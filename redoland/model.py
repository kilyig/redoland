"""Concordia LanguageModel backends for Redoland.

Two implementations of `concordia.language_model.language_model.LanguageModel`:

* ClaudeCLIModel  — every call shells out to the `claude -p` CLI on THIS session's
                    auth. It NEVER reads ANTHROPIC_API_KEY (stripped from the child
                    env) and never uses the anthropic SDK, so a full run costs $0 on
                    the metered key. The per-agent "intelligence" dial is the heritable
                    thinking-token budget, passed via the MAX_THINKING_TOKENS env var.
* StubModel       — deterministic, scriptable. Used ONLY in tests (the FakeBackend
                    analog); never calls the network, never the metered key.

Also provides `dummy_embedder`: Concordia's AssociativeMemoryBank / Simulation want a
sentence embedder, but Redoland uses its own lightweight text memory and never calls
associative retrieval, so a cheap deterministic vector suffices (keeps numpy as the
only extra dependency and the checkpoints embedder-free).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Optional


class ModelUnavailable(RuntimeError):
    """Raised when `claude -p` cannot be reached or returns an error (out of usage
    credits, expired auth, CLI missing, persistent timeout). We HALT the run rather
    than let the failure be silently read as an empty 'pass' action — otherwise a dead
    model grinds the simulation forward on noise (600 empty passes per year)."""

import numpy as np

from concordia.language_model import language_model

_EMBED_DIM = 16


def dummy_embedder(text: str) -> np.ndarray:
    """Deterministic unit-norm vector from a hash. Never semantically meaningful —
    Redoland does not use associative retrieval; this only satisfies the API."""
    h = hashlib.sha256(text.encode("utf-8")).digest()      # 32 bytes
    v = np.frombuffer(h, dtype=np.uint8)[:_EMBED_DIM].astype(np.float64)
    v = v / (np.linalg.norm(v) or 1.0)
    return v


# --------------------------------------------------------------------------- #
# Robust JSON extraction (shared) — a malformed/empty/truncated reply must     #
# never crash a multi-hour run.                                                #
# --------------------------------------------------------------------------- #


def safe_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}


# --------------------------------------------------------------------------- #
# claude -p backend.                                                           #
# --------------------------------------------------------------------------- #


class ClaudeCLIModel(language_model.LanguageModel):
    """All inference via `claude -p` (session auth). thinking_tokens = the agent's
    heritable intelligence dial, applied via MAX_THINKING_TOKENS per call."""

    def __init__(self, *, thinking_tokens: int = 0, model: str = "claude-haiku-4-5",
                 cli: Optional[str] = None, timeout: int = 180,
                 retries: int = 3, retry_wait: float = 2.0):
        self._thinking = max(0, int(thinking_tokens))
        self._model = model
        self._cli = cli or os.environ.get("REDOLAND_CLAUDE_BIN", "claude")
        self._timeout = timeout
        self._retries = max(1, int(retries))        # transient blips get a few retries
        self._retry_wait = max(0.0, float(retry_wait))
        self._think_choices = False    # gating CHOICEs are cheap unless a caller opts in

    @contextlib.contextmanager
    def deliberate_on_choices(self):
        """Within this block, sample_choice spends the agent's thinking budget instead
        of snap-judging — for the rare CHOICE that deserves real deliberation (e.g.
        a life-or-death decision to join a fight)."""
        prev = self._think_choices
        self._think_choices = True
        try:
            yield
        finally:
            self._think_choices = prev

    # -- core CLI call --------------------------------------------------- #
    def _run(self, prompt: str, *, thinking: Optional[int] = None) -> str:
        """Call `claude -p` and return its text. A FAILED call (non-zero exit, error
        envelope, timeout, or missing CLI) is retried a few times, then raised as
        ModelUnavailable — NOT swallowed into an empty string, so the run halts instead
        of disguising the failure as a 'pass'. A successful empty result is fine."""
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)          # never the metered key
        env["MAX_THINKING_TOKENS"] = str(self._thinking if thinking is None else thinking)
        cmd = [self._cli, "-p", prompt, "--model", self._model,
               "--output-format", "json", "--no-session-persistence"]
        last = "unknown error"
        for attempt in range(self._retries):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=self._timeout, env=env)
            except subprocess.TimeoutExpired:
                last = f"timed out after {self._timeout}s"
            except Exception as e:                  # CLI missing / cannot spawn
                last = f"could not run '{self._cli}': {e}"
            else:
                if proc.returncode == 0:
                    ok, text = self._envelope(proc.stdout)
                    if ok:
                        return text                 # success — empty text is valid
                    last = "error response from claude -p"
                else:
                    msg = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
                    last = f"exit {proc.returncode}: {msg[:200]}"
            if attempt + 1 < self._retries:
                time.sleep(self._retry_wait)
        raise ModelUnavailable(
            f"claude -p ({self._model}) gave no usable response after {self._retries} "
            f"attempts — {last}. Halting the run (likely out of usage credits, expired "
            f"auth, or the CLI is unavailable) rather than committing empty 'pass' steps.")

    @staticmethod
    def _envelope(stdout: str):
        """Unwrap the `claude -p --output-format json` envelope. Returns (ok, text):
        ok=False marks an error envelope (a failed call); a successful empty result is
        (True, '')."""
        raw = (stdout or "").strip()
        if not raw:
            return True, ""                         # exit 0 + empty = a valid empty answer
        try:
            env = json.loads(raw)
            if isinstance(env, dict):
                if env.get("is_error") or str(env.get("subtype", "")).startswith("error"):
                    return False, ""
                if "result" in env:
                    return True, env.get("result") or ""
        except Exception:
            pass
        return True, raw

    # -- LanguageModel interface ----------------------------------------- #
    def sample_text(self, prompt: str, *, max_tokens: int = 5000,
                    terminators: Collection[str] = (), temperature: float = 1.0,
                    top_p: float = 0.95, top_k: int = 64, timeout: float = 60,
                    seed: Optional[int] = None) -> str:
        out = self._run(prompt)
        for t in terminators:                        # honor the API's terminator contract
            i = out.find(t)
            if i != -1:
                out = out[:i]
        return out

    def sample_choice(self, prompt: str, responses: Sequence[str], *,
                      seed: Optional[int] = None) -> tuple[int, str, Mapping[str, Any]]:
        """Concordia passes single letters ('a','b',…) as `responses`, with the
        lettered menu already in `prompt`. Pick one and return its index. (Also
        works if called directly with full-text options.)"""
        opts = [str(r) for r in responses]
        q = (prompt.rstrip() + "\n\nReply with ONLY your choice, exactly as written "
             "(no explanation): " + " | ".join(opts))
        order = sorted(range(len(opts)), key=lambda j: -len(opts[j]))   # longest first
        # Gating / enum picks (willing?, press-or-flee, …) are snap judgments — they run
        # with NO extended thinking. The agent's intelligence budget is spent on the
        # substantive generative decisions (sample_text), and on any CHOICE a caller has
        # wrapped in deliberate_on_choices() (e.g. accept_join).
        for attempt in range(3):
            think = self._thinking if (self._think_choices and attempt == 0) else 0
            raw = self._run(q, thinking=think).strip().lower()
            for i in order:
                o = opts[i].lower()
                if len(o) == 1:
                    if re.search(r"(?<![a-z0-9])" + re.escape(o) + r"(?![a-z0-9])", raw):
                        return i, opts[i], {"raw": raw}
                elif o and o in raw:
                    return i, opts[i], {"raw": raw}
        return 0, opts[0], {"raw": "fallback", "fallback": True}


# --------------------------------------------------------------------------- #
# Deterministic test stub.                                                     #
# --------------------------------------------------------------------------- #


class StubModel(language_model.LanguageModel):
    """Scriptable, deterministic model for tests. `choice_fn(prompt, responses)`
    returns an index (or None → 0); `text_fn(prompt)` returns a string. Defaults
    pick the first option / empty text so the engine stays well-defined."""

    def __init__(self, choice_fn=None, text_fn=None):
        self._choice_fn = choice_fn or (lambda prompt, responses: 0)
        self._text_fn = text_fn or (lambda prompt: "")
        self.calls: list[tuple[str, str]] = []

    @contextlib.contextmanager
    def deliberate_on_choices(self):
        yield                          # no-op for the deterministic stub

    def sample_text(self, prompt: str, *, max_tokens: int = 5000,
                    terminators: Collection[str] = (), temperature: float = 1.0,
                    top_p: float = 0.95, top_k: int = 64, timeout: float = 60,
                    seed: Optional[int] = None) -> str:
        self.calls.append(("text", prompt))
        return self._text_fn(prompt) or ""

    def sample_choice(self, prompt: str, responses: Sequence[str], *,
                      seed: Optional[int] = None) -> tuple[int, str, Mapping[str, Any]]:
        self.calls.append(("choice", prompt))
        idx = self._choice_fn(prompt, list(responses))
        idx = 0 if idx is None else int(idx) % len(responses)
        return idx, responses[idx], {}
