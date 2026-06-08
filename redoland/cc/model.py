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

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Optional

import numpy as np

from concordia.language_model import language_model

_EMBED_DIM = 16


def dummy_embedder(text: str) -> np.ndarray:
    """Deterministic unit-norm vector from a hash. Never semantically meaningful —
    Redoland does not use associative retrieval; this only satisfies the API."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    v = np.frombuffer(h[:_EMBED_DIM * 4], dtype=np.uint32).astype(np.float64)
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
                 cli: Optional[str] = None, timeout: int = 180):
        self._thinking = max(0, int(thinking_tokens))
        self._model = model
        self._cli = cli or os.environ.get("REDOLAND_CLAUDE_BIN", "claude")
        self._timeout = timeout

    # -- core CLI call --------------------------------------------------- #
    def _run(self, prompt: str, *, thinking: Optional[int] = None) -> str:
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)          # never the metered key
        env["MAX_THINKING_TOKENS"] = str(self._thinking if thinking is None else thinking)
        try:
            proc = subprocess.run(
                [self._cli, "-p", prompt,
                 "--model", self._model,
                 "--output-format", "json",
                 "--no-session-persistence"],
                capture_output=True, text=True, timeout=self._timeout, env=env)
        except Exception:
            return ""
        return self._envelope(proc.stdout)

    @staticmethod
    def _envelope(stdout: str) -> str:
        """Unwrap the `claude -p --output-format json` envelope to the result text;
        retry-on-truncation is handled by callers checking for empty output."""
        raw = (stdout or "").strip()
        if not raw:
            return ""
        try:
            env = json.loads(raw)
            if isinstance(env, dict) and "result" in env:
                return env.get("result") or ""
        except Exception:
            pass
        return raw

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
        letters = [chr(ord("a") + i) for i in range(len(responses))]
        menu = "\n".join(f"  ({l}) {r}" for l, r in zip(letters, responses))
        q = (f"{prompt}\n\nChoose exactly ONE option. Respond with ONLY its letter "
             f"({letters[0]}-{letters[-1]}) and nothing else:\n{menu}")
        for attempt in range(3):
            raw = self._run(q, thinking=0 if attempt else self._thinking).strip().lower()
            m = re.search(r"[a-z]", raw)
            if m:
                idx = ord(m.group(0)) - ord("a")
                if 0 <= idx < len(responses):
                    return idx, responses[idx], {"raw": raw}
            # fallback: maybe it echoed the option text
            for i, r in enumerate(responses):
                if r.lower() in raw:
                    return i, responses[i], {"raw": raw}
        return 0, responses[0], {"raw": "fallback", "fallback": True}


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
