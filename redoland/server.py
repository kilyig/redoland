"""Read-only web UI server for Redoland.

Serves the whole runs/ directory: per-year transcripts, metrics, and stat
distributions, with the in-progress year exposed as a tab that auto-updates as new
action-commits land (the UI just polls — there is no live stream). The ONLY write
action is Start/Pause a branch's run. MULTIPLE branches can run concurrently: each
running branch gets its own background worker and (for branches of the same world)
its own git worktree, so their commits don't clash. Creating, forking, and injecting
worldlines is done by the AI via the CLI, not here.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .metrics import snapshot_metrics, distributions
from .sim import Simulation
from .store import safe_branch

_STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

# Background runners, keyed by "<run>/<branch>". Each entry is one running branch;
# multiple may run at once (concurrency). A branch leaves the registry the moment it
# is paused or finishes — so the "active branches" picker only ever lists what is
# genuinely running right now. All branch CREATION (fork / inject / new world) is the
# AI's job via the CLI; the UI only runs/pauses.
JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _job_key(run: str, branch: str) -> str:
    return f"{run}/{safe_branch(branch)}"


class Manager:
    def __init__(self, runs_dir: str):
        self.runs_dir = os.path.abspath(runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)

    def path(self, run: str) -> str:
        return os.path.join(self.runs_dir, safe_branch(run))

    def worktree_path(self, run: str, branch: str) -> str:
        # worktrees live OUTSIDE the run repos (sibling dir) so they never nest inside
        # a working tree or get picked up by a commit.
        return os.path.join(self.runs_dir, ".worktrees", safe_branch(run), safe_branch(branch))

    def model_factory(self, run=None):
        """A per-agent model factory using the model this run was created with (so a
        Sonnet village runs on Sonnet, a Haiku village on Haiku)."""
        import json
        from .cli import make_model_factory
        model = None
        if run is not None:
            try:
                meta = json.load(open(os.path.join(self.path(run), "meta.json")))
                model = meta.get("params", {}).get("model")
            except Exception:
                model = None
        return make_model_factory(model) if model else make_model_factory()

    # -- reads (no model needed) ----------------------------------------- #
    def list_runs(self):
        out = []
        for n in sorted(os.listdir(self.runs_dir)):
            # `.git` is a dir in a normal run repo, a FILE in a submodule/worktree checkout
            if os.path.exists(os.path.join(self.runs_dir, n, ".git")):
                out.append(n)
        return out

    def store(self, run):
        return Simulation.open(self.path(run), model_factory=None).store

    def _years(self, s, b):
        """Tagged (completed) years PLUS the in-progress year at the branch tip, so a
        growing branch shows its current year as a tab that auto-updates."""
        years = set(s.years_for_branch(b))
        try:
            tip_year = s.load_world(safe_branch(b)).year
            if tip_year and tip_year > 0:
                years.add(tip_year)
        except Exception:
            pass
        return sorted(years)

    def tree(self, run):
        s = self.store(run)
        out = []
        for b in sorted(s.list_branches()):
            years = self._years(s, b)
            if not years:
                continue
            tip = s.load_world(safe_branch(b))     # the live tip (incl. in-progress year)
            out.append({"branch": b, "years": years, "tip": snapshot_metrics(tip)})
        return {"branches": out}

    def year(self, run, b, y):
        s = self.store(run)
        tagged = s.years_for_branch(b)
        # events come from the branch TIP (it holds every year's cumulative file, incl.
        # the in-progress year as it grows) — that's what makes the tab auto-update.
        raw = s.read_at(safe_branch(b), f"events/year-{y}.jsonl")
        events = [json.loads(l) for l in raw.splitlines() if l.strip()] if raw else []
        ref = s.tag(b, y) if y in tagged else safe_branch(b)   # completed -> tag; live -> tip
        w = s.load_world(ref)
        return {"events": events,
                "metrics": snapshot_metrics(w),
                "dist": distributions(w),
                "years": self._years(s, b)}

    # -- start / pause branches (many at once) --------------------------- #
    def active(self):
        """The branches running right now — what the top-right picker lists. Each entry
        carries the world (run) and branch name, plus the current year for context."""
        with _JOBS_LOCK:
            return {"active": [
                {"run": j["run"], "branch": j["branch"], "year": j.get("year"),
                 "phase": j["phase"], "error": j.get("error")}
                for j in JOBS.values() if j["phase"] in ("running", "error")]}

    def start(self, run, branch="main"):
        """Run the given branch forward (one commit per action) until paused or extinct.
        Concurrent with any other running branch. Branches of the SAME world each run in
        their own git worktree so their commits don't clash."""
        branch = safe_branch(branch)
        key = _job_key(run, branch)
        with _JOBS_LOCK:
            j = JOBS.get(key)
            if j and j["phase"] == "running":
                return {"ok": False, "error": "That branch is already running."}
            JOBS[key] = {"run": run, "branch": branch, "phase": "running",
                         "error": None, "pause": False, "year": None}
        threading.Thread(target=self._run, args=(run, branch, key), daemon=True).start()
        return {"ok": True}

    def pause(self, run, branch="main"):
        """Stop a branch at the next step boundary; it then drops out of `active`."""
        key = _job_key(run, branch)
        j = JOBS.get(key)
        if not j:
            return {"ok": False, "error": "That branch is not running."}
        j["pause"] = True
        return {"ok": True}

    def _run(self, run, branch, key):
        try:
            # one worktree per running branch (so same-world branches don't clash)
            main_store = self.store(run)
            wt = main_store.ensure_worktree(branch, self.worktree_path(run, branch))
            sim = Simulation.open(wt, self.model_factory(run))
            sim.store.checkout_branch(branch)
            sim.run(years=None, log=self._progress(key),
                    should_stop=lambda: JOBS.get(key, {}).get("pause", True))
        except Exception as e:  # noqa
            j = JOBS.get(key)
            if j:
                j["phase"], j["error"] = "error", str(e)
            return
        # clean stop (paused or extinct) -> leave the registry so it drops from the picker
        with _JOBS_LOCK:
            if JOBS.get(key, {}).get("phase") == "running":
                JOBS.pop(key, None)

    def _progress(self, key):
        def log(year, world):
            j = JOBS.get(key)
            if j:
                j["year"] = world.year
        return log


def serve(runs_dir="runs", port=8000, host="127.0.0.1"):
    mgr = Manager(runs_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, obj, code=200, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # never cache — the page and data change constantly; avoids stale UI
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    with open(os.path.join(_STATIC, "redoland.html"), "rb") as fh:
                        return self._send(fh.read(), ctype="text/html; charset=utf-8")
                if u.path == "/api/runs":
                    return self._send({"runs": mgr.list_runs()})
                if u.path == "/api/active":
                    return self._send(mgr.active())
                if u.path == "/api/tree":
                    return self._send(mgr.tree(q["run"]))
                if u.path == "/api/year":
                    return self._send(mgr.year(q["run"], q["b"], int(q["y"])))
                return self._send({"error": "not found"}, 404)
            except Exception as e:  # noqa
                return self._send({"error": str(e)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = self._read_json()
                if u.path == "/api/start":
                    return self._send(mgr.start(body["run"], body.get("branch", "main")))
                if u.path == "/api/pause":
                    return self._send(mgr.pause(body["run"], body.get("branch", "main")))
                return self._send({"error": "not found"}, 404)
            except Exception as e:  # noqa
                return self._send({"error": str(e)}, 500)

    # Loopback by default: there is no authentication, and POST /api/start triggers paid
    # `claude -p` inference. Pass host="0.0.0.0" to expose it (e.g. when running inside
    # a container and viewing from the host) — anyone who can reach the port can Start.
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Redoland UI: http://{host}:{port}  (runs dir: {mgr.runs_dir})")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
