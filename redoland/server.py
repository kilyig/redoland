"""Web UI server for Redoland (Concordia engine).

Serves the whole runs/ directory so the UI can browse, START new worldlines,
CONTINUE a worldline (run more years), FORK a past year into a new worldline, and
inject events — plus per-year transcripts, metrics, and stat distributions. Running
is long (claude -p), so run/fork/continue execute in a single background worker
thread and stream turn-by-turn to live.jsonl (SSE). Browsing needs no model.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .metrics import snapshot_metrics, distributions
from .sim import Simulation
from .store import safe_branch

_STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

# single global background runner (start/pause a branch). All branch CREATION
# (fork / inject / new world) is done by the AI via the CLI; the UI only runs/pauses.
JOB = {"busy": False, "run": None, "branch": None, "phase": "idle", "error": None,
       "pause": False}
_JOB_LOCK = threading.Lock()


class Manager:
    def __init__(self, runs_dir: str):
        self.runs_dir = os.path.abspath(runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)

    def path(self, run: str) -> str:
        return os.path.join(self.runs_dir, safe_branch(run))

    def model_factory(self):
        from .cli import make_model_factory
        return make_model_factory()

    # -- reads (no model needed) ----------------------------------------- #
    def list_runs(self):
        out = []
        for n in sorted(os.listdir(self.runs_dir)):
            if os.path.isdir(os.path.join(self.runs_dir, n, ".git")):
                out.append(n)
        return out

    def store(self, run):
        return Simulation.open(self.path(run), model_factory=None).store

    def tree(self, run):
        s = self.store(run)
        out = []
        for b in sorted(s.list_branches()):
            years = s.years_for_branch(b)
            if not years:
                continue
            tip = s.load_world(s.tag(b, years[-1]))
            out.append({"branch": b, "years": years, "tip": snapshot_metrics(tip)})
        return {"branches": out}

    def year(self, run, b, y):
        s = self.store(run)
        years = s.years_for_branch(b)
        if not years:
            return {"events": [], "metrics": {}, "dist": {}}
        raw = s.read_at(s.tag(b, years[-1]), f"events/year-{y}.jsonl")
        events = [json.loads(l) for l in raw.splitlines() if l.strip()] if raw else []
        w = s.load_world(s.tag(b, y)) if y in years else None
        return {"events": events,
                "metrics": snapshot_metrics(w) if w else {},
                "dist": distributions(w) if w else {},
                "years": years}

    # -- start / pause a branch ------------------------------------------ #
    def start(self, run, branch="main"):
        """Run the given branch forward (one commit per action) until paused or
        extinct. Branch creation/intervention is the AI's job; this just runs."""
        with _JOB_LOCK:
            if JOB["busy"]:
                return {"ok": False, "error": "A branch is already running. Pause it first."}
            JOB.update(busy=True, paused=False, error=None, phase="running",
                       run=run, branch=branch, pause=False)
        threading.Thread(target=self._run, args=(run, branch), daemon=True).start()
        return {"ok": True}

    def pause(self):
        JOB["pause"] = True            # the run loop stops at the next step boundary
        return {"ok": True}

    def _run(self, run, branch):
        try:
            sim = Simulation.open(self.path(run), self.model_factory())
            sim.store.checkout_branch(branch)
            sim.run(years=None, log=self._progress, should_stop=lambda: JOB["pause"])
        except Exception as e:  # noqa
            JOB["error"] = str(e)
        finally:
            JOB.update(busy=False, phase="idle", pause=False)

    def _progress(self, year, world):
        JOB["branch"] = world.branch


def serve(runs_dir="runs", port=8000):
    mgr = Manager(runs_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, obj, code=200, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def _sse_live(self, run):
            path = os.path.join(mgr.path(run), "live.jsonl")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
            except Exception:
                return
            pos = 0
            idle = 0
            try:
                while True:
                    if os.path.exists(path):
                        size = os.path.getsize(path)
                        if size < pos:
                            pos = 0
                        if size > pos:
                            with open(path) as f:
                                f.seek(pos)
                                data = f.read()
                                pos = f.tell()
                            for ln in data.splitlines():
                                ln = ln.strip()
                                if not ln:
                                    continue
                                try:
                                    self.wfile.write(b"data: " + ln.encode() + b"\n\n")
                                except Exception:
                                    return
                            try:
                                self.wfile.flush()
                            except Exception:
                                return
                            idle = 0
                            time.sleep(0.2)
                            continue
                    idle += 1
                    if idle % 10 == 0:
                        try:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                        except Exception:
                            return
                    time.sleep(0.3)
            except Exception:
                return

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/live":
                    return self._sse_live(q.get("run", ""))
                if u.path in ("/", "/index.html"):
                    with open(os.path.join(_STATIC, "redoland.html"), "rb") as fh:
                        return self._send(fh.read(), ctype="text/html; charset=utf-8")
                if u.path == "/api/runs":
                    return self._send({"runs": mgr.list_runs()})
                if u.path == "/api/status":
                    return self._send(dict(JOB))
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
                    return self._send(mgr.pause())
                return self._send({"error": "not found"}, 404)
            except Exception as e:  # noqa
                return self._send({"error": str(e)}, 500)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Redoland UI: http://127.0.0.1:{port}  (runs dir: {mgr.runs_dir})")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
