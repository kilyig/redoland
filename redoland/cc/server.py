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

from ..core import Params
from .metrics import snapshot_metrics, distributions
from .sim import Simulation
from .store import safe_branch

_STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")

# single global background job (one run/fork/continue at a time)
JOB = {"busy": False, "run": None, "branch": None, "phase": "idle", "error": None,
       "target_years": 0}
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

    # -- background jobs ------------------------------------------------- #
    def start_job(self, kind, **kw):
        with _JOB_LOCK:
            if JOB["busy"]:
                return {"ok": False, "error": "A run is already in progress."}
            JOB.update(busy=True, error=None, phase="starting",
                       run=kw.get("run") or kw.get("name"), branch=None,
                       target_years=int(kw.get("years", 0) or 0))
        threading.Thread(target=self._run_job, args=(kind, kw), daemon=True).start()
        return {"ok": True}

    def _run_job(self, kind, kw):
        try:
            mf = self.model_factory()
            if kind == "new":
                name = safe_branch(kw["name"])
                params = Params(founders=int(kw.get("founders", 6)),
                                ratio=float(kw.get("ratio", Params().ratio)))
                JOB.update(run=name, branch="main", phase="founding")
                sim = Simulation.create(self.path(name), params, int(kw.get("seed", 1)), mf)
                JOB.update(phase="running")
                if int(kw.get("years", 0) or 0):
                    sim.run(int(kw["years"]), log=self._progress)
            elif kind == "continue":
                run, branch = kw["run"], kw["branch"]
                JOB.update(run=run, branch=branch, phase="running")
                sim = Simulation.open(self.path(run), mf)
                sim.store.checkout_branch(branch)
                sim.run(int(kw["years"]), log=self._progress)
            elif kind == "fork":
                run = kw["run"]
                sim = Simulation.open(self.path(run), mf)
                nb = sim.fork(kw["parent"], int(kw["year"]), kw.get("name"))
                JOB.update(run=run, branch=nb, phase="running")
                if int(kw.get("years", 0) or 0):
                    sim.replay(nb, int(kw["years"]), log=self._progress)
        except Exception as e:  # noqa
            JOB["error"] = str(e)
        finally:
            JOB.update(busy=False, phase="idle")

    def _progress(self, year, world):
        JOB["branch"] = world.branch

    def inject(self, run, branch, ratio=None, pile=None, narrate=None):
        sim = Simulation.open(self.path(run), model_factory=None)
        return sim.inject(branch, ratio=ratio, pile=pile, narrate=narrate)


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
                if u.path == "/api/new":
                    return self._send(mgr.start_job("new", **body))
                if u.path == "/api/continue":
                    return self._send(mgr.start_job("continue", **body))
                if u.path == "/api/fork":
                    return self._send(mgr.start_job("fork", **body))
                if u.path == "/api/inject":
                    ch = mgr.inject(body["run"], body["branch"], ratio=body.get("ratio"),
                                    pile=body.get("pile"), narrate=body.get("narrate"))
                    return self._send({"ok": True, "changes": ch})
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
