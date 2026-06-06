"""Minimal stdlib web timeline viewer for a Redoland run.

No third-party deps: http.server + a single static page. Exposes a small JSON
API the page calls to render the branch tree, read a year's transcript, edit an
utterance, fork / inject / replay, and diff two worldlines.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .backend import FakeBackend
from .metrics import snapshot_metrics
from .sim import Simulation

_LOCK = threading.Lock()
_STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


class Viewer:
    def __init__(self, path):
        self.path = path

    def sim(self):
        return Simulation.open(self.path, FakeBackend())

    def tree(self):
        s = self.sim().store
        out = []
        for b in sorted(s.list_branches()):
            years = s.years_for_branch(b)
            if not years:
                continue
            tip = s.load_world(s.tag(b, years[-1]))
            out.append({"branch": b, "years": years,
                        "tip": snapshot_metrics(tip)})
        return {"branches": out}

    def year(self, b, y):
        s = self.sim().store
        years = s.years_for_branch(b)
        if not years:
            return {"events": [], "metrics": {}}
        tip = years[-1]
        raw = s.read_at(s.tag(b, tip), f"events/year-{y}.jsonl")
        events = [json.loads(l) for l in raw.splitlines() if l.strip()] if raw else []
        w = s.load_world(s.tag(b, y)) if y in years else None
        return {"events": events, "metrics": snapshot_metrics(w) if w else {},
                "years": years}

    def series(self, b):
        s = self.sim().store
        years = s.years_for_branch(b)
        pts = []
        for y in years:
            w = s.load_world(s.tag(b, y))
            pts.append(snapshot_metrics(w))
        return {"branch": b, "series": pts}

    def agent(self, b, aid):
        s = self.sim().store
        years = s.years_for_branch(b)
        if not years:
            return {}
        data = s.read_at(s.tag(b, years[-1]), f"agents/{aid}.json")
        return json.loads(data) if data else {}


def serve(path, port=8000):
    viewer = Viewer(os.path.abspath(path))

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

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    with open(os.path.join(_STATIC, "index.html"), "rb") as fh:
                        return self._send(fh.read(), ctype="text/html; charset=utf-8")
                if u.path == "/api/tree":
                    return self._send(viewer.tree())
                if u.path == "/api/year":
                    return self._send(viewer.year(q["b"], int(q["y"])))
                if u.path == "/api/series":
                    return self._send(viewer.series(q["b"]))
                if u.path == "/api/agent":
                    return self._send(viewer.agent(q["b"], q["id"]))
                if u.path == "/api/diff":
                    with _LOCK:
                        d = viewer.sim().diff(q["a"], int(q["ya"]), q["b"], int(q["yb"]))
                    return self._send(d)
                return self._send({"error": "not found"}, 404)
            except Exception as e:  # noqa
                return self._send({"error": str(e)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = self._read_json()
                with _LOCK:
                    sim = viewer.sim()
                    if u.path == "/api/fork":
                        nb = sim.fork(body["parent"], int(body["year"]), body.get("name"))
                        return self._send({"ok": True, "branch": nb})
                    if u.path == "/api/inject":
                        ch = sim.inject(body["branch"], ratio=body.get("ratio"),
                                        pile=body.get("pile"), narrate=body.get("narrate"))
                        return self._send({"ok": True, "changes": ch})
                    if u.path == "/api/edit":
                        n = sim.edit_utterance(body["branch"], body["eid"], body["text"])
                        return self._send({"ok": True, "patched": n})
                    if u.path == "/api/replay":
                        sim.replay(body["branch"], int(body["years"]))
                        return self._send({"ok": True})
                return self._send({"error": "not found"}, 404)
            except Exception as e:  # noqa
                return self._send({"error": str(e)}, 500)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Redoland viewer: http://127.0.0.1:{port}  (run: {path})")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
