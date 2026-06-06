"""Redoland command line.

    python -m redoland init world --years 30 --preset stable
    python -m redoland run world --years 10
    python -m redoland log world
    python -m redoland fork world --parent main --year 12 --name drought
    python -m redoland inject world --branch drought --ratio 0.55 --narrate "drought"
    python -m redoland replay world --branch drought --years 18
    python -m redoland edit world --branch drought --eid e000123 --text "new words"
    python -m redoland diff world drought 30 control 30
    python -m redoland metrics world --branch main --year 30
    python -m redoland serve world --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .core import Params, PRESETS
from .metrics import snapshot_metrics


def run_path(name: str) -> str:
    if os.path.isabs(name) or name.startswith("./") or name.startswith("runs/"):
        return name
    return os.path.join("runs", name)


def make_backend(name: str):
    if name == "anthropic":
        from .backend import AnthropicBackend
        return AnthropicBackend()
    from .backend import FakeBackend
    return FakeBackend()


def _sim(name, backend="fake"):
    from .sim import Simulation
    return Simulation.open(run_path(name), make_backend(backend))


def cmd_init(a):
    from .sim import Simulation
    ratio = PRESETS.get(a.preset, a.ratio) if a.preset else a.ratio
    params = Params(founders=a.founders, ratio=ratio)
    sim = Simulation.create(run_path(a.name), params, a.seed, make_backend(a.backend))
    print(f"created run '{a.name}' (ratio {ratio}, {a.founders} founders, seed {a.seed})")
    if a.years:
        sim.run(a.years, log=lambda y, w: print(f"  year {y}: {len(w.living())} living"))
        print(f"ran {a.years} years.")


def cmd_run(a):
    sim = _sim(a.name, a.backend)
    sim.run(a.years, log=lambda y, w: print(f"  year {y}: {len(w.living())} living"))


def cmd_log(a):
    sim = _sim(a.name)
    store = sim.store
    print(f"run: {a.name}  (git branch: {store.current_branch()})")
    for b in sorted(store.list_branches()):
        years = store.years_for_branch(b)
        if not years:
            continue
        w = store.load_world(store.tag(b, years[-1]))
        m = snapshot_metrics(w)
        print(f"  {b:12} y{years[0]}..y{years[-1]}  pop={m['population']:2d} "
              f"born={m['births']:2d} starv={m['deaths_starvation']:2d} "
              f"nat={m['deaths_natural']:2d} gen={m.get('max_generation')}")


def cmd_fork(a):
    sim = _sim(a.name)
    nb = sim.fork(a.parent, a.year, a.fork_name)
    print(f"forked '{nb}' from {a.parent}@y{a.year}")


def cmd_inject(a):
    sim = _sim(a.name)
    ch = sim.inject(a.branch, ratio=a.ratio, pile=a.pile, narrate=a.narrate)
    print(f"injected into '{a.branch}': {', '.join(ch) or 'nothing'}")


def cmd_replay(a):
    sim = _sim(a.name, a.backend)
    sim.replay(a.branch, a.years,
               log=lambda y, w: print(f"  {a.branch} year {y}: {len(w.living())} living"))


def cmd_edit(a):
    sim = _sim(a.name)
    n = sim.edit_utterance(a.branch, a.eid, a.text)
    print(f"patched {n} location(s) for {a.eid}. Replay '{a.branch}' to see downstream "
          f"effects (behavioural divergence needs the anthropic backend).")


def cmd_diff(a):
    sim = _sim(a.name)
    d = sim.diff(a.branch_a, a.year_a, a.branch_b, a.year_b)
    print(json.dumps(d, indent=2))


def cmd_metrics(a):
    sim = _sim(a.name)
    w = sim.store.load_world(sim.store.tag(a.branch, a.year))
    print(json.dumps(snapshot_metrics(w), indent=2))


def cmd_serve(a):
    from .server import serve
    serve(run_path(a.name), port=a.port)


def build_parser():
    p = argparse.ArgumentParser(prog="redoland")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a new run")
    s.add_argument("name")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--founders", type=int, default=10)
    s.add_argument("--ratio", type=float, default=Params().ratio)
    s.add_argument("--preset", choices=list(PRESETS), default=None)
    s.add_argument("--years", type=int, default=0)
    s.add_argument("--backend", default="fake", choices=["fake", "anthropic"])
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("run", help="simulate forward on the current branch")
    s.add_argument("name"); s.add_argument("--years", type=int, default=10)
    s.add_argument("--backend", default="fake", choices=["fake", "anthropic"])
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("log", help="show branches and metrics")
    s.add_argument("name"); s.set_defaults(fn=cmd_log)

    s = sub.add_parser("fork", help="fork a worldline at a past year")
    s.add_argument("name"); s.add_argument("--parent", default="main")
    s.add_argument("--year", type=int, required=True)
    s.add_argument("--name", dest="fork_name", default=None)
    s.set_defaults(fn=cmd_fork)

    s = sub.add_parser("inject", help="inject an event at a branch tip")
    s.add_argument("name"); s.add_argument("--branch", required=True)
    s.add_argument("--ratio", type=float, default=None)
    s.add_argument("--pile", type=int, default=None)
    s.add_argument("--narrate", default=None)
    s.set_defaults(fn=cmd_inject)

    s = sub.add_parser("replay", help="replay/run a branch forward")
    s.add_argument("name"); s.add_argument("--branch", required=True)
    s.add_argument("--years", type=int, default=10)
    s.add_argument("--backend", default="fake", choices=["fake", "anthropic"])
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("edit", help="edit a recorded utterance")
    s.add_argument("name"); s.add_argument("--branch", required=True)
    s.add_argument("--eid", required=True); s.add_argument("--text", required=True)
    s.set_defaults(fn=cmd_edit)

    s = sub.add_parser("diff", help="compare two branch/year snapshots")
    s.add_argument("name")
    s.add_argument("branch_a"); s.add_argument("year_a", type=int)
    s.add_argument("branch_b"); s.add_argument("year_b", type=int)
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("metrics", help="metrics for one branch/year")
    s.add_argument("name"); s.add_argument("--branch", default="main")
    s.add_argument("--year", type=int, required=True)
    s.set_defaults(fn=cmd_metrics)

    s = sub.add_parser("serve", help="launch the web timeline viewer")
    s.add_argument("name"); s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
