"""Redoland command line (Concordia engine).

    python -m redoland init world --founders 6 --years 5
    python -m redoland run world --years 5
    python -m redoland fork world --parent main --year 3 --name drought
    python -m redoland inject world --branch drought --ratio 0.6 --narrate "drought"
    python -m redoland replay world --branch drought --years 5
    python -m redoland log world
    python -m redoland serve            # web UI over the whole runs/ dir

All inference runs through `claude -p` (session auth, never the metered key).
"""

from __future__ import annotations

import argparse
import json
import os

from .core import Params, PRESETS
from .metrics import snapshot_metrics


def run_path(name: str) -> str:
    if os.path.isabs(name) or name.startswith("./") or name.startswith("runs/"):
        return name
    return os.path.join("runs", name)


def make_model_factory(model: str = "claude-haiku-4-5"):
    """Production factory: each agent gets a claude -p model whose thinking budget
    is its heritable intelligence dial. Never the metered key."""
    from .model import ClaudeCLIModel
    return lambda thinking_tokens: ClaudeCLIModel(thinking_tokens=thinking_tokens, model=model)


def _sim(name):
    from .sim import Simulation
    return Simulation.open(run_path(name), make_model_factory())


def cmd_init(a):
    from .sim import Simulation
    ratio = PRESETS.get(a.preset, a.ratio) if a.preset else a.ratio
    params = Params(founders=a.founders, ratio=ratio)
    sim = Simulation.create(run_path(a.name), params, a.seed, make_model_factory())
    print(f"created run '{a.name}' (ratio {ratio}, {a.founders} founders, seed {a.seed})")
    if a.years:
        sim.run(a.years, log=lambda y, w: print(f"  year {y}: {len(w.living())} living"))


def cmd_run(a):
    sim = _sim(a.name)
    if getattr(a, "branch", None):
        sim.store.checkout_branch(a.branch)
    sim.run(a.years, log=lambda y, w: print(f"  year {y}: {len(w.living())} living"))


def cmd_log(a):
    s = _sim(a.name).store
    print(f"run: {a.name}  (git branch: {s.current_branch()})")
    for b in sorted(s.list_branches()):
        years = s.years_for_branch(b)
        if not years:
            continue
        m = snapshot_metrics(s.load_world(s.tag(b, years[-1])))
        print(f"  {b:12} y{years[0]}..y{years[-1]}  pop={m['population']:2d} "
              f"born={m['births']:2d} starv={m['deaths_starvation']:2d} "
              f"combat={m['deaths_combat']:2d} nat={m['deaths_natural']:2d}")


def cmd_fork(a):
    sim = _sim(a.name)
    if a.at:                              # fork from any action-commit
        print("forked:", sim.fork_at(a.at, a.fork_name or f"fork_{a.at[:6]}"))
    else:                                 # fork from a year tag
        print("forked:", sim.fork(a.parent, a.year, a.fork_name))


def cmd_inject(a):
    changes = json.loads(a.changes) if a.changes else {}
    res = _sim(a.name).inject(a.branch, changes=changes, narrative=a.narrate,
                              ratio=a.ratio, pile=a.pile)
    print("injected:", res["text"])
    print("effects:", res["effects"])


def cmd_timeline(a):
    for r in _sim(a.name).store.timeline(a.branch, n=a.n):
        print(f"  {r['commit']}  {r['desc']}")


def cmd_state(a):
    s = _sim(a.name).store
    ref = a.at or (s.tag(a.branch, a.year) if a.year is not None else None)
    w = s.load_world(ref)
    print(f"branch {w.branch}  year {w.year}  phase {w.cursor.get('phase')}  pile {w.pile}  "
          f"living {len(w.living())}")
    for x in w.living():
        print(f"  {x.id} {x.name:9} {x.sex:6} age {x.age:>2} food {x.food:>2} "
              f"hp {int(x.hp):>3} str {x.strength:>3} sat {x.health}/{w.params.health_max} "
              f"int {x.intelligence_tokens} mem {x.memory_tokens}")


def cmd_replay(a):
    _sim(a.name).replay(a.branch, a.years,
                        log=lambda y, w: print(f"  {a.branch} year {y}: {len(w.living())} living"))


def cmd_metrics(a):
    s = _sim(a.name).store
    print(json.dumps(snapshot_metrics(s.load_world(s.tag(a.branch, a.year))), indent=2))


def cmd_serve(a):
    from .server import serve
    serve(os.path.abspath(a.runs_dir), port=a.port)


def build_parser():
    p = argparse.ArgumentParser(prog="redoland")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init"); s.add_argument("name")
    s.add_argument("--seed", type=int, default=1); s.add_argument("--founders", type=int, default=6)
    s.add_argument("--ratio", type=float, default=Params().ratio)
    s.add_argument("--preset", choices=list(PRESETS), default=None)
    s.add_argument("--years", type=int, default=0); s.set_defaults(fn=cmd_init)

    s = sub.add_parser("run"); s.add_argument("name"); s.add_argument("--years", type=int, default=5)
    s.add_argument("--branch", default=None); s.set_defaults(fn=cmd_run)

    s = sub.add_parser("log"); s.add_argument("name"); s.set_defaults(fn=cmd_log)

    s = sub.add_parser("timeline", help="per-action commit history (fork points)")
    s.add_argument("name"); s.add_argument("--branch", default=None)
    s.add_argument("--n", type=int, default=40); s.set_defaults(fn=cmd_timeline)

    s = sub.add_parser("state", help="world snapshot (roster) at a branch/year/commit")
    s.add_argument("name"); s.add_argument("--branch", default="main")
    s.add_argument("--year", type=int, default=None); s.add_argument("--at", default=None)
    s.set_defaults(fn=cmd_state)

    s = sub.add_parser("fork"); s.add_argument("name"); s.add_argument("--parent", default="main")
    s.add_argument("--year", type=int, default=None)
    s.add_argument("--at", default=None, help="commit hash to fork from (any action)")
    s.add_argument("--name", dest="fork_name", default=None)
    s.set_defaults(fn=cmd_fork)

    s = sub.add_parser("inject"); s.add_argument("name"); s.add_argument("--branch", required=True)
    s.add_argument("--changes", default=None, help="JSON: agents/kill/pile/spawn/params")
    s.add_argument("--narrate", default=None, help="the explanation shown to all agents")
    s.add_argument("--ratio", type=float, default=None); s.add_argument("--pile", type=int, default=None)
    s.set_defaults(fn=cmd_inject)

    s = sub.add_parser("replay"); s.add_argument("name"); s.add_argument("--branch", required=True)
    s.add_argument("--years", type=int, default=5); s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("metrics"); s.add_argument("name"); s.add_argument("--branch", default="main")
    s.add_argument("--year", type=int, required=True); s.set_defaults(fn=cmd_metrics)

    s = sub.add_parser("serve"); s.add_argument("--runs-dir", default="runs")
    s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    from .env import load_dotenv
    load_dotenv()
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
