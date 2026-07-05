"""Redoland command line.

Run `python -m redoland -h` for the full overview (mental model, the fork/inject/run
workflow, and the inject --changes schema), and `python -m redoland <command> -h` for
any command. All inference runs through `claude -p` (session auth, never the metered key).
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


def run_model(name: str) -> str:
    """The Claude model this run was created with (stored in its params; default Haiku
    for legacy runs created before model selection existed)."""
    import json
    import os
    try:
        meta = json.load(open(os.path.join(run_path(name), "meta.json")))
        return meta.get("params", {}).get("model") or "claude-haiku-4-5"
    except Exception:
        return "claude-haiku-4-5"


def _sim(name):
    from .sim import Simulation
    # think with the model the village was created with (Sonnet villages stay Sonnet).
    return Simulation.open(run_path(name), make_model_factory(run_model(name)))


def cmd_init(a):
    from .sim import Simulation
    model = a.model or Params().model
    # Default food model is the fountain F = max(food_base, round(food_floor_ratio*N)).
    # Passing --ratio or --preset means the user wants the CLASSIC proportional model
    # (F = round(ratio*N)), so we zero the fountain params to let `ratio` take effect.
    if a.preset is not None or a.ratio is not None:
        ratio = PRESETS.get(a.preset) if a.preset is not None else a.ratio
        params = Params(founders=a.founders, ratio=ratio, model=model,
                        food_base=0, food_floor_ratio=0.0)
        food_desc = f"classic ratio {ratio}"
    else:
        params = Params(founders=a.founders, model=model)   # fountain defaults
        food_desc = (f"fountain max({params.food_base}, round({params.food_floor_ratio}*N))")
    premise = (a.premise or "").strip()
    sim = Simulation.create(run_path(a.name), params, a.seed, make_model_factory(model),
                            premise=premise)
    print(f"created run '{a.name}' ({food_desc}, {a.founders} founders, seed {a.seed}, "
          f"model {model})" + (f"\n  stage: {premise}" if premise else ""))
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


def cmd_diff(a):
    d = _sim(a.name).diff(a.branch_a, a.year_a, a.branch_b, a.year_b)
    print(json.dumps(d, indent=2))


def cmd_serve(a):
    from .server import serve
    serve(os.path.abspath(a.runs_dir), port=a.port, host=a.host)


_OVERVIEW = """\
Redoland — a generational LLM-agent village simulation with git-backed worldlines.

MENTAL MODEL
  * A RUN is a village, stored as its own git repo under runs/<name>/.
  * Time advances in YEARS: food appears in a central pile, agents take/give/talk/
    fight/reproduce, then at year's end they eat, age, and may die.
  * Every single agent ACTION is its own git commit. A WORLDLINE is a git branch.
  * You FORK a new worldline from ANY action, optionally INJECT an event (a change +
    a public explanation), then RUN it forward and COMPARE branches. This is the
    counterfactual "what if X had happened?" loop.
  * All agent inference runs through the `claude -p` CLI on this session's auth —
    NEVER the metered ANTHROPIC_API_KEY.

TYPICAL WORKFLOW (what an operating AI does)
  1. redoland timeline <run> --branch main          # find a fork point (a commit)
  2. redoland fork <run> --at <commit> --name whatif # branch from that exact action
  3. redoland inject <run> --branch whatif \\
        --narrate "An earthquake strikes." \\
        --changes '{"kill":["a005"],"pile":{"set":0}}'   # an event (public to all)
  4. redoland run <run> --branch whatif --years 5    # run the new worldline forward
  5. redoland diff <run> main 10 whatif 15           # compare outcomes

INJECT --changes SCHEMA (all keys optional; everything is PUBLIC to all agents)
  {
    "agents": {"a003": {"food": 5, "hp": 10, "satiation": 1}},  # set per-agent values
    "kill":   ["a005"],                                          # agents who die
    "pile":   {"set": 0} | {"add": 20},                          # the plaza's food
    "spawn":  [{"sex":"female","age":25,"strength":60,"food":4}],# a newcomer arrives
    "params": {"ratio": 0.5},                                    # a rule change, going forward
    "actions":[{"actor":"a001","kind":"take","amount":3},        # FORCE specific agent moves:
               {"actor":"a001","kind":"talk","partners":["a002"]}]#  take/give/talk/attack/child
  }
  System-style changes (agents/kill/pile/spawn/params) apply first, then "actions".
  take/give resolve instantly; talk/attack/child play out via the model (claude -p).
  Agent ids (a001, a002, …) come from `redoland state` / `redoland timeline`.

NOTES
  * One writer per BRANCH. The web UI can run many branches at once (each in its own
    git worktree), but don't `run`/`inject` a branch from here while the UI is running
    that same branch. Different branches in parallel are fine.
  * Don't restart a run while a year is well underway — the uncommitted in-progress
    year is discarded. Committed years (git) are always safe.
  * The web UI (`redoland serve`) is read-only + a per-branch Start/Pause button;
    creating and forking worldlines is done here, via this CLI.
"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="redoland", description=_OVERVIEW,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    s = sub.add_parser("init", help="create a new village (run) and optionally run it",
                       description="Create a new run under runs/<name>/ with founders.")
    s.add_argument("name", help="run name (a folder under runs/)")
    s.add_argument("--seed", type=int, default=1, help="RNG seed for the founding (default 1)")
    s.add_argument("--founders", type=int, default=6, help="number of founding villagers (default 6)")
    s.add_argument("--ratio", type=float, default=None,
                   help="use the CLASSIC food model pile = round(ratio * population) at this "
                        "ratio (overrides the default fountain model). Unset = fountain default.")
    s.add_argument("--premise", default=None,
                   help="set the stage: a premise/backstory for this world (e.g. "
                        "'Survivors of a flood that drowned the old kingdom'). Shown to "
                        "every agent and announced in year 1.")
    s.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="named scarcity preset (classic ratio model; overrides --ratio)")
    s.add_argument("--model", default=None,
                   help="Claude model the village's agents think with, e.g. "
                        "claude-sonnet-4-6 or claude-haiku-4-5 (default %s). Stored with "
                        "the run, so it always thinks with this model." % Params().model)
    s.add_argument("--years", type=int, default=0, help="run this many years immediately (default 0)")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("run", help="run a branch forward N years (continues from where it is)",
                       description="Continue a worldline forward, committing after every action.")
    s.add_argument("name", help="run name")
    s.add_argument("--years", type=int, default=5, help="how many years to advance (default 5)")
    s.add_argument("--branch", default=None, help="branch to run (default: current/checked-out)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("log", help="list branches with summary metrics",
                       description="Show every worldline (branch) and its tip metrics.")
    s.add_argument("name", help="run name"); s.set_defaults(fn=cmd_log)

    s = sub.add_parser("timeline", help="per-action commit history — these are the fork points",
                       description="List recent action-commits (hash + the event). Pick one to fork from.")
    s.add_argument("name", help="run name")
    s.add_argument("--branch", default=None, help="branch to list (default: current)")
    s.add_argument("--n", type=int, default=40, help="how many recent commits to show (default 40)")
    s.set_defaults(fn=cmd_timeline)

    s = sub.add_parser("state", help="world snapshot — the living roster + each agent's stats",
                       description="Print who is alive and their food/HP/strength/age/etc. and "
                                   "their agent ids (a001…). Use ids in `inject`.")
    s.add_argument("name", help="run name")
    s.add_argument("--branch", default="main", help="branch to inspect (default main)")
    s.add_argument("--year", type=int, default=None, help="a completed year's end-state")
    s.add_argument("--at", default=None, help="a specific commit hash (any action)")
    s.set_defaults(fn=cmd_state)

    s = sub.add_parser("fork", help="branch a new worldline from any action (or a year)",
                       description="Create a new branch from a commit (--at, any action) or a "
                                   "year tag (--year). Then inject and/or run it.")
    s.add_argument("name", help="run name")
    s.add_argument("--at", default=None, help="commit hash to fork from (from `timeline`) — forks at ANY action")
    s.add_argument("--parent", default="main", help="parent branch when forking by --year (default main)")
    s.add_argument("--year", type=int, default=None, help="fork from the end of this completed year")
    s.add_argument("--name", dest="fork_name", default=None, help="name for the new worldline")
    s.set_defaults(fn=cmd_fork)

    s = sub.add_parser("inject", help="inject an event (a change + a public explanation) as a commit",
                       description="Apply a manipulation at the branch tip. The explanation AND the "
                                   "mechanical effects are recorded into every agent's memory. See the "
                                   "--changes schema in the top-level `redoland -h`.")
    s.add_argument("name", help="run name")
    s.add_argument("--branch", required=True, help="branch to inject into")
    s.add_argument("--narrate", default=None, help="the human explanation shown to ALL agents (the 'why')")
    s.add_argument("--changes", default=None,
                   help="JSON of the changes: keys agents/kill/pile/spawn/params (see `redoland -h`)")
    s.add_argument("--ratio", type=float, default=None, help="shortcut: change the food ratio")
    s.add_argument("--pile", type=int, default=None, help="shortcut: set the plaza's food")
    s.set_defaults(fn=cmd_inject)

    s = sub.add_parser("replay", help="checkout a branch and run it forward N years",
                       description="Same as `run --branch B`: checkout then advance N years.")
    s.add_argument("name", help="run name")
    s.add_argument("--branch", required=True, help="branch to replay")
    s.add_argument("--years", type=int, default=5, help="years to advance (default 5)")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("metrics", help="full JSON metrics for one branch/year",
                       description="Population, deaths, food Gini, trait/cognition means, etc.")
    s.add_argument("name", help="run name")
    s.add_argument("--branch", default="main", help="branch (default main)")
    s.add_argument("--year", type=int, required=True, help="completed year to report")
    s.set_defaults(fn=cmd_metrics)

    s = sub.add_parser("diff", help="compare two branch/year snapshots side by side",
                       description="Metrics for A@yearA vs B@yearB — the payoff of a counterfactual.")
    s.add_argument("name", help="run name")
    s.add_argument("branch_a", help="first branch"); s.add_argument("year_a", type=int, help="first year")
    s.add_argument("branch_b", help="second branch"); s.add_argument("year_b", type=int, help="second year")
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("serve", help="launch the read-only web UI (Start/Pause a branch)",
                       description="Web viewer over the whole runs/ dir: year tabs, transcripts, "
                                   "live stream, stat distributions, and a Start/Pause button.")
    s.add_argument("--runs-dir", default="runs", help="directory of runs to serve (default ./runs)")
    s.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    s.add_argument("--host", default="0.0.0.0",
                   help="bind address (default 0.0.0.0 — reachable from the host in a container)")
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    import sys
    from .env import load_dotenv
    from .model import ModelUnavailable
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except ModelUnavailable as e:
        # the model went dark (out of credits / auth / CLI) — halt with a clear message
        # rather than a traceback. Progress is safe: every completed step is committed.
        print(f"\nHALTED: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
