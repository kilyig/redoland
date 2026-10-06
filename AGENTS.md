# Redoland — operator guide for AIs

You are operating **Redoland**, a generational LLM-agent village simulation with
git-backed worldline branching. This file is everything you need to start working
immediately. Run all commands from the repo root with the project venv:
`.venv/bin/python -m redoland <command>` (the venv has the one dependency,
Concordia). `python -m redoland -h` prints the same mental model + the inject
schema; `python -m redoland <command> -h` documents any command.

Setup (once):
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
plus the `claude` CLI installed and logged in — every agent decision is a `claude -p`
call on that login (set `REDOLAND_CLAUDE_BIN` in `.env.local` if it is not on PATH;
`.env.local` is auto-loaded and git-ignored). **Cost:** a simulated year is many
`claude -p` calls per agent, so start with `--years 1` on a small village.

## What it is

A village of LLM-driven people under scarcity. Each **year**: food appears in a
central pile; agents **take** from it, **give** food, **talk** privately, **attack**
to raid food, or have a **child** with a partner; at year's end they **eat**, **age**,
and may **die** (starvation, combat, or old age). Personalities (Big Five), strength,
and two cognition dials — **intelligence** (the agent's thinking-token budget) and
**memory** (its context length) — are heritable via Gaussian crossover (parents'
mean + noise); sex is a 50/50 draw.

The point of the project is **counterfactual experiments**: rewind to any moment,
change something, and watch how history diverges.

## The mental model (this is the whole thing)

- A **run** is a village = its own git repo under `runs/<name>/`.
- **Every single agent action is its own git commit.** A **worldline** is a git **branch**.
- The loop you run for the human:
  1. **find** a moment — `timeline` lists per-action commits (these are the fork points);
  2. **fork** a new worldline from that exact action — `fork --at <commit>`;
  3. optionally **inject** an event (a state change + a public explanation) — `inject`;
  4. **run** the new worldline forward — `run --branch <b> --years N`;
  5. **compare** — `diff`.
- A fork with **no** injected event still diverges (the LLM is stochastic). That
  natural variation is the *control* — re-run a fork N times to get the outcome
  distribution, then see whether an injected event moves it beyond that spread.

## Commands (see `-h` on each for full args)

| Command | What it does |
|---|---|
| `init <run> [--founders 6] [--seed 1] [--premise "…"] [--model M] [--ratio R \| --preset P] [--years N]` | create a new village (**set the stage** with a premise; pick the **model** it thinks with — default `claude-haiku-4-5`, stored with the run). `--ratio` / `--preset {crisis,tense,stable,abundant}` switch to the classic proportional food model (see below) |
| `run <run> [--branch B] [--years N]` | advance a branch N years (default 5; 1 commit per action). `--branch` defaults to the checked-out branch |
| `replay <run> --branch B [--years N]` | same as `run --branch B` (checkout, then advance; default 5) |
| `timeline <run> [--branch B] [--n 40]` | per-action commit history → **fork points** (shows agent **names**, not ids; `--branch` defaults to the checked-out branch) |
| `state <run> --branch B (--year Y \| --at <commit>)` | living roster + each agent's stats + **ids**. **`--branch` only matters together with `--year`/`--at`**; with neither, `state` prints the main working tree's checked-out snapshot whatever `--branch` says |
| `fork <run> --at <commit> --name <new>` | new worldline from **any** action; or `--year Y [--parent P]` from a completed year's end (parent default `main`) |
| `inject <run> --branch B --narrate "…" --changes '<json>'` | inject an event (public to all). Shortcuts: `--pile N` (set the plaza's food), `--ratio R` (set `params.ratio`) |
| `diff <run> A yA B yB` | compare two branch/year snapshots |
| `log <run>` / `metrics <run> --branch B --year Y` | branch summaries / full JSON metrics |
| `serve [--port 8000] [--host 127.0.0.1]` | read-only web UI + per-branch Start/Pause (**many branches run at once**); loopback only by default — `--host 0.0.0.0` exposes Start (paid inference, no auth) to anyone who can reach the port, needed e.g. inside a container viewed from the host |

`<run>` is a bare name (→ `runs/<name>`), a `runs/x` path, or an absolute/`./` path.
Agent ids (`a001`, `a002`, …) come from `state` (`timeline` prints names only).

If the model goes dark (CLI missing, expired login, persistent timeout) the run
**halts** with `HALTED: …` on stderr and exit code 2; nothing is lost, every completed
step is already committed. Usage/rate-limit errors are not failures: the call is
retried indefinitely with ~90 s sleeps, so a run simply pauses until the quota resets.

## Injecting an event — `--changes` JSON

Everything is **PUBLIC**: the `--narrate` explanation **and** the mechanical effects
are written into every agent's memory as one event. v1 manipulable parameters:

```json
{
  "agents": {"a003": {"food": 5, "hp": 10, "satiation": 1}},
  "kill": ["a005"],
  "pile": {"set": 0},
  "spawn": [{"sex": "female", "age": 25, "strength": 60, "food": 4}],
  "params": {"food_base": 8, "food_floor_ratio": 0.6},
  "actions": [
    {"actor": "a001", "kind": "take", "amount": 3},
    {"actor": "a001", "kind": "give", "target": "a002", "amount": 2},
    {"actor": "a001", "kind": "talk", "partners": ["a002", "a003"]},
    {"actor": "a004", "kind": "attack", "target": "a005", "demand": 2},
    {"actor": "a002", "kind": "child", "partner": "a003", "my_share": 2}
  ]
}
```
All keys optional. `agents` sets per-agent values (`satiation` is the 0–3 hunger bar;
`hp` 0 kills). `kill` lists agents who die. `pile` is the plaza's food: `{"set": N}`,
`{"add": N}` or a bare number. `spawn` adds newcomers (unset fields are randomised).
`params` sets any `Params` field by name, going forward. `actions` **forces** specific
agent moves, in order: `take` (`amount`), `give` (`target`, `amount`), `talk`
(`partners`, a list), `attack` (`target`, `demand` — default: all the target's food),
`child` (`partner`, `my_share` of the 3-food cost). Forced moves bypass the yearly
action/speaking budgets.

**Food supply.** The default is the *fountain* model: each year the pile is
`max(food_base, round(food_floor_ratio × living))` = `max(15, round(0.9 × N))`. The
classic proportional model `round(ratio × N)` is used **only** when `food_base` and
`food_floor_ratio` are both 0 — i.e. runs created with `init --ratio`/`--preset`. So
`"params": {"ratio": 0.5}` (or `inject --ratio`) does nothing on a default run; change
`food_base`/`food_floor_ratio` instead, or zero both to make `ratio` take effect.

**Pile timing.** The pile is refilled by the next year's setup step, so a `pile`
change injected while the branch sits at a year boundary (`state` shows
`phase year_start` — right after `year_end`, a `--year` fork, or genesis) is overwritten
before anyone can take from it. Inject `pile` mid-year (phase `scramble`), or use
`params` for a lasting change.

System-style changes apply first, then `actions`. So you can author **any** event:
a system event ("an earthquake empties the pile") *and/or* a specific agent move
("a001 takes 3 from the pile", "a001 gathers a002 to talk"). `take`/`give` resolve
instantly; `talk`/`attack`/`child` play out through the model (so they use `claude -p`).

### Worked example
```bash
PY=.venv/bin/python
$PY -m redoland timeline myrun --branch main          # -> e.g. commit 3fd3e37 "Eron attacks Kesh"
$PY -m redoland fork     myrun --at 3fd3e37 --name elephant
$PY -m redoland state    myrun --branch elephant --at 3fd3e37   # -> find Quill's id, say a008
$PY -m redoland inject   myrun --branch elephant \
      --narrate "A rogue elephant tramples through the village." \
      --changes '{"kill":["a008"],"agents":{"a003":{"hp":15}}}'
$PY -m redoland run      myrun --branch elephant --years 5
$PY -m redoland diff     myrun main 12 elephant 12
```

## Hard rules

- **Concurrency is via git worktrees, one writer per branch.** The web UI can run
  **multiple branches at once** — each running branch gets its own linked worktree
  under `runs/.worktrees/<run>/<branch>/`, so commits to different branches never
  clash (refs are shared, so reads from the main repo see them). The rule is one
  writer **per branch**: don't `run`/`inject` a branch from the CLI while the UI is
  running that same branch (or start the same branch twice). Different branches —
  same world or not — are fine in parallel.
- **A UI-started branch stays claimed by its worktree, even after Pause.** Nothing
  removes `runs/.worktrees/<run>/<branch>/`, and git refuses to check out a branch
  that a worktree holds, so CLI `run --branch B` / `replay` / `inject` on that branch
  fail with `fatal: 'B' is already used by worktree at …`. Worse, `run` with **no**
  `--branch` does not fail: if the UI started the branch that was checked out in the
  main tree (usually `main`), that tree is left detached, so a bare `run` loads the
  stale detached snapshot and commits on a detached HEAD (the checkout failure is
  swallowed). Either fork a new branch from the tip (`fork --at <tip-commit>`) and
  work on that, or release the branch first with
  `git -C runs/<run> worktree remove ../.worktrees/<run>/<branch>` (only when the UI
  is not running it). Reads (`timeline`, `state --at/--year`, `log`, `diff`) are
  unaffected.
- **Stopping a run loses nothing.** Every step is a commit and the run cursor is part
  of the world state, so the next `run` resumes mid-year from the last committed step
  (completed years are additionally tagged `<branch>-y<N>`). Pause/Ctrl-C freely, e.g.
  to pick up a code change.
- **Code changes are PR-first:** feature branch → PR → plain merge (no squash). Never
  force-push or rewrite `main`.

## Where things are (to dive into the code)

`redoland/`: `engine.py` (the year loop: `step()` + the scramble), `decide.py`
(agent decisions via `claude -p`), `model.py` (`ClaudeCLIModel`; reads
`REDOLAND_CLAUDE_BIN`), `components.py` (the prompt every agent sees), `villager.py`
(builds an agent's Concordia mind), `world.py` (the World container: bodies + minds +
cursor), `intervene.py` (`apply_changes` — the inject params), `store.py` (git:
per-action commits, fork, timeline, worktrees), `sim.py` (run/fork/inject/replay/diff),
`server.py` (the read-only UI server; the page is `static/redoland.html` at the repo
root), `cli.py` (these commands), `core.py` (Agent, Params, PRESETS, RNG, mortality),
`metrics.py` (snapshot metrics / distributions), `env.py` (`.env.local` loader).
`data/ssa_4c6.json` is the SSA life table behind old-age mortality. Tests:
`python tests/test_smoke.py` (no network, deterministic stub). Runs live in `runs/`
(git-ignored, local-only) —
except the **sample run**, which ships as a git submodule at `runs/sample_run`
(github.com/kilyig/redoland-sample-run). Clone with `git clone --recurse-submodules`,
or run `git submodule update --init` in an existing clone; it then appears in the UI
and the CLI (`log sample_run`) like any other run.
