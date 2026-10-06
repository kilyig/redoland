# Redoland — operator guide for AIs

You are operating **Redoland**, a generational LLM-agent village simulation with
git-backed worldline branching. This file is everything you need to start working
immediately. Run all commands from the repo root with the project venv:
`/home/dev/redoland/.venv/bin/python -m redoland <command>` (the venv has the one
dependency, Concordia). `python -m redoland -h` prints the same mental model + the
inject schema; `python -m redoland <command> -h` documents any command.

## What it is

A village of LLM-driven people under scarcity. Each **year**: food appears in a
central pile; agents **take** from it, **give** food, **talk** privately, **attack**
to raid food, or have a **child** with a partner; at year's end they **eat**, **age**,
and may **die** (starvation, combat, or old age). Personalities (Big Five), strength,
sex, and two cognition dials — **intelligence** (the agent's thinking-token budget)
and **memory** (its context length) — are heritable via Gaussian crossover.

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
| `init <run> --founders 6 --seed 1 [--premise "…"] [--model claude-sonnet-4-6] [--years N]` | create a new village (**set the stage** with a premise; pick the **model** it thinks with — stored with the run) |
| `run <run> --branch B --years N` | advance a branch N years (1 commit per action) |
| `timeline <run> --branch B [--n 40]` | per-action commit history → **fork points** |
| `state <run> --branch B [--year Y \| --at <commit>]` | living roster + each agent's stats + **ids** |
| `fork <run> --at <commit> --name <new>` | new worldline from **any** action (or `--year Y`) |
| `inject <run> --branch B --narrate "…" --changes '<json>'` | inject an event (public to all) |
| `diff <run> A yA B yB` | compare two branch/year snapshots |
| `log <run>` / `metrics <run> --branch B --year Y` | branch summaries / full JSON metrics |
| `serve [--port 8000] [--host 127.0.0.1]` | read-only web UI + per-branch Start/Pause (**many branches run at once**); loopback only by default — `--host 0.0.0.0` exposes Start (paid inference, no auth) to anyone who can reach the port, needed e.g. inside a container viewed from the host |

Agent ids (`a001`, `a002`, …) come from `state` and `timeline`.

## Injecting an event — `--changes` JSON

Everything is **PUBLIC**: the `--narrate` explanation **and** the mechanical effects
are written into every agent's memory as one event. v1 manipulable parameters:

```json
{
  "agents": {"a003": {"food": 5, "hp": 10, "satiation": 1}},   // set per-agent values
  "kill":   ["a005"],                                          // agents who die
  "pile":   {"set": 0},                                        // or {"add": 20} — the plaza's food
  "spawn":  [{"sex": "female", "age": 25, "strength": 60, "food": 4}],  // a newcomer arrives
  "params": {"ratio": 0.5},                                    // a rule change, going forward
  "actions":[{"actor":"a001","kind":"take","amount":3},        // FORCE specific agent moves —
             {"actor":"a001","kind":"talk","partners":["a002"]}] // take/give/talk/attack/child
}
```
(`satiation` is the 0–3 hunger bar; `ratio` sets pile = round(ratio × population).)
System-style changes apply first, then `actions`. So you can author **any** event:
a system event ("an earthquake empties the pile") *and/or* a specific agent move
("a001 takes 3 from the pile", "a001 gathers a002 to talk"). `take`/`give` resolve
instantly; `talk`/`attack`/`child` play out through the model (so they use `claude -p`).

### Worked example
```bash
PY=/home/dev/redoland/.venv/bin/python
$PY -m redoland timeline myrun --branch main          # -> e.g. commit 3fd3e37 "Eron attacks Kesh"
$PY -m redoland fork     myrun --at 3fd3e37 --name elephant
$PY -m redoland state    myrun --branch elephant       # -> find Quill's id, say a008
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
- **Don't interrupt an in-progress year.** Committed years (git tags `<branch>-y<N>`)
  are always safe, but the *uncommitted in-progress* year is discarded if you stop a
  run. Before restarting a run to pick up a code change, check how far the current
  year has progressed (`timeline` / the live feed) and only restart if it just began.
- **Code changes are PR-first:** feature branch → PR → plain merge (no squash). Never
  force-push or rewrite `main`.

## Where things are (to dive into the code)

`redoland/`: `engine.py` (the year loop: `step()` + the scramble), `decide.py`
(agent decisions via `claude -p`), `model.py` (`ClaudeCLIModel`), `components.py`
(the prompt every agent sees), `intervene.py` (`apply_changes` — the inject params),
`store.py` (git: per-action commits, fork, timeline), `sim.py` (run/fork/inject/diff),
`server.py` + `static/redoland.html` (the read-only UI), `cli.py` (these commands),
`core.py` (Agent, Params, RNG, mortality). Tests: `python tests/test_smoke.py`
(no network, deterministic stub). Runs live in `runs/` (git-ignored, local-only) —
except the **sample run**, which ships as a git submodule at `runs/sample_run`
(github.com/kilyig/redoland-sample-run). Clone with `git clone --recurse-submodules`,
or run `git submodule update --init` in an existing clone; it then appears in the UI
and the CLI (`log sample_run`) like any other run.
