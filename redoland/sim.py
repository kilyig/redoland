"""Simulation orchestration (Concordia engine): ties engine + git store together
and implements the worldline operations (run / fork / inject / replay / diff).

This is the "redo" layer: a year is a commit; fork branches from a past year's
tag; inject mutates the restored world at a branch tip; replay runs forward —
producing a divergent timeline. Reuses the standalone design verbatim; only the
serialized payload (bodies + globals) and the model wiring differ.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from .core import Mortality, Params
from .engine import Engine
from .store import GitStore, safe_branch


class Simulation:
    def __init__(self, store: GitStore, mortality: Optional[Mortality] = None):
        self.store = store
        self.mortality = mortality or Mortality()

    # -- lifecycle -------------------------------------------------------- #
    @classmethod
    def create(cls, path, params: Params, seed: int, model_factory: Callable[[int], object],
               mortality=None, randomize_choices: bool = True, premise: str = "") -> "Simulation":
        store = GitStore(path, model_factory=model_factory, randomize_choices=randomize_choices)
        store.init_repo()
        sim = cls(store, mortality)
        eng = Engine.found(params, model_factory, seed, sim.mortality,
                           randomize_choices=randomize_choices, premise=premise)
        # The founding population is a pre-year-1 genesis snapshot: committed (so a
        # later run can load it) but NOT tagged as a year. The simulation starts at
        # year 1, which announces the founding in its own transcript.
        store.commit_year(eng.w, f"genesis: founding population "
                                 f"({params.founders} agents, seed {seed}, ratio {params.ratio})",
                          tag_year=False)
        return sim

    @classmethod
    def open(cls, path, model_factory: Callable[[int], object], mortality=None,
             randomize_choices: bool = True) -> "Simulation":
        store = GitStore(path, model_factory=model_factory, randomize_choices=randomize_choices)
        store.adopt_clone()          # a fresh clone (e.g. the sample-run submodule) -> local branches
        return cls(store, mortality)

    def engine(self) -> Engine:
        return Engine(self.store.load_world(), self.mortality)

    # -- run forward on the current branch -------------------------------- #
    def run(self, years: int = None, log=lambda *_: None, should_stop=lambda: False):
        """Step the simulation forward, committing after EVERY step (so a fork can be
        taken at any action). A year boundary additionally gets the human-friendly
        `<branch>-y<N>` tag. Runs `years` full years (None = until extinction or
        should_stop()); stops cleanly at a step boundary when should_stop() is true."""
        eng = self.engine()
        try:
            self.store.checkout_branch(eng.w.branch)
        except Exception:
            pass
        years_done = 0
        # Every action is its own commit, so the UI just polls the committed transcript
        # (incl. the in-progress year) — there is no separate live stream to maintain.
        while years is None or years_done < years:
            if should_stop() or not eng.w.living():
                break
            label = eng.step()
            is_boundary = (label == "year_end")
            last_ev = eng.w.year_events[-1] if eng.w.year_events else None
            desc = (last_ev.get("text") or last_ev.get("kind")) if last_ev else label
            self.store.commit_year(eng.w, f"y{eng.w.year} {label}: {desc[:80]}",
                                   tag_year=is_boundary)
            if is_boundary:
                years_done += 1
                log(eng.w.year, eng.w)
        return eng.w

    # -- fork (from a year tag OR any action-commit) ---------------------- #
    def fork(self, parent_branch: str, year: int, new_branch: str = None):
        nb = safe_branch(new_branch or f"{parent_branch}_f{year}")
        nb = self.store.checkout_fork(parent_branch, year, nb)
        world = self.store.load_world()
        world.branch = nb
        self.store.commit_year(world, f"fork {nb} from {parent_branch}@y{year}",
                               retag=True, tag_year=False)
        return nb

    def fork_at(self, commit: str, new_branch: str):
        """Fork a new worldline from ANY action-commit (a hash from `timeline`)."""
        nb = self.store.checkout_fork_at(commit, new_branch)
        world = self.store.load_world()
        world.branch = nb
        self.store.commit_year(world, f"fork {nb} from {commit[:10]}", tag_year=False)
        return nb

    # -- inject an event (the v1 manipulable params; public to all) ------- #
    def inject(self, branch: str, changes: dict = None, narrative: str = None,
               ratio: float = None, pile: int = None, narrate: str = None):
        """Apply an authored event at the branch tip as a new commit. `changes` may carry:
          * structured state changes (intervene.apply_changes): agents/kill/pile/spawn/params
          * "actions": a list of FORCED agent moves, e.g.
              [{"actor":"a001","kind":"take","amount":3},
               {"actor":"a001","kind":"talk","partners":["a002"]}]
            each runs through the real engine (take/give resolve instantly; talk/attack/child
            play out, which uses claude -p). State changes apply first, then actions.
        Everything is PUBLIC: an explanation event (with the mechanical effects) is recorded
        into every agent's memory, alongside whatever events the forced actions produce."""
        from .intervene import apply_changes
        changes = dict(changes or {})
        if ratio is not None:
            changes.setdefault("params", {})["ratio"] = ratio
        if pile is not None:
            changes["pile"] = {"set": pile}
        actions = changes.pop("actions", None) or []
        self.store.checkout_branch(branch)
        world = self.store.load_world()
        effects = apply_changes(world, changes)
        # an explanation event for the system-style changes (skip if it's only actions)
        text = None
        if narrative or narrate or effects:
            text = (narrative or narrate or "An event befalls the village.") + \
                   (" (" + "; ".join(effects) + ")" if effects else "")
            world.record("inject", "village", text, phase="inject")   # public → all agents
        # forced agent actions (scripted moves)
        forced = []
        if actions:
            eng = Engine(world, self.mortality)
            for act in actions:
                actor = world.agents.get(act.get("actor"))
                if actor and actor.alive:
                    eng._initiate(actor, {k: v for k, v in act.items() if k != "actor"})
                    forced.append(act)
        msg = (narrative or narrate or (f"{len(forced)} action(s)" if forced else "event"))
        self.store.commit_year(world, f"inject @y{world.year}: {str(msg)[:50]}", tag_year=False)
        return {"effects": effects, "actions": forced, "text": text}

    # -- replay forward --------------------------------------------------- #
    def replay(self, branch: str, years: int, log=lambda *_: None):
        self.store.checkout_branch(branch)
        return self.run(years, log)

    # -- compare ---------------------------------------------------------- #
    def diff(self, branch_a: str, year_a: int, branch_b: str, year_b: int):
        from .metrics import snapshot_metrics
        wa = self.store.load_world(self.store.tag(branch_a, year_a))
        wb = self.store.load_world(self.store.tag(branch_b, year_b))
        return {
            f"{branch_a}@y{year_a}": snapshot_metrics(wa),
            f"{branch_b}@y{year_b}": snapshot_metrics(wb),
        }
