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
               mortality=None, randomize_choices: bool = True) -> "Simulation":
        store = GitStore(path, model_factory=model_factory, randomize_choices=randomize_choices)
        store.init_repo()
        sim = cls(store, mortality)
        eng = Engine.found(params, model_factory, seed, sim.mortality,
                           randomize_choices=randomize_choices)
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
        return cls(GitStore(path, model_factory=model_factory,
                            randomize_choices=randomize_choices), mortality)

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
        live = self._open_live(eng.w)
        eng.w.event_sink = live["sink"]
        years_done = 0
        try:
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
        finally:
            eng.w.event_sink = None
            live["close"]()
        return eng.w

    def _open_live(self, world):
        path = os.path.join(self.store.path, "live.jsonl")
        try:
            with open(path, "w") as fh:
                fh.write(json.dumps({
                    "eid": "live", "year": world.year, "phase": "meta",
                    "kind": "run_start", "speaker": "village", "who": "the village",
                    "audience": "public", "payload": {},
                    "text": f"live run — branch {world.branch}, continuing after year {world.year}",
                }) + "\n")
            fh = open(path, "a")
        except OSError:
            return {"sink": lambda ev: None, "close": lambda: None}

        def sink(ev):
            try:
                fh.write(json.dumps(ev) + "\n")
                fh.flush()
            except Exception:
                pass

        def close():
            try:
                fh.close()
            except Exception:
                pass
        return {"sink": sink, "close": close}

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
        """Apply an authored event at the branch tip as a new commit. `changes` is the
        structured spec (see intervene.apply_changes). Legacy ratio/pile/narrate kwargs
        are still accepted. The explanation + the mechanical effects are recorded as one
        PUBLIC event in every agent's memory."""
        from .intervene import apply_changes
        changes = dict(changes or {})
        if ratio is not None:
            changes.setdefault("params", {})["ratio"] = ratio
        if pile is not None:
            changes["pile"] = {"set": pile}
        narrative = narrative or narrate or "An event befalls the village."
        self.store.checkout_branch(branch)
        world = self.store.load_world()
        effects = apply_changes(world, changes)
        text = narrative + (" (" + "; ".join(effects) + ")" if effects else "")
        world.record("inject", "village", text, phase="inject")   # public → all agents
        self.store.commit_year(world, f"inject @y{world.year}: {narrative[:50]}", tag_year=False)
        return {"effects": effects, "text": text}

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
