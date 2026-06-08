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
    def run(self, years: int, log=lambda *_: None):
        eng = self.engine()
        try:
            self.store.checkout_branch(eng.w.branch)
        except Exception:
            pass
        live = self._open_live(eng.w)
        eng.w.event_sink = live["sink"]
        try:
            for _ in range(years):
                eng.run_year()
                self.store.commit_year(eng.w, f"{eng.w.branch} year {eng.w.year}")
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

    # -- fork ------------------------------------------------------------- #
    def fork(self, parent_branch: str, year: int, new_branch: str = None):
        nb = safe_branch(new_branch or f"{parent_branch}_f{year}")
        nb = self.store.checkout_fork(parent_branch, year, nb)
        world = self.store.load_world()
        world.branch = nb
        self.store.commit_year(world, f"fork {nb} from {parent_branch}@y{year}", retag=True)
        return nb

    # -- inject (environmental / state change at the branch tip) ---------- #
    def inject(self, branch: str, ratio: float = None, pile: int = None, narrate: str = None):
        self.store.checkout_branch(branch)
        world = self.store.load_world()
        changes = []
        if ratio is not None:
            world.params.ratio = ratio
            changes.append(f"ratio->{ratio}")
        if pile is not None:
            world.pile = pile
            changes.append(f"pile->{pile}")
        if narrate:
            world.record("narrate", "village", f"[INJECTED] {narrate}")
            changes.append("narrate")
        self.store.commit_year(world, f"inject @{world.year}: {', '.join(changes)}", retag=True)
        return changes

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
