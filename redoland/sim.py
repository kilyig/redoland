"""Simulation orchestration: ties engine + git store + backend together and
implements the worldline operations (run / fork / inject / edit / replay / diff).

Edit-and-replay semantics (MVP_PLAN.md §7.2): worldline operations happen at a
*year boundary*. You fork at the checkpoint after year k, optionally edit a past
utterance (which patches the recorded transcript AND the carried memory every
witness holds) or inject an event (drought, etc.), then replay year k+1 onward —
everything downstream is regenerated. With the FakeBackend, downstream behaviour
is a function of world state, so INJECTIONS visibly diverge offline; an
utterance EDIT changes the record + carried memory and is wired for replay, but
its *behavioural* divergence shows with the Anthropic backend (the FakeBackend
does not read memory text). See README.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .core import Mortality, Params
from .engine import Engine, World
from .store import GitStore, safe_branch


class Simulation:
    def __init__(self, store: GitStore, backend, mortality: Optional[Mortality] = None):
        self.store = store
        self.backend = backend
        self.mortality = mortality or Mortality()

    # -- lifecycle -------------------------------------------------------- #
    @classmethod
    def create(cls, path, params: Params, seed: int, backend,
               mortality=None) -> "Simulation":
        store = GitStore(path)
        store.init_repo()
        sim = cls(store, backend, mortality)
        eng = Engine.found(params, backend, seed, sim.mortality)
        store.commit_year(eng.w, f"found: {params.founders} agents, seed {seed}, "
                                 f"ratio {params.ratio}")
        return sim

    @classmethod
    def open(cls, path, backend, mortality=None) -> "Simulation":
        return cls(GitStore(path), backend, mortality)

    def engine(self) -> Engine:
        return Engine(self.store.load_world(), self.backend, self.mortality)

    # -- run forward on the current branch -------------------------------- #
    def run(self, years: int, log=lambda *_: None):
        eng = self.engine()
        # make sure git is on the right branch for this world
        try:
            self.store.checkout_branch(eng.w.branch)
        except Exception:
            pass
        for _ in range(years):
            eng.run_year()
            self.store.commit_year(eng.w, f"{eng.w.branch} year {eng.w.year}")
            log(eng.w.year, eng.w)
        return eng.w

    # -- fork ------------------------------------------------------------- #
    def fork(self, parent_branch: str, year: int, new_branch: str = None):
        nb = safe_branch(new_branch or f"{parent_branch}_f{year}")
        nb = self.store.checkout_fork(parent_branch, year, nb)
        world = self.store.load_world()
        world.branch = nb
        # re-stamp the boundary commit so the world knows its new branch
        self.store.commit_year(world, f"fork {nb} from {parent_branch}@y{year}",
                               retag=True)
        return nb

    # -- inject (environmental / state change at the branch tip) ---------- #
    def inject(self, branch: str, ratio: float = None, pile: int = None,
               narrate: str = None):
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
        self.store.commit_year(world, f"inject @{world.year}: {', '.join(changes)}",
                               retag=True)
        return changes

    # -- edit a recorded utterance --------------------------------------- #
    def edit_utterance(self, branch: str, eid: str, new_text: str):
        """Patch a past utterance in the transcript AND in every agent's carried
        memory, at the branch tip, then commit. Replay forward to see divergence
        (behavioural divergence requires the Anthropic backend)."""
        self.store.checkout_branch(branch)
        world = self.store.load_world()
        patched = 0
        # patch carried subjective memory on every agent
        for ag in world.agents.values():
            for e in ag.memory_raw:
                if e.get("eid") == eid:
                    e["text"] = new_text
                    patched += 1
        # patch the historical transcript files in the working tree
        edir = os.path.join(self.store.path, "events")
        if os.path.isdir(edir):
            for fn in os.listdir(edir):
                fp = os.path.join(edir, fn)
                lines = open(fp).read().splitlines()
                changed = False
                out = []
                for ln in lines:
                    if not ln.strip():
                        continue
                    ev = json.loads(ln)
                    if ev.get("eid") == eid:
                        ev["text"] = new_text
                        ev["edited"] = True
                        changed = True
                    out.append(json.dumps(ev))
                if changed:
                    with open(fp, "w") as fh:
                        fh.write("\n".join(out) + "\n")
                    patched += 1
        self.store.commit_year(world, f"edit {eid} @{world.year}", retag=True)
        return patched

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
