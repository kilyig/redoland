"""The simulation engine: world state plus the yearly loop.

Year = setup -> meeting (vote-distribute the pile) -> reproduction -> year-end
(eat / age / SSA mortality) -> compaction. See MVP_PLAN.md §6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .core import (Agent, Mortality, Params, RNG, BIG5, approx_tokens, make_name)


@dataclass
class Motion:
    id: str
    proposer: str
    amount: int
    recipient: str
    yes: set = field(default_factory=set)
    open: bool = True

    def to_dict(self):
        return {"id": self.id, "proposer": self.proposer, "amount": self.amount,
                "recipient": self.recipient, "yes": sorted(self.yes), "open": self.open}


class World:
    def __init__(self, params: Params, rng: RNG, branch: str = "main"):
        self.params = params
        self.rng = rng
        self.branch = branch
        self.year = 0
        self.pile = 0
        self.agents: dict[str, Agent] = {}
        self.motions: list[Motion] = []
        self.used_names: set = set()
        self.next_eid = 0
        self.next_aid = 0
        self.next_mid = 0
        self.year_events: list[dict] = []   # events for the current year (reset each year)

    # -- queries ---------------------------------------------------------- #
    def living(self) -> list[Agent]:
        return [self.agents[i] for i in sorted(self.agents) if self.agents[i].alive]

    def threshold(self) -> int:
        return math.ceil(len(self.living()) / 2)

    # -- id helpers ------------------------------------------------------- #
    def new_aid(self) -> str:
        self.next_aid += 1
        return f"a{self.next_aid:03d}"

    def new_mid(self) -> str:
        self.next_mid += 1
        return f"m{self.year}_{self.next_mid}"

    # -- events / memory -------------------------------------------------- #
    def record(self, kind, who, text="", audience="public", payload=None, phase="meeting"):
        self.next_eid += 1
        eid = f"e{self.next_eid:06d}"
        who_name = self.agents[who].name if who in self.agents else "the village"
        ev = {"eid": eid, "year": self.year, "phase": phase, "kind": kind,
              "speaker": who, "who": who_name, "audience": audience,
              "text": text, "payload": payload or {}}
        self.year_events.append(ev)
        # distribute into witnesses' subjective memory
        mem_entry = {"eid": eid, "year": self.year, "kind": kind,
                     "who": who_name, "text": text}
        for a in self._witnesses(who, audience):
            a.memory_raw.append(dict(mem_entry))
        return ev

    def _witnesses(self, speaker, audience):
        if audience == "public":
            return self.living()
        ids = set()
        if isinstance(audience, list):
            ids.update(audience)
        if speaker in self.agents:
            ids.add(speaker)
        return [self.agents[i] for i in ids if i in self.agents]


class Engine:
    def __init__(self, world: World, backend, mortality: Optional[Mortality] = None):
        self.w = world
        self.backend = backend
        self.mortality = mortality or Mortality()

    # ===================================================================== #
    # Founding                                                              #
    # ===================================================================== #
    @classmethod
    def found(cls, params: Params, backend, seed: int, mortality=None) -> "Engine":
        rng = RNG(seed=seed)
        world = World(params, rng)
        eng = cls(world, backend, mortality)
        for _ in range(params.founders):
            sex = "male" if rng.chance(0.5) else "female"
            name = make_name(sex, rng, world.used_names)
            world.used_names.add(name)
            traits = {t: rng.randint(15, 85) for t in BIG5}
            ag = Agent(
                id=world.new_aid(), name=name, sex=sex, traits=traits,
                intelligence_tokens=rng.randint(params.int_min, params.int_max),
                memory_tokens=rng.randint(params.mem_min, params.mem_max),
                age=rng.randint(params.founder_age_min, params.founder_age_max),
                health=params.start_health, food=params.start_food, birth_year=0,
            )
            world.agents[ag.id] = ag
        return eng

    # ===================================================================== #
    # One year                                                              #
    # ===================================================================== #
    def run_year(self):
        self.w.year += 1
        self.w.year_events = []
        self._setup()
        self._meeting()
        self._reproduction()
        self._year_end()
        self._compaction()
        return self.w.year_events

    # -- setup ------------------------------------------------------------ #
    def _setup(self):
        w = self.w
        n = len(w.living())
        f = round(n * w.params.ratio)
        w.pile = f
        w.motions = []
        for a in w.living():
            a.bore_this_year = False
            a.repro_done_year = False
        w.record("narrate", "village",
                 f"Year {w.year}: {f} food appears in the pile for {n} people.",
                 phase="setup")

    # -- meeting ---------------------------------------------------------- #
    def _meeting(self):
        w = self.w
        slots = w.params.meeting_slots_per_agent * len(w.living())
        while slots > 0 and w.pile > 0:
            raised = []
            for a in w.living():
                ok, _ = self.backend.handraise(w, a, w.rng)
                if ok:
                    raised.append(a)
            if not raised:
                break
            speaker = w.rng.choice(raised)
            action = self.backend.turn(w, speaker, w.rng)
            self._apply_action(speaker, action)
            self._resolve_motions()
            slots -= 1
        if w.pile > 0:
            w.record("narrate", "village",
                     f"{w.pile} food went unclaimed and spoiled.", phase="meeting")
            w.pile = 0
        for m in w.motions:
            m.open = False

    def _apply_action(self, speaker, action):
        w = self.w
        kind = action.get("kind", "pass")
        text = action.get("text", "")
        aud = action.get("audience", "public")
        if kind == "think":
            w.record("think", speaker.id, text, audience=[speaker.id])
        elif kind == "say":
            w.record("say", speaker.id, text, audience=aud)
        elif kind == "propose":
            rec = action.get("recipient")
            amt = int(action.get("amount", 1))
            if w.pile > 0 and rec in w.agents and w.agents[rec].alive:
                amt = max(1, min(amt, w.pile))
                m = Motion(id=w.new_mid(), proposer=speaker.id, amount=amt,
                           recipient=rec, yes={speaker.id})
                w.motions.append(m)
                rn = w.agents[rec].name
                w.record("propose", speaker.id,
                         text or f"I propose {amt} food go to {rn}.",
                         payload={"motion": m.id, "amount": amt, "recipient": rec})
        elif kind == "vote":
            mid = action.get("motion")
            for m in w.motions:
                if m.id == mid and m.open:
                    m.yes.add(speaker.id)
                    w.record("vote", speaker.id, text or "I vote yes.",
                             payload={"motion": mid})
                    break
        elif kind == "give":
            tgt = action.get("target")
            amt = int(action.get("amount", 1))
            if tgt in w.agents and w.agents[tgt].alive and amt >= 1 and speaker.food >= amt:
                speaker.food -= amt
                w.agents[tgt].food += amt
                w.record("give", speaker.id,
                         text or f"I give {amt} food to {w.agents[tgt].name}.",
                         payload={"target": tgt, "amount": amt})
        else:
            w.record("pass", speaker.id, "", audience=[speaker.id])

    def _resolve_motions(self):
        w = self.w
        thr = w.threshold()
        for m in w.motions:
            if m.open and len(m.yes) >= thr and w.pile >= m.amount:
                w.pile -= m.amount
                w.agents[m.recipient].food += m.amount
                m.open = False
                w.record("narrate", "village",
                         f"The village agrees: {m.amount} food to "
                         f"{w.agents[m.recipient].name} ({len(m.yes)} in favour).",
                         payload={"motion": m.id})

    # -- reproduction ----------------------------------------------------- #
    def _reproduction(self):
        w = self.w
        for _ in range(w.params.repro_rounds):
            eligible = [a for a in w.living() if not a.repro_done_year]
            order = list(eligible)
            w.rng.shuffle(order)
            offers = []
            for a in order:
                if a.repro_done_year:
                    continue
                cands = [c for c in eligible if c.id != a.id and not c.repro_done_year]
                off = self.backend.offer(w, a, cands, w.rng)
                if off:
                    off["proposer"] = a.id
                    offers.append(off)
            for off in offers:
                self._try_birth(off)

    def _try_birth(self, off):
        w = self.w
        a = w.agents.get(off["proposer"])
        p = w.agents.get(off.get("partner"))
        if not a or not p or not a.alive or not p.alive:
            return
        if a.repro_done_year or p.repro_done_year:
            return
        if a.sex == p.sex or a.is_close_kin(p.id):
            return
        my_share = int(off.get("my_share", 1))
        partner_share = w.params.child_cost - my_share
        if not (0 <= my_share <= w.params.child_cost):
            return
        if a.food < my_share or p.food < partner_share:
            return
        female = a if a.sex == "female" else p
        if female.bore_this_year:
            return
        if not self.backend.respond(w, p, off, w.rng):
            return
        self._birth(a, my_share, p, partner_share)

    def _birth(self, a, a_share, p, p_share):
        w = self.w
        a.food -= a_share
        p.food -= p_share
        mother = a if a.sex == "female" else p
        father = p if a.sex == "female" else a
        child = self._crossover(mother, father)
        # kinship
        child.parents = [mother.id, father.id]
        existing = [c for c in (mother.children + father.children)]
        child.siblings = sorted(set(existing))
        for sid in child.siblings:
            sib = w.agents.get(sid)
            if sib and child.id not in sib.siblings:
                sib.siblings.append(child.id)
        mother.children.append(child.id)
        father.children.append(child.id)
        # parent notes (cultural inheritance)
        child.mother_note = self.backend.note(w, mother, child, w.rng)
        child.father_note = self.backend.note(w, father, child, w.rng)
        mother.bore_this_year = True
        mother.repro_done_year = True
        father.repro_done_year = True
        w.agents[child.id] = child
        w.record("birth", "village",
                 f"{child.name} ({child.sex}) is born to {mother.name} and "
                 f"{father.name}, age {child.age}.",
                 payload={"child": child.id, "mother": mother.id, "father": father.id},
                 phase="repro")

    def _crossover(self, mother, father):
        w = self.w
        rng = w.rng
        p = w.params
        traits = {}
        for t in BIG5:
            v = round((mother.trait(t) + father.trait(t)) / 2 + rng.gauss(0, p.big5_sigma))
            traits[t] = int(max(0, min(100, v)))
        intel = round((mother.intelligence_tokens + father.intelligence_tokens) / 2
                      + rng.gauss(0, p.int_sigma))
        intel = int(max(p.int_min, min(p.int_max, intel)))
        mem = round((mother.memory_tokens + father.memory_tokens) / 2
                    + rng.gauss(0, p.mem_sigma))
        mem = int(max(p.mem_min, min(p.mem_max, mem)))
        sex = "male" if rng.chance(0.5) else "female"
        name = make_name(sex, rng, w.used_names)
        w.used_names.add(name)
        return Agent(id=w.new_aid(), name=name, sex=sex, traits=traits,
                     intelligence_tokens=intel, memory_tokens=mem,
                     age=p.child_age, health=p.child_health, food=0,
                     birth_year=w.year)

    # -- year end --------------------------------------------------------- #
    def _year_end(self):
        w = self.w
        # newborns of this year do not eat/age/roll this year
        participants = [a for a in w.living() if a.birth_year != w.year]
        # eat
        for a in participants:
            eat = self.backend.eat_choice(w, a, w.rng)
            eat = max(0, min(a.food, int(eat)))
            new_h = a.health - 1 + eat
            a.food -= eat
            w.record("eat", a.id, f"eats {eat} food", audience=[a.id], phase="eat")
            if new_h <= 0:
                a.health = 0
                self._die(a, "starvation")
            else:
                a.health = min(w.params.health_max, new_h)
        # age
        for a in participants:
            if a.alive:
                a.age += 1
        # natural mortality (SSA)
        for a in participants:
            if a.alive and w.rng.chance(self.mortality.q(a.age, a.sex)):
                self._die(a, "natural")

    def _die(self, a, cause):
        w = self.w
        a.alive = False
        a.death_year = w.year
        a.death_cause = cause
        if w.params.dead_food == "pile":
            w.pile += a.food
            a.food = 0
        elif w.params.dead_food == "lost":
            a.food = 0
        verb = "starves" if cause == "starvation" else "dies"
        w.record("death", "village", f"{a.name} {verb} at age {a.age}.",
                 payload={"agent": a.id, "cause": cause}, phase="mortality")

    # -- compaction ------------------------------------------------------- #
    def _compaction(self):
        w = self.w
        for a in w.living():
            budget = a.memory_tokens
            def total():
                return approx_tokens(a.memory_summary) + sum(
                    approx_tokens(e.get("text", "")) for e in a.memory_raw)
            if total() > budget and len(a.memory_raw) > 6:
                fold_n = len(a.memory_raw) // 2
                fold = a.memory_raw[:fold_n]
                a.memory_raw = a.memory_raw[fold_n:]
                a.memory_summary = self.backend.compact(w, a, fold, w.rng)
                w.record("compaction", a.id,
                         f"({a.name} consolidates older memories)",
                         audience=[a.id], phase="meeting")
