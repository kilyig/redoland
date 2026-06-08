"""The simulation engine — v2 force-and-conversation model (MVP_PLAN_V2.md).

Year = food into the pile → SCRAMBLE (a discrete-event willingness lottery over
TAKE / GIVE / CONVO / ATTACK) → boundary (eat / HP-recover / age / SSA mortality
/ compaction / commit). Conversations and fights are atomic multi-step chunks.
There is no voting; distribution is free TAKE from the pile and raiding by force.
"""

from __future__ import annotations

from typing import Optional

from .core import (Agent, Mortality, Params, RNG, BIG5, approx_tokens, make_name)


class World:
    def __init__(self, params: Params, rng: RNG, branch: str = "main"):
        self.params = params
        self.rng = rng
        self.branch = branch
        self.year = 0
        self.pile = 0
        self.agents: dict[str, Agent] = {}
        self.used_names: set = set()
        self.next_eid = 0
        self.next_aid = 0
        self.year_events: list[dict] = []

    def living(self) -> list[Agent]:
        return [self.agents[i] for i in sorted(self.agents) if self.agents[i].alive]

    def new_aid(self) -> str:
        self.next_aid += 1
        return f"a{self.next_aid:03d}"

    def record(self, kind, who, text="", audience="public", payload=None, phase="scramble"):
        self.next_eid += 1
        eid = f"e{self.next_eid:06d}"
        who_name = self.agents[who].name if who in self.agents else "the village"
        ev = {"eid": eid, "year": self.year, "phase": phase, "kind": kind,
              "speaker": who, "who": who_name, "audience": audience,
              "text": text, "payload": payload or {}}
        self.year_events.append(ev)
        mem = {"eid": eid, "year": self.year, "kind": kind, "who": who_name, "text": text}
        for a in self._witnesses(who, audience):
            a.memory_raw.append(dict(mem))
        return ev

    def _witnesses(self, speaker, audience):
        if audience == "public":
            return self.living()
        ids = set(audience) if isinstance(audience, list) else set()
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
                strength=rng.randint(15, 85), hp=params.hp_max,
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
        self._scramble()
        self._year_end()
        self._compaction()
        return self.w.year_events

    def _setup(self):
        w = self.w
        n = len(w.living())
        f = round(n * w.params.ratio)
        w.pile = f
        for a in w.living():
            a.bore_this_year = False
            a.repro_done_year = False
        w.record("narrate", "village",
                 f"Year {w.year}: {f} food appears in the pile for {n} people.",
                 phase="setup")

    # -- the scramble: discrete-event willingness lottery ----------------- #
    def _scramble(self):
        w = self.w
        last = None
        steps = 0
        while steps < w.params.max_events_per_year:
            willing = [a for a in w.living()
                       if a.id != last and self.backend.willing(w, a, w.rng)]
            if not willing:
                break
            actor = w.rng.choice(willing)
            self._initiate(actor)
            last = actor.id
            steps += 1

    def _initiate(self, actor):
        w = self.w
        action = self.backend.choose_action(w, actor, w.rng)
        kind = action.get("kind", "pass")
        if kind == "take":
            amt = max(0, min(int(action.get("amount", 1)), w.pile))
            if amt > 0:
                actor.food += amt
                w.pile -= amt
                w.record("take", actor.id,
                         f"{actor.name} takes {amt} from the pile "
                         f"(pile now {w.pile}).", payload={"amount": amt})
        elif kind == "give":
            tgt = w.agents.get(action.get("target"))
            amt = int(action.get("amount", 1))
            if tgt and tgt.alive and amt >= 1 and actor.food >= amt:
                actor.food -= amt
                tgt.food += amt
                w.record("give", actor.id,
                         f"{actor.name} gives {amt} food to {tgt.name}.",
                         payload={"target": tgt.id, "amount": amt})
        elif kind == "convo":
            self._convo_chunk(actor, action)
        elif kind == "attack":
            tgt = w.agents.get(action.get("target"))
            if tgt and tgt.alive and tgt.id != actor.id:
                self._fight_chunk(actor, tgt, int(action.get("demand", tgt.food)))
        else:
            w.record("pass", actor.id, "", audience=[actor.id])

    # -- conversation chunk (v1: reproduction negotiation) ---------------- #
    def _convo_chunk(self, initiator, action):
        w = self.w
        partner = w.agents.get(action.get("partner"))
        if not partner or not partner.alive or partner.id == initiator.id:
            return
        grp = [initiator.id, partner.id]
        w.record("convo", initiator.id,
                 f"{initiator.name} draws {partner.name} aside to talk.",
                 audience=grp, phase="convo")
        share = max(0, min(int(action.get("my_share", 1)), w.params.child_cost))
        w.record("propose", initiator.id,
                 f"{initiator.name}: have a child with me — I'll put in {share} "
                 f"of {w.params.child_cost} food.", audience=grp, phase="convo")
        if self.backend.respond_child(w, partner, initiator, share, w.rng):
            if not self._birth(initiator, partner, share):
                w.record("convo", partner.id,
                         f"{partner.name} agrees, but they cannot spare the food.",
                         audience=grp, phase="convo")
        else:
            w.record("reject", partner.id, f"{partner.name} declines.",
                     audience=grp, phase="convo")

    # -- fight chunk ------------------------------------------------------ #
    def _fight_chunk(self, initiator, target, demand):
        w = self.w
        p = w.params
        attackers = {initiator.id}
        defenders = {target.id}
        w.record("attack", "village",
                 f"{initiator.name} moves to attack {target.name}.",
                 payload={"initiator": initiator.id, "target": target.id},
                 phase="fight")
        w.record("under_attack", target.id,
                 f"You are under attack by {initiator.name}.",
                 audience=[target.id], phase="fight")

        # MUSTER (escalating arms race)
        for _ in range(p.muster_passes_cap):
            added = self._recruit(attackers, defenders, "attack")
            added = self._recruit(defenders, attackers, "defend") or added
            if not added:
                break
        sA = self._roster_str(attackers)
        sB = self._roster_str(defenders)
        w.record("muster", "village",
                 f"Attackers [{self._names(attackers)}] (str {sA}) vs "
                 f"defenders [{self._names(defenders)}] (str {sB}).",
                 payload={"attackers": sorted(attackers), "defenders": sorted(defenders)},
                 phase="fight")

        # OFF-RAMP
        if self.backend.attacker_decision(w, initiator, attackers, defenders, w.rng) == "cancel":
            w.record("cancel", "village",
                     f"{initiator.name} thinks better of it and calls off the attack.",
                     phase="fight")
            return
        if self.backend.defender_decision(w, target, attackers, defenders, w.rng) == "submit":
            if self.backend.attacker_on_submit(w, initiator, attackers, defenders, w.rng) == "accept":
                amt = min(demand, target.food)
                target.food -= amt
                initiator.food += amt
                w.record("submit", "village",
                         f"{target.name} submits; {initiator.name} takes {amt} food "
                         f"without a fight.", payload={"amount": amt}, phase="fight")
                return
            w.record("presson", "village",
                     f"{target.name} submits, but {initiator.name} attacks anyway.",
                     phase="fight")

        # BLOW ROUNDS
        fa, fd = set(attackers), set(defenders)
        for _ in range(p.blow_rounds_cap):
            fa = {i for i in fa if w.agents[i].alive}
            fd = {i for i in fd if w.agents[i].alive}
            if not fa or not fd:
                break
            SA = sum(w.agents[i].strength for i in fa)
            SB = sum(w.agents[i].strength for i in fd)
            dmg_a = (p.c_lethality * SB) / max(1, len(fa))
            dmg_d = (p.c_lethality * SA) / max(1, len(fd))
            for i in fa:
                w.agents[i].hp -= dmg_a
            for i in fd:
                w.agents[i].hp -= dmg_d
            w.record("blow", "village",
                     f"Blows are traded — attackers lose {dmg_a:.0f} HP each, "
                     f"defenders {dmg_d:.0f} each.", phase="fight")
            for i in list(fa | fd):
                if w.agents[i].hp <= 0:
                    self._die(w.agents[i], "combat")
                    fa.discard(i); fd.discard(i)
            # morale
            for side, enemy in ((fa, fd), (fd, fa)):
                for i in list(side):
                    a = w.agents[i]
                    if not a.alive:
                        side.discard(i); continue
                    d = self.backend.morale(w, a, side, enemy, w.rng)
                    if d in ("flee", "yield"):
                        side.discard(i)
                        w.record(d, "village",
                                 f"{a.name} {'flees' if d=='flee' else 'yields'}.",
                                 phase="fight")

        # OUTCOME
        if fa and not fd:
            amt = min(demand, max(0, target.food))
            if target.id in w.agents:
                target.food = max(0, target.food - amt)
            initiator.food += amt
            w.record("outcome", "village",
                     f"The attackers prevail; {initiator.name} takes {amt} food"
                     f"{' from ' + target.name if target.alive else ' (loot)'}.",
                     payload={"amount": amt}, phase="fight")
        else:
            w.record("outcome", "village",
                     f"The defenders hold; {initiator.name}'s raid fails.",
                     phase="fight")

    def _recruit(self, side, opposing, label):
        w = self.w
        added = False
        for m in list(side):
            for inv in self.backend.recruit_invites(w, w.agents[m], side, opposing, label, w.rng):
                a = w.agents.get(inv)
                if not a or not a.alive or inv in side or inv in opposing:
                    continue
                if self.backend.accept_join(w, a, side, opposing, label, w.rng):
                    side.add(inv)
                    added = True
                    w.record("join", "village",
                             f"{a.name} joins the {label}ers.", phase="fight")
        return added

    def _roster_str(self, ids):
        return sum(self.w.agents[i].strength for i in ids if self.w.agents[i].alive)

    def _names(self, ids):
        return ", ".join(self.w.agents[i].name for i in ids if i in self.w.agents)

    # -- reproduction (called from a convo) ------------------------------- #
    def _birth(self, proposer, partner, proposer_share):
        w = self.w
        partner_share = w.params.child_cost - proposer_share
        if proposer.sex == partner.sex or proposer.is_close_kin(partner.id):
            return False
        if proposer.food < proposer_share or partner.food < partner_share:
            return False
        mother = proposer if proposer.sex == "female" else partner
        father = partner if proposer.sex == "female" else proposer
        if mother.bore_this_year:
            return False
        proposer.food -= proposer_share
        partner.food -= partner_share
        child = self._crossover(mother, father)
        child.parents = [mother.id, father.id]
        child.siblings = sorted(set(mother.children + father.children))
        for sid in child.siblings:
            sib = w.agents.get(sid)
            if sib and child.id not in sib.siblings:
                sib.siblings.append(child.id)
        mother.children.append(child.id)
        father.children.append(child.id)
        child.mother_note = self.backend.note(w, mother, child, w.rng)
        child.father_note = self.backend.note(w, father, child, w.rng)
        mother.bore_this_year = True
        mother.repro_done_year = True
        father.repro_done_year = True
        w.agents[child.id] = child
        w.record("birth", "village",
                 f"{child.name} ({child.sex}, str {child.strength}) is born to "
                 f"{mother.name} and {father.name}, age {child.age}.",
                 payload={"child": child.id, "mother": mother.id, "father": father.id},
                 phase="convo")
        return True

    def _crossover(self, mother, father):
        w = self.w
        rng = w.rng
        p = w.params
        traits = {}
        for t in BIG5:
            v = round((mother.trait(t) + father.trait(t)) / 2 + rng.gauss(0, p.big5_sigma))
            traits[t] = int(max(0, min(100, v)))
        intel = int(max(p.int_min, min(p.int_max,
                    round((mother.intelligence_tokens + father.intelligence_tokens) / 2
                          + rng.gauss(0, p.int_sigma)))))
        mem = int(max(p.mem_min, min(p.mem_max,
                  round((mother.memory_tokens + father.memory_tokens) / 2
                        + rng.gauss(0, p.mem_sigma)))))
        strength = int(max(0, min(100,
                      round((mother.strength + father.strength) / 2
                            + rng.gauss(0, p.strength_sigma)))))
        sex = "male" if rng.chance(0.5) else "female"
        name = make_name(sex, rng, w.used_names)
        w.used_names.add(name)
        return Agent(id=w.new_aid(), name=name, sex=sex, traits=traits,
                     intelligence_tokens=intel, memory_tokens=mem,
                     age=p.child_age, health=p.child_health, food=0,
                     birth_year=w.year, strength=strength, hp=p.hp_max)

    # -- year end --------------------------------------------------------- #
    def _year_end(self):
        w = self.w
        p = w.params
        agents = w.living()
        # eat
        for a in agents:
            eat = max(0, min(a.food, int(self.backend.eat_choice(w, a, w.rng))))
            new_h = a.health - 1 + eat
            a.food -= eat
            w.record("eat", a.id, f"eats {eat} food", audience=[a.id], phase="eat")
            if new_h <= 0:
                a.health = 0
                self._die(a, "starvation")
            else:
                a.health = min(p.health_max, new_h)
        # HP recovery (gated on satiation)
        for a in agents:
            if a.alive and a.health >= p.hp_recovery_min_satiation and a.hp < p.hp_max:
                a.hp = min(p.hp_max, a.hp + p.hp_recovery)
        # age all survivors (incl. newborns)
        for a in agents:
            if a.alive:
                a.age += 1
        # SSA natural mortality
        for a in agents:
            if a.alive and w.rng.chance(self.mortality.q(a.age, a.sex)):
                self._die(a, "natural")

    def _die(self, a, cause):
        w = self.w
        a.alive = False
        a.death_year = w.year
        a.death_cause = cause
        a.hp = 0
        if w.params.dead_food == "pile":
            w.pile += a.food
            a.food = 0
        elif w.params.dead_food == "lost":
            a.food = 0
        verb = {"starvation": "starves", "combat": "is killed", "natural": "dies"}[cause]
        w.record("death", "village", f"{a.name} {verb} at age {a.age}.",
                 payload={"agent": a.id, "cause": cause},
                 phase="mortality" if cause != "combat" else "fight")

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
                         audience=[a.id], phase="boundary")
