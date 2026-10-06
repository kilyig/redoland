"""The Redoland engine, driving Concordia entities.

Deterministic physics (food, Lanchester combat, satiation, HP, SSA mortality,
crossover) computed in code here; every *decision* is delegated to an agent's
Concordia mind via the `decide` helpers (decide.X(world, a)). The engine's own
randomness uses the seeded, checkpointed `world.rng` so physics replays
deterministically (LLM choices are never deterministic).
"""

from __future__ import annotations

from typing import Optional

from .core import Agent, BIG5, Mortality, Params, RNG, approx_tokens, make_name
from . import decide
from .world import World


class Engine:
    def __init__(self, world: World, mortality: Optional[Mortality] = None):
        self.w = world
        self.mortality = mortality or Mortality()

    # ===================================================================== #
    # Founding                                                              #
    # ===================================================================== #
    @classmethod
    def found(cls, params: Params, model_factory, seed: int, mortality=None,
              randomize_choices: bool = True, premise: str = "") -> "Engine":
        rng = RNG(seed=seed)
        world = World(params, rng, model_factory=model_factory,
                      randomize_choices=randomize_choices, premise=premise)
        eng = cls(world, mortality)
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
            world.attach_mind(ag.id)
        return eng

    # ===================================================================== #
    # One year                                                              #
    # ===================================================================== #
    def run_year(self):
        """Run one full year by stepping to the next year boundary (convenience for
        tests / offline use). Assumes the cursor is at a year boundary."""
        while True:
            if self.step() == "year_end":
                return self.w.year_events

    # -- one atomic, committable step ------------------------------------- #
    def step(self):
        """Advance the simulation by ONE atomic step and return a short label:
        'setup' (a year opens, food appears), 'action' (one agent took their move),
        'talk'/'talk_end' (one utterance in a group conversation / the conversation
        closing), 'fight'/'fight_end' (one muster pass or blow round / the fight
        resolving), 'quiescent' (no one else wants to act), or 'year_end' (eat / age /
        mortality resolved). The world's run cursor makes this resumable: a fork taken
        after any single step — including mid-conversation or mid-fight — continues from
        exactly that point, because the talk/fight sub-state is part of the cursor."""
        w = self.w
        cur = w.cursor
        phase = cur.get("phase", "year_start")
        if phase == "year_start":
            w.year += 1
            w.year_events = []
            self._setup()
            cur["phase"], cur["last"], cur["steps"] = "scramble", None, 0
            return "setup"
        if phase == "scramble":
            if cur["steps"] >= w.params.max_events_per_year:
                cur["phase"] = "year_end"
                return "quiescent"
            actor = self._next_actor(cur["last"])
            if actor is None:
                cur["phase"] = "year_end"
                return "quiescent"
            # A talk/attack move may hand control to the talk/fight state machine
            # (phase becomes 'talk'/'fight'); instant moves resolve inline here.
            self._initiate(actor, stepwise=True)
            cur["last"], cur["steps"] = actor.id, cur["steps"] + 1
            return "action"
        if phase == "talk":
            t = cur.get("talk")
            if t is None or not self._talk_step(t):     # one utterance, or the talk ends
                cur.pop("talk", None)
                cur["phase"] = "scramble"
                return "talk_end"
            return "talk"
        if phase == "fight":
            f = cur.get("fight")
            if f is None or not self._fight_step(f):     # one muster/blow, or it resolves
                cur.pop("fight", None)
                cur["phase"] = "scramble"
                return "fight_end"
            return "fight"
        if phase == "year_end":
            self._year_end()
            self._compaction()
            cur["phase"], cur["last"], cur["steps"] = "year_start", None, 0
            return "year_end"
        return "idle"

    def _setup(self):
        w = self.w
        p = w.params
        n = len(w.living())
        if p.food_base or p.food_floor_ratio:
            f = max(p.food_base, round(p.food_floor_ratio * n))
        else:
            f = round(n * p.ratio)
        # an inject made while the year was closed may have set the coming harvest
        if w.next_pile_set is not None:
            f = w.next_pile_set
            w.next_pile_set = None
        # food banked for this year rolls into the pile: the stores of those who died at
        # the end of last year (dead_food="pile") and anything injected into the pile
        # while the year was closed (World.add_pile_food). Only those two paths write
        # next_pile_bonus, so it is honoured whatever dead_food is. Unclaimed pile food
        # otherwise just spoils on reset.
        bonus = w.next_pile_bonus
        w.next_pile_bonus = 0
        w.pile = f + bonus
        for a in w.living():
            a.bore_this_year = False
            a.repro_done_year = False
            a.says_used_year = 0               # refill each agent's yearly speaking budget
            a.actions_used_year = 0            # refill each agent's yearly ACTION budget
        if w.year == 1:            # year 1 is the start; announce the founding here
            if w.premise:          # the creator's "stage" — public to all from the start
                w.record("narrate", "village", w.premise, phase="setup")
            w.record("narrate", "village",
                     f"The village is founded by {n} people.", phase="setup")
        extra = f" ({bonus} of it carried over from last year)" if bonus else ""
        w.record("narrate", "village",
                 f"Year {w.year}: {w.pile} food appears in the pile for {n} people{extra}.",
                 phase="setup")

    # -- the scramble: one-at-a-time willingness draw --------------------- #
    def _next_actor(self, last):
        """Draw eligible agents ONE AT A TIME (respecting the one-step cooldown) and
        return the FIRST that wants to act — or None if no one does (quiescent). The
        uniform shuffle makes the actor a uniform pick among the willing. If only the
        just-acted agent still wants to act (sole survivor / last willing person),
        they continue rather than ending the year on the cooldown alone."""
        w = self.w
        cap = w.params.actions_per_year
        # An agent out of ACTIONS for the year can no longer INITIATE a move (though it can
        # still be pulled into a talk / defend / join a fight — those don't go through here),
        # so it drops out of the scramble draw. The year goes quiescent once no one both has
        # actions left AND wants to act.
        pool = [a for a in w.living() if a.id != last and a.actions_used_year < cap]
        w.rng.shuffle(pool)
        actor = next((a for a in pool if decide.willing(w, a)), None)
        if actor is None:
            la = w.agents.get(last) if last else None
            if la and la.alive and la.actions_used_year < cap and decide.willing(w, la):
                actor = la
        return actor

    def _initiate(self, actor, action=None, stepwise=False):
        # action=None: the agent chooses (normal scramble). Otherwise the action is
        # FORCED (used by inject to script a specific agent move into the timeline).
        # stepwise=True (the scramble loop): talk/attack hand off to the talk/fight
        # state machine, so each utterance/blow becomes its own resumable commit.
        # stepwise=False (inject / direct calls): talk/attack play out atomically.
        w = self.w
        agent_chosen = action is None       # a normal scramble move (vs. an inject-forced one)
        if action is None:
            action = decide.choose_action(w, actor)
        kind = action.get("kind", "pass")
        # `acted` = did the initiator actually spend a move? Only a real initiation counts
        # toward the yearly ACTION budget (charged once, below, to this actor only). A pass
        # or a no-op (empty take, invalid target) costs nothing.
        acted = False
        if kind == "take":
            amt = max(0, min(int(action.get("amount", 1) or 0), w.pile))
            if amt > 0:
                actor.food += amt
                w.pile -= amt
                w.record("take", actor.id,
                         f"{actor.name} takes {amt} from the pile (pile now {w.pile}).",
                         payload={"amount": amt})
                acted = True
        elif kind == "give":
            tgt = w.agents.get(action.get("target"))
            amt = int(action.get("amount", 1) or 0)
            if tgt and tgt.alive and amt >= 1 and actor.food >= amt:
                actor.food -= amt
                tgt.food += amt
                w.record("give", actor.id, f"{actor.name} gives {amt} food to {tgt.name}.",
                         payload={"target": tgt.id, "amount": amt})
                acted = True
        elif kind == "child":
            # A child OFFER counts as an action for the OFFERER only (the partner being
            # offered spends nothing — they never reach here). Even a doomed offer (stated
            # and refused by the engine) is a spent move; only an invalid partner is a no-op.
            acted = self._convo_chunk(actor, action)
        elif kind == "talk":
            # The speaking budget is GLOBAL and per-utterance, charged in `_say` (opener
            # and every reply alike). Here we only refuse to OPEN a talk when the actor has
            # no words left to speak the opener. Being pulled into someone else's talk,
            # offering a child (kind=="child"), and giving/taking/attacking never touch the
            # speaking budget. Inject-forced talks (agent_chosen=False) bypass the gate.
            if agent_chosen and actor.says_used_year >= w.params.says_per_year:
                w.record("pass", actor.id, "", audience=[actor.id])
                return
            t = self._enter_talk(actor, action)
            if t is None:
                return
            acted = True                              # opening a talk is one action (opener only)
            if stepwise:
                w.cursor["talk"] = t                  # the state machine drives it
                w.cursor["phase"] = "talk"
            else:
                while self._talk_step(t):             # play it out atomically
                    pass
        elif kind == "attack":
            tgt = w.agents.get(action.get("target"))
            if tgt and tgt.alive and tgt.id != actor.id:
                demand = int(action.get("demand", tgt.food) or 0)
                f = self._enter_fight(actor, tgt, demand)
                acted = True                          # STARTING the attack is the initiator's
                                                      # action; defenders, co-attackers who join,
                                                      # and the blows themselves cost nothing.
                if stepwise:
                    w.cursor["fight"] = f             # the state machine drives it
                    w.cursor["phase"] = "fight"
                else:
                    while self._fight_step(f):        # play it out atomically
                        pass
        else:
            w.record("pass", actor.id, "", audience=[actor.id])
        # Charge the yearly action budget once, to the initiator, for a real scramble move.
        # Inject-forced moves (agent_chosen=False) bypass the budget, like the speaking gate.
        if acted and agent_chosen:
            actor.actions_used_year += 1

    # -- reproduction proposal -------------------------------------------- #
    def _can_bear_child(self, proposer, partner, proposer_share):
        """Single source of truth for whether a proposed birth can happen — used to gate
        the offer/accept affordances AND to validate at birth. Returns (ok, reason); reason
        is "" when ok. Because a proposal resolves atomically (offer -> accept -> birth in
        one call), every reason here is knowable at offer time, so the 'accept' affordance
        is never presented for a doomed offer."""
        w = self.w
        cost = w.params.child_cost
        proposer_share = max(0, min(int(proposer_share), cost))
        partner_share = cost - proposer_share
        if proposer.sex == partner.sex:
            return False, "two people of the same sex cannot have a child"
        if proposer.is_close_kin(partner.id):
            return False, "they are close kin"
        mother = proposer if proposer.sex == "female" else partner
        if mother.bore_this_year:
            return False, f"{mother.name} has already borne a child this year"
        if mother.age > w.params.max_maternal_age:
            return False, (f"{mother.name} is past childbearing age "
                           f"(over {w.params.max_maternal_age})")
        if proposer.food < proposer_share:
            return False, (f"{proposer.name} cannot spare the {proposer_share} food "
                           f"they offered (holds {proposer.food})")
        if partner.food < partner_share:
            return False, (f"{partner.name} cannot spare the remaining {partner_share} of "
                           f"{cost} food (holds {partner.food})")
        return True, ""

    def _convo_chunk(self, initiator, action):
        """Make a child offer. Returns True if an offer was actually attempted (a real move
        for the offerer — even a doomed one the engine refuses), False on an invalid partner
        (a no-op that costs the offerer nothing)."""
        w = self.w
        partner = w.agents.get(action.get("partner"))
        if not partner or not partner.alive or partner.id == initiator.id:
            return False
        grp = [initiator.id, partner.id]
        share = max(0, min(int(action.get("my_share", 1) or 0), w.params.child_cost))
        # Affordance gate: a doomed offer is never made — say exactly why and ask nothing.
        ok, reason = self._can_bear_child(initiator, partner, share)
        if not ok:
            w.record("proposal", "village",
                     f"{initiator.name} cannot offer a child to {partner.name}: {reason}.",
                     audience=grp, phase="proposal")
            return True
        w.record("proposal", initiator.id, f"{initiator.name} draws {partner.name} aside to talk.",
                 audience=grp, phase="proposal")
        w.record("propose", initiator.id,
                 f"{initiator.name}: have a child with me — I'll put in {share} of "
                 f"{w.params.child_cost} food.", audience=grp, phase="proposal")
        if decide.respond_child(w, partner, initiator, share):
            if not self._birth(initiator, partner, share):
                # Should be unreachable (the offer was vetted above); re-state the precise
                # reason rather than the old vague "food or age" message if it ever trips.
                _, reason = self._can_bear_child(initiator, partner, share)
                w.record("proposal", "village",
                         f"{partner.name} agrees, but no child comes of it: "
                         f"{reason or 'conditions changed'}.",
                         audience=grp, phase="proposal")
        else:
            w.record("reject", partner.id, f"{partner.name} declines.",
                     audience=grp, phase="proposal")
        return True

    # -- free-form group talk (a resumable, per-utterance state machine) --- #
    # The conversation is driven one utterance at a time. `_enter_talk` records the
    # opener and returns a small JSON-serializable state dict; `_talk_step` produces
    # exactly one utterance (or ends the talk). The scramble loop parks this dict in
    # `world.cursor["talk"]`, so each utterance is its own commit and a fork taken
    # mid-conversation resumes the talk exactly where it left off.
    def _enter_talk(self, initiator, action):
        """Validate the group and record the 'gathers to talk' opener. Returns the talk
        state dict, or None if fewer than two live participants (nothing happens)."""
        w = self.w
        ids = action.get("partners") or ([action["partner"]] if action.get("partner") else [])
        group, seen = [initiator], {initiator.id}
        for pid in ids:
            o = w.agents.get(pid)
            if o and o.alive and o.id not in seen:
                group.append(o)
                seen.add(o.id)
        if len(group) < 2:
            return None
        gids = [g.id for g in group]                  # FIXED roster: the audience key
        others_names = ", ".join(g.name for g in group[1:])
        w.record("convo", initiator.id, f"{initiator.name} gathers {others_names} to talk.",
                 audience=gids, phase="convo")
        return {"group": gids, "initiator": initiator.id,
                "last": None, "guard": 0, "opened": False}

    def _talk_step(self, t):
        """Produce ONE utterance (or end the talk). Returns True while the conversation
        continues, False once it has closed."""
        w = self.w
        gids = t["group"]                             # canonical roster (audience key)
        alive = [w.agents[i] for i in gids if i in w.agents and w.agents[i].alive]
        if len(alive) < 2:                            # the group collapsed (e.g. deaths)
            return False
        if not t["opened"]:                           # the initiator speaks first
            t["opened"] = True
            initiator = w.agents.get(t["initiator"])
            if initiator and initiator.alive:
                t["last"] = initiator.id if self._say(initiator, alive, gids) else None
            return True
        # The per-agent global word budget is what ends most talks (speakers drop out as
        # their words run out); convo_safety_cap is just the hard runaway guard.
        if t["guard"] >= int(w.params.convo_safety_cap):
            return False
        t["guard"] += 1
        history = self._tail(w, gids)
        willing = [g for g in alive if g.id != t["last"]
                   and g.says_used_year < w.params.says_per_year
                   and decide.want_to_speak(w, g, [o for o in alive if o.id != g.id], history)]
        if not willing:
            return False
        speaker = w.rng.choice(willing)
        self._say(speaker, alive, gids)
        t["last"] = speaker.id
        return True

    def _say(self, agent, alive, gids):
        # Every recorded utterance spends one of the speaker's global yearly words. Out of
        # words => they stay silent (the willing-filter and opener-gate normally prevent
        # reaching here, but this keeps the budget authoritative).
        if agent.says_used_year >= self.w.params.says_per_year:
            return ""
        others = [g for g in alive if g.id != agent.id]
        text = (decide.say(self.w, agent, others, self._tail(self.w, gids)) or "").strip()
        if text:
            agent.says_used_year += 1
            self.w.record("say", agent.id, text, audience=gids, phase="convo")
        return text

    def _talk_chunk(self, initiator, action):
        """Atomic group talk (inject / direct/test calls): enter then run to the end."""
        t = self._enter_talk(initiator, action)
        if t is None:
            return
        while self._talk_step(t):
            pass

    def _tail(self, w, gids):
        lines = [f"{e['who']}: {e['text']}" for e in w.year_events
                 if e.get("kind") == "say" and e.get("audience") == gids]
        return "\n".join(lines)

    # -- fight (a resumable, per-step state machine) ---------------------- #
    # A raid plays out as a sequence of steps: one muster pass at a time, then a single
    # decide step (cancel / submit / press on), then one blow round at a time, then the
    # outcome. `_enter_fight` records the opening and returns a JSON-serializable state
    # dict; `_fight_step` advances it by one step. The scramble loop parks this dict in
    # `world.cursor["fight"]`, so every muster pass and blow round is its own commit and
    # a fork taken mid-fight resumes from exactly that point.
    def _enter_fight(self, initiator, target, demand):
        w = self.w
        w.record("attack", "village", f"{initiator.name} moves to attack {target.name}.",
                 payload={"initiator": initiator.id, "target": target.id}, phase="fight")
        w.record("under_attack", target.id, f"You are under attack by {initiator.name}.",
                 audience=[target.id], phase="fight")
        return {"initiator": initiator.id, "target": target.id, "demand": int(demand),
                "attackers": [initiator.id], "defenders": [target.id],
                "sub": "muster", "muster_pass": 0, "round": 0, "fallen": []}

    def _fight_step(self, f):
        """Advance the fight by one step. Returns True while it continues, False once
        the raid has resolved (cancel / submit / outcome)."""
        w = self.w
        p = w.params
        attackers = {i for i in f["attackers"] if i in w.agents and w.agents[i].alive}
        defenders = {i for i in f["defenders"] if i in w.agents and w.agents[i].alive}
        initiator = w.agents.get(f["initiator"])
        target = w.agents.get(f["target"])
        if not attackers or not defenders:               # a side vanished — resolve now
            return self._fight_outcome(f, attackers, defenders)
        sub = f["sub"]

        if sub == "muster":                              # one muster pass per step
            added = self._recruit(attackers, defenders, "attack")
            added = self._recruit(defenders, attackers, "defend") or added
            f["attackers"], f["defenders"] = sorted(attackers), sorted(defenders)
            f["muster_pass"] += 1
            if not added or f["muster_pass"] >= p.muster_passes_cap:
                w.record("muster", "village",
                         f"Attackers [{self._names(attackers)}] (str {self._roster_str(attackers)}) "
                         f"vs defenders [{self._names(defenders)}] (str {self._roster_str(defenders)}).",
                         payload={"attackers": sorted(attackers), "defenders": sorted(defenders)},
                         phase="fight")
                f["sub"] = "decide"
            return True

        if sub == "decide":                              # cancel / submit / press on
            if decide.attacker_decision(w, initiator, attackers, defenders) == "cancel":
                w.record("cancel", "village",
                         f"{initiator.name} thinks better of it and calls off the attack.",
                         phase="fight")
                return False
            if decide.defender_decision(w, target, attackers, defenders) == "submit":
                if decide.attacker_on_submit(w, initiator, attackers, defenders) == "accept":
                    amt = min(f["demand"], target.food)
                    target.food -= amt
                    initiator.food += amt
                    w.record("submit", "village",
                             f"{target.name} submits; {initiator.name} takes {amt} food "
                             f"without a fight.", payload={"amount": amt}, phase="fight")
                    return False
                w.record("presson", "village",
                         f"{target.name} submits, but {initiator.name} attacks anyway.", phase="fight")
            f["sub"] = "blows"
            return True

        if sub == "blows":                               # one blow round per step
            if f["round"] >= p.blow_rounds_cap:
                return self._fight_outcome(f, attackers, defenders)
            f["round"] += 1
            fa, fd = set(attackers), set(defenders)
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
                    f["fallen"].append(i)
                    fa.discard(i); fd.discard(i)
            for side, enemy in ((fa, fd), (fd, fa)):
                for i in list(side):
                    a = w.agents[i]
                    if not a.alive:
                        side.discard(i); continue
                    d = decide.morale(w, a, side, enemy)
                    if d in ("flee", "yield"):
                        side.discard(i)
                        w.record(d, "village",
                                 f"{a.name} {'flees' if d == 'flee' else 'yields'}.", phase="fight")
            f["attackers"], f["defenders"] = sorted(fa), sorted(fd)
            if not fa or not fd:
                return self._fight_outcome(f, fa, fd)
            return True

        return self._fight_outcome(f, attackers, defenders)

    def _fight_outcome(self, f, attackers, defenders):
        w = self.w
        mode = w.params.dead_food
        initiator = w.agents.get(f["initiator"])
        target = w.agents.get(f["target"])
        won = bool(attackers and not defenders)
        iname = initiator.name if initiator else "the attacker"

        if won and target is not None and not target.alive \
                and initiator is not None and initiator.alive and target.food > 0:
            # the TARGET was slain: the victor loots the body — and chooses how much to
            # take (up to everything), no longer capped by the demand. Whatever they
            # leave drops into this year's pile (below).
            avail = max(0, target.food)
            loot = max(0, min(avail, int(decide.loot_choice(w, initiator, target, avail))))
            initiator.food += loot
            target.food -= loot
            w.record("loot", "village",
                     f"{iname} takes {loot} of {avail} food from {target.name}'s body.",
                     payload={"amount": loot, "available": avail}, phase="fight")
        elif won and target is not None and target.alive:
            # the target survived (routed/fled) but the attackers hold the field: the
            # victor seizes up to the demanded amount; the target keeps the rest.
            amt = min(f["demand"], max(0, target.food))
            target.food = max(0, target.food - amt)
            if initiator:
                initiator.food += amt
            w.record("outcome", "village",
                     f"The attackers prevail; {iname} seizes {amt} food from {target.name}.",
                     payload={"amount": amt}, phase="fight")

        # everyone who fell in the fight: their remaining stores drop into THIS year's
        # pile (others can still grab them this year; unclaimed = spoiled at the reset).
        if mode == "pile":
            spoils = 0
            for i in f.get("fallen", []):
                a = w.agents.get(i)
                if a is not None and not a.alive and a.food > 0:
                    spoils += a.food
                    w.pile += a.food
                    a.food = 0
            if spoils:
                w.record("spoils", "village",
                         f"{spoils} food from the fallen is left in the plaza for the taking.",
                         payload={"amount": spoils}, phase="fight")

        if won:
            if not (target is not None and target.alive):
                w.record("outcome", "village",
                         f"The attackers prevail; {iname}'s raid succeeds.", phase="fight")
        else:
            w.record("outcome", "village",
                     f"The defenders hold; {iname}'s raid fails.", phase="fight")
        return False

    def _fight_chunk(self, initiator, target, demand):
        """Atomic fight (inject / direct/test calls): enter then run to the end."""
        f = self._enter_fight(initiator, target, demand)
        while self._fight_step(f):
            pass

    def _recruit(self, side, opposing, label):
        w = self.w
        added = False
        for m in list(side):
            for inv in decide.recruit_invites(w, w.agents[m], side, opposing, label):
                a = w.agents.get(inv)
                if not a or not a.alive or inv in side or inv in opposing:
                    continue
                if decide.accept_join(w, a, side, opposing, label):
                    side.add(inv)
                    added = True
                    w.record("join", "village", f"{a.name} joins the {label}ers.", phase="fight")
        return added

    def _roster_str(self, ids):
        return sum(self.w.agents[i].strength for i in ids if self.w.agents[i].alive)

    def _names(self, ids):
        return ", ".join(self.w.agents[i].name for i in ids if i in self.w.agents)

    # -- reproduction ----------------------------------------------------- #
    def _birth(self, proposer, partner, proposer_share):
        w = self.w
        ok, _ = self._can_bear_child(proposer, partner, proposer_share)
        if not ok:                                   # single source of truth (see above)
            return False
        partner_share = w.params.child_cost - proposer_share
        mother = proposer if proposer.sex == "female" else partner
        father = partner if proposer.sex == "female" else proposer
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
        w.agents[child.id] = child
        w.attach_mind(child.id)                       # mid-year birth: register the new mind
        child.mother_note = decide.note(w, mother, child)
        child.father_note = decide.note(w, father, child)
        mother.bore_this_year = True
        mother.repro_done_year = True
        father.repro_done_year = True
        big5 = " ".join(f"{t[0].upper()}{child.trait(t)}" for t in BIG5)
        w.record("birth", "village",
                 f"{child.name} ({child.sex}, age {child.age}; str {child.strength}, "
                 f"int {child.intelligence_tokens}, mem {child.memory_tokens}; "
                 f"{big5}) is born to {mother.name} and {father.name}.",
                 payload={"child": child.id, "mother": mother.id, "father": father.id},
                 phase="proposal")
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
        for a in agents:
            eat = max(0, min(a.food, int(decide.eat_choice(w, a))))
            new_h = a.health - 1 + eat
            a.food -= eat
            w.record("eat", a.id, f"eats {eat} food", audience=[a.id], phase="eat")
            if new_h <= 0:
                a.health = 0
                self._die(a, "starvation")
            else:
                a.health = min(p.health_max, new_h)
        for a in agents:
            if a.alive and a.health >= p.hp_recovery_min_satiation and a.hp < p.hp_max:
                a.hp = min(p.hp_max, a.hp + p.hp_recovery)
        for a in agents:
            if a.alive:
                a.age += 1
        for a in agents:
            if a.alive and w.rng.chance(self.mortality.q(a.age, a.sex)):
                self._die(a, "natural")

    def _die(self, a, cause):
        w = self.w
        a.alive = False
        a.death_year = w.year
        a.death_cause = cause
        a.hp = 0
        mode = w.params.dead_food
        if mode == "lost":
            a.food = 0
        elif mode == "pile":
            if cause == "combat":
                pass               # food stays on the body; the fight resolution loots it
                                   # (victor first) and drops the rest into THIS year's pile
            else:                  # year-end death: stores roll into NEXT year's pile
                w.add_pile_food(a.food)     # (banked: the year is closed by now)
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
                a.memory_summary = decide.compact(w, a, fold)
                w.record("compaction", a.id, f"({a.name} consolidates older memories)",
                         audience=[a.id], phase="boundary")
