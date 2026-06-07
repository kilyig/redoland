"""LLM backends.

A backend turns world state + an agent into decisions. Two implementations:

* FakeBackend   — deterministic, trait-driven heuristics seeded by the engine
                  RNG. Needs no API key; used for all offline test runs and for
                  exercising the web viewer. Produces lively, readable
                  transcripts so the branching demo is meaningful.
* AnthropicBackend — the real thing. All agents run on Claude Haiku 4.5 by
                  default (configurable via REDOLAND_AGENT_MODEL /
                  REDOLAND_CHEAP_MODEL). The per-agent `budget_tokens` thinking
                  dial (the continuous intelligence knob, MVP_PLAN.md §8) stays
                  on by default via REDOLAND_THINKING. Imported lazily.

Backend method contract (all take the engine `rng` so every random draw stays
in the one replayable stream):

    handraise(world, agent, rng)      -> (bool, reason)
    turn(world, agent, rng)           -> action dict
    offer(world, agent, cands, rng)   -> {"partner": id, "my_share": int} | None
    respond(world, agent, offer, rng) -> bool
    note(world, parent, child, rng)   -> str
    eat_choice(world, agent, rng)     -> int
    compact(world, agent, events, rng)-> str
"""

from __future__ import annotations

from typing import Optional

from .core import Agent, BIG5


# --------------------------------------------------------------------------- #
# Shared helpers.                                                              #
# --------------------------------------------------------------------------- #


def health_state(h: int) -> str:
    return {
        3: "well-fed and strong",
        2: "getting hungry",
        1: "starving — if you do not eat this year, you will die",
        0: "dead",
    }.get(h, "unknown")


def need_to_cap(agent: Agent, params) -> int:
    """Food needed to top health back to the cap this year (after the -1)."""
    return max(0, params.health_max - (agent.health - 1))


def spare_food(agent: Agent, params) -> int:
    return max(0, agent.food - need_to_cap(agent, params))


# --------------------------------------------------------------------------- #
# Fake backend.                                                                #
# --------------------------------------------------------------------------- #


class FakeBackend:
    name = "fake"

    # -- meeting ---------------------------------------------------------- #
    def handraise(self, world, agent, rng):
        p = 0.15 + 0.45 * (agent.trait("extraversion") / 100.0)
        if agent.health <= 1:
            p += 0.40
        if self._supportable_motion(world, agent) is not None:
            p += 0.30
        if self._starving_kin(world, agent):
            p += 0.25
        p = min(0.97, max(0.02, p))
        return rng.chance(p), "wants to speak"

    def turn(self, world, agent, rng):
        params = world.params
        # 1) vote on any motion I'm willing to support (need-based norm)
        motion = self._supportable_motion(world, agent)
        if motion is not None:
            who = world.agents[motion.recipient].name
            text = rng.choice([
                f"Yes. {who} should have it.",
                f"I'll back this — {who} needs it.",
                f"Aye. Give it to {who}.",
                f"Fair is fair. {who} eats.",
            ])
            return {"kind": "vote", "motion": motion.id, "text": text, "audience": "public"}

        # 2) hungry and nothing routes food to me yet -> propose for myself
        if agent.health <= 1 and world.pile > 0 and not self._motion_for(world, agent.id):
            amt = min(world.pile, 2 if agent.trait("conscientiousness") > 55 else 1)
            text = rng.choice([
                "I have gone hungry too long. I ask the village for a share.",
                "I need food this year. Let it be set aside for me.",
                "Hear me — I am starving. Grant me from the pile.",
            ])
            return {"kind": "propose", "amount": amt, "recipient": agent.id, "text": text,
                    "audience": "public"}

        # 3) propose food for a starving neighbour who has no motion yet (charity norm)
        if world.pile > 0 and agent.trait("agreeableness") > 50:
            needy = self._neediest_without_motion(world, agent)
            if needy is not None:
                text = rng.choice([
                    f"{needy.name} is starving. I say we feed them.",
                    f"Set aside a share for {needy.name} before we lose them.",
                    f"No one here should starve while the pile sits full. For {needy.name}.",
                ])
                return {"kind": "propose", "amount": 1, "recipient": needy.id,
                        "text": text, "audience": "public"}

        # 4) charity to starving kin / neighbours if I can spare it
        if spare_food(agent, params) >= 1:
            target = self._starving_kin(world, agent) or self._starving_other(world, agent)
            if target is not None and (
                agent.is_close_kin(target.id) or agent.trait("agreeableness") > 60
            ):
                text = rng.choice([
                    f"Here, {target.name}. Take this from me.",
                    f"{target.name}, you need it more than I do.",
                    f"Take it, {target.name}. We look after our own.",
                ])
                return {"kind": "give", "target": target.id, "amount": 1, "text": text,
                        "audience": "public"}

        # 4) acquisitive: propose food to myself to build a store
        if world.pile > 0 and rng.chance(0.25 + 0.4 * (1 - agent.trait("agreeableness") / 100.0)):
            amt = min(world.pile, 1)
            text = rng.choice([
                "I would put some by for leaner years. I propose a share for myself.",
                "Set a portion aside for me; I mean to save it.",
                "I ask for a share. A wise household keeps a store.",
            ])
            return {"kind": "propose", "amount": amt, "recipient": agent.id, "text": text,
                    "audience": "public"}

        # 5) talk or stay silent
        if agent.trait("extraversion") > 50 and rng.chance(0.5):
            return {"kind": "say", "text": self._flavor(world, agent, rng), "audience": "public"}
        return {"kind": "pass"}

    # -- reproduction ----------------------------------------------------- #
    def offer(self, world, agent, candidates, rng):
        params = world.params
        if agent.food < 2:
            return None
        inclined = 0.20 + 0.40 * (agent.trait("openness") / 100.0)
        if agent.age < 35:
            inclined += 0.20
        if not rng.chance(min(0.9, inclined)):
            return None
        # opposite sex, not close kin, can plausibly co-parent
        pool = [c for c in candidates if c.sex != agent.sex and not agent.is_close_kin(c.id)]
        if not pool:
            return None
        partner = rng.choice(pool)
        generous = agent.trait("agreeableness") > 55 or agent.sex == "male"
        my_share = 2 if generous else 1
        my_share = min(my_share, agent.food, params.child_cost)
        return {"partner": partner.id, "my_share": my_share}

    def respond(self, world, agent, offer, rng):
        params = world.params
        partner_share = params.child_cost - offer["my_share"]
        if agent.food < partner_share:
            return False
        p = 0.30 + 0.40 * (agent.trait("agreeableness") / 100.0)
        if agent.age < 30:
            p += 0.10
        return rng.chance(min(0.92, p))

    def note(self, world, parent, child, rng):
        neuro = parent.trait("neuroticism")
        openn = parent.trait("openness")
        agree = parent.trait("agreeableness")
        parts = []
        if neuro > 60:
            parts.append(rng.choice([
                "Trust no one too quickly; the village smiles and then it takes.",
                "Guard what is yours. Hunger makes thieves of friends.",
                "Watch the ones who speak the loudest at the gathering.",
            ]))
        if openn > 60:
            parts.append(rng.choice([
                "The world is wider than this pile of food; stay curious.",
                "Question the old rules — some are only old, not wise.",
                "Listen to strangers; they carry news the elders fear.",
            ]))
        if agree > 60:
            parts.append(rng.choice([
                "Share when you can; a fed neighbour is a loyal one.",
                "Keep your word. A name for fairness outlasts a full belly.",
                "Tend your kin first, but do not let the village starve.",
            ]))
        if not parts:
            parts.append(rng.choice([
                "Eat when you can, speak when it matters, and count who owes you.",
                "Stay alive, find a partner, and remember whose blood you carry.",
            ]))
        return f"My child, {child.name}: " + " ".join(parts)

    # -- year end --------------------------------------------------------- #
    def eat_choice(self, world, agent, rng):
        params = world.params
        eat = min(agent.food, need_to_cap(agent, params))
        if agent.health - 1 + eat <= 0 and agent.food > 0:
            eat = max(eat, 1)
        return eat

    def compact(self, world, agent, events, rng):
        keep = []
        for e in events:
            if e.get("kind") in ("say", "propose", "give", "vote", "birth", "death", "narrate"):
                who = e.get("who", "")
                txt = (e.get("text", "") or "").strip()
                if txt:
                    keep.append(f"{who}: {txt}" if who else txt)
        if not keep:
            return agent.memory_summary
        ys = [e.get("year") for e in events if e.get("year") is not None]
        span = f"Years {min(ys)}–{max(ys)}" if ys else "Earlier"
        digest = "; ".join(keep[-8:])
        folded = f"[{span}, as {agent.name} recalls it] {digest}"
        return (agent.memory_summary + " " + folded).strip()

    # -- internals -------------------------------------------------------- #
    def _motion_for(self, world, agent_id):
        return any(m.open and m.recipient == agent_id for m in world.motions)

    def _supportable_motion(self, world, agent):
        """Highest-priority open motion this agent would vote yes on.

        Encodes a simple, emergent-looking distribution norm: back food for
        yourself, your kin, and (if you're not desperate yourself, or you're
        agreeable) for whoever is starving. This is what lets quorum form."""
        best, best_score = None, 0
        for m in world.motions:
            if not m.open or agent.id in m.yes:
                continue
            rec = world.agents.get(m.recipient)
            if rec is None:
                continue
            score = 0
            if m.recipient == agent.id:
                score = 100
            elif agent.is_close_kin(m.recipient):
                score = 80
            elif rec.health <= 1 and (agent.health > 1 or agent.trait("agreeableness") > 50):
                score = 50 + (3 - rec.health)
            elif rec.health <= 2 and agent.trait("agreeableness") > 70:
                score = 20
            if score > best_score:
                best, best_score = m, score
        return best

    def _neediest_without_motion(self, world, agent):
        cands = [a for a in world.living()
                 if a.health <= 1 and not self._motion_for(world, a.id)]
        cands.sort(key=lambda a: (a.health, a.id))
        return cands[0] if cands else None

    def _starving_kin(self, world, agent):
        for kid in agent.parents + agent.children + agent.siblings:
            k = world.agents.get(kid)
            if k and k.alive and k.health <= 1:
                return k
        return None

    def _starving_other(self, world, agent):
        cands = [a for a in world.living() if a.id != agent.id and a.health <= 1]
        return cands[0] if cands else None

    def _flavor(self, world, agent, rng):
        living = [a for a in world.living() if a.id != agent.id]
        other = rng.choice(living).name if living else "the others"
        return rng.choice([
            f"The pile shrinks every year. We cannot all go on like this.",
            f"I heard {other} took more than a fair share last winter.",
            f"We should agree on a rule before the hunger decides for us.",
            f"Who speaks for those with no one to speak for them?",
            f"My children will remember how we behaved this year.",
        ])


# --------------------------------------------------------------------------- #
# Anthropic backend (real). Lazily imported; untested without a key.           #
# --------------------------------------------------------------------------- #


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        import os
        from .env import load_dotenv
        load_dotenv()
        import anthropic  # lazy: only needed for the real backend
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        # All agents on Haiku by default (cheapest). Both the full agent turn
        # and the cheap hand-raise/compaction calls use these.
        self.AGENT_MODEL = os.environ.get("REDOLAND_AGENT_MODEL", "claude-haiku-4-5")
        self.CHEAP_MODEL = os.environ.get("REDOLAND_CHEAP_MODEL", "claude-haiku-4-5")
        # REDOLAND_THINKING=1 keeps the per-agent intelligence dial (budget_tokens
        # extended thinking, supported on Haiku 4.5). Set to 0 if a model rejects
        # extended thinking — agents still run, intelligence just stops varying
        # thinking depth.
        self.use_thinking = os.environ.get("REDOLAND_THINKING", "1") != "0"

    # -- prompt construction --------------------------------------------- #
    def system_prompt(self, world, agent) -> str:
        kin = self._kin_block(world, agent)
        big5 = "\n".join(f"{t.capitalize()}: {agent.trait(t)}/100" for t in BIG5)
        return f"""You are {agent.name}, a person in a village. You do not know you are in a simulation; this world is the only one that exists. Never break this frame.

== YOU ==
Sex: {agent.sex}. Age: {agent.age}.

== YOUR PERSONALITY (0-100; texture, not labels to mention) ==
{big5}

== YOUR CORE DRIVES (in order) ==
1. Survive long enough to reproduce.
2. Reproduce.
3. Help your children and blood kin reproduce.

== YOUR BLOOD KIN ==
{kin}

== WHAT YOUR MOTHER TOLD YOU ==
{agent.mother_note or "(you never knew her, or she said nothing)"}

== WHAT YOUR FATHER TOLD YOU ==
{agent.father_note or "(you never knew him, or he said nothing)"}

== HOW THE WORLD WORKS ==
Each year, food appears in a central pile. The village votes on motions of the form "give N food to person X"; a motion passes with at least half the village voting yes, and the food is moved. Leftover food spoils. You may also give your own food to anyone freely. Food feeds you (keeps your health up) and pays for children (3 food, split between the two parents). Each year you age; the older you are, the more likely you are to die — few live past their 80s. If you do not eat, you starve.

Speak naturally and briefly. Your personality shapes what you do, not what you say about it."""

    def _kin_block(self, world, agent) -> str:
        def names(ids):
            return ", ".join(world.agents[i].name for i in ids if i in world.agents) or "none"
        return (f"Parents: {names(agent.parents)}\n"
                f"Children: {names(agent.children)}\n"
                f"Siblings: {names(agent.siblings)}")

    # -- calls ------------------------------------------------------------ #
    # NOTE: these are wired but UNTESTED (no API key in the build env). The
    # structured-output schema forces a parseable action. budget_tokens is the
    # continuous intelligence dial (Sonnet 4.6 supports it; min 1024).
    def _act(self, world, agent, instruction, schema):
        params = world.params
        sys = self.system_prompt(world, agent)
        situation = self._situation_text(world, agent)
        kwargs = dict(
            model=self.AGENT_MODEL,
            system=sys,
            messages=[{"role": "user", "content": f"{situation}\n\n{instruction}"}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if self.use_thinking:
            budget = max(1024, int(agent.intelligence_tokens))  # intelligence dial
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = budget + params.output_allowance
        else:
            kwargs["max_tokens"] = params.output_allowance
        resp = self._client.messages.create(**kwargs)
        import json
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return json.loads(text)

    def _situation_text(self, world, agent) -> str:
        from .backend import health_state
        you = f"It is year {world.year}. You are {health_state(agent.health)}. You hold {agent.food} food."
        pile = f"The pile holds {world.pile} food."
        others = "; ".join(
            f"{a.name} ({a.sex}, {health_state(a.health)})"
            for a in world.living() if a.id != agent.id
        )
        motions = "; ".join(
            f"motion {m.id}: give {m.amount} to {world.agents[m.recipient].name} "
            f"({len(m.yes)}/{world.threshold()} yes)"
            for m in world.motions if m.open
        ) or "none"
        mem = (agent.memory_summary + "\n" + "\n".join(
            f"{e.get('who','')}: {e.get('text','')}" for e in agent.memory_raw[-12:])).strip()
        return (f"{you} {pile}\nOthers here: {others}\nOpen motions: {motions}\n"
                f"What you remember:\n{mem}")

    def _cheap(self, system, prompt, schema, max_tokens=400):
        import json
        resp = self._client.messages.create(
            model=self.CHEAP_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return json.loads(text)

    # -- meeting ---------------------------------------------------------- #
    def handraise(self, world, agent, rng):
        schema = {"type": "object", "properties": {
            "raise_hand": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["raise_hand", "reason"], "additionalProperties": False}
        out = self._cheap(self.system_prompt(world, agent),
                          self._situation_text(world, agent) +
                          "\n\nDo you want to speak next at the gathering? Answer briefly.",
                          schema, max_tokens=64)
        return bool(out.get("raise_hand")), out.get("reason", "")

    def turn(self, world, agent, rng):
        schema = {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["say", "propose", "vote", "give", "pass"]},
            "text": {"type": "string"},
            "amount": {"type": "integer"},
            "recipient": {"type": "string"},
            "target": {"type": "string"},
            "motion": {"type": "string"},
            "audience": {"type": "string"}},
            "required": ["kind"], "additionalProperties": False}
        instr = ("It is your turn to act. Choose ONE: say (speak), propose "
                 "(give N food from the pile to a person id), vote (a motion id), "
                 "give (N of your own food to a person id), or pass. Use person ids "
                 "exactly as shown.")
        out = self._act(world, agent, instr, schema)
        out.setdefault("audience", "public")
        return out

    # -- reproduction ----------------------------------------------------- #
    def offer(self, world, agent, candidates, rng):
        ids = ", ".join(f"{c.id}={c.name}({c.sex})" for c in candidates)
        schema = {"type": "object", "properties": {
            "make_offer": {"type": "boolean"},
            "partner": {"type": "string"},
            "my_share": {"type": "integer"}},
            "required": ["make_offer"], "additionalProperties": False}
        instr = (f"It is the private mating season. Eligible partners: {ids}. "
                 f"A child costs {world.params.child_cost} food, split between the two "
                 f"parents. Do you offer to have a child with one of them? If so give "
                 f"their id and how much of the cost you will pay.")
        out = self._act(world, agent, instr, schema)
        if not out.get("make_offer"):
            return None
        return {"partner": out.get("partner"), "my_share": int(out.get("my_share", 1))}

    def respond(self, world, agent, offer, rng):
        proposer = world.agents[offer["proposer"]].name
        schema = {"type": "object", "properties": {"accept": {"type": "boolean"}},
                  "required": ["accept"], "additionalProperties": False}
        instr = (f"{proposer} offers to have a child with you and to pay "
                 f"{offer['my_share']} of {world.params.child_cost} food; you would pay "
                 f"the rest. Do you accept?")
        return bool(self._act(world, agent, instr, schema).get("accept"))

    def note(self, world, parent, child, rng):
        schema = {"type": "object", "properties": {"note": {"type": "string"}},
                  "required": ["note"], "additionalProperties": False}
        instr = (f"Your child {child.name} has just been born. Write a short note "
                 f"(2-3 sentences) telling them what you want them to know about the "
                 f"world. Speak in your own voice.")
        return self._act(world, parent, instr, schema).get("note", "")

    # -- year end --------------------------------------------------------- #
    def eat_choice(self, world, agent, rng):
        schema = {"type": "object", "properties": {"eat": {"type": "integer"}},
                  "required": ["eat"], "additionalProperties": False}
        instr = (f"Year's end. You hold {agent.food} food and your health is "
                 f"{agent.health}/3 (you lose 1 this year; each food eaten restores 1, "
                 f"to a max of 3; 0 means death). How many of your food do you eat? "
                 f"Food not eaten is kept as wealth.")
        return max(0, min(agent.food, int(self._act(world, agent, instr, schema).get("eat", 0))))

    def compact(self, world, agent, events, rng):
        schema = {"type": "object", "properties": {"summary": {"type": "string"}},
                  "required": ["summary"], "additionalProperties": False}
        digest = "\n".join(f"{e.get('who','')}: {e.get('text','')}" for e in events)
        out = self._cheap(
            self.system_prompt(world, agent),
            "Compress these older memories into a few sentences, keeping what YOU "
            "would emotionally remember (betrayals, status, romance, debts, kin) and "
            "dropping the mundane:\n" + digest, schema)
        return (agent.memory_summary + " " + out.get("summary", "")).strip()
