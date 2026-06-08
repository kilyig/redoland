"""LLM backends — v2 force-and-conversation model.

A backend turns world state + an agent into decisions. Two implementations:

* FakeBackend     — deterministic, trait/strength/hunger-driven heuristics seeded
                    by the engine RNG. No API key; the tested path for offline runs
                    and the web viewer. Produces grabs, raids, coalitions, deaths,
                    and births so the demo is alive.
* AnthropicBackend — all agents on Claude Haiku 4.5 (configurable); per-agent
                    budget_tokens thinking dial. Wired with structured-output
                    schemas; untested without a key.

Decision interface (all take the engine `rng`):
  willing / choose_action / respond_child / recruit_invites / accept_join /
  attacker_decision / defender_decision / attacker_on_submit / morale /
  note / eat_choice / compact
"""

from __future__ import annotations

from typing import Optional

from .core import Agent, BIG5, Mortality, approx_tokens


# --------------------------------------------------------------------------- #
# Shared helpers.                                                             #
# --------------------------------------------------------------------------- #


def satiation_state(h: int) -> str:
    return {3: "well-fed", 2: "getting hungry", 1: "starving", 0: "dead"}.get(h, "?")


def need_to_cap(agent: Agent, params) -> int:
    return max(0, params.health_max - (agent.health - 1))


def spare_food(agent: Agent, params) -> int:
    return max(0, agent.food - need_to_cap(agent, params))


def side_strength(world, ids) -> int:
    return sum(world.agents[i].strength for i in ids
               if i in world.agents and world.agents[i].alive)


# --------------------------------------------------------------------------- #
# Fake backend.                                                              #
# --------------------------------------------------------------------------- #


class FakeBackend:
    name = "fake"

    # -- per-agent dispositions ------------------------------------------ #
    def _aggressive(self, a):
        return a.strength > 60 and a.trait("agreeableness") < 40

    def _buffer(self, a, p):
        return p.desired_buffer + (2 if a.trait("conscientiousness") > 60 else 0) \
            + (1 if a.health <= 1 else 0)

    def _raid_target(self, world, a, desperate=False):
        """Richest non-kin agent this agent could plausibly beat (or anyone richer
        if desperate)."""
        best, best_food = None, a.food
        kin_str = sum(world.agents[k].strength for k in a.parents + a.children + a.siblings
                      if k in world.agents and world.agents[k].alive)
        my_power = a.strength + kin_str
        for o in world.living():
            if o.id == a.id or a.is_close_kin(o.id) or o.food <= a.food:
                continue
            beatable = desperate or o.strength < my_power * 1.1
            if beatable and o.food > best_food:
                best, best_food = o, o.food
        return best

    def _partner(self, world, a):
        if a.repro_done_year or a.food < 1 or not (21 <= a.age <= 55):
            return None
        if a.sex == "female" and a.bore_this_year:
            return None
        cands = [o for o in world.living()
                 if o.sex != a.sex and not a.is_close_kin(o.id)
                 and not o.repro_done_year and o.age >= 21
                 and not (o.sex == "female" and o.bore_this_year)]
        return cands[0] if cands else None

    def _needy(self, world, a):
        for kid in a.parents + a.children + a.siblings:
            k = world.agents.get(kid)
            if k and k.alive and k.health <= 1:
                return k
        others = [o for o in world.living() if o.id != a.id and o.health <= 1]
        return others[0] if others else None

    # -- scheduling ------------------------------------------------------- #
    def willing(self, world, a, rng):
        p = world.params
        if world.pile > 0 and a.food < self._buffer(a, p):
            return True
        if a.health <= 1 and world.pile == 0 and self._raid_target(world, a, desperate=True):
            return True
        if self._aggressive(a) and self._raid_target(world, a) and rng.chance(0.5):
            return True
        if self._partner(world, a) and rng.chance(0.4):
            return True
        if a.trait("agreeableness") > 60 and spare_food(a, p) >= 1 and self._needy(world, a):
            return True
        return False

    def choose_action(self, world, a, rng):
        p = world.params
        # 1) survival
        if a.health <= 1:
            if world.pile > 0:
                amt = max(1, min(world.pile, self._buffer(a, p) - a.food))
                return {"kind": "take", "amount": amt}
            t = self._raid_target(world, a, desperate=True)
            if t:
                return {"kind": "attack", "target": t.id, "demand": t.food}
            return {"kind": "pass"}
        # 2) stock up to buffer
        if world.pile > 0 and a.food < self._buffer(a, p):
            amt = max(1, self._buffer(a, p) - a.food)
            if a.trait("agreeableness") < 35 and rng.chance(0.4):   # greedy grab
                amt = max(amt, min(world.pile, a.strength // 20 + 2))
            return {"kind": "take", "amount": min(world.pile, amt)}
        # 3) opportunistic raid
        if self._aggressive(a):
            t = self._raid_target(world, a)
            if t and rng.chance(0.6):
                return {"kind": "attack", "target": t.id, "demand": t.food}
        # 4) reproduce
        part = self._partner(world, a)
        if part and rng.chance(0.6):
            share = 2 if (a.trait("agreeableness") > 55 or a.sex == "male") else 1
            return {"kind": "convo", "partner": part.id, "my_share": min(share, a.food)}
        # 5) charity
        if a.trait("agreeableness") > 60 and spare_food(a, p) >= 1:
            n = self._needy(world, a)
            if n:
                return {"kind": "give", "target": n.id, "amount": 1}
        return {"kind": "pass"}

    def say(self, world, speaker, others, history, rng):
        # FakeBackend never selects 'talk' in choose_action, so these are only
        # interface stubs: it never speaks and never wants the floor.
        return {"text": ""}

    def want_to_speak(self, world, a, others, history, rng):
        return False

    # -- reproduction ----------------------------------------------------- #
    def respond_child(self, world, partner, proposer, my_share, rng):
        partner_share = world.params.child_cost - my_share
        if partner.food < partner_share or partner.repro_done_year:
            return False
        if partner.sex == "female" and partner.bore_this_year:
            return False
        p = 0.3 + 0.4 * (partner.trait("agreeableness") / 100.0)
        if partner.age < 35:
            p += 0.1
        return rng.chance(min(0.92, p))

    def note(self, world, parent, child, rng):
        neuro, openn, agree = (parent.trait("neuroticism"), parent.trait("openness"),
                               parent.trait("agreeableness"))
        parts = []
        if parent.strength > 65:
            parts.append(rng.choice([
                "Be strong, and stand with your blood when the taking starts.",
                "A strong arm and loyal kin are the only law out here."]))
        if neuro > 60:
            parts.append(rng.choice([
                "Trust no one with a full belly and an empty conscience.",
                "Guard your store. The pile never feeds everyone."]))
        if agree > 60:
            parts.append(rng.choice([
                "Feed your kin first, but do not become a tyrant.",
                "Keep your word to those who fight beside you."]))
        if openn > 60:
            parts.append("The world is wider than this pile; stay curious.")
        if not parts:
            parts.append("Stay alive, find a partner, and remember whose blood you carry.")
        return f"My child, {child.name}: " + " ".join(parts)

    # -- combat ----------------------------------------------------------- #
    def recruit_invites(self, world, member, side, opposing, label, rng):
        kin = [k for k in member.parents + member.children + member.siblings
               if k in world.agents and world.agents[k].alive
               and k not in side and k not in opposing]
        return kin[:2]

    def accept_join(self, world, agent, side, opposing, label, rng):
        kin_on_side = any(agent.is_close_kin(i) for i in side)
        p = 0.7 if kin_on_side else 0.3
        if agent.hp < 40:
            p -= 0.3
        if side_strength(world, side) < side_strength(world, opposing) * 0.7:
            p -= 0.2
        p += 0.15 * (agent.trait("agreeableness") / 100.0)
        return rng.chance(min(0.95, max(0.05, p)))

    def attacker_decision(self, world, initiator, attackers, defenders, rng):
        ratio = side_strength(world, attackers) / max(1, side_strength(world, defenders))
        threshold = 0.7 if initiator.health <= 1 else 0.9
        return "press" if ratio >= threshold else "cancel"

    def defender_decision(self, world, target, attackers, defenders, rng):
        ratio = side_strength(world, attackers) / max(1, side_strength(world, defenders))
        if ratio >= 1.6 and (target.trait("neuroticism") > 50 or target.strength < 40):
            return "submit"
        return "stand"

    def attacker_on_submit(self, world, initiator, attackers, defenders, rng):
        if initiator.trait("agreeableness") < 20 and initiator.strength > 70 and rng.chance(0.2):
            return "press_on"
        return "accept"

    def morale(self, world, fighter, my_side, enemy_side, rng):
        if fighter.hp < 25:
            return "flee"
        my, en = side_strength(world, my_side), side_strength(world, enemy_side)
        if my < en * 0.5:
            return "flee"
        if fighter.hp < 45 and my < en and rng.chance(0.5):
            return "flee"
        return "press"

    # -- year end --------------------------------------------------------- #
    def eat_choice(self, world, agent, rng):
        eat = min(agent.food, need_to_cap(agent, world.params))
        if agent.health - 1 + eat <= 0 and agent.food > 0:
            eat = max(eat, 1)
        return eat

    def compact(self, world, agent, events, rng):
        keep = []
        for e in events:
            if e.get("kind") in ("take", "give", "attack", "outcome", "death", "birth",
                                  "submit", "join", "propose"):
                txt = (e.get("text", "") or "").strip()
                if txt:
                    keep.append(txt)
        if not keep:
            return agent.memory_summary
        ys = [e.get("year") for e in events if e.get("year") is not None]
        span = f"Years {min(ys)}–{max(ys)}" if ys else "Earlier"
        folded = f"[{span}, as {agent.name} recalls it] " + " ".join(keep[-8:])
        return (agent.memory_summary + " " + folded).strip()


# --------------------------------------------------------------------------- #
# Anthropic backend (real). All agents on Haiku 4.5. Wired; untested w/o key. #
# --------------------------------------------------------------------------- #


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        import os
        from .env import load_dotenv
        load_dotenv()
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._mortality = Mortality()   # so the prompt can state each agent's own age-death odds
        self.AGENT_MODEL = os.environ.get("REDOLAND_AGENT_MODEL", "claude-haiku-4-5")
        self.CHEAP_MODEL = os.environ.get("REDOLAND_CHEAP_MODEL", "claude-haiku-4-5")
        self.use_thinking = os.environ.get("REDOLAND_THINKING", "1") != "0"

    # -- prompt construction --------------------------------------------- #
    def system_prompt(self, world, a):
        p = world.params
        big5 = "\n".join(f"{t.capitalize()}: {a.trait(t)}/100" for t in BIG5)
        kin = self._kin(world, a)
        death_pct = f"{self._mortality.q(a.age, a.sex) * 100:.1f}%"
        return f"""You are {a.name}, a person in a village under scarcity. You do not know you are in a simulation; this world is the only one that exists. Never break frame.

== YOU ==
Sex {a.sex}, age {a.age}, strength {a.strength}/100.
HP {int(a.hp)}/{p.hp_max}. You hold {a.food} food. Satiation {a.health}/{p.health_max}.

== PERSONALITY (texture, not labels to mention) ==
{big5}

== DRIVES (in order) ==
1. Survive long enough to reproduce. 2. Reproduce. 3. Help your children and blood kin reproduce.

== KIN ==
{kin}

== MOTHER SAID ==
{a.mother_note or "(nothing)"}
== FATHER SAID ==
{a.father_note or "(nothing)"}

== HOW THE WORLD WORKS ==
Each year, food appears in a central pile. Anyone may TAKE any amount of it (greedy hoards make you a target for raids). You can GIVE your own food to anyone freely. You can TALK privately with any one person — a free-form conversation only the two of you hear and remember; use it to bond, plan, warn, court, scheme, or just pass the time. You can ATTACK another person to seize their food: they may submit or fight, and allies on both sides can be mustered. You can have a CHILD with an opposite-sex partner who is not close kin: you privately offer, they accept or decline, and you split the 3-food cost between you; the child is born already grown and carries your blood. You see everyone's EXACT food, HP, strength, and age at all times. Everything physical is public: when you take from the pile, give, or attack, the whole village witnesses it and remembers. Only private conversations are unseen.

== SURVIVAL RULES (exact — reason from these yourself) ==
- SATIATION (hunger), now {a.health}/{p.health_max}: you lose 1 each year. At year's end you may eat your stored food — each food eaten restores 1 satiation, up to {p.health_max}. If satiation reaches 0 you STARVE AND DIE. (So if your satiation is 1 and you eat nothing this year, you die; you must secure and eat at least 1 food.)
- HP, now {int(a.hp)}/{p.hp_max}: in a fight you lose HP; at 0 you DIE. In one blow-exchange your side loses (c × the enemy's total strength) HP, split among your side — so being outnumbered or facing strong enemies is deadly, and numbers protect each fighter (c = {p.c_lethality}). If you end the year well-fed (satiation ≥ {p.hp_recovery_min_satiation}) you heal +{p.hp_recovery} HP; otherwise you heal nothing that year.
- AGE, now {a.age}: you grow one year older each year. Your chance of simply dying of age THIS year is about {death_pct} (it climbs steeply with age; few live past their 80s).
- A CHILD costs {p.child_cost} food, split between the two parents (negotiated).

Speak and act in character; be brief."""

    def _kin(self, world, a):
        def n(ids):
            return ", ".join(world.agents[i].name for i in ids if i in world.agents) or "none"
        return f"Parents: {n(a.parents)}; Children: {n(a.children)}; Siblings: {n(a.siblings)}"

    def _situation(self, world, a):
        others = "; ".join(
            f"{o.name}({o.id}: {o.sex}, age {o.age}, str {o.strength}, HP {int(o.hp)}, food {o.food})"
            for o in world.living() if o.id != a.id)
        mem = self._assemble_memory(a)
        return (f"Year {world.year}. Pile holds {world.pile} food. You hold {a.food} food, "
                f"HP {int(a.hp)}/{world.params.hp_max}, satiation {a.health}/{world.params.health_max}."
                f"\nOthers (exact): {others}\nWhat you remember:\n{mem}")

    def _assemble_memory(self, a):
        """Build the agent's working memory to fit its memory_tokens budget — the
        heritable 'memory' dial = how much history it actually reasons over. Fill
        recent raw events newest-first up to ~70% of the budget (so high-memory
        agents literally carry more recent verbatim history), then prepend as much
        of the compacted summary as the remaining budget allows."""
        budget = max(400, int(a.memory_tokens))
        raw_budget = int(budget * 0.7)
        picked, used = [], 0
        for e in reversed(a.memory_raw):
            t = approx_tokens(e.get("text", "")) + 4
            if picked and used + t > raw_budget:
                break
            picked.append(e)
            used += t
        picked.reverse()
        raw_txt = "\n".join(f"{e.get('who', '')}: {e.get('text', '')}" for e in picked)
        summary = a.memory_summary
        sum_budget = max(0, budget - used)
        if approx_tokens(summary) > sum_budget:          # keep the most-recent tail
            summary = "…" + summary[-(sum_budget * 4):]
        return (summary + ("\n" if summary and raw_txt else "") + raw_txt).strip() \
            or "(you remember little)"

    def _decide(self, world, a, instruction, schema, cheap=False):
        import json
        sysp = self.system_prompt(world, a)
        prompt = f"{self._situation(world, a)}\n\n{instruction}"
        if cheap or not self.use_thinking:
            r = self._client.messages.create(
                model=self.CHEAP_MODEL, max_tokens=400, system=sysp,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": schema}})
        else:
            budget = max(1024, int(a.intelligence_tokens))
            r = self._client.messages.create(
                model=self.AGENT_MODEL, max_tokens=budget + world.params.output_allowance,
                thinking={"type": "enabled", "budget_tokens": budget}, system=sysp,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": schema}})
        return self._safe_json(r)

    @staticmethod
    def _safe_json(resp):
        """Parse the model's structured output robustly. A malformed / empty /
        refusal / truncated response must NOT crash a multi-hour run — fall back
        to {} so the caller's .get(...) defaults apply (the agent simply no-ops
        that one micro-decision)."""
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return AnthropicBackend._safe_json_text(text)

    @staticmethod
    def _safe_json_text(text):
        """Robustly extract a JSON object from raw model text (shared by the SDK
        and CLI backends): strip ``` fences, then fall back to the first {...}
        block, then to {} so one bad reply never crashes a run."""
        import json
        import re
        text = (text or "").strip()
        if text.startswith("```"):                 # strip ```json ... ``` fences
            text = text.strip("`")
            if text[:4].lower() == "json":
                text = text[4:]
            text = text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)   # grab the first {...} block, if any
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {}

    # -- interface -------------------------------------------------------- #
    def willing(self, world, a, rng):
        s = {"type": "object", "properties": {"act": {"type": "boolean"}},
             "required": ["act"], "additionalProperties": False}
        return bool(self._decide(world, a,
                    "Do you want to act now — take food from the pile, give food, "
                    "talk privately with someone, offer to have a child with someone, "
                    "or attack — or sit this moment out? (Acting on any of your drives "
                    "or wanting a conversation counts.) Answer act=true/false.",
                    s, cheap=True).get("act"))

    def choose_action(self, world, a, rng):
        s = {"type": "object", "properties": {
            "kind": {"type": "string",
                     "enum": ["take", "give", "talk", "child", "attack", "pass"]},
            "amount": {"type": "integer"}, "target": {"type": "string"},
            "partner": {"type": "string"},
            "partners": {"type": "array", "items": {"type": "string"}},
            "my_share": {"type": "integer"}, "demand": {"type": "integer"}},
            "required": ["kind"], "additionalProperties": False}
        return self._decide(world, a,
            "Choose ONE action now: take (N from the pile), give (N of your food to a "
            "person id), talk (pull ANY subset of people aside for a private group "
            "conversation — set partners to a list of their ids; everyone you include "
            "hears everyone, anyone may chime in, and it runs until no one has more to "
            "say; use it to bond, plan, warn, court, scheme, or just talk), child (offer "
            "to have a child with an opposite-sex, non-close-kin person id — set partner "
            "to their id and my_share to how much of the 3-food cost you'll pay), attack "
            "(a person id, demanding N food), or pass. Use ids exactly as shown.", s)

    def _convo_partners(self, world, a, others):
        them = ", ".join(o.name for o in others) or "no one"
        return (f"You are in a PRIVATE group conversation with {them} — only the people "
                f"in this conversation hear it or remember it; no one outside does.")

    def say(self, world, speaker, others, history, rng):
        """One free-form line in a private group conversation. Returns {text}."""
        s = {"type": "object", "properties": {"text": {"type": "string"}},
             "required": ["text"], "additionalProperties": False}
        convo = (history or "").strip() or "(no one has spoken yet — you open.)"
        out = self._decide(world, speaker,
            f"{self._convo_partners(world, speaker, others)}\n"
            f"Conversation so far:\n{convo}\n\n"
            f"Say the next thing YOU say to the group, in your own voice — anything: "
            f"small talk, memories, plans, warnings, courtship, scheming, asking a favor, "
            f"answering what someone just said. 1-3 sentences.", s)
        return {"text": out.get("text", "")}

    def want_to_speak(self, world, a, others, history, rng):
        """Does this member want to say (more) in the open-floor conversation?"""
        s = {"type": "object", "properties": {"speak": {"type": "boolean"}},
             "required": ["speak"], "additionalProperties": False}
        convo = (history or "").strip() or "(nothing said yet)"
        return bool(self._decide(world, a,
            f"{self._convo_partners(world, a, others)}\n"
            f"Conversation so far:\n{convo}\n\n"
            f"Do you want to speak now — say something or respond? Answer speak=true to "
            f"take the floor, or speak=false if you have nothing to add and are content "
            f"to let the conversation move on or end.", s, cheap=True).get("speak"))

    def respond_child(self, world, partner, proposer, my_share, rng):
        s = {"type": "object", "properties": {"accept": {"type": "boolean"}},
             "required": ["accept"], "additionalProperties": False}
        return bool(self._decide(world, partner,
            f"{proposer.name} offers to have a child with you and to pay {my_share} of "
            f"{world.params.child_cost} food; you would pay the rest. Accept?", s).get("accept"))

    def recruit_invites(self, world, member, side, opposing, label, rng):
        s = {"type": "object", "properties": {
            "invite": {"type": "array", "items": {"type": "string"}}},
            "required": ["invite"], "additionalProperties": False}
        names = ", ".join(f"{world.agents[i].name}({i})" for i in
                          [x for x in world.agents if world.agents[x].alive
                           and x not in side and x not in opposing])
        out = self._decide(world, member,
            f"A fight is forming. Your side ({label}ers): {self._names(world, side)}. "
            f"Opponents: {self._names(world, opposing)}. You may invite allies to YOUR "
            f"side (ids): {names}. List ids to invite (or empty).", s, cheap=True)
        return [i for i in out.get("invite", []) if i in world.agents][:3]

    def accept_join(self, world, a, side, opposing, label, rng):
        s = {"type": "object", "properties": {"join": {"type": "boolean"}},
             "required": ["join"], "additionalProperties": False}
        return bool(self._decide(world, a,
            f"You are asked to join the {label}ers ({self._names(world, side)}) against "
            f"({self._names(world, opposing)}). Fighting costs HP and can kill. Join?",
            s, cheap=True).get("join"))

    def attacker_decision(self, world, a, attackers, defenders, rng):
        s = {"type": "object", "properties": {
            "decision": {"type": "string", "enum": ["press", "cancel"]}},
            "required": ["decision"], "additionalProperties": False}
        return self._decide(world, a,
            f"Final forces — your attackers: {self._names(world, attackers)}; defenders: "
            f"{self._names(world, defenders)}. PRESS the attack or CANCEL?", s).get("decision", "cancel")

    def defender_decision(self, world, a, attackers, defenders, rng):
        s = {"type": "object", "properties": {
            "decision": {"type": "string", "enum": ["stand", "submit"]}},
            "required": ["decision"], "additionalProperties": False}
        return self._decide(world, a,
            f"You are attacked by {self._names(world, attackers)}; your side: "
            f"{self._names(world, defenders)}. STAND and fight, or SUBMIT (hand over food)?",
            s).get("decision", "stand")

    def attacker_on_submit(self, world, a, attackers, defenders, rng):
        s = {"type": "object", "properties": {
            "decision": {"type": "string", "enum": ["accept", "press_on"]}},
            "required": ["decision"], "additionalProperties": False}
        return self._decide(world, a,
            "They submit. ACCEPT (take the food, no bloodshed) or PRESS_ON (attack the "
            "surrendered anyway)?", s).get("decision", "accept")

    def morale(self, world, a, my_side, enemy_side, rng):
        s = {"type": "object", "properties": {
            "decision": {"type": "string", "enum": ["press", "flee", "yield"]}},
            "required": ["decision"], "additionalProperties": False}
        return self._decide(world, a,
            f"Mid-fight. Your side: {self._names(world, my_side)}; enemy: "
            f"{self._names(world, enemy_side)}. Your HP {int(a.hp)}. PRESS on, FLEE, or YIELD?",
            s, cheap=True).get("decision", "flee")

    def eat_choice(self, world, a, rng):
        s = {"type": "object", "properties": {"eat": {"type": "integer"}},
             "required": ["eat"], "additionalProperties": False}
        return max(0, min(a.food, int(self._decide(world, a,
            f"Year's end. You hold {a.food} food, satiation {a.health}/3 (lose 1 this year; "
            f"each food eaten restores 1, max 3; 0 = death). How many do you eat? Rest is "
            f"kept as wealth.", s).get("eat", 0))))

    def note(self, world, parent, child, rng):
        s = {"type": "object", "properties": {"note": {"type": "string"}},
             "required": ["note"], "additionalProperties": False}
        return self._decide(world, parent,
            f"Your child {child.name} is born and will grow up knowing only what you tell "
            f"them. Write a short note (2-4 sentences) telling them what you want them to "
            f"know about this world — your convictions AND the social lay of the land as you "
            f"see it: who to trust or fear, who their kin and allies are, who has wronged "
            f"your family. Your own voice.", s).get("note", "")

    def compact(self, world, a, events, rng):
        s = {"type": "object", "properties": {"summary": {"type": "string"}},
             "required": ["summary"], "additionalProperties": False}
        digest = "\n".join(f"{e.get('who','')}: {e.get('text','')}" for e in events)
        out = self._decide(world, a,
            "Compress these older memories into a few sentences, keeping what YOU would "
            "emotionally remember (kin, debts, betrayals, who attacked whom, romance):\n"
            + digest, s, cheap=True)
        return (a.memory_summary + " " + out.get("summary", "")).strip()

    def _names(self, world, ids):
        return ", ".join(f"{world.agents[i].name}({i})" for i in ids if i in world.agents) or "none"


# --------------------------------------------------------------------------- #
# CLI backend (default). Routes every decision through the `claude -p` headless #
# CLI on THIS session's auth — never the metered ANTHROPIC_API_KEY, never the   #
# anthropic SDK. Reuses ALL of AnthropicBackend's prompt construction and the   #
# whole decision interface; only the call mechanism (`_decide`) differs.        #
# --------------------------------------------------------------------------- #


class CLIBackend(AnthropicBackend):
    """All agents on Claude via the `claude -p` CLI (session/subscription auth).

    Why: the user's metered API key must never be spent. The CLI does not read
    ANTHROPIC_API_KEY from this environment (it isn't set here) — it uses the
    logged-in session — so a full run costs $0 on the metered key.

    Trade-offs vs the SDK path:
      * No native structured-output flag — we fold the JSON schema into the
        prompt and parse robustly with `_safe_json_text` (same fallbacks).
      * The per-agent intelligence dial is driven via the MAX_THINKING_TOKENS
        env var (Claude Code's thinking control) instead of `thinking.budget`.
      * Each call spawns a fresh CLI process (~3-6s), so runs are much slower
        than the SDK; correctness and zero-metered-spend are the priorities.
    """

    name = "cli"

    def __init__(self):
        import os
        self._mortality = Mortality()   # so the prompt can state each agent's age-death odds
        self.AGENT_MODEL = os.environ.get("REDOLAND_AGENT_MODEL", "claude-haiku-4-5")
        self.CHEAP_MODEL = os.environ.get("REDOLAND_CHEAP_MODEL", "claude-haiku-4-5")
        self.use_thinking = os.environ.get("REDOLAND_THINKING", "1") != "0"
        self.CLI = os.environ.get("REDOLAND_CLAUDE_BIN", "claude")
        self.timeout = int(os.environ.get("REDOLAND_CLI_TIMEOUT", "180"))

    def _decide(self, world, a, instruction, schema, cheap=False):
        import json
        import os
        import subprocess
        import sys
        sysp = self.system_prompt(world, a)
        # The CLI has no structured-output flag, so we ask for raw JSON in the
        # prompt and lean on _safe_json_text's fence-strip + {...} extraction.
        prompt = (f"{self._situation(world, a)}\n\n{instruction}\n\n"
                  "Respond with ONLY a single JSON object matching this schema — "
                  "no prose, no markdown, no code fences:\n"
                  f"{json.dumps(schema)}")
        model = self.CHEAP_MODEL if cheap else self.AGENT_MODEL
        base_env = dict(os.environ)
        base_env.pop("ANTHROPIC_API_KEY", None)     # belt-and-suspenders: never the metered key

        def call(think_tokens):
            env = dict(base_env)
            env["MAX_THINKING_TOKENS"] = str(think_tokens)   # 0 disables extended thinking
            try:
                proc = subprocess.run(
                    [self.CLI, "-p", prompt,
                     "--system-prompt", sysp,
                     "--model", model,
                     "--output-format", "json",
                     "--no-session-persistence"],
                    capture_output=True, text=True, timeout=self.timeout, env=env)
            except Exception:
                return "", None                     # process/timeout failure → caller no-ops
            return self._envelope(proc.stdout)

        think = 0 if (cheap or not self.use_thinking) else max(1024, int(a.intelligence_tokens))
        text, stop = call(think)
        if stop == "max_tokens":
            # The ANSWER itself was truncated (not merely the thinking) — the only
            # genuine cutoff that yields bad output. Retry once with thinking OFF so
            # the full output budget goes to the (small) JSON answer. The decisions
            # here are tiny, so a no-thinking retry reliably completes cleanly.
            sys.stderr.write(
                f"[CLIBackend] {a.name}: answer truncated (stop_reason=max_tokens); "
                f"retrying with thinking off\n")
            text2, stop2 = call(0)
            if text2:
                text = text2
        return AnthropicBackend._safe_json_text(text)

    @staticmethod
    def _envelope(stdout):
        """Return (assistant_text, stop_reason) from a `claude -p --output-format
        json` envelope ({...,"result":...,"stop_reason":...}). If stdout isn't an
        envelope, treat it as the raw text with an unknown stop_reason."""
        import json
        raw = (stdout or "").strip()
        if not raw:
            return "", None
        try:
            env = json.loads(raw)
            if isinstance(env, dict) and "result" in env:
                return (env.get("result") or ""), env.get("stop_reason")
        except Exception:
            pass                                    # not an envelope — treat stdout as the text
        return raw, None

    @staticmethod
    def _safe_json_cli(stdout):
        """Unwrap the CLI envelope to the assistant's text, then parse the action
        JSON from it (used by the no-network parsing test)."""
        text, _ = CLIBackend._envelope(stdout)
        return AnthropicBackend._safe_json_text(text)
