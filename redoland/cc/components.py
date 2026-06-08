"""Villager context component.

One Concordia `ActionSpecIgnored` component assembles the entire decision context
for a villager each time it acts: persona + drives + world rules + exact survival
mechanics + its own state + everyone else's exact public state + its scaled memory.
This is the port of the old backend.system_prompt + _situation + _assemble_memory,
rebuilt fresh each act so kin/notes/stats are always current. It contributes NO
behavioural nudges — it states the rules and facts; the agent decides.
"""

from __future__ import annotations

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component

from ..core import BIG5, Mortality, approx_tokens

_MORTALITY = Mortality()   # shared; states each agent's own age-death odds


class VillagerContext(action_spec_ignored.ActionSpecIgnored,
                      entity_component.ComponentWithLogging):
    """Builds the full situation text for villager `agent_id` from `world`."""

    def __init__(self, world, agent_id: str, pre_act_label: str = "\nSituation"):
        super().__init__(pre_act_label)
        self._world = world
        self._id = agent_id

    # -- the agent's own subjective memory, scaled by its memory dial ----- #
    def _assemble_memory(self, a) -> str:
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
        if approx_tokens(summary) > sum_budget:
            summary = "…" + summary[-(sum_budget * 4):]
        return (summary + ("\n" if summary and raw_txt else "") + raw_txt).strip() \
            or "(you remember little)"

    def _kin(self, world, a) -> str:
        def n(ids):
            return ", ".join(world.agents[i].name for i in ids if i in world.agents) or "none"
        return f"Parents: {n(a.parents)}; Children: {n(a.children)}; Siblings: {n(a.siblings)}"

    def _make_pre_act_value(self) -> str:
        world = self._world
        p = world.params
        a = world.agents[self._id]
        big5 = "\n".join(f"{t.capitalize()}: {a.trait(t)}/100" for t in BIG5)
        death_pct = f"{_MORTALITY.q(a.age, a.sex) * 100:.1f}%"
        others = "; ".join(
            f"{o.name}({o.id}: {o.sex}, age {o.age}, str {o.strength}, HP {int(o.hp)}, "
            f"food {o.food})" for o in world.living() if o.id != a.id) or "(no one else)"
        mem = self._assemble_memory(a)
        text = f"""You are {a.name}, a person in a village under scarcity. You do not know you are in a simulation; this world is the only one that exists. Never break frame.

== YOU ==
Sex {a.sex}, age {a.age}, strength {a.strength}/100. HP {int(a.hp)}/{p.hp_max}. You hold {a.food} food. Satiation {a.health}/{p.health_max}.

== PERSONALITY (texture, not labels to mention) ==
{big5}

== DRIVES (in order) ==
1. Survive long enough to reproduce. 2. Reproduce. 3. Help your children and blood kin reproduce.

== KIN ==
{self._kin(world, a)}

== MOTHER SAID ==
{a.mother_note or "(nothing)"}
== FATHER SAID ==
{a.father_note or "(nothing)"}

== HOW THE WORLD WORKS ==
Each year, food appears in a central pile. Anyone may TAKE any amount of it (greedy hoards make you a target for raids). You can GIVE your own food to anyone freely. You can TALK privately with any subset of people — a free-form conversation only those people hear and remember; use it to bond, plan, warn, court, scheme, or just pass the time. You can ATTACK another person to seize their food: they may submit or fight, and allies on both sides can be mustered. You can have a CHILD with an opposite-sex partner who is not close kin: you privately offer, they accept or decline, and you split the {p.child_cost}-food cost between you; the child is born already grown and carries your blood. You see everyone's EXACT food, HP, strength, and age at all times. Everything physical is public; only private conversations are unseen.

== SURVIVAL RULES (exact — reason from these yourself) ==
- SATIATION (hunger), now {a.health}/{p.health_max}: you lose 1 each year. At year's end you may eat your stored food — each food eaten restores 1 satiation, up to {p.health_max}. If satiation reaches 0 you STARVE AND DIE.
- HP, now {int(a.hp)}/{p.hp_max}: in a fight you lose HP; at 0 you DIE. In one blow-exchange your side loses (c × the enemy's total strength) HP, split among your side — so being outnumbered or facing strong enemies is deadly, and numbers protect each fighter (c = {p.c_lethality}). If you end the year well-fed (satiation ≥ {p.hp_recovery_min_satiation}) you heal +{p.hp_recovery} HP; otherwise you heal nothing.
- AGE, now {a.age}: you grow one year older each year. Your chance of simply dying of age THIS year is about {death_pct}.
- A CHILD costs {p.child_cost} food, split between the two parents (negotiated).

== CURRENT SITUATION ==
Year {world.year}. The pile holds {world.pile} food.
Others (exact): {others}

== WHAT YOU REMEMBER ==
{mem}"""
        self._logging_channel({"Key": self.get_pre_act_label(), "Value": text})
        return text
