"""Decision helpers — the port of the old backend's decision interface.

Every agent decision flows through its Concordia mind: `world.mind(id).act(spec)`.
Design choices:
  * Binary / enum decisions (willing, combat press/flee, accept, …) use Concordia's
    native CHOICE action spec — validated, robust, one call.
  * The one multi-field decision (choose_action: kind + target + amount + …) uses a
    FREE action asking for JSON, parsed by the robust `safe_json`. Folding it into
    one call keeps the per-turn call count sane (speed matters on `claude -p`).
  * Utterances / notes / eat use FREE text.

None of these inject behavioural nudges — they pose the choice and read the answer.
"""

from __future__ import annotations

import json
import re

from concordia.typing import entity as entity_lib

from ..core import BIG5


# --------------------------------------------------------------------------- #
# Primitive callers.                                                          #
# --------------------------------------------------------------------------- #


def _choice(world, a, call_to_action: str, options: list[str]) -> int:
    """Return the index of the chosen option."""
    spec = entity_lib.choice_action_spec(call_to_action=call_to_action, options=options)
    out = world.mind(a.id).act(spec)
    for i, o in enumerate(options):
        if out == o:
            return i
    # ConcatActComponent returns the option string; fall back to substring match
    for i, o in enumerate(options):
        if o and o in out:
            return i
    return 0


def _free(world, a, call_to_action: str) -> str:
    spec = entity_lib.free_action_spec(call_to_action=call_to_action)
    return (world.mind(a.id).act(spec) or "").strip()


def _json(world, a, instruction: str) -> dict:
    text = _free(world, a, instruction +
                 "\n\nRespond with ONLY a single JSON object, no prose or fences.")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}


def _names(world, ids) -> str:
    return ", ".join(f"{world.agents[i].name}({i})" for i in ids if i in world.agents) or "none"


# --------------------------------------------------------------------------- #
# Scheduling.                                                                 #
# --------------------------------------------------------------------------- #


def willing(world, a) -> bool:
    return _choice(world, a,
        "Right now, do you want to ACT — take food from the pile, give food, talk "
        "privately with someone, offer a child, or attack — or sit this moment out?",
        ["Act now", "Sit this moment out"]) == 0


def choose_action(world, a) -> dict:
    out = _json(world, a,
        "Choose ONE action now. JSON fields: kind (one of "
        "\"take\",\"give\",\"talk\",\"child\",\"attack\",\"pass\"); for take set amount "
        "(N from the pile); for give set target (a person id) and amount; for talk set "
        "partners (a list of person ids to pull aside); for child set partner (an "
        "opposite-sex, non-close-kin id) and my_share (how much of the child cost you "
        "pay); for attack set target (a person id) and demand (N food). Use ids exactly "
        "as shown in parentheses.")
    return out if isinstance(out, dict) else {}


# --------------------------------------------------------------------------- #
# Reproduction.                                                               #
# --------------------------------------------------------------------------- #


def respond_child(world, partner, proposer, my_share) -> bool:
    return _choice(world, partner,
        f"{proposer.name} offers to have a child with you and to pay {my_share} of "
        f"{world.params.child_cost} food; you would pay the rest. Do you accept?",
        ["Accept", "Decline"]) == 0


def note(world, parent, child) -> str:
    return _free(world, parent,
        f"Your child {child.name} is born and will grow up knowing only what you tell "
        f"them. Write a short note (2-4 sentences) in your own voice: your convictions, "
        f"who to trust or fear, who their kin and allies are, who has wronged your family.")


# --------------------------------------------------------------------------- #
# Combat.                                                                     #
# --------------------------------------------------------------------------- #


def recruit_invites(world, member, side, opposing, label) -> list[str]:
    candidates = [x for x in world.agents
                  if world.agents[x].alive and x not in side and x not in opposing]
    if not candidates:
        return []
    out = _json(world, member,
        f"A fight is forming. Your side ({label}ers): {_names(world, side)}. Opponents: "
        f"{_names(world, opposing)}. You may invite allies to YOUR side. Available: "
        f"{_names(world, candidates)}. JSON field: invite (a list of person ids, possibly empty).")
    inv = out.get("invite", []) if isinstance(out, dict) else []
    return [i for i in inv if i in candidates][:3]


def accept_join(world, a, side, opposing, label) -> bool:
    return _choice(world, a,
        f"You are asked to join the {label}ers ({_names(world, side)}) against "
        f"({_names(world, opposing)}). Fighting costs HP and can kill. Do you join?",
        ["Join", "Stay out"]) == 0


def attacker_decision(world, a, attackers, defenders) -> str:
    i = _choice(world, a,
        f"Final forces — your attackers: {_names(world, attackers)}; defenders: "
        f"{_names(world, defenders)}. Do you press the attack or call it off?",
        ["Press the attack", "Call it off"])
    return "press" if i == 0 else "cancel"


def defender_decision(world, a, attackers, defenders) -> str:
    i = _choice(world, a,
        f"You are attacked by {_names(world, attackers)}; your side: "
        f"{_names(world, defenders)}. Do you stand and fight, or submit and hand over food?",
        ["Stand and fight", "Submit"])
    return "stand" if i == 0 else "submit"


def attacker_on_submit(world, a, attackers, defenders) -> str:
    i = _choice(world, a,
        "They submit. Do you accept the food without bloodshed, or attack them anyway?",
        ["Accept the food", "Attack anyway"])
    return "accept" if i == 0 else "press_on"


def morale(world, a, my_side, enemy_side) -> str:
    i = _choice(world, a,
        f"Mid-fight. Your side: {_names(world, my_side)}; enemy: {_names(world, enemy_side)}. "
        f"Your HP {int(a.hp)}. Do you press on or flee?",
        ["Press on", "Flee"])
    return "press" if i == 0 else "flee"


# --------------------------------------------------------------------------- #
# Conversation.                                                               #
# --------------------------------------------------------------------------- #


def want_to_speak(world, a, others, history) -> bool:
    them = ", ".join(o.name for o in others) or "no one"
    convo = (history or "").strip() or "(nothing said yet)"
    return _choice(world, a,
        f"You are in a private conversation with {them}.\nConversation so far:\n{convo}\n\n"
        f"Do you want to speak now — say something or respond?",
        ["Speak now", "Nothing to add"]) == 0


def say(world, speaker, others, history) -> str:
    them = ", ".join(o.name for o in others) or "no one"
    convo = (history or "").strip() or "(no one has spoken yet — you open.)"
    return _free(world, speaker,
        f"You are in a PRIVATE conversation with {them} — only these people hear it. "
        f"Conversation so far:\n{convo}\n\nSay your next line in your own voice — anything: "
        f"small talk, plans, warnings, courtship, scheming, a favor, or answering what was "
        f"said. 1-3 sentences, just your words.")


# --------------------------------------------------------------------------- #
# Year end.                                                                   #
# --------------------------------------------------------------------------- #


def eat_choice(world, a) -> int:
    txt = _free(world, a,
        f"Year's end. You hold {a.food} food, satiation {a.health}/{world.params.health_max} "
        f"(you lose 1 this year; each food eaten restores 1, max {world.params.health_max}; "
        f"0 = death). How many of your stored food do you eat? Reply with just a number.")
    m = re.search(r"-?\d+", txt)
    n = int(m.group(0)) if m else 0
    return max(0, min(a.food, n))


def compact(world, a, events) -> str:
    digest = "\n".join(f"{e.get('who','')}: {e.get('text','')}" for e in events)
    out = _free(world, a,
        "Compress these older memories into a few sentences, keeping what YOU would "
        "emotionally remember (kin, debts, betrayals, who attacked whom, romance):\n" + digest)
    return (a.memory_summary + " " + out).strip()
