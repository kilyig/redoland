"""Apply a human/AI-authored intervention (an injected event) to the world.

v1 manipulable parameters (owner-approved list):
  - per agent: food, satiation (hunger), HP, kill (death)
  - the plaza pile (food)
  - spawn a new agent (a stranger arrives)
  - the food ratio (a regime change, going forward)

Everything is PUBLIC: `apply_changes` mutates the world and returns a list of plain
human-readable effects; the caller records ONE narrate event combining the human
explanation with these effects into every living agent's memory.

`changes` schema (all keys optional):
  {
    "agents": {"a003": {"food": 5, "hp": 10, "satiation": 1}},   # set values
    "kill":   ["a005"],
    "pile":   {"set": 0} | {"add": 20} | 12,
    "spawn":  [{"sex":"female","age":25,"strength":60,"food":4, ...}],
    "params": {"ratio": 0.5}                                       # rule change
  }

Timing: between two years (the cursor at 'year_end'/'year_start' — e.g. right after a
`<branch>-y<N>` tag, or right after `init`) the pile is dead: the next engine step,
_setup, rebuilds it. So a kill's stores and any `pile` change made then are aimed at
the COMING year's pile ("set" replaces its harvest, "add" is carried over on top),
exactly as a year-end death's food is. Mid-year they hit the live pile. The effects
text says which happened.
"""

from __future__ import annotations

from .core import Agent, BIG5, make_name


def _clampi(v, lo, hi) -> int:
    return max(lo, min(hi, int(round(float(v)))))


def apply_changes(world, changes: dict) -> list[str]:
    w = world
    p = w.params
    effects: list[str] = []

    # -- per-agent state -------------------------------------------------- #
    for aid, fields in (changes.get("agents") or {}).items():
        a = w.agents.get(aid)
        if not a or not a.alive:
            continue
        for key, val in (fields or {}).items():
            k = key.lower()
            if k == "food":
                a.food = max(0, int(val))
                effects.append(f"{a.name}'s food is now {a.food}")
            elif k in ("satiation", "health", "hunger"):
                a.health = _clampi(val, 0, p.health_max)
                effects.append(f"{a.name}'s satiation is now {a.health}/{p.health_max}")
            elif k == "hp":
                a.hp = _clampi(val, 0, p.hp_max)
                if a.hp <= 0:
                    effects.append(_kill(w, a))
                else:
                    effects.append(f"{a.name}'s HP is now {int(a.hp)}")

    # -- kills ------------------------------------------------------------ #
    for aid in (changes.get("kill") or []):
        a = w.agents.get(aid)
        if a and a.alive:
            effects.append(_kill(w, a))

    # -- the plaza pile --------------------------------------------------- #
    pile = changes.get("pile")
    if pile is not None:
        set_to = add = None
        if isinstance(pile, dict):
            if "set" in pile:
                set_to = max(0, int(pile["set"]))
            if "add" in pile:
                add = int(pile["add"])
        else:
            set_to = max(0, int(pile))
        if w.year_closed():
            # The year is over: engine._setup rebuilds the pile next, so a change to
            # w.pile here would vanish. Aim it at the coming year instead — "set"
            # replaces that year's harvest, "add" is carried over on top of it.
            if set_to is not None:
                w.next_pile_set = set_to
            if add is not None:
                w.next_pile_bonus = max(0, w.next_pile_bonus + add)
            base = (f"{w.next_pile_set} food" if w.next_pile_set is not None
                    else "the usual harvest")
            extra = f" plus {w.next_pile_bonus} carried over" if w.next_pile_bonus else ""
            effects.append(f"the plaza will hold {base}{extra} when the new year opens")
        else:
            if set_to is not None:
                w.pile = set_to
            if add is not None:
                w.pile = max(0, w.pile + add)
            effects.append(f"the plaza now holds {w.pile} food")

    # -- spawn newcomers -------------------------------------------------- #
    for spec in (changes.get("spawn") or []):
        a = _spawn(w, spec or {})
        effects.append(f"{a.name} ({a.sex}, age {a.age}, str {a.strength}) arrives")

    # -- rule changes (regime) — v1: food ratio --------------------------- #
    for key, val in (changes.get("params") or {}).items():
        if hasattr(p, key):
            cur = getattr(p, key)
            setattr(p, key, type(cur)(val))
            effects.append(f"the world changes: {key} is now {getattr(p, key)}")

    return effects


def _kill(w, a) -> str:
    """Kill `a` and return the effect text. The dead's stores are routed exactly like a
    natural death's (engine._die): into the live pile mid-year, banked for NEXT year's
    pile when the year is closed (World.add_pile_food) — never left to be overwritten."""
    a.alive = False
    a.death_year = w.year
    a.death_cause = "injected"
    a.hp = 0
    text = f"{a.name} dies"
    if w.params.dead_food == "pile":
        if a.food:
            banked = w.add_pile_food(a.food)
            text += (f" ({a.food} food rolls into next year's pile)" if banked
                     else f" ({a.food} food falls to the pile)")
        a.food = 0
    elif w.params.dead_food == "lost":
        a.food = 0
    return text


def _spawn(w, spec: dict) -> Agent:
    rng = w.rng
    p = w.params
    sex = spec.get("sex") or ("male" if rng.chance(0.5) else "female")
    name = spec.get("name") or make_name(sex, rng, w.used_names)
    w.used_names.add(name)
    traits = spec.get("traits") or {t: rng.randint(15, 85) for t in BIG5}
    a = Agent(
        id=w.new_aid(), name=name, sex=sex, traits=traits,
        intelligence_tokens=int(spec.get("intelligence",
                                spec.get("intelligence_tokens", rng.randint(p.int_min, p.int_max)))),
        memory_tokens=int(spec.get("memory",
                          spec.get("memory_tokens", rng.randint(p.mem_min, p.mem_max)))),
        age=int(spec.get("age", p.child_age)),
        health=int(spec.get("satiation", spec.get("health", p.health_max))),
        food=int(spec.get("food", 0)),
        strength=int(spec.get("strength", rng.randint(15, 85))),
        hp=int(spec.get("hp", p.hp_max)),
        birth_year=w.year,
    )
    w.agents[a.id] = a
    w.attach_mind(a.id)                # the newcomer can act from now on
    return a
