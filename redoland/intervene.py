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
                    _kill(w, a)
                    effects.append(f"{a.name} dies")
                else:
                    effects.append(f"{a.name}'s HP is now {int(a.hp)}")

    # -- kills ------------------------------------------------------------ #
    for aid in (changes.get("kill") or []):
        a = w.agents.get(aid)
        if a and a.alive:
            _kill(w, a)
            effects.append(f"{a.name} dies")

    # -- the plaza pile --------------------------------------------------- #
    pile = changes.get("pile")
    if pile is not None:
        if isinstance(pile, dict):
            if "set" in pile:
                w.pile = max(0, int(pile["set"]))
            if "add" in pile:
                w.pile = max(0, w.pile + int(pile["add"]))
        else:
            w.pile = max(0, int(pile))
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


def _kill(w, a):
    a.alive = False
    a.death_year = w.year
    a.death_cause = "injected"
    a.hp = 0
    if w.params.dead_food == "pile":
        w.pile += a.food
        a.food = 0
    elif w.params.dead_food == "lost":
        a.food = 0


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
