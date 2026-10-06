"""Observation metrics over a world snapshot: population and deaths by cause, food
total and Gini, mean traits/strength/HP/cognition, lineage depth, and per-agent
stat distributions for the UI."""

from __future__ import annotations

import statistics as st

from .core import BIG5


def gini(values):
    vals = sorted(v for v in values)
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(vals, 1):
        cum += i * v
    return (2 * cum) / (n * sum(vals)) - (n + 1) / n


def snapshot_metrics(world) -> dict:
    living = world.living()
    ever = list(world.agents.values())
    dead = [a for a in ever if not a.alive]
    m = {
        "year": world.year,
        "branch": world.branch,
        "population": len(living),
        "ever_lived": len(ever),
        "births": sum(1 for a in ever if a.birth_year > 0),
        "deaths_total": len(dead),
        "deaths_starvation": sum(1 for a in dead if a.death_cause == "starvation"),
        "deaths_natural": sum(1 for a in dead if a.death_cause == "natural"),
        "deaths_combat": sum(1 for a in dead if a.death_cause == "combat"),
        "food_gini": round(gini([a.food for a in living]), 3),
        "food_total": sum(a.food for a in living),
        "mean_age": round(st.mean([a.age for a in living]), 1) if living else 0,
    }
    if living:
        m["mean_strength"] = round(st.mean(a.strength for a in living), 1)
        m["mean_hp"] = round(st.mean(a.hp for a in living), 1)
        for t in BIG5:
            m[f"mean_{t}"] = round(st.mean(a.trait(t) for a in living), 1)
        m["mean_intelligence_tokens"] = round(st.mean(a.intelligence_tokens for a in living))
        m["mean_memory_tokens"] = round(st.mean(a.memory_tokens for a in living))
        # crude lineage depth = generations
        memo = {}

        def depth(aid):
            if aid in memo:
                return memo[aid]
            a = world.agents.get(aid)
            if not a or not a.parents:
                memo[aid] = 0
                return 0
            d = 1 + max((depth(p) for p in a.parents if p in world.agents), default=0)
            memo[aid] = d
            return d
        m["max_generation"] = max((depth(a.id) for a in ever), default=0)
    return m


def distributions(world) -> dict:
    """Raw per-agent value arrays for the living population, so the UI can render
    distributions of wealth / intelligence / age / memory / strength / personality."""
    living = world.living()
    dist = {
        "food": [a.food for a in living],
        "strength": [a.strength for a in living],
        "hp": [int(a.hp) for a in living],
        "age": [a.age for a in living],
        "intelligence": [a.intelligence_tokens for a in living],
        "memory": [a.memory_tokens for a in living],
    }
    for t in BIG5:
        dist[t] = [a.trait(t) for a in living]
    dist["_agents"] = [
        {"id": a.id, "name": a.name, "sex": a.sex, "age": a.age, "food": a.food,
         "strength": a.strength, "hp": int(a.hp), "intelligence": a.intelligence_tokens,
         "memory": a.memory_tokens, **{t: a.trait(t) for t in BIG5}}
        for a in living
    ]
    return dist


def trait_drift(world_start, world_end) -> dict:
    """Mean-trait change between two snapshots (founders vs later population)."""
    def means(w):
        L = w.living()
        if not L:
            return {t: None for t in BIG5}
        return {t: st.mean(a.trait(t) for a in L) for t in BIG5}
    a, b = means(world_start), means(world_end)
    return {t: (round(b[t] - a[t], 1) if a[t] is not None and b[t] is not None else None)
            for t in BIG5}
