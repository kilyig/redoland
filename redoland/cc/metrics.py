"""Metrics over a world snapshot. Reuses the framework-agnostic standalone metrics
(they operate on core.Agent bodies, which cc.World also holds) and adds per-year
distributions for the web UI's stats tab."""

from __future__ import annotations

from ..core import BIG5
from ..metrics import gini, snapshot_metrics, trait_drift   # noqa: F401 (re-export)


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
