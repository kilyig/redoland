"""World container for the Concordia-based engine.

Design (see CONCORDIA_MIGRATION_PLAN.md §2.1): every villager is a pair —
  * a BODY: a plain `core.Agent` dataclass (numeric + genome state) that the
    deterministic engine mutates directly. This is the serializable state.
  * a MIND: a Concordia `EntityAgent` that produces the agent's decisions through
    Concordia's component→prompt→model→action pipeline. Minds are rebuilt from
    bodies on load, so only bodies + world globals are checkpointed.

Keeping bodies as `core.Agent` lets the combat/crossover/year-end logic port over
almost unchanged; only "ask the agent" calls change (backend.X → decide.X, which
calls mind.act()).
"""

from __future__ import annotations

from typing import Callable, Optional

from .core import Agent, Params, RNG


class World:
    def __init__(self, params: Params, rng: RNG, branch: str = "main",
                 model_factory: Optional[Callable[[int], object]] = None,
                 randomize_choices: bool = True, premise: str = ""):
        self.params = params
        self.rng = rng
        self.branch = branch
        # the "stage": an optional premise/backstory the world's creator sets at
        # founding. Woven into every agent's prompt (so newborns learn it too) and
        # announced in year 1's transcript. Empty = the default scarcity setting.
        self.premise = premise or ""
        # Concordia shuffles CHOICE options to reduce position bias (good for the
        # real model; turn OFF for deterministic tests with the scripted stub).
        self.randomize_choices = randomize_choices
        self.year = 0
        self.pile = 0
        self.agents: dict[str, Agent] = {}        # bodies (serializable, engine-mutated)
        self.minds: dict[str, object] = {}        # Concordia EntityAgents (decision-makers)
        self.models: dict[str, object] = {}       # per-agent LanguageModel (deliberation toggles)
        self.used_names: set = set()
        self.next_eid = 0
        self.next_aid = 0
        self.year_events: list[dict] = []
        # food held by those who died at year-end (natural/starvation) waiting to roll
        # into NEXT year's pile (dead_food="pile"). Mid-year combat spoils go straight to
        # the live pile instead, so they are claimable within the same year.
        self.next_pile_bonus = 0
        self.event_sink = None                    # optional callable(ev) for live streaming
        # builds a per-agent LanguageModel given the agent's intelligence dial
        self.model_factory = model_factory
        # resumable run cursor: lets the simulation be checkpointed/forked after ANY
        # single action and resumed exactly (phase of the year + scramble position).
        self.cursor = {"phase": "year_start", "last": None, "steps": 0}

    # -- population ------------------------------------------------------- #
    def living(self) -> list[Agent]:
        return [self.agents[i] for i in sorted(self.agents) if self.agents[i].alive]

    def new_aid(self) -> str:
        self.next_aid += 1
        return f"a{self.next_aid:03d}"

    # -- transcript + subjective memory ----------------------------------- #
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
        if self.event_sink is not None:
            try:
                self.event_sink(ev)
            except Exception:
                pass
        return ev

    def _witnesses(self, speaker, audience) -> list[Agent]:
        if audience == "public":
            return self.living()
        ids = set(audience) if isinstance(audience, list) else set()
        if speaker in self.agents:
            ids.add(speaker)
        return [self.agents[i] for i in ids if i in self.agents]

    # -- minds ------------------------------------------------------------ #
    def attach_mind(self, agent_id: str):
        """(Re)build the Concordia EntityAgent for a body and register it."""
        from .villager import build_villager
        self.minds[agent_id] = build_villager(self, agent_id)
        return self.minds[agent_id]

    def mind(self, agent_id: str):
        m = self.minds.get(agent_id)
        if m is None:
            m = self.attach_mind(agent_id)
        return m
