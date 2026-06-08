"""Build a villager as a Concordia EntityAgent.

A villager's mind = one `VillagerContext` component (the full situation) feeding a
`ConcatActComponent` that calls the per-agent model. The model is built by
`world.model_factory(intelligence_tokens)` so each agent's heritable intelligence
dial becomes its own thinking-token budget. Decisions are produced by `mind.act(spec)`.
"""

from __future__ import annotations

from concordia.agents import entity_agent
from concordia.components.agent import concat_act_component

from .components import VillagerContext

_SITUATION_KEY = "situation"


def build_villager(world, agent_id: str):
    body = world.agents[agent_id]
    model = world.model_factory(int(body.intelligence_tokens))
    context = VillagerContext(world, agent_id)
    act = concat_act_component.ConcatActComponent(
        model=model,
        component_order=[_SITUATION_KEY],
        prefix_entity_name=False,                 # we want raw utterances/choices
        randomize_choices=getattr(world, "randomize_choices", True),
    )
    return entity_agent.EntityAgent(
        agent_name=body.name,
        act_component=act,
        context_components={_SITUATION_KEY: context},
    )
