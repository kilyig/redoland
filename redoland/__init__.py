"""Redoland — a generational LLM social simulation with worldline branching,
built on Google DeepMind's Concordia framework.

Villagers are Concordia EntityAgents whose decisions flow through Concordia's
component->prompt->model->action pipeline; the world's physics (free TAKE, Lanchester
combat, satiation, SSA mortality, Gaussian-crossover reproduction) is deterministic
code in the engine. All inference runs through the `claude -p` CLI on session auth
— never the metered ANTHROPIC_API_KEY. Each simulated year is a git commit; forking
a worldline = a git branch from a past year (rewind / fork / inject / replay).

Entry points:
    python -m redoland init <name> --founders 6 --years 5
    python -m redoland serve            # web UI to start / continue / fork worldlines
"""

__version__ = "0.2.0"
