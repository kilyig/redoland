"""Redoland — a generational LLM social simulation with worldline branching,
built on Google DeepMind's Concordia framework.

Villagers are Concordia EntityAgents whose decisions flow through Concordia's
component->prompt->model->action pipeline; the world's physics (free TAKE, Lanchester
combat, satiation, SSA mortality, Gaussian-crossover reproduction) is deterministic
code in the engine. All inference runs through the `claude -p` CLI on session auth
— never the metered ANTHROPIC_API_KEY. Every agent action is a git commit (completed
years are additionally tagged `<branch>-y<N>`); a worldline is a git branch, forked
from any action (fork / inject / run / diff).

Entry points:
    python -m redoland -h               # the mental model + every command
    python -m redoland init <name> --founders 6 --years 5
    python -m redoland serve            # read-only web UI + per-branch Start/Pause;
                                        # creating/forking worldlines is done via the CLI
"""

__version__ = "0.2.0"
