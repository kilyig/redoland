"""Core types, parameters, seeded RNG, and the SSA mortality model.

Everything here is pure-stdlib and JSON-serializable so a whole world can be
checkpointed into a git commit and replayed deterministically (for the engine's
own randomness — LLM sampling is never deterministic; see MVP_PLAN.md §7.3).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Parameters (defaults reflect the decisions in MVP_PLAN.md §2 and §13).       #
# --------------------------------------------------------------------------- #


@dataclass
class Params:
    founders: int = 10
    founder_age_min: int = 21
    founder_age_max: int = 45
    start_food: int = 2
    start_health: int = 3
    health_max: int = 3
    # F = round(N_living * ratio). 1.15 is the recommended "stable + selective"
    # setting: across seeds the population self-regulates (carrying-capacity
    # equilibrium, no extinction, no explosion) while natural selection drives
    # mean agreeableness down (~50 -> ~35; scarcity punishes costly altruism)
    # and conscientiousness up (thrift favored). See PRESETS below.
    ratio: float = 1.15
    child_cost: int = 3
    child_age: int = 21
    child_health: int = 3
    repro_rounds: int = 2
    meeting_slots_per_agent: int = 10
    # cognition ranges (continuous heritable dials)
    int_min: int = 1024
    int_max: int = 8000
    mem_min: int = 4000
    mem_max: int = 32000
    # crossover sigmas
    big5_sigma: float = 8.0
    int_sigma: float = 600.0
    mem_sigma: float = 3000.0
    # api
    output_allowance: int = 1500
    dead_food: str = "lost"       # "lost" | "pile" | "inherit"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        known = {f for f in cls().__dict__}
        return cls(**{k: v for k, v in d.items() if k in known})


BIG5 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Named F/N presets (the political-intensity + selection dial). Measured over
# 60-100 simulated years x 8 seeds with the FakeBackend:
#   crisis    0.90 — mostly goes extinct; brutal, dramatic decline (high pressure)
#   tense     1.00 — survives ~half the seeds; constant scarcity politics
#   balanced  1.05 — usually survives; lively governance, frequent starvation
#   stable    1.15 — survives every seed; carrying-capacity equilibrium + strong,
#                     interpretable selection (agreeableness down, thrift up)  [DEFAULT]
#   abundant  1.25 — always survives; food often spoils, weaker selection
PRESETS = {"crisis": 0.90, "tense": 1.00, "balanced": 1.05, "stable": 1.15, "abundant": 1.25}


# --------------------------------------------------------------------------- #
# Seeded, serializable RNG.                                                    #
# --------------------------------------------------------------------------- #


class RNG:
    """Thin wrapper over random.Random whose state we can checkpoint."""

    def __init__(self, seed: Optional[int] = None, state: Optional[list] = None):
        self._r = random.Random()
        if state is not None:
            # JSON turns tuples into lists; random wants nested tuples.
            self._r.setstate(_to_tuple(state))
        elif seed is not None:
            self._r.seed(seed)

    def get_state(self) -> list:
        return _to_list(self._r.getstate())

    # convenience pass-throughs
    def random(self) -> float:
        return self._r.random()

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._r.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._r.gauss(mu, sigma)

    def choice(self, seq):
        return self._r.choice(list(seq))

    def shuffle(self, seq):
        self._r.shuffle(seq)

    def chance(self, p: float) -> bool:
        return self._r.random() < p


def _to_list(x):
    if isinstance(x, tuple):
        return ["__tuple__"] + [_to_list(i) for i in x]
    return x


def _to_tuple(x):
    if isinstance(x, list) and x and x[0] == "__tuple__":
        return tuple(_to_tuple(i) for i in x[1:])
    if isinstance(x, list):
        return [_to_tuple(i) for i in x]
    return x


# --------------------------------------------------------------------------- #
# Agent.                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Agent:
    id: str
    name: str
    sex: str                                   # "male" | "female"
    traits: dict                               # big five, 0..100
    intelligence_tokens: int
    memory_tokens: int
    age: int
    health: int
    food: int
    alive: bool = True
    birth_year: int = 0
    death_year: Optional[int] = None
    death_cause: Optional[str] = None          # "starvation" | "natural"
    parents: list = field(default_factory=list)
    children: list = field(default_factory=list)
    siblings: list = field(default_factory=list)
    mother_note: str = ""                      # note this agent's mother wrote TO it
    father_note: str = ""
    # subjective memory
    memory_summary: str = ""                   # compacted older memories (persona-biased)
    memory_raw: list = field(default_factory=list)   # recent visible events [{eid, year, kind, who, text}]
    bore_this_year: bool = False               # females: already gave birth this year
    repro_done_year: bool = False              # made/accepted a child this year

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Agent":
        return cls(**d)

    # cognition helpers ----------------------------------------------------- #
    @property
    def thinking_budget(self) -> int:
        return int(self.intelligence_tokens)

    def trait(self, name: str) -> int:
        return int(self.traits.get(name, 50))

    def is_close_kin(self, other_id: str) -> bool:
        return other_id in self.parents or other_id in self.children or other_id in self.siblings


# --------------------------------------------------------------------------- #
# Token approximation (offline; no tokenizer available without the API).       #
# --------------------------------------------------------------------------- #


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# SSA mortality model.                                                         #
# --------------------------------------------------------------------------- #


class Mortality:
    def __init__(self, table_path: Optional[str] = None):
        if table_path is None:
            table_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "ssa_4c6.json"
            )
        with open(table_path) as fh:
            data = json.load(fh)
        self._male = {int(k): float(v) for k, v in data["male"].items()}
        self._female = {int(k): float(v) for k, v in data["female"].items()}
        self._anchors_male = sorted(self._male)
        self._anchors_female = sorted(self._female)

    def q(self, age: int, sex: str) -> float:
        table = self._male if sex == "male" else self._female
        anchors = self._anchors_male if sex == "male" else self._anchors_female
        if age >= 120:
            return 1.0
        if age <= anchors[0]:
            return table[anchors[0]]
        # find bracketing anchors and linearly interpolate
        lo = anchors[0]
        for a in anchors:
            if a == age:
                return table[a]
            if a > age:
                hi = a
                frac = (age - lo) / (hi - lo)
                return table[lo] + frac * (table[hi] - table[lo])
            lo = a
        return table[anchors[-1]]


# --------------------------------------------------------------------------- #
# Name generation.                                                             #
# --------------------------------------------------------------------------- #

_MALE_NAMES = [
    "Kael", "Joren", "Bram", "Doran", "Eron", "Fen", "Garrick", "Hale", "Ivo",
    "Jarl", "Korin", "Lem", "Mako", "Nols", "Orin", "Perr", "Quill", "Roan",
    "Sten", "Taric", "Ulf", "Varo", "Wim", "Yarl", "Zeb", "Aldo", "Cael", "Dav",
]
# guard against any accidental non-ascii entries
_MALE_NAMES = [n for n in _MALE_NAMES if n.isascii()] or ["Kael", "Joren", "Bram"]
_FEMALE_NAMES = [
    "Mara", "Ryn", "Sela", "Tova", "Una", "Vela", "Wenna", "Yara", "Zinn",
    "Ada", "Bryn", "Cira", "Dell", "Esa", "Fina", "Gwen", "Hesta", "Isa",
    "Juna", "Kesh", "Lira", "Nessa", "Orla", "Pell", "Rhea", "Senna", "Talia",
]


def make_name(sex: str, rng: RNG, used: set) -> str:
    pool = _MALE_NAMES if sex == "male" else _FEMALE_NAMES
    candidates = [n for n in pool if n not in used]
    if not candidates:
        # append a generational suffix to recycle
        base = rng.choice(pool)
        i = 2
        while f"{base} {_roman(i)}" in used:
            i += 1
        return f"{base} {_roman(i)}"
    return rng.choice(candidates)


def _roman(n: int) -> str:
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in numerals:
        while n >= v:
            out += s
            n -= v
    return out
