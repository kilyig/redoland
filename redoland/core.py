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
    # --- fountain / carrying-capacity food model (optional; overrides `ratio`) ---
    # If either is set, F = max(food_base, round(food_floor_ratio * N)):
    #   food_base       — a fixed fountain output (abundance bootstrap, e.g. 25)
    #   food_floor_ratio— grow food so food-per-person never drops below this
    #                     (e.g. 0.9 → mild-scarcity carrying-capacity equilibrium)
    # With both 0, the classic F = round(N * ratio) is used.
    food_base: int = 0
    food_floor_ratio: float = 0.0
    # F = round(N_living * ratio). v2 default 1.5: combat adds a large mortality
    # channel, so the v1 "1.15" tuning is invalid — at 1.5/c=0.3 the FakeBackend
    # village survives every seed over 50y (small, ~4-6, clan-feud-driven, with
    # strength self-domesticating downward). Provisional; needs a proper ratio×c
    # sweep with the real LLM backend. See PRESETS below.
    ratio: float = 1.5
    child_cost: int = 3
    child_age: int = 21
    child_health: int = 3
    max_maternal_age: int = 45    # a woman can bear children only up to this age (men: no limit)
    repro_rounds: int = 2
    meeting_slots_per_agent: int = 10
    # Conversations end when no one wants to speak next. Two bounds back that up:
    #   convo_turns_per_person — a SOFT cap: a conversation may run at most this many
    #     utterances per participant (so cap = turns_per_person × group size); trims the
    #     repetitive tail that sets in once the substance is said. Scales with group size
    #     so larger groups get room for everyone to weigh in.
    #   convo_safety_cap — a hard runaway guard so a year cannot hang forever.
    convo_turns_per_person: int = 6
    convo_safety_cap: int = 200
    # cognition ranges (continuous heritable dials)
    int_min: int = 1024
    int_max: int = 8000
    mem_min: int = 4000
    mem_max: int = 32000
    # crossover sigmas
    big5_sigma: float = 8.0
    int_sigma: float = 600.0
    mem_sigma: float = 3000.0
    # --- v2 force-and-combat model (see MVP_PLAN_V2.md) ---
    strength_sigma: float = 8.0   # crossover sigma for Strength (0..100)
    hp_max: int = 100             # combat life pool
    hp_recovery: int = 25         # HP healed per year IF fed
    hp_recovery_min_satiation: int = 2   # heal only when satiation (hunger bar) >= this
    c_lethality: float = 0.3      # Lanchester constant: damage = c * enemy_strength
    # halting backstops (termination guarantees, NOT cost limits)
    max_events_per_year: int = 600
    muster_passes_cap: int = 5
    blow_rounds_cap: int = 12
    convo_turns_cap: int = 6
    # how much food an agent wants stored before it stops grabbing the pile
    desired_buffer: int = 2
    # api
    output_allowance: int = 1500
    # which Claude model drives this village's agents (via `claude -p --model`).
    # Stored with the world so a run always thinks with the model it was created with.
    model: str = "claude-haiku-4-5"
    # dead food:
    #   "lost"  — a dead person's uneaten food vanishes.
    #   "pile"  — it returns to the commons. A year-end death (starvation/old age) rolls
    #             into NEXT year's pile (this year's claiming window is closed). A mid-year
    #             combat death drops into THIS year's pile (others can still grab it; it
    #             spoils at the year reset) — after the victor loots the body first.
    dead_food: str = "lost"       # "lost" | "pile"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        known = {f for f in cls().__dict__}
        return cls(**{k: v for k, v in d.items() if k in known})


BIG5 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Named F/N presets (political-intensity + selection dial). Re-tuned for the v2
# combat model (FakeBackend, 40-50y x 6-8 seeds) — combat dominates mortality, so
# survivable ratios are higher than v1's. Provisional until a real-backend sweep:
#   crisis    1.20 — frequent extinction; brutal clan-feud collapse
#   tense     1.40 — usually survives; tiny, violent
#   stable    1.50 — survives every seed; small (~4-6) clan-feud village  [DEFAULT]
#   abundant  1.70 — survives; larger, food-rich, still violent
PRESETS = {"crisis": 1.20, "tense": 1.40, "stable": 1.50, "abundant": 1.70}


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
    health: int                                # satiation / hunger bar, 0..3 (0 = starve)
    food: int
    strength: int = 50                         # heritable, 0..100 (drives combat damage)
    hp: int = 100                              # combat life pool, 0..hp_max (0 = die in combat)
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
