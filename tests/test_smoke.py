"""Smoke tests for the Concordia-based engine — run with:
    python3 tests/test_cc.py        (no pytest, no network, no metered key)

Every decision is driven by a deterministic scripted StubModel, so these test the
engine/worldline mechanics, not the LLM. Mirrors the standalone smoke suite.
"""

import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redoland.core import Params, Agent, RNG, BIG5, Mortality
from redoland.model import StubModel, ClaudeCLIModel, dummy_embedder, safe_json
from redoland.engine import Engine
from redoland.world import World
from redoland.sim import Simulation
from redoland.metrics import snapshot_metrics, distributions


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def _parse(p, pat, d=0):
    m = re.search(pat, p)
    return int(m.group(1)) if m else d


def survival_stub():
    """A scripted model: act when hungry, take to a buffer, eat to survive."""
    def choice_fn(prompt, responses):
        if "do you want to ACT" in prompt:
            f = _parse(prompt, r"You hold (\d+) food"); pile = _parse(prompt, r"pile holds (\d+) food")
            return 0 if (f < 2 and pile > 0) else 1
        return 0
    def text_fn(prompt):
        if "Choose ONE action now" in prompt:
            f = _parse(prompt, r"You hold (\d+) food"); pile = _parse(prompt, r"pile holds (\d+) food")
            return ('{"kind":"take","amount":%d}' % min(pile, 2 - f)) if (pile > 0 and f < 2) else '{"kind":"pass"}'
        if "How many of your stored food" in prompt:
            s = _parse(prompt, r"satiation (\d+)/"); f = _parse(prompt, r"You hold (\d+) food")
            return str(1 if s <= 2 and f > 0 else 0)
        return ""
    return StubModel(choice_fn=choice_fn, text_fn=text_fn)


def mkbody(w, aid, name, sex, **kw):
    d = dict(traits={t: 50 for t in BIG5}, intelligence_tokens=1500, memory_tokens=6000,
             age=30, health=3, food=4)
    d.update(kw)
    b = Agent(id=aid, name=name, sex=sex, **d)
    w.agents[aid] = b
    w.used_names.add(name)
    w.next_aid = max(w.next_aid, int(aid[1:]))
    w.attach_mind(aid)
    return b


# --------------------------------------------------------------------------- #


def test_model_parsing():
    b = ClaudeCLIModel(thinking_tokens=0)
    check(b._envelope('{"result":"hi","stop_reason":"end_turn"}') == "hi", "envelope unwraps result")
    check(safe_json("```json\n{\"a\":1}\n```") == {"a": 1}, "safe_json strips fences")
    check(safe_json("nope") == {}, "safe_json falls back to {}")
    v = dummy_embedder("x")
    check(hasattr(v, "shape") and v.shape[0] == 16, "dummy embedder returns a 16-d vector")


def test_engine_invariants():
    stub = survival_stub()
    eng = Engine.found(Params(founders=5, ratio=1.6, start_food=1),
                       model_factory=lambda t: stub, seed=7, randomize_choices=False)
    for _ in range(4):
        eng.run_year()
    for a in eng.w.agents.values():
        check(0 <= a.health <= 3, f"satiation in range ({a.name})")
        check(0 <= a.hp <= 100, f"HP in range ({a.name})")
        check(0 <= a.strength <= 100, f"strength in range ({a.name})")
        check(a.food >= 0, f"food non-negative ({a.name})")
        check(a.sex in ("male", "female"), "sex valid")
        check(a.id not in (a.parents + a.children + a.siblings), "no self-kinship")
    check(len(eng.w.minds) >= len(eng.w.living()), "every living agent has a mind")


def test_determinism():
    def world():
        stub = survival_stub()
        e = Engine.found(Params(founders=5, ratio=1.5, start_food=1),
                         model_factory=lambda t: stub, seed=99, randomize_choices=False)
        for _ in range(4):
            e.run_year()
        return e.w
    a, b = world(), world()
    check(sorted(a.agents) == sorted(b.agents) and a.pile == b.pile,
          "same seed + scripted model -> identical world (rng-driven physics)")
    check([a.agents[i].food for i in sorted(a.agents)] ==
          [b.agents[i].food for i in sorted(b.agents)], "identical food vector")


def test_combat_resolves():
    def cf(prompt, responses):
        p = prompt.lower()
        if "press the attack or call it off" in p: return 0
        if "stand and fight, or submit" in p: return 0
        if "press on or flee" in p: return 0
        if "do you join" in p: return 1
        return 0
    stub = StubModel(choice_fn=cf, text_fn=lambda p: '{"invite":[]}' if "invite" in p else "")
    w = World(Params(), RNG(5), model_factory=lambda t: stub, randomize_choices=False); w.year = 1
    mkbody(w, "a001", "Eron", "male", strength=90, hp=100, food=2)
    mkbody(w, "a002", "Kesh", "female", strength=30, hp=100, food=5)
    eng = Engine(w); w.year_events = []
    eng._fight_chunk(w.agents["a001"], w.agents["a002"], demand=5)
    kinds = [e["kind"] for e in w.year_events]
    check("blow" in kinds or "submit" in kinds, "combat resolves (blows or submission)")
    check(w.agents["a002"].hp < 100 or "submit" in kinds, "defender takes damage or submits")


def test_birth_crossover():
    stub = StubModel(choice_fn=lambda p, r: 0,
                     text_fn=lambda p: "Be strong." if "note" in p.lower() else "")
    w = World(Params(child_cost=3), RNG(5), model_factory=lambda t: stub, randomize_choices=False); w.year = 2
    mkbody(w, "a001", "Eron", "male", strength=80, intelligence_tokens=3000)
    mkbody(w, "a002", "Kesh", "female", strength=40, intelligence_tokens=1200)
    eng = Engine(w); w.year_events = []
    before = set(w.agents)
    eng._convo_chunk(w.agents["a001"], {"partner": "a002", "my_share": 1})
    new = [i for i in w.agents if i not in before]
    check(len(new) == 1, "a child is born")
    c = w.agents[new[0]]
    check(0 <= c.strength <= 100 and c.intelligence_tokens >= 1024, "child genome within bounds")
    check(new[0] in w.minds, "child mind attached mid-year")
    check(set(c.parents) == {"a001", "a002"}, "child kin links set")


def test_group_talk():
    speak = {"n": 0}
    def cf(prompt, responses):
        if "do you want to speak now" in prompt.lower():
            speak["n"] += 1
            return 0 if speak["n"] <= 3 else 1
        return 0
    stub = StubModel(choice_fn=cf, text_fn=lambda p: "We must look after kin." if "say your next line" in p.lower() else "")
    w = World(Params(), RNG(5), model_factory=lambda t: stub, randomize_choices=False); w.year = 1
    mkbody(w, "a001", "Eron", "male"); mkbody(w, "a002", "Kesh", "female"); mkbody(w, "a003", "Ivo", "male")
    eng = Engine(w); w.year_events = []
    eng._talk_chunk(w.agents["a001"], {"partners": ["a002", "a003"]})
    says = [e for e in w.year_events if e["kind"] == "say"]
    check(len(says) >= 1, "group conversation produces utterances")
    check(all(e["audience"] == ["a001", "a002", "a003"] for e in says), "every line heard by the whole group")
    check(says[0]["speaker"] == "a001", "initiator opens")


def test_worldline_fork_inject_replay():
    path = "/tmp/redoland_cc_test_run"
    shutil.rmtree(path, ignore_errors=True)
    stub = survival_stub()
    mf = lambda t: stub
    sim = Simulation.create(path, Params(founders=5, ratio=1.6, start_food=1), seed=7,
                            model_factory=mf, randomize_choices=False)
    sim.run(2)
    check(sim.store.years_for_branch("main") == [0, 1, 2], "main worldline committed years 0-2")
    sim.fork("main", 1, "famine")
    sim.inject("famine", ratio=0.2, narrate="A blight ruins the harvest.")
    sim.replay("famine", 2)
    ma = snapshot_metrics(sim.store.load_world(sim.store.tag("main", 2)))
    fa = snapshot_metrics(sim.store.load_world(sim.store.tag("famine", 3)))
    check(ma["food_total"] != fa["food_total"] or ma["population"] != fa["population"],
          "fork + inject + replay produces a divergent timeline")
    shutil.rmtree(path, ignore_errors=True)


def test_distributions():
    stub = survival_stub()
    eng = Engine.found(Params(founders=5, ratio=1.6), model_factory=lambda t: stub,
                       seed=3, randomize_choices=False)
    eng.run_year()
    d = distributions(eng.w)
    for key in ["food", "intelligence", "memory", "age", "strength"] + BIG5:
        check(key in d and len(d[key]) == len(eng.w.living()), f"distribution '{key}' present")
    check(len(d["_agents"]) == len(eng.w.living()), "per-agent rows present")


if __name__ == "__main__":
    for fn in [test_model_parsing, test_engine_invariants, test_determinism,
               test_combat_resolves, test_birth_crossover, test_group_talk,
               test_worldline_fork_inject_replay, test_distributions]:
        print(fn.__name__)
        fn()
    print("\nALL CC TESTS PASSED")
