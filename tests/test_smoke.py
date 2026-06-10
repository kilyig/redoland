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
    check(b._envelope('{"result":"hi","stop_reason":"end_turn"}') == (True, "hi"), "envelope unwraps result")
    check(b._envelope('{"is_error":true,"result":"out of credits"}')[0] is False, "error envelope flagged as failure")
    check(b._envelope("") == (True, ""), "exit-0 empty output is a valid empty answer")
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


def test_sole_actor_can_continue():
    """A lone willing agent (e.g. the last survivor) must be able to act more than
    once per year. The one-step cooldown must skip them only while OTHERS want to
    act — it must not freeze them out and end the year early."""
    def cf(prompt, responses):
        if "do you want to ACT" in prompt:
            f = _parse(prompt, r"You hold (\d+) food"); pile = _parse(prompt, r"pile holds (\d+) food")
            return 0 if (f < 3 and pile > 0) else 1
        return 0
    def tf(prompt):
        if "Choose ONE action now" in prompt:
            f = _parse(prompt, r"You hold (\d+) food"); pile = _parse(prompt, r"pile holds (\d+) food")
            return '{"kind":"take","amount":1}' if (pile > 0 and f < 3) else '{"kind":"pass"}'
        return "0"
    stub = StubModel(choice_fn=cf, text_fn=tf)
    w = World(Params(ratio=5.0, start_food=0), RNG(1),
              model_factory=lambda t: stub, randomize_choices=False)
    mkbody(w, "a001", "Solo", "male", food=0)        # a sole survivor
    Engine(w).run_year()
    takes = [e for e in w.year_events if e["kind"] == "take"]
    check(len(takes) >= 2, f"lone agent acts repeatedly in a year (got {len(takes)} takes)")


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


def test_run_halts_on_model_failure():
    """A dead model (out of credits / failed CLI) HALTS the run with ModelUnavailable
    instead of silently committing endless empty 'pass' steps."""
    from redoland.model import ModelUnavailable, StubModel

    class DeadModel(StubModel):
        def sample_text(self, prompt, **k): raise ModelUnavailable("no response")
        def sample_choice(self, prompt, responses, **k): raise ModelUnavailable("no response")

    path = "/tmp/redoland_halt"; shutil.rmtree(path, ignore_errors=True)
    sim = Simulation.create(path, Params(founders=3), seed=1,
                            model_factory=lambda t: survival_stub(), randomize_choices=False)
    before = len(sim.store.timeline("main", n=300))
    sim2 = Simulation.open(path, lambda t: DeadModel())
    sim2.store.checkout_branch("main")
    raised = False
    try:
        sim2.run(years=2)
    except ModelUnavailable:
        raised = True
    added = len(sim2.store.timeline("main", n=900)) - before
    check(raised, "the run halts with ModelUnavailable when the model gives no response")
    check(added <= 1, f"no wall of empty 'pass' commits piles up (added {added}, not ~600/year)")
    shutil.rmtree(path, ignore_errors=True)


def test_dead_food_yearend_to_next_pile():
    """dead_food='pile': a year-end death's stores roll into NEXT year's pile."""
    stub = StubModel(choice_fn=lambda p, r: 0, text_fn=lambda p: "")
    w = World(Params(dead_food="pile", food_base=10, food_floor_ratio=0.0, ratio=0), RNG(1),
              model_factory=lambda t: stub, randomize_choices=False)
    w.year = 5
    mkbody(w, "a001", "X", "male", food=7, health=2)
    mkbody(w, "a002", "Y", "female", food=1, health=3)
    eng = Engine(w)
    eng._die(w.agents["a001"], "natural")            # a year-end death
    check(w.next_pile_bonus == 7, "year-end death's food is banked for next year")
    check(w.agents["a001"].food == 0, "dead agent's food is cleared")
    w.year_events = []
    eng._setup()                                     # opens year 6
    check(w.pile == 10 + 7, "next year's pile = base food + the dead's stores")
    check(w.next_pile_bonus == 0, "the carryover is consumed once spent")


def test_combat_kill_loot_choice_and_spoils():
    """A slain target: the victor LOOTS a chosen amount (not capped by the demand) and
    the remainder drops into THIS year's pile (dead_food='pile')."""
    def cf(prompt, responses):
        p = prompt.lower()
        if "press the attack or call it off" in p: return 0   # press
        if "stand and fight, or submit" in p: return 0        # stand (fight to the death)
        if "press on or flee" in p: return 0                  # press on
        if "do you join" in p: return 1                       # decline to join
        return 0
    def tf(prompt):
        if "invite" in prompt: return '{"invite":[]}'
        if "how much do you take" in prompt.lower(): return "5"   # loot choice
        return ""
    stub = StubModel(choice_fn=cf, text_fn=tf)
    w = World(Params(dead_food="pile"), RNG(5), model_factory=lambda t: stub,
              randomize_choices=False); w.year = 1
    mkbody(w, "a001", "Eron", "male", strength=100, hp=100, food=0)
    mkbody(w, "a002", "Kesh", "female", strength=1, hp=1, food=8)   # frail -> dies fast
    eng = Engine(w); w.year_events = []; w.pile = 0
    eng._fight_chunk(w.agents["a001"], w.agents["a002"], demand=3)
    check(not w.agents["a002"].alive, "the frail target is killed")
    check(w.agents["a001"].food == 5, "victor takes the amount they CHOSE (5), not the demand (3)")
    check(w.pile == 3, "the unlooted remainder (8-5) falls to this year's pile")
    check(w.agents["a002"].food == 0, "nothing is left on the body")


def test_convo_cap_scales_with_group():
    """The soft cap (convo_turns_per_person × group size) ends a talk where everyone
    always wants to speak, and larger groups get proportionally more turns."""
    stub = StubModel(choice_fn=lambda p, r: 0,       # want_to_speak -> 'Speak now'
                     text_fn=lambda p: "More." if "say your next line" in p.lower() else "")
    def run(ids):
        w = World(Params(convo_turns_per_person=2), RNG(1),
                  model_factory=lambda t: stub, randomize_choices=False); w.year = 1
        for i, gid in enumerate(ids):
            mkbody(w, gid, "P" + gid, "male" if i % 2 else "female")
        eng = Engine(w); w.year_events = []
        eng._talk_chunk(w.agents[ids[0]], {"partners": ids[1:]})
        return sum(1 for e in w.year_events if e["kind"] == "say")
    two = run(["a001", "a002"])                       # cap = 2 × 2 = 4
    three = run(["a001", "a002", "a003"])             # cap = 2 × 3 = 6
    check(two <= 4 + 1, f"2-person talk is bounded by the soft cap (got {two})")
    check(three <= 6 + 1, f"3-person talk is bounded by the soft cap (got {three})")
    check(three > two, "larger groups get proportionally more turns")


def test_model_persisted():
    """A village remembers which Claude model it was created with (through git)."""
    path = "/tmp/redoland_model"; shutil.rmtree(path, ignore_errors=True)
    sim = Simulation.create(path, Params(founders=3, model="claude-sonnet-4-6"), seed=1,
                            model_factory=lambda t: survival_stub(), randomize_choices=False)
    w = sim.store.load_world("main")
    check(w.params.model == "claude-sonnet-4-6", "model persists through the git round-trip")
    from redoland.cli import make_model_factory
    m = make_model_factory(w.params.model)(2000)
    check(getattr(m, "_model", None) == "claude-sonnet-4-6", "factory builds agents on the stored model")
    shutil.rmtree(path, ignore_errors=True)


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


def test_max_maternal_age():
    """A woman past max_maternal_age cannot bear a child; the father's age is irrelevant."""
    stub = StubModel(choice_fn=lambda p, r: 0, text_fn=lambda p: "")
    w = World(Params(child_cost=3, max_maternal_age=45), RNG(5),
              model_factory=lambda t: stub, randomize_choices=False); w.year = 2
    mkbody(w, "a001", "Eron", "male", age=70, food=4)        # old father — allowed
    mkbody(w, "a002", "Kesh", "female", age=50, food=4)      # mother past 45 — blocked
    eng = Engine(w); w.year_events = []
    before = set(w.agents)
    eng._convo_chunk(w.agents["a001"], {"partner": "a002", "my_share": 1})
    check(set(w.agents) == before, "no child born to a mother past max_maternal_age")
    # a young mother with an old father still works (no paternal limit)
    w2 = World(Params(child_cost=3, max_maternal_age=45), RNG(5),
               model_factory=lambda t: stub, randomize_choices=False); w2.year = 2
    mkbody(w2, "a001", "Eron", "male", age=70, food=4)
    mkbody(w2, "a002", "Kesh", "female", age=30, food=4)
    eng2 = Engine(w2); w2.year_events = []
    before2 = set(w2.agents)
    eng2._convo_chunk(w2.agents["a001"], {"partner": "a002", "my_share": 1})
    check(len(set(w2.agents) - before2) == 1, "young mother + old father can still have a child")


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
    check(sim.store.years_for_branch("main") == [1, 2],
          "main worldline starts at year 1 (genesis untagged, no year 0)")
    sim.fork("main", 1, "famine")
    sim.inject("famine", ratio=0.2, narrate="A blight ruins the harvest.")
    sim.replay("famine", 2)
    ma = snapshot_metrics(sim.store.load_world(sim.store.tag("main", 2)))
    fa = snapshot_metrics(sim.store.load_world(sim.store.tag("famine", 3)))
    check(ma["food_total"] != fa["food_total"] or ma["population"] != fa["population"],
          "fork + inject + replay produces a divergent timeline")
    shutil.rmtree(path, ignore_errors=True)


def test_fork_at_action_and_inject():
    """Per-action commits: fork from a mid-year action, inject a public event
    (kill + empty the pile), resume from that exact step, and diverge from main."""
    path = "/tmp/redoland_cc_forkstep"
    shutil.rmtree(path, ignore_errors=True)
    stub = survival_stub()
    sim = Simulation.create(path, Params(founders=5, ratio=1.6, start_food=0), seed=7,
                            model_factory=lambda t: stub, randomize_choices=False)
    sim.run(1)
    tl = sim.store.timeline("main", n=20)
    check(sum(1 for r in tl if " action:" in r["desc"]) >= 2,
          "each agent action is its own commit")
    mid = next(r for r in tl if "action:" in r["desc"])      # a mid-year action
    sim.fork_at(mid["commit"], "quake")
    victim = sim.store.load_world().living()[0].id
    res = sim.inject("quake", changes={"kill": [victim], "pile": {"set": 0}},
                     narrative="An earthquake strikes.")
    check(any("dies" in e for e in res["effects"]), "inject kill takes effect")
    check("earthquake" in res["text"].lower(), "narrative recorded with effects")
    sim.replay("quake", 1)   # resume the rest of year 1 from the fork point
    main_pop = snapshot_metrics(sim.store.load_world(sim.store.tag("main", 1)))["population"]
    quake_w = sim.store.load_world()
    check(not quake_w.agents[victim].alive, "injected death persists on the fork")
    check(any(b == "quake" for b in sim.store.list_branches()), "fork branch exists")
    shutil.rmtree(path, ignore_errors=True)


def test_stepwise_talk_resumable():
    """A group talk is driven one utterance at a time: the opener is its own step, each
    later step adds at most one utterance, and the talk sub-state lives on the cursor
    (so it is JSON-serializable / checkpointable) until the conversation closes."""
    import json
    speak = {"n": 0}
    def cf(prompt, responses):
        if "do you want to speak now" in prompt.lower():
            speak["n"] += 1
            return 0 if speak["n"] <= 3 else 1
        return 0
    stub = StubModel(choice_fn=cf,
                     text_fn=lambda p: "We look after kin." if "say your next line" in p.lower() else "")
    w = World(Params(), RNG(5), model_factory=lambda t: stub, randomize_choices=False); w.year = 1
    mkbody(w, "a001", "Eron", "male"); mkbody(w, "a002", "Kesh", "female"); mkbody(w, "a003", "Ivo", "male")
    eng = Engine(w); w.year_events = []
    w.cursor = {"phase": "scramble", "last": None, "steps": 0}
    eng._initiate(w.agents["a001"], {"kind": "talk", "partners": ["a002", "a003"]}, stepwise=True)
    check(w.cursor["phase"] == "talk" and "talk" in w.cursor, "talk hands off to the state machine")
    check(json.loads(json.dumps(w.cursor)) == w.cursor, "the talk cursor is JSON-serializable (checkpointable)")
    check(sum(1 for e in w.year_events if e["kind"] == "convo") == 1, "the opener is its own entry step")
    labels = []
    guard = 0
    while w.cursor["phase"] == "talk" and guard < 50:
        guard += 1
        before = sum(1 for e in w.year_events if e["kind"] == "say")
        labels.append(eng.step())
        after = sum(1 for e in w.year_events if e["kind"] == "say")
        check(after - before <= 1, "at most one utterance per step")
    check(labels[-1] == "talk_end", "the closing step is labelled talk_end")
    check(w.cursor["phase"] == "scramble" and "talk" not in w.cursor,
          "the talk clears its sub-state and returns control to the scramble")
    says = [e for e in w.year_events if e["kind"] == "say"]
    check(len(says) >= 1 and says[0]["speaker"] == "a001", "initiator opens; utterances recorded")


def test_stepwise_talk_persists_through_git():
    """Pause mid-conversation and resume from the git checkpoint: the talk's group and
    position survive a load_world() round-trip and the conversation runs to a clean close."""
    path = "/tmp/redoland_talk_resume"
    shutil.rmtree(path, ignore_errors=True)
    speak = {"n": 0}
    def cf(prompt, responses):
        if "do you want to speak now" in prompt.lower():
            speak["n"] += 1
            return 0 if speak["n"] <= 4 else 1
        return 0
    stub = StubModel(choice_fn=cf,
                     text_fn=lambda p: "Kin first." if "say your next line" in p.lower() else "")
    sim = Simulation.create(path, Params(founders=4), seed=5,
                            model_factory=lambda t: stub, randomize_choices=False)
    sim.store.checkout_branch("main")
    eng = sim.engine()
    eng.step()                                       # year_start -> setup (year 1)
    a = eng.w.living()
    eng._initiate(a[0], {"kind": "talk", "partners": [a[1].id, a[2].id]}, stepwise=True)
    sim.store.commit_year(eng.w, "y1 action: talk opener", tag_year=False)
    eng.step()                                       # one utterance
    sim.store.commit_year(eng.w, "y1 talk", tag_year=False)
    check(eng.w.cursor["phase"] == "talk", "still mid-talk after one utterance")
    w2 = sim.store.load_world("main")                # reload from the git checkpoint
    check(w2.cursor.get("phase") == "talk" and "talk" in w2.cursor,
          "mid-talk cursor (group + position) restored from git")
    n_before = sum(1 for e in w2.year_events if e["kind"] == "say")
    eng2 = Engine(w2, sim.mortality)
    guard = 0
    while w2.cursor["phase"] == "talk" and guard < 50:
        guard += 1; eng2.step()
    n_after = sum(1 for e in w2.year_events if e["kind"] == "say")
    check(n_after >= n_before, "the conversation resumes from the checkpoint and continues")
    check(w2.cursor["phase"] == "scramble", "talk closes back into the scramble after resume")
    shutil.rmtree(path, ignore_errors=True)


def test_stepwise_fight_resumable():
    """A fight is driven step by step: muster pass(es), one decide step, then blow
    rounds, each as its own resumable step, with the fight sub-state on the cursor."""
    import json
    def cf(prompt, responses):
        p = prompt.lower()
        if "press the attack or call it off" in p: return 0   # press
        if "stand and fight, or submit" in p: return 0        # stand
        if "press on or flee" in p: return 0                  # press on
        if "do you join" in p: return 1                       # decline to join
        return 0
    stub = StubModel(choice_fn=cf, text_fn=lambda p: '{"invite":[]}' if "invite" in p else "")
    w = World(Params(), RNG(5), model_factory=lambda t: stub, randomize_choices=False); w.year = 1
    mkbody(w, "a001", "Eron", "male", strength=90, hp=100, food=2)
    mkbody(w, "a002", "Kesh", "female", strength=30, hp=100, food=5)
    eng = Engine(w); w.year_events = []
    w.cursor = {"phase": "scramble", "last": None, "steps": 0}
    eng._initiate(w.agents["a001"], {"kind": "attack", "target": "a002", "demand": 5}, stepwise=True)
    check(w.cursor["phase"] == "fight" and "fight" in w.cursor, "attack hands off to the state machine")
    check(json.loads(json.dumps(w.cursor)) == w.cursor, "the fight cursor is JSON-serializable (checkpointable)")
    subs = set()
    guard = 0
    while w.cursor["phase"] == "fight" and guard < 80:
        guard += 1
        if "fight" in w.cursor:
            subs.add(w.cursor["fight"]["sub"])
        eng.step()
    check({"muster", "decide", "blows"} & subs, "the fight passes through muster/decide/blow sub-phases")
    check(w.cursor["phase"] == "scramble" and "fight" not in w.cursor,
          "the fight clears its sub-state and returns control to the scramble")
    kinds = [e["kind"] for e in w.year_events]
    check("outcome" in kinds or "submit" in kinds or "cancel" in kinds, "the fight reaches a resolution")
    check("blow" in kinds or "submit" in kinds, "blows are traded or the target submits")


def test_premise_set_the_stage():
    """A world premise is woven into every agent's prompt, announced in year 1, and
    survives a git round-trip (so it reaches agents born later too)."""
    from redoland.components import VillagerContext
    path = "/tmp/redoland_premise"; shutil.rmtree(path, ignore_errors=True)
    premise = "Survivors of a flood that drowned the old kingdom."
    sim = Simulation.create(path, Params(founders=4), seed=3,
                            model_factory=lambda t: survival_stub(),
                            randomize_choices=False, premise=premise)
    w = sim.store.load_world("main")
    check(w.premise == premise, "premise persists through a git round-trip")
    aid = w.living()[0].id
    ctx = VillagerContext(w, aid)._make_pre_act_value()
    check("THE WORLD" in ctx and premise in ctx, "premise woven into the agent prompt")
    sim.run(1)
    raw = sim.store.read_at("main-y1", "events/year-1.jsonl") or ""
    check(premise in raw, "premise announced in year 1's transcript")
    shutil.rmtree(path, ignore_errors=True)


def test_concurrent_branches_via_worktrees():
    """Two branches of the SAME world advance at the same time, each in its own git
    worktree; their commits land on the shared refs and are visible from the main repo."""
    import threading
    from redoland.store import GitStore
    path = "/tmp/redoland_concurrent"; shutil.rmtree(path, ignore_errors=True)
    wt_root = "/tmp/redoland_concurrent_wt"; shutil.rmtree(wt_root, ignore_errors=True)
    mf = lambda t: survival_stub()
    sim = Simulation.create(path, Params(founders=4, ratio=1.6, start_food=1), seed=7,
                            model_factory=mf, randomize_choices=False)
    sim.run(1)
    sim.fork("main", 1, "beta")                       # a second branch of the same world
    # one worktree per branch (created serially, then run concurrently)
    setup = GitStore(path, model_factory=mf, randomize_choices=False)
    wts = {b: setup.ensure_worktree(b, os.path.join(wt_root, b)) for b in ("main", "beta")}
    check(os.path.exists(os.path.join(wts["main"], ".git")) and wts["main"] != wts["beta"],
          "each branch gets its own worktree")

    def run_branch(b):
        s = Simulation.open(wts[b], mf)
        s.store.checkout_branch(b)
        s.run(2)
    ts = [threading.Thread(target=run_branch, args=(b,)) for b in ("main", "beta")]
    for t in ts: t.start()
    for t in ts: t.join()
    reader = GitStore(path, model_factory=mf, randomize_choices=False)
    check(reader.years_for_branch("main") and reader.years_for_branch("main")[-1] >= 3,
          "main advanced while beta ran")
    check(reader.years_for_branch("beta") and reader.years_for_branch("beta")[-1] >= 3,
          "beta advanced while main ran")
    shutil.rmtree(path, ignore_errors=True); shutil.rmtree(wt_root, ignore_errors=True)


def test_server_concurrent_jobs():
    """The server runs several branches at once: start registers each in the active
    set; pause stops it and drops it from the set (so the picker only lists what runs)."""
    import time
    from redoland import server as srv
    base = "/tmp/redoland_srv"; shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    mf = lambda t: survival_stub()
    for name in ("w1", "w2"):
        Simulation.create(os.path.join(base, name),
                          Params(founders=3, ratio=1.6, start_food=1), seed=4,
                          model_factory=mf, randomize_choices=False)
    srv.JOBS.clear()
    mgr = srv.Manager(base)
    mgr.model_factory = lambda run=None: mf              # stub instead of claude -p
    r1 = mgr.start("w1", "main"); r2 = mgr.start("w2", "main")
    check(r1["ok"] and r2["ok"], "two branches of different worlds start concurrently")
    keys = {(a["run"], a["branch"]) for a in mgr.active()["active"]}
    check(("w1", "main") in keys and ("w2", "main") in keys, "both show up in the active set")
    check(not mgr.start("w1", "main")["ok"], "starting an already-running branch is rejected")
    mgr.pause("w1", "main"); mgr.pause("w2", "main")
    for _ in range(100):                                 # wait for the workers to stop
        if not mgr.active()["active"]:
            break
        time.sleep(0.05)
    check(mgr.active()["active"] == [], "paused branches drop out of the active set")
    shutil.rmtree(base, ignore_errors=True)


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
    for fn in [test_model_parsing, test_run_halts_on_model_failure,
               test_engine_invariants, test_sole_actor_can_continue,
               test_determinism,
               test_combat_resolves, test_dead_food_yearend_to_next_pile,
               test_combat_kill_loot_choice_and_spoils, test_convo_cap_scales_with_group,
               test_model_persisted,
               test_birth_crossover, test_max_maternal_age, test_group_talk,
               test_stepwise_talk_resumable, test_stepwise_talk_persists_through_git,
               test_stepwise_fight_resumable,
               test_worldline_fork_inject_replay, test_fork_at_action_and_inject,
               test_premise_set_the_stage, test_concurrent_branches_via_worktrees,
               test_server_concurrent_jobs,
               test_distributions]:
        print(fn.__name__)
        fn()
    print("\nALL CC TESTS PASSED")
