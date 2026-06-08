"""Smoke tests — run with: python3 tests/test_smoke.py  (no pytest needed)."""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redoland.core import Params, RNG, Mortality
from redoland.backend import FakeBackend
from redoland.engine import Engine
from redoland.sim import Simulation
from redoland.metrics import snapshot_metrics


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def test_mortality_monotonic():
    m = Mortality()
    for sex in ("male", "female"):
        qs = [m.q(a, sex) for a in range(21, 110)]
        check(all(b >= a - 1e-9 for a, b in zip(qs, qs[1:])),
              f"{sex} mortality non-decreasing with age")


def test_rng_serialization():
    r = RNG(seed=1)
    s = r.get_state()
    r2 = RNG(state=s)
    check(all(r.random() == r2.random() for _ in range(50)), "rng replays from state")


def test_engine_invariants():
    eng = Engine.found(Params(), FakeBackend(), seed=11)
    for _ in range(25):
        eng.run_year()
    w = eng.w
    check(all(0 <= a.health <= 3 for a in w.agents.values()), "satiation in [0,3]")
    check(all(0 <= a.hp <= w.params.hp_max for a in w.agents.values()), "HP in [0,100]")
    check(all(0 <= a.strength <= 100 for a in w.agents.values()), "strength in [0,100]")
    check(all(a.food >= 0 for a in w.agents.values()), "food non-negative")
    check(all(a.intelligence_tokens >= 1024 for a in w.agents.values()),
          "intelligence >= API floor")
    check(all((a.sex in ("male", "female")) for a in w.agents.values()), "sex valid")
    check(all(a.id not in a.parents + a.children + a.siblings for a in w.agents.values()),
          "no self-kinship")


def test_combat_and_repro_occur():
    eng = Engine.found(Params(), FakeBackend(), seed=11)
    kinds = set()
    for _ in range(25):
        for e in eng.run_year():
            kinds.add(e["kind"])
    check("attack" in kinds and "blow" in kinds, "raids and blows happen")
    check("birth" in kinds, "reproduction (via convos) happens")
    dead = [a for a in eng.w.agents.values() if not a.alive]
    check(any(a.death_cause == "combat" for a in dead), "combat deaths occur")


def test_determinism():
    a = Engine.found(Params(ratio=1.15), FakeBackend(), seed=99)
    b = Engine.found(Params(ratio=1.15), FakeBackend(), seed=99)
    for _ in range(15):
        a.run_year(); b.run_year()
    check(sorted(a.w.agents) == sorted(b.w.agents) and a.w.pile == b.w.pile,
          "same seed -> identical world")


def test_branching_diverges():
    path = "/tmp/redoland_smoke_run"
    shutil.rmtree(path, ignore_errors=True)
    sim = Simulation.create(path, Params(ratio=1.15), seed=5, backend=FakeBackend())
    sim.run(15)
    sim.fork("main", 10, "drought")
    sim.inject("drought", ratio=0.5)
    sim.replay("drought", 12)
    sim.fork("main", 10, "control")
    sim.replay("control", 12)
    d = sim.diff("drought", 22, "control", 22)
    dp = d["drought@y22"]["population"]
    cp = d["control@y22"]["population"]
    check(dp != cp, f"drought ({dp}) diverges from control ({cp})")
    shutil.rmtree(path, ignore_errors=True)


def test_cli_backend_parsing():
    """CLIBackend unwraps the `claude -p --output-format json` envelope and
    parses the action JSON robustly — no network, no metered key."""
    from redoland.backend import CLIBackend
    import json
    b = CLIBackend()
    check(b.name == "cli", "CLIBackend reports name 'cli'")
    env = json.dumps({"type": "result", "is_error": False,
                      "result": "```json\n{\"kind\":\"take\",\"amount\":2}\n```"})
    check(b._safe_json_cli(env) == {"kind": "take", "amount": 2},
          "envelope + ```json fence parsed")
    check(b._safe_json_cli('{"act": true}') == {"act": True}, "bare json parsed")
    check(b._safe_json_cli(json.dumps({"result": "ok: {\"kind\":\"pass\"} done"}))
          == {"kind": "pass"}, "prose-wrapped json extracted")
    check(b._safe_json_cli("not json") == {}, "garbage -> {} (no-op fallback)")
    check(b._safe_json_cli("") == {}, "empty -> {} (no-op fallback)")
    check(not hasattr(b, "_client"), "CLIBackend holds no anthropic SDK client")
    # _envelope surfaces stop_reason so _decide can detect a truncated answer
    text, stop = b._envelope(json.dumps(
        {"result": "{\"act\": true}", "stop_reason": "max_tokens"}))
    check((text, stop) == ('{"act": true}', "max_tokens"),
          "_envelope returns (text, stop_reason)")
    check(b._envelope("raw text") == ("raw text", None),
          "_envelope falls back to (raw, None) for non-envelope stdout")


if __name__ == "__main__":
    for fn in [test_mortality_monotonic, test_rng_serialization,
               test_engine_invariants, test_combat_and_repro_occur,
               test_determinism, test_branching_diverges,
               test_cli_backend_parsing]:
        print(fn.__name__)
        fn()
    print("\nALL SMOKE TESTS PASSED")
