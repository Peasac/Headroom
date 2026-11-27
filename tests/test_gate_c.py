import pytest
from conftest import ROOT
from helpers import make_run

from headroom.analysis.gate_c import evaluate_gate_c, format_gate_c
from headroom.backends.fake import FakeTurn
from headroom.experiments import load_experiment
from headroom.schemas.chat import ToolCall
from headroom.schemas.config import BaselineCondition
from headroom.schemas.gate import load_gate_a, load_gate_c
from headroom.schemas.run import ConditionRecord
from headroom.tasks.suite import HELDOUT_TASKS, HeldOutError, heldout_run_tasks

pytestmark = pytest.mark.anyio

RULE = load_gate_c(ROOT / "experiments" / "gate_c.toml")
GATE_A = load_gate_a(ROOT / "experiments" / RULE.claims.no_adaptation_gain.gate_a_rule.removeprefix("experiments/"))
GOOD = ToolCall(name="rename_file", arguments={"src": "drafts/plan.md", "dst": "archive/plan.md"})
TASKS = [f"t{i}" for i in range(12)]
D, V, K, P = "qwen-direct", "qwen-validated", "qwen-validated-2k", "qwen-plan-act"
B, VE, SR = "baseline", "video-export", "screen-record"

# Shaped like the tuning half: every arm ~15% slower under load, plan_act ~45% slower than direct,
# validated equal to direct, the 2k arm slightly faster but failing 2 tasks everywhere.
SLOW = {B: 1.0, VE: 1.17, SR: 1.12}
SPEED = {D: 1.0, V: 1.0, K: 0.98, P: 1.45}
TUNING_LIKE = {(a, c): SPEED[a] * SLOW[c] for a in SPEED for c in SLOW}
K_FAILS = {(K, c, t) for c in SLOW for t in ("t0", "t1")}


@pytest.fixture
async def base(experiment, provenance):
    return await make_run(experiment, provenance, [FakeTurn(tool_calls=(GOOD,)), "ok"])


def build(base, times, *, fail=frozenset()):
    runs = []
    for (arm, cond), t in times.items():
        for i, task in enumerate(TASKS):
            for rep in range(3):
                # Per-task and per-repeat jitter that differs between cells, so intervals are not degenerate.
                jitter = 1 + 0.004 * ((i * 7 + rep * 3 + sum(map(ord, arm + cond)) % 5) % 5 - 2)
                grade = base.grade.model_copy(update={"success": (arm, cond, task) not in fail})
                runs.append(base.model_copy(update={
                    "run_id": f"{arm}-{cond}-{task}-{rep}", "arm_id": arm, "condition_id": cond, "task_id": task,
                    "repeat": rep, "wall_s": t * (1 + 0.1 * i) * jitter, "grade": grade, "success": grade.success,
                    "condition": ConditionRecord(
                        condition=BaselineCondition(condition_id=cond, kind="baseline", realism="real"),
                        injector_alive=True, at_start={},
                    ),
                }))
    return runs


def plan_of(times):
    return {(a, c, t) for (a, c) in times for t in TASKS}


def claims(res):
    return {c.name: c.status for c in res.claims}


def test_registered_rule_matches_the_heldout_experiment_and_gate_a_real():
    exp = load_experiment(ROOT / "experiments" / "gate_c_heldout.toml")
    assert exp.experiment_id == RULE.experiment_id and exp.task_split == "heldout"
    assert sorted(a.arm_id for a in exp.arms) == sorted(RULE.arms)
    assert [c.condition_id for c in exp.conditions] == RULE.conditions == ["baseline", "video-export", "screen-record"]
    assert GATE_A.rule_id == "gate_a_real_v1"
    assert sorted(GATE_A.scope.arms) == sorted(RULE.arms) and GATE_A.scope.conditions == RULE.conditions


def test_heldout_experiment_is_the_tuning_experiment_on_other_tasks():
    held = load_experiment(ROOT / "experiments" / "gate_c_heldout.toml").model_dump()
    tune = load_experiment(ROOT / "experiments" / "week3_real_workloads.toml").model_dump()
    for d in (held, tune):
        d.pop("experiment_id"), d.pop("task_split")
    assert held == tune


def test_heldout_run_takes_every_heldout_task_once_and_nothing_else():
    assert heldout_run_tasks(allow_heldout=True, overrides=[], earlier_sessions=0) == [t.task_id for t in HELDOUT_TASKS]
    assert len(HELDOUT_TASKS) == 26
    with pytest.raises(HeldOutError, match="pass --heldout"):
        heldout_run_tasks(allow_heldout=False, overrides=[], earlier_sessions=0)
    with pytest.raises(HeldOutError, match="--repeats"):
        heldout_run_tasks(allow_heldout=True, overrides=["--repeats"], earlier_sessions=0)
    with pytest.raises(HeldOutError, match="run once"):
        heldout_run_tasks(allow_heldout=True, overrides=[], earlier_sessions=1)


async def test_tuning_like_results_replicate_every_claim(base):
    res = evaluate_gate_c(build(base, TUNING_LIKE, fail=K_FAILS), RULE, GATE_A, planned=plan_of(TUNING_LIKE))
    assert set(claims(res).values()) == {"REPLICATED"}, format_gate_c(res)
    assert res.verdict == "PASS"
    assert "Gate C (gate_c_v1): PASS: 5 of 5 claims replicated" in format_gate_c(res)


async def test_plan_act_as_fast_as_direct_is_not_replicated(base):
    times = {**TUNING_LIKE, (P, SR): TUNING_LIKE[(D, SR)]}
    res = evaluate_gate_c(build(base, times, fail=K_FAILS), RULE, GATE_A)
    assert claims(res)["plan_act_slower"] == "NOT REPLICATED" and res.verdict == "FAIL"


async def test_success_may_move_by_one_task_but_not_two(base):
    one = K_FAILS | {(D, VE, "t5")}
    assert claims(evaluate_gate_c(build(base, TUNING_LIKE, fail=one), RULE, GATE_A))[
        "success_unchanged_under_load"] == "REPLICATED"
    two = one | {(D, VE, "t6")}
    assert claims(evaluate_gate_c(build(base, TUNING_LIKE, fail=two), RULE, GATE_A))[
        "success_unchanged_under_load"] == "NOT REPLICATED"


async def test_validated_ten_percent_slower_breaks_equivalence(base):
    times = {**TUNING_LIKE, (V, B): 1.10}
    res = evaluate_gate_c(build(base, times, fail=K_FAILS), RULE, GATE_A)
    assert claims(res)["direct_validated_equivalent"] == "NOT REPLICATED"


async def test_a_workload_that_does_not_slow_an_arm_is_not_replicated(base):
    times = {**TUNING_LIKE, (K, SR): TUNING_LIKE[(K, B)]}
    res = evaluate_gate_c(build(base, times, fail=K_FAILS), RULE, GATE_A)
    assert claims(res)["workloads_slow_every_arm"] == "NOT REPLICATED"


async def test_an_adaptation_gain_on_heldout_breaks_the_first_claim(base):
    # direct stays the best fixed config (lowest median over conditions), but a different config wins
    # each workload by far: validated under video-export, the 2k arm under screen-record.
    times = {**TUNING_LIKE, (V, B): 2.0, (V, VE): 1.0, (V, SR): 2.0, (K, B): 2.0, (K, VE): 2.0, (K, SR): 0.9}
    res = evaluate_gate_c(build(base, times), RULE, GATE_A)
    assert claims(res)["no_adaptation_gain"] == "NOT REPLICATED" and res.verdict == "FAIL"


async def test_a_condition_that_could_not_run_is_inconclusive(base):
    times = {k: v for k, v in TUNING_LIKE.items() if k[1] != SR}
    res = evaluate_gate_c(
        build(base, times, fail=K_FAILS), RULE, GATE_A, planned=plan_of(TUNING_LIKE),
        failures={SR: "InjectorError: no ffmpeg progress (after 0/312 runs)"},
    )
    assert res.verdict == "INCONCLUSIVE"
    assert claims(res)["plan_act_slower"] == "INCONCLUSIVE"
    assert "missing: qwen-direct screen-record: condition FAILED" in format_gate_c(res)
