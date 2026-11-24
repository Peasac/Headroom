import pytest
from conftest import ROOT
from helpers import make_run

from headroom.analysis.gate_a import analysed_runs, evaluate_gate_a, format_gate_a
from headroom.backends.fake import FakeTurn
from headroom.schemas.chat import ToolCall
from headroom.schemas.config import BaselineCondition, CpuContention, MemoryPressure, VramPressure
from headroom.schemas.gate import GateScope, load_gate_a
from headroom.schemas.run import ConditionRecord

pytestmark = pytest.mark.anyio

REGISTERED = load_gate_a(ROOT / "experiments" / "gate_a.toml")
GOOD = ToolCall(name="rename_file", arguments={"src": "drafts/plan.md", "dst": "archive/plan.md"})
COND = {
    "baseline": BaselineCondition(condition_id="baseline", kind="baseline", realism="real"),
    "cpu": CpuContention(condition_id="cpu", kind="cpu_contention", realism="emulated", workers=4,
                         min_free_logical_cpus=2),
    "memory": MemoryPressure(condition_id="memory", kind="memory_pressure", realism="emulated",
                             target_available_mb=1024, min_available_mb=512, max_ballast_mb=8192),
    "vram": VramPressure(condition_id="vram", kind="vram_pressure", realism="emulated", ballast_mb=2048,
                         touch_interval_s=0.5, min_available_ram_mb=1024),
    "vram-1g": VramPressure(condition_id="vram-1g", kind="vram_pressure", realism="emulated", gate_a=False,
                            ballast_mb=1024, touch_interval_s=0.5, min_available_ram_mb=1024),
}
TASKS = ["t1", "t2", "t3", "t4", "t5", "t6"]


def rule(conditions=("baseline", "cpu", "vram"), arms=("a", "b")):
    """The registered rule (thresholds, bootstrap) with a test scope."""
    return REGISTERED.model_copy(update={"scope": GateScope(arms=list(arms), conditions=list(conditions))})


@pytest.fixture
async def base(experiment, provenance):
    return await make_run(experiment, provenance, [FakeTurn(tool_calls=(GOOD,)), "ok"])


def build(base, times, *, fail=(), cache=()):
    """times: {(arm, condition): seconds for task t1; other tasks scale by 1 + 0.1*i}."""
    runs = []
    for (arm, cond), t in times.items():
        for i, task in enumerate(TASKS):
            for rep in range(3):
                wall = t * (1 + 0.1 * i) * (1 + 0.01 * rep)
                grade = base.grade.model_copy(update={"success": (arm, cond, task) not in fail})
                runs.append(base.model_copy(update={
                    "run_id": f"{arm}-{cond}-{task}-{rep}", "arm_id": arm, "condition_id": cond, "task_id": task,
                    "repeat": rep, "wall_s": wall, "grade": grade, "success": grade.success,
                    "cache_reused": (arm, cond, task, rep) in cache,
                    "condition": ConditionRecord(condition=COND[cond], injector_alive=True, at_start={}),
                }))
    return runs


def plan_of(times):
    return {(a, c, t) for (a, c) in times for t in TASKS}


TIMES = {
    ("a", "baseline"): 1.0, ("b", "baseline"): 1.05,
    ("a", "cpu"): 1.1, ("b", "cpu"): 1.15,
    ("a", "vram"): 3.0, ("b", "vram"): 1.5,
    ("a", "vram-1g"): 2.0, ("b", "vram-1g"): 1.2,
}


async def test_one_big_gap_is_not_enough(base):
    res = evaluate_gate_a(build(base, TIMES), rule(), planned=plan_of(TIMES))
    assert res.best_fixed == "a"  # equal success; a has the lower median across the scoped conditions
    gaps = {g.condition_id: g for g in res.conditions}
    assert gaps["vram"].oracle == "b" and gaps["vram"].gap == pytest.approx(1.0, rel=1e-6)
    assert gaps["vram"].passes and gaps["vram"].ci[0] > 0
    assert gaps["baseline"].oracle == "a" and gaps["baseline"].gap == 0
    assert res.passing == ("vram",) and res.verdict == "FAIL"  # needs 2 scoped conditions


async def test_conditions_outside_the_scope_never_count(base):
    res = evaluate_gate_a(build(base, TIMES), rule())
    extra = next(g for g in res.conditions if g.condition_id == "vram-1g")
    assert not extra.in_scope and extra.passes  # big gap, but it is a sensitivity check
    assert "vram-1g" not in res.passing


async def test_arms_outside_the_scope_are_not_used(base):
    times = {**TIMES, ("c", "baseline"): 0.1, ("c", "cpu"): 0.1, ("c", "vram"): 0.1}
    res = evaluate_gate_a(build(base, times), rule())
    assert res.best_fixed == "a" and all(g.oracle in ("a", "b", None) for g in res.conditions)
    assert res.out_of_scope_runs == 3 * len(TASKS) * 3


async def test_two_conditions_pass(base):
    # a is best fixed (median over baseline, memory, cpu, vram: 1.5 vs b's 1.55), b wins under cpu and vram.
    times = {
        ("a", "baseline"): 1.0, ("b", "baseline"): 1.6, ("a", "memory"): 1.0, ("b", "memory"): 1.6,
        ("a", "cpu"): 2.0, ("b", "cpu"): 1.2, ("a", "vram"): 3.0, ("b", "vram"): 1.5,
    }
    res = evaluate_gate_a(build(base, times), rule(("baseline", "memory", "cpu", "vram")))
    assert res.best_fixed == "a"
    assert set(res.passing) == {"cpu", "vram"} and res.verdict == "PASS"


async def test_best_fixed_winning_everywhere_means_no_gap(base):
    times = {**TIMES, ("a", "cpu"): 2.0, ("b", "cpu"): 1.2}  # b now wins 2 of 3 conditions: b is best fixed
    res = evaluate_gate_a(build(base, times), rule())
    assert res.best_fixed == "b" and res.verdict == "FAIL"


async def test_a_faster_config_with_lower_success_cannot_be_the_oracle(base):
    fail = {("b", "vram", "t3")}  # b is faster under vram but fails a task there
    res = evaluate_gate_a(build(base, TIMES, fail=fail), rule())
    vram = next(g for g in res.conditions if g.condition_id == "vram")
    assert res.cells[("b", "vram")].task_success == 5 and res.cells[("a", "vram")].task_success == 6
    assert vram.oracle == "a" and vram.gap == 0 and not vram.passes


async def test_best_fixed_prefers_success_over_speed(base):
    fail = {("a", "baseline", "t1")}
    res = evaluate_gate_a(build(base, TIMES, fail=fail), rule())
    assert res.best_fixed == "b"


async def test_a_condition_that_could_not_run_is_failed_not_no_gap(base):
    # The cpu block never ran. Over the conditions that did, a is best fixed (2.0 vs 2.25) and b wins vram.
    times = {("a", "baseline"): 1.0, ("b", "baseline"): 3.0, ("a", "vram"): 3.0, ("b", "vram"): 1.5}
    planned = plan_of({**times, ("a", "cpu"): 0, ("b", "cpu"): 0})
    res = evaluate_gate_a(build(base, times), rule(), planned=planned,
                          failures={"cpu": "InjectorError: cannot start (after 0/36 runs)"})
    assert res.best_fixed == "a"
    cpu = next(g for g in res.conditions if g.condition_id == "cpu")
    assert cpu.status == "failed" and cpu.gap is None and not cpu.passes
    assert "cannot start" in cpu.reason
    # vram passes, cpu FAILED: had cpu run it could have been the second pass, so not a plain FAIL.
    assert res.passing == ("vram",) and res.verdict == "INCONCLUSIVE"
    text = format_gate_a(res)
    assert "Gate A (gate_a_v2): INCONCLUSIVE" in text and "FAILED conditions: cpu" in text
    assert "cpu           FAILED: InjectorError" in text


async def test_missing_planned_runs_make_a_condition_failed(base):
    runs = [r for r in build(base, TIMES) if not (r.condition_id == "baseline" and r.task_id == "t2" and r.arm_id == "b")]
    res = evaluate_gate_a(runs, rule(), planned=plan_of(TIMES))
    baseline = next(g for g in res.conditions if g.condition_id == "baseline")
    assert baseline.status == "failed" and "b: 1 planned task(s)" in baseline.reason


async def test_cache_reused_runs_are_excluded_and_verified_absent(base):
    cache = {("a", "baseline", "t1", 0), ("b", "vram", "t2", 1)}
    runs = build(base, TIMES, cache=cache)
    kept, excluded = analysed_runs(runs)
    assert excluded == {"cache_reused": 2} and len(kept) == len(runs) - 2
    assert not any(r.cache_reused for r in kept)
    res = evaluate_gate_a(runs, rule(), planned=plan_of(TIMES))  # other repeats remain: not FAILED
    assert res.analysed == len(runs) - 2 and res.excluded == {"cache_reused": 2}
    assert all(g.status == "ok" for g in res.conditions)


async def test_report_shows_gap_success_noise_and_the_verdict(base):
    text = format_gate_a(evaluate_gate_a(build(base, TIMES), rule()))
    assert "Gate A (gate_a_v2): FAIL: 1 of 3 scoped conditions" in text
    assert "success, checked separately" in text and "6/6" in text
    assert "+100.0% [" in text and "noise CV med/max" in text
    assert "outside the scope (sensitivity check, reported separately, never counted)" in text
    assert "verified: no analysed run has cache_reused = true" in text
