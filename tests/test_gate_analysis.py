import pytest
from conftest import ROOT
from helpers import make_run

from headroom.analysis.gate_a import analysed_runs, evaluate_gate_a, format_gate_a
from headroom.backends.fake import FakeTurn
from headroom.schemas.chat import ToolCall
from headroom.schemas.config import BaselineCondition, CpuContention, MemoryPressure, VramPressure
from headroom.schemas.gate import load_gate_a
from headroom.schemas.run import ConditionRecord

pytestmark = pytest.mark.anyio

RULE = load_gate_a(ROOT / "experiments" / "gate_a.toml")
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


TIMES = {
    ("a", "baseline"): 1.0, ("b", "baseline"): 1.05,
    ("a", "cpu"): 1.1, ("b", "cpu"): 1.15,
    ("a", "vram"): 3.0, ("b", "vram"): 1.5,
    ("a", "vram-1g"): 2.0, ("b", "vram-1g"): 1.2,
}


async def test_one_big_gap_is_not_enough(base):
    res = evaluate_gate_a(build(base, TIMES), RULE)
    assert res.best_fixed == "a"  # equal success; a has the lower median across Gate A conditions
    gaps = {g.condition_id: g for g in res.conditions}
    assert gaps["vram"].oracle == "b" and gaps["vram"].gap == pytest.approx(1.0, rel=1e-6)
    assert gaps["vram"].passes and gaps["vram"].ci[0] > 0
    assert gaps["baseline"].oracle == "a" and gaps["baseline"].gap == 0
    assert res.passing == ("vram",) and not res.passed  # needs 2 Gate A conditions


async def test_dose_response_never_counts(base):
    res = evaluate_gate_a(build(base, TIMES), RULE)
    dose = next(g for g in res.conditions if g.condition_id == "vram-1g")
    assert not dose.counts_for_gate and dose.passes  # big gap, but it does not count
    assert "vram-1g" not in res.passing


async def test_two_conditions_pass(base):
    # a is best fixed (median over baseline, memory, cpu, vram: 1.5 vs b's 1.55), b wins under cpu and vram.
    times = {
        ("a", "baseline"): 1.0, ("b", "baseline"): 1.6, ("a", "memory"): 1.0, ("b", "memory"): 1.6,
        ("a", "cpu"): 2.0, ("b", "cpu"): 1.2, ("a", "vram"): 3.0, ("b", "vram"): 1.5,
    }
    res = evaluate_gate_a(build(base, times), RULE)
    assert res.best_fixed == "a"
    assert set(res.passing) == {"cpu", "vram"} and res.passed


async def test_best_fixed_winning_everywhere_means_no_gap(base):
    times = {**TIMES, ("a", "cpu"): 2.0, ("b", "cpu"): 1.2}  # b now wins 2 of 3 conditions: b is best fixed
    res = evaluate_gate_a(build(base, times), RULE)
    assert res.best_fixed == "b" and not res.passed


async def test_a_faster_config_with_lower_success_cannot_be_the_oracle(base):
    fail = {("b", "vram", "t3")}  # b is faster under vram but fails a task there
    res = evaluate_gate_a(build(base, TIMES, fail=fail), RULE)
    vram = next(g for g in res.conditions if g.condition_id == "vram")
    assert res.cells[("b", "vram")].task_success == 5 and res.cells[("a", "vram")].task_success == 6
    assert vram.oracle == "a" and vram.gap == 0 and not vram.passes


async def test_best_fixed_prefers_success_over_speed(base):
    fail = {("a", "baseline", "t1")}
    res = evaluate_gate_a(build(base, TIMES, fail=fail), RULE)
    assert res.best_fixed == "b"


async def test_cache_reused_runs_are_excluded_and_verified_absent(base):
    cache = {("a", "baseline", "t1", 0), ("b", "vram", "t2", 1)}
    runs = build(base, TIMES, cache=cache)
    kept, excluded = analysed_runs(runs)
    assert excluded == {"cache_reused": 2} and len(kept) == len(runs) - 2
    assert not any(r.cache_reused for r in kept)
    res = evaluate_gate_a(runs, RULE)
    assert res.analysed == len(runs) - 2 and res.excluded == {"cache_reused": 2}


async def test_report_shows_gap_success_noise_and_the_verdict(base):
    text = format_gate_a(evaluate_gate_a(build(base, TIMES), RULE))
    assert "Gate A (gate_a_v1): FAIL: 1 of 3 Gate A conditions" in text
    assert "success, checked separately" in text and "6/6" in text
    assert "+100.0% [" in text and "noise CV med/max" in text
    assert "dose-response (reported separately; does not count toward Gate A)" in text
    assert "verified: no analysed run has cache_reused = true" in text
