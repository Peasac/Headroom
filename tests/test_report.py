import pytest
from helpers import make_run

from headroom.backends.fake import FakeTurn
from headroom.runner.report import (
    aa_gap_p95, clean_success, condition_effects, format_condition_effects, format_report, paired_differences,
    run_symbol, state_success, summarize_runs,
)
from headroom.schemas.common import measured
from headroom.schemas.chat import ToolCall

pytestmark = pytest.mark.anyio

GOOD = ToolCall(name="rename_file", arguments={"src": "drafts/plan.md", "dst": "archive/plan.md"})
ADD = ToolCall(name="add_event", arguments={"title": "Design review", "start": "2026-12-03T14:00", "end": "2026-12-03T15:30"})
TEXT_CALL = '{"name": "get_event", "parameters": {"title": "Design review"}}'


@pytest.fixture
def mk(experiment, provenance):
    async def _mk(turns, **kw):
        return await make_run(experiment, provenance, turns, **kw)
    return _mk


async def test_malformed_text_call_does_not_fail_a_run_by_itself(mk):
    # direct: the event is added, then a text call ends the run. The state grader decides: pass.
    run = await mk([FakeTurn(tool_calls=(ADD,)), TEXT_CALL], task_id="calendar.add_01", arm_id="direct")
    assert run.status == "malformed_text_call"
    assert run.success and state_success(run)
    assert not clean_success(run)
    assert run_symbol(run) == "p"


async def test_clean_success_needs_zero_text_calls(mk):
    run = await mk([FakeTurn(tool_calls=(ADD,)), "Done."], task_id="calendar.add_01")
    assert state_success(run) and clean_success(run) and run_symbol(run) == "P"


async def test_state_failure_is_a_failure_whatever_the_status(mk):
    run = await mk(["I scheduled it."], task_id="calendar.add_01")
    assert run.status == "completed" and not run.success and run_symbol(run) == "F"


async def test_context_exceeded_passes_on_state_but_is_not_clean(mk):
    run = await mk([FakeTurn(tool_calls=(GOOD,), prompt_tokens=10), FakeTurn(prompt_tokens=900)], context_cap=512)
    assert run.status == "context_exceeded"
    assert run.malformed_text_calls == 0
    assert run.success and state_success(run)  # the state grader decides success
    assert not clean_success(run)  # but a run over the context cap is never clean
    assert run_symbol(run) == "p"


async def test_report_counts_tasks_not_runs(mk):
    runs = [
        # files.rename_01, validated: passes on both repeats
        await mk([FakeTurn(tool_calls=(GOOD,)), "ok"], repeat=0, order_index=0),
        await mk([FakeTurn(tool_calls=(GOOD,)), "ok"], repeat=1, order_index=1),
        # calendar.add_01, validated: passes, but wrote a text call on one repeat
        await mk([FakeTurn(tool_calls=(ADD,)), "ok"], task_id="calendar.add_01", repeat=0, order_index=2),
        await mk([FakeTurn(tool_calls=(ADD,)), TEXT_CALL, "ok"], task_id="calendar.add_01", repeat=1, order_index=3),
        # files.append_01, validated: repeats disagree (fails, then passes)
        await mk(["nothing"], task_id="files.append_01", repeat=0, order_index=4),
        await mk([FakeTurn(tool_calls=(ToolCall(name="write_file", arguments={
            "path": "shopping.txt", "content": "eggs\nbread\nmilk"}),)), "ok"],
            task_id="files.append_01", repeat=1, order_index=5),
        # direct arm: one task, fails
        await mk(["I moved it."], arm_id="direct", order_index=6),
    ]
    by_arm = {s.arm_id: s for s in summarize_runs(runs)}
    v, d = by_arm["validated"], by_arm["direct"]
    assert (v.tasks, v.runs) == (3, 6)
    assert v.task_successes == 2  # rename + add; append is inconsistent -> not counted
    assert v.task_clean_successes == 1  # only rename is clean on every repeat
    assert v.inconsistent_tasks == ("files.append_01",)
    assert v.per_task == {"calendar.add_01": "Pp", "files.append_01": "FP", "files.rename_01": "PP"}
    assert v.malformed_text_calls == 1 and v.runs_with_malformed_text == 1
    assert (d.tasks, d.task_successes, d.claimed_done_without_state_change) == (1, 0, 1)

    text = format_report(summarize_runs(runs))
    assert "2/3 tasks" in text and "1/3 tasks" in text and "0/1 tasks" in text  # task counts beside every rate
    assert "(n=3)" in text and "(n=1)" in text  # task counts beside every timing figure
    assert "INCONSISTENT repeats: files.append_01" in text
    assert "calendar.add_01" in text and "Pp" in text


def with_wall(run, wall_s, repeat):
    return run.model_copy(update={"wall_s": wall_s, "repeat": repeat})


async def test_timing_uses_per_task_medians_with_a_bootstrap_interval_over_tasks(mk):
    base = {
        "files.rename_01": await mk([FakeTurn(tool_calls=(GOOD,)), "ok"]),
        "calendar.add_01": await mk([FakeTurn(tool_calls=(ADD,)), "ok"], task_id="calendar.add_01"),
    }
    walls = {"files.rename_01": [1.0, 1.1, 0.9, 1.0, 5.0], "calendar.add_01": [2.0, 2.2, 1.8, 2.0, 2.1]}
    runs = [with_wall(base[t], w, i) for t, ws in walls.items() for i, w in enumerate(ws)]
    (s,) = summarize_runs(runs)
    assert s.task_median_wall_s == {"files.rename_01": 1.0, "calendar.add_01": 2.0}  # the 5.0 outlier is ignored
    assert s.median_wall_s == 1.5
    assert s.median_wall_ci[0] <= 1.5 <= s.median_wall_ci[1]
    assert s.task_cv["files.rename_01"] > s.task_cv["calendar.add_01"]
    assert s.aa_gap_p95 is not None and s.aa_gap_p95 >= 0
    assert s.peak_private_mb == 900.0  # helpers.SAMPLE
    text = format_report([s])
    assert "1.50 [" in text and "(n=2)" in text
    assert "repeat-to-repeat CV of one task" in text and "files.rename_01" in text
    assert "A/A split gap" in text


async def test_paired_difference_between_arms(mk):
    a = await mk([FakeTurn(tool_calls=(GOOD,)), "ok"])
    b = await mk([FakeTurn(tool_calls=(GOOD,)), "ok"], arm_id="direct")
    runs = [with_wall(a, 1.0, 0), with_wall(a, 1.0, 1), with_wall(b, 1.2, 0), with_wall(b, 1.2, 1)]
    (d,) = paired_differences(summarize_runs(runs))
    assert (d.arm_a, d.arm_b, d.tasks) == ("direct", "validated", 1)
    assert d.median_ratio_minus_1 == pytest.approx(1.0 / 1.2 - 1)


async def test_condition_effect_is_paired_against_the_same_arm_at_baseline(mk):
    from headroom.schemas.config import CpuContention
    from headroom.schemas.run import ConditionRecord

    cpu = CpuContention(condition_id="cpu", kind="cpu_contention", realism="emulated", workers=4, min_free_logical_cpus=2)
    a = await mk([FakeTurn(tool_calls=(GOOD,)), "ok"])
    b = await mk([FakeTurn(tool_calls=(ADD,)), "ok"], task_id="calendar.add_01")
    rec = ConditionRecord(condition=cpu, injector_alive=True,
                          at_start={"sys_cpu_pct": measured(85, "%"), "workers_alive": measured(4, "processes")})
    runs = [
        with_wall(a, 1.0, 0), with_wall(b, 2.0, 0),
        with_wall(a, 1.2, 0).model_copy(update={"condition_id": "cpu", "condition": rec}),
        with_wall(b, 2.6, 0).model_copy(update={"condition_id": "cpu", "condition": rec}),
    ]
    effects = {e.condition_id: e for e in condition_effects(runs)}
    assert effects["baseline"].latency_change is None
    cpu_e = effects["cpu"]
    assert cpu_e.tasks == 2 and cpu_e.latency_change == pytest.approx(0.25)  # median of +20% and +30%
    assert cpu_e.pressure["sys_cpu_pct"] == 85 and cpu_e.pressure["workers_alive"] == 4
    text = format_condition_effects(list(effects.values()))
    assert "+25.0%" in text and "workers_alive 4" in text


def test_aa_gap_is_zero_without_noise_and_needs_two_repeats():
    assert aa_gap_p95({"t1": (1.0, 1.0, 1.0, 1.0), "t2": (2.0, 2.0, 2.0, 2.0)}) == 0
    assert aa_gap_p95({"t1": (1.0,)}) is None
