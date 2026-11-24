import pytest
from helpers import make_run

from headroom.backends.fake import FakeTurn
from headroom.runner.report import clean_success, format_report, run_symbol, state_success, summarize_runs
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
    assert v.successful_runs == 5 and v.malformed_text_calls == 1 and v.runs_with_malformed_text == 1
    assert (d.tasks, d.task_successes, d.claimed_done_without_state_change) == (1, 0, 1)
    assert d.p95_time_to_success_s is None

    text = format_report(summarize_runs(runs))
    assert "2/3 tasks" in text and "1/3 tasks" in text and "0/1 tasks" in text  # task counts beside every rate
    assert "(n=6)" in text and "(n=5)" in text  # run counts beside every p95
    assert "INCONSISTENT repeats: files.append_01" in text
    assert "calendar.add_01" in text and "Pp" in text
