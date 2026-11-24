import pytest
from conftest import make_config

from headroom.backends.fake import FakeBackend, FakeTurn
from headroom.runner.agent_loop import run_task
from headroom.schemas.chat import ToolCall
from headroom.tasks.checks import grade
from headroom.tasks.suite import get_task

pytestmark = pytest.mark.anyio

RENAME = get_task("files.rename_01")
GOOD = ToolCall(name="rename_file", arguments={"src": "drafts/plan.md", "dst": "archive/plan.md"})
BAD_WRITE = ToolCall(name="rename_file", arguments={"src": "drafts/plan.md"})  # missing dst: invalid, state-changing
BAD_READ = ToolCall(name="read_file", arguments={"file": "drafts/plan.md"})  # wrong arg name: invalid, read-only
MISSING = ToolCall(name="read_file", arguments={"path": "nope.md"})  # valid, tool error


def calls(*cs: ToolCall) -> FakeTurn:
    return FakeTurn(tool_calls=cs)


async def run(turns, mode="validated", retry_limit=2, task=RENAME, sampling=None, llama=None, context_cap=8192):
    backend = FakeBackend(turns)
    res = await run_task(
        backend, task, model=llama, config=make_config(mode, retry_limit, context_cap), sampling=sampling, seed=3
    )
    return res, backend


@pytest.fixture
def go(sampling, llama):
    async def _go(turns, **kw):
        return await run(turns, sampling=sampling, llama=llama, **kw)
    return _go


async def test_completes_and_passes_grading(go):
    res, backend = await go([calls(GOOD), "Moved."])
    assert res.status == "completed"
    assert len(res.steps) == 2
    assert grade(RENAME, res.final_state).success
    # The tool result went back to the model before its final reply.
    tool_msg = backend.requests[1].messages[-1]
    assert tool_msg.role == "tool" and tool_msg.tool_name == "rename_file"
    assert [r.seed for r in backend.requests] == [3, 3]


async def test_direct_mode_fails_on_invalid_state_changing_call(go):
    res, backend = await go([calls(BAD_WRITE), calls(GOOD), "done"], mode="direct")
    assert res.status == "invalid_state_call"
    assert res.invalid_state_changing_calls == 1
    assert res.final_state == RENAME.initial_state
    assert len(backend.requests) == 1  # no second chance


async def test_validated_mode_feeds_error_back_and_spends_a_retry(go):
    res, backend = await go([calls(BAD_WRITE), calls(GOOD), "done"], mode="validated")
    assert res.status == "completed"
    assert res.retries_used == 1
    assert res.invalid_state_changing_calls == 1
    assert backend.requests[1].messages[-1].content.startswith("Invalid call:")
    assert grade(RENAME, res.final_state).success


async def test_validated_mode_fails_once_retries_are_spent(go):
    res, _ = await go([calls(BAD_WRITE), calls(BAD_WRITE), calls(GOOD)], mode="validated", retry_limit=1)
    assert res.status == "invalid_state_call"
    assert res.retries_used == 1
    assert res.invalid_state_changing_calls == 2


@pytest.mark.parametrize("mode", ["direct", "validated"])
async def test_invalid_read_only_call_is_fed_back_and_counted_not_fatal(go, mode):
    res, backend = await go([calls(BAD_READ), calls(GOOD), "done"], mode=mode, retry_limit=0)
    assert res.status == "completed"
    assert res.invalid_other_calls == 1
    assert res.invalid_state_changing_calls == 0
    assert res.retries_used == 0
    assert backend.requests[1].messages[-1].content.startswith("Invalid call:")


@pytest.mark.parametrize("mode", ["direct", "validated"])
async def test_tool_error_is_fed_back_and_not_counted_as_invalid(go, mode):
    res, backend = await go([calls(MISSING), calls(GOOD), "done"], mode=mode)
    assert res.status == "completed"
    assert res.invalid_other_calls == res.invalid_state_changing_calls == 0
    assert backend.requests[1].messages[-1].content == "Error: no such file: nope.md"


async def test_calls_after_a_fatal_call_in_the_same_reply_do_not_run(go):
    res, _ = await go([calls(BAD_WRITE, GOOD)], mode="direct")
    assert res.status == "invalid_state_call"
    assert len(res.steps[0].outcomes) == 1
    assert res.final_state == RENAME.initial_state


async def test_max_steps(go):
    res, _ = await go([calls(MISSING)] * RENAME.max_steps)
    assert res.status == "max_steps"
    assert len(res.steps) == RENAME.max_steps


async def test_backend_error_ends_the_run(go):
    res, _ = await go([calls(MISSING)])  # script runs out on the 2nd call
    assert res.status == "backend_error"
    assert len(res.steps) == 1


async def test_context_cap_is_enforced(go):
    res, _ = await go([FakeTurn(content="x", prompt_tokens=600)], context_cap=512)
    assert res.status == "context_exceeded"


TEXT_CALL = '{"name": "rename_file", "parameters": {"src": "drafts/plan.md", "dst": "archive/plan.md"}}'


async def test_text_call_in_direct_mode_ends_the_run(go):
    res, backend = await go([TEXT_CALL, calls(GOOD)], mode="direct")
    assert res.status == "malformed_text_call"
    assert res.malformed_text_calls == 1
    assert res.steps[0].malformed_text_call.startswith('{"name": "rename_file"')
    assert len(backend.requests) == 1


async def test_text_call_in_validated_mode_gets_short_error_and_spends_a_retry(go):
    res, backend = await go([TEXT_CALL, calls(GOOD), "done"], mode="validated", retry_limit=2)
    assert res.status == "completed"
    assert res.malformed_text_calls == 1 and res.retries_used == 1
    feedback = backend.requests[1].messages[-1]
    assert feedback.role == "user" and "written as text" in feedback.content
    assert grade(RENAME, res.final_state).success


async def test_text_calls_and_invalid_calls_share_the_retry_budget(go):
    res, _ = await go([calls(BAD_WRITE), TEXT_CALL, calls(GOOD)], mode="validated", retry_limit=1)
    assert res.status == "malformed_text_call"
    assert res.retries_used == 1


async def test_claimed_done_without_state_change(go):
    # Reads, then claims success without the rename (like the calendar.find_slot run).
    read = ToolCall(name="read_file", arguments={"path": "drafts/plan.md"})
    res, _ = await go([calls(read), "I have moved the file."])
    assert res.status == "completed"
    assert res.state_changes == 0
    assert res.claimed_done_without_state_change


async def test_done_after_a_state_change_is_not_flagged(go):
    res, _ = await go([calls(GOOD), "Moved."])
    assert res.state_changes == 1
    assert not res.claimed_done_without_state_change


async def test_runs_that_did_not_complete_are_not_claimed_done(go):
    res, _ = await go([calls(BAD_WRITE)], mode="direct")
    assert not res.claimed_done_without_state_change


async def test_unimplemented_mode_is_refused(go):
    with pytest.raises(NotImplementedError):
        await go(["x"], mode="plan_act")
