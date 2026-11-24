import pytest

from headroom.runner.policy import disposition
from headroom.schemas.chat import ToolCall
from headroom.schemas.run import ToolOutcome


def outcome(status: str, mutates: bool | None) -> ToolOutcome:
    return ToolOutcome(call=ToolCall(name="t", arguments={}), mutates_state=mutates, status=status, error="e")


@pytest.mark.parametrize("mode", ["direct", "validated", "plan_act", "plan_act_verify"])
@pytest.mark.parametrize("status,mutates", [
    ("ok", True), ("ok", False),
    ("tool_error", True), ("tool_error", False),
    ("invalid", False),  # read-only: nothing changed, feed back, never fatal
    ("invalid", None),   # unknown tool: same
])
def test_non_fatal_outcomes_continue_in_every_mode(mode, status, mutates):
    assert disposition(outcome(status, mutates), mode, retries_left=0) == "continue"


def test_invalid_state_changing_call_fails_direct_even_with_retries_left():
    assert disposition(outcome("invalid", True), "direct", retries_left=5) == "fail_run"


@pytest.mark.parametrize("mode", ["validated", "plan_act", "plan_act_verify"])
def test_feedback_modes_retry_until_budget_is_spent(mode):
    assert disposition(outcome("invalid", True), mode, retries_left=1) == "retry"
    assert disposition(outcome("invalid", True), mode, retries_left=0) == "fail_run"
