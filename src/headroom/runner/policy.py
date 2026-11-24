"""What the agent loop does after each tool call, and after a tool call written as text.

- ok / tool_error: the result (or error text) goes back to the model. Not a failure.
- invalid, read-only or unknown tool: nothing changed, so it goes back to the model like a
  tool error, in every mode. Counted as an invalid call, never a run failure.
- invalid, state-changing: 'direct' fails the run. Feedback modes send the error back and
  spend one retry; once the run's retry budget is spent, the run fails.
- tool call written as text (backend did not parse it): same rule as an invalid
  state-changing call. 'direct' fails; feedback modes send a short error and spend a retry.
"""

from typing import Literal

from headroom.schemas.config import FEEDBACK_MODES, Mode
from headroom.schemas.run import ToolOutcome

Disposition = Literal["continue", "retry", "fail_run"]


def disposition(outcome: ToolOutcome, mode: Mode, retries_left: int) -> Disposition:
    if outcome.status != "invalid" or outcome.mutates_state is not True:
        return "continue"
    return _retry_or_fail(mode, retries_left)


def text_call_disposition(mode: Mode, retries_left: int) -> Disposition:
    return _retry_or_fail(mode, retries_left)


def _retry_or_fail(mode: Mode, retries_left: int) -> Disposition:
    if mode in FEEDBACK_MODES and retries_left > 0:
        return "retry"
    return "fail_run"
