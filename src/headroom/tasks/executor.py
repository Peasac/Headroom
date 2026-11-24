"""Executes one tool call against the environment.

Safety rule: every call is validated before it runs, and there is deliberately no parameter to
skip validation. Harness modes differ only in what the agent loop does with an 'invalid'
outcome (feed the error back, or fail the run), never in whether validation happens.
"""

from collections.abc import Collection

from pydantic import ValidationError

from headroom.schemas.chat import ToolCall
from headroom.schemas.env import EnvState
from headroom.schemas.run import ToolOutcome
from headroom.tasks.tools import REGISTRY, ToolError


def execute(state: EnvState, call: ToolCall, allowed: Collection[str]) -> tuple[EnvState, ToolOutcome]:
    spec = REGISTRY.get(call.name) if call.name in allowed else None
    if spec is None:
        return state, ToolOutcome(
            call=call, mutates_state=None, status="invalid",
            error=f"unknown tool {call.name!r}; available: {sorted(allowed)}",
        )

    def invalid(error: str) -> tuple[EnvState, ToolOutcome]:
        return state, ToolOutcome(call=call, mutates_state=spec.mutates_state, status="invalid", error=error)

    if call.parse_error:
        return invalid(call.parse_error)
    try:
        args = spec.args_model.model_validate(call.arguments)
    except ValidationError as e:
        return invalid(_describe(e))

    try:
        new_state, result = spec.fn(state, args)
    except ToolError as e:
        return state, ToolOutcome(call=call, mutates_state=spec.mutates_state, status="tool_error", error=str(e))
    return new_state, ToolOutcome(call=call, mutates_state=spec.mutates_state, status="ok", result=result)


def _describe(e: ValidationError) -> str:
    """Short, model-readable error: 'field: message; ...'."""
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
