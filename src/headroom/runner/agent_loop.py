"""Runs one task to completion under one AgentConfig."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from headroom.backends.base import Backend, BackendError
from headroom.runner.policy import disposition, text_call_disposition
from headroom.runner.textcalls import detect_text_tool_call
from headroom.schemas.chat import ChatMessage, ToolCall
from headroom.schemas.common import Measured
from headroom.schemas.config import AgentConfig, ModelSpec, Sampling
from headroom.schemas.env import EnvState
from headroom.schemas.run import RunStatus, Step, ToolOutcome
from headroom.schemas.task import Task
from headroom.tasks.executor import execute
from headroom.tasks.tools import tool_schemas

SYSTEM_PROMPT = (
    "You complete tasks by calling the provided tools. Keep calling tools until the task is done, "
    "then reply with one short sentence and no tool call."
)
TEXT_CALL_FEEDBACK = (
    "Your last message contained a tool call written as text. It was not executed. "
    "Use the tool-calling interface to call tools."
)
IMPLEMENTED_MODES = ("direct", "validated")


@dataclass
class LoopResult:
    status: RunStatus
    final_state: EnvState
    steps: list[Step] = field(default_factory=list)
    error: str | None = None
    retries_used: int = 0
    invalid_state_changing_calls: int = 0
    invalid_other_calls: int = 0
    malformed_text_calls: int = 0
    state_changes: int = 0

    @property
    def claimed_done_without_state_change(self) -> bool:
        return self.status == "completed" and self.state_changes == 0


async def run_task(
    backend: Backend, task: Task, *, model: ModelSpec, config: AgentConfig, sampling: Sampling, seed: int
) -> LoopResult:
    if config.mode not in IMPLEMENTED_MODES:
        raise NotImplementedError(f"mode {config.mode!r} is not implemented yet (have {IMPLEMENTED_MODES})")
    if model.model_id != config.model_id:
        raise ValueError(f"model {model.model_id!r} does not match config.model_id {config.model_id!r}")

    tools = tool_schemas(task.tools)
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=task.prompt),
    ]
    res = LoopResult(status="max_steps", final_state=task.initial_state)

    for index in range(task.max_steps):
        try:
            reply = await backend.chat(model, messages, sampling=sampling, seed=seed, tools=tools)
        except BackendError as e:
            res.status, res.error = "backend_error", str(e)
            return res

        if isinstance(reply.prompt_tokens, Measured) and reply.prompt_tokens.value > config.context_cap:
            res.steps.append(Step(index=index, reply=reply))
            res.status = "context_exceeded"
            res.error = f"prompt {reply.prompt_tokens.value} tokens > context_cap {config.context_cap}"
            return res

        messages.append(reply.message)

        if not reply.message.tool_calls:
            text_call = detect_text_tool_call(reply.message.content)
            res.steps.append(Step(index=index, reply=reply, malformed_text_call=text_call))
            if text_call is None:
                res.status = "completed"
                return res
            res.malformed_text_calls += 1
            if text_call_disposition(config.mode, config.retry_limit - res.retries_used) == "fail_run":
                res.status = "malformed_text_call"
                res.error = f"tool call written as text: {text_call}"
                return res
            res.retries_used += 1
            messages.append(ChatMessage(role="user", content=TEXT_CALL_FEEDBACK))
            continue

        outcomes, failed = _run_calls(res, reply.message.tool_calls, task, config, messages)
        res.steps.append(Step(index=index, reply=reply, outcomes=tuple(outcomes)))
        if failed:
            res.status = "invalid_state_call"
            res.error = outcomes[-1].error
            return res

    return res


def _run_calls(
    res: LoopResult, calls: Sequence[ToolCall], task: Task, config: AgentConfig, messages: list[ChatMessage]
) -> tuple[list[ToolOutcome], bool]:
    outcomes: list[ToolOutcome] = []
    for call in calls:
        res.final_state, outcome = execute(res.final_state, call, task.tools)
        outcomes.append(outcome)
        if outcome.status == "invalid":
            if outcome.mutates_state is True:
                res.invalid_state_changing_calls += 1
            else:
                res.invalid_other_calls += 1
        elif outcome.status == "ok" and outcome.mutates_state:
            res.state_changes += 1

        d = disposition(outcome, config.mode, config.retry_limit - res.retries_used)
        if d == "fail_run":
            return outcomes, True  # later calls in the same reply are not executed
        if d == "retry":
            res.retries_used += 1
        messages.append(ChatMessage(role="tool", content=_feedback(outcome), tool_name=call.name))
    return outcomes, False


def _feedback(o: ToolOutcome) -> str:
    if o.status == "ok":
        return o.result or ""
    if o.status == "tool_error":
        return f"Error: {o.error}"
    return f"Invalid call: {o.error}"
