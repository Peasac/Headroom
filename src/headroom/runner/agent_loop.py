"""Runs one task to completion under one AgentConfig.

Modes (cumulative):
  direct           tool loop; an invalid state-changing or text-written call ends the run.
  validated        tool loop; those errors are fed back and spend a retry.
  plan_act         validated, preceded by one planning call with no tools offered.
  plan_act_verify  plan_act, then, once the model says it is done, a verification pass with tools.
Each tool-loop phase (act, verify) has its own budget of task.max_steps calls.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from headroom.backends.base import Backend, BackendError
from headroom.runner.policy import disposition, text_call_disposition
from headroom.runner.textcalls import detect_text_tool_call
from headroom.schemas.chat import BackendReply, ChatMessage, ToolCall, ToolSchema
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
PLAN_PROMPT = (
    "Before acting, write a short numbered plan of the tool calls you will make to complete the task. "
    "Do not call any tools yet. These tools will be available:\n{tools}"
)
ACT_PROMPT = "Now carry out your plan using the tools."
VERIFY_PROMPT = (
    "Check that the task is fully and correctly done: use the tools to inspect the result. If anything is "
    "wrong or missing, fix it with the tools. When everything is correct, reply with one short sentence "
    "and no tool call."
)
PLANNING_MODES = ("plan_act", "plan_act_verify")

Phase = Literal["plan", "act", "verify"]


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


@dataclass
class _Run:
    backend: Backend
    task: Task
    model: ModelSpec
    config: AgentConfig
    sampling: Sampling
    seed: int
    tools: list[ToolSchema]
    messages: list[ChatMessage]
    res: LoopResult
    num_thread: int | None = None


async def run_task(
    backend: Backend,
    task: Task,
    *,
    model: ModelSpec,
    config: AgentConfig,
    sampling: Sampling,
    seed: int,
    num_thread: int | None = None,
) -> LoopResult:
    """num_thread comes from the condition (reduced compute), not from the config."""
    if model.model_id != config.model_id:
        raise ValueError(f"model {model.model_id!r} does not match config.model_id {config.model_id!r}")

    r = _Run(
        backend=backend, task=task, model=model, config=config, sampling=sampling, seed=seed, num_thread=num_thread,
        tools=tool_schemas(task.tools),
        messages=[ChatMessage(role="system", content=SYSTEM_PROMPT), ChatMessage(role="user", content=task.prompt)],
        res=LoopResult(status="max_steps", final_state=task.initial_state),
    )

    if config.mode in PLANNING_MODES:
        if not await _plan(r):
            return r.res
        r.messages.append(ChatMessage(role="user", content=ACT_PROMPT))

    await _tool_loop(r, "act")
    if config.mode == "plan_act_verify" and r.res.status == "completed":
        r.messages.append(ChatMessage(role="user", content=VERIFY_PROMPT))
        await _tool_loop(r, "verify")
    return r.res


async def _call(r: _Run, phase: Phase, tools: Sequence[ToolSchema]) -> BackendReply | None:
    """One model call. Returns None (status set) on backend error or a full context window."""
    try:
        reply = await r.backend.chat(
            r.model, r.messages, sampling=r.sampling, seed=r.seed, tools=tools, num_ctx=r.config.context_cap,
            num_thread=r.num_thread,
        )
    except BackendError as e:
        r.res.status, r.res.error = "backend_error", str(e)
        return None
    used = _tokens_used(reply)
    if used is not None and used >= r.config.context_cap:
        r.res.steps.append(Step(index=len(r.res.steps), phase=phase, reply=reply))
        r.res.status = "context_exceeded"
        r.res.error = f"prompt + reply = {used} tokens filled context_cap {r.config.context_cap}"
        return None
    return reply


def _tokens_used(reply: BackendReply) -> int | None:
    """Prompt plus reply tokens. The backend truncates a prompt longer than num_ctx, so a full window
    (not just prompt > cap) is the signal that the context was exceeded."""
    if isinstance(reply.prompt_tokens, Measured) and isinstance(reply.completion_tokens, Measured):
        return int(reply.prompt_tokens.value + reply.completion_tokens.value)
    if isinstance(reply.prompt_tokens, Measured):
        return int(reply.prompt_tokens.value)
    return None


async def _plan(r: _Run) -> bool:
    listing = "\n".join(f"- {t.name}: {t.description}" for t in r.tools)
    r.messages.append(ChatMessage(role="user", content=PLAN_PROMPT.format(tools=listing)))
    reply = await _call(r, "plan", tools=())  # no tools offered: the model can only write a plan
    if reply is None:
        return False
    r.res.steps.append(Step(index=len(r.res.steps), phase="plan", reply=reply))
    r.messages.append(reply.message)
    return True


async def _tool_loop(r: _Run, phase: Phase) -> None:
    res, config = r.res, r.config
    res.status = "max_steps"
    for _ in range(r.task.max_steps):
        reply = await _call(r, phase, r.tools)
        if reply is None:
            return
        index = len(res.steps)
        r.messages.append(reply.message)

        if not reply.message.tool_calls:
            text_call = detect_text_tool_call(reply.message.content)
            res.steps.append(Step(index=index, phase=phase, reply=reply, malformed_text_call=text_call))
            if text_call is None:
                res.status = "completed"
                return
            res.malformed_text_calls += 1
            if text_call_disposition(config.mode, config.retry_limit - res.retries_used) == "fail_run":
                res.status = "malformed_text_call"
                res.error = f"tool call written as text: {text_call}"
                return
            res.retries_used += 1
            r.messages.append(ChatMessage(role="user", content=TEXT_CALL_FEEDBACK))
            continue

        outcomes, failed = _run_calls(res, reply.message.tool_calls, r.task, config, r.messages)
        res.steps.append(Step(index=index, phase=phase, reply=reply, outcomes=tuple(outcomes)))
        if failed:
            res.status = "invalid_state_call"
            res.error = outcomes[-1].error
            return


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
