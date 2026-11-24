"""Shared test helpers: a task-aware fake backend and a quick way to make real Run records."""

from conftest import FakeSource, quiet_ok

from headroom.backends.base import BackendError
from headroom.backends.fake import FakeBackend, FakeTurn
from headroom.runner.single import execute_run
from headroom.schemas.chat import BackendReply, ToolCall
from headroom.schemas.common import measured
from headroom.schemas.run import PlannedRun
from headroom.tasks.suite import get_task

SOLUTIONS = {
    "files.rename_01": [ToolCall(name="rename_file", arguments={"src": "drafts/plan.md", "dst": "archive/plan.md"})],
    "calendar.add_01": [ToolCall(
        name="add_event", arguments={"title": "Design review", "start": "2026-12-03T14:00", "end": "2026-12-03T15:30"}
    )],
}
SAMPLE = {"sys_mem_used_mb": 8000.0, "backend_rss_mb": 300.0, "backend_private_mb": 900.0, "vram_used_mb": 3000.0}


class TaskBackend:
    """Answers by task prompt, not by call order, so it works under a shuffled plan.
    First call of a run: the task's solution. After a tool result: 'Done.' Warm-up: 'OK'."""

    def __init__(self, prompts_to_task: dict[str, str]):
        self.prompts_to_task = prompts_to_task
        self.requests: list[dict] = []

    async def version(self):
        return "fake-0"

    async def verify_model(self, model):
        return None

    async def chat(self, model, messages, *, sampling, seed, tools=()):
        self.requests.append({"model": model.model_id, "messages": tuple(messages), "tools": tuple(tools),
                              "sampling": sampling, "seed": seed})
        if sampling.max_tokens == 1:
            return _reply(model, FakeTurn(content="OK"))
        if messages[-1].role == "tool":
            return _reply(model, FakeTurn(content="Done."))
        task_id = self.prompts_to_task.get(messages[1].content)
        if task_id is None:
            raise BackendError("unknown prompt")
        return _reply(model, FakeTurn(tool_calls=tuple(SOLUTIONS[task_id])))


def _reply(model, turn: FakeTurn) -> BackendReply:
    return BackendReply(
        model_id=model.model_id, message={"role": "assistant", "content": turn.content, "tool_calls": turn.tool_calls},
        done_reason="stop", latency_s=turn.latency_s, ttft_s=measured(turn.ttft_s, "s"),
        prompt_tokens=measured(turn.prompt_tokens, "tokens"), cached_prompt_tokens=measured(0, "tokens"),
        completion_tokens=measured(turn.completion_tokens, "tokens"), load_s=measured(0.0, "s"),
        prefill_s=measured(0.0, "s"), decode_s=measured(0.0, "s"),
    )


async def make_run(experiment, provenance, turns, *, task_id="files.rename_01", arm_id="validated",
                   order_index=0, repeat=0, session_id="s1", context_cap=None):
    if context_cap is not None:
        arms = [a.model_copy(update={"config": a.config.model_copy(update={"context_cap": context_cap})})
                for a in experiment.arms]
        experiment = experiment.model_copy(update={"arms": arms})
    planned = PlannedRun(order_index=order_index, task_id=task_id, repeat=repeat, condition_id="baseline", arm_id=arm_id)
    return await execute_run(
        FakeBackend(turns), get_task(task_id), experiment=experiment, planned=planned, session_id=session_id,
        order_seed=7, quiet=quiet_ok(), backend_version="fake-0", provenance=provenance,
        sources=[FakeSource([SAMPLE])],
    )
