"""Scripted backend for tests. Deterministic: replies, latencies and token counts come from the script."""

from collections.abc import Sequence

from pydantic import Field

from headroom.backends.base import BackendError
from headroom.schemas.chat import BackendReply, ChatMessage, ToolCall, ToolSchema
from headroom.schemas.common import Frozen, measured
from headroom.schemas.config import ModelSpec, Sampling


class FakeTurn(Frozen):
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    latency_s: float = Field(default=0.05, ge=0)
    ttft_s: float = Field(default=0.01, ge=0)
    prompt_tokens: int = 10
    completion_tokens: int = 5


class FakeRequest(Frozen):
    model_id: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSchema, ...]
    sampling: Sampling
    seed: int
    num_ctx: int | None = None


class FakeBackend:
    def __init__(self, turns: Sequence[FakeTurn | str], *, version: str = "fake-0") -> None:
        self._turns = [FakeTurn(content=t) if isinstance(t, str) else t for t in turns]
        self._version = version
        self.requests: list[FakeRequest] = []

    async def version(self) -> str:
        return self._version

    async def verify_model(self, model: ModelSpec) -> None:
        return None

    async def chat(
        self,
        model: ModelSpec,
        messages: Sequence[ChatMessage],
        *,
        sampling: Sampling,
        seed: int,
        tools: Sequence[ToolSchema] = (),
        num_ctx: int | None = None,
    ) -> BackendReply:
        self.requests.append(FakeRequest(
            model_id=model.model_id, messages=tuple(messages), tools=tuple(tools), sampling=sampling, seed=seed,
            num_ctx=num_ctx,
        ))
        if not self._turns:
            raise BackendError("FakeBackend script exhausted")
        turn = self._turns.pop(0)
        return BackendReply(
            model_id=model.model_id,
            message=ChatMessage(role="assistant", content=turn.content, tool_calls=turn.tool_calls),
            done_reason="stop",
            latency_s=turn.latency_s,
            ttft_s=measured(turn.ttft_s, "s"),
            prompt_tokens=measured(turn.prompt_tokens, "tokens"),
            cached_prompt_tokens=measured(0, "tokens"),
            completion_tokens=measured(turn.completion_tokens, "tokens"),
            load_s=measured(0.0, "s"),
            prefill_s=measured(0.0, "s"),
            decode_s=measured(0.0, "s"),
        )
