from collections.abc import Sequence
from typing import Protocol

from headroom.schemas.chat import BackendReply, ChatMessage, ToolSchema
from headroom.schemas.common import Measured
from headroom.schemas.config import ModelSpec, Sampling


class BackendError(RuntimeError):
    """The backend failed to produce a reply (transport, HTTP or protocol error)."""


class ModelMismatchError(BackendError):
    """The model the backend serves is not the pinned one."""


class Backend(Protocol):
    async def version(self) -> str: ...

    async def verify_model(self, model: ModelSpec) -> None:
        """Raise ModelMismatchError unless the backend serves exactly the pinned weights."""
        ...

    async def chat(
        self,
        model: ModelSpec,
        messages: Sequence[ChatMessage],
        *,
        sampling: Sampling,
        seed: int,
        tools: Sequence[ToolSchema] = (),
    ) -> BackendReply: ...


async def warm_up(backend: Backend, model: ModelSpec, sampling: Sampling) -> BackendReply:
    """One unscored call so the model is loaded before any measured call."""
    return await backend.chat(
        model,
        [ChatMessage(role="user", content="Reply with OK.")],
        sampling=sampling.model_copy(update={"max_tokens": 1}),
        seed=0,
    )


def is_warm(reply: BackendReply, max_load_s: float) -> bool | None:
    """True if the call did not pay a model load. None if the backend did not report load time."""
    if not isinstance(reply.load_s, Measured):
        return None
    return reply.load_s.value <= max_load_s
