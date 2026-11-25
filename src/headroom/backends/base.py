from collections.abc import Sequence
from typing import Protocol

from headroom.schemas.chat import BackendReply, ChatMessage, ToolSchema
from headroom.schemas.common import Measured
from headroom.schemas.config import ModelSpec, Sampling


class BackendError(RuntimeError):
    """The backend failed to produce a reply (transport, HTTP or protocol error)."""


class ModelMismatchError(BackendError):
    """The model the backend serves is not the pinned one."""


class ContextExceededError(BackendError):
    """The request is larger than the context window, so the backend refused it rather than truncate it.

    requested_tokens is the backend's own count of the request, when it reports one."""

    def __init__(self, message: str, *, requested_tokens: int | None, context: int | None) -> None:
        super().__init__(message)
        self.requested_tokens = requested_tokens
        self.context = context


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
        num_ctx: int | None = None,
        num_thread: int | None = None,
    ) -> BackendReply:
        """num_ctx: context window for this call (an arm's context_cap); None uses the backend's maximum.
        num_thread: CPU threads for the runner (reduced-compute condition); None leaves the default.
        Changing either from the loaded value makes the backend reload the model."""
        ...


async def warm_up(
    backend: Backend, model: ModelSpec, sampling: Sampling, num_ctx: int | None = None, num_thread: int | None = None
) -> BackendReply:
    """One unscored call so the model is loaded, with the options the next run uses, before any measured call."""
    return await backend.chat(
        model,
        [ChatMessage(role="user", content="Reply with OK.")],
        sampling=sampling.model_copy(update={"max_tokens": 1}),
        seed=0,
        num_ctx=num_ctx,
        num_thread=num_thread,
    )


def is_warm(reply: BackendReply, max_load_s: float) -> bool | None:
    """True if the call did not pay a model load. None if the backend did not report load time."""
    if not isinstance(reply.load_s, Measured):
        return None
    return reply.load_s.value <= max_load_s
