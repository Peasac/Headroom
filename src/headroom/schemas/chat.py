from typing import Any, Literal

from pydantic import Field

from headroom.schemas.common import SCHEMA_VERSION, Frozen, Reading


class ToolSchema(Frozen):
    """What the model is shown for one tool. JSON Schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(Frozen):
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    index: int | None = Field(default=None, description="Position among parallel calls, as the backend returned it.")
    parse_error: str | None = Field(
        default=None, description="Set when the backend returned arguments that are not a JSON object."
    )


class ChatMessage(Frozen):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = Field(default=None, description="For role='tool': which tool produced this result.")
    thinking: str | None = None


class BackendReply(Frozen):
    """One model call, as recorded. Every agent step embeds one of these."""

    schema_version: int = SCHEMA_VERSION
    model_id: str
    message: ChatMessage
    done_reason: str | None
    latency_s: float = Field(ge=0, description="Client wall time from request sent to final chunk.")
    ttft_s: Reading = Field(
        description="Client time to first chunk carrying content, thinking or a tool call. "
        "Ollama emits a tool call whole, so for tool calls this is close to latency_s; "
        "use prefill_s for the model-side time to first token."
    )
    prompt_tokens: Reading = Field(description="Total prompt tokens, cached ones included.")
    cached_prompt_tokens: Reading = Field(description="Prompt tokens served from the backend's prompt cache.")
    completion_tokens: Reading
    load_s: Reading = Field(description="Server-reported model load time. Near zero when warm.")
    prefill_s: Reading = Field(description="Server-reported prompt evaluation time.")
    decode_s: Reading = Field(description="Server-reported generation time.")
