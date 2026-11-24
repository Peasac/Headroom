"""Ollama adapter over the native /api/chat endpoint.

The native API (not the OpenAI-compatible /v1) is used because it is the one that accepts
keep_alive, num_ctx and think, and reports load, prefill and decode timings.
"""

import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from headroom.backends.base import BackendError, ModelMismatchError
from headroom.schemas.chat import BackendReply, ChatMessage, ToolCall, ToolSchema
from headroom.schemas.common import Reading, measured, unavailable
from headroom.schemas.config import BackendSettings, ModelSpec, Sampling

_FROM_BLOB = re.compile(r"^FROM\s.*?sha256[-:]([0-9a-f]{64})\s*$", re.MULTILINE)


class OllamaBackend:
    def __init__(
        self,
        settings: BackendSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._client = httpx.AsyncClient(
            base_url=settings.base_url, timeout=settings.request_timeout_s, transport=transport
        )

    async def __aenter__(self) -> "OllamaBackend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def version(self) -> str:
        data = await self._get_json("GET", "/api/version")
        return str(data["version"])

    async def verify_model(self, model: ModelSpec) -> None:
        data = await self._get_json("POST", "/api/show", json={"model": model.backend_name})
        blobs = _FROM_BLOB.findall(data.get("modelfile", ""))
        if model.gguf_sha256 not in blobs:
            raise ModelMismatchError(
                f"{model.backend_name}: pinned weights sha256 {model.gguf_sha256} not served; "
                f"backend has {blobs or 'no blob hashes'}"
            )
        quant = (data.get("details") or {}).get("quantization_level")
        if quant != model.quant:
            raise ModelMismatchError(f"{model.backend_name}: quantization {quant!r}, pinned {model.quant!r}")

    def request_body(
        self,
        model: ModelSpec,
        messages: Sequence[ChatMessage],
        *,
        sampling: Sampling,
        seed: int,
        tools: Sequence[ToolSchema] = (),
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model.backend_name,
            "messages": [_to_ollama_message(m) for m in messages],
            "stream": True,
            "keep_alive": self.settings.keep_alive,
            "options": {
                "num_ctx": self.settings.num_ctx,
                "seed": seed,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "repeat_penalty": sampling.repeat_penalty,
                "presence_penalty": sampling.presence_penalty,
                "frequency_penalty": sampling.frequency_penalty,
                "num_predict": sampling.max_tokens,
            },
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        if model.think is not None:
            body["think"] = model.think
        return body

    async def chat(
        self,
        model: ModelSpec,
        messages: Sequence[ChatMessage],
        *,
        sampling: Sampling,
        seed: int,
        tools: Sequence[ToolSchema] = (),
    ) -> BackendReply:
        body = self.request_body(model, messages, sampling=sampling, seed=seed, tools=tools)
        content: list[str] = []
        thinking: list[str] = []
        calls: list[ToolCall] = []
        final: dict[str, Any] | None = None
        first: float | None = None

        t0 = self._clock()
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode(errors="replace")
                    raise BackendError(f"Ollama /api/chat returned {resp.status_code}: {detail[:500]}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise BackendError(f"Ollama error: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    if first is None and (msg.get("content") or msg.get("thinking") or msg.get("tool_calls")):
                        first = self._clock()
                    content.append(msg.get("content") or "")
                    thinking.append(msg.get("thinking") or "")
                    calls.extend(_parse_tool_call(tc) for tc in msg.get("tool_calls") or [])
                    if chunk.get("done"):
                        final = chunk
                        break
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama request failed: {e!r}") from e
        t1 = self._clock()

        if final is None:
            raise BackendError("Ollama stream ended without a final (done) chunk")

        return BackendReply(
            model_id=model.model_id,
            message=ChatMessage(
                role="assistant",
                content="".join(content),
                tool_calls=tuple(calls),
                thinking="".join(thinking) or None,
            ),
            done_reason=final.get("done_reason"),
            latency_s=t1 - t0,
            ttft_s=measured(first - t0, "s") if first is not None else unavailable("reply carried no output"),
            prompt_tokens=_count(final, "prompt_eval_count"),
            cached_prompt_tokens=_count(final, "prompt_eval_cached_count"),
            completion_tokens=_count(final, "eval_count"),
            load_s=_seconds(final, "load_duration"),
            prefill_s=_seconds(final, "prompt_eval_duration"),
            decode_s=_seconds(final, "eval_duration"),
        )

    async def _get_json(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            resp = await self._client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama {path} failed: {e!r}") from e
        if resp.status_code != 200:
            raise BackendError(f"Ollama {path} returned {resp.status_code}: {resp.text[:500]}")
        return resp.json()


def _to_ollama_message(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [_to_ollama_call(c) for c in m.tool_calls]
    if m.tool_name is not None:
        out["tool_name"] = m.tool_name
    return out


def _to_ollama_call(c: ToolCall) -> dict[str, Any]:
    """Send a call back as Ollama returned it (docs: append response.message unchanged): id and
    function.index included when the backend gave them."""
    fn: dict[str, Any] = {"name": c.name, "arguments": c.arguments}
    if c.index is not None:
        fn = {"index": c.index, **fn}
    return {"id": c.call_id, "function": fn} if c.call_id is not None else {"function": fn}


def _parse_tool_call(tc: dict[str, Any]) -> ToolCall:
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    error = None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            error = f"arguments are not valid JSON: {e}"
    if not isinstance(args, dict):
        error = error or f"arguments are {type(args).__name__}, not an object"
        args = {}
    index = fn.get("index")
    return ToolCall(
        name=str(fn.get("name", "")), arguments=args, call_id=tc.get("id"),
        index=index if isinstance(index, int) else None, parse_error=error,
    )


def _count(final: dict[str, Any], key: str) -> Reading:
    if isinstance(final.get(key), int):
        return measured(final[key], "tokens")
    return unavailable(f"Ollama did not report {key}")


def _seconds(final: dict[str, Any], key: str) -> Reading:
    if isinstance(final.get(key), int):
        return measured(final[key] / 1e9, "s")
    return unavailable(f"Ollama did not report {key}")
