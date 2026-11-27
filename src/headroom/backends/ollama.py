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

from headroom.backends.base import BackendError, ContextExceededError, ModelMismatchError, parse_tool_call
from headroom.schemas.chat import BackendReply, ChatMessage, ToolCall, ToolSchema
from headroom.schemas.common import Reading, measured, unavailable
from headroom.schemas.config import ModelSpec, OllamaSettings, Sampling

_FROM_BLOB = re.compile(r"^FROM\s.*?sha256[-:]([0-9a-f]{64})\s*$", re.MULTILINE)


class OllamaBackend:
    def __init__(
        self,
        settings: OllamaSettings,
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
        num_ctx: int | None = None,
        num_thread: int | None = None,
    ) -> dict[str, Any]:
        if num_ctx is not None and num_ctx > self.settings.num_ctx:
            raise ValueError(f"num_ctx {num_ctx} exceeds the backend maximum {self.settings.num_ctx}")
        body: dict[str, Any] = {
            "model": model.backend_name,
            "messages": [_to_ollama_message(m) for m in messages],
            "stream": True,
            "keep_alive": self.settings.keep_alive,
            # Without this, Ollama silently cuts a prompt longer than num_ctx down to num_ctx/2 + 2
            # tokens and answers from the remnant. With it, an oversized request fails with HTTP 400.
            "truncate": False,
            "options": {
                "num_ctx": num_ctx if num_ctx is not None else self.settings.num_ctx,
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
        if num_thread is not None:
            body["options"]["num_thread"] = num_thread
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
        num_ctx: int | None = None,
        num_thread: int | None = None,
    ) -> BackendReply:
        body = self.request_body(
            model, messages, sampling=sampling, seed=seed, tools=tools, num_ctx=num_ctx, num_thread=num_thread
        )
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
                    _raise_if_context_exceeded(detail)
                    raise BackendError(f"Ollama /api/chat returned {resp.status_code}: {detail[:500]}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        _raise_if_context_exceeded(str(chunk["error"]))
                        raise BackendError(f"Ollama error: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    if first is None and (msg.get("content") or msg.get("thinking") or msg.get("tool_calls")):
                        first = self._clock()
                    content.append(msg.get("content") or "")
                    thinking.append(msg.get("thinking") or "")
                    calls.extend(parse_tool_call(tc) for tc in msg.get("tool_calls") or [])
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

    async def unload(self, model: ModelSpec) -> None:
        """Unload the model now (keep_alive 0), so the next call loads it under current conditions."""
        await self._get_json("POST", "/api/generate", json={"model": model.backend_name, "keep_alive": 0})

    async def loaded(self, model: ModelSpec) -> tuple[Reading, Reading]:
        """(total size, size in VRAM) of the loaded model, from /api/ps."""
        try:
            data = await self._get_json("GET", "/api/ps")
        except BackendError as e:
            return unavailable(str(e)), unavailable(str(e))
        for m in data.get("models", []):
            if m.get("name") == model.backend_name or m.get("model") == model.backend_name:
                return measured(m["size"] / 2**20, "MB"), measured(m["size_vram"] / 2**20, "MB")
        reason = f"{model.backend_name} is not loaded"
        return unavailable(reason), unavailable(reason)

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


def _count(final: dict[str, Any], key: str) -> Reading:
    if isinstance(final.get(key), int):
        return measured(final[key], "tokens")
    return unavailable(f"Ollama did not report {key}")


def _seconds(final: dict[str, Any], key: str) -> Reading:
    if isinstance(final.get(key), int):
        return measured(final[key] / 1e9, "s")
    return unavailable(f"Ollama did not report {key}")


_CONTEXT_EXCEEDED = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")


def _raise_if_context_exceeded(detail: str) -> None:
    """Ollama's refusal of an oversized request (sent with truncate=false)."""
    m = _CONTEXT_EXCEEDED.search(detail)
    if m:
        raise ContextExceededError(
            f"request of {m[1]} tokens exceeds the context window of {m[2]} tokens",
            requested_tokens=int(m[1]), context=int(m[2]),
        )
