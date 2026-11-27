"""Adapter for a stock llama.cpp server (llama-server), started and stopped by the harness.

What differs from the Ollama path is the serving stack, not the inference library (Ollama 0.40 runs
llama.cpp too): the prompt is rendered from the model's own Jinja chat template, tool calls are parsed
by llama.cpp, and llama.cpp chooses layer placement itself. The weights file is the same one Ollama
serves, pinned by sha256.

llama-server loads one model with a fixed context size and thread count, so a call that needs a
different (model, num_ctx, num_thread) restarts it. The restart time is reported as that call's
load_s; a call on an already-running server reports load_s = 0. unload() stops the server, which also
drops its prompt cache.
"""

import asyncio
import hashlib
import json
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx

from headroom.backends.base import BackendError, ContextExceededError, ModelMismatchError, parse_tool_call
from headroom.schemas.chat import BackendReply, ChatMessage, ToolSchema
from headroom.schemas.common import Reading, measured, unavailable
from headroom.schemas.config import LlamaCppSettings, ModelSpec, Sampling


class Process(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


Launcher = Callable[[list[str], Path], Process]


def launch(args: list[str], log_path: Path) -> Process:
    """Start llama-server with its output in log_path, bound to the harness (killed if the harness dies)."""
    from headroom.conditions.procs import bind_to_harness

    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
    bind_to_harness(proc)
    return proc


def exe_version(exe: str) -> str:
    """'b12777 (caea914e9)' from `llama-server --version`."""
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    m = re.search(r"build (\d+), commit (\w+)", out.stdout + out.stderr)
    if not m:
        raise BackendError(f"could not read the llama-server version: {(out.stdout + out.stderr)[:300]!r}")
    return f"b{m[1]} ({m[2]})"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


class LlamaCppBackend:
    def __init__(
        self,
        settings: LlamaCppSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
        launcher: Launcher = launch,
        version_reader: Callable[[str], str] = exe_version,
        hasher: Callable[[Path], str] = sha256_file,
        poll_s: float = 0.1,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._launch = launcher
        self._read_version = version_reader
        self._hash = hasher
        self._poll_s = poll_s
        self._client = httpx.AsyncClient(
            base_url=f"http://{settings.host}:{settings.port}", timeout=settings.request_timeout_s, transport=transport
        )
        self._proc: Process | None = None
        self._key: tuple[str, int, int | None] | None = None
        self._logs = Path(tempfile.mkdtemp(prefix="headroom-llama-server-"))
        self._starts = 0
        self._verified: set[str] = set()

    async def __aenter__(self) -> "LlamaCppBackend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._stop()
        await self._client.aclose()

    # ---- identity ---------------------------------------------------------------------------

    async def version(self) -> str:
        """Checks the exe against its pinned hash and build, then returns e.g. 'b12777 (caea914e9)'."""
        exe = Path(self.settings.server_exe)
        if not exe.is_file():
            raise BackendError(f"llama-server not found at {exe}")
        digest = await asyncio.to_thread(self._hash, exe)
        if digest != self.settings.server_sha256:
            raise BackendError(f"{exe}: sha256 {digest}, pinned {self.settings.server_sha256}")
        version = await asyncio.to_thread(self._read_version, str(exe))
        if version.split()[0] != self.settings.build:
            raise BackendError(f"{exe} reports {version}, pinned build {self.settings.build}")
        return version

    def weights_path(self, model: ModelSpec) -> Path:
        return Path(self.settings.model_dir) / f"sha256-{model.gguf_sha256}"

    async def verify_model(self, model: ModelSpec) -> None:
        """Hashes the weights file (once per backend). The quantization is fixed by the hash: the same file
        Ollama reports as model.quant."""
        if model.gguf_sha256 in self._verified:
            return
        path = self.weights_path(model)
        if not path.is_file():
            raise ModelMismatchError(f"{model.backend_name}: no weights file {path}")
        digest = await asyncio.to_thread(self._hash, path)
        if digest != model.gguf_sha256:
            raise ModelMismatchError(f"{path}: sha256 {digest}, pinned {model.gguf_sha256}")
        self._verified.add(model.gguf_sha256)

    # ---- server lifecycle -------------------------------------------------------------------

    def launch_args(self, model: ModelSpec, num_ctx: int, num_thread: int | None) -> list[str]:
        s = self.settings
        args = [s.server_exe, "-m", str(self.weights_path(model)), "-c", str(num_ctx), "--host", s.host, "--port", str(s.port)]
        if num_thread is not None:
            args += ["-t", str(num_thread)]
        return args + list(s.server_args)

    async def _ensure(self, model: ModelSpec, num_ctx: int, num_thread: int | None) -> float:
        """Make sure the server runs with these settings. Returns the seconds spent (re)starting it."""
        key = (model.gguf_sha256, num_ctx, num_thread)
        if self._proc is not None and self._proc.poll() is None and self._key == key:
            return 0.0
        self._stop()
        if await self._healthy():
            raise BackendError(
                f"something else is already serving on {self.settings.host}:{self.settings.port}; refusing to start"
            )
        self._starts += 1
        log = self._logs / f"start-{self._starts:05d}.log"
        t0 = self._clock()
        self._proc = self._launch(self.launch_args(model, num_ctx, num_thread), log)
        self._key = key
        while not await self._healthy():
            code = self._proc.poll()
            if code is not None:
                self._proc, self._key = None, None
                raise BackendError(f"llama-server exited with code {code} while loading: {_tail(log)}")
            if self._clock() - t0 > self.settings.load_timeout_s:
                self._stop()
                raise BackendError(f"llama-server not healthy after {self.settings.load_timeout_s}s: {_tail(log)}")
            await asyncio.sleep(self._poll_s)
        return self._clock() - t0

    async def _healthy(self) -> bool:
        try:
            return (await self._client.get("/health", timeout=2.0)).status_code == 200
        except httpx.HTTPError:
            return False

    def _stop(self) -> None:
        proc, self._proc, self._key = self._proc, None, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)

    async def unload(self, model: ModelSpec) -> None:
        """Stop the server (any model), so the next call loads under current conditions with an empty cache."""
        self._stop()

    @property
    def log_dir(self) -> Path:
        return self._logs

    # ---- chat -------------------------------------------------------------------------------

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
            "messages": to_openai_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": seed,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repeat_penalty": sampling.repeat_penalty,
            "presence_penalty": sampling.presence_penalty,
            "frequency_penalty": sampling.frequency_penalty,
            "max_tokens": sampling.max_tokens,
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        if model.think is not None:
            body["chat_template_kwargs"] = {"enable_thinking": model.think}
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
        ctx = num_ctx if num_ctx is not None else self.settings.num_ctx
        if ctx > self.settings.num_ctx:
            raise ValueError(f"num_ctx {ctx} exceeds the backend maximum {self.settings.num_ctx}")
        load_s = await self._ensure(model, ctx, num_thread)
        body = self.request_body(model, messages, sampling=sampling, seed=seed, tools=tools)

        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish: str | None = None
        final: dict[str, Any] | None = None
        first: float | None = None

        t0 = self._clock()
        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode(errors="replace")
                    _raise_if_context_exceeded(detail)
                    raise BackendError(f"llama-server returned {resp.status_code}: {detail[:500]}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        _raise_if_context_exceeded(json.dumps(chunk["error"]))
                        raise BackendError(f"llama-server error: {chunk['error']}")
                    if "timings" in chunk or "usage" in chunk:
                        final = chunk
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                            first = self._clock()
                        content.append(delta.get("content") or "")
                        reasoning.append(delta.get("reasoning_content") or "")
                        for tc in delta.get("tool_calls") or []:
                            _merge_call(calls, tc)
                        finish = choice.get("finish_reason") or finish
        except httpx.HTTPError as e:
            raise BackendError(f"llama-server request failed: {e!r}{self._died()}") from e
        t1 = self._clock()

        if final is None:
            raise BackendError(f"llama-server stream ended without usage/timings{self._died()}")
        usage, timings = final.get("usage") or {}, final.get("timings") or {}
        return BackendReply(
            model_id=model.model_id,
            message=ChatMessage(
                role="assistant",
                content="".join(content),
                tool_calls=tuple(
                    parse_tool_call({"id": c["id"], "function": {"index": i, "name": c["name"], "arguments": c["arguments"]}})
                    for i, c in sorted(calls.items())
                ),
                thinking="".join(reasoning) or None,
            ),
            done_reason=finish,
            latency_s=t1 - t0,
            ttft_s=measured(first - t0, "s") if first is not None else unavailable("reply carried no output"),
            prompt_tokens=_tokens(usage.get("prompt_tokens"), "usage.prompt_tokens"),
            cached_prompt_tokens=_tokens(timings.get("cache_n"), "timings.cache_n"),
            completion_tokens=_tokens(usage.get("completion_tokens"), "usage.completion_tokens"),
            load_s=measured(load_s, "s"),
            prefill_s=_ms(timings.get("prompt_ms"), "timings.prompt_ms"),
            decode_s=_ms(timings.get("predicted_ms"), "timings.predicted_ms"),
        )

    def _died(self) -> str:
        if self._proc is not None and (code := self._proc.poll()) is not None:
            return f" (llama-server exited with code {code}: {_tail(self._logs / f'start-{self._starts:05d}.log')})"
        return ""


def to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """OpenAI chat format. Tool results carry no call id in our history, so each one is paired, in order,
    with the calls of the assistant message before it (the agent loop answers calls in the order made).
    Calls without an id get a deterministic one."""
    out: list[dict[str, Any]] = []
    pending: list[str] = []
    for i, m in enumerate(messages):
        if m.role == "assistant":
            ids = [c.call_id or f"call_{i}_{j}" for j, c in enumerate(m.tool_calls)]
            msg: dict[str, Any] = {"role": "assistant", "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": cid, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for cid, c in zip(ids, m.tool_calls, strict=True)
                ]
            pending = ids
            out.append(msg)
        elif m.role == "tool":
            if not pending:
                raise ValueError(f"message {i}: tool result with no unanswered tool call before it")
            out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _merge_call(calls: dict[int, dict[str, Any]], tc: dict[str, Any]) -> None:
    """Streamed tool calls arrive as fragments keyed by index: id and name once, arguments in pieces."""
    c = calls.setdefault(int(tc.get("index", 0)), {"id": None, "name": "", "arguments": ""})
    c["id"] = tc.get("id") or c["id"]
    fn = tc.get("function") or {}
    c["name"] += fn.get("name") or ""
    c["arguments"] += fn.get("arguments") or ""


def _tokens(v: Any, what: str) -> Reading:
    return measured(v, "tokens") if isinstance(v, int) else unavailable(f"llama-server did not report {what}")


def _ms(v: Any, what: str) -> Reading:
    return measured(v / 1000, "s") if isinstance(v, (int, float)) else unavailable(f"llama-server did not report {what}")


def _tail(log: Path, n: int = 600) -> str:
    try:
        return log.read_text(encoding="utf-8", errors="replace")[-n:].strip()
    except OSError as e:
        return f"(no log: {e})"


_CONTEXT_EXCEEDED = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")


def _raise_if_context_exceeded(detail: str) -> None:
    """llama-server refuses a prompt larger than the context (HTTP 400, exceed_context_size_error)."""
    m = _CONTEXT_EXCEEDED.search(detail)
    if m:
        raise ContextExceededError(
            f"request of {m[1]} tokens exceeds the context window of {m[2]} tokens",
            requested_tokens=int(m[1]), context=int(m[2]),
        )
