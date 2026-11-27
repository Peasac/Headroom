"""llama.cpp adapter against a mocked HTTP transport and a fake server process.

Chunk shapes copied from llama-server b12777 (caea914e9), 2026-11-26, model path shortened."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from headroom.backends import BackendError, ContextExceededError, ModelMismatchError, is_warm, make_backend
from headroom.backends.llamacpp import LlamaCppBackend, to_openai_messages
from headroom.schemas.chat import ChatMessage, ToolCall, ToolSchema
from headroom.schemas.common import Measured
from headroom.schemas.config import LlamaCppSettings

pytestmark = pytest.mark.anyio

META = {"created": 1795711337, "id": "chatcmpl-x", "model": "m.gguf", "system_fingerprint": "b12777-caea914e9",
        "object": "chat.completion.chunk"}


def chunk(delta=None, finish=None, **extra):
    choices = [] if delta is None and finish is None else [{"finish_reason": finish, "index": 0, "delta": delta or {}}]
    return {"choices": choices, **META, **extra}


TOOL_STREAM = [
    chunk({"role": "assistant", "content": None}),
    chunk({"tool_calls": [{"index": 0, "id": "nPF1rdhM", "type": "function",
                           "function": {"name": "read_file", "arguments": "{"}}]}),
    *(chunk({"tool_calls": [{"index": 0, "function": {"arguments": a}}]}) for a in ['"path":"', "notes", "/a", ".txt", '"', "}"]),
    chunk({}, "tool_calls"),
    chunk(usage={"completion_tokens": 27, "prompt_tokens": 271, "total_tokens": 298,
                 "prompt_tokens_details": {"cached_tokens": 0}},
          timings={"cache_n": 0, "prompt_n": 271, "prompt_ms": 5018.802, "predicted_n": 27, "predicted_ms": 3406.714}),
]
TEXT_STREAM = [
    chunk({"role": "assistant", "content": None}),
    *(chunk({"content": s}) for s in ["Hello", ",", " friend", "."]),
    chunk({}, "stop"),
    chunk(usage={"completion_tokens": 5, "prompt_tokens": 273, "total_tokens": 278,
                 "prompt_tokens_details": {"cached_tokens": 255}},
          timings={"cache_n": 255, "prompt_n": 18, "prompt_ms": 2140.52, "predicted_n": 5, "predicted_ms": 125.202}),
]
OVERSIZED = {"error": {"code": 400, "message": "request (2767 tokens) exceeds the available context size (2048 tokens), "
                       "try increasing it", "type": "exceed_context_size_error", "n_prompt_tokens": 2767, "n_ctx": 2048}}
READ_FILE = ToolSchema(name="read_file", description="Read a file.", parameters={"type": "object", "properties": {}})
USER = [ChatMessage(role="user", content="hi")]


class FakeProcess:
    def __init__(self, server, args):
        self.server, self.args, self.code = server, args, None

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 1
        self.server.running = None

    kill = terminate

    def wait(self, timeout=None):
        return self.code


class Server:
    """Plays llama-server: /health is 503 for `loading_polls` polls after a launch, then 200."""

    def __init__(self, *streams, status=200, loading_polls=1, crash_on_launch=False):
        self.streams, self.status, self.loading_polls, self.crash = list(streams), status, loading_polls, crash_on_launch
        self.running: FakeProcess | None = None
        self.launches: list[list[str]] = []
        self.bodies: list[dict] = []
        self._polls = 0

    def launch(self, args, log: Path):
        self.launches.append(args)
        log.write_text("load_tensors: offloaded 33/33 layers to GPU\n" + ("CUDA error: out of memory\n" if self.crash else ""))
        proc = FakeProcess(self, args)
        if self.crash:
            proc.code = 1
        else:
            self.running, self._polls = proc, 0
        return proc

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.running is None:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/health":
            self._polls += 1
            return httpx.Response(200 if self._polls > self.loading_polls else 503, json={})
        self.bodies.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, json=OVERSIZED)
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in self.streams.pop(0)) + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode())


@pytest.fixture
def lsettings(tmp_path) -> LlamaCppSettings:
    return LlamaCppSettings(
        kind="llamacpp", server_exe=str(tmp_path / "llama-server.exe"), server_sha256="e" * 64, build="b12777",
        model_dir=str(tmp_path), port=8090, num_ctx=8192, server_args=["-np", "1", "--no-webui"],
        request_timeout_s=30, load_timeout_s=60, max_warm_load_s=0.5,
    )


def counter():
    t = [0.0]

    def tick():
        t[0] += 0.5
        return t[0]
    return tick


def backend(settings, server, **kw):
    return LlamaCppBackend(settings, transport=httpx.MockTransport(server), launcher=server.launch,
                           clock=counter(), poll_s=0, **kw)


async def test_streamed_tool_call_is_assembled(lsettings, qwen, sampling):
    async with backend(lsettings, Server(TOOL_STREAM)) as b:
        reply = await b.chat(qwen, USER, sampling=sampling, seed=1, tools=[READ_FILE], num_ctx=2048)
    (call,) = reply.message.tool_calls
    assert (call.name, call.arguments, call.call_id, call.parse_error) == ("read_file", {"path": "notes/a.txt"}, "nPF1rdhM", None)
    assert reply.done_reason == "tool_calls" and reply.message.content == ""
    assert reply.prompt_tokens == Measured(value=271, unit="tokens")
    assert reply.completion_tokens.value == 27 and reply.cached_prompt_tokens.value == 0
    assert reply.prefill_s.value == pytest.approx(5.018802) and reply.decode_s.value == pytest.approx(3.406714)


async def test_text_reply_and_cached_prompt(lsettings, qwen, sampling):
    async with backend(lsettings, Server(TEXT_STREAM)) as b:
        reply = await b.chat(qwen, USER, sampling=sampling, seed=1)
    assert reply.message.content == "Hello, friend." and reply.message.tool_calls == ()
    assert reply.prompt_tokens.value == 273 and reply.cached_prompt_tokens.value == 255


async def test_server_starts_once_and_restarts_only_when_settings_change(lsettings, qwen, sampling):
    server = Server(TEXT_STREAM, TEXT_STREAM, TEXT_STREAM, TEXT_STREAM)
    async with backend(lsettings, server) as b:
        first = await b.chat(qwen, USER, sampling=sampling, seed=1, num_ctx=2048)
        second = await b.chat(qwen, USER, sampling=sampling, seed=1, num_ctx=2048)
        third = await b.chat(qwen, USER, sampling=sampling, seed=1, num_ctx=8192, num_thread=2)
        await b.unload(qwen)
        fourth = await b.chat(qwen, USER, sampling=sampling, seed=1, num_ctx=8192, num_thread=2)
    assert len(server.launches) == 3
    assert first.load_s.value > 0 and not is_warm(first, 0.5)
    assert second.load_s == Measured(value=0.0, unit="s") and is_warm(second, 0.5)
    assert third.load_s.value > 0 and fourth.load_s.value > 0
    a, b2 = server.launches[0], server.launches[1]
    assert a[a.index("-c") + 1] == "2048" and "-t" not in a
    assert b2[b2.index("-c") + 1] == "8192" and b2[b2.index("-t") + 1] == "2"
    assert a[a.index("-m") + 1].endswith(f"sha256-{qwen.gguf_sha256}")
    assert a[-3:] == ["-np", "1", "--no-webui"]


async def test_a_server_that_dies_while_loading_is_a_backend_error(lsettings, qwen, sampling):
    async with backend(lsettings, Server(crash_on_launch=True)) as b:
        with pytest.raises(BackendError, match="(?s)exited with code 1 while loading.*out of memory"):
            await b.chat(qwen, USER, sampling=sampling, seed=1)


async def test_refuses_to_start_when_the_port_is_taken(lsettings, qwen, sampling):
    server = Server(TEXT_STREAM)
    server.running = FakeProcess(server, ["someone else"])
    server._polls = 99
    async with backend(lsettings, server) as b:
        with pytest.raises(BackendError, match="already serving"):
            await b.chat(qwen, USER, sampling=sampling, seed=1)
    assert server.launches == []


async def test_an_oversized_request_raises_context_exceeded(lsettings, qwen, sampling):
    async with backend(lsettings, Server(status=400)) as b:
        with pytest.raises(ContextExceededError) as e:
            await b.chat(qwen, USER, sampling=sampling, seed=1, num_ctx=2048)
    assert e.value.requested_tokens == 2767 and e.value.context == 2048


async def test_num_ctx_above_the_backend_maximum_is_rejected(lsettings, qwen, sampling):
    async with backend(lsettings, Server()) as b:
        with pytest.raises(ValueError, match="exceeds the backend maximum"):
            await b.chat(qwen, USER, sampling=sampling, seed=1, num_ctx=16384)


def test_request_body_sends_every_sampling_value_and_the_thinking_switch(lsettings, qwen, sampling):
    body = LlamaCppBackend(lsettings).request_body(qwen, USER, sampling=sampling, seed=7, tools=[READ_FILE])
    assert body["seed"] == 7 and body["max_tokens"] == sampling.max_tokens
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty"):
        assert body[k] == getattr(sampling, k)
    assert body["chat_template_kwargs"] == {"enable_thinking": qwen.think}
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


def test_tool_results_are_paired_with_the_calls_in_order():
    calls = (ToolCall(name="read_file", arguments={"path": "a"}, call_id="x1"),
             ToolCall(name="read_file", arguments={"path": "b"}))
    msgs = to_openai_messages([
        ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u"),
        ChatMessage(role="assistant", tool_calls=calls),
        ChatMessage(role="tool", content="A", tool_name="read_file"),
        ChatMessage(role="tool", content="B", tool_name="read_file"),
    ])
    assistant = msgs[2]
    assert [c["id"] for c in assistant["tool_calls"]] == ["x1", "call_2_1"]
    assert json.loads(assistant["tool_calls"][1]["function"]["arguments"]) == {"path": "b"}
    assert [(m["tool_call_id"], m["content"]) for m in msgs[3:]] == [("x1", "A"), ("call_2_1", "B")]


def test_a_tool_result_without_a_call_is_rejected():
    with pytest.raises(ValueError, match="no unanswered tool call"):
        to_openai_messages([ChatMessage(role="user", content="u"), ChatMessage(role="tool", content="A")])


async def test_version_checks_the_pinned_exe_hash_and_build(lsettings):
    Path(lsettings.server_exe).write_bytes(b"exe")
    good = LlamaCppBackend(lsettings, hasher=lambda p: "e" * 64, version_reader=lambda exe: "b12777 (caea914e9)")
    assert await good.version() == "b12777 (caea914e9)"
    wrong_hash = LlamaCppBackend(lsettings, hasher=lambda p: "f" * 64, version_reader=lambda exe: "b12777 (x)")
    with pytest.raises(BackendError, match="pinned"):
        await wrong_hash.version()
    wrong_build = LlamaCppBackend(lsettings, hasher=lambda p: "e" * 64, version_reader=lambda exe: "b12000 (x)")
    with pytest.raises(BackendError, match="pinned build b12777"):
        await wrong_build.version()


async def test_verify_model_hashes_the_weights_file(lsettings, qwen):
    b = LlamaCppBackend(lsettings, hasher=lambda p: qwen.gguf_sha256)
    with pytest.raises(ModelMismatchError, match="no weights file"):
        await b.verify_model(qwen)
    b.weights_path(qwen).write_bytes(b"gguf")
    await b.verify_model(qwen)
    bad = LlamaCppBackend(lsettings, hasher=lambda p: "0" * 64)
    with pytest.raises(ModelMismatchError, match="pinned"):
        await bad.verify_model(qwen)


def test_settings_reject_launch_args_the_harness_manages(lsettings):
    with pytest.raises(ValidationError, match="may not set"):
        LlamaCppSettings(**{**lsettings.model_dump(), "server_args": ["-c", "4096"]})


def test_make_backend_picks_the_adapter(lsettings, settings):
    from headroom.backends.ollama import OllamaBackend
    assert isinstance(make_backend(lsettings), LlamaCppBackend)
    assert isinstance(make_backend(settings), OllamaBackend)
