"""Ollama adapter against a mocked HTTP transport. Chunk shapes copied from Ollama 0.40.16."""

import json

import httpx
import pytest

from headroom.backends import BackendError, ContextExceededError, ModelMismatchError, is_warm
from headroom.backends.ollama import OllamaBackend
from headroom.schemas.chat import BackendReply, ChatMessage, ToolCall, ToolSchema
from headroom.schemas.common import Measured, Unavailable

pytestmark = pytest.mark.anyio

FINAL = {
    "model": "llama3.2:3b", "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
    "total_duration": 263222900, "load_duration": 13564400, "prompt_eval_count": 151,
    "prompt_eval_cached_count": 150, "prompt_eval_duration": 17657000, "eval_count": 14, "eval_duration": 224660000,
}
TOOL_CHUNK = {
    "model": "llama3.2:3b", "done": False,
    "message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_g68ag0xg", "function": {"index": 0, "name": "read_file", "arguments": {"path": "a.txt"}}}
    ]},
}
READ_FILE = ToolSchema(name="read_file", description="Read a file.", parameters={"type": "object", "properties": {}})


def text_chunk(s: str) -> dict:
    return {"model": "llama3.2:3b", "message": {"role": "assistant", "content": s}, "done": False}


class Server:
    """Records requests and answers /api/chat with the given chunks as NDJSON."""

    def __init__(self, chunks=(), status=200, show=None):
        self.chunks, self.status, self.show = list(chunks), status, show
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.bodies.append(body)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/show":
            return httpx.Response(200, json=self.show) if self.show else httpx.Response(404, json={"error": "not found"})
        ndjson = "\n".join(json.dumps(c) for c in self.chunks) + "\n"
        return httpx.Response(self.status, content=ndjson.encode())


def clock(*ticks: float):
    it = iter(ticks)
    return lambda: next(it)


def backend(settings, server, ticks=(0.0, 0.25, 1.0)):
    return OllamaBackend(settings, transport=httpx.MockTransport(server), clock=clock(*ticks))


USER = [ChatMessage(role="user", content="hi")]


async def test_text_reply_records_reply_latency_and_tokens(settings, llama, sampling):
    server = Server([text_chunk("Hello"), text_chunk(" there"), FINAL])
    async with backend(settings, server) as b:
        reply = await b.chat(llama, USER, sampling=sampling, seed=1)

    assert reply.message.content == "Hello there"
    assert reply.message.tool_calls == ()
    assert reply.latency_s == 1.0
    assert reply.ttft_s == Measured(value=0.25, unit="s")
    assert reply.prompt_tokens == Measured(value=151, unit="tokens")
    assert reply.cached_prompt_tokens == Measured(value=150, unit="tokens")
    assert reply.completion_tokens == Measured(value=14, unit="tokens")
    assert reply.load_s.value == pytest.approx(0.0135644)
    assert reply.prefill_s.value == pytest.approx(0.017657)
    assert reply.decode_s.value == pytest.approx(0.22466)
    assert reply.done_reason == "stop"
    assert is_warm(reply, settings.max_warm_load_s) is True
    assert BackendReply.model_validate_json(reply.model_dump_json()) == reply


async def test_tool_call_reply_is_parsed(settings, llama, sampling):
    server = Server([TOOL_CHUNK, FINAL])
    async with backend(settings, server) as b:
        reply = await b.chat(llama, USER, sampling=sampling, seed=1, tools=[READ_FILE])
    assert reply.message.tool_calls == (
        ToolCall(name="read_file", arguments={"path": "a.txt"}, call_id="call_g68ag0xg", index=0),
    )
    assert reply.ttft_s == Measured(value=0.25, unit="s")


async def test_string_arguments_that_are_not_json_are_recorded_not_raised(settings, llama, sampling):
    bad = {"model": "m", "done": False, "message": {"role": "assistant", "content": "",
           "tool_calls": [{"function": {"name": "read_file", "arguments": "{path: a.txt"}}]}}
    async with backend(settings, Server([bad, FINAL])) as b:
        reply = await b.chat(llama, USER, sampling=sampling, seed=1)
    (call,) = reply.message.tool_calls
    assert call.arguments == {} and call.parse_error


async def test_request_pins_keep_alive_num_ctx_seed_and_every_sampling_value(settings, llama, sampling):
    server = Server([text_chunk("x"), FINAL])
    async with backend(settings, server) as b:
        await b.chat(llama, USER, sampling=sampling, seed=42, tools=[READ_FILE])
    body = server.bodies[0]
    assert body["stream"] is True
    assert body["keep_alive"] == "30m"
    assert body["options"] == {
        "num_ctx": 8192, "seed": 42, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "num_predict": 256,
    }
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "think" not in body  # llama has no thinking switch


async def test_per_call_num_ctx_overrides_the_maximum_but_cannot_exceed_it(settings, llama, sampling):
    server = Server([text_chunk("x"), FINAL])
    async with backend(settings, server) as b:
        await b.chat(llama, USER, sampling=sampling, seed=1, num_ctx=2048)
        with pytest.raises(ValueError):
            await b.chat(llama, USER, sampling=sampling, seed=1, num_ctx=16384)
    assert server.bodies[0]["options"]["num_ctx"] == 2048


async def test_think_is_sent_when_pinned(settings, qwen, sampling):
    server = Server([text_chunk("x"), FINAL])
    async with backend(settings, server) as b:
        await b.chat(qwen, USER, sampling=sampling, seed=1)
    assert server.bodies[0]["think"] is False


async def test_tool_history_is_serialized(settings, llama, sampling):
    history = [
        ChatMessage(role="user", content="read a.txt"),
        ChatMessage(role="assistant", tool_calls=(ToolCall(name="read_file", arguments={"path": "a.txt"}),)),
        ChatMessage(role="tool", content="contents", tool_name="read_file"),
    ]
    server = Server([text_chunk("done"), FINAL])
    async with backend(settings, server) as b:
        await b.chat(llama, history, sampling=sampling, seed=1)
    msgs = server.bodies[0]["messages"]
    assert msgs[1]["tool_calls"] == [{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]
    assert msgs[2] == {"role": "tool", "content": "contents", "tool_name": "read_file"}


async def test_returned_tool_call_goes_back_exactly_as_ollama_sent_it(settings, llama, sampling):
    """Docs: append response.message unchanged. The id and function.index must survive the round trip."""
    async with backend(settings, Server([TOOL_CHUNK, FINAL])) as b:
        reply = await b.chat(llama, USER, sampling=sampling, seed=1, tools=[READ_FILE])
    server = Server([text_chunk("done"), FINAL])
    history = [*USER, reply.message, ChatMessage(role="tool", content="x", tool_name="read_file")]
    async with backend(settings, server) as b:
        await b.chat(llama, history, sampling=sampling, seed=1, tools=[READ_FILE])
    sent = server.bodies[0]["messages"][1]
    assert sent["role"] == "assistant"
    assert sent["tool_calls"] == TOOL_CHUNK["message"]["tool_calls"]


async def test_missing_counters_are_unavailable_not_zero(settings, llama, sampling):
    final = {k: v for k, v in FINAL.items() if k not in ("prompt_eval_count", "prompt_eval_cached_count", "load_duration")}
    async with backend(settings, Server([text_chunk("x"), final])) as b:
        reply = await b.chat(llama, USER, sampling=sampling, seed=1)
    assert isinstance(reply.prompt_tokens, Unavailable)
    assert isinstance(reply.cached_prompt_tokens, Unavailable)
    assert isinstance(reply.load_s, Unavailable)
    assert is_warm(reply, settings.max_warm_load_s) is None


async def test_empty_reply_has_unavailable_ttft(settings, llama, sampling):
    async with backend(settings, Server([FINAL]), ticks=(0.0, 1.0)) as b:
        reply = await b.chat(llama, USER, sampling=sampling, seed=1)
    assert isinstance(reply.ttft_s, Unavailable)


async def test_cold_load_is_detected(settings, llama, sampling):
    cold = {**FINAL, "load_duration": 7_049_813_900}
    async with backend(settings, Server([text_chunk("x"), cold])) as b:
        reply = await b.chat(llama, USER, sampling=sampling, seed=1)
    assert is_warm(reply, settings.max_warm_load_s) is False


@pytest.mark.parametrize("chunks,status", [
    ([{"error": "model 'x' not found"}], 200),
    ([], 404),
    ([text_chunk("truncated")], 200),  # no done chunk
])
async def test_backend_failures_raise(settings, llama, sampling, chunks, status):
    async with backend(settings, Server(chunks, status=status)) as b:
        with pytest.raises(BackendError):
            await b.chat(llama, USER, sampling=sampling, seed=1)


OVERSIZED = "request (2307 tokens) exceeds the available context size (2048 tokens), try increasing it"


@pytest.mark.parametrize("chunks,status", [([{"error": OVERSIZED}], 400), ([{"error": OVERSIZED}], 200)])
async def test_an_oversized_request_raises_context_exceeded(settings, llama, sampling, chunks, status):
    async with backend(settings, Server(chunks, status=status)) as b:
        with pytest.raises(ContextExceededError) as e:
            await b.chat(llama, USER, sampling=sampling, seed=1, num_ctx=2048)
    assert e.value.requested_tokens == 2307 and e.value.context == 2048


def test_requests_never_let_ollama_truncate(settings, llama, sampling):
    body = OllamaBackend(settings).request_body(llama, USER, sampling=sampling, seed=1)
    assert body["truncate"] is False


def show(*blobs: str, quant="Q4_K_M") -> dict:
    modelfile = "\n".join(f"FROM C:\\Users\\u\\.ollama\\models\\blobs\\sha256-{b}" for b in blobs) + "\nTEMPLATE x\n"
    return {"modelfile": modelfile, "details": {"quantization_level": quant}}


async def test_verify_model_accepts_pinned_weights_among_blobs(settings, llama):
    async with backend(settings, Server(show=show("c" * 64, llama.gguf_sha256))) as b:
        await b.verify_model(llama)
        assert await b.version() == "0.40.16"


@pytest.mark.parametrize("payload", [show("c" * 64), show("a" * 64, quant="Q8_0")])
async def test_verify_model_rejects_wrong_weights_or_quant(settings, llama, payload):
    async with backend(settings, Server(show=payload)) as b:
        with pytest.raises(ModelMismatchError):
            await b.verify_model(llama)


async def test_verify_model_missing_model_raises(settings, llama):
    async with backend(settings, Server()) as b:
        with pytest.raises(BackendError):
            await b.verify_model(llama)
