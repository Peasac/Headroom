import pytest

from headroom.backends import BackendError, is_warm, warm_up
from headroom.backends.fake import FakeBackend, FakeTurn
from headroom.schemas.chat import BackendReply, ChatMessage, ToolCall
from headroom.schemas.common import Measured

pytestmark = pytest.mark.anyio


async def test_reply_records_text_latency_and_token_counts(llama, sampling):
    backend = FakeBackend([FakeTurn(content="hello", latency_s=0.42, ttft_s=0.07, prompt_tokens=31, completion_tokens=4)])
    reply = await backend.chat(llama, [ChatMessage(role="user", content="hi")], sampling=sampling, seed=3)

    assert reply.model_id == "llama3.2-3b"
    assert reply.message.content == "hello"
    assert reply.latency_s == 0.42
    assert reply.ttft_s == Measured(value=0.07, unit="s")
    assert reply.prompt_tokens == Measured(value=31, unit="tokens")
    assert reply.completion_tokens == Measured(value=4, unit="tokens")
    # The record survives a JSON round trip unchanged (it is what goes into the run log).
    assert BackendReply.model_validate_json(reply.model_dump_json()) == reply


async def test_reply_records_tool_calls(llama, sampling):
    call = ToolCall(name="read_file", arguments={"path": "a.txt"})
    backend = FakeBackend([FakeTurn(tool_calls=(call,))])
    reply = await backend.chat(llama, [ChatMessage(role="user", content="read a.txt")], sampling=sampling, seed=0)
    assert reply.message.tool_calls == (call,)


async def test_fake_records_requests_with_seed_and_sampling(llama, sampling):
    backend = FakeBackend(["a", "b"])
    await backend.chat(llama, [ChatMessage(role="user", content="x")], sampling=sampling, seed=7)
    await backend.chat(llama, [ChatMessage(role="user", content="y")], sampling=sampling, seed=8)
    assert [r.seed for r in backend.requests] == [7, 8]
    assert backend.requests[0].sampling == sampling


async def test_exhausted_script_raises(llama, sampling):
    backend = FakeBackend([])
    with pytest.raises(BackendError):
        await backend.chat(llama, [], sampling=sampling, seed=0)


async def test_warm_up_is_a_one_token_call(llama, sampling):
    backend = FakeBackend(["OK"])
    reply = await warm_up(backend, llama, sampling)
    assert backend.requests[0].sampling.max_tokens == 1
    assert is_warm(reply, max_load_s=0.5) is True
