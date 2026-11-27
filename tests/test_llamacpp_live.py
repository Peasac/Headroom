"""Against the pinned llama.cpp server (experiments/week3_llamacpp.toml). Run with: uv run pytest -m live"""

from pathlib import Path

import pytest

from headroom.backends import is_warm, warm_up
from headroom.backends.llamacpp import LlamaCppBackend
from headroom.experiments import load_experiment
from headroom.schemas.chat import ChatMessage, ToolSchema

pytestmark = [pytest.mark.live, pytest.mark.anyio]

CFG = load_experiment(Path(__file__).parent.parent / "experiments" / "week3_llamacpp.toml")
READ_FILE = ToolSchema(
    name="read_file", description="Read a text file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


@pytest.mark.skipif(not Path(CFG.backend.server_exe).is_file(), reason="pinned llama-server not installed")
async def test_pinned_server_verifies_warms_and_calls_a_tool_after_a_tool_result():
    spec = CFG.model(CFG.run_models[0])
    async with LlamaCppBackend(CFG.backend) as b:
        assert (await b.version()).startswith(CFG.backend.build)
        await b.verify_model(spec)
        await warm_up(b, spec, CFG.sampling, num_ctx=2048)
        msgs = [ChatMessage(role="user", content="Read notes/a.txt, then tell me what it says.")]
        first = await b.chat(spec, msgs, sampling=CFG.sampling, seed=0, tools=[READ_FILE], num_ctx=2048)
        assert is_warm(first, CFG.backend.max_warm_load_s)
        (call,) = first.message.tool_calls
        assert call.name == "read_file" and call.arguments == {"path": "notes/a.txt"}
        msgs += [first.message, ChatMessage(role="tool", content="Meeting moved to Friday.", tool_name="read_file")]
        second = await b.chat(spec, msgs, sampling=CFG.sampling, seed=0, tools=[READ_FILE], num_ctx=2048)
        assert "friday" in second.message.content.lower() and second.cached_prompt_tokens.value > 0
