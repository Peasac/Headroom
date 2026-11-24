"""Against the real local Ollama server. Run with: uv run pytest -m live"""

import pytest

from headroom.backends import is_warm, warm_up
from headroom.backends.ollama import OllamaBackend
from headroom.schemas.chat import ChatMessage, ToolSchema
from headroom.schemas.common import Measured

pytestmark = [pytest.mark.live, pytest.mark.anyio]

READ_FILE = ToolSchema(
    name="read_file",
    description="Read a text file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


async def test_real_run_has_real_vram_and_backend_memory(week1):
    from pathlib import Path

    from headroom.runner.quiet import machine_reader, quiet_check
    from headroom.runner.single import execute_run
    from headroom.schemas.run import PlannedRun
    from headroom.tasks.suite import get_task
    from headroom.telemetry.provenance import collect_provenance
    from headroom.telemetry.sources import NvmlSource, PsutilSource

    arm = week1.arms[0]
    spec = week1.model(arm.config.model_id)
    nvml = NvmlSource()
    sources = [PsutilSource(week1.telemetry.backend_process_prefix, week1.telemetry.backend_process_arg), nvml]
    planned = PlannedRun(order_index=0, task_id="files.rename_01", repeat=0, condition_id="baseline", arm_id=arm.arm_id)
    try:
        async with OllamaBackend(week1.backend) as b:
            check = quiet_check(week1.quiet.model_copy(update={"max_attempts": 1}), machine_reader(nvml, 0.5))
            await warm_up(b, spec, week1.sampling)
            run = await execute_run(
                b, get_task("files.rename_01"), experiment=week1, planned=planned, session_id="live-test",
                order_seed=0, quiet=check, backend_version=await b.version(),
                provenance=collect_provenance(Path.cwd(), nvml), sources=sources,
            )
    finally:
        for s in sources:
            s.close()
    r = run.telemetry.readings
    assert isinstance(r["peak_vram_used_mb"], Measured), r["peak_vram_used_mb"]
    assert r["peak_vram_used_mb"].value > 1000  # model weights are on the GPU
    assert isinstance(r["peak_backend_rss_mb"], Measured), r["peak_backend_rss_mb"]
    assert isinstance(r["peak_backend_private_mb"], Measured), r["peak_backend_private_mb"]
    assert isinstance(run.quiet.gpu_temp_c, Measured) and isinstance(run.quiet.power_plan, str)
    assert isinstance(r["peak_sys_mem_used_mb"], Measured)
    assert run.backend.keep_alive == week1.backend.keep_alive
    assert run.cold_start is False
    assert "RTX 3050" in run.provenance.gpu


async def test_pinned_weights_are_served(week1):
    async with OllamaBackend(week1.backend) as b:
        for model in week1.models:
            await b.verify_model(model)


async def test_warm_call_records_reply_latency_tokens_and_no_cold_start(week1):
    spec = week1.model(week1.run_models[0])
    async with OllamaBackend(week1.backend) as b:
        await warm_up(b, spec, week1.sampling)
        reply = await b.chat(
            spec, [ChatMessage(role="user", content="Read the file a.txt.")],
            sampling=week1.sampling, seed=week1.sampling.seed, tools=[READ_FILE],
        )
    assert reply.latency_s > 0
    assert isinstance(reply.prompt_tokens, Measured) and reply.prompt_tokens.value > 0
    assert isinstance(reply.completion_tokens, Measured) and reply.completion_tokens.value > 0
    assert is_warm(reply, week1.backend.max_warm_load_s) is True
    assert [c.name for c in reply.message.tool_calls] == ["read_file"]
