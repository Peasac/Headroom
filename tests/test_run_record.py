import pytest
from helpers import make_run

from headroom.backends.fake import FakeTurn
from headroom.runner.single import _cold_start
from headroom.schemas.chat import ToolCall
from headroom.schemas.common import Measured
from headroom.schemas.run import Run
from headroom.tasks.checks import GRADER_VERSION

pytestmark = pytest.mark.anyio

GOOD = ToolCall(name="rename_file", arguments={"src": "drafts/plan.md", "dst": "archive/plan.md"})


async def test_run_record_carries_everything_the_log_needs(experiment, provenance):
    run = await make_run(experiment, provenance, [FakeTurn(tool_calls=(GOOD,)), "Moved."])
    assert run.status == "completed" and run.success
    # config, backend settings (keep_alive), machine details + git commit, quiet check, grader version
    assert run.config.mode == "validated" and run.arm_id == "validated" and run.controller == "fixed"
    assert run.backend.keep_alive == "30m" and run.backend.num_ctx == 8192
    assert run.sampling.seed == 0 and run.sampling.temperature == 0.7
    assert run.provenance.gpu == "Test GPU" and run.provenance.git_commit.status == "unavailable"
    assert run.quiet.passed and run.quiet.power_plan == "Balanced"
    assert run.grade.grader_version == GRADER_VERSION
    assert run.cold_start is False
    # Ollama process memory with every run
    assert run.telemetry.readings["peak_backend_private_mb"] == Measured(value=900.0, unit="MB")
    assert run.telemetry.readings["end_backend_rss_mb"].value == 300.0
    assert run.state_changes == 1 and not run.claimed_done_without_state_change
    assert Run.model_validate_json(run.model_dump_json()) == run


async def test_success_is_the_state_graders_verdict_and_status_is_separate(experiment, provenance):
    # The rename happens, then the next prompt is over the cap.
    run = await make_run(
        experiment, provenance, [FakeTurn(tool_calls=(GOOD,), prompt_tokens=100), FakeTurn(prompt_tokens=600)],
        context_cap=512,
    )
    assert run.status == "context_exceeded"
    assert run.success is run.grade.success is True
    assert run.schema_version == 2


@pytest.mark.parametrize("warm,expected", [
    ([True, True], False), ([True, False], True), ([True, None], None), ([None, False], True), ([], None),
])
def test_cold_start_flag(warm, expected):
    assert _cold_start(warm) is expected
