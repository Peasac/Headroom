from datetime import datetime
from typing import Literal

from pydantic import Field

from headroom.schemas.chat import BackendReply, ToolCall
from headroom.schemas.common import SCHEMA_VERSION, Frozen, Reading, Unavailable
from headroom.schemas.config import AgentConfig, BackendSettings, ExperimentConfig, ModelSpec, Sampling
from headroom.schemas.env import EnvState
from headroom.schemas.result import GradeResult
from headroom.schemas.telemetry import TelemetrySummary


class ToolOutcome(Frozen):
    """What happened to one tool call.

    invalid:    arguments failed validation (or unknown tool); nothing executed, state unchanged.
    tool_error: arguments valid, but the tool refused (e.g. file missing); state unchanged.
    ok:         executed.
    """

    call: ToolCall
    mutates_state: bool | None  # None when the tool is unknown
    status: Literal["ok", "invalid", "tool_error"]
    result: str | None = None
    error: str | None = None


class Step(Frozen):
    index: int = Field(ge=0)
    phase: Literal["plan", "act", "verify"] = Field(
        default="act", description="plan: no tools offered, the model writes a plan. verify: after the model "
        "says it is done (plan_act_verify), it checks and fixes the result with tools."
    )
    reply: BackendReply
    outcomes: tuple[ToolOutcome, ...] = ()
    malformed_text_call: str | None = Field(
        default=None, description="Snippet of a tool call the model wrote as text and the backend did not parse."
    )


class Provenance(Frozen):
    os: str
    cpu: str
    cpu_count: int
    ram_total_mb: float
    gpu: str | Unavailable
    gpu_driver: str | Unavailable
    vram_total_mb: Reading
    power_scheme: str | Unavailable
    python: str
    headroom_version: str
    git_commit: str | Unavailable
    git_dirty: bool | Unavailable


class QuietCheck(Frozen):
    """Quiet-machine check made just before the run. Values are from the last attempt."""

    passed: bool
    attempts: int = Field(ge=1)
    waited_s: float = Field(ge=0)
    gpu_util_pct: Reading = Field(description="Mean over the check window.")
    gpu_temp_c: Reading
    on_ac: Reading
    power_plan: str | Unavailable
    failures: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = Field(default=(), description="Checks skipped because the sensor is unavailable.")


class Session(Frozen):
    """One invocation of the paired runner. One line in results/<experiment>/sessions.jsonl."""

    schema_version: int = SCHEMA_VERSION
    session_id: str
    experiment_id: str
    started_at: datetime
    order_seed: int
    plan: tuple["PlannedRun", ...]
    experiment: ExperimentConfig


class PlannedRun(Frozen):
    order_index: int = Field(ge=0, description="Position in the randomized run order.")
    task_id: str
    repeat: int = Field(ge=0)
    condition_id: str
    arm_id: str

    @property
    def pair_key(self) -> tuple[str, int, str]:
        """Runs with the same pair key differ only in arm."""
        return (self.task_id, self.repeat, self.condition_id)


RunStatus = Literal[
    "completed",            # model stopped calling tools (graded separately; not the same as success)
    "max_steps",            # task's step budget used up
    "invalid_state_call",   # invalid state-changing call: direct mode, or retries exhausted
    "malformed_text_call",  # tool call written as text: direct mode, or retries exhausted
    "context_exceeded",     # a prompt went over config.context_cap
    "backend_error",
]


class Run(Frozen):
    """One task, under one arm, condition and repeat. One line in results/<experiment>/runs.jsonl."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    session_id: str
    order_seed: int
    order_index: int
    task_id: str
    repeat: int
    condition_id: str
    arm_id: str
    controller: str
    config: AgentConfig
    model: ModelSpec
    sampling: Sampling = Field(description="Includes the fixed backend seed.")
    backend: BackendSettings = Field(description="Includes keep_alive and num_ctx, so cold starts can be filtered.")
    backend_version: str
    provenance: Provenance = Field(description="Machine details and git commit.")
    quiet: QuietCheck
    started_at: datetime
    wall_s: float = Field(ge=0)
    status: RunStatus
    error: str | None = None
    steps: tuple[Step, ...]
    retries_used: int = Field(ge=0)
    invalid_state_changing_calls: int = Field(ge=0)
    invalid_other_calls: int = Field(ge=0, description="Invalid read-only or unknown-tool calls (fed back, never fatal).")
    malformed_text_calls: int = Field(ge=0, description="Tool calls written as text that the backend did not parse.")
    state_changes: int = Field(ge=0, description="State-changing tool calls that executed.")
    claimed_done_without_state_change: bool = Field(
        description="The model ended the run (final reply, no tool call) without any state-changing call executing."
    )
    cold_start: bool | None = Field(
        description="True if any step's load time exceeded backend.max_warm_load_s; None if not reported."
    )
    prompt_tokens_total: int | None = Field(default=None, description="Sum of prompt_eval_count over all calls.")
    cached_prompt_tokens_total: int | None = Field(
        default=None, description="Sum of cached prompt tokens over all calls (later calls reuse earlier ones: normal)."
    )
    first_call_cached_tokens: int | None = Field(
        default=None, description="Cached prompt tokens on the run's first call: anything > 0 came from an earlier run."
    )
    cache_reused: bool | None = Field(
        default=None, description="First call reused prompt cache left by an earlier run. None if not reported."
    )
    final_state: EnvState
    telemetry: TelemetrySummary = Field(description="Includes Ollama process memory (working set and private).")
    grade: GradeResult = Field(description="Carries grader_version.")
    success: bool = Field(
        description="The state grader's verdict alone (schema v2). Status records how the run ended; it does not "
        "decide success. clean success = success and malformed_text_calls == 0 (computed in reports)."
    )


Session.model_rebuild()
