from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from headroom.schemas.common import Frozen

MAX_MODELS = 2  # CLAUDE.md: keep the config space small.

Mode = Literal["direct", "validated", "plan_act", "plan_act_verify"]
# Modes are cumulative: each adds to the one before, so every mode after 'direct' feeds
# validation errors back to the model.
FEEDBACK_MODES: frozenset[str] = frozenset({"validated", "plan_act", "plan_act_verify"})


class ModelSpec(Frozen):
    model_id: str = Field(description="Stable id used in configs and logs, e.g. 'llama3.2-3b'.")
    backend_name: str = Field(description="Name the backend serves it under, e.g. Ollama tag 'llama3.2:3b'.")
    gguf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="Pinned hash of the weights file.")
    quant: str
    params_b: float = Field(gt=0)
    think: bool | None = Field(
        default=None,
        description="Thinking switch, sent on every call when set. None: model has no switch, never sent.",
    )


class Sampling(Frozen):
    """Every value is sent explicitly so backend and modelfile defaults never apply."""

    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0, le=1)
    repeat_penalty: float = Field(ge=0)
    presence_penalty: float
    frequency_penalty: float
    max_tokens: int = Field(gt=0)
    seed: int = Field(description="Fixed backend seed, sent on every measured call.")


class BackendSettings(Frozen):
    kind: Literal["ollama"]
    base_url: str
    keep_alive: str = Field(
        pattern=r"^[1-9][0-9]*(s|m|h)$",
        description="How long the model stays loaded after a call. Must be positive.",
    )
    num_ctx: int = Field(
        gt=0,
        description="Largest context any arm may use. Each call sends its arm's context_cap as num_ctx; "
        "a change reloads the model, which the unscored warm-up before every run absorbs.",
    )
    request_timeout_s: float = Field(gt=0)
    max_warm_load_s: float = Field(
        ge=0, description="A measured call reporting a longer load time counts as a cold start."
    )


class AgentConfig(Frozen):
    """One point in the config space. What a controller chooses."""

    model_id: str
    mode: Mode
    context_cap: int = Field(
        gt=0,
        description="Context window (sent as num_ctx). A call whose prompt plus reply fills it ends the run "
        "as context_exceeded.",
    )
    retry_limit: int = Field(
        ge=0, description="Per run, in feedback modes: invalid state-changing or text-written tool calls "
        "fed back before the run fails."
    )


class Arm(Frozen):
    """A controller plus the configuration it uses. Every arm runs on identical inputs."""

    arm_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    controller: Literal["fixed"]
    config: AgentConfig


class _ConditionBase(Frozen):
    condition_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    realism: Literal["real", "emulated"]


class BaselineCondition(_ConditionBase):
    kind: Literal["baseline"]


class MemoryPressure(_ConditionBase):
    """A ballast process takes RAM until available memory is down to target_available_mb, and
    keeps it resident. Safety valve: it gives memory back whenever available drops below
    min_available_mb, so other programs can always allocate."""

    kind: Literal["memory_pressure"]
    target_available_mb: int = Field(gt=0)
    min_available_mb: int = Field(gt=0)
    max_ballast_mb: int = Field(gt=0)

    @model_validator(mode="after")
    def _valve_below_target(self) -> "MemoryPressure":
        if self.min_available_mb >= self.target_available_mb:
            raise ValueError("min_available_mb must be below target_available_mb")
        return self


class CpuContention(_ConditionBase):
    """Busy-loop worker processes at normal priority. At least min_free_logical_cpus stay free."""

    kind: Literal["cpu_contention"]
    workers: int = Field(ge=1)
    min_free_logical_cpus: int = Field(ge=1)


class ReducedCompute(_ConditionBase):
    """Fewer CPU threads for the model runner (Ollama option num_thread; a change reloads the model)."""

    kind: Literal["reduced_compute"]
    num_thread: int = Field(ge=1)


class VramPressure(_ConditionBase):
    """A process allocates ballast_mb on the GPU through the CUDA driver API and rewrites it every
    touch_interval_s so it stays resident. The model is unloaded at the start of the block so it
    reloads under pressure. Refuses to start below min_available_ram_mb of free RAM, because the
    driver spills over-committed VRAM into system RAM."""

    kind: Literal["vram_pressure"]
    ballast_mb: int = Field(gt=0)
    touch_interval_s: float = Field(gt=0)
    min_available_ram_mb: int = Field(gt=0)


Condition = Annotated[
    BaselineCondition | MemoryPressure | CpuContention | ReducedCompute | VramPressure,
    Field(discriminator="kind"),
]


class TelemetrySettings(Frozen):
    interval_s: float = Field(ge=0.05)
    backend_process_prefix: str = Field(description="Name prefix of the backend server process, e.g. 'ollama'.")
    backend_process_arg: str = Field(description="Argument that marks the server process, e.g. 'serve'.")


class QuietSettings(Frozen):
    """Checked before every run. A run that never passes still runs, flagged."""

    max_gpu_util_pct: float = Field(gt=0, description="Mean GPU utilization over window_s must be below this.")
    max_gpu_temp_c: float = Field(gt=0, description="GPU temperature must be below this.")
    require_ac: bool
    power_plan: str | None = Field(description="Required OS power plan name; None to only record it.")
    window_s: float = Field(gt=0, description="GPU utilization is averaged over this window.")
    wait_s: float = Field(ge=0, description="Wait between attempts.")
    max_attempts: int = Field(ge=1)


class ExperimentConfig(Frozen):
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    run_models: list[str] = Field(min_length=1, max_length=MAX_MODELS)
    repeats: int = Field(ge=1)
    min_same_task_gap: int = Field(
        default=1, ge=1,
        description="Runs of the same task are at least this many positions apart in the run order. "
        "1 = unconstrained (sessions before 2026-11-24 evening).",
    )
    backend: BackendSettings
    sampling: Sampling
    telemetry: TelemetrySettings
    quiet: QuietSettings
    arms: list[Arm] = Field(min_length=1)
    conditions: list[Condition] = Field(min_length=1)
    models: list[ModelSpec] = Field(min_length=1, max_length=MAX_MODELS)

    @field_validator("models")
    @classmethod
    def _unique_model_ids(cls, models: list[ModelSpec]) -> list[ModelSpec]:
        ids = [m.model_id for m in models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate model_id in {ids}")
        return models

    @model_validator(mode="after")
    def _consistent(self) -> "ExperimentConfig":
        known = {m.model_id for m in self.models}
        unknown = [m for m in self.run_models if m not in known]
        if unknown:
            raise ValueError(f"run_models not in models: {unknown}")
        for ids, what in ((([a.arm_id for a in self.arms]), "arm_id"), ([c.condition_id for c in self.conditions], "condition_id")):
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {what} in {ids}")
        # CLAUDE.md: keep the config space small.
        for name, values in (
            ("model", {a.config.model_id for a in self.arms}),
            ("context cap", {a.config.context_cap for a in self.arms}),
            ("retry limit", {a.config.retry_limit for a in self.arms if a.config.mode in FEEDBACK_MODES}),
        ):
            if len(values) > 2:
                raise ValueError(f"config space allows at most 2 values per {name}; arms use {sorted(values)}")
        for arm in self.arms:
            if arm.config.model_id not in self.run_models:
                raise ValueError(f"arm {arm.arm_id}: model_id {arm.config.model_id!r} not in run_models")
            if arm.config.context_cap > self.backend.num_ctx:
                raise ValueError(
                    f"arm {arm.arm_id}: context_cap {arm.config.context_cap} exceeds num_ctx {self.backend.num_ctx}"
                )
        return self

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.model_id == model_id:
                return m
        raise KeyError(f"unknown model_id {model_id!r}; known: {[m.model_id for m in self.models]}")

    def condition(self, condition_id: str) -> Condition:
        for c in self.conditions:
            if c.condition_id == condition_id:
                return c
        raise KeyError(f"unknown condition_id {condition_id!r}; known: {[c.condition_id for c in self.conditions]}")

    def arm(self, arm_id: str) -> Arm:
        for a in self.arms:
            if a.arm_id == arm_id:
                return a
        raise KeyError(f"unknown arm_id {arm_id!r}; known: {[a.arm_id for a in self.arms]}")
