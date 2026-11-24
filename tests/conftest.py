from pathlib import Path

import pytest

from headroom.experiments import load_experiment
from headroom.schemas.common import measured, unavailable
from headroom.schemas.config import (
    AgentConfig, Arm, BackendSettings, Condition, ExperimentConfig, ModelSpec, QuietSettings, Sampling,
    TelemetrySettings,
)
from headroom.schemas.run import Provenance, QuietCheck

ROOT = Path(__file__).resolve().parents[1]
WEEK1 = ROOT / "experiments" / "week1_baseline.toml"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def week1():
    return load_experiment(WEEK1)


@pytest.fixture
def sampling() -> Sampling:
    return Sampling(
        temperature=0.7, top_p=0.9, top_k=40, min_p=0.0,
        repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, max_tokens=256, seed=0,
    )


@pytest.fixture
def settings() -> BackendSettings:
    return BackendSettings(
        kind="ollama", base_url="http://ollama.test", keep_alive="30m",
        num_ctx=8192, request_timeout_s=30, max_warm_load_s=0.5,
    )


@pytest.fixture
def llama() -> ModelSpec:
    return ModelSpec(model_id="llama3.2-3b", backend_name="llama3.2:3b", gguf_sha256="a" * 64, quant="Q4_K_M", params_b=3.2)


@pytest.fixture
def qwen() -> ModelSpec:
    return ModelSpec(
        model_id="qwen3.5-4b", backend_name="qwen3.5:4b", gguf_sha256="b" * 64, quant="Q4_K_M", params_b=4.2, think=False
    )


@pytest.fixture
def telemetry_settings() -> TelemetrySettings:
    return TelemetrySettings(interval_s=0.05, backend_process_prefix="ollama", backend_process_arg="serve")


def make_config(mode: str = "validated", retry_limit: int = 2, context_cap: int = 8192) -> AgentConfig:
    return AgentConfig(model_id="llama3.2-3b", mode=mode, context_cap=context_cap, retry_limit=retry_limit)


@pytest.fixture
def quiet_settings() -> QuietSettings:
    return QuietSettings(
        max_gpu_util_pct=10, max_gpu_temp_c=70, require_ac=True, power_plan="Balanced",
        window_s=1.0, wait_s=10, max_attempts=3,
    )


@pytest.fixture
def experiment(settings, sampling, telemetry_settings, quiet_settings, llama) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="test", run_models=["llama3.2-3b"], repeats=2, backend=settings, sampling=sampling,
        telemetry=telemetry_settings, quiet=quiet_settings,
        arms=[
            Arm(arm_id="validated", controller="fixed", config=make_config("validated", 2)),
            Arm(arm_id="direct", controller="fixed", config=make_config("direct", 0)),
        ],
        conditions=[Condition(condition_id="baseline", kind="baseline", realism="real")],
        models=[llama],
    )


def quiet_ok(**update) -> QuietCheck:
    base = dict(
        passed=True, attempts=1, waited_s=0.0, gpu_util_pct=measured(3.0, "%"), gpu_temp_c=measured(55.0, "C"),
        on_ac=measured(1, "bool"), power_plan="Balanced",
    )
    return QuietCheck(**{**base, **update})


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(
        os="TestOS", cpu="Test CPU", cpu_count=8, ram_total_mb=16384.0, gpu="Test GPU", gpu_driver="1.0",
        vram_total_mb=measured(6144.0, "MB"), power_scheme="Balanced", python="3.12.2", headroom_version="0.1.0",
        git_commit=unavailable("no commits"), git_dirty=unavailable("no commits"),
    )


class FakeSource:
    """Scripted telemetry source: returns the given samples in turn, then repeats the last one."""

    def __init__(self, samples, unavailable=None):
        self.samples = list(samples)
        self.unavailable = dict(unavailable or {})
        self.started = self.closed = False
        self._i = 0

    def start(self):
        self.started = True

    def sample(self):
        s = self.samples[min(self._i, len(self.samples) - 1)]
        self._i += 1
        if isinstance(s, Exception):
            raise s
        return dict(s)

    def close(self):
        self.closed = True
