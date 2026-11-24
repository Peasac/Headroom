import pytest
from pydantic import TypeAdapter, ValidationError

from headroom.schemas.common import Measured, Reading, Unavailable, measured, unavailable
from headroom.schemas.config import BackendSettings, ExperimentConfig, ModelSpec


def test_reading_round_trips_both_variants():
    ta = TypeAdapter(Reading)
    for r in (measured(1.5, "s"), unavailable("no sensor")):
        assert ta.validate_json(ta.dump_json(r)) == r
    assert isinstance(ta.validate_python({"status": "ok", "value": 3, "unit": "tokens"}), Measured)


def test_unavailable_cannot_carry_a_value():
    with pytest.raises(ValidationError):
        Unavailable(reason="x", value=0)  # type: ignore[call-arg]


def test_model_spec_rejects_bad_hash(llama):
    with pytest.raises(ValidationError):
        ModelSpec(**{**llama.model_dump(), "gguf_sha256": "not-a-hash"})


@pytest.mark.parametrize("keep_alive", ["0", "0m", "-1", "forever", ""])
def test_keep_alive_must_be_positive_duration(settings, keep_alive):
    with pytest.raises(ValidationError):
        BackendSettings(**{**settings.model_dump(), "keep_alive": keep_alive})


def _experiment(experiment, **update):
    return ExperimentConfig.model_validate({**experiment.model_dump(), **update})


def test_experiment_fixture_is_valid(experiment):
    assert _experiment(experiment) == experiment


def test_experiment_caps_models_at_two(experiment, llama, qwen):
    third = ModelSpec(**{**llama.model_dump(), "model_id": "third"})
    with pytest.raises(ValidationError):
        _experiment(experiment, models=[m.model_dump() for m in (llama, qwen, third)])


def test_experiment_rejects_duplicate_and_unknown_models(experiment, llama):
    with pytest.raises(ValidationError):
        _experiment(experiment, models=[llama.model_dump(), llama.model_dump()])
    with pytest.raises(ValidationError):
        _experiment(experiment, run_models=["nope"])


def _arm(experiment, **config_update):
    arm = experiment.arms[0].model_dump()
    return {**arm, "config": {**arm["config"], **config_update}}


def test_arms_must_use_a_run_model_and_fit_num_ctx(experiment):
    with pytest.raises(ValidationError):
        _experiment(experiment, arms=[_arm(experiment, model_id="qwen3.5-4b")])
    with pytest.raises(ValidationError):
        _experiment(experiment, arms=[_arm(experiment, context_cap=16384)])


def test_arm_and_condition_ids_are_unique(experiment):
    with pytest.raises(ValidationError):
        _experiment(experiment, arms=[_arm(experiment), _arm(experiment)])
    cond = experiment.conditions[0].model_dump()
    with pytest.raises(ValidationError):
        _experiment(experiment, conditions=[cond, cond])


def test_week1_config_loads_and_pins_what_we_decided(week1):
    # llama3.2:3b is out of the config space (docs/PROJECT.md, decision log); qwen3.5:4b, thinking off.
    assert week1.run_models == ["qwen3.5-4b"]
    assert [m.model_id for m in week1.models] == ["qwen3.5-4b"]
    assert week1.backend.kind == "ollama"
    assert week1.backend.keep_alive == "30m"
    assert week1.backend.num_ctx == 8192
    qwen = week1.model("qwen3.5-4b")
    assert qwen.think is False and qwen.quant == "Q4_K_M"
    assert [(a.arm_id, a.config.model_id, a.config.mode) for a in week1.arms] == [
        ("qwen-validated", "qwen3.5-4b", "validated"), ("qwen-direct", "qwen3.5-4b", "direct"),
    ]
    assert week1.sampling.temperature == 0.0
    assert week1.repeats == 5
    assert week1.quiet.max_gpu_util_pct == 10 and week1.quiet.max_gpu_temp_c == 70
    assert week1.telemetry.backend_process_prefix == "ollama"
