import pytest
from pydantic import TypeAdapter, ValidationError

from headroom.schemas.common import Measured, Reading, Unavailable, measured, unavailable
from headroom.schemas.config import ExperimentConfig, ModelSpec, OllamaSettings


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
        OllamaSettings(**{**settings.model_dump(), "keep_alive": keep_alive})


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


def test_config_space_allows_two_context_caps_and_two_retry_limits(experiment):
    def arm(arm_id, mode="validated", cap=8192, retries=2):
        return {"arm_id": arm_id, "controller": "fixed",
                "config": {"model_id": "llama3.2-3b", "mode": mode, "context_cap": cap, "retry_limit": retries}}

    ok = [arm("a", cap=2048), arm("b", cap=8192), arm("c", retries=1), arm("d", mode="direct", retries=7)]
    assert len(_experiment(experiment, arms=ok).arms) == 4  # direct's retry_limit is unused, so not counted
    with pytest.raises(ValidationError, match="context cap"):
        _experiment(experiment, arms=[arm("a", cap=2048), arm("b", cap=4096), arm("c", cap=8192)])
    with pytest.raises(ValidationError, match="retry limit"):
        _experiment(experiment, arms=[arm("a", retries=0), arm("b", retries=1), arm("c", retries=2)])


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
    assert [(a.arm_id, a.config.mode, a.config.context_cap) for a in week1.arms] == [
        ("qwen-validated", "validated", 8192), ("qwen-direct", "direct", 8192),
        ("qwen-plan-act", "plan_act", 8192), ("qwen-plan-act-verify", "plan_act_verify", 8192),
        ("qwen-validated-2k", "validated", 2048),
    ]
    assert {a.config.model_id for a in week1.arms} == {"qwen3.5-4b"}
    assert week1.sampling.temperature == 0.0
    assert week1.repeats == 5
    assert week1.quiet.max_gpu_util_pct == 10 and week1.quiet.max_gpu_temp_c == 70
    assert week1.telemetry.backend_process_prefix == "ollama"
