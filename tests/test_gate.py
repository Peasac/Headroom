from datetime import date

from conftest import ROOT

from headroom.experiments import load_experiment
from headroom.schemas.gate import load_gate_a


def test_gate_a_rule_is_the_registered_one():
    rule = load_gate_a(ROOT / "experiments" / "gate_a.toml")
    assert rule.rule_id == "gate_a_v2" and rule.supersedes == "gate_a_v1"
    assert rule.registered_on == date(2026, 11, 25)
    assert rule.scope.arms == ["qwen-direct", "qwen-validated", "qwen-validated-2k", "qwen-plan-act"]
    assert rule.scope.conditions == ["baseline", "memory", "cpu", "reduced", "vram"]
    # Unchanged from v1:
    assert rule.metric.timing == "per_task_median_wall_s"
    assert rule.pass_.min_gap == 0.10
    assert rule.pass_.min_conditions == 2
    assert rule.pass_.ci_level == 0.95 and rule.pass_.ci_must_exclude == 0.0
    assert rule.success.checked == "separately"
    assert rule.noise.min_single_task_difference == 0.16


def test_gate_scope_names_real_arms_and_conditions():
    rule = load_gate_a(ROOT / "experiments" / "gate_a.toml")
    cfg = load_experiment(ROOT / "experiments" / "week2_conditions.toml")
    scoped = {a: cfg.arm(a).config for a in rule.scope.arms}
    assert {(c.mode, c.context_cap) for c in scoped.values()} == {
        ("direct", 8192), ("validated", 8192), ("validated", 2048), ("plan_act", 8192),
    }
    assert {cfg.condition(c).kind for c in rule.scope.conditions} == {
        "baseline", "memory_pressure", "cpu_contention", "reduced_compute", "vram_pressure",
    }
    assert cfg.condition("vram").ballast_mb == 2048
