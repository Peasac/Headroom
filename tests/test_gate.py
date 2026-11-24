from datetime import date

from conftest import ROOT

from headroom.schemas.gate import load_gate_a


def test_gate_a_rule_is_the_registered_one():
    rule = load_gate_a(ROOT / "experiments" / "gate_a.toml")
    assert rule.rule_id == "gate_a_v1"
    assert rule.registered_on == date(2026, 11, 24)
    assert rule.metric.timing == "per_task_median_wall_s"
    assert rule.pass_.min_gap == 0.10
    assert rule.pass_.min_conditions == 2
    assert rule.pass_.ci_level == 0.95 and rule.pass_.ci_must_exclude == 0.0
    assert rule.success.checked == "separately"
    assert rule.noise.min_single_task_difference == 0.16
