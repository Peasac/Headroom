"""The pre-registered Gate A rule (experiments/gate_a.toml)."""

import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field

from headroom.schemas.common import Frozen


class GateMetric(Frozen):
    timing: Literal["per_task_median_wall_s"]
    condition_figure: Literal["median_over_tasks"]


class GateDefinitions(Frozen):
    best_fixed: str
    oracle: str
    gap: str


class GatePass(Frozen):
    min_gap: float = Field(gt=0)
    min_conditions: int = Field(ge=1)
    ci_level: float = Field(gt=0, lt=1)
    ci_must_exclude: float
    bootstrap: str
    bootstrap_resamples: int = Field(ge=1000)
    bootstrap_seed: int


class GateSuccess(Frozen):
    checked: Literal["separately"]


class GateNoise(Frozen):
    baseline_arm_noise: float = Field(ge=0)
    min_single_task_difference: float = Field(ge=0)
    remeasure_noise_per_condition: bool


class GateScope(Frozen):
    arms: list[str] = Field(min_length=2)
    conditions: list[str] = Field(min_length=2)


class GateARule(Frozen):
    rule_id: str
    supersedes: str | None = None
    registered_on: date
    question: str
    scope: GateScope
    metric: GateMetric
    definitions: GateDefinitions
    pass_: GatePass = Field(alias="pass")
    success: GateSuccess
    noise: GateNoise


def load_gate_a(path: str | Path = "experiments/gate_a.toml") -> GateARule:
    with open(path, "rb") as f:
        return GateARule.model_validate(tomllib.load(f))


class NoAdaptationGain(Frozen):
    statement: str
    gate_a_rule: str
    replicates_when: str


class SuccessUnchanged(Frozen):
    statement: str
    max_task_difference: int = Field(ge=0)


class Slower(Frozen):
    statement: str
    slower: str
    faster: str
    min_ci_lower: float


class Equivalent(Frozen):
    statement: str
    a: str
    b: str
    margin: float = Field(gt=0)


class WorkloadsSlow(Frozen):
    statement: str
    min_ci_lower: float


class GateCClaims(Frozen):
    no_adaptation_gain: NoAdaptationGain
    success_unchanged_under_load: SuccessUnchanged
    plan_act_slower: Slower
    direct_validated_equivalent: Equivalent
    workloads_slow_every_arm: WorkloadsSlow


class GateCRule(Frozen):
    rule_id: str
    registered_on: date
    question: str
    experiment_id: str
    arms: list[str] = Field(min_length=2)
    conditions: list[str] = Field(min_length=2)
    baseline: str
    ci_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=1000)
    bootstrap_seed: int
    claims: GateCClaims


def load_gate_c(path: str | Path = "experiments/gate_c.toml") -> GateCRule:
    with open(path, "rb") as f:
        rule = GateCRule.model_validate(tomllib.load(f))
    c = rule.claims
    for arm in (c.plan_act_slower.slower, c.plan_act_slower.faster,
                c.direct_validated_equivalent.a, c.direct_validated_equivalent.b):
        if arm not in rule.arms:
            raise ValueError(f"claim arm {arm!r} not in arms {rule.arms}")
    if rule.baseline not in rule.conditions:
        raise ValueError(f"baseline {rule.baseline!r} not in conditions")
    return rule
