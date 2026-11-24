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


class GateARule(Frozen):
    rule_id: str
    registered_on: date
    question: str
    metric: GateMetric
    definitions: GateDefinitions
    pass_: GatePass = Field(alias="pass")
    success: GateSuccess
    noise: GateNoise


def load_gate_a(path: str | Path = "experiments/gate_a.toml") -> GateARule:
    with open(path, "rb") as f:
        return GateARule.model_validate(tomllib.load(f))
