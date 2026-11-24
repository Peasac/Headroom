import tomllib
from pathlib import Path

from headroom.schemas.config import ExperimentConfig


def load_experiment(path: str | Path) -> ExperimentConfig:
    with open(path, "rb") as f:
        return ExperimentConfig.model_validate(tomllib.load(f))
