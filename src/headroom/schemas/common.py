from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# v2: Run.success is the state grader's verdict alone (v1 also required status == 'completed').
# Reports derive success from Run.grade, so v1 and v2 records are counted the same way.
SCHEMA_VERSION = 2


class Frozen(BaseModel):
    """Base for every schema: immutable, and unknown fields are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Measured(Frozen):
    status: Literal["ok"] = "ok"
    value: int | float
    unit: str


class Unavailable(Frozen):
    """A sensor or counter that could not be read. Never stands in for zero."""

    status: Literal["unavailable"] = "unavailable"
    reason: str


Reading = Annotated[Measured | Unavailable, Field(discriminator="status")]


def measured(value: int | float, unit: str) -> Measured:
    return Measured(value=value, unit=unit)


def unavailable(reason: str) -> Unavailable:
    return Unavailable(reason=reason)
