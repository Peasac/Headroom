"""Three mock tool families over EnvState: files, notes, calendar."""

from collections.abc import Iterable

from headroom.schemas.chat import ToolSchema
from headroom.tasks.tools import calendar, files, notes
from headroom.tasks.tools.base import ToolError, ToolSpec

REGISTRY: dict[str, ToolSpec] = {s.name: s for s in (*files.SPECS, *notes.SPECS, *calendar.SPECS)}


def tool_schemas(names: Iterable[str]) -> list[ToolSchema]:
    return [REGISTRY[n].schema() for n in names]


__all__ = ["REGISTRY", "ToolError", "ToolSpec", "tool_schemas"]
