from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from headroom.schemas.chat import ToolSchema
from headroom.schemas.env import EnvState


class ToolError(Exception):
    """Valid arguments, but the tool refuses (missing file, duplicate title...). State is unchanged."""


class ToolArgs(BaseModel):
    """Base for tool argument models. Unknown arguments are a validation error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


ToolFn = Callable[[EnvState, Any], tuple[EnvState, str]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    mutates_state: bool
    fn: ToolFn

    def schema(self) -> ToolSchema:
        """What the model sees. Derived from args_model, so shown schema and validation cannot drift."""
        return ToolSchema(
            name=self.name, description=self.description, parameters=_strip_titles(self.args_model.model_json_schema())
        )


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto-generated 'title' annotations (token noise), keeping properties named 'title'."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "properties":
                out[k] = {name: _strip_titles(sub) for name, sub in v.items()}
            elif k == "title" and isinstance(v, str):
                continue
            else:
                out[k] = _strip_titles(v)
        return out
    if isinstance(node, list):
        return [_strip_titles(x) for x in node]
    return node
