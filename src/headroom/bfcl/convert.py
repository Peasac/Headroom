"""BFCL function docs -> the tool schema a function-calling model is shown, as BFCL does for
function-calling (FC) models: types mapped to JSON Schema, '.' in names replaced by '_', and the
Python-syntax note appended to each description."""

import copy
from typing import Any

from headroom.schemas.chat import ToolSchema

# BFCL's GORILLA_TO_OPENAPI type map.
TYPE_MAP = {
    "integer": "integer", "number": "number", "float": "number", "string": "string", "boolean": "boolean",
    "bool": "boolean", "array": "array", "list": "array", "dict": "object", "object": "object", "tuple": "array",
    "any": "string", "byte": "integer", "short": "integer", "long": "integer", "double": "number",
    "char": "string", "ArrayList": "array", "Array": "array", "HashMap": "object", "Hashtable": "object",
    "Queue": "array", "Stack": "array", "Any": "string", "String": "string", "Bigint": "integer",
}
PYTHON_NOTE = " Note that the provided function is in Python 3 syntax."


def fc_name(name: str) -> str:
    return name.replace(".", "_")


def _map_types(schema: Any) -> Any:
    if isinstance(schema, dict):
        out = {k: _map_types(v) for k, v in schema.items()}
        t = schema.get("type")
        if isinstance(t, str):
            out["type"] = TYPE_MAP.get(t, t)
            if t == "float":
                out["description"] = (out.get("description") or "") + " This is a float type value."
        return out
    if isinstance(schema, list):
        return [_map_types(x) for x in schema]
    return schema


def to_tool(function: dict[str, Any]) -> ToolSchema:
    params = _map_types(copy.deepcopy(function["parameters"]))
    return ToolSchema(
        name=fc_name(function["name"]),
        description=(function.get("description") or "") + PYTHON_NOTE,
        parameters=params,
    )
