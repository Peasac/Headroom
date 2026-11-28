"""AST checker for the BFCL single-call categories (simple, multiple, live_simple), following the
rules of BFCL's own ast_checker:

- exactly one function call, to the expected function (names compared with '.' -> '_', since the
  model is shown underscored names);
- every required parameter present; no parameter that the possible answer does not list;
- each value has the declared type (an integer is accepted for a float) and is one of the possible
  answers. Strings are compared after BFCL's standardisation (lowercase; spaces and , . / - _ * ^
  removed); lists element by element in order; dicts key by key;
- a possible-answer parameter the model left out must list "" (meaning optional).

It reads only the model's tool calls, never its text, so it checks what the model would have executed.
"""

import re
from dataclasses import dataclass
from typing import Any

from headroom.bfcl.convert import fc_name
from headroom.schemas.chat import ToolCall

_STRIP = re.compile(r"[ ,./\-_*^]")


@dataclass(frozen=True)
class Verdict:
    valid: bool
    error_type: str | None = None
    error: str | None = None


def standardize(s: str) -> str:
    return _STRIP.sub("", s).lower().replace("'", '"')


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return standardize(v)
    if isinstance(v, list | tuple):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _type_ok(value: Any, schema: dict[str, Any]) -> bool:
    t = schema.get("type", "any")
    if t in ("any", "Any"):
        return True
    if t in ("string", "char", "String"):
        return isinstance(value, str)
    if t in ("integer", "byte", "short", "long", "Bigint"):
        return isinstance(value, int) and not isinstance(value, bool)
    if t in ("float", "number", "double"):
        return isinstance(value, int | float) and not isinstance(value, bool)
    if t in ("boolean", "bool"):
        return isinstance(value, bool)
    if t in ("array", "list", "tuple", "ArrayList", "Array", "Queue", "Stack"):
        if not isinstance(value, list):
            return False
        items = schema.get("items")
        return items is None or all(_type_ok(x, items) for x in value)
    if t in ("dict", "object", "HashMap", "Hashtable"):
        return isinstance(value, dict)
    return True


def _value_ok(value: Any, possible: list[Any], schema: dict[str, Any]) -> bool:
    t = schema.get("type")
    if isinstance(value, dict) and t in ("dict", "object", "HashMap", "Hashtable"):
        return any(_dict_ok(value, p) for p in possible if isinstance(p, dict))
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        # BFCL's list_dict_checker: element by element, each a dict of possible values.
        return any(
            isinstance(p, list) and len(p) == len(value)
            and all(isinstance(pd, dict) and _dict_ok(v, pd) for v, pd in zip(value, p))
            for p in possible
        )
    if isinstance(value, float | int) and not isinstance(value, bool) and t in ("float", "number", "double"):
        return any(isinstance(p, int | float) and not isinstance(p, bool) and float(p) == float(value) for p in possible)
    return _norm(value) in [_norm(p) for p in possible if p != "" or value == ""]


def _dict_ok(value: dict[str, Any], possible: dict[str, list[Any]]) -> bool:
    for k, v in value.items():
        if k not in possible or _norm(v) not in [_norm(p) for p in possible[k]]:
            return False
    return all("" in opts for k, opts in possible.items() if k not in value)


def check_call(calls: tuple[ToolCall, ...], functions: list[dict[str, Any]], answer: list[dict[str, Any]]) -> Verdict:
    """calls: what the model returned. functions: the BFCL function docs shown. answer: BFCL ground_truth
    (one {function name: {param: [possible values]}})."""
    if len(answer) != 1:
        raise ValueError("single-call categories have exactly one ground-truth call")
    if not calls:
        return Verdict(False, "no_function_call", "the model made no function call")
    if len(calls) != 1:
        return Verdict(False, "wrong_count", f"{len(calls)} function calls, expected 1")
    call = calls[0]
    if call.parse_error:
        return Verdict(False, "unparseable_arguments", call.parse_error)
    (want_name, want_params), = answer[0].items()
    doc = next((f for f in functions if f["name"] == want_name), None)
    if doc is None:  # upstream data issue (e.g. simple_363): unpassable, as in BFCL's own checker
        return Verdict(False, "answer_not_offered", f"ground truth names {want_name}, which the item does not offer")
    if call.name != fc_name(want_name):
        return Verdict(False, "wrong_function", f"called {call.name}, expected {fc_name(want_name)}")
    props: dict[str, Any] = doc["parameters"].get("properties", {})
    for req in doc["parameters"].get("required", []):
        if req not in call.arguments:
            return Verdict(False, "missing_required", f"missing required parameter {req!r}")
    for name, value in call.arguments.items():
        if name not in want_params:
            return Verdict(False, "unexpected_param", f"unexpected parameter {name!r}")
        schema = props.get(name, {})
        if not _type_ok(value, schema):
            return Verdict(False, "type_error", f"{name}={value!r} is not of type {schema.get('type')!r}")
        if not _value_ok(value, want_params[name], schema):
            return Verdict(False, "value_error", f"{name}={value!r} not in {want_params[name]!r}")
    for name, opts in want_params.items():
        if name not in call.arguments and "" not in opts:
            return Verdict(False, "missing_optional", f"parameter {name!r} omitted but has no default")
    return Verdict(True)
