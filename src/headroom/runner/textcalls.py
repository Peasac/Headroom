"""Detects tool calls the model wrote as plain text, which the backend did not parse into tool_calls.

Seen with llama3.2: content = '{"name": "cat", "parameters": {"file": "shopping.txt"}}'.
"""

import json
import re
from typing import Any

_MARKERS = re.compile(r"<tool_call>|<\|python_tag\|>|<function[=>]", re.IGNORECASE)
SNIPPET_CHARS = 200


def detect_text_tool_call(content: str) -> str | None:
    """Return a snippet of the first tool-call-like text, or None."""
    if not content:
        return None
    marker = _MARKERS.search(content)
    if marker:
        return content[marker.start() : marker.start() + SNIPPET_CHARS]
    decoder = json.JSONDecoder()
    for i, ch in enumerate(content):
        if ch not in "{[":
            continue
        try:
            obj, end = decoder.raw_decode(content, i)
        except ValueError:
            continue
        if _looks_like_call(obj):
            return content[i:end][:SNIPPET_CHARS]
    return None


def _looks_like_call(obj: Any) -> bool:
    if isinstance(obj, list):
        return any(_looks_like_call(x) for x in obj)
    if not isinstance(obj, dict):
        return False
    if isinstance(obj.get("function"), dict):
        return _looks_like_call(obj["function"])
    return isinstance(obj.get("name"), str) and (
        isinstance(obj.get("parameters"), dict) or isinstance(obj.get("arguments"), (dict, str))
    )
