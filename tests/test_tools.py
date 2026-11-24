import inspect
import json

import pytest

from headroom.schemas.chat import ToolCall
from headroom.schemas.env import EnvState, Note
from headroom.tasks.executor import execute
from headroom.tasks.tools import REGISTRY, tool_schemas

ALL = tuple(REGISTRY)
STATE = EnvState(files={"a.txt": "A", "b.txt": "B"}, notes=(Note(title="Ideas", body="x"),))


def c(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def test_state_changing_tools_are_flagged():
    mutating = {n for n, s in REGISTRY.items() if s.mutates_state}
    assert mutating == {"write_file", "rename_file", "add_note", "add_event"}


def test_executor_has_no_way_to_skip_validation():
    # Safety rule: no mode/flag can turn validation off. Guard against one being added.
    assert list(inspect.signature(execute).parameters) == ["state", "call", "allowed"]


def test_schemas_drop_pydantic_titles_but_keep_title_properties():
    schemas = {s.name: s.parameters for s in tool_schemas(ALL)}
    json.dumps(schemas)  # serializable for the backend
    assert "title" not in schemas["add_note"]
    assert set(schemas["add_note"]["properties"]) == {"title", "body"}
    assert "title" not in schemas["add_note"]["properties"]["title"]
    assert schemas["add_event"]["required"] == ["title", "start", "end"]


@pytest.mark.parametrize("call", [
    c("write_file", path="x.txt"),                                       # missing argument
    c("write_file", path="x.txt", content="y", mode="append"),           # unknown argument
    c("write_file", path=5, content="y"),                                # wrong type
    c("write_file", path="../etc/passwd", content="y"),                  # escapes the root
    c("rename_file", src="a.txt", dst=""),                               # empty path
    c("add_note", title="", body="y"),                                   # empty title
    c("add_event", title="t", start="2026-12-03T15:00", end="2026-12-03T14:00"),        # end before start
    c("add_event", title="t", start="2026-12-03T14:00+02:00", end="2026-12-03T15:00+02:00"),  # timezone
    c("add_event", title="t", start="tomorrow 2pm", end="2026-12-03T15:00"),             # unparseable
    c("delete_file", path="a.txt"),                                      # unknown tool
    ToolCall(name="write_file", arguments={}, parse_error="arguments are not valid JSON"),
])
def test_invalid_calls_are_rejected_and_change_nothing(call):
    state, outcome = execute(STATE, call, ALL)
    assert outcome.status == "invalid"
    assert outcome.error
    assert state == STATE


def test_tool_not_offered_by_the_task_is_invalid():
    _, outcome = execute(STATE, c("read_file", path="a.txt"), allowed=("add_note",))
    assert outcome.status == "invalid"


@pytest.mark.parametrize("call", [
    c("read_file", path="missing.txt"),
    c("rename_file", src="missing.txt", dst="c.txt"),
    c("rename_file", src="a.txt", dst="b.txt"),  # destination exists
    c("add_note", title="ideas", body="dup"),    # duplicate title, case-insensitive
])
def test_tool_errors_change_nothing(call):
    state, outcome = execute(STATE, call, ALL)
    assert outcome.status == "tool_error"
    assert state == STATE


def test_paths_are_normalized():
    state, outcome = execute(STATE, c("write_file", path="./dir\\new.txt", content="n"), ALL)
    assert outcome.status == "ok"
    assert state.files["dir/new.txt"] == "n"


def test_valid_mutation_does_not_touch_the_input_state():
    before = STATE.model_copy(deep=True)
    state, _ = execute(STATE, c("rename_file", src="a.txt", dst="z.txt"), ALL)
    assert STATE == before
    assert state.files == {"b.txt": "B", "z.txt": "A"}


def test_list_events_is_sorted_and_scoped_to_the_day():
    s = EnvState()
    for title, start, end in [("Late", "2026-12-04T15:00", "2026-12-04T16:00"),
                              ("Early", "2026-12-04T09:00", "2026-12-04T10:00"),
                              ("Other day", "2026-12-05T09:00", "2026-12-05T10:00")]:
        s, _ = execute(s, c("add_event", title=title, start=start, end=end), ALL)
    _, out = execute(s, c("list_events", date="2026-12-04"), ALL)
    assert out.result == "09:00-10:00 Early\n15:00-16:00 Late"
