"""Every task has a reference solution that passes and a plausible wrong solution that fails
for the intended reason. Solutions run through the real executor, so tools and graders are
tested together."""

from collections import Counter

import pytest

from headroom.schemas.chat import ToolCall
from headroom.schemas.env import EnvState
from headroom.schemas.task import NoteExists, Task
from headroom.tasks.checks import grade
from headroom.tasks.executor import execute
from headroom.tasks.suite import ALL_TASKS, get_task
from headroom.tasks.tools import REGISTRY


def c(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def play(task: Task, calls: list[ToolCall]) -> EnvState:
    state = task.initial_state
    for call in calls:
        state, outcome = execute(state, call, task.tools)
        assert outcome.status == "ok", outcome  # scripted solutions use only valid calls
    return state


def file_of(task_id: str, path: str) -> str:
    return get_task(task_id).initial_state.files[path]


PLAN = file_of("files.rename_01", "drafts/plan.md")
BASE = file_of("files.copy_01", "config/base.yaml")
B_MD = file_of("files.find_and_move_01", "drafts/b.md")
RECIPE = get_task("notes.to_file_01").initial_state.notes[0].body

# task_id -> (passing calls, the targeted wrong solution, kinds of the checks it must break)
CASES: dict[str, tuple[list[ToolCall], list[ToolCall], set[str]]] = {
    # ---------------- tier 1 ----------------
    "files.rename_01": (
        [c("rename_file", src="drafts/plan.md", dst="archive/plan.md")],
        # Copies instead of moving: the original stays behind.
        [c("read_file", path="drafts/plan.md"), c("write_file", path="archive/plan.md", content=PLAN)],
        {"file_absent", "count"},
    ),
    "files.create_01": (
        [c("write_file", path="todo/today.txt", content="call the bank")],
        # Ignores the folder.
        [c("write_file", path="today.txt", content="call the bank")],
        {"file_equals"},
    ),
    "files.overwrite_01": (
        [c("write_file", path="status.txt", content="deployed")],
        # Appends instead of replacing.
        [c("read_file", path="status.txt"), c("write_file", path="status.txt", content="pending\nchecked by Ana\ndeployed")],
        {"file_equals"},
    ),
    "notes.add_01": (
        [c("add_note", title="Wifi", body="Network: HomeNet\nPassword: river42")],
        # Wrong title.
        [c("add_note", title="Wifi password", body="Network: HomeNet\nPassword: river42")],
        {"note_exists"},
    ),
    "calendar.add_01": (
        [c("add_event", title="Design review", start="2026-12-03T14:00", end="2026-12-03T15:30")],
        # Wrong end time.
        [c("add_event", title="Design review", start="2026-12-03T14:00", end="2026-12-03T15:00")],
        {"event_exists"},
    ),
    "calendar.add_relative_01": (
        [c("add_event", title="Dentist", start="2026-12-02T09:00", end="2026-12-02T09:45")],
        # Books it today instead of tomorrow.
        [c("add_event", title="Dentist", start="2026-12-01T09:00", end="2026-12-01T09:45")],
        {"event_exists"},
    ),
    # ---------------- tier 2 ----------------
    "files.append_01": (
        [c("read_file", path="shopping.txt"), c("write_file", path="shopping.txt", content="eggs\nbread\nmilk")],
        # Overwrites without reading first: existing items lost.
        [c("write_file", path="shopping.txt", content="milk\n")],
        {"file_equals"},
    ),
    "files.copy_01": (
        [c("read_file", path="config/base.yaml"), c("write_file", path="config/prod.yaml", content=BASE)],
        # Renames instead of copying: the original is gone.
        [c("rename_file", src="config/base.yaml", dst="config/prod.yaml")],
        {"preserved", "count"},
    ),
    "files.edit_01": (
        [c("read_file", path="settings.ini"),
         c("write_file", path="settings.ini", content="[ui]\ntheme=dark\nfont_size=14\nlanguage=en\n")],
        # Writes only the changed line: the other lines are lost.
        [c("write_file", path="settings.ini", content="theme=dark\n")],
        {"file_equals"},
    ),
    "files.merge_01": (
        [c("read_file", path="a.txt"), c("read_file", path="b.txt"),
         c("write_file", path="all.txt", content="alpha\nbeta\ngamma\n")],
        # Wrong order.
        [c("read_file", path="a.txt"), c("read_file", path="b.txt"),
         c("write_file", path="all.txt", content="gamma\nalpha\nbeta\n")],
        {"file_equals"},
    ),
    "notes.from_file_01": (
        [
            c("read_file", path="meeting.txt"),
            c("add_note", title="Meeting actions", body=(
                "- Tom: email the Q3 budget to Priya\n- Ana: book room 4B for Friday\n- Priya: renew the parking permit"
            )),
        ],
        # Misses one action item.
        [
            c("read_file", path="meeting.txt"),
            c("add_note", title="Meeting actions", body="- email budget to Priya\n- book room 4B for Friday"),
        ],
        {"note_exists"},
    ),
    "notes.contacts_01": (
        [c("read_file", path="contacts.txt"), c("add_note", title="Call list", body="Ana: 555-0142\nTom: 555-0199")],
        # Only one of the two numbers.
        [c("read_file", path="contacts.txt"), c("add_note", title="Call list", body="Ana: 555-0142")],
        {"note_exists"},
    ),
    "notes.to_file_01": (
        [c("search_notes", query="Recipe"), c("write_file", path="recipe.txt", content=RECIPE)],
        # Saves the other note that the search also returns.
        [c("search_notes", query="eggs"), c("write_file", path="recipe.txt", content="eggs, flour, milk, butter")],
        {"file_equals"},
    ),
    # ---------------- tier 3 ----------------
    "calendar.find_slot_01": (
        [
            c("list_events", date="2026-12-04"),
            c("add_event", title="Sync with Sam", start="2026-12-04T11:00", end="2026-12-04T12:00"),
        ],
        # Skips the calendar check and books over the interview.
        [c("add_event", title="Sync with Sam", start="2026-12-04T10:00", end="2026-12-04T11:00")],
        {"event_exists"},
    ),
    "calendar.from_note_01": (
        [c("search_notes", query="kickoff"),
         c("add_event", title="Kickoff", start="2026-12-08T10:00", end="2026-12-08T11:00")],
        # Takes the time from the wrong note.
        [c("search_notes", query="2026"),
         c("add_event", title="Kickoff", start="2026-12-15T16:00", end="2026-12-15T17:00")],
        {"event_exists"},
    ),
    "calendar.prep_before_01": (
        [c("list_events", date="2026-12-10"),
         c("add_event", title="Prep", start="2026-12-10T14:30", end="2026-12-10T15:00")],
        # Uses the other Board meeting's time (2026-12-17 10:00).
        [c("add_event", title="Prep", start="2026-12-10T09:30", end="2026-12-10T10:00")],
        {"event_exists"},
    ),
    "files.find_and_move_01": (
        [c("list_files", prefix="drafts/"), c("read_file", path="drafts/a.md"), c("read_file", path="drafts/b.md"),
         c("rename_file", src="drafts/b.md", dst="final/b.md")],
        # Moves the Q3 budget file instead.
        [c("list_files", prefix="drafts/"), c("rename_file", src="drafts/c.md", dst="final/c.md")],
        {"file_equals", "file_absent"},
    ),
    "files.latest_log_01": (
        [c("list_files", prefix="logs/"), c("read_file", path="logs/2026-11-30.log"),
         c("write_file", path="logs/2026-11-30.log", content="boot ok\ndisk 91%\nreviewed\n")],
        # Right file, but writes without reading: valid call, wrong effect (log content lost).
        [c("list_files", prefix="logs/"), c("write_file", path="logs/2026-11-30.log", content="reviewed\n")],
        {"file_equals"},
    ),
    "notes.find_code_01": (
        [c("search_notes", query="gate"), c("write_file", path="gate.txt", content="4471")],
        # Takes the garage code.
        [c("search_notes", query="code"), c("write_file", path="gate.txt", content="9910")],
        {"file_equals"},
    ),
    "files.safe_write_01": (
        [c("list_files", prefix="archive/"), c("write_file", path="archive/report-2.md", content="Final numbers attached.")],
        # Writes straight to report.md: valid call, overwrites the signed-off report.
        [c("write_file", path="archive/report.md", content="Final numbers attached.")],
        {"file_equals", "count"},
    ),
}

IDS = [t.task_id for t in ALL_TASKS]

# Note tasks: a note that mentions every keyword, but with one fact wrong. Keyword matching alone
# would pass these; the grader must not.
NOTE_NEGATIVES: dict[str, str] = {
    # Takes the travel-budget discussion line as the budget action item.
    "notes.from_file_01": "- Ana: present the travel budget\n- Ana: book room 4B for Friday\n- Priya: renew the parking permit",
    # Network and password swapped.
    "notes.add_01": "Network: river42\nPassword: HomeNet",
    # Numbers swapped between Ana and Tom.
    "notes.contacts_01": "Ana: 555-0199\nTom: 555-0142",
}


def test_suite_shape():
    assert len(ALL_TASKS) == 20
    assert Counter(t.tier for t in ALL_TASKS) == {1: 6, 2: 7, 3: 7}
    # The held-out half is not written yet: everything here is tuning.
    assert {t.split for t in ALL_TASKS} == {"tune"}


def test_every_task_has_a_pass_and_a_fail_case():
    assert set(CASES) == set(IDS)


@pytest.mark.parametrize("task_id", IDS)
def test_reference_solution_passes(task_id):
    task = get_task(task_id)
    result = grade(task, play(task, CASES[task_id][0]))
    assert result.success, [r.detail for r in result.checks if not r.passed]


@pytest.mark.parametrize("task_id", IDS)
def test_wrong_solution_fails_for_the_intended_reason(task_id):
    task = get_task(task_id)
    _, wrong, expected_kinds = CASES[task_id]
    result = grade(task, play(task, wrong))
    assert not result.success
    assert {r.check.kind for r in result.checks if not r.passed} == expected_kinds


@pytest.mark.parametrize("task_id", IDS)
def test_doing_nothing_fails(task_id):
    task = get_task(task_id)
    assert not grade(task, task.initial_state).success


@pytest.mark.parametrize("task_id", IDS)
def test_task_tools_exist(task_id):
    assert set(get_task(task_id).tools) <= set(REGISTRY)


def test_grade_result_round_trips():
    task = get_task("calendar.add_01")
    result = grade(task, task.initial_state)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def _note_checks(task: Task) -> list[NoteExists]:
    return [ch for ch in task.checks if isinstance(ch, NoteExists)]


def test_every_note_task_has_a_negative_case():
    assert set(NOTE_NEGATIVES) == {t.task_id for t in ALL_TASKS if _note_checks(t)}


@pytest.mark.parametrize("task_id", sorted(NOTE_NEGATIVES))
def test_note_with_right_keywords_but_wrong_fact_fails(task_id):
    task = get_task(task_id)
    (check,) = _note_checks(task)
    body = NOTE_NEGATIVES[task_id]
    # The case is a real gap probe: every keyword is present somewhere in the body.
    assert all(term.casefold() in body.casefold() for fact in check.facts for term in fact)
    state = play(task, [c("add_note", title=check.title, body=body)])
    result = grade(task, state)
    assert not result.success
    assert {r.check.kind for r in result.checks if not r.passed} == {"note_exists"}
