"""Every task has a reference solution that passes and a plausible wrong solution that fails
for the intended reason. Solutions run through the real executor, so tools and graders are
tested together."""

from collections import Counter

import pytest
from helpers import c, play

from headroom.schemas.chat import ToolCall
from headroom.schemas.task import NoteExists, Task
from headroom.tasks.checks import grade
from headroom.tasks.executor import execute
from headroom.tasks.suite import TUNE_TASKS, get_task, hard
from headroom.tasks.tools import REGISTRY


def file_of(task_id: str, path: str) -> str:
    return get_task(task_id).initial_state.files[path]


PLAN = file_of("files.rename_01", "drafts/plan.md")
BASE = file_of("files.copy_01", "config/base.yaml")
B_MD = file_of("files.find_and_move_01", "drafts/b.md")
RECIPE = get_task("notes.to_file_01").initial_state.notes[0].body
SCANS = [p for p in get_task("files.batch_rename_01").initial_state.files if p.startswith("scans/") and p.endswith(".txt")]
DECISIONS = (
    "Kestrel payroll vendor - Priya\nOffsite in Lisbon - Tom\nRelease train every second Tuesday - Ana\n"
    "Two support engineers in Q1 - Marta"
)

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
    # ---------------- tier 4 ----------------
    "files.log_errors_01": (
        [c("read_file", path="logs/app.log"), c("write_file", path="errors.txt", content=hard.APP_LOG_ERRORS)],
        # Takes the WARN lines too.
        [c("read_file", path="logs/app.log"), c("write_file", path="errors.txt", content="\n".join(
            line for line in hard.APP_LOG.splitlines() if " ERROR " in line or " WARN " in line))],
        {"file_equals"},
    ),
    "notes.minutes_decisions_01": (
        [c("read_file", path="minutes.txt"), c("add_note", title="Decisions", body=DECISIONS)],
        # Misses the last decision (the one a truncated read would lose).
        [c("read_file", path="minutes.txt"), c("add_note", title="Decisions", body=DECISIONS.rsplit("\n", 1)[0])],
        {"note_exists"},
    ),
    "files.batch_rename_01": (
        [c("list_files", prefix="scans/"), *(c("rename_file", src=p, dst=p.replace(".txt", ".md")) for p in SCANS)],
        # Stops after four of the five files.
        [c("list_files", prefix="scans/"),
         *(c("rename_file", src=p, dst=p.replace(".txt", ".md")) for p in SCANS[:4])],
        {"file_equals", "file_absent"},
    ),
    "calendar.weekday_series_01": (
        [c("add_event", title="Standup", start=f"2026-12-{d}T09:00", end=f"2026-12-{d}T09:15") for d in range(14, 19)],
        # Monday to Thursday only.
        [c("add_event", title="Standup", start=f"2026-12-{d}T09:00", end=f"2026-12-{d}T09:15") for d in range(14, 18)],
        {"event_exists", "count"},
    ),
    "notes.title_taken_01": (
        [c("search_notes", query="Ideas"), c("add_note", title="Ideas 2", body="Try the night market on Friday.")],
        # Uses a different fallback title.
        [c("add_note", title="Ideas (new)", body="Try the night market on Friday.")],
        {"note_exists"},
    ),
    "files.move_conflict_01": (
        [c("rename_file", src="done/invoice.txt", dst="done/invoice-old.txt"),
         c("rename_file", src="inbox/invoice.txt", dst="done/invoice.txt")],
        # Renames the old invoice but never moves the new one.
        [c("rename_file", src="done/invoice.txt", dst="done/invoice-old.txt")],
        {"file_equals", "file_absent"},
    ),
}

IDS = [t.task_id for t in TUNE_TASKS]

# Note tasks: a note that mentions every keyword, but with one fact wrong. Keyword matching alone
# would pass these; the grader must not.
NOTE_NEGATIVES: dict[str, str] = {
    # Takes the travel-budget discussion line as the budget action item.
    "notes.from_file_01": "- Ana: present the travel budget\n- Ana: book room 4B for Friday\n- Priya: renew the parking permit",
    # Network and password swapped.
    "notes.add_01": "Network: river42\nPassword: HomeNet",
    # Numbers swapped between Ana and Tom.
    "notes.contacts_01": "Ana: 555-0199\nTom: 555-0142",
    # Owners rotated by one decision.
    "notes.minutes_decisions_01": "Kestrel - Tom\nLisbon - Ana\nTuesday - Marta\nQ1 - Priya",
    # Night market and Friday both stated, but not together.
    "notes.title_taken_01": "Try the night market.\nFriday is busy.",
}


def test_suite_shape():
    assert len(TUNE_TASKS) == 26
    assert Counter(t.tier for t in TUNE_TASKS) == {1: 6, 2: 7, 3: 7, 4: 6}
    assert {t.split for t in TUNE_TASKS} == {"tune"}


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
    assert set(NOTE_NEGATIVES) == {t.task_id for t in TUNE_TASKS if _note_checks(t)}


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


# Tier 4 "recovery": the natural first call fails with a tool error (definition in suite/files.py).
RECOVERY_FIRST_CALL = {
    "notes.title_taken_01": c("add_note", title="Ideas", body="Try the night market on Friday."),
    "files.move_conflict_01": c("rename_file", src="inbox/invoice.txt", dst="done/invoice.txt"),
}


@pytest.mark.parametrize("task_id", sorted(RECOVERY_FIRST_CALL))
def test_recovery_tasks_start_with_a_tool_error(task_id):
    task = get_task(task_id)
    state, outcome = execute(task.initial_state, RECOVERY_FIRST_CALL[task_id], task.tools)
    assert outcome.status == "tool_error" and state == task.initial_state


# Tier 4 "long": token counts measured 2026-11-25 with qwen3.5:4b's tokenizer (llama-server /tokenize,
# no inference). Character counts pin the content: if it changes, re-measure and update both.
BASE_PROMPT_TOKENS = 585  # system prompt + tools + chat template, reported by Ollama on a tier-1 task
LONG_CONTENT = {"app.log": (hard.APP_LOG, 4711, 2694), "minutes.txt": (hard.MINUTES, 9223, 2616)}


@pytest.mark.parametrize("name", sorted(LONG_CONTENT))
def test_long_content_overflows_2048_but_fits_8192_twice(name):
    text, chars, tokens = LONG_CONTENT[name]
    assert len(text) == chars
    assert BASE_PROMPT_TOKENS + tokens > 2048
    assert BASE_PROMPT_TOKENS + 2 * tokens + 1024 < 8192  # read twice, plus a full reply


def test_tier_4_has_two_of_each_kind():
    t4 = {t.task_id for t in TUNE_TASKS if t.tier == 4}
    chain = {"files.batch_rename_01", "calendar.weekday_series_01"}
    long = {"files.log_errors_01", "notes.minutes_decisions_01"}
    assert t4 == chain | long | set(RECOVERY_FIRST_CALL)
