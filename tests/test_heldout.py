"""Held-out tasks: graders tested with scripted solutions only. No model runs these tasks until the
final held-out run. The freeze test fails on any edit to a held-out task or to grader semantics."""

import hashlib
import json
import tomllib
from collections import Counter

import pytest
from conftest import ROOT
from helpers import c, play

from headroom.schemas.chat import ToolCall
from headroom.schemas.task import NoteExists
from headroom.tasks.checks import GRADER_VERSION, grade
from headroom.tasks.executor import execute
from headroom.tasks.suite import HELDOUT_TASKS, TUNE_TASKS, HeldOutError, get_task, heldout, select_tasks

FREEZE = ROOT / "experiments" / "heldout_freeze.toml"


def heldout_hash() -> str:
    blob = json.dumps([t.model_dump(mode="json") for t in HELDOUT_TASKS], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def f(task_id: str, path: str) -> str:
    return get_task(task_id).initial_state.files[path]


PROJECTS_ATLAS = f("files.find_owner_copy_01", "projects/atlas.md")
PROJECTS_BOREAL = f("files.find_owner_copy_01", "projects/boreal.md")
APP_INI = f("files.config_from_note_01", "app.ini")
ADDRESS = f("notes.file_to_note_01", "address.txt")
Q3 = f("files.read_missing_01", "archive/q3-final.txt")
Q2 = f("files.read_missing_01", "archive/q2-final.txt")
BIRTHDAYS = {"Ana": "1990-03-14", "Tom": "1988-07-02", "Priya": "1993-11-25", "Marta": "1991-01-09"}
SESSIONS = [("Opening talk", "09:30", "10:30"), ("Hands-on lab", "10:45", "12:00"), ("Panel", "13:00", "14:00"),
            ("Closing", "14:15", "15:00")]


def ev(title: str, start: str, end: str) -> ToolCall:
    return c("add_event", title=title, start=start, end=end)


# task_id -> (passing calls, a plausible wrong solution, kinds of the checks it must break)
CASES: dict[str, tuple[list[ToolCall], list[ToolCall], set[str]]] = {
    # ---------------- tier 1 ----------------
    "files.create_csv_01": (
        [c("write_file", path="people.csv", content="name,age\nAna,31\nTom,28\n")],
        # No header line.
        [c("write_file", path="people.csv", content="Ana,31\nTom,28\n")],
        {"file_equals"},
    ),
    "files.blank_out_01": (
        [c("write_file", path="cache.txt", content="")],
        # Gets rid of the file instead of emptying it.
        [c("rename_file", src="cache.txt", dst="cache.old")],
        {"file_equals"},
    ),
    "files.move_up_01": (
        [c("rename_file", src="reports/2026/q3.txt", dst="q3.txt")],
        # Moves it up one level only.
        [c("rename_file", src="reports/2026/q3.txt", dst="reports/q3.txt")],
        {"file_equals"},
    ),
    "notes.add_rota_01": (
        [c("add_note", title="Rota", body="Ana: Monday\nTom: Tuesday")],
        # Leaves Tom out.
        [c("add_note", title="Rota", body="Ana: Monday")],
        {"note_exists"},
    ),
    "calendar.add_overnight_01": (
        [ev("Night shift", "2026-12-05T22:00", "2026-12-06T06:00")],
        # Anchors the shift to the wrong night.
        [ev("Night shift", "2026-12-06T22:00", "2026-12-07T06:00")],
        {"event_exists"},
    ),
    "calendar.add_duration_01": (
        [ev("Workshop", "2026-12-09T13:15", "2026-12-09T15:45")],
        # Drops the 15 minutes from the start time.
        [ev("Workshop", "2026-12-09T13:15", "2026-12-09T15:30")],
        {"event_exists"},
    ),
    # ---------------- tier 2 ----------------
    "files.prepend_01": (
        [c("read_file", path="stock.txt"),
         c("write_file", path="stock.txt", content="# Inventory\nbolts 40\nnuts 120\nwashers 75\n")],
        # Appends instead of prepending.
        [c("read_file", path="stock.txt"),
         c("write_file", path="stock.txt", content="bolts 40\nnuts 120\nwashers 75\n# Inventory\n")],
        {"file_equals"},
    ),
    "files.drop_line_01": (
        [c("read_file", path="groceries.txt"), c("write_file", path="groceries.txt", content="milk\nbread\neggplant\n")],
        # Also drops 'eggplant', which starts like 'eggs'.
        [c("read_file", path="groceries.txt"), c("write_file", path="groceries.txt", content="milk\nbread\n")],
        {"file_equals"},
    ),
    "files.sort_lines_01": (
        [c("read_file", path="names.txt"), c("write_file", path="names.txt", content="Ana\nMarta\nPriya\nTom\n")],
        # Reverse order.
        [c("read_file", path="names.txt"), c("write_file", path="names.txt", content="Tom\nPriya\nMarta\nAna\n")],
        {"file_equals"},
    ),
    "files.extract_value_01": (
        [c("read_file", path="server.ini"), c("write_file", path="port.txt", content="8443")],
        # Takes admin_port.
        [c("read_file", path="server.ini"), c("write_file", path="port.txt", content="9000")],
        {"file_equals"},
    ),
    "notes.file_to_note_01": (
        [c("read_file", path="address.txt"), c("add_note", title="Address", body=ADDRESS)],
        # Loses the postcode line.
        [c("read_file", path="address.txt"), c("add_note", title="Address", body="Marta Silva\nRua das Flores 12")],
        {"note_exists"},
    ),
    "notes.lookup_number_01": (
        [c("search_notes", query="Contacts"), c("write_file", path="priya.txt", content="555-0187")],
        # Takes Priyanka's number.
        [c("search_notes", query="Priya"), c("write_file", path="priya.txt", content="555-0111")],
        {"file_equals"},
    ),
    "notes.filter_expenses_01": (
        [c("read_file", path="expenses.txt"),
         c("add_note", title="Big expenses", body="rent 950\nlaptop repair 180\ntrain pass 120")],
        # Misses one expense.
        [c("read_file", path="expenses.txt"), c("add_note", title="Big expenses", body="rent 950\nlaptop repair 180")],
        {"note_exists"},
    ),
    # ---------------- tier 3 ----------------
    "files.collect_lines_01": (
        [c("list_files", prefix="feedback/"), c("read_file", path="feedback/01.txt"),
         c("read_file", path="feedback/02.txt"), c("read_file", path="feedback/03.txt"),
         c("write_file", path="summary.txt", content="Fast checkout.\nLove the dark mode.\nSearch is slow.\n")],
        # Misses a file.
        [c("list_files", prefix="feedback/"), c("write_file", path="summary.txt", content="Fast checkout.\nSearch is slow.\n")],
        {"file_equals"},
    ),
    "files.find_owner_copy_01": (
        [c("list_files", prefix="projects/"), c("read_file", path="projects/boreal.md"),
         c("write_file", path="handover/boreal.md", content=PROJECTS_BOREAL)],
        # Copies the project Priya reviews.
        [c("list_files", prefix="projects/"), c("write_file", path="handover/atlas.md", content=PROJECTS_ATLAS)],
        {"file_equals"},
    ),
    "files.config_from_note_01": (
        [c("search_notes", query="new server"), c("read_file", path="app.ini"),
         c("write_file", path="app.ini", content=APP_INI.replace("port=8080", "port=9443"))],
        # Uses the old server's port.
        [c("search_notes", query="server"), c("write_file", path="app.ini", content=APP_INI.replace("8080", "7000"))],
        {"file_equals"},
    ),
    "notes.latest_decision_01": (
        [c("search_notes", query="launch"), c("add_note", title="Launch date", body="Launch: 2027-01-26.")],
        # Takes the highest-numbered note, not the most recent decision.
        [c("search_notes", query="launch"), c("add_note", title="Launch date", body="Launch: 2027-01-19.")],
        {"note_exists"},
    ),
    "calendar.after_last_01": (
        [c("list_events", date="2026-12-11"), ev("Wrap-up", "2026-12-11T16:30", "2026-12-11T17:15")],
        # Uses the next day's last event.
        [ev("Wrap-up", "2026-12-11T18:00", "2026-12-11T18:45")],
        {"event_exists"},
    ),
    "calendar.free_afternoon_01": (
        [c("list_events", date="2026-12-14"), c("list_events", date="2026-12-15"),
         ev("Deep work", "2026-12-15T14:00", "2026-12-15T16:00")],
        # Picks the first completely empty day instead.
        [ev("Deep work", "2026-12-16T14:00", "2026-12-16T16:00")],
        {"event_exists"},
    ),
    "calendar.from_invite_01": (
        [c("list_files", prefix="inbox/"), c("read_file", path="inbox/b.txt"),
         ev("Budget review", "2026-12-18T10:00", "2026-12-18T11:30")],
        # Takes the slides deadline from the reminder.
        [c("read_file", path="inbox/c.txt"), ev("Budget review", "2026-12-16T17:00", "2026-12-16T18:00")],
        {"event_exists"},
    ),
    # ---------------- tier 4 ----------------
    "files.filter_csv_01": (
        [c("read_file", path="sales.csv"), c("write_file", path="islands.csv", content=heldout.SALES_ISLANDS)],
        # Rows without the header.
        [c("read_file", path="sales.csv"),
         c("write_file", path="islands.csv", content=heldout.SALES_ISLANDS.split("\n", 1)[1])],
        {"file_equals"},
    ),
    "notes.long_handbook_01": (
        [c("search_notes", query="wifi"), c("write_file", path="guest-wifi.txt", content="lantern-58")],
        # Takes the staff password.
        [c("search_notes", query="wifi"), c("write_file", path="guest-wifi.txt", content="harbor-21")],
        {"file_equals"},
    ),
    "notes.birthday_batch_01": (
        [c("read_file", path="birthdays.txt"), *(c("add_note", title=f"{n} birthday", body=d) for n, d in BIRTHDAYS.items())],
        # Stops after three people.
        [c("read_file", path="birthdays.txt"),
         *(c("add_note", title=f"{n} birthday", body=d) for n, d in list(BIRTHDAYS.items())[:3])],
        {"note_exists", "count"},
    ),
    "calendar.schedule_file_01": (
        [c("read_file", path="schedule.txt"), *(ev(t, f"2026-12-12T{s}", f"2026-12-12T{e}") for t, s, e in SESSIONS)],
        # Skips the last session.
        [c("read_file", path="schedule.txt"), *(ev(t, f"2026-12-12T{s}", f"2026-12-12T{e}") for t, s, e in SESSIONS[:3])],
        {"event_exists", "count"},
    ),
    "files.swap_names_01": (
        [c("rename_file", src="a.txt", dst="tmp.txt"), c("rename_file", src="b.txt", dst="a.txt"),
         c("rename_file", src="tmp.txt", dst="b.txt")],
        # Forgets the last step: the temporary file stays and b.txt is missing.
        [c("rename_file", src="a.txt", dst="tmp.txt"), c("rename_file", src="b.txt", dst="a.txt")],
        {"file_equals"},
    ),
    "files.read_missing_01": (
        [c("read_file", path="pointer.txt"), c("list_files", prefix="archive/"),
         c("read_file", path="archive/q3-final.txt"), c("write_file", path="backup/q3-final.txt", content=Q3)],
        # Copies the wrong archived report.
        [c("read_file", path="pointer.txt"), c("write_file", path="backup/q2-final.txt", content=Q2)],
        {"file_equals"},
    ),
}

IDS = [t.task_id for t in HELDOUT_TASKS]

# Note tasks: notes that contain every keyword but state a fact wrongly. The grader must fail them.
NOTE_NEGATIVES: dict[str, list[tuple[str, str]]] = {
    "notes.add_rota_01": [("Rota", "Ana: Tuesday\nTom: Monday")],
    "notes.file_to_note_01": [("Address", "Marta Silva\nRua das Flores 21\n1200-195 Porto\nLisboa")],
    "notes.filter_expenses_01": [("Big expenses", "rent 180\nlaptop repair 120\ntrain pass 950")],
    "notes.latest_decision_01": [("Launch date", "Launch: 2027-01-19.\nEarlier plan: 2027-01-26.")],
    # Ana's and Tom's dates swapped.
    "notes.birthday_batch_01": [
        ("Ana birthday", BIRTHDAYS["Tom"]), ("Tom birthday", BIRTHDAYS["Ana"]),
        ("Priya birthday", BIRTHDAYS["Priya"]), ("Marta birthday", BIRTHDAYS["Marta"]),
    ],
}

RECOVERY_FIRST_CALL = {
    "files.swap_names_01": c("rename_file", src="a.txt", dst="b.txt"),
    "files.read_missing_01": c("read_file", path="reports/q3-final.txt"),
}

# Token counts measured 2026-11-25 with qwen3.5:4b's tokenizer (llama-server /tokenize, no inference).
BASE_PROMPT_TOKENS = 585
LONG_CONTENT = {"sales.csv": (heldout.SALES, 3833, 2690), "handbook": (heldout.HANDBOOK, 10391, 2301)}


def test_freeze_hash_matches():
    frozen = tomllib.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["tasks"] == len(HELDOUT_TASKS)
    assert frozen["grader_version"] == GRADER_VERSION, "grader semantics changed after the held-out freeze"
    assert frozen["sha256"] == heldout_hash(), "a held-out task changed after it was frozen"


def test_shape_mirrors_the_tuning_half():
    assert len(HELDOUT_TASKS) == 26
    assert Counter(t.tier for t in HELDOUT_TASKS) == Counter(t.tier for t in TUNE_TASKS) == {1: 6, 2: 7, 3: 7, 4: 6}

    def family_tiers(tasks):
        return Counter((t.task_id.split(".")[0], t.tier) for t in tasks if t.tier < 4)

    assert family_tiers(HELDOUT_TASKS) == family_tiers(TUNE_TASKS)
    assert not {t.template_id for t in HELDOUT_TASKS} & {t.template_id for t in TUNE_TASKS}


def test_every_heldout_task_has_a_pass_and_a_fail_case():
    assert set(CASES) == set(IDS)


@pytest.mark.parametrize("task_id", IDS)
def test_reference_solution_passes(task_id):
    task = get_task(task_id)
    result = grade(task, play(task, CASES[task_id][0]))
    assert result.success, [r.detail for r in result.checks if not r.passed]


@pytest.mark.parametrize("task_id", IDS)
def test_wrong_solution_fails_for_the_intended_reason(task_id):
    task = get_task(task_id)
    _, wrong, kinds = CASES[task_id]
    result = grade(task, play(task, wrong))
    assert not result.success
    assert {r.check.kind for r in result.checks if not r.passed} == kinds


@pytest.mark.parametrize("task_id", IDS)
def test_doing_nothing_fails(task_id):
    task = get_task(task_id)
    assert not grade(task, task.initial_state).success


def test_every_note_task_has_a_negative_case():
    assert set(NOTE_NEGATIVES) == {t.task_id for t in HELDOUT_TASKS if any(isinstance(x, NoteExists) for x in t.checks)}


@pytest.mark.parametrize("task_id", sorted(NOTE_NEGATIVES))
def test_note_with_right_keywords_but_wrong_fact_fails(task_id):
    task = get_task(task_id)
    notes = NOTE_NEGATIVES[task_id]
    bodies = " ".join(b for _, b in notes).casefold()
    terms = {t for ch in task.checks if isinstance(ch, NoteExists) for fact in ch.facts for t in fact}
    assert all(t.casefold() in bodies for t in terms)  # a real probe: every keyword is present
    result = grade(task, play(task, [c("add_note", title=title, body=body) for title, body in notes]))
    assert not result.success
    assert {r.check.kind for r in result.checks if not r.passed} == {"note_exists"}


@pytest.mark.parametrize("task_id", sorted(RECOVERY_FIRST_CALL))
def test_recovery_tasks_start_with_a_tool_error(task_id):
    task = get_task(task_id)
    state, outcome = execute(task.initial_state, RECOVERY_FIRST_CALL[task_id], task.tools)
    assert outcome.status == "tool_error" and state == task.initial_state


@pytest.mark.parametrize("name", sorted(LONG_CONTENT))
def test_long_content_overflows_2048_but_fits_8192_twice(name):
    text, chars, tokens = LONG_CONTENT[name]
    assert len(text) == chars
    assert BASE_PROMPT_TOKENS + tokens > 2048
    assert BASE_PROMPT_TOKENS + 2 * tokens + 1024 < 8192


def test_tier_4_has_two_of_each_kind():
    t4 = {t.task_id for t in HELDOUT_TASKS if t.tier == 4}
    long = {"files.filter_csv_01", "notes.long_handbook_01"}
    chain = {"notes.birthday_batch_01", "calendar.schedule_file_01"}
    assert t4 == long | chain | set(RECOVERY_FIRST_CALL)


def test_runner_default_excludes_heldout_and_refuses_it_without_the_flag():
    assert select_tasks(None) == [t.task_id for t in TUNE_TASKS]
    with pytest.raises(HeldOutError):
        select_tasks(["files.rename_01", "files.swap_names_01"])
    assert select_tasks(["files.swap_names_01"], allow_heldout=True) == ["files.swap_names_01"]
    with pytest.raises(KeyError):
        select_tasks(["no.such_01"])
