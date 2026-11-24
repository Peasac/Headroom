"""Note tasks (tiers as defined in suite/files.py)."""

from headroom.schemas.env import EnvState, Note
from headroom.schemas.task import Count, FileEquals, NoteExists, Task, Unchanged

NOTE_TOOLS = ("read_file", "list_files", "add_note", "search_notes")
NOTE_FILE_TOOLS = ("search_notes", "add_note", "read_file", "write_file", "list_files")

_MEETING = """Weekly sync, 2026-11-23
Attendees: Priya, Tom, Ana

Discussed the Q3 numbers; revenue is flat.
Ana presented the travel budget; no changes needed.
Action: Tom to email the Q3 budget to Priya.
Tom mentioned the office move is delayed.
Action: Ana to book room 4B for Friday's review.
Action: Priya to renew the parking permit.
"""
_CONTACTS = """Ana Ruiz - phone 555-0142 - ana@example.com
Tom Berg - phone 555-0199 - tom@example.com
Li Wei - phone 555-0175 - li@example.com
"""
_RECIPE = "Pancakes: 2 eggs, 1 cup flour, 1 cup milk. Rest the batter 10 minutes."

TASKS = [
    # ---- tier 1 ----
    Task(
        task_id="notes.add_01",
        template_id="notes.add",
        split="tune",
        tier=1,
        prompt="Create a note titled 'Wifi' whose body has two lines: 'Network: HomeNet' and 'Password: river42'.",
        tools=NOTE_TOOLS,
        initial_state=EnvState(notes=(Note(title="Ideas", body="Try a 4-day sprint."),)),
        checks=(
            NoteExists(title="Wifi", facts=(("network", "homenet"), ("password", "river42"))),
            Count(scope="notes", n=2),
            Unchanged(scope="files"),
            Unchanged(scope="events"),
        ),
        max_steps=4,
    ),
    # ---- tier 2 ----
    Task(
        task_id="notes.from_file_01",
        template_id="notes.from_file",
        split="tune",
        tier=2,
        prompt="Read meeting.txt and create a note titled 'Meeting actions' that lists every action item in it.",
        tools=NOTE_TOOLS,
        initial_state=EnvState(
            files={"meeting.txt": _MEETING},
            notes=(Note(title="Ideas", body="Try a 4-day sprint."),),
        ),
        checks=(
            # One fact per action item: distinctive entities that must share a line or sentence,
            # so paraphrase passes but the travel-budget distractor does not count as the budget item.
            NoteExists(title="Meeting actions", facts=(("budget", "priya"), ("4B", "friday"), ("parking",))),
            Count(scope="notes", n=2),
            Unchanged(scope="files"),
            Unchanged(scope="events"),
        ),
        max_steps=6,
    ),
    Task(
        task_id="notes.contacts_01",
        template_id="notes.contacts",
        split="tune",
        tier=2,
        prompt="Read contacts.txt and create a note titled 'Call list' with Ana's and Tom's phone numbers.",
        tools=NOTE_TOOLS,
        initial_state=EnvState(
            files={"contacts.txt": _CONTACTS},
            notes=(Note(title="Groceries", body="eggs, rice, tea"),),
        ),
        checks=(
            NoteExists(title="Call list", facts=(("ana", "555-0142"), ("tom", "555-0199"))),
            Count(scope="notes", n=2),
            Unchanged(scope="files"),
            Unchanged(scope="events"),
        ),
        max_steps=6,
    ),
    Task(
        task_id="notes.to_file_01",
        template_id="notes.to_file",
        split="tune",
        tier=2,
        prompt="Save the body of my note titled 'Recipe' to recipe.txt.",
        tools=NOTE_FILE_TOOLS,
        initial_state=EnvState(notes=(
            Note(title="Recipe", body=_RECIPE),
            Note(title="Shopping", body="eggs, flour, milk, butter"),
        )),
        checks=(
            FileEquals(path="recipe.txt", content=_RECIPE),
            Count(scope="files", n=1),
            Unchanged(scope="notes"),
            Unchanged(scope="events"),
        ),
        max_steps=6,
    ),
    # ---- tier 3 ----
    Task(
        task_id="notes.find_code_01",
        template_id="notes.find_code",
        split="tune",
        tier=3,
        prompt="One of my notes has the gate code. Save just the gate code, nothing else, to gate.txt.",
        tools=NOTE_FILE_TOOLS,
        initial_state=EnvState(notes=(
            Note(title="Building", body="Gate code is 4471, enter it on the keypad. Parking is behind the gym."),
            Note(title="Garage", body="Garage door code is 9910."),
            Note(title="Wifi", body="Password: river42"),
        )),
        checks=(
            FileEquals(path="gate.txt", content="4471"),
            Count(scope="files", n=1),
            Unchanged(scope="notes"),
            Unchanged(scope="events"),
        ),
        max_steps=10,
    ),
]
