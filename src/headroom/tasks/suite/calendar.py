"""Calendar tasks (tiers as defined in suite/files.py)."""

from datetime import datetime

from headroom.schemas.env import EnvState, Event, Note
from headroom.schemas.task import Count, EventExists, Preserved, Task, Unchanged

CAL_TOOLS = ("add_event", "list_events")


def _ev(title: str, start: str, end: str) -> Event:
    return Event(title=title, start=datetime.fromisoformat(start), end=datetime.fromisoformat(end))


def _exists(title: str, start: str, end: str) -> EventExists:
    return EventExists(title=title, start=datetime.fromisoformat(start), end=datetime.fromisoformat(end))


TASKS = [
    # ---- tier 1 ----
    Task(
        task_id="calendar.add_01",
        template_id="calendar.add",
        split="tune",
        tier=1,
        prompt="Schedule 'Design review' on 2026-12-03 from 14:00 to 15:30.",
        tools=CAL_TOOLS,
        initial_state=EnvState(events=(_ev("Lunch", "2026-12-03T12:00", "2026-12-03T13:00"),)),
        checks=(
            _exists("Design review", "2026-12-03T14:00", "2026-12-03T15:30"),
            Preserved(scope="events"),
            Count(scope="events", n=2),
            Unchanged(scope="files"),
            Unchanged(scope="notes"),
        ),
        max_steps=4,
    ),
    Task(
        task_id="calendar.add_relative_01",
        template_id="calendar.add_relative",
        split="tune",
        tier=1,
        prompt="Today is Tuesday 2026-12-01. Add 'Dentist' tomorrow from 09:00 to 09:45.",
        tools=CAL_TOOLS,
        initial_state=EnvState(events=(_ev("Gym", "2026-12-01T18:00", "2026-12-01T19:00"),)),
        checks=(
            _exists("Dentist", "2026-12-02T09:00", "2026-12-02T09:45"),
            Preserved(scope="events"),
            Count(scope="events", n=2),
            Unchanged(scope="files"),
            Unchanged(scope="notes"),
        ),
        max_steps=4,
    ),
    # ---- tier 3 ----
    Task(
        task_id="calendar.find_slot_01",
        template_id="calendar.find_slot",
        split="tune",
        tier=3,
        prompt=(
            "On 2026-12-04, book a 1-hour event titled 'Sync with Sam' in the earliest free hour between "
            "09:00 and 12:00. Check that day's calendar first and do not change existing events."
        ),
        tools=CAL_TOOLS,
        initial_state=EnvState(
            events=(
                _ev("Standup", "2026-12-04T09:00", "2026-12-04T10:00"),
                _ev("Interview", "2026-12-04T10:00", "2026-12-04T11:00"),
                _ev("Dentist", "2026-12-05T09:00", "2026-12-05T10:00"),  # other day: free at 09:00 there
            )
        ),
        checks=(
            _exists("Sync with Sam", "2026-12-04T11:00", "2026-12-04T12:00"),
            Preserved(scope="events"),
            Count(scope="events", n=4),
            Unchanged(scope="files"),
            Unchanged(scope="notes"),
        ),
        max_steps=8,
    ),
    Task(
        task_id="calendar.from_note_01",
        template_id="calendar.from_note",
        split="tune",
        tier=3,
        prompt=(
            "Find my note about the kickoff and add the kickoff to the calendar as 'Kickoff' "
            "at the date and time the note gives."
        ),
        tools=("search_notes", "add_event", "list_events"),
        initial_state=EnvState(notes=(
            Note(title="Project K plan", body="Kickoff with the design team on 2026-12-08 from 10:00 to 11:00 in room 4B."),
            Note(title="Retro", body="Retro on 2026-12-15 from 16:00 to 17:00."),
        )),
        checks=(
            _exists("Kickoff", "2026-12-08T10:00", "2026-12-08T11:00"),
            Count(scope="events", n=1),
            Unchanged(scope="notes"),
            Unchanged(scope="files"),
        ),
        max_steps=10,
    ),
    Task(
        task_id="calendar.prep_before_01",
        template_id="calendar.prep_before",
        split="tune",
        tier=3,
        prompt=(
            "On 2026-12-10, add a 30-minute event titled 'Prep' that ends exactly when the "
            "'Board meeting' that day starts."
        ),
        tools=CAL_TOOLS,
        initial_state=EnvState(events=(
            _ev("Lunch", "2026-12-10T12:00", "2026-12-10T13:00"),
            _ev("Board meeting", "2026-12-10T15:00", "2026-12-10T16:00"),
            _ev("Board meeting", "2026-12-17T10:00", "2026-12-17T11:00"),  # a different day
        )),
        checks=(
            _exists("Prep", "2026-12-10T14:30", "2026-12-10T15:00"),
            Preserved(scope="events"),
            Count(scope="events", n=4),
            Unchanged(scope="files"),
            Unchanged(scope="notes"),
        ),
        max_steps=10,
    ),
]
