import datetime as dt
from typing import Annotated

from pydantic import AfterValidator, Field, model_validator

from headroom.schemas.env import EnvState, Event
from headroom.tasks.tools.base import ToolArgs, ToolSpec


def _naive(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is not None:
        raise ValueError("use local time without a timezone, e.g. 2026-12-01T14:00")
    return value


LocalDateTime = Annotated[
    dt.datetime, AfterValidator(_naive), Field(description="Local time, ISO 8601, e.g. 2026-12-01T14:00")
]


class AddEventArgs(ToolArgs):
    title: str = Field(min_length=1)
    start: LocalDateTime
    end: LocalDateTime

    @model_validator(mode="after")
    def _end_after_start(self) -> "AddEventArgs":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


def add_event(state: EnvState, a: AddEventArgs) -> tuple[EnvState, str]:
    event = Event(title=a.title, start=a.start, end=a.end)
    return state.model_copy(update={"events": (*state.events, event)}), f"added {_fmt(event)} on {a.start.date()}"


class ListEventsArgs(ToolArgs):
    date: dt.date = Field(description="Day to list, YYYY-MM-DD.")


def list_events(state: EnvState, a: ListEventsArgs) -> tuple[EnvState, str]:
    day = sorted((e for e in state.events if e.start.date() == a.date), key=lambda e: e.start)
    return state, "\n".join(_fmt(e) for e in day) if day else f"(no events on {a.date})"


def _fmt(e: Event) -> str:
    return f"{e.start:%H:%M}-{e.end:%H:%M} {e.title}"


SPECS = [
    ToolSpec("add_event", "Add a calendar event.", AddEventArgs, True, add_event),
    ToolSpec("list_events", "List the events on one day.", ListEventsArgs, False, list_events),
]
