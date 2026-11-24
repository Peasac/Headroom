from datetime import datetime

from pydantic import Field

from headroom.schemas.common import Frozen


class Note(Frozen):
    title: str
    body: str


class Event(Frozen):
    title: str
    start: datetime
    end: datetime


class EnvState(Frozen):
    """The whole mock environment. Graders look only at this, never at the model's words."""

    files: dict[str, str] = Field(default_factory=dict)
    notes: tuple[Note, ...] = ()
    events: tuple[Event, ...] = ()
