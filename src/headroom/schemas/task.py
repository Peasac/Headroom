from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from headroom.schemas.common import Frozen
from headroom.schemas.env import EnvState

Scope = Literal["files", "notes", "events"]


class FileEquals(Frozen):
    """File exists with this content (line endings and trailing whitespace normalized)."""

    kind: Literal["file_equals"] = "file_equals"
    path: str
    content: str


class FileAbsent(Frozen):
    kind: Literal["file_absent"] = "file_absent"
    path: str


class NoteExists(Frozen):
    """A note with this title (case-insensitive) that states every fact.

    A fact is a group of terms that must all appear (case-insensitive) within one line or
    sentence of the body. Grouping ties a keyword to its context: ('budget',) passes on any
    mention of a budget, ('budget', 'priya') only where the budget and Priya are in the same
    item. Use distinctive entities (names, ids), not phrases, so paraphrase still passes.
    """

    kind: Literal["note_exists"] = "note_exists"
    title: str
    facts: tuple[tuple[str, ...], ...] = ()


class EventExists(Frozen):
    kind: Literal["event_exists"] = "event_exists"
    title: str
    start: datetime
    end: datetime


class Unchanged(Frozen):
    """This part of the state is exactly as it started."""

    kind: Literal["unchanged"] = "unchanged"
    scope: Scope


class Preserved(Frozen):
    """Every item that existed at the start still exists, unmodified. Additions are allowed."""

    kind: Literal["preserved"] = "preserved"
    scope: Scope


class Count(Frozen):
    kind: Literal["count"] = "count"
    scope: Scope
    n: int = Field(ge=0)


Check = Annotated[
    FileEquals | FileAbsent | NoteExists | EventExists | Unchanged | Preserved | Count,
    Field(discriminator="kind"),
]


class Task(Frozen):
    task_id: str = Field(pattern=r"^[a-z_]+\.[a-z_]+_\d{2}$")
    template_id: str = Field(description="Tasks are split into tune/heldout by template, never by instance.")
    split: Literal["tune", "heldout"]
    tier: Literal[1, 2, 3, 4]
    prompt: str
    tools: tuple[str, ...] = Field(min_length=1)
    initial_state: EnvState
    checks: tuple[Check, ...] = Field(min_length=1)
    max_steps: int = Field(gt=0)

    @model_validator(mode="after")
    def _id_matches_template(self) -> "Task":
        if not self.task_id.startswith(self.template_id + "_"):
            raise ValueError(f"task_id {self.task_id!r} must start with template_id {self.template_id!r} + '_'")
        return self
