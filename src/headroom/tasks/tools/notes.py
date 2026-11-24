from pydantic import Field

from headroom.schemas.env import EnvState, Note
from headroom.tasks.tools.base import ToolArgs, ToolError, ToolSpec


class AddNoteArgs(ToolArgs):
    title: str = Field(min_length=1)
    body: str


def add_note(state: EnvState, a: AddNoteArgs) -> tuple[EnvState, str]:
    if any(n.title.casefold() == a.title.casefold() for n in state.notes):
        raise ToolError(f"a note titled {a.title!r} already exists")
    return state.model_copy(update={"notes": (*state.notes, Note(title=a.title, body=a.body))}), f"added note {a.title!r}"


class SearchNotesArgs(ToolArgs):
    query: str = Field(min_length=1, description="Case-insensitive text to find in titles or bodies.")


def search_notes(state: EnvState, a: SearchNotesArgs) -> tuple[EnvState, str]:
    q = a.query.casefold()
    hits = [n for n in state.notes if q in n.title.casefold() or q in n.body.casefold()]
    return state, "\n\n".join(f"# {n.title}\n{n.body}" for n in hits) if hits else "(no matching notes)"


SPECS = [
    ToolSpec("add_note", "Create a note. Titles are unique.", AddNoteArgs, True, add_note),
    ToolSpec("search_notes", "Search notes by text.", SearchNotesArgs, False, search_notes),
]
