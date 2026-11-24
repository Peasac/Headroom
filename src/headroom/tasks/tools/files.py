from typing import Annotated

from pydantic import AfterValidator, Field

from headroom.schemas.env import EnvState
from headroom.tasks.tools.base import ToolArgs, ToolError, ToolSpec


def _norm_path(p: str) -> str:
    p = p.strip().replace("\\", "/")
    parts = [x for x in p.split("/") if x and x != "."]
    if not parts or ".." in parts:
        raise ValueError("must be a relative file path without '..'")
    return "/".join(parts)


FilePath = Annotated[str, AfterValidator(_norm_path), Field(description="Relative path, e.g. notes/today.txt")]


class ReadFileArgs(ToolArgs):
    path: FilePath


def read_file(state: EnvState, a: ReadFileArgs) -> tuple[EnvState, str]:
    if a.path not in state.files:
        raise ToolError(f"no such file: {a.path}")
    return state, state.files[a.path]


class WriteFileArgs(ToolArgs):
    path: FilePath
    content: str


def write_file(state: EnvState, a: WriteFileArgs) -> tuple[EnvState, str]:
    verb = "overwrote" if a.path in state.files else "created"
    return state.model_copy(update={"files": {**state.files, a.path: a.content}}), f"{verb} {a.path}"


class RenameFileArgs(ToolArgs):
    src: FilePath
    dst: FilePath


def rename_file(state: EnvState, a: RenameFileArgs) -> tuple[EnvState, str]:
    if a.src not in state.files:
        raise ToolError(f"no such file: {a.src}")
    if a.dst in state.files:
        raise ToolError(f"destination exists: {a.dst}")
    files = {p: c for p, c in state.files.items() if p != a.src}
    files[a.dst] = state.files[a.src]
    return state.model_copy(update={"files": files}), f"renamed {a.src} to {a.dst}"


class ListFilesArgs(ToolArgs):
    prefix: str = Field(default="", description="Only list paths starting with this, e.g. 'drafts/'.")


def list_files(state: EnvState, a: ListFilesArgs) -> tuple[EnvState, str]:
    paths = sorted(p for p in state.files if p.startswith(a.prefix))
    return state, "\n".join(paths) if paths else "(no files)"


SPECS = [
    ToolSpec("read_file", "Read a text file.", ReadFileArgs, False, read_file),
    ToolSpec("write_file", "Create a text file, or overwrite it entirely.", WriteFileArgs, True, write_file),
    ToolSpec("rename_file", "Rename or move a file. Fails if the destination exists.", RenameFileArgs, True, rename_file),
    ToolSpec("list_files", "List file paths.", ListFilesArgs, False, list_files),
]
