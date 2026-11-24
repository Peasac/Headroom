"""Append-only JSONL logs under results/.

Records are only ever appended: files are opened in append mode, and nothing here rewrites,
truncates or deletes a line. A torn last line (crash mid-write) is left in place; the next
record starts on a fresh line and readers report the torn line instead of dropping it silently.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from headroom.schemas.run import Run, Session

M = TypeVar("M", bound=BaseModel)


class AppendOnlyLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: BaseModel) -> None:
        line = record.model_dump_json().encode("utf-8")
        if b"\n" in line:
            raise ValueError("a record must serialize to a single line")
        with open(self.path, "ab") as f:  # append mode: writes can only land at the end
            if f.tell() > 0 and not _ends_with_newline(self.path):
                f.write(b"\n")  # isolate a torn previous line; never modify it
            f.write(line + b"\n")
            f.flush()
            os.fsync(f.fileno())


class RunLog(AppendOnlyLog):
    """results/<experiment>/runs.jsonl. Refuses a run_id that is already in the file."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._run_ids = {r.run_id for r in read_jsonl(self.path, Run)[0]} if self.path.exists() else set()

    def append(self, record: Run) -> None:  # type: ignore[override]
        if record.run_id in self._run_ids:
            raise ValueError(f"run_id {record.run_id} is already logged")
        super().append(record)
        self._run_ids.add(record.run_id)


def experiment_dir(results_root: str | Path, experiment_id: str) -> Path:
    return Path(results_root) / experiment_id


def run_log(results_root: str | Path, experiment_id: str) -> RunLog:
    return RunLog(experiment_dir(results_root, experiment_id) / "runs.jsonl")


def session_log(results_root: str | Path, experiment_id: str) -> AppendOnlyLog:
    return AppendOnlyLog(experiment_dir(results_root, experiment_id) / "sessions.jsonl")


def read_jsonl(path: str | Path, model: type[M]) -> tuple[list[M], list[tuple[int, str]]]:
    """(records, bad lines as (line number, error)). Bad lines are reported, never removed."""
    records, bad = [], []
    for lineno, raw in _lines(Path(path)):
        try:
            records.append(model.model_validate_json(raw))
        except (ValidationError, json.JSONDecodeError) as e:
            bad.append((lineno, str(e).splitlines()[0][:200]))
    return records, bad


def read_sessions(results_root: str | Path, experiment_id: str) -> list[Session]:
    path = experiment_dir(results_root, experiment_id) / "sessions.jsonl"
    return read_jsonl(path, Session)[0] if path.exists() else []


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if raw.strip():
                yield lineno, raw


def _ends_with_newline(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"
