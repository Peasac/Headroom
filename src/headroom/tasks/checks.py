"""Deterministic graders. They read the final EnvState only."""

import re

from headroom.schemas.env import EnvState
from headroom.schemas.result import CheckResult, GradeResult
from headroom.schemas.task import (
    Check, Count, EventExists, FileAbsent, FileEquals, NoteExists, Preserved, Task, Unchanged,
)

# Bump whenever a check's semantics change, so old grades can be told apart and regenerated.
# v2: NoteExists checks facts (terms co-occurring in one line/sentence), not loose keywords.
GRADER_VERSION = 2


def grade(task: Task, final: EnvState) -> GradeResult:
    results = tuple(_result(c, task.initial_state, final) for c in task.checks)
    return GradeResult(
        task_id=task.task_id, grader_version=GRADER_VERSION, success=all(r.passed for r in results), checks=results
    )


def _result(check: Check, initial: EnvState, final: EnvState) -> CheckResult:
    passed, detail = evaluate(check, initial, final)
    return CheckResult(check=check, passed=passed, detail=detail)


def evaluate(check: Check, initial: EnvState, final: EnvState) -> tuple[bool, str]:
    match check:
        case FileEquals(path=path, content=want):
            if path not in final.files:
                return False, f"{path} missing"
            got = final.files[path]
            ok = _norm_text(got) == _norm_text(want)
            return ok, f"{path} matches" if ok else f"{path}: expected {_clip(want)}, got {_clip(got)}"
        case FileAbsent(path=path):
            ok = path not in final.files
            return ok, f"{path} absent" if ok else f"{path} still exists"
        case NoteExists(title=title, facts=facts):
            notes = [n for n in final.notes if _same(n.title, title)]
            if not notes:
                return False, f"no note titled {title!r}; titles: {[n.title for n in final.notes]}"
            units = [u.casefold() for n in notes for u in _units(n.body)]
            missing = [f for f in facts if not any(all(t.casefold() in u for t in f) for u in units)]
            return not missing, (
                f"note {title!r} ok" if not missing else f"note {title!r}: no line or sentence states {missing}"
            )
        case EventExists(title=title, start=start, end=end):
            ok = any(_same(e.title, title) and e.start == start and e.end == end for e in final.events)
            got = [f"{e.title} {e.start:%Y-%m-%dT%H:%M}-{e.end:%H:%M}" for e in final.events]
            return ok, f"event {title!r} found" if ok else f"no {title!r} {start:%Y-%m-%dT%H:%M}-{end:%H:%M}; have {got}"
        case Unchanged(scope=scope):
            ok = getattr(final, scope) == getattr(initial, scope)
            return ok, f"{scope} unchanged" if ok else f"{scope} changed"
        case Preserved(scope=scope):
            lost = _lost(scope, initial, final)
            return not lost, f"{scope} preserved" if not lost else f"{scope} lost or modified: {lost}"
        case Count(scope=scope, n=n):
            got = len(getattr(final, scope))
            return got == n, f"{scope}: {got} (want {n})"
    raise TypeError(f"unhandled check {check!r}")


def _lost(scope: str, initial: EnvState, final: EnvState) -> list[str]:
    if scope == "files":
        return [p for p, c in initial.files.items() if final.files.get(p) != c]
    items = getattr(final, scope)
    return [x.title for x in getattr(initial, scope) if x not in items]


def _units(body: str) -> list[str]:
    """Lines, ';'-separated parts and sentences: the scope within which a fact's terms must co-occur."""
    return [u for u in re.split(r"\n|;|(?<=[.!?])\s+", body) if u.strip()]


def _norm_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").split("\n")).rstrip()


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _clip(s: str, n: int = 80) -> str:
    return repr(s if len(s) <= n else s[:n] + "...")
