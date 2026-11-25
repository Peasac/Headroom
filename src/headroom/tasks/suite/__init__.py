from collections.abc import Sequence

from headroom.schemas.task import Task
from headroom.tasks.suite import calendar, files, hard, heldout, notes

TUNE_TASKS: tuple[Task, ...] = (*files.TASKS, *notes.TASKS, *calendar.TASKS, *hard.TASKS)
HELDOUT_TASKS: tuple[Task, ...] = tuple(heldout.TASKS)
ALL_TASKS: tuple[Task, ...] = (*TUNE_TASKS, *HELDOUT_TASKS)
_BY_ID = {t.task_id: t for t in ALL_TASKS}
if len(_BY_ID) != len(ALL_TASKS):
    raise RuntimeError("duplicate task_id in suite")
if {t.split for t in TUNE_TASKS} != {"tune"} or {t.split for t in HELDOUT_TASKS} != {"heldout"}:
    raise RuntimeError("a task is in the wrong split module")
if {t.template_id for t in TUNE_TASKS} & {t.template_id for t in HELDOUT_TASKS}:
    raise RuntimeError("a template appears in both splits; split by template, never by instance")


def get_task(task_id: str) -> Task:
    return _BY_ID[task_id]


class HeldOutError(ValueError):
    """A held-out task was requested without explicitly allowing it."""


def select_tasks(task_ids: Sequence[str] | None, *, allow_heldout: bool = False) -> list[str]:
    """Task ids to run. Default: every tuning task. Held-out tasks stay frozen until the final
    held-out run, so asking for one needs allow_heldout=True."""
    if not task_ids:
        return [t.task_id for t in TUNE_TASKS]
    ids = list(task_ids)
    unknown = [t for t in ids if t not in _BY_ID]
    if unknown:
        raise KeyError(f"unknown task id(s): {unknown}")
    frozen = [t for t in ids if _BY_ID[t].split == "heldout"]
    if frozen and not allow_heldout:
        raise HeldOutError(f"held-out task(s) {frozen} are frozen until the final held-out run; pass --heldout")
    return ids
