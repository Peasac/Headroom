from headroom.schemas.task import Task
from headroom.tasks.suite import calendar, files, notes

ALL_TASKS: tuple[Task, ...] = (*files.TASKS, *notes.TASKS, *calendar.TASKS)
_BY_ID = {t.task_id: t for t in ALL_TASKS}
if len(_BY_ID) != len(ALL_TASKS):
    raise RuntimeError("duplicate task_id in suite")


def get_task(task_id: str) -> Task:
    return _BY_ID[task_id]
