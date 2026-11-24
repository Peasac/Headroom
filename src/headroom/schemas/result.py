from headroom.schemas.common import Frozen
from headroom.schemas.task import Check


class CheckResult(Frozen):
    check: Check
    passed: bool
    detail: str


class GradeResult(Frozen):
    task_id: str
    grader_version: int
    success: bool
    checks: tuple[CheckResult, ...]
