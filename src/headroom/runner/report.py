"""Per-condition, per-arm summary of logged runs.

Success is decided by the state grader alone; clean success also requires zero tool calls written
as text. Temperature is 0 with a fixed seed, so repeats are timing samples: success is counted
per task (a task passes when every repeat passes), latency per run.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from headroom.schemas.common import Measured
from headroom.schemas.run import Run
from headroom.stats import nearest_rank


def state_success(run: Run) -> bool:
    """Derived from the grade, so records from every schema version count the same way."""
    return run.grade.success


def clean_success(run: Run) -> bool:
    """State passes, no tool call written as text, and the run did not exceed the context cap."""
    return run.grade.success and run.malformed_text_calls == 0 and run.status != "context_exceeded"


def run_symbol(run: Run) -> str:
    """P: passes, clean. p: passes, not clean (text-written tool call or context exceeded). F: fails."""
    if not state_success(run):
        return "F"
    return "P" if clean_success(run) else "p"


@dataclass(frozen=True)
class ArmSummary:
    condition_id: str
    arm_id: str
    model_id: str
    mode: str
    tasks: int
    task_successes: int  # every repeat passes the state grader
    task_clean_successes: int  # every repeat passes cleanly
    inconsistent_tasks: tuple[str, ...]  # repeats disagree on success (counted as not passing)
    per_task: dict[str, str]  # task_id -> one symbol per repeat, in repeat order
    runs: int
    statuses: dict[str, int]
    p95_wall_s: float | None  # over all runs
    successful_runs: int
    p95_time_to_success_s: float | None  # over runs that pass the state grader
    invalid_state_changing: int
    invalid_other: int
    malformed_text_calls: int
    runs_with_malformed_text: int
    claimed_done_without_state_change: int
    context_exceeded: int
    cold_starts: int
    quiet_flagged: int
    backend_private_mb: tuple[float, float] | None  # (first run start, last run end)
    backend_rss_mb: tuple[float, float] | None


def summarize_runs(runs: Sequence[Run]) -> list[ArmSummary]:
    groups: dict[tuple[str, str], list[Run]] = {}
    for r in runs:
        groups.setdefault((r.condition_id, r.arm_id), []).append(r)
    out = []
    for (condition_id, arm_id), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: (r.session_id, r.order_index))
        by_task: dict[str, list[Run]] = {}
        for r in sorted(rs, key=lambda r: (r.task_id, r.repeat, r.session_id)):
            by_task.setdefault(r.task_id, []).append(r)
        passing = [r for r in rs if state_success(r)]
        out.append(ArmSummary(
            condition_id=condition_id,
            arm_id=arm_id,
            model_id=rs[0].config.model_id,
            mode=rs[0].config.mode,
            tasks=len(by_task),
            task_successes=sum(all(state_success(r) for r in t) for t in by_task.values()),
            task_clean_successes=sum(all(clean_success(r) for r in t) for t in by_task.values()),
            inconsistent_tasks=tuple(
                tid for tid, t in by_task.items() if len({state_success(r) for r in t}) > 1
            ),
            per_task={tid: "".join(run_symbol(r) for r in t) for tid, t in by_task.items()},
            runs=len(rs),
            statuses=dict(Counter(r.status for r in rs)),
            p95_wall_s=nearest_rank([r.wall_s for r in rs], 0.95) if rs else None,
            successful_runs=len(passing),
            p95_time_to_success_s=nearest_rank([r.wall_s for r in passing], 0.95) if passing else None,
            invalid_state_changing=sum(r.invalid_state_changing_calls for r in rs),
            invalid_other=sum(r.invalid_other_calls for r in rs),
            malformed_text_calls=sum(r.malformed_text_calls for r in rs),
            runs_with_malformed_text=sum(r.malformed_text_calls > 0 for r in rs),
            claimed_done_without_state_change=sum(r.claimed_done_without_state_change for r in rs),
            context_exceeded=sum(r.status == "context_exceeded" for r in rs),
            cold_starts=sum(r.cold_start is True for r in rs),
            quiet_flagged=sum(not r.quiet.passed for r in rs),
            backend_private_mb=_span(rs, "backend_private_mb"),
            backend_rss_mb=_span(rs, "backend_rss_mb"),
        ))
    return out


def _span(rs: list[Run], metric: str) -> tuple[float, float] | None:
    start = rs[0].telemetry.readings.get(f"start_{metric}")
    end = rs[-1].telemetry.readings.get(f"end_{metric}")
    if isinstance(start, Measured) and isinstance(end, Measured):
        return (start.value, end.value)
    return None


def _rate(k: int, n: int) -> str:
    return f"{k}/{n} tasks {k / n:4.0%}" if n else "0/0 tasks"


def _p(value: float | None, n: int) -> str:
    return f"{value:.2f} (n={n})" if value is not None else f"- (n={n})"


def format_report(summaries: Sequence[ArmSummary]) -> str:
    lines = []
    for condition_id in sorted({s.condition_id for s in summaries}):
        rows = [s for s in summaries if s.condition_id == condition_id]
        w = max(16, *(len(s.arm_id) + 2 for s in rows))
        lines.append(f"condition: {condition_id}")
        lines.append(
            f"  {'arm':<{w}}{'success':>18}{'clean success':>18}{'p95 wall s':>16}{'p95 to-success s':>19}"
            f"{'invalid st/oth':>16}{'malformed txt':>16}{'claimed done':>14}{'ctx exceeded':>14}"
        )
        for s in rows:
            lines.append(
                f"  {s.arm_id:<{w}}"
                f"{_rate(s.task_successes, s.tasks):>18}"
                f"{_rate(s.task_clean_successes, s.tasks):>18}"
                f"{_p(s.p95_wall_s, s.runs):>16}"
                f"{_p(s.p95_time_to_success_s, s.successful_runs):>19}"
                f"{f'{s.invalid_state_changing}/{s.invalid_other}':>16}"
                f"{f'{s.malformed_text_calls} ({s.runs_with_malformed_text} runs)':>16}"
                f"{s.claimed_done_without_state_change:>14}"
                f"{s.context_exceeded:>14}"
            )

        tasks = sorted({t for s in rows for t in s.per_task})
        lines.append("")
        lines.append(f"  {'per task (one symbol per repeat)':<34}" + "".join(f"{s.arm_id:>{w}}" for s in rows))
        for t in tasks:
            lines.append(f"  {t:<34}" + "".join(f"{s.per_task.get(t, '-'):>{w}}" for s in rows))

        lines.append("")
        for s in rows:
            statuses = ", ".join(f"{k} {v}" for k, v in sorted(s.statuses.items()))
            mem = (
                f"private {s.backend_private_mb[0]:.0f} -> {s.backend_private_mb[1]:.0f} MB"
                if s.backend_private_mb else "private unavailable"
            )
            rss = f"working set {s.backend_rss_mb[0]:.0f} -> {s.backend_rss_mb[1]:.0f} MB" if s.backend_rss_mb else ""
            inconsistent = f" | INCONSISTENT repeats: {', '.join(s.inconsistent_tasks)}" if s.inconsistent_tasks else ""
            lines.append(
                f"  {s.arm_id} ({s.model_id}, {s.mode}): {s.runs} runs | statuses: {statuses} | "
                f"cold starts {s.cold_starts} | quiet-check flagged {s.quiet_flagged}/{s.runs} | "
                f"Ollama {mem}, {rss}{inconsistent}"
            )
        lines.append("")
    lines.append("success: the state grader passes on every repeat of the task. clean success: also no tool call")
    lines.append("written as text and no context_exceeded. Symbols: P clean pass, p pass but not clean, F fail.")
    lines.append("Repeats are timing samples (temperature 0, fixed seed); p95 uses nearest rank, n = runs.")
    lines.append("invalid st/oth = invalid state-changing / read-only+unknown-tool calls.")
    return "\n".join(lines)
