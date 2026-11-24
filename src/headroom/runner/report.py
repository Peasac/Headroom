"""Per-condition, per-arm summary of logged runs.

Success is decided by the state grader alone; clean success also requires zero tool calls written
as text and no context_exceeded. Temperature is 0 with a fixed seed, so repeats are timing samples:
success is counted per task (a task passes when every repeat passes).

Timing: each task's median wall time over its repeats, then the median over tasks, with a bootstrap
95% interval that resamples tasks. Noise: the spread of one task's repeats (CV), an A/A estimate
from splitting each task's repeats into two halves, and paired differences between arms.
"""

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from headroom.schemas.common import Measured
from headroom.schemas.run import Run
from headroom.stats import bootstrap_ci, cv, median, nearest_rank

AA_SPLITS = 2000


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
    context_cap: int
    tasks: int
    task_successes: int  # every repeat passes the state grader
    task_clean_successes: int  # every repeat passes cleanly
    inconsistent_tasks: tuple[str, ...]  # repeats disagree on success (counted as not passing)
    per_task: dict[str, str]  # task_id -> one symbol per repeat, in repeat order
    task_walls: dict[str, tuple[float, ...]]  # task_id -> wall time per repeat, in repeat order
    task_median_wall_s: dict[str, float]
    task_cv: dict[str, float | None]  # repeat-to-repeat spread of one task
    median_wall_s: float  # median over tasks of the per-task medians
    median_wall_ci: tuple[float, float]  # bootstrap 95% over tasks
    aa_gap_p95: float | None  # 95th percentile |A/B - 1| when each task's repeats are split in two
    runs: int
    statuses: dict[str, int]
    p95_wall_s: float  # over all runs (secondary)
    peak_private_mb: float | None  # highest Ollama private memory seen in any run
    invalid_state_changing: int
    invalid_other: int
    malformed_text_calls: int
    runs_with_malformed_text: int
    claimed_done_without_state_change: int
    context_exceeded: int
    cold_starts: int
    quiet_flagged: int
    cache_reused: int  # runs whose first call reused prompt cache from an earlier run
    cache_unknown: int  # runs that do not record it (logged before the field existed)


@dataclass(frozen=True)
class PairedDifference:
    """Arm b against arm a on the same tasks: median over tasks of (b's task median / a's) - 1."""

    condition_id: str
    arm_a: str
    arm_b: str
    tasks: int
    median_ratio_minus_1: float
    ci: tuple[float, float]


def summarize_runs(runs: Sequence[Run]) -> list[ArmSummary]:
    groups: dict[tuple[str, str], list[Run]] = {}
    for r in runs:
        groups.setdefault((r.condition_id, r.arm_id), []).append(r)
    return [_summarize(cid, aid, rs) for (cid, aid), rs in sorted(groups.items())]


def _summarize(condition_id: str, arm_id: str, rs: list[Run]) -> ArmSummary:
    by_task: dict[str, list[Run]] = {}
    for r in sorted(rs, key=lambda r: (r.task_id, r.repeat, r.session_id)):
        by_task.setdefault(r.task_id, []).append(r)
    walls = {t: tuple(r.wall_s for r in group) for t, group in by_task.items()}
    task_medians = {t: median(w) for t, w in walls.items()}
    privates = [
        r.telemetry.readings["peak_backend_private_mb"].value for r in rs
        if isinstance(r.telemetry.readings.get("peak_backend_private_mb"), Measured)
    ]
    config = rs[0].config
    return ArmSummary(
        condition_id=condition_id,
        arm_id=arm_id,
        model_id=config.model_id,
        mode=config.mode,
        context_cap=config.context_cap,
        tasks=len(by_task),
        task_successes=sum(all(state_success(r) for r in t) for t in by_task.values()),
        task_clean_successes=sum(all(clean_success(r) for r in t) for t in by_task.values()),
        inconsistent_tasks=tuple(tid for tid, t in by_task.items() if len({state_success(r) for r in t}) > 1),
        per_task={tid: "".join(run_symbol(r) for r in t) for tid, t in by_task.items()},
        task_walls=walls,
        task_median_wall_s=task_medians,
        task_cv={t: cv(w) for t, w in walls.items()},
        median_wall_s=median(list(task_medians.values())),
        median_wall_ci=bootstrap_ci(list(task_medians.values())),
        aa_gap_p95=aa_gap_p95(walls),
        runs=len(rs),
        statuses=dict(Counter(r.status for r in rs)),
        p95_wall_s=nearest_rank([r.wall_s for r in rs], 0.95),
        peak_private_mb=max(privates) if privates else None,
        invalid_state_changing=sum(r.invalid_state_changing_calls for r in rs),
        invalid_other=sum(r.invalid_other_calls for r in rs),
        malformed_text_calls=sum(r.malformed_text_calls for r in rs),
        runs_with_malformed_text=sum(r.malformed_text_calls > 0 for r in rs),
        claimed_done_without_state_change=sum(r.claimed_done_without_state_change for r in rs),
        context_exceeded=sum(r.status == "context_exceeded" for r in rs),
        cold_starts=sum(r.cold_start is True for r in rs),
        quiet_flagged=sum(not r.quiet.passed for r in rs),
        cache_reused=sum(r.cache_reused is True for r in rs),
        cache_unknown=sum(r.cache_reused is None for r in rs),
    )


def aa_gap_p95(walls: dict[str, tuple[float, ...]], splits: int = AA_SPLITS, seed: int = 0) -> float | None:
    """A/A noise: split each task's repeats at random into two halves, compute the arm statistic
    (median over tasks of per-task medians) for each half, and record |A/B - 1|. Returns the 95th
    percentile over random splits. Each half has only about half the repeats, so this is a
    conservative (high) estimate of the noise in a comparison with all repeats on both sides."""
    if any(len(w) < 2 for w in walls.values()) or not walls:
        return None
    rng = random.Random(seed)
    gaps = []
    for _ in range(splits):
        a_meds, b_meds = [], []
        for w in walls.values():
            shuffled = list(w)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            a_meds.append(median(shuffled[:half]))
            b_meds.append(median(shuffled[half:]))
        gaps.append(abs(median(a_meds) / median(b_meds) - 1))
    return nearest_rank(gaps, 0.95)


def paired_differences(summaries: Sequence[ArmSummary]) -> list[PairedDifference]:
    out = []
    for condition_id in sorted({s.condition_id for s in summaries}):
        rows = [s for s in summaries if s.condition_id == condition_id]
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                tasks = sorted(set(a.task_median_wall_s) & set(b.task_median_wall_s))
                if not tasks:
                    continue
                ratios = [b.task_median_wall_s[t] / a.task_median_wall_s[t] for t in tasks]
                stat = lambda xs: median(xs) - 1  # noqa: E731
                out.append(PairedDifference(
                    condition_id=condition_id, arm_a=a.arm_id, arm_b=b.arm_id, tasks=len(tasks),
                    median_ratio_minus_1=stat(ratios), ci=bootstrap_ci(ratios, stat),
                ))
    return out


def _rate(k: int, n: int) -> str:
    return f"{k}/{n} tasks {k / n:4.0%}" if n else "0/0 tasks"


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "-"


def format_report(summaries: Sequence[ArmSummary]) -> str:
    lines = []
    for condition_id in sorted({s.condition_id for s in summaries}):
        rows = [s for s in summaries if s.condition_id == condition_id]
        w = max(16, *(len(s.arm_id) + 2 for s in rows))
        lines.append(f"condition: {condition_id}")
        lines.append(
            f"  {'arm':<{w}}{'success':>18}{'clean success':>18}{'median wall s [95% CI] (tasks)':>34}"
            f"{'peak Ollama private MB':>24}{'invalid st/oth':>16}{'malformed txt':>16}{'claimed done':>14}"
            f"{'ctx exceeded':>14}"
        )
        for s in rows:
            timing = f"{s.median_wall_s:.2f} [{s.median_wall_ci[0]:.2f}, {s.median_wall_ci[1]:.2f}] (n={s.tasks})"
            mem = f"{s.peak_private_mb:.0f}" if s.peak_private_mb is not None else "unavailable"
            lines.append(
                f"  {s.arm_id:<{w}}"
                f"{_rate(s.task_successes, s.tasks):>18}"
                f"{_rate(s.task_clean_successes, s.tasks):>18}"
                f"{timing:>34}"
                f"{mem:>24}"
                f"{f'{s.invalid_state_changing}/{s.invalid_other}':>16}"
                f"{f'{s.malformed_text_calls} ({s.runs_with_malformed_text} runs)':>16}"
                f"{s.claimed_done_without_state_change:>14}"
                f"{s.context_exceeded:>14}"
            )

        tasks = sorted({t for s in rows for t in s.per_task})
        cell = max(w, 24)
        lines.append("")
        lines.append(
            f"  {'per task: results, median wall s, CV':<34}" + "".join(f"{s.arm_id:>{cell}}" for s in rows)
        )
        for t in tasks:
            cells = []
            for s in rows:
                if t not in s.per_task:
                    cells.append(f"{'-':>{cell}}")
                    continue
                cells.append(f"{f'{s.per_task[t]} {s.task_median_wall_s[t]:5.2f}s {_pct(s.task_cv[t]):>4}':>{cell}}")
            lines.append(f"  {t:<34}" + "".join(cells))

        lines.append("")
        lines.append("  noise")
        for s in rows:
            cvs = {t: c for t, c in s.task_cv.items() if c is not None}
            if cvs:
                worst = max(cvs, key=cvs.get)
                spread = (
                    f"repeat-to-repeat CV of one task: median {_pct(median(list(cvs.values())))}, "
                    f"max {_pct(cvs[worst])} ({worst})"
                )
            else:
                spread = "repeat-to-repeat CV: needs 2+ repeats"
            lines.append(f"  {s.arm_id}: {spread}; A/A split gap, 95th pct: {_pct(s.aa_gap_p95)}")
        for d in paired_differences(rows):
            lines.append(
                f"  {d.arm_b} vs {d.arm_a}: {d.median_ratio_minus_1:+.1%} "
                f"[{d.ci[0]:+.1%}, {d.ci[1]:+.1%}] (paired over {d.tasks} tasks)"
            )

        lines.append("")
        for s in rows:
            statuses = ", ".join(f"{k} {v}" for k, v in sorted(s.statuses.items()))
            inconsistent = f" | INCONSISTENT repeats: {', '.join(s.inconsistent_tasks)}" if s.inconsistent_tasks else ""
            cache = f"cache reused (first call) {s.cache_reused}/{s.runs - s.cache_unknown}"
            if s.cache_unknown:
                cache += f" ({s.cache_unknown} not recorded)"
            lines.append(
                f"  {s.arm_id} ({s.model_id}, {s.mode}, ctx {s.context_cap}): {s.runs} runs | statuses: {statuses} | "
                f"p95 wall {s.p95_wall_s:.2f}s | cold starts {s.cold_starts} | "
                f"quiet-check flagged {s.quiet_flagged}/{s.runs} | {cache}{inconsistent}"
            )
        lines.append("")
    lines.append("success: the state grader passes on every repeat of the task. clean success: also no tool call")
    lines.append("written as text and no context_exceeded. Symbols: P clean pass, p pass but not clean, F fail.")
    lines.append("Timing: per-task median wall over repeats; arm figure = median over tasks, bootstrap 95% CI over")
    lines.append("tasks. CV = stdev/mean of one task's repeats. A/A split gap: |A/B - 1| when each task's repeats are")
    lines.append("randomly halved (conservative). Paired: median over tasks of (b/a - 1), bootstrap 95% CI over tasks.")
    return "\n".join(lines)
