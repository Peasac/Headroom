"""Gate A, applied exactly as pre-registered in experiments/gate_a.toml (rule gate_a_v1).

  figure(config, condition)  median over tasks of the task's median wall time over its repeats
  task success               tasks whose every repeat passes the state grader (checked separately)
  best_fixed                 highest task success summed over the Gate A conditions; ties broken by the
                             lowest median, across those conditions, of the figure
  oracle(condition)          lowest figure among configs whose task success there is not below best_fixed's
  gap(condition)             paired median over tasks of (best_fixed / oracle per-task median) - 1,
                             percentile bootstrap interval resampling tasks
  pass                       gap >= min_gap and the interval excludes 0, in >= min_conditions conditions

Runs flagged cache_reused or suspend_suspected are excluded before anything is computed.
Conditions with gate_a = false (dose-response levels) are evaluated the same way but reported
separately and never counted.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from headroom.runner.report import state_success
from headroom.schemas.gate import GateARule
from headroom.schemas.run import Run
from headroom.stats import bootstrap_ci, cv, median


def exclusion_reasons(run: Run) -> list[str]:
    reasons = []
    if run.cache_reused is True:
        reasons.append("cache_reused")
    if run.suspend_suspected is True:
        reasons.append("suspended")
    return reasons


def analysed_runs(runs: Sequence[Run]) -> tuple[list[Run], Counter]:
    kept, excluded = [], Counter()
    for r in runs:
        reasons = exclusion_reasons(r)
        if reasons:
            excluded.update(reasons)
        else:
            kept.append(r)
    if any(r.cache_reused is True for r in kept):  # the guarantee the analysis rests on
        raise AssertionError("an analysed run has cache_reused = true")
    return kept, excluded


@dataclass(frozen=True)
class Cell:
    """One config under one condition."""

    task_medians: dict[str, float]
    task_success: int
    tasks: int
    figure: float
    task_cv: dict[str, float | None]


@dataclass(frozen=True)
class ConditionGap:
    condition_id: str
    counts_for_gate: bool
    best_fixed_figure: float
    oracle: str
    oracle_figure: float
    eligible: tuple[str, ...]  # configs whose task success here is not below best_fixed's
    tasks: int
    gap: float
    ci: tuple[float, float]
    passes: bool
    noise_cv_median: float | None  # repeat-to-repeat spread inside this condition, over all (config, task)
    noise_cv_max: float | None


@dataclass(frozen=True)
class GateAResult:
    rule_id: str
    best_fixed: str
    conditions: tuple[ConditionGap, ...]
    passing: tuple[str, ...]  # Gate A conditions that pass
    passed: bool
    min_conditions: int
    cells: dict[tuple[str, str], Cell]  # (config, condition)
    analysed: int
    excluded: dict[str, int]


def evaluate_gate_a(runs: Sequence[Run], rule: GateARule) -> GateAResult:
    kept, excluded = analysed_runs(runs)
    cells = _cells(kept)
    configs = sorted({a for a, _ in cells})
    counts = _gate_flags(kept)
    conditions = sorted(counts)
    gate_conditions = [c for c in conditions if counts[c]]

    complete = [a for a in configs if all((a, c) in cells for c in gate_conditions)]
    if not complete or not gate_conditions:
        raise ValueError("need at least one config measured in every Gate A condition")
    best_fixed = min(
        complete,
        key=lambda a: (
            -sum(cells[(a, c)].task_success for c in gate_conditions),
            median([cells[(a, c)].figure for c in gate_conditions]),
            a,
        ),
    )

    p = rule.pass_
    out = []
    for c in conditions:
        if (best_fixed, c) not in cells:
            continue
        bf = cells[(best_fixed, c)]
        eligible = tuple(a for a in configs if (a, c) in cells and cells[(a, c)].task_success >= bf.task_success)
        oracle = min(eligible, key=lambda a: (cells[(a, c)].figure, a != best_fixed, a))
        oc = cells[(oracle, c)]
        shared = sorted(set(bf.task_medians) & set(oc.task_medians))
        ratios = [bf.task_medians[t] / oc.task_medians[t] for t in shared]
        stat = lambda xs: median(xs) - 1  # noqa: E731
        gap = stat(ratios)
        ci = bootstrap_ci(ratios, stat, level=p.ci_level, resamples=p.bootstrap_resamples, seed=p.bootstrap_seed)
        cvs = [v for a in configs if (a, c) in cells for v in cells[(a, c)].task_cv.values() if v is not None]
        out.append(ConditionGap(
            condition_id=c, counts_for_gate=counts[c], best_fixed_figure=bf.figure, oracle=oracle,
            oracle_figure=oc.figure, eligible=eligible, tasks=len(shared), gap=gap, ci=ci,
            passes=gap >= p.min_gap and ci[0] > p.ci_must_exclude,
            noise_cv_median=median(cvs) if cvs else None, noise_cv_max=max(cvs) if cvs else None,
        ))
    passing = tuple(g.condition_id for g in out if g.counts_for_gate and g.passes)
    return GateAResult(
        rule_id=rule.rule_id, best_fixed=best_fixed, conditions=tuple(out), passing=passing,
        passed=len(passing) >= p.min_conditions, min_conditions=p.min_conditions, cells=cells,
        analysed=len(kept), excluded=dict(excluded),
    )


def _cells(runs: Sequence[Run]) -> dict[tuple[str, str], Cell]:
    grouped: dict[tuple[str, str], dict[str, list[Run]]] = {}
    for r in runs:
        grouped.setdefault((r.arm_id, r.condition_id), {}).setdefault(r.task_id, []).append(r)
    cells = {}
    for key, by_task in grouped.items():
        medians = {t: median([r.wall_s for r in rs]) for t, rs in by_task.items()}
        cells[key] = Cell(
            task_medians=medians,
            task_success=sum(all(state_success(r) for r in rs) for rs in by_task.values()),
            tasks=len(by_task),
            figure=median(list(medians.values())),
            task_cv={t: cv([r.wall_s for r in rs]) for t, rs in by_task.items()},
        )
    return cells


def _gate_flags(runs: Sequence[Run]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for r in runs:
        flags.setdefault(r.condition_id, r.condition.condition.gate_a if r.condition else True)
    return flags


def format_gate_a(res: GateAResult) -> str:
    gate = [g for g in res.conditions if g.counts_for_gate]
    extra = [g for g in res.conditions if not g.counts_for_gate]
    lines = [
        f"Gate A ({res.rule_id}): {'PASS' if res.passed else 'FAIL'}: {len(res.passing)} of {len(gate)} Gate A "
        f"conditions have an oracle gap >= 10% with the 95% interval excluding 0 (need {res.min_conditions})"
        + (f": {', '.join(res.passing)}" if res.passing else ""),
        f"analysed runs: {res.analysed}; excluded: "
        + (", ".join(f"{k} {v}" for k, v in sorted(res.excluded.items())) or "none")
        + "; verified: no analysed run has cache_reused = true",
        f"best fixed config: {res.best_fixed}",
        "",
    ]
    configs = sorted({a for a, _ in res.cells})
    conds = [g.condition_id for g in res.conditions]

    def table(title: str, cell_text) -> None:
        lines.append(title)
        lines.append(f"  {'config':<26}" + "".join(f"{c:>12}" for c in conds))
        for a in configs:
            lines.append(f"  {a:<26}" + "".join(f"{cell_text(a, c):>12}" for c in conds))
        lines.append("")

    table(
        "success, checked separately (tasks whose every repeat passes, of tasks measured)",
        lambda a, c: f"{res.cells[(a, c)].task_success}/{res.cells[(a, c)].tasks}" if (a, c) in res.cells else "-",
    )
    table(
        "condition figure: median over tasks of per-task median wall s",
        lambda a, c: f"{res.cells[(a, c)].figure:.2f}" if (a, c) in res.cells else "-",
    )

    def gap_rows(title: str, rows: Sequence[ConditionGap]) -> None:
        if not rows:
            return
        lines.append(title)
        lines.append(
            f"  {'condition':<12}{'best fixed s':>13}  {'oracle':<26}{'oracle s':>9}{'gap [95% CI] (tasks)':>32}"
            f"{'pass':>6}{'noise CV med/max':>18}"
        )
        for g in rows:
            noise = f"{g.noise_cv_median:.1%}/{g.noise_cv_max:.1%}" if g.noise_cv_median is not None else "-"
            lines.append(
                f"  {g.condition_id:<12}{g.best_fixed_figure:>13.2f}  {g.oracle:<26}{g.oracle_figure:>9.2f}"
                f"{f'{g.gap:+.1%} [{g.ci[0]:+.1%}, {g.ci[1]:+.1%}] (n={g.tasks})':>32}"
                f"{'yes' if g.passes else 'no':>6}{noise:>18}"
            )
        lines.append("")

    gap_rows("oracle gap per Gate A condition", gate)
    gap_rows("dose-response (reported separately; does not count toward Gate A)", extra)
    lines.append("noise CV = repeat-to-repeat spread of one task's wall time, over all configs in that condition.")
    return "\n".join(lines)
