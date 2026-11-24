"""Gate A, applied exactly as pre-registered in experiments/gate_a.toml.

Only the rule's scope counts: its arms, across its conditions.

  figure(config, condition)  median over tasks of the task's median wall time over its repeats
  task success               tasks whose every repeat passes the state grader (checked separately)
  best_fixed                 highest task success summed over the scoped conditions; ties broken by the
                             lowest median, across those conditions, of the figure
  oracle(condition)          lowest figure among scoped configs whose task success there is not below
                             best_fixed's
  gap(condition)             paired median over tasks of (best_fixed / oracle per-task median) - 1,
                             percentile bootstrap interval resampling tasks
  pass                       gap >= min_gap and the interval excludes 0, in >= min_conditions conditions

A scoped condition is FAILED (never "no gap") when its block could not run (a recorded condition
event: injector failure, model failed to load) or when a scoped arm has no analysed runs for a
planned task there. Verdict: PASS, FAIL, or INCONCLUSIVE when FAILED conditions could have
changed a FAIL.

Runs flagged cache_reused or suspend_suspected are excluded before anything is computed.
Conditions outside the scope (e.g. vram-1g) are evaluated the same way with the scoped arms and
reported separately as a sensitivity check; arms outside the scope are not used here.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from headroom.runner.report import state_success
from headroom.schemas.gate import GateARule
from headroom.schemas.run import Run
from headroom.stats import bootstrap_ci, cv, median

Planned = set[tuple[str, str, str]]  # (arm_id, condition_id, task_id)


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
    in_scope: bool
    status: Literal["ok", "failed"]
    reason: str | None = None  # why FAILED
    best_fixed_figure: float | None = None
    oracle: str | None = None
    oracle_figure: float | None = None
    eligible: tuple[str, ...] = ()  # scoped configs whose task success here is not below best_fixed's
    tasks: int = 0
    gap: float | None = None
    ci: tuple[float, float] | None = None
    passes: bool = False
    noise_cv_median: float | None = None  # repeat-to-repeat spread inside this condition
    noise_cv_max: float | None = None


@dataclass(frozen=True)
class GateAResult:
    rule_id: str
    scope_arms: tuple[str, ...]
    best_fixed: str | None
    conditions: tuple[ConditionGap, ...]
    passing: tuple[str, ...]
    failed: dict[str, str]
    verdict: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    min_conditions: int
    cells: dict[tuple[str, str], Cell]  # (config, condition), scoped arms only
    analysed: int
    excluded: dict[str, int]
    out_of_scope_runs: int


def evaluate_gate_a(
    runs: Sequence[Run],
    rule: GateARule,
    *,
    planned: Planned | None = None,
    failures: dict[str, str] | None = None,
) -> GateAResult:
    """planned: what the session(s) planned, to detect missing runs. failures: recorded condition events."""
    kept, excluded = analysed_runs(runs)
    arms = list(rule.scope.arms)
    scoped = [r for r in kept if r.arm_id in arms]
    cells = _cells(scoped)
    failed = dict(failures or {})

    for c in rule.scope.conditions:
        if c in failed:
            continue
        for a in arms:
            if (a, c) not in cells:
                failed[c] = f"no analysed runs for {a}"
                break
            if planned is not None:
                missing = {t for (pa, pc, t) in planned if pa == a and pc == c} - set(cells[(a, c)].task_medians)
                if missing:
                    failed[c] = f"{a}: {len(missing)} planned task(s) have no analysed runs"
                    break

    ok = [c for c in rule.scope.conditions if c not in failed]
    best_fixed = None
    if ok:
        best_fixed = min(
            arms,
            key=lambda a: (
                -sum(cells[(a, c)].task_success for c in ok),
                median([cells[(a, c)].figure for c in ok]),
                a,
            ),
        )

    out = []
    for c in rule.scope.conditions:
        if c in failed or best_fixed is None:
            out.append(ConditionGap(condition_id=c, in_scope=True, status="failed",
                                    reason=failed.get(c, "no scoped condition ran")))
        else:
            out.append(_gap(c, True, arms, cells, best_fixed, rule))
    extra = sorted({cid for (_, cid) in cells} - set(rule.scope.conditions))
    for c in extra:
        if best_fixed is not None and all((a, c) in cells for a in arms):
            out.append(_gap(c, False, arms, cells, best_fixed, rule))
        else:
            out.append(ConditionGap(condition_id=c, in_scope=False, status="failed",
                                    reason="not every scoped arm has analysed runs here"))

    passing = tuple(g.condition_id for g in out if g.in_scope and g.passes)
    n_failed = sum(1 for g in out if g.in_scope and g.status == "failed")
    need = rule.pass_.min_conditions
    verdict = "PASS" if len(passing) >= need else ("INCONCLUSIVE" if len(passing) + n_failed >= need else "FAIL")
    return GateAResult(
        rule_id=rule.rule_id, scope_arms=tuple(arms), best_fixed=best_fixed, conditions=tuple(out),
        passing=passing, failed={g.condition_id: g.reason for g in out if g.status == "failed"},
        verdict=verdict, min_conditions=need, cells=cells, analysed=len(kept), excluded=dict(excluded),
        out_of_scope_runs=len(kept) - len(scoped),
    )


def _gap(c: str, in_scope: bool, arms: list[str], cells, best_fixed: str, rule: GateARule) -> ConditionGap:
    p = rule.pass_
    bf = cells[(best_fixed, c)]
    eligible = tuple(a for a in arms if (a, c) in cells and cells[(a, c)].task_success >= bf.task_success)
    oracle = min(eligible, key=lambda a: (cells[(a, c)].figure, a != best_fixed, a))
    oc = cells[(oracle, c)]
    shared = sorted(set(bf.task_medians) & set(oc.task_medians))
    ratios = [bf.task_medians[t] / oc.task_medians[t] for t in shared]
    stat = lambda xs: median(xs) - 1  # noqa: E731
    gap = stat(ratios)
    ci = bootstrap_ci(ratios, stat, level=p.ci_level, resamples=p.bootstrap_resamples, seed=p.bootstrap_seed)
    cvs = [v for a in arms if (a, c) in cells for v in cells[(a, c)].task_cv.values() if v is not None]
    return ConditionGap(
        condition_id=c, in_scope=in_scope, status="ok", best_fixed_figure=bf.figure, oracle=oracle,
        oracle_figure=oc.figure, eligible=eligible, tasks=len(shared), gap=gap, ci=ci,
        passes=gap >= p.min_gap and ci[0] > p.ci_must_exclude,
        noise_cv_median=median(cvs) if cvs else None, noise_cv_max=max(cvs) if cvs else None,
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


def format_gate_a(res: GateAResult) -> str:
    scoped = [g for g in res.conditions if g.in_scope]
    extra = [g for g in res.conditions if not g.in_scope]
    failed_note = (
        f"; FAILED conditions: {', '.join(g.condition_id for g in scoped if g.status == 'failed')}"
        if any(g.status == "failed" for g in scoped) else ""
    )
    lines = [
        f"Gate A ({res.rule_id}): {res.verdict}: {len(res.passing)} of {len(scoped)} scoped conditions have an "
        f"oracle gap >= 10% with the 95% interval excluding 0 (need {res.min_conditions})"
        + (f": {', '.join(res.passing)}" if res.passing else "") + failed_note,
        f"scope: {', '.join(res.scope_arms)}",
        f"analysed runs: {res.analysed} ({res.out_of_scope_runs} from arms outside the scope, not used); excluded: "
        + (", ".join(f"{k} {v}" for k, v in sorted(res.excluded.items())) or "none")
        + "; verified: no analysed run has cache_reused = true",
        f"best fixed config: {res.best_fixed or '- (no scoped condition ran)'}",
        "",
    ]
    conds = [g.condition_id for g in res.conditions]

    def table(title: str, cell_text) -> None:
        lines.append(title)
        lines.append(f"  {'config':<22}" + "".join(f"{c:>12}" for c in conds))
        for a in res.scope_arms:
            lines.append(f"  {a:<22}" + "".join(f"{cell_text(a, c):>12}" for c in conds))
        lines.append("")

    table(
        "success, checked separately (tasks whose every repeat passes, of tasks measured)",
        lambda a, c: f"{res.cells[(a, c)].task_success}/{res.cells[(a, c)].tasks}" if (a, c) in res.cells else "FAILED",
    )
    table(
        "condition figure: median over tasks of per-task median wall s",
        lambda a, c: f"{res.cells[(a, c)].figure:.2f}" if (a, c) in res.cells else "FAILED",
    )

    def gap_rows(title: str, rows: Sequence[ConditionGap]) -> None:
        if not rows:
            return
        lines.append(title)
        lines.append(
            f"  {'condition':<12}{'best fixed s':>13}  {'oracle':<22}{'oracle s':>9}{'gap [95% CI] (tasks)':>32}"
            f"{'pass':>6}{'noise CV med/max':>18}"
        )
        for g in rows:
            if g.status == "failed":
                lines.append(f"  {g.condition_id:<12}  FAILED: {g.reason}")
                continue
            noise = f"{g.noise_cv_median:.1%}/{g.noise_cv_max:.1%}" if g.noise_cv_median is not None else "-"
            lines.append(
                f"  {g.condition_id:<12}{g.best_fixed_figure:>13.2f}  {g.oracle:<22}{g.oracle_figure:>9.2f}"
                f"{f'{g.gap:+.1%} [{g.ci[0]:+.1%}, {g.ci[1]:+.1%}] (n={g.tasks})':>32}"
                f"{'yes' if g.passes else 'no':>6}{noise:>18}"
            )
        lines.append("")

    gap_rows("oracle gap per scoped condition", scoped)
    gap_rows("outside the scope (sensitivity check, reported separately, never counted)", extra)
    lines.append("noise CV = repeat-to-repeat spread of one task's wall time, over the scoped configs in that condition.")
    return "\n".join(lines)
