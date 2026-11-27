"""Gate C, applied exactly as pre-registered in experiments/gate_c.toml: do the tuning-half findings
hold on the held-out tasks?

Each claim is a set of checks. A check is "pass", "fail", or "missing" when its data is absent (a
condition block that could not run, or a scoped arm without analysed runs for a planned task). A claim
is NOT REPLICATED when any check fails, INCONCLUSIVE when none fails but one is missing, otherwise
REPLICATED. Verdict: PASS when every claim replicates, FAIL when any is not replicated, otherwise
INCONCLUSIVE.

Runs flagged cache_reused or suspend_suspected are excluded first (the same exclusions as Gate A).
Timing comparisons are paired over tasks: median over tasks of (x / y per-task median) - 1, with a
percentile bootstrap interval that resamples tasks.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from headroom.analysis.gate_a import Cell, Planned, _cells, analysed_runs, evaluate_gate_a
from headroom.schemas.gate import GateARule, GateCRule
from headroom.schemas.run import Run
from headroom.stats import bootstrap_ci, median

Status = Literal["pass", "fail", "missing"]


@dataclass(frozen=True)
class Check:
    label: str
    status: Status
    detail: str


@dataclass(frozen=True)
class ClaimResult:
    name: str
    statement: str
    checks: tuple[Check, ...]

    @property
    def status(self) -> Literal["REPLICATED", "NOT REPLICATED", "INCONCLUSIVE"]:
        states = {c.status for c in self.checks}
        if "fail" in states:
            return "NOT REPLICATED"
        return "INCONCLUSIVE" if "missing" in states or not states else "REPLICATED"


@dataclass(frozen=True)
class GateCResult:
    rule_id: str
    verdict: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    claims: tuple[ClaimResult, ...]
    analysed: int
    excluded: dict[str, int]
    missing: dict[tuple[str, str], str] = field(default_factory=dict)  # (arm, condition) -> why


def evaluate_gate_c(
    runs: Sequence[Run],
    rule: GateCRule,
    gate_a_rule: GateARule,
    *,
    planned: Planned | None = None,
    failures: dict[str, str] | None = None,
) -> GateCResult:
    kept, excluded = analysed_runs(runs)
    cells = _cells([r for r in kept if r.arm_id in rule.arms])
    missing: dict[tuple[str, str], str] = {}
    for c in rule.conditions:
        for a in rule.arms:
            if failures and c in failures:
                missing[(a, c)] = f"condition FAILED: {failures[c]}"
            elif (a, c) not in cells:
                missing[(a, c)] = "no analysed runs"
            elif planned is not None:
                gone = {t for (pa, pc, t) in planned if pa == a and pc == c} - set(cells[(a, c)].task_medians)
                if gone:
                    missing[(a, c)] = f"{len(gone)} planned task(s) have no analysed runs"

    def cell(a: str, c: str) -> Cell | None:
        return None if (a, c) in missing else cells[(a, c)]

    def paired(x: Cell, y: Cell) -> tuple[float, tuple[float, float], int]:
        shared = sorted(set(x.task_medians) & set(y.task_medians))
        ratios = [x.task_medians[t] / y.task_medians[t] for t in shared]
        stat = lambda xs: median(xs) - 1  # noqa: E731
        ci = bootstrap_ci(ratios, stat, level=rule.ci_level, resamples=rule.bootstrap_resamples,
                          seed=rule.bootstrap_seed)
        return stat(ratios), ci, len(shared)

    def fmt(est: float, ci: tuple[float, float], n: int) -> str:
        return f"{est:+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}] (n={n})"

    workloads = [c for c in rule.conditions if c != rule.baseline]
    cl = rule.claims
    claims = []

    # 1. no adaptation gain: the Gate A rule, unchanged.
    ga = evaluate_gate_a(kept, gate_a_rule, planned=planned, failures=failures)
    gaps = ", ".join(
        f"{g.condition_id} {g.gap:+.1%}" if g.gap is not None else f"{g.condition_id} FAILED"
        for g in ga.conditions if g.in_scope
    )
    status: Status = {"FAIL": "pass", "PASS": "fail", "INCONCLUSIVE": "missing"}[ga.verdict]
    claims.append(ClaimResult("no_adaptation_gain", cl.no_adaptation_gain.statement, (Check(
        gate_a_rule.rule_id, status, f"verdict {ga.verdict}; best fixed {ga.best_fixed}; oracle gap: {gaps}",
    ),)))

    # 2. success unchanged under load.
    s = cl.success_unchanged_under_load
    checks = []
    for a in rule.arms:
        base = cell(a, rule.baseline)
        for c in workloads:
            here = cell(a, c)
            label = f"{a} {c}"
            if base is None or here is None:
                checks.append(Check(label, "missing", missing.get((a, c)) or missing.get((a, rule.baseline), "")))
                continue
            diff = here.task_success - base.task_success
            checks.append(Check(
                label, "pass" if abs(diff) <= s.max_task_difference else "fail",
                f"{here.task_success}/{here.tasks} vs baseline {base.task_success}/{base.tasks} ({diff:+d})",
            ))
    claims.append(ClaimResult("success_unchanged_under_load", s.statement, tuple(checks)))

    # 3. plan_act slower than direct.
    p = cl.plan_act_slower
    checks = []
    for c in rule.conditions:
        x, y = cell(p.slower, c), cell(p.faster, c)
        if x is None or y is None:
            checks.append(Check(c, "missing", missing.get((p.slower, c)) or missing.get((p.faster, c), "")))
            continue
        est, ci, n = paired(x, y)
        checks.append(Check(c, "pass" if ci[0] > p.min_ci_lower else "fail", f"{p.slower} vs {p.faster} {fmt(est, ci, n)}"))
    claims.append(ClaimResult("plan_act_slower", p.statement, tuple(checks)))

    # 4. direct and validated equivalent.
    e = cl.direct_validated_equivalent
    checks = []
    for c in rule.conditions:
        x, y = cell(e.b, c), cell(e.a, c)
        if x is None or y is None:
            checks.append(Check(c, "missing", missing.get((e.a, c)) or missing.get((e.b, c), "")))
            continue
        est, ci, n = paired(x, y)
        inside = -e.margin < ci[0] and ci[1] < e.margin
        checks.append(Check(c, "pass" if inside else "fail", f"{e.b} vs {e.a} {fmt(est, ci, n)}, margin +-{e.margin:.0%}"))
    claims.append(ClaimResult("direct_validated_equivalent", e.statement, tuple(checks)))

    # 5. workloads slow every arm.
    w = cl.workloads_slow_every_arm
    checks = []
    for a in rule.arms:
        base = cell(a, rule.baseline)
        for c in workloads:
            here = cell(a, c)
            label = f"{a} {c}"
            if base is None or here is None:
                checks.append(Check(label, "missing", missing.get((a, c)) or missing.get((a, rule.baseline), "")))
                continue
            est, ci, n = paired(here, base)
            checks.append(Check(label, "pass" if ci[0] > w.min_ci_lower else "fail", f"vs baseline {fmt(est, ci, n)}"))
    claims.append(ClaimResult("workloads_slow_every_arm", w.statement, tuple(checks)))

    states = {c.status for c in claims}
    verdict = "FAIL" if "NOT REPLICATED" in states else ("INCONCLUSIVE" if "INCONCLUSIVE" in states else "PASS")
    return GateCResult(rule.rule_id, verdict, tuple(claims), len(kept), dict(excluded), missing)


def format_gate_c(res: GateCResult) -> str:
    n_rep = sum(c.status == "REPLICATED" for c in res.claims)
    lines = [
        f"Gate C ({res.rule_id}): {res.verdict}: {n_rep} of {len(res.claims)} claims replicated on the held-out tasks",
        f"analysed runs: {res.analysed}; excluded: "
        + (", ".join(f"{k} {v}" for k, v in sorted(res.excluded.items())) or "none"),
    ]
    for (a, c), why in sorted(res.missing.items()):
        lines.append(f"missing: {a} {c}: {why}")
    for claim in res.claims:
        lines += ["", f"{claim.name}: {claim.status}", f"  {claim.statement}"]
        for ch in claim.checks:
            lines.append(f"  {ch.status:<8}{ch.label:<34}{ch.detail}")
    return "\n".join(lines)
