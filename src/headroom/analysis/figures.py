"""Release figures, regenerated from the raw run logs (results/) with the committed rules.

Each figure is computed by the same code the gates use (evaluate_gate_a, condition_effects), so a
figure never shows a number the gate commands would not print. Needs the optional plots group:
`uv run --group plots headroom figures`.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from headroom.analysis.gate_a import analysed_runs, evaluate_gate_a
from headroom.runner.log import read_jsonl
from headroom.runner.report import condition_effects
from headroom.schemas.gate import load_gate_a
from headroom.schemas.run import Run


@dataclass(frozen=True)
class GateTest:
    label: str
    experiment_id: str
    session_id: str
    rule: str


GATE_TESTS = (
    GateTest("Gate A, emulated pressure (tuning tasks)", "week2_conditions", "20261124T184111Z-844a18",
             "experiments/gate_a.toml"),
    GateTest("Gate A, real workloads (tuning tasks)", "week3_real_workloads", "20261127T045705Z-9203a0",
             "experiments/gate_a_real.toml"),
    GateTest("Gate C, real workloads (held-out tasks)", "gate_c_heldout", "20261127T130044Z-31f87f",
             "experiments/gate_a_real.toml"),
)
ARMS = ("qwen-direct", "qwen-validated", "qwen-validated-2k", "qwen-plan-act")
ARM_LABEL = {"qwen-direct": "direct", "qwen-validated": "validated", "qwen-validated-2k": "validated, ctx 2048",
             "qwen-plan-act": "plan-act"}
# Reference categorical palette, slots 1-4 in fixed order (validated: adjacent CVD dE >= 8, light mode).
ARM_COLOR = {"qwen-direct": "#257de0", "qwen-validated": "#f47230", "qwen-validated-2k": "#00b894",
             "qwen-plan-act": "#f2a600"}
TEXT, TEXT_2, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"
MARKER = {"qwen-direct": "o", "qwen-validated": "s", "qwen-validated-2k": "D", "qwen-plan-act": "^"}


def _runs(results: Path, t: GateTest) -> list[Run]:
    runs = read_jsonl(results / t.experiment_id / "runs.jsonl", Run)[0]
    return [r for r in runs if r.session_id == t.session_id]


def gap_rows(results: Path, root: Path = Path(".")) -> list[tuple[str, str, float, tuple[float, float]]]:
    """(test label, condition, gap, ci) for every scoped condition of every gate test."""
    rows = []
    for t in GATE_TESTS:
        res = evaluate_gate_a(_runs(results, t), load_gate_a(root / t.rule))
        for g in res.conditions:
            if g.in_scope and g.gap is not None:
                rows.append((t.label, g.condition_id, g.gap, g.ci))
    return rows


def slowdown_rows(results: Path) -> list[tuple[str, str, str, float, tuple[float, float]]]:
    """(test label, condition, arm, latency change vs baseline, ci), non-baseline conditions only."""
    rows = []
    for t in GATE_TESTS:
        kept, _ = analysed_runs(_runs(results, t))
        for e in condition_effects(kept):
            if e.arm_id in ARMS and e.latency_change is not None:
                rows.append((t.label, e.condition_id, e.arm_id, e.latency_change, e.ci))
    return rows


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_gaps(rows, out: Path, min_gap: float = 0.10) -> Path:
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(r[0] for r in rows))
    fig, axes = plt.subplots(len(labels), 1, figsize=(7.5, 1.2 + 0.42 * len(rows)), sharex=True,
                             gridspec_kw={"height_ratios": [sum(r[0] == lab for r in rows) for lab in labels]})
    fig.patch.set_facecolor(SURFACE)
    for ax, lab in zip(axes, labels):
        sub = [r for r in rows if r[0] == lab][::-1]
        _style(ax)
        for i, (_, cond, gap, ci) in enumerate(sub):
            ax.plot([ci[0] * 100, ci[1] * 100], [i, i], color=ARM_COLOR["qwen-direct"], linewidth=2,
                    solid_capstyle="round")
            ax.plot(gap * 100, i, "o", color=ARM_COLOR["qwen-direct"], markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2)
            ax.annotate(f"{gap:+.1%}", (max(gap, ci[1]) * 100, i), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_yticks(range(len(sub)), [r[1] for r in sub], color=TEXT, fontsize=9)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.axvline(min_gap * 100, color=TEXT_2, linestyle=(0, (4, 3)), linewidth=1)
        ax.axvline(0, color=TEXT_2, linewidth=0.8)
        ax.set_title(lab, loc="left", fontsize=10, color=TEXT, pad=4)
    axes[0].annotate("pass threshold (10%)", (min_gap * 100, len([r for r in rows if r[0] == labels[0]]) - 0.3),
                     xytext=(4, 2), textcoords="offset points", fontsize=8, color=TEXT_2, va="bottom")
    axes[-1].set_xlabel("oracle gap: best fixed config vs best config per condition, % (95% CI over tasks)",
                        fontsize=9, color=TEXT_2)
    axes[-1].set_xlim(-3, 16)
    fig.suptitle("Adapting per condition never gains 10% in two conditions", x=0.02, ha="left",
                 fontsize=12, color=TEXT, fontweight="bold")
    fig.tight_layout()
    path = out / "oracle_gap.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_slowdowns(rows, out: Path) -> Path:
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys((r[0], r[1]) for r in rows))
    fig, ax = plt.subplots(figsize=(7.5, 1.4 + 0.5 * len(groups)))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    offsets = {a: (k - 1.5) * 0.16 for k, a in enumerate(ARMS)}
    ticks = []
    for i, (lab, cond) in enumerate(groups[::-1]):
        ticks.append(f"{cond}  ({'held-out' if 'held-out' in lab else 'tuning'})")
        for arm in ARMS:
            hit = [r for r in rows if r[0] == lab and r[1] == cond and r[2] == arm]
            if not hit:
                continue
            _, _, _, ch, ci = hit[0]
            y = i + offsets[arm]
            ax.plot([ci[0] * 100, ci[1] * 100], [y, y], color=ARM_COLOR[arm], linewidth=2)
            ax.plot(ch * 100, y, MARKER[arm], color=ARM_COLOR[arm], markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=ARM_LABEL[arm] if i == 0 else None)
    ax.set_yticks(range(len(groups)), ticks, color=TEXT, fontsize=9)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlim(-2, 170)
    ax.set_xticks([0, 5, 10, 15, 20, 50, 100, 150], ["0", "5", "10", "15", "20", "50", "100", "150"])
    ax.set_xlabel("wall time vs the same config at baseline, % (95% CI over tasks; linear to 20%, log above)",
                  fontsize=9, color=TEXT_2)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("Each condition slows every config by about the same amount", x=0.02, ha="left",
                 fontsize=12, color=TEXT, fontweight="bold")
    fig.tight_layout()
    path = out / "condition_slowdown.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_figures(results: Path, out: Path, root: Path = Path(".")) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    gaps, slows = gap_rows(results, root), slowdown_rows(results)
    paths = [plot_gaps(gaps, out), plot_slowdowns(slows, out)]
    table = out / "figures_data.csv"
    with open(table, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["figure", "test", "condition", "arm", "estimate", "ci_low", "ci_high"])
        for lab, cond, g, ci in gaps:
            w.writerow(["oracle_gap", lab, cond, "best fixed vs oracle", f"{g:.4f}", f"{ci[0]:.4f}", f"{ci[1]:.4f}"])
        for lab, cond, arm, ch, ci in slows:
            w.writerow(["condition_slowdown", lab, cond, arm, f"{ch:.4f}", f"{ci[0]:.4f}", f"{ci[1]:.4f}"])
    return [*paths, table]
