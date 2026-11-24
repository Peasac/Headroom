import asyncio
import json
from pathlib import Path

import typer

from headroom.backends import is_warm, warm_up
from headroom.backends.ollama import OllamaBackend
from headroom.experiments import load_experiment
from headroom.runner.log import read_jsonl, read_sessions, run_log, session_log
from headroom.runner.paired import new_session, run_session
from headroom.runner.quiet import machine_reader, quiet_check
from headroom.runner.report import condition_effects, format_condition_effects, format_report, summarize_runs
from headroom.runner.single import execute_run
from headroom.schemas.chat import ChatMessage, ToolSchema
from headroom.schemas.config import ExperimentConfig, ModelSpec
from headroom.schemas.run import PlannedRun, Run
from headroom.tasks.suite import ALL_TASKS, get_task
from headroom.telemetry.provenance import collect_provenance
from headroom.telemetry.sources import NvmlSource, PsutilSource

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_EXPERIMENT = Path("experiments/week1_baseline.toml")
RESULTS = Path("results")

SMOKE_TOOL = ToolSchema(
    name="read_file",
    description="Read a text file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


@app.callback()
def main() -> None:
    """Headroom experiment harness."""


@app.command()
def smoke(
    experiment: Path = typer.Option(DEFAULT_EXPERIMENT, help="Experiment config (TOML)."),
    model: str | None = typer.Option(None, help="model_id; defaults to the first of run_models."),
    prompt: str = typer.Option("Read the file notes/today.txt.", help="User message."),
) -> None:
    """Verify the pinned model, warm it up, then make one measured call and print the record as JSON."""
    cfg = load_experiment(experiment)
    spec = cfg.model(model or cfg.run_models[0])
    version, reply = asyncio.run(_smoke(cfg, spec, prompt))
    print(json.dumps({"backend_version": version, "reply": reply.model_dump(mode="json")}, indent=2))
    warm = is_warm(reply, cfg.backend.max_warm_load_s)
    if warm is None:
        typer.echo("WARNING: backend did not report load time; cannot confirm the call was warm", err=True)
    elif not warm:
        typer.echo(
            f"COLD START: measured call reported load_s={reply.load_s.value:.3f}s "  # type: ignore[union-attr]
            f"> max_warm_load_s={cfg.backend.max_warm_load_s}s",
            err=True,
        )
        raise typer.Exit(code=2)


async def _smoke(cfg: ExperimentConfig, spec: ModelSpec, prompt: str):
    async with OllamaBackend(cfg.backend) as backend:
        version = await backend.version()
        await backend.verify_model(spec)
        await warm_up(backend, spec, cfg.sampling)
        reply = await backend.chat(
            spec, [ChatMessage(role="user", content=prompt)],
            sampling=cfg.sampling, seed=cfg.sampling.seed, tools=[SMOKE_TOOL],
        )
    return version, reply


@app.command("run-task")
def run_task_cmd(
    task: str = typer.Option(..., help="task_id, e.g. files.rename_01"),
    arm: str | None = typer.Option(None, help="arm_id; defaults to the first arm."),
    experiment: Path = typer.Option(DEFAULT_EXPERIMENT, help="Experiment config (TOML)."),
    full: bool = typer.Option(False, help="Print the whole run record instead of a summary."),
) -> None:
    """Run one task once (quiet check, telemetry, grading) and print it. Writes nothing to results/."""
    cfg = load_experiment(experiment)
    planned = PlannedRun(
        order_index=0, task_id=task, repeat=0, condition_id=cfg.conditions[0].condition_id,
        arm_id=arm or cfg.arms[0].arm_id,
    )
    run = asyncio.run(_one_run(cfg, planned))
    if full:
        print(json.dumps(run.model_dump(mode="json"), indent=2))
        return
    print(json.dumps({
        "task_id": run.task_id, "arm": run.arm_id, "success": run.success, "status": run.status,
        "error": run.error, "steps": len(run.steps), "wall_s": round(run.wall_s, 3), "cold_start": run.cold_start,
        "quiet_passed": run.quiet.passed, "quiet_failures": run.quiet.failures,
        "malformed_text_calls": run.malformed_text_calls,
        "claimed_done_without_state_change": run.claimed_done_without_state_change,
        "failed_checks": [c.detail for c in run.grade.checks if not c.passed],
    }, indent=2))


async def _one_run(cfg: ExperimentConfig, planned: PlannedRun) -> Run:
    nvml = NvmlSource()
    sources = [PsutilSource(cfg.telemetry.backend_process_prefix, cfg.telemetry.backend_process_arg), nvml]
    try:
        async with OllamaBackend(cfg.backend) as backend:
            version = await _verify(backend, cfg)
            check = quiet_check(cfg.quiet, machine_reader(nvml, cfg.quiet.window_s))
            config = cfg.arm(planned.arm_id).config
            await warm_up(backend, cfg.model(config.model_id), cfg.sampling, num_ctx=config.context_cap)
            return await execute_run(
                backend, get_task(planned.task_id), experiment=cfg, planned=planned, session_id="adhoc",
                order_seed=0, quiet=check, backend_version=version,
                provenance=collect_provenance(Path.cwd(), nvml), sources=sources,
            )
    finally:
        for s in sources:
            s.close()


@app.command("run-experiment")
def run_experiment_cmd(
    experiment: Path = typer.Option(DEFAULT_EXPERIMENT, help="Experiment config (TOML)."),
    repeats: int | None = typer.Option(None, help="Override the config's repeats."),
    arm: list[str] | None = typer.Option(None, help="Run only these arm_ids (repeatable). Default: every arm."),
    task: list[str] | None = typer.Option(None, help="Run only these task_ids (repeatable). Default: every task."),
    condition: list[str] | None = typer.Option(
        None, help="Run only these condition_ids (repeatable). Default: every condition."
    ),
    order_seed: int | None = typer.Option(None, help="Seed for the run order; random (and recorded) if omitted."),
    results: Path = typer.Option(RESULTS, help="Results root (append-only logs)."),
) -> None:
    """Run every task x repeat x condition under every arm, append each run to the log, then print
    the per-condition report for this session."""
    cfg = load_experiment(experiment)
    if repeats is not None:
        cfg = cfg.model_copy(update={"repeats": repeats})
    update = {}
    if arm:
        update["arms"] = [cfg.arm(a).model_dump() for a in arm]
    if condition:
        update["conditions"] = [cfg.condition(c).model_dump() for c in condition]
    if update:
        cfg = ExperimentConfig.model_validate({**cfg.model_dump(), **update})
    task_ids = list(task) if task else [t.task_id for t in ALL_TASKS]
    for t in task_ids:
        get_task(t)  # fail fast on an unknown task id
    session = new_session(cfg, task_ids, order_seed)
    typer.echo(f"session {session.session_id}: {len(session.plan)} runs, order_seed {session.order_seed}", err=True)
    runs = asyncio.run(_run_session(cfg, session, results))
    print(f"session {session.session_id} (order_seed {session.order_seed}, {len(runs)} runs)\n")
    print(format_report(summarize_runs(runs)))
    if effects := format_condition_effects(condition_effects(runs)):
        print("\n" + effects)


async def _run_session(cfg: ExperimentConfig, session, results: Path) -> list[Run]:
    nvml = NvmlSource()
    sources = [PsutilSource(cfg.telemetry.backend_process_prefix, cfg.telemetry.backend_process_arg), nvml]
    total = len(session.plan)

    def progress(run: Run) -> None:
        q = "" if run.quiet.passed else f" QUIET-FLAGGED({'; '.join(run.quiet.failures)})"
        cache = f" CACHE-REUSED({run.first_call_cached_tokens} tok)" if run.cache_reused else ""
        typer.echo(
            f"[{run.order_index + 1:>3}/{total}] {run.condition_id:<10} {run.arm_id:<18} {run.task_id:<24} "
            f"r{run.repeat} {'PASS' if run.success else 'fail'} {run.status:<20} {run.wall_s:5.2f}s "
            f"waited {run.quiet.waited_s:.0f}s{q}{cache}",
            err=True,
        )

    try:
        async with OllamaBackend(cfg.backend) as backend:
            version = await _verify(backend, cfg)
            return await run_session(
                backend, session, {t.task_id: t for t in ALL_TASKS},
                run_log=run_log(results, cfg.experiment_id), session_log=session_log(results, cfg.experiment_id),
                quiet=lambda ignore: quiet_check(cfg.quiet, machine_reader(nvml, cfg.quiet.window_s), ignore=ignore),
                sources=sources, provenance=collect_provenance(Path.cwd(), nvml), backend_version=version,
                on_run=progress, on_event=lambda msg: typer.echo(msg, err=True), nvml=nvml,
            )
    finally:
        for s in sources:
            s.close()


async def _verify(backend: OllamaBackend, cfg: ExperimentConfig) -> str:
    version = await backend.version()
    for model_id in {a.config.model_id for a in cfg.arms}:
        await backend.verify_model(cfg.model(model_id))
    return version


@app.command()
def report(
    experiment_id: str = typer.Option("week1_baseline", "--experiment-id"),
    session: str | None = typer.Option(None, help="session_id; defaults to the latest session."),
    all_sessions: bool = typer.Option(False, "--all", help="Report every logged run."),
    results: Path = typer.Option(RESULTS, help="Results root."),
) -> None:
    """Print the per-condition report from the run log."""
    path = results / experiment_id / "runs.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"no run log at {path}")
    runs, bad = read_jsonl(path, Run)
    if bad:
        typer.echo(f"WARNING: {len(bad)} unreadable line(s) in {path}: {bad[:3]}", err=True)
    if not all_sessions:
        session = session or (read_sessions(results, experiment_id)[-1].session_id)
        runs = [r for r in runs if r.session_id == session]
    print(f"{'all sessions' if all_sessions else f'session {session}'}: {len(runs)} runs\n")
    print(format_report(summarize_runs(runs)))
    if effects := format_condition_effects(condition_effects(runs)):
        print("\n" + effects)


if __name__ == "__main__":
    app()
