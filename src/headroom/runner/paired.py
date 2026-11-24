"""Paired runner: every arm runs every (task, repeat, condition) on identical inputs, in a
randomized, recorded order."""

import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import groupby

from headroom.backends.base import Backend, warm_up
from headroom.conditions.injectors import Injector, InjectorError, make_injector
from headroom.conditions.snapshot import start_readings
from headroom.runner.log import AppendOnlyLog, RunLog
from headroom.runner.single import execute_run
from headroom.schemas.config import Condition, ExperimentConfig, ModelSpec
from headroom.schemas.run import ConditionRecord, PlannedRun, Provenance, QuietCheck, Run, Session
from headroom.schemas.task import Task
from headroom.telemetry.sources import NvmlSource, Source


Item = tuple[str, int, str]  # (task_id, repeat, arm_id)
SPACING_ATTEMPTS = 200


def build_plan(
    task_ids: Sequence[str],
    repeats: int,
    condition_ids: Sequence[str],
    arm_ids: Sequence[str],
    order_seed: int,
    min_same_task_gap: int = 1,
) -> list[PlannedRun]:
    """Conditions run as blocks (injectors are set up once per block), in shuffled order.
    Within a block, every (task, repeat, arm) run is shuffled, so no arm systematically runs
    first, warmest or right after its pair.

    min_same_task_gap: two runs of the same task are at least this many positions apart (across
    block boundaries too). 1 = no constraint. The backend keeps several recent prompts cached, so
    a nearby run of the same task can reuse the cached prompt and run faster."""
    rng = random.Random(order_seed)
    conditions = list(condition_ids)
    rng.shuffle(conditions)
    plan: list[PlannedRun] = []
    for condition_id in conditions:
        items = [(t, r, a) for t in task_ids for r in range(repeats) for a in arm_ids]
        history = [p.task_id for p in plan]
        for task_id, repeat, arm_id in _spaced_order(items, rng, min_same_task_gap, history):
            plan.append(PlannedRun(
                order_index=len(plan), task_id=task_id, repeat=repeat, condition_id=condition_id, arm_id=arm_id
            ))
    return plan


def _spaced_order(items: list[Item], rng: random.Random, gap: int, history: list[str]) -> list[Item]:
    if gap <= 1:
        order = list(items)
        rng.shuffle(order)
        return order
    for _ in range(SPACING_ATTEMPTS):  # random order, weighted toward tasks with more runs left
        order = _try_spaced(items, rng, gap, history, weighted=True)
        if order is not None:
            return order
    order = _try_spaced(items, rng, gap, history, weighted=False)  # most-runs-left first: tightest packing
    if order is None:
        raise ValueError(f"cannot order {len(items)} runs with min_same_task_gap={gap}; too few distinct tasks")
    return order


def _try_spaced(
    items: list[Item], rng: random.Random, gap: int, history: list[str], *, weighted: bool
) -> list[Item] | None:
    by_task: dict[str, list[Item]] = {}
    for it in items:
        by_task.setdefault(it[0], []).append(it)
    for runs in by_task.values():
        rng.shuffle(runs)  # random arm and repeat order within a task
    recent = list(history[-(gap - 1):])
    out: list[Item] = []
    while len(out) < len(items):
        blocked = set(recent[-(gap - 1):])
        allowed = [t for t, runs in by_task.items() if runs and t not in blocked]
        if not allowed:
            return None
        if weighted:
            task = rng.choices(allowed, weights=[len(by_task[t]) for t in allowed])[0]
        else:
            most = max(len(by_task[t]) for t in allowed)
            task = rng.choice([t for t in allowed if len(by_task[t]) == most])
        out.append(by_task[task].pop())
        recent.append(task)
    return out


def new_session(experiment: ExperimentConfig, task_ids: Sequence[str], order_seed: int | None = None) -> Session:
    seed = order_seed if order_seed is not None else random.SystemRandom().randrange(2**31)
    plan = build_plan(
        task_ids, experiment.repeats, [c.condition_id for c in experiment.conditions],
        [a.arm_id for a in experiment.arms], seed, experiment.min_same_task_gap,
    )
    return Session(
        session_id=f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}",
        experiment_id=experiment.experiment_id, started_at=datetime.now(UTC), order_seed=seed,
        plan=tuple(plan), experiment=experiment,
    )


async def run_session(
    backend: Backend,
    session: Session,
    tasks: Mapping[str, Task],
    *,
    run_log: RunLog,
    session_log: AppendOnlyLog,
    quiet: Callable[[frozenset[str]], QuietCheck],
    sources: Sequence[Source],
    provenance: Provenance,
    backend_version: str,
    on_run: Callable[[Run], None] | None = None,
    on_event: Callable[[str], None] | None = None,
    nvml: NvmlSource | None = None,
    injector_factory: Callable[[Condition], Injector] = make_injector,
) -> list[Run]:
    """quiet(ignore) runs the quiet-machine check, ignoring checks the condition itself causes to fail.

    Each condition block runs inside its injector. If an injector cannot start (or dies mid-block),
    the rest of that block is skipped and reported through on_event; the load is always removed."""
    experiment = session.experiment
    say = on_event or (lambda msg: None)
    session_log.append(session)  # the plan is on disk before the first run
    runs: list[Run] = []
    for condition_id, block_iter in groupby(session.plan, key=lambda p: p.condition_id):
        block = list(block_iter)
        condition = experiment.condition(condition_id)
        models = {experiment.arm(p.arm_id).config.model_id: None for p in block}
        injector = injector_factory(condition)
        say(f"condition {condition_id}: starting ({condition.kind})")
        try:
            with injector:
                if injector.reload_model:
                    await _unload(backend, [experiment.model(m) for m in models])
                say(f"condition {condition_id}: in place; {len(block)} runs")
                for planned in block:
                    if not injector.alive():
                        raise InjectorError(f"{condition_id}: injector stopped during the block")
                    runs.append(await _one(
                        backend, experiment, session, tasks, planned, injector, quiet, sources,
                        provenance, backend_version, nvml, condition,
                    ))
                    run_log.append(runs[-1])
                    if on_run:
                        on_run(runs[-1])
        except InjectorError as e:
            say(f"condition {condition_id}: SKIPPED ({e})")
        finally:
            if injector.reload_model:  # after the load is gone, so the next block loads clean
                await _unload(backend, [experiment.model(m) for m in models])
    return runs


async def _one(
    backend, experiment, session, tasks, planned, injector, quiet, sources, provenance, backend_version, nvml, condition,
) -> Run:
    check = quiet(injector.quiet_ignore)
    config = experiment.arm(planned.arm_id).config
    model = experiment.model(config.model_id)
    # Unscored: loads the model with this run's options if needed (a different num_ctx or
    # num_thread reloads it), and replaces part of the backend's prompt cache.
    await warm_up(backend, model, experiment.sampling, num_ctx=config.context_cap, num_thread=injector.num_thread)
    at_start = {**await start_readings(backend, model, nvml), **injector.readings()}
    record = ConditionRecord(
        condition=condition, injector_alive=injector.alive(), at_start=at_start, notes=tuple(injector.notes)
    )
    return await execute_run(
        backend, tasks[planned.task_id], experiment=experiment, planned=planned,
        session_id=session.session_id, order_seed=session.order_seed, quiet=check,
        backend_version=backend_version, provenance=provenance, sources=sources,
        condition=record, num_thread=injector.num_thread,
    )


async def _unload(backend: Backend, models: Sequence[ModelSpec]) -> None:
    unload = getattr(backend, "unload", None)
    if unload is not None:
        for m in models:
            await unload(m)
