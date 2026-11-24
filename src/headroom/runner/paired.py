"""Paired runner: every arm runs every (task, repeat, condition) on identical inputs, in a
randomized, recorded order."""

import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from headroom.backends.base import Backend, warm_up
from headroom.runner.log import AppendOnlyLog, RunLog
from headroom.runner.single import execute_run
from headroom.schemas.config import ExperimentConfig
from headroom.schemas.run import PlannedRun, Provenance, QuietCheck, Run, Session
from headroom.schemas.task import Task
from headroom.telemetry.sources import Source

SUPPORTED_CONDITIONS = ("baseline",)


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
    quiet: Callable[[], QuietCheck],
    sources: Sequence[Source],
    provenance: Provenance,
    backend_version: str,
    on_run: Callable[[Run], None] | None = None,
) -> list[Run]:
    experiment = session.experiment
    unsupported = [c.condition_id for c in experiment.conditions if c.kind not in SUPPORTED_CONDITIONS]
    if unsupported:
        raise NotImplementedError(f"no injector yet for conditions {unsupported}")

    session_log.append(session)  # the plan is on disk before the first run
    runs = []
    for planned in session.plan:
        check = quiet()
        config = experiment.arm(planned.arm_id).config
        model = experiment.model(config.model_id)
        # Unscored: loads the model at this arm's context size if needed (a different num_ctx
        # reloads it), and replaces the backend's prompt cache with an unrelated prompt, so no run
        # inherits the cached prefix of the run before it.
        await warm_up(backend, model, experiment.sampling, num_ctx=config.context_cap)
        run = await execute_run(
            backend, tasks[planned.task_id], experiment=experiment, planned=planned,
            session_id=session.session_id, order_seed=session.order_seed, quiet=check,
            backend_version=backend_version, provenance=provenance, sources=sources,
        )
        run_log.append(run)
        runs.append(run)
        if on_run:
            on_run(run)
    return runs
