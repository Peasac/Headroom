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


def build_plan(
    task_ids: Sequence[str], repeats: int, condition_ids: Sequence[str], arm_ids: Sequence[str], order_seed: int
) -> list[PlannedRun]:
    """Conditions run as blocks (injectors are set up once per block), in shuffled order.
    Within a block, every (task, repeat, arm) run is shuffled, so no arm systematically runs
    first, warmest or right after its pair."""
    rng = random.Random(order_seed)
    conditions = list(condition_ids)
    rng.shuffle(conditions)
    plan: list[PlannedRun] = []
    for condition_id in conditions:
        block = [(t, r, a) for t in task_ids for r in range(repeats) for a in arm_ids]
        rng.shuffle(block)
        for task_id, repeat, arm_id in block:
            plan.append(PlannedRun(
                order_index=len(plan), task_id=task_id, repeat=repeat, condition_id=condition_id, arm_id=arm_id
            ))
    return plan


def new_session(experiment: ExperimentConfig, task_ids: Sequence[str], order_seed: int | None = None) -> Session:
    seed = order_seed if order_seed is not None else random.SystemRandom().randrange(2**31)
    plan = build_plan(
        task_ids, experiment.repeats, [c.condition_id for c in experiment.conditions],
        [a.arm_id for a in experiment.arms], seed,
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
        model = experiment.model(experiment.arm(planned.arm_id).config.model_id)
        # Unscored: loads the model if needed, and replaces the backend's prompt cache with an
        # unrelated prompt, so no run inherits the cached prefix of the run before it.
        await warm_up(backend, model, experiment.sampling)
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
