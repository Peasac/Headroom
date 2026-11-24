"""One run with telemetry: the unit the paired runner repeats."""

import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from headroom.backends.base import Backend, is_warm
from headroom.runner.agent_loop import run_task
from headroom.runner.awake import unbiased_s

SUSPEND_THRESHOLD_S = 0.5  # more time suspended than this inside a run: its wall time is not usable
from headroom.schemas.common import Measured
from headroom.schemas.config import ExperimentConfig
from headroom.schemas.run import ConditionRecord, PlannedRun, Provenance, QuietCheck, Run
from headroom.schemas.task import Task
from headroom.tasks.checks import grade
from headroom.telemetry.sampler import TelemetrySampler
from headroom.telemetry.sources import Source


async def execute_run(
    backend: Backend,
    task: Task,
    *,
    experiment: ExperimentConfig,
    planned: PlannedRun,
    session_id: str,
    order_seed: int,
    quiet: QuietCheck,
    backend_version: str,
    provenance: Provenance,
    sources: Sequence[Source],
    condition: ConditionRecord | None = None,
    num_thread: int | None = None,
) -> Run:
    if planned.task_id != task.task_id:
        raise ValueError(f"planned task {planned.task_id!r} != {task.task_id!r}")
    if condition is not None and condition.condition.condition_id != planned.condition_id:
        raise ValueError(f"condition {condition.condition.condition_id!r} != planned {planned.condition_id!r}")
    arm = experiment.arm(planned.arm_id)
    model = experiment.model(arm.config.model_id)
    started_at = datetime.now(UTC)
    with TelemetrySampler(sources, experiment.telemetry.interval_s) as sampler:
        u0, c0, t0 = unbiased_s(), time.time(), time.perf_counter()
        loop = await run_task(
            backend, task, model=model, config=arm.config, sampling=experiment.sampling,
            seed=experiment.sampling.seed, num_thread=num_thread,
        )
        wall_s = time.perf_counter() - t0
        c1, u1 = time.time(), unbiased_s()
    # Whichever of the run timer and the system clock counted sleep, minus time that excludes sleep.
    suspended_s = None if u0 is None or u1 is None else max(0.0, max(wall_s, c1 - c0) - (u1 - u0))

    result = grade(task, loop.final_state)
    return Run(
        run_id=uuid.uuid4().hex,
        experiment_id=experiment.experiment_id,
        session_id=session_id,
        order_seed=order_seed,
        order_index=planned.order_index,
        task_id=task.task_id,
        repeat=planned.repeat,
        condition_id=planned.condition_id,
        arm_id=arm.arm_id,
        controller=arm.controller,
        config=arm.config,
        model=model,
        sampling=experiment.sampling,
        backend=experiment.backend,
        backend_version=backend_version,
        provenance=provenance,
        quiet=quiet,
        condition=condition,
        started_at=started_at,
        wall_s=wall_s,
        status=loop.status,
        error=loop.error,
        steps=tuple(loop.steps),
        retries_used=loop.retries_used,
        invalid_state_changing_calls=loop.invalid_state_changing_calls,
        invalid_other_calls=loop.invalid_other_calls,
        malformed_text_calls=loop.malformed_text_calls,
        state_changes=loop.state_changes,
        claimed_done_without_state_change=loop.claimed_done_without_state_change,
        cold_start=_cold_start([is_warm(s.reply, experiment.backend.max_warm_load_s) for s in loop.steps]),
        **_cache_fields(loop.steps),
        suspended_s=suspended_s,
        suspend_suspected=None if suspended_s is None else suspended_s > SUSPEND_THRESHOLD_S,
        final_state=loop.final_state,
        telemetry=sampler.summary(),
        grade=result,
        success=result.success,
    )


def _cache_fields(steps) -> dict:
    def total(attr: str) -> int | None:
        values = [getattr(s.reply, attr) for s in steps]
        if not values or not all(isinstance(v, Measured) for v in values):
            return None
        return int(sum(v.value for v in values))

    first = steps[0].reply.cached_prompt_tokens if steps else None
    first_cached = int(first.value) if isinstance(first, Measured) else None
    return {
        "prompt_tokens_total": total("prompt_tokens"),
        "cached_prompt_tokens_total": total("cached_prompt_tokens"),
        "first_call_cached_tokens": first_cached,
        "cache_reused": None if first_cached is None else first_cached > 0,
    }


def _cold_start(warm: list[bool | None]) -> bool | None:
    if any(w is False for w in warm):
        return True
    if not warm or any(w is None for w in warm):
        return None
    return False
