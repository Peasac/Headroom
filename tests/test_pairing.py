from collections import Counter

import pytest
from conftest import FakeSource, quiet_ok
from helpers import SAMPLE, SOLUTIONS, TaskBackend

from headroom.runner.log import AppendOnlyLog, RunLog, read_jsonl
from headroom.runner.paired import build_plan, new_session, run_session
from headroom.schemas.run import Run, Session
from headroom.tasks.suite import get_task

TASKS = ["files.rename_01", "calendar.add_01"]


def test_plan_covers_every_pair_under_every_arm_exactly_once():
    plan = build_plan(TASKS, repeats=3, condition_ids=["baseline", "other"], arm_ids=["a", "b"], order_seed=1)
    keys = Counter((p.task_id, p.repeat, p.condition_id, p.arm_id) for p in plan)
    assert len(keys) == 2 * 3 * 2 * 2 and set(keys.values()) == {1}
    assert [p.order_index for p in plan] == list(range(len(plan)))


def test_conditions_run_as_contiguous_blocks():
    plan = build_plan(TASKS, 3, ["baseline", "other"], ["a", "b"], order_seed=1)
    conditions = [p.condition_id for p in plan]
    assert conditions == sorted(conditions, key=conditions.index)  # each condition appears in one run


def test_order_is_random_but_reproducible_from_the_seed():
    a = build_plan(TASKS, 5, ["baseline"], ["a", "b"], order_seed=1)
    assert a == build_plan(TASKS, 5, ["baseline"], ["a", "b"], order_seed=1)
    assert a != build_plan(TASKS, 5, ["baseline"], ["a", "b"], order_seed=2)
    unshuffled = [(t, r, arm) for t in TASKS for r in range(5) for arm in ["a", "b"]]
    assert [(p.task_id, p.repeat, p.arm_id) for p in a] != unshuffled


@pytest.mark.anyio
async def test_session_runs_arms_on_identical_inputs_and_logs_everything(tmp_path, experiment, provenance):
    tasks = {t: get_task(t) for t in TASKS}
    backend = TaskBackend({tasks[t].prompt: t for t in TASKS})
    session = new_session(experiment, TASKS, order_seed=3)
    runs_log, sessions_log = RunLog(tmp_path / "runs.jsonl"), AppendOnlyLog(tmp_path / "sessions.jsonl")
    runs = await run_session(
        backend, session, tasks, run_log=runs_log, session_log=sessions_log, quiet=quiet_ok,
        sources=[FakeSource([SAMPLE])], provenance=provenance, backend_version="fake-0",
    )

    # Every planned run happened, in plan order, and is in the log with its order recorded.
    assert [r.order_index for r in runs] == [p.order_index for p in session.plan]
    assert read_jsonl(runs_log.path, Run)[0] == runs
    assert all(r.session_id == session.session_id and r.order_seed == 3 for r in runs)
    (logged_session,), _ = read_jsonl(sessions_log.path, Session)
    assert logged_session.plan == session.plan

    # Identical inputs within each pair: same first request (messages, tools, sampling, seed).
    first_requests = [r for r in backend.requests if r["sampling"].max_tokens != 1 and len(r["messages"]) == 2]
    assert len(first_requests) == len(runs)
    by_pair: dict = {}
    for run, req in zip(runs, first_requests, strict=True):
        by_pair.setdefault((run.task_id, run.repeat, run.condition_id), {})[run.arm_id] = req
    assert len(by_pair) == len(TASKS) * experiment.repeats
    for arms in by_pair.values():
        assert set(arms) == {"validated", "direct"}
        a, b = arms["validated"], arms["direct"]
        assert (a["messages"], a["tools"], a["sampling"], a["seed"]) == (b["messages"], b["tools"], b["sampling"], b["seed"])
    assert {r["seed"] for r in first_requests} == {experiment.sampling.seed}

    # A warm-up (prompt-cache reset) precedes every run.
    warmups = [i for i, r in enumerate(backend.requests) if r["sampling"].max_tokens == 1]
    assert len(warmups) == len(runs)
    assert all(r.success for r in runs) and set(SOLUTIONS) >= set(TASKS)
