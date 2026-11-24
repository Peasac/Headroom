import inspect

import pytest
from helpers import make_run

from headroom.backends.fake import FakeTurn
from headroom.runner.log import AppendOnlyLog, RunLog, read_jsonl
from headroom.schemas.chat import ToolCall
from headroom.schemas.run import Run

pytestmark = pytest.mark.anyio

GOOD = ToolCall(name="rename_file", arguments={"src": "drafts/plan.md", "dst": "archive/plan.md"})


@pytest.fixture
async def runs(experiment, provenance):
    return [await make_run(experiment, provenance, [FakeTurn(tool_calls=(GOOD,)), "ok"], order_index=i) for i in range(3)]


async def test_appending_never_changes_earlier_bytes(tmp_path, runs):
    log = RunLog(tmp_path / "runs.jsonl")
    log.append(runs[0])
    first = log.path.read_bytes()
    log.append(runs[1])
    after = log.path.read_bytes()
    assert after.startswith(first)
    assert after.count(b"\n") == 2


async def test_round_trip(tmp_path, runs):
    log = RunLog(tmp_path / "runs.jsonl")
    for r in runs:
        log.append(r)
    read, bad = read_jsonl(log.path, Run)
    assert read == runs and bad == []


async def test_reopening_keeps_existing_records(tmp_path, runs):
    RunLog(tmp_path / "runs.jsonl").append(runs[0])
    log = RunLog(tmp_path / "runs.jsonl")  # a new session opening the same file
    log.append(runs[1])
    assert [r.run_id for r in read_jsonl(log.path, Run)[0]] == [runs[0].run_id, runs[1].run_id]


async def test_duplicate_run_id_is_refused(tmp_path, runs):
    log = RunLog(tmp_path / "runs.jsonl")
    log.append(runs[0])
    with pytest.raises(ValueError):
        RunLog(tmp_path / "runs.jsonl").append(runs[0])
    assert len(read_jsonl(log.path, Run)[0]) == 1


async def test_torn_last_line_is_kept_and_reported_not_overwritten(tmp_path, runs):
    path = tmp_path / "runs.jsonl"
    RunLog(path).append(runs[0])
    with open(path, "ab") as f:
        f.write(b'{"run_id": "torn')  # simulated crash mid-write
    torn = path.read_bytes()
    RunLog(path).append(runs[1])
    assert path.read_bytes().startswith(torn)
    read, bad = read_jsonl(path, Run)
    assert [r.run_id for r in read] == [runs[0].run_id, runs[1].run_id]
    assert [lineno for lineno, _ in bad] == [2]


def test_log_has_no_rewrite_or_delete_api():
    public = {n for n, _ in inspect.getmembers(RunLog) if not n.startswith("_")}
    assert public == {"append"}
    assert {n for n, _ in inspect.getmembers(AppendOnlyLog) if not n.startswith("_")} == {"append"}
