import os
import subprocess
import sys
import time

import psutil
import pytest
from conftest import ROOT, FakeSource, quiet_ok
from helpers import SAMPLE, TaskBackend

from headroom.conditions.injectors import (
    CpuInjector, Injector, InjectorError, MemoryInjector, VramInjector, make_injector,
)
from headroom.conditions.procs import WORKER_MODULE, kill_strays
from headroom.experiments import load_experiment
from headroom.runner.log import AppendOnlyLog, RunLog
from headroom.runner.paired import new_session, run_session
from headroom.runner.quiet import QuietReading, evaluate
from headroom.schemas.common import measured
from headroom.schemas.config import CpuContention, MemoryPressure, VramPressure
from headroom.tasks.suite import get_task

WEEK2 = ROOT / "experiments" / "week2_conditions.toml"


def _gone(pid: int, timeout_s: float = 10) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


def test_week2_levels_are_the_ones_set_once():
    cfg = load_experiment(WEEK2)
    by_id = {c.condition_id: c for c in cfg.conditions}
    assert [c.condition_id for c in cfg.conditions] == ["baseline", "memory", "cpu", "reduced", "vram", "vram-1g"]
    assert (by_id["memory"].target_available_mb, by_id["memory"].min_available_mb) == (1024, 512)
    assert (by_id["cpu"].workers, by_id["cpu"].min_free_logical_cpus) == (14, 2)
    assert by_id["reduced"].num_thread == 2
    assert (by_id["vram"].ballast_mb, by_id["vram"].min_available_ram_mb) == (2048, 1024)
    # The one extra dose-response level: reported separately, never counted for Gate A.
    assert by_id["vram-1g"].ballast_mb == 1024 and by_id["vram-1g"].gate_a is False
    assert all(c.gate_a for c in cfg.conditions if c.condition_id != "vram-1g")


def test_week2_full_run_design():
    cfg = load_experiment(WEEK2)
    assert cfg.repeats == 3 and cfg.min_same_task_gap == 5 and cfg.unload_before_each_run
    assert sorted((a.config.mode, a.config.context_cap) for a in cfg.arms) == sorted(
        (m, c) for m in ("direct", "validated", "plan_act", "plan_act_verify") for c in (2048, 8192)
    )
    assert {a.config.retry_limit for a in cfg.arms if a.config.mode != "direct"} == {2}


def test_week2_injectors():
    cfg = load_experiment(WEEK2)
    assert {type(make_injector(c)).__name__ for c in cfg.conditions} == {
        "BaselineInjector", "MemoryInjector", "CpuInjector", "ReducedComputeInjector", "VramInjector",
    }


def test_memory_valve_must_sit_below_the_target():
    with pytest.raises(ValueError):
        MemoryPressure(condition_id="m", kind="memory_pressure", realism="emulated",
                       target_available_mb=500, min_available_mb=600, max_ballast_mb=1000)


def test_cpu_workers_start_and_stop_cleanly():
    inj = CpuInjector(CpuContention(condition_id="c", kind="cpu_contention", realism="emulated",
                                    workers=2, min_free_logical_cpus=1))
    with inj:
        pids = [p.pid for p in inj.procs]
        assert inj.alive() and len(pids) == 2
        assert inj.readings()["workers_alive"].value == 2
        assert all(psutil.Process(pid).cpu_percent(0.3) > 20 for pid in pids)  # they really spin
    assert not inj.procs
    assert all(_gone(pid) for pid in pids)


def test_cpu_guard_keeps_headroom_and_leaves_nothing_running():
    ncpu = psutil.cpu_count()
    inj = CpuInjector(CpuContention(condition_id="c", kind="cpu_contention", realism="emulated",
                                    workers=ncpu, min_free_logical_cpus=1))
    with pytest.raises(InjectorError, match="logical CPUs free"):
        with inj:
            pass
    assert not inj.procs


def test_memory_ballast_holds_and_releases():
    # target 2 MB available: grows to max_ballast_mb (256 MB here) and holds it.
    inj = MemoryInjector(MemoryPressure(condition_id="m", kind="memory_pressure", realism="emulated",
                                        target_available_mb=2, min_available_mb=1, max_ballast_mb=256))
    with inj:
        held = inj.readings()["ballast_held_mb"]
        assert held.status == "ok" and held.value >= 200
        pid = inj.procs[0].pid
    assert _gone(pid)


def test_vram_guard_refuses_when_ram_is_short():
    inj = VramInjector(VramPressure(condition_id="v", kind="vram_pressure", realism="emulated",
                                    ballast_mb=64, touch_interval_s=0.5, min_available_ram_mb=10**9))
    with pytest.raises(InjectorError, match="RAM available"):
        with inj:
            pass
    assert not inj.procs


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object cleanup is Windows-only")
def test_workers_die_when_the_harness_crashes():
    # A throwaway "harness" starts a worker, then dies without any cleanup (os._exit).
    code = (
        "import os, sys; from headroom.conditions.procs import spawn_worker;"
        "p = spawn_worker('cpu'); print(p.pid, flush=True); p.stdout.readline(); os._exit(1)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    worker_pid = int(out.stdout.split()[0])
    assert _gone(worker_pid), "worker outlived the crashed harness"


def test_worker_is_a_single_process_we_can_stop():
    inj = CpuInjector(CpuContention(condition_id="c", kind="cpu_contention", realism="emulated",
                                    workers=1, min_free_logical_cpus=1))
    with inj:
        pid = inj.procs[0].pid
        # Not a venv launcher with the real worker underneath (Windows may attach a conhost.exe).
        assert not [c for c in psutil.Process(pid).children() if "python" in c.name().lower()]
        assert WORKER_MODULE in psutil.Process(pid).cmdline()
    assert _gone(pid)


def test_kill_strays_removes_workers_of_another_harness():
    other = os.getppid()  # a live pid that is not us
    exe = getattr(sys, "_base_executable", None) or sys.executable
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    proc = subprocess.Popen([exe, "-m", WORKER_MODULE, str(other), "cpu"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        proc.stdout.readline()
        assert kill_strays() >= 1
        assert _gone(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()


def test_quiet_check_ignores_what_the_condition_causes(quiet_settings):
    hot = QuietReading(gpu_util_pct=measured(60, "%"), gpu_temp_c=measured(80, "C"),
                       on_ac=measured(1, "bool"), power_plan="Balanced")
    failures, unchecked, _ = evaluate(hot, quiet_settings, frozenset({"gpu_util"}))
    assert failures == ["GPU temp 80C >= 70C"]
    assert "gpu_util (caused by condition)" in unchecked


# --- runner integration with fake injectors -------------------------------------------------

class FakeInjector(Injector):
    log: list[str] = []

    def __init__(self, condition, *, fail_start=False, die_after=None, num_thread=None, reload_model=False):
        super().__init__(condition)
        self.fail_start, self.die_after, self.calls = fail_start, die_after, 0
        self.num_thread, self.reload_model = num_thread, reload_model
        self.running = False

    def start(self):
        FakeInjector.log.append(f"start {self.condition.condition_id}")
        if self.fail_start:
            raise InjectorError("cannot apply")
        self.running = True

    def stop(self):
        FakeInjector.log.append(f"stop {self.condition.condition_id}")
        self.running = False

    def alive(self):
        self.calls += 1
        return self.running and (self.die_after is None or self.calls <= self.die_after)

    def readings(self):
        return {"level_marker": measured(7, "x")}


class UnloadingBackend(TaskBackend):
    def __init__(self, prompts):
        super().__init__(prompts)
        self.unloads: list[str] = []

    async def unload(self, model):
        self.unloads.append(model.model_id)


@pytest.mark.anyio
async def test_blocks_run_inside_their_injector_and_are_skipped_safely(tmp_path, experiment, provenance):
    conds = [
        {"condition_id": "baseline", "kind": "baseline", "realism": "real"},
        {"condition_id": "reduced", "kind": "reduced_compute", "realism": "emulated", "num_thread": 2},
        {"condition_id": "vram", "kind": "vram_pressure", "realism": "emulated",
         "ballast_mb": 64, "touch_interval_s": 1, "min_available_ram_mb": 1},
        {"condition_id": "cpu", "kind": "cpu_contention", "realism": "emulated", "workers": 1, "min_free_logical_cpus": 1},
    ]
    exp = experiment.model_validate({**experiment.model_dump(), "conditions": conds, "repeats": 1,
                                     "arms": [experiment.arms[0].model_dump()]})
    tasks = {t: get_task(t) for t in ("files.rename_01", "calendar.add_01")}
    backend = UnloadingBackend({tasks[t].prompt: t for t in tasks})

    def factory(c):
        return {
            "baseline": lambda: FakeInjector(c),
            "reduced": lambda: FakeInjector(c, num_thread=2),
            "vram": lambda: FakeInjector(c, reload_model=True),
            "cpu": lambda: FakeInjector(c, fail_start=True),
        }[c.condition_id]()

    FakeInjector.log = []
    events: list[str] = []
    runs = await run_session(
        backend, new_session(exp, list(tasks), order_seed=1), tasks,
        run_log=RunLog(tmp_path / "runs.jsonl"), session_log=AppendOnlyLog(tmp_path / "sessions.jsonl"),
        quiet=lambda ignore: quiet_ok(), sources=[FakeSource([SAMPLE])], provenance=provenance,
        backend_version="fake-0", on_event=events.append, injector_factory=factory,
    )
    assert {r.condition_id for r in runs} == {"baseline", "reduced", "vram"}  # cpu could not start: skipped
    assert any("cpu: SKIPPED" in e for e in events)
    assert sorted(FakeInjector.log) == sorted(f"{a} {c}" for c in ("baseline", "reduced", "vram", "cpu")
                                              for a in ("start", "stop"))
    for r in runs:
        assert r.condition is not None and r.condition.condition.condition_id == r.condition_id
        assert r.condition.at_start["level_marker"].value == 7 and "sys_mem_available_mb" in r.condition.at_start
    assert {q["num_thread"] for q in backend.requests} == {None, 2}  # only the reduced block sends num_thread
    assert sum(1 for r in runs if r.condition_id == "reduced") == 2
    assert backend.unloads.count("llama3.2-3b") == 2  # vram block: before (load under pressure) and after


@pytest.mark.anyio
async def test_unload_before_each_run_and_warm_up_never_uses_a_task_prompt(tmp_path, experiment, provenance):
    exp = experiment.model_validate({**experiment.model_dump(), "repeats": 2, "unload_before_each_run": True})
    tasks = {t: get_task(t) for t in ("files.rename_01", "calendar.add_01")}
    backend = UnloadingBackend({tasks[t].prompt: t for t in tasks})
    runs = await run_session(
        backend, new_session(exp, list(tasks), order_seed=1), tasks,
        run_log=RunLog(tmp_path / "runs.jsonl"), session_log=AppendOnlyLog(tmp_path / "sessions.jsonl"),
        quiet=lambda ignore: quiet_ok(), sources=[FakeSource([SAMPLE])], provenance=provenance,
        backend_version="fake-0",
    )
    assert len(backend.unloads) == len(runs) == 8
    warmups = [q for q in backend.requests if q["sampling"].max_tokens == 1]
    prompts = {t.prompt for t in tasks.values()}
    assert len(warmups) == len(runs)
    assert all(m.content not in prompts for q in warmups for m in q["messages"])
    assert {m.content for q in warmups for m in q["messages"]} == {"Reply with OK."}


@pytest.mark.anyio
async def test_resume_runs_only_what_is_missing_in_the_original_order(tmp_path, experiment, provenance):
    exp = experiment.model_validate({**experiment.model_dump(), "repeats": 2})
    tasks = {t: get_task(t) for t in ("files.rename_01", "calendar.add_01")}
    session = new_session(exp, list(tasks), order_seed=4)
    run_log_, session_log_ = RunLog(tmp_path / "runs.jsonl"), AppendOnlyLog(tmp_path / "sessions.jsonl")
    common = dict(run_log=run_log_, session_log=session_log_, quiet=lambda ignore: quiet_ok(),
                  sources=[FakeSource([SAMPLE])], provenance=provenance, backend_version="fake-0")
    session_log_.append(session)  # as if the first attempt wrote it, then crashed after 3 runs
    first = await run_session(TaskBackend({tasks[t].prompt: t for t in tasks}),
                              session.model_copy(update={"plan": session.plan[:3]}), tasks, done=frozenset(), **common)
    assert [r.order_index for r in first] == [0, 1, 2]
    rest = await run_session(TaskBackend({tasks[t].prompt: t for t in tasks}), session, tasks,
                             done=frozenset(r.order_index for r in first), **common)
    assert [r.order_index for r in rest] == [p.order_index for p in session.plan[3:]]
    from headroom.runner.log import read_jsonl
    from headroom.schemas.run import Run, Session
    logged = read_jsonl(run_log_.path, Run)[0]
    assert sorted(r.order_index for r in logged) == list(range(len(session.plan)))
    assert len(read_jsonl(session_log_.path, Session)[0]) == 1  # the session is not written again


def test_sleep_detection_and_keep_awake_are_available_on_windows():
    from headroom.runner.awake import keep_awake, unbiased_s

    if sys.platform != "win32":
        pytest.skip("Windows only")
    a = unbiased_s()
    time.sleep(0.05)
    assert a is not None and 0.03 < unbiased_s() - a < 1.0
    with keep_awake() as ok:
        assert ok


@pytest.mark.anyio
async def test_a_dying_injector_ends_its_block(tmp_path, experiment, provenance):
    exp = experiment.model_validate({**experiment.model_dump(), "repeats": 2, "arms": [experiment.arms[0].model_dump()]})
    tasks = {t: get_task(t) for t in ("files.rename_01", "calendar.add_01")}
    events: list[str] = []
    FakeInjector.log = []
    runs = await run_session(
        TaskBackend({tasks[t].prompt: t for t in tasks}), new_session(exp, list(tasks), order_seed=1), tasks,
        run_log=RunLog(tmp_path / "runs.jsonl"), session_log=AppendOnlyLog(tmp_path / "sessions.jsonl"),
        quiet=lambda ignore: quiet_ok(), sources=[FakeSource([SAMPLE])], provenance=provenance,
        backend_version="fake-0", on_event=events.append, injector_factory=lambda c: FakeInjector(c, die_after=4),
    )
    assert len(runs) == 2  # 4 planned; the injector died after 2 (alive() is asked twice per run)
    assert any("stopped during the block" in e for e in events)
    assert FakeInjector.log[-1] == "stop baseline"
