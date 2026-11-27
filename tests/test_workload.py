"""Background workload condition: a real program for the whole block, with its throughput recorded."""

import hashlib
import sys
from pathlib import Path

import psutil
import pytest

from headroom.conditions.injectors import InjectorError, WorkloadInjector, make_injector
from headroom.schemas.config import BackgroundWorkload

PY = Path(getattr(sys, "_base_executable", None) or sys.executable)  # the real interpreter, not the venv launcher
PY_SHA = hashlib.sha256(PY.read_bytes()).hexdigest()
# Prints ffmpeg-style -progress blocks: about 50 frames per second.
FAKE_FFMPEG = (
    "import sys, time\n"
    "f = 0\n"
    "while True:\n"
    "    f += 5; print(f'frame={f}'); print('fps=50.0'); print('progress=continue'); sys.stdout.flush(); time.sleep(0.1)\n"
)


def workload(**kw) -> BackgroundWorkload:
    base = dict(condition_id="fake-export", kind="background_workload", realism="real", exe=str(PY), exe_sha256=PY_SHA,
                args=["-c", FAKE_FFMPEG], progress="ffmpeg", settle_s=1.5, quiet_ignore=["gpu_temp"])
    return BackgroundWorkload(**{**base, **kw})


def test_workload_runs_reports_fps_and_stops():
    inj = make_injector(workload())
    assert isinstance(inj, WorkloadInjector) and inj.quiet_ignore == frozenset({"gpu_temp"})
    with inj:
        pid = inj.proc.pid
        r = inj.readings()
        assert r["workload_alive"].value == 1
        assert 25 < r["workload_fps"].value < 75
        assert r["workload_private_mb"].value > 0
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert not inj.alive()


def test_a_workload_that_exits_early_fails_the_block():
    with pytest.raises(InjectorError, match="exited with code 3"):
        with make_injector(workload(args=["-c", "import sys; sys.exit(3)"], progress="none", settle_s=1.0)):
            pass


def test_no_progress_fails_the_block():
    with pytest.raises(InjectorError, match="no ffmpeg progress"):
        with make_injector(workload(args=["-c", "import time; time.sleep(30)"], settle_s=1.0)):
            pass


def test_an_unpinned_exe_is_refused():
    with pytest.raises(InjectorError, match="pinned"):
        with make_injector(workload(exe_sha256="0" * 64)):
            pass


def test_fps_is_unavailable_once_the_workload_died():
    inj = make_injector(workload())
    with inj:
        inj.proc.kill()
        inj.proc.wait(10)
        r = inj.readings()
    assert r["workload_alive"].value == 0 and r["workload_fps"].status == "unavailable"
