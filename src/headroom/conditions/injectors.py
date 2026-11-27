"""One injector per condition kind. Use as a context manager around a condition block:
start() puts the load in place (and waits until it is), stop() removes it, also on errors."""

import collections
import hashlib
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import psutil

from headroom.conditions.procs import bind_to_harness, kill_strays, spawn_worker, stop_worker
from headroom.schemas.common import Reading, measured, unavailable
from headroom.schemas.config import (
    BackgroundWorkload, BaselineCondition, Condition, CpuContention, MemoryPressure, ReducedCompute, VramPressure,
)

MB = 2**20
READY_TIMEOUT_S = 120


class InjectorError(RuntimeError):
    """The condition could not be applied safely. Its block is skipped; nothing is left running."""


class Injector:
    quiet_ignore: frozenset[str] = frozenset()  # quiet checks this condition itself causes to fail
    reload_model: bool = False  # unload the model at block start/stop so it loads under the condition
    num_thread: int | None = None  # backend option for every call in the block

    def __init__(self, condition: Condition) -> None:
        self.condition = condition
        self.notes: list[str] = []

    def __enter__(self) -> "Injector":
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def alive(self) -> bool:
        return True

    def readings(self) -> dict[str, Reading]:
        """Injector-specific readings at the start of a run (the level actually in place)."""
        return {}


class BaselineInjector(Injector):
    pass


class _WorkerInjector(Injector):
    def __init__(self, condition: Condition) -> None:
        super().__init__(condition)
        self.procs: list[subprocess.Popen] = []

    def _spawn(self, *args: str) -> str:
        proc = spawn_worker(*args)
        self.procs.append(proc)
        line = _read_line(proc, READY_TIMEOUT_S)
        if not line.startswith("READY"):
            err = proc.stderr.read() if proc.poll() is not None and proc.stderr else ""
            raise InjectorError(f"{self.condition.condition_id}: worker did not start: {line or err or 'timeout'}")
        return line

    def start(self) -> None:
        if killed := kill_strays():
            self.notes.append(f"killed {killed} stray worker(s) from an earlier session")

    def stop(self) -> None:
        for proc in self.procs:
            stop_worker(proc)
        self.procs = []

    def alive(self) -> bool:
        return bool(self.procs) and all(p.poll() is None for p in self.procs)


class MemoryInjector(_WorkerInjector):
    condition: MemoryPressure

    def start(self) -> None:
        super().start()
        c = self.condition
        self._spawn("memory", str(c.target_available_mb), str(c.min_available_mb), str(c.max_ballast_mb))

    def readings(self) -> dict[str, Reading]:
        out: dict[str, Reading] = {"target_available_mb": measured(self.condition.target_available_mb, "MB")}
        if not self.alive():
            out["ballast_held_mb"] = unavailable("ballast process is not running")
            return out
        try:
            mi = psutil.Process(self.procs[0].pid).memory_info()
            held = getattr(mi, "private", mi.rss)
            out["ballast_held_mb"] = measured(held / MB, "MB")
        except psutil.Error as e:
            out["ballast_held_mb"] = unavailable(f"cannot read ballast process: {e}")
        return out


class CpuInjector(_WorkerInjector):
    condition: CpuContention
    quiet_ignore = frozenset({"gpu_temp"})  # busy CPUs heat the shared laptop cooling

    def start(self) -> None:
        super().start()
        c = self.condition
        ncpu = psutil.cpu_count() or 1
        if c.workers > ncpu - c.min_free_logical_cpus:
            raise InjectorError(
                f"{c.condition_id}: {c.workers} workers would leave fewer than {c.min_free_logical_cpus} "
                f"of {ncpu} logical CPUs free"
            )
        for _ in range(c.workers):
            self._spawn("cpu")

    def readings(self) -> dict[str, Reading]:
        return {
            "workers_requested": measured(self.condition.workers, "processes"),
            "workers_alive": measured(sum(p.poll() is None for p in self.procs), "processes"),
        }


class ReducedComputeInjector(Injector):
    condition: ReducedCompute

    def __init__(self, condition: ReducedCompute) -> None:
        super().__init__(condition)
        self.num_thread = condition.num_thread

    def readings(self) -> dict[str, Reading]:
        return {"num_thread": measured(self.condition.num_thread, "threads")}


class VramInjector(_WorkerInjector):
    condition: VramPressure
    quiet_ignore = frozenset({"gpu_util"})  # the ballast is rewritten on the GPU periodically
    reload_model = True

    def start(self) -> None:
        super().start()
        c = self.condition
        avail = psutil.virtual_memory().available / MB
        if avail < c.min_available_ram_mb:
            raise InjectorError(
                f"{c.condition_id}: only {avail:.0f} MB RAM available, need {c.min_available_ram_mb} MB "
                "(over-committed VRAM spills into system RAM)"
            )
        self._spawn("vram", str(c.ballast_mb), str(c.touch_interval_s))

    def readings(self) -> dict[str, Reading]:
        held = self.condition.ballast_mb if self.alive() else None
        return {"vram_ballast_mb": measured(held, "MB") if held else unavailable("ballast process is not running")}


class WorkloadInjector(Injector):
    """Runs a real program for the whole block. With progress='ffmpeg' it reads ffmpeg's `-progress`
    key=value stream and reports the frame rate over the last FPS_WINDOW_S seconds before each run."""

    condition: BackgroundWorkload
    FPS_WINDOW_S = 5.0

    def __init__(self, condition: BackgroundWorkload, *, clock=time.monotonic, sleep=time.sleep) -> None:
        super().__init__(condition)
        self.quiet_ignore = frozenset(condition.quiet_ignore)
        self._clock, self._sleep = clock, sleep
        self.proc: subprocess.Popen | None = None
        self._frames: collections.deque[tuple[float, int]] = collections.deque(maxlen=4096)
        self._log: Path | None = None

    def start(self) -> None:
        c = self.condition
        exe = Path(c.exe)
        if not exe.is_file():
            raise InjectorError(f"{c.condition_id}: {exe} not found")
        digest = _sha256(exe)
        if digest != c.exe_sha256:
            raise InjectorError(f"{c.condition_id}: {exe} sha256 {digest}, pinned {c.exe_sha256}")
        self._log = Path(tempfile.mkstemp(prefix=f"headroom-{c.condition_id}-", suffix=".log")[1])
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with open(self._log, "wb") as err:
            self.proc = subprocess.Popen(
                [str(exe), *c.args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                text=True, creationflags=flags,
            )
        bind_to_harness(self.proc)
        if c.progress == "ffmpeg":
            threading.Thread(target=self._read_progress, name=f"{c.condition_id}-progress", daemon=True).start()
        self._sleep(c.settle_s)
        if not self.alive():
            raise InjectorError(f"{c.condition_id}: workload exited with code {self.proc.poll()}: {self._tail()}")
        if c.progress == "ffmpeg" and not self._frames:
            raise InjectorError(f"{c.condition_id}: no ffmpeg progress after {c.settle_s}s: {self._tail()}")

    def _read_progress(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "frame" and value.isdigit():
                self._frames.append((self._clock(), int(value)))

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(10)
        if proc.stdout:
            proc.stdout.close()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def recent_fps(self) -> float | None:
        if len(self._frames) < 2:
            return None
        now = self._frames[-1][0]
        window = [(t, f) for t, f in self._frames if t >= now - self.FPS_WINDOW_S]
        if len(window) < 2 or window[-1][0] <= window[0][0]:
            return None
        return (window[-1][1] - window[0][1]) / (window[-1][0] - window[0][0])

    def readings(self) -> dict[str, Reading]:
        out: dict[str, Reading] = {
            "workload_alive": measured(1 if self.alive() else 0, "bool"),
        }
        if self.condition.progress == "ffmpeg":
            fps = self.recent_fps() if self.alive() else None
            out["workload_fps"] = (
                measured(fps, "frames/s") if fps is not None
                else unavailable("workload not running or no recent progress")
            )
        try:
            if self.alive():
                p = psutil.Process(self.proc.pid)  # type: ignore[union-attr]
                mi = p.memory_info()
                out["workload_private_mb"] = measured(getattr(mi, "private", mi.rss) / MB, "MB")
        except psutil.Error as e:
            out["workload_private_mb"] = unavailable(f"cannot read workload process: {e}")
        return out

    def _tail(self, n: int = 400) -> str:
        try:
            return self._log.read_text(errors="replace")[-n:].strip() if self._log else ""
        except OSError:
            return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def make_injector(condition: Condition) -> Injector:
    match condition:
        case BaselineCondition():
            return BaselineInjector(condition)
        case MemoryPressure():
            return MemoryInjector(condition)
        case CpuContention():
            return CpuInjector(condition)
        case ReducedCompute():
            return ReducedComputeInjector(condition)
        case VramPressure():
            return VramInjector(condition)
        case BackgroundWorkload():
            return WorkloadInjector(condition)
    raise TypeError(f"no injector for {condition!r}")


def _read_line(proc: subprocess.Popen, timeout_s: float) -> str:
    q: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: q.put(proc.stdout.readline() if proc.stdout else ""), daemon=True).start()
    try:
        return q.get(timeout=timeout_s).strip()
    except queue.Empty:
        return ""
