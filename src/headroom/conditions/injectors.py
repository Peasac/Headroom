"""One injector per condition kind. Use as a context manager around a condition block:
start() puts the load in place (and waits until it is), stop() removes it, also on errors."""

import queue
import subprocess
import threading

import psutil

from headroom.conditions.procs import kill_strays, spawn_worker, stop_worker
from headroom.schemas.common import Reading, measured, unavailable
from headroom.schemas.config import (
    BaselineCondition, Condition, CpuContention, MemoryPressure, ReducedCompute, VramPressure,
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
    raise TypeError(f"no injector for {condition!r}")


def _read_line(proc: subprocess.Popen, timeout_s: float) -> str:
    q: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: q.put(proc.stdout.readline() if proc.stdout else ""), daemon=True).start()
    try:
        return q.get(timeout=timeout_s).strip()
    except queue.Empty:
        return ""
