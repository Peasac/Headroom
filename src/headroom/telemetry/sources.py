"""Sensor sources. Each declares up front which metrics it cannot provide, and why."""

from typing import Protocol

import psutil

MB = 2**20


class Source(Protocol):
    unavailable: dict[str, str]  # metric -> reason it can never be read on this machine

    def start(self) -> None: ...

    def sample(self) -> dict[str, float]:
        """Values readable right now. A metric missing from one sample is a transient gap."""
        ...

    def close(self) -> None: ...


class PsutilSource:
    """System memory/CPU, the backend server's process tree, battery, CPU temperature."""

    def __init__(self, process_prefix: str, process_arg: str) -> None:
        self.unavailable: dict[str, str] = {}
        self._ncpu = psutil.cpu_count() or 1
        self._procs: dict[int, psutil.Process] = {}

        if not hasattr(psutil, "sensors_temperatures"):
            self.unavailable["cpu_temp_c"] = "psutil cannot read CPU temperature on this OS"
        battery = psutil.sensors_battery()
        if battery is None:
            self.unavailable["battery_pct"] = self.unavailable["on_ac"] = "no battery reported by the OS"
        elif battery.power_plugged is None:
            self.unavailable["on_ac"] = "OS did not report AC power state"

        self._has_private = hasattr(psutil.Process().memory_info(), "private")
        if not self._has_private:
            self.unavailable["backend_private_mb"] = "psutil reports process private bytes only on Windows"

        self._root = _find_process(process_prefix, process_arg)
        if self._root is None:
            reason = f"no process named {process_prefix}* with {process_arg!r} in its command line"
            for m in ("backend_rss_mb", "backend_private_mb", "backend_cpu_pct"):
                self.unavailable[m] = reason

    def start(self) -> None:
        psutil.cpu_percent(None)  # prime: the next call reports usage since now
        self._procs = {}
        self._refresh()

    def _refresh(self) -> set[int]:
        """Track the backend tree (server + model runners). Returns pids first seen now."""
        try:
            tree = [self._root, *self._root.children(recursive=True)] if self._root else []
        except psutil.Error:
            tree = []
        current, new = {}, set()
        for p in tree:
            if p.pid in self._procs:
                current[p.pid] = self._procs[p.pid]
                continue
            try:
                p.cpu_percent(None)  # prime; its first real reading comes next sample
            except psutil.Error:
                continue
            current[p.pid] = p
            new.add(p.pid)
        self._procs = current
        return new

    def sample(self) -> dict[str, float]:
        vm = psutil.virtual_memory()
        out = {"sys_mem_used_mb": (vm.total - vm.available) / MB, "sys_cpu_pct": psutil.cpu_percent(None)}

        if self._root is not None:
            new = self._refresh()
            rss, private, cpu, seen = 0, 0, 0.0, False
            for pid, p in self._procs.items():
                try:
                    mi = p.memory_info()
                    rss += mi.rss
                    private += getattr(mi, "private", 0)
                    if pid not in new:
                        cpu += p.cpu_percent(None)
                    seen = True
                except psutil.Error:
                    continue
            if seen:
                out["backend_rss_mb"] = rss / MB
                out["backend_cpu_pct"] = cpu / self._ncpu
                if self._has_private:
                    out["backend_private_mb"] = private / MB

        if "battery_pct" not in self.unavailable:
            b = psutil.sensors_battery()
            if b is not None:
                out["battery_pct"] = float(b.percent)
                if b.power_plugged is not None:
                    out["on_ac"] = 1.0 if b.power_plugged else 0.0

        if "cpu_temp_c" not in self.unavailable:
            temps = [t.current for group in psutil.sensors_temperatures().values() for t in group]
            if temps:
                out["cpu_temp_c"] = max(temps)
        return out

    def close(self) -> None:
        self._procs = {}


class NvmlSource:
    """NVIDIA GPU via NVML: device-wide VRAM used, utilization, temperature."""

    METRICS = ("vram_used_mb", "gpu_util_pct", "gpu_temp_c")

    def __init__(self, device_index: int = 0) -> None:
        self.unavailable: dict[str, str] = {}
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml = pynvml
        except ImportError:
            reason = "nvidia-ml-py is not installed"
        except Exception as e:  # NVMLError: no driver, no device, ...
            reason = f"NVML unavailable: {e}"
        if self._handle is None:
            self.unavailable = {m: reason for m in self.METRICS}

    def device(self) -> tuple[str, str, float] | None:
        """(name, driver version, total VRAM in MB), or None without NVML."""
        if self._handle is None:
            return None
        n = self._nvml
        return (
            n.nvmlDeviceGetName(self._handle),
            n.nvmlSystemGetDriverVersion(),
            n.nvmlDeviceGetMemoryInfo(self._handle).total / MB,
        )

    def start(self) -> None:
        pass

    def sample(self) -> dict[str, float]:
        if self._handle is None:
            return {}
        n, h = self._nvml, self._handle
        readers = {
            "vram_used_mb": lambda: n.nvmlDeviceGetMemoryInfo(h).used / MB,
            "gpu_util_pct": lambda: n.nvmlDeviceGetUtilizationRates(h).gpu,
            "gpu_temp_c": lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
        }
        out = {}
        for metric, read in readers.items():
            try:
                out[metric] = float(read())
            except n.NVMLError:
                continue
        return out

    def close(self) -> None:
        if self._handle is not None:
            self._nvml.nvmlShutdown()
            self._handle = None


def _find_process(prefix: str, arg: str) -> psutil.Process | None:
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            if name.startswith(prefix.lower()) and arg in (p.info["cmdline"] or []):
                return p
        except psutil.Error:
            continue
    return None
