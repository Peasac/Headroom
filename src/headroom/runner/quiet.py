"""Quiet-machine check before each run: GPU utilization and temperature, AC power, power plan."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import psutil

from headroom.schemas.common import Measured, Reading, Unavailable, measured, unavailable
from headroom.schemas.config import QuietSettings
from headroom.schemas.run import QuietCheck
from headroom.telemetry.provenance import power_scheme
from headroom.telemetry.sources import NvmlSource

POLL_S = 0.1


@dataclass(frozen=True)
class QuietReading:
    gpu_util_pct: Reading
    gpu_temp_c: Reading
    on_ac: Reading
    power_plan: str | Unavailable


def evaluate(
    r: QuietReading, s: QuietSettings, ignore: frozenset[str] = frozenset()
) -> tuple[list[str], list[str], bool]:
    """(failures, unchecked, any failure is transient i.e. may clear by waiting).

    ignore: checks the current condition causes to fail by design (e.g. gpu_util under VRAM pressure).
    They are still read and recorded, and listed as unchecked."""
    failures, unchecked, transient = [], [], False
    if "gpu_util" in ignore:
        unchecked.append("gpu_util (caused by condition)")
    elif isinstance(r.gpu_util_pct, Measured):
        if r.gpu_util_pct.value >= s.max_gpu_util_pct:
            failures.append(f"GPU util {r.gpu_util_pct.value:.0f}% >= {s.max_gpu_util_pct:g}%")
            transient = True
    else:
        unchecked.append("gpu_util")
    if "gpu_temp" in ignore:
        unchecked.append("gpu_temp (caused by condition)")
    elif isinstance(r.gpu_temp_c, Measured):
        if r.gpu_temp_c.value >= s.max_gpu_temp_c:
            failures.append(f"GPU temp {r.gpu_temp_c.value:.0f}C >= {s.max_gpu_temp_c:g}C")
            transient = True
    else:
        unchecked.append("gpu_temp")
    if s.require_ac:
        if isinstance(r.on_ac, Measured):
            if r.on_ac.value != 1:
                failures.append("not on AC power")
        else:
            unchecked.append("on_ac")
    if s.power_plan is not None:
        if isinstance(r.power_plan, str):
            if r.power_plan != s.power_plan:
                failures.append(f"power plan {r.power_plan!r} != {s.power_plan!r}")
        else:
            unchecked.append("power_plan")
    return failures, unchecked, transient


def quiet_check(
    settings: QuietSettings,
    read: Callable[[], QuietReading],
    sleep: Callable[[float], None] = time.sleep,
    ignore: frozenset[str] = frozenset(),
) -> QuietCheck:
    """Read; if it fails, wait and retry. Stops early when no failure can clear by waiting.
    Never raises: a check that does not pass returns passed=False and the run goes ahead, flagged."""
    waited = 0.0
    attempt = 0
    while True:
        attempt += 1
        r = read()
        failures, unchecked, transient = evaluate(r, settings, ignore)
        if not failures or not transient or attempt >= settings.max_attempts:
            return QuietCheck(
                passed=not failures, attempts=attempt, waited_s=waited, gpu_util_pct=r.gpu_util_pct,
                gpu_temp_c=r.gpu_temp_c, on_ac=r.on_ac, power_plan=r.power_plan,
                failures=tuple(failures), unchecked=tuple(unchecked),
            )
        sleep(settings.wait_s)
        waited += settings.wait_s


def machine_reader(nvml: NvmlSource, window_s: float) -> Callable[[], QuietReading]:
    """Reads the real machine. GPU utilization is averaged over window_s (it is bursty at idle)."""

    def read() -> QuietReading:
        utils, temp = [], None
        end = time.perf_counter() + window_s
        while True:
            s = nvml.sample()
            if "gpu_util_pct" in s:
                utils.append(s["gpu_util_pct"])
            temp = s.get("gpu_temp_c", temp)
            if time.perf_counter() >= end:
                break
            time.sleep(POLL_S)
        gpu_reason = nvml.unavailable.get("gpu_util_pct", "no GPU reading")
        return QuietReading(
            gpu_util_pct=measured(sum(utils) / len(utils), "%") if utils else unavailable(gpu_reason),
            gpu_temp_c=measured(temp, "C") if temp is not None else unavailable(gpu_reason),
            on_ac=_on_ac(),
            power_plan=power_scheme(),
        )

    return read


def _on_ac() -> Reading:
    b = psutil.sensors_battery()
    if b is None:
        return unavailable("no battery reported by the OS")
    if b.power_plugged is None:
        return unavailable("OS did not report AC power state")
    return measured(1 if b.power_plugged else 0, "bool")
