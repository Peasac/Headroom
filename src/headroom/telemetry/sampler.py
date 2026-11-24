"""Background sampler: polls sources on a thread while a run executes, then summarizes."""

import threading
import time
from collections.abc import Callable, Sequence

from headroom.schemas.common import Reading, measured, unavailable
from headroom.schemas.telemetry import METRICS, TelemetrySummary
from headroom.stats import nearest_rank
from headroom.telemetry.sources import Source

MAX_ERRORS_KEPT = 5


class TelemetrySampler:
    """Use as a context manager around one run. Samples every interval_s and once more at exit,
    so even a run shorter than the interval has one sample."""

    def __init__(self, sources: Sequence[Source], interval_s: float, clock: Callable[[], float] = time.perf_counter):
        self._sources = list(sources)
        self._interval = interval_s
        self._clock = clock
        self._samples: list[dict[str, float]] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = self._t1 = 0.0

    def __enter__(self) -> "TelemetrySampler":
        for s in self._sources:
            s.start()
        self._t0 = self._clock()
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._take()
        self._t1 = self._clock()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._take()

    def _take(self) -> None:
        merged: dict[str, float] = {}
        for s in self._sources:
            try:
                merged.update(s.sample())
            except Exception as e:  # a broken sensor must not kill the run; it is reported instead
                if len(self._errors) < MAX_ERRORS_KEPT:
                    self._errors.append(f"{type(s).__name__}: {e!r}")
        self._samples.append(merged)

    def summary(self) -> TelemetrySummary:
        declared = {m: r for s in self._sources for m, r in s.unavailable.items()}
        return summarize(self._samples, declared, self._interval, self._t1 - self._t0, tuple(self._errors))


def summarize(
    samples: Sequence[dict[str, float]],
    declared_unavailable: dict[str, str],
    interval_s: float,
    duration_s: float,
    errors: tuple[str, ...] = (),
) -> TelemetrySummary:
    readings: dict[str, Reading] = {}
    for metric, (unit, aggs) in METRICS.items():
        values = [s[metric] for s in samples if metric in s]
        for agg in aggs:
            name = f"{agg}_{metric}"
            if metric in declared_unavailable:
                readings[name] = unavailable(declared_unavailable[metric])
            elif not values:
                readings[name] = unavailable("no sensor value was read during the run")
            else:
                readings[name] = measured(_aggregate(agg, values), unit)
    return TelemetrySummary(
        interval_s=interval_s, samples=len(samples), duration_s=duration_s, readings=readings, sampler_errors=errors
    )


def _aggregate(agg: str, values: list[float]) -> float:
    if agg == "start":
        return values[0]
    if agg == "end":
        return values[-1]
    if agg == "peak":
        return max(values)
    if agg == "mean":
        return sum(values) / len(values)
    if agg == "p95":
        return nearest_rank(values, 0.95)
    raise ValueError(f"unknown aggregation {agg!r}")
