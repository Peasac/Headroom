from pydantic import Field, model_validator

from headroom.schemas.common import Frozen, Reading

# metric -> (unit, aggregations). Each aggregation becomes a summary reading named "<agg>_<metric>".
# vram_used_mb is device-wide: Windows (WDDM) does not report per-process GPU memory, so it
# includes other apps on the GPU. Compare against start_vram_used_mb, not zero.
METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "sys_mem_used_mb": ("MB", ("start", "peak")),
    # Ollama server + model runner processes. RSS is the Windows working set, which the OS trims
    # when idle; private is committed memory, steadier across runs.
    "backend_rss_mb": ("MB", ("start", "peak", "end")),
    "backend_private_mb": ("MB", ("start", "peak", "end")),
    "sys_cpu_pct": ("%", ("mean", "p95")),
    "backend_cpu_pct": ("% of all cores", ("mean", "p95")),
    "vram_used_mb": ("MB", ("start", "peak")),
    "gpu_util_pct": ("%", ("mean", "p95")),
    "gpu_temp_c": ("C", ("peak",)),
    "cpu_temp_c": ("C", ("peak",)),
    "battery_pct": ("%", ("start", "end")),
    "on_ac": ("bool", ("start", "end")),
}
SUMMARY_NAMES: frozenset[str] = frozenset(f"{agg}_{m}" for m, (_, aggs) in METRICS.items() for agg in aggs)


class TelemetrySummary(Frozen):
    """Device telemetry over one run. Every reading is always present: measured or unavailable."""

    interval_s: float
    samples: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    readings: dict[str, Reading]
    sampler_errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _all_readings_present(self) -> "TelemetrySummary":
        if set(self.readings) != SUMMARY_NAMES:
            missing, extra = SUMMARY_NAMES - set(self.readings), set(self.readings) - SUMMARY_NAMES
            raise ValueError(f"readings mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}")
        return self
