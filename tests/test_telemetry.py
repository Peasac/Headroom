import sys

import pytest
from conftest import FakeSource

from headroom.schemas.common import Measured, Unavailable
from headroom.schemas.telemetry import SUMMARY_NAMES, TelemetrySummary
from headroom.telemetry.sampler import TelemetrySampler, summarize
from headroom.telemetry.sources import NvmlSource, PsutilSource


def test_aggregations():
    samples = [
        {"sys_mem_used_mb": 100, "vram_used_mb": 700, "sys_cpu_pct": float(v), "battery_pct": 90 - i}
        for i, v in enumerate(range(1, 21))
    ]
    samples[5]["sys_mem_used_mb"] = 300
    s = summarize(samples, {}, interval_s=0.2, duration_s=4.0)
    r = s.readings
    assert r["start_sys_mem_used_mb"] == Measured(value=100, unit="MB")
    assert r["peak_sys_mem_used_mb"].value == 300
    assert r["mean_sys_cpu_pct"].value == pytest.approx(10.5)
    assert r["p95_sys_cpu_pct"].value == 19  # nearest rank of 1..20
    assert r["start_battery_pct"].value == 90 and r["end_battery_pct"].value == 71
    assert s.samples == 20


def test_every_reading_is_present_and_missing_sensors_say_why():
    s = summarize([{"sys_mem_used_mb": 1.0}], {"vram_used_mb": "no NVIDIA GPU"}, 0.2, 0.1)
    assert set(s.readings) == SUMMARY_NAMES
    assert s.readings["peak_vram_used_mb"] == Unavailable(reason="no NVIDIA GPU")
    # Not declared unavailable but never read: still unavailable, never zero.
    assert isinstance(s.readings["mean_sys_cpu_pct"], Unavailable)


def test_summary_rejects_missing_readings():
    s = summarize([], {}, 0.2, 0.0)
    with pytest.raises(ValueError):
        TelemetrySummary(**{**s.model_dump(), "readings": {}})


def test_sampler_polls_in_background_and_samples_at_exit():
    src = FakeSource([{"sys_mem_used_mb": 1.0}, {"sys_mem_used_mb": 5.0}, {"sys_mem_used_mb": 2.0}])
    import time
    with TelemetrySampler([src], interval_s=0.05) as sampler:
        time.sleep(0.2)
    s = sampler.summary()
    assert src.started
    assert s.samples >= 3
    assert s.readings["peak_sys_mem_used_mb"].value == 5.0
    assert s.duration_s >= 0.2


def test_short_run_still_gets_one_sample():
    with TelemetrySampler([FakeSource([{"sys_mem_used_mb": 1.0}])], interval_s=10) as sampler:
        pass
    assert sampler.summary().samples == 1


def test_broken_source_is_reported_not_fatal():
    good, bad = FakeSource([{"sys_mem_used_mb": 1.0}]), FakeSource([RuntimeError("sensor fell off")])
    with TelemetrySampler([good, bad], interval_s=10) as sampler:
        pass
    s = sampler.summary()
    assert s.readings["peak_sys_mem_used_mb"].value == 1.0
    assert "sensor fell off" in s.sampler_errors[0]


def test_psutil_source_without_backend_process_says_so():
    src = PsutilSource("no-such-process-xyz", "serve")
    assert "no process named" in src.unavailable["backend_rss_mb"]
    src.start()
    out = src.sample()
    assert out["sys_mem_used_mb"] > 0
    assert "backend_rss_mb" not in out


def test_nvml_source_without_library_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # makes `import pynvml` raise ImportError
    src = NvmlSource()
    assert src.unavailable == {m: "nvidia-ml-py is not installed" for m in NvmlSource.METRICS}
    assert src.sample() == {} and src.device() is None
