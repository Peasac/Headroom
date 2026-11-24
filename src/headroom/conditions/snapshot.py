"""Machine readings taken just before each run, recorded with the run's condition."""

import asyncio

import psutil

from headroom.backends.base import Backend
from headroom.schemas.common import Reading, measured, unavailable
from headroom.schemas.config import ModelSpec
from headroom.telemetry.sources import NvmlSource

MB = 2**20
CPU_WINDOW_S = 0.5


async def start_readings(backend: Backend, model: ModelSpec, nvml: NvmlSource | None) -> dict[str, Reading]:
    out: dict[str, Reading] = {
        "sys_mem_available_mb": measured(psutil.virtual_memory().available / MB, "MB"),
        "sys_cpu_pct": measured(await asyncio.to_thread(psutil.cpu_percent, CPU_WINDOW_S), "%"),
    }
    gpu = nvml.sample() if nvml is not None else {}
    reason = (nvml.unavailable.get("vram_used_mb") if nvml is not None else None) or "no GPU source"
    out["vram_used_mb"] = measured(gpu["vram_used_mb"], "MB") if "vram_used_mb" in gpu else unavailable(reason)
    loaded = getattr(backend, "loaded", None)
    if loaded is None:
        out["model_size_mb"] = out["model_vram_mb"] = unavailable("backend does not report loaded models")
    else:
        out["model_size_mb"], out["model_vram_mb"] = await loaded(model)
    return out
