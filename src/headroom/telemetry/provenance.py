"""Facts about the machine and code, logged with every run."""

import platform
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import psutil

from headroom.schemas.common import Unavailable, measured, unavailable
from headroom.schemas.run import Provenance
from headroom.telemetry.sources import MB, NvmlSource


def collect_provenance(repo_root: Path, nvml: NvmlSource) -> Provenance:
    device = nvml.device()
    no_gpu = unavailable(nvml.unavailable.get("vram_used_mb", "no GPU"))
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    return Provenance(
        os=platform.platform(),
        cpu=_cpu_name(),
        cpu_count=psutil.cpu_count() or 0,
        ram_total_mb=psutil.virtual_memory().total / MB,
        gpu=device[0] if device else no_gpu,
        gpu_driver=device[1] if device else no_gpu,
        vram_total_mb=measured(device[2], "MB") if device else no_gpu,
        power_scheme=power_scheme(),
        python=sys.version.split()[0],
        headroom_version=version("headroom"),
        git_commit=commit,
        git_dirty=status if isinstance(status, Unavailable) else bool(status),
    )


def _git(repo_root: Path, *args: str) -> str | Unavailable:
    try:
        out = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return unavailable(f"git failed: {e}")
    if out.returncode != 0:
        return unavailable(f"git {' '.join(args)}: {out.stderr.strip()[:200] or 'failed'}")
    return out.stdout.strip()


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def power_scheme() -> str | Unavailable:
    if sys.platform != "win32":
        return unavailable("power scheme readout is only implemented for Windows")
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return unavailable(f"powercfg failed: {e}")
    match = re.search(r"\((.+)\)", out.stdout)
    return match.group(1) if match else unavailable(f"could not parse powercfg output: {out.stdout.strip()[:100]}")
