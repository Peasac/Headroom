"""Worker processes that cannot outlive the harness.

Three layers of cleanup:
1. Injectors stop their workers in a finally block (normal exit and Python exceptions).
2. On Windows every worker is put in a Job Object with KILL_ON_JOB_CLOSE. The job handle belongs
   to the harness process, so when the harness dies for any reason (crash, kill, closed terminal)
   Windows closes the handle and kills every worker.
3. Each worker polls the harness pid about once a second and exits when it is gone.
kill_strays() removes workers left over from anything else.
"""

import ctypes
import os
import subprocess
import sys

import psutil

WORKER_MODULE = "headroom.conditions.worker"
_JOB = None


def spawn_worker(*args: str) -> subprocess.Popen:
    """Start a worker as one process. In a Windows venv, sys.executable is a launcher that starts the
    real interpreter as a child, so the pid we hold would not be the worker, and terminating it
    would leave the worker running. Start the base interpreter with our import path instead."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    exe = getattr(sys, "_base_executable", None) or sys.executable
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    proc = subprocess.Popen(
        [exe, "-m", WORKER_MODULE, str(os.getpid()), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags, env=env,
    )
    if sys.platform == "win32":
        _assign_to_job(proc)
    return proc


def stop_worker(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    """Stop the worker and anything it started."""
    try:
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()
    except psutil.Error:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout_s)
    for stream in (proc.stdout, proc.stderr):
        if stream:
            stream.close()


def kill_strays() -> int:
    """Kill worker processes that are not ours (e.g. left by an earlier crash). Returns the count."""
    killed = 0
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if WORKER_MODULE in cmd and (len(cmd) < 4 or cmd[cmd.index(WORKER_MODULE) + 1] != str(me)):
                p.kill()
                killed += 1
        except (psutil.Error, ValueError, IndexError):
            continue
    return killed


# --- Windows Job Object (ctypes; no extra packages) -------------------------------------------

def _assign_to_job(proc: subprocess.Popen) -> None:
    global _JOB
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    if _JOB is None:
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = _ExtendedLimitInfo()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # ExtendedLimitInformation
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        _JOB = job  # deliberately never closed: it closes when this process ends, killing the workers
    if not k32.AssignProcessToJobObject(_JOB, int(proc._handle)):  # type: ignore[attr-defined]
        err = ctypes.get_last_error()
        proc.kill()
        raise OSError(err, "AssignProcessToJobObject failed; worker killed")


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _BasicLimitInfo(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInfo(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInfo), ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]
