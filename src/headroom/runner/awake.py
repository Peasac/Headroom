"""Keep the machine awake during a session, and detect sleep inside a run.

keep_awake(): Windows SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) stops idle sleep
while a session runs (the display may still turn off). Lid close or a manual sleep can still
suspend the machine, so each run also measures it:

unbiased_s(): Windows QueryUnbiasedInterruptTime, which does not advance while the machine is
asleep. perf_counter minus unbiased time over a run = time spent suspended.
"""

import ctypes
import sys
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake():
    if sys.platform != "win32":
        yield False
        return
    k32 = ctypes.WinDLL("kernel32")
    k32.SetThreadExecutionState.restype = ctypes.c_uint32
    ok = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    try:
        yield ok
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


def unbiased_s() -> float | None:
    """Seconds of interrupt time excluding sleep, or None where unavailable."""
    if sys.platform != "win32":
        return None
    t = ctypes.c_ulonglong()
    if not ctypes.WinDLL("kernel32").QueryUnbiasedInterruptTime(ctypes.byref(t)):
        return None
    return t.value / 1e7
