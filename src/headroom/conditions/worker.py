"""Worker process entry point: python -m headroom.conditions.worker <harness_pid> <mode> [args].

Modes:
  cpu                                         busy loop on one core
  memory <target_avail_mb> <min_avail_mb> <max_mb>  RAM ballast with a safety valve
  vram <mb> <touch_interval_s>                 GPU ballast via the CUDA driver API (ctypes)

Every mode exits on its own once the harness pid is gone (checked about once a second).
Workers print "READY ..." on stdout when the load is in place, or "ERROR ..." and exit 2.
"""

import ctypes
import sys
import time

import psutil

MB = 2**20
CHUNK_MB = 128
PAGE = 4096


def main(argv: list[str]) -> int:
    parent, mode, args = int(argv[1]), argv[2], argv[3:]

    def alive() -> bool:
        return psutil.pid_exists(parent)

    if mode == "cpu":
        return _cpu(alive)
    if mode == "memory":
        return _memory(alive, int(args[0]), int(args[1]), int(args[2]))
    if mode == "vram":
        return _vram(alive, int(args[0]), float(args[1]))
    print(f"ERROR unknown mode {mode!r}", flush=True)
    return 2


def _cpu(alive) -> int:
    print("READY", flush=True)
    x, next_check = 0, time.monotonic() + 1.0
    while True:
        for _ in range(200_000):
            x += 1
        if time.monotonic() >= next_check:
            if not alive():
                return 0
            next_check = time.monotonic() + 1.0


def _touch(buf: bytearray) -> None:
    buf[::PAGE] = b"\x01" * ((len(buf) + PAGE - 1) // PAGE)  # one write per page: makes it resident


def _memory(alive, target_avail_mb: int, min_avail_mb: int, max_mb: int) -> int:
    chunks: list[bytearray] = []
    ready = False
    while alive():
        avail = psutil.virtual_memory().available // MB
        held = len(chunks) * CHUNK_MB
        if avail < min_avail_mb and chunks:
            chunks.pop()  # safety valve: someone else needs memory
            continue
        if avail > target_avail_mb + CHUNK_MB and held + CHUNK_MB <= max_mb:
            buf = bytearray(CHUNK_MB * MB)
            _touch(buf)
            chunks.append(buf)
            continue
        if not ready:  # stopped growing: at the target, or at max_mb
            print(f"READY {held}", flush=True)
            ready = True
        for buf in chunks:
            _touch(buf)  # keep the ballast in the working set so the OS trims others instead
        time.sleep(0.25)
    return 0


def _vram(alive, mb: int, interval_s: float) -> int:
    try:
        cu = ctypes.WinDLL("nvcuda") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        print(f"ERROR CUDA driver library not found: {e}", flush=True)
        return 2
    u64, sz = ctypes.c_uint64, ctypes.c_size_t
    cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(u64), sz]
    cu.cuMemsetD8_v2.argtypes = [u64, ctypes.c_ubyte, sz]
    cu.cuMemFree_v2.argtypes = [u64]

    def check(code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} failed with CUDA error {code}")

    n = mb * MB
    ptr, ctx = u64(), ctypes.c_void_p()
    try:
        check(cu.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        check(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
        check(cu.cuMemAlloc_v2(ctypes.byref(ptr), n), f"cuMemAlloc({mb} MB)")
        check(cu.cuMemsetD8_v2(ptr, 1, n), "cuMemsetD8")
        check(cu.cuCtxSynchronize(), "cuCtxSynchronize")
    except RuntimeError as e:
        print(f"ERROR {e}", flush=True)
        return 2
    print(f"READY {mb}", flush=True)
    try:
        while alive():
            cu.cuMemsetD8_v2(ptr, 1, n)  # rewrite so the driver keeps it resident in VRAM
            cu.cuCtxSynchronize()
            time.sleep(interval_s)
    finally:
        cu.cuMemFree_v2(ptr)
        cu.cuCtxDestroy_v2(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
