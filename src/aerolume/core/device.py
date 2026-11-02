"""Compute device and precision selection.

"auto" picks CUDA when present, then Apple MPS, then CPU. Precision "auto" is
float32 on GPUs (bandwidth-bound, and the projection holds divergence to ~1e-6,
far below anything visible in forces) and float64 on CPU (where it costs little
and makes validation runs bit-for-bit stable).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class DeviceInfo:
    kind: str           # "cuda" | "mps" | "cpu"
    name: str           # human readable, e.g. "NVIDIA GeForce RTX 4060 Ti (8.0 GB)"
    memory_gb: float    # 0 when unknown
    threads: int        # CPU threads torch uses


def _torch():
    import torch  # lazy: keeps model/core importable without torch
    return torch


@cache
def available() -> tuple[str, ...]:
    """Device kinds usable in this process, best first."""
    torch = _torch()
    kinds = []
    if torch.cuda.is_available():
        kinds.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        kinds.append("mps")
    kinds.append("cpu")
    return tuple(kinds)


def resolve(device: str | None = "auto") -> str:
    """Map "auto"/None/"cuda"/... to a concrete torch device string that exists.

    A CUDA index is checked too: a simulation saved with "cuda:1" on a two-GPU
    machine must fall back (the planner catches the RuntimeError and warns) on
    a machine with one GPU, not fail later inside ``get_device_properties``
    with "Invalid device id".
    """
    kinds = available()
    if device in (None, "", "auto"):
        return kinds[0]
    dev, sep, index = device.partition(":")
    if dev not in kinds:
        raise RuntimeError(f"device {device!r} not available (have: {', '.join(kinds)})")
    if dev == "cuda" and sep:
        try:
            i = int(index)
        except ValueError:
            raise RuntimeError(f"device {device!r} is not a valid CUDA device name (use cuda or cuda:N)") from None
        count = _torch().cuda.device_count()
        if not 0 <= i < count:
            raise RuntimeError(f"device {device!r} not available (this machine has {count} CUDA "
                               f"device{'s' if count != 1 else ''})")
    return device


def resolve_dtype(precision: str | None, device: str):
    """torch dtype for a precision setting on a resolved device."""
    torch = _torch()
    if precision in (None, "", "auto"):
        return torch.float32 if device.startswith(("cuda", "mps")) else torch.float64
    if precision == "float64" and device.startswith("mps"):
        raise RuntimeError("MPS does not support float64")
    return {"float32": torch.float32, "float64": torch.float64}[precision]


@cache
def system_memory_gb() -> float:
    """Installed physical memory [GiB], 0 when it cannot be determined.

    The planner refuses CPU grids that would not fit (a "Cells per chord" of
    4096 asked for 806 M cells; allocating a fraction of that pushes a machine
    into heavy paging long before any error appears). Standard library only.
    """
    try:
        import sys
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return 0.0
            return st.ullTotalPhys / 1024**3
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return 0.0


@cache
def info(device: str = "auto") -> DeviceInfo:
    """Description of a device for the status bar and reports."""
    torch = _torch()
    dev = resolve(device)
    if dev.startswith("cuda"):
        p = torch.cuda.get_device_properties(torch.device(dev))
        gb = p.total_memory / 1024**3
        return DeviceInfo("cuda", f"{p.name} ({gb:.1f} GB)", gb, torch.get_num_threads())
    if dev.startswith("mps"):
        return DeviceInfo("mps", "Apple GPU (MPS)", 0.0, torch.get_num_threads())
    import platform
    cpu = platform.processor() or platform.machine() or "CPU"
    return DeviceInfo("cpu", f"CPU · {cpu} · {torch.get_num_threads()} threads", 0.0, torch.get_num_threads())
