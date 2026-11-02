"""Subnormal (denormal) handling of the CPU floating-point unit, switched per block of code.

The CPU solvers flush subnormals to zero while they step: the decaying tails of
smooth fields (eddy viscosity, dye, wakes) otherwise reach the subnormal range,
where x86 arithmetic is ~100x slower, and flushing them cannot change any result
that matters at single or double precision. ``torch.set_flush_denormal`` sets
the flush-to-zero / denormals-are-zero bits of the *calling thread* and stays
set until someone clears it -- and other numerical code on that thread is not
built for it: scipy's ``cKDTree`` then recurses until the native stack
overflows, a fatal process crash rather than an exception (reproduced by
``geometry3d.design_from_meshes`` on three boxes after a CPU run). So:

* :func:`flush_denormal` switches flushing on for a block and restores the
  thread's previous mode afterwards. Every CPU stepping loop (``FlowSolver``,
  ``FlowSolver3D``, the sensitivity sweeps, shape optimisation) runs inside it,
  so no Aerolume code leaves the mode changed.
* :func:`ieee_subnormals` does the opposite for code that needs IEEE behaviour
  (the k-d trees of :mod:`aerolume.geometry3d`), in case a caller -- a user
  script, another library -- left flushing on.

torch has no getter for the mode, so :func:`flushing_subnormals` probes it: the
product of a subnormal and 1.0 is zero exactly when the bits are set. The probe
uses numpy (same thread, same SSE control register) so it needs no torch import;
it costs about a microsecond.
"""
from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator

import numpy as np

_SUBNORMAL = np.float64(1e-310)
_ONE = np.float64(1.0)

__all__ = ["flush_denormal", "flushing_subnormals", "ieee_subnormals"]


def flushing_subnormals() -> bool:
    """Whether subnormals are flushed to zero on the calling thread right now."""
    return float(_SUBNORMAL * _ONE) == 0.0


@contextlib.contextmanager
def flush_denormal(enabled: bool = True) -> Iterator[None]:
    """Flush subnormals to zero on this thread inside the block (``enabled``), then
    restore the previous mode. Nested blocks are fine; ``enabled=False`` does nothing
    (GPU runs: the GPU kernels are not affected by the CPU's control register)."""
    if not enabled:
        yield
        return
    import torch

    before = flushing_subnormals()
    if not before:
        torch.set_flush_denormal(True)
    try:
        yield
    finally:
        if not before:
            torch.set_flush_denormal(False)


@contextlib.contextmanager
def ieee_subnormals() -> Iterator[None]:
    """Run the block with IEEE subnormals on this thread (scipy k-d trees need them),
    restoring flush-to-zero afterwards if it was on. Never imports torch: if the mode is
    on, torch (the only library here that sets it) is already loaded."""
    torch = sys.modules.get("torch")
    if torch is None or not flushing_subnormals():
        yield
        return
    torch.set_flush_denormal(False)
    try:
        yield
    finally:
        torch.set_flush_denormal(True)
