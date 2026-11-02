"""Differentiable multi-step 3D rollouts with gradient checkpointing.

Backpropagating through N steps naively keeps every intermediate of every step
(a few dozen full 3D fields each), which exhausts memory after a handful of
steps on any useful grid. :func:`rollout3d` splits the trajectory into chunks
and wraps each in :func:`torch.utils.checkpoint.checkpoint`, so only the state at
chunk boundaries is kept and each chunk is recomputed once in the backward
pass: memory O(N / chunk + chunk) step-states for ~1.3x the compute (the 2D
design, ``aerolume.solvers.ns2d.rollout``).

The time step is fixed over the rollout: letting the CFL rule choose dt inside
a differentiated rollout would make dt a non-smooth function of the design.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .core import Ops3D, State3D, cfl_dt3d, step3d
from .geometry import Geometry3D


def _chunk(state: State3D, geom: Geometry3D, ops: Ops3D, dt: float, m: int) -> tuple[State3D, torch.Tensor]:
    out = []
    for _ in range(m):
        state, c = step3d(state, geom, ops, dt)
        out.append(c)
    return state, torch.stack(out)


def rollout3d(state: State3D, geom: Geometry3D, ops: Ops3D, n_steps: int, dt: float | None = None, *,
              checkpoint_every: int = 10) -> tuple[State3D, torch.Tensor]:
    """Advance ``n_steps`` differentiably; returns (final state, coefficients (n_steps, 6)).

    ``dt=None`` takes the CFL / viscous step of the initial state once, as a
    plain float. ``checkpoint_every=0`` disables checkpointing (fastest, most
    memory). Checkpointing only kicks in when autograd is recording.
    """
    if dt is None:
        dt = float(cfl_dt3d(state.u.detach(), state.v.detach(), state.w.detach(), ops))
    hist = []
    k = 0
    use_ckpt = checkpoint_every > 0 and torch.is_grad_enabled()
    while k < n_steps:
        m = min(checkpoint_every if checkpoint_every > 0 else n_steps, n_steps - k)
        if use_ckpt:
            state, c = checkpoint(_chunk, state, geom, ops, dt, m, use_reentrant=False)
        else:
            state, c = _chunk(state, geom, ops, dt, m)
        hist.append(c)
        k += m
    return state, torch.cat(hist) if hist else torch.zeros(0, 6, device=ops.device, dtype=ops.dtype)
