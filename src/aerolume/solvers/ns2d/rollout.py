"""Differentiable multi-step rollouts with gradient checkpointing.

Backpropagating through N solver steps naively stores every intermediate of
every step (a few dozen full fields each), which exhausts memory after a few
hundred steps on a fine grid. :func:`rollout` splits the trajectory into chunks
and wraps each chunk in :func:`torch.utils.checkpoint.checkpoint`, so only the
state at chunk boundaries is kept and each chunk is recomputed once during the
backward pass: memory O(N / chunk + chunk) fields for ~1.3x the compute.

The time step is held fixed over the rollout. Letting the CFL rule pick dt
inside a differentiated rollout would make dt itself a (non-smooth) function of
the design through ``max|u|``, which only adds noise to sensitivities.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .core import Ops, State, cfl_dt, step
from .geometry import Geometry


def _chunk(state: State, geom: Geometry, ops: Ops, dt: float, m: int) -> tuple[State, torch.Tensor]:
    out = []
    for _ in range(m):
        state, c = step(state, geom, ops, dt)
        out.append(c)
    return state, torch.stack(out)


def rollout(state: State, geom: Geometry, ops: Ops, n_steps: int, dt: float | None = None, *,
            checkpoint_every: int = 10) -> tuple[State, torch.Tensor]:
    """Advance ``n_steps`` differentiably; returns (final state, coefficients (n_steps, 3)).

    ``dt=None`` takes the CFL / viscous step of the initial state once, as a
    plain float. ``checkpoint_every=0`` disables checkpointing (fastest, most
    memory). Checkpointing only kicks in when autograd is recording.
    """
    if dt is None:
        dt = float(cfl_dt(state.u.detach(), state.v.detach(), ops))
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
    return state, torch.cat(hist) if hist else torch.zeros(0, 3, device=ops.device, dtype=ops.dtype)
