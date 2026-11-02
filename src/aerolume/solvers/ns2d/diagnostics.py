"""Solver diagnostics (AeroJAX-style) and the PID-controlled time step.

DESIGN.md 10.4 asks for the numbers an engineer watches while a run converges:
the divergence the projection leaves, how much each field still changes per
step, the CFL number, the step size, the pressure-solve residual and the
throughput - recorded without synchronising the GPU with the host every step.

* :func:`diagnostics_row` computes one row of :data:`DIAG_COLUMNS` from the
  states before and after a step (device tensors only). The live solver
  evaluates it on the last step of every block of steps (every
  ``DIAG_EVERY`` steps, the CUDA-graph block), writes it into a device ring
  buffer inside the captured graph, and copies the ring to the host once per
  ``advance`` - so recording costs about one tenth of a step's reductions per
  step (measured overhead in theory doc 20.3).
* The pressure "residual" is ``||L psi - f|| / ||f||`` of the last stage's
  Poisson problem. The solver's Poisson solve is direct (exact DCT), so this is
  a round-off number: it proves the solve is exact on this grid rather than
  measuring an iteration error, and it would expose a broken compatibility
  condition (unbalanced sources in a closed domain) immediately.
* :func:`pid_dt` is the ``adaptive_dt="pid"`` step: a PID controller in log
  space that steers the CFL number of the step toward ``PID_TARGET`` x the
  configured CFL limit, never above the instantaneous stability limit and never
  growing faster than ``PID_MAX_GROWTH`` per step. The plain CFL rule jumps with
  every change of ``max |u|`` (for example when a vortex passes the probe of
  the maximum); the PID step changes smoothly, which keeps time-step-dependent
  numerical damping and output sampling steady (theory doc 20.2).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

#: diagnostics are recorded on the last step of every block of this many steps
DIAG_EVERY = 10
#: columns of one diagnostics row (the ring stores "step" first, filled by the solver)
DIAG_COLUMNS = ("step", "t", "dt", "cfl", "div_rms", "div_max", "p_residual",
                "du_l2", "du_rms", "du_max", "du_rel",
                "dv_l2", "dv_rms", "dv_max", "dv_rel",
                "dp_l2", "dp_rms", "dp_max", "dp_rel",
                "dtheta_l2", "dtheta_rms", "dtheta_max", "dtheta_rel")
#: PID step: target CFL as a fraction of the configured limit, gains (log space), growth bound
PID_TARGET = 0.9
PID_GAINS = (0.10, 0.25, 0.02)        # (kP, kI, kD)
PID_MAX_GROWTH = 1.03


def _acc_dtype(t: torch.Tensor) -> torch.dtype:
    return torch.float32 if t.device.type == "mps" else torch.float64


def _change(new: torch.Tensor, old: torch.Tensor, h: float) -> list[torch.Tensor]:
    """[L2, RMS, max, relative] of the per-step change of one field (domain norms).

    Three reductions per field (norms accumulate in float64 without a cast pass)."""
    d = new - old
    acc = _acc_dtype(d)
    nrm = torch.linalg.vector_norm(d, dtype=acc)
    l2 = nrm * h
    rms = nrm / (d.numel() ** 0.5)
    mx = d.abs().amax().to(acc)
    ref = torch.linalg.vector_norm(new, dtype=acc) * h
    rel = l2 / ref.clamp_min(1e-300 if acc == torch.float64 else 1e-30)
    return [l2, rms, mx, rel]


def diagnostics_row(old, new, psi: torch.Tensor | None, rhs: torch.Tensor | None, ops, source=None) -> torch.Tensor:
    """One diagnostics row (without "step"): float64 device tensor of len(DIAG_COLUMNS) - 1.

    ``old`` / ``new`` are the solver states before and after the step, ``psi``
    and ``rhs`` the last stage's pressure potential and Poisson right-hand side,
    ``source`` the mass source of ports (the divergence the velocity should have).
    """
    h = ops.h
    div = (new.u[:, 1:] - new.u[:, :-1] + new.v[1:] - new.v[:-1]) * (1.0 / h)
    if source is not None:
        div = div - source
    acc = _acc_dtype(div)
    div_rms = torch.linalg.vector_norm(div, dtype=acc) / (div.numel() ** 0.5)
    div_max = div.abs().amax().to(acc)
    dt = new.dt.to(acc)
    umax = new.u.abs().amax().to(acc) + new.v.abs().amax().to(acc)
    cfl = dt * umax / h
    if psi is not None and rhs is not None:
        r = ops.poisson.laplacian(psi) - rhs
        res = torch.linalg.vector_norm(r, dtype=acc) / torch.linalg.vector_norm(rhs, dtype=acc).clamp_min(1e-30)
    else:
        res = torch.zeros((), dtype=acc, device=div.device)
    vals = [new.t.to(acc), dt, cfl, div_rms, div_max, res]
    vals += _change(new.u, old.u, h)
    vals += _change(new.v, old.v, h)
    vals += _change(new.p, old.p, h)
    if getattr(new, "theta", None) is not None and getattr(old, "theta", None) is not None:
        vals += _change(new.theta, old.theta, h)
    else:
        nan = torch.full((), float("nan"), dtype=acc, device=div.device)
        vals += [nan, nan, nan, nan]
    return torch.stack([v.reshape(()) for v in vals])


def pid_dt(dt_limit: torch.Tensor, dt_prev: torch.Tensor, ctrl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PID-controlled step (device scalars only): returns (dt, new ctrl).

    The controlled error is ``e = log(PID_TARGET dt_limit / dt_prev)``: zero when
    the previous step, applied to the current velocity, would sit exactly at the
    target CFL. ``ctrl[0:2]`` holds the previous two errors. The first step
    (dt_prev = 0) takes the target directly.
    """
    kp, ki, kd = PID_GAINS
    target = PID_TARGET * dt_limit
    first = dt_prev <= 0
    safe_prev = torch.where(first, target, dt_prev)
    e = torch.log(target / safe_prev)
    e1, e2 = ctrl[0], ctrl[1]
    step = ki * e + kp * (e - e1) + kd * (e - 2.0 * e1 + e2)
    grow = torch.exp(step.clamp(max=math.log(PID_MAX_GROWTH)))
    dt = torch.where(first, target, torch.minimum(safe_prev * grow, dt_limit))
    e = torch.where(first, torch.zeros_like(e), e)
    ctrl = torch.cat([torch.stack([e, e1]), ctrl[2:]])
    return dt, ctrl


def write_csv(path, columns: list[str], rows: np.ndarray) -> Path:
    """Write a diagnostics table (rows x columns) as CSV with a header."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([f"{v:.9g}" for v in r])
    return p


__all__ = ["DIAG_COLUMNS", "DIAG_EVERY", "PID_GAINS", "PID_MAX_GROWTH", "PID_TARGET", "diagnostics_row", "pid_dt",
           "write_csv"]
