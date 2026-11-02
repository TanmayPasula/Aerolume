"""Experimental: shape optimisation driven by the autograd CFD sensitivity.

:func:`optimize_shape_cfd` improves any 2D design (airfoils at low Reynolds
numbers, bluff bodies, vehicle side profiles) with the gradient that
:mod:`aerolume.optimize.sensitivity` computes through the Navier-Stokes
solver -- the AeroJAX idea: one reverse sweep gives the derivative with
respect to every surface point, so the cost of an iteration does not depend
on the number of design variables.

Method
------
* **Design variables:** amplitudes ``a_k`` of smooth normal-displacement modes
  (periodic Gaussian bumps along each outline,
  :func:`aerolume.geometry.normal_basis`); the displacement of the refined
  outline is ``d = sum_k a_k phi_k``. Smooth modes are what keeps the outline
  free of grid-scale wiggles that the per-vertex gradient alone would create
  (it aliases with the grid, see the sensitivity module). Corners sharper than
  30 degrees (an airfoil's trailing edge, a car's sharp edges) are held fixed,
  with the modes fading in over ~1.5 cells (:func:`corner_taper`): the normal is
  undefined there, and moving a sharp trailing edge along its bisector splits
  it into a fishtail. Wheels keep their shape.
* **Gradient:** dJ/da_k = sum_i dJ/dd_i phi_k(i) from one reverse sweep through
  the sensitivity window (the same windowed objective as the sensitivity map).
* **Area preservation** (``keep_area``): a normal displacement changes the
  enclosed area by ``sum_i d_i ds_i`` to first order, a linear function of the
  amplitudes; the step is projected onto the plane where that is zero, so the
  optimiser re-distributes material instead of simply removing it, and a small
  uniform offset then restores the exact polygon area (the curvature term).
* **Step:** steepest descent with a fixed largest surface movement per
  iteration (default 0.75 cells), accepted when the windowed objective of the
  displaced shape -- run from the same starting flow -- is better, halved up to
  twice otherwise. After a step is accepted the flow is advanced on the new
  shape for the sensitivity lead time, so the next gradient starts from a flow
  that has adapted to it.

It is *experimental* because the objective is the short-window average the
sensitivity uses (for steady flows a few per cent from the steady value; for
shedding flows an indicative one) and because shapes change by fractions of a
cell per step: confirm any gain with a normal CFD run of the result
(``result.extra["design"]``).
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time

import numpy as np

from ..model import Body, Design, Simulation
from .common import Cancelled, OptimizationResult, Progress, ProgressFn

__all__ = ["corner_taper", "optimize_shape_cfd"]

#: Default largest surface movement per accepted step, in grid cells.
STEP_CELLS = 0.75
#: Default bound on the total movement of any surface point, in grid cells.
MAX_TOTAL_CELLS = 4.0
#: Step halvings tried before an iteration gives up.
MAX_HALVINGS = 2
#: Outline corners sharper than this (turning angle, degrees) are held fixed.
CORNER_DEG = 30.0


def _rotate(p: np.ndarray, deg: float, about) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    q = np.asarray(p, float) - np.asarray(about, float)
    return np.c_[c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]] + np.asarray(about, float)


def optimize_shape_cfd(design: Design, sim: Simulation, *, objective: str | None = None, n_modes: int = 12,
                       n_iter: int = 4, step_cells: float = STEP_CELLS, max_total_cells: float = MAX_TOTAL_CELLS,
                       keep_area: bool = True, window: float | tuple[float, float] | None = None,
                       plan=None, device: str | None = None, progress: ProgressFn | None = None,
                       cancel: threading.Event | None = None) -> OptimizationResult:
    """Improve ``design`` for the CFD objective of ``sim`` with autograd gradients (experimental).

    ``objective``: "cd" (minimise), "cl", "ld" (maximise) or "cm" (minimise);
    default ``sim.cfd.sensitivity_objective``. ``n_modes``: smooth bumps per
    body; ``n_iter``: gradient steps; ``step_cells`` / ``max_total_cells``:
    largest movement per step / in total, in grid cells; ``keep_area``: keep
    each body's area to first order. ``window`` as in
    :func:`~aerolume.optimize.compute_sensitivity`. Uses the simulation's own
    plan (Draft quality keeps an iteration to ~30 s on a GPU).

    Returns an :class:`OptimizationResult` whose ``best_coords`` are the new
    outlines (design frame, metres, one per body, refined to about half a cell)
    and whose ``extra["design"]`` is a ready-to-simulate :class:`Design`.
    ``cancel`` stops after the current solver chunk and returns the best shape
    so far with a warning.
    """
    from ..core.device import resolve
    from ..simulation.live import cpu_threads
    from ..simulation.plan import plan_simulation
    from ..solvers.fpmode import flush_denormal
    from .sensitivity import (
        OBJECTIVES,
        WindowedObjective,
        choose_window,
        develop_flow,
        normal_modes,
        shedding_period,
    )

    wall0 = time.perf_counter()
    obj = (objective or getattr(sim.cfd, "sensitivity_objective", None) or "cd").lower()
    if obj not in OBJECTIVES:
        raise ValueError(f"unknown objective {obj!r}; use one of {', '.join(OBJECTIVES)}")
    sense = 1.0 if obj in ("cl", "ld") else -1.0            # +1: maximise
    rep = Progress(progress)
    if plan is None:
        plan = plan_simulation(design, sim, device=device)
    cfg = plan.solver_config
    if device not in (None, "", "auto"):
        cfg = dataclasses.replace(cfg, device=resolve(device), bodies=list(cfg.bodies))
    L = float(plan.length_ref)
    h = cfg.h
    unit = max(float(plan.strouhal_length) / L, 1e-6)
    dev_name = str(cfg.device if cfg.device not in ("auto", None, "") else resolve("auto"))
    dclass = "cpu" if dev_name.startswith("cpu") else "gpu"
    notes: list[str] = []
    warnings: list[str] = []
    history: list[dict] = []

    with cpu_threads(dev_name), flush_denormal(dclass == "cpu"):
        solver = develop_flow(cfg, max(float(plan.t_avg_start), 4.0 * unit), rep.sub(0.0, 0.15), cancel)
        period = shedding_period(solver, 0.5 * solver.t)
        lead, avg, _ = choose_window(unit, dclass, period, window, 1)
        if period is not None:
            warnings.append(f"The flow sheds vortices (period {period:.3g} L/U): the objective is a short-time "
                            "average, so gains are indicative. Confirm them with a full CFD run.")
        prob = WindowedObjective(cfg, solver.state.u, solver.state.v, objective=obj, lead=lead, average=avg,
                                 margin_cells=max_total_cells + 2.0)
        del solver
        movable = [k for k, b in enumerate(prob.bodies) if b.role != "wheel"]
        if not movable:
            raise ValueError("The design has no body that can change shape (wheels keep theirs).")
        tapers = {k: corner_taper(prob.points[k], 1.5 * h) for k in movable}
        modes = {k: normal_modes(prob, n_modes, body=k) * tapers[k] for k in movable}
        area0 = [_area(p) for p in prob.points]
        total = [np.zeros(len(p)) for p in prob.points]          # accumulated displacement (non-dim)
        step = step_cells * h
        limit = max_total_cells * h

        def mode_disp(amps: dict[int, np.ndarray]) -> list[np.ndarray]:
            out = [np.zeros(len(p)) for p in prob.points]
            for k, a in amps.items():
                out[k] = a @ modes[k]
            return out

        initial = best_val = None
        best_points = [p.copy() for p in prob.points]
        best_total = [t.copy() for t in total]
        pending = False                      # an accepted shape whose settled value is not measured yet
        stop_msg = f"{n_iter} iteration(s) done"
        cancelled = False

        def settled(val) -> None:
            """Book the settled objective of the current shape; keep it if it is the best."""
            nonlocal best_val, best_points, best_total
            history.append(_hist(len(history), val, total, prob, area0, L, wall0))
            if best_val is None or (val.value - best_val.value) * sense > 0:
                best_val = val
                best_points = [p.copy() for p in prob.points]
                best_total = [t.copy() for t in total]

        try:
            for it in range(1, int(n_iter) + 1):
                share = rep.sub(0.15 + 0.85 * (it - 1) / n_iter, 0.15 + 0.85 * it / n_iter)
                val, vg = prob.gradient(progress=share.sub(0.0, 0.8), cancel=cancel)
                if initial is None:
                    initial = val
                settled(val)
                pending = False
                # mode gradient, area projection, steepest step of fixed largest movement
                amps: dict[int, np.ndarray] = {}
                for k in movable:
                    g = modes[k] @ vg[k]
                    if keep_area:
                        c = modes[k] @ prob.dual[k]
                        g = g - (g @ c) / max(c @ c, 1e-300) * c
                    amps[k] = sense * g
                disp = mode_disp(amps)
                peak = max(float(np.abs(d).max()) for d in disp)
                if not peak > 0:
                    stop_msg = "the gradient vanished"
                    break
                scale = step / peak
                accepted = False
                for _ in range(MAX_HALVINGS + 1):
                    trial = [d * scale for d in disp]
                    if keep_area:
                        trial = _restore_area(prob, trial, area0, tapers, movable)
                    if max(float(np.abs(t + d).max()) for t, d in zip(total, trial)) > limit + 1e-12:
                        scale *= 0.5
                        continue
                    tv = prob.evaluate(trial, cancel=cancel)
                    if (tv.value - val.value) * sense > 0:
                        accepted = True
                        break
                    scale *= 0.5
                if not accepted:
                    stop_msg = "no step along the gradient improved the objective"
                    break
                prob.rebase(trial)
                total = [t + d for t, d in zip(total, trial)]
                prob.advance(lead, cancel=cancel, progress=share.sub(0.85, 1.0))
                pending = True
            if pending:
                settled(prob.evaluate(cancel=cancel))
        except Cancelled:
            cancelled = True
            stop_msg = "stopped by the user"
        except ValueError as exc:          # the shape left the mask window: stop cleanly
            stop_msg = str(exc)

    if initial is None:
        raise Cancelled("cancelled before the first gradient", None)
    v0, v1 = initial.value, best_val.value
    improvement = sense * (v1 - v0) / abs(v0) if v0 else 0.0
    sym = OBJECTIVES[obj]
    outlines = [p * L for p in best_points]                 # wind frame, metres
    ground = bool(plan.ground)
    alpha = float(sim.conditions.alpha_deg)
    design_frame = []
    for b, p in zip(prob.bodies, outlines):
        keep = ground and b.role == "wheel"
        design_frame.append(p if (keep or not alpha) else _rotate(p, alpha, design.moment_ref))
    new_bodies = [Body(name=ob.name, points=pts, role=ob.role, omega=ob.omega, center=ob.center)
                  for ob, pts in zip(design.bodies, design_frame)]
    new_design = Design(name=f"{design.name} (optimised {sym})", kind=design.kind, bodies=new_bodies,
                        length_ref=design.length_ref, force_ref=design.force_ref, moment_ref=design.moment_ref,
                        source={"type": "optimized", "from": design.id, "objective": obj, "method": "cfd-autograd"},
                        notes=f"Optimised for {sym} from '{design.name}' with the experimental CFD optimiser.")
    notes.append(f"Experimental CFD shape optimisation: {n_modes} smooth normal modes per body, gradients by "
                 f"automatic differentiation through the solver; objective = {sym} averaged over "
                 f"{prob.average:.3g} L/U after a {prob.lead:.3g} L/U lead ({prob.steps:,} steps per gradient)"
                 + ("; area held constant to first order" if keep_area else "") + ".")
    notes.append("The objective is the sensitivity window's short-time average: confirm the gain with a normal "
                 "CFD run of the optimised design.")
    if cancelled:
        warnings.append("Stopped by the user: this is the best shape found so far.")
    if improvement <= 0:
        warnings.append("No improving step was found; the original shape is returned.")
    move = max(float(np.abs(t).max()) for t in best_total) * L
    message = (f"{sym} {v0:.4g} -> {v1:.4g} ({100 * improvement:+.1f} %) after {len(history) - 1} accepted step(s); "
               f"largest surface movement {move * 1000:.3g} mm; {stop_msg}.")
    return OptimizationResult(
        method="cfd-autograd-modes", objective=obj,
        initial={"value": v0, "cd": float(initial.coefficients[0]), "cl": float(initial.coefficients[1]),
                 "cm": float(initial.coefficients[2])},
        best={"value": v1, "cd": float(best_val.coefficients[0]), "cl": float(best_val.coefficients[1]),
              "cm": float(best_val.coefficients[2])},
        improvement=float(improvement), best_coords=design_frame, history=history, x=None,
        converged=stop_msg.startswith(("no step", "the gradient")), message=message,
        evaluations=len(history), wall_seconds=time.perf_counter() - wall0, notes=notes, warnings=warnings,
        extra={"design": new_design, "displacement_m": [t * L for t in best_total], "window": (prob.lead, prob.average),
               "cancelled": cancelled, "max_movement_m": move})


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _restore_area(prob, disp: list[np.ndarray], area0: list[float], tapers: dict[int, np.ndarray],
                  movable: list[int]) -> list[np.ndarray]:
    """Add a uniform (corner-tapered) normal offset per body that puts its area back to ``area0``.

    The projected gradient step keeps the area to first order only; on a coarse
    grid a step is a sizeable fraction of a small body and the curvature term
    (~ kappa d^2 / 2 per unit length) adds up to per cent over a few steps. One
    Newton correction with the exact polygon area removes it to third order.
    """
    out = [np.array(d, float) for d in disp]
    outlines = prob.outlines(out)
    for k in movable:
        w = tapers[k]
        den = float(np.sum(w * prob.dual[k]))
        if den > 0:
            out[k] = out[k] + (area0[k] - _area(outlines[k])) / den * w
    return out


def corner_taper(points: np.ndarray, width: float, angle_deg: float = CORNER_DEG) -> np.ndarray:
    """1 along smooth outline, 0 at corners sharper than ``angle_deg``, Gaussian dip of ``width`` (arc length).

    A normal displacement is ill-defined at a corner (the normal jumps): moving
    the corner vertex along the bisector while its neighbours move along their
    own normals splits a sharp trailing edge into a "fishtail". Corners are
    therefore held fixed -- which also keeps an airfoil's chord and trailing
    edge where they are -- and the modes fade in over ~1.5 cells.
    """
    p = np.asarray(points, float)
    e_in = p - np.roll(p, 1, axis=0)
    e_out = np.roll(p, -1, axis=0) - p
    ang = np.degrees(np.abs(np.arctan2(e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0],
                                       np.sum(e_in * e_out, axis=1))))
    seg = np.hypot(*e_out.T)
    arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    per = float(seg.sum())
    w = np.ones(len(p))
    for c in np.nonzero(ang > angle_deg)[0]:
        d = np.abs(arc - arc[c])
        d = np.minimum(d, per - d)
        w *= 1.0 - np.exp(-0.5 * (d / width) ** 2)
    return w


def _hist(it: int, val, total, prob, area0, L: float, wall0: float) -> dict:
    return {"iter": it, "objective": float(val.value), "cd": float(val.coefficients[0]),
            "cl": float(val.coefficients[1]), "cm": float(val.coefficients[2]),
            "max_movement_m": max(float(np.abs(t).max()) for t in total) * L,
            "area_change": [(_area(p) - a0) / a0 for p, a0 in zip(prob.points, area0)],
            "seconds": time.perf_counter() - wall0}

