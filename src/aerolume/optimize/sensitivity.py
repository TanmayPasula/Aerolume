"""Shape sensitivity maps: dJ/dn by automatic differentiation through the Navier-Stokes solver.

The map answers "where does pushing the surface outward change the objective,
and by how much?" for J = Cd, Cl, L/D or Cm. It is the AirShaper-style
optimisation view, computed AeroJAX-style: the flow solver is differentiable
(``docs/theory_ns2d.md`` section 13), so one reverse sweep through a short
window of the simulation yields dJ/d(displacement) for every surface point at
once -- the cost of about six forward runs of that window (measured), independent
of the number of surface points. ``docs/theory_sensitivity.md`` derives the
method, shows its verification against finite differences and lists its
limits.

Pipeline
--------
1. **Developed flow.** Either the caller's (the runner passes its finished
   :class:`~aerolume.solvers.ns2d.FlowSolver`), or the flow is developed here up
   to the plan's ``t_avg_start``.
2. **Objective.** From that state (held fixed), the solver runs ``lead + average``
   convective times with a fixed time step; J is the objective of the force
   coefficients averaged over the last ``average`` of them. The lead lets the
   flow respond to the shape change before forces are counted (the first steps
   after a geometry change carry an impulsive start-up transient, see the
   theory doc).
3. **Parametrisation.** Every body outline is refined so that vertices are at
   most half a cell apart (inserted points lie on the original edges, so the
   shape is unchanged) and each vertex gets a displacement ``d_i`` along its
   fixed outward normal. ``d -> polygon -> SDF -> smoothed masks ->
   penalisation -> force impulse`` is differentiable end to end.
4. **Reverse sweep** with memory O(sqrt(N)) states: the forward pass keeps the
   state every ~sqrt(N) steps, the reverse pass recomputes one segment at a
   time and pulls the adjoint back step by step (a vector-Jacobian product of
   one solver step). On CUDA that step - forward and backward - is a captured
   CUDA graph (``torch.cuda.make_graphed_callables``): ~3-4 ms per step on a
   50 k-cell grid instead of ~75 ms eager, bit-identical results.
   Gradients reach the masks only; one final backward through the geometry
   build turns them into dJ/dd_i.
5. **Map.** dJ/dd_i is a *point* derivative that scales with the local vertex
   spacing; dividing by the dual length of each vertex gives the shape
   derivative density g(s) with ``dJ = integral g dn ds``. It is smoothed over
   about one cell along the outline (the mask lives on the grid, so per-vertex
   values carry grid-scale noise) and evaluated at the points that
   ``FlowSolver.surface()`` returns, in the same order, so the arrays drop
   straight into :attr:`SurfaceData.sensitivity`.

Units and sign
--------------
The returned values are ``L_ref * g`` in **1/m**: the change of J per metre of
outward normal displacement applied over a stretch of outline one reference
length long. Equivalently, for a small displacement field dn(s) [m]::

    dJ ~= (1 / L_ref) * sum_i sens_i * dn_i * ds_i       (ds_i: point spacing [m])

Positive = pushing the surface outward (adding material) here increases J.
For Cd, blue (negative) regions are where adding material lowers the drag.

**The pattern is reliable, the size is indicative.** Measured on a steady
cylinder (Re 40, 40 cells/D): whole-cell finite differences of the run's Cd
give dCd/dR = 2.23-2.33, the adjoint of the default 6 + 2 window 4.32 (16 + 4:
3.82). About 1.23x of that is the finite window (the windowed derivative relaxes
toward ~3.5 with a time constant of ~12 D/U) and about 1.5-1.6x the immersed
boundary: the discrete J(R) has a cell-scale ripple, so its local (sub-cell)
derivative, which is what the adjoint returns, is larger than the slope over
whole cells. The peaks, signs and the place of each feature agree with the
finite differences; the GUI and the report therefore show the map relative to
its 98th-percentile magnitude, and every map carries a warning that says so.
The 1/m values stay in :class:`SensitivityMaps` and the exports.

How the simulation runner calls it
----------------------------------
When ``sim.cfd.compute_sensitivity`` is set, after :func:`run_cfd` has assembled
its result and while its solver is still alive::

    from aerolume.optimize import Cancelled, attach_sensitivity, compute_sensitivity
    try:
        maps = compute_sensitivity(plan.design, plan.sim, plan=plan, state=solver,
                                   objective=plan.sim.cfd.sensitivity_objective,
                                   n_per_body=[len(sd.x) for sd in result.surface],
                                   progress=sub_progress, cancel=cancel)
        attach_sensitivity(result, maps)        # fills SurfaceData.sensitivity, adds a note
    except Cancelled:
        result.warnings.append("Sensitivity map skipped: stopped by the user.")
    except Exception as exc:                     # the forces are still valid
        result.warnings.append(f"Sensitivity map failed: {exc}")

Reusing the run's solver skips the flow development (the expensive part) and
makes the map belong to exactly the flow the result describes. The map costs
about ``(lead + average) / dt`` reverse steps: ~30 s for a Draft airfoil run on
the GPU (2,000 steps, 131 k cells), about a minute on a CPU (shorter default
window there). ``progress`` receives fractions 0..1 of the
sensitivity work; map them onto the tail of the run's own bar.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from ..model import CFDResult, Design, Simulation
from ..solvers.ns2d import FlowSolver, ImmersedBody, SolverConfig, State, build_geometry, make_operators
from ..solvers.ns2d.core import Ops, cfl_dt, step_full
from ..solvers.ns2d.geometry import Geometry, body_window
from ..solvers.ns2d.solver import resample_closed
from .common import Progress, ProgressFn, check_cancel

__all__ = [
    "MAGNITUDE_WARNING",
    "OBJECTIVES",
    "SensitivityMaps",
    "WindowedObjective",
    "attach_sensitivity",
    "choose_window",
    "compute_sensitivity",
    "density_on_outline",
    "develop_flow",
    "estimate_seconds",
    "finite_difference_check",
    "normal_modes",
    "refine_polygon",
    "shedding_period",
    "surface_point_count",
]

#: Objectives and their display names. J is computed from the window-mean force
#: coefficients (summed over all bodies); "ld" is mean Cl / mean Cd.
OBJECTIVES = {"cd": "Cd", "cl": "Cl", "ld": "L/D", "cm": "Cm"}
#: Default windows, in units of the body's own convective time (chord / U for
#: airfoils, frontal height / U for vehicles and bluff bodies), per device class.
#: Steady flows: (lead, average). The truncated-window sensitivity over-estimates
#: the relaxed one by a factor that decays with the lead and is nearly the same
#: along the whole outline. Measured against sub-cell finite differences of the
#: same discrete objective (docs/theory_sensitivity.md section 5): cylinder at
#: Re 30, 16 cells/D: 3 + 1 -> +13-15 %, 6 + 2 -> +5-6 %, 10 + 2 -> +2-3 %;
#: cylinder at Re 40, 40 cells/D, with a longer relaxation (~12 D/U): 6 + 2 ->
#: about +23 %, 16 + 4 -> about +9 %. That is the window's share only, not the
#: error against the physical slope: the immersed boundary adds another ~1.5x
#: (module docstring), which is why the map's size is reported as indicative.
#: The CPU default is shorter because a reverse step costs ~120 ms there.
STEADY_WINDOW = {"gpu": (6.0, 2.0), "cpu": (3.0, 1.0)}
#: The warning every map carries (see the module docstring for the measurements).
MAGNITUDE_WARNING = ("Read the sensitivity map's pattern, not its size: the magnitude can be about twice the "
                     "physical derivative. Part comes from the finite averaging window (about +20 % at the default "
                     "window on a Re 40 cylinder) and part from how the immersed boundary responds to sub-cell "
                     "shape changes; signs and the places that matter most are reliable.")
#: Unsteady (shedding) flows: a lead of UNSTEADY_LEAD time units (at most one
#: period, at least one unit: it only has to let the start-up impulse of the
#: shape change pass), average whole periods covering at least this many time
#: units, and an ensemble of windows started at evenly spaced phases of one
#: period, averaged. A single
#: window's map depends on where in the cycle it starts (phase scatter ~50-80 %
#: of the mean on the dominant modes for 3 periods, NACA 0012 Re 1e5 LES); the
#: ensemble and the longer window bring that down to ~20-30 %.
UNSTEADY_MIN_AVERAGE = {"gpu": 4.0, "cpu": 2.0}
UNSTEADY_LEAD = 2.0
UNSTEADY_ENSEMBLE = 3
#: Refined outline spacing [cells]: below the grid scale, so the map can show
#: everything the masks can feel.
REFINE_CELLS = 0.5
#: Smoothing of the map along the outline (Gaussian sigma) [cells]. Per-vertex
#: derivatives alias with the grid (each vertex feels only the one or two faces
#: in its part of the one-cell mask band); 1.5 cells removes that noise while the
#: map stays grid-converged (cylinder, 16 / 32 / 48 cells per diameter).
SMOOTH_CELLS = 1.5
#: Extra cells around the bodies in the (fixed) mask window, so displaced
#: shapes (finite-difference checks, optimiser steps) stay inside it.
WINDOW_MARGIN_CELLS = 4.0
#: The window runs at a fixed time step (dt must not depend on the design):
#: this fraction of the CFL step of the starting state, leaving room for the
#: flow to speed up a little during the window.
DT_SAFETY = 0.9
#: Safety cap on the number of differentiated steps (time, not memory, is the limit).
MAX_STEPS = 40_000
#: Solver steps per cancel / progress check.
CHECK_EVERY = 32
#: The reverse sweep's checkpoints move to host memory when the states it holds
#: would need more than this fraction of the free GPU memory.
OFFLOAD_FRACTION = 0.4

_GEOM_FIELDS = ("chi_u", "chi_v", "a_u", "a_v", "k_u", "k_v")


# ---------------------------------------------------------------------------
# outline helpers
# ---------------------------------------------------------------------------

def refine_polygon(points: np.ndarray, max_spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert points on edges longer than ``max_spacing`` (closed polygon, any orientation).

    Inserted points lie exactly on the original edges, so the outline -- and the
    solver's masks -- are unchanged; only the displacement parametrisation gets
    finer. Returns ``(refined (M, 2), original_index (N,))`` where
    ``refined[original_index[k]] == points[k]``.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.roll(p, -1, axis=0)
    seg = np.hypot(*(q - p).T)
    out, idx = [], []
    for k in range(len(p)):
        idx.append(sum(len(o) for o in out))
        m = max(1, math.ceil(seg[k] / max_spacing - 1e-9)) if max_spacing > 0 else 1
        t = np.arange(m)[:, None] / m
        out.append(p[k] + t * (q[k] - p[k]))
    return np.vstack(out), np.asarray(idx, dtype=int)


def _vertex_frame(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Outward unit normals (central-difference tangent rotated clockwise, CCW outline),
    dual lengths (half the two adjacent edges), arc length of each vertex from
    vertex 0, and the perimeter."""
    nxt, prv = np.roll(p, -1, axis=0), np.roll(p, 1, axis=0)
    t = nxt - prv
    n = np.c_[t[:, 1], -t[:, 0]]
    n /= np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-300)[:, None]
    e = np.hypot(*(nxt - p).T)                 # edge i -> i+1
    dual = 0.5 * (e + np.roll(e, 1))
    arc = np.concatenate([[0.0], np.cumsum(e)[:-1]])
    return n, dual, arc, float(e.sum())


def density_on_outline(arc_v: np.ndarray, grad_v: np.ndarray, dual_v: np.ndarray, perimeter: float,
                       s_query: np.ndarray, sigma: float) -> np.ndarray:
    """Shape-derivative density g(s) at arc positions ``s_query`` from per-vertex derivatives.

    ``grad_v[i] = dJ/dd_i`` is the derivative with respect to moving vertex i
    (a hat function of the outline) along its normal; it is approximately
    ``g(s_i) * dual_v[i]``. The density is the kernel-weighted ratio
    ``sum K(s - s_i) grad_i / sum K(s - s_i) dual_i`` with a periodic Gaussian
    K of standard deviation ``sigma``: a smoothed derivative divided by the
    smoothed length it acts on, which stays well defined where vertices
    cluster (airfoil noses) and reduces to ``grad_i / dual_i`` as sigma -> 0
    (``sigma <= 0`` interpolates that ratio linearly instead).
    """
    arc_v = np.asarray(arc_v, float)
    s_query = np.asarray(s_query, float)
    if sigma <= 0:
        ratio = np.asarray(grad_v, float) / np.maximum(dual_v, 1e-300)
        xs = np.concatenate([arc_v - perimeter, arc_v, arc_v + perimeter])
        return np.interp(s_query, xs, np.tile(ratio, 3))
    out = np.empty(s_query.size)
    step = max(1, 4_000_000 // max(arc_v.size, 1))
    for a in range(0, s_query.size, step):
        d = np.abs(s_query[a:a + step, None] - arc_v[None, :])
        d = np.minimum(d, perimeter - d)
        k = np.exp(-0.5 * (d / sigma) ** 2)
        out[a:a + step] = (k @ grad_v) / np.maximum(k @ dual_v, 1e-300)
    return out


def surface_point_count(cfg: SolverConfig) -> int:
    """Points per body that :func:`aerolume.simulation.run_cfd` asks ``FlowSolver.surface`` for
    (about two per grid cell along the longest outline, 200..2000)."""
    per = [float(np.hypot(*np.diff(np.vstack([b.points, b.points[:1]]), axis=0).T).sum()) for b in cfg.bodies]
    return int(np.clip(2.0 * max(per, default=0.0) / cfg.h, 200, 2000))


# ---------------------------------------------------------------------------
# one differentiable step, optionally CUDA-graphed
# ---------------------------------------------------------------------------

class _Stepper:
    """One solver step as a pure function of (u, v, mask tensors), with its VJP.

    ``chi_c`` (the cell-centre solid fraction, used only to switch the LES
    eddy viscosity off inside solids) is passed *without* gradient: its effect
    on the forces is confined to the inside of the body. (Older solver
    versions also updated the eddy viscosity in place in a way autograd could
    not differentiate with respect to chi_c; the 0.1.0 solver no longer does.)
    Everything that sets the wall (chi_u, chi_v, the solid velocity a_u, a_v
    and the force weights k_u, k_v) is differentiated.
    """

    def __init__(self, ops: Ops, window, nbody: int, dt: float, *, use_graphs: bool | None = None):
        self.ops, self.window, self.nbody = ops, tuple(window), int(nbody)
        self.dt = torch.tensor(float(dt), device=ops.device, dtype=ops.dtype)
        self._zero = torch.zeros((), device=ops.device, dtype=ops.dtype)
        self.fn: Callable = self._step
        self.graphed = False
        self.note: str | None = None
        self._want_graphs = (ops.device.type == "cuda") if use_graphs is None else bool(use_graphs)
        self._tried = False

    def _step(self, u, v, chi_u, chi_v, a_u, a_v, k_u, k_v, chi_c, dt):
        g = Geometry(self.window, chi_u, chi_v, chi_c, a_u, a_v, k_u, k_v, self.nbody)
        new, c = step_full(State(u, v, self._zero, None, self._zero, self._zero), g, self.ops, dt)
        return new.u, new.v, c

    def prepare(self, u: torch.Tensor, v: torch.Tensor, geom_t: Sequence[torch.Tensor], chi_c: torch.Tensor) -> None:
        """Capture the CUDA graph (once) from sample tensors shaped like these."""
        if self._tried or not self._want_graphs:
            return
        self._tried = True
        try:
            with torch.enable_grad():
                sample = ([u.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)]
                          + [t.detach().clone().requires_grad_(True) for t in geom_t]
                          + [chi_c.detach().clone(), self.dt.clone()])
                self.fn = torch.cuda.make_graphed_callables(self._step, tuple(sample), num_warmup_iters=3,
                                                            allow_unused_input=True)
            self.graphed = True
        except Exception as exc:  # noqa: BLE001 - any capture failure: the eager path computes the same thing
            self.fn = self._step
            self.graphed = False
            self.note = (f"CUDA graphs were not available for the reverse sweep ({type(exc).__name__}); "
                         "it ran in eager mode, which is slower but gives the same result.")
            if self.ops.device.type == "cuda":
                torch.cuda.synchronize(self.ops.device)

    def forward(self, u, v, geom_t, chi_c):
        """(u, v, per-body coefficients) after one step, without autograd.

        With CUDA graphs the outputs are the graph's static buffers: they are
        overwritten by the next call, so callers clone what they keep.
        """
        with torch.no_grad():
            return self.fn(u, v, *geom_t, chi_c, self.dt)

    def vjp(self, u, v, geom_t, chi_c, lam_u, lam_v, w):
        """Pull (lam_u, lam_v, w) on the step outputs back to (u, v, mask tensors).

        Returns a tuple of gradients (None where an input does not influence the
        outputs). With CUDA graphs these alias static buffers: consume them before
        the next call.
        """
        u = u.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        with torch.enable_grad():
            un, vn, c = self.fn(u, v, *geom_t, chi_c, self.dt)
            return torch.autograd.grad((un, vn, c), (u, v, *geom_t), (lam_u, lam_v, w), allow_unused=True)


# ---------------------------------------------------------------------------
# the windowed objective
# ---------------------------------------------------------------------------

def _objective_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "cd":
        return lambda m: m[0]
    if name == "cl":
        return lambda m: m[1]
    if name == "cm":
        return lambda m: m[2]
    if name == "ld":
        return lambda m: m[1] / m[0]
    raise ValueError(f"unknown objective {name!r}; use one of {', '.join(OBJECTIVES)}")


@dataclasses.dataclass
class WindowValue:
    """J and the window-mean coefficients [Cd, Cl, Cm] (summed over bodies)."""
    value: float
    coefficients: np.ndarray
    per_body: np.ndarray


class WindowedObjective:
    """J(d): the time-averaged force objective after a shape change, as a function of
    per-vertex normal displacements ``d`` (non-dimensional, one array per body).

    Built from a solver config (non-dimensional, wind frame) and a developed
    starting state (u0, v0), which is held fixed. :meth:`evaluate` runs the
    window forward; :meth:`gradient` returns J and dJ/dd for every refined
    vertex by one reverse sweep. Both use the same discrete function -- same
    fixed mask window, time step and step count -- so finite differences of
    :meth:`evaluate` check :meth:`gradient` exactly (:func:`finite_difference_check`).
    """

    def __init__(self, cfg: SolverConfig, u0: torch.Tensor, v0: torch.Tensor, *, objective: str = "cd",
                 lead: float = 1.0, average: float = 1.0, dt: float | None = None,
                 refine_cells: float = REFINE_CELLS, margin_cells: float = WINDOW_MARGIN_CELLS,
                 use_graphs: bool | None = None, max_steps: int = MAX_STEPS):
        if not cfg.bodies:
            raise ValueError("there is no body to compute a sensitivity for")
        if not (lead >= 0 and average > 0):
            raise ValueError("the window needs lead >= 0 and average > 0 convective times")
        self.objective = objective
        self._fobj = _objective_fn(objective)
        self.cfg = dataclasses.replace(cfg, bodies=list(cfg.bodies), dye=False, use_cuda_graphs=False)
        self.ops = make_operators(self.cfg)
        dev, dtype = self.ops.device, self.ops.dtype
        self.u0 = u0.detach().to(device=dev, dtype=dtype).clone()
        self.v0 = v0.detach().to(device=dev, dtype=dtype).clone()
        if self.u0.shape != (cfg.ny, cfg.nx + 1) or self.v0.shape != (cfg.ny + 1, cfg.nx):
            raise ValueError("the starting state does not match the solver grid")
        h = cfg.h
        self.bodies: list[ImmersedBody] = []
        self.points: list[np.ndarray] = []           # refined outlines (non-dimensional)
        self.original_index: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.dual: list[np.ndarray] = []
        self.arc: list[np.ndarray] = []
        self.perimeter: list[float] = []
        for b in self.cfg.bodies:
            p, idx = refine_polygon(b.points, refine_cells * h)
            n, dual, arc, per = _vertex_frame(p)
            self.bodies.append(ImmersedBody(points=p, name=b.name, role=b.role, omega=b.omega, center=b.center))
            self.points.append(p)
            self.original_index.append(idx)
            self.normals.append(n)
            self.dual.append(dual)
            self.arc.append(arc)
            self.perimeter.append(per)
        self._p0 = [torch.as_tensor(p, device=dev, dtype=dtype) for p in self.points]
        self._n0 = [torch.as_tensor(n, device=dev, dtype=dtype) for n in self.normals]
        margin = self.ops.mask_width + 3.0 + abs(self.ops.mask_shift) + margin_cells
        self.window = body_window(self.points, self.ops, margin_cells=margin)
        self.dt = float(dt) if dt is not None else DT_SAFETY * float(cfl_dt(self.u0, self.v0, self.ops))
        if not (self.dt > 0 and math.isfinite(self.dt)):
            raise ValueError("could not determine a time step from the starting state")
        self.n_lead = math.ceil(lead / self.dt - 1e-9)
        self.n_avg = max(1, math.ceil(average / self.dt - 1e-9))
        if self.n_lead + self.n_avg > max_steps:
            raise ValueError(f"the window needs {self.n_lead + self.n_avg:,} steps (limit {max_steps:,}); "
                             "shorten it")
        self.lead, self.average = self.n_lead * self.dt, self.n_avg * self.dt
        self._stepper = _Stepper(self.ops, self.window, len(self.bodies), self.dt, use_graphs=use_graphs)

    # -------------------------------------------------------------- geometry
    @property
    def steps(self) -> int:
        return self.n_lead + self.n_avg

    @property
    def cuda_graphs(self) -> bool:
        return self._stepper.graphed

    @property
    def notes(self) -> list[str]:
        return [self._stepper.note] if self._stepper.note else []

    def _as_disp(self, disp) -> list[torch.Tensor]:
        dev, dtype = self.ops.device, self.ops.dtype
        if disp is None:
            return [torch.zeros(len(p), device=dev, dtype=dtype) for p in self.points]
        if len(disp) != len(self.points):
            raise ValueError("one displacement array per body is needed")
        out = []
        for d, p in zip(disp, self.points):
            t = d if isinstance(d, torch.Tensor) else torch.as_tensor(np.asarray(d, float))
            t = t.to(device=dev, dtype=dtype)
            if t.shape != (len(p),):
                raise ValueError(f"displacement for a {len(p)}-vertex outline has shape {tuple(t.shape)}")
            out.append(t)
        return out

    def polygons(self, disp=None) -> list[torch.Tensor]:
        """Displaced outlines ``p_i + d_i n_i`` (non-dimensional tensors; keeps autograd)."""
        return [p + d[:, None] * n for p, d, n in zip(self._p0, self._as_disp(disp), self._n0)]

    def outlines(self, disp=None) -> list[np.ndarray]:
        """Displaced outlines as numpy arrays (non-dimensional)."""
        return [p.detach().cpu().double().numpy() for p in self.polygons(disp)]

    def geometry(self, disp=None) -> Geometry:
        """Masks of the displaced bodies on the fixed window (differentiable in ``disp``)."""
        bodies = []
        for b, p in zip(self.bodies, self.polygons(disp)):
            nb = dataclasses.replace(b)          # validated numpy copy (omega, centre, role kept) ...
            nb.points = p                        # ... carrying the tensor outline: build_geometry keeps its graph
            bodies.append(nb)
        return build_geometry(bodies, self.ops, window=self.window)

    # ------------------------------------------------------------- forward
    def _forward(self, geom_t, chi_c, *, keep_every: int = 0, keep_on=None, progress=None, cancel=None,
                 work=(0.0, 1.0)):
        """Run the window; returns (sum of per-body coefficients over the averaging
        steps, list of stored (u, v) states every ``keep_every`` steps, held on
        device ``keep_on`` (default: the solver's))."""
        st = self._stepper
        st.prepare(self.u0, self.v0, geom_t, chi_c)
        u, v = self.u0, self.v0
        acc = torch.zeros(len(self.bodies), 3, device=self.ops.device, dtype=self.ops.dtype)
        kept = []
        n = self.steps
        for i in range(n):
            if keep_every and i % keep_every == 0:
                if keep_on is None:
                    kept.append((u.clone(), v.clone()))
                else:
                    kept.append((u.to(keep_on, copy=True), v.to(keep_on, copy=True)))
            u, v, c = st.forward(u, v, geom_t, chi_c)
            if i >= self.n_lead:
                acc += c
            if i % CHECK_EVERY == 0:
                check_cancel(cancel)
                if progress is not None:
                    progress(work[0] + (work[1] - work[0]) * i / n, "Running the sensitivity window")
        return acc, kept

    def _window_value(self, acc: torch.Tensor) -> tuple[torch.Tensor, WindowValue]:
        per_body = acc / self.n_avg
        mean = per_body.sum(0).detach().requires_grad_(True)
        with torch.enable_grad():
            j = self._fobj(mean)
            (dj,) = torch.autograd.grad(j, mean)
        val = WindowValue(float(j.detach()), mean.detach().cpu().double().numpy(),
                          per_body.detach().cpu().double().numpy())
        return dj, val

    def evaluate(self, disp=None, *, cancel: threading.Event | None = None) -> WindowValue:
        """J for displaced outlines (forward only)."""
        with torch.no_grad():
            g = self.geometry(disp)
            geom_t = [getattr(g, k) for k in _GEOM_FIELDS]
            acc, _ = self._forward(geom_t, g.chi_c, cancel=cancel)
        return self._window_value(acc)[1]

    def advance(self, duration: float, disp=None, *, cancel: threading.Event | None = None,
                progress: ProgressFn | None = None) -> WindowValue | None:
        """Move the starting state forward by ``duration`` (L/U) with the (displaced) shape.

        Used to start ensemble windows at other phases of a shedding cycle and,
        by the CFD optimiser, to let the flow settle on an accepted shape. Runs
        at the window's fixed time step on the same (graphed) step. Returns the
        mean coefficients over the advanced span (None when it is empty).
        """
        n = round(duration / self.dt)
        if n <= 0:
            return None
        with torch.no_grad():
            g = self.geometry(disp)
            geom_t = [getattr(g, k) for k in _GEOM_FIELDS]
            st = self._stepper
            st.prepare(self.u0, self.v0, geom_t, g.chi_c)
            u, v = self.u0, self.v0
            acc = torch.zeros(len(self.bodies), 3, device=self.ops.device, dtype=self.ops.dtype)
            for i in range(n):
                u, v, c = st.forward(u, v, geom_t, g.chi_c)
                acc += c
                if i % CHECK_EVERY == 0:
                    check_cancel(cancel)
                    if progress is not None:
                        progress(i / n, "Letting the flow settle")
            self.u0, self.v0 = u.clone(), v.clone()
            if not (bool(torch.isfinite(self.u0).all()) and bool(torch.isfinite(self.v0).all())):
                raise RuntimeError("The flow became unstable while advancing the sensitivity start state.")
        per_body = (acc / n).cpu().double().numpy()
        return WindowValue(float("nan"), per_body.sum(0), per_body)

    def rebase(self, disp) -> None:
        """Make the displaced outlines the new reference shape (normals, spacing re-derived).

        The mask window, time step, step counts and the captured CUDA graph stay
        as they are, so an optimiser can take many steps without re-capturing.
        Raises ValueError when the new shape leaves the mask window.
        """
        new = self.outlines(disp)
        win = body_window(new, self.ops, margin_cells=self.ops.mask_width + 3.0 + abs(self.ops.mask_shift))
        j0, j1, i0, i1 = self.window
        if not (win[0] >= j0 and win[1] <= j1 and win[2] >= i0 and win[3] <= i1):
            raise ValueError("the displaced shape no longer fits the sensitivity window")
        dev, dtype = self.ops.device, self.ops.dtype
        for k, p in enumerate(new):
            n, dual, arc, per = _vertex_frame(p)
            self.points[k], self.normals[k], self.dual[k], self.arc[k], self.perimeter[k] = p, n, dual, arc, per
            b = self.bodies[k]
            self.bodies[k] = ImmersedBody(points=p, name=b.name, role=b.role, omega=b.omega, center=b.center)
        self._p0 = [torch.as_tensor(p, device=dev, dtype=dtype) for p in self.points]
        self._n0 = [torch.as_tensor(n, device=dev, dtype=dtype) for n in self.normals]

    # ------------------------------------------------------------- reverse
    def gradient(self, disp=None, *, progress: ProgressFn | None = None,
                 cancel: threading.Event | None = None) -> tuple[WindowValue, list[np.ndarray]]:
        """J and dJ/dd_i for every refined vertex of every body (non-dimensional).

        Memory: about 2 sqrt(N) velocity fields plus one step's intermediates.
        Raises :class:`~aerolume.optimize.Cancelled` when ``cancel`` is set.
        """
        rep = Progress(progress)
        dvec = [d.detach().clone().requires_grad_(True) for d in self._as_disp(disp)]
        with torch.enable_grad():
            g_full = self.geometry(dvec)
        full = [getattr(g_full, k) for k in _GEOM_FIELDS]
        geom_t = [t.detach().clone().requires_grad_(True) for t in full]
        chi_c = g_full.chi_c.detach()
        n = self.steps
        seg = max(1, math.ceil(math.sqrt(n)))
        # progress weights: the forward pass is ~1/6 of the work; recomputing a segment
        # and the reverse steps (a forward and a backward replay each) are the rest
        f_fwd = 1.0 / 6.0
        keep_on = self._offload_target(n, seg)
        acc, kept = self._forward(geom_t, chi_c, keep_every=seg, keep_on=keep_on, progress=rep, cancel=cancel,
                                  work=(0.0, f_fwd))
        dj, val = self._window_value(acc)
        nb = len(self.bodies)
        w_avg = (dj / self.n_avg).expand(nb, 3).contiguous()
        w_zero = torch.zeros_like(w_avg)
        lam_u = torch.zeros_like(self.u0)
        lam_v = torch.zeros_like(self.v0)
        grads = [torch.zeros_like(t) for t in geom_t]
        st = self._stepper
        done = 0
        for b in reversed(range(len(kept))):
            start, stop = b * seg, min(n, (b + 1) * seg)
            u, v = kept[b]
            if keep_on is not None:
                u, v = u.to(self.ops.device), v.to(self.ops.device)
            states = [(u, v)]
            for _ in range(start, stop - 1):
                u, v, _c = st.forward(u, v, geom_t, chi_c)
                states.append((u.clone(), v.clone()))
            kept[b] = None
            for i in reversed(range(start, stop)):
                ui, vi = states[i - start]
                out = st.vjp(ui, vi, geom_t, chi_c, lam_u, lam_v, w_avg if i >= self.n_lead else w_zero)
                lam_u.copy_(out[0])
                lam_v.copy_(out[1])
                for a, gk in zip(grads, out[2:]):
                    if gk is not None:
                        a.add_(gk)
                done += 1
                if done % CHECK_EVERY == 0:
                    check_cancel(cancel)
                    rep(f_fwd + (1.0 - f_fwd) * done / n, "Differentiating the flow (reverse sweep)")
            del states
        pairs = [(t, gk) for t, gk in zip(full, grads) if t.requires_grad]
        if not all(bool(torch.isfinite(gk).all()) for _, gk in pairs):
            raise RuntimeError("The sensitivity is not finite: the differentiated flow model has a "
                               "non-smooth point here (this happens with the Vreman and Smagorinsky "
                               "sub-grid models where the strain vanishes). Use laminar or WALE.")
        if pairs:
            torch.autograd.backward([t for t, _ in pairs], [gk for _, gk in pairs])
        out = []
        for d in dvec:
            gd = d.grad if d.grad is not None else torch.zeros_like(d)
            out.append(gd.detach().cpu().double().numpy())
        if not all(np.all(np.isfinite(o)) for o in out):
            raise RuntimeError("The sensitivity is not finite (the geometry derivative overflowed).")
        rep(1.0, "Sensitivity computed")
        return val, out

    # ------------------------------------------------------------- helpers
    def _offload_target(self, n: int, seg: int):
        """'cpu' when the checkpointed states would take more than OFFLOAD_FRACTION of the
        free GPU memory (fine grids, long windows), else None (keep them on the device).

        The reverse sweep holds ~n/seg checkpoints plus one recomputed segment of seg
        states; only the checkpoints can wait in host memory, which halves the device
        footprint at the cost of one copy per segment.
        """
        if self.ops.device.type != "cuda":
            return None
        per_state = (self.u0.numel() + self.v0.numel()) * self.u0.element_size()
        need = per_state * (math.ceil(n / seg) + seg)
        try:
            free, _ = torch.cuda.mem_get_info(self.ops.device)
        except Exception:  # noqa: BLE001 - no reliable reading: keep everything on the device
            return None
        return "cpu" if need > OFFLOAD_FRACTION * free else None

    def density(self, vertex_grad: Sequence[np.ndarray], s_query: Sequence[np.ndarray],
                sigma: float | None = None) -> list[np.ndarray]:
        """Map per-vertex derivatives to the density g (non-dimensional) at arc positions per body."""
        sig = SMOOTH_CELLS * self.cfg.h if sigma is None else float(sigma)
        return [density_on_outline(a, gv, dl, per, sq, sig)
                for a, gv, dl, per, sq in zip(self.arc, vertex_grad, self.dual, self.perimeter, s_query)]


# ---------------------------------------------------------------------------
# finite-difference verification
# ---------------------------------------------------------------------------

def normal_modes(problem: WindowedObjective, n_modes: int = 8, width: float | None = None,
                 body: int = 0) -> np.ndarray:
    """Smooth displacement modes of one body's refined outline: (n_modes, n_vertices).

    Periodic Gaussian bumps in arc length (:func:`aerolume.geometry.normal_basis`),
    peak 1 (non-dimensional), centred at equal arc-length intervals from vertex 0.
    """
    from ..geometry.params import normal_basis
    return normal_basis(problem.points[body], n_modes, width).cpu().double().numpy()


def finite_difference_check(problem: WindowedObjective, modes: np.ndarray, *, eps: float = 1e-4,
                            body: int = 0, vertex_grad: Sequence[np.ndarray] | None = None,
                            progress: ProgressFn | None = None,
                            cancel: threading.Event | None = None) -> list[dict]:
    """Compare autograd with central differences of the same J along displacement modes.

    For each mode phi (an array over the refined vertices of ``body``), returns
    ``{"mode", "ad", "fd", "rel_err", "j_plus", "j_minus"}`` with
    ``ad = sum_i dJ/dd_i phi_i`` and ``fd = (J(+eps phi) - J(-eps phi)) / (2 eps)``.
    ``eps`` is a non-dimensional displacement amplitude (1e-4 L is ~1/100 of a
    cell on typical grids: small enough for the central difference to be
    second-order accurate, large enough to stay above float64 round-off).
    """
    rep = Progress(progress)
    if vertex_grad is None:
        _, vertex_grad = problem.gradient(progress=rep.sub(0.0, 0.3), cancel=cancel)
    out = []
    nb = len(problem.points)
    modes = np.atleast_2d(np.asarray(modes, float))
    for k, phi in enumerate(modes):
        disp = [np.zeros(len(p)) for p in problem.points]
        disp[body] = eps * phi
        jp = problem.evaluate(disp, cancel=cancel).value
        disp[body] = -eps * phi
        jm = problem.evaluate(disp, cancel=cancel).value
        fd = (jp - jm) / (2.0 * eps)
        ad = float(np.dot(vertex_grad[body], phi)) if body < nb else float("nan")
        rel = abs(ad - fd) / max(abs(fd), 1e-300)
        out.append({"mode": k, "ad": ad, "fd": fd, "rel_err": rel, "j_plus": jp, "j_minus": jm})
        rep(0.3 + 0.7 * (k + 1) / len(modes), f"Finite-difference check, mode {k + 1} of {len(modes)}")
    return out


# ---------------------------------------------------------------------------
# public driver
# ---------------------------------------------------------------------------

class SensitivityMaps(list):
    """``list[np.ndarray]``: one sensitivity map per body, dJ/dn [1/m] (see the module
    docstring for the exact definition), at the points ``FlowSolver.surface()``
    returns (same order). Metadata rides along as attributes:

    ``objective``, ``value`` (J over the window), ``coefficients`` ({"cd","cl","cm"}
    window means), ``window`` ((lead, average) in L/U), ``dt``, ``steps``,
    ``wall_seconds``, ``peak_memory_bytes`` / ``extra_memory_bytes`` (CUDA: the
    process peak during the reverse sweep, and its rise above the memory in use
    before it; None on CPU), ``device``, ``cuda_graphs``, ``bodies`` (names),
    ``points`` / ``normals`` (wind frame, metres) and ``s`` (arc length, metres)
    of the map points, ``vertex_points`` / ``vertex_gradient`` (refined outlines,
    non-dimensional, and dJ/dd per vertex), ``period`` (shedding period in L/U,
    or None), ``ensemble`` (number of averaged start phases), ``notes`` and
    ``warnings``.
    """

    def __init__(self, arrays: Sequence[np.ndarray] = (), **meta):
        super().__init__(np.asarray(a, dtype=np.float64) for a in arrays)
        self.objective: str = meta.pop("objective", "cd")
        self.value: float = meta.pop("value", float("nan"))
        self.coefficients: dict[str, float] = meta.pop("coefficients", {})
        self.window: tuple[float, float] = meta.pop("window", (float("nan"), float("nan")))
        self.dt: float = meta.pop("dt", float("nan"))
        self.steps: int = meta.pop("steps", 0)
        self.wall_seconds: float = meta.pop("wall_seconds", float("nan"))
        self.peak_memory_bytes: int | None = meta.pop("peak_memory_bytes", None)
        self.extra_memory_bytes: int | None = meta.pop("extra_memory_bytes", None)
        self.device: str = meta.pop("device", "")
        self.cuda_graphs: bool = meta.pop("cuda_graphs", False)
        self.bodies: list[str] = meta.pop("bodies", [])
        self.points: list[np.ndarray] = meta.pop("points", [])
        self.normals: list[np.ndarray] = meta.pop("normals", [])
        self.s: list[np.ndarray] = meta.pop("s", [])
        self.vertex_points: list[np.ndarray] = meta.pop("vertex_points", [])
        self.vertex_gradient: list[np.ndarray] = meta.pop("vertex_gradient", [])
        self.period: float | None = meta.pop("period", None)
        self.ensemble: int = meta.pop("ensemble", 1)
        self.length_ref: float = meta.pop("length_ref", 1.0)
        self.notes: list[str] = meta.pop("notes", [])
        self.warnings: list[str] = meta.pop("warnings", [])
        if meta:
            raise TypeError(f"unknown SensitivityMaps fields: {', '.join(meta)}")

    def summary(self) -> str:
        """One sentence for result notes and logs."""
        sym = OBJECTIVES.get(self.objective, self.objective)
        mem = (f", peak GPU memory {self.peak_memory_bytes / 2**20:.0f} MB"
               if self.peak_memory_bytes else "")
        return (f"Shape sensitivity of {sym} by automatic differentiation through the solver: "
                f"{sym} averaged over t = {self.window[0]:.3g} ... {self.window[0] + self.window[1]:.3g} "
                f"L/U after the shape change ({self.steps:,} steps of {self.dt:.3g} L/U), computed in "
                f"{self.wall_seconds:.1f} s{mem}.")


def _state_tensors(state) -> tuple[torch.Tensor, torch.Tensor, FlowSolver | None]:
    if isinstance(state, FlowSolver):
        s = state.state
        return s.u, s.v, state
    if isinstance(state, State):
        return state.u, state.v, None
    if isinstance(state, dict) and "u" in state and "v" in state:
        return torch.as_tensor(state["u"]), torch.as_tensor(state["v"]), None
    raise TypeError("state must be a FlowSolver, a solver State or a state_dict with 'u' and 'v'")


def develop_flow(cfg: SolverConfig, t_end: float, rep: Progress, cancel) -> FlowSolver:
    """Develop the flow from the impulsive start to ``t_end`` (fast path, CUDA graphs)."""
    solver = FlowSolver(dataclasses.replace(cfg, dye=False))
    rate = 200 if solver.device.type == "cpu" else 2000
    while solver.t < t_end - 1e-12:
        check_cancel(cancel)
        n = max(1, min(rate, math.ceil((t_end - solver.t) / max(solver.dt, 1e-12))))
        if solver.device.type == "cuda" and n >= 20:
            n -= n % 10
        solver.advance(n)
        c = solver.coefficients()
        if not np.all(np.isfinite(c)):
            raise RuntimeError(f"The flow became unstable at t = {solver.t:.3g} while developing it.")
        rep(min(1.0, solver.t / t_end), f"Developing the flow for the sensitivity (t = {solver.t:.1f} of {t_end:g})")
    return solver


def shedding_period(solver: FlowSolver | None, t_from: float) -> float | None:
    """Period of the lift oscillation in the solver's history (L/U), or None when steady.

    Uses the runner's own rules for a Strouhal number (``simulation.runner``:
    ``ST_MIN_PEAK_SHARE``, ``ST_MIN_RMS``, ``MIN_CYCLES``) plus a decay test, so
    the map is computed as "shedding" exactly when the result reports shedding.
    Without them a slowly decaying start-up oscillation of a steady flow counted:
    the steady Re-40 cylinder (Cl rms 0.57 % of Cd, falling to x0.59 over the
    window) got the 3x-costlier shedding branch and a warning that its flow was
    unsteady, next to a result that reported no Strouhal number.
    """
    if solver is None:
        return None
    try:
        from ..analysis import dominant_frequency
        from ..simulation.runner import AMPLITUDE_BAND, MIN_CYCLES, ST_MIN_PEAK_SHARE, ST_MIN_RMS
        h = solver.history()
        if h["t"].size < 64:
            return None
        t0 = max(t_from, float(h["t"][0]))
        f = dominant_frequency(h["t"], h["cl"], t_start=t0, min_fraction=ST_MIN_PEAK_SHARE)
        if not f or f <= 0:
            return None
        sel = np.asarray(h["t"]) >= t0
        t = np.asarray(h["t"], dtype=np.float64)[sel]
        cl = np.asarray(h["cl"], dtype=np.float64)[sel]
        cd = np.asarray(h["cd"], dtype=np.float64)[sel]
        ok = np.isfinite(t) & np.isfinite(cl) & np.isfinite(cd)
        t, cl, cd = t[ok], cl[ok], cd[ok]
        if t.size < 16 or f * float(t[-1] - t[0]) < MIN_CYCLES:
            return None
        resid = cl - np.polyval(np.polyfit(t - t[0], cl, 1), t - t[0])
        if float(resid.std()) < ST_MIN_RMS * max(abs(float(cl.mean())), abs(float(cd.mean())), 1e-6):
            return None                     # a ripple, not shedding
        # dying out: a start-up oscillation of a steady flow. Laminar flows only: the rms
        # of a chaotic LES wake wanders by more than this between two half windows
        laminar = getattr(getattr(solver, "cfg", None), "turbulence", "none") in ("none", None, "")
        half = resid.size // 2
        ra, rb = float(resid[:half].std()), float(resid[half:].std())
        if laminar and ra > 0 and rb / ra < AMPLITUDE_BAND[0]:
            return None
    except Exception:  # noqa: BLE001 - periodicity is advisory: treat as steady
        return None
    return 1.0 / f


def choose_window(unit: float, dclass: str, period: float | None,
                  window: float | tuple[float, float] | None = None,
                  ensemble: int | None = None) -> tuple[float, float, int]:
    """(lead, average, ensemble members) in L/U for a body time unit ``unit`` (L/U).

    Steady flows: :data:`STEADY_WINDOW` for the device class ("gpu" / "cpu").
    Shedding flows (``period`` in L/U): a lead of :data:`UNSTEADY_LEAD` units
    (clipped to [one unit, one period]) and whole periods of averaging covering
    :data:`UNSTEADY_MIN_AVERAGE`, from :data:`UNSTEADY_ENSEMBLE` phases. An
    explicit ``window`` (average, or (lead, average)) and ``ensemble`` override
    the defaults.
    """
    unsteady_lead = max(unit, min(UNSTEADY_LEAD * unit, period)) if period is not None else 0.0
    if window is None:
        if period is None:
            lead, avg = (w * unit for w in STEADY_WINDOW[dclass])
        else:
            lead = unsteady_lead
            avg = math.ceil(UNSTEADY_MIN_AVERAGE[dclass] * unit / period - 1e-9) * period
    elif isinstance(window, (tuple, list)):
        lead, avg = float(window[0]), float(window[1])
    else:
        avg = float(window)
        lead = STEADY_WINDOW[dclass][0] * unit if period is None else unsteady_lead
    members = int(ensemble) if ensemble is not None else (UNSTEADY_ENSEMBLE if period is not None else 1)
    return float(lead), float(avg), max(1, members)


def estimate_seconds(plan) -> float:
    """Rough wall time [s] of :func:`compute_sensitivity` reusing the run's solver.

    Assumes a steady flow (shedding flows cost ~2-4x: whole periods and three
    phases). A reverse step costs about six forward steps with CUDA graphs (0.8 ms
    forward, 3.4 ms backward replay plus the recomputation, 50 k cells) and seven
    on a CPU (eager autograd), measured against the planner's throughput model.
    Meant for the Setup page's run-time estimate.
    """
    from ..simulation.plan import estimate_steps_per_second
    cfg = plan.solver_config
    gpu = not str(plan.device).startswith("cpu")
    unit = max(float(plan.strouhal_length) / float(plan.length_ref), 1e-6)
    lead, avg = STEADY_WINDOW["gpu" if gpu else "cpu"]
    steps = (lead + avg) * unit / max(DT_SAFETY * float(plan.dt), 1e-12)
    sps = estimate_steps_per_second(plan.device, plan.cells, advection=cfg.advection,
                                    turbulence=cfg.turbulence, dtype=cfg.dtype)
    return float(steps * (6.0 if gpu else 7.0) / max(sps, 1e-9) + (3.0 if gpu else 1.0))


def compute_sensitivity(design: Design, sim: Simulation, *, objective: str | None = None, plan=None,
                        state=None, window: float | tuple[float, float] | None = None,
                        n_per_body: int | Sequence[int] | None = None, device: str | None = None,
                        progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                        ensemble: int | None = None, smooth_cells: float = SMOOTH_CELLS) -> SensitivityMaps:
    """Surface sensitivity map of ``objective`` for every body of a CFD set-up, in 1/m.

    The values are the derivative of a windowed time average from a fixed
    developed flow, taken through the immersed boundary's masks: their pattern
    and signs are reliable, their size is not an exact dJ/dn. It is biased high,
    by up to about 2x (window truncation, ~+23 % at the default window on a
    Re 40 cylinder, and the masks' stronger response to sub-cell than to
    whole-cell shape changes, ~1.5x; module docstring), and every map carries
    :data:`MAGNITUDE_WARNING`. Show it relative to its largest values.

    ``objective``: "cd", "cl", "ld" (mean Cl / mean Cd) or "cm"; default
    ``sim.cfd.sensitivity_objective``. ``plan``: the :class:`SimPlan` (planned here
    when None). ``state``: a developed :class:`FlowSolver` (the runner's, after its
    run; it is only read, never advanced), a solver ``State`` or a
    ``state_dict``; None develops the flow to ``plan.t_avg_start`` first.
    ``window``: averaging time, or ``(lead, average)``, in convective times L/U.
    Default: for a steady flow 6 + 2 body time units on a GPU, 3 + 1 on a CPU
    (body time unit = chord or frontal height over U); for a shedding flow a
    short lead (2 units, at most a period) and whole periods of averaging,
    repeated from ``ensemble`` phases of the cycle (default 3) and averaged. ``n_per_body``: points per body
    (default: what the runner uses for ``SurfaceData``). ``progress(fraction,
    message)`` and ``cancel`` (a ``threading.Event``; raises
    :class:`~aerolume.optimize.Cancelled`) as usual.

    Returns a :class:`SensitivityMaps` (a list with one array per body plus
    metadata). Positive values: pushing the surface outward there increases J.
    """
    from ..core.device import resolve
    from ..simulation.live import cpu_threads
    from ..simulation.plan import plan_simulation
    from ..solvers.fpmode import flush_denormal

    wall0 = time.perf_counter()
    obj = (objective or getattr(sim.cfd, "sensitivity_objective", None) or "cd").lower()
    _objective_fn(obj)                                   # validates
    rep = Progress(progress)
    if plan is None:
        plan = plan_simulation(design, sim, device=device)
    cfg = plan.solver_config
    if device not in (None, "", "auto"):
        cfg = dataclasses.replace(cfg, device=resolve(device), bodies=list(cfg.bodies))
    L = float(plan.length_ref)
    unit = max(float(plan.strouhal_length) / L, 1e-6)      # body convective time in L/U
    notes: list[str] = []
    warn: list[str] = [MAGNITUDE_WARNING]
    dev_name = str(cfg.device if cfg.device not in ("auto", None, "") else resolve("auto"))
    dclass = "cpu" if dev_name.startswith("cpu") else "gpu"

    with cpu_threads(dev_name), flush_denormal(dclass == "cpu"):
        if state is None:
            solver = develop_flow(cfg, max(float(plan.t_avg_start), 4.0 * unit), rep.sub(0.0, 0.3), cancel)
            u0, v0 = solver.state.u, solver.state.v
            rest = rep.sub(0.3, 1.0)
        else:
            u0, v0, solver = _state_tensors(state)
            rest = rep
        t_state = float(solver.t) if solver is not None else float("nan")

        # -- window: steady flows want a long lead; shedding flows whole periods + an ensemble
        period = shedding_period(solver, 0.5 * t_state if math.isfinite(t_state) else 0.0)
        lead, avg, members = choose_window(unit, dclass, period, window, ensemble)
        if period is not None:
            warn.append(f"The flow is unsteady (the lift oscillates with a period of {period:.3g} L/U). The map "
                        f"is the sensitivity of a time average over {avg / period:.0f} period(s), averaged over "
                        f"{members} start phase(s); it also reflects how the shape shifts the shedding, so read "
                        "it as a guide to where the shape matters rather than as exact numbers.")

        problem = WindowedObjective(cfg, u0, v0, objective=obj, lead=lead, average=avg)
        is_cuda = problem.ops.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(problem.ops.device)
            before = torch.cuda.memory_allocated(problem.ops.device)
            torch.cuda.reset_peak_memory_stats(problem.ops.device)
        vals, grads = [], []
        shift = (period or 0.0) / members
        for m in range(members):
            if m:
                problem.advance(shift, cancel=cancel)
            sub = rest.sub(0.97 * m / members, 0.97 * (m + 1) / members)
            val_m, g_m = problem.gradient(progress=sub, cancel=cancel)
            vals.append(val_m)
            grads.append(g_m)
        peak = extra = None
        if is_cuda:
            torch.cuda.synchronize(problem.ops.device)
            peak = int(torch.cuda.max_memory_allocated(problem.ops.device))
            extra = max(0, peak - int(before))
        notes.extend(problem.notes)
        vgrad = [np.mean([g[k] for g in grads], axis=0) for k in range(len(grads[0]))]
        coef = np.mean([v.coefficients for v in vals], axis=0)
        value = float(np.mean([v.value for v in vals]))

        # -- map onto the surface points of FlowSolver.surface()
        nb = len(cfg.bodies)
        if n_per_body is None:
            counts = [surface_point_count(cfg)] * nb
        elif isinstance(n_per_body, (int, np.integer)):
            counts = [int(n_per_body)] * nb
        else:
            counts = [int(c) for c in n_per_body]
            if len(counts) != nb:
                raise ValueError(f"n_per_body has {len(counts)} entries for {nb} bodies")
        pts, nrm, s_q = [], [], []
        for b, n in zip(cfg.bodies, counts):
            p, s = resample_closed(b.points, n)
            tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
            tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-30)[:, None]
            pts.append(p)
            nrm.append(np.c_[tang[:, 1], -tang[:, 0]])
            s_q.append(s)
        dens = problem.density(vgrad, s_q, sigma=smooth_cells * cfg.h)
        arrays = [g / L for g in dens]

    coeffs = {"cd": float(coef[0]), "cl": float(coef[1]), "cm": float(coef[2])}
    maps = SensitivityMaps(
        arrays, objective=obj, value=value, coefficients=coeffs, window=(problem.lead, problem.average),
        dt=problem.dt, steps=problem.steps * members, wall_seconds=time.perf_counter() - wall0,
        peak_memory_bytes=peak, extra_memory_bytes=extra, device=str(problem.ops.device),
        cuda_graphs=problem.cuda_graphs, bodies=[b.name for b in cfg.bodies],
        points=[p * L for p in pts], normals=nrm, s=[s * L for s in s_q],
        vertex_points=problem.points, vertex_gradient=vgrad, period=period, ensemble=members,
        length_ref=L, notes=notes, warnings=warn)
    maps.notes.insert(0, maps.summary())
    rep(1.0, "Sensitivity map ready")
    return maps


def attach_sensitivity(result: CFDResult, arrays: Sequence[np.ndarray]) -> CFDResult:
    """Fill ``SurfaceData.sensitivity`` of ``result`` from :func:`compute_sensitivity` output.

    Bodies are matched by name when ``arrays`` is a :class:`SensitivityMaps`
    (else by order). An array whose length differs from the body's surface data
    is resampled by normalised arc length (both start at vertex 0 and run
    counter-clockwise). With a :class:`SensitivityMaps`, its summary and
    warnings are added to the result's notes and warnings. Returns ``result``.
    """
    meta = arrays if isinstance(arrays, SensitivityMaps) else None
    names = list(meta.bodies) if meta is not None else []
    seq = [np.asarray(a, dtype=np.float64) for a in arrays]
    for i, sd in enumerate(result.surface):
        j = names.index(sd.body) if sd.body in names else i
        if j >= len(seq):
            continue
        a = seq[j]
        n = len(sd.x)
        if a.size != n:
            if a.size < 2 or n < 1:
                continue
            # both point sets are equally spaced in arc length from vertex 0
            xa = np.arange(a.size + 1) / a.size
            a = np.interp(np.arange(n) / n, xa, np.concatenate([a, a[:1]]))
        sd.sensitivity = a
    if meta is not None:
        for line in meta.notes:
            if line not in result.notes:
                result.notes.append(line)
        for line in meta.warnings:
            if line not in result.warnings:
                result.warnings.append(line)
    return result
