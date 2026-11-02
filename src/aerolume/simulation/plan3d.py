"""Automatic 3D simulation set-up: from a 3D Design + Simulation to a ready solver config.

This is the 3D counterpart of :mod:`.plan` (DESIGN.md 7.7) and follows the same
recipe, so a user who has run a 2D case meets the same reasoning, the same
words and the same summary table:

1. **Wind frame** (DESIGN.md 7.2). The geometry is yawed about +z by -yaw and
   then pitched about +y by +alpha, both through ``Design.moment_ref3``. On a
   road the wheels only yaw (a pitched car still stands on its wheels). A pitch
   that pushes the body into the road is refused with a message.
2. **Symmetry plane.** A design marked ``symmetric`` at zero yaw with
   ``cfd.symmetry == "auto"`` is simulated in the half domain y >= 0 with a
   symmetry plane at y = 0 (half the cells for the same resolution); the solver
   reports full-body forces. Yaw, an asymmetric shape or "full" use both halves.
3. **Domain** by geometry class -- vehicles (lengths ahead and behind, the
   cross-section from the blockage limit), bluff bodies (multiples of their
   equivalent frontal diameter) and wings (multiples of the mean chord). The
   cross-section keeps the same clearance above and beside the body, sized so
   that the frontal area over the tunnel cross-section stays under the
   quality's blockage limit; that bounds the blockage in y and in z at once
   (a long narrow tunnel would pass the area test and still squeeze the flow
   over the roof). Below the limit, a minimum clearance in body sizes applies.
4. **Resolution**: cells per vehicle height / per frontal size / per mean
   chord per quality and device, raised for thin features (ground clearance,
   a wing's thickness), then fitted into a cell budget (GPU ~2 / 6 / 14 M
   cells for Draft / Standard / Fine on an 8 GB card; CPU Draft-size only) by
   first shortening the domain ahead and behind toward its minimum and, for
   bluff bodies and wings, narrowing the cross-section toward 1.6 times the
   blockage limit -- vehicles keep theirs, as in 2D -- and only then by
   coarsening. 3D grids are uniform (the exact pressure solve needs it), so a
   tunnel cross-section costs cells everywhere; that is why the budgets bind.
5. **Reynolds number and flow model** exactly as in 2D (:mod:`.plan`): the real
   Re when the grid resolves the laminar boundary layer, a WALE large-eddy model
   with upwind advection when it does not; "capped" runs the cap.
6. **Duration** in convective times, early stop on the 2D convergence rules
   (:mod:`.runner`), and a **run-time estimate** from a throughput table
   measured on the reference machine (RTX 4060 Ti, fused kernels and CUDA
   graphs) corrected by this machine's measurements near the planned grid size
   (the 2D machinery; 3D keeps its own observations).

Everything here is cheap: the two geometry measurements that are not (the
mirror error behind the symmetry plane and the gaps between parts) do not
depend on the grid and are cached per shape, so the Setup page can re-plan on
every settings change. The full geometry report is the Geometry page's
:func:`aerolume.geometry3d.check_design3d`.
"""
from __future__ import annotations

import contextlib
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..core import device as devmod
from ..core import units
from ..model import Design, Fluid, Mesh3D, Simulation
from . import plan as _p2
from .physics import (
    PhysicsScales,
    Placed,
    PlacedPhysicsPlan,
    cavity_nusselt,
    closed_box,
    draft_buoyant_duration,
    has_placed_physics,
    merge_warnings,
    peak_source,
    peak_speed,
    physics_cost,
    physics_dt,
    physics_rows,
    physics_scales,
    placed_physics_3d,
    plan_monitors,
    thermal_checks,
    unbounded_heating_note,
)
from .plan import (
    BL_CELLS_MIN,
    CENTRAL_CELL_RE,
    CFL,
    CL_RTOL_FACTOR,
    CONVERGENCE_RTOL,
    LES_MODEL,
    MEMORY_FRACTION,
    NUT_PEAK,
    QUALITY_ORDER,
    THIN_SLACK,
    fmt_duration,
    fmt_estimate,
    fmt_time,
    stop_rule_text,
)

__all__ = [
    "SimPlan3D", "estimate_steps_per_second3d", "geometry_class3d", "loop_efficiency3d", "plan_live3d",
    "plan_simulation3d", "record_throughput3d", "rotation3", "wind_frame_meshes",
]

AirfoilCoords = Callable[[str], np.ndarray]

# ---------------------------------------------------------------------------
# tunables (each one justified where it is used)
# ---------------------------------------------------------------------------

#: Largest blockage (frontal area / tunnel cross-section, the full cross-section
#: also for half-domain runs) per class and quality (Draft, Standard, Fine). In 3D
#: the air passes around the sides, so blockage costs far less than in 2D, but at
#: low Re it is still measurable: the sphere at Re 100 in a 2.85 %-blockage box
#: reads +3.4 % Cd against an unconfined extrapolation (theory_ns3d.md 10).
#: Vehicles get more: their domains are long and the budget is volume-limited.
BLOCKAGE_MAX3 = {
    "vehicle": (0.05, 0.04, 0.03),
    "generic": (0.03, 0.02, 0.02),
    "wing": (0.03, 0.02, 0.02),
}
#: Largest share of the tunnel's width (and, separately, of its height) the body
#: may fill. The area ratio alone allows a long, thin body to nearly touch two
#: walls: a 10 m wing passes a 3 % area limit in a 12 m wide tunnel, with its tip
#: vortices a metre from the walls. Wind-tunnel practice keeps a wing's span under
#: ~80 % of the test section; the wing limits are tighter because the CFD walls
#: are frictionless mirrors that bind the tip vortices.
DIRECTION_MAX3 = {
    "vehicle": (0.5, 0.45, 0.4),
    "generic": (0.4, 0.35, 0.3),
    "wing": (0.6, 0.55, 0.5),
}
#: Explore sessions trade blockage for a small, fast grid.
BLOCKAGE_LIVE3 = 0.08
#: Minimum gap between the body and a side or top wall, in body sizes (vehicle
#: height, frontal size, mean chord) -- binding only where blockage alone would
#: allow a tighter tunnel (thin wings, low flat bodies).
MIN_CLEARANCE3 = {"vehicle": (1.5, 1.5, 2.0), "generic": (1.5, 1.75, 2.0), "wing": (2.0, 2.5, 3.0)}
#: Resolution presets: cells per vehicle *height* (road to roof), per equivalent
#: frontal diameter of a bluff body, per mean aerodynamic chord of a wing. The
#: sphere converges to ~2 % at 16 cells/D and ~1 % at 24 at Re 100
#: (theory_ns3d.md 10); a car's wake and shear layers scale with its height.
CELLS_PER_SIZE3 = {
    "vehicle": {"gpu": (24, 32, 40), "cpu": (12, 16, 20)},
    "generic": {"gpu": (20, 24, 32), "cpu": (10, 12, 16)},
    "wing": {"gpu": (16, 24, 32), "cpu": (8, 10, 12)},
}
#: Cell budgets per device class and quality. GPU: DESIGN.md 7.1 for an 8 GB card
#: (Draft ~1-2 M, Standard ~4-6 M, Fine ~10-14 M cells; float32 LES peaks at
#: ~200-260 B/cell, so 14 M cells stay under 4 GB). CPU: Draft-size grids only
#: (~13-30 steps/s at 0.2 M cells with 4 threads); the budgets keep a CPU Draft
#: run in minutes, not hours.
CELL_BUDGET3 = {
    "gpu": (2_000_000, 6_000_000, 14_000_000),
    "cpu": (150_000, 300_000, 500_000),
}
LIVE_BUDGET3 = {"gpu": 400_000, "cpu": 40_000}
#: Cells across a thin feature (ground clearance, wing thickness, a thin part).
#: Lower than 2D's 4 / 6 / 8: every extra cell across costs the cube in 3D, and
#: the voxeliser keeps a feature solid down to ~2 cells.
THIN_CELLS3 = (3.0, 4.0, 6.0)
LIVE_THIN_CELLS3 = 2.0
#: ... but a thin feature raises the preset resolution by at most this factor (wings
#: less: their thickness is the whole wing, and the domain pays for every raise).
THIN_RAISE_MAX = {"vehicle": 1.5, "generic": 1.5, "wing": 1.25}
#: Domain ahead of / behind the body: vehicles in vehicle lengths, bluff bodies in
#: frontal sizes, wings in mean chords (preferred per quality, and the minimum the
#: budget may shrink them to). A 3D wake recovers far faster than a 2D one, so
#: shorter runs behind than in 2D suffice.
DOMAIN3 = {
    "vehicle": {"up": (1.0, 1.25, 1.5), "down": (2.0, 2.5, 3.0), "min": {"up": 0.75, "down": 1.5}},
    "generic": {"up": (5.0, 6.0, 6.0), "down": (10.0, 12.0, 12.0), "min": {"up": 3.0, "down": 6.0}},
    "wing": {"up": (3.0, 4.0, 5.0), "down": (5.0, 6.0, 8.0), "min": {"up": 2.0, "down": 3.0}},
}
#: Run length (t_end, average_from): vehicles and wings in L/U, bluff bodies in
#: frontal sizes / U (scaled to L/U). A 3D wake loses its start-up transient in a
#: few flow-throughs of the body and its drag fluctuates far less than behind a
#: 2D vortex street (Ahmed-type body at Draft: Cd rms 3 % of the mean), so drag
#: converges quickly; the lift takes longer. Measured on the Draft Ahmed-type body
#: (25 deg slant, moving road, 2026-10): with 12 L/U the transient ended at
#: t = 4.2 and Cl was "not converged" (9 effective samples, drifting); with 20 L/U
#: the run stopped early, converged, at t = 19 (Cd +-0.8 %, Cl +-0.02).
DURATION3 = {
    "vehicle": ((20.0, 5.0), (25.0, 6.0), (30.0, 8.0)),
    "generic": ((40.0, 20.0), (60.0, 30.0), (80.0, 40.0)),
    "wing": ((20.0, 6.0), (25.0, 8.0), (30.0, 10.0)),
}
DRAFT_T_END_MAX3 = 60.0
#: (max|u| + max|v| + max|w|) / U that sets the CFL step (the solver sums the three
#: maxima). Measured over whole runs (2026-10): 2.1 for the sphere at Re 100 (20
#: cells/D), 3.0 for the wall-unresolved LES of the Draft Ahmed-type body on a
#: moving road; the others are estimates. Refined by measurement (record_throughput3d).
SPEED_FACTOR3 = {
    ("generic", "laminar"): 2.1, ("generic", "les"): 2.8,
    ("vehicle", "laminar"): 2.3, ("vehicle", "les"): 3.0,
    ("wing", "laminar"): 2.2, ("wing", "les"): 2.8,
    # closed boxes, as in 2D (plan.CLOSED_SPEED_FACTOR): the cubic cavity's centre-line maxima are
    # ~0.2 and ~0.25 of the free-fall velocity (Tric et al. 2000), forced boxes are driven at U
    ("closed-natural", "laminar"): 0.6, ("closed-natural", "les"): 0.7,
    # a box warmed by heaters runs on their plumes (2D: 2.35 of the free-fall velocity, plan.py);
    # not measured in 3D, where the three maxima are summed: the 2D value scaled by 3/2, an estimate
    ("closed-natural+heater", "laminar"): 3.5, ("closed-natural+heater", "les"): 3.5,
    ("closed-forced", "laminar"): 1.8, ("closed-forced", "les"): 1.8,
    # relative to the peak speed (physics.peak_speed): the Draft 3D room (a 1 m/s ceiling fan, 1 m/s
    # supply, a 200 W radiator) stepped at 0.0019 L/U where (max|u| + max|v| + max|w|) = 1.8 U had
    # planned 0.0072, i.e. 6.7 U, 3.4 times its peak (twice the fan's disc velocity; critic, 2026-10)
    ("closed-fan", "laminar"): 3.4, ("closed-fan", "les"): 3.4,
    # a tunnel run planned on its peak speed: the Draft hot sphere (D 0.1 m, Re 100, 60 °C in 20 °C
    # air, Ri 627, half domain) held (max|u| + max|v| + max|w|) at 117-119 U = 4.7 times its buoyant
    # velocity 25 U from t = 0.4 to 3 L/U: the plume reaches the tunnel's ceiling 0.21 m above the
    # sphere at 1.8 u_b and spreads under it at 1.5 u_b (fix round 3.2, 2026-10)
    ("generic-plume", "laminar"): 4.7, ("generic-plume", "les"): 4.7,
    # a jet or a fan's slipstream outrunning the stream: not measured yet (the jet's speed plus the
    # cross-flow it entrains); the first run of the class replaces it
    ("generic-jet", "laminar"): 2.0, ("generic-jet", "les"): 2.4,
}
#: The solver's explicit viscous limit h^2 / (6 nu) (ns3d.core.VISCOUS_DT).
VISCOUS_DT3 = 1.0 / 6.0
#: Grid sizes are rounded up to multiples of this (the dense-transform pressure
#: solve takes any size; even sizes keep the half-domain tunnels symmetric).
GRID_MULTIPLE = 4
#: Live sessions: attitude range the mask window and the domain are sized for.
LIVE_PITCH3 = 10.0
LIVE_PITCH_GROUND3 = 3.0
LIVE_YAW3 = 15.0
#: Largest grid a 3D sensitivity map is attempted on when the sensitivity module does
#: not say (DESIGN.md 7.1: ~1.5 M cells on 8 GB); the module's own budget
#: (optimize.sensitivity3d.DEFAULT_MAX_CELLS) wins when it is importable.
SENSITIVITY_MAX_CELLS = 1_500_000
#: Wall-unresolved 3D LES of a car: how to read it (stated with every vehicle plan).
#: Measured 2026-10 on an Ahmed-type body (25 deg slant, 1.044 m, moving road, 40 m/s,
#: half domain): Cd 0.637 +- 0.005 at Draft (1.9 M cells) and 0.611 +- 0.006 at Standard
#: (5.9 M), against about 0.29 in Ahmed's wind tunnel; Cl +0.08 and -0.06.
VEHICLE_3D_NOTE = (
    "Reading the numbers: this is a wall-unresolved large-eddy simulation on a uniform grid with no wall or "
    "transition model, so friction drag and separation from smooth surfaces are approximate. On an Ahmed-type "
    "reference body (25° slant, moving road) it gave Cd 0.64 at Draft and 0.61 at Standard where the wind "
    "tunnel measures about 0.29, so treat absolute Cd and Cl as indicative only, and compare designs "
    "simulated at the same quality and the same conditions.")
WING_CFD_NOTE = (
    "For a wing's lift and drag at flight Reynolds numbers use the Wing analysis (vortex lattice with "
    "viscous section polars): a CFD grid this coarse cannot resolve the boundary layers of a lifting "
    "surface, which sets both. Use the CFD for the flow pattern (tip vortices, separation, interference).")

# ---------------------------------------------------------------------------
# throughput model (3D) -- measured, then corrected by measurement
# ---------------------------------------------------------------------------

#: FlowSolver3D.advance throughput on the reference machine [cells -> steps/s]:
#: RTX 4060 Ti, float32, fused kernels and CUDA graphs, running means on, a
#: sphere in a half domain (idle GPU, 2026-10); CPU: i7-13700F, 4 threads,
#: float32, reference core. Interpolated in log-log like the 2D table.
_REF_RATES3: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("gpu", "upwind3"): [(65536, 1334), (221184, 1014), (524288, 668), (1024000, 322), (2000000, 136),
                         (4194304, 57.3), (8192000, 26.4)],
    ("gpu", "les"): [(65536, 1081), (221184, 770), (524288, 462), (1024000, 199), (2000000, 81.0),
                     (4194304, 33.8), (8192000, 16.2)],
    ("gpu", "central"): [(65536, 1054), (221184, 1096), (524288, 761), (1024000, 345), (2000000, 137),
                         (4194304, 57.4), (8192000, 26.4)],
    ("cpu", "upwind3"): [(8192, 92.1), (27648, 66.8), (65536, 53.5), (221184, 30.2)],
    ("cpu", "les"): [(8192, 52.5), (27648, 34.0), (65536, 25.7), (221184, 13.1)],
    ("cpu", "central"): [(8192, 95.0), (27648, 69.0), (65536, 55.0), (221184, 31.0)],
}
#: dtype factors against the float32 tables (float64 is ~2x the memory traffic;
#: consumer GPUs also run it at a fraction of the float32 rate).
_DTYPE_FACTOR3 = {("cpu", "float64"): 0.6, ("gpu", "float64"): 0.3}
#: Share of the loop's wall time spent stepping: a 3D step costs far more than
#: rendering one slice, so the live view takes less than in 2D.
LOOP_EFFICIENCY3 = {"live": 0.88, "headless": 0.97}


def _model_rate3(dev_class: str, flavour: str, dtype: str, cells: float) -> float:
    table = _REF_RATES3[(dev_class, flavour)]
    xs = np.log([c for c, _ in table])
    ys = np.log([r for _, r in table])
    x = math.log(max(cells, 1.0))
    if x <= xs[0]:
        y = ys[0] + 0.3 * (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    elif x >= xs[-1]:
        y = ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    else:
        y = float(np.interp(x, xs, ys))
    return math.exp(y) * _DTYPE_FACTOR3.get((dev_class, dtype), 1.0)


def estimate_steps_per_second3d(device: str, cells: float, *, advection: str = "upwind3",
                                turbulence: str = "none", dtype: str = "float32") -> float:
    """Expected FlowSolver3D throughput [steps/s] on ``device`` for ``cells`` cells.

    The measured reference table, corrected by this machine's 3D measurements
    near this grid size (the kernel-weighted scheme of :func:`.plan._local`;
    3D observations live in their own stores, so 2D estimates never see them).
    """
    dc, fl = _p2._device_class(device), _p2._flavour(advection, turbulence)
    base = _model_rate3(dc, fl, dtype, cells)
    with _p2._CACHE_LOCK:
        corr = _p2._local(_p2._RATE_CACHE3D, ("3d", dc, fl, dtype), cells,
                          lambda k: k[1] == dc and k[3] == dtype)
    return base * (corr if corr is not None else 1.0)


def loop_efficiency3d(device: str, cells: float, *, live_view: bool = True) -> float:
    """Stepping share of a 3D run's loop (see LOOP_EFFICIENCY3), measured when available."""
    dc, mode = _p2._device_class(device), ("live" if live_view else "headless")
    with _p2._CACHE_LOCK:
        v = _p2._local(_p2._LOOP_CACHE3D, ("3d", dc, mode), cells, lambda k: k == ("3d", dc, mode))
    return float(np.clip(v if v is not None else LOOP_EFFICIENCY3[mode], 0.2, 1.0))


def record_throughput3d(device: str, cells: float, steps_per_s: float | None, *, advection: str = "upwind3",
                        turbulence: str = "none", dtype: str = "float32", kind: str | None = None,
                        speed_factor: float | None = None, cells_per_ref: float | None = None,
                        loop_efficiency: float | None = None, live_view: bool = False) -> None:
    """Feed a 3D run's measurements back into the 3D estimates (see :func:`.plan.record_throughput`)."""
    if not (math.isfinite(cells) and cells > 0):
        return
    dc, fl = _p2._device_class(device), _p2._flavour(advection, turbulence)
    model = "les" if fl == "les" else "laminar"

    def ok(v: float | None) -> bool:
        return v is not None and math.isfinite(v) and v > 0

    changed = False
    with _p2._CACHE_LOCK:
        if ok(steps_per_s):
            _p2._observe(_p2._RATE_CACHE3D, ("3d", dc, fl, dtype), cells,
                         float(steps_per_s) / _model_rate3(dc, fl, dtype, cells))
            changed = True
        if kind and ok(speed_factor):
            cpl = float(cells_per_ref) if ok(cells_per_ref) else 32.0
            current = _speed_factor3_locked(kind, model, cpl)
            _p2._observe(_p2._SPEED_CACHE3D, ("3d", kind, model), cpl, _p2.clamp_update(float(speed_factor), current))
            changed = True
        if ok(loop_efficiency):
            _p2._observe(_p2._LOOP_CACHE3D, ("3d", dc, "live" if live_view else "headless"), cells,
                         min(float(loop_efficiency), 1.0))
            changed = True
    if changed:
        _p2._save_calibration()


def _default_speed_factor3(kind: str, model: str) -> float:
    """The class default of a 3D calibration key (before this machine has measured it): the most
    specific signature SPEED_FACTOR3 holds (:func:`.plan.signature_defaults`), else the class's."""
    for key in _p2.signature_defaults(kind)[:-1]:
        if (key, model) in SPEED_FACTOR3:
            return SPEED_FACTOR3[(key, model)]
    base = _p2.base_kind(kind)
    fallback = 2.5 if model == "laminar" else 3.0
    for src in ("plume", "jet"):
        if base.endswith("-" + src):
            # relative to the peak speed: other classes start from the generic body's measurement
            return SPEED_FACTOR3.get((base, model), SPEED_FACTOR3[("generic-" + src, model)])
    return SPEED_FACTOR3.get((base, model), SPEED_FACTOR3.get((base.removesuffix("-physics"), model), fallback))


def _speed_factor3_locked(kind: str, model: str, cells_per_ref: float) -> float:
    """:func:`_speed_factor3` for a caller that holds the calibration lock."""
    default = _default_speed_factor3(kind, model)
    v = _p2.speed_median(_p2._SPEED_CACHE3D, ("3d", kind, model), cells_per_ref, prior=default)
    return v if v is not None else default


def _speed_factor3(kind: str, model: str, cells_per_ref: float) -> float:
    """The 3D CFL speed factor to plan with: this key's measured median near this resolution, else
    the class default (the 2D rules of :func:`.plan._speed_factor`)."""
    with _p2._CACHE_LOCK:
        return _speed_factor3_locked(kind, model, cells_per_ref)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def geometry_class3d(design: Design) -> str:
    """Layout rules of a 3D design: "vehicle", "generic" (bluff bodies) or "wing"."""
    return design.kind if design.kind in ("vehicle", "wing") else "generic"


def rotation3(alpha_deg: float, yaw_deg: float) -> np.ndarray:
    """DESIGN.md 7.2 attitude as a 3x3 matrix: yaw about +z by -yaw, then pitch about +y by +alpha.

    Positive alpha lifts the nose (which points to -x); positive yaw turns the
    nose toward +y, so the wind comes from the vehicle's left.
    """
    a, b = math.radians(alpha_deg), math.radians(-yaw_deg)
    ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    return ry @ rz


def _wheel_frame(m: Mesh3D) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Centre, axle and radius of a wheel mesh (stored values first, measured otherwise)."""
    from ..geometry3d import wheel_params
    try:
        c, ax, r = wheel_params(m)
    except ValueError:
        b = m.bounds()
        c = (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
        ax, r = (0.0, 1.0, 0.0), 0.5 * (b[5] - b[4])
    if m.center is not None:
        c = tuple(float(v) for v in m.center)
    if m.axis is not None and np.linalg.norm(m.axis) > 0:
        a = np.asarray(m.axis, dtype=np.float64)
        ax = tuple(float(v) for v in a / np.linalg.norm(a))
    return c, ax, float(r)  # type: ignore[return-value]


def wind_frame_meshes(meshes: list[Mesh3D], alpha_deg: float, yaw_deg: float,
                      about: tuple[float, float, float], *, ground: bool = False) -> list[Mesh3D]:
    """The meshes in the wind frame (metres): yawed, then pitched about ``about``.

    On a road the wheels only yaw (a pitched car still stands on its wheels).
    Vertex order is kept, so per-vertex results map straight back onto the
    user's mesh. Wheel centres and axes rotate with their mesh; ``omega`` is
    copied unchanged (the planner sets it).
    """
    ref = np.asarray(about, dtype=np.float64)
    r_full = rotation3(alpha_deg, yaw_deg)
    r_yaw = rotation3(0.0, yaw_deg)
    out = []
    for m in meshes:
        rot = r_yaw if (ground and m.role == "wheel") else r_full
        v = (np.asarray(m.vertices, dtype=np.float64) - ref) @ rot.T + ref
        center = axis = None
        if m.center is not None:
            center = tuple(float(x) for x in (np.asarray(m.center) - ref) @ rot.T + ref)
        if m.axis is not None:
            axis = tuple(float(x) for x in np.asarray(m.axis, dtype=np.float64) @ rot.T)
        out.append(Mesh3D(name=m.name, vertices=v, faces=np.asarray(m.faces), role=m.role, omega=m.omega,
                          center=center, axis=axis, thermal=getattr(m, "thermal", "adiabatic"),
                          thermal_value=getattr(m, "thermal_value", 0.0)))
    return out


def _unique_mesh_names(meshes: list[Mesh3D]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, m in enumerate(meshes):
        name = (m.name or f"Part {i + 1}").strip() or f"Part {i + 1}"
        k = seen.get(name, 0)
        seen[name] = k + 1
        out.append(name if k == 0 else f"{name} ({k + 1})")
    return out


def _bbox(meshes: list[Mesh3D]) -> tuple[float, float, float, float, float, float]:
    lo = np.min([np.asarray(m.vertices).min(axis=0) for m in meshes], axis=0)
    hi = np.max([np.asarray(m.vertices).max(axis=0) for m in meshes], axis=0)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])


def _frontal_area(meshes: list[Mesh3D], fallback: float) -> float:
    """Projected area along x of the wind-frame geometry (union of all parts), m²."""
    a = 0.0
    with contextlib.suppress(Exception):     # a planning estimate must not fail on an odd mesh
        from ..geometry3d import frontal_area
        a = float(frontal_area(meshes, axis="x", resolution=320))
    return a if (a > 0 and math.isfinite(a)) else fallback


def _thickness(m: Mesh3D) -> float:
    """Typical solid thickness of one part: 2 V / A for a closed part (a plate of
    thickness t gives t, a sphere a third of its diameter), else its smallest
    bounding-box extent (open sheets have no volume)."""
    from ..geometry3d import area, is_watertight, volume
    b = m.bounds()
    smallest = min(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    with contextlib.suppress(Exception):     # diagnostics only
        if is_watertight(m):
            v, a = abs(volume(m)), area(m)
            if v > 0 and a > 0:
                return min(2.0 * v / a, smallest)
    return smallest


def _wing_thickness(design: Design) -> float:
    """Thickness of the thinnest wing section [m] (NACA t/c from the code, 12 % otherwise)."""
    from ..geometry3d import naca_code
    best = math.inf
    for s in design.wing.sections if design.wing is not None else []:
        code = naca_code(s.airfoil)
        tc = int(code[-2:]) / 100.0 if code else 0.12
        best = min(best, tc * float(s.chord))
    return best


def _resolve_precision3(precision: str | None, device: str, warnings: list[str]) -> str:
    """3D runs use float32 unless float64 is asked for explicitly: a 3D field is
    up to 14 M cells, float64 doubles the memory and, on consumer GPUs, divides
    the speed by several; the solver is validated in float32 (theory_ns3d.md)."""
    if precision in (None, "", "auto", "float32"):
        return "float32"
    try:
        dt = devmod.resolve_dtype(precision, device)
    except (RuntimeError, KeyError):
        warnings.append(f"Precision '{precision}' is not supported on this device; using float32.")
        return "float32"
    return "float64" if "64" in str(dt) else "float32"


def _clearance_for(area: float, w: float, h: float, limit: float, *, ground: bool) -> float:
    """Uniform wall clearance c [m] giving blockage ``limit``: A / ((w + 2c)(h + c or 2c)) = limit."""
    target = area / max(limit, 1e-9)
    # (w + 2c)(h + k c) = target, k = 1 on a road (the floor is the road), 2 in free flight
    k = 1.0 if ground else 2.0
    a2, a1, a0 = 2.0 * k, 2.0 * h + k * w, w * h - target
    if a0 >= 0:
        return 0.0
    return (-a1 + math.sqrt(a1 * a1 - 4.0 * a2 * a0)) / (2.0 * a2)


def _round_up(n: float, m: int = GRID_MULTIPLE) -> int:
    return max(m * math.ceil(max(n, 1.0) / m - 1e-9), 2 * m)


# Geometry facts the plan needs that cost more than a keystroke: the measured mirror
# error and the gaps between parts. They do not depend on the grid, so they are
# measured once per design (and shape) and the grid-dependent warnings are derived
# from them on every re-plan. (check_design3d with a resolution measured the same
# gaps by dense surface sampling: 2.5 s for a five-part car, quadratic in the number
# of parts -- too slow for the Setup page, which re-plans on every settings change.
# The Geometry page shows check_design3d's full report.)
_FACTS_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_FACTS_LOCK = threading.Lock()
#: Frontal areas of the wind-frame geometry per (shape, attitude, road): rasterising
#: the projection of a 200 k-triangle mesh takes ~0.14 s, too much for every re-plan.
_AREA_CACHE: OrderedDict[tuple, float] = OrderedDict()
#: Mirror error (RMS, of the largest dimension) up to which a half domain is used;
#: the threshold design_from_meshes uses to set ``symmetric``.
SYMMETRY_TOL3 = 0.005
#: Parts farther apart than this (in body sizes, bounding boxes) are not measured.
GAP_SEARCH = 0.15
#: Fewer cells than this across a gap between parts: closed (2D: CLOSED_GAP_CELLS).
CLOSED_GAP_CELLS3 = 2.5


def _design_key(design: Design, meshes: list[Mesh3D]) -> tuple:
    return (design.id, design.kind, float(design.length_ref),
            tuple((m.name, m.role, len(m.vertices), len(m.faces),
                   tuple(round(float(x), 9) for x in m.bounds())) for m in meshes))


def _gap(a: Mesh3D, b: Mesh3D, margin: float, size: float) -> float:
    """Smallest distance between the surfaces of two parts (inf beyond ``margin``, or when
    they touch or interpenetrate: a mirror pushed into the body is not a gap); the same
    measurement as the Geometry page's report (:func:`aerolume.geometry3d.part_gap`)."""
    from ..geometry3d import part_gap
    return part_gap(a, b, margin, size=size, samples=3000)


def _geometry_facts(design: Design, meshes: list[Mesh3D]) -> dict:
    """{"sym_err": RMS mirror error [m] or None, "gaps": [(name_a, name_b, gap_m)]}, cached per shape."""
    key = _design_key(design, meshes)
    with _FACTS_LOCK:
        if key in _FACTS_CACHE:
            _FACTS_CACHE.move_to_end(key)
            return _FACTS_CACHE[key]
    from ..geometry3d import symmetry_error
    facts: dict = {"sym_err": None, "gaps": [],
                   # per-part thickness does not depend on the attitude (0.2 s for 200 k triangles)
                   "thickness": {m.name: _thickness(m) for m in meshes if m.role != "wheel"}}
    bb = _bbox(meshes)
    size = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
    if design.symmetric:
        with_err = None
        with contextlib.suppress(Exception):     # a failed measurement keeps the user's flag
            nf = sum(len(m.faces) for m in meshes)
            with_err = float(symmetry_error(meshes, plane_y=0.0, n_samples=6000 if nf < 2_000_000 else 3000))
        facts["sym_err"] = with_err
    parts = [m for m in meshes if m.role != "wheel"]
    if 1 < len(parts) <= 40:
        margin = GAP_SEARCH * size
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                a, b = parts[i], parts[j]
                ba, bb2 = a.bounds(), b.bounds()
                apart = max(ba[0] - bb2[1], bb2[0] - ba[1], ba[2] - bb2[3], bb2[2] - ba[3], ba[4] - bb2[5],
                            bb2[4] - ba[5])
                if apart > margin:
                    continue
                with contextlib.suppress(Exception):
                    g = _gap(a, b, margin, size)
                    if math.isfinite(g):
                        facts["gaps"].append((a.name, b.name, g))
    with _FACTS_LOCK:
        _FACTS_CACHE[key] = facts
        while len(_FACTS_CACHE) > 16:
            _FACTS_CACHE.popitem(last=False)
    return facts


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

@dataclass
class SimPlan3D(PlacedPhysicsPlan):
    """Everything the 3D runner needs, plus the rationale shown in the Setup page.

    Same public members as :class:`~aerolume.simulation.plan.SimPlan` where they
    apply (times in L/U, lengths in metres) plus the 3D ones: ``grid3``,
    ``symmetry_plane``, ``extent3`` (the *solved* domain: y >= 0 only with a
    symmetry plane), ``bodies_wind3`` (wind-frame meshes in the order of the
    design's meshes, or the lofted wing) and ``focus_extent3``. ``extent`` is
    the side view (x, z) of the domain and ``bodies_wind`` is empty (2D
    outlines do not exist here), so 2D-minded callers degrade gracefully.
    """
    solver_config: object                 # ns3d.SolverConfig3D
    design: Design
    sim: Simulation
    length_ref: float
    speed: float
    fluid: Fluid
    re_real: float
    re_sim: float
    turbulence: str                       # "laminar" | "les-<model>"
    t_end: float
    t_avg_start: float
    dt: float
    steps_estimate: int
    est_seconds: float
    cells: int
    cells_per_ref: float
    bodies_wind: list = field(default_factory=list)
    extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # --- extras (the 2D names) -------------------------------------------------
    device: str = "cpu"
    device_name: str = ""
    precision: str = "float32"
    advection: str = "upwind3"
    quality: str = "draft"
    ground: bool = False
    ground_moving: bool = False
    strouhal_length: float = 1.0
    blockage: float = float("nan")
    ref_label: str = "reference length"
    live: bool = False
    capped: bool = False
    geometry_class: str = "generic"
    conv_rtol: float = 0.05
    blockage_limit: float = 0.03
    est_seconds_headless: float = float("nan")
    bl_resolved: bool = True
    # --- 3D ---------------------------------------------------------------------
    symmetry_plane: bool = False
    extent3: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    bodies_wind3: list[Mesh3D] = field(default_factory=list)
    #: the design-frame meshes the plan was made from (lofted for a wing), names made unique
    meshes_design: list[Mesh3D] = field(default_factory=list)
    alpha_deg: float = 0.0
    yaw_deg: float = 0.0
    ref_area: float = 1.0
    span_ref: float = 1.0
    moment_ref3: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: blockage split by direction: body height / tunnel height, body width / tunnel width
    blockage_z: float = float("nan")
    blockage_y: float = float("nan")
    #: default slice positions [m] per normal axis (the runner's three mean slices)
    slice_positions: dict[str, float] = field(default_factory=dict)
    #: (xmin, xmax, ymin, ymax, zmin, zmax) of the wind-frame body (full body), metres
    body_bounds: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    # --- placed physics (DESIGN.md 10; the 2D SimPlan's fields, simulation/physics.py) --------------
    domain_mode: str = "tunnel"
    physics: PhysicsScales | None = None
    placed: Placed | None = None
    monitors: tuple[str, ...] = ()
    calibration_kind: str = ""
    step_cost: float = 1.0
    #: peak speed over U and what sets it (the 2D SimPlan's fields, physics.peak_speed)
    speed_ratio: float = 1.0
    peak_basis: str = ""
    #: the planner's own estimate (``est_seconds`` becomes the runner's measured re-estimate)
    est_seconds_planned: float = float("nan")
    est_after_seconds: float = 0.0
    #: the 2D SimPlan's field: the thermal checks' grid, for re-deriving them on the measured rise
    thermal_grid: tuple[float, float, float] | None = None

    dimension = 3                          # class attribute: run_cfd / LiveSession dispatch on it

    @property
    def time_unit(self) -> float:
        """Seconds per convective time unit (length_ref / speed)."""
        return self.length_ref / self.speed

    @property
    def grid3(self) -> tuple[int, int, int]:
        cfg = self.solver_config
        return (cfg.nx, cfg.ny, cfg.nz)

    @property
    def grid(self) -> tuple[int, int, int]:
        """(nx, ny, nz) -- the 3D grid (2D plans return (nx, ny))."""
        return self.grid3

    @property
    def extent3_full(self) -> tuple[float, float, float, float, float, float]:
        """The domain including the mirrored half when a symmetry plane is used."""
        x0, x1, _, y1, z0, z1 = self.extent3
        return (x0, x1, -y1, y1, z0, z1) if self.symmetry_plane else self.extent3

    @property
    def focus_extent3(self) -> tuple[float, float, float, float, float, float]:
        """A default 3D view box (metres): the body and its near wake, inside the (full) domain."""
        ex = self.extent3_full
        bx0, bx1, by0, by1, bz0, bz1 = self.body_bounds
        lb = max(bx1 - bx0, 1e-9)
        size = max(lb, by1 - by0, bz1 - bz0)
        x0, x1 = max(ex[0], bx0 - 0.5 * lb), min(ex[1], bx1 + 2.0 * lb)
        m = 0.6 * size
        y0, y1 = max(ex[2], by0 - m), min(ex[3], by1 + m)
        z0 = ex[4] if self.ground else max(ex[4], bz0 - m)
        z1 = min(ex[5], bz1 + m)
        return (x0, x1, y0, y1, z0, z1)

    @property
    def focus_extent(self) -> tuple[float, float, float, float]:
        """Default 2D view window (x, z) of the side view, metres."""
        f = self.focus_extent3
        return (f[0], f[1], f[4], f[5])

    def focus_window(self, axis: str) -> tuple[float, float, float, float]:
        """Default 2D view window of a slice normal to ``axis`` (SliceData orientation)."""
        f = self.focus_extent3
        if axis == "z":
            return (f[0], f[1], f[2], f[3])
        if axis == "x":
            return (f[2], f[3], f[4], f[5])
        return (f[0], f[1], f[4], f[5])

    @property
    def estimate_label(self) -> str:
        return "" if self.live else fmt_estimate(self.est_seconds)

    def summary(self) -> dict[str, str]:
        """Label -> formatted value for the Setup page (shown verbatim)."""
        cfg = self.solver_config
        if self.turbulence.startswith("les"):
            model = self.turbulence.split("-", 1)[1].upper() if "-" in self.turbulence else "sub-grid"
            flow = f"LES ({model} sub-grid model), {self.advection} advection"
        else:
            flow = f"Laminar (all scales resolved), {self.advection} advection"
        x0, x1, y0, y1, z0, z1 = self.extent3_full
        sim_re = units.fmt_reynolds(self.re_sim) + (" (capped)" if self.capped else "")
        full_cells = self.cells * (2 if self.symmetry_plane else 1)
        rows = {
            "Real Reynolds number": units.fmt_reynolds(self.re_real),
            "Simulated Reynolds number": sim_re,
            "Flow model": flow,
            "Attitude": f"α {self.alpha_deg:g}°, yaw {self.yaw_deg:g}°",
            "Grid": f"{cfg.nx:,} × {cfg.ny:,} × {cfg.nz:,}",
            "Cells": f"{self.cells:,}",
            "Cells (3D)": (f"{self.cells / 1e6:.2f} M solved (half domain; {full_cells / 1e6:.2f} M for both halves)"
                           if self.symmetry_plane else f"{self.cells / 1e6:.2f} M"),
            f"Cells per {self.ref_label}": f"{self.cells_per_ref:.0f}",
            "Domain": f"{units.sig(x1 - x0, 3)} m × {units.sig(y1 - y0, 3)} m × {units.sig(z1 - z0, 3)} m",
            "Symmetry plane": ("y = 0 (half domain simulated)" if self.symmetry_plane else "none (full domain)"),
            "Blockage": (f"{100 * self.blockage:.1f} % (height {100 * self.blockage_z:.0f} %, "
                         f"width {100 * self.blockage_y:.0f} %)"),
            "Road": ("moving belt" if self.ground_moving else "fixed floor") if self.ground else "none (free flow)",
            "Time step": f"{units.sig(self.dt, 3)} L/U ({fmt_time(self.dt * self.time_unit)})",
            "Duration": f"{self.t_end:.4g} L/U ({fmt_time(self.t_end * self.time_unit)} of flow)",
            "Averaging from": f"t = {self.t_avg_start:.4g} L/U",
            "Stops early when": (f"Cd known to ±{100 * self.conv_rtol:g} %, "
                                 f"Cl to ±{100 * CL_RTOL_FACTOR * self.conv_rtol:g} %"),
            "Steps (estimate)": f"≈ {self.steps_estimate:,}",
            "Estimated run time": fmt_duration(self.est_seconds_planned if math.isfinite(self.est_seconds_planned)
                                               else self.est_seconds),
            "Device": f"{self.device_name} · {self.precision}",
        }
        if self.closed:
            rows.pop("Real Reynolds number", None)
            rows["Simulated Reynolds number"] = f"{units.fmt_reynolds(self.re_sim)} (U L / ν with U the reference velocity)"
            for k in ("Blockage", "Attitude"):
                rows.pop(k, None)
            rows["Road"] = "moving floor" if self.ground_moving else "fixed walls"
        if self.monitors and not self.live:
            rows["Stops early when"] = stop_rule_text(self)
        rows.update(physics_rows(self.physics, self.placed, closed=self.closed))
        if self.live:
            for k in ("Duration", "Averaging from", "Stops early when", "Steps (estimate)", "Estimated run time"):
                rows.pop(k, None)
        return rows


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _design_meshes(design: Design, airfoil_coords: AirfoilCoords | None) -> list[Mesh3D]:
    """The design's meshes, or the wing lofted from its planform."""
    if design.meshes:
        ms = [m for m in design.meshes if len(m.faces) and len(m.vertices)]
        if not ms:
            raise ValueError("The design's meshes have no triangles loaded (their arrays are missing); "
                             "re-open the project or re-import the geometry.")
        return ms
    if design.kind == "wing" and design.wing is not None:
        from ..geometry3d import loft
        try:
            return [loft(design.wing, airfoil_coords=airfoil_coords, name=design.name or "Wing")]
        except ValueError as exc:
            if "airfoil_coords" in str(exc):
                raise ValueError("The wing uses an airfoil from the project (\"design:<id>\"); lofting it for 3D CFD "
                                 "needs the project. Run it from the application, or use NACA sections.") from exc
            raise ValueError(f"The wing cannot be lofted to a mesh: {exc}") from exc
    raise ValueError("The design has no meshes to simulate.")


def _on_plane(v, tol: float) -> bool:
    """Whether a point (or a direction) has no y component (within ``tol``)."""
    a = np.asarray(v if v is not None else (0.0, 0.0, 0.0), dtype=np.float64).ravel()
    return len(a) < 2 or abs(float(a[1])) <= tol


def _mirror_pair(a, b, tol: float) -> bool:
    """Whether point ``b`` is the mirror image of point ``a`` across y = 0 (same x and z, opposite y)."""
    pa = np.asarray(a, dtype=np.float64).ravel()
    pb = np.asarray(b, dtype=np.float64).ravel()
    if len(pa) < 3 or len(pb) < 3:
        return False
    return abs(pa[0] - pb[0]) <= tol and abs(pa[2] - pb[2]) <= tol and abs(pa[1] + pb[1]) <= tol


def _normal_symmetric(normal, tol: float) -> bool:
    """Whether a facing direction maps to itself (or its opposite, the same axis) under y -> -y:
    it lies in the plane or points straight across it."""
    if normal is None:
        return True
    n = np.asarray(normal, dtype=np.float64).ravel()
    return not (len(n) >= 3 and abs(n[1]) > tol * max(np.linalg.norm(n), 1e-12) and np.hypot(n[0], n[2]) > 1e-9)


def _mirror_symmetric(kind: str, x, tol: float) -> bool:
    """Whether one placed item is its own mirror image across y = 0 (wind frame = design frame
    at zero yaw: a pitch turns about y and keeps every y), i.e. whether its point set maps to
    itself under y -> -y.

    Either its centre, its line's end and its direction lie in the plane, or (a smoke line or
    rake) its two ends are each other's mirror image: a spanwise rake from y = -0.08 to +0.08
    has nozzles k and count-1-k at mirrored places, so the half domain carries exactly the
    nozzles at y >= 0 and the mirror shows the rest (the smoke_3d example's rake used to switch
    the symmetry plane off, doubling the cells and reading a non-zero CS 1.7e-7, critic round
    3.3). Smoke only: the solver reflects a tracer particle released at y < 0 back into the
    solved half, where it would join the streak of its mirror nozzle and double that streak's
    density, so a particle rake must lie in the plane. A fan must not swirl (the mirror image
    swirls the other way); an emitter's facing direction (disc, plane, nozzle axis) lies in the
    plane or across it. The solver normalises each footprint over the whole shape before the half
    domain clips it, so the solved half gets exactly half the item (ns3d.physics), and fans and
    heaters report their full-body values."""
    if kind == "emitter":
        if not _normal_symmetric(getattr(x, "normal", None), tol):
            return False
        if (x.kind in ("line", "rake") and x.end is not None and getattr(x, "species", "dye") == "dye"
                and _mirror_pair(x.position, x.end, tol)):
            return True
        if not _on_plane(x.position, tol):
            return False
        return not (x.end is not None and not _on_plane(x.end, tol))
    if not _on_plane(x.position, tol):
        return False
    if kind in ("fan", "port"):
        if not _on_plane(x.direction, 1e-9 * max(float(np.linalg.norm(np.asarray(x.direction, float))), 1.0)):
            return False
        return not (kind == "fan" and abs(float(getattr(x, "swirl", 0.0) or 0.0)) > 0.0)
    return True                                   # heaters: boxes and spheres centred on the plane


def _physics_breaks_symmetry(sim: Simulation, length: float = 1.0) -> str:
    """Why the placed physics rules out a half domain ("" when it does not).

    A symmetry plane mirrors everything on the y >= 0 side, so an emitter, fan, jet or heater
    off the plane would get a twin on the other side, and different conditions on the front and
    back walls are not mirror-symmetric. Items that are their own mirror image (centred on
    y = 0, directions in the plane, :func:`_mirror_symmetric`) keep it: the hot sphere's smoke
    rake along the centre line used to switch it off, so its plan (without the rake) and its
    run (with it) chose different grids for one body (critic round 3.2).
    """
    tol = 1e-6 * max(float(length), 1e-9)
    off: list[str] = []
    for what, kind, items in (("smoke emitters", "emitter", sim.emitters), ("fans", "fan", sim.fans),
                              ("jets and vents", "port", sim.ports), ("heat sources", "heater", sim.heat_sources)):
        live = [x for x in items if getattr(x, "enabled", True)]
        if any(not _mirror_symmetric(kind, x, tol) for x in live):
            off.append(what)
    if off:
        return ("the case has placed " + ", ".join(off) + " off the centre plane y = 0 (or swirling, or blowing "
                "across it)")
    if sim.thermal.enabled:
        walls = {w.side: (w.mode, w.value) for w in sim.thermal.walls}
        if walls.get("front", ("adiabatic", 0.0)) != walls.get("back", ("adiabatic", 0.0)):
            return "the front and back walls have different thermal conditions"
    return ""


def _memory_kw(placed: Placed | None) -> dict:
    """``ns3d.estimate_memory_bytes`` keywords of the placed physics (temperature, smoke channels, particles)."""
    if placed is None:
        return {}
    n_part = sum(1 for e in placed.emitters if e.species == "particles")
    return {"thermal": placed.thermal is not None, "dye_channels": max(1, len(placed.dye_channels)),
            "particles": 65536 * n_part}


def _solver_physics_3d(placed: Placed | None, cfd) -> dict:
    """SolverConfig3D keywords of the placed physics and of the diagnostics / PID step. Without
    emitters the v0.2 smoke rake stays (``emitters=None``); with them the emitters replace it."""
    kw = {"diagnostics": bool(getattr(cfd, "record_diagnostics", False)),
          "adaptive_dt": "pid" if getattr(cfd, "adaptive_dt", "cfl") == "pid" else "cfl"}
    if placed is not None:
        kw.update(thermal=placed.thermal, heat_sources=list(placed.heat_sources), fans=list(placed.fans),
                  ports=list(placed.ports), emitters=list(placed.emitters) if placed.emitters else None)
    return kw


#: Cells across the smaller side of a closed 3D box per quality and device class. The cubic cavity
#: (theory_ns3d.md 15.6) reads Nu within 0.04 % of the spectral benchmark at Ra 10^4 on 32^3 and
#: within 1 % at Ra 10^5 on 48^3; the thermal boundary-layer rule raises the count for higher Ra.
CLOSED_CELLS3 = {"gpu": (32, 48, 64), "cpu": (16, 24, 32)}
LIVE_CLOSED_CELLS3 = {"gpu": 48, "cpu": 20}


def _round4(n: float) -> int:
    return max(GRID_MULTIPLE, GRID_MULTIPLE * round(n / GRID_MULTIPLE))


def _plan3_closed(design: Design, sim: Simulation, device: str | None, *, live: bool,
                  airfoil_coords: AirfoilCoords | None = None) -> SimPlan3D:
    """Plan a closed 3D box (a room, a cubic cavity): the 2D rules of :func:`.plan._plan_closed`
    in three dimensions, gravity along −z. Bodies are optional."""
    from ..solvers.ns3d.config import ImmersedMesh, SolverConfig3D
    from ..solvers.ns3d.solver import estimate_memory_bytes
    cond, cfd = sim.conditions, sim.cfd
    L = float(design.length_ref)
    if not (L > 0 and math.isfinite(L)):
        raise ValueError("The design's reference length must be positive.")
    if not (cond.fluid.density > 0 and cond.fluid.viscosity > 0):
        raise ValueError("The fluid needs a positive density and viscosity.")
    notes: list[str] = []
    warnings: list[str] = []
    quality = cfd.quality if cfd.quality in QUALITY_ORDER else "standard"
    qi = QUALITY_ORDER.index(quality)
    dev = _p2._resolve_device(device if device not in (None, "", "auto") else cfd.device, warnings)
    dclass = _p2._device_class(dev)
    precision = _resolve_precision3(cfd.precision, dev, warnings)
    dinfo = devmod.info(dev)
    device_name = _p2._device_name(dev, dinfo)
    ground = bool(sim.ground.enabled)
    moving = ground and bool(sim.ground.moving)
    alpha, yaw = float(cond.alpha_deg), float(getattr(cond, "yaw_deg", 0.0) or 0.0)
    ref = tuple(float(v) for v in design.moment_ref3)
    meshes: list[Mesh3D] = []
    if design.meshes or (design.kind == "wing" and design.wing is not None):
        meshes = _design_meshes(design, airfoil_coords)
        names = _unique_mesh_names(meshes)
        meshes = [Mesh3D(name=n, vertices=m.vertices, faces=m.faces, role=m.role, omega=m.omega, center=m.center,
                         axis=m.axis, thermal=m.thermal, thermal_value=m.thermal_value) for m, n in zip(meshes, names)]
    wind = wind_frame_meshes(meshes, alpha, yaw, ref) if meshes else []
    bb = _bbox(wind) if wind else None
    box, explicit = closed_box(design, sim, 3, bb, alpha, yaw)
    x0, x1, y0, y1, z0, z1 = box
    W, D, H = x1 - x0, y1 - y0, z1 - z0
    if bb is not None and (bb[0] < x0 or bb[1] > x1 or bb[2] < y0 or bb[3] > y1 or bb[4] < z0 or bb[5] > z1):
        warnings.append("The design reaches through the walls of the closed box: the walls cut it. Enlarge the box "
                        "(Domain ▸ Extent) if the whole body matters.")
    scales = physics_scales(design, sim, 3, box=box)
    U = scales.speed

    # -- resolution
    n_box = LIVE_CLOSED_CELLS3[dclass] if live else CLOSED_CELLS3[dclass][qi]
    h_t = min(W, D, H) / n_box
    why = f"{n_box} cells across the box's smaller side ({quality.capitalize()} quality on the {dclass.upper()})"
    n_thin = LIVE_THIN_CELLS3 if live else THIN_CELLS3[qi]
    if scales.thermal and scales.natural and scales.ra >= 1e3 and scales.surfaces > 0 and not live:
        # (a box warmed only by heaters has no heated wall whose boundary layer this would size)
        nu_est = cavity_nusselt(scales.ra, scales.pr)
        h_bl = scales.length_ra / (2.0 * nu_est) / n_thin
        if h_bl < h_t:
            h_t, why = h_bl, (f"{n_thin:.0f} cells across the thermal boundary layer of the heated walls "
                              f"(H / (2 Nu), Nu ≈ {nu_est:.2g} at Ra {units.fmt_reynolds(scales.ra)})")
    if bb is not None:
        size = max(min(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]), 1e-6 * L)
        per = CELLS_PER_SIZE3["generic"][dclass][0 if live else qi]
        if size / per < h_t:
            h_t, why = size / per, f"{per} cells across the body ({units.sig(size, 3)} m)"
    user_cpl = bool(cfd.cells_per_ref)
    if user_cpl:
        h_t, why = L / float(cfd.cells_per_ref), f"{float(cfd.cells_per_ref):g} cells per reference length (set by hand)"
    budget = LIVE_BUDGET3[dclass] if live else CELL_BUDGET3[dclass][qi]
    if dclass == "gpu" and dinfo.memory_gb and dinfo.memory_gb < 7.5:
        budget = int(budget * dinfo.memory_gb / 8.0)
    if not user_cpl and W * D * H / h_t ** 3 > budget:
        h_new = (W * D * H / budget) ** (1.0 / 3.0)
        if not live:
            warnings.append(f"The {dclass.upper()} cell budget ({budget:,}) limits the grid to cells of "
                            f"{units.sig(1000 * h_new, 3)} mm instead of {units.sig(1000 * h_t, 3)} mm (for {why}).")
        h_t = h_new
    nz = _round4(H / h_t)
    h_m = H / nz
    nx, ny = _round4(W / h_m), _round4(D / h_m)
    while not user_cpl and nx * ny * nz > budget and nz > GRID_MULTIPLE:
        nz -= GRID_MULTIPLE
        h_m = H / nz
        nx, ny = _round4(W / h_m), _round4(D / h_m)
    h = h_m / L
    cells = nx * ny * nz
    extent3 = (x0, x0 + nx * h_m, y0, y0 + ny * h_m, z0, z0 + nz * h_m)
    if abs(nx * h_m - W) > 1e-6 * W or abs(ny * h_m - D) > 1e-6 * D:
        notes.append(f"The box is {units.sig(W, 4)} × {units.sig(D, 4)} × {units.sig(H, 4)} m; the uniform grid "
                     f"(cells in multiples of {GRID_MULTIPLE}) fits it as {units.sig(nx * h_m, 4)} × "
                     f"{units.sig(ny * h_m, 4)} × {units.sig(H, 4)} m.")

    # -- flow model and duration (the 2D rules)
    re_sim = U * L / scales.nu
    n_bl = 5.0 / (h * math.sqrt(re_sim))
    resolved = n_bl >= BL_CELLS_MIN
    if cfd.turbulence == "les":
        model = "les"
    elif cfd.turbulence == "laminar" or (scales.natural and scales.ra < 1e9):
        model = "laminar"
    else:
        model = "laminar" if resolved else "les"
    advection = "central" if (model == "laminar" and re_sim * h <= CENTRAL_CELL_RE and not live) else "upwind3"
    turbulence = "laminar" if model == "laminar" else f"les-{LES_MODEL}"
    solver_turb = "none" if model == "laminar" else LES_MODEL
    natural = scales.natural and scales.thermal
    # (time unit of the duration label, its name): the label is built from the duration used (2D rules)
    if natural and scales.ra >= 1e3 and scales.buoyancy:
        a, b, cap = _p2.NATURAL_T
        unit_lu, unit_name = scales.length_ra / L, "free-fall times H/U"
        n_ff = min(a + b * scales.pe * scales.length_ra / L, cap)
        if (qi == 0 and not live and scales.ra >= _p2.NATURAL_DRAFT_RA and n_ff > _p2.NATURAL_DRAFT_FF
                and not cfd.duration):
            notes.append(f"Draft runs this room for {_p2.NATURAL_DRAFT_FF:g} free-fall times instead of the "
                         f"{n_ff:.0f} a steady laminar box would need: at Ra {units.fmt_reynolds(scales.ra)} the flow "
                         "is unsteady, so it never reaches the steady state the longer rule waits for, and the "
                         "large-scale circulation is established within a few tens of free-fall times. Enough to "
                         "see the flow and the temperatures; Standard quality, or a Duration set in Setup, averages "
                         "longer.")
            n_ff = _p2.NATURAL_DRAFT_FF
        t_end = n_ff * unit_lu
    elif natural:
        # the speed scale is κ/H here, so one diffusion time H²/κ is H/L reference times
        unit_lu, unit_name = scales.length_ra / L, "diffusion times H²/κ"
        t_end = _p2.CONDUCTION_T * unit_lu
    else:
        unit_lu, unit_name = max(W, D, H) / L, "box turnovers max(W, D, H)/U"
        t_end = _p2.FORCED_TURNOVERS[qi] * unit_lu
    t_avg = 0.5 * t_end
    if cfd.duration:
        t_end = float(cfd.duration)
        if not cfd.average_from:
            t_avg = 0.5 * t_end
    dur_txt = (f"{units.sig(t_end / unit_lu, 3)} {unit_name}" + (", set in Setup" if cfd.duration else "")
               if unit_lu > 0 else "set in Setup")
    if cfd.average_from is not None and cfd.average_from > 0:
        t_avg = float(cfd.average_from)
    if t_avg >= 0.9 * t_end:
        t_avg = 0.5 * t_end
    if live:
        t_end, t_avg = math.inf, 0.0

    placed = placed_physics_3d(design, sim, scales, alpha_deg=alpha, yaw_deg=yaw, extent3=extent3, h_m=h_m,
                               bodies_wind=wind, closed=True)
    if not scales.thermal and not placed.fans and not placed.ports and not moving:
        raise ValueError("Nothing drives the flow in this closed box: add a fan or a jet, heat a wall or a body with "
                         "heat transfer on, or move the floor.")
    need = estimate_memory_bytes(nx, ny, nz, dtype=precision, turbulence=solver_turb, **_memory_kw(placed))
    have_gb = float(getattr(dinfo, "memory_gb", 0.0) or 0.0) if dclass == "gpu" else devmod.system_memory_gb()
    if have_gb > 0 and need > MEMORY_FRACTION * have_gb * 1024 ** 3:
        raise ValueError(f"This grid of {cells:,} cells needs about {need / 1024 ** 3:.1f} GB, more than "
                         f"{MEMORY_FRACTION:.0%} of the {have_gb:.1f} GB this device has. Choose a lower quality.")
    u_peak, peak_basis = peak_speed(scales, sim, 3)
    ratio = max(1.0, u_peak / U)
    cal_kind = _p2.calibration_kind("generic", placed, closed=True, natural=natural, speed_ratio=ratio,
                                    bodies=bool(design.meshes), cells_across=min(nx, ny, nz))
    kfac = _speed_factor3(cal_kind, model, float(L / h_m)) * ratio
    nu_max = 1.0 / re_sim + (NUT_PEAK * h if model == "les" else 0.0)
    dt = min(CFL * h / kfac, VISCOUS_DT3 * h * h / nu_max)
    dt = min(dt, physics_dt(h, kfac, scales, placed, sim, closed=True, cfl=CFL, dim=3,
                            nut=NUT_PEAK * h if model == "les" else 0.0))
    steps = math.ceil(t_end / dt) if math.isfinite(t_end) else 0
    cost = physics_cost(placed, bool(cfd.record_diagnostics), cfd.adaptive_dt == "pid", 3)
    sps = estimate_steps_per_second3d(dev, cells, advection=advection, turbulence=solver_turb, dtype=precision) / cost
    overhead = (6.0 if dclass == "gpu" else 3.0) + cells * (1.5e-6 if dclass == "gpu" else 1e-5)
    est = steps / (sps * loop_efficiency3d(dev, cells, live_view=True)) + overhead if steps else float("nan")
    est_headless = steps / (sps * loop_efficiency3d(dev, cells, live_view=False)) + overhead if steps else float("nan")
    imm = []
    for m, (th_mode, th_val) in zip(wind, placed.body_thermal):
        b = m.bounds()
        center = m.center or (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
        m.center, m.axis = tuple(center), tuple(m.axis or (0.0, 1.0, 0.0))
        imm.append(ImmersedMesh(vertices=np.asarray(m.vertices) / L, faces=np.asarray(m.faces), name=m.name, role=m.role,
                                omega=float(m.omega or 0.0) * L / U, center=tuple(float(v) / L for v in m.center),
                                axis=m.axis, thermal=th_mode, thermal_value=th_val))
    cfg = SolverConfig3D(
        nx=nx, ny=ny, nz=nz, h=h, x0=extent3[0] / L, y0=extent3[2] / L, z0=extent3[4] / L, re=re_sim, bodies=imm,
        domain="closed", bc_x="noslip", bc_y="noslip", bc_zmin="moving" if moving else "noslip", bc_zmax="noslip",
        ground_speed=float(cond.speed) / U if moving else 1.0, advection=advection, turbulence=solver_turb, cfl=CFL,
        ref_area=max(float(design.ref_area), 1e-12) / L ** 2, length_ref=1.0,
        span_ref=max(float(design.span_ref), 1e-12) / L, moment_ref=tuple(v / L for v in ref), dtype=precision,
        device=dev, use_cuda_graphs=True, perturb=True, dye=bool(placed.dye_channels), **_solver_physics_3d(placed, cfd))
    notes.insert(0, f"Closed box: {units.sig(W, 3)} × {units.sig(D, 3)} × {units.sig(H, 3)} m with no-slip walls on every "
                    "side " + ("(from Domain ▸ Extent)" if explicit else "(around the design and the placed items; set "
                               "Domain ▸ Extent to fix the walls)")
                    + (", the floor moving at the flow speed" if moving else "") + "; gravity along −z. The reference "
                    f"velocity U = {units.sig(U, 3)} m/s is the {scales.speed_basis}; coefficients use q = ½ρU².")
    notes.append(f"Grid: {nx:,} × {ny:,} × {nz:,} = {cells:,} cells of {units.sig(h_m * 1000, 3)} mm, set by {why}.")
    notes.append(f"Flow model: {'WALE large-eddy model' if model == 'les' else 'laminar'} at Re = "
                 f"{units.fmt_reynolds(re_sim)} (U L/ν).")
    if not live:
        notes.append(f"Duration: {t_end:.4g} L/U ({dur_txt}, {fmt_time(t_end * L / U)} of real flow); averages from "
                     f"t = {t_avg:.4g}; the run stops early once the monitored quantities have settled. Time step ≈ "
                     f"{units.sig(dt, 2)} L/U: about {steps:,} steps, estimated {fmt_duration(est)} on {device_name}"
                     + (f" (the placed physics costs about {100 * (cost - 1):.0f} % per step)." if cost > 1.02 else ".")
                     + _p2.peak_note(ratio, u_peak, peak_basis))
        if placed.fans:
            notes.append("A fan in a closed box drives the air until the walls' friction balances its thrust, so the "
                         "room recirculates faster than the fan's own disc velocity (the reference used here). The "
                         "time step is planned on the peak such rooms measured (the sum of the three velocity maxima "
                         "about 3.4 times twice the disc velocity at Draft); later runs of the same kind of room (a "
                         "fan alone, or with jets or heaters, with or without bodies, on a similar grid) refine it, "
                         "by at most 1.5 times per run.")
    notes.extend(placed.notes)
    warnings.extend(w for w in placed.warnings if w not in warnings)
    t_notes, t_warn = thermal_checks(scales, sim, h_m=h_m, n_bl=n_bl, bl_target=BL_CELLS_MIN, closed=True, live=live,
                                     dim=3)
    notes.extend(t_notes)
    warnings.extend(t_warn)
    unbounded = unbounded_heating_note(scales, placed, sim, box, 3)
    if unbounded and not live:
        notes.append(unbounded)
    warnings = merge_warnings(warnings)
    bounds = bb if bb is not None else tuple(extent3)
    slice_pos = {"y": 0.5 * (extent3[2] + extent3[3]), "z": 0.5 * (extent3[4] + extent3[5]),
                 "x": 0.5 * (extent3[0] + extent3[1])}
    return SimPlan3D(
        solver_config=cfg, design=design, sim=sim, length_ref=L, speed=U, fluid=cond.fluid, re_real=re_sim,
        re_sim=re_sim, turbulence=turbulence, t_end=t_end, t_avg_start=t_avg, dt=dt, steps_estimate=steps,
        est_seconds=est, cells=cells, cells_per_ref=float(L / h_m), bodies_wind=[],
        extent=(extent3[0], extent3[1], extent3[4], extent3[5]), notes=notes, warnings=warnings, device=dev,
        device_name=device_name, precision=precision, advection=advection, quality=quality, ground=ground,
        ground_moving=moving, strouhal_length=L, blockage=float("nan"), ref_label="reference length", live=live,
        capped=False, geometry_class=geometry_class3d(design), conv_rtol=CONVERGENCE_RTOL[qi],
        blockage_limit=float("nan"), est_seconds_headless=est_headless, bl_resolved=bool(resolved),
        symmetry_plane=False, extent3=extent3, bodies_wind3=wind, meshes_design=meshes, alpha_deg=alpha, yaw_deg=yaw,
        ref_area=float(design.ref_area) if design.ref_area > 0 else 1.0,
        span_ref=float(design.span_ref) if design.span_ref > 0 else 1.0, moment_ref3=ref,
        blockage_z=float("nan"), blockage_y=float("nan"), slice_positions=slice_pos, body_bounds=bounds,
        domain_mode="closed", physics=scales, placed=placed, monitors=plan_monitors(scales, placed, closed=True),
        calibration_kind=cal_kind, step_cost=cost, speed_ratio=ratio, peak_basis=peak_basis, est_seconds_planned=est,
        thermal_grid=(float(h_m), float(n_bl), float(BL_CELLS_MIN)) if scales.thermal else None)


def _plan3(design: Design, sim: Simulation, device: str | None, *, live: bool,
           airfoil_coords: AirfoilCoords | None = None) -> SimPlan3D:
    """Shared implementation of :func:`plan_simulation3d` and :func:`plan_live3d`."""
    from ..solvers.ns3d.config import ImmersedMesh, SolverConfig3D
    if getattr(sim.cfd, "domain_mode", "tunnel") == "closed":
        return _plan3_closed(design, sim, device, live=live, airfoil_coords=airfoil_coords)
    cond, cfd = sim.conditions, sim.cfd
    U, L = float(cond.speed), float(design.length_ref)
    if not (U > 0 and math.isfinite(U)):
        raise ValueError("The flow speed must be positive.")
    if not (L > 0 and math.isfinite(L)):
        raise ValueError("The design's reference length must be positive.")
    if not (cond.fluid.density > 0 and cond.fluid.viscosity > 0):
        raise ValueError("The fluid needs a positive density and viscosity.")
    notes: list[str] = []
    warnings: list[str] = []
    quality = cfd.quality if cfd.quality in QUALITY_ORDER else "standard"
    qi = QUALITY_ORDER.index(quality)
    geo = geometry_class3d(design)

    dev = _p2._resolve_device(device if device not in (None, "", "auto") else cfd.device, warnings)
    dclass = _p2._device_class(dev)
    precision = _resolve_precision3(cfd.precision, dev, warnings)
    dinfo = devmod.info(dev)
    device_name = _p2._device_name(dev, dinfo)
    if dclass == "cpu" and qi > 0 and not live:
        warnings.append(f"{quality.capitalize()} quality on a CPU: 3D grids that fit a CPU's speed are coarse "
                        f"(budget {CELL_BUDGET3['cpu'][qi]:,} cells), so this is barely finer than Draft and takes "
                        "much longer. A GPU runs 3D Standard and Fine grids.")

    # -- 1. wind frame ------------------------------------------------------------
    meshes = _design_meshes(design, airfoil_coords)
    names = _unique_mesh_names(meshes)
    meshes = [Mesh3D(name=n, vertices=m.vertices, faces=m.faces, role=m.role, omega=m.omega, center=m.center,
                     axis=m.axis, thermal=m.thermal, thermal_value=m.thermal_value) for m, n in zip(meshes, names)]
    ground = bool(sim.ground.enabled)
    moving = ground and bool(sim.ground.moving)
    alpha, yaw = float(cond.alpha_deg), float(getattr(cond, "yaw_deg", 0.0) or 0.0)
    ref = tuple(float(v) for v in design.moment_ref3)
    wind = wind_frame_meshes(meshes, alpha, yaw, ref, ground=ground)
    bb = _bbox(wind)
    if ground:
        low = min(float(np.asarray(m.vertices)[:, 2].min()) for m in wind)
        size = max(bb[1] - bb[0], bb[5] - bb[4], 1e-9)
        if low < -1e-6 * size:
            raise ValueError(f"At α {alpha:g}° (yaw {yaw:g}°) the geometry reaches {units.sig(-low * 1000, 3)} mm below "
                             "the road. Reduce the pitch angle or raise the body (Geometry: place on road).")
    elif design.kind == "vehicle":
        notes.append("The road is switched off: the vehicle is simulated in free air, as if in flight. Turn the "
                     "road on in Setup for ground effect.")

    # -- 2. symmetry plane ------------------------------------------------------------
    sym_setting = str(getattr(cfd, "symmetry", "auto") or "auto")
    symmetric = bool(design.symmetric)
    facts = _geometry_facts(design, meshes)
    err = facts.get("sym_err")
    size_all = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
    if symmetric and err is not None and err > SYMMETRY_TOL3 * size_all:
        # the flag says symmetric, the shape no longer is (moved or edited after import):
        # a half domain would simulate a different body
        warnings.append(f"The design is marked mirror-symmetric, but its mirror image across y = 0 is off by "
                        f"{units.sig(1000 * err, 3)} mm (RMS), so both halves are simulated. Centre it on y = 0 "
                        "(Geometry) to use a symmetry plane and half the cells.")
        symmetric = False
    use_sym = symmetric and abs(yaw) < 1e-9 and sym_setting == "auto" and not live
    asym = _physics_breaks_symmetry(sim, L)
    if use_sym and asym:
        use_sym = False
        notes.append(f"Full domain: {asym}, so the symmetry plane is not used (it would mirror them).")
    elif use_sym and has_placed_physics(sim) and any(getattr(x, "enabled", True) for x in
                                                      (*sim.emitters, *sim.fans, *sim.ports, *sim.heat_sources)):
        notes.append("The placed items lie on the centre plane y = 0 or span it symmetrically, so they are their "
                     "own mirror images and the symmetry plane stays: the solved half gets half of each, and fans, "
                     "jets and heaters are reported for the whole.")
    if use_sym:
        err_txt = (f" (mirror error {units.sig(1000 * float(err), 2)} mm RMS)"
                   if err is not None and err > 1e-6 * size_all else "")
        notes.append("Symmetry plane at y = 0: the design is mirror-symmetric" + err_txt + " and the wind comes "
                     "straight on, so only the half y ≥ 0 is simulated (half the cells) and forces are reported "
                     "for the whole body. Side force, roll and yaw moments are zero by construction, and wake "
                     "motions that break the symmetry (a sideways-swinging wake) are suppressed; mean drag and "
                     "lift are not affected by that in the cases measured (theory_ns3d.md 10).")
    elif not live:
        if symmetric and abs(yaw) >= 1e-9:
            notes.append(f"Full domain: the design is symmetric, but a yaw of {yaw:g}° makes the flow asymmetric, so "
                         "both halves are simulated.")
        elif symmetric and sym_setting == "full":
            notes.append("Full domain (both halves) as set in Setup; the symmetric design could also be run with a "
                         "symmetry plane for half the cells.")
        elif not symmetric:
            notes.append("Full domain: the design is not marked mirror-symmetric about y = 0, so both halves are "
                         "simulated.")

    # live sessions must contain every attitude the sliders can reach
    if live:
        pitch_lim = LIVE_PITCH_GROUND3 if ground else LIVE_PITCH3
        env = [m for a in (-pitch_lim, alpha, pitch_lim) for yw in (-LIVE_YAW3, yaw, LIVE_YAW3)
               for m in wind_frame_meshes(meshes, a, yw, ref, ground=ground)]
        bbd = _bbox(env)
    else:
        pitch_lim = 0.0
        bbd = bb
    xmin, xmax, ymin, ymax, zmin, zmax = bbd

    # -- 3. sizes and domain (metres) --------------------------------------------------
    hz_body = (bb[5] - max(bb[4], 0.0)) if ground else (bb[5] - bb[4])
    wy_body = bb[3] - bb[2]
    fallback_area = max(hz_body * wy_body * 0.8, 1e-12)
    akey = (_design_key(design, meshes), round(alpha, 9), round(yaw, 9), ground)
    with _FACTS_LOCK:
        area_front = _AREA_CACHE.get(akey)
    if area_front is None:
        area_front = _frontal_area(wind, fallback_area)
        with _FACTS_LOCK:
            _AREA_CACHE[akey] = area_front
            while len(_AREA_CACHE) > 64:
                _AREA_CACHE.popitem(last=False)
    if geo == "vehicle":
        size = max(bb[5] if ground else hz_body, 1e-6 * L)          # vehicle height (road to roof)
        size_label = "vehicle height"
    elif geo == "wing":
        size = L                                                     # mean aerodynamic chord
        size_label = "mean chord"
    else:
        size = max(math.sqrt(4.0 * area_front / math.pi), 0.25 * (bb[1] - bb[0]), 1e-6 * L)
        size_label = "frontal size"
    dom = DOMAIN3[geo]
    sx = L if geo in ("vehicle", "wing") else size
    pref = {"up": dom["up"][qi] * sx, "down": dom["down"][qi] * sx}
    mins = {"up": dom["min"]["up"] * sx, "down": dom["min"]["down"] * sx}
    blk_limit = BLOCKAGE_LIVE3 if live else BLOCKAGE_MAX3[geo][qi]
    dir_limit = DIRECTION_MAX3[geo][0 if live else qi]
    c_min = MIN_CLEARANCE3[geo][0 if live else qi] * size
    c_blk = _clearance_for(area_front, wy_body, hz_body, blk_limit, ground=ground)
    # per direction: the body may fill at most dir_limit of the tunnel's width and of its
    # height (a 10 m wing in a 12 m tunnel passes the area test and still feels the walls)
    c_dir_y = 0.5 * wy_body * (1.0 / dir_limit - 1.0)
    c_dir_z = hz_body * (1.0 / dir_limit - 1.0) / (1.0 if ground else 2.0)
    pref["side"] = max(c_blk, c_min, c_dir_y)
    pref["top"] = max(c_blk, c_min, c_dir_z)
    if geo == "vehicle":
        # the cross-section comes from the blockage limit and is never traded for cells (as in 2D)
        mins["side"], mins["top"] = pref["side"], pref["top"]
    else:
        c_loose = max(_clearance_for(area_front, wy_body, hz_body, 1.6 * blk_limit, ground=ground),
                      MIN_CLEARANCE3[geo][0] * size * 0.75)
        mins["side"] = max(c_loose, c_dir_y)
        mins["top"] = max(c_loose, c_dir_z)
    user_dom: dict[str, float] = {}
    if cfd.domain_upstream:
        user_dom["up"] = float(cfd.domain_upstream) * L
    if cfd.domain_downstream:
        user_dom["down"] = float(cfd.domain_downstream) * L
    if cfd.domain_height:
        # total height (free flight) or height above the road
        h_user = float(cfd.domain_height) * L
        user_dom["top"] = max((h_user - zmax) if ground else 0.5 * (h_user - (zmax - zmin)), 0.05 * size)
    if getattr(cfd, "domain_width", None):
        w_user = float(cfd.domain_width) * L
        user_dom["side"] = max(0.5 * (w_user - (ymax - ymin)), 0.05 * size)
    pref.update(user_dom)
    mins.update(user_dom)
    mins = {k: min(mins[k], v) for k, v in pref.items()}
    span_x = xmax - xmin

    def lengths(e: dict) -> tuple[float, float, float]:
        lx = e["up"] + span_x + e["down"]
        if use_sym:
            ly = max(abs(ymin), abs(ymax)) + e["side"]
        else:
            ly = (ymax - ymin) + 2.0 * e["side"]
        lz = (zmax + e["top"]) if ground else (zmax - zmin) + 2.0 * e["top"]
        return lx, ly, lz

    def vol(e: dict) -> float:
        a, b, c = lengths(e)
        return a * b * c

    # -- 4. resolution ---------------------------------------------------------------------
    def preset_cpl(q: int) -> float:
        per = CELLS_PER_SIZE3[geo][dclass][q]
        return per * L / size

    user_cpl = bool(cfd.cells_per_ref)
    cpl_target = float(cfd.cells_per_ref) if user_cpl else preset_cpl(qi)
    if live:
        cpl_target = min(cpl_target, preset_cpl(0))
    thin: list[tuple[str, float]] = []
    if ground and geo == "vehicle":
        bodies_only = [m for m in wind if m.role != "wheel"] or wind
        clr = min(float(np.asarray(m.vertices)[:, 2].min()) for m in bodies_only)
        if clr > 1e-6 * size:
            thin.append(("the ground clearance", clr))
    if geo == "wing":
        t_w = _wing_thickness(design) if design.wing is not None else math.inf
        if not math.isfinite(t_w):
            t_w = min(facts["thickness"].get(m.name, math.inf) for m in meshes)
        thin.append(("the thinnest wing section", t_w))
    else:
        for m in meshes:
            if m.role == "wheel":
                continue
            t = facts["thickness"].get(m.name, 0.0)
            if t > 0 and t < 0.25 * size:
                thin.append((f"'{m.name}' ({units.sig(t * 1000, 3)} mm thick)", t))
    n_thin = LIVE_THIN_CELLS3 if live else THIN_CELLS3[qi]
    # a thin feature may raise the resolution by at most THIN_RAISE_MAX: beyond that the
    # budget would take the cells back from the domain (2 chords of upstream room and
    # a wall a chord above a wing cost more accuracy than a 2-cell-thick section)
    cpl_need = min(max((n_thin * L / t for _, t in thin if t > 0), default=0.0), THIN_RAISE_MAX[geo] * cpl_target)
    cpl = cpl_target if user_cpl else max(cpl_target, cpl_need)
    budget = LIVE_BUDGET3[dclass] if live else CELL_BUDGET3[dclass][qi]
    if dclass == "gpu" and dinfo.memory_gb and dinfo.memory_gb < 7.5:
        budget = int(budget * dinfo.memory_gb / 8.0)          # smaller GPUs: stay well inside memory
    ext = dict(pref)
    shrunk = budget_limited = False

    def cells_at(c: float, e: dict) -> float:
        return vol(e) * (c / L) ** 3

    if not user_cpl and cells_at(cpl, ext) > budget:
        lo, hi = 0.0, 1.0
        if cells_at(cpl, {k: mins[k] for k in ext}) <= budget:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                e = {k: pref[k] - mid * (pref[k] - mins[k]) for k in ext}
                lo, hi = (lo, mid) if cells_at(cpl, e) <= budget else (mid, hi)
            lam = hi
        else:
            lam = 1.0
        ext = {k: pref[k] - lam * (pref[k] - mins[k]) for k in ext}
        shrunk = lam > 1e-3
        if cells_at(cpl, ext) > budget:
            cpl = (budget / vol(ext)) ** (1.0 / 3.0) * L
            budget_limited = True
    cpl = max(round(cpl) if (budget_limited or user_cpl) else math.ceil(cpl - 1e-6), 4)
    lx, ly, lz = lengths(ext)

    def grid_at(c: int) -> tuple[int, int, int]:
        # the full width is rounded per half, so a full-domain run of a symmetric body has
        # exactly the grid of the half-domain run plus its mirror image (comparable runs)
        ny_ = _round_up(ly * c / L) if use_sym else 2 * _round_up(0.5 * ly * c / L)
        return _round_up(lx * c / L), ny_, _round_up(lz * c / L)

    nx, ny, nz = grid_at(cpl)
    if not user_cpl:
        while nx * ny * nz > budget and cpl > 4:
            cpl -= 1
            nx, ny, nz = grid_at(cpl)
    h_m = L / cpl
    h = 1.0 / cpl
    x0_m = xmin - ext["up"]
    if use_sym:
        y0_m = 0.0
    else:
        y0_m = 0.5 * (ymin + ymax) - 0.5 * ny * h_m
    z0_m = 0.0 if ground else 0.5 * (zmin + zmax) - 0.5 * nz * h_m
    extent3 = (x0_m, x0_m + nx * h_m, y0_m, y0_m + ny * h_m, z0_m, z0_m + nz * h_m)
    cells = nx * ny * nz
    width_full = (2.0 * ny * h_m) if use_sym else ny * h_m
    height = nz * h_m
    blockage = area_front / (width_full * height)
    blockage_y = wy_body / width_full
    blockage_z = hz_body / height
    # placed physics (DESIGN.md 10): scales on the free stream, solver specs, checks
    scales: PhysicsScales | None = None
    placed: Placed | None = None
    if has_placed_physics(sim):
        scales = physics_scales(design, sim, 3)
        placed = placed_physics_3d(design, sim, scales, alpha_deg=alpha, yaw_deg=yaw, extent3=extent3, h_m=h_m,
                                   bodies_wind=wind, closed=False)

    # -- 5. Reynolds number and flow model (the 2D rules) -----------------------------------------
    re_real = cond.reynolds(L)
    re_resolvable = (5.0 / (BL_CELLS_MIN * h)) ** 2
    capped = cap_from_grid = False
    if cfd.reynolds_mode == "capped":
        if cfd.reynolds_cap:
            cap = float(cfd.reynolds_cap)
        else:
            cap, cap_from_grid = re_resolvable, True
        re_sim = min(re_real, cap)
        capped = re_sim < 0.999 * re_real
    else:
        re_sim = re_real
    re_check = re_sim * (_p2.LIVE_SPEED_RANGE[1] if live and not capped else 1.0)
    n_bl = 5.0 / (h * math.sqrt(re_check))
    cell_re = re_check * h
    resolved = n_bl >= BL_CELLS_MIN
    if cfd.turbulence == "les":
        model = "les"
    elif cfd.turbulence == "laminar":
        model = "laminar"
    else:
        model = "laminar" if resolved else "les"
    advection = "central" if (model == "laminar" and cell_re <= CENTRAL_CELL_RE and not live) else "upwind3"
    turbulence = "laminar" if model == "laminar" else f"les-{LES_MODEL}"
    solver_turb = "none" if model == "laminar" else LES_MODEL

    # memory: the solver's own estimate (state, temporaries, graph pool, means, mask window)
    from ..solvers.ns3d.solver import estimate_memory_bytes
    win_cells = 1
    for a, b in ((bbd[0], bbd[1]), (bbd[2], bbd[3]), (bbd[4], bbd[5])):
        win_cells *= int((b - a) / h_m + 12)
    need = estimate_memory_bytes(nx, ny, nz, dtype=precision, turbulence=solver_turb,
                                 mask_window_cells=win_cells * max(1, len(wind)), **_memory_kw(placed))
    have_gb = float(getattr(dinfo, "memory_gb", 0.0) or 0.0) if dclass == "gpu" else devmod.system_memory_gb()
    if have_gb > 0 and need > MEMORY_FRACTION * have_gb * 1024 ** 3:
        what = "GPU memory" if dclass == "gpu" else "memory (RAM)"
        fix = (f"Lower the cells per {'mean chord' if geo == 'wing' else 'reference length'} in Setup "
               "(Mesh & time)" if user_cpl else "Choose a lower quality")
        raise ValueError(f"This grid of {cells:,} cells needs about {need / 1024 ** 3:.1f} GB of {what}, more than "
                         f"{MEMORY_FRACTION:.0%} of the {have_gb:.1f} GB this device has. {fix}, or run on a device "
                         "with more memory.")

    # -- 6. duration -----------------------------------------------------------------------
    t_end, t_avg = DURATION3[geo][qi]
    if geo == "generic":
        # frontal sizes / U -> L/U, rounded so the plan reads "40 L/U", not "39.949"
        t_end, t_avg = float(f"{t_end * size / L:.2g}"), float(f"{t_avg * size / L:.2g}")
    if qi == 0 and t_end > DRAFT_T_END_MAX3:
        t_end, t_avg = DRAFT_T_END_MAX3, float(f"{t_avg * DRAFT_T_END_MAX3 / t_end:.2g}")
    if cfd.duration:
        t_end = float(cfd.duration)
        if not cfd.average_from:
            t_avg = min(t_avg, 0.5 * t_end)
    if cfd.average_from is not None and cfd.average_from > 0:
        t_avg = float(cfd.average_from)
    if t_avg >= 0.9 * t_end:
        warnings.append(f"Averaging was set to start at t = {t_avg:g}, too close to the end (t = {t_end:g}); the last "
                        "half of the run is averaged instead.")
        t_avg = 0.5 * t_end
    if live:
        t_end, t_avg = math.inf, 0.0

    # -- time step and run-time estimate ------------------------------------------------------
    u_peak, peak_basis = peak_speed(scales, sim, 3) if scales is not None else (U, "free-stream speed")
    if qi == 0 and not live and not cfd.duration:
        # a Draft run whose plume outruns its free stream: the step follows the plume, so the free
        # stream's duration would cost u_peak/U times the steps (physics.draft_buoyant_duration)
        t_cap, cap_note = draft_buoyant_duration(scales, t_end, u_peak)
        if cap_note:
            t_end = t_cap
            if not (cfd.average_from and 0 < float(cfd.average_from) < 0.9 * t_end):
                t_avg = 0.5 * t_end
            notes.append(cap_note)
    ratio = max(1.0, u_peak / U)
    cal_kind = _p2.calibration_kind(geo, placed, closed=False, natural=False, speed_ratio=ratio,
                                    source=peak_source(peak_basis))
    if ratio <= _p2.PEAK_RATIO_MIN:
        ratio = 1.0                       # barely faster than the stream: the class factor covers it
    kfac = _speed_factor3(cal_kind, model, float(cpl)) * ratio
    nu_max = 1.0 / re_sim + (NUT_PEAK * h if model == "les" else 0.0)
    dt = min(CFL * h / kfac, VISCOUS_DT3 * h * h / nu_max)
    if placed is not None:
        dt = min(dt, physics_dt(h, kfac, scales, placed, sim, closed=False, cfl=CFL, dim=3,
                                nut=NUT_PEAK * h if model == "les" else 0.0))
    steps = math.ceil(t_end / dt) if math.isfinite(t_end) else 0
    cost = physics_cost(placed, bool(cfd.record_diagnostics), cfd.adaptive_dt == "pid", 3)
    sps = estimate_steps_per_second3d(dev, cells, advection=advection, turbulence=solver_turb, dtype=precision) / cost
    nfaces = sum(len(m.faces) for m in wind)
    # build (voxelisation, kernel compile, graph capture) and outputs (surface, slices, volume)
    overhead = ((6.0 if dclass == "gpu" else 3.0) + cells * (1.5e-6 if dclass == "gpu" else 1e-5)
                + nfaces * (2e-6 if dclass == "gpu" else 2e-5))
    est = steps / (sps * loop_efficiency3d(dev, cells, live_view=True)) + overhead if steps else float("nan")
    est_headless = (steps / (sps * loop_efficiency3d(dev, cells, live_view=False)) + overhead
                    if steps else float("nan"))

    # -- solver bodies ---------------------------------------------------------------------------
    rotate_wheels = moving and bool(sim.ground.rotate_wheels)
    omega_note = None
    imm = []
    body_thermal = placed.body_thermal if placed is not None else [("adiabatic", 0.0)] * len(wind)
    for m, (th_mode, th_val) in zip(wind, body_thermal):
        omega_si = float(m.omega or 0.0)
        center = m.center
        axis = m.axis
        if m.role == "wheel":
            c, ax, r = _wheel_frame(m)
            center, axis = c, ax
            if ground:
                omega_si = 0.0
                axy2 = ax[0] ** 2 + ax[1] ** 2
                if rotate_wheels and r > 0 and axy2 > 1e-6:
                    # roll with the belt: the contact point moves at the road's speed along the
                    # wheel's rolling direction; about the +y axle that is omega = -U / r
                    omega_si = -U * ax[1] / (r * axy2)
                    omega_note = (f"Wheels roll with the belt (|ω| = {units.sig(abs(omega_si), 3)} rad/s, "
                                  "ω = −U/r about the +y axle by the right-hand rule).")
        if center is None:
            b = m.bounds()
            center = (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
        if axis is None:
            axis = (0.0, 1.0, 0.0)
        m.center, m.axis, m.omega = tuple(center), tuple(axis), omega_si
        imm.append(ImmersedMesh(vertices=np.asarray(m.vertices) / L, faces=np.asarray(m.faces), name=m.name,
                                role=m.role, omega=omega_si * L / U,
                                center=tuple(float(v) / L for v in center), axis=tuple(axis),
                                thermal=th_mode, thermal_value=th_val))
    cfg = SolverConfig3D(
        nx=nx, ny=ny, nz=nz, h=h, x0=x0_m / L, y0=y0_m / L, z0=z0_m / L, re=re_sim, bodies=imm,
        inlet="uniform", bc_y="symmetry_ymin" if use_sym else "slip",
        bc_zmin=("moving" if moving else "noslip") if ground else "slip", bc_zmax="slip", ground_speed=1.0,
        advection=advection, turbulence=solver_turb, cfl=CFL,
        ref_area=max(float(design.ref_area), 1e-12) / L ** 2, length_ref=1.0,
        span_ref=max(float(design.span_ref), 1e-12) / L, moment_ref=tuple(v / L for v in ref),
        dtype=precision, device=dev, use_cuda_graphs=True, perturb=True,
        window_pitch_deg=pitch_lim if live else 0.0, window_yaw_deg=LIVE_YAW3 if live else 0.0,
        **_solver_physics_3d(placed, cfd))

    # -- 7. explanations --------------------------------------------------------------------------
    ref_label = {"vehicle": "vehicle length", "wing": "mean chord"}.get(geo, "reference length")
    st_len = float(design.force_ref) if geo == "vehicle" and design.force_ref > 0 else (L if geo == "wing" else size)
    if not design.ref_area > 0:
        warnings.append("The design has no reference area; coefficients use 1 m². Set it on the Geometry page "
                        "(Auto = frontal area).")
    up_txt, down_txt = units.sig(ext["up"] / L, 3), units.sig((extent3[1] - xmax) / L, 3)
    half_txt = (f"{units.sig(extent3[3], 3)} m to the side of the symmetry plane (the tunnel is "
                f"{units.sig(width_full, 3)} m wide)" if use_sym else f"{units.sig(width_full, 3)} m wide")
    if ground:
        notes.append(f"Domain: {up_txt} {ref_label}s of road ahead of the body, {down_txt} behind it, {half_txt} and "
                     f"{units.sig(height, 3)} m tall under a frictionless ceiling: {100 * blockage:.1f} % blockage "
                     f"(the body fills {100 * blockage_z:.0f} % of the height and {100 * blockage_y:.0f} % of the "
                     "width). The "
                     + ("belt moves with the free stream, as on a rolling-road wind tunnel." if moving else
                        "floor is fixed, like a wind tunnel without a belt: a boundary layer grows along it."))
    else:
        notes.append(f"Domain: {up_txt} {ref_label}s upstream, {down_txt} downstream, {half_txt} and "
                     f"{units.sig(height, 3)} m tall with frictionless walls: {100 * blockage:.1f} % blockage (the "
                     f"body fills {100 * blockage_z:.0f} % of the height and {100 * blockage_y:.0f} % of the width).")
    if ground and moving and abs(yaw) > 1e-9:
        notes.append("With yaw, the moving belt still runs along the wind (x), not along the car: the road moves "
                     "past the car at the yaw angle. A turntable belt would move with the car; the difference "
                     "affects mostly the underbody and the wheels.")
    if not live and blockage > 1.05 * blk_limit:
        warnings.append(f"Blockage {100 * blockage:.1f} % is above the {100 * blk_limit:g} % this quality aims for; the "
                        "walls speed up the flow around the body, so drag and lift read high. A wider and taller "
                        "domain (Setup) reduces the effect.")
    if shrunk:
        notes.append(f"The domain was made more compact than usual to keep the grid within the {dclass.upper()} "
                     f"budget of {budget:,} cells; this costs a little accuracy (inlet and outlet proximity"
                     + (", blockage" if geo != "vehicle" else "") + ") but keeps the resolution.")
    preset_txt = (f"{cpl} cells per {ref_label} (set by hand)" if user_cpl else
                  f"{cpl} cells per {ref_label} ({quality.capitalize()} quality on the {dclass.upper()})")
    per_size = size / h_m
    notes.append(f"Grid: {nx:,} × {ny:,} × {nz:,} = {cells:,} cells of {units.sig(h_m * 1000, 3)} mm; {preset_txt}, "
                 f"{per_size:.0f} cells per {size_label}. The grid is uniform (the exact pressure solver needs it), "
                 "so the cells far from the body cost as much as the ones next to it.")
    for what, t in thin:
        nc = t / h_m
        if nc < THIN_SLACK * n_thin and not live:
            warnings.append(f"{what[0].upper() + what[1:]} spans only {nc:.1f} grid cells (target {n_thin:.0f}); the "
                            "flow there is under-resolved"
                            + (" and the underbody flow, which sets much of the lift, is the least trustworthy part "
                               "of the result" if what == "the ground clearance" else "")
                            + ". " + ("Increase the cells per reference length." if user_cpl else
                                      "Choose a finer quality or run on a GPU." if dclass == "cpu" else
                                      "Choose a finer quality." if qi < 2 else
                                      "Fine quality is the finest preset; set the cells per reference length by "
                                      "hand for more."))
        elif not live and (cpl > cpl_target + 0.5) and abs(cpl - n_thin * L / t) < 1.0:
            notes.append(f"The resolution was raised so that {what} spans {nc:.0f} cells.")
    if cpl < THIN_SLACK * cpl_target and not live:
        warnings.append(f"The {dclass.upper()} cell budget ({budget:,}) limits the grid to {cpl} cells per {ref_label} "
                        f"instead of {cpl_target:.0f}. Results are coarser than the {quality.capitalize()} preset "
                        "intends.")
    if per_size < 12 and not live:
        warnings.append(f"Only {per_size:.0f} cells per {size_label}: forces need about 16 (a few per cent at 24, "
                        "theory_ns3d.md 12) to be more than indicative.")
    if user_cpl and cells > 1.5 * CELL_BUDGET3[dclass][-1]:
        warnings.append(f"{cells:,} cells is far above what this device handles comfortably "
                        f"({CELL_BUDGET3[dclass][-1]:,}); expect a long run and high memory use.")
    if omega_note:
        notes.append(omega_note)

    re_txt = units.fmt_reynolds
    bl_txt = (f"a laminar boundary layer is about {units.sig(5.0 / math.sqrt(re_check), 2)} {ref_label}s thick, "
              f"i.e. {n_bl:.1f} cells")
    if capped:
        notes.append(f"Reynolds number capped at {re_txt(re_sim)} (real: {re_txt(re_real)}). The flow is simulated as "
                     f"if the fluid were {units.sig(re_real / re_sim, 3)}× more viscous; coefficients come from the "
                     "capped flow and forces are scaled to the real speed and density.")
        if cap_from_grid and not live:
            warnings.append(f"No Reynolds cap is set, so the cap is the highest Reynolds number this grid resolves "
                            f"({re_txt(re_sim)}): it changes with the quality and the device. Set the Reynolds cap "
                            "in Setup (Solver) to keep the flow fixed.")
        elif not resolved and not live and model == "les":
            warnings.append(f"The Reynolds cap ({re_txt(re_sim)}) is above what this grid resolves "
                            f"({re_txt(re_resolvable)} at {cpl} cells per {ref_label}): {bl_txt}, so a large-eddy "
                            "model represents the unresolved scales.")
    if model == "les":
        if not resolved:
            notes.append(f"Flow model: at Re = {re_txt(re_check)} {bl_txt} -- too thin for this grid -- so a WALE "
                         "large-eddy model represents the unresolved scales and third-order upwind advection keeps "
                         "the run stable.")
        else:
            notes.append(f"Flow model: WALE large-eddy model at Re = {re_txt(re_sim)}, as requested in the settings "
                         "(the grid would resolve this flow without one).")
    else:
        why = "central differences (cell Reynolds number ≤ 2)" if advection == "central" else \
            "third-order upwind differences"
        notes.append(f"Flow model: laminar at Re = {re_txt(re_sim)}; {bl_txt}, so no turbulence model is needed. "
                     f"Advection uses {why}.")
        if not resolved:
            warnings.append(f"Laminar flow was requested, but {bl_txt}: the boundary layers are not resolved and the "
                            "run may be noisy or unstable. 'Auto' would use a large-eddy model.")
    if re_sim > 1e4 and not resolved:
        if geo == "wing":
            warnings.append(f"Lift and drag are not reliable at Re = {re_txt(re_sim)}: the boundary layers are much "
                            "thinner than a grid cell and there is no wall or transition model, so a wing's lift "
                            "comes out low and its drag high. " + WING_CFD_NOTE)
        else:
            warnings.append(f"At Re = {re_txt(re_sim)} the boundary layers are much thinner than a grid cell and there "
                            "is no wall or transition model: separation from smooth surfaces and friction drag are "
                            "approximate, so absolute Cd and Cl are indicative; differences between designs run "
                            "the same way are the robust result.")
    elif geo == "generic" and 210 < re_sim <= 1e4 and not live:
        notes.append(f"At Re = {re_txt(re_sim)} the wake of a bluff body is unsteady and three-dimensional; the "
                     "averaged forces describe that unsteady flow.")
    if geo == "vehicle" and not live:
        notes.append(VEHICLE_3D_NOTE)
    if geo == "wing" and not live and not (re_sim > 1e4 and not resolved):
        notes.append(WING_CFD_NOTE)
    if geo == "generic" and not live and re_sim <= 300:
        notes.append(f"Confinement: at low Reynolds numbers the walls raise the drag noticeably even at small "
                     f"blockage (measured: a sphere at Re 100 in a 2.85 % box reads about 3-4 % above an unconfined "
                     f"one, theory_ns3d.md 10); this domain has {100 * blockage:.1f} %.")

    conv_rtol = CONVERGENCE_RTOL[qi]
    if not live:
        tu = L / U
        notes.append(f"Duration: {t_end:g} convective times L/U ({fmt_time(t_end * tu)} of real flow); forces are "
                     f"averaged from t = {t_avg:g} (or later, where the start-up transient is found to end), and the "
                     f"run stops early once the mean drag is known to ±{100 * conv_rtol:g} % and the mean lift to "
                     f"±{100 * CL_RTOL_FACTOR * conv_rtol:g} % (one standard error) and neither drifts.")
        calibrated = bool(_p2._RATE_CACHE3D)
        notes.append(f"Time step ≈ {units.sig(dt, 2)} L/U from the stability limit (CFL {CFL:g}): about {steps:,} "
                     f"steps, estimated {fmt_duration(est)} on {device_name}"
                     + ("." if calibrated else " (reference-machine estimate; refined after each run).")
                     + _p2.peak_note(ratio, u_peak, peak_basis))
        if est > 3600:
            warnings.append(f"Estimated run time {fmt_duration(est)}. Draft quality"
                            + (" or a GPU" if dclass == "cpu" else "") + " would be much faster.")
    # gaps between parts that this grid closes or barely resolves (the other geometry
    # diagnostics are on the Geometry page: check_design3d)
    if not live:
        for na, nb, gap in facts.get("gaps", []):
            nc = gap / h_m
            if nc < CLOSED_GAP_CELLS3:
                warnings.append(f"The {units.sig(gap * 1000, 3)} mm gap between '{na}' and '{nb}' is only {nc:.1f} "
                                "cells wide: the solver treats it as closed (no flow through it). Use a finer grid "
                                "or a wider gap if that flow matters.")
            elif nc < 6:
                notes.append(f"The gap between '{na}' and '{nb}' spans {nc:.1f} cells: flow passes but its losses "
                             "are under-resolved.")

    # default slices: the symmetry plane (or the body's centre plane), mid-height, just behind the body
    monitors: tuple[str, ...] = ()
    if placed is not None:
        notes.extend(placed.notes)
        warnings.extend(w for w in placed.warnings if w not in warnings)
        t_notes, t_warn = thermal_checks(scales, sim, h_m=h_m, n_bl=n_bl, bl_target=BL_CELLS_MIN, closed=False,
                                         live=live, dim=3)
        notes.extend(t_notes)
        warnings.extend(t_warn)
        monitors = plan_monitors(scales, placed, closed=False)
        if cost > 1.02 and not live:
            notes.append(f"The placed physics costs about {100 * (cost - 1):.0f} % more time per step (measured per "
                         "feature, theory_ns3d.md 18); the estimate includes it.")
    slice_pos = {
        "y": 0.0 if use_sym else 0.5 * (bb[2] + bb[3]),
        "z": 0.5 * (max(bb[4], 0.0) + bb[5]) if ground else 0.5 * (bb[4] + bb[5]),
        "x": min(bb[1] + max(2.0 * h_m, 0.1 * (bb[1] - bb[0])), extent3[1] - h_m),
    }
    plan = SimPlan3D(
        solver_config=cfg, design=design, sim=sim, length_ref=L, speed=U, fluid=cond.fluid,
        re_real=re_real, re_sim=re_sim, turbulence=turbulence, t_end=t_end, t_avg_start=t_avg, dt=dt,
        steps_estimate=steps, est_seconds=est, cells=cells, cells_per_ref=float(cpl), bodies_wind=[],
        extent=(extent3[0], extent3[1], extent3[4], extent3[5]), notes=notes, warnings=warnings,
        device=dev, device_name=device_name, precision=precision, advection=advection, quality=quality,
        ground=ground, ground_moving=moving, strouhal_length=float(st_len), blockage=blockage, ref_label=ref_label,
        live=live, capped=capped, geometry_class=geo, conv_rtol=conv_rtol, blockage_limit=blk_limit,
        est_seconds_headless=est_headless, bl_resolved=bool(resolved), symmetry_plane=use_sym, extent3=extent3,
        bodies_wind3=wind, meshes_design=meshes, alpha_deg=alpha, yaw_deg=yaw,
        ref_area=float(design.ref_area) if design.ref_area > 0 else 1.0,
        span_ref=float(design.span_ref) if design.span_ref > 0 else 1.0, moment_ref3=ref,
        blockage_z=blockage_z, blockage_y=blockage_y, slice_positions=slice_pos, body_bounds=bb,
        domain_mode="tunnel", physics=scales, placed=placed, monitors=monitors, calibration_kind=cal_kind,
        step_cost=cost, speed_ratio=ratio, peak_basis=peak_basis,
        thermal_grid=(float(h_m), float(n_bl), float(BL_CELLS_MIN)) if scales is not None and scales.thermal else None)
    plan.warnings = merge_warnings(plan.warnings)
    if not live and cal_kind.endswith("-plume"):
        pn = _p2.plume_note(extent3[5] - bb[5], u_peak, L / U)
        if pn:
            plan.notes.append(pn)
    if not live and getattr(cfd, "compute_sensitivity", False):
        _add_sensitivity_estimate3(plan)
    plan.est_seconds_planned = plan.est_seconds
    return plan


def sensitivity_max_cells(device: str) -> int:
    """The largest grid the 3D sensitivity map runs on, for this device class."""
    try:
        from ..optimize.sensitivity3d import DEFAULT_MAX_CELLS
        return int(DEFAULT_MAX_CELLS[_p2._device_class(device)])
    except Exception:  # noqa: BLE001 - module missing or changed: the DESIGN.md figure
        return SENSITIVITY_MAX_CELLS


def _add_sensitivity_estimate3(plan: SimPlan3D) -> None:
    """Fold the 3D sensitivity map into the estimate, or say why it will be skipped.

    The map needs a reverse sweep through the solver with checkpoints, which
    DESIGN.md 7.1 limits to coarse grids (``optimize.sensitivity3d.DEFAULT_MAX_CELLS``:
    2.1 M cells on an 8 GB GPU, 0.4 M on a CPU); a larger grid is skipped by the
    runner with a warning, and the plan says so in advance.
    """
    limit = sensitivity_max_cells(plan.device)
    if plan.cells > limit:
        plan.warnings.append(f"The sensitivity map is limited to grids of {limit / 1e6:.2g} M cells (this one has "
                             f"{plan.cells / 1e6:.2f} M), so the run will skip it. Draft quality, a symmetry plane "
                             "or fewer cells per reference length make it fit.")
        return
    try:
        from ..optimize.sensitivity3d import estimate_seconds as sens_seconds
        extra = float(sens_seconds(plan))
    except Exception as exc:  # noqa: BLE001 - advisory: the run itself does not depend on it
        plan.notes.append(f"(Sensitivity-map time not included in the estimate: {type(exc).__name__}.)")
        return
    if not math.isfinite(extra) or extra <= 0:
        return
    plan.est_seconds += extra
    plan.est_after_seconds += extra
    if math.isfinite(plan.est_seconds_headless):
        plan.est_seconds_headless += extra
    plan.notes.append(f"Sensitivity map: about {fmt_duration(extra)} more after the flow solve (included in the "
                      "estimate).")


def plan_simulation3d(design: Design, sim: Simulation, *, device: str | None = None,
                      airfoil_coords: AirfoilCoords | None = None) -> SimPlan3D:
    """Plan a production 3D CFD run of ``sim`` on ``design`` (DESIGN.md 7.7).

    ``device`` overrides ``sim.cfd.device``; ``airfoil_coords(name)`` resolves
    "design:<id>" airfoils when a wing planform has to be lofted. Raises
    ``ValueError`` with a user-readable message for impossible set-ups (no
    meshes, zero speed, a pitch into the road, a grid that does not fit).
    """
    return _plan3(design, sim, device, live=False, airfoil_coords=airfoil_coords)


def plan_live3d(design: Design, sim: Simulation, *, device: str | None = None,
                airfoil_coords: AirfoilCoords | None = None) -> SimPlan3D:
    """Plan for a 3D explore session: a small full-domain grid (yaw changes need
    both halves), sized for every attitude the sliders reach (±10° pitch, ±3° on
    a road, ±15° yaw), at up to 8 % blockage, float32, upwind advection."""
    return _plan3(design, sim, device, live=True, airfoil_coords=airfoil_coords)

