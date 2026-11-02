"""Functional, differentiable core of the 2D incompressible Navier-Stokes solver.

Equations (non-dimensional, Re = U L / nu)::

    du/dt + div(u u) = -grad p + (1/Re) lap u + div(2 nu_t S) - (chi/eta)(u - u_s)
    div u = 0

discretised on a uniform staggered (MAC) grid:

* ``u[j, i]`` on x-faces at (x0 + i h, y0 + (j + 1/2) h), shape (ny, nx + 1)
* ``v[j, i]`` on y-faces at (x0 + (i + 1/2) h, y0 + j h), shape (ny + 1, nx)
* ``p[j, i]`` at cell centres, shape (ny, nx)

Time integration is the three-stage strong-stability-preserving Runge-Kutta
scheme of Shu & Osher with a projection after every stage, so every stage
value is discretely divergence-free. Each stage is::

    u_hat = lerp(u_prev, u_0, a) + b dt L(u_prev)       (explicit advection + diffusion)
    u_hat <- (u_hat + k chi u_s) / (1 + k chi),  k = b dt / eta   (implicit penalisation)
    u     = u_hat - grad psi,  lap psi = div u_hat      (exact projection)

with (a, b) = (0, 1), (3/4, 1/4), (1/3, 2/3). The momentum removed by the
penalisation, combined over the stages with the weights (1/6, 2/3, 1) that the
Shu-Osher form assigns to each stage's increment, is the force on the bodies.

Every function here is pure torch with static (Python) control flow only, so a
step can be captured in a CUDA graph, differentiated with autograd and run on
CPU, CUDA or MPS. ``docs/theory_ns2d.md`` derives and justifies each choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from . import physics as phx
from .config import SolverConfig
from .diagnostics import diagnostics_row, pid_dt
from .geometry import Geometry, build_geometry
from .physics import Physics
from .poisson import PoissonSolver

# Shu-Osher SSP-RK3: stage value = lerp(u_prev, u0, a) + b dt L(u_prev)
SSP_RK3 = ((0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0))
# weight with which each stage's penalisation increment reaches u^(n+1)
FORCE_WEIGHTS = (1.0 / 6.0, 2.0 / 3.0, 1.0)
# default subgrid-model constants (Lilly; Nicoud & Ducros; Vreman)
LES_CONSTANTS = {"smagorinsky": 0.17, "wale": 0.5, "vreman": 0.07}
# automatic Brinkman permeability (theory doc, "choosing eta"):
#   eta = min(ETA_PER_H h, h^2 / (nu * STIFFNESS_MIN))
# keeps the inertial penetration (U eta) below 5 % of a cell and the stiffness
# S = h^2 / (nu eta) - penalisation rate over viscous supply rate across a cell -
# above STIFFNESS_MIN, where the effective wall position no longer depends on Re
ETA_PER_H = 0.05
STIFFNESS_MIN = 60.0
# the smoothed mask is centred MASK_SHIFT * h inside the true surface so the
# *effective* no-slip wall of the stiff penalisation sits on it (theory doc)
MASK_SHIFT = 0.32


# ---------------------------------------------------------------------------
# operators and state
# ---------------------------------------------------------------------------

@dataclass
class Ops:
    """Everything about the grid and the physics that a step needs but that is
    not part of the evolving state. Python scalars are static (baked into a
    captured CUDA graph); tensors (``nu``) may be changed in place while a
    graph is live.
    """
    cfg: SolverConfig
    nx: int
    ny: int
    h: float
    x0: float
    y0: float
    device: torch.device
    dtype: torch.dtype
    periodic: bool
    advection: str
    turbulence: str
    bc_top: str
    bc_bottom: str
    ground_speed: float
    cfl: float
    dt_fixed: float | None
    force_ref: float
    moment_ref: tuple[float, float]
    mask_width: float
    mask_shift: float
    eta_fixed: float | None
    les_c: float
    nu: torch.Tensor                       # 0-d, 1/Re (live-editable)
    eta: torch.Tensor                      # 0-d Brinkman permeability (follows nu unless fixed)
    u_inlet: torch.Tensor                  # (ny, 1) inlet u profile
    dye_inlet: torch.Tensor | None         # (ny, 1) dye comb at the inlet
    poisson: PoissonSolver
    forcing: tuple[float, float] | None = None   # uniform body force (verification only)
    extras: dict = field(default_factory=dict)
    closed: bool = False                   # walls on all four sides (no inlet / outlet)
    bc_left: str = "noslip"                # closed domains only
    bc_right: str = "noslip"
    adaptive_dt: str = "cfl"
    phys: Physics | None = None            # placed physics (DESIGN.md 10), None = off
    wall_model: bool = False               # Werner-Wengle shear on the no-slip domain walls (opt-in)


class State(NamedTuple):
    """Evolving solver state. ``p`` is the step-averaged pressure of the step
    that produced this state (for output only: it does not feed back into the
    dynamics); ``dt`` is that step (0 for an initial state).

    Placed physics adds (None when off, so six-field construction still works):
    ``theta`` the temperature (ny, nx), ``dye_extra`` smoke channels 1.. (n-1,
    ny, nx) next to channel 0 in ``dye``, ``ctrl`` the controller memory (PID
    step, fan PI integrators and pressure jumps)."""
    u: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    dye: torch.Tensor | None
    t: torch.Tensor
    dt: torch.Tensor
    theta: torch.Tensor | None = None
    dye_extra: torch.Tensor | None = None
    ctrl: torch.Tensor | None = None


def resolve_device_dtype(cfg: SolverConfig) -> tuple[torch.device, torch.dtype]:
    """Concrete torch device and dtype for a config (``device="auto"`` aware)."""
    try:
        from aerolume.core.device import resolve
    except ImportError:                     # solver used stand-alone, without the app core
        dev = cfg.device if cfg.device not in ("auto", None, "") else (
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = resolve(cfg.device)           # raises if the requested device is unavailable
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    return torch.device(dev), dtype


def inlet_profile(cfg: SolverConfig) -> np.ndarray:
    """Inlet u at the u-face rows: 1, or a parabola with mean 1 (peak 1.5)."""
    if cfg.inlet == "uniform":
        return np.ones(cfg.ny)
    eta = (np.arange(cfg.ny) + 0.5) / cfg.ny
    prof = 6.0 * eta * (1.0 - eta)
    return prof / prof.mean()          # discrete mean exactly 1: Re is based on the bulk velocity


def dye_comb(ny: int) -> np.ndarray:
    """Inlet smoke: thin streaks every few cells (about 40 over the height)."""
    spacing = max(6, ny // 40)
    width = max(1, spacing // 4)
    j = np.arange(ny)
    return (((j + spacing // 2) % spacing) < width).astype(np.float64)


def auto_eta(h: float, re: float) -> float:
    """Default Brinkman permeability for cell size h at Reynolds number re."""
    return min(ETA_PER_H * h, h * h * re / STIFFNESS_MIN)


def set_reynolds(ops: Ops, re: float) -> None:
    """Change Re in place (nu and, unless fixed, eta; with heat transfer also the
    diffusivity 1/(Re Pr) and the thermal eta): safe while a CUDA graph is live."""
    ops.nu.fill_(1.0 / re)
    ops.eta.fill_(ops.eta_fixed if ops.eta_fixed is not None else auto_eta(ops.h, re))
    ph = ops.phys
    if ph is not None and ph.thermal:
        pe = re * ph.prandtl
        ph.kappa.fill_(1.0 / pe)
        ph.eta_t.fill_(ph.eta_t_fixed if ph.eta_t_fixed is not None else phx.thermal_eta(ops.h, pe))


def make_operators(cfg: SolverConfig) -> Ops:
    """Build the static operators (grid, Poisson solver, BC data, placed physics) for a config."""
    dev, dtype = resolve_device_dtype(cfg)
    nu = torch.tensor(1.0 / cfg.re, device=dev, dtype=dtype)
    eta = torch.tensor(cfg.eta if cfg.eta is not None else auto_eta(cfg.h, cfg.re), device=dev, dtype=dtype)
    u_in = torch.as_tensor(inlet_profile(cfg), device=dev, dtype=dtype).reshape(cfg.ny, 1)
    comb = cfg.dye and cfg.dye_inlet and not cfg.closed
    dye_in = torch.as_tensor(dye_comb(cfg.ny) if comb else np.zeros(cfg.ny), device=dev,
                             dtype=dtype).reshape(cfg.ny, 1) if cfg.dye else None
    poisson = PoissonSolver(cfg.nx, cfg.ny, cfg.h, periodic=cfg.periodic, device=dev, dtype=dtype)
    les_c = cfg.les_constant if cfg.les_constant is not None else LES_CONSTANTS.get(cfg.turbulence, 0.0)
    extras = {}
    if cfg.dt is not None:
        extras["dt_fixed"] = torch.tensor(float(cfg.dt), device=dev, dtype=dtype)
    ops = Ops(cfg=cfg, nx=cfg.nx, ny=cfg.ny, h=cfg.h, x0=cfg.x0, y0=cfg.y0, device=dev, dtype=dtype,
              periodic=cfg.periodic, advection=cfg.advection, turbulence=cfg.turbulence,
              bc_top=cfg.bc_top, bc_bottom=cfg.bc_bottom, ground_speed=float(cfg.ground_speed),
              cfl=cfg.cfl, dt_fixed=cfg.dt, force_ref=cfg.force_ref, moment_ref=cfg.moment_ref,
              mask_width=cfg.mask_width, mask_shift=MASK_SHIFT, eta_fixed=cfg.eta, les_c=les_c, nu=nu, eta=eta,
              u_inlet=u_in, dye_inlet=dye_in, poisson=poisson, extras=extras,
              closed=cfg.closed, bc_left=cfg.bc_left, bc_right=cfg.bc_right, adaptive_dt=cfg.adaptive_dt,
              wall_model=cfg.uses_wall_model)
    ops.phys = phx.build_physics(cfg, ops)
    return ops


def face_coordinates(ops: Ops) -> dict[str, torch.Tensor]:
    """1D coordinate vectors: xc/yc (centres), xu (u-face x), yv (v-face y)."""
    d, t, h = ops.device, ops.dtype, ops.h
    return {
        "xc": ops.x0 + (torch.arange(ops.nx, device=d, dtype=t) + 0.5) * h,
        "yc": ops.y0 + (torch.arange(ops.ny, device=d, dtype=t) + 0.5) * h,
        "xu": ops.x0 + torch.arange(ops.nx + 1, device=d, dtype=t) * h,
        "yv": ops.y0 + torch.arange(ops.ny + 1, device=d, dtype=t) * h,
    }


def init_state(cfg: SolverConfig, ops: Ops, *, perturb: bool | None = None) -> State:
    """Impulsive start: the inlet profile everywhere (v = 0), optionally with a
    weak, deterministic, asymmetric perturbation, projected to be discretely
    divergence-free.

    The perturbation is a Lamb-Oseen vortex (peak swirl 5 % of U) centred
    on the bodies. It gives the bodies a small circulation, so the wake is
    asymmetric from the first step and periodic shedding locks in after a few
    convective times instead of waiting for round-off to grow (which takes
    100+ time units and depends on the hardware). It is a smooth function of
    nothing but the body bounding box, so runs are reproducible.
    """
    perturb = cfg.perturb if perturb is None else perturb
    dev, dt = ops.device, ops.dtype
    if cfg.periodic or ops.closed:
        u = torch.zeros(ops.ny, ops.nx + 1, device=dev, dtype=dt)    # no free stream
    else:
        u = ops.u_inlet.expand(ops.ny, ops.nx + 1).clone()
    v = torch.zeros(ops.ny + 1, ops.nx, device=dev, dtype=dt)
    if perturb and cfg.bodies and not cfg.periodic and not ops.closed:
        pts = np.vstack([b.points for b in cfg.bodies])
        cx, cy = 0.5 * (pts[:, 0].min() + pts[:, 0].max()), 0.5 * (pts[:, 1].min() + pts[:, 1].max())
        size = max(pts[:, 1].max() - pts[:, 1].min(), 0.25 * (pts[:, 0].max() - pts[:, 0].min()), 4 * ops.h)
        rc = 0.75 * size
        gam = 0.05 * 2.0 * math.pi * rc / 0.638          # peak u_theta of Lamb-Oseen = 0.638 G/(2 pi rc)
        c = face_coordinates(ops)

        def swirl(x, y):
            dx, dy = x - cx, y - cy
            r2 = (dx * dx + dy * dy).clamp_min(1e-12)
            f = gam / (2 * math.pi * r2) * (1 - torch.exp(-r2 / rc**2))
            return -f * dy, f * dx

        yu, xu = torch.meshgrid(c["yc"], c["xu"], indexing="ij")
        yv, xv = torch.meshgrid(c["yv"], c["xc"], indexing="ij")
        du, _ = swirl(xu, yu)
        _, dv = swirl(xv, yv)
        u = u + du
        v = v + dv
        u, v = _apply_boundary_values(u, v, ops)
        u, v, _ = project(u, v, ops)
    p = torch.zeros(ops.ny, ops.nx, device=dev, dtype=dt)
    ph = ops.phys
    n_dye = cfg.dye_channels
    dye = torch.zeros(ops.ny, ops.nx, device=dev, dtype=dt) if n_dye else None
    zero = torch.zeros((), device=dev, dtype=dt)
    theta = dye_extra = ctrl = None
    if ph is not None:
        if ph.thermal:
            theta = torch.full((ops.ny, ops.nx), ph.theta_init if ops.closed or cfg.periodic else
                               float(cfg.thermal.theta_inlet), device=dev, dtype=dt)
        if n_dye > 1:
            dye_extra = torch.zeros(n_dye - 1, ops.ny, ops.nx, device=dev, dtype=dt)
        if ph.ctrl_size:
            ctrl = torch.zeros(ph.ctrl_size, device=dev, dtype=dt)
    return State(u, v, p, dye, zero.clone(), zero.clone(), theta, dye_extra, ctrl)


# ---------------------------------------------------------------------------
# boundary conditions: ghost-padded copies of the velocity
# ---------------------------------------------------------------------------

def _wall_ghost(rows: torch.Tensor, kind: str, speed: float) -> torch.Tensor:
    """Ghost rows of u (cell-centred in y) mirrored across a wall."""
    if kind == "slip":
        return rows                                   # du/dy = 0
    if kind == "noslip":
        return -rows                                  # u = 0 on the wall
    return 2.0 * speed - rows                         # u = speed on the wall (moving road)


def pad_u(u: torch.Tensor, ops: Ops) -> torch.Tensor:
    """u with two ghost layers on every side: (ny + 4, nx + 5).

    Padded index [j + 2, i + 2] holds u[j, i] for j = -2..ny+1, i = -2..nx+2.
    Inlet ghosts repeat the inlet value, outlet ghosts extrapolate at zero
    gradient, wall ghosts mirror according to the wall type.
    """
    if ops.periodic:
        up = u[:, :-1]
        up = torch.cat([up[:, -2:], up, up[:, :3]], dim=1)
        return torch.cat([up[-2:], up, up[:2]], dim=0)
    if ops.closed:
        # side walls: u is the wall-normal velocity there (zero), odd reflection
        ux = torch.cat([-u[:, 1:3].flip(1), u, -u[:, -3:-1].flip(1)], dim=1)
    else:
        ux = torch.cat([u[:, :1].expand(-1, 2), u, u[:, -1:].expand(-1, 2)], dim=1)
    bot = _wall_ghost(ux[:2].flip(0), ops.bc_bottom, ops.ground_speed)
    top = _wall_ghost(ux[-2:].flip(0), ops.bc_top, 0.0)
    return torch.cat([bot, ux, top], dim=0)


def pad_v(v: torch.Tensor, ops: Ops) -> torch.Tensor:
    """v with two ghost layers on every side: (ny + 5, nx + 4).

    Padded index [j + 2, i + 2] holds v[j, i] for j = -2..ny+2, i = -2..nx+1.
    v vanishes on walls and at the inlet (odd reflection) and has zero
    gradient at the outlet.
    """
    if ops.periodic:
        vp = v[:-1]
        vp = torch.cat([vp[-2:], vp, vp[:3]], dim=0)
        return torch.cat([vp[:, -2:], vp, vp[:, :2]], dim=1)
    vy = torch.cat([-v[1:3].flip(0), v, -v[-3:-1].flip(0)], dim=0)
    if ops.closed:
        # side walls: v is tangential; no-slip = odd reflection, slip = even reflection
        sl = -1.0 if ops.bc_left == "noslip" else 1.0
        sr = -1.0 if ops.bc_right == "noslip" else 1.0
        return torch.cat([sl * vy[:, :2].flip(1), vy, sr * vy[:, -2:].flip(1)], dim=1)
    return torch.cat([-vy[:, :2].flip(1), vy, vy[:, -1:].expand(-1, 2)], dim=1)


def _apply_boundary_values(u: torch.Tensor, v: torch.Tensor, ops: Ops) -> tuple[torch.Tensor, torch.Tensor]:
    """Impose the Dirichlet boundary values (inlet u, wall v) and outlet mass balance."""
    if ops.periodic:
        return torch.cat([u[:, :-1], u[:, :1]], dim=1), torch.cat([v[:-1], v[:1]], dim=0)
    u = _inlet_outlet(u, ops)
    zero = torch.zeros_like(v[:1])
    return u, torch.cat([zero, v[1:-1], zero], dim=0)


def _inlet_outlet(u: torch.Tensor, ops: Ops) -> torch.Tensor:
    """Impose the inlet profile and balance the outlet mass flux.

    The inlet column is taken from ``ops.u_inlet`` at every stage (not carried
    along in the state), so the solution is differentiable with respect to
    the inflow. The outlet column is shifted uniformly so that outflow equals
    inflow plus the net volume flow of the jets and vents (ports): walls are
    impermeable, so this makes the net boundary flux minus the sources - and
    hence the sum of the Poisson right-hand side - exactly zero, which the
    all-Neumann pressure problem requires. A device reduction, no host sync.
    In a closed domain both side columns are walls (u = 0).
    """
    if ops.closed:
        zc = torch.zeros_like(u[:, :1])
        return torch.cat([zc, u[:, 1:-1], zc], dim=1)
    ph = ops.phys
    if ph is not None and ph.ports:
        corr = (ops.u_inlet.sum() + ph.q_net * (1.0 / ops.h) - u[:, -1].sum()) / ops.ny
    else:
        corr = (ops.u_inlet.sum() - u[:, -1].sum()) / ops.ny
    return torch.cat([ops.u_inlet, u[:, 1:-1], u[:, -1:] + corr], dim=1)


# ---------------------------------------------------------------------------
# right-hand side: advection, diffusion, subgrid stress
# ---------------------------------------------------------------------------

def _face_flux(vsum: torch.Tensor | None, a, b, c, d, upwind: bool) -> torch.Tensor:
    """Scaled flux of a quantity through the face between b and c (neighbours a, b | c, d).

    ``vsum`` is *twice* the advecting velocity at the face (the sum of the two
    values it is interpolated from); ``None`` means self-advection, where it is
    ``b + c``. The result is the true flux times 4 (central) or 24/7 (upwind3);
    :func:`_advection` folds that constant into its final difference, which
    saves a full-field pass per flux.

    Central: 2nd-order average (with the divergence form and discretely
    divergence-free velocities this conserves kinetic energy). Upwind3: the
    3rd-order upwind-biased reconstruction (-a + 5b + 2c)/6 for flow from the
    left (mirror for flow from the right), written branch-free as a 4th-order
    central part plus |velocity| times a 4th-difference dissipation::

        phi_face = (7(b + c) - (a + d))/12 + sign(V) (3(b - c) - (a - d))/12
        24/7 * flux = vsum * [(b + c) - (a + d)/7] + 3/7 |vsum| [(b - c) - (a - d)/3]
    """
    s = b + c
    vsum = s if vsum is None else vsum
    if not upwind:
        return vsum * s
    cen = torch.add(s, a + d, alpha=-1.0 / 7.0)
    dis = torch.add(b - c, a - d, alpha=-1.0 / 3.0)
    return torch.addcmul(vsum * cen, vsum.abs(), dis, value=3.0 / 7.0)


def _advection(U: torch.Tensor, V: torch.Tensor, ops: Ops) -> tuple[torch.Tensor, torch.Tensor]:
    """-div(u u) at the u-faces and v-faces (full shapes), conservative form."""
    ny, nx = ops.ny, ops.nx
    up = ops.advection == "upwind3"
    k = (7.0 / 24.0 if up else 0.25) / ops.h          # undo the flux scaling, divide by h
    # u-momentum, x-flux at cell centres c = -1..nx (rows 0..ny-1)
    fx = _face_flux(None, U[2:-2, 0:nx + 2], U[2:-2, 1:nx + 3], U[2:-2, 2:nx + 4], U[2:-2, 3:nx + 5], up)
    # u-momentum, y-flux at corners j = 0..ny, faces i = 0..nx
    vs = V[2:ny + 3, 1:nx + 2] + V[2:ny + 3, 2:nx + 3]
    gy = _face_flux(vs, U[0:ny + 1, 2:-2], U[1:ny + 2, 2:-2], U[2:ny + 3, 2:-2], U[3:ny + 4, 2:-2], up)
    au = (fx[:, :-1] - fx[:, 1:]).add_(gy[:-1]).sub_(gy[1:]).mul_(k)
    # v-momentum, x-flux at corners j = 0..ny, i = 0..nx
    us = U[1:ny + 2, 2:-2] + U[2:ny + 3, 2:-2]
    fx = _face_flux(us, V[2:-2, 0:nx + 1], V[2:-2, 1:nx + 2], V[2:-2, 2:nx + 3], V[2:-2, 3:nx + 4], up)
    # v-momentum, y-flux at cell centres c = -1..ny (cols 0..nx-1)
    gy = _face_flux(None, V[0:ny + 2, 2:-2], V[1:ny + 3, 2:-2], V[2:ny + 4, 2:-2], V[3:ny + 5, 2:-2], up)
    av = (fx[:, :-1] - fx[:, 1:]).add_(gy[:-1]).sub_(gy[1:]).mul_(k)
    return au, av


def _add_diffusion(acc: torch.Tensor, P: torch.Tensor, nu_h2: torch.Tensor) -> torch.Tensor:
    """acc + nu * lap(P) at the interior of a 2-ghost-padded array (5-point stencil).

    ``nu_h2`` = nu / h^2 as a device scalar, so a live viscosity change needs no
    re-capture; the scaling and the accumulation are a single addcmul.
    """
    ring = (P[2:-2, 3:-1] + P[2:-2, 1:-3]).add_(P[3:-1, 2:-2]).add_(P[1:-3, 2:-2])
    return torch.addcmul(acc, ring.add_(P[2:-2, 2:-2], alpha=-4.0), nu_h2)


def _gradients(U: torch.Tensor, V: torch.Tensor, ops: Ops):
    """Velocity-gradient tensor at cells -1..ny x -1..nx and shear at corners.

    Returns (g11, g12, g21, g22) at cell centres (each (ny + 2, nx + 2)) where
    g12 = du/dy, g21 = dv/dx, plus the corner values of du/dy + dv/dx on the
    (ny + 3, nx + 3) corner grid -1..ny+1 x -1..nx+1.
    """
    ny, nx, inv_h = ops.ny, ops.nx, 1.0 / ops.h
    g11 = (U[1:ny + 3, 2:nx + 4] - U[1:ny + 3, 1:nx + 3]) * inv_h
    g22 = (V[2:ny + 4, 1:nx + 3] - V[1:ny + 3, 1:nx + 3]) * inv_h
    dudy = (U[1:ny + 4, 1:nx + 4] - U[0:ny + 3, 1:nx + 4]) * inv_h
    dvdx = (V[1:ny + 4, 1:nx + 4] - V[1:ny + 4, 0:nx + 3]) * inv_h

    def avg(q):
        return (q[:-1, :-1] + q[1:, :-1] + q[:-1, 1:] + q[1:, 1:]) * 0.25
    return g11, avg(dudy), avg(dvdx), g22, dudy + dvdx


#: Added under the square roots of the eddy-viscosity models so they stay differentiable
#: where their argument is exactly zero (forward values change by ~1e-15 relative).
_SQRT_EPS = 1e-30


def eddy_viscosity(g11, g12, g21, g22, ops: Ops) -> torch.Tensor:
    """Subgrid eddy viscosity of the configured model from the gradient tensor.

    The 2D field is treated as a 3D field with no third component, so the
    textbook 3D formulas apply unchanged. In 2D, WALE and Vreman both vanish
    in pure shear (their reason to exist: no eddy viscosity in laminar
    boundary layers), while Smagorinsky does not.
    """
    delta2 = ops.h * ops.h
    c = ops.les_c
    s12 = 0.5 * (g12 + g21)
    ss = g11 * g11 + g22 * g22 + 2.0 * s12 * s12                     # S_ij S_ij
    if ops.turbulence == "smagorinsky":
        # + _SQRT_EPS: the derivative of sqrt is infinite at exactly zero strain (still
        # flow, symmetry lines), which would turn shape gradients into NaN; it shifts
        # nu_t by ~1e-15 c^2 h^2, far below float32 resolution of any real value.
        return (c * c * delta2) * torch.sqrt(2.0 * ss + _SQRT_EPS)
    if ops.turbulence == "wale":
        q11 = g11 * g11 + g12 * g21                                 # (g g)_11
        q22 = g21 * g12 + g22 * g22
        q12 = 0.5 * (g11 * g12 + g12 * g22 + g21 * g11 + g22 * g21)  # sym part of (g g)_12
        tr3 = (q11 + q22) * (1.0 / 3.0)
        sd = (q11 - tr3) ** 2 + (q22 - tr3) ** 2 + 2.0 * q12 * q12 + tr3 * tr3
        num = sd.clamp_min(0.0) ** 1.5
        den = ss.clamp_min(0.0) ** 2.5 + sd.clamp_min(0.0) ** 1.25 + 1e-30
        return (c * c * delta2) * num / den
    if ops.turbulence == "vreman":
        # alpha_ij = du_j/dx_i ; beta = delta^2 alpha^T alpha ; B = det(beta) in 2D
        b11 = g11 * g11 + g12 * g12
        b22 = g21 * g21 + g22 * g22
        b12 = g11 * g21 + g12 * g22
        bb = (b11 * b22 - b12 * b12).clamp_min(0.0)
        aa = g11 * g11 + g12 * g12 + g21 * g21 + g22 * g22
        return (c * delta2) * torch.sqrt(bb / (aa + 1e-30) + _SQRT_EPS)     # safe sqrt, see above
    raise ValueError(ops.turbulence)


def _sgs_stress_divergence(U, V, geom: Geometry, ops: Ops, return_nut: bool = False):
    """div(2 nu_t S) at u- and v-faces (full shapes) plus the max of nu_t (and,
    with ``return_nut``, the eddy viscosity at cells -1..ny x -1..nx)."""
    ny, nx, inv_h = ops.ny, ops.nx, 1.0 / ops.h
    g11, g12, g21, g22, shear_k = _gradients(U, V, ops)
    nut = eddy_viscosity(g11, g12, g21, g22, ops)                    # cells -1..ny x -1..nx
    if not geom.empty:
        j0, j1, i0, i1 = geom.window
        # The product is formed from the un-cloned tensor before the slice write, so autograd
        # never saves a view that the in-place assignment then modifies: that keeps the
        # switch-off inside solids differentiable with respect to chi_c (shape gradients).
        win = nut[j0 + 1:j1 + 1, i0 + 1:i1 + 1] * (1.0 - geom.chi_c)
        nut = nut.clone()
        nut[j0 + 1:j1 + 1, i0 + 1:i1 + 1] = win
    nut_k = (nut[:-1, :-1] + nut[1:, :-1] + nut[:-1, 1:] + nut[1:, 1:]) * 0.25   # corners 0..ny x 0..nx
    txx = 2.0 * nut[1:ny + 1] * g11[1:ny + 1]                        # rows 0..ny-1, cells -1..nx
    tyy = 2.0 * nut[:, 1:nx + 1] * g22[:, 1:nx + 1]                  # cells -1..ny, cols 0..nx-1
    txy = nut_k * shear_k[1:ny + 2, 1:nx + 2]                        # corners 0..ny x 0..nx
    fu = (txx[:, 1:] - txx[:, :-1] + txy[1:] - txy[:-1]) * inv_h
    fv = (txy[:, 1:] - txy[:, :-1] + tyy[1:] - tyy[:-1]) * inv_h
    if return_nut:
        return fu, fv, nut.amax(), nut
    return fu, fv, nut.amax()


# ---------------------------------------------------------------------------
# wall model: the shear of an under-resolved turbulent wall layer
# ---------------------------------------------------------------------------

#: Werner & Wengle (1991) power law u+ = A (y+)^B above the viscous sublayer
WW_A = 8.3
WW_B = 1.0 / 7.0
#: y+ where the power law meets u+ = y+ (A^(1/(1-B)) = 11.81)
WW_YPLUS_CROSS = WW_A ** (1.0 / (1.0 - WW_B))


def wall_shear_excess(us: torch.Tensor, y: float, nu) -> torch.Tensor:
    """Wall shear stress the Werner-Wengle law adds to the resolved one, signed like ``us``.

    ``us`` is the wall-parallel velocity relative to the wall at distance ``y``
    (the first u- or v-face row, half a cell from the wall). The resolved no-slip
    stress there is nu |us| / y (the ghost-cell gradient). Inverting
    u+ = A (y+)^B for the friction velocity gives the explicit
    tau = (|us| (nu / y)^B / A)^(2 / (1 + B)), which equals nu |us| / y exactly at
    y+ = 11.81 and exceeds it above. So the excess is zero wherever the first
    face lies in the viscous sublayer - a resolved wall is left alone - and grows
    to the turbulent friction of the log region where the grid is coarse. In
    solver units (stress in rho U^2, nu = 1/Re).
    """
    a = us.abs()
    tau_lam = a * (nu / y)
    tau_ww = (a * (nu / y) ** WW_B * (1.0 / WW_A)) ** (2.0 / (1.0 + WW_B))
    return torch.sign(us) * (tau_ww - tau_lam).clamp_min(0.0)


def wall_model_force(u: torch.Tensor, v: torch.Tensor, ops: Ops) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Body force (full u- and v-face shapes; v None when no side wall is modelled) that adds
    the Werner-Wengle excess shear to the faces next to the no-slip domain walls.

    Why: a coarse grid puts the first face far outside the viscous sublayer
    (y+ ~ 170 at 2 m/s for a fan-driven room on a 4 cm grid), where the no-slip
    ghost gradient nu u / y is ~10 times below the turbulent friction and
    scales with 1/h. Opt-in (``SolverConfig.wall_model``): in a 2D room the
    correct wall friction does not make the speeds physical - the missing 3D
    interior dissipation does - so theory doc 18.4 measures both and explains
    why the planner leaves it off. The excess stress acts on the face row as a
    force per unit volume tau / h.
    Bottom / top walls act on u (a moving road relative to its speed), the side
    walls of a closed domain on v. Pure tensor algebra: CUDA-graph safe and
    differentiable.
    """
    y = 0.5 * ops.h
    inv_h = 1.0 / ops.h
    zrow = torch.zeros_like(u[:1])
    bottom = top = zrow
    if ops.bc_bottom in ("noslip", "moving"):
        us = u[:1] - ops.ground_speed if ops.bc_bottom == "moving" else u[:1]
        bottom = -wall_shear_excess(us, y, ops.nu) * inv_h
    if ops.bc_top == "noslip":
        top = -wall_shear_excess(u[-1:], y, ops.nu) * inv_h
    fu = torch.cat([bottom, torch.zeros_like(u[1:-1]), top], dim=0)
    if not ops.closed or (ops.bc_left != "noslip" and ops.bc_right != "noslip"):
        return fu, None
    zcol = torch.zeros_like(v[:, :1])
    left = -wall_shear_excess(v[:, :1], y, ops.nu) * inv_h if ops.bc_left == "noslip" else zcol
    right = -wall_shear_excess(v[:, -1:], y, ops.nu) * inv_h if ops.bc_right == "noslip" else zcol
    fv = torch.cat([left, torch.zeros_like(v[:, 1:-1]), right], dim=1)
    return fu, fv


def momentum_rhs(u: torch.Tensor, v: torch.Tensor, geom: Geometry, ops: Ops, *, return_nut: bool = False):
    """Explicit tendencies (Lu, Lv) with boundary rows/columns already set, and
    the largest eddy viscosity (a tensor, or None without a subgrid model).
    ``return_nut=True`` appends the eddy-viscosity field (cells -1..ny x -1..nx,
    None without a model) for the temperature equation's nu_t / Pr_t.

    Boundary handling: the inlet u and the wall v are Dirichlet (tendency 0);
    the outlet u obeys the convective condition u_t + u_x = 0 (convection
    speed 1 = free stream), integrated by the same Runge-Kutta stages. In a
    closed domain the side-wall u is Dirichlet too. With ``ops.wall_model`` the
    faces next to no-slip domain walls also feel the Werner-Wengle excess shear
    (:func:`wall_model_force`).
    """
    U, V = pad_u(u, ops), pad_v(v, ops)
    au, av = _advection(U, V, ops)
    nu_h2 = ops.nu * (1.0 / ops.h**2)
    lu = _add_diffusion(au, U, nu_h2)
    lv = _add_diffusion(av, V, nu_h2)
    nut_max = nut = None
    if ops.turbulence != "none":
        if return_nut:
            fu, fv, nut_max, nut = _sgs_stress_divergence(U, V, geom, ops, return_nut=True)
        else:
            fu, fv, nut_max = _sgs_stress_divergence(U, V, geom, ops)
        lu = lu + fu
        lv = lv + fv
    if ops.forcing is not None:
        lu = lu + ops.forcing[0]
        lv = lv + ops.forcing[1]
    if ops.wall_model:
        wu, wv = wall_model_force(u, v, ops)
        lu = lu + wu
        if wv is not None:
            lv = lv + wv
    if ops.periodic:
        return (lu, lv, nut_max, nut) if return_nut else (lu, lv, nut_max)
    if ops.closed:
        out = torch.zeros_like(u[:, :1])
    else:
        out = (u[:, -2:-1] - u[:, -1:]) * (1.0 / ops.h)             # convective outlet, speed 1
    lu = torch.cat([torch.zeros_like(out), lu[:, 1:-1], out], dim=1)
    zero = torch.zeros_like(lv[:1])
    lv = torch.cat([zero, lv[1:-1], zero], dim=0)
    return (lu, lv, nut_max, nut) if return_nut else (lu, lv, nut_max)


# ---------------------------------------------------------------------------
# projection, penalisation, time step
# ---------------------------------------------------------------------------

def divergence(u: torch.Tensor, v: torch.Tensor, h: float) -> torch.Tensor:
    """Discrete divergence at cell centres, (ny, nx)."""
    return (u[:, 1:] - u[:, :-1] + v[1:] - v[:-1]) * (1.0 / h)


def project(u: torch.Tensor, v: torch.Tensor, ops: Ops, source: torch.Tensor | None = None, *,
            return_rhs: bool = False):
    """Remove the gradient part: returns (u, v, psi) with div = 0 to round-off.

    ``psi`` solves lap psi = div(u) - source with homogeneous Neumann (or
    periodic) conditions, so boundary-normal velocities are untouched and the
    projected velocity has div u = source (the mass source of jets and vents,
    which must sum to the net boundary inflow; see :func:`_inlet_outlet`).
    ``return_rhs=True`` also returns the right-hand side (diagnostics).
    """
    rhs = divergence(u, v, ops.h)
    if source is not None:
        rhs = rhs - source
    psi = ops.poisson.solve(rhs)
    if return_rhs:
        return (*_correct(u, v, psi, ops), rhs)
    return _correct(u, v, psi, ops)


def _correct(u: torch.Tensor, v: torch.Tensor, psi: torch.Tensor, ops: Ops):
    """u - grad psi on the faces (boundary-normal faces untouched)."""
    inv_h = 1.0 / ops.h
    if ops.periodic:
        gx = (psi - torch.roll(psi, 1, dims=1)) * inv_h              # faces 0..nx-1
        gy = (psi - torch.roll(psi, 1, dims=0)) * inv_h
        u = u - torch.cat([gx, gx[:, :1]], dim=1)
        v = v - torch.cat([gy, gy[:1]], dim=0)
        return u, v, psi
    gx = (psi[:, 1:] - psi[:, :-1]) * inv_h
    gy = (psi[1:] - psi[:-1]) * inv_h
    zc = torch.zeros_like(psi[:, :1])
    zr = torch.zeros_like(psi[:1])
    u = u - torch.cat([zc, gx, zc], dim=1)
    v = v - torch.cat([zr, gy, zr], dim=0)
    return u, v, psi


def penalise(u: torch.Tensor, v: torch.Tensor, geom: Geometry, ops: Ops, bdt):
    """Implicit Brinkman penalisation on the body window.

    Solves ``u_new = u - k chi (u_new - u_s)`` exactly, i.e.
    ``u_new = (u + k chi u_s) / (1 + k chi)`` with ``k = b dt / eta``: stable for
    any eta, and for eta -> 0 it pins u to u_s wherever chi > 0. Returns the
    updated (u, v) and the momentum increments (du, dv) on the window, which
    are the (negative) impulse of the bodies on the fluid. ``u`` and ``v`` must
    be fresh tensors (they are updated in place on the window).
    """
    j0, j1, i0, i1 = geom.window
    k = bdt / ops.eta
    uw = u[j0:j1, i0:i1 + 1]
    vw = v[j0:j1 + 1, i0:i1]
    un = (uw + k * geom.a_u) / (1.0 + k * geom.chi_u)
    vn = (vw + k * geom.a_v) / (1.0 + k * geom.chi_v)
    du, dv = un - uw, vn - vw
    u[j0:j1, i0:i1 + 1] = un
    v[j0:j1 + 1, i0:i1] = vn
    return u, v, du, dv


def cfl_dt(u: torch.Tensor, v: torch.Tensor, ops: Ops, nut_max: torch.Tensor | None = None) -> torch.Tensor:
    """Stable time step as a device scalar (no host synchronisation).

    Advective limit ``cfl h / (max|u| + max|v|)`` and the explicit viscous
    limit ``0.25 h^2 / (nu + max nu_t)`` (SSP-RK3 is stable to ~0.31 on the
    negative real axis for the 2D 5-point Laplacian).
    """
    umax = (u.abs().amax() + v.abs().amax()).clamp_min(1e-6)
    dt = ops.cfl * ops.h / umax
    nu = ops.nu if nut_max is None else ops.nu + nut_max
    return torch.minimum(dt, 0.25 * ops.h * ops.h / nu)


# ---------------------------------------------------------------------------
# passive dye
# ---------------------------------------------------------------------------

def _van_leer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Van Leer limited slope: harmonic mean of one-sided differences, 0 at extrema."""
    ab = a * b
    s = a + b
    return torch.where(ab > 0, 2.0 * ab / torch.where(ab > 0, s, torch.ones_like(s)), torch.zeros_like(ab))


def advect_dye(c: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt, ops: Ops) -> torch.Tensor:
    """One forward-Euler step of the dye with a MUSCL / van Leer TVD flux.

    Purely visual, so first order in time is fine; TVD keeps it bounded and
    the final clamp removes the small over/undershoots that unsplit 2D TVD
    can still produce at CFL ~ 0.5. Concentrations below 1e-6 (invisible) are
    set to exactly zero: the limiter otherwise lets the far tails decay into
    subnormal floats, which are ~100x slower on x86 CPUs.
    """
    if ops.periodic:
        cx = torch.cat([c[:, -2:], c, c[:, :2]], dim=1)
        cy = torch.cat([c[-2:], c, c[:2]], dim=0)
    else:
        cx = torch.cat([ops.dye_inlet.expand(-1, 2), c, c[:, -1:].expand(-1, 2)], dim=1)
        cy = torch.cat([c[:1].expand(2, -1), c, c[-1:].expand(2, -1)], dim=0)
    s = _van_leer(cx[:, 1:-1] - cx[:, :-2], cx[:, 2:] - cx[:, 1:-1])        # cells -1..nx
    left = cx[:, 1:-2] + 0.5 * s[:, :-1]
    right = cx[:, 2:-1] - 0.5 * s[:, 1:]
    fx = u * torch.where(u > 0, left, right)
    s = _van_leer(cy[1:-1] - cy[:-2], cy[2:] - cy[1:-1])
    low = cy[1:-2] + 0.5 * s[:-1]
    high = cy[2:-1] - 0.5 * s[1:]
    fy = v * torch.where(v > 0, low, high)
    c = c - (dt * (1.0 / ops.h)) * (fx[:, 1:] - fx[:, :-1] + fy[1:] - fy[:-1])
    return torch.nn.functional.threshold(c, 1e-6, 0.0).clamp(max=1.0)


# ---------------------------------------------------------------------------
# the time step
# ---------------------------------------------------------------------------

def _centre_u(u: torch.Tensor) -> torch.Tensor:
    return 0.5 * (u[:, 1:] + u[:, :-1])


def _centre_v(v: torch.Tensor) -> torch.Tensor:
    return 0.5 * (v[1:] + v[:-1])


def _trace_put(trace: dict, key: str, u=None, v=None, c=None) -> None:
    """Store cell-centred copies of a stage's fields in a solver trace."""
    if u is not None:
        trace[f"{key}_u"] = _centre_u(u).detach().clone()
    if v is not None:
        trace[f"{key}_v"] = _centre_v(v).detach().clone()
    if c is not None:
        trace[key] = c.detach().clone()


def _zero_dye_in_solids(dye: torch.Tensor, geom: Geometry) -> torch.Tensor:
    """Smoke is zero inside solids (chi > 1/2): numerical spreading puts a little into
    the diffuse interface, where the Brinkman velocity is ~0 and it would linger.

    ``dye`` must be a fresh tensor (the output of :func:`advect_dye`): the window is
    multiplied in place, which keeps this on the default path down to two small
    window kernels instead of a full-field copy. That is autograd-safe because the
    clamp that produced ``dye`` does not save its output for the backward pass.
    """
    if geom.empty:
        return dye
    j0, j1, i0, i1 = geom.window
    dye[..., j0:j1, i0:i1].mul_((geom.chi_c <= 0.5).to(dye.dtype))
    return dye


def _add_buoyancy(uh, vh, theta, geom: Geometry, ops: Ops, ph, bdt, w, acc_u, acc_v, paint_acc):
    """Add the stage's Boussinesq impulse ``b dt Ri theta`` to the penalised stage velocity,
    everywhere -- also inside solids -- and book the solid's share against the body force.

    Why after the penalisation and also in solids (the 3D solver's rule, theory_ns3d.md 15.3):
    buoyancy is mostly balanced by pressure, and the projection that supplies the pressure
    runs after the penalisation. Added to the tendency before it (as this solver did until
    critic round 5), the penalisation removed the buoyancy of the fictitious fluid inside each
    body and booked it as a force on the body -- a spurious Archimedes lift Ri theta_s A (Cl
    +1.77 for a hot cylinder in still cold fluid whose true force is ~0) -- and the forcing was
    no longer a gradient across the interface, so a uniformly warm fluid around a body was
    not at rest (|u| 1e-2). Added here, a stratification is an exact discrete rest state, and
    the momentum balance of the body's area gives its force as the penalisation impulse minus
    ``Ri integral_body(theta)``: the share ``w b dt Ri vol theta_f`` (``vol`` the unshifted area
    fraction, :class:`.geometry.ThermalGeometry`) joins the impulse accumulators, which the
    force sums turn into -Ri A theta for a uniform theta. Painted obstacles get the same share.
    Returns (uh, vh, acc_u, acc_v, paint_acc)."""
    fu, fv = phx.buoyancy(theta, ph, ops)
    if fu is not None:
        uh = uh + bdt * fu
    if fv is not None:
        vh = vh + bdt * fv
    if not geom.empty:
        j0, j1, i0, i1 = geom.window
        tg = geom.thermal
        vu = tg.vol_u if tg is not None and tg.vol_u is not None else geom.chi_u
        vv = tg.vol_v if tg is not None and tg.vol_v is not None else geom.chi_v
        if fu is not None:
            share = fu[j0:j1, i0:i1 + 1] * vu * (bdt * w)
            acc_u = share if acc_u is None else acc_u + share
        if fv is not None:
            share = fv[j0:j1 + 1, i0:i1] * vv * (bdt * w)
            acc_v = share if acc_v is None else acc_v + share
    if ph.paint and paint_acc is not None:
        zero = torch.zeros((), device=uh.device, dtype=uh.dtype)
        su = (fu * ph.paint_u * ph.paint_wu).sum() if fu is not None else zero
        sv = (fv * ph.paint_v * ph.paint_wv).sum() if fv is not None else zero
        paint_acc = paint_acc + torch.stack([su, sv]) * (bdt * w)
    return uh, vh, acc_u, acc_v, paint_acc


def step_full(state: State, geom: Geometry, ops: Ops, dt=None, *, out: dict | None = None, diag: bool = False,
              trace: dict | None = None) -> tuple[State, torch.Tensor]:
    """Advance one SSP-RK3 step. Returns the new state and per-body
    coefficients, shape (nbody, 3) = [Cd, Cl, Cm] (Cm positive nose-up).

    ``dt`` may be a float, a 0-d tensor, or None for the CFL / viscous limit
    evaluated on the device from the current state (or the PID step,
    ``adaptive_dt="pid"``).

    Placed physics (``ops.phys``, DESIGN.md 10) enters every stage: the
    temperature is integrated with the same SSP-RK3 stages as the momentum
    (bounded advection, diffusion, heaters, implicit penalisation toward hot
    bodies and jets), fan forces join the momentum tendency, Boussinesq buoyancy is
    added after the penalisation (:func:`_add_buoyancy`), jets / vents and painted
    obstacles are penalised after the bodies,
    and the projection carries the ports' mass source. With ``out`` (a dict)
    the step also reports, as device tensors: "heat" (heat rate of each
    temperature body into the fluid), "theta_wall" (mean skin temperature of each
    heat-flux body), "wall_heat" (left, right, bottom, top), "fan_thrust",
    "fan_ud" (disc velocities), "paint_force" ([Cd, Cl] of painted obstacles) and,
    with ``diag=True``, the diagnostics row "diag" (diagnostics.DIAG_COLUMNS[1:]).
    ``trace`` (a dict, eager use) collects every stage's fields: explicit update,
    after penalisation, Poisson right-hand side, pressure potential, correction,
    projected velocity.
    """
    ph = ops.phys
    u0, v0 = state.u, state.v
    up, vp = u0, v0
    thermal = ph is not None and ph.thermal and state.theta is not None
    th0 = state.theta if thermal else None
    thp = th0
    acc_u = acc_v = None
    psi_w = None
    heat_acc = paint_acc = jet_acc = None
    ctrl = state.ctrl
    fan_f = fan_thrust = fan_ud = None
    want_nut = thermal and ops.turbulence != "none"
    src = ph.src if (ph is not None and ph.ports) else None
    extra_pen = ph is not None and (ph.ports or ph.paint)
    last_psi = last_rhs = None
    for s, ((a, b), w) in enumerate(zip(SSP_RK3, FORCE_WEIGHTS)):
        nut = None
        if want_nut:
            lu, lv, nut_max, nut = momentum_rhs(up, vp, geom, ops, return_nut=True)
        else:
            lu, lv, nut_max = momentum_rhs(up, vp, geom, ops)
        if s == 0:
            if dt is None:
                # a fixed dt lives in a pre-built device tensor: creating one here
                # would be a host->device copy, illegal inside CUDA-graph capture
                if ops.dt_fixed is not None:
                    dt = ops.extras["dt_fixed"]
                else:
                    dt = cfl_dt(u0, v0, ops, nut_max)
                    if ph is not None:
                        dt = phx.physics_dt_limit(dt, u0, v0, th0, nut_max, ph, ops)
                        if ph.pid:
                            dt, ctrl = pid_dt(dt, state.dt, ctrl)
            if not torch.is_tensor(dt):
                dt = torch.tensor(float(dt), device=ops.device, dtype=ops.dtype)
            if ph is not None and ph.nfan:
                fan_thrust, ctrl, fan_ud = phx.fan_control(u0, v0, ctrl, dt, ph)
                fan_f = phx.fan_forces(fan_thrust, ph)
        lt = None
        if ph is not None:
            # (buoyancy is not a tendency: it is added after the penalisation, below)
            if fan_f is not None:
                lu = lu + fan_f[0]
                lv = lv + fan_f[1]
            if thermal:
                lt, _ = phx.theta_rhs(thp, up, vp, nut, geom, ph, ops)
        bdt = dt * b
        if a == 0.0:
            uh = torch.addcmul(up, lu, bdt)
            vh = torch.addcmul(vp, lv, bdt)
        else:
            uh = torch.addcmul(torch.lerp(up, u0, a), lu, bdt)
            vh = torch.addcmul(torch.lerp(vp, v0, a), lv, bdt)
        if lt is not None:
            thh = torch.addcmul(thp if a == 0.0 else torch.lerp(thp, th0, a), lt, bdt)
        if not ops.periodic:
            uh = _inlet_outlet(uh, ops)
        if trace is not None:
            _trace_put(trace, f"stage{s + 1}/explicit", uh, vh)
            if lt is not None:
                _trace_put(trace, f"stage{s + 1}/explicit_theta", c=thh)
        if not geom.empty:
            uh, vh, du, dv = penalise(uh, vh, geom, ops, bdt)
            acc_u = du * w if acc_u is None else acc_u + du * w
            acc_v = dv * w if acc_v is None else acc_v + dv * w
        if extra_pen:
            uh, vh, imp = phx.penalise_extra(uh, vh, ph, bdt / ops.eta)
            if imp is not None:
                paint_acc = imp * w if paint_acc is None else paint_acc + imp * w
        if thermal and ph.buoyant:
            uh, vh, acc_u, acc_v, paint_acc = _add_buoyancy(uh, vh, thp, geom, ops, ph, bdt, w,
                                                            acc_u, acc_v, paint_acc)
        if ops.periodic and (extra_pen or not geom.empty):   # keep the duplicate periodic faces identical
            uh, vh = _apply_boundary_values(uh, vh, ops)
        if lt is not None:
            thh, heat, jet_heat = phx.penalise_theta(thh, geom, ph, bdt)
            if heat is not None:
                heat_acc = heat * w if heat_acc is None else heat_acc + heat * w
            if jet_heat is not None:
                jet_acc = jet_heat * w if jet_acc is None else jet_acc + jet_heat * w
            thp = thh
        if trace is not None:
            _trace_put(trace, f"stage{s + 1}/penalised", uh, vh)
        if (diag and s == 2) or trace is not None:
            up, vp, psi, rhs = project(uh, vh, ops, src, return_rhs=True)
            last_psi, last_rhs = psi, rhs
        else:
            up, vp, psi = project(uh, vh, ops, src)
        if trace is not None:
            _trace_put(trace, f"stage{s + 1}/divergence", c=last_rhs)
            _trace_put(trace, f"stage{s + 1}/pressure_potential", c=psi)
            _trace_put(trace, f"stage{s + 1}/correction", up - uh, vp - vh)
            _trace_put(trace, f"stage{s + 1}/projected", up, vp)
            if thermal:
                _trace_put(trace, f"stage{s + 1}/theta", c=thp)
        psi_w = psi * w if psi_w is None else torch.add(psi_w, psi, alpha=w)
    # Step-averaged pressure. The stage gradients reach u^(n+1) with the same
    # weights as the penalisation increments, u^(n+1) = u^n + ... - grad(sum w_s psi_s),
    # so p = sum w_s psi_s / dt is exactly the pressure of the discrete momentum
    # balance. The last stage alone (psi_3 / (b_3 dt)) is not: each stage also
    # re-cleans the velocity the previous projection left inside the solid, which
    # adds a dipole-like potential (control-volume drag 6 % low, cylinder Re 40).
    p = psi_w * (1.0 / dt)
    if geom.empty:
        coeffs = torch.zeros(geom.nbody, 3, device=ops.device, dtype=ops.dtype)
    else:
        scale = -(ops.h * ops.h) / (dt * 0.5 * ops.force_ref)
        ru = torch.einsum("bkij,ij->bk", geom.k_u, acc_u)
        rv = torch.einsum("bkij,ij->bk", geom.k_v, acc_v)
        coeffs = torch.stack([ru[:, 0], rv[:, 0], ru[:, 1] + rv[:, 1]], dim=1) * scale
    dye, dye_extra = state.dye, state.dye_extra
    t_new = state.t + dt
    if ph is not None and ph.n_dye:
        c = dye[None] if dye_extra is None else torch.cat([dye[None], dye_extra])
        c = phx.update_dye(c, up, vp, dt, t_new, geom, ph, ops)
        dye, dye_extra = c[0], (c[1:] if c.shape[0] > 1 else None)
    elif dye is not None:
        dye = _zero_dye_in_solids(advect_dye(dye, up, vp, dt, ops), geom)
        if ph is not None and ph.paint:
            dye = dye * (ph.paint_c <= 0.5).to(dye.dtype)
    new = State(up, vp, p, dye, t_new, dt * torch.ones_like(state.t), thp, dye_extra, ctrl)
    if out is not None:
        _report(out, state, new, geom, ops, dt, heat_acc, paint_acc, fan_thrust, fan_ud, diag, last_psi, last_rhs,
                src, jet_acc)
    if trace is not None:
        _trace_put(trace, "final", up, vp)
        _trace_put(trace, "final_p", c=p)
        if thermal:
            _trace_put(trace, "final_theta", c=thp)
    return new, coeffs


def _report(out: dict, old: State, new: State, geom: Geometry, ops: Ops, dt, heat_acc, paint_acc, fan_thrust,
            fan_ud, diag: bool, psi, rhs, src, jet_acc=None) -> None:
    """Fill the optional per-step outputs of :func:`step_full` (device tensors).

    With heat transfer the step also reports the terms of the energy balance
    (theory doc 16.6): "heat_storage" (rate of change of the heat content over the
    step), "jet_heat" (enthalpy the jets add), "flow_heat" (advective heat in at the
    inlet and out at the outlet) and "vent_heat" (heat leaving with the vents' fluid, and
    with the uniform leak that balances an unbalanced closed box: ``physics._build_ports``).
    """
    ph = ops.phys
    h2 = ops.h * ops.h
    if ph is not None and ph.thermal and new.theta is not None:
        tg = geom.thermal
        if tg is not None and tg.temp_ids:
            out["heat"] = heat_acc * (h2 / dt)
        if tg is not None and tg.flux_ids and not geom.empty:
            j0, j1, i0, i1 = geom.window
            out["theta_wall"] = torch.einsum("bij,ij->b", tg.w_f, new.theta[j0:j1, i0:i1])
        out["wall_heat"] = phx.wall_heat_rates(new.theta, ph, ops)
        out["heat_storage"] = ((new.theta.sum() - old.theta.sum()) * (h2 / dt)).reshape(1)
        out["flow_heat"] = phx.flow_heat_rates(new.theta, new.u, ph, ops)
        if ph.ports:
            out["vent_heat"] = (-(ph.src_neg * new.theta).sum() * h2).reshape(1)
        if jet_acc is not None:
            out["jet_heat"] = jet_acc * (h2 / dt)
    if fan_thrust is not None:
        out["fan_thrust"] = fan_thrust
        out["fan_ud"] = fan_ud
    if paint_acc is not None:
        out["paint_force"] = paint_acc * (-h2 / (dt * 0.5 * ops.force_ref))
    if diag:
        out["diag"] = diagnostics_row(old, new, psi, rhs, ops, src)


def step(state: State, geom: Geometry, ops: Ops, dt=None) -> tuple[State, torch.Tensor]:
    """One time step; returns the new state and the total [Cd, Cl, Cm] over all
    bodies (a differentiable tensor of shape (3,))."""
    new, coeffs = step_full(state, geom, ops, dt)
    return new, coeffs.sum(0)


__all__ = [
    "Ops",
    "State",
    "advect_dye",
    "auto_eta",
    "build_geometry",
    "cfl_dt",
    "divergence",
    "eddy_viscosity",
    "face_coordinates",
    "init_state",
    "make_operators",
    "momentum_rhs",
    "pad_u",
    "pad_v",
    "project",
    "set_reynolds",
    "step",
    "step_full",
    "wall_model_force",
    "wall_shear_excess",
]
