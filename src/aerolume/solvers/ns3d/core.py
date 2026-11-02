"""Functional, differentiable core of the 3D incompressible Navier-Stokes solver.

Equations (non-dimensional, Re = U L / nu), exactly the 2D model in three
dimensions::

    du/dt + div(u u) = -grad p + (1/Re) lap u + div(2 nu_t S) - (chi/eta)(u - u_s)
    div u = 0

on a uniform staggered (MAC) grid of nx x ny x nz cells of size h:

* ``u[k, j, i]`` on x-faces at (x0 + i h, y0 + (j+1/2) h, z0 + (k+1/2) h), (nz, ny, nx+1)
* ``v[k, j, i]`` on y-faces at (x0 + (i+1/2) h, y0 + j h, z0 + (k+1/2) h), (nz, ny+1, nx)
* ``w[k, j, i]`` on z-faces at (x0 + (i+1/2) h, y0 + (j+1/2) h, z0 + k h), (nz+1, ny, nx)
* ``p[k, j, i]`` at cell centres, (nz, ny, nx)

Time integration is the three-stage SSP Runge-Kutta scheme with a projection
after every stage (the 2D scheme, ``docs/theory_ns2d.md`` 5)::

    u_hat = lerp(u_prev, u_0, a) + b dt L(u_prev)
    u_hat <- (u_hat + k chi u_s) / (1 + k chi),  k = b dt / eta     (implicit penalisation)
    u     = u_hat - grad psi,  lap psi = div u_hat                   (exact projection)

with (a, b) = (0, 1), (3/4, 1/4), (1/3, 2/3); the penalisation increments,
weighted (1/6, 2/3, 1), are the momentum the bodies remove: the force.

Every function here is pure torch with static control flow, so a step can be
captured in a CUDA graph, differentiated with autograd and run on CPU or CUDA.
The fast path of :class:`~aerolume.solvers.ns3d.FlowSolver3D` on CUDA replaces
the right-hand side and the projection updates with fused kernels
(:mod:`.fused`) that compute the same arithmetic; this module is the reference.
``docs/theory_ns3d.md`` derives and verifies each choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from ..ns2d.core import ETA_PER_H, FORCE_WEIGHTS, LES_CONSTANTS, MASK_SHIFT, SSP_RK3, STIFFNESS_MIN
from .config import SolverConfig3D
from .geometry import Geometry3D, build_geometry3d
from .poisson import PoissonSolver3D

#: SSP-RK3 is stable to about 2.51 on the negative real axis; the 7-point Laplacian
#: reaches 12 nu / h^2, so the explicit viscous limit is 0.209 h^2 / nu. 1/6 keeps
#: the same 80 % margin the 2D solver keeps (0.25 of 0.31).
VISCOUS_DT = 1.0 / 6.0
#: names of the six coefficients, in order
COEFF_NAMES = ("cd", "cs", "cl", "c_roll", "cm", "cn")
_SQRT_EPS = 1e-30


# ---------------------------------------------------------------------------
# operators and state
# ---------------------------------------------------------------------------

@dataclass
class Ops3D:
    """Everything about the grid and the physics that a step needs but that is
    not part of the evolving state. Python scalars are static (baked into a
    captured CUDA graph); tensors (``nu``, ``eta``) may be changed in place.

    ``ghost`` holds the ghost-cell rule of every velocity component along every
    axis (see :func:`pad3`); ``nu_value`` mirrors ``nu`` as a Python float for
    the fused kernels, which take the viscosity as a launch argument.
    """
    cfg: SolverConfig3D
    nx: int
    ny: int
    nz: int
    h: float
    x0: float
    y0: float
    z0: float
    device: torch.device
    dtype: torch.dtype
    periodic: bool
    periodic_axes: tuple[bool, bool, bool]
    symmetry: bool
    advection: str
    turbulence: str
    bc_y: str
    bc_zmin: str
    bc_zmax: str
    ground_speed: float
    cfl: float
    dt_fixed: float | None
    ref_area: float
    length_ref: float
    span_ref: float
    moment_ref: tuple[float, float, float]
    mask_width: float
    mask_shift: float
    eta_fixed: float | None
    les_c: float
    nu: torch.Tensor
    eta: torch.Tensor
    nu_value: float
    u_inlet: torch.Tensor                  # (nz, ny, 1)
    poisson: PoissonSolver3D
    ghost: dict
    forcing: tuple[float, float, float] | None = None
    extras: dict = field(default_factory=dict)
    closed: bool = False                   # walls at x0 / x1 (no inflow), DESIGN.md 10


class State3D(NamedTuple):
    """Evolving solver state. ``p`` is the step-averaged pressure of the step that
    produced this state (output only); ``dt`` is that step (0 initially)."""
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor
    t: torch.Tensor
    dt: torch.Tensor


def resolve_device_dtype(cfg: SolverConfig3D) -> tuple[torch.device, torch.dtype]:
    """Concrete torch device and dtype for a config (``device="auto"`` aware)."""
    try:
        from aerolume.core.device import resolve
    except ImportError:                     # pragma: no cover - solver used without the app core
        dev = cfg.device if cfg.device not in ("auto", None, "") else (
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = resolve(cfg.device)
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    return torch.device(dev), dtype


def auto_eta(h: float, re: float) -> float:
    """Default Brinkman permeability (the 2D calibration, theory_ns2d.md 6.3)."""
    return min(ETA_PER_H * h, h * h * re / STIFFNESS_MIN)


def set_reynolds(ops: Ops3D, re: float) -> None:
    """Change Re in place (nu, eta and the fused kernels' launch value)."""
    ops.nu.fill_(1.0 / re)
    ops.nu_value = 1.0 / re
    ops.eta.fill_(ops.eta_fixed if ops.eta_fixed is not None else auto_eta(ops.h, re))


def ghost_rules(cfg: SolverConfig3D) -> dict[str, tuple[tuple, tuple, tuple]]:
    """Ghost-cell rule per velocity component and axis: ``{comp: (x, y, z)}``, each
    axis a ``(lo, hi)`` pair of ``(kind, value)``.

    Kinds (2 ghost layers each side): ``edge`` repeats the boundary value (inlet
    value, zero-gradient outlet); ``even`` / ``odd`` mirror a cell-centred
    quantity across a boundary half a cell away (zero gradient / zero value);
    ``affine`` mirrors to a prescribed value (moving road); ``face_odd`` mirrors a
    face-located normal velocity across a wall *on* that face (value 0);
    ``periodic_cell`` / ``periodic_face`` wrap around (face arrays carry a
    duplicate last face). The y = y0 symmetry plane is a free-slip wall: the same
    rules as ``slip``.
    """
    px, py, pz = (True, True, True) if cfg.periodic else (False, cfg.bc_y == "periodic", False)
    pc, pf = ("periodic_cell", 0.0), ("periodic_face", 0.0)
    even, odd, edge, fodd = ("even", 0.0), ("odd", 0.0), ("edge", 0.0), ("face_odd", 0.0)
    zlo_tan = {"slip": even, "noslip": odd, "moving": ("affine", float(cfg.ground_speed))}[cfg.bc_zmin]
    zlo_v = {"slip": even, "noslip": odd, "moving": odd}[cfg.bc_zmin]
    zhi = {"slip": even, "noslip": odd}[cfg.bc_zmax]
    y_tan = odd if cfg.bc_y == "noslip" else even
    if cfg.domain == "closed":           # walls at x0 and x1: u = 0 on the faces, tangential odd / even
        x_tan = odd if cfg.bc_x == "noslip" else even
        ux, vx = (fodd, fodd), (x_tan, x_tan)
    else:                                # inlet (u given, v = w = 0) / zero-gradient outlet
        ux, vx = (edge, edge), (odd, edge)
    u = ((pf, pf) if px else ux,
         (pc, pc) if py else (y_tan, y_tan),
         (pc, pc) if pz else (zlo_tan, zhi))
    v = ((pc, pc) if px else vx,
         (pf, pf) if py else (fodd, fodd),
         (pc, pc) if pz else (zlo_v, zhi))
    w = ((pc, pc) if px else vx,
         (pc, pc) if py else (y_tan, y_tan),
         (pf, pf) if pz else (fodd, fodd))
    return {"u": u, "v": v, "w": w}


def make_operators3d(cfg: SolverConfig3D) -> Ops3D:
    """Build the static operators (grid, Poisson solver, BC data) for a config."""
    dev, dtype = resolve_device_dtype(cfg)
    nu = torch.tensor(1.0 / cfg.re, device=dev, dtype=dtype)
    eta = torch.tensor(cfg.eta if cfg.eta is not None else auto_eta(cfg.h, cfg.re), device=dev, dtype=dtype)
    u_in = torch.ones(cfg.nz, cfg.ny, 1, device=dev, dtype=dtype)
    pax = cfg.periodic_axes
    poisson = PoissonSolver3D(cfg.nx, cfg.ny, cfg.nz, cfg.h, periodic=pax, device=dev, dtype=dtype)
    les_c = cfg.les_constant if cfg.les_constant is not None else LES_CONSTANTS.get(cfg.turbulence, 0.0)
    extras = {}
    if cfg.dt is not None:
        extras["dt_fixed"] = torch.tensor(float(cfg.dt), device=dev, dtype=dtype)
    return Ops3D(cfg=cfg, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, h=cfg.h, x0=cfg.x0, y0=cfg.y0, z0=cfg.z0,
                 device=dev, dtype=dtype, periodic=cfg.periodic, periodic_axes=pax, symmetry=cfg.symmetry,
                 advection=cfg.advection, turbulence=cfg.turbulence, bc_y=cfg.bc_y, bc_zmin=cfg.bc_zmin,
                 bc_zmax=cfg.bc_zmax, ground_speed=float(cfg.ground_speed), cfl=cfg.cfl, dt_fixed=cfg.dt,
                 ref_area=cfg.ref_area, length_ref=cfg.length_ref, span_ref=cfg.span_ref,
                 moment_ref=cfg.moment_ref, mask_width=cfg.mask_width, mask_shift=MASK_SHIFT,
                 eta_fixed=cfg.eta, les_c=les_c, nu=nu, eta=eta, nu_value=1.0 / cfg.re, u_inlet=u_in,
                 poisson=poisson, ghost=ghost_rules(cfg), extras=extras, closed=cfg.domain == "closed")


def face_coordinates3d(ops: Ops3D) -> dict[str, torch.Tensor]:
    """1D coordinate vectors: xc/yc/zc (centres), xu (u-face x), yv (v-face y), zw (w-face z)."""
    d, t, h = ops.device, ops.dtype, ops.h
    return {
        "xc": ops.x0 + (torch.arange(ops.nx, device=d, dtype=t) + 0.5) * h,
        "yc": ops.y0 + (torch.arange(ops.ny, device=d, dtype=t) + 0.5) * h,
        "zc": ops.z0 + (torch.arange(ops.nz, device=d, dtype=t) + 0.5) * h,
        "xu": ops.x0 + torch.arange(ops.nx + 1, device=d, dtype=t) * h,
        "yv": ops.y0 + torch.arange(ops.ny + 1, device=d, dtype=t) * h,
        "zw": ops.z0 + torch.arange(ops.nz + 1, device=d, dtype=t) * h,
    }


# ---------------------------------------------------------------------------
# boundary conditions: ghost-padded copies of the velocity
# ---------------------------------------------------------------------------

def _ghost_slabs(x: torch.Tensor, dim: int, lo: tuple, hi: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """The two ghost layers below and above ``x`` along ``dim`` (index order -2, -1 and n, n+1)."""
    n = x.shape[dim]

    def make(rule, side):
        kind, val = rule
        if kind == "edge":
            return x.narrow(dim, 0 if side == "lo" else n - 1, 1)
        if kind in ("even", "odd", "affine"):
            m = x.narrow(dim, 0 if side == "lo" else n - 2, 2).flip(dim)
            if kind == "even":
                return m
            if kind == "odd":
                return -m
            return 2.0 * val - m
        if kind == "face_odd":
            return -x.narrow(dim, 1 if side == "lo" else n - 3, 2).flip(dim)
        if kind == "periodic_cell":
            return x.narrow(dim, n - 2 if side == "lo" else 0, 2)
        if kind == "periodic_face":
            return x.narrow(dim, n - 3 if side == "lo" else 1, 2)
        raise ValueError(kind)

    return make(lo, "lo"), make(hi, "hi")


def pad3(x: torch.Tensor, rules: tuple[tuple, tuple, tuple]) -> torch.Tensor:
    """``x`` with two ghost layers on every side: shape + 4 in each dimension.

    Padded index ``[k+2, j+2, i+2]`` holds ``x[k, j, i]``. Ghosts are filled axis
    by axis (x, then y on the x-padded rows, then z on everything), so edge and
    corner ghosts compose the rules of the axes involved. Built with one
    allocation and slice writes (no ghost rule saves a tensor for backward, so
    this is autograd-safe).
    """
    nz, ny, nx = x.shape
    out = x.new_empty(nz + 4, ny + 4, nx + 4)
    rx, ry, rz = rules
    lo, hi = _ghost_slabs(x, 2, *rx)
    out[2:-2, 2:-2, 2:-2] = x
    out[2:-2, 2:-2, 0:2] = lo
    out[2:-2, 2:-2, nx + 2:] = hi
    mid = out[2:-2]
    lo, hi = _ghost_slabs(mid[:, 2:-2], 1, *ry)
    out[2:-2, 0:2] = lo
    out[2:-2, ny + 2:] = hi
    lo, hi = _ghost_slabs(out[2:-2], 0, *rz)
    out[0:2] = lo
    out[nz + 2:] = hi
    return out


def pad_velocity(u, v, w, ops: Ops3D):
    """Ghost-padded copies (U, V, W) of the three velocity components."""
    g = ops.ghost
    return pad3(u, g["u"]), pad3(v, g["v"]), pad3(w, g["w"])


def inlet_outlet(u: torch.Tensor, ops: Ops3D, source: torch.Tensor | None = None) -> torch.Tensor:
    """Impose the inlet profile and balance the outlet mass flux (as in 2D).

    The outlet plane is shifted uniformly so that outflow equals inflow plus
    ``source`` (``h * sum m`` of the ports' mass sources, None = none); the
    other boundaries are impermeable or periodic, so the net boundary flux
    balances the sources exactly and the sum of the Poisson right-hand side is
    zero, as the all-Neumann pressure problem requires. A device reduction, no
    host sync. A closed domain has walls at both x ends: u = 0 there.
    """
    if ops.closed:
        z = torch.zeros_like(u[..., :1])
        return torch.cat([z, u[..., 1:-1], z], dim=2)
    corr = (ops.u_inlet.sum() - u[..., -1].sum()) * (1.0 / (ops.nz * ops.ny))
    if source is not None:
        corr = corr + source * (1.0 / (ops.nz * ops.ny))
    return torch.cat([ops.u_inlet, u[..., 1:-1], u[..., -1:] + corr], dim=2)


def sync_periodic(u, v, w, ops: Ops3D):
    """Make the duplicate faces of periodic axes bit-identical to their partners."""
    px, py, pz = ops.periodic_axes
    if px:
        u = torch.cat([u[..., :-1], u[..., :1]], dim=2)
    if py:
        v = torch.cat([v[:, :-1], v[:, :1]], dim=1)
    if pz:
        w = torch.cat([w[:-1], w[:1]], dim=0)
    return u, v, w


def apply_boundary_values(u, v, w, ops: Ops3D):
    """All Dirichlet values: inlet u and outlet mass balance, zero normal velocity
    on walls (and the symmetry plane), periodic duplicates."""
    px, py, pz = ops.periodic_axes
    if not px:
        u = inlet_outlet(u, ops)
    if not py:
        z = torch.zeros_like(v[:, :1])
        v = torch.cat([z, v[:, 1:-1], z], dim=1)
    if not pz:
        z = torch.zeros_like(w[:1])
        w = torch.cat([z, w[1:-1], z], dim=0)
    return sync_periodic(u, v, w, ops)


# ---------------------------------------------------------------------------
# right-hand side: advection, diffusion, subgrid stress
# ---------------------------------------------------------------------------

def face_flux(vsum: torch.Tensor | None, a, b, c, d, upwind: bool) -> torch.Tensor:
    """Scaled flux through the face between b and c (neighbours a, b | c, d).

    Identical to the 2D flux: ``vsum`` is twice the advecting velocity (None =
    self-advection, ``b + c``); the result is the flux times 4 (central) or 24/7
    (upwind3: 4th-order central part plus |V| times a 4th-difference
    dissipation, i.e. the 3rd-order upwind-biased reconstruction).
    """
    s = b + c
    vsum = s if vsum is None else vsum
    if not upwind:
        return vsum * s
    cen = torch.add(s, a + d, alpha=-1.0 / 7.0)
    dis = torch.add(b - c, a - d, alpha=-1.0 / 3.0)
    return torch.addcmul(vsum * cen, vsum.abs(), dis, value=3.0 / 7.0)


def advection_scale(ops: Ops3D) -> float:
    """Factor that undoes the flux scaling of :func:`face_flux` and divides by h."""
    return (7.0 / 24.0 if ops.advection == "upwind3" else 0.25) / ops.h


def advection3(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, ops: Ops3D):
    """-div(u u) at the u-, v- and w-faces (full shapes), conservative form.

    Normal fluxes live at cell centres, cross fluxes on cell edges with the
    advecting velocity averaged to the edge (``docs/theory_ns3d.md`` 3).
    """
    nz, ny, nx = ops.nz, ops.ny, ops.nx
    up = ops.advection == "upwind3"
    k = advection_scale(ops)
    m = slice(2, -2)
    # --- u-momentum (nz, ny, nx+1)
    fx = face_flux(None, U[m, m, 0:nx + 2], U[m, m, 1:nx + 3], U[m, m, 2:nx + 4], U[m, m, 3:nx + 5], up)
    au = fx[..., :-1] - fx[..., 1:]
    vs = V[m, 2:ny + 3, 1:nx + 2] + V[m, 2:ny + 3, 2:nx + 3]
    g = face_flux(vs, U[m, 0:ny + 1, m], U[m, 1:ny + 2, m], U[m, 2:ny + 3, m], U[m, 3:ny + 4, m], up)
    au = au + (g[:, :-1] - g[:, 1:])
    ws = W[2:nz + 3, m, 1:nx + 2] + W[2:nz + 3, m, 2:nx + 3]
    g = face_flux(ws, U[0:nz + 1, m, m], U[1:nz + 2, m, m], U[2:nz + 3, m, m], U[3:nz + 4, m, m], up)
    au = (au + (g[:-1] - g[1:])) * k
    # --- v-momentum (nz, ny+1, nx)
    us = U[m, 1:ny + 2, m] + U[m, 2:ny + 3, m]
    g = face_flux(us, V[m, m, 0:nx + 1], V[m, m, 1:nx + 2], V[m, m, 2:nx + 3], V[m, m, 3:nx + 4], up)
    av = g[..., :-1] - g[..., 1:]
    g = face_flux(None, V[m, 0:ny + 2, m], V[m, 1:ny + 3, m], V[m, 2:ny + 4, m], V[m, 3:ny + 5, m], up)
    av = av + (g[:, :-1] - g[:, 1:])
    ws = W[2:nz + 3, 1:ny + 2, m] + W[2:nz + 3, 2:ny + 3, m]
    g = face_flux(ws, V[0:nz + 1, m, m], V[1:nz + 2, m, m], V[2:nz + 3, m, m], V[3:nz + 4, m, m], up)
    av = (av + (g[:-1] - g[1:])) * k
    # --- w-momentum (nz+1, ny, nx)
    us = U[1:nz + 2, m, m] + U[2:nz + 3, m, m]
    g = face_flux(us, W[m, m, 0:nx + 1], W[m, m, 1:nx + 2], W[m, m, 2:nx + 3], W[m, m, 3:nx + 4], up)
    aw = g[..., :-1] - g[..., 1:]
    vs = V[1:nz + 2, m, m] + V[2:nz + 3, m, m]
    g = face_flux(vs, W[m, 0:ny + 1, m], W[m, 1:ny + 2, m], W[m, 2:ny + 3, m], W[m, 3:ny + 4, m], up)
    aw = aw + (g[:, :-1] - g[:, 1:])
    g = face_flux(None, W[0:nz + 2, m, m], W[1:nz + 3, m, m], W[2:nz + 4, m, m], W[3:nz + 5, m, m], up)
    aw = (aw + (g[:-1] - g[1:])) * k
    return au, av, aw


def laplacian_padded(P: torch.Tensor) -> torch.Tensor:
    """h^2 times the 7-point Laplacian at the interior of a 2-ghost-padded array."""
    m = slice(2, -2)
    ring = (P[m, m, 3:-1] + P[m, m, 1:-3]) + (P[m, 3:-1, m] + P[m, 1:-3, m]) + (P[3:-1, m, m] + P[1:-3, m, m])
    return ring - 6.0 * P[m, m, m]


def velocity_gradient(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, h: float) -> dict[str, torch.Tensor]:
    """Velocity-gradient tensor at cell centres -1..n (one ghost ring) and edge shears.

    Works on any z-slab of the padded arrays (sizes are taken from the shapes),
    which lets field outputs be computed in chunks. Returns ``g`` (a 3 x 3 nested
    tuple, ``g[i][j] = du_i/dx_j``, each (nz+2, ny+2, nx+2)) and the edge sums
    ``sxy = du/dy + dv/dx`` (x-y edges, (nz+2, ny+3, nx+3)), ``sxz`` (x-z edges,
    (nz+3, ny+2, nx+3)), ``syz`` (y-z edges, (nz+3, ny+3, nx+2)) on the same
    extended index ranges (array index = position + 1).
    """
    nz, ny, nx = U.shape[0] - 4, U.shape[1] - 4, U.shape[2] - 5
    inv = 1.0 / h
    g11 = (U[1:nz + 3, 1:ny + 3, 2:nx + 4] - U[1:nz + 3, 1:ny + 3, 1:nx + 3]) * inv
    g22 = (V[1:nz + 3, 2:ny + 4, 1:nx + 3] - V[1:nz + 3, 1:ny + 3, 1:nx + 3]) * inv
    g33 = (W[2:nz + 4, 1:ny + 3, 1:nx + 3] - W[1:nz + 3, 1:ny + 3, 1:nx + 3]) * inv
    dudy = (U[1:nz + 3, 1:ny + 4, 1:nx + 4] - U[1:nz + 3, 0:ny + 3, 1:nx + 4]) * inv
    dvdx = (V[1:nz + 3, 1:ny + 4, 1:nx + 4] - V[1:nz + 3, 1:ny + 4, 0:nx + 3]) * inv
    dudz = (U[1:nz + 4, 1:ny + 3, 1:nx + 4] - U[0:nz + 3, 1:ny + 3, 1:nx + 4]) * inv
    dwdx = (W[1:nz + 4, 1:ny + 3, 1:nx + 4] - W[1:nz + 4, 1:ny + 3, 0:nx + 3]) * inv
    dvdz = (V[1:nz + 4, 1:ny + 4, 1:nx + 3] - V[0:nz + 3, 1:ny + 4, 1:nx + 3]) * inv
    dwdy = (W[1:nz + 4, 1:ny + 4, 1:nx + 3] - W[1:nz + 4, 0:ny + 3, 1:nx + 3]) * inv

    def a_xy(q):
        return (q[:, :-1, :-1] + q[:, 1:, :-1] + q[:, :-1, 1:] + q[:, 1:, 1:]) * 0.25

    def a_xz(q):
        return (q[:-1, :, :-1] + q[1:, :, :-1] + q[:-1, :, 1:] + q[1:, :, 1:]) * 0.25

    def a_yz(q):
        return (q[:-1, :-1] + q[1:, :-1] + q[:-1, 1:] + q[1:, 1:]) * 0.25

    g = ((g11, a_xy(dudy), a_xz(dudz)),
         (a_xy(dvdx), g22, a_yz(dvdz)),
         (a_xz(dwdx), a_yz(dwdy), g33))
    return {"g": g, "sxy": dudy + dvdx, "sxz": dudz + dwdx, "syz": dvdz + dwdy}


def traceless(g):
    """The gradient tensor with ``g33`` replaced by ``-(g11 + g22)``.

    The models assume incompressible flow; at interior cells the discrete
    divergence is zero to round-off after every projection, so this changes
    nothing there, and it leaves eight independent components - which is what
    lets the fused CUDA path evaluate a model in one kernel (8-input limit).
    """
    return (g[0], g[1], (g[2][0], g[2][1], -(g[0][0] + g[1][1])))


def eddy_viscosity3d(g, ops: Ops3D) -> torch.Tensor:
    """Subgrid eddy viscosity of the configured model from ``g[i][j] = du_i/dx_j``.

    Smagorinsky (Lilly), WALE (Nicoud & Ducros 1999) and Vreman (2004) with the
    filter width h and the 2D solver's constants. WALE and Vreman vanish in pure
    shear (laminar boundary layers); Smagorinsky does not. The solver passes the
    :func:`traceless` gradient.
    """
    delta2 = ops.h * ops.h
    c = ops.les_c
    s = [[0.5 * (g[i][j] + g[j][i]) for j in range(3)] for i in range(3)]
    ss = sum(s[i][j] * s[i][j] for i in range(3) for j in range(3))              # S_ij S_ij
    if ops.turbulence == "smagorinsky":
        return (c * c * delta2) * torch.sqrt(2.0 * ss + _SQRT_EPS)
    if ops.turbulence == "wale":
        g2 = [[sum(g[i][k] * g[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        tr3 = (g2[0][0] + g2[1][1] + g2[2][2]) * (1.0 / 3.0)
        sd = 0.0
        for i in range(3):
            for j in range(3):
                sdij = 0.5 * (g2[i][j] + g2[j][i])
                if i == j:
                    sdij = sdij - tr3
                sd = sd + sdij * sdij
        num = sd.clamp_min(0.0) ** 1.5
        den = ss.clamp_min(0.0) ** 2.5 + sd.clamp_min(0.0) ** 1.25 + 1e-30
        return (c * c * delta2) * num / den
    if ops.turbulence == "vreman":
        # alpha_ij = du_j/dx_i = g[j][i]; beta_ij = sum_m alpha_mi alpha_mj = sum_m g[i][m] g[j][m]
        b = [[sum(g[i][m] * g[j][m] for m in range(3)) for j in range(3)] for i in range(3)]
        bb = (b[0][0] * b[1][1] - b[0][1] * b[0][1] + b[0][0] * b[2][2] - b[0][2] * b[0][2]
              + b[1][1] * b[2][2] - b[1][2] * b[1][2]).clamp_min(0.0)
        aa = sum(g[i][j] * g[i][j] for i in range(3) for j in range(3))
        return (c * delta2) * torch.sqrt(bb / (aa + 1e-30) + _SQRT_EPS)
    raise ValueError(ops.turbulence)


def sgs_divergence3d(U, V, W, geom: Geometry3D, ops: Ops3D, out: dict | None = None):
    """div(2 nu_t S) at the u-, v- and w-faces (full shapes) plus the max of nu_t.

    nu_t lives at cell centres (with a ghost ring); diagonal stresses at centres,
    shear stresses on edges with nu_t averaged from the four cells around each
    edge: conservative and dissipative, as in 2D. nu_t is zero inside solids.
    ``out`` (a dict) receives ``nut`` (nz+2, ny+2, nx+2), which the heat
    equation's turbulent diffusivity needs.
    """
    nz, ny, nx = ops.nz, ops.ny, ops.nx
    inv = 1.0 / ops.h
    gr = velocity_gradient(U, V, W, ops.h)
    g = gr["g"]
    nut = eddy_viscosity3d(traceless(g), ops)                            # (nz+2, ny+2, nx+2)
    if not geom.empty:
        k0, k1, j0, j1, i0, i1 = geom.window
        # product formed before the slice write (see the 2D core): differentiable in chi_c
        win = nut[k0 + 1:k1 + 1, j0 + 1:j1 + 1, i0 + 1:i1 + 1] * (1.0 - geom.chi_c)
        nut = nut.clone()
        nut[k0 + 1:k1 + 1, j0 + 1:j1 + 1, i0 + 1:i1 + 1] = win
    i = slice(1, -1)
    # edge viscosities (4-cell averages) on the edges the stresses need
    n_xy = nut[i]
    n_xy = (n_xy[:, :-1, :-1] + n_xy[:, 1:, :-1] + n_xy[:, :-1, 1:] + n_xy[:, 1:, 1:]) * 0.25   # (nz, ny+1, nx+1)
    n_xz = nut[:, i]
    n_xz = (n_xz[:-1, :, :-1] + n_xz[1:, :, :-1] + n_xz[:-1, :, 1:] + n_xz[1:, :, 1:]) * 0.25   # (nz+1, ny, nx+1)
    n_yz = nut[:, :, i]
    n_yz = (n_yz[:-1, :-1] + n_yz[1:, :-1] + n_yz[:-1, 1:] + n_yz[1:, 1:]) * 0.25               # (nz+1, ny+1, nx)
    t12 = n_xy * gr["sxy"][1:nz + 1, 1:ny + 2, 1:nx + 2]
    t13 = n_xz * gr["sxz"][1:nz + 2, 1:ny + 1, 1:nx + 2]
    t23 = n_yz * gr["syz"][1:nz + 2, 1:ny + 2, 1:nx + 1]
    t11 = 2.0 * nut[i, i] * g[0][0][i, i]                                # (nz, ny, nx+2)
    t22 = 2.0 * nut[i, :, i] * g[1][1][i, :, i]                          # (nz, ny+2, nx)
    t33 = 2.0 * nut[:, i, i] * g[2][2][:, i, i]                          # (nz+2, ny, nx)
    fu = (t11[..., 1:] - t11[..., :-1] + t12[:, 1:] - t12[:, :-1] + t13[1:] - t13[:-1]) * inv
    fv = (t12[..., 1:] - t12[..., :-1] + t22[:, 1:] - t22[:, :-1] + t23[1:] - t23[:-1]) * inv
    fw = (t13[..., 1:] - t13[..., :-1] + t23[:, 1:] - t23[:, :-1] + t33[1:] - t33[:-1]) * inv
    if out is not None:
        out["nut"] = nut
    return fu, fv, fw, nut.amax()


def boundary_tendencies(u, lu, lv, lw, ops: Ops3D):
    """Tendencies on boundary faces: inlet u Dirichlet (0), convective outlet
    ``u_t + u_x = 0`` (speed 1), zero on walls (also the x walls of a closed
    domain) and the symmetry plane."""
    px, py, pz = ops.periodic_axes
    if not px:
        out = (u[..., -2:-1] - u[..., -1:]) * (1.0 / ops.h)
        if ops.closed:
            out = torch.zeros_like(out)
        lu = torch.cat([torch.zeros_like(out), lu[..., 1:-1], out], dim=2)
    if not py:
        z = torch.zeros_like(lv[:, :1])
        lv = torch.cat([z, lv[:, 1:-1], z], dim=1)
    if not pz:
        z = torch.zeros_like(lw[:1])
        lw = torch.cat([z, lw[1:-1], z], dim=0)
    return lu, lv, lw


def momentum_rhs3d(u, v, w, geom: Geometry3D, ops: Ops3D, *, phys=None, theta=None, out: dict | None = None):
    """Explicit tendencies (Lu, Lv, Lw) with boundary faces already set, and the
    largest eddy viscosity (a tensor, or None without a subgrid model).

    ``phys`` (a :class:`~.physics.Physics3D`) adds the placed-physics momentum
    sources of fans and ports before the boundary faces are set (buoyancy is added
    after the penalisation, see :func:`step3d_phys`); ``out`` receives the eddy
    viscosity field (``"nut"``) with LES.
    """
    U, V, W = pad_velocity(u, v, w, ops)
    au, av, aw = advection3(U, V, W, ops)
    nu_h2 = ops.nu * (1.0 / ops.h ** 2)
    lu = torch.addcmul(au, laplacian_padded(U), nu_h2)
    lv = torch.addcmul(av, laplacian_padded(V), nu_h2)
    lw = torch.addcmul(aw, laplacian_padded(W), nu_h2)
    nut_max = None
    if ops.turbulence != "none":
        fu, fv, fw, nut_max = sgs_divergence3d(U, V, W, geom, ops, out=out)
        lu, lv, lw = lu + fu, lv + fv, lw + fw
    if ops.forcing is not None:
        lu, lv, lw = lu + ops.forcing[0], lv + ops.forcing[1], lw + ops.forcing[2]
    if phys is not None:
        lu, lv, lw = phys.momentum_sources(lu, lv, lw, u, v, w, theta, geom)
    lu, lv, lw = boundary_tendencies(u, lu, lv, lw, ops)
    return lu, lv, lw, nut_max


# ---------------------------------------------------------------------------
# projection, penalisation, time step
# ---------------------------------------------------------------------------

def divergence3d(u, v, w, h: float) -> torch.Tensor:
    """Discrete divergence at cell centres, (nz, ny, nx)."""
    return ((u[..., 1:] - u[..., :-1]) + (v[:, 1:] - v[:, :-1]) + (w[1:] - w[:-1])) * (1.0 / h)


def pressure_gradient(psi: torch.Tensor, ops: Ops3D):
    """grad psi on all u-, v-, w-faces (zero on non-periodic boundary faces)."""
    inv = 1.0 / ops.h
    px, py, pz = ops.periodic_axes
    if px:
        gx = (psi - torch.roll(psi, 1, dims=2)) * inv
        gx = torch.cat([gx, gx[..., :1]], dim=2)
    else:
        gx = F.pad((psi[..., 1:] - psi[..., :-1]) * inv, (1, 1))
    if py:
        gy = (psi - torch.roll(psi, 1, dims=1)) * inv
        gy = torch.cat([gy, gy[:, :1]], dim=1)
    else:
        gy = F.pad((psi[:, 1:] - psi[:, :-1]) * inv, (0, 0, 1, 1))
    if pz:
        gz = (psi - torch.roll(psi, 1, dims=0)) * inv
        gz = torch.cat([gz, gz[:1]], dim=0)
    else:
        gz = F.pad((psi[1:] - psi[:-1]) * inv, (0, 0, 0, 0, 1, 1))
    return gx, gy, gz


def project3d(u, v, w, ops: Ops3D):
    """Remove the gradient part: returns (u, v, w, psi) with div = 0 to round-off.

    ``psi`` solves lap psi = div(u) with homogeneous Neumann (or periodic)
    conditions, so boundary-normal velocities are untouched.
    """
    psi = ops.poisson.solve(divergence3d(u, v, w, ops.h))
    gx, gy, gz = pressure_gradient(psi, ops)
    return u - gx, v - gy, w - gz, psi


def penalise3d(u, v, w, geom: Geometry3D, ops: Ops3D, bdt):
    """Implicit Brinkman penalisation on the body window (as in 2D).

    Solves ``u_new = (u + k chi u_s) / (1 + k chi)`` with ``k = b dt / eta``:
    stable for any eta. Returns the updated fields and the momentum increments
    on the window. ``u``, ``v``, ``w`` must be fresh tensors (updated in place).
    """
    k0, k1, j0, j1, i0, i1 = geom.window
    k = bdt / ops.eta
    uw = u[k0:k1, j0:j1, i0:i1 + 1]
    vw = v[k0:k1, j0:j1 + 1, i0:i1]
    ww = w[k0:k1 + 1, j0:j1, i0:i1]
    un = (uw + k * geom.a_u) / (1.0 + k * geom.chi_u)
    vn = (vw + k * geom.a_v) / (1.0 + k * geom.chi_v)
    wn = (ww + k * geom.a_w) / (1.0 + k * geom.chi_w)
    du, dv, dw = un - uw, vn - vw, wn - ww
    u[k0:k1, j0:j1, i0:i1 + 1] = un
    v[k0:k1, j0:j1 + 1, i0:i1] = vn
    w[k0:k1 + 1, j0:j1, i0:i1] = wn
    return u, v, w, du, dv, dw


def cfl_dt3d(u, v, w, ops: Ops3D, nut_max: torch.Tensor | None = None) -> torch.Tensor:
    """Stable time step as a device scalar (no host synchronisation).

    Advective limit ``cfl h / (max|u| + max|v| + max|w|)`` and the explicit
    viscous limit ``h^2 / (6 (nu + max nu_t))``.
    """
    umax = (u.abs().amax() + v.abs().amax() + w.abs().amax()).clamp_min(1e-6)
    dt = ops.cfl * ops.h / umax
    nu = ops.nu if nut_max is None else ops.nu + nut_max
    return torch.minimum(dt, VISCOUS_DT * ops.h * ops.h / nu)


# ---------------------------------------------------------------------------
# forces and moments
# ---------------------------------------------------------------------------

def force_coefficients(acc: tuple[torch.Tensor, torch.Tensor, torch.Tensor], geom: Geometry3D, ops: Ops3D,
                       dt) -> torch.Tensor:
    """Per-body ``[CD, CS, CL, C_roll, Cm, Cn]`` from the weighted penalisation increments.

    ``F = -(h^3 / dt) sum w_b (1/6 D1 + 2/3 D2 + D3)``; moments use the lever
    arms baked into the geometry weights. With a symmetry plane (y = y0) the
    half-domain values are converted to the full body: Fx, Fz, My doubled, Fy =
    0, and Mx = -(yr - y0) Fz, Mz = (yr - y0) Fx (zero when the moment reference
    lies on the plane).
    """
    acc_u, acc_v, acc_w = acc
    scale = -(ops.h ** 3) / dt
    ru = torch.einsum("bkzyx,zyx->bk", geom.k_u, acc_u) * scale       # [Fx, My_u, Mz_u]
    rv = torch.einsum("bkzyx,zyx->bk", geom.k_v, acc_v) * scale       # [Fy, Mx_v, Mz_v]
    rw = torch.einsum("bkzyx,zyx->bk", geom.k_w, acc_w) * scale       # [Fz, Mx_w, My_w]
    fx, fy, fz = ru[:, 0], rv[:, 0], rw[:, 0]
    mx, my, mz = rv[:, 1] + rw[:, 1], ru[:, 1] + rw[:, 2], ru[:, 2] + rv[:, 2]
    if ops.symmetry:
        dy = ops.moment_ref[1] - ops.y0
        fx, fz, my = 2.0 * fx, 2.0 * fz, 2.0 * my
        fy = torch.zeros_like(fy)
        mx = -dy * fz
        mz = dy * fx
    q_a = 0.5 * ops.ref_area
    return torch.stack([fx / q_a, fy / q_a, fz / q_a, mx / (q_a * ops.span_ref), my / (q_a * ops.length_ref),
                        mz / (q_a * ops.span_ref)], dim=1)


# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------

def _vertex_bounds(bodies) -> np.ndarray | None:
    pts = [b.vertices_numpy() for b in bodies]
    if not pts:
        return None
    p = np.vstack(pts)
    return np.stack([p.min(axis=0), p.max(axis=0)])


def init_state3d(cfg: SolverConfig3D, ops: Ops3D, *, perturb: bool | None = None) -> State3D:
    """Impulsive start: the inlet velocity everywhere (v = w = 0), optionally with a
    weak, deterministic perturbation, projected to be discretely divergence-free.

    The perturbation is a Lamb-Oseen vortex with its axis along y (peak swirl
    5 % of U), centred on the bodies and confined spanwise by a Gaussian. It
    breaks the up-down symmetry so an unsteady wake locks in after a few
    convective times, while staying mirror-symmetric about the y = y0 plane
    when that plane is a symmetry plane (so half- and full-domain runs evolve
    identically). It depends only on the body bounding box: reproducible.
    """
    perturb = cfg.perturb if perturb is None else perturb
    dev, dt = ops.device, ops.dtype
    nz, ny, nx = ops.nz, ops.ny, ops.nx
    if cfg.periodic or cfg.domain == "closed":           # no free stream: start from rest
        u = torch.zeros(nz, ny, nx + 1, device=dev, dtype=dt)
    else:
        u = ops.u_inlet.expand(nz, ny, nx + 1).clone()
    v = torch.zeros(nz, ny + 1, nx, device=dev, dtype=dt)
    w = torch.zeros(nz + 1, ny, nx, device=dev, dtype=dt)
    bb = _vertex_bounds(cfg.bodies)
    if perturb and bb is not None and not cfg.periodic and cfg.domain != "closed":
        (xa, ya, za), (xb, yb, zb) = bb
        cx, cz = 0.5 * (xa + xb), 0.5 * (za + zb)
        cy = ops.y0 if ops.symmetry else 0.5 * (ya + yb)
        size = max(zb - za, 0.25 * (xb - xa), 4.0 * ops.h)
        rc = 0.75 * size
        gam = 0.05 * 2.0 * math.pi * rc / 0.638            # peak u_theta of Lamb-Oseen = 0.638 G/(2 pi rc)
        ly = max(0.5 * (yb - ya), rc)
        c = face_coordinates3d(ops)

        def swirl(x, y, z):
            dx, dz = x - cx, z - cz
            r2 = (dx * dx + dz * dz).clamp_min(1e-12)
            f = gam / (2 * math.pi * r2) * (1 - torch.exp(-r2 / rc ** 2)) * torch.exp(-((y - cy) / ly) ** 2)
            return -f * dz, f * dx

        du, _ = swirl(c["xu"][None, None, :], c["yc"][None, :, None], c["zc"][:, None, None])
        _, dw = swirl(c["xc"][None, None, :], c["yc"][None, :, None], c["zw"][:, None, None])
        u = u + du
        w = w + dw
        u, v, w = apply_boundary_values(u, v, w, ops)
        u, v, w, _ = project3d(u, v, w, ops)
    p = torch.zeros(nz, ny, nx, device=dev, dtype=dt)
    zero = torch.zeros((), device=dev, dtype=dt)
    return State3D(u, v, w, p, zero.clone(), zero.clone())


# ---------------------------------------------------------------------------
# the time step
# ---------------------------------------------------------------------------

class StepOut3D(NamedTuple):
    """What :func:`step3d_phys` returns: the new state, per-body coefficients (nbody, 6)
    and the new temperature field (None without heat transfer)."""
    state: State3D
    coeffs: torch.Tensor
    theta: torch.Tensor | None


def step3d_full(state: State3D, geom: Geometry3D, ops: Ops3D, dt=None) -> tuple[State3D, torch.Tensor]:
    """Advance one SSP-RK3 step. Returns the new state and per-body coefficients,
    shape (nbody, 6) = [CD, CS, CL, C_roll, Cm, Cn] (full-body values with a
    symmetry plane).

    ``dt`` may be a float, a 0-d tensor, or None for the CFL / viscous limit
    evaluated on the device from the current state.
    """
    out = step3d_phys(state, geom, ops, dt)
    return out.state, out.coeffs


def step3d_phys(state: State3D, geom: Geometry3D, ops: Ops3D, dt=None, *, phys=None, theta=None,
                trace: list | None = None, diag: bool = False) -> StepOut3D:
    """One SSP-RK3 step with the placed physics of DESIGN.md section 10.

    With ``phys=None`` (and no ``trace``) this executes exactly the operations of
    the v0.2 step -- :func:`step3d_full` is this function -- so the default path
    is unchanged. ``phys`` (a :class:`~.physics.Physics3D`) adds, per stage, the
    momentum sources of fans and ports, the ports' mass sources in the
    projection and the outlet balance, the painted obstacles, the buoyancy of
    ``theta`` (after the penalisation, theory_ns3d.md 15.3), and advances
    ``theta`` (heat transfer) in the same stages. ``trace`` (a list) receives
    one dict per stage with the intermediate staggered fields: the explicit
    tendencies, the stage value before and after penalisation (and buoyancy),
    the projection's right-hand side, psi and the projected velocity (the
    AeroJAX-style solver trace). ``diag`` stores the last stage's
    projection-RHS norm in ``phys.div_pre``.
    """
    u0, v0, w0 = state.u, state.v, state.w
    up, vp, wp = u0, v0, w0
    acc = None
    psi_w = None
    px = ops.periodic_axes[0]
    any_periodic = any(ops.periodic_axes)
    thermal = phys is not None and phys.thermal is not None and theta is not None
    t0 = theta
    tp = theta
    want_nut = thermal and ops.turbulence != "none"
    source = phys.mass_flux if phys is not None else None
    for s, ((a, b), wgt) in enumerate(zip(SSP_RK3, FORCE_WEIGHTS)):
        extra = {} if want_nut else None
        if phys is None:
            lu, lv, lw, nut_max = momentum_rhs3d(up, vp, wp, geom, ops)
        else:
            if s == 0 and dt is None:
                # the dt rule and the fan controllers act before any source is evaluated
                lu, lv, lw, nut_max = momentum_rhs3d(up, vp, wp, geom, ops, out=extra)
                dt = (ops.extras["dt_fixed"] if ops.dt_fixed is not None
                      else phys.dt_rule(u0, v0, w0, ops, nut_max, theta=t0))
                if not torch.is_tensor(dt):
                    dt = torch.tensor(float(dt), device=ops.device, dtype=ops.dtype)
                phys.begin_step(u0, v0, w0, dt)
                lu, lv, lw = phys.momentum_sources(lu, lv, lw, up, vp, wp, tp, geom)
                lu, lv, lw = boundary_tendencies(up, lu, lv, lw, ops)
            else:
                if s == 0:
                    if not torch.is_tensor(dt):
                        dt = torch.tensor(float(dt), device=ops.device, dtype=ops.dtype)
                    phys.begin_step(u0, v0, w0, dt)
                lu, lv, lw, nut_max = momentum_rhs3d(up, vp, wp, geom, ops, phys=phys, theta=tp, out=extra)
        if s == 0:
            if dt is None:
                dt = ops.extras["dt_fixed"] if ops.dt_fixed is not None else cfl_dt3d(u0, v0, w0, ops, nut_max)
            if not torch.is_tensor(dt):
                dt = torch.tensor(float(dt), device=ops.device, dtype=ops.dtype)
        bdt = dt * b
        if a == 0.0:
            uh, vh, wh = torch.addcmul(up, lu, bdt), torch.addcmul(vp, lv, bdt), torch.addcmul(wp, lw, bdt)
        else:
            uh = torch.addcmul(torch.lerp(up, u0, a), lu, bdt)
            vh = torch.addcmul(torch.lerp(vp, v0, a), lv, bdt)
            wh = torch.addcmul(torch.lerp(wp, w0, a), lw, bdt)
        if not px:
            uh = inlet_outlet(uh, ops) if source is None else inlet_outlet(uh, ops, source)
        rec = None
        if trace is not None:
            rec = {"stage": s, "a": a, "b": b, "dt": dt.detach().clone(),
                   "tendency": (lu.detach().clone(), lv.detach().clone(), lw.detach().clone()),
                   "u_star": (uh.detach().clone(), vh.detach().clone(), wh.detach().clone())}
        if not geom.empty:
            uh, vh, wh, du, dv, dw = penalise3d(uh, vh, wh, geom, ops, bdt)
            acc = (du * wgt, dv * wgt, dw * wgt) if acc is None else (
                acc[0] + du * wgt, acc[1] + dv * wgt, acc[2] + dw * wgt)
            if any_periodic:
                uh, vh, wh = sync_periodic(uh, vh, wh, ops)
        if phys is not None and phys.paint is not None:
            phys.penalise(uh, vh, wh, bdt, ops)
            if any_periodic:
                uh, vh, wh = sync_periodic(uh, vh, wh, ops)
        if thermal and phys.buoyant:
            # after the penalisation, everywhere (physics.ThermalOps.add_buoyancy); the solid's
            # share leaves the body force
            wh, share = phys.buoyancy(wh, tp, bdt, wgt, geom, ops)
            if share is not None:
                acc = (acc[0], acc[1], acc[2] + share)
        if thermal:
            tp_new = phys.theta_stage(tp, t0, up, vp, wp, a, bdt, wgt, ops,
                                      nut=extra.get("nut") if extra else None)
        if phys is None and trace is None:
            up, vp, wp, psi = project3d(uh, vh, wh, ops)
        else:
            div = divergence3d(uh, vh, wh, ops.h)
            if phys is not None:
                div = phys.continuity(div)
                if diag and s == 2:
                    phys.record_divergence(div)
            psi = ops.poisson.solve(div)
            gx, gy, gz = pressure_gradient(psi, ops)
            if rec is not None:
                rec["u_penalised"] = (uh.detach().clone(), vh.detach().clone(), wh.detach().clone())
                rec["rhs"] = div.detach().clone()
                rec["psi"] = psi.detach().clone()
            up, vp, wp = uh - gx, vh - gy, wh - gz
            if rec is not None:
                rec["u_projected"] = (up.detach().clone(), vp.detach().clone(), wp.detach().clone())
                if thermal:
                    rec["theta"] = tp_new.detach().clone()
                trace.append(rec)
        if thermal:
            tp = tp_new
        psi_w = psi * wgt if psi_w is None else torch.add(psi_w, psi, alpha=wgt)
    p = psi_w * (1.0 / dt)                 # step-averaged pressure (theory_ns2d.md 4)
    if geom.empty:
        coeffs = torch.zeros(geom.nbody, 6, device=ops.device, dtype=ops.dtype)
    else:
        coeffs = force_coefficients(acc, geom, ops, dt)
    if phys is not None:
        phys.end_step(dt, ops)
    return StepOut3D(State3D(up, vp, wp, p, state.t + dt, dt * torch.ones_like(state.t)), coeffs,
                     tp if thermal else None)


def step3d(state: State3D, geom: Geometry3D, ops: Ops3D, dt=None) -> tuple[State3D, torch.Tensor]:
    """One time step; returns the new state and the total coefficients over all
    bodies (a differentiable tensor of shape (6,))."""
    new, coeffs = step3d_full(state, geom, ops, dt)
    return new, coeffs.sum(0)


__all__ = [
    "COEFF_NAMES",
    "Ops3D",
    "State3D",
    "StepOut3D",
    "advection3",
    "apply_boundary_values",
    "auto_eta",
    "build_geometry3d",
    "cfl_dt3d",
    "divergence3d",
    "eddy_viscosity3d",
    "face_coordinates3d",
    "force_coefficients",
    "init_state3d",
    "make_operators3d",
    "momentum_rhs3d",
    "pad3",
    "pad_velocity",
    "project3d",
    "set_reynolds",
    "step3d",
    "step3d_full",
    "step3d_phys",
    "traceless",
    "velocity_gradient",
]
