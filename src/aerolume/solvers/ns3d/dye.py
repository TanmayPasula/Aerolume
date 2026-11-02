"""Passive smoke ("dye") for the 3D solver: emitters, bounded advection, sources.

Users asked for smoke *in the space around the body*, the way a wind tunnel
shows it: streams released from nozzles upstream that wrap around the model and
break up in its wake (DESIGN.md 7.13). This module transports a passive scalar
``c`` (0 = clean air, 1 = undiluted smoke) with the solver's velocity:

    dc/dt + div(u c) = 0                         (no diffusion beyond the numerics)

* **Where it lives.** ``c`` is cell-centred on the MAC grid, so the face
  velocities the solver already has are exactly the advecting velocities of a
  finite-volume flux: no interpolation, and with the discretely divergence-free
  velocity after the projection a uniform ``c`` stays uniform to round-off.
* **Scheme.** MUSCL reconstruction with the van Leer limiter and one forward
  Euler step per solver step, using the end-of-step velocity -- the 2D solver's
  dye (``ns2d.core.advect_dye``) carried to three dimensions. Second order in
  space where the field is smooth, first order at extrema (no new maxima). A
  forward Euler step with a limited slope is total-variation diminishing when
  the sum of the face Courant numbers is at most 1 / (1 + psi_max / 2) = 1/2 for
  van Leer (psi_max = 2); the solver's time step is ``cfl h / (max|u| + max|v| +
  max|w|)`` with cfl = 0.5, which is exactly that bound. Smoke is a picture,
  not a measurement, so first order in time costs nothing visible, and it keeps
  the cost to one flux pass per direction plus one update (theory_ns3d.md 14).
* **Bounds.** Every update is clipped to [0, 1] (the unsplit 3D TVD limit is
  sharp, so rare round-off excursions are removed rather than allowed to grow)
  and values below :data:`DYE_FLOOR` -- invisible -- are set to exactly zero:
  the limiter otherwise lets far tails decay into subnormal floats, which run
  ~100x slower on x86 CPUs.
* **Solids.** Cells whose solid fraction exceeds one half are set to zero after
  every step, so ``field("dye")`` is zero inside bodies. The Brinkman velocity
  there is ~0, so almost nothing flows in: this is a clean-up of what numerical
  spreading puts into the diffuse interface, not a sink that eats smoke
  hugging the surface.
* **Boundaries.** Clean air enters at the inlet (zero ghost cells), the outlet
  is zero-gradient, walls, the road and the symmetry plane mirror the field
  (zero normal velocity makes their flux zero anyway; the mirror only keeps the
  limiter's slopes symmetric, so a half-domain run equals the full one),
  periodic axes wrap.
* **Emitters** (:class:`DyeRake`): nozzles upstream of the body -- a row across
  the span ("rake"), a span x height grid ("grid") or a full band ("plane").
  After each step the smoke inside a nozzle is relaxed to 1 with a smooth
  (anti-aliased) radial profile, so a nozzle between cell centres still emits
  and moving it by a fraction of a cell moves the stream. Optional on/off
  pulses cut the streams into time lines.

The fused CUDA kernels of the same arithmetic are in :mod:`.fused`
(``dye_flux`` / ``dye_update``); :class:`~aerolume.solvers.ns3d.FlowSolver3D`
runs this inside its CUDA graphs. The dye is solver state, not part of the
differentiable :class:`~aerolume.solvers.ns3d.core.State3D`: it never feeds
back into the flow, so shape sensitivities never need it.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

from .config import DYE_KINDS, DyeRake

#: smoke concentrations below this are invisible and set to exactly zero (see the module docstring)
DYE_FLOOR = 1e-8
#: default nozzle radius in cells: the stream is ~2.5 cells across, the thinnest the
#: limiter carries over a body length without smearing into a haze at Draft grids
NOZZLE_RADIUS_CELLS = 1.25
#: half-thickness of the emitting slab along x, in cells (a nozzle is a short tube)
NOZZLE_DEPTH_CELLS = 1.0


def default_dye_rake(bounds, extent, h: float, *, kind: str = "rake", symmetry: bool = False,
                     ground: bool = False, n: int | None = None) -> DyeRake:
    """Emitters chosen from the bodies' bounding box (solver units), as a tunnel engineer would place them.

    The release plane sits 0.4 of the body's smaller dimension (length or
    cross-stream size) upstream of its nose (close enough that numerical spreading over the approach stays
    small, far enough that the stagnation region does not push smoke back into
    the nozzles), at least four cells from the inlet. "rake": one row across the
    span at 40 % of the body height, 20 % wider than the body on each side --
    the classic smoke-rake photograph: the streams split at the nose, over the
    top, round the sides and under the body, and roll up in the wake; "grid": the frontal box (10 % margins) in a span x
    height grid; "plane": a band over the same box. With a symmetry plane only
    nozzles on the solved side (y >= y0) are kept; one on the plane is half a
    nozzle, mirrored by the boundary condition. On a road the lowest nozzle stays
    two cells above it.
    """
    bx0, bx1, by0, by1, bz0, bz1 = (float(b) for b in bounds)
    ex0, _, ey0, ey1, ez0, ez1 = (float(e) for e in extent)
    wy, hz = max(by1 - by0, h), max(bz1 - bz0, h)
    size = max(wy, hz)
    radius = NOZZLE_RADIUS_CELLS * h
    x = bx0 - max(0.4 * min(size, max(bx1 - bx0, h)), 4.0 * h)
    x = min(max(x, ex0 + 4.0 * h), bx0 - 2.0 * h)
    # the span of the whole body: a half-domain run holds the full meshes, so these bounds
    # are the mirrored body's and the row below is the same as in a full-domain run
    ya, yb = by0 - 0.2 * wy, by1 + 0.2 * wy
    if symmetry:
        ya = max(ya, 2.0 * ey0 - (ey1 - radius - h))
    else:
        ya = max(ya, ey0 + radius + h)
    yb = min(yb, ey1 - radius - h)
    za, zb = bz0 + 0.1 * hz, bz1 + 0.1 * hz
    if ground:
        za = max(za, ez0 + 2.0 * h + radius)
    za, zb = max(za, ez0 + radius + h), min(zb, ez1 - radius - h)
    if kind == "plane":
        return DyeRake("plane", [(x, max(ya, ey0) if symmetry else ya, za), (x, yb, zb)], radius=radius)
    # spacing: about six nozzle radii between centres (at least 4.8 after rounding the count), so the
    # streams stay distinct over a body length
    gap = max(6.0 * radius, size / 7.0)
    if kind == "rake":
        k = int(n) if n else max(3, min(17, round((yb - ya) / gap) + 1))
        ys = _solved_side(np.linspace(ya, yb, k), symmetry, ey0)
        z = bz0 + 0.40 * hz
        z = min(max(z, ez0 + radius + 2 * h), ez1 - radius - h)
        return DyeRake("rake", [(x, y, z) for y in ys], radius=radius)
    if kind == "grid":
        ky = max(3, min(11, round((yb - ya) / gap) + 1))
        kz = max(2, min(9, round((zb - za) / gap) + 1))
        if n:
            ky = max(2, round(math.sqrt(n * (yb - ya) / max(zb - za, 1e-9))))
            kz = max(2, round(n / ky))
        ys = _solved_side(np.linspace(ya, yb, ky), symmetry, ey0)
        zs = np.linspace(za, zb, kz)
        return DyeRake("grid", [(x, y, z) for z in zs for y in ys], radius=radius)
    raise ValueError(f"unknown dye rake kind {kind!r}; use one of {', '.join(DYE_KINDS)}")


def _solved_side(ys: np.ndarray, symmetry: bool, y0: float) -> np.ndarray:
    """With a symmetry plane only nozzles on the solved side (y >= y0) are emitted;
    the boundary condition mirrors them (and completes a nozzle on the plane)."""
    return ys[ys >= y0 - 1e-9] if symmetry else ys


# ---------------------------------------------------------------------------
# emitter weights on the grid
# ---------------------------------------------------------------------------

class DyeSources(NamedTuple):
    """Emitter weights on a window of cells (k0, k1, j0, j1, i0, i1) and the pulse
    parameters as a device tensor [t_on, period] (continuous: t_on > period)."""
    window: tuple[int, int, int, int, int, int]
    weight: torch.Tensor
    pulse: torch.Tensor

    @property
    def empty(self) -> bool:
        k0, k1, j0, j1, i0, i1 = self.window
        return k1 <= k0 or j1 <= j0 or i1 <= i0


def _ramp(d: np.ndarray, edge: float, h: float) -> np.ndarray:
    """1 inside ``edge`` - h/2, 0 beyond ``edge`` + h/2, linear between: an
    anti-aliased edge, so sub-cell moves of an emitter change what it emits."""
    return np.clip((edge + 0.5 * h - d) / h, 0.0, 1.0)


def source_weights(rake: DyeRake, *, nx: int, ny: int, nz: int, h: float, x0: float, y0: float, z0: float
                   ) -> tuple[tuple[int, int, int, int, int, int], np.ndarray]:
    """(window, weights) of a rake on a grid: weights in [0, 1] on the window's cells (numpy)."""
    pos = np.asarray(rake.positions, dtype=np.float64).reshape(-1, 3)
    if not len(pos):
        return (0, 0, 0, 0, 0, 0), np.zeros((0, 0, 0))
    r = rake.radius if rake.radius > 0 else NOZZLE_RADIUS_CELLS * h
    depth = NOZZLE_DEPTH_CELLS * h
    if rake.kind == "plane":
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        lo = lo - np.array([depth, 0.0, 0.0])
        hi = hi + np.array([depth, 0.0, 0.0])
        pad = np.array([h, h, h])
    else:
        lo, hi = pos.min(axis=0) - np.array([depth, r, r]), pos.max(axis=0) + np.array([depth, r, r])
        pad = np.array([h, h, h])
    lo, hi = lo - pad, hi + pad
    i0, j0, k0 = (math.floor((lo[a] - o) / h) for a, o in ((0, x0), (1, y0), (2, z0)))
    i1, j1, k1 = (math.ceil((hi[a] - o) / h) for a, o in ((0, x0), (1, y0), (2, z0)))
    i0, i1 = max(i0, 0), min(i1, nx)
    j0, j1 = max(j0, 0), min(j1, ny)
    k0, k1 = max(k0, 0), min(k1, nz)
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return (0, 0, 0, 0, 0, 0), np.zeros((0, 0, 0))
    xc = x0 + (np.arange(i0, i1) + 0.5) * h
    yc = y0 + (np.arange(j0, j1) + 0.5) * h
    zc = z0 + (np.arange(k0, k1) + 0.5) * h
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
    if rake.kind == "plane":
        (_, ya, za), (_, yb, zb) = pos[0], pos[1]
        ya, yb = min(ya, yb), max(ya, yb)
        za, zb = min(za, zb), max(za, zb)
        wy = _ramp(np.maximum(ya - Y, Y - yb), 0.0, h)
        wz = _ramp(np.maximum(za - Z, Z - zb), 0.0, h)
        wx = _ramp(np.abs(X - 0.5 * (pos[0, 0] + pos[1, 0])), depth, h)
        w = wx * wy * wz
    else:
        w = np.zeros_like(X)
        for px, py, pz in pos:
            rr = np.sqrt((Y - py) ** 2 + (Z - pz) ** 2)
            w = np.maximum(w, _ramp(rr, r, h) * _ramp(np.abs(X - px), depth, h))
    if not w.any():
        return (0, 0, 0, 0, 0, 0), np.zeros((0, 0, 0))
    return (k0, k1, j0, j1, i0, i1), w


def build_dye_sources(rake: DyeRake | None, ops) -> DyeSources:
    """Emitter weights of ``rake`` on the solver grid (an empty window when it misses the domain)."""
    dev, dt = ops.device, ops.dtype
    if rake is None:
        win, w = (0, 0, 0, 0, 0, 0), np.zeros((0, 0, 0))
    else:
        win, w = source_weights(rake, nx=ops.nx, ny=ops.ny, nz=ops.nz, h=ops.h, x0=ops.x0, y0=ops.y0, z0=ops.z0)
    pulse = pulse_tensor(rake, dev, dt)
    return DyeSources(win, torch.as_tensor(w, device=dev, dtype=dt), pulse)


def pulse_tensor(rake: DyeRake | None, device, dtype) -> torch.Tensor:
    """[t_on, period] on the device; a continuous rake gets t_on > period (always on).

    The pulse lives in a tensor rather than in Python floats so a captured CUDA
    graph reads the current values: changing the pulses never re-captures.
    """
    if rake is not None and rake.pulsed:
        vals = [rake.pulse_on, rake.pulse_on + rake.pulse_off]
    else:
        vals = [2.0, 1.0]
    return torch.tensor(vals, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# advection (reference implementation; the fused kernels compute the same arithmetic)
# ---------------------------------------------------------------------------

def pad_dye(c: torch.Tensor, ops) -> torch.Tensor:
    """``c`` with two ghost layers on every face: (nz+4, ny+4, nx+4).

    x: clean inflow (zeros) at the inlet, zero gradient at the outlet; y and z:
    mirrored (walls, road, symmetry plane: zero normal velocity, so the mirror
    only shapes the limiter's slopes); periodic axes wrap. Edge and corner
    ghosts are not filled: the face stencils read along one axis at a time.
    """
    nz, ny, nx = c.shape
    P = c.new_empty(nz + 4, ny + 4, nx + 4)
    P[2:-2, 2:-2, 2:-2] = c
    px, py, pz = ops.periodic_axes
    m = slice(2, -2)
    if px:
        P[m, m, 0:2] = c[..., nx - 2:]
        P[m, m, nx + 2:] = c[..., :2]
    else:
        P[m, m, 0:2] = 0.0
        P[m, m, nx + 2:] = c[..., nx - 1:].expand(nz, ny, 2)
    if py:
        P[m, 0:2, m] = c[:, ny - 2:]
        P[m, ny + 2:, m] = c[:, :2]
    else:
        P[m, 0:2, m] = c[:, :2].flip(1)
        P[m, ny + 2:, m] = c[:, ny - 2:].flip(1)
    if pz:
        P[0:2, m, m] = c[nz - 2:]
        P[nz + 2:, m, m] = c[:2]
    else:
        P[0:2, m, m] = c[:2].flip(0)
        P[nz + 2:, m, m] = c[nz - 2:].flip(0)
    return P


def _van_leer(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Van Leer limited slope: the harmonic mean of the one-sided differences, 0 at extrema."""
    pq = p * q
    pos = pq > 0
    return torch.where(pos, 2.0 * pq / torch.where(pos, p + q, torch.ones_like(p)), torch.zeros_like(p))


def face_flux(a, b, c, d, vel):
    """Upwind MUSCL flux through the face between cells b and c (neighbours a | d)."""
    up = b + 0.5 * _van_leer(b - a, c - b)
    dn = c - 0.5 * _van_leer(c - b, d - c)
    return vel * torch.where(vel > 0, up, dn)


def dye_fluxes(c: torch.Tensor, u, v, w, ops):
    """(Fx, Fy, Fz): MUSCL fluxes on all x-, y- and z-faces (the shapes of u, v, w)."""
    nz, ny, nx = c.shape
    P = pad_dye(c, ops)
    m = slice(2, -2)
    fx = face_flux(P[m, m, 0:nx + 1], P[m, m, 1:nx + 2], P[m, m, 2:nx + 3], P[m, m, 3:nx + 4], u)
    fy = face_flux(P[m, 0:ny + 1, m], P[m, 1:ny + 2, m], P[m, 2:ny + 3, m], P[m, 3:ny + 4, m], v)
    fz = face_flux(P[0:nz + 1, m, m], P[1:nz + 2, m, m], P[2:nz + 3, m, m], P[3:nz + 4, m, m], w)
    return fx, fy, fz


def advect_dye3d(c: torch.Tensor, u, v, w, dt, ops) -> torch.Tensor:
    """One forward Euler step of the smoke with van Leer MUSCL fluxes, clipped to [0, 1]
    and floored (values below :data:`DYE_FLOOR` become 0). Returns a new tensor."""
    fx, fy, fz = dye_fluxes(c, u, v, w, ops)
    k = dt * (1.0 / ops.h)
    r = c - k * ((fx[..., 1:] - fx[..., :-1]) + (fy[:, 1:] - fy[:, :-1]) + (fz[1:] - fz[:-1]))
    r = r.clamp(max=1.0)
    return torch.where(r < DYE_FLOOR, torch.zeros_like(r), r)


def apply_solids_and_sources(c: torch.Tensor, t, geom, src: DyeSources | None) -> None:
    """In place: smoke set to zero in solid cells (solid fraction > 1/2) and relaxed to 1
    inside the nozzles (``c += s (1 - c)``, s the nozzle weight times the pulse switch)."""
    if geom is not None and not geom.empty:
        k0, k1, j0, j1, i0, i1 = geom.window
        win = c[k0:k1, j0:j1, i0:i1]
        win.masked_fill_(geom.chi_c > 0.5, 0.0)
    if src is not None and not src.empty:
        k0, k1, j0, j1, i0, i1 = src.window
        win = c[k0:k1, j0:j1, i0:i1]
        on = (torch.remainder(t, src.pulse[1]) < src.pulse[0]).to(c.dtype)
        win.add_(src.weight * on * (1.0 - win))


__all__ = [
    "DYE_FLOOR",
    "DYE_KINDS",
    "NOZZLE_RADIUS_CELLS",
    "DyeRake",
    "DyeSources",
    "advect_dye3d",
    "apply_solids_and_sources",
    "build_dye_sources",
    "default_dye_rake",
    "dye_fluxes",
    "face_flux",
    "pad_dye",
    "pulse_tensor",
    "source_weights",
]
