"""Placed physics of the 3D solver (DESIGN.md section 10): heat transfer with
Boussinesq buoyancy, heated bodies, heat sources, actuator-disc fans, jets and
vents, painted obstacles and the PID time-step controller.

Everything here is optional. A :class:`Physics3D` is built by
:class:`~aerolume.solvers.ns3d.FlowSolver3D` only when the configuration asks
for one of these features; without it the step functions run the v0.2 solver
unchanged (every hook is a Python ``if phys is not None``), so the validated
default path neither changes nor slows down.

Heat transfer (``docs/theory_ns3d.md`` 15)
------------------------------------------
theta = (T - T_amb) / dT_ref is cell-centred and advanced inside the same
SSP-RK3 stages as the momentum, so buoyancy and transport are coupled at the
scheme's order::

    d theta/dt + div(u theta) = div((1/Pe + nu_t/Pr_t) grad theta) + q
    du/dt + ...               = ... + Ri theta e_z          (gravity along -z)

* **Advection** is the smoke's MUSCL / van Leer flux (bounded, second order
  where smooth) with the stage's discretely divergence-free face velocity;
  **diffusion** the 7-point flux form, so the scheme is conservative: the heat
  that enters through walls, bodies and sources is accounted for exactly. The
  time step obeys the combined bound C / (1/2) + 6 D <= 1 (:meth:`ThermalOps.dt_limit`),
  which makes every stage a convex combination: theta obeys a discrete maximum
  principle away from sources.
* **Domain walls** enter through ghost cells: Dirichlet ``2 theta_w - theta``
  (temperature), mirror (adiabatic, symmetry plane), the mirror plus ``G h``
  (first layer) and ``3 G h`` (second) for a heat flux G in conduction units, so
  that ``-d theta/dn = G`` at the wall, the inflow value at a tunnel inlet, zero
  gradient at the outlet.
* **Heated bodies**: "temperature" bodies are penalised implicitly towards
  their temperature with the momentum's mask, wall offset and stiffness rule
  (Pe in place of Re, ``eta_T = min(0.05 h, h^2 Pe / 60)``): the heat the
  penalisation adds is the body's heat-transfer rate, exactly as its momentum
  impulse is its force. "adiabatic" bodies block conduction: the diffusive
  flux through a face is multiplied by (1 - chi) there. "heat_flux" bodies
  release their flux in a one-mask-width skin centred on the effective wall,
  normalised so the total is exactly flux x area.
* **Buoyancy** ``Ri theta`` acts on the w-faces everywhere, added to each stage
  value after the penalisation (like the pressure that balances most of it), so
  a stratified fluid at rest around a body is an exact discrete equilibrium; the
  buoyancy of the fictitious fluid inside a solid is taken out of the body's
  force (F = penalisation impulse - Ri integral(chi theta)), which leaves the
  Archimedes force of the surrounding fluid in it.

Fans and ports
--------------
* A fan is an actuator disc: a uniform force density over a disc of thickness
  t, normalised separately on the u-, v- and w-faces so the force adds up to
  exactly ``T n``. Mode "velocity" regulates T with a PI controller on the
  measured disc-averaged axial velocity, inside the step (no host sync).
* A jet is a mass source ``m = Q phi`` carrying momentum ``m U_j n``,
  temperature ``m theta_j`` and smoke; a vent a sink ``m < 0`` carrying the
  local velocity, temperature and smoke. The projection solves ``lap psi =
  div u_hat - m``; a tunnel's outlet carries the inflow plus ``sum m``.

Footprints are anti-aliased (one-cell linear ramps) and normalised over the
*whole* shape before the domain clips it, so a fan or jet centred on a symmetry
plane puts exactly half of itself into a half-domain run, as the mirror image
requires. Faces on non-periodic domain boundaries never carry a source (they
carry boundary conditions).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..ns2d.core import ETA_PER_H, STIFFNESS_MIN
from .config import WALL_SIDES, Fan3D, HeatSource3D, Port3D, SolverConfig3D
from .geometry import grid_signed_distance, smooth_step

#: PID gains of the adaptive time step on the log of (target dt / current dt): integral,
#: proportional, derivative (Soederlind-style digital controller; integral 1 = jump to the target)
PID_GAINS = (0.3, 0.4, 0.05)
#: largest growth / smallest shrink factor of the PID-controlled dt per step
PID_LIMITS = (0.5, 1.2)
#: footprints below this fraction of their total are dropped (an emitter or fan outside the domain)
_EMPTY = 1e-9
#: the van Leer MUSCL step's TVD bound on the Courant sum (theory_ns3d.md 14, 15.2)
THETA_COURANT_MAX = 0.5
#: (max|u| + max|v| + max|w|) of a buoyancy-dominated heated body, in units of the buoyant
#: velocity sqrt(g beta dT L) = U sqrt(Ri): the speed sum the step's CFL / heat-equation
#: limits see once the plume has formed. Measured on the heated sphere (theory_ns3d.md 15.8,
#: 2026-10): 2.5 at Ri 10, 3.6 at Ri 100, 3.9-4.6 at Ri 627 (Re 20-100, 8-16 cells/D). With
#: Ri >> 1 the free stream no longer sets the step: a Re 100 sphere at Ri 627 steps ~10x
#: shorter than the free-fall bound sqrt(h / Ri) suggests.
PLUME_SPEED_FACTOR = 4.0


def planning_speed_sum(ri: float, forced: float) -> float:
    """The speed sum (max|u| + max|v| + max|w|) / U to plan a run's time step with:
    ``forced`` (the free-stream case's factor) or, when buoyancy drives faster flow,
    ``PLUME_SPEED_FACTOR * sqrt(Ri)`` -- the measured plume speeds, whichever is larger.
    Within a factor 1.6 of the measured late-time sums from Ri 1 to 627 (theory_ns3d.md 15.8)."""
    ri = abs(float(ri))
    return max(float(forced), PLUME_SPEED_FACTOR * math.sqrt(ri))


def thermal_dt_estimate(h: float, speed_sum: float, pe: float, *, nut: float = 0.0, prt: float = 0.85) -> float:
    """The heat equation's step limit (:meth:`ThermalOps.dt_limit`) for a planned speed sum: the
    step a heat-transfer run takes in its developed state, ``1 / (S / (h / 2) + 6 kappa / h^2)``
    with kappa = 1 / Pe (+ nu_t / Pr_t). Planners use it with :func:`planning_speed_sum`."""
    from .core import VISCOUS_DT
    kap = 1.0 / float(pe) + float(nut) / float(prt)
    return 1.0 / (float(speed_sum) / (THETA_COURANT_MAX * h) + kap / (VISCOUS_DT * h * h))


# ---------------------------------------------------------------------------
# footprints on the staggered grid
# ---------------------------------------------------------------------------

def ramp(d, edge: float, h: float):
    """1 inside ``edge - h/2``, 0 beyond ``edge + h/2``, linear between (anti-aliased)."""
    return np.clip((edge + 0.5 * h - d) / h, 0.0, 1.0)


def _axes_of(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing ``n`` to a right-handed orthonormal frame (e1, e2, n)."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def grid_window(lo, hi, ops, margin: float = 2.0, *, clip: bool = True) -> tuple[int, int, int, int, int, int]:
    """Cell window (k0, k1, j0, j1, i0, i1) covering the box [lo, hi] plus ``margin`` cells,
    clipped to the domain (``clip``) or not (it may then reach outside)."""
    h = ops.h
    i0 = math.floor((lo[0] - ops.x0) / h - margin)
    i1 = math.ceil((hi[0] - ops.x0) / h + margin)
    j0 = math.floor((lo[1] - ops.y0) / h - margin)
    j1 = math.ceil((hi[1] - ops.y0) / h + margin)
    k0 = math.floor((lo[2] - ops.z0) / h - margin)
    k1 = math.ceil((hi[2] - ops.z0) / h + margin)
    if clip:
        i0, i1 = max(i0, 0), min(i1, ops.nx)
        j0, j1 = max(j0, 0), min(j1, ops.ny)
        k0, k1 = max(k0, 0), min(k1, ops.nz)
    return (k0, k1, j0, j1, i0, i1)


def window_empty(win) -> bool:
    k0, k1, j0, j1, i0, i1 = win
    return k1 <= k0 or j1 <= j0 or i1 <= i0


def grid_coordinates(win, ops, grid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcastable (Z, Y, X) coordinates of the entries of ``grid`` ("c" centres, "u" /
    "v" / "w" faces) over a cell window: x-faces i0..i1 for "u", and so on."""
    k0, k1, j0, j1, i0, i1 = win
    h = ops.h
    xs = ops.x0 + (np.arange(i0, i1 + 1) * h if grid == "u" else (np.arange(i0, i1) + 0.5) * h)
    ys = ops.y0 + (np.arange(j0, j1 + 1) * h if grid == "v" else (np.arange(j0, j1) + 0.5) * h)
    zs = ops.z0 + (np.arange(k0, k1 + 1) * h if grid == "w" else (np.arange(k0, k1) + 0.5) * h)
    return zs[:, None, None], ys[None, :, None], xs[None, None, :]


def _crop(arr: np.ndarray, full, win, grid: str) -> np.ndarray:
    """The part of ``arr`` (on the unclipped window ``full``) that lies on ``win``."""
    fk0, _, fj0, _, fi0, _ = full
    k0, k1, j0, j1, i0, i1 = win
    ex = (1 if grid == "w" else 0, 1 if grid == "v" else 0, 1 if grid == "u" else 0)
    return arr[k0 - fk0:k1 - fk0 + ex[0], j0 - fj0:j1 - fj0 + ex[1], i0 - fi0:i1 - fi0 + ex[2]]


def _zero_boundary_faces(arr: np.ndarray, win, ops, grid: str) -> np.ndarray:
    """Faces on non-periodic domain boundaries carry boundary conditions, never sources."""
    k0, k1, j0, j1, i0, i1 = win
    px, py, pz = ops.periodic_axes
    arr = arr.copy()
    if grid == "u" and not px:
        if i0 == 0:
            arr[..., 0] = 0.0
        if i1 == ops.nx:
            arr[..., -1] = 0.0
    if grid == "v" and not py:
        if j0 == 0:
            arr[:, 0] = 0.0
        if j1 == ops.ny:
            arr[:, -1] = 0.0
    if grid == "w" and not pz:
        if k0 == 0:
            arr[0] = 0.0
        if k1 == ops.nz:
            arr[-1] = 0.0
    return arr


class Footprint:
    """A smooth shape evaluated on a cell window of the staggered grid.

    ``values[g]`` (numpy) are the shape's weights on grid ``g`` over the clipped
    window ``win``; ``totals[g]`` the sum of the weights over the *whole* shape
    (also the part outside the domain), so ``values / (totals h^3)`` integrates
    to 1 over the shape and to the in-domain fraction ``fractions[g]`` over the
    domain.
    """

    def __init__(self, fn, lo, hi, ops, grids: Sequence[str]):
        full = grid_window(lo, hi, ops, clip=False)
        self.win = grid_window(lo, hi, ops, clip=True)
        self.values: dict[str, np.ndarray] = {}
        self.totals: dict[str, float] = {}
        self.fractions: dict[str, float] = {}
        for g in grids:
            Z, Y, X = grid_coordinates(full, ops, g)
            w = np.broadcast_to(fn(X, Y, Z), np.broadcast_shapes(Z.shape, Y.shape, X.shape)).astype(np.float64)
            tot = float(w.sum())
            if window_empty(self.win) or tot <= 0:
                self.values[g] = np.zeros((0, 0, 0))
                self.totals[g], self.fractions[g] = max(tot, 0.0), 0.0
                continue
            part = _zero_boundary_faces(_crop(w, full, self.win, g), self.win, ops, g)
            self.values[g] = part
            self.totals[g] = tot
            self.fractions[g] = float(part.sum()) / tot
        self.empty = window_empty(self.win) or all(f <= _EMPTY for f in self.fractions.values())

    def normalised(self, g: str, h: float) -> np.ndarray:
        """Weights that integrate to 1 over the whole shape (h^3 sum)."""
        return self.values[g] / (self.totals[g] * h ** 3) if self.totals[g] > 0 else self.values[g]

    def slices(self, g: str) -> tuple[slice, slice, slice]:
        """Index slices of grid ``g`` arrays (full domain shapes) covered by the window."""
        return face_slices(self.win, g)


def face_slices(win, g: str) -> tuple[slice, slice, slice]:
    """Slices of a full-domain array of grid ``g`` ("c", "u", "v", "w") over cell window ``win``."""
    k0, k1, j0, j1, i0, i1 = win
    return (slice(k0, k1 + (1 if g == "w" else 0)), slice(j0, j1 + (1 if g == "v" else 0)),
            slice(i0, i1 + (1 if g == "u" else 0)))


def disc_shape(center, normal, radius: float, thickness: float, h: float):
    """Footprint function of a disc (radius, thickness) facing ``normal`` and its bounding box."""
    c = np.asarray(center, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    half = 0.5 * thickness

    def fn(X, Y, Z):
        dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
        s = dx * n[0] + dy * n[1] + dz * n[2]
        r = np.sqrt(np.maximum(dx * dx + dy * dy + dz * dz - s * s, 0.0))
        return ramp(r, radius, h) * ramp(np.abs(s), half, h)
    ext = radius + half + h
    return fn, c - ext, c + ext


def box_shape(center, size, h: float, axes=None):
    """Profile of a box (ramped one cell at its faces) and its bounding box. ``axes``: a 3x3
    rotation whose columns are the box's own axes (None = the grid's); the bounding box then
    covers the turned box (half-extents ``|R| hs``)."""
    c = np.asarray(center, dtype=np.float64)
    hs = 0.5 * np.asarray(size, dtype=np.float64)
    if axes is None:
        def fn(X, Y, Z):
            return ramp(np.abs(X - c[0]), hs[0], h) * ramp(np.abs(Y - c[1]), hs[1], h) * ramp(np.abs(Z - c[2]), hs[2], h)
        return fn, c - hs - h, c + hs + h
    r = np.asarray(axes, dtype=np.float64).reshape(3, 3)

    def fn(X, Y, Z):
        dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
        out = 1.0
        for i in range(3):
            out = out * ramp(np.abs(dx * r[0, i] + dy * r[1, i] + dz * r[2, i]), hs[i], h)
        return out
    ext = np.abs(r) @ hs
    return fn, c - ext - h, c + ext + h


def sphere_shape(center, radius: float, h: float):
    c = np.asarray(center, dtype=np.float64)

    def fn(X, Y, Z):
        return ramp(np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2), radius, h)
    return fn, c - radius - h, c + radius + h


# ---------------------------------------------------------------------------
# heat transfer
# ---------------------------------------------------------------------------

def auto_eta_thermal(h: float, pe: float) -> float:
    """Thermal penalisation permeability: the momentum rule (theory_ns2d.md 6.3) with Pe for
    Re, so the thermal stiffness h^2 Pe / eta never falls below 60 and the effective wall of a
    heated body sits where the calibrated mask puts it (verified in theory_ns3d.md 15)."""
    return min(ETA_PER_H * h, h * h * pe / STIFFNESS_MIN)


def _mesh_area(item) -> float:
    v = np.asarray(item.vertices_numpy() if hasattr(item, "vertices_numpy") else item[0], dtype=np.float64)
    f = np.asarray(item.faces if hasattr(item, "faces") else item[1], dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _body_grid(ops, win, g: str):
    """``grid_signed_distance`` grid tuple of grid ``g`` over cell window ``win``."""
    k0, k1, j0, j1, i0, i1 = win
    h = ops.h
    X0, Y0, Z0 = ops.x0 + i0 * h, ops.y0 + j0 * h, ops.z0 + k0 * h
    wx, wy, wz = i1 - i0, j1 - j0, k1 - k0
    return {"u": (wx + 1, wy, wz, h, X0 - 0.5 * h, Y0, Z0), "v": (wx, wy + 1, wz, h, X0, Y0 - 0.5 * h, Z0),
            "w": (wx, wy, wz + 1, h, X0, Y0, Z0 - 0.5 * h), "c": (wx, wy, wz, h, X0, Y0, Z0)}[g]


class ThermalGeometry:
    """Per-body thermal masks on the momentum geometry's window (all tensors on the device).

    * ``chi_t`` / ``a_t``: centre solid fraction of the "temperature" bodies and
      ``sum chi_b theta_b`` (the penalisation target, pre-multiplied);
      ``w_t`` (nT, ...) each such body's share for its heat-transfer rate;
      ``t_index`` their indices among all bodies;
    * ``adia`` = (chi_u, chi_v, chi_w): face solid fractions of "adiabatic" bodies,
      which scale the diffusive flux by (1 - chi);
    * ``skin``: heat-flux source density of the "heat_flux" bodies in conduction
      units (G delta, times kappa when added to the heat equation), and
      ``flux_rate`` (nb,) the heat each puts into the solved domain (conduction
      units, G x the in-domain part of its area);
    * ``vol_w``: the bodies' volume fraction on the w-faces -- the smoothed
      indicator *without* the calibrated wall offset, whose integral is the true
      volume to O(h^2) (the momentum mask sits ~0.25 h inside the surface and
      would under-count it by ~20 % at 6 cells per diameter). It weighs the
      buoyancy that leaves the body force (Archimedes, theory_ns3d.md 15.3).
    """

    def __init__(self, bodies, ops, window, *, use_shared: bool = True):
        dev, dt = ops.device, ops.dtype
        self.window = tuple(window)
        nb = len(bodies)
        self.nbody = nb
        self.area = np.array([_mesh_area(b) for b in bodies], dtype=np.float64)
        self.modes = [getattr(b, "thermal", "adiabatic") for b in bodies]
        self.values = np.array([float(getattr(b, "thermal_value", 0.0)) for b in bodies])
        self.t_index = [i for i, m in enumerate(self.modes) if m == "temperature"]
        k0, k1, j0, j1, i0, i1 = self.window
        empty = window_empty(self.window) or nb == 0
        h = ops.h
        w = ops.mask_width * h
        shift = ops.mask_shift * h
        band = (ops.mask_width + abs(ops.mask_shift) + 1.0) * h
        self.chi_t = self.a_t = self.w_t = None
        self.adia = None
        self.skin = None
        self.vol_w = None
        # skin cells of the heat-flux bodies: flat index into the full field, normalised
        # release weight and owning body (:meth:`wall_theta`); None without such bodies
        self.wall_idx = self.wall_w = self.wall_seg = None
        self.flux_rate = torch.zeros(nb, device=dev, dtype=torch.float64)
        if empty:
            self.t_index = []
            return
        from .geometry import _mesh_tensors
        items = [_mesh_tensors(b, dev, dt) for b in bodies]
        sdf = {}

        def dist(b: int, g: str) -> torch.Tensor:
            if (b, g) not in sdf:
                vt, ft, _ = items[b]
                with torch.no_grad():
                    sdf[(b, g)] = grid_signed_distance([(vt.detach(), ft)], _body_grid(ops, self.window, g), band,
                                                       use_shared=use_shared)
            return sdf[(b, g)]

        if self.t_index:
            cc = torch.stack([smooth_step(dist(b, "c") + shift, w) for b in self.t_index])
            self.chi_t = cc.sum(0).clamp(max=1.0)
            theta = torch.tensor([self.values[b] for b in self.t_index], device=dev, dtype=dt)
            self.a_t = (cc * theta[:, None, None, None]).sum(0)
            self.w_t = torch.ones_like(cc) if len(self.t_index) == 1 else cc / (cc.sum(0, keepdim=True) + 1e-30)
        adia = [i for i, m in enumerate(self.modes) if m == "adiabatic"]
        if adia:
            win = self.window
            fr = []
            for g in ("u", "v", "w"):
                chi = torch.stack([smooth_step(dist(b, g) + shift, w) for b in adia]).sum(0).clamp(max=1.0)
                arr = _zero_boundary_faces(chi.detach().cpu().double().numpy(), win, ops, g)
                fr.append(torch.as_tensor(arr, device=dev, dtype=dt))
            self.adia = tuple(fr)
        vol = torch.stack([smooth_step(dist(b, "w"), w) for b in range(nb)]).sum(0).clamp(max=1.0)
        self.vol_w = torch.as_tensor(_zero_boundary_faces(vol.detach().cpu().double().numpy(), self.window, ops, "w"),
                                     device=dev, dtype=dt)
        flux = [i for i, m in enumerate(self.modes) if m == "heat_flux"]
        if flux:
            skin = torch.zeros(k1 - k0, j1 - j0, i1 - i0, device=dev, dtype=dt)
            rates = np.zeros(nb)
            idx, wts, seg = [], [], []
            # flat index of window cell (k, j, i) in the full (nz, ny, nx) field
            kk = torch.arange(k0, k1, device=dev).reshape(-1, 1, 1)
            jj = torch.arange(j0, j1, device=dev).reshape(1, -1, 1)
            ii = torch.arange(i0, i1, device=dev).reshape(1, 1, -1)
            flat = (kk * ops.ny + jj) * ops.nx + ii
            for b in flux:
                d_win, total = self._skin_delta(items[b], ops, band, use_shared, w)
                if total <= 0:
                    continue
                dens = d_win * (self.values[b] * self.area[b] / (total * h ** 3))      # integrates to G A
                skin = skin + dens
                rates[b] = float(dens.sum()) * h ** 3
                sel = d_win > 0
                if bool(sel.any()):
                    wv = d_win[sel]
                    idx.append(flat.expand_as(d_win)[sel])
                    wts.append(wv / wv.sum())
                    seg.append(torch.full((int(sel.sum()),), b, device=dev, dtype=torch.long))
            self.skin = skin
            self.flux_rate = torch.as_tensor(rates, device=dev, dtype=torch.float64)
            if idx:
                self.wall_idx = torch.cat(idx)
                self.wall_w = torch.cat(wts).to(dt)
                self.wall_seg = torch.cat(seg)

    def wall_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Wall temperature of every body (nb,), a device tensor: the held value of a
        "temperature" body; for a "heat_flux" body the mean of theta over the skin its heat
        is released into, weighted like the release (the diffuse wall's own temperature);
        0 for adiabatic bodies (not measured).

        Why not the fluid probes of :meth:`FlowSolver3D.surface` (theta 1.5 h and 2.5 h off
        the wall, extrapolated linearly): when the thermal boundary layer is about one cell
        thick -- any Draft run of a body metres long at a few m/s -- both probes sit in
        nearly ambient fluid and the extrapolation returns almost the ambient temperature,
        so Nu = Q / (A theta_w) came out ~10x too high (a 0.5 m box at 200 W/m^2 read Nu 1,821
        with a 2 K wall excess while the fluid next to it was 13 K warm). The skin is where
        the model puts the wall's heat; its temperature is the wall temperature of the
        model at any resolution, and at resolved boundary layers it agrees with the
        uniform-flux flat-plate and sphere references (theory_ns3d.md 15.7)."""
        out = torch.zeros(self.nbody, device=theta.device, dtype=theta.dtype)
        if self.t_index:
            vals = torch.as_tensor(self.values[self.t_index], device=theta.device, dtype=theta.dtype)
            out[torch.as_tensor(self.t_index, device=theta.device)] = vals
        if self.wall_idx is not None:
            out.index_add_(0, self.wall_seg, theta.reshape(-1)[self.wall_idx] * self.wall_w)
        return out

    def _skin_delta(self, item, ops, band: float, use_shared: bool, w: float):
        """The surface delta (derivative of the unshifted mask) on the window, and its sum over
        an unclipped grid holding the whole body (the normaliser: total = flux x area exactly)."""
        vt, ft, _ = item
        v = vt.detach().double().cpu().numpy()
        full = grid_window(v.min(axis=0), v.max(axis=0), ops, margin=ops.mask_width + 3.0, clip=False)
        with torch.no_grad():
            d = grid_signed_distance([(vt.detach(), ft)], _body_grid(ops, full, "c"), band, use_shared=use_shared)
        s = (d / w).clamp(-1.0, 1.0)
        delta = torch.where(s.abs() < 1.0, (math.pi / (4.0 * w)) * torch.cos(0.5 * math.pi * s), torch.zeros_like(s))
        total = float(delta.sum())
        k0, k1, j0, j1, i0, i1 = self.window
        fk0, fk1, fj0, fj1, fi0, fi1 = full
        out = torch.zeros(k1 - k0, j1 - j0, i1 - i0, device=delta.device, dtype=delta.dtype)
        a = (max(k0, fk0), min(k1, fk1), max(j0, fj0), min(j1, fj1), max(i0, fi0), min(i1, fi1))
        if not window_empty(a):
            out[a[0] - k0:a[1] - k0, a[2] - j0:a[3] - j0, a[4] - i0:a[5] - i0] = \
                delta[a[0] - fk0:a[1] - fk0, a[2] - fj0:a[3] - fj0, a[4] - fi0:a[5] - fi0]
        return out, total


class HeatSourceSet:
    """Volumetric heaters: per source a cell window and a density (conduction units, times
    kappa in the heat equation) that integrates exactly to its power over the whole shape."""

    def __init__(self, sources: Sequence[HeatSource3D], ops):
        dev, dt = ops.device, ops.dtype
        self.items: list[tuple[tuple, torch.Tensor]] = []
        self.powers: list[float] = []
        self.notes: list[str] = []
        for s in sources:
            if not s.enabled:
                continue
            if s.kind == "box":
                fn, lo, hi = box_shape(s.center, s.size, ops.h, getattr(s, "axes", None))
            else:
                fn, lo, hi = sphere_shape(s.center, s.size[0], ops.h)
            fp = Footprint(fn, lo, hi, ops, ("c",))
            if fp.empty:
                self.notes.append(f"heat source {s.name or s.kind!r} lies outside the domain; ignored")
                continue
            q = fp.normalised("c", ops.h) * s.power
            self.items.append((fp.win, torch.as_tensor(q, device=dev, dtype=dt)))
            self.powers.append(float(q.sum()) * ops.h ** 3)

    @property
    def total(self) -> float:
        """Heat released into the solved domain (conduction units, k dT L)."""
        return float(sum(self.powers))


class ThermalOps:
    """Everything the heat equation needs: coefficients, wall rules, masks, sources and the
    device accumulators of the bodies' heat-transfer rates.

    Internally the equation is in convective units (theta per L/U); heat rates,
    fluxes and powers cross this interface in conduction units (k dT_ref L), so
    a body's Nusselt number is its rate over (area x theta).
    """

    def __init__(self, cfg: SolverConfig3D, ops, geom, bodies):
        tc = cfg.thermal
        dev, dt = ops.device, ops.dtype
        self.cfg = tc
        self.prandtl = float(tc.prandtl)
        self.pe = float(cfg.re) * self.prandtl
        self.kappa = 1.0 / self.pe
        self.ri = float(tc.richardson)
        self.prt = float(tc.prandtl_turbulent)
        self.initial = float(tc.theta_initial)
        self.eta_fixed = tc.eta
        self.eta = torch.tensor(tc.eta if tc.eta is not None else auto_eta_thermal(ops.h, self.pe),
                                device=dev, dtype=dt)
        self.notes: list[str] = []
        sides = cfg.side_walls
        kinds, vals = [], []
        for side in WALL_SIDES:
            vel = sides[side]
            if vel == "periodic":
                kind, val = "periodic", 0.0
            elif vel == "inflow":
                kind, val = "inflow", tc.theta_inlet
            elif vel == "outflow":
                kind, val = "outflow", 0.0
            elif vel == "symmetry":
                kind, val = "adiabatic", 0.0
            else:
                kind, val = tc.wall(side)
            if side in tc.walls and kind != tc.walls[side][0]:
                self.notes.append(f"the {side} side is a {vel} boundary: its thermal condition "
                                  f"{tc.walls[side][0]!r} is not used")
            kinds.append(kind)
            vals.append(val)
        self.side_kinds = tuple(kinds)
        self.side_vals = torch.tensor(vals, device=dev, dtype=dt)
        self._flux_w = {}
        for i, kind in enumerate(kinds):
            if kind == "heat_flux":
                dim = (2, 2, 1, 1, 0, 0)[i]
                lo = i % 2 == 0
                shape = [1, 1, 1]
                shape[dim] = 2
                # ghost = theta + G h (first layer) and + 3 G h (second): -d theta/dn = G at the wall
                wts = torch.tensor([3.0, 1.0] if lo else [1.0, 3.0], device=dev, dtype=dt).reshape(shape)
                self._flux_w[i] = wts * ops.h
        self.sources = HeatSourceSet(cfg.heat_sources, ops)
        self.notes.extend(self.sources.notes)
        self.geom = ThermalGeometry(bodies, ops, geom.window)
        nb = len(bodies)
        self.heat_acc = torch.zeros(nb, device=dev, dtype=dt)
        self.heat_rate = torch.zeros(nb, device=dev, dtype=dt)
        self.t_index = torch.as_tensor(self.geom.t_index, device=dev, dtype=torch.long)
        self.symmetry = bool(ops.symmetry)
        self.q_max = self._source_density_max(ops)
        # heating rate of the strongest source cell (theta per unit time), a device scalar for the
        # step rule (:meth:`Physics3D.dt_rule`); refreshed in place by set_reynolds (graph-safe)
        self.heat_max = (torch.tensor(self.kappa * self.q_max, device=dev, dtype=dt)
                         if self.q_max > 0.0 else None)

    def _source_density_max(self, ops) -> float:
        """Largest |source density| (conduction units) any cell receives from the heaters and the
        heat-flux skins together: their densities add in :meth:`stage`, so overlapping sources are
        summed on the union of their windows (once, at build time) before the maximum is taken."""
        parts = [(win, q) for win, q in self.sources.items]
        if self.geom.skin is not None:
            parts.append((self.geom.window, self.geom.skin))
        if not parts:
            return 0.0
        if len(parts) == 1:
            return float(parts[0][1].abs().amax())
        lo = [min(w[2 * a] for w, _ in parts) for a in range(3)]
        hi = [max(w[2 * a + 1] for w, _ in parts) for a in range(3)]
        acc = torch.zeros(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], device=ops.device, dtype=ops.dtype)
        for (k0, k1, j0, j1, i0, i1), q in parts:
            acc[k0 - lo[0]:k1 - lo[0], j0 - lo[1]:j1 - lo[1], i0 - lo[2]:i1 - lo[2]] += q
        return float(acc.abs().amax())

    def set_reynolds(self, re: float, ops) -> None:
        """Follow a live Reynolds-number change: Pe = Re Pr, kappa and the thermal eta."""
        self.pe = float(re) * self.prandtl
        self.kappa = 1.0 / self.pe
        self.eta.fill_(self.eta_fixed if self.eta_fixed is not None else auto_eta_thermal(ops.h, self.pe))
        if self.heat_max is not None:
            self.heat_max.fill_(self.kappa * self.q_max)

    # ----------------------------------------------------------------- ghosts
    def _ghosts(self, c: torch.Tensor, dim: int, side: int) -> torch.Tensor:
        n = c.shape[dim]
        lo = side % 2 == 0
        kind = self.side_kinds[side]
        if kind == "periodic":
            return c.narrow(dim, n - 2, 2) if lo else c.narrow(dim, 0, 2)
        if kind == "outflow":
            e = c.narrow(dim, 0 if lo else n - 1, 1)
            return torch.cat([e, e], dim=dim)
        mirror = c.narrow(dim, 0, 2).flip(dim) if lo else c.narrow(dim, n - 2, 2).flip(dim)
        if kind == "adiabatic":
            return mirror
        val = self.side_vals[side]
        if kind == "inflow":
            return torch.zeros_like(mirror) + val
        if kind == "temperature":
            return 2.0 * val - mirror
        return mirror + val * self._flux_w[side]                     # heat flux into the fluid

    def pad(self, c: torch.Tensor) -> torch.Tensor:
        """theta with two ghost layers on the six faces (edges and corners are never read)."""
        nz, ny, nx = c.shape
        P = c.new_empty(nz + 4, ny + 4, nx + 4)
        m = slice(2, -2)
        P[m, m, m] = c
        P[m, m, 0:2] = self._ghosts(c, 2, 0)
        P[m, m, nx + 2:] = self._ghosts(c, 2, 1)
        P[m, 0:2, m] = self._ghosts(c, 1, 2)
        P[m, ny + 2:, m] = self._ghosts(c, 1, 3)
        P[0:2, m, m] = self._ghosts(c, 0, 4)
        P[nz + 2:, m, m] = self._ghosts(c, 0, 5)
        return P

    # ----------------------------------------------------------------- fluxes
    def face_kappa(self, nut: torch.Tensor | None, ops):
        """kappa / h on the x-, y- and z-faces: a Python float (laminar), or tensors with the
        turbulent part nu_t / Pr_t averaged from the two cells (zero on non-periodic boundary
        faces, where the wall condition uses the molecular value)."""
        kh = self.kappa / ops.h
        if nut is None:
            return kh, kh, kh
        c = 0.5 / (self.prt * ops.h)
        m = slice(1, -1)
        kx = kh + c * (nut[m, m, :-1] + nut[m, m, 1:])
        ky = kh + c * (nut[m, :-1, m] + nut[m, 1:, m])
        kz = kh + c * (nut[:-1, m, m] + nut[1:, m, m])
        px, py, pz = ops.periodic_axes
        if not px:
            kx[..., 0] = kh
            kx[..., -1] = kh
        if not py:
            ky[:, 0] = kh
            ky[:, -1] = kh
        if not pz:
            kz[0] = kh
            kz[-1] = kh
        return kx, ky, kz

    def fluxes(self, c, u, v, w, ops, *, nut=None, kern=None):
        """Total (advective + diffusive) theta fluxes on every x-, y- and z-face."""
        from .dye import face_flux as muscl
        nz, ny, nx = c.shape
        P = self.pad(c)
        m = slice(2, -2)
        kx, ky, kz = self.face_kappa(nut, ops)
        sx = (P[m, m, 0:nx + 1], P[m, m, 1:nx + 2], P[m, m, 2:nx + 3], P[m, m, 3:nx + 4])
        sy = (P[m, 0:ny + 1, m], P[m, 1:ny + 2, m], P[m, 2:ny + 3, m], P[m, 3:ny + 4, m])
        sz = (P[0:nz + 1, m, m], P[1:nz + 2, m, m], P[2:nz + 3, m, m], P[3:nz + 4, m, m])
        out = []
        for st, vel, kf in ((sx, u, kx), (sy, v, ky), (sz, w, kz)):
            if kern is not None:
                f = kern.th_flux_s(*st, vel, kf=kf) if isinstance(kf, float) else kern.th_flux_t(*st, vel, kf)
            else:
                f = muscl(*st, vel) - kf * (st[2] - st[1])
            out.append(f)
        g = self.geom
        if g.adia is not None:                       # adiabatic bodies: no conduction through their faces
            k0, k1, j0, j1, i0, i1 = g.window
            K, J, I = slice(k0 + 2, k1 + 2), slice(j0 + 2, j1 + 2), slice(i0 + 2, i1 + 2)
            cu, cv, cw = g.adia
            dx = P[K, J, i0 + 2:i1 + 3] - P[K, J, i0 + 1:i1 + 2]
            dy = P[K, j0 + 2:j1 + 3, I] - P[K, j0 + 1:j1 + 2, I]
            dz = P[k0 + 2:k1 + 3, J, I] - P[k0 + 1:k1 + 2, J, I]
            for f, kf, d, chi, g_ in ((out[0], kx, dx, cu, "u"), (out[1], ky, dy, cv, "v"), (out[2], kz, dz, cw, "w")):
                sl = face_slices(g.window, g_)
                kw = kf if isinstance(kf, float) else kf[sl]
                f[sl] += kw * chi * d
        return out

    def stage(self, c_prev, c0, up, vp, wp, a: float, bdt, wgt: float, ops, *, nut=None, kern=None, ports=None):
        """One Runge-Kutta stage of the heat equation (the momentum's (a, b) pair):
        ``lerp(c_prev, c0, a) + b dt L(c_prev)``, sources, then the implicit penalisation of
        the heated bodies, whose increments (times the stage weight) accumulate the
        bodies' heat-transfer rates."""
        fx, fy, fz = self.fluxes(c_prev, up, vp, wp, ops, nut=nut, kern=kern)
        inv_h = 1.0 / ops.h
        if kern is not None:
            D = kern.div(fx[..., 1:], fx[..., :-1], fy[:, 1:], fy[:, :-1], fz[1:], fz[:-1], inv_h=inv_h)
            ch = kern.th_upd(c_prev, c0, D, bdt, al=a)
        else:
            D = ((fx[..., 1:] - fx[..., :-1]) + (fy[:, 1:] - fy[:, :-1]) + (fz[1:] - fz[:-1])) * inv_h
            base = c_prev if a == 0.0 else torch.lerp(c_prev, c0, a)
            ch = base - bdt * D
        bk = bdt * self.kappa                    # conduction-unit sources -> the heat equation's units
        for win, q in self.sources.items:
            ch[face_slices(win, "c")] += bk * q
        g = self.geom
        if g.skin is not None:
            ch[face_slices(g.window, "c")] += bk * g.skin
        if ports is not None:
            ports.scalar_sources(ch, c_prev, bdt, "theta")
        if g.chi_t is not None:
            sl = face_slices(g.window, "c")
            win = ch[sl]
            k = bdt / self.eta
            new = (win + k * g.a_t) / (1.0 + k * g.chi_t)
            inc = (new - win) * wgt
            ch[sl] = new
            self.heat_acc.index_add_(0, self.t_index, torch.einsum("bzyx,zyx->b", g.w_t, inc))
        return ch

    def begin_step(self) -> None:
        self.heat_acc.zero_()

    def end_step(self, dt, ops) -> None:
        """Heat-transfer rate of every body in conduction units (k dT_ref L): the
        penalisation's heat (temperature bodies), the released flux (heat-flux bodies),
        doubled to the full body with a symmetry plane."""
        scale = (2.0 if self.symmetry else 1.0) * ops.h ** 3 * self.pe
        q = self.heat_acc * (scale / dt)
        if self.geom.flux_rate.numel():
            q = q + self.geom.flux_rate.to(q.dtype) * (2.0 if self.symmetry else 1.0)
        self.heat_rate.copy_(q)

    def dt_limit(self, nut_max, ops, umax_sum=None):
        """Time-step limit of the heat equation, a device scalar (or a float when laminar and
        ``umax_sum`` is None: the diffusion limit alone).

        Why a combined rule and not min(advective, diffusive): a forward-Euler stage of the
        MUSCL / van Leer advection is TVD for a Courant sum C <= 1/2 and the 7-point
        diffusion is positive for 6 D <= 1 (D = kappa dt / h^2); together the update is a
        convex combination -- bounded, no new extrema -- only when ``C / (1/2) + 6 D <= 1``,
        and SSP-RK3 keeps that property because its stages are convex combinations of such
        steps. Taking the two limits separately allowed C = 0.49 with 6 D = 1 at once, which
        is outside the RK3 stability region: a heated sphere at 24 cells per diameter (Re 50,
        Pr 0.71, where both limits coincide) grew an oscillation x1.13 per step until theta
        overflowed (theory_ns3d.md 15.2). The rule is
        ``dt <= 1 / (umax_sum / (C_max h) + kappa_eff / (VISCOUS_DT h^2))`` with C_max = 1/2."""
        from .core import VISCOUS_DT
        kap = self.kappa if nut_max is None else self.kappa + nut_max / self.prt
        diff_rate = kap / (VISCOUS_DT * ops.h * ops.h)
        if umax_sum is None:
            return 1.0 / diff_rate
        return 1.0 / (umax_sum * (1.0 / (THETA_COURANT_MAX * ops.h)) + diff_rate)

    def add_buoyancy(self, wh: torch.Tensor, theta: torch.Tensor, bdt, wgt: float, geom, ops, *,
                     inplace: bool = False):
        """Add the stage's buoyancy impulse ``b dt Ri theta`` (theta averaged to the w-faces) to
        the penalised stage value ``wh``, everywhere -- also inside solids -- and return it
        with the solid's share for the force accumulator (None without bodies).

        Why after the penalisation and also in solids (theory_ns3d.md 15.3): buoyancy is
        mostly balanced by pressure, and the projection that supplies the pressure runs
        after the penalisation. Penalising the buoyancy first, or switching it off inside
        the diffuse solid, makes the forcing non-conservative across the interface, so a
        stratified fluid around a body would never be at rest (measured: spurious currents
        of 1-2 % of the buoyancy velocity and 20-40 % error in the Archimedes force at 6-10
        cells per diameter). Added here, a linear stratification is an exact discrete rest
        state. The fictitious fluid inside a solid then feels buoyancy too; the momentum
        balance of the body's volume gives the force on the body as the penalisation
        impulse minus ``Ri integral_body(theta)`` -- the returned share (``wgt b dt Ri
        vol_w theta_f`` on the body window, ``vol_w`` the volume fraction of
        :class:`ThermalGeometry`), which the step adds to its force accumulator.
        Boundary faces of non-periodic z carry no source (w = 0 there)."""
        if self.ri == 0.0 or theta is None:
            return wh, None
        tf = 0.5 * (theta[1:] + theta[:-1])
        if inplace:
            wh[1:-1].add_(tf * (bdt * self.ri))
        else:
            wh = torch.cat([wh[:1], wh[1:-1] + tf * (bdt * self.ri), wh[-1:]], dim=0)
        if ops.periodic_axes[2]:
            edge = (theta[:1] + theta[-1:]) * (0.5 * self.ri) * bdt
            if inplace:
                wh[:1].add_(edge)
                wh[-1:].add_(edge)
            else:
                wh = torch.cat([wh[:1] + edge, wh[1:-1], wh[-1:] + edge], dim=0)
        if geom.empty:
            return wh, None
        k0, k1, j0, j1, i0, i1 = geom.window
        lo, hi = max(k0 - 1, 0), min(k1, ops.nz - 1)
        tc = theta[lo:hi + 1, j0:j1, i0:i1]
        pz = ops.periodic_axes[2]
        if k0 == 0:                                   # the face below cell 0: wrapped, or a boundary (chi = 0)
            tc = torch.cat([theta[-1:, j0:j1, i0:i1] if pz else tc[:1], tc], dim=0)
        if k1 == ops.nz:
            tc = torch.cat([tc, theta[:1, j0:j1, i0:i1] if pz else tc[-1:]], dim=0)
        tw = 0.5 * (tc[1:] + tc[:-1])                                    # faces k0..k1
        vol = self.geom.vol_w if self.geom.vol_w is not None else geom.chi_w
        share = (tw * vol) * (bdt * (self.ri * wgt))
        return wh, share

    # ----------------------------------------------------------------- reports
    def wall_heat_rates(self, theta: torch.Tensor, state, ops) -> dict[str, float]:
        """Heat entering the fluid through each domain side (conduction units, k dT_ref L)
        at the current state: the diffusive wall flux from the ghost-cell rule (and at a
        tunnel inlet / outlet the advected heat). Positive = into the fluid; full-body
        values with a symmetry plane."""
        P = self.pad(theta.double())
        h, kap = ops.h, self.kappa
        m = slice(2, -2)
        out = {}
        for i, side in enumerate(WALL_SIDES):
            dim = (2, 2, 1, 1, 0, 0)[i]
            lo = i % 2 == 0
            n = theta.shape[dim]
            inner = P.narrow(dim, 2 if lo else n + 1, 1)
            ghost = P.narrow(dim, 1 if lo else n + 2, 1)
            sl = [m, m, m]
            sl[dim] = slice(None)
            inner, ghost = inner[tuple(sl)], ghost[tuple(sl)]
            kind = self.side_kinds[i]
            if kind == "periodic":
                out[side] = 0.0
                continue
            diff = kap * (ghost - inner) / h                                  # into the fluid
            q = float(diff.sum()) * h * h
            if kind in ("inflow", "outflow"):
                un = state.u[..., 0] if lo else state.u[..., -1]
                un = un.double()
                face_val = ghost[..., 0] if kind == "inflow" else inner[..., 0]
                adv = (un * face_val).sum() * h * h
                q = (q + float(adv)) if lo else (-float(adv))                 # outlet: no diffusion
            out[side] = q * self.pe * (2.0 if (self.symmetry and side != "front") else 1.0)
        return out


# ---------------------------------------------------------------------------
# fans (actuator discs)
# ---------------------------------------------------------------------------

class FanSet:
    """Actuator discs with device-side thrust, measured disc velocity and PI state.

    Per fan: the force weights ``f_g = n_g phi_g / (sum phi h^3)`` on the u-, v-,
    w-faces of its window (so the force is exactly ``T n``), swirl weights, and
    measurement weights for the disc-averaged axial velocity (renormalised by
    the in-domain fraction, so a fan on a symmetry plane measures the mirrored
    average). Tensors (nfan,): ``thrust`` (current T), ``ud`` (measured
    velocity), ``e_prev`` (PI error), ``swirl_amp``.
    """

    def __init__(self, fans: Sequence[Fan3D], ops, cfg: SolverConfig3D):
        dev, dt = ops.device, ops.dtype
        h = ops.h
        self.items: list[dict] = []
        self.notes: list[str] = []
        self.fans: list[Fan3D] = []
        u_inf = 0.0 if cfg.closed or cfg.periodic else 1.0
        thrust0, target, kp, ki, vel = [], [], [], [], []
        for f in fans:
            if not f.enabled:
                continue
            t = f.thickness if f.thickness is not None else max(f.diameter / 8.0, 2.0 * h)
            fn, lo, hi = disc_shape(f.center, f.direction, 0.5 * f.diameter, t, h)
            fp = Footprint(fn, lo, hi, ops, ("u", "v", "w"))
            if fp.empty:
                self.notes.append(f"fan {f.name or 'fan'!r} lies outside the domain; ignored")
                continue
            n = np.asarray(f.direction, dtype=np.float64)
            R = 0.5 * f.diameter
            area = f.area
            item = {"win": fp.win}
            frac = sum(fp.fractions[g] * abs(n[i]) for i, g in enumerate("uvw")) / max(np.abs(n).sum(), 1e-30)
            for i, g in enumerate("uvw"):
                ph = fp.normalised(g, h)
                Z, Y, X = grid_coordinates(fp.win, ops, g)
                rel = [np.broadcast_to(X - f.center[0], ph.shape), np.broadcast_to(Y - f.center[1], ph.shape),
                       np.broadcast_to(Z - f.center[2], ph.shape)]
                cross = np.cross(n, np.stack(rel, axis=-1))[..., i]          # (n x (x - c))_g
                sw = (area / R) * cross * ph
                # n x (x - c) integrates to zero over the disc, but its samples on a staggered
                # grid do not when the centre is off the grid: remove the net, so the swirl is a
                # pure torque (no side force) exactly
                if ph.sum() > 0:
                    sw = sw - ph * (sw.sum() / ph.sum())
                item["f" + g] = torch.as_tensor(n[i] * ph, device=dev, dtype=dt)
                item["s" + g] = torch.as_tensor(sw, device=dev, dtype=dt)
                fr = fp.fractions[g] if fp.fractions[g] > _EMPTY else 1.0
                item["m" + g] = torch.as_tensor(n[i] * ph * h ** 3 / fr, device=dev, dtype=dt)
            item["frac"] = frac
            if f.swirl != 0 and ops.symmetry and min(fp.fractions.values()) < 1.0 - 1e-6:
                # swirl about an axis in (or across) the mirror plane is not mirror-symmetric: a
                # half-domain run cannot represent it (run the full domain instead)
                self.notes.append(f"fan {f.name or 'fan'!r} is cut by the symmetry plane: its swirl cannot be "
                                  "mirrored and is ignored (run the full domain to include it)")
                f = Fan3D(**{**f.__dict__, "swirl": 0.0})
            self.items.append(item)
            self.fans.append(f)
            un = u_inf * n[0]
            if f.mode == "thrust":
                t0 = f.value
            elif f.mode == "pressure_jump":
                t0 = f.value * area
            else:                                                         # momentum-theory first guess
                t0 = 2.0 * area * f.value * (f.value - un)
            gain = 2.0 * area * max(abs(2.0 * f.value - un), abs(un), 0.1) if f.mode == "velocity" else 0.0
            tau = 4.0 * f.diameter / max(abs(f.value), abs(un), 0.1)
            thrust0.append(t0)
            target.append(f.value if f.mode == "velocity" else 0.0)
            kp.append(0.5 * gain)
            ki.append(gain / tau)
            vel.append(f.mode == "velocity")
        k = len(self.fans)
        self.u_inf = u_inf
        self.thrust = torch.tensor(thrust0, device=dev, dtype=dt).reshape(k)
        self.thrust0 = self.thrust.clone()
        self.target = torch.tensor(target, device=dev, dtype=dt).reshape(k)
        self.kp = torch.tensor(kp, device=dev, dtype=dt).reshape(k)
        self.ki = torch.tensor(ki, device=dev, dtype=dt).reshape(k)
        self.velocity_mode = torch.tensor(vel, device=dev, dtype=torch.bool).reshape(k)
        self.tmax = torch.tensor([50.0 * f.area * (abs(f.value) + 1.0) ** 2 + abs(t0)
                                  for f, t0 in zip(self.fans, thrust0)], device=dev, dtype=dt).reshape(k)
        self.swirl = torch.tensor([f.swirl for f in self.fans], device=dev, dtype=dt).reshape(k)
        self.ud = torch.zeros(k, device=dev, dtype=dt)
        self.e_prev = torch.zeros(k, device=dev, dtype=dt)
        self.swirl_amp = torch.zeros(k, device=dev, dtype=dt)
        self.any_swirl = bool(any(f.swirl != 0 for f in self.fans))

    def __len__(self) -> int:
        return len(self.fans)

    def measure(self, u, v, w) -> torch.Tensor:
        """Disc-averaged axial velocity of every fan (device tensor (nfan,))."""
        vals = []
        for it in self.items:
            ws = it["win"]
            s = (it["mu"] * u[face_slices(ws, "u")]).sum() + (it["mv"] * v[face_slices(ws, "v")]).sum() \
                + (it["mw"] * w[face_slices(ws, "w")]).sum()
            vals.append(s)
        return torch.stack(vals)

    def begin_step(self, u0, v0, w0, dt) -> None:
        """Measure the disc velocities and update the regulated fans (PI on the thrust)."""
        ud = self.measure(u0, v0, w0)
        self.ud.copy_(ud)
        e = self.target - ud
        new = (self.thrust + self.kp * (e - self.e_prev) + self.ki * dt * e)
        new = torch.maximum(torch.minimum(new, self.tmax), -self.tmax)
        self.thrust.copy_(torch.where(self.velocity_mode, new, self.thrust))
        self.e_prev.copy_(torch.where(self.velocity_mode, e, self.e_prev))
        if self.any_swirl:
            self.swirl_amp.copy_(self.swirl * ud * ud.abs())

    def add_forces(self, lu, lv, lw) -> None:
        for i, it in enumerate(self.items):
            ws = it["win"]
            T = self.thrust[i]
            for arr, g in ((lu, "u"), (lv, "v"), (lw, "w")):
                sl = face_slices(ws, g)
                arr[sl] += it["f" + g] * T
                if self.any_swirl:
                    arr[sl] += it["s" + g] * self.swirl_amp[i]

    def reset(self) -> None:
        self.thrust.copy_(self.thrust0)
        self.ud.zero_()
        self.e_prev.zero_()
        self.swirl_amp.zero_()

    def report(self, symmetry: bool) -> list[dict[str, Any]]:
        """Per fan: thrust T, pressure jump T/A, volume flow, mean induced velocity and ideal
        power (solver units; full-body / mirrored values with a symmetry plane)."""
        T = self.thrust.detach().double().cpu().numpy()
        ud = self.ud.detach().double().cpu().numpy()
        out = []
        for i, f in enumerate(self.fans):
            mult = (2.0 if symmetry else 1.0) * self.items[i]["frac"]
            un = self.u_inf * f.direction[0]
            out.append({"name": f.name, "mode": f.mode, "thrust": float(T[i] * mult),
                        "pressure_jump": float(T[i] / f.area), "disc_velocity": float(ud[i]),
                        "flow": float(ud[i] * f.area * mult), "induced_velocity": float(ud[i] - un),
                        "power": float(T[i] * ud[i] * mult), "target": float(f.value) if f.mode == "velocity" else
                        float("nan"), "in_domain": float(self.items[i]["frac"])})
        return out


# ---------------------------------------------------------------------------
# ports: jets and vents
# ---------------------------------------------------------------------------

class PortSet:
    """Jets (mass sources with momentum, temperature and smoke) and vents (sinks)."""

    def __init__(self, ports: Sequence[Port3D], ops, cfg: SolverConfig3D):
        dev, dt = ops.device, ops.dtype
        h = ops.h
        self.items: list[dict] = []
        self.notes: list[str] = []
        self.ports: list[Port3D] = []
        total = 0.0
        for p in ports:
            if not p.enabled or p.speed == 0:
                continue
            t = p.thickness if p.thickness is not None else max(p.diameter / 4.0, 2.0 * h)
            fn, lo, hi = disc_shape(p.center, p.direction, 0.5 * p.diameter, t, h)
            fp = Footprint(fn, lo, hi, ops, ("c", "u", "v", "w"))
            if fp.empty:
                self.notes.append(f"port {p.name or p.kind!r} lies outside the domain; ignored")
                continue
            Q = p.flow
            n = np.asarray(p.direction, dtype=np.float64)
            item = {"win": fp.win, "kind": p.kind, "m": torch.as_tensor(Q * fp.normalised("c", h), device=dev,
                                                                          dtype=dt)}
            for i, g in enumerate("uvw"):
                ph = fp.normalised(g, h)
                if p.kind == "jet":
                    item["f" + g] = torch.as_tensor(Q * p.speed * n[i] * ph, device=dev, dtype=dt)
                else:
                    item["f" + g] = torch.as_tensor(Q * ph, device=dev, dtype=dt)     # times the local velocity
            item["theta"] = 0.0 if p.theta is None else float(p.theta)
            item["dye"] = None if p.dye_channel is None else int(p.dye_channel)
            self.items.append(item)
            self.ports.append(p)
            total += float(item["m"].sum()) * h ** 3
        self.volume_flow = total                                  # net source in the solved domain (U L^2)
        self.mass_flux = torch.tensor(total / (h * h), device=dev, dtype=dt)   # in units of the outlet face sum
        # Uniform leak of an unbalanced closed (or fully periodic) box: the pressure solve drops the
        # source's mean, i.e. removes the excess volume evenly from every cell. The fluid it removes
        # must take its heat and smoke along; without this the conservative scalar update kept them
        # (a source +c q/V everywhere), and a closed box fed by a theta = 1 jet with no vent climbed
        # to theta 2.77 while its heat balance still "closed" (critic round 5).
        self.leak = 0.0
        if (cfg.closed or cfg.periodic) and total > 1e-12:          # (a balanced set nets to round-off)
            self.leak = total / (ops.nx * ops.ny * ops.nz * h ** 3)
        if (cfg.closed or cfg.periodic) and abs(total) > 1e-12:
            self.notes.append(f"the jets and vents of a closed domain do not balance (net {total:+.4g} U L^2); "
                              "the pressure solve removes the excess uniformly"
                              + (", and the removed fluid takes its heat and smoke with it" if total > 0 else ""))

    def __len__(self) -> int:
        return len(self.ports)

    def add_forces(self, lu, lv, lw, u, v, w) -> None:
        for it in self.items:
            ws = it["win"]
            for arr, g, vel in ((lu, "u", u), (lv, "v", v), (lw, "w", w)):
                sl = face_slices(ws, g)
                if it["kind"] == "jet":
                    arr[sl] += it["f" + g]
                else:
                    arr[sl] += it["f" + g] * vel[sl]

    def continuity(self, div: torch.Tensor) -> torch.Tensor:
        """``div u_hat - m``: the right-hand side of the projection with sources (in place)."""
        for it in self.items:
            div[face_slices(it["win"], "c")] -= it["m"]
        return div

    def scalar_sources(self, c_hat, c_prev, bdt, what) -> None:
        """Add ``b dt m s`` to a transported scalar: s = the jet's value (theta or smoke of
        channel ``what``) or, for a vent, the local value (the fluid leaves with it); in an
        unbalanced closed box also the uniform leak ``-b dt (Q / V) c`` (see ``leak``) over every
        cell the pressure solve's mean removal acts on. A net sink (``Q < 0``) needs no term: the
        fluid the solve creates evenly arrives at 0 and dilutes through the conservative flux."""
        for it in self.items:
            sl = face_slices(it["win"], "c")
            if it["kind"] == "vent":
                c_hat[sl] += bdt * it["m"] * c_prev[sl]
                continue
            if what == "theta":
                val = it["theta"]
            else:
                val = 1.0 if it["dye"] == what else 0.0
            if val != 0.0:
                c_hat[sl] += (bdt * val) * it["m"]
        if self.leak:
            c_hat.sub_(c_prev * (bdt * self.leak))


# ---------------------------------------------------------------------------
# painted obstacles (explore mode)
# ---------------------------------------------------------------------------

class PaintLayer:
    """Solid fraction painted with the mouse, on the whole domain (faces and centres).

    Painted solids are at rest and penalised with the bodies' permeability after
    the body penalisation; they block smoke and particles. They carry no force
    report, conduct heat like fluid at rest, and never enter the differentiable
    core (the functional step functions do not know them).
    """

    def __init__(self, ops):
        dev, dt = ops.device, ops.dtype
        nz, ny, nx = ops.nz, ops.ny, ops.nx
        self.chi_u = torch.zeros(nz, ny, nx + 1, device=dev, dtype=dt)
        self.chi_v = torch.zeros(nz, ny + 1, nx, device=dev, dtype=dt)
        self.chi_w = torch.zeros(nz + 1, ny, nx, device=dev, dtype=dt)
        self.chi_c = torch.zeros(nz, ny, nx, device=dev, dtype=dt)

    def stroke(self, ops, center, radius: float, *, erase: bool = False, shape: str = "sphere", size=None) -> bool:
        """Add (or with ``erase`` remove) a brush footprint: a ball of ``radius`` or a box of
        ``size`` with the mask's smooth edge. Returns False when it misses the domain."""
        h = ops.h
        w = ops.mask_width * h
        shift = ops.mask_shift * h
        c = np.asarray(center, dtype=np.float64)
        if shape == "box":
            hs = 0.5 * np.asarray(size if size is not None else (2 * radius,) * 3, dtype=np.float64)

            def dist(X, Y, Z):
                # exact signed distance of the box (rounded outside the corners, like the mask)
                qx, qy, qz = np.abs(X - c[0]) - hs[0], np.abs(Y - c[1]) - hs[1], np.abs(Z - c[2]) - hs[2]
                outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2)
                return outside + np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
            ext = hs
        elif shape == "sphere":
            def dist(X, Y, Z):
                return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - radius
            ext = np.array([radius] * 3)
        else:
            raise ValueError("brush shape must be 'sphere' or 'box'")
        win = grid_window(c - ext, c + ext, ops, margin=ops.mask_width + 2.0)
        if window_empty(win):
            return False
        for g, arr in (("u", self.chi_u), ("v", self.chi_v), ("w", self.chi_w), ("c", self.chi_c)):
            Z, Y, X = grid_coordinates(win, ops, g)
            d = torch.as_tensor(np.broadcast_to(dist(X, Y, Z), np.broadcast_shapes(Z.shape, Y.shape, X.shape)).copy(),
                                device=arr.device, dtype=arr.dtype)
            b = smooth_step(d + shift, w)
            if g != "c":
                b = torch.as_tensor(_zero_boundary_faces(b.cpu().double().numpy(), win, ops, g), device=arr.device,
                                    dtype=arr.dtype)
            sl = face_slices(win, g)
            arr[sl] = torch.minimum(arr[sl], 1.0 - b) if erase else torch.maximum(arr[sl], b)
        return True

    def clear(self) -> None:
        for a in (self.chi_u, self.chi_v, self.chi_w, self.chi_c):
            a.zero_()

    def penalise(self, uh, vh, wh, bdt, eta) -> None:
        """Implicit penalisation towards rest: ``u <- u / (1 + k chi)`` (in place)."""
        k = bdt / eta
        uh.div_(1.0 + k * self.chi_u)
        vh.div_(1.0 + k * self.chi_v)
        wh.div_(1.0 + k * self.chi_w)


# ---------------------------------------------------------------------------
# PID time step
# ---------------------------------------------------------------------------

class PidDt:
    """PID control of the time step towards ``target`` x the stability limit.

    The CFL rule jumps to the limit every step, so dt follows every flicker of
    max|u|. This controller works on ``e = log(dt_target / dt_prev)``:
    ``dt = dt_prev exp(kI e + kP (e - e1) + kD (e - 2 e1 + e2))``, the growth
    clamped to [0.5, 1.2] per step and dt never above the limit itself, so the
    CFL number never exceeds ``cfl``. All state is on the device (graph-safe).
    """

    def __init__(self, ops, target: float):
        dev, dt = ops.device, ops.dtype
        self.target = float(target)
        self.dt_prev = torch.zeros((), device=dev, dtype=dt)
        self.e1 = torch.zeros((), device=dev, dtype=dt)
        self.e2 = torch.zeros((), device=dev, dtype=dt)

    def __call__(self, dt_cap: torch.Tensor) -> torch.Tensor:
        ki, kp, kd = PID_GAINS
        want = dt_cap * self.target
        first = self.dt_prev <= 0
        prev = torch.where(first, want, self.dt_prev)
        e = torch.log(want / prev)
        fac = torch.exp(ki * e + kp * (e - self.e1) + kd * (e - 2.0 * self.e1 + self.e2))
        fac = fac.clamp(PID_LIMITS[0], PID_LIMITS[1])
        dt = torch.minimum(torch.where(first, want, prev * fac), dt_cap)
        self.e2.copy_(self.e1)
        self.e1.copy_(torch.where(first, torch.zeros_like(e), e))
        self.dt_prev.copy_(dt)
        return dt

    def reset(self) -> None:
        for t in (self.dt_prev, self.e1, self.e2):
            t.zero_()


# ---------------------------------------------------------------------------
# the container used by the step functions
# ---------------------------------------------------------------------------

class Physics3D:
    """The placed physics of one solver; the step functions call its hooks.

    Hooks (all device-side, graph-capturable), in the order a stage calls them:
    :meth:`dt_rule` and :meth:`begin_step` (first stage), :meth:`momentum_sources`
    (fans, ports), :attr:`mass_flux` (the outlet balance), :meth:`penalise`
    (paint), :meth:`buoyancy` (after the penalisation), :meth:`theta_stage`,
    :meth:`continuity` (the projection's sources); :meth:`end_step` after the
    last stage. :meth:`buffers` lists the tensors a step mutates in place
    (snapshotted around a CUDA-graph capture).
    """

    def __init__(self, cfg: SolverConfig3D, ops, geom, bodies):
        self.cfg = cfg
        self.ops = ops
        self.thermal = ThermalOps(cfg, ops, geom, bodies) if cfg.thermal is not None else None
        fans = FanSet(cfg.fans, ops, cfg) if cfg.fans else None
        self.fans = fans if fans is not None and len(fans) else None
        ports = PortSet(cfg.ports, ops, cfg) if cfg.ports else None
        self.ports = ports if ports is not None and len(ports) else None
        self.paint: PaintLayer | None = None
        self.pid = PidDt(ops, cfg.pid_target) if cfg.adaptive_dt == "pid" else None
        self.diag = cfg.diagnostics
        # speed scale of the momentum sources, for the CFL bound from rest (as in the 2D solver): a fan
        # or jet in still air would otherwise take its first step at the viscous limit
        floor = 0.0
        if self.ports is not None:
            floor = max(floor, max(p.speed for p in self.ports.ports))
        if self.fans is not None:
            for f in self.fans.fans:
                scale = abs(f.value) if f.mode == "velocity" else math.sqrt(
                    abs(f.value / f.area if f.mode == "thrust" else f.value))
                floor = max(floor, scale * (1.0 + abs(f.swirl)))
        self.u_floor = float(floor)
        self.div_pre = torch.zeros((), device=ops.device, dtype=ops.dtype)
        self.notes: list[str] = []
        for part in (self.thermal, fans, ports):
            if part is not None:
                self.notes.extend(part.notes)

    # -------------------------------------------------------------- the hooks
    @property
    def mass_flux(self):
        """Net port source in outlet-face units (``h sum m``), or None."""
        return self.ports.mass_flux if self.ports is not None else None

    def dt_rule(self, u0, v0, w0, ops, nut_max, theta=None):
        """The step's dt (device scalar): the momentum's CFL / viscous limits, the sources'
        speed scale (``u_floor``: jets, fans), the heat equation's combined bound
        (:meth:`ThermalOps.dt_limit`), the buoyant free-fall limit -- a parcel at rest
        accelerates at Ri |theta|, so it may move at most ``cfl`` cells in one step from rest
        (the 2D solver's rules) -- and the heat sources' rate bound (a parcel heated from rest),
        then the PID controller when enabled."""
        from .core import cfl_dt3d
        dt = cfl_dt3d(u0, v0, w0, ops, nut_max)
        umax = (u0.abs().amax() + v0.abs().amax() + w0.abs().amax()).clamp_min(1e-6)
        if self.u_floor > 0.0:
            dt = dt.clamp(max=ops.cfl * ops.h / self.u_floor)
            umax = umax.clamp(min=self.u_floor)
        if self.thermal is not None:
            dt = torch.minimum(dt, self.thermal.dt_limit(nut_max, ops, umax))   # combined advection-diffusion
            if self.thermal.ri != 0.0 and theta is not None:
                acc = (abs(self.thermal.ri) * theta.abs().amax()).clamp_min(1e-12)
                dt = torch.minimum(dt, ops.cfl * torch.sqrt(ops.h / acc))
            if self.thermal.heat_max is not None:
                # Heaters (and heat-flux skins) warming fluid at rest: on the first step u = 0 and
                # theta = 0, so neither bound above applies and only the diffusion limit was left.
                # A 300 W box heater in a still 3D room took a 68.7 L/U first step, reached theta
                # 6e5 and |u| 1.6e6, and was NaN on step 2 (critic round 5; the 2D solver got the
                # same bound in critic round 3.4, ns2d.physics.physics_dt_limit). A parcel heated
                # from rest at S = kappa max|q| has theta = S t, accelerates by Ri S t and moves
                # Ri S t^3 / 6: keep that within cfl cells, and each step's increment S dt <= 1.
                s_max = self.thermal.heat_max.clamp_min(1e-12)
                dt = torch.minimum(dt, 1.0 / s_max)
                if self.thermal.ri != 0.0:
                    dt = torch.minimum(dt, (6.0 * ops.cfl * ops.h / (abs(self.thermal.ri) * s_max)) ** (1.0 / 3.0))
        if self.pid is not None:
            dt = self.pid(dt)
        return dt

    def begin_step(self, u0, v0, w0, dt) -> None:
        if self.thermal is not None:
            self.thermal.begin_step()
        if self.fans is not None:
            self.fans.begin_step(u0, v0, w0, dt)

    def momentum_sources(self, lu, lv, lw, u, v, w, theta, geom):
        """Fans and ports, added to the explicit tendencies before the stage update.
        (Buoyancy is not a tendency here: :meth:`buoyancy` adds it after the penalisation.)"""
        if self.fans is not None:
            self.fans.add_forces(lu, lv, lw)
        if self.ports is not None:
            self.ports.add_forces(lu, lv, lw, u, v, w)
        return lu, lv, lw

    def continuity(self, div):
        return self.ports.continuity(div) if self.ports is not None else div

    def penalise(self, uh, vh, wh, bdt, ops) -> None:
        if self.paint is not None:
            self.paint.penalise(uh, vh, wh, bdt, ops.eta)

    @property
    def buoyant(self) -> bool:
        """True when the step must add buoyancy (heat transfer with Ri != 0)."""
        return self.thermal is not None and self.thermal.ri != 0.0

    def buoyancy(self, wh, theta, bdt, wgt, geom, ops, *, inplace: bool = False):
        """The stage's buoyancy impulse after the penalisation (:meth:`ThermalOps.add_buoyancy`):
        returns (wh, the solid's share for the force accumulator or None)."""
        return self.thermal.add_buoyancy(wh, theta, bdt, wgt, geom, ops, inplace=inplace)

    def theta_stage(self, c_prev, c0, up, vp, wp, a, bdt, wgt, ops, *, nut=None, kern=None):
        return self.thermal.stage(c_prev, c0, up, vp, wp, a, bdt, wgt, ops, nut=nut, kern=kern, ports=self.ports)

    def end_step(self, dt, ops) -> None:
        if self.thermal is not None:
            self.thermal.end_step(dt, ops)

    def record_divergence(self, div) -> None:
        """Diagnostics: the squared norm of the last stage's projection right-hand side."""
        self.div_pre.copy_((div * div).sum())

    # ------------------------------------------------------------------ misc
    def buffers(self) -> list[torch.Tensor]:
        out = [self.div_pre]
        if self.thermal is not None:
            out += [self.thermal.heat_acc, self.thermal.heat_rate]
        if self.fans is not None:
            out += [self.fans.thrust, self.fans.ud, self.fans.e_prev, self.fans.swirl_amp]
        if self.pid is not None:
            out += [self.pid.dt_prev, self.pid.e1, self.pid.e2]
        return out

    def reset(self) -> None:
        if self.thermal is not None:
            self.thermal.heat_acc.zero_()
            self.thermal.heat_rate.zero_()
        if self.fans is not None:
            self.fans.reset()
        if self.pid is not None:
            self.pid.reset()
        self.div_pre.zero_()


__all__ = [
    "PID_GAINS",
    "PLUME_SPEED_FACTOR",
    "THETA_COURANT_MAX",
    "FanSet",
    "Footprint",
    "HeatSourceSet",
    "PaintLayer",
    "Physics3D",
    "PidDt",
    "PortSet",
    "ThermalGeometry",
    "ThermalOps",
    "auto_eta_thermal",
    "box_shape",
    "disc_shape",
    "face_slices",
    "grid_coordinates",
    "grid_window",
    "planning_speed_sum",
    "ramp",
    "sphere_shape",
    "thermal_dt_estimate",
]
