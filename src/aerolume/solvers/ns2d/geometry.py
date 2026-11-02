"""Immersed-body geometry: signed distance -> smooth solid masks -> force weights.

Bodies enter the flow solver only through Brinkman volume penalisation, so the
whole geometric description is a set of fields on the staggered grid:

* ``chi_u``, ``chi_v`` - solid fraction at the u-faces and v-faces, where the
  momentum lives. They are evaluated directly at the face positions (not
  averaged from the centres), which keeps the effective wall where the SDF puts
  it for both velocity components.
* ``a_u``, ``a_v``  - ``sum_b chi_b * u_solid_b``: the penalisation target
  pre-multiplied by the mask, so rigid rotation (spinning wheels) costs nothing
  extra in the time step.
* ``k_u``, ``k_v``  - per-body force / moment weights that turn the momentum
  removed by the penalisation into ``[Cd, Cl, Cm]`` (see core.step).
* ``chi_c``         - solid fraction at cell centres (LES switch-off, display).

The mask is a compact C1 "half-sine" step of the signed distance d::

    chi(d) = 1                          d <= -w
           = (1 - sin(pi d / 2w)) / 2   |d| < w
           = 0                          d >= w

with half-width ``w = mask_width * h``. Compact support matters: with an
implicit (stiff) penalisation any non-zero tail of chi would act as a wall, so
an infinite-support profile such as tanh would move the effective surface
outwards. The profile is smooth in the vertex positions (through the SDF), so
the masks - and every force computed from them - are differentiable.

All fields are stored only on a *window*: the cell block covering the bodies
plus a margin. Penalisation, force sums and SDF evaluations are therefore
proportional to the body size, not the domain size.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

from .config import ImmersedBody

# ---------------------------------------------------------------------------
# signed distance
# ---------------------------------------------------------------------------

_SHARED_SDF = None          # resolved lazily: aerolume.geometry.sdf.signed_distance or False


def _shared_sdf():
    global _SHARED_SDF
    if _SHARED_SDF is None:
        try:
            from aerolume.geometry.sdf import signed_distance as fn
            _SHARED_SDF = fn
        except Exception:  # noqa: BLE001 - module missing or broken while being written: use ours
            _SHARED_SDF = False
    return _SHARED_SDF or None


def polygon_sdf(poly: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, chunk: int = 1 << 22) -> torch.Tensor:
    """Signed distance from points (x, y) to one closed polygon, negative inside.

    The magnitude is the exact Euclidean distance to the nearest edge, which is
    differentiable with respect to the vertices almost everywhere; the sign
    comes from an even-odd crossing test (piecewise constant, no gradient
    needed). Work is chunked so that at most ``chunk`` point-edge pairs are
    alive at once.
    """
    a = poly
    b = torch.roll(poly, -1, dims=0)
    ab = b - a                                             # (n, 2)
    ab2 = (ab * ab).sum(-1).clamp_min(1e-30)               # (n,)
    px, py = x.reshape(-1), y.reshape(-1)
    n = a.shape[0]
    step = max(1, chunk // max(n, 1))
    out = []
    for s in range(0, px.numel(), step):
        qx, qy = px[s:s + step, None], py[s:s + step, None]
        apx, apy = qx - a[None, :, 0], qy - a[None, :, 1]
        t = ((apx * ab[None, :, 0] + apy * ab[None, :, 1]) / ab2[None]).clamp(0.0, 1.0)
        dx, dy = apx - t * ab[None, :, 0], apy - t * ab[None, :, 1]
        d2 = (dx * dx + dy * dy).amin(dim=1)
        ay, by = a[None, :, 1], b[None, :, 1]
        straddle = (ay > qy) != (by > qy)
        xcross = a[None, :, 0] + (qy - ay) * ab[None, :, 0] / torch.where(by != ay, by - ay, 1e-30)
        crossings = (straddle & (qx < xcross)).sum(dim=1)
        sign = 1.0 - 2.0 * (crossings % 2).to(d2.dtype)
        out.append(sign * torch.sqrt(d2.clamp_min(1e-24)))
    return torch.cat(out).reshape(x.shape)


def signed_distance(polys: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Signed distance to the union of polygons (negative inside any of them).

    Uses the shared :func:`aerolume.geometry.sdf.signed_distance` when that
    module is importable, otherwise the local implementation above.
    """
    fn = _shared_sdf()
    if fn is not None:
        return fn(list(polys), x, y)
    d = None
    for p in polys:
        dp = polygon_sdf(p, x, y)
        d = dp if d is None else torch.minimum(d, dp)
    return d


def smooth_step(d: torch.Tensor, w: float) -> torch.Tensor:
    """Compact C1 solid fraction of a signed distance (1 inside, 0 outside)."""
    s = (d / w).clamp(-1.0, 1.0)
    return 0.5 - 0.5 * torch.sin(0.5 * math.pi * s)


# ---------------------------------------------------------------------------
# geometry container
# ---------------------------------------------------------------------------

class ThermalGeometry(NamedTuple):
    """Thermal conditions of the bodies on the same window as :class:`Geometry`.

    * ``chi_t`` / ``a_t``: solid fraction of the "temperature" bodies at cell
      centres and ``sum chi_b theta_b`` (implicit penalisation target, as ``a_u``).
    * ``w_t`` (nT, wy, wx): each temperature body's share of ``chi_t`` (heat-rate split).
    * ``ins_u`` / ``ins_v``: ``1 - chi`` of every other body at the faces. The
      diffusivity is multiplied by it, so no heat crosses an adiabatic or
      heat-flux body (the Neumann penalisation of Kadoch et al. 2012).
    * ``q_skin``: heat released by the "heat_flux" bodies per unit area and unit
      time, divided by the diffusivity (it is multiplied by 1/Pe at run time, so
      it follows live Reynolds-number changes). It is spread over the interface
      with the weight ``-d chi / d d``, whose integral across the interface is
      exactly 1, and normalised per body so the released heat is exactly
      ``G P / Pe`` (theory doc 16.3).
    * ``w_f`` (nF, wy, wx): each heat-flux body's normalised skin weight, to
      average its wall temperature.
    * ``temp_ids`` / ``flux_ids``: body indices of each kind; ``theta_s`` (nT,),
      ``flux_g`` (nF,) their values; ``perimeter`` (nb,) all outline lengths.
    * ``vol_u`` / ``vol_v``: the bodies' area fraction on the u- / v-faces -- the smoothed
      indicator *without* the calibrated wall offset, whose integral is the true area to
      O(h^2) (the momentum mask sits 0.32 h inside the surface). It weighs the buoyancy
      that leaves the body force (Archimedes; the 3D solver's ``vol_w``, theory_ns3d.md 15.3).
    """
    chi_t: torch.Tensor
    a_t: torch.Tensor
    w_t: torch.Tensor
    ins_u: torch.Tensor
    ins_v: torch.Tensor
    q_skin: torch.Tensor
    w_f: torch.Tensor
    theta_s: torch.Tensor
    flux_g: torch.Tensor
    perimeter: torch.Tensor
    temp_ids: tuple[int, ...]
    flux_ids: tuple[int, ...]
    vol_u: torch.Tensor | None = None
    vol_v: torch.Tensor | None = None


class Geometry(NamedTuple):
    """Masks and weights of all immersed bodies on the solver window.

    ``window = (j0, j1, i0, i1)`` is a block of *cells* (rows j0..j1-1, columns
    i0..i1-1). The u-face arrays cover rows j0..j1-1 and faces i0..i1; the
    v-face arrays cover faces j0..j1 and columns i0..i1-1. An empty window
    (``j1 == j0``) means "no bodies". ``thermal`` holds the bodies' heat-transfer
    conditions when the solver carries temperature (None otherwise).
    """
    window: tuple[int, int, int, int]
    chi_u: torch.Tensor          # (wy, wx+1)
    chi_v: torch.Tensor          # (wy+1, wx)
    chi_c: torch.Tensor          # (wy, wx)
    a_u: torch.Tensor            # (wy, wx+1)   sum_b chi_b * u_solid
    a_v: torch.Tensor            # (wy+1, wx)
    k_u: torch.Tensor            # (nb, 2, wy, wx+1)  [Fx weight, Cm weight]
    k_v: torch.Tensor            # (nb, 2, wy+1, wx)  [Fy weight, Cm weight]
    nbody: int
    thermal: ThermalGeometry | None = None

    @property
    def empty(self) -> bool:
        return self.window[1] <= self.window[0] or self.window[3] <= self.window[2]

    def tensors(self) -> list[torch.Tensor]:
        """Every mask / weight tensor (the live solver copies these in place)."""
        out = [self.chi_u, self.chi_v, self.chi_c, self.a_u, self.a_v, self.k_u, self.k_v]
        if self.thermal is not None:
            out += [t for t in self.thermal if torch.is_tensor(t)]
        return out

    def signature(self) -> tuple:
        """Shapes and static data: equal signatures can be copied into each other."""
        th = None
        if self.thermal is not None:
            th = (self.thermal.temp_ids, self.thermal.flux_ids,
                  tuple(tuple(t.shape) for t in self.thermal if torch.is_tensor(t)))
        return (self.window, self.nbody, tuple(tuple(t.shape) for t in self.tensors()[:7]), th)


def _as_tensor(points, device, dtype) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        return points.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(points, dtype=np.float64), device=device, dtype=dtype)


def body_window(polys: Sequence[np.ndarray], ops, margin_cells: float | None = None) -> tuple[int, int, int, int]:
    """Cell window covering all polygons plus a margin, clipped to the domain."""
    if not polys:
        return (0, 0, 0, 0)
    pts = np.vstack([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys])
    margin = (ops.mask_width + 3.0 + abs(getattr(ops, "mask_shift", 0.0))) if margin_cells is None else margin_cells
    h = ops.h
    i0 = math.floor((pts[:, 0].min() - ops.x0) / h - margin)
    i1 = math.ceil((pts[:, 0].max() - ops.x0) / h + margin)
    j0 = math.floor((pts[:, 1].min() - ops.y0) / h - margin)
    j1 = math.ceil((pts[:, 1].max() - ops.y0) / h + margin)
    i0, i1 = max(i0, 0), min(i1, ops.nx)
    j0, j1 = max(j0, 0), min(j1, ops.ny)
    if i1 <= i0 or j1 <= j0:
        return (0, 0, 0, 0)
    return (j0, j1, i0, i1)


def window_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    """True when cell window ``inner`` lies inside ``outer``."""
    return (outer[0] <= inner[0] and inner[1] <= outer[1] and outer[2] <= inner[2] and inner[3] <= outer[3])


def build_geometry(bodies, ops, *, window: tuple[int, int, int, int] | None = None) -> Geometry:
    """Masks, solid velocities and force weights for a list of bodies.

    ``bodies`` items may be :class:`ImmersedBody`, (N, 2) arrays or (N, 2)
    tensors (tensors may require grad: the result is then differentiable with
    respect to the vertex positions). Rotation data (``omega``, ``center``) is
    taken from ImmersedBody items; bare polygons are static.

    ``window`` forces a particular cell window (it must contain the bodies);
    the live solver uses that to keep tensor shapes - and hence captured CUDA
    graphs - unchanged when a body is rotated slightly.
    """
    dev, dt = ops.device, ops.dtype
    polys, omegas, centers, thermal = [], [], [], []
    for b in bodies:
        if isinstance(b, ImmersedBody):
            polys.append(_as_tensor(b.points, dev, dt))
            omegas.append(b.omega)
            centers.append(b.center)
            # the length is measured below, from the polygon actually used, and only with heat
            # transfer on: the sensitivity sweep hands in vertex tensors that require grad (and
            # may live on the GPU), which ``ImmersedBody.perimeter``'s NumPy cannot read
            thermal.append((b.thermal, b.thermal_value, None))
        else:
            polys.append(_as_tensor(b, dev, dt))
            omegas.append(0.0)
            centers.append((0.0, 0.0))
            thermal.append(("adiabatic", 0.0, None))     # perimeter measured only if needed
    nb = len(polys)
    if window is None:
        window = body_window([p.detach().cpu().numpy() for p in polys], ops)
    j0, j1, i0, i1 = window
    wy, wx = j1 - j0, i1 - i0
    h = ops.h
    if nb == 0 or wy <= 0 or wx <= 0:
        z = torch.zeros(0, 0, device=dev, dtype=dt)
        return Geometry((0, 0, 0, 0), z, z, z, z, z, torch.zeros(0, 2, 0, 0, device=dev, dtype=dt),
                        torch.zeros(0, 2, 0, 0, device=dev, dtype=dt), nb)

    def grid(nr, nc, r0, c0, xoff, yoff):
        ys = ops.y0 + (torch.arange(nr, device=dev, dtype=dt) + r0 + yoff) * h
        xs = ops.x0 + (torch.arange(nc, device=dev, dtype=dt) + c0 + xoff) * h
        return torch.meshgrid(ys, xs, indexing="ij")

    yu, xu = grid(wy, wx + 1, j0, i0, 0.0, 0.5)      # u faces: x = i h, y = (j + 1/2) h
    yv, xv = grid(wy + 1, wx, j0, i0, 0.5, 0.0)      # v faces: x = (i + 1/2) h, y = j h
    yc, xc = grid(wy, wx, j0, i0, 0.5, 0.5)

    w = ops.mask_width * h
    shift = getattr(ops, "mask_shift", 0.0) * h      # mask centre sits this far inside the surface
    phys = getattr(ops, "phys", None)
    want_thermal = phys is not None and getattr(phys, "thermal", False)
    chis_u, chis_v, chis_c, dist_c = [], [], [], []
    vols_u, vols_v = [], []
    for p in polys:
        du_ = signed_distance([p], xu, yu)
        dv_ = signed_distance([p], xv, yv)
        chis_u.append(smooth_step(du_ + shift, w))
        chis_v.append(smooth_step(dv_ + shift, w))
        dc = signed_distance([p], xc, yc) + shift
        chis_c.append(smooth_step(dc, w))
        if want_thermal:
            dist_c.append(dc)
            vols_u.append(smooth_step(du_, w))
            vols_v.append(smooth_step(dv_, w))
    cu = torch.stack(chis_u)          # (nb, wy, wx+1)
    cv = torch.stack(chis_v)
    cc = torch.stack(chis_c)

    if not ops.periodic:
        # boundary faces carry boundary conditions, never penalisation
        fu = torch.ones(wx + 1, device=dev, dtype=dt)
        if i0 == 0:
            fu[0] = 0.0
        if i1 == ops.nx:
            fu[-1] = 0.0
        fv = torch.ones(wy + 1, device=dev, dtype=dt)
        if j0 == 0:
            fv[0] = 0.0
        if j1 == ops.ny:
            fv[-1] = 0.0
        cu = cu * fu[None, None, :]
        cv = cv * fv[None, :, None]
        if vols_u:
            vols_u = [c * fu[None, :] for c in vols_u]
            vols_v = [c * fv[:, None] for c in vols_v]

    chi_u = cu.sum(0).clamp(max=1.0)
    chi_v = cv.sum(0).clamp(max=1.0)
    chi_c = cc.sum(0).clamp(max=1.0)

    om = torch.tensor(omegas, device=dev, dtype=dt)[:, None, None]
    cx = torch.tensor([c[0] for c in centers], device=dev, dtype=dt)[:, None, None]
    cy = torch.tensor([c[1] for c in centers], device=dev, dtype=dt)[:, None, None]
    a_u = (cu * (-om * (yu[None] - cy))).sum(0)      # rigid rotation: u_s = -omega (y - cy)
    a_v = (cv * (om * (xv[None] - cx))).sum(0)       #                 v_s =  omega (x - cx)

    # per-body share of the penalisation at each face (exactly 1 on the body's
    # own support when bodies do not overlap)
    if nb == 1:
        wu, wv = torch.ones_like(cu), torch.ones_like(cv)
    else:
        wu = cu / (cu.sum(0, keepdim=True) + 1e-30)
        wv = cv / (cv.sum(0, keepdim=True) + 1e-30)
    xr, yr = ops.moment_ref
    k_u = torch.stack([wu, wu * (yu[None] - yr)], dim=1)       # Fx ; +(y - yr) Fx part of -Mz
    k_v = torch.stack([wv, -wv * (xv[None] - xr)], dim=1)      # Fy ; -(x - xr) Fy part of -Mz
    tg = None
    if want_thermal:
        extent = (ops.x0, ops.x0 + ops.nx * h, ops.y0, ops.y0 + ops.ny * h)
        for k, (mode, val, _) in enumerate(thermal):
            # the outline length inside the domain (equal to the perimeter for bodies inside it)
            thermal[k] = (mode, val, clipped_length(polys[k].detach().double().cpu().numpy(), extent))
        tg = _thermal_geometry(thermal, cu, cv, cc, dist_c, w, h, dev, dt)
        tg = tg._replace(vol_u=torch.stack(vols_u).sum(0).clamp(max=1.0),
                         vol_v=torch.stack(vols_v).sum(0).clamp(max=1.0))
    return Geometry((j0, j1, i0, i1), chi_u, chi_v, chi_c, a_u, a_v, k_u, k_v, nb, tg)


def clipped_length(points: np.ndarray, extent: tuple[float, float, float, float]) -> float:
    """Length of a closed outline inside the box ``extent`` = (xmin, xmax, ymin, ymax).

    A heat-flux body releases G * (this length) / Pe: only the part of its surface
    inside the domain exchanges heat with the fluid (Liang-Barsky clipping per edge).
    """
    p = np.asarray(points, dtype=np.float64)
    a, b = p, np.roll(p, -1, axis=0)
    xmin, xmax, ymin, ymax = extent
    total = 0.0
    for (x1, y1), (x2, y2) in zip(a, b):
        dx, dy = x2 - x1, y2 - y1
        t0, t1 = 0.0, 1.0
        ok = True
        for pk, qk in ((-dx, x1 - xmin), (dx, xmax - x1), (-dy, y1 - ymin), (dy, ymax - y1)):
            if pk == 0.0:
                if qk < 0.0:
                    ok = False
                    break
                continue
            r = qk / pk
            if pk < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
        if ok and t1 > t0:
            total += (t1 - t0) * math.hypot(dx, dy)
    return total


def skin_weight(d: torch.Tensor, w: float) -> torch.Tensor:
    """``-d chi / d d`` of :func:`smooth_step`: the interface 'delta' whose integral
    across the interface is exactly 1 (it is zero outside the smoothed band)."""
    s = (d / w).clamp(-1.0, 1.0)
    return torch.where(s.abs() < 1.0, (0.25 * math.pi / w) * torch.cos(0.5 * math.pi * s), torch.zeros_like(s))


def _thermal_geometry(thermal, cu, cv, cc, dist_c, w: float, h: float, dev, dt) -> ThermalGeometry:
    """Heat-transfer masks of the bodies (see :class:`ThermalGeometry`)."""
    temp = tuple(i for i, (mode, _, _) in enumerate(thermal) if mode == "temperature")
    flux = tuple(i for i, (mode, _, _) in enumerate(thermal) if mode == "heat_flux")
    other = [i for i in range(len(thermal)) if i not in temp]
    wy, wx = cc.shape[1:]
    zc = torch.zeros(wy, wx, device=dev, dtype=dt)
    if temp:
        ct = cc[list(temp)]
        chi_t = ct.sum(0).clamp(max=1.0)
        th_s = torch.tensor([thermal[i][1] for i in temp], device=dev, dtype=dt)
        a_t = (ct * th_s[:, None, None]).sum(0)
        w_t = ct / (ct.sum(0, keepdim=True) + 1e-30) if len(temp) > 1 else torch.ones_like(ct)
    else:
        chi_t, a_t = zc, zc.clone()
        th_s = torch.zeros(0, device=dev, dtype=dt)
        w_t = torch.zeros(0, wy, wx, device=dev, dtype=dt)
    if other:
        ins_u = (1.0 - cu[other].sum(0)).clamp(0.0, 1.0)
        ins_v = (1.0 - cv[other].sum(0)).clamp(0.0, 1.0)
    else:
        ins_u, ins_v = torch.ones_like(cu[0]), torch.ones_like(cv[0])
    if flux:
        sk = torch.stack([skin_weight(dist_c[i], w) for i in flux])          # (nF, wy, wx)
        w_f = sk / (sk.sum((1, 2), keepdim=True) + 1e-30)
        g = torch.tensor([thermal[i][1] for i in flux], device=dev, dtype=dt)
        per = torch.tensor([thermal[i][2] for i in flux], device=dev, dtype=dt)
        # heat rate of body b = (G_b / Pe) P_b, spread with the normalised skin weight per unit area
        q_skin = (w_f * (g * per)[:, None, None]).sum(0) / (h * h)
    else:
        w_f = torch.zeros(0, wy, wx, device=dev, dtype=dt)
        g = torch.zeros(0, device=dev, dtype=dt)
        q_skin = zc.clone()
    perim = torch.tensor([p for _, _, p in thermal], device=dev, dtype=dt)
    return ThermalGeometry(chi_t, a_t, w_t, ins_u, ins_v, q_skin, w_f, th_s, g, perim, temp, flux)
