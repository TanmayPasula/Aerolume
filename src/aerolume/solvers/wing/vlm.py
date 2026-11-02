"""Vortex-lattice method (ring vortices on the mean surface, steady free-stream wake).

Discretisation (Katz & Plotkin, Low-Speed Aerodynamics, 2nd ed., section 12.3):
each panel of the cambered mean surface carries a vortex ring whose leading
segment lies on the panel's quarter-chord line and whose trailing segment lies
on the next panel's quarter-chord line (a quarter panel behind the trailing
edge for the last row); the collocation point is the panel's three-quarter-
chord point (the 1/4-3/4 rule, exact in 2D for flat and parabolic mean lines).
The rings of the trailing-edge row continue into semi-infinite trailing legs
along the free-stream direction, so the steady wake is aligned with the flow at
every angle of attack and sideslip. Neighbouring rings share segments; each
geometric segment is evaluated once with its net circulation.

Kutta condition: implicit — the trailing-edge rings shed their full strength
into the wake legs, so the loading vanishes at the trailing edge.

Forces: Kutta–Joukowski F = rho G (V x dl) on every bound segment (spanwise and
chordwise), with V the free stream plus the velocity induced by the whole
system at the segment midpoint ("near field"); lift, side force and moments come
from these. Induced drag comes from the Trefftz plane instead: the near-field
drag of a vortex lattice converges poorly (it needs the leading-edge suction a
lattice cannot resolve), while the far-field integral of the trailing vortex
sheet is accurate on coarse lattices.

Compressibility: Prandtl–Glauert (Göthert) — influence coefficients are
evaluated on the geometry stretched by 1/beta in x and velocities mapped back
(u / beta, v, w); the tangency condition uses the physical normals. The
circulation, hence the Kutta–Joukowski lift and the Trefftz drag, carry over
unchanged. Valid for subcritical flow (DESIGN.md 7.6: M < 0.6).

Units: the solver works with a unit free-stream speed and unit density; the
caller turns forces into coefficients with q = 1/2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .geometry import WingGeometry
from .vortex import (
    segment_velocity,
    segment_velocity_sum,
    semi_infinite_velocity,
    semi_infinite_velocity_sum,
    stretch,
    to_physical,
)

EVAL_CACHE_PAIRS = 2_500_000      # cache midpoint influences up to this many segment pairs (~60 MB)


@dataclass
class LatticeSolution:
    """One solved operating point of the lattice (unit speed and density, body frame)."""
    gamma: np.ndarray            # ring circulations (n_panels,)
    gamma_te: np.ndarray         # strip (trailing-edge ring) circulations (n_strips,)
    force: np.ndarray            # Kutta–Joukowski force on the whole lattice (3,)
    moment: np.ndarray           # ... and its moment about the reference point (3,)
    induced_drag: float          # Trefftz-plane induced drag (unit q = 1/2), along the free stream
    downwash: np.ndarray         # Trefftz-plane normal-wash per strip (positive up)
    strip_force: np.ndarray      # (n_strips, 3) KJ force per strip


def trefftz(starts: np.ndarray, strengths: np.ndarray, strip_legs: np.ndarray, gamma_strip: np.ndarray,
            vhat: np.ndarray, frac: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Induced drag and normal-wash of a trailing vortex sheet in the Trefftz plane.

    ``starts`` are the leg origins (L, 3) with strengths (L,) along +vhat;
    ``strip_legs`` (S, 2) the two leg indices bounding each strip. The trace is
    the legs projected on the plane normal to ``vhat``; each leg is a 2D point
    vortex there. Returns (D_i, w_n) with D_i = -(rho/2) sum G_s (v.n)_s ds_s
    (rho = 1) and w_n the normal-wash at the strip collocation points, n = vhat x t.
    ``frac`` places those points between the strip's legs (the lattice's
    semicircle control-point fractions; 0.5 = midpoint): evaluating the
    discrete sheet where the lattice collocated keeps the far-field drag
    consistent with the near-field solution (e = 1 for an elliptic wing).
    """
    vh = vhat / np.linalg.norm(vhat)
    up = np.array([0.0, 0.0, 1.0]) - vh[2] * vh
    if np.linalg.norm(up) < 1e-9:                      # flow along z: any perpendicular will do
        up = np.array([1.0, 0.0, 0.0]) - vh[0] * vh
    e2 = up / np.linalg.norm(up)
    e1 = np.cross(e2, vh)
    pts = np.column_stack([starts @ e1, starts @ e2])             # 2D trace coordinates
    a = pts[strip_legs[:, 0]]
    b = pts[strip_legs[:, 1]]
    f = np.full(len(a), 0.5) if frac is None else np.asarray(frac, dtype=float)
    mid = a + f[:, None] * (b - a)
    t = b - a
    ds = np.hypot(t[:, 0], t[:, 1])
    t = t / np.where(ds > 0, ds, 1.0)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])                       # vhat x t, in-plane
    r = mid[:, None, :] - pts[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", r, r)
    lim = (1e-9 * max(float(np.max(ds)), 1e-12)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(r2 > lim, strengths[None, :] / (2.0 * math.pi * r2), 0.0)
    # 2D point vortex along +vhat: v = G/(2 pi r^2) (vhat x r) = G/(2 pi r^2) (-r2 e1 + r1 e2)
    v1 = -np.sum(f * r[:, :, 1], axis=1)
    v2 = np.sum(f * r[:, :, 0], axis=1)
    wn = v1 * n[:, 0] + v2 * n[:, 1]
    di = -0.5 * float(np.sum(gamma_strip * wn * ds))
    return di, wn


class VortexLattice:
    """Influence system of a wing's mean surface; solve it for any free-stream direction.

    The bound part of the influence matrix does not depend on the flow
    direction and is built once; only the wake legs (aligned with the free
    stream) are rebuilt per operating point. ``beta`` is the Prandtl–Glauert
    factor sqrt(1 - M^2).
    """

    def __init__(self, geom: WingGeometry, *, beta: float = 1.0, moment_ref=(0.0, 0.0, 0.0)) -> None:
        self.geom = geom
        self.beta = float(beta)
        self.ref = np.asarray(moment_ref, dtype=float)
        nc = geom.n_chord
        cps, nrm, seg_a, seg_b, wake = [], [], [], [], []
        rows_f, cols_f, vals_f = [], [], []
        rows_w, cols_w, vals_w = [], [], []
        strip_legs = []
        te_panel = []
        seg_strip = []
        p_off = s_off = l_off = strip_off = 0
        for P, CP, CN in zip(geom.surfaces, geom.cp_points, geom.cp_normals):
            nn = P.shape[1]
            nsp = nn - 1
            R = np.empty_like(P)
            R[:-1] = P[:-1] + 0.25 * (P[1:] - P[:-1])
            R[-1] = P[-1] + 0.25 * (P[-1] - P[-2])
            cps.append(CP.reshape(-1, 3))
            nrm.append(CN.reshape(-1, 3))

            def pid(i, j, _off=p_off, _nsp=nsp):
                return _off + i * _nsp + j

            # spanwise (bound) segments Sp[i, j]: G[i,j] - G[i-1,j]
            for i in range(nc):
                for j in range(nsp):
                    seg_a.append(R[i, j])
                    seg_b.append(R[i, j + 1])
                    k = s_off
                    rows_f.append(k), cols_f.append(pid(i, j)), vals_f.append(1.0)
                    if i > 0:
                        rows_f.append(k), cols_f.append(pid(i - 1, j)), vals_f.append(-1.0)
                    seg_strip.append(strip_off + j)
                    s_off += 1
            # chordwise segments Ch[i, j]: G[i,j-1] - G[i,j]
            for i in range(nc):
                for j in range(nn):
                    seg_a.append(R[i, j])
                    seg_b.append(R[i + 1, j])
                    k = s_off
                    if j > 0:
                        rows_f.append(k), cols_f.append(pid(i, j - 1)), vals_f.append(1.0)
                    if j < nsp:
                        rows_f.append(k), cols_f.append(pid(i, j)), vals_f.append(-1.0)
                    seg_strip.append(strip_off + min(j, nsp - 1))
                    s_off += 1
            # wake legs Wk[j] from R[nc, j]: G[nc-1,j-1] - G[nc-1,j]
            for j in range(nn):
                wake.append(R[nc, j])
                if j > 0:
                    rows_w.append(l_off + j), cols_w.append(pid(nc - 1, j - 1)), vals_w.append(1.0)
                if j < nsp:
                    rows_w.append(l_off + j), cols_w.append(pid(nc - 1, j)), vals_w.append(-1.0)
            for j in range(nsp):
                strip_legs.append((l_off + j, l_off + j + 1))
                te_panel.append(pid(nc - 1, j))
            p_off += nc * nsp
            l_off += nn
            strip_off += nsp
        self.n_panels = p_off
        self.cp = np.vstack(cps)
        self.normal = np.vstack(nrm)
        self.seg_a = np.array(seg_a)
        self.seg_b = np.array(seg_b)
        self.seg_mid = 0.5 * (self.seg_a + self.seg_b)
        self.seg_dl = self.seg_b - self.seg_a
        self.seg_strip = np.array(seg_strip, dtype=int)
        self.wake_start = np.array(wake)
        self.strip_legs = np.array(strip_legs, dtype=int)
        self.te_panel = np.array(te_panel, dtype=int)
        self.M = sparse.csr_matrix((vals_f, (rows_f, cols_f)), shape=(len(seg_a), p_off))
        self.Mw = sparse.csr_matrix((vals_w, (rows_w, cols_w)), shape=(len(wake), p_off))
        span = max(float(np.ptp(self.cp[:, 1])), 1e-9)
        self.length_scale = span
        # stretched copies for the influence evaluations
        b = self.beta
        self._cp_s = stretch(self.cp, b)
        self._a_s = stretch(self.seg_a, b)
        self._b_s = stretch(self.seg_b, b)
        self._mid_s = stretch(self.seg_mid, b)
        self._wake_s = stretch(self.wake_start, b)
        v = to_physical(segment_velocity(self._cp_s, self._a_s, self._b_s), b)
        wfix = np.einsum("psk,pk->ps", v, self.normal)
        self._A_fixed = np.asarray((self.M.T @ wfix.T).T)
        # bound-on-bound velocities at the segment midpoints (for Kutta–Joukowski) do not depend on
        # the flow direction either: kept when small (default lattices: ~15 MB), else recomputed per point
        n_pairs = len(self.seg_a) ** 2
        self._v_mid = (to_physical(segment_velocity(self._mid_s, self._a_s, self._b_s), b)
                       if n_pairs <= EVAL_CACHE_PAIRS else None)

    def wake_direction(self, vhat: np.ndarray) -> np.ndarray:
        """Unit wake direction in the stretched space for a physical free-stream direction."""
        d = np.array(vhat, dtype=float)
        d[0] /= self.beta
        return d / np.linalg.norm(d)

    def solve(self, vhat: np.ndarray) -> LatticeSolution:
        """Solve for the unit free-stream direction ``vhat`` (body frame) and integrate the loads."""
        vhat = np.asarray(vhat, dtype=float)
        vhat = vhat / np.linalg.norm(vhat)
        d = self.wake_direction(vhat)
        vw = to_physical(semi_infinite_velocity(self._cp_s, self._wake_s, d, length_scale=self.length_scale),
                         self.beta)
        ww = np.einsum("psk,pk->ps", vw, self.normal)
        A = self._A_fixed + np.asarray((self.Mw.T @ ww.T).T)
        rhs = -self.normal @ vhat
        gamma = np.linalg.solve(A, rhs)
        g_seg = self.M @ gamma
        g_wake = self.Mw @ gamma
        if self._v_mid is not None:
            v_ind = np.einsum("psk,s->pk", self._v_mid, g_seg)
        else:
            v_ind = to_physical(segment_velocity_sum(self._mid_s, self._a_s, self._b_s, g_seg), self.beta)
        v_ind += to_physical(semi_infinite_velocity_sum(self._mid_s, self._wake_s, d, g_wake,
                                                        length_scale=self.length_scale), self.beta)
        v_loc = vhat[None, :] + v_ind
        f_seg = g_seg[:, None] * np.cross(v_loc, self.seg_dl)
        force = f_seg.sum(axis=0)
        moment = np.cross(self.seg_mid - self.ref, f_seg).sum(axis=0)
        strip_force = np.zeros((len(self.te_panel), 3))
        np.add.at(strip_force, self.seg_strip, f_seg)
        gte = gamma[self.te_panel]
        di, wn = trefftz(self.wake_start, g_wake, self.strip_legs, gte, vhat, self.geom.strips.cp_frac)
        return LatticeSolution(gamma=gamma, gamma_te=gte, force=force, moment=moment, induced_drag=di,
                               downwash=wn, strip_force=strip_force)


__all__ = ["LatticeSolution", "VortexLattice", "trefftz"]
