"""Nonlinear lifting-line method (Phillips & Snyder 2000).

W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic
lifting-line theory", Journal of Aircraft 37(4), 2000. The wing is a chain of
horseshoe vortices whose bound segments lie on the quarter-chord line (between
the spanwise nodes of the lattice) and whose trailing legs follow the free
stream to infinity. At each segment's control point the 3D vortex lifting law
and the 2D section lift are equated:

    R_i = 2 |V_i x dl_i| G_i - |V_i|^2 cl_i(alpha_i) dA_i = 0,
    V_i = v_inf + sum_j v_ij G_j,   alpha_i = atan2(V_i . u_n, V_i . u_a),

with ``cl_i`` the section lift curve (the viscous polar of the strip's airfoil
at its Reynolds number, or thin-airfoil theory 2 pi (alpha - alpha_L0) / beta
for an inviscid run), ``dA_i`` the strip area, ``u_a`` / ``u_n`` the section's
chordwise and normal directions. Unlike Prandtl's linear theory the equations
keep the full local velocity, so sweep, dihedral, sideslip and a nonlinear lift
curve enter directly. The system is solved by Newton's method with the analytic
Jacobian (Phillips & Snyder eq. 13-16) and a backtracking line search, started
from the linearised solution; a damped fixed-point iteration is the fallback.

Control points sit at the semicircle (theta-) midpoints of the bound segments
(see :func:`~aerolume.solvers.wing.geometry.spanwise_nodes`), which makes the
discrete solution of an elliptic wing exactly elliptic.

Validity: straight or moderately swept wings of moderate-to-high aspect ratio
(a lifting line has no chordwise loading: below AR ~4, or beyond ~20 deg of
sweep, use the vortex lattice). Section lift beyond the polar's cl_max is
clipped (held at cl_max); the solution is then *not* a stall prediction and
the caller warns. Compressibility enters only through the section lift curves.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .geometry import WingGeometry
from .vlm import trefftz
from .vortex import segment_velocity, semi_infinite_velocity

LiftFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

NEWTON_MAX_IT = 60
FIXED_POINT_MAX_IT = 3000
RTOL = 1e-9               # converged when max |R_i| / dA_i is below this


@dataclass
class LineSolution:
    """One solved operating point of the lifting line (unit speed and density, body frame)."""
    gamma: np.ndarray            # (n,) strip circulations
    alpha: np.ndarray            # (n,) effective section angle [rad]
    alpha_geo: np.ndarray        # (n,) geometric section angle [rad] (free stream only)
    velocity: np.ndarray         # (n, 3) local velocity at the control points
    force: np.ndarray            # (n, 3) vortex-lifting-law force per strip
    induced_drag: float          # Trefftz-plane induced drag (unit q = 1/2)
    downwash: np.ndarray         # Trefftz normal-wash per strip
    converged: bool
    iterations: int
    residual: float


class LiftingLine:
    """Horseshoe system on the quarter-chord line of a :class:`WingGeometry`."""

    def __init__(self, geom: WingGeometry) -> None:
        self.geom = geom
        st = geom.strips
        a_list, b_list, legs, starts = [], [], [], []
        off = 0
        for q in geom.c4_nodes:
            n = len(q) - 1
            a_list.append(q[:-1])
            b_list.append(q[1:])
            starts.append(q)
            legs.extend((off + j, off + j + 1) for j in range(n))
            off += len(q)
        self.a = np.vstack(a_list)
        self.b = np.vstack(b_list)
        self.leg_start = np.vstack(starts)
        self.strip_legs = np.array(legs, dtype=int)
        self.dl = self.b - self.a
        f = st.cp_frac[:, None]
        self.cp = self.a + f * self.dl
        self.u_a = st.u_a
        self.u_n = st.u_n
        self.u_s = self.dl / np.linalg.norm(self.dl, axis=1, keepdims=True)
        self.dA = st.area
        self.chord = st.chord
        self.length_scale = max(float(np.ptp(self.cp[:, 1])), 1e-9)
        self._bound = segment_velocity(self.cp, self.a, self.b)        # independent of the flow direction
        n = len(self.a)
        # leg -> strip incidence: leg j carries G_{j-1} - G_j of its surface
        self._leg_from = np.zeros((len(self.leg_start), n))
        for s, (la, lb) in enumerate(self.strip_legs):
            self._leg_from[lb, s] += 1.0
            self._leg_from[la, s] -= 1.0

    def influence(self, d: np.ndarray) -> np.ndarray:
        """v_ij (n, n, 3): velocity at control point i of unit horseshoe j with legs along ``d``."""
        semi_b = semi_infinite_velocity(self.cp, self.b, d, length_scale=self.length_scale)
        semi_a = semi_infinite_velocity(self.cp, self.a, d, length_scale=self.length_scale)
        return self._bound + semi_b - semi_a

    def geometric_alpha(self, vhat: np.ndarray) -> np.ndarray:
        """Section angle of attack from the free stream alone [rad]."""
        return np.arctan2(self.u_n @ vhat, self.u_a @ vhat)

    def solve(self, vhat: np.ndarray, lift: LiftFn, *, slope0: np.ndarray, alpha0: np.ndarray,
              gamma0: np.ndarray | None = None) -> LineSolution:
        """Solve the nonlinear lifting-line equations for free-stream direction ``vhat``.

        ``lift(alpha)`` returns (cl, dcl/dalpha) per strip for an array of angles
        [rad]; ``slope0`` [1/rad] and ``alpha0`` [rad] (zero-lift angle) per strip
        define the linearised start; ``gamma0`` overrides it (a neighbouring
        angle's solution in a sweep).
        """
        vhat = np.asarray(vhat, float) / np.linalg.norm(vhat)
        v = self.influence(vhat)
        dl, ua, un, dA = self.dl, self.u_a, self.u_n, self.dA
        n = len(dA)
        a_geo = self.geometric_alpha(vhat)

        def state(g):
            V = vhat[None, :] + np.einsum("ijk,j->ik", v, g)
            w = np.cross(V, dl)
            wn = np.linalg.norm(w, axis=1)
            va, vn = np.einsum("ik,ik->i", V, ua), np.einsum("ik,ik->i", V, un)
            al = np.arctan2(vn, va)
            cl, dcl = lift(al)
            V2 = np.einsum("ik,ik->i", V, V)
            R = 2.0 * wn * g - V2 * cl * dA
            return V, w, wn, va, vn, al, cl, dcl, V2, R

        if gamma0 is None:
            # linearised Prandtl system: 2|vhat x dl| G - dA a sum_j (v_ij . u_n) G_j = dA a (alpha_geo - alpha_0)
            wn0 = np.linalg.norm(np.cross(vhat[None, :], dl), axis=1)
            vn_ij = np.einsum("ijk,ik->ij", v, un)
            A = np.diag(2.0 * wn0) - (dA * slope0)[:, None] * vn_ij
            try:
                g = np.linalg.solve(A, dA * slope0 * (a_geo - alpha0))
            except np.linalg.LinAlgError:
                g = np.zeros(n)
        else:
            g = np.array(gamma0, dtype=float)
        V, w, wn, va, vn, al, cl, dcl, V2, R = state(g)
        scale = np.maximum(dA, 1e-30)
        res = float(np.max(np.abs(R) / scale))
        it = 0
        ok = res < RTOL
        while not ok and it < NEWTON_MAX_IT:
            it += 1
            vxdl = np.cross(v, dl[:, None, :])                               # (n, n, 3): v_ij x dl_i
            dwn = np.einsum("ik,ijk->ij", w, vxdl) / np.maximum(wn, 1e-300)[:, None]
            dal = (va[:, None] * np.einsum("ijk,ik->ij", v, un) - vn[:, None] * np.einsum("ijk,ik->ij", v, ua)) \
                / np.maximum(va * va + vn * vn, 1e-300)[:, None]
            dV2 = 2.0 * np.einsum("ik,ijk->ij", V, v)
            J = 2.0 * g[:, None] * dwn - dA[:, None] * (dV2 * cl[:, None] + V2[:, None] * dcl[:, None] * dal)
            J[np.diag_indices(n)] += 2.0 * wn
            try:
                step = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                break
            lam, best = 1.0, None
            for _ in range(8):
                trial = state(g + lam * step)
                r_t = float(np.max(np.abs(trial[-1]) / scale))
                if best is None or r_t < best[0]:
                    best = (r_t, lam, trial)
                if r_t < res:
                    break
                lam *= 0.5
            res, lam, trial = best
            g = g + lam * step
            V, w, wn, va, vn, al, cl, dcl, V2, R = trial
            ok = res < RTOL
        if not ok:
            # damped fixed point (robust, slow): G <- G + w (|V|^2 cl dA / (2|V x dl|) - G)
            for k in range(FIXED_POINT_MAX_IT):
                target = V2 * cl * dA / (2.0 * np.maximum(wn, 1e-300))
                g = g + 0.05 * (target - g)
                V, w, wn, va, vn, al, cl, dcl, V2, R = state(g)
                res = float(np.max(np.abs(R) / scale))
                if res < 10 * RTOL:
                    ok = True
                    break
            it += k + 1
        force = g[:, None] * np.cross(V, dl)
        g_leg = self._leg_from @ g
        di, wnt = trefftz(self.leg_start, g_leg, self.strip_legs, g, vhat, self.geom.strips.cp_frac)
        return LineSolution(gamma=g, alpha=al, alpha_geo=a_geo, velocity=V, force=force, induced_drag=di,
                            downwash=wnt, converged=bool(ok and np.all(np.isfinite(g))), iterations=it,
                            residual=res)


__all__ = ["LiftingLine", "LineSolution"]
