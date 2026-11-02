"""Vectorised drop-in kernels for the hot paths of the vendored mfoil.

mfoil is written as straightforward per-panel / per-station Python loops, which
makes it easy to read and verify but costs ~6 s per viscous operating point at
200 panels. Profiling shows three hot spots:

* the O(N^2) panel influence loops (``build_gamma``, ``build_wake``,
  ``calc_ue_m``) — ~1.6 s per point, of which the airfoil-only part does not
  even depend on angle of attack;
* ``build_glob_sys`` — ~0.4 s per Newton iteration: ~230 scalar
  ``residual_station`` calls, each building dozens of tiny numpy arrays, plus
  element-wise writes into ``scipy.sparse.lil_matrix``;
* the sparse assembly / solve in ``solve_glob``.

This module re-implements exactly those pieces with numpy broadcasting. Every
function mirrors its mfoil namesake line by line (same formulas, same clipping,
same special cases, same order of the global unknowns) so that results agree
with the reference to round-off; ``tests/test_airfoil_fast.py`` checks each one
against the original on real solution states. Everything that is inherently
sequential (boundary-layer initialisation, stagnation-point moves, transition
marching) stays in ``_mfoil`` untouched.

Conventions follow mfoil: a boundary-layer state is ``U = [th, ds, sa, ue]``
(momentum thickness, displacement thickness, amplification factor n or root
shear stress sqrt(ctau), edge speed). Vectorised states have shape (4, n) and
derivatives w.r.t. the state have shape (4, n), or (8, n) for two-state
(interval) quantities, matching the 1x4 / 1x8 row vectors of the original.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from . import _mfoil as mf

TWO_PI = 2.0 * np.pi
_EP = 1e-9          # mfoil's "point sits on a panel end" threshold
_LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# panel influence kernels (mfoil: panel_info, panel_*_stream, panel_*_velocity)
# ---------------------------------------------------------------------------

class _PanelGeo:
    """Panel-frame geometry of every (target point, panel) pair.

    ``p1``/``p2`` are panel end points, shape (2, P); ``xi`` are target points,
    shape (2, T). All derived arrays broadcast to (T, P).
    """

    __slots__ = ("d", "n", "r1", "r2", "t", "th1", "th2", "x", "z")

    def __init__(self, p1: np.ndarray, p2: np.ndarray, xi: np.ndarray):
        dxp, dzp = p2[0] - p1[0], p2[1] - p1[1]
        d = np.sqrt(dxp ** 2 + dzp ** 2)
        tx, tz = dxp / d, dzp / d
        rx = xi[0][:, None] - p1[0][None, :]
        rz = xi[1][:, None] - p1[1][None, :]
        x = rx * tx + rz * tz
        z = rx * (-tz) + rz * tx
        self.t = (tx, tz)
        self.n = (-tz, tx)
        self.x, self.z, self.d = x, z, d
        self.r1 = np.sqrt(x ** 2 + z ** 2)
        self.r2 = np.sqrt((x - d) ** 2 + z ** 2)
        self.th1 = np.arctan2(z, x)
        self.th2 = np.arctan2(z, x - d)

    def to_global(self, u: np.ndarray, w: np.ndarray, vdir: np.ndarray | None):
        """Panel-frame velocity (u, w) -> global (2, T, P), or dotted with vdir (2, T)."""
        tx, tz = self.t
        nx, nz = self.n
        vx = u * tx + w * nx
        vz = u * tz + w * nz
        if vdir is None:
            return np.stack((vx, vz))
        return vx * vdir[0][:, None] + vz * vdir[1][:, None]


def _safe_log(r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(r < _EP, 0.0, np.log(np.maximum(r, _EP)))


def linvortex_stream(g: _PanelGeo) -> tuple[np.ndarray, np.ndarray]:
    """Streamfunction influence (a, b) of linear-vortex panels (mfoil panel_linvortex_stream)."""
    x, z, d, r1, r2 = g.x, g.z, g.d, g.r1, g.r2
    logr1, logr2 = _safe_log(r1), _safe_log(r2)
    P1 = (0.5 / np.pi) * (z * (g.th2 - g.th1) - d + x * logr1 - (x - d) * logr2)
    P2 = x * P1 + (0.5 / np.pi) * (0.5 * r2 ** 2 * logr2 - 0.5 * r1 ** 2 * logr1 - r2 ** 2 / 4 + r1 ** 2 / 4)
    return P1 - P2 / d, P2 / d


def _source_angles(g: _PanelGeo, branch_cut: bool):
    """mfoil's sequential endpoint special-casing for source panels."""
    th1, th2 = g.th1, g.th2
    if branch_cut:
        th1 = np.where(th1 < 0, th1 + TWO_PI, th1)
        th2 = np.where(th2 < 0, th2 + TWO_PI, th2)
    on1 = g.r1 < _EP
    logr1 = _safe_log(g.r1)
    th1 = np.where(on1, np.pi, th1)
    th2 = np.where(on1, np.pi, th2)
    on2 = g.r2 < _EP
    logr2 = _safe_log(g.r2)
    th1 = np.where(on2, 0.0, th1)
    th2 = np.where(on2, 0.0, th2)
    return logr1, logr2, th1, th2


def constsource_stream(g: _PanelGeo) -> np.ndarray:
    """Streamfunction influence of constant-source panels (mfoil panel_constsource_stream)."""
    logr1, logr2, th1, th2 = _source_angles(g, branch_cut=False)
    x, z, d = g.x, g.z, g.d
    P = (x * (th1 - th2) + d * th2 + z * logr1 - z * logr2) / TWO_PI
    return np.where((th1 + th2) > np.pi, P - 0.25 * d, P + 0.75 * d)


def linsource_stream(g: _PanelGeo) -> tuple[np.ndarray, np.ndarray]:
    """Streamfunction influence (a, b) of linear-source panels (mfoil panel_linsource_stream)."""
    logr1, logr2, th1, th2 = _source_angles(g, branch_cut=True)
    x, z, d, r1, r2 = g.x, g.z, g.d, g.r1, g.r2
    P1 = (0.5 / np.pi) * (x * (th1 - th2) + th2 * d + z * logr1 - z * logr2)
    P2 = x * P1 + (0.5 / np.pi) * (0.5 * r2 ** 2 * th2 - 0.5 * r1 ** 2 * th1 - 0.5 * z * d)
    return P1 - P2 / d, P2 / d


def linvortex_velocity(g: _PanelGeo, vdir=None):
    """Velocity influence (a, b) of linear-vortex panels (mfoil panel_linvortex_velocity, onmid=False)."""
    x, z, d, r1, r2 = g.x, g.z, g.d, g.r1, g.r2
    dth = g.th2 - g.th1
    lr12 = np.log(r1 / r2)
    temp1 = dth / TWO_PI
    temp2 = (2 * z * lr12 - 2 * x * dth) / (4 * np.pi * d)
    ug1, ug2 = temp1 + temp2, -temp2
    temp1 = np.log(r2 / r1) / TWO_PI
    temp2 = (x * lr12 - d + z * dth) / (TWO_PI * d)
    wg1, wg2 = temp1 + temp2, -temp2
    return g.to_global(ug1, wg1, vdir), g.to_global(ug2, wg2, vdir)


def constsource_velocity(g: _PanelGeo, vdir=None):
    """Velocity influence of constant-source panels (mfoil panel_constsource_velocity)."""
    logr1, logr2, th1, th2 = _source_angles(g, branch_cut=False)
    u = (0.5 / np.pi) * (logr1 - logr2)
    w = (0.5 / np.pi) * (th2 - th1)
    return g.to_global(u, w, vdir)


def linsource_velocity(g: _PanelGeo, vdir=None):
    """Velocity influence (a, b) of linear-source panels (mfoil panel_linsource_velocity)."""
    x, z, d, r1, r2 = g.x, g.z, g.d, g.r1, g.r2
    dth = g.th2 - g.th1
    lr12 = np.log(r1 / r2)
    temp1 = lr12 / TWO_PI
    temp2 = (x * lr12 - d + z * dth) / (TWO_PI * d)
    ug1, ug2 = temp1 - temp2, temp2
    temp1 = dth / TWO_PI
    temp2 = (-z * lr12 + x * dth) / (TWO_PI * d)
    wg1, wg2 = temp1 - temp2, temp2
    return g.to_global(ug1, wg1, vdir), g.to_global(ug2, wg2, vdir)


# ---------------------------------------------------------------------------
# inviscid solution (mfoil: build_gamma, inviscid_velocity, build_wake)
# ---------------------------------------------------------------------------

class GeometryCache:
    """Angle-of-attack independent matrices of one paneled airfoil.

    The vortex-panel influence matrix (AIC), its LU factors and the airfoil
    part of the source streamfunction matrix depend only on the panel nodes,
    so a polar sweep builds them once instead of once per angle.
    """

    def __init__(self, M):
        X = M.foil.x
        N = M.foil.N
        self.N = N
        self.X = X.copy()
        A = np.zeros((N + 1, N + 1))
        _, hTE, _, tcp, tdp = mf.TE_info(X)
        nogap = abs(hTE) < 1e-10 * M.geom.chord
        g = _PanelGeo(X[:, :-1], X[:, 1:], X)
        a, b = linvortex_stream(g)
        A[:N, :N - 1] += a
        A[:N, 1:N] += b
        A[:N, N] = -1.0
        rhs = np.zeros((N + 1, 2))
        rhs[:N, 0] = -X[1]
        rhs[:N, 1] = X[0]
        if not nogap:   # a sharp TE has no TE panel (mfoil P9)
            gte = _PanelGeo(X[:, [N - 1]], X[:, [0]], X)
            a = constsource_stream(gte)[:, 0]
            A[:N, 0] += -a * (0.5 * tcp)
            A[:N, N - 1] += a * (0.5 * tcp)
            a, b = linvortex_stream(gte)
            ab = (a + b)[:, 0]
            A[:N, 0] += -ab * (-0.5 * tdp)
            A[:N, N - 1] += ab * (-0.5 * tdp)
        if nogap:
            A[N - 1, :] = 0.0
            A[N - 1, [0, 1, 2, N - 3, N - 2, N - 1]] = [1, -2, 1, -1, 2, -1]
            rhs[N - 1, :] = 0.0    # the extrapolation condition is homogeneous (mfoil P10)
        self.nogap = nogap
        A[N, 0] = 1.0
        A[N, N - 1] = 1.0
        self.AIC = A
        self.lu = scipy.linalg.lu_factor(A)
        self.gamref = scipy.linalg.lu_solve(self.lu, rhs)[:N, :]
        # airfoil-panel constant-source streamfunction block of calc_ue_m's B
        self.B_foil = constsource_stream(_PanelGeo(X[:, :-1], X[:, 1:], X))  # (N, N-1)
        self.te = mf.TE_info(X)


def build_gamma(M, cache: GeometryCache) -> None:
    """Fill M.isol AIC / gamref / gam for the current alpha (mfoil build_gamma)."""
    alpha = M.oper.alpha
    M.isol.AIC = cache.AIC
    M.isol.gamref = cache.gamref.copy()
    M.isol.gam = M.isol.gamref[:, 0] * mf.cosd(alpha) + M.isol.gamref[:, 1] * mf.sind(alpha)


def inviscid_velocity(X, G, Vinf, alpha, pts, dolin=False, te=None):
    """Velocity at points pts (2, T) from vortex strengths G on nodes X (mfoil inviscid_velocity).

    Returns V (2, T) and, with ``dolin``, V_G (2, T, N).
    """
    N = X.shape[1]
    _, hTE, _, tcp, tdp = te if te is not None else mf.TE_info(X)
    g = _PanelGeo(X[:, :-1], X[:, 1:], pts)
    a, b = linvortex_velocity(g)                      # (2, T, N-1)
    V_G = np.zeros((2, pts.shape[1], N))
    V_G[:, :, :-1] += a
    V_G[:, :, 1:] += b
    if abs(hTE) >= 1e-10 * np.ptp(X[0, :]):            # no TE panel on a sharp TE (mfoil P9)
        gte = _PanelGeo(X[:, [N - 1]], X[:, [0]], pts)
        a = constsource_velocity(gte)[:, :, 0]        # (2, T)
        V_G[:, :, 0] += a * (-0.5 * tcp)
        V_G[:, :, N - 1] += a * 0.5 * tcp
        a, b = linvortex_velocity(gte)
        ab = (a + b)[:, :, 0]
        V_G[:, :, 0] += ab * (0.5 * tdp)
        V_G[:, :, N - 1] += ab * (-0.5 * tdp)
    V = V_G @ G
    V = V + Vinf * np.array([mf.cosd(alpha), mf.sind(alpha)])[:, None]
    return (V, V_G) if dolin else V


def build_wake(M, cache: GeometryCache) -> None:
    """Wake streamline, tangents and inviscid wake ue (mfoil build_wake)."""
    N = M.foil.N
    Vinf = M.oper.Vinf
    Nw = int(np.ceil(N / 10 + 10 * M.geom.wakelen))
    S = M.foil.s
    ds1 = 0.5 * (S[1] - S[0] + S[N - 1] - S[N - 2])
    sv = mf.space_geom(ds1, M.geom.wakelen * M.geom.chord, Nw)
    xyw = np.zeros((2, Nw))
    tw = xyw.copy()
    xy1, xyN = M.foil.x[:, 0], M.foil.x[:, N - 1]
    xyte = 0.5 * (xy1 + xyN)
    n = xyN - xy1
    t = np.array([n[1], -n[0]])
    if abs(cache.te[1]) < 1e-10 * M.geom.chord:      # sharp TE: leave along the bisector (mfoil P9)
        t = cache.te[0]
    assert t[0] > 0, 'Wrong wake direction; ensure airfoil points are CCW'
    xyw[:, 0] = xyte + 1e-5 * t * M.geom.chord
    sw = S[N - 1] + sv
    X, gam, alpha, te = M.foil.x, M.isol.gam, M.oper.alpha, cache.te
    for i in range(Nw - 1):
        v1 = inviscid_velocity(X, gam, Vinf, alpha, xyw[:, [i]], te=te)[:, 0]
        v1 = v1 / mf.norm2(v1)
        tw[:, i] = v1
        xyw[:, i + 1] = xyw[:, i] + (sv[i + 1] - sv[i]) * v1
        v2 = inviscid_velocity(X, gam, Vinf, alpha, xyw[:, [i + 1]], te=te)[:, 0]
        v2 = v2 / mf.norm2(v2)
        tw[:, i + 1] = v2
        xyw[:, i + 1] = xyw[:, i] + (sv[i + 1] - sv[i]) * 0.5 * (v1 + v2)
    v = inviscid_velocity(X, gam, Vinf, alpha, xyw, te=te)
    uewi = np.sum(v * tw, axis=0)[:, None]
    uewiref = np.zeros((Nw, 2))
    v = inviscid_velocity(X, M.isol.gamref[:, 0], Vinf, 0, xyw, te=te)
    uewiref[:, 0] = np.sum(v * tw, axis=0)
    v = inviscid_velocity(X, M.isol.gamref[:, 1], Vinf, 90, xyw, te=te)
    uewiref[:, 1] = np.sum(v * tw, axis=0)
    M.wake.N = Nw
    M.wake.x = xyw
    M.wake.s = sw
    M.wake.t = tw
    M.isol.uewi = uewi
    M.isol.uewiref = uewiref


def calc_ue_m(M, cache: GeometryCache) -> None:
    """Sensitivity of ue to the transpiration sources (mfoil calc_ue_m)."""
    N, Nw = M.foil.N, M.wake.N
    assert Nw > 0, 'No wake'
    X, W, Wt = M.foil.x, M.wake.x, M.wake.t

    # Cgam = d(wake uei)/d(gamma)   [Nw x N]
    _, V_G = inviscid_velocity(X, M.isol.gam, 0, 0, W, dolin=True, te=cache.te)
    Cgam = V_G[0] * Wt[0][:, None] + V_G[1] * Wt[1][:, None]

    # B = d(airfoil surf streamfunction)/d(source)  [(N+1) x (N+Nw-2)]
    B = np.zeros((N + 1, N + Nw - 2))
    B[:N, :N - 1] = cache.B_foil
    for j in range(Nw - 1):
        Xj = W[:, [j, j + 1]]
        Xm = 0.5 * (Xj[:, 0] + Xj[:, 1])
        Xj = np.column_stack((Xj[:, 0], Xm, Xj[:, 1]))
        if j == (Nw - 2):
            Xj[:, 2] = 2 * Xj[:, 2] - Xj[:, 1]
        a, b = linsource_stream(_PanelGeo(Xj[:, [0]], Xj[:, [1]], X))
        a, b = a[:, 0], b[:, 0]
        if j > 0:
            B[:N, N - 1 + j] += 0.5 * a + b
            B[:N, N - 1 + j - 1] += 0.5 * a
        else:
            B[:N, N - 1 + j] += b
        a, b = linsource_stream(_PanelGeo(Xj[:, [1]], Xj[:, [2]], X))
        a, b = a[:, 0], b[:, 0]
        B[:N, N - 1 + j] += a + 0.5 * b
        if j < Nw - 2:
            B[:N, N - 1 + j + 1] += 0.5 * b
        else:
            B[:N, N - 1 + j] += 0.5 * b

    if cache.nogap:
        B[N - 1, :] = 0.0      # sharp TE: row N-1 is the gamma-extrapolation condition (mfoil P10)

    # Bp = -inv(AIC) * B = d(airfoil gamma)/d(source)
    Bp = -scipy.linalg.lu_solve(cache.lu, B)
    Bp = Bp[:-1, :]

    # Csig = d(wake uei)/d(source) [Nw x (N+Nw-2)]
    Csig = np.zeros((Nw, N + Nw - 2))
    gfoil = _PanelGeo(X[:, :-1], X[:, 1:], W)
    csv = constsource_velocity(gfoil, Wt)                 # (Nw, N-1)
    Csig[1:, :N - 1] = csv[1:, :]
    Csig[0, 1:N - 2] = csv[0, 1:N - 2]
    # wake sources: for each wake node j, its two half panels act on every wake
    # point at once; the singular self terms (i == j) use mfoil's special values
    ii = np.arange(Nw)
    for j in range(Nw):
        I = [max(j - 1, 0), j, min(j + 1, Nw - 1)]
        Xj = W[:, I].copy()
        Xj[:, 0] = 0.5 * (Xj[:, 0] + Xj[:, 1])
        Xj[:, 2] = 0.5 * (Xj[:, 1] + Xj[:, 2])
        if j == Nw - 1:
            Xj[:, 2] = 2 * Xj[:, 1] - Xj[:, 0]
        d1 = mf.norm2(Xj[:, 1] - Xj[:, 0])
        d2 = mf.norm2(Xj[:, 2] - Xj[:, 1])
        o = ii != j                                         # off-diagonal targets
        if j == 0:
            a, b = linsource_velocity(_PanelGeo(Xj[:, [1]], Xj[:, [2]], W), Wt)
            a, b = a[:, 0], b[:, 0]
            Csig[o, N - 1] += b[o]
            Csig[o, 0] += a[o]
            Csig[o, N - 2] += a[o]
            # self: first point, three panels meet at the TE
            dl = mf.norm2(X[:, 1] - X[:, 0])
            du = mf.norm2(X[:, N - 1] - X[:, N - 2])
            Csig[j, 0] += (0.5 / np.pi) * (np.log(dl / d2) + 1)
            Csig[j, N - 2] += (0.5 / np.pi) * (np.log(du / d2) + 1)
            Csig[j, N - 1] += -0.5 / np.pi
        elif j == Nw - 1:
            a = constsource_velocity(_PanelGeo(Xj[:, [0]], Xj[:, [2]], W), Wt)[:, 0]
            Csig[o, N + Nw - 3] += a[o]
            # self: no self effect of the last (ghost-extended) panel
        else:
            a1, b1 = linsource_velocity(_PanelGeo(Xj[:, [0]], Xj[:, [1]], W), Wt)
            a2, b2 = linsource_velocity(_PanelGeo(Xj[:, [1]], Xj[:, [2]], W), Wt)
            Csig[o, N - 1 + j - 1] += a1[o, 0] + 0.5 * b1[o, 0]
            Csig[o, N - 1 + j] += 0.5 * a2[o, 0] + b2[o, 0]
            aa = (0.25 / np.pi) * np.log(d1 / d2)
            Csig[j, N - 1 + j - 1] += aa + 0.5 / np.pi
            Csig[j, N - 1 + j] += aa - 0.5 / np.pi

    Dw = Cgam @ Bp + Csig
    Dw[0, :] = Bp[-1, :]
    M.vsol.ue_sigma = np.vstack((Bp, Dw))
    rebuild_ue_m(M)


def rebuild_ue_m(M) -> None:
    """ue_m = diag(sgue) ue_sigma sigma_m with a dense sigma_m (mfoil rebuild_ue_m).

    sigma_m is stored as a dense ndarray; mfoil's own rebuild_ue_m (called by
    stagpoint_move when the stagnation panel changes) works unchanged on it.
    """
    N, Nw = M.foil.N, M.wake.N
    sig = np.zeros((N + Nw - 2, N + Nw))
    i = np.arange(N - 1)
    ds = M.foil.s[1:] - M.foil.s[:-1]
    sig[i, i] = -M.isol.sgnue[i] / ds
    sig[i, i + 1] = M.isol.sgnue[i + 1] / ds
    k = np.arange(Nw - 1)
    dsw = M.wake.s[1:] - M.wake.s[:-1]
    sig[N - 1 + k, N + k] = -1.0 / dsw
    sig[N - 1 + k, N + k + 1] = 1.0 / dsw
    M.vsol.sigma_m = sig
    sgue = np.concatenate((M.isol.sgnue, np.ones(Nw)))
    M.vsol.ue_m = sgue[:, None] * (M.vsol.ue_sigma @ sig)


# ---------------------------------------------------------------------------
# vectorised boundary-layer closures (mfoil: the get_* functions)
# ---------------------------------------------------------------------------

class Flags:
    """Per-station flags plus the scalar parameters the closures need.

    Mirrors the fields of mfoil's Param that the closures read; ``turb`` and
    ``wake`` are boolean arrays (one entry per station).
    """

    def __init__(self, param, turb: np.ndarray, wake: np.ndarray):
        self.p = param
        self.turb = np.asarray(turb, dtype=bool)
        self.wake = np.asarray(wake, dtype=bool)
        self.comp = param.Minf > 0

    def with_turb(self, turb) -> Flags:
        return Flags(self.p, np.broadcast_to(turb, self.turb.shape), self.wake)


def _e(n: int, k: int) -> np.ndarray:
    """Unit derivative row e_k, shape (4, n)."""
    out = np.zeros((4, n))
    out[k] = 1.0
    return out


def get_uk(u, f: Flags):
    p = f.p
    if f.comp:
        lam, Vinf = p.KTl, p.Vinf
        den = 1 - lam * (u / Vinf) ** 2
        den_u = -2 * lam * u / Vinf ** 2
        uk = u * (1 - lam) / den
        uk_u = (1 - lam) / den - (uk / den) * den_u
        return uk, uk_u
    return u, np.ones_like(u)


def get_Mach2(U, f: Flags):
    n = U.shape[1]
    if f.comp:
        p = f.p
        H0, g = p.H0, p.gam
        uk, uk_u = get_uk(U[3], f)
        c2 = (g - 1) * (H0 - 0.5 * uk ** 2)
        c2_uk = (g - 1) * (-uk)
        M2 = uk ** 2 / c2
        M2_uk = 2 * uk / c2 - M2 / c2 * c2_uk
        M2_U = np.zeros((4, n))
        M2_U[3] = M2_uk * uk_u
        return M2, M2_U
    return np.zeros(n), np.zeros((4, n))


def get_H(U):
    H = U[1] / U[0]
    H_U = np.zeros_like(U)
    H_U[0] = -H / U[0]
    H_U[1] = 1 / U[0]
    return H, H_U


def get_Hw(U, wgap):
    Hw = wgap / U[0]
    Hw_U = np.zeros_like(U)
    Hw_U[0] = -Hw / U[0]
    return Hw, Hw_U


def get_Hk(U, f: Flags):
    H, H_U = get_H(U)
    if f.comp:
        M2, M2_U = get_Mach2(U, f)
        den = 1 + 0.113 * M2
        Hk = (H - 0.29 * M2) / den
        Hk_U = (H_U - 0.29 * M2_U) / den - Hk / den * 0.113 * M2_U
        return Hk, Hk_U
    return H, H_U


def _clip_hk(Hk, Hk_U, wake):
    """mfoil's 'limit Hk' block: 1.00005 in the wake, 1.05 elsewhere; zero slope when clipped."""
    lim = np.where(wake, 1.00005, 1.05)
    c = Hk < lim
    return np.where(c, lim, Hk), np.where(c, 0.0, Hk_U)


def get_Ret(U, f: Flags):
    p = f.p
    if f.comp:
        M2, M2_U = get_Mach2(U, f)
        uk, uk_u = get_uk(U[3], f)
        H0, gmi, Ts = p.H0, p.gam - 1, p.Tsrat
        Tr = 1 - 0.5 * uk ** 2 / H0
        Tr_uk = -uk / H0
        fs = Tr ** 1.5 * (1 + Ts) / (Tr + Ts)
        f_Tr = 1.5 * fs / Tr - fs / (Tr + Ts)
        mu = p.mu0 * fs
        mu_uk = p.mu0 * f_Tr * Tr_uk
        den = 1 + 0.5 * gmi * M2
        rho = p.rho0 / den ** (1 / gmi)
        rho_U = (-1 / gmi) * rho / den * (0.5 * gmi) * M2_U
        Ret = rho * uk * U[0] / mu
        Ret_U = rho_U * uk * U[0] / mu
        Ret_U[3] += (rho * U[0] / mu - Ret / mu * mu_uk) * uk_u
        Ret_U[0] += rho * uk / mu
        return Ret, Ret_U
    Ret = p.rho0 * U[0] * U[3] / p.mu0
    Ret_U = np.zeros_like(U)
    Ret_U[0] = U[3] / p.mu0     # (sic) mfoil omits rho0 here; rho0 == 1 in mfoil's units
    Ret_U[3] = U[0] / p.mu0
    return Ret, Ret_U


def get_Hs(U, f: Flags):
    Hk, Hk_U = get_Hk(U, f)
    Hk, Hk_U = _clip_hk(Hk, Hk_U, f.wake)
    # turbulent
    Hsmin, dHsinf = 1.5, 0.015
    Ret, Ret_U = get_Ret(U, f)
    big = Ret > 400
    Ho = np.where(big, 3 + 400.0 / Ret, 4.0)
    Ho_U = np.where(big, -400.0 / Ret ** 2 * Ret_U, 0.0)
    small = Ret < 200
    Reb = np.where(small, 200.0, Ret)
    Reb_U = np.where(small, 0.0, Ret_U)
    # attached branch
    Hr = (Ho - Hk) / (Ho - 1)
    Hr_U = (Ho_U - Hk_U) / (Ho - 1) - (Ho - Hk) / (Ho - 1) ** 2 * Ho_U
    aa = (2 - Hsmin - 4 / Reb) * Hr ** 2
    aa_U = (4 / Reb ** 2 * Reb_U) * Hr ** 2 + (2 - Hsmin - 4 / Reb) * 2 * Hr * Hr_U
    Hs_a = Hsmin + 4 / Reb + aa * 1.5 / (Hk + 0.5)
    Hs_a_U = -4 / Reb ** 2 * Reb_U + aa_U * 1.5 / (Hk + 0.5) - aa * 1.5 / (Hk + 0.5) ** 2 * Hk_U
    # separated branch
    lrb = np.log(Reb)
    lrb_U = 1 / Reb * Reb_U
    aa = Hk - Ho + 4 / lrb
    aa_U = Hk_U - Ho_U - 4 / lrb ** 2 * lrb_U
    bb = 0.007 * lrb / aa ** 2 + dHsinf / Hk
    bb_U = 0.007 * (lrb_U / aa ** 2 - 2 * lrb / aa ** 3 * aa_U) - dHsinf / Hk ** 2 * Hk_U
    Hs_s = Hsmin + 4 / Reb + (Hk - Ho) ** 2 * bb
    Hs_s_U = -4 / Reb ** 2 * Reb_U + 2 * (Hk - Ho) * (Hk_U - Ho_U) * bb + (Hk - Ho) ** 2 * bb_U
    att = Hk < Ho
    Hs_t = np.where(att, Hs_a, Hs_s)
    Hs_t_U = np.where(att, Hs_a_U, Hs_s_U)
    M2, M2_U = get_Mach2(U, f)
    den = 1 + 0.014 * M2
    Hs_t = (Hs_t + 0.028 * M2) / den
    Hs_t_U = (Hs_t_U + 0.028 * M2_U) / den - Hs_t / den * 0.014 * M2_U
    # laminar
    a = Hk - 4.35
    num = 0.0111 * a ** 2 - 0.0278 * a ** 3
    lo = Hk < 4.35
    Hs_l = np.where(lo, num / (Hk + 1) + 1.528 - 0.0002 * (a * Hk) ** 2, 0.015 * a ** 2 / Hk + 1.528)
    Hs_Hk = np.where(lo,
                     (0.0111 * 2 * a - 0.0278 * 3 * a ** 2) / (Hk + 1) - num / (Hk + 1) ** 2
                     - 0.0002 * 2 * a * Hk * (Hk + a),
                     0.015 * 2 * a / Hk - 0.015 * a ** 2 / Hk ** 2)
    Hs_l_U = Hs_Hk * Hk_U
    return np.where(f.turb, Hs_t, Hs_l), np.where(f.turb, Hs_t_U, Hs_l_U)


def get_Hss(U, f: Flags):
    M2, M2_U = get_Mach2(U, f)
    Hk, Hk_U = get_Hk(U, f)
    num = 0.064 / (Hk - 0.8) + 0.251
    num_U = -0.064 / (Hk - 0.8) ** 2 * Hk_U
    return M2 * num, M2_U * num + M2 * num_U


def get_de(U, f: Flags):
    Hk, Hk_U = get_Hk(U, f)
    aa = 3.15 + 1.72 / (Hk - 1)
    aa_U = -1.72 / (Hk - 1) ** 2 * Hk_U
    de = U[0] * aa + U[1]
    de_U = U[0] * aa_U
    de_U[0] += aa
    de_U[1] += 1
    cap = de > 12.0 * U[0]
    de = np.where(cap, 12.0 * U[0], de)
    de_U = np.where(cap, 12.0 * _e(U.shape[1], 0), de_U)
    return de, de_U


def get_cf(U, f: Flags):
    p = f.p
    n = U.shape[1]
    Hk, Hk_U = get_Hk(U, f)
    Ret, Ret_U = get_Ret(U, f)
    # turbulent
    M2, M2_U = get_Mach2(U, f)
    Fc = np.sqrt(1 + 0.5 * (p.gam - 1) * M2)
    Fc_U = 0.5 / Fc * 0.5 * (p.gam - 1) * M2_U
    aa = -1.33 * Hk
    aa_U = -1.33 * Hk_U
    sm = aa < -17
    aa = np.where(sm, -20 + 3 * np.exp((aa + 17) / 3), aa)
    aa_U = np.where(sm, (aa + 20) / 3 * aa_U, aa_U)
    bb = np.log(Ret / Fc)
    bb_U = Ret_U / Ret - Fc_U / Fc
    lo = bb < 3
    bb = np.where(lo, 3.0, bb)
    bb_U = np.where(lo, 0.0, bb_U)
    bb = bb / _LN10
    bb_U = bb_U / _LN10
    cc = -1.74 - 0.31 * Hk
    cc_U = -0.31 * Hk_U
    dd = np.tanh(4.0 - Hk / 0.875)
    dd_U = (1 - dd ** 2) * (-Hk_U / 0.875)
    cf0 = 0.3 * np.exp(aa) * bb ** cc
    cf0_U = cf0 * aa_U + 0.3 * np.exp(aa) * cc * bb ** (cc - 1) * bb_U + cf0 * np.log(bb) * cc_U
    cf_t = (cf0 + 1.1e-4 * (dd - 1)) / Fc
    cf_t_U = (cf0_U + 1.1e-4 * dd_U) / Fc - cf_t / Fc * Fc_U
    # laminar
    lo = Hk < 5.5
    num = np.where(lo, 0.0727 * (5.5 - Hk) ** 3 / (Hk + 1) - 0.07,
                   0.015 * (1 - 1.0 / (Hk - 4.5)) ** 2 - 0.07)
    num_Hk = np.where(lo, 0.0727 * (3 * (5.5 - Hk) ** 2 / (Hk + 1) * (-1) - (5.5 - Hk) ** 3 / (Hk + 1) ** 2),
                      0.015 * 2 * (1 - 1.0 / (Hk - 4.5)) / (Hk - 4.5) ** 2)
    cf_l = num / Ret
    cf_l_U = num_Hk / Ret * Hk_U - num / Ret ** 2 * Ret_U
    cf = np.where(f.turb, cf_t, cf_l)
    cf_U = np.where(f.turb, cf_t_U, cf_l_U)
    cf = np.where(f.wake, 0.0, cf)
    cf_U = np.where(f.wake, np.zeros((4, n)), cf_U)
    return cf, cf_U


def get_cfxt(U, x, f: Flags):
    cf, cf_U = get_cf(U, f)
    cfxt = cf * x / U[0]
    cfxt_U = cf_U * x / U[0]
    cfxt_U[0] -= cfxt / U[0]
    return cfxt, cfxt_U, cf / U[0]


def get_Us(U, f: Flags):
    Hs, Hs_U = get_Hs(U, f)
    Hk, Hk_U = get_Hk(U, f)
    H, H_U = get_H(U)
    Hk, Hk_U = _clip_hk(Hk, Hk_U, f.wake)
    bi = 1.0 / f.p.GB
    Us = 0.5 * Hs * (1 - bi * (Hk - 1) / H)
    Us_U = 0.5 * Hs_U * (1 - bi * (Hk - 1) / H) + 0.5 * Hs * (-bi * Hk_U / H + bi * (Hk - 1) / H ** 2 * H_U)
    c = (~f.wake) & (Us > 0.95)
    Us = np.where(c, 0.98, Us)
    Us_U = np.where(c, 0.0, Us_U)
    c = (~f.wake) & (Us > 0.99995)
    Us = np.where(c, 0.99995, Us)
    Us_U = np.where(c, 0.0, Us_U)
    return Us, Us_U


def get_cDi_lam(U, f: Flags):
    Hk, Hk_U = get_Hk(U, f)
    Ret, Ret_U = get_Ret(U, f)
    lo = Hk < 4
    Hk1 = Hk - 4
    with np.errstate(invalid="ignore"):
        num = np.where(lo, 0.00205 * np.abs(4 - Hk) ** 5.5 + 0.207,
                       -0.0016 * Hk1 ** 2 / (1 + 0.02 * Hk1 ** 2) + 0.207)
        num_Hk = np.where(lo, 0.00205 * 5.5 * np.abs(4 - Hk) ** 4.5 * (-1),
                          -0.0016 * (2 * Hk1 / (1 + 0.02 * Hk1 ** 2)
                                     - Hk1 ** 2 / (1 + 0.02 * Hk1 ** 2) ** 2 * 0.02 * 2 * Hk1))
    return num / Ret, num_Hk / Ret * Hk_U - num / Ret ** 2 * Ret_U


def get_cDi_lamwake(U, f: Flags):
    fl = f.with_turb(False)
    Hk, Hk_U = get_Hk(U, fl)
    Hs, Hs_U = get_Hs(U, fl)
    Ret, Ret_U = get_Ret(U, fl)
    HsRet = Hs * Ret
    HsRet_U = Hs_U * Ret + Hs * Ret_U
    num = 2 * 1.1 * (1 - 1 / Hk) ** 2 * (1 / Hk)
    num_Hk = 2 * 1.1 * (2 * (1 - 1 / Hk) * (1 / Hk ** 2) * (1 / Hk) + (1 - 1 / Hk) ** 2 * (-1 / Hk ** 2))
    return num / HsRet, num_Hk * Hk_U / HsRet - num / HsRet ** 2 * HsRet_U


def get_cDi_turbwall(U, f: Flags):
    cf, cf_U = get_cf(U, f)
    Hk, Hk_U = get_Hk(U, f)
    Hs, Hs_U = get_Hs(U, f)
    Us, Us_U = get_Us(U, f)
    Ret, Ret_U = get_Ret(U, f)
    lr = np.log(Ret)
    lr_U = Ret_U / Ret
    Hmin = 1 + 2.1 / lr
    Hmin_U = -2.1 / lr ** 2 * lr_U
    aa = np.tanh((Hk - 1) / (Hmin - 1))
    fac = 0.5 + 0.5 * aa
    fac_U = 0.5 * (1 - aa ** 2) * (Hk_U / (Hmin - 1) - (Hk - 1) / (Hmin - 1) ** 2 * Hmin_U)
    cDi = 0.5 * cf * Us * (2 / Hs) * fac
    cDi_U = cf_U * Us / Hs * fac + cf * Us_U / Hs * fac - cDi / Hs * Hs_U + cf * Us / Hs * fac_U
    w = f.wake
    return np.where(w, 0.0, cDi), np.where(w, 0.0, cDi_U)


def get_cDi_outer(U, f: Flags):
    Hs, Hs_U = get_Hs(U, f)
    Us, Us_U = get_Us(U, f)
    ct = U[2] ** 2
    ct_U = 2 * U[2] * _e(U.shape[1], 2)
    cDi = ct * (0.995 - Us) * 2 / Hs
    cDi_U = ct_U * (0.995 - Us) * 2 / Hs + ct * (-Us_U) * 2 / Hs - ct * (0.995 - Us) * 2 / Hs ** 2 * Hs_U
    t = f.turb
    return np.where(t, cDi, 0.0), np.where(t, cDi_U, 0.0)


def get_cDi_lamstress(U, f: Flags):
    Hs, Hs_U = get_Hs(U, f)
    Us, Us_U = get_Us(U, f)
    Ret, Ret_U = get_Ret(U, f)
    HsRet = Hs * Ret
    HsRet_U = Hs_U * Ret + Hs * Ret_U
    num = 0.15 * (0.995 - Us) ** 2 * 2
    num_Us = 0.15 * 2 * (0.995 - Us) * (-1) * 2
    return num / HsRet, num_Us * Us_U / HsRet - num / HsRet ** 2 * HsRet_U


def get_cDi(U, f: Flags):
    # turbulent (wall + outer + laminar stress, floored by the laminar value)
    c_w, c_w_U = get_cDi_turbwall(U, f)
    c_l, c_l_U = get_cDi_lam(U, f)
    c_lw, c_lw_U = get_cDi_lamwake(U, f)
    cDil = np.where(f.wake, c_lw, c_l)
    cDil_U = np.where(f.wake, c_lw_U, c_l_U)
    c_o, c_o_U = get_cDi_outer(U, f)
    c_s, c_s_U = get_cDi_lamstress(U, f)
    cDi_t = np.where(f.wake, 0.0, c_w) + c_o + c_s
    cDi_t_U = np.where(f.wake, 0.0, c_w_U) + c_o_U + c_s_U
    mx = cDil > cDi_t
    cDi_t = np.where(mx, cDil, cDi_t)
    cDi_t_U = np.where(mx, cDil_U, cDi_t_U)
    cDi_t = np.where(f.wake, 2 * cDi_t, cDi_t)
    cDi_t_U = np.where(f.wake, 2 * cDi_t_U, cDi_t_U)
    return np.where(f.turb, cDi_t, c_l), np.where(f.turb, cDi_t_U, c_l_U)


def get_cDixt(U, x, f: Flags):
    cDi, cDi_U = get_cDi(U, f)
    cDixt = cDi * x / U[0]
    cDixt_U = cDi_U * x / U[0]
    cDixt_U[0] -= cDixt / U[0]
    return cDixt, cDixt_U, cDi / U[0]


def get_cteq(U, f: Flags):
    p = f.p
    CC, C = 0.5 / (p.GA ** 2 * p.GB), p.GC
    Hk, Hk_U = get_Hk(U, f)
    Hs, Hs_U = get_Hs(U, f)
    H, H_U = get_H(U)
    Ret, Ret_U = get_Ret(U, f)
    Us, Us_U = get_Us(U, f)
    Hk, Hk_U = _clip_hk(Hk, Hk_U, f.wake)
    Hkc_n = Hk - 1.0 - C / Ret
    Hkc_n_U = Hk_U + C / Ret ** 2 * Ret_U
    lo = Hkc_n < 0.01
    Hkc_n = np.where(lo, 0.01, Hkc_n)
    Hkc_n_U = np.where(lo, 0.0, Hkc_n_U)
    Hkc = np.where(f.wake, Hk - 1.0, Hkc_n)
    Hkc_U = np.where(f.wake, Hk_U, Hkc_n_U)
    num = CC * Hs * (Hk - 1) * Hkc ** 2
    num_U = CC * (Hs_U * (Hk - 1) * Hkc ** 2 + Hs * Hk_U * Hkc ** 2 + Hs * (Hk - 1) * 2 * Hkc * Hkc_U)
    den = (1 - Us) * H * Hk ** 2
    den_U = (-Us_U) * H * Hk ** 2 + (1 - Us) * H_U * Hk ** 2 + (1 - Us) * H * 2 * Hk * Hk_U
    cteq = np.sqrt(num / den)
    cteq_U = 0.5 / cteq * (num_U / den - num / den ** 2 * den_U)
    return cteq, cteq_U


def get_damp(U, f: Flags):
    n = U.shape[1]
    Hk, Hk_U = get_Hk(U, f)
    Ret, Ret_U = get_Ret(U, f)
    th = U[0]
    c = Hk < 1.05
    Hk = np.where(c, 1.05, Hk)
    Hk_U = np.where(c, 0.0, Hk_U)
    Hmi = 1.0 / (Hk - 1)
    Hmi_U = -Hmi ** 2 * Hk_U
    aa = 2.492 * Hmi ** 0.43
    aa_U = 0.43 * aa / Hmi * Hmi_U
    bb = np.tanh(14 * Hmi - 9.24)
    bb_U = (1 - bb ** 2) * 14 * Hmi_U
    lrc = aa + 0.7 * (bb + 1)
    lrc_U = aa_U + 0.7 * bb_U
    lr = np.log(Ret) / _LN10
    lr_U = (1 / Ret) * Ret_U / _LN10
    dl = 0.1
    rn = (lr - (lrc - dl)) / (2 * dl)
    rn_U = (lr_U - lrc_U) / (2 * dl)
    one = rn >= 1
    rf = np.where(one, 1.0, 3 * rn ** 2 - 2 * rn ** 3)
    rf_U = np.where(one, 0.0, (6 * rn - 6 * rn ** 2) * rn_U)
    ar = 3.87 * Hmi - 2.52
    ar_U = 3.87 * Hmi_U
    ex = np.exp(-ar ** 2)
    ex_U = ex * (-2 * ar * ar_U)
    da = 0.028 * (Hk - 1) - 0.0345 * ex
    da_U = 0.028 * Hk_U - 0.0345 * ex_U
    af = -0.05 + 2.7 * Hmi - 5.5 * Hmi ** 2 + 3 * Hmi ** 3 + 0.1 * np.exp(-20 * Hmi)
    af_U = (2.7 - 11 * Hmi + 9 * Hmi ** 2 - 1 * np.exp(-20 * Hmi)) * Hmi_U
    damp = rf * af * da / th
    damp_U = (rf_U * af * da + rf * af_U * da + rf * af * da_U) / th
    damp_U[0] -= damp / th
    on = lr >= lrc - dl
    damp = np.where(on, damp, 0.0)
    damp_U = np.where(on, damp_U, 0.0)
    nx = 5.0 * (U[2] - f.p.ncrit)
    tnx = np.tanh(nx)
    eex = 1 + tnx
    ed = eex * 0.001 / th
    ed_U = np.zeros((4, n))
    ed_U[2] = (1 - tnx ** 2) * 5.0 * 0.001 / th
    ed_U[0] -= ed / th
    return damp + ed, damp_U + ed_U


def get_upw(U1, U2, f: Flags):
    Hk1, Hk1_U1 = get_Hk(U1, f)
    Hk2, Hk2_U2 = get_Hk(U2, f)
    C = np.where(f.wake, 1.0, 5.0)
    Huc = C * 1.0 / Hk2 ** 2
    Huc_U = np.concatenate((np.zeros_like(Hk1_U1), -2 * Huc / Hk2 * Hk2_U2))
    aa = (Hk2 - 1.0) / (Hk1 - 1.0)
    la = np.log(np.sign(aa) * aa)
    la_U = np.concatenate((-1.0 / (Hk1 - 1.0) * Hk1_U1, 1.0 / (Hk2 - 1.0) * Hk2_U2))
    Hls = la ** 2
    Hls_U = 2 * la * la_U
    big = Hls > 15
    Hls = np.where(big, 15.0, Hls)
    Hls_U = np.where(big, 0.0, Hls_U)
    ex = np.exp(-Hls * Huc)
    upw = 1.0 - 0.5 * ex
    upw_U = -0.5 * ex * (-Hls_U * Huc - Hls * Huc_U)
    return upw, upw_U


def upwind(upw, upw_U, f1, f1_U1, f2, f2_U2):
    f = (1 - upw) * f1 + upw * f2
    f_U = np.concatenate(((1 - upw) * f1_U1, upw * f2_U2))
    if upw_U is not None:
        f_U = f_U + (-upw_U) * f1 + upw_U * f2
    return f, f_U


def get_uq(ds, ds_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, f: Flags):
    p = f.p
    beta = p.GB
    A = np.where(f.wake, p.GA * p.Dlr, p.GA)
    C = np.where(f.wake, 0.0, p.GC)
    Hk, Hk_U = _clip_hk(Hk, Hk_U, f.wake)
    Hkc = Hk - 1.0 - C / Ret
    Hkc_U = Hk_U + C / Ret ** 2 * Ret_U
    lo = Hkc < 0.01
    Hkc = np.where(lo, 0.01, Hkc)
    Hkc_U = np.where(lo, 0.0, Hkc_U)
    ut = 0.5 * cf - (Hkc / (A * Hk)) ** 2
    ut_U = 0.5 * cf_U - 2 * (Hkc / (A * Hk)) * (Hkc_U / (A * Hk) - Hkc / (A * Hk ** 2) * Hk_U)
    uq = ut / (beta * ds)
    uq_U = ut_U / (beta * ds) - uq / ds * ds_U
    return uq, uq_U


def residual_station(f: Flags, x: np.ndarray, Uin: np.ndarray, Aux: np.ndarray):
    """Viscous residuals of n non-transition, non-similarity intervals (mfoil residual_station).

    x: (2, n) xi at the interval ends; Uin: (4, 2, n) states; Aux: (2, n) wake gap.
    Returns R (3, n), R_U (3, 8, n), R_x (3, 2, n).
    """
    U = Uin.copy()
    U[1] -= Aux
    U1, U2 = U[:, 0, :], U[:, 1, :]
    Um = 0.5 * (U1 + U2)
    th, ds, sa = U[0], U[1], U[2]
    n = U.shape[2]
    uk1, uk1_u = get_uk(U1[3], f)
    uk2, uk2_u = get_uk(U2[3], f)

    thlog = np.log(th[1] / th[0])
    thlog_U = np.zeros((8, n))
    thlog_U[0] = -1.0 / th[0]
    thlog_U[4] = 1.0 / th[1]
    uelog = np.log(uk2 / uk1)
    uelog_U = np.zeros((8, n))
    uelog_U[3] = -uk1_u / uk1
    uelog_U[7] = uk2_u / uk2
    xlog = np.log(x[1] / x[0])
    xlog_x = np.stack((-1.0 / x[0], 1.0 / x[1]))
    dx = x[1] - x[0]
    dx_x = np.stack((-np.ones(n), np.ones(n)))

    upw, upw_U = get_upw(U1, U2, f)

    H1, H1_U1 = get_H(U1)
    H2, H2_U2 = get_H(U2)
    H = 0.5 * (H1 + H2)
    H_U = 0.5 * np.concatenate((H1_U1, H2_U2))

    Hs1, Hs1_U1 = get_Hs(U1, f)
    Hs2, Hs2_U2 = get_Hs(U2, f)
    Hs, Hs_U = upwind(0.5, None, Hs1, Hs1_U1, Hs2, Hs2_U2)
    Hslog = np.log(Hs2 / Hs1)
    Hslog_U = np.concatenate((-1.0 / Hs1 * Hs1_U1, 1.0 / Hs2 * Hs2_U2))

    Hw1, Hw1_U1 = get_Hw(U1, Aux[0])
    Hw2, Hw2_U2 = get_Hw(U2, Aux[1])
    Hw = 0.5 * (Hw1 + Hw2)
    Hw_U = 0.5 * np.concatenate((Hw1_U1, Hw2_U2))

    # --- turbulent: shear-lag equation
    salog = np.log(sa[1] / sa[0])
    salog_U = np.zeros((8, n))
    salog_U[2] = -1.0 / sa[0]
    salog_U[6] = 1.0 / sa[1]
    de1, de1_U1 = get_de(U1, f)
    de2, de2_U2 = get_de(U2, f)
    de, de_U = upwind(0.5, None, de1, de1_U1, de2, de2_U2)
    Us1, Us1_U1 = get_Us(U1, f)
    Us2, Us2_U2 = get_Us(U2, f)
    Us, Us_U = upwind(0.5, None, Us1, Us1_U1, Us2, Us2_U2)
    Hk1, Hk1_U1 = get_Hk(U1, f)
    Hk2, Hk2_U2 = get_Hk(U2, f)
    Hk, Hk_U = upwind(upw, upw_U, Hk1, Hk1_U1, Hk2, Hk2_U2)
    Ret1, Ret1_U1 = get_Ret(U1, f)
    Ret2, Ret2_U2 = get_Ret(U2, f)
    Ret, Ret_U = upwind(0.5, None, Ret1, Ret1_U1, Ret2, Ret2_U2)
    cf1, cf1_U1 = get_cf(U1, f)
    cf2, cf2_U2 = get_cf(U2, f)
    cf, cf_U = upwind(upw, upw_U, cf1, cf1_U1, cf2, cf2_U2)
    dsa = 0.5 * (ds[0] + ds[1])
    dsa_U = np.zeros((8, n))
    dsa_U[1] = 0.5
    dsa_U[5] = 0.5
    uq, uq_U = get_uq(dsa, dsa_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, f)
    cteq1, cteq1_U1 = get_cteq(U1, f)
    cteq2, cteq2_U2 = get_cteq(U2, f)
    cteq, cteq_U = upwind(upw, upw_U, cteq1, cteq1_U1, cteq2, cteq2_U2)
    e2 = np.zeros((4, n))
    e2[2] = 1.0
    saa, saa_U = upwind(upw, upw_U, sa[0], e2, sa[1], e2)
    Klag, beta = f.p.SlagK, f.p.GB
    Clag = Klag / beta * 1.0 / (1.0 + Us)
    Clag_U = -Clag / (1.0 + Us) * Us_U
    ald = np.where(f.wake, f.p.Dlr, 1.0)
    Cuq = f.p.Cuq
    Rlag_t = Clag * (cteq - ald * saa) * dx - 2 * de * salog + 2 * de * (uq * dx - uelog) * Cuq
    Rlag_t_U = (Clag_U * (cteq - ald * saa) * dx + Clag * (cteq_U - ald * saa_U) * dx
                - 2 * de_U * salog - 2 * de * salog_U
                + 2 * de_U * (uq * dx - uelog) * Cuq + 2 * de * (uq_U * dx - uelog_U) * Cuq)
    Rlag_t_x = Clag * (cteq - ald * saa) * dx_x + 2 * de * uq * dx_x

    # --- laminar: amplification equation
    damp1, damp1_U1 = get_damp(U1, f)
    damp2, damp2_U2 = get_damp(U2, f)
    damp, damp_U = upwind(0.5, None, damp1, damp1_U1, damp2, damp2_U2)
    Rlag_l = sa[1] - sa[0] - damp * dx
    Rlag_l_U = -damp_U * dx
    Rlag_l_U[2] -= 1.0
    Rlag_l_U[6] += 1.0
    Rlag_l_x = -damp * dx_x

    t = f.turb
    Rlag = np.where(t, Rlag_t, Rlag_l)
    Rlag_U = np.where(t, Rlag_t_U, Rlag_l_U)
    Rlag_x = np.where(t, Rlag_t_x, Rlag_l_x)

    Ms1, Ms1_U1 = get_Mach2(U1, f)
    Ms2, Ms2_U2 = get_Mach2(U2, f)
    Ms, Ms_U = upwind(0.5, None, Ms1, Ms1_U1, Ms2, Ms2_U2)

    cfxt1, cfxt1_U1, cfxt1_x1 = get_cfxt(U1, x[0], f)
    cfxt2, cfxt2_U2, cfxt2_x2 = get_cfxt(U2, x[1], f)
    cfxtm, cfxtm_Um, cfxtm_xm = get_cfxt(Um, 0.5 * (x[0] + x[1]), f)
    cfxt = 0.25 * cfxt1 + 0.5 * cfxtm + 0.25 * cfxt2
    cfxt_U = 0.25 * np.concatenate((cfxt1_U1 + cfxtm_Um, cfxtm_Um + cfxt2_U2))
    cfxt_x = 0.25 * np.stack((cfxt1_x1 + cfxtm_xm, cfxtm_xm + cfxt2_x2))

    Rmom = thlog + (2 + H + Hw - Ms) * uelog - 0.5 * xlog * cfxt
    Rmom_U = thlog_U + (H_U + Hw_U - Ms_U) * uelog + (2 + H + Hw - Ms) * uelog_U - 0.5 * xlog * cfxt_U
    Rmom_x = -0.5 * xlog_x * cfxt - 0.5 * xlog * cfxt_x

    cDixt1, cDixt1_U1, cDixt1_x1 = get_cDixt(U1, x[0], f)
    cDixt2, cDixt2_U2, cDixt2_x2 = get_cDixt(U2, x[1], f)
    cDixt, cDixt_U = upwind(upw, upw_U, cDixt1, cDixt1_U1, cDixt2, cDixt2_U2)
    cDixt_x = np.stack(((1.0 - upw) * cDixt1_x1, upw * cDixt2_x2))
    cfxtu, cfxtu_U = upwind(upw, upw_U, cfxt1, cfxt1_U1, cfxt2, cfxt2_U2)
    cfxtu_x = np.stack(((1.0 - upw) * cfxt1_x1, upw * cfxt2_x2))

    Hss1, Hss1_U1 = get_Hss(U1, f)
    Hss2, Hss2_U2 = get_Hss(U2, f)
    Hss, Hss_U = upwind(0.5, None, Hss1, Hss1_U1, Hss2, Hss2_U2)

    Rshape = Hslog + (2 * Hss / Hs + 1 - H - Hw) * uelog + xlog * (0.5 * cfxtu - cDixt)
    Rshape_U = (Hslog_U + (2 * Hss_U / Hs - 2 * Hss / Hs ** 2 * Hs_U - H_U - Hw_U) * uelog
                + (2 * Hss / Hs + 1 - H - Hw) * uelog_U + xlog * (0.5 * cfxtu_U - cDixt_U))
    Rshape_x = xlog_x * (0.5 * cfxtu - cDixt) + xlog * (0.5 * cfxtu_x - cDixt_x)

    R = np.stack((Rmom, Rshape, Rlag))
    R_U = np.stack((Rmom_U, Rshape_U, Rlag_U))
    R_x = np.stack((Rmom_x, Rshape_x, Rlag_x))
    return R, R_U, R_x


# ---------------------------------------------------------------------------
# global Newton system (mfoil: build_glob_sys, solve_glob)
# ---------------------------------------------------------------------------

def build_glob_sys(M) -> tuple[np.ndarray, np.ndarray]:
    """Residual (3 Nsys) and dense Jacobian dR/dU (3 Nsys x 4 Nsys) (mfoil build_glob_sys).

    Regular intervals of all three surfaces are evaluated in one vectorised
    call; the few special stations (similarity/stagnation, wake start,
    transition) go through mfoil's scalar routines exactly as in the original.
    Stores M.glob.R and returns (R, R_U).
    """
    Nsys = M.glob.Nsys
    R = np.zeros(3 * Nsys)
    R_U = np.zeros((3 * Nsys, 4 * Nsys))
    R_x = np.zeros((3 * Nsys, Nsys))
    Ug = M.glob.U

    rows, xs, Us, Auxs, turbs, wakes = [], [], [], [], [], []
    param = None
    for si in range(3):
        Is = np.asarray(M.vsol.Is[si])
        xi = M.isol.xi[Is]
        N = len(Is)
        U = Ug[:, Is]
        Aux = np.zeros(N)
        param = mf.build_param(M, si)
        if si == 2:
            Aux[:] = M.vsol.wgap
        i0 = 1 if (si < 2) and (xi[0] < 1e-8 * xi[-1]) else 0

        if si < 2:
            Ip = [i0, i0 + 1]
            Ust, Ust_U, Ust_x, xst = mf.stagnation_state(U[:, Ip], xi[Ip])
            param.turb, param.simi = False, True
            R1, R1_Ut, R1_x = mf.residual_station(param, np.r_[xst, xst], np.stack((Ust, Ust), axis=-1),
                                                  Aux[[i0, i0]])
            param.simi = False
            R1_Ust = R1_Ut[:, 0:4] + R1_Ut[:, 4:8]
            R1_U = R1_Ust @ Ust_U
            R1_x = R1_Ust @ Ust_x
            J = [Is[i0], Is[i0 + 1]]
            if i0 == 1:
                Ig = slice(3 * Is[0], 3 * Is[0] + 3)
                R[Ig] = U[0:3, 0] - Ust[0:3]
                R_U[Ig, 4 * Is[0]:(4 * Is[0] + 4)] += np.eye(3, 4)
                R_U[Ig, 4 * J[0]:(4 * J[0] + 4)] -= Ust_U[0:3, 0:4]
                R_U[Ig, 4 * J[1]:(4 * J[1] + 4)] -= Ust_U[0:3, 4:8]
                R_x[Ig, J] = -Ust_x[0:3, :]
        else:
            R1, R1_U, J = mf.wake_sys(M, param)
            R1_x = None
            param.turb, param.wake = True, True

        Ig = slice(3 * Is[i0], 3 * Is[i0] + 3)
        R[Ig] = R1
        for j in range(len(J)):
            R_U[Ig, 4 * J[j]:(4 * J[j] + 4)] += R1_U[:, 4 * j:(4 * j + 4)]
            if R1_x is not None:
                R_x[Ig, J[j]] += R1_x[:, j]

        # march: regular intervals are collected, transition intervals done now
        turb_now = bool(param.turb)
        for i in range(i0 + 1, N):
            Ip = [i - 1, i]
            tran = bool(M.vsol.turb[Is[i - 1]]) ^ bool(M.vsol.turb[Is[i]])
            if tran:
                param.turb = turb_now
                Ri, Ri_U, Ri_x = mf.residual_transition(M, param, xi[Ip], U[:, Ip], Aux[Ip])
                mf.store_transition(M, si, i)
                Ig = slice(3 * Is[i], 3 * Is[i] + 3)
                R[Ig] += Ri
                R_U[Ig, 4 * Is[i - 1]:(4 * Is[i - 1] + 4)] += Ri_U[:, 0:4]
                R_U[Ig, 4 * Is[i]:(4 * Is[i] + 4)] += Ri_U[:, 4:8]
                R_x[Ig, Is[i - 1]] += Ri_x[:, 0]
                R_x[Ig, Is[i]] += Ri_x[:, 1]
                turb_now = True
            else:
                rows.append((Is[i - 1], Is[i]))
                xs.append(xi[Ip])
                Us.append(U[:, Ip])
                Auxs.append(Aux[Ip])
                turbs.append(turb_now)
                wakes.append(si == 2)

    if rows:
        f = Flags(param, np.array(turbs), np.array(wakes))
        x = np.array(xs).T                                   # (2, n)
        Ust = np.transpose(np.array(Us), (1, 2, 0))          # (4, 2, n)
        Aux = np.array(Auxs).T                               # (2, n)
        Rv, Rv_U, Rv_x = residual_station(f, x, Ust, Aux)
        ia = np.array([r[0] for r in rows])
        ib = np.array([r[1] for r in rows])
        for k in range(3):
            R[3 * ib + k] += Rv[k]
            for c in range(4):
                R_U[3 * ib + k, 4 * ia + c] += Rv_U[k, c]
                R_U[3 * ib + k, 4 * ib + c] += Rv_U[k, 4 + c]
            R_x[3 * ib + k, ia] += Rv_x[k, 0]
            R_x[3 * ib + k, ib] += Rv_x[k, 1]

    # stagnation-point sensitivity folded into the ue columns (chain rule)
    x_st = np.concatenate((-M.isol.sgnue, -np.ones(M.wake.N)))
    R_st = R_x @ x_st
    Ist, st_ue = M.isol.Istag, M.isol.sstag_ue
    R_U[:, 4 * Ist[0] + 3] += R_st * st_ue[0]
    R_U[:, 4 * Ist[1] + 3] += R_st * st_ue[1]

    M.glob.R = R
    return R, R_U


def solve_glob(M, R_U: np.ndarray) -> None:
    """Newton update dU from the augmented (BL + edge velocity) system (mfoil solve_glob).

    Prescribed-alpha mode only (Aerolume never uses mfoil's cl-target mode).
    The matrix (~920 x 920 at 200 panels, ~13 % non-zero: banded BL blocks plus
    the two dense ue-source coupling blocks) is solved with SuperLU like the
    original. A dense LAPACK solve is no faster single-threaded, and with a
    many-thread OpenBLAS on a busy machine it was measured 50-100x slower.
    """
    assert not M.oper.givencl, 'cl-constrained mode is not supported by the fast path'
    Nsys = M.glob.Nsys
    ue = M.glob.U[3, :]
    ds = M.glob.U[1, :]
    uemax = np.max(np.abs(ue))
    np.maximum(ue, 1e-10 * uemax, out=ue)       # in place, as in mfoil
    ueinv = mf.get_ueinv(M)
    ue_m = np.asarray(M.vsol.ue_m)
    Rfull = np.concatenate((M.glob.R, ue - (ueinv + ue_m @ (ds * ue))))
    J = np.zeros((4 * Nsys, 4 * Nsys))
    J[:3 * Nsys, :] = R_U
    rows = slice(3 * Nsys, 4 * Nsys)
    J[rows, 3::4] = np.eye(Nsys) - ue_m * ds[None, :]
    J[rows, 1::4] = -ue_m * ue[None, :]
    if not (np.all(np.isfinite(J)) and np.all(np.isfinite(Rfull))):
        raise FloatingPointError('non-finite Newton system')
    dV = -scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(J), Rfull)
    if not np.all(np.isfinite(dV)):
        raise FloatingPointError('singular Newton system')
    M.glob.dU = np.reshape(dV, (4, Nsys), order='F')


# ---------------------------------------------------------------------------
# transition (mfoil: march_amplification, update_transition)
# ---------------------------------------------------------------------------

def march_amplification(M, si: int) -> int:
    """March the e^N amplification equation along surface si (mfoil march_amplification).

    The amplification rate dn/dx depends on the state only through Hk, Re_theta
    and theta — except for mfoil's small "extra amplification near ncrit" term,
    which depends on n itself. So the base rate is evaluated once for every
    station (vectorised, with that term switched off by n = -1e3, where
    tanh saturates to exactly -1), and the per-station Newton solve for n runs
    on plain floats. Same equations, same iteration and step limits as mfoil.
    Returns the index of the last laminar station.
    """
    Is = np.asarray(M.vsol.Is[si])
    N = len(Is)
    param = mf.build_param(M, si)
    param.turb, param.wake = False, False
    U = M.glob.U[:, Is]
    turb = np.asarray(M.vsol.turb)[Is]
    f = Flags(param, np.zeros(N, dtype=bool), np.zeros(N, dtype=bool))
    Ub = U.copy()
    Ub[2, :] = -1e3
    base = get_damp(Ub, f)[0].tolist()
    th = U[0].tolist()
    nsa = U[2].tolist()
    xi = M.isol.xi[Is].tolist()
    ncrit, xift = float(param.ncrit), float(param.xift)
    tanh = math.tanh
    n_prev = 0.0
    i = 1
    while i < N:
        n2 = n_prev * 1.01 if turb[i] else nsa[i]
        dx = xi[i] - xi[i - 1]
        d1 = base[i - 1] + (1 + tanh(5.0 * (n_prev - ncrit))) * 0.001 / th[i - 1]
        b2, th2 = base[i], th[i]
        for it in range(20):
            t2 = tanh(5.0 * (n2 - ncrit))
            d2 = b2 + (1 + t2) * 0.001 / th2
            Ramp = n2 - n_prev - 0.5 * (d1 + d2) * dx
            if abs(Ramp) < 1e-12:
                break
            dU = -Ramp / (1.0 - 0.5 * (1 - t2 * t2) * 5.0 * 0.001 / th2 * dx)
            dmax = 0.5 * (1.01 - it / 20)
            if abs(dU) > dmax:
                dU *= dmax / abs(dU)
            n2 += dU
        if n2 > ncrit or xi[i] > xift:
            break
        M.glob.U[2, Is[i]] = n2
        n_prev = n2
        i += 1
    return i - 1


def update_transition(M) -> None:
    """Update laminar/turbulent flags from the current state (mfoil update_transition)."""
    for si in range(2):
        Is = np.asarray(M.vsol.Is[si])
        N = len(Is)
        param = mf.build_param(M, si)
        turb_side = np.asarray(M.vsol.turb)[Is]
        nz = np.flatnonzero(turb_side)
        ilam0 = int(nz[0]) - 1 if len(nz) else N - 1
        sa = M.glob.U[2, Is].copy()
        ilam = march_amplification(M, si)
        if ilam == ilam0:
            M.glob.U[2, Is] = sa
            continue
        if ilam < ilam0:
            param.turb = True
            sa0, _ = mf.get_cttr(M.glob.U[:, Is[ilam + 1]], param)
            sa1 = M.glob.U[2, Is[ilam0 + 1]] if (ilam0 < N - 1) else sa0
            xi = M.isol.xi[Is]
            dx = xi[min(ilam0 + 1, N - 1)] - xi[ilam + 1]
            for i in range(ilam + 1, ilam0 + 1):
                fr = 0 if (dx == 0) or (i == ilam + 1) else (xi[i] - xi[ilam + 1]) / dx
                if (ilam + 1) == ilam0:
                    fr = 1
                M.glob.U[2, Is[i]] = sa0 + fr * (sa1 - sa0)
                assert M.glob.U[2, Is[i]] > 0, 'negative ctau in update_transition'
                M.vsol.turb[Is[i]] = True
        else:
            for i in range(ilam0, ilam + 1):
                M.vsol.turb[Is[i]] = False
