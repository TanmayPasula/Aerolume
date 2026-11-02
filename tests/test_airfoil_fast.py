"""The vectorised kernels in ``_fast`` against their line-by-line originals in ``_mfoil``.

``_fast`` re-implements mfoil's hot loops (panel influence matrices, wake,
ue-source sensitivities, the boundary-layer residual/Jacobian assembly, the
Newton solve and the transition march). It is only trustworthy if it is the
*same* discretisation, so every kernel is run side by side with the mfoil
original on the same, non-trivial solution state (after a few Newton
iterations, with laminar, turbulent and wake stations present) and compared
to round-off. A coarse panelling keeps the reference (pure-Python) side quick.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from aerolume.solvers.airfoil import _fast as fa
from aerolume.solvers.airfoil import _mfoil as mf
from aerolume.solvers.airfoil import prepare_coords


def _naca(code: str, n: int = 61, closed: bool = False) -> np.ndarray:
    """Local NACA 4-digit outline (Selig order; blunt TE unless closed)."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    x = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, n)))
    a4 = -0.1036 if closed else -0.1015
    yt = t / 0.2 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    up = np.c_[x - yt * np.sin(th), yc + yt * np.cos(th)][::-1]
    lo = np.c_[x + yt * np.sin(th), yc - yt * np.cos(th)][1:]
    pts = np.vstack([up, lo])
    return pts[:-1] if closed else pts


def _model(coords: np.ndarray, *, alpha: float, re: float = 1e6, mach: float = 0.0,
           xtr=(1.0, 1.0), npanel: int = 99):
    with np.errstate(all="ignore"):
        M = mf.mfoil(coords=coords, npanel=npanel)
    M.param.verb = 0
    M.param.doplot = False
    M.geom.chord = 1.0
    M.oper.alpha, M.oper.Re, M.oper.Ma = alpha, re, mach
    M.oper.xtr = np.array(xtr, dtype=float)
    return M


CASES = [
    pytest.param("2412", False, 5.0, 0.0, (1.0, 1.0), id="2412-blunt-free"),
    pytest.param("2412", True, 5.0, 0.0, (1.0, 1.0), id="2412-sharp-free"),
    pytest.param("4412", False, 3.0, 0.3, (1.0, 1.0), id="4412-blunt-mach0.3"),
    pytest.param("0012", False, 4.0, 0.0, (0.3, 0.05), id="0012-blunt-tripped"),
]


@pytest.mark.parametrize("code, closed, alpha, mach, xtr", CASES)
def test_fast_kernels_match_mfoil(code, closed, alpha, mach, xtr):
    xy, _, _ = prepare_coords(_naca(code, closed=closed))
    ref = _model(xy, alpha=alpha, mach=mach, xtr=xtr)
    with np.errstate(all="ignore"):
        # --- inviscid: AIC, 0/90-deg basis solutions, gamma at alpha -------------
        mf.solve_inviscid(ref)
        new = copy.deepcopy(ref)
        cache = fa.GeometryCache(new)
        fa.build_gamma(new, cache)
        assert cache.nogap == closed
        scale = np.abs(ref.isol.gamref).max()
        assert np.abs(new.isol.AIC - ref.isol.AIC).max() < 1e-12 * np.abs(ref.isol.AIC).max()
        assert np.abs(new.isol.gamref - ref.isol.gamref).max() < 1e-10 * scale
        assert np.abs(new.isol.gam - ref.isol.gam).max() < 1e-10 * scale

        # --- wake streamline and inviscid wake ue ---------------------------------
        for m in (ref, new):
            m.oper.viscous = True
            mf.init_thermo(m)
        mf.build_wake(ref)
        fa.build_wake(new, cache)
        assert new.wake.N == ref.wake.N
        assert np.abs(new.wake.x - ref.wake.x).max() < 1e-12
        assert np.abs(new.wake.t - ref.wake.t).max() < 1e-10
        assert np.abs(np.ravel(new.isol.uewi) - np.ravel(ref.isol.uewi)).max() < 1e-10
        assert np.abs(new.isol.uewiref - ref.isol.uewiref).max() < 1e-10

        # --- ue sensitivity to the transpiration sources ----------------------------
        for m in (ref, new):
            mf.stagpoint_find(m)
            mf.identify_surfaces(m)
            mf.set_wake_gap(m)
        mf.calc_ue_m(ref)
        fa.calc_ue_m(new, cache)
        ue_m_ref = np.asarray(ref.vsol.ue_m.toarray() if hasattr(ref.vsol.ue_m, "toarray") else ref.vsol.ue_m)
        # both sides solve with the AIC (LAPACK LU here, mfoil's own solve there): the
        # differences are round-off amplified by its condition number (~1e-11 relative)
        assert np.abs(np.asarray(new.vsol.ue_m) - ue_m_ref).max() < 1e-9 * np.abs(ue_m_ref).max()
        assert np.abs(new.vsol.ue_sigma - ref.vsol.ue_sigma).max() < 1e-9 * np.abs(ref.vsol.ue_sigma).max()

        # --- a few reference Newton iterations to get a non-trivial BL state --------
        mf.init_boundary_layer(ref)
        mf.stagpoint_move(ref)
        for _ in range(3):
            mf.build_glob_sys(ref)
            mf.solve_glob(ref)
            mf.update_state(ref)
            ref.glob.realloc = False
            mf.stagpoint_move(ref)
            mf.update_transition(ref)
        turb = np.asarray(ref.vsol.turb)
        assert turb.any() and not turb.all()        # laminar and turbulent stations both present

        # --- residual + Jacobian assembly -----------------------------------------------
        new = copy.deepcopy(ref)
        mf.build_glob_sys(ref)
        R, R_U = fa.build_glob_sys(new)
        RU_ref = ref.glob.R_U.toarray()
        assert np.abs(R - np.ravel(ref.glob.R)).max() < 1e-12 * max(1.0, np.abs(ref.glob.R).max())
        assert np.abs(R_U - RU_ref).max() < 1e-12 * np.abs(RU_ref).max()

        # --- Newton update -------------------------------------------------------------
        mf.solve_glob(ref)
        fa.solve_glob(new, R_U)
        dU_ref = ref.glob.dU
        assert np.abs(new.glob.dU - dU_ref).max() < 1e-7 * np.abs(dU_ref).max()

        # --- transition march / laminar-turbulent flags ---------------------------------
        mf.update_state(ref)
        ref.glob.realloc = False
        mf.stagpoint_move(ref)
        new = copy.deepcopy(ref)
        mf.update_transition(ref)
        fa.update_transition(new)
        assert np.array_equal(np.asarray(new.vsol.turb), np.asarray(ref.vsol.turb))
        assert np.abs(new.glob.U - ref.glob.U).max() < 1e-12 * np.abs(ref.glob.U).max()
