"""analyze / polar / inviscid: API contract, sweep strategy, robustness, cancel/progress, threads.

Accuracy against wind-tunnel data lives in ``test_airfoil_validation.py``;
these tests check behaviour, so most of them run on a coarser panelling
(``PANELS``) to stay fast.
"""
from __future__ import annotations

import math
import subprocess
import sys
import threading

import numpy as np
import pytest

from aerolume.model import PolarPoint, PolarResult
from aerolume.solvers.airfoil import AirfoilGeometryError, alpha_range, analyze, inviscid, polar
from aerolume.solvers.airfoil import solver as S

PANELS = 119


def naca4(code: str, n: int = 81, closed: bool = True) -> np.ndarray:
    """Local NACA 4-digit generator (Selig order, CCW, implicitly closed)."""
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


@pytest.fixture(scope="module")
def n2412() -> np.ndarray:
    return naca4("2412")


# ---------------------------------------------------------------------------
# inviscid
# ---------------------------------------------------------------------------

def test_inviscid_symmetric_section():
    p0 = inviscid(naca4("0012"), 0.0)
    p4 = inviscid(naca4("0012"), 4.0)
    m4 = inviscid(naca4("0012"), -4.0)
    assert isinstance(p4, PolarPoint) and p4.converged
    assert abs(p0.cl) < 1e-6 and abs(p0.cm) < 1e-6
    assert m4.cl == pytest.approx(-p4.cl, abs=1e-6)
    # thickness raises the thin-airfoil slope 2*pi by roughly (1 + 0.77 t/c)
    thin = 2 * math.pi * math.radians(4.0)
    assert 1.05 * thin < p4.cl < 1.12 * thin
    assert abs(p4.cm) < 0.01                                   # a.c. near the quarter chord
    assert math.isnan(p4.cd) and p4.cf is None                 # no boundary layer, no drag
    assert p4.cp is not None and p4.x is not None and len(p4.cp) == len(p4.x)
    assert np.max(p4.cp) == pytest.approx(1.0, abs=0.02)        # stagnation point
    assert np.array_equal(p4.cp, p4.cp_inviscid)


def test_inviscid_camber_and_compressibility(n2412):
    p = inviscid(n2412, 0.0)
    assert 0.25 < p.cl < 0.32 and -0.065 < p.cm < -0.045    # camber lift and nose-down moment
    pm = inviscid(n2412, 2.0, mach=0.3)
    pi = inviscid(n2412, 2.0)
    # Karman-Tsien: lift grows a little faster than Prandtl-Glauert 1/sqrt(1 - M^2) = 1.048
    assert 1.04 < pm.cl / pi.cl < 1.07


def test_inviscid_distribution_order(n2412):
    p = inviscid(n2412, 3.0)
    ile = int(np.argmin(p.x))
    assert p.x[0] > 0.99 and p.x[-1] > 0.99
    assert p.y[0] >= p.y[-1]                                  # upper TE first
    assert np.mean(p.y[1:ile]) > np.mean(p.y[ile + 1:-1])     # upper surface first
    assert np.mean(p.cp[5:ile - 5]) < np.mean(p.cp[ile + 5:-5])   # suction side is the upper one


def test_inviscid_independent_of_scale_and_order(n2412):
    # Off the design frame the chord line runs from the TE to the farthest contour point,
    # which on a cambered nose is ~0.1 deg off the NACA chord line (see prepare_coords):
    # compare two off-frame copies, and the design-frame copy with a loose tolerance.
    a = inviscid(n2412 + [0.5, 0.0], 4.0)
    b = inviscid((n2412 * 3.0 + [5.0, 2.0])[::-1], 4.0)
    assert b.cl == pytest.approx(a.cl, abs=1e-6) and b.cm == pytest.approx(a.cm, abs=1e-6)
    assert inviscid(n2412, 4.0).cl == pytest.approx(a.cl, abs=0.02)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def test_analyze_point_contents(n2412, capsys):
    p = analyze(n2412, 1e6, 4.0, panels=PANELS)
    assert capsys.readouterr().out == ""                       # no stdout noise
    assert isinstance(p, PolarPoint) and p.converged and p.alpha == 4.0
    assert 0.6 < p.cl < 0.8 and 0.005 < p.cd < 0.010 and -0.07 < p.cm < -0.03
    assert p.cd == pytest.approx(p.cdp + p.cdf, rel=1e-12)
    assert 0 < p.cdp < p.cd and 0 < p.cdf < p.cd
    assert 0.0 < p.xtr_upper < 0.7 and 0.5 < p.xtr_lower <= 1.0
    n = len(p.x)
    for arr in (p.y, p.cp, p.cp_inviscid, p.cf):
        assert arr is not None and len(arr) == n and np.all(np.isfinite(arr))
    ile = int(np.argmin(p.x))
    # the boundary layer decambers the section: less suction than inviscid on the upper side
    assert np.min(p.cp[:ile]) > np.min(p.cp_inviscid[:ile])
    assert np.median(p.cf[:ile]) > 0                            # attached flow
    assert PolarPoint.from_dict(p.to_dict()).cl == p.cl          # JSON round trip


def test_analyze_inviscid_flag_matches_inviscid(n2412):
    assert analyze(n2412, 1e6, 3.0, viscous=False).cl == pytest.approx(inviscid(n2412, 3.0).cl, abs=1e-12)


def test_forced_transition_moves_transition_forward(n2412):
    free = analyze(n2412, 3e6, 2.0, panels=PANELS)
    trip = analyze(n2412, 3e6, 2.0, panels=PANELS, xtr_upper=0.1, xtr_lower=0.1)
    assert free.converged and trip.converged
    assert trip.xtr_upper <= 0.1 + 0.02 and trip.xtr_lower <= 0.1 + 0.02
    assert free.xtr_upper > 0.2
    assert trip.cd > 1.2 * free.cd                              # more turbulent wetted area


def test_ncrit_moves_transition(n2412):
    lo = analyze(n2412, 1e6, 2.0, panels=PANELS, ncrit=4.0)
    hi = analyze(n2412, 1e6, 2.0, panels=PANELS, ncrit=11.0)
    assert lo.xtr_upper < hi.xtr_upper


def test_any_chord_and_position_give_the_same_answer(n2412):
    a = analyze(n2412, 1e6, 3.0, panels=PANELS)
    b = analyze(n2412 * 0.25, 1e6, 3.0, panels=PANELS)        # scaled about the LE: still the design frame
    assert b.cl == pytest.approx(a.cl, abs=1e-6) and b.cd == pytest.approx(a.cd, rel=1e-5)


def test_blunt_trailing_edge(n2412):
    p = analyze(naca4("2412", closed=False), 1e6, 3.0, panels=PANELS)
    q = analyze(n2412, 1e6, 3.0, panels=PANELS)
    assert p.converged and abs(p.cl - q.cl) < 0.03 and abs(p.cd - q.cd) < 0.1 * q.cd


@pytest.mark.parametrize("kwargs, fragment", [
    ({"re": -1.0}, "Reynolds"),
    ({"mach": 0.95}, "Mach"),
    ({"ncrit": 0.0}, "Ncrit"),
    ({"xtr_upper": -0.1}, "xtr_upper"),
    ({"panels": 10}, "panels"),
])
def test_argument_checks(n2412, kwargs, fragment):
    args = {"re": 1e6, "alpha_deg": 2.0} | kwargs
    with pytest.raises(ValueError, match=fragment):
        analyze(n2412, **args)


def test_non_airfoil_is_rejected():
    box = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    box = np.vstack([np.linspace(box[i], box[(i + 1) % 4], 10, endpoint=False) for i in range(4)])
    with pytest.raises(AirfoilGeometryError, match="Polar analysis needs a single airfoil-like body"):
        polar(box, 1e6, [0.0])


# ---------------------------------------------------------------------------
# polar: sweep strategy and robustness
# ---------------------------------------------------------------------------

def _record_solves(monkeypatch, fail: set[float] = frozenset()):
    """Wrap _Case.solve_viscous: log (alpha, warm) and force failures at given angles."""
    calls: list[tuple[float, bool]] = []
    orig = S._Case.solve_viscous

    def wrapped(self, alpha, *, warm, cancel=None):
        calls.append((round(float(alpha), 6), bool(warm and self.has_bl)))
        if round(float(alpha), 6) in fail:
            self.restore(None)
            return False
        return orig(self, alpha, warm=warm, cancel=cancel)

    monkeypatch.setattr(S._Case, "solve_viscous", wrapped)
    return calls


def test_polar_sweeps_outward_from_zero_with_warm_starts(n2412, monkeypatch):
    calls = _record_solves(monkeypatch)
    alphas = [4.0, -2.0, 1.0, 0.5, 2.0, -1.0, 3.0]            # deliberately unordered
    res = polar(n2412, 1e6, alphas, panels=PANELS)
    assert isinstance(res, PolarResult)
    assert [p.alpha for p in res.points] == alphas            # returned in the caller's order
    assert all(p.converged for p in res.points)
    order = [a for a, _ in calls]
    assert order == [0.5, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0]     # nearest zero first, up, then down
    assert calls[0][1] is False                                # cold start
    assert all(w for _, w in calls[1:])                        # every other point warm-started
    t = res.table()
    assert np.all(np.diff(t["cl"][np.argsort(t["alpha"])]) > 0)
    assert res.wall_seconds > 0 and res.re == 1e6
    assert any("Solver" in n for n in res.notes)


def test_warm_start_matches_cold_start(n2412):
    swept = polar(n2412, 1e6, [0.0, 1.0, 2.0, 3.0], panels=PANELS).points[-1]
    cold = analyze(n2412, 1e6, 3.0, panels=PANELS)
    assert swept.cl == pytest.approx(cold.cl, abs=2e-6)
    assert swept.cd == pytest.approx(cold.cd, rel=2e-5)


def test_failed_point_is_retried_then_marked(n2412, monkeypatch):
    calls = _record_solves(monkeypatch, fail={2.0})
    res = polar(n2412, 1e6, [0.0, 1.0, 2.0, 3.0], panels=PANELS)
    by = {p.alpha: p for p in res.points}
    assert not by[2.0].converged and math.isnan(by[2.0].cl)
    assert by[3.0].converged and by[1.0].converged            # the sweep carried on past it
    tried = [w for a, w in calls if a == 2.0]
    assert len(tried) >= 3                                     # warm, sub-stepped / cold, ramp
    assert any("Not converged at alpha = 2 deg" in w for w in res.warnings)
    assert 2.0 not in res.table()["alpha"]


def test_retry_with_intermediate_angles_recovers(n2412, monkeypatch):
    """A big warm step that fails is split into sub-steps (intermediate alphas)."""
    orig = S._Case.solve_viscous
    seen: list[float] = []

    def flaky(self, alpha, *, warm, cancel=None):
        seen.append(float(alpha))
        if float(alpha) == 6.0 and len(seen) and seen.count(6.0) == 1:
            self.restore(None)
            return False                                       # the direct 0 -> 6 jump "fails" once
        return orig(self, alpha, warm=warm, cancel=cancel)

    monkeypatch.setattr(S._Case, "solve_viscous", flaky)
    res = polar(n2412, 1e6, [0.0, 6.0], panels=PANELS)
    assert all(p.converged for p in res.points)
    assert 3.0 in seen                                         # half-way intermediate angle


def test_deep_stall_points_are_marked_not_raised():
    res = polar(naca4("0006"), 2e5, [0.0, 2.0, 28.0], panels=PANELS)
    by = {p.alpha: p for p in res.points}
    assert by[0.0].converged and by[2.0].converged
    if not by[28.0].converged:                                  # expected; never an exception
        assert any("28 deg" in w for w in res.warnings)


def test_progress_and_cancel(n2412):
    cancel = threading.Event()
    seen: list[tuple[float, str]] = []

    def progress(frac: float, msg: str) -> None:
        seen.append((frac, msg))
        if len(seen) == 2:
            cancel.set()

    res = polar(n2412, 1e6, alpha_range(0, 6, 1), panels=PANELS, progress=progress, cancel=cancel)
    assert len(seen) == 2 and seen[0][0] == pytest.approx(1 / 7) and "alpha = 0 deg" in seen[0][1]
    assert len(res.points) == 2
    assert any("Cancelled after 2 of 7 angles" in w for w in res.warnings)


def test_cancel_before_start_returns_empty(n2412):
    ev = threading.Event()
    ev.set()
    res = polar(n2412, 1e6, [0.0, 1.0], panels=PANELS, cancel=ev)
    assert res.points == [] and any("Cancelled" in w for w in res.warnings)


def test_threads_are_independent_and_deterministic(n2412):
    """Concurrent sweeps (as the app's worker threads run them) give bit-identical results."""
    alphas = [0.0, 2.0]
    ref = polar(n2412, 1e6, alphas, panels=PANELS, distributions=False)
    codes = ("2412", "0012")
    out: dict[int, PolarResult] = {}
    errors: list[BaseException] = []

    def run(k: int) -> None:
        try:
            out[k] = polar(naca4(codes[k]), 1e6, alphas, panels=PANELS, distributions=False)
        except BaseException as exc:          # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(len(codes))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert [p.cl for p in out[0].points] == [p.cl for p in ref.points]
    assert [p.cd for p in out[0].points] == [p.cd for p in ref.points]
    assert abs(out[1].points[0].cl) < 1e-6                     # 0012 at 0 deg: no cross-talk


def test_alpha_range_and_duplicates(n2412):
    a = alpha_range(-2, 2, 0.5)
    assert len(a) == 9 and a[-1] == pytest.approx(2.0)
    res = polar(n2412, 1e6, [1.0, 1.0, 0.0], panels=PANELS)
    assert [p.alpha for p in res.points] == [1.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        polar(n2412, 1e6, [])


def test_import_does_not_pull_in_matplotlib_or_print():
    code = ("import sys; import aerolume.solvers.airfoil as a; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib imported'; "
            "assert 'matplotlib.pyplot' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""
