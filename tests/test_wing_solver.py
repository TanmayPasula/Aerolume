"""Finite-wing solvers: geometry, vortex kernels, VLM / LLT theory checks, conventions, API behaviour.

Fast tests (no 2D viscous solver): viscous coupling is exercised with an analytic
section-polar provider; tests/test_wing_polars.py covers the real cache and
tests/test_wing_validation.py the published reference cases.
"""
from __future__ import annotations

import itertools
import json
import math
import threading

import numpy as np
import pytest

from aerolume.model import Fluid, WingPoint, WingResult, WingSection, WingSpec
from aerolume.solvers.wing import (
    SectionPolarError,
    WingGeometryError,
    analyze_wing,
    freestream_direction,
    planform_metrics,
    wing_polar,
)
from aerolume.solvers.wing.geometry import build_geometry, flat_plate, section_shape, spanwise_nodes
from aerolume.solvers.wing.polars import _make_table, airfoil_key
from aerolume.solvers.wing.vortex import segment_velocity, semi_infinite_velocity

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rect(ar=6.0, chord=1.0, *, sweep=0.0, dihedral=0.0, twist=0.0, airfoil="0012", n_span=16, n_chord=4,
         spacing="cosine", symmetric=True, y0=0.0) -> WingSpec:
    s = 0.5 * ar * chord
    return WingSpec(sections=[
        WingSection(y=y0, chord=chord, airfoil=airfoil),
        WingSection(y=y0 + s, chord=chord, x_le=s * math.tan(math.radians(sweep)),
                    z_le=s * math.tan(math.radians(dihedral)), twist_deg=twist, airfoil=airfoil)],
        symmetric=symmetric, n_chord=n_chord, n_span=n_span, spacing=spacing)


def elliptic(ar, n=16, nc=4, b=1.0) -> WingSpec:
    c0 = 4 * b / (math.pi * ar)
    ys = 0.5 * b * np.sin(np.linspace(0, math.pi / 2, n + 1))
    secs = [WingSection(y=float(y), chord=c0 * math.sqrt(max(0.0, 1 - (2 * y / b) ** 2)),
                        x_le=0.25 * c0 * (1 - math.sqrt(max(0.0, 1 - (2 * y / b) ** 2)))) for y in ys]
    return WingSpec(sections=secs, n_chord=nc, n_span=n)


class FakePolars:
    """Analytic section polars (no mfoil): a 0.1/deg lift curve that stalls at ``stall`` deg,
    cd = 0.006 (1e6/Re)^0.2 (1 + cl^2), cm = -0.05 (cambered) or 0. Duck-types SectionPolars."""

    def __init__(self, stall: float = 12.0, fail: bool = False) -> None:
        self.stall = stall
        self.fail = fail
        self.calls: list[dict] = []
        self.ncrit, self.mach = 9.0, 0.0
        self.coords_provider = None
        self.warnings: list[str] = []
        self.points_computed = 0
        self.seconds = 0.0

    def matches(self, *, ncrit, mach):
        return True

    def coords(self, name):
        from aerolume.geometry.airfoils import naca
        return naca(airfoil_key(name), 200)       # same outline as SectionPolars

    def ensure(self, demands, *, progress=None, cancel=None):
        self.calls.append({k: list(v) for k, v in demands.items()})
        if progress:
            progress(1.0, "fake polars")
        return 0

    def node_res(self, name):
        return [1e6]

    def section(self, name, re, alpha):
        a0 = -2.0 if airfoil_key(name) == "2412" else 0.0
        a = np.asarray(alpha, float)
        cl = 0.1 * (a - a0)
        over = a > self.stall
        cl = np.where(over, 0.1 * (self.stall - a0) - 0.05 * (a - self.stall), cl)
        cd = 0.006 * (1e6 / re) ** 0.2 * (1 + cl ** 2)
        cm = np.full_like(a, -0.05 if a0 else 0.0)
        return cl, cd, cm

    def table(self, name, re):
        if self.fail:
            raise SectionPolarError("nothing converged")
        alpha = np.arange(-14.0, 21.0, 1.0)
        cl, cd, cm = self.section(name, re, alpha)
        return _make_table(alpha, cl, cd, cm, re)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_planform_metrics_rectangle_and_trapezoid():
    m = planform_metrics(rect(6.0, 0.5))
    assert m["area"] == pytest.approx(1.5) and m["span"] == pytest.approx(3.0)
    assert m["aspect_ratio"] == pytest.approx(6.0) and m["mac"] == pytest.approx(0.5)
    assert m["mean_chord"] == pytest.approx(0.5) and m["sweep_c4_deg"] == pytest.approx(0.0)
    cr, lam, s = 1.0, 0.4, 3.0
    w = WingSpec(sections=[WingSection(y=0, chord=cr), WingSection(y=s, chord=cr * lam, x_le=0.5)])
    m = planform_metrics(w)
    assert m["area"] == pytest.approx(2 * s * cr * (1 + lam) / 2)
    assert m["mac"] == pytest.approx(2 / 3 * cr * (1 + lam + lam * lam) / (1 + lam))
    assert m["mac_y"] == pytest.approx(s / 3 * (1 + 2 * lam) / (1 + lam))
    assert m["mac_le_x"] == pytest.approx(0.5 * m["mac_y"] / s)
    assert m["taper"] == pytest.approx(lam)
    half = planform_metrics(WingSpec(sections=w.sections, symmetric=False))
    assert half["area"] == pytest.approx(m["area"] / 2) and half["span"] == pytest.approx(s)


@pytest.mark.parametrize("sections, symmetric, match", [
    ([WingSection(y=0, chord=1)], True, "at least two"),
    ([WingSection(y=0, chord=1), WingSection(y=0, chord=1)], True, "increase"),
    ([WingSection(y=-1, chord=1), WingSection(y=1, chord=1)], True, "right half"),
    ([WingSection(y=0, chord=-1), WingSection(y=1, chord=1)], True, "negative"),
    ([WingSection(y=0, chord=0), WingSection(y=1, chord=0)], True, "no area"),
    ([WingSection(y=0, chord=1), WingSection(y=1, chord=1, twist_deg=60)], True, "twist"),
])
def test_bad_wings_are_rejected_with_reasons(sections, symmetric, match):
    w = WingSpec(sections=sections, symmetric=symmetric)
    with pytest.raises(WingGeometryError, match=match):
        planform_metrics(w)
    with pytest.raises(ValueError):
        wing_polar(w, [0.0], viscous=False)


def test_spanwise_nodes_keep_stations_and_cluster_at_tip():
    y, seg, frac = spanwise_nodes([0.0, 0.7, 2.0], 20, "cosine", root_is_edge=False)
    assert len(y) == 21 and 0.7 in y and y[0] == 0 and y[-1] == 2.0
    assert np.all(np.diff(y) > 0) and set(seg) == {0, 1}
    d = np.diff(y)
    assert d[-1] < d[0] / 5                       # clustered at the tip only
    assert np.all((frac > 0) & (frac < 1))
    y2, _, frac2 = spanwise_nodes([0.0, 1.0], 10, "uniform", root_is_edge=False)
    assert np.allclose(np.diff(y2), 0.1) and np.allclose(frac2, 0.5)
    y3, _, _ = spanwise_nodes([0.0, 1.0], 10, "cosine", root_is_edge=True)
    d3 = np.diff(y3)
    assert d3[0] < d3[5] / 3 and d3[-1] < d3[5] / 3  # both ends are edges
    # more segments than panels: one panel per segment
    y4, seg4, _ = spanwise_nodes(np.linspace(0, 1, 8), 3, "cosine", root_is_edge=False)
    assert len(y4) == 8 and list(seg4) == list(range(7))


def test_section_shape_thin_airfoil_values():
    from aerolume.geometry.airfoils import naca
    s0 = section_shape("0012", naca("0012", 200))
    assert np.max(np.abs(s0.z)) < 1e-9 and abs(s0.alpha_l0) < 1e-9 and abs(s0.cm0) < 1e-9
    assert s0.thickness == pytest.approx(0.12, abs=0.002)
    s1 = section_shape("2412", naca("2412", 200))
    assert math.degrees(s1.alpha_l0) == pytest.approx(-2.08, abs=0.06)    # thin-airfoil theory
    assert s1.cm0 == pytest.approx(-0.053, abs=0.002)
    assert np.max(s1.z) == pytest.approx(0.02, abs=0.001)
    s2 = section_shape("4412", naca("4412", 200))
    assert math.degrees(s2.alpha_l0) == pytest.approx(-4.15, abs=0.1)


def test_geometry_mirrors_twists_and_tilts():
    w = rect(4.0, twist=5.0, dihedral=10.0, n_span=6, n_chord=3)
    g = build_geometry(w, [flat_plate()] * 2)
    assert len(g.surfaces) == 1 and g.surfaces[0].shape == (4, 13, 3)
    st = g.strips
    assert np.allclose(st.y, -st.y[::-1]) and np.allclose(st.chord, st.chord[::-1])
    assert np.allclose(st.area.sum(), 2 * 2.0 / math.cos(math.radians(10)) * 1.0, rtol=1e-9)
    tip = g.surfaces[0][:, -1, :]
    assert tip[-1, 2] < tip[0, 2]                 # +5 deg twist: trailing edge below leading edge (nose up)
    n = np.vstack([c.reshape(-1, 3) for c in g.cp_normals])
    assert np.all(n[:, 2] > 0.9)
    right = st.y > 0
    assert np.all(st.u_n[right, 1] < 0) and np.all(st.u_n[~right, 1] > 0)    # dihedral tilts normals inboard
    # a gap at the centre gives two separate lifting surfaces
    g2 = build_geometry(rect(4.0, y0=0.5), [flat_plate()] * 2)
    assert len(g2.surfaces) == 2 and np.all(np.abs(g2.strips.y) > 0.5)


# ---------------------------------------------------------------------------
# vortex kernels
# ---------------------------------------------------------------------------

def test_biot_savart_limits_and_directions():
    p = np.array([[0.0, 0.0, 0.3]])
    v = segment_velocity(p, np.array([[0.0, -500.0, 0.0]]), np.array([[0.0, 500.0, 0.0]]))[0, 0]
    assert v == pytest.approx([1 / (2 * math.pi * 0.3), 0, 0], rel=1e-5)    # +y line, point above: +x
    vs = semi_infinite_velocity(p, np.array([[0.0, 0.0, 0.0]]), np.array([1.0, 0, 0]), length_scale=1.0)[0, 0]
    assert vs == pytest.approx([0, -1 / (4 * math.pi * 0.3), 0], rel=1e-9)  # half an infinite line
    on_line = segment_velocity(np.array([[0.0, 0.2, 0.0], [0.0, 3.0, 0.0]]), np.array([[0.0, -1.0, 0.0]]),
                               np.array([[0.0, 1.0, 0.0]]))
    assert np.all(on_line == 0)                   # on the segment and on its extension


# ---------------------------------------------------------------------------
# theory checks (inviscid)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ar", [4.0, 8.0])
def test_elliptic_wing_lifting_line_and_lattice(ar):
    w = elliptic(ar)
    llt = wing_polar(w, [3.0], method="llt", viscous=False).points[0]
    vlm = wing_polar(w, [3.0], method="vlm", viscous=False).points[0]
    formula = 2 * math.pi * ar / (ar + 2)
    assert llt.cl / math.radians(3) == pytest.approx(formula, rel=5e-3)
    assert llt.e > 0.999 and vlm.e > 0.99
    # a lifting surface carries less lift than a lifting line at low AR (circle / Falkner checks: validation)
    assert vlm.cl < llt.cl
    # span loading is elliptic where the strips are collocated (semicircle points, not strip centres)
    st = build_geometry(w, [flat_plate()] * len(w.sections)).strips
    eta = 2 * (st.y - 0.5 * st.dy + st.cp_frac * st.dy) / 1.0
    ell = 4 / math.pi * np.sqrt(1 - eta ** 2)
    # (piecewise-constant strip circulation: within a step of the ellipse at the tips; inboard the
    # lifting line is elliptic, the lifting surface of a low-AR elliptic wing very nearly so, e ~ 0.997)
    inboard = np.abs(eta) < 0.7
    for p, tol in ((vlm, 0.035), (llt, 0.01)):
        dev = np.abs(p.c_cl / p.cl - ell)
        assert np.max(dev) < 0.05 and np.max(dev[inboard]) < tol


def test_cambered_wing_reaches_the_2d_limit():
    """AR 40: zero-lift angle and cm tend to thin-airfoil theory (camber slope at the collocation points)."""
    w = rect(40.0, airfoil="2412", n_span=24, n_chord=8)
    r = wing_polar(w, [-2.0, 0.0, 2.0], viscous=False, moment_ref=(0.25, 0, 0))
    cl = np.array([p.cl for p in r.points])
    s = (cl[2] - cl[0]) / 4
    assert -cl[1] / s == pytest.approx(-2.12, abs=0.06)
    assert r.points[1].cm == pytest.approx(-0.052, abs=0.003)


def test_symmetric_wing_without_sideslip_has_no_lateral_loads_and_zero_lift_at_zero_alpha():
    p = analyze_wing(rect(6.0, sweep=20.0, dihedral=5.0), alpha_deg=0.0, viscous=False)
    assert abs(p.cl) < 1e-9 and abs(p.cdi) < 1e-12
    p = analyze_wing(rect(6.0, sweep=20.0, dihedral=5.0), alpha_deg=5.0, viscous=False)
    assert abs(p.cy) < 1e-9 and abs(p.c_roll) < 1e-9 and abs(p.cn) < 1e-9
    assert p.cl > 0 and p.cdi > 0


def test_sideslip_signs_and_antisymmetry():
    w = rect(6.0, dihedral=5.0)
    pp = analyze_wing(w, alpha_deg=4.0, beta_deg=5.0, viscous=False)
    pm = analyze_wing(w, alpha_deg=4.0, beta_deg=-5.0, viscous=False)
    assert pp.cl == pytest.approx(pm.cl, rel=1e-9)
    for k in ("cy", "c_roll", "cn"):
        assert getattr(pp, k) == pytest.approx(-getattr(pm, k), abs=1e-9)
    # beta > 0: the free stream gains a +y component, the -y (left, windward) half of a dihedral wing
    # sees more incidence, so it lifts more: rolling moment about +x is negative
    assert pp.c_roll < -1e-3
    left = pp.y < 0
    assert np.sum(pp.cl_local[left]) > np.sum(pp.cl_local[~left])


def test_freestream_direction_matches_the_rotation_convention():
    assert freestream_direction(0, 0) == pytest.approx([1, 0, 0])
    a, b = math.radians(5), math.radians(10)
    assert freestream_direction(5, 10) == pytest.approx([math.cos(a) * math.cos(b), math.cos(a) * math.sin(b),
                                                         math.sin(a)])


def test_pitching_moment_reference_transfer_and_sign():
    w = rect(6.0, airfoil="2412")
    p1 = analyze_wing(w, alpha_deg=0.0, viscous=False, moment_ref=(0.25, 0, 0))
    p2 = analyze_wing(w, alpha_deg=0.0, viscous=False, moment_ref=(1.25, 0, 0))
    assert p2.cm - p1.cm == pytest.approx(1.0 * p1.cl / 1.0, rel=1e-9)    # lift ahead of the point: nose up
    assert p1.cl > 0 and p2.cm > p1.cm


def test_prandtl_glauert_matches_the_stretched_wing():
    """Göthert's rule: CL(M) of a wing = CL of the wing stretched by 1/beta in x (incompressible) / beta."""
    m = 0.5
    beta = math.sqrt(1 - m * m)
    w = rect(6.0, n_span=12, n_chord=4)
    ws = rect(6.0 / (1 / beta), chord=1 / beta, n_span=12, n_chord=4)       # same span, chord / beta
    p = analyze_wing(w, alpha_deg=1.0, mach=m, viscous=False)
    ps = analyze_wing(ws, alpha_deg=1.0, viscous=False)
    assert p.cl == pytest.approx(ps.cl / beta, rel=2e-3)
    p0 = analyze_wing(w, alpha_deg=1.0, viscous=False)
    assert 1.05 < p.cl / p0.cl < 1 / beta


def test_mach_limits():
    w = rect(6.0)
    r = wing_polar(w, [2.0], mach=0.7, viscous=False)
    assert any("Prandtl" in s for s in r.warnings)
    with pytest.raises(ValueError, match="Mach"):
        wing_polar(w, [2.0], mach=0.95, viscous=False)
    with pytest.raises(ValueError, match="Mach"):
        wing_polar(w, [2.0], mach=-0.1, viscous=False)


def test_lifting_line_warns_outside_its_validity():
    r = wing_polar(rect(6.0, sweep=35.0), [2.0], method="llt", viscous=False)
    assert any("sweep" in s for s in r.warnings)
    r = wing_polar(rect(2.0), [2.0], method="llt", viscous=False)
    assert any("Aspect ratio" in s for s in r.warnings)
    r = wing_polar(rect(6.0), [2.0], method="vlm", viscous=False)
    assert not r.warnings


# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------

def test_input_validation():
    w = rect(6.0)
    for bad in ([], [float("nan")], [50.0]):
        with pytest.raises(ValueError):
            wing_polar(w, bad, viscous=False)
    with pytest.raises(ValueError, match="method"):
        wing_polar(w, [0.0], method="panel", viscous=False)
    with pytest.raises(ValueError, match="speed"):
        wing_polar(w, [0.0], speed=0.0)
    w2 = rect(6.0, airfoil="design:abc123")
    with pytest.raises(ValueError, match="coords_provider"):
        wing_polar(w2, [0.0], viscous=False)


def test_design_airfoils_come_from_the_coords_provider():
    from aerolume.geometry.airfoils import naca
    seen = []

    def provider(name):
        seen.append(name)
        return naca("2412", 200)                  # the outline SectionPolars generates for "2412"

    w = rect(8.0, airfoil="design:abc123")
    r = wing_polar(w, [0.0], viscous=False, coords_provider=provider)
    assert seen == ["design:abc123"] and r.points[0].cl > 0.1       # cambered: lift at zero incidence
    ref = wing_polar(rect(8.0, airfoil="2412"), [0.0], viscous=False)
    assert r.points[0].cl == pytest.approx(ref.points[0].cl, rel=1e-6)


def test_points_follow_the_input_order_and_analyze_wing_agrees():
    w = rect(6.0, airfoil="2412")
    r = wing_polar(w, [4.0, -2.0, 4.0, 0.0], viscous=False)
    assert [p.alpha for p in r.points] == [4.0, -2.0, 4.0, 0.0]
    p = analyze_wing(w, alpha_deg=-2.0, viscous=False)
    assert p.cl == pytest.approx(r.points[1].cl, rel=1e-12)
    assert r.area == pytest.approx(6.0) and r.aspect_ratio == pytest.approx(6.0) and r.mac == pytest.approx(1.0)
    assert r.re_mac == pytest.approx(Fluid().density * 30.0 / Fluid().viscosity)
    assert any("Inviscid" in n for n in r.notes)


def test_span_arrays_and_loading_integral():
    w = rect(6.0, airfoil="2412", n_span=12)
    p = analyze_wing(w, alpha_deg=4.0, viscous=False)
    n = 24
    for k in ("y", "cl_local", "c_cl", "alpha_i"):
        assert getattr(p, k).shape == (n,)
    assert p.cd_local is None and p.cdv == 0.0 and p.cd == p.cdi
    g = build_geometry(w, [flat_plate()] * 2)
    # the mean of c*cl/c_mean over the span is the circulation lift coefficient (~CL)
    assert np.sum(p.c_cl * g.strips.dy) / 6.0 == pytest.approx(p.cl, rel=0.01)
    assert np.all(p.alpha_i > 0)                 # downwash everywhere on a lifting untwisted wing


def test_result_round_trips_through_json():
    r = wing_polar(rect(6.0, airfoil="2412"), [0.0, 4.0], viscous=False, sim_id="s1")
    d = json.loads(json.dumps(r.to_dict()))
    r2 = WingResult.from_dict(d)
    assert r2.points[1].cl == pytest.approx(r.points[1].cl)
    assert np.allclose(r2.points[1].cl_local, r.points[1].cl_local)
    assert r2.table()["alpha"].tolist() == [0.0, 4.0]


def test_progress_and_cancel():
    w = rect(6.0)
    seen = []
    wing_polar(w, [0.0, 2.0, 4.0], viscous=False, progress=lambda f, m: seen.append(f))
    assert seen and seen[-1] == pytest.approx(1.0) and all(b >= a for a, b in itertools.pairwise(seen))
    ev = threading.Event()
    ev.set()
    r = wing_polar(w, [0.0, 2.0], viscous=False, cancel=ev)
    assert r.points == [] and any("Cancelled" in s for s in r.warnings)
    ev2 = threading.Event()

    def stop_after_two(f, m):
        if f >= 0.5:
            ev2.set()

    r = wing_polar(w, [0.0, 1.0, 2.0, 3.0], viscous=False, progress=stop_after_two, cancel=ev2)
    assert 1 <= len(r.points) < 4 and all(p.converged for p in r.points)
    assert analyze_wing(w, alpha_deg=1.0, viscous=False, cancel=ev).converged is False


# ---------------------------------------------------------------------------
# viscous coupling (analytic section polars)
# ---------------------------------------------------------------------------

def test_vlm_strip_drag_is_taken_where_the_polar_gives_the_strip_lift():
    fake = FakePolars()
    w = rect(6.0, chord=0.5, airfoil="2412", n_span=8)
    d = {}
    r = wing_polar(w, [2.0, 6.0], speed=20.0, polar_provider=fake, details=d)
    p = r.points[1]
    re = d["strip_re"]
    for j in (0, 3, 7):
        alpha = (p.cl_local[j] / 0.1) - 2.0                      # inverse of the fake lift curve
        _, cd, _ = fake.section("2412", re[j], alpha)
        assert p.cd_local[j] == pytest.approx(float(cd), rel=2e-3)    # cd(alpha) is linear on the 1 deg grid
    g = build_geometry(w, [flat_plate()] * 2)
    assert p.cdv == pytest.approx(np.sum(p.cd_local * g.strips.area) / 1.5, rel=1e-9)     # S = 1.5 m^2
    assert p.cd == pytest.approx(p.cdi + p.cdv) and p.cdv > 0.005
    # the requested section ranges cover every strip's estimated angle
    (dem,) = fake.calls
    lo = min(a for _, a, _ in dem["2412"])
    hi = max(b for _, _, b in dem["2412"])
    assert lo <= -2.0 + p.cl_local.min() / (2 * math.pi) * 180 / math.pi - 1 and hi >= 4.0
    assert any("Viscous strip theory" in n for n in r.notes)


def test_viscous_lift_is_inviscid_for_the_lattice():
    fake = FakePolars()
    w = rect(6.0, airfoil="2412")
    ref = (0.25, 0.0, 0.5)                                # half a chord above the wing
    pv = analyze_wing(w, alpha_deg=4.0, polar_provider=fake, moment_ref=ref)
    pi = analyze_wing(w, alpha_deg=4.0, viscous=False, moment_ref=ref)
    assert pv.cl == pytest.approx(pi.cl, rel=1e-9) and pv.cdi == pytest.approx(pi.cdi, rel=1e-9)
    # profile drag acting below the reference point pitches the nose down by ~ CDv * 0.5 / c
    assert pv.cm - pi.cm == pytest.approx(-pv.cdv * 0.5 * math.cos(math.radians(4)), rel=0.02)


def test_lifting_line_clips_post_stall_and_estimates_stall_onset():
    fake = FakePolars(stall=8.0)
    w = rect(8.0, airfoil="0012")
    d = {}
    r = wing_polar(w, np.arange(0.0, 17.0, 2.0), method="llt", polar_provider=fake, details=d)
    assert all(p.converged for p in r.points)
    cl = np.array([p.cl for p in r.points])
    assert np.all(np.diff(cl) > -1e-9)                 # clipped, never a post-stall drop
    assert any("clipped" in s for s in r.warnings)
    st = d["stall"]
    assert st is not None and 8.0 < st["alpha"] < 11.0 and 0.6 < st["cl"] < 0.85
    assert any("Stall onset" in n for n in r.notes)
    # below stall the section lift curve is linear (0.1/deg): Prandtl's result for a rectangular wing,
    # CL = a alpha / (1 + a (1 + tau) / (pi AR)) with a small positive Glauert factor tau at AR 8
    a0 = 0.1 * 180 / math.pi
    p4 = r.points[2]
    bounds = [a0 * math.radians(4) / (1 + a0 * (1 + tau) / (math.pi * 8)) for tau in (0.3, 0.0)]
    assert bounds[0] < p4.cl < bounds[1]


def test_vlm_flags_strips_beyond_section_cl_max():
    fake = FakePolars(stall=6.0)
    d = {}
    r = wing_polar(rect(8.0, airfoil="0012"), [2.0, 4.0, 8.0, 12.0], polar_provider=fake, details=d)
    assert any("beyond cl_max" in s for s in r.warnings)
    assert d["stall"] is not None and 4.0 < d["stall"]["alpha"] < 8.0


def test_failed_section_polars_mark_points_unconverged_instead_of_raising():
    r = wing_polar(rect(6.0), [0.0, 2.0], polar_provider=FakePolars(fail=True))
    assert len(r.points) == 2 and not any(p.converged for p in r.points)
    assert all(math.isnan(p.cl) for p in r.points)
    assert any("Not converged" in s for s in r.warnings)
    r = wing_polar(rect(6.0), [0.0, 2.0], method="llt", polar_provider=FakePolars(fail=True))
    assert not any(p.converged for p in r.points) and r.warnings


def test_blended_airfoils_between_sections():
    fake = FakePolars()
    w = WingSpec(sections=[WingSection(y=0, chord=1.0, airfoil="2412"),
                           WingSection(y=4.0, chord=1.0, airfoil="0012")], n_span=12, n_chord=4)
    r = wing_polar(w, [0.0], polar_provider=fake, viscous=False)
    p = r.points[0]
    right = p.y > 0
    cl = p.cl_local[right]
    assert cl[0] > cl[-2] > 0                       # camber fades toward the 0012 tip
    rv = wing_polar(w, [0.0], polar_provider=fake)
    assert set(rv.points[0].cd_local.round(12)) and rv.points[0].converged


def test_wingpoint_defaults_are_unconverged():
    assert WingPoint(alpha=1.0).converged is False


def test_rounded_tip_strips_do_not_decide_the_stall_onset():
    """A tip whose chord falls to zero gets upwash from the neighbouring trailing leg; the lifting line then
    shows local cl far above the inboard values there. Those strips are left out of the stall estimate."""
    secs = [WingSection(y=0.0, chord=1.0)]           # taper 0.4 with a parabolic tip, as NACA TN 1270's wings
    for eta, c in ((0.95, 0.43), (0.97, 0.30), (0.985, 0.16), (0.995, 0.06), (1.0, 0.0)):
        secs.append(WingSection(y=4.0 * eta, chord=c, x_le=0.25 * (1.0 - c)))
    w = WingSpec(sections=secs, n_span=24)
    inv = analyze_wing(w, alpha_deg=6.0, method="llt", viscous=False)
    right = inv.y > 0
    assert inv.cl_local[right][-1] > 1.3 * np.median(inv.cl_local[right])     # the artefact this guards against
    d = {}
    r = wing_polar(w, np.arange(0.0, 15.0, 1.0), method="llt", polar_provider=FakePolars(stall=8.0), details=d)
    assert d["stall"] is not None and abs(d["stall"]["y"]) < 0.95 * 4.0
    assert d["stall"]["alpha"] > 6.0
    assert any("within 5 % of the semispan" in n for n in r.notes)     # the tip strips' clipping is a note


def test_lifting_line_newton_converges_quadratically_on_a_nonlinear_case():
    """Swept, twisted, dihedral wing in sideslip with a nonlinear lift curve: the analytic Jacobian
    (checked against finite differences to 5e-10 during development) gives a handful of iterations."""
    from aerolume.solvers.wing.llt import LiftingLine

    w = WingSpec(sections=[WingSection(y=0, chord=1.0),
                           WingSection(y=4, chord=0.5, x_le=1.0, z_le=0.4, twist_deg=-3)], n_span=12)
    line = LiftingLine(build_geometry(w, [flat_plate()] * 2))
    n = len(line.dA)

    def lift(al):
        return math.pi * np.sin(2 * al), 2 * math.pi * np.cos(2 * al)

    sol = line.solve(freestream_direction(8.0, 4.0), lift, slope0=np.full(n, 2 * math.pi), alpha0=np.zeros(n))
    assert sol.converged and sol.iterations <= 8 and sol.residual < 1e-9
