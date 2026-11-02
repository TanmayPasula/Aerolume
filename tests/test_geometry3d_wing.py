"""Wing planform metrics and lofting (aerolume.geometry3d.wing)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import aerolume.geometry3d as G
from aerolume.geometry import naca4
from aerolume.model import WingSection, WingSpec


def _rect(span_half=5.0, chord=1.0, airfoil="0012", **kw) -> WingSpec:
    return WingSpec([WingSection(0.0, chord, airfoil=airfoil), WingSection(span_half, chord, airfoil=airfoil)], **kw)


def _tapered_swept() -> WingSpec:
    t30 = math.tan(math.radians(30.0))
    return WingSpec([WingSection(0.0, 2.0, 0.0, 0.0, 0.0, "2412"), WingSection(5.0, 1.0, 5.0 * t30, 0.0, -3.0, "0012")])


# ---------------------------------------------------------------------------
# planform metrics vs hand calculation
# ---------------------------------------------------------------------------

def test_planform_rectangular():
    p = G.planform(_rect())
    assert p["area"] == pytest.approx(10.0) and p["span"] == pytest.approx(10.0)
    assert p["aspect_ratio"] == pytest.approx(10.0) and p["mac"] == pytest.approx(1.0)
    assert p["mac_le_x"] == pytest.approx(0.0) and p["mac_y"] == pytest.approx(2.5)
    assert p["taper"] == pytest.approx(1.0) and p["sweep_c4_deg"] == pytest.approx(0.0, abs=1e-12)
    assert p["mean_chord"] == pytest.approx(1.0)


def test_planform_tapered_swept_matches_textbook_formulas():
    p = G.planform(_tapered_swept())
    lam, cr, b = 0.5, 2.0, 10.0
    S = 0.5 * (cr + cr * lam) * b
    assert p["area"] == pytest.approx(S) and p["span"] == pytest.approx(b)
    assert p["aspect_ratio"] == pytest.approx(b * b / S)
    assert p["taper"] == pytest.approx(lam)
    mac = 2 / 3 * cr * (1 + lam + lam**2) / (1 + lam)                     # 1.5556
    ymac = b / 6 * (1 + 2 * lam) / (1 + lam)                              # 2.2222
    assert p["mac"] == pytest.approx(mac) and p["mac_y"] == pytest.approx(ymac)
    assert p["mac_le_x"] == pytest.approx(ymac * math.tan(math.radians(30)))
    tan_c4 = math.tan(math.radians(30)) - 4 / p["aspect_ratio"] * 0.25 * (1 - lam) / (1 + lam)
    assert p["sweep_c4_deg"] == pytest.approx(math.degrees(math.atan(tan_c4)))   # 27.80 deg
    assert p["sweep_c4_deg"] == pytest.approx(27.80, abs=0.01)
    assert p["sweep_le_deg"] == pytest.approx(30.0)


def test_planform_cranked_wing_matches_numerical_integration():
    w = WingSpec([WingSection(0.0, 2.0, 0.0), WingSection(1.5, 1.6, 0.3, 0.1), WingSection(6.0, 0.6, 1.9, 0.5)])
    p = G.planform(w)
    y = np.linspace(0, 6, 600001)
    c = np.interp(y, [0, 1.5, 6], [2.0, 1.6, 0.6])
    x = np.interp(y, [0, 1.5, 6], [0.0, 0.3, 1.9])
    Sh = np.trapezoid(c, y)
    assert p["area"] == pytest.approx(2 * Sh, rel=1e-9)
    assert p["mac"] == pytest.approx(np.trapezoid(c * c, y) / Sh, rel=1e-9)
    assert p["mac_le_x"] == pytest.approx(np.trapezoid(c * x, y) / Sh, rel=1e-9)
    assert p["mac_y"] == pytest.approx(np.trapezoid(c * y, y) / Sh, rel=1e-9)
    assert p["dihedral_deg"] == pytest.approx(math.degrees(math.atan2(0.5, 6.0)))


def test_planform_non_symmetric_and_validation():
    w = WingSpec([WingSection(-2.0, 1.0), WingSection(3.0, 1.0)], symmetric=False)
    p = G.planform(w)
    assert p["span"] == pytest.approx(5.0) and p["area"] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        G.planform(WingSpec([WingSection(0, 1.0)]))
    with pytest.raises(ValueError):
        G.planform(WingSpec([WingSection(1, 1.0), WingSection(0.5, 1.0)]))
    with pytest.raises(ValueError):
        G.planform(WingSpec([WingSection(0, 1.0), WingSection(1, 0.0), WingSection(2, 0.0)]))
    with pytest.raises(ValueError):
        G.planform(WingSpec([WingSection(-1, 1.0), WingSection(1, 1.0)], symmetric=True))
    # a pointed tip has a planform (the wing solver analyses it) but no loft
    pointed = WingSpec([WingSection(0, 1.0), WingSection(1, 0.0)])
    assert G.planform(pointed)["taper"] == 0.0 and G.planform(pointed)["area"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        G.loft(pointed)


def test_planform_is_the_wing_solvers():
    """design_from_wing's references are the wing solver's defaults: the same definitions."""
    from aerolume.solvers.wing import planform_metrics

    for w in (_rect(), _tapered_swept(), WingSpec([WingSection(0.0, 2.0, 0.0, 0.0), WingSection(2.0, 1.4, 0.3, 0.1),
                                                   WingSection(5.0, 0.6, 1.1, 0.5)]),
              WingSpec([WingSection(-2.0, 1.0), WingSection(3.0, 0.5, 0.4)], symmetric=False)):
        p, q = G.planform(w), planform_metrics(w)
        for k, v in q.items():
            assert p[k] == pytest.approx(v, rel=1e-12, abs=1e-12), k
        d = G.design_from_wing(w)
        assert (d.ref_area, d.length_ref, d.span_ref) == (q["area"], q["mac"], q["span"])
        assert d.moment_ref3[0] == pytest.approx(q["mac_le_x"] + 0.25 * q["mac"])


def test_tapered_wing_is_the_wing_people_describe():
    """tapered_wing: span tip to tip, quarter-chord sweep, tip twist -- read back by planform."""
    w = G.tapered_wing(10.0, 1.2, 0.6, sweep_deg=5.0, dihedral_deg=3.0, twist_deg=-2.0, airfoil="2412",
                       tip_airfoil="0012")
    p = G.planform(w)
    assert p["span"] == pytest.approx(10.0) and p["area"] == pytest.approx(0.5 * (1.2 + 0.6) * 10.0)
    assert p["taper"] == pytest.approx(0.5) and p["sweep_c4_deg"] == pytest.approx(5.0, abs=1e-12)
    assert p["dihedral_deg"] == pytest.approx(3.0) and p["root_chord"] == 1.2 and p["tip_chord"] == 0.6
    root, tip = w.sections
    assert (root.y, root.x_le, root.twist_deg, root.airfoil) == (0.0, 0.0, 0.0, "2412")
    assert tip.twist_deg == -2.0 and tip.airfoil == "0012" and w.symmetric
    # untapered, unswept default; a one-sided panel uses the whole span as its length
    rect = G.planform(G.tapered_wing(4.0, 0.5))
    assert rect["taper"] == 1.0 and rect["aspect_ratio"] == pytest.approx(8.0) and rect["sweep_le_deg"] == 0.0
    half = G.tapered_wing(3.0, 1.0, symmetric=False)
    assert half.sections[-1].y == 3.0 and G.planform(half)["span"] == pytest.approx(3.0)
    # the command line's wing uses the same definitions (sweep at the quarter chord, tip twist)
    import argparse

    from aerolume.cli import _wing_design
    cli = _wing_design(argparse.Namespace(naca="2412", span=10.0, chord=1.2, taper=0.5, sweep=5.0, dihedral=3.0,
                                          twist=-2.0)).wing
    same = G.tapered_wing(10.0, 1.2, 0.6, sweep_deg=5.0, dihedral_deg=3.0, twist_deg=-2.0)
    assert [s.to_dict() for s in cli.sections] == [s.to_dict() for s in same.sections]
    d = G.design_from_wing(w)
    assert d.ref_area == pytest.approx(p["area"]) and d.span_ref == pytest.approx(10.0)
    for bad in ({"span": 0.0, "root_chord": 1.0}, {"span": 1.0, "root_chord": -1.0},
                {"span": 1.0, "root_chord": 1.0, "tip_chord": 0.0}, {"span": 1.0, "root_chord": float("nan")},
                {"span": 1.0, "root_chord": 1.0, "sweep_deg": 85.0},
                {"span": 1.0, "root_chord": 1.0, "twist_deg": float("inf")}):
        with pytest.raises(ValueError):
            G.tapered_wing(**bad)


# ---------------------------------------------------------------------------
# lofting
# ---------------------------------------------------------------------------

def test_section_outline():
    o = G.section_outline("2412", 30)
    assert o.shape == (61, 2)
    np.testing.assert_allclose(o[0], o[-1])                     # sharp TE closed
    np.testing.assert_allclose(o[30], [0.0, 0.0], atol=1e-12)    # LE at index n
    assert o[:30, 1].min() > o[31:, 1].max() - 0.2              # upper first
    with pytest.raises(ValueError):
        G.section_outline("design:abc", 30)
    blunt = G.section_outline("design:x", 20, airfoil_coords=lambda name: naca4("0012", 120, closed_te=False))
    assert blunt.shape == (41, 2) and blunt[0, 1] > blunt[-1, 1]


def test_loft_rectangular_is_closed_outward_and_has_the_right_size():
    w = _rect()
    m = G.loft(w, n_chord=40)
    info = G.mesh_info(m)
    assert info["watertight"] and info["inconsistent_edges"] == 0 and info["components"] == 1
    assert G.volume(m) > 0
    x0, x1, y0, y1, _z0, _z1 = m.bounds()
    assert (x0, x1, y0, y1) == pytest.approx((0.0, 1.0, -5.0, 5.0), abs=1e-9)
    # NACA 0012 section area 0.0822 c^2 (exact integral of the thickness form: 0.68525 t)
    assert G.volume(m) == pytest.approx(0.68525 * 0.12 * 10.0, rel=0.01)
    # the lofted planform is exactly the trapezoid planform() integrates
    # (rasterised: grid-aligned edges cost up to half a pixel each, so use a fine raster)
    assert G.frontal_area(m, axis="z", resolution=4096) == pytest.approx(G.planform(w)["area"], rel=0.002)


def test_loft_tapered_swept_twisted_wing():
    w = _tapered_swept()
    m = G.loft(w, n_chord=30, n_span_per_panel=6)
    assert G.is_watertight(m) and G.volume(m) > 0 and G.inconsistent_edges(m) == 0
    assert G.frontal_area(m, axis="z", resolution=2048) == pytest.approx(G.planform(w)["area"], rel=0.005)
    # tip: chord 1, LE at x = 5 tan30, twisted -3 deg (nose down): TE above the LE
    tip = m.vertices[np.isclose(m.vertices[:, 1], 5.0)]
    le = tip[np.argmin(tip[:, 0])]
    te = tip[np.argmax(tip[:, 0])]
    assert le[0] == pytest.approx(5 * math.tan(math.radians(30)), abs=1e-9)
    assert te[2] - le[2] == pytest.approx(math.sin(math.radians(3.0)), abs=0.01)
    assert np.hypot(te[0] - le[0], te[2] - le[2]) == pytest.approx(1.0, abs=1e-6)
    # left half mirrors the right one exactly (same diagonals mirrored), as a
    # half-domain simulation needs
    left = m.vertices[np.isclose(m.vertices[:, 1], -5.0)]
    np.testing.assert_allclose(np.sort(left[:, 0]), np.sort(tip[:, 0]))
    assert G.symmetry_error(m) < 1e-12


def test_loft_root_gap_dihedral_blunt_te_and_one_sided():
    coords = naca4("4412", 160, closed_te=False)
    w = WingSpec([WingSection(0.3, 1.0, 0.0, 0.0, 2.0, "design:mine"),
                  WingSection(3.0, 0.5, 0.2, 0.4, 0.0, "design:mine")])
    m = G.loft(w, n_chord=24, airfoil_coords=lambda name: coords)
    info = G.mesh_info(m)
    assert info["watertight"] and info["components"] == 2 and info["inconsistent_edges"] == 0
    assert G.volume(m) > 0
    assert m.bounds()[2:4] == pytest.approx((-3.0, 3.0))
    one = G.loft(WingSpec(w.sections, symmetric=False), n_chord=24, airfoil_coords=lambda name: coords)
    assert G.is_watertight(one) and G.volume(one) == pytest.approx(G.volume(m) / 2, rel=1e-9)
    open_ = G.loft(_rect(), n_chord=16, closed_tips=False)
    assert G.mesh_info(open_)["holes"] == 2


def test_loft_voxelises_to_its_volume():
    torch = pytest.importorskip("torch")
    from aerolume.geometry3d import voxel as VX

    m = G.loft(_rect(span_half=1.0, chord=1.0), n_chord=40)
    h = 1.0 / 48                                   # the 12 % section is ~6 cells thick
    g = VX.grid_around(m, h)
    info: dict = {}
    frac = VX.voxelize(m, g, device="cpu", info=info)
    # measured +0.53 % here (+3.7 % at 24 cells per chord, +0.15 % at 96): the ramp's
    # second-order curvature term around the nose and the sub-cell trailing edge
    assert float(frac.sum()) * h**3 == pytest.approx(G.volume(m), rel=0.01)
    assert info["sheet_faces"] == 0                 # a closed wing is never thickened
    assert torch.is_tensor(frac)


# ---------------------------------------------------------------------------
# designs
# ---------------------------------------------------------------------------

def test_design_from_wing_and_checks():
    w = _tapered_swept()
    d = G.design_from_wing(w, "Swept")
    p = G.planform(w)
    assert d.kind == "wing" and d.dimension == 3 and d.wing is w and d.meshes == []
    assert d.ref_area == pytest.approx(p["area"]) and d.span_ref == pytest.approx(10.0)
    assert d.length_ref == pytest.approx(p["mac"]) and d.symmetric
    assert d.moment_ref3 == pytest.approx((p["mac_le_x"] + 0.25 * p["mac"], 0.0, 0.0))
    assert G.check_design3d(d) == []
    bad = G.design_from_wing(WingSpec([WingSection(0, 2.0, airfoil="banana"), WingSection(0.8, 2.0)],
                                      n_span=4), "Stubby")
    msgs = " ".join(G.check_design3d(bad))
    assert "'banana'" in msgs and "Aspect ratio" in msgs and "Coarse lattice" in msgs
    thin = " ".join(G.check_design3d(d, cells_per_ref=8))
    assert "tip section is only" in thin
