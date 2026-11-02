"""prepare_coords: airfoil outline normalisation, TE/LE detection and rejection of non-airfoils."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.model import Body, Design
from aerolume.solvers.airfoil import AirfoilGeometryError, prepare_coords
from aerolume.solvers.airfoil.solver import adapt_trailing_edge


def naca4(code: str, n: int = 81, closed: bool = True) -> np.ndarray:
    """Local NACA 4-digit generator (Selig order, CCW, implicitly closed) — no dependency on geometry/."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    b = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    a4 = -0.1036 if closed else -0.1015
    yt = t / 0.2 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    pts = np.vstack([np.c_[xu[::-1], yu[::-1]], np.c_[xl[1:], yl[1:]]])
    return pts[:-1] if closed else pts


def _check_selig(q: np.ndarray) -> None:
    # the TE (the midpoint of the two corners of a blunt TE) is at (1, 0); a cambered
    # NACA's corners are offset along x by +/- yt sin(camber slope) (~2e-4 on 4412)
    assert np.allclose(0.5 * (q[0] + q[-1]), [1.0, 0.0], atol=1e-9)
    assert abs(q[0, 0] - 1.0) < 1e-3 and abs(q[-1, 0] - 1.0) < 1e-3
    ile = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
    # the LE is the contour point farthest from the TE; on a cambered nose it may lie
    # between vertices, but never more than a panel away from one
    assert np.hypot(*q[ile]) < 5e-3
    assert np.all(q[:, 0] > -1e-9) and np.all(np.hypot(q[:, 0] - 1, q[:, 1]) <= 1 + 1e-9)
    assert np.all(q[1:ile, 1] > q[ile + 1:-1, 1].min() - 1e-9)        # upper first
    area = 0.5 * np.sum(q[:-1, 0] * q[1:, 1] - q[1:, 0] * q[:-1, 1])
    assert area > 0                                                     # CCW


def test_sharp_naca_is_identity_in_design_frame():
    p = naca4("2412")
    q, chord, info = prepare_coords(p)
    assert chord == pytest.approx(1.0, abs=1e-12)
    assert info["te_type"] == "sharp" and info["te_gap"] == 0.0
    assert info["incidence_deg"] == pytest.approx(0.0, abs=1e-9)
    assert np.array_equal(q[0], q[-1]) and np.allclose(q[0], [1, 0])
    assert len(q) == len(p) + 1
    # same points (the design frame is kept): every input point appears in the output
    assert np.allclose(np.sort(q[:-1, 0]), np.sort(p[:, 0]), atol=1e-12)
    assert info["thickness"] == pytest.approx(0.12, abs=2e-3)
    assert info["camber"] == pytest.approx(0.02, abs=1e-3)
    assert info["camber_x"] == pytest.approx(0.4, abs=0.05)
    _check_selig(q)


def test_blunt_te_corners_and_gap():
    p = naca4("0012", closed=False)
    q, chord, info = prepare_coords(p)
    assert info["te_type"] == "blunt"
    assert info["te_gap"] == pytest.approx(0.00252, rel=0.02)
    assert q[0, 1] > 0 > q[-1, 1]                     # upper corner first, lower corner last
    assert chord == pytest.approx(1.0, abs=1e-6)
    _check_selig(q)


@pytest.mark.parametrize("closed", [True, False])
def test_any_scale_position_orientation_and_order(closed):
    p = naca4("4412", closed=closed)
    ang = math.radians(7.0)                             # 7 deg nose-up incidence
    R = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
    moved = (p @ R.T) * 0.35 + np.array([2.0, -1.0])
    moved = np.roll(moved[::-1], 17, axis=0)          # clockwise and starting mid-surface
    # Reference: the same section merely shifted, so that (like the moved copy) it is
    # not in the design frame and its LE is the contour point farthest from the TE.
    # For a cambered nose that point is not the NACA origin: on NACA 4412 it sits
    # ~0.003 c above it, which tilts the chord line by 0.18 deg and lengthens it by
    # 3e-4 c relative to the design-frame chord (the identity test above).
    ref, chord0, info0 = prepare_coords(p + np.array([0.5, 0.3]))
    assert info0["incidence_deg"] == pytest.approx(0.18, abs=0.02)
    q, chord, info = prepare_coords(moved)
    assert chord == pytest.approx(0.35 * chord0, rel=1e-5)
    assert info["incidence_deg"] == pytest.approx(7.0 + info0["incidence_deg"], abs=0.01)
    assert q.shape == ref.shape and np.allclose(q, ref, atol=1e-9)   # same contour, same order
    assert info["te_type"] == info0["te_type"]
    assert info["thickness"] == pytest.approx(info0["thickness"], abs=1e-3)
    assert info["camber"] == pytest.approx(info0["camber"], abs=1e-3)
    _check_selig(q)


def test_explicitly_closed_and_duplicate_points():
    p = naca4("0012")
    p = np.vstack([p[:30], p[29:30], p[30:], p[:1]])   # duplicate + explicit closing point
    q, _, info = prepare_coords(p)
    assert info["te_type"] == "sharp"
    assert len(q) == len(naca4("0012")) + 1


def test_mirrored_section_faces_the_flow():
    p = naca4("2412") * np.array([-1.0, 1.0])          # LE on the right, camber still up
    q, _, info = prepare_coords(p)
    assert info["mirrored"]
    assert info["camber"] == pytest.approx(0.02, abs=1e-3)   # still positive camber
    assert any("mirrored" in w for w in info["warnings"])
    _check_selig(q)


def test_design_and_body_inputs():
    p = naca4("2412") * 0.5
    d = Design(name="wing", kind="airfoil", bodies=[Body("main", p)], length_ref=0.5, force_ref=0.5,
               moment_ref=(0.125, 0.0))
    q, chord, info = prepare_coords(d)
    assert chord == pytest.approx(0.5) and info["name"] == "wing"
    q2, chord2, _ = prepare_coords(d.bodies[0])
    assert np.allclose(q, q2) and chord2 == chord


@pytest.mark.parametrize("make, fragment", [
    (lambda: Design(name="car", kind="vehicle", bodies=[Body("b", naca4("0012"))], length_ref=1, force_ref=1),
     "vehicle"),
    (lambda: Design(name="two", kind="airfoil", bodies=[Body("a", naca4("0012")), Body("b", naca4("0012") + 2)],
                    length_ref=1, force_ref=1), "2 bodies"),
    (lambda: np.c_[np.cos(np.linspace(0, 2 * np.pi, 80, endpoint=False)),
                   np.sin(np.linspace(0, 2 * np.pi, 80, endpoint=False))], "rounded"),
    (lambda: np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), "points"),
    (lambda: np.c_[np.linspace(0, 1, 30), np.zeros(30)], "no area"),
    (lambda: np.array([[0, 0], [4, 0], [4, 1], [3.9, 1.3], [0.5, 1.4], [0.1, 1.0]] * 1, float), "points"),
])
def test_rejections_have_clear_messages(make, fragment):
    with pytest.raises(AirfoilGeometryError) as e:
        prepare_coords(make())
    msg = str(e.value)
    assert msg.startswith("Polar analysis needs a single airfoil-like body")
    assert fragment in msg


def test_rejects_bluff_box_and_crossing_outline():
    # a car-like box: blunt "trailing edge" far too thick
    box = np.array([[0, 0], [4.5, 0], [4.5, 1.2], [3.0, 1.4], [1.0, 1.4], [0.0, 0.8]], float)
    box = np.vstack([np.linspace(box[i], box[(i + 1) % len(box)], 12, endpoint=False) for i in range(len(box))])
    with pytest.raises(AirfoilGeometryError):
        prepare_coords(box)
    bow = naca4("0012").copy()
    bow[10:20, 1] *= -3.0                               # upper surface folded through the lower one
    with pytest.raises(AirfoilGeometryError):
        prepare_coords(bow)


def test_adapt_trailing_edge_open_and_close():
    q, _, _ = prepare_coords(naca4("0012"))
    o = adapt_trailing_edge(q, 0.0, 5e-4)
    assert o[0, 1] - o[-1, 1] == pytest.approx(5e-4, rel=1e-9)
    assert np.allclose(o[len(o) // 2], q[len(q) // 2])  # LE untouched
    assert np.max(np.abs(o - q)) <= 2.5e-4 + 1e-15
    qb, _, infob = prepare_coords(naca4("0012", closed=False))
    s = adapt_trailing_edge(qb, infob["te_gap"], 0.0)
    assert np.allclose(s[0], s[-1]) and np.allclose(s[0], [1.0, 0.0], atol=1e-12)
