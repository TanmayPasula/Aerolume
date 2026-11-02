"""NACA generators, .dat files, airfoil normalisation and CST."""
from __future__ import annotations

import numpy as np
import pytest

from aerolume.geometry import (
    cst_eval,
    cst_fit,
    design_from_airfoil,
    format_airfoil_dat,
    naca,
    naca4,
    naca5,
    read_airfoil_dat,
)
from aerolume.geometry import polygon as P


def surfaces(c):
    """Upper and lower surface (LE -> TE) of a Selig-ordered section, as y(x) on a fine grid."""
    ile = int(np.argmin(c[:, 0]))
    up = c[:ile + 1][::-1]
    lo = c[ile:]
    d = c[0] - c[-1]
    if abs(d[1]) <= abs(d[0]):                     # sharp TE: point 0 closes the lower surface too
        lo = np.vstack([lo, c[:1]])
    x = np.linspace(0.0, 1.0, 20001)
    return x, np.interp(x, up[:, 0], up[:, 1]), np.interp(x, lo[:, 0], lo[:, 1])


def test_naca0012_thickness_and_symmetry():
    c = naca4("0012", 400)
    assert len(c) == 400
    assert P.signed_area(c) > 0 and not P.is_self_intersecting(c)
    assert np.allclose(c[0], [1, 0]) and np.allclose(c[np.argmin(c[:, 0])], [0, 0])
    x, yu, yl = surfaces(c)
    t = yu - yl
    assert t.max() == pytest.approx(0.12, abs=2e-5)
    assert x[np.argmax(t)] == pytest.approx(0.30, abs=0.01)
    # symmetric: lower surface mirrors upper
    assert np.allclose(yu, -yl, atol=1e-12)
    # closed TE: zero thickness at x = 1; blunt TE: 0.252 % (published open form)
    b = naca4("0012", 400, closed_te=False)
    assert len(b) == 401
    assert b[0, 1] - b[-1, 1] == pytest.approx(2 * 5 * 0.12 * 0.0021 * 1.0, abs=1e-12)


def test_naca2412_camber():
    c = naca4("2412", 400)
    x, yu, yl = surfaces(c)
    camber = 0.5 * (yu + yl)
    i = int(np.argmax(camber))
    assert camber[i] == pytest.approx(0.02, abs=2e-4)
    assert x[i] == pytest.approx(0.40, abs=0.01)
    assert (yu - yl).max() == pytest.approx(0.12, abs=3e-4)


def test_naca23012_camber_position_and_reflex():
    c = naca5("23012", 400)
    x, yu, yl = surfaces(c)
    camber = 0.5 * (yu + yl)
    assert x[int(np.argmax(camber))] == pytest.approx(0.15, abs=0.01)
    assert (yu - yl).max() == pytest.approx(0.12, abs=5e-4)
    # reflex mean line: camber turns back up near the trailing edge (ends ~0 with negative slope reversed)
    r = naca5("23112", 400)
    xr, yur, ylr = surfaces(r)
    cam_r = 0.5 * (yur + ylr)
    assert np.diff(cam_r[xr > 0.9]).mean() > np.diff(camber[x > 0.9]).mean()
    assert not P.is_self_intersecting(r)


def test_naca_dispatch_and_errors():
    assert np.allclose(naca("NACA 2412"), naca4("2412"))
    assert np.allclose(naca("23012", chord=2.0), 2.0 * naca5("23012"))
    for bad in ("12", "2a12", "123456", "23212", "26012"):
        with pytest.raises(ValueError):
            naca(bad)


@pytest.mark.parametrize("fmt", ["selig", "lednicer"])
@pytest.mark.parametrize("closed", [True, False])
def test_dat_round_trip(fmt, closed, tmp_path):
    c = naca("2412", 120, closed_te=closed)
    text = format_airfoil_dat("NACA 2412", c, fmt)
    name, back = read_airfoil_dat(text)
    assert name == "NACA 2412"
    assert back.shape == c.shape
    assert np.allclose(back, c, atol=1e-7)
    f = tmp_path / "foil.dat"
    f.write_text(text)
    _name2, back2 = read_airfoil_dat(f)
    assert np.allclose(back2, c, atol=1e-7)


def test_dat_tolerates_messy_files():
    text = ("﻿My Foil  (exported)\n# comment\n\n"
            "1,000000\t0,001300\n"
            " 0,500000   0,060000\n"
            "0,000000 0,000000\n\n"
            "0,500000\t-0,040000\n"
            "1,000000   -0,001300\n")
    name, c = read_airfoil_dat(text)
    assert name.startswith("My Foil")
    assert c.shape == (5, 2) and P.signed_area(c) > 0
    # comma-separated with dots, explicitly closed, clockwise (lower surface first)
    text2 = "foil\n1.0, 0.0\n0.5, -0.05\n0.0, 0.0\n0.5, 0.07\n1.0, 0.0\n"
    _, c2 = read_airfoil_dat(text2)
    assert len(c2) == 4 and P.signed_area(c2) > 0
    with pytest.raises(ValueError):
        read_airfoil_dat("just a title\n")


def test_design_from_airfoil_normalises_position_scale_rotation():
    c = naca("4412", 200)
    moved = P.transform(c, scale=0.37, rotate_deg=12.0, translate=(3.0, -1.0))
    d = design_from_airfoil(moved, "x", chord=0.5)
    q = d.bodies[0].points
    assert d.kind == "airfoil" and d.length_ref == d.force_ref == 0.5
    assert d.moment_ref == (0.125, 0.0)
    assert np.allclose(q[0], [0.5, 0.0], atol=1e-12)          # starts at the TE
    le = q[np.argmin(np.hypot(*q.T))]
    assert np.allclose(le, [0, 0], atol=1e-12)
    assert P.bbox(q)[1] == pytest.approx(0.5, abs=1e-9)
    assert P.signed_area(q) > 0
    # same shape as normalising the original directly
    d0 = design_from_airfoil(c, "y", chord=0.5)
    assert np.allclose(np.sort(q[:, 1]), np.sort(d0.bodies[0].points[:, 1]), atol=1e-9)


def test_cst_fit_naca2412_order8():
    p = cst_fit(naca("2412", 240), order=8)
    assert p.residual < 1e-4
    assert p.max_error < 5e-4
    assert len(p.upper) == len(p.lower) == 9
    c = cst_eval(p, 200)
    assert P.signed_area(c) > 0 and not P.is_self_intersecting(c)
    _x, yu, yl = surfaces(c)
    assert (yu - yl).max() == pytest.approx(0.12, abs=1e-3)


def test_cst_blunt_te_and_dict_round_trip():
    p = cst_fit(naca("0012", 240, closed_te=False), order=8)
    assert p.te_thickness == pytest.approx(0.00252, abs=2e-5)
    c = cst_eval(p, 160)
    assert len(c) == 161 and c[0, 1] - c[-1, 1] == pytest.approx(p.te_thickness)
    q = type(p).from_dict(p.to_dict())
    assert np.allclose(cst_eval(q, 160), c)
