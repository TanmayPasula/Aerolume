"""Exports: CSV round trips, VTK ImageData well-formedness and content, NPZ, SVG, DXF, Selig."""
from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from aerolume.export import (
    design_dat,
    design_dxf,
    design_svg,
    fields_npz,
    fields_vtk,
    history_csv,
    polar_csv,
    read_csv,
    result_csv,
    selig_order,
    surface_csv,
)
from aerolume.model import Body, CFDResult, Design, FieldData, PolarPoint, PolarResult, Stat, SurfaceData


def naca4_ccw(code: str = "2412", n: int = 81, chord: float = 0.5, closed_te: bool = True) -> np.ndarray:
    """NACA 4-digit outline, counter-clockwise (Aerolume order), rolled so it does not start at the TE."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2)) if p else 0 * x
    xu, yu, xl, yl = x, yc + yt, x, yc - yt
    upper = np.c_[xu[::-1], yu[::-1]]                 # TE -> LE over the top
    lower = np.c_[xl[1:], yl[1:]]                     # LE -> TE underneath (LE not repeated)
    pts = np.vstack([upper, lower])
    if closed_te:
        pts = pts[:-1]                                # implicit closure: TE appears once
    return np.roll(pts, 17, axis=0) * chord


def signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


@pytest.fixture
def fd() -> FieldData:
    ny, nx = 6, 10
    yy, xx = np.mgrid[0:ny, 0:nx]
    fields = {"u": (xx + 0.5).astype(float), "v": (yy * 0.1).astype(float), "cp": np.sin(xx + yy * 1.0),
              "p": 1e5 * np.cos(xx * 1.0)}
    mask = np.zeros((ny, nx))
    mask[2:4, 3:5] = 1
    return FieldData(extent=(-1.0, 4.0, -1.5, 1.5), fields=fields, mask=mask, label="mean")


# --- VTK ----------------------------------------------------------------------------

def _decode_block(text: str) -> np.ndarray:
    raw = base64.b64decode(text)
    nbytes = int(np.frombuffer(raw[:8], "<u8")[0])
    assert nbytes == len(raw) - 8
    return np.frombuffer(raw[8:], "<f4")


@pytest.mark.parametrize("encoding", ["appended", "inline", "ascii"])
def test_vtk_imagedata_well_formed(tmp_path, fd, encoding):
    path = fields_vtk(fd, tmp_path / f"f_{encoding}.vti", encoding=encoding)
    root = ET.parse(path).getroot()                                  # well-formed XML
    assert root.tag == "VTKFile" and root.get("type") == "ImageData"
    assert root.get("byte_order") == "LittleEndian" and root.get("header_type") == "UInt64"
    img = root.find("ImageData")
    assert img.get("WholeExtent") == "0 10 0 6 0 0"
    origin = [float(v) for v in img.get("Origin").split()]
    spacing = [float(v) for v in img.get("Spacing").split()]
    assert origin == [-1.0, -1.5, 0.0]
    assert spacing[:2] == pytest.approx([0.5, 0.5])
    assert img.find("Piece").get("Extent") == "0 10 0 6 0 0"
    cd = img.find("Piece/CellData")
    arrays = {a.get("Name"): a for a in cd.findall("DataArray")}
    assert set(arrays) == {"u", "v", "cp", "p", "mask", "velocity"}
    assert cd.get("Vectors") == "velocity" and cd.get("Scalars") == "cp"
    assert arrays["velocity"].get("NumberOfComponents") == "3"

    appended = root.find("AppendedData")
    blob = appended.text.strip()[1:] if appended is not None else None   # strip the "_" marker
    for name, a in arrays.items():
        ncomp = int(a.get("NumberOfComponents"))
        if encoding == "appended":
            off = int(a.get("offset"))
            hdr = base64.b64decode(blob[off:off + 12])               # 12 chars -> 9 bytes, enough for 8
            nbytes = int(np.frombuffer(hdr[:8], "<u8")[0])
            nchars = 4 * math.ceil((nbytes + 8) / 3)
            vals = _decode_block(blob[off:off + nchars])
        elif encoding == "inline":
            vals = _decode_block(a.text.strip())
        else:
            vals = np.array(a.text.split(), dtype=np.float32)
        assert vals.size == 10 * 6 * ncomp
        if name in fd.fields:
            np.testing.assert_allclose(vals, fd.fields[name].ravel().astype(np.float32), rtol=1e-7)
    if encoding == "appended":
        # x-fastest cell order with row 0 at ymin: cell (i=3, j=2) is mask[2, 3]
        off = int(arrays["mask"].get("offset"))
        nchars = 4 * math.ceil((10 * 6 * 4 + 8) / 3)
        assert _decode_block(blob[off:off + nchars])[2 * 10 + 3] == 1.0


def test_vtk_rejects_bad_input(tmp_path, fd):
    with pytest.raises(ValueError):
        fields_vtk(fd, tmp_path / "x.vti", encoding="raw")
    bad = FieldData(extent=(0, 1, 0, 1), fields={"a": np.zeros((3, 4)), "b": np.zeros((4, 3))})
    with pytest.raises(ValueError, match="different shapes"):
        fields_vtk(bad, tmp_path / "y.vti")


def test_fields_npz(tmp_path, fd):
    path = fields_npz(fd, tmp_path / "f.npz")
    with np.load(path) as z:
        np.testing.assert_allclose(z["p"], fd.fields["p"].astype(np.float32))
        np.testing.assert_array_equal(z["mask"], fd.mask)
        assert z["x"][0] == pytest.approx(-0.75) and z["y"][-1] == pytest.approx(1.25)
        assert tuple(z["extent"]) == fd.extent and str(z["label"]) == "mean"


# --- CSV ----------------------------------------------------------------------------

def make_result() -> CFDResult:
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.001, 0.01, 5000))
    hist = {"t": t, "cd": rng.standard_normal(5000) * 1e-3 + 1.0, "cl": rng.standard_normal(5000) / 3,
            "cm": np.full(5000, -1 / 7), "cd_body2": np.zeros(5000)}
    s = np.linspace(0, 1, 50)
    surf = [SurfaceData("main", s, s, s**2, -s, s, 1 - s * 2.5, s / 1e3),
            SurfaceData("flap", s[:10], s[:10], s[:10], s[:10], s[:10], s[:10], s[:10], sensitivity=s[:10] * 3)]
    return CFDResult(sim_id="r1", cd=Stat(1.0, 0.001, 0.99, 1.01, 2e-5), cl=Stat(0.0, 0.33, -1, 1, float("nan")),
                     drag=551.25, lift=0.0, strouhal=None, grid=(512, 256), converged=True, device="cpu",
                     per_body={"main": {"cd": 0.9}}, warnings=['Re capped, "quotes", commas'],
                     history=hist, surface=surf)


def test_history_csv_round_trip_is_exact(tmp_path):
    r = make_result()
    header, rows = read_csv(history_csv(r, tmp_path / "h.csv"))
    assert header == ["t", "cd", "cl", "cm", "cd_body2"]
    data = np.array(rows, dtype=float)
    for j, k in enumerate(header):
        np.testing.assert_array_equal(data[:, j], r.history[k])      # bit-exact via repr
    with pytest.raises(ValueError):
        history_csv({"cd": np.zeros(3)}, tmp_path / "bad.csv")


def test_surface_csv_round_trip(tmp_path):
    r = make_result()
    header, rows = read_csv(surface_csv(r, tmp_path / "s.csv"))
    assert header == ["body", "s", "x", "y", "nx", "ny", "cp", "cf", "sensitivity"]
    assert len(rows) == 60 and rows[0][0] == "main" and rows[-1][0] == "flap"
    cp = np.array([float(row[6]) for row in rows[:50]])
    np.testing.assert_array_equal(cp, r.surface[0].cp)
    assert rows[0][8] == "nan" and float(rows[-1][8]) == pytest.approx(r.surface[1].sensitivity[-1])


def test_result_csv(tmp_path):
    r = make_result()
    header, rows = read_csv(result_csv(r, tmp_path / "r.csv"))
    assert header == ["quantity", "value", "stderr", "unit", "description"]
    by = {row[0]: row for row in rows}
    assert float(by["cd"][1]) == 1.0 and float(by["cd"][2]) == 2e-5
    assert by["cl"][2] == "nan" and by["strouhal"][1] == "nan"
    assert by["drag"][3] == "N/m" and by["converged"][1] == "true" and by["grid_nx"][1] == "512"
    assert by["body:main:cd"][1] == "0.9"
    assert by["warning"][1] == 'Re capped, "quotes", commas'           # CSV quoting survives


def test_polar_csv_and_summary(tmp_path):
    pts = [PolarPoint(alpha=a, cl=0.11 * a + 0.2, cd=0.008 + 0.001 * a * a, cm=-0.05, converged=a != 2)
           for a in (3, -2, 0, 2, 1)]
    pol = PolarResult(sim_id="p", re=5e5, points=pts)
    header, rows = read_csv(polar_csv(pol, tmp_path / "p.csv"))
    assert header[:3] == ["alpha", "cl", "cd"] and header[-1] == "converged"
    assert [float(r[0]) for r in rows] == [-2, 0, 1, 2, 3]            # sorted by alpha
    assert rows[3][-1] == "false" and float(rows[1][6]) == pytest.approx(0.2 / 0.008)
    _, rows2 = read_csv(result_csv(pol, tmp_path / "ps.csv"))
    by = {row[0]: row for row in rows2}
    assert float(by["re"][1]) == 5e5 and by["alpha_zero_lift"][3] == "deg"


# --- geometry ------------------------------------------------------------------------

def airfoil_design(closed_te: bool = True) -> Design:
    pts = naca4_ccw(closed_te=closed_te)
    assert signed_area(pts) > 0
    return Design(name="NACA 2412  test", kind="airfoil", bodies=[Body("main", pts)], length_ref=0.5,
                  force_ref=0.5, moment_ref=(0.125, 0.0))


def test_selig_dat_sharp_te(tmp_path):
    d = airfoil_design(True)
    lines = design_dat(d, tmp_path / "a.dat").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "NACA 2412 test"
    xy = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    assert xy.shape == (len(d.bodies[0].points) + 1, 2)
    np.testing.assert_allclose(xy[0], [1.0, 0.0], atol=1e-6)            # starts at the TE ...
    np.testing.assert_allclose(xy[-1], xy[0])                           # ... and returns to it
    le = int(np.argmin(xy[:, 0]))
    np.testing.assert_allclose(xy[le], [0.0, 0.0], atol=1e-7)
    assert np.all(xy[1:le, 1] > 0) and np.all(xy[le + 1:-1, 1] < 0.03)  # upper first, then lower
    assert np.all(np.diff(xy[:le + 1, 0]) <= 0) and np.all(np.diff(xy[le:, 0]) >= 0)
    assert signed_area(xy[:-1]) > 0                                     # Selig order is CCW
    raw = design_dat(d, tmp_path / "b.dat", normalize=False).read_text(encoding="utf-8").splitlines()
    assert float(raw[1].split()[0]) == pytest.approx(0.5)


def test_selig_order_blunt_te():
    pts = naca4_ccw(closed_te=False)
    q = selig_order(pts)
    assert len(q) == len(pts)                                           # no repeat for a blunt TE
    assert q[0, 1] > 0 and q[-1, 1] < 0
    assert q[0, 0] == pytest.approx(0.5) and q[-1, 0] == pytest.approx(0.5)
    assert np.array_equal(selig_order(pts[::-1]), q)                    # CW input is fixed first


def test_design_svg(tmp_path):
    d = airfoil_design()
    d.bodies.append(Body("flap", d.bodies[0].points * 0.3 + [0.55, -0.05], role="element"))
    root = ET.parse(design_svg(d, tmp_path / "a.svg")).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert root.get("width").endswith("mm")
    w = float(root.get("width")[:-2])
    xmin, xmax, _, _ = d.bounds()
    assert w == pytest.approx((xmax - xmin) * 1.1 * 1e3, rel=1e-3)
    paths = root.findall(".//s:path", ns)
    assert [p.get("id") for p in paths] == ["main", "flap"] and paths[1].get("data-role") == "element"
    first = paths[0].get("d").split(" L ")[0][2:].split(",")
    p0 = d.bodies[0].points[0]
    assert float(first[0]) == pytest.approx((p0[0] - xmin + 0.05 * (xmax - xmin)) * 1e3, abs=1e-3)


@pytest.mark.parametrize("units,scale,code", [("m", 1.0, 6), ("mm", 1000.0, 4)])
def test_design_dxf(tmp_path, units, scale, code):
    import ezdxf

    d = airfoil_design()
    d.bodies.append(Body("wheel", np.c_[np.cos(np.linspace(0, 6, 20)), np.sin(np.linspace(0, 6, 20))],
                         role="wheel"))
    path = design_dxf(d, tmp_path / f"a_{units}.dxf", units=units)
    doc = ezdxf.readfile(path)
    assert doc.header["$INSUNITS"] == code
    polys = list(doc.modelspace().query("LWPOLYLINE"))
    assert len(polys) == 2 and all(p.closed for p in polys)
    assert [p.dxf.layer for p in polys] == ["BODY", "WHEEL"]
    got = np.array([(x, y) for x, y, *_ in polys[0].get_points()])
    np.testing.assert_allclose(got / scale, d.bodies[0].points, atol=1e-12)


def test_result_csv_room_heater_is_an_input_and_temperatures_are_exported(tmp_path):
    """A heater-only room: its heat rate is the power set (no error bar), and the fluid temperatures,
    the set heater power and the vents' heat are in the CSV like the report's numbers."""
    diag = {"heat_sources_set": 300.0, "heat_in_sources": 300.00003, "t_fluid_max": 41.2, "t_fluid_mean": 24.5,
            "t_fluid_min": 20.0, "t_fluid_dev_max": 21.2, "heat_out_ports": 120.0, "heat_in_ports": 0.0}
    r = CFDResult(sim_id="room", heat_rate=300.00003, diagnostics=diag,
           history={"t": np.linspace(0, 10, 50), "heat_rate": np.full(50, 300.00003)})
    _, rows = read_csv(result_csv(r, tmp_path / "room.csv"))
    by = {row[0]: row for row in rows}
    assert by["heat_rate"][2] == "nan" and "set (an input" in by["heat_rate"][4]
    assert float(by["heater_power_set"][1]) == 300.0
    assert float(by["t_fluid_max"][1]) == 41.2 and by["t_fluid_max"][3] == "degC" and by["t_fluid_dev_max"][3] == "K"
    assert float(by["vent_heat_out"][1]) == 120.0
