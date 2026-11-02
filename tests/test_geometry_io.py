"""Shape import: SVG, DXF, image silhouettes, CSV point lists, and the dispatcher."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aerolume.geometry import import_file, read_points_csv, read_svg
from aerolume.geometry import polygon as P
from aerolume.geometry.svg import parse_path_data, parse_transform

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "geometry"


def svg(body: str, attrs: str = 'viewBox="0 0 100 100"') -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" {attrs}>{body}</svg>'


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

def test_svg_rect_y_flip_and_viewbox():
    (p,) = read_svg(svg('<rect x="10" y="20" width="30" height="10"/>'))
    assert P.signed_area(p) == pytest.approx(300.0)
    # y-down 20..30 inside a 100-high viewBox -> y-up 70..80
    assert P.bbox(p) == pytest.approx((10, 40, 70, 80))


def test_svg_relative_absolute_and_implicit_lineto_agree():
    a = read_svg(svg('<path d="M10 10 L 60 10 L60 40 L 10 40 Z"/>'))[0]
    b = read_svg(svg('<path d="m10,10 50,0 0,30 -50,0z"/>'))[0]       # implicit relative lineto
    c = read_svg(svg('<path d="M10 10H60V40H10Z"/>'))[0]
    d = read_svg(svg('<path d="M10 10h50v30h-50z"/>'))[0]
    for q in (b, c, d):
        assert P.signed_area(q) == pytest.approx(P.signed_area(a))
        assert P.bbox(q) == pytest.approx(P.bbox(a))


def test_svg_circle_ellipse_and_arcs_are_accurate():
    (c,) = read_svg(svg('<circle cx="50" cy="50" r="40"/>'), tolerance=1e-3)
    assert P.signed_area(c) == pytest.approx(math.pi * 1600, rel=1e-4)
    assert np.allclose(np.hypot(*(c - [50, 50]).T), 40, atol=2e-3)
    (e,) = read_svg(svg('<ellipse cx="50" cy="50" rx="40" ry="20"/>'), tolerance=1e-3)
    assert P.signed_area(e) == pytest.approx(math.pi * 800, rel=1e-4)
    # a circle from two arcs (relative, packed flags) equals the circle element
    (a,) = read_svg(svg('<path d="M10,50 a40,40 0 1,0 80,0 a40 40 0 10-80 0z"/>'), tolerance=1e-3)
    assert P.signed_area(a) == pytest.approx(math.pi * 1600, rel=1e-4)
    # arc with too-small radii is scaled up (SVG F.6.6): half circle of radius 40
    (h,) = read_svg(svg('<path d="M10,50 A1,1 0 0,1 90,50 Z"/>'), tolerance=1e-3)
    assert P.signed_area(h) == pytest.approx(0.5 * math.pi * 1600, rel=1e-4)


def test_svg_rounded_rect():
    (r,) = read_svg(svg('<rect x="0" y="0" width="40" height="20" rx="5"/>'), tolerance=1e-4)
    assert P.signed_area(r) == pytest.approx(800 - (4 - math.pi) * 25, rel=1e-5)


def test_svg_curves_smooth_and_quadratic():
    # cubic + smooth cubic forming a symmetric lens; quadratic + smooth quadratic
    (c,) = read_svg(svg('<path d="M0,50 C 20,0 80,0 100,50 S 20,100 0,50 Z"/>'), tolerance=1e-4)
    assert P.bbox(c)[0] == pytest.approx(0, abs=1e-9)
    # S reflects the previous control point (80,0) about (100,50) -> (120,100): compare with
    # a dense evaluation of the same two cubics
    t = np.linspace(0, 1, 20001)[:, None]

    def bez(p0, p1, p2, p3):
        p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    dense = np.vstack([bez((0, 50), (20, 0), (80, 0), (100, 50)), bez((100, 50), (120, 100), (20, 100), (0, 50))])
    assert abs(P.signed_area(c)) == pytest.approx(abs(P.signed_area(dense)), rel=1e-4)
    (q,) = read_svg(svg('<path d="M0,50 Q50,0 100,50 T 200,50 L200,90 L0,90 Z"/>', 'viewBox="0 0 200 100"'))
    assert P.bbox(q)[3] - P.bbox(q)[2] == pytest.approx(65.0, abs=1e-6)   # Q peak at y=25 -> 90-25


def test_svg_nested_group_transforms():
    body = ('<g transform="translate(100,0) scale(2)">'
            '  <g transform="rotate(90 10 10)"><rect x="0" y="0" width="20" height="10"/></g>'
            '</g>'
            '<g transform="matrix(1 0 0 1 0 0) skewX(45)"><rect x="0" y="50" width="10" height="10"/></g>')
    polys = read_svg(svg(body, 'viewBox="0 0 300 300"'))
    areas = sorted(abs(P.signed_area(p)) for p in polys)
    assert areas == pytest.approx([100.0, 800.0])            # skew keeps area; scale 2 -> x4
    big = max(polys, key=lambda p: abs(P.signed_area(p)))
    # rotate(90 about 10,10) maps the 20x10 rect to x 0..10, y 0..20 (y down); scale 2,
    # shift 100 -> x 100..120 ... checks size and position
    x0, x1, y0, y1 = P.bbox(big)
    assert (x1 - x0, y1 - y0) == pytest.approx((20.0, 40.0))
    assert x0 == pytest.approx(120.0)            # rotated rect spans x 10..20 -> *2 + 100
    m = parse_transform("translate(5) scale(2,3)")
    assert np.allclose(m, [[2, 0, 5], [0, 3, 0], [0, 0, 1]])


def test_svg_open_paths_are_chained_and_closed_with_note():
    # two open halves of a square, drawn separately: chained into one
    body = '<path d="M0,0 L50,0 L50,50"/><path d="M50,50 L0,50 L0,0"/>'
    polys = read_svg(svg(body))
    assert len(polys) == 1 and P.signed_area(polys[0]) == pytest.approx(2500)
    notes = []
    polys = read_svg(svg('<polyline points="0,0 50,0 50,50"/>'), notes=notes)
    assert len(polys) == 1 and len(notes) == 1


def test_svg_ignores_hidden_and_non_geometry():
    body = ('<defs><rect width="5" height="5"/></defs><text>hi</text>'
            '<rect width="5" height="5" style="display:none"/><rect width="10" height="10"/>')
    assert len(read_svg(svg(body))) == 1


def test_svg_units_m():
    s = svg('<rect x="0" y="0" width="100" height="40"/>', 'width="200mm" height="80mm" viewBox="0 0 100 40"')
    (p,) = read_svg(s, units="m")
    x0, x1, y0, y1 = P.bbox(p)
    assert (x1 - x0, y1 - y0) == pytest.approx((0.2, 0.08))


def test_parse_path_data_pieces():
    sub = parse_path_data("M0 0 C1 1 2 1 3 0 Q 4 -1 5 0 L 6 0 z M 10 10 l 1 0")
    assert len(sub) == 2
    pieces, closed = sub[0]
    assert closed and [len(c) for c in pieces] == [4, 3, 2, 2]
    assert not sub[1][1]
    with pytest.raises(ValueError):
        parse_path_data("M0 0 A 1 1 0 2 0 3 3")             # bad flag


def test_svg_example_file():
    polys = read_svg(EXAMPLES / "car_profile.svg")
    assert len(polys) == 3                                    # body + 2 wheels
    for p in polys:
        assert P.signed_area(p) > 0 and not P.is_self_intersecting(p)


# ---------------------------------------------------------------------------
# DXF
# ---------------------------------------------------------------------------

def test_dxf_chaining_bulges_arcs(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    from aerolume.geometry import read_dxf
    doc = ezdxf.new()
    msp = doc.modelspace()
    # square 0..10 as four LINEs, shuffled and partly reversed, with 1e-6 endpoint noise
    msp.add_line((10, 0), (10, 10))
    msp.add_line((0, 0), (10, 0.000001))
    msp.add_line((0, 10), (10, 10))                           # reversed
    msp.add_line((0, 10), (0, 0))
    # stadium as LWPOLYLINE with bulges (two semicircles of r = 1)
    msp.add_lwpolyline([(20, 0, 0, 0, 1.0), (22, 0, 0, 0, 1.0)], format="xyseb", close=True)
    # half disk: ARC + closing LINE
    msp.add_arc((40, 0), 2.0, 0, 180)
    msp.add_line((38, 0), (42, 0))
    msp.add_circle((60, 0), 3.0)
    msp.add_ellipse((80, 0), major_axis=(4, 0), ratio=0.5)
    f = tmp_path / "t.dxf"
    doc.saveas(f)
    polys = read_dxf(f, tolerance=1e-4)
    areas = sorted(abs(P.signed_area(p)) for p in polys)
    assert areas == pytest.approx(sorted([100.0, math.pi * 1.0, 0.5 * math.pi * 4, math.pi * 9, math.pi * 8]),
                                  rel=1e-4)
    for p in polys:
        assert P.signed_area(p) > 0


def test_dxf_spline_and_example_file():
    pytest.importorskip("ezdxf")
    from aerolume.geometry import read_dxf
    notes = []
    polys = read_dxf(EXAMPLES / "car_profile.dxf", notes=notes)
    assert len(polys) == 4                                    # body, 2 wheels, stadium
    body = max(polys, key=lambda p: abs(P.signed_area(p)))
    assert not P.is_self_intersecting(body)
    x0, x1, _y0, _y1 = P.bbox(body)
    assert x1 - x0 == pytest.approx(4100.0, rel=1e-3)        # hatchback, mm
    assert any("millimetres" in n for n in notes)
    # only the body layer
    only = read_dxf(EXAMPLES / "car_profile.dxf", layers=["0"])
    assert len(only) == 1


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

def test_image_disk_area_within_one_percent(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw

    from aerolume.geometry import read_image_silhouette
    img = Image.new("L", (400, 300), 255)
    ImageDraw.Draw(img).ellipse((80, 40, 280, 240), fill=0)       # 201 px wide (inclusive box)
    f = tmp_path / "disk.png"
    img.save(f)
    polys = read_image_silhouette(f)
    assert len(polys) == 1
    a = P.signed_area(polys[0])
    exact = math.pi * 100.5**2
    assert abs(a - exact) / exact < 0.01
    c = P.centroid(polys[0])
    assert np.allclose(c, [180, 299 - 140], atol=1.0)            # y flipped up


def test_image_light_on_dark_alpha_and_holes(tmp_path):
    from PIL import Image, ImageDraw

    from aerolume.geometry import read_image_silhouette
    # light object on dark background with a hole and a speck of dust
    img = Image.new("L", (300, 200), 10)
    d = ImageDraw.Draw(img)
    d.rectangle((50, 50, 250, 150), fill=240)
    d.rectangle((120, 80, 180, 120), fill=10)                    # hole: ignored
    d.point((10, 10), fill=255)                                 # dust: filtered
    f = tmp_path / "inv.png"
    img.save(f)
    polys = read_image_silhouette(f)
    assert len(polys) == 1
    assert P.signed_area(polys[0]) == pytest.approx(201 * 101, rel=0.02)
    inv = read_image_silhouette(f, invert=True)                  # background becomes the object
    assert len(inv) >= 1
    # alpha cut-out
    rgba = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
    ImageDraw.Draw(rgba).ellipse((50, 50, 150, 150), fill=(255, 255, 255, 255))   # white on transparent
    g = tmp_path / "alpha.png"
    rgba.save(g)
    (p,) = read_image_silhouette(g)
    assert P.signed_area(p) == pytest.approx(math.pi * 50.5**2, rel=0.01)


def test_image_example_file():
    from aerolume.geometry import read_image_silhouette
    polys = read_image_silhouette(EXAMPLES / "car_silhouette.png")
    assert len(polys) == 1 and not P.is_self_intersecting(polys[0])


# ---------------------------------------------------------------------------
# CSV and dispatcher
# ---------------------------------------------------------------------------

def test_csv_blocks_groups_and_locales(tmp_path):
    text = "x,y\n0,0\n1,0\n1,1\n0,1\n0,0\n\n5,5\n6,5\n6,6\n"
    polys = read_points_csv(text)
    assert [len(p) for p in polys] == [4, 3]
    assert all(P.signed_area(p) > 0 for p in polys)
    semi = "x;y;id\n0,0;0,0;1\n1,5;0,0;1\n1,5;1,0;1\n0;0;2\n2;0;2\n2;2;2\n"
    polys = read_points_csv(semi)
    assert len(polys) == 2 and P.signed_area(polys[0]) == pytest.approx(0.75)
    f = tmp_path / "pts.txt"
    f.write_text("0 0\n2 0\n2 1\n0 1\n")
    (p,) = read_points_csv(f)
    assert P.signed_area(p) == pytest.approx(2.0)


def test_import_file_dispatch_examples():
    expect = {"naca2412_selig.dat": "airfoil", "naca23012_lednicer.dat": "airfoil",
              "car_profile.svg": "vehicle", "car_profile.dxf": "vehicle",
              "car_silhouette.png": "vehicle", "two_element_wing.csv": "airfoil"}
    for name, kind in expect.items():
        r = import_file(EXAMPLES / name)
        assert r.suggested_kind == kind, name
        assert r.polygons and all(P.signed_area(p) > 0 for p in r.polygons)
    with pytest.raises(ValueError):
        import_file(EXAMPLES / "make_examples.py")
