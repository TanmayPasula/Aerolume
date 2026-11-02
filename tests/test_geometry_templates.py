"""Templates, design builders and design diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.geometry import (
    check_design,
    design_from_airfoil,
    design_from_polygons,
    list_templates,
    make,
    naca,
    templates,
)
from aerolume.geometry import polygon as P
from aerolume.model import Design

VEHICLES = ["sedan", "hatchback", "fastback", "suv", "pickup", "van", "box_truck", "ahmed"]


def assert_valid(d: Design) -> None:
    assert isinstance(d, Design)
    for b in d.bodies:
        q = b.points
        assert q.shape[1] == 2 and len(q) >= 3 and np.isfinite(q).all()
        assert P.signed_area(q) > 0, f"{d.name}/{b.name} not CCW"
        assert not P.is_self_intersecting(q), f"{d.name}/{b.name} self-intersects"
        assert np.hypot(*(q[-1] - q[0])) > 0, "must be implicitly closed"
    assert d.length_ref > 0 and d.force_ref > 0
    # the model round-trips
    d2 = Design.from_dict(d.to_dict())
    assert np.allclose(d2.bodies[0].points, d.bodies[0].points)


def test_catalogue_is_complete_and_well_formed():
    infos = list_templates()
    names = [i.name for i in infos]
    for required in VEHICLES + ["front_wing", "naca", "cylinder", "square", "ellipse", "flat_plate",
                                "rounded_rectangle"]:
        assert required in names
    for info in infos:
        assert info.kind in ("airfoil", "vehicle", "generic") and info.description and info.label
        for p in info.params:
            assert p.unit_kind in templates.UNIT_KINDS
            if p.unit_kind in ("length", "angle", "ratio", "count"):
                assert p.min <= p.default <= p.max and p.step > 0
            if p.unit_kind == "choice":
                assert p.default in p.choices


@pytest.mark.parametrize("name", [i.name for i in list_templates()])
def test_every_template_default_is_valid(name):
    d = make(name)
    assert_valid(d)
    assert d.source["template"] == name and set(d.source["params"]) == set(templates.get_template(name).defaults())


@pytest.mark.parametrize("name", [n for n in VEHICLES if n != "ahmed"])
def test_vehicle_references_and_clearance(name):
    info = templates.get_template(name)
    p = info.defaults()
    d = make(name)
    x0, x1, y0, y1 = d.bounds()
    assert d.kind == "vehicle"
    assert x0 == pytest.approx(0.0, abs=1e-12) and x1 == pytest.approx(p["length"], rel=1e-12)
    assert y0 == pytest.approx(p["ground_clearance"], rel=1e-12)
    assert y1 == pytest.approx(p["height"], rel=1e-12)
    assert d.length_ref == pytest.approx(p["length"])
    assert d.force_ref == pytest.approx(p["height"] - p["ground_clearance"])
    assert d.moment_ref == pytest.approx((0.5 * p["length"], 0.0))
    assert len(d.bodies) == 1                                  # wheels off by default
    assert check_design(d) == []


def test_ahmed_geometry():
    d = make("ahmed", slant_angle=35.0)
    q = d.bodies[0].points
    x0, x1, y0, y1 = d.bounds()
    assert (x1 - x0, y1 - y0) == pytest.approx((1.044, 0.288))
    assert y0 == pytest.approx(0.05)
    # the slant: points on the rear upper edge follow the requested angle
    top_rear = q[(q[:, 0] > x1 - 0.222 * math.cos(math.radians(35)) + 0.01) & (q[:, 1] > y1 - 0.12)]
    slope = np.polyfit(top_rear[:, 0], top_rear[:, 1], 1)[0]
    assert math.degrees(math.atan(-slope)) == pytest.approx(35.0, abs=0.2)
    # front corner radius: area deficit of two quarter circles vs a sharp box
    a = P.signed_area(q)
    box = 1.044 * 0.288
    slant_tri = 0.5 * 0.222 * math.cos(math.radians(35)) * 0.222 * math.sin(math.radians(35))
    assert a == pytest.approx(box - slant_tri - 2 * (1 - math.pi / 4) * 0.1**2, rel=1e-3)


def test_wheels_option_and_road_contact_diagnostic():
    d = make("sedan", wheels=True)
    assert_valid(d)
    wheels = [b for b in d.bodies if b.role == "wheel"]
    assert len(wheels) == 2 and all(b.center is not None for b in wheels)
    assert min(float(b.points[:, 1].min()) for b in wheels) == pytest.approx(0.0, abs=1e-9)
    msgs = check_design(d)
    assert any("road" in m and "seals" in m for m in msgs)
    lifted = make("sedan", wheels=True, wheel_gap=0.02)
    assert not any("seals" in m for m in check_design(lifted))
    assert any("2 grid cells" in m for m in check_design(lifted, cells_per_ref=256))


@pytest.mark.parametrize("name", [i.name for i in list_templates()])
def test_parameter_ranges_always_give_valid_shapes(name):
    """Every numeric parameter at its min and max, plus random combinations of all of
    them, gives a valid CCW non-self-intersecting design (no exception): the GUI form
    can never produce a broken shape from in-range values."""
    info = templates.get_template(name)
    for p in info.params:
        if p.unit_kind in ("length", "angle", "ratio", "count"):
            for v in (p.min, p.max):
                assert_valid(make(name, **{p.name: v}))
    rng = np.random.default_rng(42)
    for _ in range(12):
        kw = {}
        for p in info.params:
            if p.unit_kind in ("length", "angle", "ratio"):
                kw[p.name] = float(rng.uniform(p.min, p.max))
            elif p.unit_kind == "bool":
                kw[p.name] = bool(rng.integers(2))
            elif p.unit_kind == "choice":
                kw[p.name] = p.choices[int(rng.integers(len(p.choices)))]
        assert_valid(make(name, **kw))


def test_parameter_validation_and_variation():
    with pytest.raises(ValueError):
        make("sedan", lenght=4.0)                               # typo
    with pytest.raises(ValueError):
        make("sedan", length=100.0)                             # out of range
    with pytest.raises(KeyError):
        make("spaceship")
    # shallow glass on a short car: steepened to fit, and the user is told
    d = make("sedan", length=3.0, windscreen_angle=15.0, rear_angle=10.0)
    assert_valid(d)
    assert any("steepened" in m for m in check_design(d))
    for ang in (40.0, 60.0, 85.0):
        assert_valid(make("hatchback", rear_angle=ang))
        assert_valid(make("suv", rear_angle=ang))
    assert_valid(make("pickup", bed_length=2.4))
    assert_valid(make("box_truck", box_edge_radius=0.0))
    assert_valid(make("van", rear_radius=0.0))
    assert_valid(make("fastback", nose_radius=0.6))             # clamped to fit


def test_front_wing_in_ground_effect():
    d = make("front_wing", ride_height=0.05, flap=True)
    assert_valid(d)
    assert d.kind == "vehicle" and d.source["ground"] is True
    assert d.length_ref == d.force_ref == pytest.approx(0.3)
    assert min(float(b.points[:, 1].min()) for b in d.bodies) == pytest.approx(0.05)
    assert [b.role for b in d.bodies] == ["element", "element"]
    main = d.bodies[0].points
    # inverted camber: the lower surface bulges more than the upper (camber toward the road)
    assert P.centroid(main)[1] < 0.5 * (main[:, 1].min() + main[:, 1].max()) + 1e-3


def test_basic_shapes_references():
    c = make("cylinder", diameter=2.0)
    assert c.length_ref == c.force_ref == 2.0 and c.moment_ref == (0.0, 0.0)
    assert P.signed_area(c.bodies[0].points) == pytest.approx(math.pi, rel=1e-3)
    sq = make("square", side=1.0, corner_radius=0.0)
    assert P.signed_area(sq.bodies[0].points) == pytest.approx(1.0, rel=1e-12)
    rr = make("rounded_rectangle", length=2.0, height=1.0, corner_radius=0.2)
    assert P.signed_area(rr.bodies[0].points) == pytest.approx(2.0 - (4 - math.pi) * 0.04, rel=1e-3)
    fp = make("flat_plate", orientation="normal")
    _x0, _x1, y0, y1 = fp.bounds()
    assert y1 - y0 == pytest.approx(1.0) and fp.force_ref == 1.0
    with pytest.raises(ValueError):
        make("flat_plate", edges="jagged")
    n = make("naca", naca="0012", chord=0.5)
    assert n.kind == "airfoil" and n.moment_ref == (0.125, 0.0)


# ---------------------------------------------------------------------------
# design builders and check_design
# ---------------------------------------------------------------------------

def test_design_from_polygons_vehicle_scaling_wheels_and_holes():
    body = make("hatchback").bodies[0].points * 250.0            # "pixels", any offset
    body = body + [37.0, -12.0]
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    wheel = lambda xc: np.column_stack([xc + 75 * np.cos(th), 63 + 75 * np.sin(th)])
    window = np.array([[550, 150], [750, 150], [750, 250], [550, 250]], float)    # inside the body
    d = design_from_polygons([body, wheel(220.0), wheel(820.0), window], "Imported", "vehicle",
                             length=4.1, ground_clearance=0.12)
    assert d.kind == "vehicle"
    x0, x1, y0, y1 = d.bounds()
    assert x0 == pytest.approx(0.0) and x1 - x0 == pytest.approx(4.1)
    assert y0 == pytest.approx(0.12)
    roles = sorted(b.role for b in d.bodies)
    assert roles == ["body", "wheel", "wheel"]                   # window dropped
    assert any("inside another" in m for m in d.source["import_notes"])
    assert d.length_ref == pytest.approx(4.1)
    assert d.force_ref == pytest.approx(y1 - y0)
    assert d.moment_ref == pytest.approx((2.05, 0.0))


def test_design_from_polygons_generic_airfoil_and_repair():
    sq = np.array([[0, 0], [3, 0], [3, 1], [0, 1]], float)
    d = design_from_polygons([sq[::-1]], "Box", "generic")       # clockwise input
    assert P.signed_area(d.bodies[0].points) > 0
    assert (d.length_ref, d.force_ref) == (3.0, 1.0) and d.moment_ref == (0.0, 0.0)
    k = np.arange(5)
    star = np.column_stack([np.cos(np.pi / 2 + 4 * np.pi * k / 5), np.sin(np.pi / 2 + 4 * np.pi * k / 5)])
    d2 = design_from_polygons([star], "Star", "generic")
    assert not P.is_self_intersecting(d2.bodies[0].points)
    assert any("crossed itself" in m for m in check_design(d2))
    a = design_from_polygons([naca("2412", 100) * 3 + 5], "Foil", "airfoil", length=0.25)
    assert a.kind == "airfoil" and a.length_ref == 0.25
    multi = design_from_polygons([naca("4412", 100), naca("4412", 80, 0.3) + [0.95, -0.05]], "2el", "airfoil")
    assert [b.role for b in multi.bodies] == ["element", "element"]
    with pytest.raises(ValueError):
        design_from_polygons([np.zeros((3, 2))], "nothing", "generic")


def test_check_design_resolution_and_te():
    d = design_from_airfoil(naca("0012", 200), "NACA 0012", 1.0)
    assert check_design(d) == []
    msgs = check_design(d, cells_per_ref=128)
    assert len(msgs) == 1 and "Trailing edge" in msgs[0]
    blunt = design_from_airfoil(naca("0012", 200, closed_te=False), "blunt", 1.0)
    assert any("blunt trailing-edge base" in m for m in check_design(blunt, cells_per_ref=128))
    # a thin plate is flagged as a thin feature, a thick body is not
    plate = make("flat_plate", thickness_ratio=0.01)
    assert any("thinner than 2 grid cells" in m for m in check_design(plate, cells_per_ref=100))
    assert check_design(make("cylinder"), cells_per_ref=100) == []


def test_check_design_overlap_gap_road():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    d = design_from_polygons([sq, sq + [0.5, 0.0]], "Two", "generic")
    # overlapping squares are merged-looking but kept as two bodies -> overlap reported
    assert any("overlap" in m for m in check_design(d))
    d2 = design_from_polygons([sq, sq + [1.004, 0.0]], "Gap", "generic")
    assert any("gap between" in m for m in check_design(d2, cells_per_ref=100))
    car = make("sedan")
    car.bodies[0].points[:, 1] -= 0.2                            # push it into the road
    assert any("below the road" in m for m in check_design(car))
