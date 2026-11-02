"""Mesh operations, primitives, references and checks (aerolume.geometry3d, numpy parts)."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

import aerolume.geometry3d as G
from aerolume.model import Design, Mesh3D, WingSection, WingSpec


def test_package_import_is_light():
    """The GUI imports geometry3d at start-up: no torch, no scipy until a function needs them."""
    import subprocess
    import sys

    code = ("import sys, aerolume.geometry3d as G; "
            "print(int('torch' in sys.modules), int('scipy.spatial' in sys.modules)); "
            "G.voxelize; print(int('torch' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert out.stdout.split() == ["0", "0", "1"]


# ---------------------------------------------------------------------------
# frame and sign conventions (DESIGN.md 7.2)
# ---------------------------------------------------------------------------

def test_positive_rotation_about_y_lifts_a_nose_at_minus_x():
    nose = np.array([-1.0, 0.0, 0.0])
    R = G.rotation_matrix((0.0, 10.0, 0.0))
    p = R @ nose
    assert p[2] > 0 and p[0] < 0                     # +alpha about +y: nose up (7.2)
    np.testing.assert_allclose(p, [-math.cos(math.radians(10)), 0, math.sin(math.radians(10))], atol=1e-15)


def test_right_hand_rule_about_z_and_x():
    np.testing.assert_allclose(G.rotation_matrix((0, 0, 90)) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(G.rotation_matrix((0, 0, 90)) @ [-1, 0, 0], [0, -1, 0], atol=1e-15)
    np.testing.assert_allclose(G.rotation_matrix((90, 0, 0)) @ [0, 1, 0], [0, 0, 1], atol=1e-15)


def test_rotation_order_is_x_then_y_then_z():
    # (0,1,0): x first -> (0,0,1), then y -> (1,0,0); the other order would give (0,0,1)
    np.testing.assert_allclose(G.rotation_matrix((90, 90, 0)) @ [0, 1, 0], [1, 0, 0], atol=1e-15)
    R = G.rotation_matrix((20, -35, 50))
    rx, ry, rz = (G.rotation_matrix(a) for a in ((20, 0, 0), (0, -35, 0), (0, 0, 50)))
    np.testing.assert_allclose(R, rz @ ry @ rx, atol=1e-15)


def test_transform_about_point_scale_translate_and_wheel_data():
    m = G.box(2, 1, 1, center=(1, 0, 0))
    m.center, m.axis = (1.0, 0.5, 0.0), (0.0, 1.0, 0.0)
    out = G.transform(m, scale=2.0, rotate_deg=(0, 0, 90), about=(1, 0, 0), translate=(0, 0, 5))
    lo = out.vertices.min(0)
    hi = out.vertices.max(0)
    np.testing.assert_allclose([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]], [0, 2, -2, 2, 4, 6], atol=1e-12)
    np.testing.assert_allclose(out.center, (0.0, 0.0, 5.0), atol=1e-12)     # (1,.5,0) -> about-rotated
    np.testing.assert_allclose(out.axis, (-1.0, 0.0, 0.0), atol=1e-12)
    assert G.volume(out) == pytest.approx(8 * 2.0, rel=1e-12)
    assert m.vertices.min() == -0.5                                          # input untouched


@pytest.mark.parametrize("kw", [{"mirror_axis": "y"}, {"scale": (-1.0, 1.0, 1.0)}, {"scale": (1, -2, -1)}])
def test_reflections_keep_outward_winding(kw):
    s = G.sphere(1.0, 16, center=(0.3, 0.2, 0.0))
    out = G.transform(s, **kw)
    assert G.volume(out) > 0
    assert G.inconsistent_edges(out) == 0


def test_mirror_about_plane():
    b = G.box(1, 1, 1, center=(0, 2, 0))
    m = G.mirror(b, "y", about=0.5)
    assert m.bounds()[2:4] == pytest.approx((-1.5, -0.5))
    assert G.volume(m) == pytest.approx(1.0)


def test_presets_are_proper_rotations():
    s = G.sphere(1.0, 12, center=(1, 2, 3))
    for name, rot in G.ORIENT_PRESETS.items():
        assert np.linalg.det(G.rotation_matrix(rot)) == pytest.approx(1.0)
        assert G.volume(G.orient_preset(s, name)) > 0
    up = G.orient_preset(Mesh3D("p", [[0, 1, 0], [0, 1, 0.1], [0.1, 1, 0]], [[0, 1, 2]]), "y-up->z-up")
    np.testing.assert_allclose(up.vertices[0], [0, 0, 1], atol=1e-15)        # file y (up) becomes z
    nose = G.orient_preset(Mesh3D("n", [[1, 0, 0], [1, 0.1, 0], [1, 0, 0.1]], [[0, 1, 2]]), "nose+x->nose-x")
    np.testing.assert_allclose(nose.vertices[0], [-1, 0, 0], atol=1e-15)
    with pytest.raises(ValueError):
        G.orient_preset(s, "sideways")


def test_place_on_road_centre_y_place_nose():
    b = G.box(4, 2, 1, center=(3, 1, 7))
    m = G.place_nose(G.centre_y(G.place_on_road(b, 0.15)))
    x0, _x1, y0, y1, z0, _z1 = G.bbox(m)
    assert (x0, y0 + y1, z0) == pytest.approx((0.0, 0.0, 0.15))


# ---------------------------------------------------------------------------
# measures and topology
# ---------------------------------------------------------------------------

def test_primitive_volumes_areas_and_normals():
    b = G.box(2, 3, 4)
    assert G.volume(b) == pytest.approx(24.0) and G.area(b) == pytest.approx(52.0)
    fn = G.face_normals(b)
    c = b.vertices[b.faces].mean(1)
    assert np.all(np.einsum("ij,ij->i", fn, c) > 0)                           # outward
    n = 24
    cyl = G.cylinder(0.5, 2.0, axis="x", n=n)
    prism = 0.5 * n * 0.25 ** 2 * math.sin(2 * math.pi / n) * 2.0             # exact n-gon prism
    assert G.volume(cyl) == pytest.approx(prism, rel=1e-12)
    assert cyl.bounds() == pytest.approx((-1, 1, -0.25, 0.25, -0.25, 0.25), abs=1e-12)
    s = G.sphere(2.0, 48)
    assert G.volume(s) == pytest.approx(4 / 3 * math.pi, rel=0.01)
    vn = G.vertex_normals(s)
    radial = s.vertices / np.linalg.norm(s.vertices, axis=1, keepdims=True)
    assert np.einsum("ij,ij->i", vn, radial).min() > 0.99
    for m in (b, cyl, s):
        info = G.mesh_info(m)
        assert info["watertight"] and info["holes"] == 0 and info["inconsistent_edges"] == 0
        assert info["components"] == 1 and info["volume"] > 0


def test_sedan_primitive_is_a_closed_symmetric_car_on_the_road():
    """The bundled 3D example's car: closed parts, outward winding, nose at x = 0,
    mirror-symmetric (so its run uses the symmetry plane), wheels recognised."""
    parts = G.sedan()
    assert [m.name for m in parts] == ["body", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
    for m in parts:
        info = G.mesh_info(m)
        assert info["watertight"] and info["inconsistent_edges"] == 0 and info["volume"] > 0
    assert G.bbox(parts) == pytest.approx((0.0, 4.5, -0.91, 0.91, 0.0, 1.41), abs=1e-6)
    d = G.design_from_meshes(parts, "Sedan", "vehicle")
    assert d.symmetric and d.length_ref == pytest.approx(4.5)
    assert [m.role for m in d.meshes] == ["body"] + ["wheel"] * 4
    assert 2.0 < d.ref_area < 2.4                                   # frontal area of a mid-size car [m²]
    small = G.sedan(2.25, wheels=False)
    assert len(small) == 1 and G.bbox(small)[1] == pytest.approx(2.25)
    with pytest.raises(ValueError):
        G.sedan(0.0)


def test_holes_nonmanifold_and_flipped_faces_are_counted():
    s = G.sphere(1.0, 16)
    open_ = Mesh3D("o", s.vertices, s.faces[1:])
    info = G.mesh_info(open_)
    assert not info["watertight"] and info["boundary_edges"] == 3 and info["holes"] == 1
    assert len(G.boundary_edges(open_)) == 3
    f = s.faces.copy()
    f[5] = f[5][[0, 2, 1]]
    assert G.inconsistent_edges(Mesh3D("f", s.vertices, f)) == 3
    # a fin glued along one existing edge: that edge now has three faces
    e = s.faces[0][:2]
    v = np.vstack([s.vertices, [[2.0, 2.0, 2.0]]])
    fin = Mesh3D("fin", v, np.vstack([s.faces, [[e[0], e[1], len(v) - 1]]]))
    info = G.mesh_info(fin)
    assert info["nonmanifold_edges"] == 1 and not G.is_watertight(fin)


def test_orient_outward_repairs_random_flips_per_component():
    rng = np.random.default_rng(3)
    a = G.sphere(1.0, 20)
    b = G.box(0.5, 0.5, 0.5, center=(3, 0, 0))
    b = Mesh3D("b", b.vertices, b.faces[:, [0, 2, 1]])                          # inside out
    m = G.merge([a, b])
    f = m.faces.copy()
    flip = rng.random(len(f)) < 0.4
    f[flip] = f[flip][:, [0, 2, 1]]
    bad = Mesh3D("bad", m.vertices, f)
    assert G.inconsistent_edges(bad) > 0
    good = G.orient_outward(bad)
    assert G.inconsistent_edges(good) == 0
    assert G.volume(good) == pytest.approx(G.volume(a) + 0.125, rel=1e-12)
    np.testing.assert_array_equal(np.sort(good.faces, 1), np.sort(bad.faces, 1))   # same triangles


def test_weld_remove_degenerate_merge_compact():
    s = G.sphere(1.0, 12)
    t = s.vertices[s.faces].reshape(-1, 3)
    soup = Mesh3D("soup", t + np.random.default_rng(0).normal(0, 1e-9, t.shape),
                  np.arange(len(t)).reshape(-1, 3))
    assert not G.is_watertight(soup)
    w = G.weld(soup, tol=1e-6)
    assert len(w.vertices) == len(s.vertices) and G.is_watertight(w)
    dup = Mesh3D("d", s.vertices, np.vstack([s.faces, s.faces[:3, [0, 2, 1]], [[0, 0, 1]]]))
    clean = G.remove_degenerate(dup)
    assert len(clean.faces) == len(s.faces) and G.is_watertight(clean)
    two = G.merge([G.box(1, 1, 1), G.box(1, 1, 1, center=(3, 0, 0))], name="two")
    assert two.name == "two" and len(two.faces) == 24 and G.components(two)[0] == 2
    extra = Mesh3D("x", np.vstack([s.vertices, [[9, 9, 9]]]), s.faces)
    assert len(G.compact(extra).vertices) == len(s.vertices)


def test_decimate_keeps_shape():
    s = G.sphere(1.0, 64)
    d = G.decimate(s, 600)
    assert 0.5 * 600 <= len(d.faces) <= 600
    assert G.volume(d) == pytest.approx(G.volume(s), rel=0.08)
    assert G.decimate(s, 10**6).faces.shape == s.faces.shape


def test_wheel_params():
    w = G.cylinder(0.6, 0.2, axis="y", n=32, center=(1.0, -0.7, 0.3))
    c, ax, r = G.wheel_params(w)
    np.testing.assert_allclose(c, (1.0, -0.7, 0.3), atol=1e-9)
    np.testing.assert_allclose(ax, (0.0, 1.0, 0.0), atol=1e-9)
    assert r == pytest.approx(0.3, rel=1e-3)


# ---------------------------------------------------------------------------
# symmetry, frontal area, auto-orientation
# ---------------------------------------------------------------------------

def test_symmetry_error():
    b = G.box(2.0, 1.0, 0.5)
    assert G.symmetry_error(b) < 1e-12
    s = G.sphere(1.0, 32)
    # the UV sphere's quad diagonals all turn the same way, so its mirror image is a
    # different triangulation of the same quads: tiny, but real
    assert G.symmetry_error(s) < 1e-3
    # tessellation does not matter, only the surface: a rotated sphere is still symmetric
    r = G.transform(s, rotate_deg=(0, 0, 7.0))
    assert G.symmetry_error(r) < 0.01
    lump = G.merge([s, G.box(0.2, 0.2, 0.2, center=(0, 0.55, 0))])
    e = G.symmetry_error(lump)
    assert 0.005 < e < 0.2
    shifted = G.transform(s, translate=(0, 0.3, 0))
    assert G.symmetry_error(shifted) > 0.1
    assert G.symmetry_error(shifted, plane_y=None) < 1e-3          # about its own centre


def test_frontal_area_sphere_box_and_union():
    s = G.sphere(1.0, 64)
    a = G.frontal_area(s)
    assert a == pytest.approx(math.pi / 4, rel=0.01)
    ngon = 0.125 * 64 * math.sin(2 * math.pi / 64)                    # the exact silhouette (n-gon)
    assert a == pytest.approx(ngon, rel=0.002)
    assert G.frontal_area(s, budget=500) == a                         # chunking does not change it
    b = G.box(3.0, 1.5, 0.8)
    assert G.frontal_area(b) == pytest.approx(1.2, rel=0.003)
    assert G.frontal_area(b, axis="z") == pytest.approx(4.5, rel=0.003)
    tandem = [s, G.sphere(1.0, 64, center=(3, 0, 0))]                  # one behind the other
    assert G.frontal_area(tandem) == pytest.approx(a, rel=1e-9)
    side = [s, G.sphere(1.0, 64, center=(0, 3, 0))]
    assert G.frontal_area(side) == pytest.approx(2 * a, rel=0.003)


def _car_like(length_axis: int, up_axis: int) -> Mesh3D:
    """Body 4 long, 1.8 wide, 1.2 high with a 2 x 1.4 x 0.6 cabin on top, in file axes."""
    body = G.box(4.0, 1.8, 1.2, center=(0, 0, 0.6))
    cabin = G.box(2.0, 1.4, 0.6, center=(0.3, 0, 1.5))
    m = G.merge([body, cabin], name="car")
    lat = 3 - length_axis - up_axis
    v = np.zeros_like(m.vertices)
    v[:, length_axis], v[:, lat], v[:, up_axis] = m.vertices[:, 0], m.vertices[:, 1], m.vertices[:, 2]
    f = m.faces
    if np.linalg.det(np.eye(3)[[length_axis, lat, up_axis]]) < 0:
        f = f[:, [0, 2, 1]]
    return Mesh3D("car", v, f)


@pytest.mark.parametrize("length_axis,up_axis", [(0, 2), (2, 1), (1, 0), (0, 1)])
def test_auto_orient_finds_length_and_up(length_axis, up_axis):
    m = _car_like(length_axis, up_axis)
    notes: list[str] = []
    out = G.auto_orient(m, notes=notes)
    x0, x1, y0, y1, z0, z1 = G.bbox(out)
    assert (x1 - x0, y1 - y0, z1 - z0) == pytest.approx((4.0, 1.8, 1.8))
    # the cabin (narrower, on top) must end up at the top: the top slice is narrower
    top = out.vertices[out.vertices[:, 2] > z1 - 0.1]
    assert np.ptp(top[:, 1]) == pytest.approx(1.4)
    assert G.volume(out) > 0 and notes


def test_auto_orient_tall_truck_uses_symmetry_not_size():
    # taller (3.5) than wide (2.5): the symmetric axis still becomes y
    body = G.box(6.0, 2.5, 3.5, center=(0, 0, 1.75))
    bump = G.box(1.0, 2.5, 0.5, center=(-2.0, 0, 3.75))               # asymmetric top-bottom
    m = G.merge([body, bump])
    file_ = G.transform(m, rotate_deg=(90, 0, 0))                      # now y-up, z lateral
    out = G.auto_orient(file_)
    x0, x1, y0, y1, z0, z1 = G.bbox(out)
    assert (x1 - x0, y1 - y0, z1 - z0) == pytest.approx((6.0, 2.5, 4.0))


def test_auto_orient_keeps_an_already_good_orientation():
    m = _car_like(0, 2)
    notes: list[str] = []
    out = G.auto_orient(m, notes=notes)
    np.testing.assert_array_equal(out.vertices, m.vertices)
    assert "kept" in notes[0]


# ---------------------------------------------------------------------------
# designs
# ---------------------------------------------------------------------------

def _symmetric_car() -> list[Mesh3D]:
    body = G.box(4.0, 1.7, 1.0, center=(2.0, 0, 0.75), name="body")
    wheels = [G.cylinder(0.6, 0.22, axis="y", n=24, center=(x, y, 0.3), name=f"wheel {k}")
              for k, (x, y) in enumerate([(0.8, -0.75), (0.8, 0.75), (3.2, -0.75), (3.2, 0.75)])]
    return [body, *wheels]


def test_design_from_meshes_vehicle():
    ms = G.transform(_symmetric_car(), scale=1000.0, translate=(500, 300, 200))    # mm, off-centre
    d = G.design_from_meshes(ms, "Car", "vehicle", units="mm")
    assert d.dimension == 3 and d.kind == "vehicle" and d.bodies == []
    x0, _x1, y0, y1, z0, _z1 = d.bounds3()
    assert (x0, y0 + y1, z0) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert d.length_ref == pytest.approx(4.0) and d.span_ref == pytest.approx(1.72)
    assert d.ref_area == pytest.approx(G.frontal_area(d.meshes), rel=1e-12)
    # body 1.7 x 1.0, plus the wheels below it (0.22 x 0.25 per side) and beside it (0.01 x 0.35)
    assert d.ref_area == pytest.approx(1.7 + 2 * (0.22 * 0.25 + 0.01 * 0.35), rel=0.005)
    assert d.moment_ref3 == pytest.approx((2.0, 0.0, 0.0))
    assert d.symmetric and d.source["symmetry_error"] < 1e-6
    roles = [m.role for m in d.meshes]
    assert roles.count("wheel") == 4
    w = next(m for m in d.meshes if m.role == "wheel")
    assert w.axis == pytest.approx((0, 1, 0)) and w.center[2] == pytest.approx(0.3)
    assert d.force_ref == pytest.approx(1.25)
    # one wheel moved: no longer symmetric
    ms2 = _symmetric_car()
    ms2[1] = G.transform(ms2[1], translate=(0.4, 0, 0))
    assert not G.design_from_meshes(ms2, "Car", "vehicle").symmetric


def test_design_from_meshes_compact_bluff_body_and_length():
    s = G.sphere(100.0, 48)                                   # a 100 mm sphere in mm
    d = G.design_from_meshes(s, "Ball", "generic", units="auto")
    assert d.length_ref == pytest.approx(0.1)                # diameter, not "length"
    assert d.ref_area == pytest.approx(math.pi * 0.1 ** 2 / 4, rel=0.01)
    assert d.moment_ref3 == pytest.approx((0, 0, 0), abs=1e-12) and d.symmetric
    notes = d.source["import_notes"]
    assert any("millimetres" in n for n in notes) and any("Compact" in n for n in notes)
    cyl = G.design_from_meshes(G.cylinder(0.1, 0.6, axis="y"), "Cyl", "generic")
    assert cyl.length_ref == pytest.approx(0.1)               # a cylinder across the flow: its diameter
    vert = G.design_from_meshes(G.cylinder(0.2, 0.6, axis="z"), "Post", "generic")
    assert vert.length_ref == pytest.approx(0.2)              # standing cylinder: its diameter
    plate = G.design_from_meshes(G.box(0.01, 1.0, 0.5), "Plate", "generic")
    assert plate.length_ref == pytest.approx(0.5)             # a plate facing the flow: its smaller side
    wing = G.loft(WingSpec([WingSection(0.0, 1.0), WingSection(2.0, 1.0)]))
    assert G.design_from_meshes(wing, "W", "generic").length_ref == pytest.approx(1.0)   # chord, not thickness
    long_ = G.design_from_meshes(G.box(5, 1, 1), "Bar", "generic", length=10.0)
    assert long_.length_ref == pytest.approx(10.0) and long_.ref_area == pytest.approx(4.0, rel=0.005)
    with pytest.raises(ValueError):
        G.design_from_meshes(s, "x", "airfoil")


def test_design_from_meshes_fixes_winding_but_not_vertices():
    s = G.sphere(1.0, 16)
    f = s.faces.copy()
    f[::3] = f[::3][:, [0, 2, 1]]
    d = G.design_from_meshes(Mesh3D("s", s.vertices, f), "S", "generic")
    m = d.meshes[0]
    assert G.inconsistent_edges(m) == 0 and G.volume(m) > 0
    assert len(m.vertices) == len(s.vertices)
    assert any("re-wound" in n for n in d.source["import_notes"])


def test_design_round_trips_through_dict():
    d = G.design_from_meshes(_symmetric_car(), "Car", "vehicle")
    d2 = Design.from_dict(d.to_dict())
    d2.attach_mesh_arrays(d.mesh_arrays())
    assert d2.ref_area == d.ref_area and d2.symmetric == d.symmetric
    assert [m.role for m in d2.meshes] == [m.role for m in d.meshes]
    np.testing.assert_array_equal(d2.meshes[1].vertices, d.meshes[1].vertices)


def _design(meshes, **kw) -> Design:
    base = {"name": "t", "kind": "vehicle", "bodies": [], "length_ref": 4.0, "force_ref": 1.0, "dimension": 3,
            "meshes": meshes, "ref_area": 2.0, "span_ref": 1.8}
    base.update(kw)
    return Design(**base)


def test_check_design3d_clean_and_problems():
    car = G.design_from_meshes(_symmetric_car(), "Car", "vehicle")
    car.source["import_notes"] = []
    assert G.check_design3d(car) == []
    # leaky + flipped + below the road + wrongly marked symmetric
    s = G.sphere(1.0, 24, center=(2.0, 0.6, 0.3))
    f = s.faces[5:].copy()
    f[::4] = f[::4][:, [0, 2, 1]]
    msgs = G.check_design3d(_design([Mesh3D("shell", s.vertices, f)], symmetric=True))
    text = " ".join(msgs)
    assert "not closed" in text and "opposite winding" in text
    assert "below the road" in text and "marked symmetric" in text
    assert any("reference area" in m for m in G.check_design3d(_design([G.box(1, 1, 1)], ref_area=0.0)))
    assert any("check the units" in m for m in G.check_design3d(_design([G.box(4000, 1800, 1400)])))
    assert G.check_design3d(_design([])) == ["The design has no meshes."]


def test_check_design3d_resolution_messages():
    plate = G.box(0.02, 1.0, 0.5, center=(2.0, -1.5, 1.0), name="fin")      # 2 cm thick
    body = G.box(4.0, 1.6, 1.0, center=(2.0, 0.2, 0.62), name="body")      # 12 cm clearance
    d = _design([body, plate], kind="vehicle")
    msgs = " ".join(G.check_design3d(d, cells_per_ref=64))                 # h = 6.25 cm
    assert "'fin' is only 2 cm thick" in msgs
    assert "Ground clearance" in msgs
    coarse = " ".join(G.check_design3d(d, cells_per_ref=8))
    assert "grid cells across the height" in coarse
    gap = _design([G.box(1, 1, 1, center=(0.5, -0.53, 0.6), name="a"),
                   G.box(1, 1, 1, center=(0.5, 0.53, 0.6), name="b")], kind="generic")
    assert "gap between 'a' and 'b'" in " ".join(G.check_design3d(gap, cells_per_ref=32))


def test_part_gaps_are_measured_where_parts_are_close():
    """Gaps: exact both ways, only between parts whose boxes come within the margin, and never
    for parts that touch or push into each other (a mirror pushed into the door)."""
    body = G.box(4.0, 1.8, 0.8, center=(2.0, 0.0, 0.7), name="body")
    mirror_gap = G.box(0.2, 0.25, 0.15, center=(2.9, 0.98, 1.2), name="mirror")     # 2.5 cm above the body
    mirror_in = G.box(0.2, 0.25, 0.15, center=(2.9, 0.98, 1.05), name="mirror")     # pushed into it
    far = G.box(0.2, 0.2, 0.2, center=(9.0, 0.0, 0.5), name="far")
    assert G.part_gap(body, mirror_gap, 0.3) == pytest.approx(0.025, abs=1e-9)
    assert G.part_gap(mirror_gap, body, 0.3) == pytest.approx(0.025, abs=1e-9)
    assert G.part_gap(body, mirror_in, 0.3) == math.inf and G.part_gap(mirror_in, body, 0.3) == math.inf
    assert G.part_gap(body, far, 0.3) == math.inf
    cabin = G.box(2.0, 1.6, 0.6, center=(2.2, 0.0, 1.4), name="cabin")             # sits on the body: touching
    assert G.part_gap(body, cabin, 0.3) == math.inf
    # nine parts with the resolution checks: well under a second (it took 4.4 s with every pair
    # sampled densely); the messages name exactly the real gaps
    parts = [body, cabin, G.box(0.3, 1.9, 0.05, center=(3.9, 0.0, 1.45), name="spoiler"),
             mirror_gap, G.box(0.2, 0.25, 0.15, center=(2.9, -0.98, 1.2), name="mirror_r")]
    parts += [G.cylinder(0.6, 0.25, axis="y", n=32, center=(x, y, 0.3), name=f"wheel {x} {y}")
              for x in (0.8, 3.2) for y in (-0.8, 0.8)]
    d = G.design_from_meshes(parts, "car", kind="vehicle", units="m")
    t0 = time.perf_counter()
    msgs = [m for m in G.check_design3d(d, cells_per_ref=24) if "gap between" in m]
    assert time.perf_counter() - t0 < 2.0
    named = sorted(m.split("'")[1] + "/" + m.split("'")[3] for m in msgs)
    assert named == ["body/mirror", "body/mirror_r", "body/spoiler", "cabin/mirror", "cabin/mirror_r"]
