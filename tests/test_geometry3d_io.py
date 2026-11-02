"""Mesh file readers/writers and the units guess (aerolume.geometry3d.meshio)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import aerolume.geometry3d as G
from aerolume.model import Mesh3D

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "geometry3d"


def _same_surface(a: Mesh3D, b: Mesh3D, tol: float = 0.0) -> None:
    """Same triangles (as sets of corner coordinates, winding included up to rotation)."""
    def canon(m):
        t = m.vertices[m.faces]
        rows = []
        for tri in t:
            k = int(np.lexsort(tri.T[::-1])[0])          # rotate so the smallest corner is first
            rows.append(np.roll(tri, -k, axis=0).reshape(-1))
        r = np.array(rows)
        return r[np.lexsort(r.T[::-1])]
    ca, cb = canon(a), canon(b)
    assert ca.shape == cb.shape
    np.testing.assert_allclose(ca, cb, atol=tol, rtol=0)


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------

def test_stl_binary_round_trip_welds_vertices(tmp_path):
    s = G.sphere(1.0, 16)
    p = G.write_stl(s, tmp_path / "s.stl", binary=True)
    assert p.stat().st_size == 84 + 50 * len(s.faces)
    (m,) = G.read_stl(p)
    assert m.name == "s"
    # STL repeats every corner; the reader welds them back to the shared vertices
    assert len(m.vertices) == len(s.vertices) and len(m.faces) == len(s.faces)
    assert G.is_watertight(m)
    s32 = Mesh3D("s32", s.vertices.astype(np.float32).astype(np.float64), s.faces)
    _same_surface(m, s32)                                 # binary STL stores float32
    assert G.volume(m) > 0                                # winding (outward) preserved


def test_stl_binary_header_starting_with_solid_is_still_binary(tmp_path):
    s = G.box(1, 2, 3)
    p = G.write_stl(s, tmp_path / "b.stl")
    data = bytearray(p.read_bytes())
    data[:80] = b"solid exported by a CAD tool that writes ascii keywords".ljust(80)
    p.write_bytes(bytes(data))
    (m,) = G.read_mesh(p)
    assert len(m.faces) == 12 and G.is_watertight(m)


def test_stl_ascii_multi_solid_round_trip(tmp_path):
    a = G.box(1.0, 0.5, 0.25, name="body part")
    b = G.cylinder(0.3, 0.2, axis="y", n=10, center=(2.0, 0, 0), name="wheel")
    p = G.write_stl([a, b], tmp_path / "two.stl", binary=False)
    assert p.read_text().count("endsolid") == 2
    notes: list[str] = []
    ms = G.read_mesh(p, notes=notes)
    assert [m.name for m in ms] == ["body_part", "wheel"]
    _same_surface(ms[0], a)                                # ASCII keeps full float64 precision
    _same_surface(ms[1], b)
    assert any("2 solids" in n for n in notes)


def test_stl_ascii_tolerates_missing_endsolid_and_truncated_facet(tmp_path):
    txt = ("solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n"
           "   vertex 0 1 0\n  endloop\n endfacet\n facet normal 0 0 1\n  outer loop\n"
           "   vertex 0 0 1\n   vertex 1 0 1\n")
    p = tmp_path / "t.stl"
    p.write_text(txt)
    notes: list[str] = []
    (m,) = G.read_stl(p, notes=notes)
    assert len(m.faces) == 1
    assert any("incomplete facet" in n for n in notes)


def test_stl_drops_degenerate_triangles_with_note(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0.0]])
    m = Mesh3D("d", v, np.arange(6).reshape(2, 3))
    p = G.write_stl(m, tmp_path / "d.stl")
    notes: list[str] = []
    (r,) = G.read_stl(p, notes=notes)
    assert len(r.faces) == 1
    assert any("degenerate" in n for n in notes)


def test_negative_zero_welds_with_zero(tmp_path):
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-0.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    p = tmp_path / "z.stl"
    G.write_stl(Mesh3D("z", v, np.arange(6).reshape(2, 3)), p, binary=False)
    (m,) = G.read_stl(p)
    assert len(m.vertices) == 4


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------

def test_obj_groups_quads_negative_indices(tmp_path):
    txt = """# test
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vn 0 0 1
o thing
g body
f 1/1/1 2/1/1 3/1/1 4/1/1
v 5 0 0
v 6 0 0
v 5 1 0
g wheel
f -3 -2 -1
g body
f 1 3 4
"""
    p = tmp_path / "g.obj"
    p.write_text(txt)
    notes: list[str] = []
    ms = G.read_obj(p, notes=notes)
    assert [m.name for m in ms] == ["body", "wheel"]
    body, wheel = ms
    assert len(body.faces) == 3                              # quad fan (2) + later face (1)
    assert len(body.vertices) == 4                           # only the vertices it uses
    np.testing.assert_allclose(wheel.vertices, [[5, 0, 0], [6, 0, 0], [5, 1, 0]])
    np.testing.assert_array_equal(wheel.faces, [[0, 1, 2]])
    assert any("split into triangles" in n for n in notes)


def test_obj_round_trip(tmp_path):
    a = G.sphere(0.5, 12, name="ball")
    b = G.box(0.2, 0.2, 0.2, center=(1, 0, 0), name="cube one")
    p = G.write_obj([a, b], tmp_path / "w.obj")
    ms = G.read_mesh(p)
    assert [m.name for m in ms] == ["ball", "cube one"]
    for m, ref in zip(ms, (a, b)):
        np.testing.assert_array_equal(m.vertices, ref.vertices)   # full precision, same order
        np.testing.assert_array_equal(m.faces, ref.faces)


def test_obj_bad_faces_skipped(tmp_path):
    p = tmp_path / "bad.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 1 2 9\nf 1 2\nf 0 1 2\n")
    notes: list[str] = []
    (m,) = G.read_obj(p, notes=notes)
    assert len(m.faces) == 1
    assert any("skipped 3 face" in n for n in notes)


def test_obj_without_faces_raises(tmp_path):
    p = tmp_path / "pts.obj"
    p.write_text("v 0 0 0\nv 1 0 0\n")
    with pytest.raises(ValueError):
        G.read_obj(p)


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("binary", [True, False])
def test_ply_round_trip(tmp_path, binary):
    s = G.cylinder(0.4, 1.0, n=12)
    p = G.write_ply(s, tmp_path / "c.ply", binary=binary, double=True)
    (m,) = G.read_mesh(p)
    np.testing.assert_allclose(m.vertices, s.vertices, atol=1e-15)
    np.testing.assert_array_equal(m.faces, s.faces)


def test_ply_ascii_mixed_polygons_and_extra_properties(tmp_path):
    txt = """ply
format ascii 1.0
comment mixed
element vertex 5
property float x
property float y
property float z
property uchar red
element face 2
property uchar flags
property list uchar int vertex_indices
property float quality
element edge 1
property int vertex1
property int vertex2
end_header
0 0 0 255
1 0 0 255
1 1 0 255
0 1 0 255
0.5 0.5 1 255
7 4 0 1 2 3 0.5
7 3 0 1 4 0.25
0 1
"""
    p = tmp_path / "m.ply"
    p.write_text(txt)
    notes: list[str] = []
    (m,) = G.read_ply(p, notes=notes)
    assert len(m.faces) == 3                       # quad -> 2, triangle -> 1
    np.testing.assert_array_equal(m.faces[2], [0, 1, 4])
    assert any("split into triangles" in n for n in notes)


def test_ply_binary_big_endian_with_variable_lists(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], dtype=">f4")
    header = ("ply\nformat binary_big_endian 1.0\nelement vertex 5\nproperty float x\nproperty float y\n"
              "property float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n")
    body = v.tobytes()
    body += struct.pack(">B4i", 4, 0, 1, 2, 3) + struct.pack(">B3i", 3, 0, 1, 4)
    p = tmp_path / "be.ply"
    p.write_bytes(header.encode() + body)
    (m,) = G.read_ply(p)
    assert len(m.faces) == 3
    np.testing.assert_allclose(m.vertices[4], [0.5, 0.5, 1])


def test_ply_point_cloud_raises(tmp_path):
    p = tmp_path / "pc.ply"
    p.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                 "property float z\nend_header\n0 0 0\n")
    with pytest.raises(ValueError, match="no faces"):
        G.read_ply(p)


# ---------------------------------------------------------------------------
# dispatch, units, sample files
# ---------------------------------------------------------------------------

def test_read_mesh_units(tmp_path):
    s = G.sphere(100.0, 8)                                 # a 100 mm sphere in mm
    p = G.write_stl(s, tmp_path / "mm.stl", binary=False)
    (raw,) = G.read_mesh(p)
    assert np.ptp(raw.vertices[:, 0]) == pytest.approx(100.0)
    (m,) = G.read_mesh(p, units="mm")
    assert np.ptp(m.vertices[:, 0]) == pytest.approx(0.1)
    notes: list[str] = []
    (a,) = G.read_mesh(p, units="auto", notes=notes)
    assert np.ptp(a.vertices[:, 0]) == pytest.approx(0.1)
    assert any("millimetres" in n for n in notes)
    with pytest.raises(ValueError):
        G.read_mesh(tmp_path / "x.3ds")
    with pytest.raises(ValueError):
        G.unit_scale("furlong")


@pytest.mark.parametrize("size,scale", [(4.5, 1.0), (0.3, 1.0), (4500.0, 0.001), (300.0, 0.001),
                                        (120000.0, 0.001)])
def test_guess_units(size, scale):
    m = G.box(size, size / 3, size / 4)
    s, why = G.guess_units(m)
    assert s == scale
    assert why


def test_sample_files_read():
    """The generated samples in examples/geometry3d read with the expected structure."""
    (sph,) = G.read_mesh(EXAMPLES / "sphere_d100mm_binary.stl")
    assert G.is_watertight(sph) and G.guess_units(sph)[0] == 0.001
    two = G.read_mesh(EXAMPLES / "box_and_cylinder_ascii.stl")
    assert [m.name for m in two] == ["box", "cylinder"] and all(G.is_watertight(m) for m in two)
    obj = G.read_mesh(EXAMPLES / "body_and_wheel.obj")
    assert [m.name for m in obj] == ["body", "wheel"]
    assert all(G.is_watertight(m) and G.volume(m) > 0 for m in obj)
    (ply,) = G.read_mesh(EXAMPLES / "wedge_binary.ply")
    assert len(ply.faces) == 8 and G.is_watertight(ply)
    assert G.volume(ply) == pytest.approx(0.5 * 1.0 * 0.3 * 0.5, rel=1e-6)
    (leaky,) = G.read_mesh(EXAMPLES / "leaky_sphere.stl")
    info = G.mesh_info(leaky)
    assert not info["watertight"] and info["holes"] == 1 and info["boundary_edges"] > 10
