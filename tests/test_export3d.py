"""3D exports: STL / OBJ geometry, VTK PolyData surfaces and ImageData volumes (checked with VTK's own
readers when available), slice / wing CSVs, and result / history CSVs of 3D results."""
from __future__ import annotations

import base64
import math
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from aerolume.export import (
    history_csv,
    mesh_obj,
    mesh_stl,
    mirror_volume,
    read_csv,
    result_csv,
    slice_csv,
    surface_vtp,
    volume_npz,
    volume_vti,
    wing_csv,
    wing_span_csv,
)
from aerolume.export.vtkxml import BLOCK_SIZE, VtkXmlWriter
from aerolume.model import (
    CFD3DResult,
    Design,
    Mesh3D,
    SliceData,
    Stat,
    SurfaceData3D,
    VolumeData,
    WingPoint,
    WingResult,
    WingSection,
    WingSpec,
)


def box(lx=2.0, ly=1.0, lz=0.5, origin=(0.0, -0.5, 0.1)) -> Mesh3D:
    """A closed box, outward-wound triangles."""
    x0, y0, z0 = origin
    v = np.array([[x0 + i * lx, y0 + j * ly, z0 + k * lz] for k in (0, 1) for j in (0, 1) for i in (0, 1)], float)
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                  [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return Mesh3D("box", v, f)


def tetra(name="fin", shift=(3.0, 0.0, 0.0)) -> Mesh3D:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * 0.3 + np.asarray(shift)
    return Mesh3D(name, v, np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]), role="element")


def design3d() -> Design:
    return Design(name="Car", kind="vehicle", bodies=[], length_ref=2.0, force_ref=0.0, dimension=3,
                  meshes=[box(), tetra()], ref_area=0.5)


def surfaces() -> list[SurfaceData3D]:
    b, t = box(), tetra()
    rng = np.random.default_rng(0)
    return [SurfaceData3D("box", b.vertices, b.faces, cp=np.linspace(-1, 1, 8), cf=np.full(8, 2e-3),
                          tau=rng.standard_normal((8, 3)) * 1e-3, sensitivity=None),
            SurfaceData3D("fin", t.vertices, t.faces, cp=np.array([0.5, -0.2, 0.1, 0.0]), cf=None, tau=None,
                          sensitivity=np.array([1e-4, -2e-4, 0.0, 3e-4]))]


def stat(m: float) -> Stat:
    return Stat(mean=m, rms=0.01, min=m - 0.1, max=m + 0.1, stderr=0.002)


def cfd3d() -> CFD3DResult:
    t = np.linspace(0, 10, 101)
    hist = {"t": t, "cd": 0.3 + 0 * t, "cs": 0 * t, "cl": -0.1 + 0 * t, "c_roll": 0 * t, "cm": 0.02 + 0 * t,
            "cn": 0 * t, "extra": t * 2}
    hist = {k: hist[k] for k in ("t", "cl", "cd", "extra", "cm", "cs", "cn", "c_roll")}   # stored out of order
    return CFD3DResult(sim_id="s", cd=stat(0.3), cs=stat(0.0), cl=stat(-0.1), c_roll=stat(0.0), cm=stat(0.02),
                       cn=stat(0.0), drag=330.0, side=0.0, lift=-110.0, moments=(0.0, 44.0, 0.0), drag_area=0.15,
                       power=9900.0, re_real=4e6, re_sim=1e5, grid=(256, 64, 96), cells=256 * 64 * 96,
                       symmetry_plane=True, converged=True, notes=["n1"], warnings=["w1"],
                       per_body={"box": {"cd": 0.28}}, history=hist, surface=surfaces())


def wing_result() -> WingResult:
    pts = []
    for a in (-2.0, 0.0, 2.0, 4.0, 6.0):
        cl = 0.08 * (a + 2)
        y = np.array([-0.9, -0.3, 0.3, 0.9])
        pts.append(WingPoint(alpha=a, cl=cl, cd=0.01 + cl * cl / 25, cdi=cl * cl / 26, cdv=0.01, cm=-0.03,
                             cy=0.0, c_roll=0.0, cn=0.0, e=0.97 if a else float("nan"), converged=a != 4.0,
                             y=y if a != 6.0 else None, cl_local=cl * np.array([0.8, 1, 1, 0.8]),
                             c_cl=np.array([0.7, 1.1, 1.1, 0.7]), alpha_i=np.full(4, -0.5),
                             cd_local=None if a < 0 else np.full(4, 0.01)))
    return WingResult(sim_id="w", method="vlm", viscous=True, area=1.2, span=2.0, aspect_ratio=2.0 ** 2 / 1.2,
                      mac=0.6, re_mac=5e5, points=pts, notes=["note"], warnings=["warn"])


# --- helpers: our own decoder of VTK XML binary blocks --------------------------------

def decode(text: str, dtype: str, compressed: bool) -> tuple[np.ndarray, int]:
    """Decode one base64 block at the start of ``text``; returns (values, characters used)."""
    if not compressed:
        n = struct.unpack("<Q", base64.b64decode(text[:12])[:8])[0]
        nchars = 4 * math.ceil((8 + n) / 3)
        raw = base64.b64decode(text[:nchars])[8:]
        return np.frombuffer(raw, dtype=dtype), nchars
    head3 = np.frombuffer(base64.b64decode(text[:32])[:24], "<u8")
    nb = int(head3[0])
    hchars = 4 * math.ceil(8 * (3 + nb) / 3)
    head = np.frombuffer(base64.b64decode(text[:hchars]), "<u8")
    csum = int(head[3:].sum())
    dchars = 4 * math.ceil(csum / 3)
    data = base64.b64decode(text[hchars:hchars + dchars])
    out, pos = [], 0
    for c in head[3:]:
        out.append(zlib.decompress(data[pos:pos + int(c)]))
        pos += int(c)
    raw = b"".join(out)
    full = int(head[1]) * (nb - 1) + (int(head[2]) or int(head[1])) if nb else 0
    assert len(raw) == full
    return np.frombuffer(raw, dtype=dtype), hchars + dchars


def appended_arrays(path) -> dict[str, np.ndarray]:
    root = ET.parse(path).getroot()
    compressed = root.get("compressor") == "vtkZLibDataCompressor"
    text = path.read_text(encoding="ascii")
    blob = text[text.index("_", text.index("<AppendedData")) + 1:].split("\n", 1)[0]
    types = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8"}
    out = {}
    for da in root.iter("DataArray"):
        off = int(da.get("offset"))
        vals, _ = decode(blob[off:], types[da.get("type")], compressed)
        n = int(da.get("NumberOfComponents", "1"))
        out[da.get("Name")] = vals.reshape(-1, n) if n > 1 else vals
    return out


# --- STL / OBJ -------------------------------------------------------------------------

def test_binary_stl_round_trip(tmp_path):
    d = design3d()
    path = mesh_stl(d, tmp_path / "car.stl")
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    assert n == 16 and len(data) == 84 + 50 * n and data[:13] == b"Aerolume STL,"
    rec = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    expect = np.concatenate([m.vertices[m.faces] for m in d.meshes])
    np.testing.assert_allclose(rec["v"], expect, atol=1e-6)
    # normals: outward for the box (right-hand rule), unit length
    np.testing.assert_allclose(np.linalg.norm(rec["n"], axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(rec["n"][0], [0, 0, -1], atol=1e-6)            # bottom face
    mm = np.frombuffer(mesh_stl(d.meshes[0], tmp_path / "mm.stl", units="mm").read_bytes()[84:],
                       dtype=rec.dtype)
    np.testing.assert_allclose(mm["v"], rec["v"][:12] * 1000, rtol=1e-6)
    vtk_io = pytest.importorskip("vtkmodules.vtkIOGeometry")
    r = vtk_io.vtkSTLReader()
    r.SetFileName(str(path))
    r.Update()
    assert r.GetOutput().GetNumberOfCells() == 16


def test_ascii_stl_and_obj(tmp_path):
    d = design3d()
    text = mesh_stl(d, tmp_path / "car_ascii.stl", binary=False).read_text(encoding="ascii")
    solids = [ln for ln in text.splitlines() if ln.startswith("solid ")]
    assert solids == ["solid box", "solid fin"] and text.count("endsolid fin") == 1
    assert text.count("facet normal") == 16
    vs = np.array([list(map(float, ln.split()[1:])) for ln in text.splitlines() if ln.strip().startswith("vertex")])
    np.testing.assert_array_equal(vs, np.concatenate([m.vertices[m.faces] for m in d.meshes]).reshape(-1, 3))

    obj = mesh_obj(d, tmp_path / "car.obj").read_text(encoding="ascii")
    lines = obj.splitlines()
    assert [ln for ln in lines if ln.startswith("o ")] == ["o box", "o fin"]
    v = np.array([list(map(float, ln.split()[1:])) for ln in lines if ln.startswith("v ")])
    f = np.array([list(map(int, ln.split()[1:])) for ln in lines if ln.startswith("f ")])
    np.testing.assert_array_equal(v, np.vstack([m.vertices for m in d.meshes]))
    np.testing.assert_array_equal(f[:12] - 1, d.meshes[0].faces)
    np.testing.assert_array_equal(f[12:] - 1 - 8, d.meshes[1].faces)
    vtk_io = pytest.importorskip("vtkmodules.vtkIOGeometry")
    r = vtk_io.vtkOBJReader()
    r.SetFileName(str(tmp_path / "car.obj"))
    r.Update()
    assert r.GetOutput().GetNumberOfCells() == 16 and r.GetOutput().GetNumberOfPoints() == 12


def test_mesh_export_inputs_and_errors(tmp_path):
    r = cfd3d()
    path = mesh_obj(r, tmp_path / "solved.obj")                  # a result's solved surfaces
    assert path.read_text(encoding="ascii").count("\nf ") == 16
    mesh_stl([box(), tetra()], tmp_path / "list.stl")
    with pytest.raises(ValueError, match="units"):
        mesh_stl(box(), tmp_path / "x.stl", units="ft")
    bad = Mesh3D("bad", np.zeros((3, 3)), np.array([[0, 1, 5]]))
    with pytest.raises(ValueError, match="missing vertices"):
        mesh_obj(bad, tmp_path / "bad.obj")
    empty = Design(name="E", kind="generic", bodies=[], length_ref=1.0, force_ref=0.0, dimension=3)
    with pytest.raises(ValueError, match="no 3D meshes"):
        mesh_stl(empty, tmp_path / "e.stl")
    wing = Design(name="W", kind="wing", bodies=[], length_ref=1.0, force_ref=0.0, dimension=3,
                  wing=WingSpec([WingSection(0.0, 1.0), WingSection(2.0, 0.5)]))
    try:
        p = mesh_stl(wing, tmp_path / "wing.stl")                # lofted when geometry3d is available
        assert p.stat().st_size > 84
    except ValueError as exc:
        assert "lofting is unavailable" in str(exc)
    assert not (tmp_path / "e.stl").exists() and not (tmp_path / "x.stl").exists()


# --- surface .vtp ------------------------------------------------------------------------

@pytest.mark.parametrize("compress", [True, False])
def test_surface_vtp_content(tmp_path, compress):
    r = cfd3d()
    path = surface_vtp(r, tmp_path / f"surf_{compress}.vtp", compress=compress)
    root = ET.parse(path).getroot()
    assert root.get("type") == "PolyData" and (root.get("compressor") is not None) == compress
    piece = root.find("PolyData/Piece")
    assert piece.get("NumberOfPoints") == "12" and piece.get("NumberOfPolys") == "16"
    arr = appended_arrays(path)
    assert set(arr) == {"cp", "cf", "tau", "sensitivity", "body_index", "Points", "connectivity", "offsets"}
    np.testing.assert_allclose(arr["cp"], np.concatenate([s.cp for s in r.surface]), rtol=1e-7)
    assert np.all(np.isnan(arr["cf"][8:])) and np.all(arr["cf"][:8] == np.float32(2e-3))
    assert arr["tau"].shape == (12, 3) and np.all(np.isnan(arr["tau"][8:]))
    assert np.all(np.isnan(arr["sensitivity"][:8]))
    np.testing.assert_allclose(arr["sensitivity"][8:], r.surface[1].sensitivity, rtol=1e-6)
    np.testing.assert_array_equal(arr["connectivity"].reshape(-1, 3)[12:], r.surface[1].faces + 8)
    np.testing.assert_array_equal(arr["offsets"], 3 * np.arange(1, 17))
    np.testing.assert_array_equal(arr["body_index"], [0] * 12 + [1] * 4)
    assert "0 = box, 1 = fin" in path.read_text(encoding="ascii")

    vtk_xml = pytest.importorskip("vtkmodules.vtkIOXML")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    rd = vtk_xml.vtkXMLPolyDataReader()
    rd.SetFileName(str(path))
    rd.Update()
    out = rd.GetOutput()
    assert out.GetNumberOfPoints() == 12 and out.GetNumberOfPolys() == 16
    np.testing.assert_allclose(vtk_to_numpy(out.GetPointData().GetArray("cp")), arr["cp"])
    np.testing.assert_allclose(vtk_to_numpy(out.GetPoints().GetData()),
                               np.vstack([s.vertices for s in r.surface]), rtol=1e-7)
    assert out.GetPointData().GetVectors().GetName() == "tau"


def test_surface_vtp_inline_and_ascii(tmp_path):
    s = surfaces()[0]
    for enc in ("inline", "ascii"):
        path = surface_vtp(s, tmp_path / f"s_{enc}.vtp", encoding=enc, compress=False)
        root = ET.parse(path).getroot()
        cp = next(da for da in root.iter("DataArray") if da.get("Name") == "cp")
        vals = (decode(cp.text.strip(), "<f4", False)[0] if enc == "inline"
                else np.array(cp.text.split(), dtype=np.float32))
        np.testing.assert_allclose(vals, s.cp, rtol=1e-7)
        names = {da.get("Name") for da in root.iter("DataArray")}
        assert "sensitivity" not in names                        # no surface has it: left out
    with pytest.raises(ValueError, match="no surface"):
        surface_vtp([], tmp_path / "none.vtp")
    bad = SurfaceData3D("b", s.vertices, s.faces, cp=np.zeros(3))
    with pytest.raises(ValueError, match="values for 8 vertices"):
        surface_vtp(bad, tmp_path / "bad.vtp")


@pytest.mark.parametrize("n", [BLOCK_SIZE // 4, BLOCK_SIZE // 4 * 3 + 5, 0])
def test_compressed_blocks_full_partial_and_empty(tmp_path, n):
    """Exact multiples of the 32 KiB block (last-block size 0), partial last blocks and empty arrays."""
    vals = np.arange(n, dtype=np.float32) * 0.5
    w = VtkXmlWriter("ImageData", compress=True)
    w.line(f'  <ImageData WholeExtent="0 {n} 0 1 0 1" Origin="0 0 0" Spacing="1 1 1">')
    w.line(f'    <Piece Extent="0 {n} 0 1 0 1">')
    w.line("      <CellData>")
    w.array("a", vals)
    for t in ("      </CellData>", "    </Piece>", "  </ImageData>"):
        w.line(t)
    path = w.write(tmp_path / f"b{n}.vti")
    np.testing.assert_array_equal(appended_arrays(path)["a"], vals)
    if n:
        vtk_xml = pytest.importorskip("vtkmodules.vtkIOXML")
        from vtkmodules.util.numpy_support import vtk_to_numpy
        rd = vtk_xml.vtkXMLImageDataReader()
        rd.SetFileName(str(path))
        rd.Update()
        np.testing.assert_array_equal(vtk_to_numpy(rd.GetOutput().GetCellData().GetArray("a")), vals)


# --- volume .vti / .npz ---------------------------------------------------------------------

def volume(mirrored=True) -> VolumeData:
    nz, ny, nx = 5, 4, 6
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    f = {"u": (10.0 + x + 0.1 * y + 0.01 * z).astype(np.float32), "v": (1.0 + y).astype(np.float32),
         "w": (0.5 * z).astype(np.float32), "cp": (-0.1 * y).astype(np.float32),
         "q": (x * y * z).astype(np.float32)}
    return VolumeData(origin=(-1.0, 0.0, 0.0), spacing=(0.5, 0.25, 0.2), fields=f,
                      mask=(x < 2).astype(np.float32), mirrored=mirrored)


def test_mirror_volume():
    vol = volume()
    full = mirror_volume(vol)
    assert full.mirrored is False and full.origin == (-1.0, -1.0, 0.0) and full.spacing == vol.spacing
    assert full.fields["u"].shape == (5, 8, 6)
    np.testing.assert_array_equal(full.fields["u"][:, 4:], vol.fields["u"])
    np.testing.assert_array_equal(full.fields["u"][:, 3], vol.fields["u"][:, 0])        # even across y = 0
    np.testing.assert_array_equal(full.fields["v"][:, 3], -vol.fields["v"][:, 0])       # odd
    np.testing.assert_array_equal(full.fields["v"][:, :4], -vol.fields["v"][:, ::-1])
    np.testing.assert_array_equal(full.mask[:, 0], vol.mask[:, -1])
    assert mirror_volume(volume(mirrored=False)).fields["u"].shape == (5, 4, 6)
    off = volume()
    off.origin = (-1.0, 0.3, 0.0)
    with pytest.raises(ValueError, match="symmetry plane"):
        mirror_volume(off)


@pytest.mark.parametrize("mirror,compress", [(None, True), (False, False)])
def test_volume_vti(tmp_path, mirror, compress):
    vol = volume()
    path = volume_vti(vol, tmp_path / "v.vti", mirror=mirror, compress=compress)
    root = ET.parse(path).getroot()
    img = root.find("ImageData")
    ny = 8 if mirror is None else 4
    assert img.get("WholeExtent") == f"0 6 0 {ny} 0 5"
    assert img.get("Origin") == ("-1.0 -1.0 0.0" if mirror is None else "-1.0 0.0 0.0")
    assert img.get("Spacing") == "0.5 0.25 0.2"
    assert root.find("ImageData/Piece/CellData").get("Vectors") == "velocity"
    arr = appended_arrays(path)
    ref = mirror_volume(vol) if mirror is None else vol
    np.testing.assert_array_equal(arr["u"], ref.fields["u"].ravel())
    np.testing.assert_array_equal(arr["mask"], ref.mask.ravel())
    np.testing.assert_array_equal(arr["velocity"][:, 1], ref.fields["v"].ravel())
    vtk_xml = pytest.importorskip("vtkmodules.vtkIOXML")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    rd = vtk_xml.vtkXMLImageDataReader()
    rd.SetFileName(str(path))
    rd.Update()
    out = rd.GetOutput()
    assert out.GetDimensions() == (7, ny + 1, 6) and out.GetNumberOfCells() == 6 * ny * 5
    got = vtk_to_numpy(out.GetCellData().GetArray("u")).reshape(5, ny, 6)
    np.testing.assert_array_equal(got, ref.fields["u"])
    # the cell at (i, j, k) has its centre where the volume says it is
    ijk, pc = [0, 0, 0], [0.0, 0.0, 0.0]
    assert out.ComputeStructuredCoordinates((-1.0 + 0.5 * 4.5, ref.origin[1] + 0.25 * 2.5, 0.2 * 3.5), ijk, pc)
    assert ijk == [4, 2, 3]
    cid = out.ComputeCellId(ijk)
    assert vtk_to_numpy(out.GetCellData().GetArray("u"))[cid] == ref.fields["u"][3, 2, 4]


def test_volume_vti_and_npz_errors_and_npz_content(tmp_path):
    with pytest.raises(ValueError, match="mirrored"):
        volume_vti(volume(mirrored=False), tmp_path / "a.vti", mirror=True)
    bad = volume()
    bad.fields["u"] = bad.fields["u"][:, :, :3]
    with pytest.raises(ValueError, match="different shapes"):
        volume_vti(bad, tmp_path / "b.vti")
    p = volume_npz(volume(), tmp_path / "v.npz")
    with np.load(p) as z:
        assert z["u"].shape == (5, 8, 4 + 2) and bool(z["mirrored"]) is False
        np.testing.assert_allclose(z["y"], -1.0 + 0.25 * (np.arange(8) + 0.5))
        np.testing.assert_array_equal(z["origin"], [-1.0, -1.0, 0.0])
    with np.load(volume_npz(volume(), tmp_path / "h.npz", mirror=False)) as z:
        assert z["u"].shape == (5, 4, 6) and bool(z["mirrored"]) is True


# --- CSV ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_slice_csv_coordinates(tmp_path, axis):
    nr, nc = 3, 4
    u = np.arange(nr * nc, dtype=float).reshape(nr, nc)
    sd = SliceData(axis, 0.7, (0.0, 2.0, -0.3, 0.3), {"u": u, "cp": -u}, mask=(u > 5).astype(float))
    header, rows = read_csv(slice_csv(sd, tmp_path / f"s{axis}.csv"))
    assert header == ["x", "y", "z", "u", "cp", "mask"] and len(rows) == nr * nc
    tab = np.array(rows, dtype=float)
    col_ax, row_ax = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[axis]
    idx = {"x": 0, "y": 1, "z": 2}
    assert np.all(tab[:, idx[axis]] == 0.7)
    np.testing.assert_allclose(tab[:nc, idx[col_ax]], 0.5 * (np.arange(nc) + 0.5), atol=1e-12)
    np.testing.assert_allclose(tab[::nc, idx[row_ax]], -0.3 + 0.2 * (np.arange(nr) + 0.5), atol=1e-12)
    np.testing.assert_array_equal(tab[:, 3], u.ravel())
    np.testing.assert_array_equal(tab[:, 5], (u > 5).ravel().astype(float))


def test_slice_csv_from_a_result(tmp_path):
    r = cfd3d()
    r.slices = [SliceData("y", 0.0, (0, 1, 0, 1), {"u": np.ones((2, 2))}),
                SliceData("z", 0.4, (0, 1, 0, 1), {"u": np.full((2, 3), 2.0)})]
    _, rows = read_csv(slice_csv(r, tmp_path / "a.csv", axis="z"))
    assert len(rows) == 6 and rows[0][2] == "0.4"
    _, rows = read_csv(slice_csv(r, tmp_path / "b.csv", index=0))
    assert len(rows) == 4
    with pytest.raises(ValueError, match="no slice normal"):
        slice_csv(r, tmp_path / "c.csv", axis="x")
    with pytest.raises(ValueError, match="reserved"):
        slice_csv(SliceData("y", 0, (0, 1, 0, 1), {"x": np.ones((1, 1))}), tmp_path / "d.csv")


def test_wing_csv_and_span_csv(tmp_path):
    w = wing_result()
    header, rows = read_csv(wing_csv(w, tmp_path / "wing.csv"))
    assert header == ["alpha", "cl", "cd", "cdi", "cdv", "cm", "cy", "c_roll", "cn", "e", "ld", "converged"]
    assert len(rows) == 5 and rows[3][-1] == "false" and rows[1][9] == "nan"
    assert float(rows[2][1]) == w.points[2].cl                        # repr: exact round trip
    assert float(rows[2][10]) == pytest.approx(w.points[2].cl / w.points[2].cd)
    header, rows = read_csv(wing_span_csv(w, tmp_path / "span.csv"))
    assert header == ["alpha", "y", "cl_local", "c_cl", "alpha_i", "cd_local", "converged"]
    assert len(rows) == 4 * 4                                         # the alpha without span data is skipped
    assert rows[0][5] == "nan" and rows[4][5] == "0.01"              # inviscid strip drag missing at -2 deg
    assert [r[0] for r in rows[::4]] == ["-2.0", "0.0", "2.0", "4.0"]
    _, rows = read_csv(wing_span_csv(w, tmp_path / "span_c.csv", converged_only=True))
    assert len(rows) == 12
    assert read_csv(history_csv(w, tmp_path / "h.csv")) == read_csv(tmp_path / "wing.csv")


def test_result_and_history_csv_for_3d(tmp_path):
    r = cfd3d()
    header, rows = read_csv(result_csv(r, tmp_path / "r.csv"))
    tab = {row[0]: row for row in rows}
    assert header == ["quantity", "value", "stderr", "unit", "description"]
    for k in ("cd", "cs", "cl", "c_roll", "cm", "cn", "cn_rms", "drag", "side", "pitch_moment", "grid_nz",
              "symmetry_plane", "cells"):
        assert k in tab, k
    assert float(tab["cd"][1]) == 0.3 and float(tab["cd"][2]) == 0.002 and tab["drag"][3] == "N"
    assert tab["pitch_moment"][1] == "44.0" and tab["symmetry_plane"][1] == "true"
    assert ["note", "n1"] == tab["note"][:2] and tab["body:box:cd"][1] == "0.28"
    header, rows = read_csv(history_csv(r, tmp_path / "h.csv"))
    assert header == ["t", "cd", "cs", "cl", "c_roll", "cm", "cn", "extra"]
    assert len(rows) == 101 and float(rows[50][7]) == r.history["extra"][50]
    header, _ = read_csv(history_csv({k: r.history[k] for k in ("t", "cn", "cl", "cd")}, tmp_path / "m.csv"))
    assert header == ["t", "cd", "cl", "cn"]                          # a bare 3D mapping keeps the 3D order

    w = wing_result()
    _, rows = read_csv(result_csv(w, tmp_path / "w.csv"))
    tab = {row[0]: row for row in rows}
    for k in ("method", "area", "aspect_ratio", "re_mac", "cl_alpha_per_rad", "oswald", "e_cruise", "cd0"):
        assert k in tab, k
    assert tab["method"][1] == "vlm" and tab["cl_alpha_per_deg"][3] == "1/deg"
    assert float(tab["cl_alpha_per_deg"][1]) == pytest.approx(0.08)
    assert ["warning", "warn"] == tab["warning"][:2]
    with pytest.raises(TypeError):
        result_csv("nope", tmp_path / "x.csv")  # type: ignore[arg-type]
