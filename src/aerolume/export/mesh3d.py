"""3D geometry and surface exports: STL, OBJ (geometry) and VTK PolyData ``.vtp`` (surface results).

Geometry is written in the frame it is held in: a :class:`Design`'s meshes in the
design frame (x downstream, y left, z up, metres; DESIGN.md 7.2), so an exported
mesh re-imports exactly where it came from; a result's
:class:`SurfaceData3D` in the wind frame it was solved in (rotated by alpha /
yaw, mirrored to the full body when a symmetry plane was used), so it overlays
the exported slices and volume in ParaView.

* **STL** (binary by default, ASCII on request) holds triangles only, so every
  mesh of the selection is merged; the ASCII form keeps one ``solid`` per mesh
  (most tools read multi-solid files). Facet normals follow the vertex order
  (right-hand rule) and are zero for degenerate triangles; STL has no units, so
  ``units`` ("m" or "mm") scales the coordinates and is stated in the header.
* **OBJ** keeps each mesh as an ``o``/``g`` group named after it, which
  ``geometry3d.read_mesh`` turns back into separate meshes.
* **VTP** (VTK XML PolyData) is the format ParaView expects for surface data:
  point data ``cp``, ``cf``, ``tau`` (3 components, the wall-shear direction
  scaled by cf) and ``sensitivity`` (dJ/dn), plus cell data ``body_index``
  saying which mesh a triangle belongs to (names in an XML comment). Arrays a
  surface does not have (cf of an inviscid body, sensitivity of a run without it)
  are filled with NaN when another surface has them, and left out when none has.

These writers do not need ``geometry3d``; a ``kind == "wing"`` design without
meshes is lofted with ``geometry3d.loft`` when that module is available.
"""
from __future__ import annotations

import io
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..model import CFD3DResult, Design, Mesh3D, SurfaceData3D
from ..project.fileio import atomic_writer
from .vtkxml import VtkXmlWriter

_UNITS = {"m": 1.0, "mm": 1e3}

MeshLike = Design | Mesh3D | SurfaceData3D | CFD3DResult | Sequence[Mesh3D | SurfaceData3D]


def _named_meshes(obj: MeshLike) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, vertices float64 (N,3), faces int64 (M,3)) for every mesh of ``obj``."""
    if isinstance(obj, Design):
        meshes: list = list(obj.meshes)
        if not meshes and obj.wing is not None:
            try:
                from ..geometry3d import loft  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ValueError("this wing design has no mesh yet and wing lofting is unavailable") from exc
            meshes = [loft(obj.wing)]
        if not meshes:
            raise ValueError(f"design {obj.name!r} has no 3D meshes")
    elif isinstance(obj, CFD3DResult):
        meshes = list(obj.surface)
        if not meshes:
            raise ValueError("the result has no surface data")
    elif isinstance(obj, (Mesh3D, SurfaceData3D)):
        meshes = [obj]
    else:
        meshes = list(obj)
    out = []
    for i, m in enumerate(meshes):
        if not isinstance(m, (Mesh3D, SurfaceData3D)):
            raise TypeError(f"not a mesh: {type(m).__name__}")
        name = (m.name if isinstance(m, Mesh3D) else m.body) or f"mesh{i + 1}"
        v = np.asarray(m.vertices, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(m.faces, dtype=np.int64).reshape(-1, 3)
        if f.size and (f.min() < 0 or f.max() >= len(v)):
            raise ValueError(f"mesh {name!r} has faces indexing missing vertices")
        out.append((name, v, f))
    return out


def _facet_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)


def _scale(units: str) -> float:
    if units not in _UNITS:
        raise ValueError(f"units must be one of {tuple(_UNITS)}")
    return _UNITS[units]


def _ascii_name(name: str) -> str:
    return "".join(c if c.isascii() and c.isprintable() and not c.isspace() else "_" for c in name) or "mesh"


def mesh_stl(meshes: MeshLike, path: str | os.PathLike, *, binary: bool = True, units: str = "m") -> Path:
    """Write the meshes as one STL file (binary unless ``binary=False``).

    ``meshes``: a 3D :class:`Design` (its meshes), a :class:`CFD3DResult` (its
    solved surfaces), one :class:`Mesh3D` / :class:`SurfaceData3D` or a sequence
    of them. ``units`` "m" (default, lossless round trip within float32) or "mm".
    """
    k = _scale(units)
    parts = _named_meshes(meshes)
    path = Path(path)
    if binary:
        n_tri = sum(len(f) for _, _, f in parts)
        rec = np.zeros(n_tri, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        i = 0
        for _, v, f in parts:
            rec["n"][i:i + len(f)] = _facet_normals(v, f)
            rec["v"][i:i + len(f)] = v[f] * k
            i += len(f)
        header = f"Aerolume STL, units {units}, {len(parts)} mesh(es)".encode("ascii")[:80].ljust(80, b" ")
        with atomic_writer(path, "wb") as fh:
            fh.write(header)
            fh.write(np.uint32(n_tri).astype("<u4").tobytes())
            fh.write(rec.tobytes())
        return path
    buf = io.StringIO()
    for name, v, f in parts:
        nm = _ascii_name(name)
        buf.write(f"solid {nm}\n")
        nrm = _facet_normals(v, f)
        tri = v[f] * k
        for t in range(len(f)):
            buf.write("  facet normal {:.7e} {:.7e} {:.7e}\n    outer loop\n".format(*nrm[t].tolist()))
            for p in tri[t]:
                buf.write("      vertex {!r} {!r} {!r}\n".format(*p.tolist()))
            buf.write("    endloop\n  endfacet\n")
        buf.write(f"endsolid {nm}\n")
    with atomic_writer(path, "w") as fh:
        fh.write(buf.getvalue())
    return path


def mesh_obj(meshes: MeshLike, path: str | os.PathLike, *, units: str = "m") -> Path:
    """Write the meshes as a Wavefront OBJ, one ``o`` / ``g`` group per mesh.

    Coordinates are written with 9 significant digits (``%.9g``: below a
    micrometre on a 10 m body). Accepts the same inputs as :func:`mesh_stl`.
    """
    k = _scale(units)
    parts = _named_meshes(meshes)
    buf = io.StringIO()
    buf.write(f"# Aerolume OBJ export, units {units}, {len(parts)} mesh(es)\n")
    base = 1
    for name, v, f in parts:
        nm = _ascii_name(name)
        buf.write(f"o {nm}\ng {nm}\n")
        if len(v):
            np.savetxt(buf, v * k, fmt="v %.9g %.9g %.9g")
        if len(f):
            np.savetxt(buf, f + base, fmt="f %d %d %d")
        base += len(v)
    with atomic_writer(path, "w") as fh:
        fh.write(buf.getvalue())
    return Path(path)


_POINT_ARRAYS = (("cp", 1), ("cf", 1), ("tau", 3), ("sensitivity", 1))


def surface_vtp(result: CFD3DResult | SurfaceData3D | Sequence[SurfaceData3D], path: str | os.PathLike, *,
                encoding: str = "appended", compress: bool = True) -> Path:
    """Write 3D surface results as one VTK XML PolyData file (``.vtp``) for ParaView.

    All surfaces are merged into one piece (points Float32 in metres, triangles
    as Int64 connectivity / offsets); point data ``cp``, ``cf``, ``tau`` (vector)
    and ``sensitivity`` as described in the module docstring; cell data
    ``body_index``. ``compress`` (default) deflates the binary blocks (VTK's zlib
    compressor); ``encoding`` as in :func:`aerolume.export.fields_vtk`.
    """
    if isinstance(result, CFD3DResult):
        surfaces = list(result.surface)
    elif isinstance(result, SurfaceData3D):
        surfaces = [result]
    else:
        surfaces = list(result)
    if not surfaces:
        raise ValueError("no surface data to export")
    parts = _named_meshes(surfaces)
    n_pts = sum(len(v) for _, v, _ in parts)
    n_tri = sum(len(f) for _, _, f in parts)
    pts = np.concatenate([v for _, v, _ in parts]) if n_pts else np.zeros((0, 3))
    offs = np.cumsum([0] + [len(v) for _, v, _ in parts])
    conn = np.concatenate([f + offs[i] for i, (_, _, f) in enumerate(parts)]) if n_tri else np.zeros((0, 3), int)
    body = np.concatenate([np.full(len(f), i, dtype=np.int32) for i, (_, _, f) in enumerate(parts)])

    point_data = []
    for name, ncomp in _POINT_ARRAYS:
        if all(getattr(s, name) is None for s in surfaces):
            continue
        cols = []
        for s, (_, v, _) in zip(surfaces, parts, strict=True):
            a = getattr(s, name)
            shape = (len(v), ncomp) if ncomp > 1 else (len(v),)
            if a is None:
                cols.append(np.full(shape, np.nan, dtype=np.float32))
            else:
                a = np.asarray(a, dtype=np.float32)
                if a.size != int(np.prod(shape)):
                    raise ValueError(f"surface {s.body!r}: {name} has {a.size} values for {len(v)} vertices")
                cols.append(a.reshape(shape))
        point_data.append((name, ncomp, np.concatenate(cols)))

    w = VtkXmlWriter("PolyData", encoding=encoding, compress=compress)
    names = ", ".join(f"{i} = {nm}" for i, (nm, _, _) in enumerate(parts)).replace("--", "- -")
    w.line(f"  <!-- body_index: {names} -->")
    w.line("  <PolyData>")
    w.line(f'    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="0" '
           f'NumberOfStrips="0" NumberOfPolys="{n_tri}">')
    present = {n for n, _, _ in point_data}
    attrs = (' Scalars="cp"' if "cp" in present else "") + (' Vectors="tau"' if "tau" in present else "")
    w.line(f"      <PointData{attrs}>")
    for name, ncomp, a in point_data:
        w.array(name, a, ncomp=ncomp)
    w.line("      </PointData>")
    w.line('      <CellData Scalars="body_index">')
    w.array("body_index", body, vtk_type="Int32")
    w.line("      </CellData>")
    w.line("      <Points>")
    w.array("Points", pts, ncomp=3)
    w.line("      </Points>")
    w.line("      <Polys>")
    w.array("connectivity", conn, vtk_type="Int64")
    w.array("offsets", 3 * np.arange(1, n_tri + 1, dtype=np.int64), vtk_type="Int64")
    w.line("      </Polys>")
    w.line("    </Piece>")
    w.line("  </PolyData>")
    return w.write(path)


__all__ = ["mesh_obj", "mesh_stl", "surface_vtp"]
