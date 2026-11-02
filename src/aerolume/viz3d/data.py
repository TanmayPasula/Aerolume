"""Conversions between Aerolume's 3D data model and the arrays / VTK objects the scene draws.

Pure-numpy helpers (mesh unpacking, volume mirroring, slice extraction, slice
placement) are usable and tested without VTK; the ``*_to_vtk`` functions import
VTK lazily.

Geometry rules taken from :mod:`aerolume.model` (the WHY of each is in the
docstrings there, repeated here only where this module depends on them):

* ``VolumeData`` arrays are (nz, ny, nx), cell-centred; ``origin`` is the min
  corner of the outer cell edges. VTK image data is point-based, so the image
  is placed with its points at the cell centres (origin + spacing / 2): a
  streamline seeded at a world point then samples the cell it is in, exactly
  like the solver's own trilinear sampling.
* ``mirrored`` volumes hold only y >= 0 (symmetry-plane runs, DESIGN.md 7.2).
  For display they are reflected about y = 0, and the lateral velocity ``v``
  changes sign in the reflected half (it is the only y-odd field a volume
  stores; speed, p, cp, q and the vorticity magnitude are even).
* ``SliceData``: for "y" slices rows run along z (row 0 = z-min) and columns
  along x; "z": rows y, columns x; "x": rows z, columns y. ``extent`` is
  (col_min, col_max, row_min, row_max) at outer cell edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SLICE_AXES",
    "MeshArrays",
    "full_volume",
    "image_to_vtk",
    "mesh_arrays",
    "polydata_to_vtk",
    "slice_corners",
    "slice_from_volume",
    "volume_field",
    "volume_points",
]

#: For a slice normal: (column axis index, row axis index) in (x, y, z).
SLICE_AXES: dict[str, tuple[int, int]] = {"y": (0, 2), "z": (0, 1), "x": (1, 2)}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class MeshArrays:
    """Vertices (N, 3) float64, faces (M, 3) int64 and per-vertex fields of one surface."""
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    fields: dict[str, np.ndarray]


def mesh_arrays(mesh) -> MeshArrays:
    """Unpack a ``Mesh3D``, a ``SurfaceData3D``, a (vertices, faces) pair or a dict.

    ``SurfaceData3D`` contributes its per-vertex ``cp``, ``cf``, ``sensitivity``
    and |tau| (as "tau") as named fields, so ``add_mesh(surface, "cp")`` works.
    Faces with an index outside the vertex array are dropped rather than
    crashing the renderer: an imported mesh is untrusted input.
    """
    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        name, verts, faces, fields = "mesh", mesh[0], mesh[1], {}
    elif isinstance(mesh, dict):
        name = str(mesh.get("name") or mesh.get("body") or "mesh")
        verts, faces = mesh["vertices"], mesh["faces"]
        fields = {k: mesh[k] for k in ("cp", "cf", "sensitivity", "tau") if mesh.get(k) is not None}
    else:
        name = str(getattr(mesh, "name", None) or getattr(mesh, "body", None) or "mesh")
        verts, faces = mesh.vertices, mesh.faces
        fields = {k: getattr(mesh, k) for k in ("cp", "cf", "sensitivity", "tau")
                  if getattr(mesh, k, None) is not None}
    v = np.ascontiguousarray(np.asarray(verts, dtype=np.float64).reshape(-1, 3))
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if f.size:
        ok = (f >= 0).all(axis=1) & (f < len(v)).all(axis=1)
        if not ok.all():
            f = f[ok]
    out = {}
    for k, a in fields.items():
        a = np.asarray(a, dtype=np.float64)
        if k == "tau" and a.ndim == 2:
            a = np.linalg.norm(a, axis=1)
        a = a.reshape(-1)
        if len(a) == len(v):
            out[k] = a
    return MeshArrays(name, v, np.ascontiguousarray(f), out)


# ---------------------------------------------------------------------------
# volumes
# ---------------------------------------------------------------------------

_Y_ODD = frozenset({"v"})


def full_volume(volume) -> tuple[tuple[float, float, float], tuple[float, float, float],
                                  dict[str, np.ndarray], np.ndarray | None]:
    """(origin, spacing, fields, mask) of a VolumeData, reflected about y = 0 if mirrored.

    Arrays come back float32 (the store keeps float16; VTK filters and
    percentiles want float32). ``speed`` is added when u, v and w are present
    and it is missing, because it is the default colouring of streamlines.
    """
    origin = tuple(float(c) for c in volume.origin)
    spacing = tuple(float(c) for c in volume.spacing)
    fields = {k: np.asarray(a, dtype=np.float32) for k, a in volume.fields.items()}
    mask = None if volume.mask is None else np.asarray(volume.mask, dtype=np.float32)
    if "speed" not in fields and all(k in fields for k in ("u", "v", "w")):
        fields["speed"] = np.sqrt(fields["u"] ** 2 + fields["v"] ** 2 + fields["w"] ** 2)
    if getattr(volume, "mirrored", False):
        ny = next(iter(fields.values())).shape[1] if fields else (mask.shape[1] if mask is not None else 0)
        y_hi = origin[1] + ny * spacing[1]
        for k, a in fields.items():
            refl = a[:, ::-1, :]
            if k in _Y_ODD:
                refl = -refl
            fields[k] = np.ascontiguousarray(np.concatenate([refl, a], axis=1))
        if mask is not None:
            mask = np.ascontiguousarray(np.concatenate([mask[:, ::-1, :], mask], axis=1))
        origin = (origin[0], -y_hi, origin[2])
    return origin, spacing, fields, mask


def volume_field(fields: dict[str, np.ndarray], name: str) -> np.ndarray:
    """A named field, deriving 'speed' from u, v, w when needed (KeyError if impossible)."""
    if name in fields:
        return fields[name]
    if name == "speed" and all(k in fields for k in ("u", "v", "w")):
        return np.sqrt(fields["u"] ** 2 + fields["v"] ** 2 + fields["w"] ** 2)
    raise KeyError(f"volume has no field {name!r}; available: {', '.join(sorted(fields))}")


def volume_points(origin, spacing, shape_zyx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centre coordinate vectors (x, y, z) of a (nz, ny, nx) block."""
    nz, ny, nx = shape_zyx
    return tuple(origin[i] + (np.arange(n) + 0.5) * spacing[i]   # type: ignore[return-value]
                 for i, n in enumerate((nx, ny, nz)))


def slice_from_volume(volume, axis: str, position: float | None, field: str, *, label: str = "mean"):
    """Cut a :class:`~aerolume.model.SliceData` out of a volume, linearly in the normal direction.

    ``position`` None means the middle of the block along ``axis``. The plane
    is clamped into the block's cell centres (a cut outside the data has no
    values). The result follows the SliceData row/column rules, so slices
    from a stored result and from a volume are drawn by the same code.
    """
    from ..model import SliceData

    axis = axis.lower()
    if axis not in _AXIS_INDEX:
        raise ValueError(f"slice axis must be 'x', 'y' or 'z', not {axis!r}")
    origin, spacing, fields, mask = full_volume(volume)
    arr = volume_field(fields, field)
    nz, ny, nx = arr.shape
    k = _AXIS_INDEX[axis]
    n = (nx, ny, nz)[k]
    c0 = origin[k] + 0.5 * spacing[k]
    if position is None:
        position = origin[k] + 0.5 * n * spacing[k]
    s = (float(position) - c0) / spacing[k]
    s = min(max(s, 0.0), n - 1.0)
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, n - 1)
    t = s - i0
    np_axis = 2 - k                         # (nz, ny, nx) -> axis index in the array

    def cut(a: np.ndarray) -> np.ndarray:
        lo, hi = np.take(a, i0, axis=np_axis), np.take(a, i1, axis=np_axis)
        return (lo * (1.0 - t) + hi * t).astype(np.float32)

    plane = cut(arr)
    m = cut(mask) if mask is not None else None
    # after removing the normal axis: "x" -> (nz, ny) rows z cols y; "y" -> (nz, nx) rows z cols x;
    # "z" -> (ny, nx) rows y cols x: already the SliceData orientation in every case.
    ci, ri = SLICE_AXES[axis]
    ncol, nrow = (nx, ny, nz)[ci], (nx, ny, nz)[ri]
    extent = (origin[ci], origin[ci] + ncol * spacing[ci], origin[ri], origin[ri] + nrow * spacing[ri])
    pos = c0 + s * spacing[k]
    return SliceData(axis=axis, position=float(pos), extent=tuple(float(e) for e in extent),
                     fields={field: plane}, mask=m, label=label)


def slice_corners(axis: str, position: float, extent) -> np.ndarray:
    """World corners (4, 3) of a slice: (col0,row0), (col1,row0), (col1,row1), (col0,row1)."""
    ci, ri = SLICE_AXES[axis]
    k = _AXIS_INDEX[axis]
    c0, c1, r0, r1 = (float(e) for e in extent)
    out = np.zeros((4, 3))
    for j, (c, r) in enumerate(((c0, r0), (c1, r0), (c1, r1), (c0, r1))):
        out[j, ci], out[j, ri], out[j, k] = c, r, position
    return out


# ---------------------------------------------------------------------------
# VTK objects
# ---------------------------------------------------------------------------

def polydata_to_vtk(vertices: np.ndarray, faces: np.ndarray, point_data: dict[str, np.ndarray] | None = None):
    """A triangle ``vtkPolyData`` from numpy arrays (zero-copy where VTK allows)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    f = np.ascontiguousarray(faces, dtype=np.int64)
    offsets = np.arange(0, 3 * len(f) + 1, 3, dtype=np.int64)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(f.reshape(-1).copy(), deep=True))
    pd = vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    for name, a in (point_data or {}).items():
        va = numpy_to_vtk(np.ascontiguousarray(a, dtype=np.float32), deep=True)
        va.SetName(name)
        pd.GetPointData().AddArray(va)
    return pd


def image_to_vtk(origin, spacing, fields: dict[str, np.ndarray], *, vectors=("u", "v", "w")):
    """A ``vtkImageData`` with points at the cell centres and one point array per field.

    When u, v and w are all present they are also packed into a 3-component
    "velocity" array set as the active vectors, which is what
    ``vtkStreamTracer`` integrates.
    """
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData

    first = next(iter(fields.values()))
    nz, ny, nx = first.shape
    img = vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*spacing)
    img.SetOrigin(*(origin[i] + 0.5 * spacing[i] for i in range(3)))
    pdata = img.GetPointData()
    for name, a in fields.items():
        va = numpy_to_vtk(np.ascontiguousarray(a, dtype=np.float32).reshape(-1), deep=True)
        va.SetName(name)
        pdata.AddArray(va)
    if vectors and all(k in fields for k in vectors):
        vel = np.stack([np.asarray(fields[k], np.float32).reshape(-1) for k in vectors], axis=1)
        va = numpy_to_vtk(np.ascontiguousarray(vel), deep=True)
        va.SetName("velocity")
        pdata.AddArray(va)
        pdata.SetActiveVectors("velocity")
    return img
