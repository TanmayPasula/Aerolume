"""Flow-field exports: NumPy ``.npz`` and VTK ``.vti`` (XML ImageData, for ParaView).

The solver grid is uniform, which is exactly VTK's ImageData: an origin, a spacing
and a block of cell values, no coordinates stored. A 2D field of ``ny x nx`` cells
becomes an image with point extent ``0 nx 0 ny 0 0``; each field is a cell-data
array named after the field, plus a 3-component ``velocity`` vector when ``u`` and
``v`` are present (ParaView's Glyph and Stream Tracer filters want a vector).
Origin and spacing are in metres, wind frame, so the result overlays geometry
exported from Aerolume.

``FieldData`` arrays are (ny, nx) with row 0 at the bottom (ymin), the solver's grid
order; flattening them row-major gives exactly VTK's x-fastest cell order.

Encodings: ``"appended"`` (default; base64 blocks after the XML, compact and still
pure text, so the file is well-formed XML), ``"inline"`` (base64 inside each
``DataArray``) and ``"ascii"`` (human-readable, large). Binary blocks use a UInt64
byte-count header followed by the little-endian data, base64-encoded together,
which is how VTK's own writer lays out uncompressed blocks (``export/vtkxml.py``,
shared with the 3D surface and volume exports).
"""
from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from ..model import FieldData
from ..project.fileio import atomic_writer
from .vtkxml import ENCODINGS, VtkXmlWriter


def _grid(fd: FieldData) -> tuple[int, int, float, float, float, float]:
    shapes = {np.shape(a) for a in fd.fields.values()}
    if fd.mask is not None:
        shapes.add(np.shape(fd.mask))
    if len(shapes) != 1:
        raise ValueError(f"fields have different shapes: {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) != 2:
        raise ValueError(f"fields must be 2D (ny, nx), got shape {shape}")
    ny, nx = shape
    xmin, xmax, ymin, ymax = (float(v) for v in fd.extent)
    return nx, ny, xmin, ymin, (xmax - xmin) / nx, (ymax - ymin) / ny


def cell_centres(fd: FieldData) -> tuple[np.ndarray, np.ndarray]:
    """1D x and y coordinates [m] of the cell centres of a field grid."""
    nx, ny, x0, y0, dx, dy = _grid(fd)
    return x0 + dx * (np.arange(nx) + 0.5), y0 + dy * (np.arange(ny) + 0.5)


def fields_npz(field_data: FieldData, path: str | os.PathLike) -> Path:
    """All fields (float32) plus ``mask``, ``extent``, cell-centre ``x`` / ``y`` and ``label``.

    Load with ``np.load(path)``; arrays are (ny, nx), row 0 at ymin.
    """
    path = Path(path)
    x, y = cell_centres(field_data)
    arrays = {name: np.asarray(a, dtype=np.float32) for name, a in field_data.fields.items()}
    for reserved in ("extent", "x", "y", "label", "mask"):
        if reserved in arrays:
            raise ValueError(f"field name {reserved!r} is reserved in the npz export")
    arrays.update(extent=np.asarray(field_data.extent, dtype=np.float64), x=x, y=y,
                  label=np.array(field_data.label))
    if field_data.mask is not None:
        arrays["mask"] = np.asarray(field_data.mask, dtype=np.float32)
    with atomic_writer(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    return path


def fields_vtk(field_data: FieldData, path: str | os.PathLike, *, encoding: str = "appended",
               velocity: bool = True) -> Path:
    """Write the fields as a VTK XML ImageData file (``.vti``) readable by ParaView.

    Cell data arrays are named after the fields (Float32); the solid fraction is
    added as ``mask`` when present, and ``velocity`` (u, v, 0) when ``velocity`` is
    true and both components exist. See the module docstring for the encodings.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    path = Path(path)
    nx, ny, x0, y0, dx, dy = _grid(field_data)
    arrays: list[tuple[str, int, np.ndarray]] = [(n, 1, np.asarray(a).ravel())
                                                for n, a in field_data.fields.items()]
    if field_data.mask is not None and "mask" not in field_data.fields:
        arrays.append(("mask", 1, np.asarray(field_data.mask).ravel()))
    if velocity and "u" in field_data.fields and "v" in field_data.fields and "velocity" not in field_data.fields:
        u = np.asarray(field_data.fields["u"]).ravel()
        v = np.asarray(field_data.fields["v"]).ravel()
        arrays.append(("velocity", 3, np.stack([u, v, np.zeros_like(u)], axis=1).ravel()))

    scalars = next((n for n in ("cp", "speed", "p", "u") if n in field_data.fields), None)
    attrs = (f' Scalars={quoteattr(scalars)}' if scalars else "") + \
        (' Vectors="velocity"' if any(n == "velocity" for n, _, _ in arrays) else "")
    dz = min(dx, dy)
    w = VtkXmlWriter("ImageData", encoding=encoding)
    w.line(f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 0" Origin="{x0!r} {y0!r} 0" '
           f'Spacing="{dx!r} {dy!r} {dz!r}">')
    w.line(f'    <Piece Extent="0 {nx} 0 {ny} 0 0">')
    w.line(f'      <CellData{attrs}>')
    for name, ncomp, a in arrays:
        w.array(name, a, ncomp=ncomp)
    for t in ("      </CellData>", "    </Piece>", "  </ImageData>"):
        w.line(t)
    return w.write(path)
