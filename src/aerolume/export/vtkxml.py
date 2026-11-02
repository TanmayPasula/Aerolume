"""A small writer for VTK's XML file formats (``.vti`` ImageData, ``.vtp`` PolyData).

ParaView, VisIt and VTK itself read these; writing them needs no VTK install,
which keeps exports working on machines (and in tests) without the 3D viewer.

Layout of binary data, as VTK's own ``vtkXMLWriter`` produces it with
``header_type="UInt64"``:

* **uncompressed**: each array is one block, a UInt64 byte count followed by the
  little-endian values, base64-encoded together;
* **compressed** (``compressor="vtkZLibDataCompressor"``): the values are cut
  into 32 KiB blocks, each deflated separately; the header is
  ``[n_blocks, block_size, last_block_size, compressed_size_1, ...]`` (UInt64,
  ``last_block_size`` 0 when the last block is full) and is base64-encoded
  *separately* from the concatenated compressed blocks. Volumes of a 3D result
  are tens of megabytes, which this typically shrinks by a third to a half.

Encodings: ``"appended"`` (default; base64 blocks after the XML in one
``<AppendedData>`` section: compact, still pure text and well-formed XML),
``"inline"`` (base64 inside each ``DataArray``) and ``"ascii"`` (readable,
large; never compressed).
"""
from __future__ import annotations

import base64
import os
import zlib
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from ..project.fileio import atomic_writer

ENCODINGS = ("appended", "inline", "ascii")
#: uncompressed size of one zlib block, as VTK's writer uses
BLOCK_SIZE = 32768
#: numpy little-endian dtype per VTK type name
VTK_TYPES = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8", "UInt8": "u1"}


def _raw_block(raw: bytes) -> bytes:
    return base64.b64encode(np.uint64(len(raw)).astype("<u8").tobytes() + raw)


def _zlib_block(raw: bytes, level: int) -> bytes:
    n = len(raw)
    nblocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    last = n % BLOCK_SIZE
    comp = [zlib.compress(raw[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE], level) for i in range(nblocks)]
    header = np.array([nblocks, BLOCK_SIZE, last, *[len(c) for c in comp]], dtype="<u8").tobytes()
    return base64.b64encode(header) + base64.b64encode(b"".join(comp))


def encode(values: np.ndarray, vtk_type: str, *, compress: bool = False, level: int = 1) -> bytes:
    """One array as a base64 binary block (see the module docstring for the layout)."""
    raw = np.ascontiguousarray(values, dtype=VTK_TYPES[vtk_type]).tobytes()
    return _zlib_block(raw, level) if compress else _raw_block(raw)


class VtkXmlWriter:
    """Accumulates the XML of one VTK file and its binary blocks, then writes it atomically.

    Usage: ``w = VtkXmlWriter("ImageData", encoding=..., compress=...)``, then
    ``w.line(...)`` for structural tags and ``w.array(...)`` for each DataArray, and
    finally ``w.write(path)``.
    """

    def __init__(self, vtk_type: str, *, encoding: str = "appended", compress: bool = False,
                 level: int = 1) -> None:
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        self.vtk_type = vtk_type
        self.encoding = encoding
        self.compress = bool(compress) and encoding != "ascii"
        self.level = int(level)
        comp = ' compressor="vtkZLibDataCompressor"' if self.compress else ""
        self._lines = ['<?xml version="1.0"?>',
                       (f'<VTKFile type="{vtk_type}" version="1.0" byte_order="LittleEndian" '
                        f'header_type="UInt64"{comp}>')]
        self._appended: list[bytes] = []
        self._offset = 0

    def line(self, text: str) -> None:
        """Append one line of XML (the caller indents)."""
        self._lines.append(text)

    def array(self, name: str, values: np.ndarray, *, vtk_type: str = "Float32", ncomp: int = 1,
              indent: str = "        ") -> None:
        """Append a ``DataArray`` of ``values`` (flattened in C order)."""
        values = np.asarray(values).ravel()
        tag = (f'{indent}<DataArray type="{vtk_type}" Name={quoteattr(name)} '
               f'NumberOfComponents="{ncomp}"')
        if self.encoding == "appended":
            blk = encode(values, vtk_type, compress=self.compress, level=self.level)
            self._lines.append(f'{tag} format="appended" offset="{self._offset}"/>')
            self._appended.append(blk)
            self._offset += len(blk)
        elif self.encoding == "inline":
            self._lines.append(f'{tag} format="binary">')
            self._lines.append(indent + "  " + encode(values, vtk_type, compress=self.compress,
                                                      level=self.level).decode("ascii"))
            self._lines.append(f"{indent}</DataArray>")
        else:
            self._lines.append(f'{tag} format="ascii">')
            vals = np.asarray(values, dtype=VTK_TYPES[vtk_type])
            if vtk_type.startswith("Float"):
                fmt = "{:.9g}" if vtk_type == "Float32" else "{!r}"   # 9 digits: exact f32
                vals_l = [fmt.format(float(x)) for x in vals.tolist()]
            else:
                vals_l = [str(int(x)) for x in vals.tolist()]
            for i in range(0, len(vals_l), 12):
                self._lines.append(indent + "  " + " ".join(vals_l[i:i + 12]))
            self._lines.append(f"{indent}</DataArray>")

    def write(self, path: str | os.PathLike) -> Path:
        """Write the file atomically and return its path."""
        path = Path(path)
        with atomic_writer(path, "wb") as fh:
            fh.write(("\n".join(self._lines) + "\n").encode("utf-8"))
            if self.encoding == "appended":
                fh.write(b'  <AppendedData encoding="base64">\n   _')
                for blk in self._appended:
                    fh.write(blk)
                fh.write(b"\n  </AppendedData>\n")
            fh.write(b"</VTKFile>\n")
        return path


__all__ = ["BLOCK_SIZE", "ENCODINGS", "VTK_TYPES", "VtkXmlWriter", "encode"]
