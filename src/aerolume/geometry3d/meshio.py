"""Triangle-mesh files: STL (ASCII and binary), Wavefront OBJ and PLY, plus a units guess.

Imported 3D shapes arrive in whatever state the modelling tool left them: STL
stores every triangle with its own three corners (no shared vertices), OBJ keeps
one vertex pool for the whole file and splits it into groups and objects, PLY can
hold quads and extra per-vertex data. The readers turn all of that into a list of
:class:`~aerolume.model.Mesh3D` (one per STL solid, OBJ group/object, or PLY file)
with shared, exactly welded vertices, so the topology checks (watertight, holes,
winding) and the voxeliser see a connected surface:

* **STL**: binary or ASCII is decided from the file size (a binary file is exactly
  ``84 + 50 n`` bytes), not from the leading ``solid`` keyword, which many binary
  exporters also write. ASCII files may hold several ``solid ... endsolid`` blocks
  (one mesh each) and a missing ``endsolid`` is tolerated.
* **OBJ**: ``o`` and ``g`` start a new mesh; faces of any size are fan-triangulated
  (the convention every OBJ reader follows); negative (relative) indices and
  ``v/vt/vn`` index triples are understood; materials, texture coordinates and
  normals are ignored (normals are recomputed from the geometry where needed).
* **PLY**: ``ascii``, ``binary_little_endian`` and ``binary_big_endian``; any element
  layout (extra vertex properties, colours, extra elements are skipped correctly);
  polygons are fan-triangulated.

Coordinates are returned as stored (file units) unless ``units`` is given. Meshes
carry no unit information, which is why :func:`guess_units` exists: the GUI shows
its guess and lets the user confirm before anything is scaled.

Readers append what they repaired or assumed to an optional ``notes`` list (the
2D readers' convention), which the import dialog shows.
"""
from __future__ import annotations

import io
import math
import re
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..model import Mesh3D
from ._rows import unique_rows

__all__ = [
    "SUPPORTED_EXTENSIONS_3D",
    "UNIT_SCALES",
    "guess_units",
    "read_mesh",
    "read_obj",
    "read_ply",
    "read_stl",
    "unit_scale",
    "write_obj",
    "write_ply",
    "write_stl",
]

SUPPORTED_EXTENSIONS_3D = {
    ".stl": "STL mesh (ASCII or binary)",
    ".obj": "Wavefront OBJ mesh",
    ".ply": "PLY mesh (ASCII or binary)",
}

#: metres per file unit for the unit names the GUI offers
UNIT_SCALES = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _note(notes: list[str] | None, text: str) -> None:
    if notes is not None and text not in notes:
        notes.append(text)


def _weld_exact(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices with bit-identical coordinates (lossless); drop unused ones.

    Coordinates are compared bit for bit after ``+ 0.0`` (so -0.0 and 0.0, the
    two zeros of a symmetric model, are the same vertex). Vertices keep the order
    of their first appearance in the file.
    """
    if len(verts) == 0:
        return verts.reshape(0, 3), faces.reshape(0, 3)
    v = np.ascontiguousarray(verts, dtype=np.float64)
    _, first, inv, _ = unique_rows(v)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return v[first[order]] + 0.0, rank[inv][faces]


def _drop_degenerate(faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove triangles that repeat a vertex index (zero area by topology)."""
    bad = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    return faces[~bad], int(bad.sum())


def _compact(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the vertices the faces use (OBJ groups share one pool)."""
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    used, inv = np.unique(faces.reshape(-1), return_inverse=True)
    return verts[used], inv.reshape(-1, 3)


def _fan(poly: Sequence[int]) -> list[tuple[int, int, int]]:
    """Fan triangulation of one polygon (the OBJ/PLY reader convention)."""
    return [(poly[0], poly[i], poly[i + 1]) for i in range(1, len(poly) - 1)]


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = seen.get(n, 0)
        seen[n] = k + 1
        out.append(n if k == 0 else f"{n} ({k + 1})")
    return out


def _finish(name: str, verts: np.ndarray, faces: np.ndarray, notes: list[str] | None,
            weld: bool = True) -> Mesh3D | None:
    """Weld, drop degenerate faces, report; None when nothing is left."""
    if weld:
        verts, faces = _weld_exact(verts, faces)
    faces, nbad = _drop_degenerate(np.asarray(faces, dtype=np.int64).reshape(-1, 3))
    if nbad:
        _note(notes, f"'{name}': removed {nbad} degenerate triangle(s) with repeated corners.")
    if not np.all(np.isfinite(verts)):
        bad_v = ~np.all(np.isfinite(verts), axis=1)
        keep = ~np.any(bad_v[faces], axis=1)
        _note(notes, f"'{name}': removed {int((~keep).sum())} triangle(s) with non-numeric coordinates.")
        faces = faces[keep]
    verts, faces = _compact(verts, faces)
    if len(faces) == 0:
        _note(notes, f"'{name}': no triangles left; skipped.")
        return None
    return Mesh3D(name=name, vertices=verts, faces=faces)


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------

_STL_REC = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
_SOLID_RE = re.compile(rb"^[ \t]*solid\b[ \t]*([^\r\n]*)", re.MULTILINE | re.IGNORECASE)
_VERTEX_RE = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)


def _stl_is_binary(data: bytes) -> bool:
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            return True
    head = data[:1024].lstrip()
    if head[:5].lower() == b"solid" and _VERTEX_RE.search(data[:65536]) is not None:
        return False
    return len(data) >= 84


def read_stl(path, *, notes: list[str] | None = None, name: str | None = None) -> list[Mesh3D]:
    """Read an ASCII or binary STL file; one :class:`Mesh3D` per ASCII ``solid``.

    Vertices are welded where their coordinates are bit-identical (STL repeats
    every corner), which restores the connectivity without moving anything.
    """
    p = Path(path)
    data = p.read_bytes()
    stem = name or p.stem
    if _stl_is_binary(data):
        n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
        avail = (len(data) - 84) // 50
        if n != avail:
            _note(notes, f"STL header announces {n} triangles but the file holds {avail}; "
                         f"read {min(n, avail)}.")
            n = min(n, avail)
        rec = np.frombuffer(data, dtype=_STL_REC, count=n, offset=84)
        verts = rec["v"].reshape(-1, 3).astype(np.float64)
        faces = np.arange(3 * n, dtype=np.int64).reshape(-1, 3)
        # the 80-byte header is free text (exporter names, colour tags, junk), so the
        # mesh is named after the file
        m = _finish(stem, verts, faces, notes)
        if m is None:
            raise ValueError(f"no triangles in {p.name}")
        return [m]
    starts = [(mt.start(), mt.group(1).decode("latin-1").strip()) for mt in _SOLID_RE.finditer(data)]
    if not starts:
        starts = [(0, "")]
    out: list[Mesh3D] = []
    labels = []
    for k, (s0, label) in enumerate(starts):
        s1 = starts[k + 1][0] if k + 1 < len(starts) else len(data)
        found = _VERTEX_RE.findall(data, s0, s1)
        if not found:
            continue
        try:
            v = np.array(found, dtype=np.float64)
        except ValueError:
            # a stray non-numeric token: parse one by one and drop the facet it is in
            v = np.array([[_float_or_nan(t) for t in row] for row in found], dtype=np.float64)
        if len(v) % 3:
            _note(notes, f"Solid '{label or stem}' ends inside a facet; the incomplete facet was dropped.")
            v = v[: len(v) - len(v) % 3]
        labels.append(label or stem)
        m = _finish(label or stem, v, np.arange(len(v), dtype=np.int64).reshape(-1, 3), notes)
        if m is not None:
            out.append(m)
    if not out:
        raise ValueError(f"no triangles in {p.name}")
    for m, nm in zip(out, _unique_names([m.name for m in out])):
        m.name = nm
    if len(out) > 1:
        _note(notes, f"{p.name}: {len(out)} solids, read as separate meshes.")
    return out


def _float_or_nan(t: bytes) -> float:
    try:
        return float(t)
    except ValueError:
        return math.nan


def write_stl(meshes: Mesh3D | Sequence[Mesh3D], path, binary: bool = True) -> Path:
    """Write meshes as STL. Binary (float32, one solid) or ASCII (one solid per mesh).

    Binary STL has no notion of several solids, so all meshes are written into
    one; the ASCII form keeps each mesh as a named ``solid`` so :func:`read_stl`
    gets the same meshes back. Facet normals follow the stored winding.
    """
    meshes = [meshes] if isinstance(meshes, Mesh3D) else list(meshes)
    path = Path(path)
    if binary:
        tris = [m.vertices[m.faces] for m in meshes if len(m.faces)]
        t = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        rec = np.zeros(len(t), dtype=_STL_REC)
        rec["v"] = t.astype(np.float32)
        rec["n"] = _unit_normals(t).astype(np.float32)
        name = (meshes[0].name if len(meshes) == 1 else "Aerolume mesh").encode("ascii", "replace")[:70]
        header = (b"Aerolume STL: " + name).ljust(80, b" ")[:80]
        if header.lower().startswith(b"solid"):
            header = b"_" + header[1:]
        with open(path, "wb") as f:
            f.write(header)
            f.write(struct.pack("<I", len(t)))
            f.write(rec.tobytes())
        return path
    buf = io.StringIO()
    for m in meshes:
        nm = re.sub(r"\s+", "_", m.name.strip()) or "mesh"
        t = m.vertices[m.faces]
        nrm = _unit_normals(t)
        buf.write(f"solid {nm}\n")
        # repr of a Python float is the shortest string that reads back bit-identically
        for tri, n in zip(t.tolist(), nrm.tolist()):
            buf.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
            for q in tri:
                buf.write(f"      vertex {q[0]!r} {q[1]!r} {q[2]!r}\n")
            buf.write("    endloop\n  endfacet\n")
        buf.write(f"endsolid {nm}\n")
    path.write_text(buf.getvalue(), encoding="ascii", errors="replace")
    return path


def _unit_normals(t: np.ndarray) -> np.ndarray:
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------

def read_obj(path, *, notes: list[str] | None = None, name: str | None = None) -> list[Mesh3D]:
    """Read a Wavefront OBJ file; every distinct ``o`` / ``g`` name becomes a mesh.

    Faces met again under a name already seen (``g body`` ... ``g wheel`` ...
    ``g body``) join that mesh. Polygons are fan-triangulated; indices may be
    negative (relative to the vertices read so far) and may carry ``/vt/vn``.
    """
    p = Path(path)
    stem = name or p.stem
    text = p.read_bytes().decode("utf-8", errors="replace").replace("\\\r\n", " ").replace("\\\n", " ")
    vlines: list[str] = []
    groups: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    order: list[tuple[str, str]] = []
    obj_name, grp_name = "", ""
    bad_faces = 0
    n_poly = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        tag, _, rest = line.partition(" ")
        if tag == "v":
            vlines.append(rest)
        elif tag == "f":
            nv = len(vlines)
            idx = []
            ok = True
            for tok in rest.split():
                s = tok.split("/", 1)[0]
                try:
                    k = int(s)
                except ValueError:
                    ok = False
                    break
                if k > 0:
                    k -= 1
                elif k < 0:
                    k += nv
                else:
                    ok = False
                    break
                if not 0 <= k < nv:
                    ok = False
                    break
                idx.append(k)
            if not ok or len(idx) < 3:
                bad_faces += 1
                continue
            if len(idx) > 3:
                n_poly += 1
            key = (obj_name, grp_name)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].extend(_fan(idx))
        elif tag == "o":
            obj_name = rest.strip()
            grp_name = ""
        elif tag == "g":
            grp_name = " ".join(rest.split())
    if bad_faces:
        _note(notes, f"{p.name}: skipped {bad_faces} face(s) with invalid or missing vertex indices.")
    if n_poly:
        _note(notes, f"{p.name}: {n_poly} polygon(s) with more than 3 corners were split into triangles.")
    if not vlines or not groups:
        raise ValueError(f"no faces in {p.name}")
    try:
        verts = np.array([r.split()[:3] for r in vlines], dtype=np.float64)
    except ValueError:
        verts = np.array([[_float_or_nan(t.encode()) for t in (r.split() + ["nan"] * 3)[:3]] for r in vlines])
    out: list[Mesh3D] = []
    names = _unique_names([(g or o or stem) for o, g in order])
    for key, nm in zip(order, names):
        f = np.asarray(groups[key], dtype=np.int64).reshape(-1, 3)
        v, f = _compact(verts, f)
        m = _finish(nm, v, f, notes, weld=False)
        if m is not None:
            out.append(m)
    if not out:
        raise ValueError(f"no triangles in {p.name}")
    if len(out) > 1:
        _note(notes, f"{p.name}: {len(out)} groups/objects, read as separate meshes "
                     f"({', '.join(m.name for m in out[:6])}{', ...' if len(out) > 6 else ''}).")
    return out


def write_obj(meshes: Mesh3D | Sequence[Mesh3D], path) -> Path:
    """Write meshes to OBJ, one ``o``/``g`` block per mesh (full float64 precision)."""
    meshes = [meshes] if isinstance(meshes, Mesh3D) else list(meshes)
    path = Path(path)
    buf = io.StringIO()
    buf.write("# Aerolume OBJ export (metres)\n")
    off = 1
    for m in meshes:
        nm = " ".join(m.name.split()) or "mesh"
        buf.write(f"o {nm}\ng {nm}\n")
        np.savetxt(buf, m.vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(buf, m.faces + off, fmt="f %d %d %d")
        off += len(m.vertices)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _ply_header(data: bytes) -> tuple[str, list[tuple[str, int, list[tuple]]], int]:
    """(format, elements, body offset). Element props: ("name", type) or ("name", "list", ct, it)."""
    if not data[:3] == b"ply":
        raise ValueError("not a PLY file (missing 'ply' magic)")
    end = data.find(b"end_header")
    if end < 0:
        raise ValueError("PLY header has no end_header")
    nl = data.find(b"\n", end)
    body = nl + 1 if nl >= 0 else len(data)
    fmt = ""
    elements: list[tuple[str, int, list[tuple]]] = []
    for raw in data[:end].decode("ascii", errors="replace").splitlines():
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok[0] == "property" and elements:
            if tok[1] == "list":
                elements[-1][2].append((tok[4], "list", _PLY_TYPES[tok[2]], _PLY_TYPES[tok[3]]))
            else:
                elements[-1][2].append((tok[2], _PLY_TYPES[tok[1]]))
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY format {fmt!r}")
    return fmt, elements, body


def _ply_ascii(text: str, elements) -> dict[str, dict[str, object]]:
    """Parse the ASCII body into {element: {prop: array or list of lists}}."""
    tok = np.array(text.split(), dtype=np.float64) if text.strip() else np.zeros(0)
    pos = 0
    out: dict[str, dict[str, object]] = {}
    for ename, count, props in elements:
        has_list = any(p[1] == "list" for p in props)
        if not has_list:
            w = len(props)
            block = tok[pos:pos + count * w]
            if len(block) < count * w:
                raise ValueError(f"PLY element '{ename}' is truncated")
            block = block.reshape(count, w)
            out[ename] = {p[0]: block[:, i] for i, p in enumerate(props)}
            pos += count * w
            continue
        # fixed-width fast path: every list has the first item's length
        width, layout = 0, []
        probe = pos
        for p in props:
            if p[1] == "list":
                c = int(tok[probe]) if probe < len(tok) else 0
                layout.append(("list", width, c))
                width += 1 + c
                probe += 1 + c
            else:
                layout.append(("scalar", width, 1))
                width += 1
                probe += 1
        block = tok[pos:pos + count * width]
        if len(block) == count * width and count:
            block = block.reshape(count, width)
            if all(np.all(block[:, off] == c) for kind, off, c in layout if kind == "list"):
                d: dict[str, object] = {}
                for p, (kind, off, c) in zip(props, layout):
                    d[p[0]] = block[:, off + 1:off + 1 + c].astype(np.int64) if kind == "list" else block[:, off]
                out[ename] = d
                pos += count * width
                continue
        # general path: walk item by item
        cols: dict[str, list] = {p[0]: [] for p in props}
        for _ in range(count):
            for p in props:
                if p[1] == "list":
                    c = int(tok[pos])
                    cols[p[0]].append(tok[pos + 1:pos + 1 + c].astype(np.int64))
                    pos += 1 + c
                else:
                    cols[p[0]].append(tok[pos])
                    pos += 1
        out[ename] = {k: (np.array(v) if v and np.ndim(v[0]) == 0 else v) for k, v in cols.items()}
    return out


def _ply_binary(data: bytes, offset: int, elements, endian: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    pos = offset
    for ename, count, props in elements:
        has_list = any(p[1] == "list" for p in props)
        if not has_list:
            dt = np.dtype([(p[0], endian + p[1]) for p in props])
            arr = np.frombuffer(data, dtype=dt, count=count, offset=pos)
            out[ename] = {p[0]: arr[p[0]] for p in props}
            pos += count * dt.itemsize
            continue
        # fast path: assume every list has the first item's length, then verify
        fields, probe = [], pos
        for i, p in enumerate(props):
            if p[1] == "list":
                ct = np.dtype(endian + p[2])
                c = int(np.frombuffer(data, dtype=ct, count=1, offset=probe)[0]) if count else 0
                fields += [(f"_n{i}", endian + p[2]), (p[0], endian + p[3], (c,))]
                probe += ct.itemsize + c * np.dtype(p[3]).itemsize
            else:
                fields.append((p[0], endian + p[1]))
                probe += np.dtype(p[1]).itemsize
        dt = np.dtype(fields)
        if count and pos + count * dt.itemsize <= len(data):
            arr = np.frombuffer(data, dtype=dt, count=count, offset=pos)
            lens_ok = all(np.all(arr[f"_n{i}"] == arr[p[0]].shape[1] if arr[p[0]].ndim == 2 else True)
                          for i, p in enumerate(props) if p[1] == "list")
            if lens_ok:
                out[ename] = {p[0]: (arr[p[0]].astype(np.int64) if p[1] == "list" else arr[p[0]])
                              for p in props}
                pos += count * dt.itemsize
                continue
        cols: dict[str, list] = {p[0]: [] for p in props}
        for _ in range(count):
            for p in props:
                if p[1] == "list":
                    ct, it = np.dtype(endian + p[2]), np.dtype(endian + p[3])
                    c = int(np.frombuffer(data, dtype=ct, count=1, offset=pos)[0])
                    pos += ct.itemsize
                    cols[p[0]].append(np.frombuffer(data, dtype=it, count=c, offset=pos).astype(np.int64))
                    pos += c * it.itemsize
                else:
                    t = np.dtype(endian + p[1])
                    cols[p[0]].append(np.frombuffer(data, dtype=t, count=1, offset=pos)[0])
                    pos += t.itemsize
        out[ename] = {k: (np.array(v) if v and np.ndim(v[0]) == 0 else v) for k, v in cols.items()}
    return out


def read_ply(path, *, notes: list[str] | None = None, name: str | None = None) -> list[Mesh3D]:
    """Read a PLY mesh (ASCII, binary little- or big-endian) as one :class:`Mesh3D`."""
    p = Path(path)
    data = p.read_bytes()
    fmt, elements, body = _ply_header(data)
    if fmt == "ascii":
        el = _ply_ascii(data[body:].decode("ascii", errors="replace"), elements)
    else:
        el = _ply_binary(data, body, elements, "<" if fmt == "binary_little_endian" else ">")
    vx = el.get("vertex")
    if not vx or not all(k in vx for k in ("x", "y", "z")):
        raise ValueError(f"{p.name}: no vertex x, y, z properties")
    verts = np.column_stack([np.asarray(vx[k], dtype=np.float64) for k in ("x", "y", "z")])
    fe = el.get("face", {})
    lists = fe.get("vertex_indices", fe.get("vertex_index"))
    if lists is None:
        raise ValueError(f"{p.name}: no faces (a point cloud cannot be simulated)")
    if isinstance(lists, np.ndarray) and lists.ndim == 2:
        faces = lists if lists.shape[1] == 3 else np.array([t for row in lists for t in _fan(list(row))])
        n_poly = len(lists) if lists.shape[1] > 3 else 0
    else:
        tris = []
        n_poly = 0
        for row in lists:
            if len(row) > 3:
                n_poly += 1
            tris.extend(_fan(list(row)))
        faces = np.array(tris, dtype=np.int64).reshape(-1, 3)
    if n_poly:
        _note(notes, f"{p.name}: {n_poly} polygon(s) with more than 3 corners were split into triangles.")
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    ok = np.all((faces >= 0) & (faces < len(verts)), axis=1)
    if not ok.all():
        _note(notes, f"{p.name}: skipped {int((~ok).sum())} face(s) with out-of-range vertex indices.")
        faces = faces[ok]
    m = _finish(name or p.stem, verts, faces, notes, weld=False)
    if m is None:
        raise ValueError(f"no triangles in {p.name}")
    return [m]


def write_ply(meshes: Mesh3D | Sequence[Mesh3D], path, binary: bool = True, double: bool = False) -> Path:
    """Write meshes (merged: PLY holds one surface) as PLY, binary little-endian or ASCII.

    ``double`` stores float64 coordinates; the default float32 is what most viewers expect.
    """
    meshes = [meshes] if isinstance(meshes, Mesh3D) else list(meshes)
    path = Path(path)
    vs, fs, off = [], [], 0
    for m in meshes:
        vs.append(m.vertices)
        fs.append(m.faces + off)
        off += len(m.vertices)
    v = np.concatenate(vs) if vs else np.zeros((0, 3))
    f = np.concatenate(fs) if fs else np.zeros((0, 3), dtype=np.int64)
    ft = "double" if double else "float"
    header = ("ply\nformat {fmt} 1.0\ncomment Aerolume export (metres)\n"
              f"element vertex {len(v)}\nproperty {ft} x\nproperty {ft} y\nproperty {ft} z\n"
              f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
    if binary:
        vt = np.dtype([("x", "<f8" if double else "<f4"), ("y", "<f8" if double else "<f4"),
                       ("z", "<f8" if double else "<f4")])
        va = np.zeros(len(v), dtype=vt)
        va["x"], va["y"], va["z"] = v[:, 0], v[:, 1], v[:, 2]
        fa = np.zeros(len(f), dtype=np.dtype([("n", "u1"), ("i", "<i4", (3,))]))
        fa["n"] = 3
        fa["i"] = f
        with open(path, "wb") as fh:
            fh.write(header.format(fmt="binary_little_endian").encode("ascii"))
            fh.write(va.tobytes())
            fh.write(fa.tobytes())
        return path
    buf = io.StringIO()
    buf.write(header.format(fmt="ascii"))
    np.savetxt(buf, v, fmt="%.17g" if double else "%.9g")
    np.savetxt(buf, f, fmt="3 %d %d %d")
    path.write_text(buf.getvalue(), encoding="ascii")
    return path


# ---------------------------------------------------------------------------
# dispatch + units
# ---------------------------------------------------------------------------

def unit_scale(units: str | float | None) -> float:
    """Metres per file unit for a unit name ("mm", "cm", "m", "in", "ft") or a number."""
    if units is None:
        return 1.0
    if isinstance(units, (int, float)):
        if not units > 0:
            raise ValueError("a unit scale must be positive")
        return float(units)
    key = str(units).strip().lower()
    key = {"meter": "m", "metre": "m", "meters": "m", "metres": "m", "millimetre": "mm", "millimeter": "mm",
           "millimetres": "mm", "millimeters": "mm", "centimetre": "cm", "centimeter": "cm", "inch": "in",
           "inches": "in", '"': "in", "feet": "ft", "foot": "ft"}.get(key, key)
    if key not in UNIT_SCALES:
        raise ValueError(f"unknown unit {units!r} (use one of {', '.join(UNIT_SCALES)})")
    return UNIT_SCALES[key]


def guess_units(meshes: Mesh3D | Sequence[Mesh3D]) -> tuple[float, str]:
    """Guess metres per file unit from the overall size; ``(scale_to_m, reason)``.

    Meshes carry no units, but real objects have plausible sizes: a car is ~4.5 m,
    a drone ~0.4 m, a rider ~1.8 m, a wind-tunnel model ~0.2-1 m. A largest
    dimension between 1 cm and 50 units is taken as metres; 50-50 000 units as
    millimetres (what CAD exports by default: a 4500-unit car is 4.5 m, a
    300-unit drone 0.3 m); beyond that as millimetres still, flagged as large.
    Centimetres and inches cannot be told apart from millimetres by size alone,
    so the reason says which alternatives would also be plausible; the GUI asks
    the user to confirm.
    """
    meshes = [meshes] if isinstance(meshes, Mesh3D) else list(meshes)
    pts = [m.vertices for m in meshes if len(m.vertices)]
    if not pts:
        return 1.0, "The mesh is empty; assuming metres."
    v = np.concatenate(pts)
    size = float(np.max(np.ptp(v, axis=0)))
    if not size > 0:
        return 1.0, "The mesh has no extent; assuming metres."

    def m_(x: float) -> str:
        return f"{x:.3g} m" if x >= 1 else (f"{x * 100:.3g} cm" if x >= 0.01 else f"{x * 1000:.3g} mm")

    if size < 0.01:
        return 1.0, (f"Largest dimension {size:.3g} units: tiny in metres; check the units (kilometres, "
                     "or a model exported at a reduced scale? Set the scale by hand).")
    if size <= 50:
        alt = (f" In centimetres it would be {m_(size * 0.01)}, in inches {m_(size * 0.0254)}."
               if size < 5 else "")
        return 1.0, f"Largest dimension {size:.4g} units: plausible in metres ({m_(size)}).{alt}"
    if size <= 50_000:
        return 0.001, (f"Largest dimension {size:.4g} units: looks like millimetres ({m_(size * 0.001)}). "
                       f"In centimetres it would be {m_(size * 0.01)}, in inches {m_(size * 0.0254)}.")
    return 0.001, (f"Largest dimension {size:.4g} units: very large; assuming millimetres "
                   f"({m_(size * 0.001)}). Check the units.")


def read_mesh(path, *, units: str | float | None = None, notes: list[str] | None = None,
              name: str | None = None) -> list[Mesh3D]:
    """Read an .stl / .obj / .ply file into a list of :class:`Mesh3D`.

    ``units``: None keeps the file's numbers; a unit name ("mm", "cm", "m", "in",
    "ft") or a number (metres per unit) scales to metres; ``"auto"`` applies
    :func:`guess_units` and records its reason in ``notes``.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".stl":
        meshes = read_stl(p, notes=notes, name=name)
    elif ext == ".obj":
        meshes = read_obj(p, notes=notes, name=name)
    elif ext == ".ply":
        meshes = read_ply(p, notes=notes, name=name)
    else:
        raise ValueError(f"unsupported mesh file type {ext!r}; supported: "
                         f"{', '.join(sorted(SUPPORTED_EXTENSIONS_3D))}")
    if units is None:
        return meshes
    if isinstance(units, str) and units.strip().lower() == "auto":
        s, why = guess_units(meshes)
        _note(notes, why)
    else:
        s = unit_scale(units)
    if s != 1.0:
        for m in meshes:
            m.vertices = m.vertices * s
            if m.center is not None:
                m.center = tuple(float(c) * s for c in m.center)
        _note(notes, f"Scaled by {s:g} to metres.")
    return meshes
