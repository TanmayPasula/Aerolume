"""Mesh arrays of 3D designs: ``designs/<id>.npz`` next to ``designs/<id>.json``.

A 3D design's JSON (``Design.to_dict``) records each mesh's name, role, wheel
data and its vertex / face counts, but not the arrays: an imported car can have
a million triangles, and JSON lists of floats would be ~20x larger and slow to
parse. The arrays live in one ``.npz`` per design (DESIGN.md 7.10), keyed as
:meth:`Design.mesh_arrays` names them (``mesh{i}_vertices``, ``mesh{i}_faces``).

Why the extra bookkeeping:

* **Change detection.** ``Project.save()`` runs after every edit in the app and
  must not rewrite megabytes each time, but meshes may be edited in place. The
  store therefore keeps a SHA-1 of the canonical arrays (float64 vertices,
  int64 faces, with shapes) and rewrites the npz only when it changed. SHA-1
  runs at ~2.7 GB/s here (hardware SHA): 4 ms for a 200k-vertex mesh. It is a
  change detector, not a security measure.
* **Pairing.** The JSON carries the same digest under ``"mesh_file"`` and the
  npz stores it as ``__digest__``. Files are written npz first, JSON second,
  each atomically; a mismatch on open means a crash landed between the two
  writes (or a file was copied from another save), which the store reports.
* **Size.** Vertices (float64) and faces (int32 when they fit) are stored
  byte-shuffled (see ``project/results.py``) and deflated at level 1: vertices
  read from an STL are float32 values, whose low float64 mantissa bytes are all
  zero, so a 200k-vertex sphere-like mesh takes 1.7 MB instead of 9.6 MB, in
  45 ms. The layout (dtype, shape per array) is a JSON string inside the npz.

A plain ``np.savez(path, **design.mesh_arrays())`` file (no ``__layout__``) is
read as well, so tools and scripts can write the simple form.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..model import Design
from .fileio import npz_bytes

MESH_FORMAT = "aerolume-mesh"
#: key in the design JSON holding {"format", "digest"} of the paired npz
JSON_KEY = "mesh_file"


def canonical_arrays(design: Design) -> dict[str, np.ndarray]:
    """``design.mesh_arrays()`` as C-contiguous little-endian float64 / int64 arrays."""
    out = {}
    for k, a in design.mesh_arrays().items():
        dt = "<f8" if k.endswith("_vertices") else "<i8"
        out[k] = np.ascontiguousarray(a, dtype=dt)
    return out


def digest_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """SHA-1 over names, dtypes, shapes and bytes of canonical mesh arrays (see the module docstring)."""
    h = hashlib.sha1(usedforsecurity=False)
    for k in sorted(arrays):
        a = arrays[k]
        h.update(f"{k}|{a.dtype.str}|{a.shape};".encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def mesh_digest(design: Design) -> str:
    """Digest of a design's current mesh arrays."""
    return digest_arrays(canonical_arrays(design))


def pack_meshes(design: Design, digest: str | None = None) -> tuple[bytes, str]:
    """The npz bytes for a design's meshes and their digest."""
    arrays = canonical_arrays(design)
    digest = digest or digest_arrays(arrays)
    out: dict[str, np.ndarray] = {}
    layout: dict[str, dict] = {}
    for k, a in arrays.items():
        if k.endswith("_faces") and (a.size == 0 or int(a.max()) < 2 ** 31 - 1):
            a = a.astype("<i4")
        layout[k] = {"dtype": a.dtype.str, "shape": list(a.shape)}
        out[k] = a.reshape(-1).view(np.uint8).reshape(-1, a.itemsize).T.copy()
    out["__layout__"] = np.array(json.dumps({"format": MESH_FORMAT, "enc": "shuffle", "arrays": layout}))
    out["__digest__"] = np.array(digest)
    return npz_bytes(out, level=1), digest


def read_meshes(path: str | Path) -> tuple[dict[str, np.ndarray], str | None]:
    """Mesh arrays (float64 vertices, int64 faces) and the stored digest (None for a plain npz).

    Raises ``ValueError`` / ``OSError`` / ``zipfile.BadZipFile`` for unreadable files;
    the store turns those into a load warning.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    digest = str(data.pop("__digest__")) if "__digest__" in data else None
    if "__layout__" not in data:
        return {k: np.asarray(v) for k, v in data.items()}, digest
    layout = json.loads(str(data.pop("__layout__")))
    if not isinstance(layout, dict) or layout.get("enc") != "shuffle":
        raise ValueError("unknown mesh file encoding")
    out = {}
    for k, meta in layout.get("arrays", {}).items():
        dt = np.dtype(meta["dtype"])
        raw = data[k]
        if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[0] != dt.itemsize:
            raise ValueError(f"mesh array {k!r} does not match its layout")
        out[k] = raw.T.copy().view(dt).reshape(tuple(int(n) for n in meta["shape"]))
    return out, digest


def check_shapes(design: Design, arrays: Mapping[str, np.ndarray], counts: list[tuple[int, int]] | None) -> str | None:
    """Why ``arrays`` cannot belong to ``design`` (None when they fit).

    ``counts`` are the (n_vertices, n_faces) per mesh recorded in the design JSON
    (None when it did not record them).
    """
    for i, _m in enumerate(design.meshes):
        kv, kf = f"mesh{i}_vertices", f"mesh{i}_faces"
        if kv not in arrays or kf not in arrays:
            return f"mesh {i + 1} has no arrays"
        v, f = np.asarray(arrays[kv]), np.asarray(arrays[kf])
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            return f"mesh {i + 1} arrays have the wrong shape"
        if counts is not None and i < len(counts) and counts[i] != (len(v), len(f)):
            return (f"mesh {i + 1} has {len(v)} vertices / {len(f)} faces on disk, the design "
                    f"expects {counts[i][0]} / {counts[i][1]}")
        if f.size and (int(f.min()) < 0 or int(f.max()) >= len(v)):
            return f"mesh {i + 1} faces index vertices that do not exist"
    return None


def json_counts(doc: Mapping) -> list[tuple[int, int]] | None:
    """(n_vertices, n_faces) per mesh as recorded in a design JSON document, or None."""
    meshes = doc.get("meshes")
    if not isinstance(meshes, list):
        return None
    out = []
    for m in meshes:
        if not isinstance(m, dict) or "n_vertices" not in m or "n_faces" not in m:
            return None
        try:
            out.append((int(m["n_vertices"]), int(m["n_faces"])))
        except (TypeError, ValueError):
            return None
    return out


def design_document(design: Design, digest: str | None) -> dict:
    """The JSON document of a design: ``to_dict()`` plus the mesh-file pairing record.

    A 3D design without 2D bodies (every 3D design the program makes) leaves out the
    ``bodies`` key. Aerolume 0.1 requires it, so 0.1 skips the file with a load
    warning instead of reading it as an empty 2D design and later saving it back
    without its dimension and meshes (project/store.py, schema policy);
    :func:`design_from_document` puts the empty list back.
    """
    doc = design.to_dict()
    if design.dimension == 3 and not design.bodies:
        doc.pop("bodies", None)
    if digest is not None:
        doc[JSON_KEY] = {"format": MESH_FORMAT, "digest": digest}
    return doc


def design_from_document(doc: Mapping) -> Design:
    """``Design.from_dict`` of a design JSON document written by :func:`design_document`
    (or by an earlier version, which always wrote ``bodies``)."""
    if isinstance(doc, Mapping) and "bodies" not in doc and doc.get("dimension") == 3:
        doc = {**doc, "bodies": []}
    return Design.from_dict(doc)  # type: ignore[arg-type]


def recorded_digest(doc: Mapping) -> str | None:
    """The mesh digest a design JSON document records (None when it records none)."""
    info = doc.get(JSON_KEY)
    digest = info.get("digest") if isinstance(info, dict) else None
    return digest if isinstance(digest, str) else None


def has_mesh_data(design: Design) -> bool:
    """Whether the design has meshes whose arrays belong in an npz."""
    return bool(design.meshes)


__all__ = ["JSON_KEY", "MESH_FORMAT", "canonical_arrays", "check_shapes", "design_document", "design_from_document",
           "digest_arrays", "has_mesh_data", "json_counts", "mesh_digest", "pack_meshes", "read_meshes",
           "recorded_digest"]
