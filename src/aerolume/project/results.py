"""Result (de)serialisation: small scalars to JSON, large arrays to one ``.npz``.

``result.json`` holds everything the Compare page, the report tables and the
simulation list need (coefficients, statistics, Reynolds numbers, warnings), so
those read in milliseconds without touching the arrays. ``arrays.npz`` holds the
force histories (float64: a statistic is only as good as its samples), surface
distributions (float64, small) and flow fields (float16, the bulk of the data).

float16 has a range of +-65504 and 11 bits of mantissa. Pressure in Pa or
vorticity in 1/s easily exceed that range (water at 30 m/s: q = 450 kPa), so every
field is divided by its largest finite magnitude before the cast and the scale is
kept in the JSON; the stored relative precision is then ~5e-4 of the field's
peak everywhere, ample for colour maps, probes and VTK export. Fields come back as
float32. A temperature in °C is stored about its mid-range instead (``"offset"`` in
the layout, :data:`OFFSET_FIELDS`), because its peak is the absolute temperature,
not the variation that matters.

Every ``result.json`` also carries the result's provenance record under the
top-level key ``"provenance"`` (``project/provenance.py``): software, settings as
run, hashes, device and seed. Loading puts it back as ``result.provenance``.

``result.json`` and ``arrays.npz`` are written as two atomic files, arrays first.
Both carry the same random ``token``; a mismatch on load means a crash landed
between the two writes, and is reported instead of silently pairing new scalars
with old fields.

Result types (``result.json`` ``"type"``): ``"cfd"`` (:class:`CFDResult`),
``"polar"`` (:class:`PolarResult`), ``"cfd3d"`` (:class:`CFD3DResult`) and
``"wing"`` (:class:`WingResult`). The 2D layouts are unchanged from v0.1, so a
v0.1 Aerolume still reads 2D results written by this version.

**3D storage** (DESIGN.md 7.10). A 3D result is one to two orders of magnitude
larger than a 2D one (a 2 M-cell volume with five fields, a 200k-vertex surface),
so its arrays are stored *byte-shuffled*: an array of ``itemsize``-byte numbers is
saved as an (itemsize, n) uint8 block, all first bytes, then all second bytes, and
so on (the HDF5 / Blosc "shuffle" filter). Exponent and high-mantissa bytes of
neighbouring cells are nearly equal, so deflate finds them far more easily;
measured on a 200k-vertex surface, shuffled float32 + int32 compress 2.5x better
than the plain arrays and twice as fast, and a float16 volume 10 % better. The
JSON layout records dtype and shape of each array (``"enc": "shuffle"``), so the
encoding can change in a later version without breaking these files. The 3D
arrays are deflated at zlib level 1 (``fileio.npz_bytes``). Precision:

* histories float64 (exact), wing span distributions float64 (exact, small);
* surface vertices / cp / cf / tau / sensitivity float32 (7 significant digits;
  0.5 micrometre on a 4 m car), faces int32 (int64 above 2**31 vertices);
* slices and the volume float16 with a per-field scale, like the 2D fields; the
  solid fraction (mask) likewise. ``CFD3DResult.extra_volumes`` (the "instant"
  smoke snapshot of DESIGN.md 7.13) are stored exactly like the volume, under
  ``volumes/<i>/`` and the layout key ``"extra_volumes"``; a result without
  them (any result written before they existed) loads with an empty list.
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..model import (
    CFD3DResult,
    CFDResult,
    FieldData,
    PolarPoint,
    PolarResult,
    SliceData,
    SurfaceData,
    SurfaceData3D,
    VolumeData,
    WingPoint,
    WingResult,
)
from .fileio import ProjectCorruptError

RESULT_FORMAT = "aerolume-result"
RESULT_SCHEMA = 1
#: result types this version reads and writes (``result.json`` "type")
RESULT_TYPES = ("cfd", "polar", "cfd3d", "wing")
_SURFACE_KEYS = ("s", "x", "y", "nx", "ny", "cp", "cf", "sensitivity")
_POINT_ARRAYS = PolarPoint._arrays
_WING_ARRAYS = WingPoint._arrays
_FIELD_SLOTS = ("mean_fields", "instant_fields")
_SURFACE3D_KEYS = ("vertices", "faces", "cp", "cf", "tau", "sensitivity")

AnyResult = CFDResult | PolarResult | CFD3DResult | WingResult


def result_type(result: Any) -> str:
    """The ``result.json`` type name of a result object (``TypeError`` for anything else)."""
    if isinstance(result, CFDResult):
        return "cfd"
    if isinstance(result, PolarResult):
        return "polar"
    if isinstance(result, CFD3DResult):
        return "cfd3d"
    if isinstance(result, WingResult):
        return "wing"
    raise TypeError(f"not a result: {type(result).__name__}")


# ---------------------------------------------------------------------------
# 2D encodings (v0.1 format, unchanged)
# ---------------------------------------------------------------------------

def _to_f16(a: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.asarray(a, dtype=np.float32)
    fin = np.isfinite(a)
    scale = float(np.abs(a[fin]).max()) if fin.any() else 1.0
    if scale == 0.0:
        scale = 1.0
    return (a / scale).astype(np.float16), scale


#: Fields reported on an affine scale (°C) are stored relative to their mid-range when
#: their values sit far from zero compared with their spread: 20 °C ± 0.05 K scaled by
#: its largest magnitude keeps 0.01 K of resolution in float16, a fifth of the whole
#: variation; about its mid-range it keeps 2.5e-5 K. Only these fields, so every field
#: that Aerolume 0.1 / 0.2 know keeps their exact layout and those versions still read
#: it; results without the "offset" key decode exactly as before.
OFFSET_FIELDS = ("temperature",)
OFFSET_RATIO = 4.0


def _to_f16_offset(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(float16 array, scale, offset) with ``a ≈ offset + scale * stored`` (see :data:`OFFSET_RATIO`)."""
    a = np.asarray(a, dtype=np.float32)
    fin = np.isfinite(a)
    if not fin.any():
        return a.astype(np.float16), 1.0, 0.0
    lo, hi = float(a[fin].min()), float(a[fin].max())
    mag = max(abs(lo), abs(hi))
    half = 0.5 * (hi - lo)
    if half > 0.0 and mag > OFFSET_RATIO * half:
        offset = 0.5 * (hi + lo)
        return ((a.astype(np.float64) - offset) / half).astype(np.float16), half, offset
    stored, scale = _to_f16(a)
    return stored, scale, 0.0


def _pack_fields(fd: FieldData, prefix: str, arrays: dict[str, np.ndarray]) -> dict:
    layout: dict[str, Any] = {"extent": [float(v) for v in fd.extent], "label": fd.label, "fields": {},
                              "mask": None}
    for name, a in fd.fields.items():
        if "/" in name:
            raise ValueError(f"field name {name!r} may not contain '/'")
        if name in OFFSET_FIELDS:
            arrays[f"{prefix}/{name}"], scale, offset = _to_f16_offset(a)
        else:
            (arrays[f"{prefix}/{name}"], scale), offset = _to_f16(a), 0.0
        layout["fields"][name] = {"scale": scale, "shape": list(np.shape(a))}
        if offset:
            layout["fields"][name]["offset"] = offset
    if fd.mask is not None:
        arrays[f"{prefix}/__mask__"], scale = _to_f16(fd.mask)
        layout["mask"] = {"scale": scale}
    return layout


def _decode(stored: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
    out = stored.astype(np.float32) * np.float32(info["scale"])
    offset = info.get("offset")
    return out + np.float32(offset) if offset else out


def _unpack_fields(layout: dict, prefix: str, z: Mapping[str, np.ndarray]) -> FieldData:
    fields = {name: _decode(z[f"{prefix}/{name}"], info) for name, info in layout["fields"].items()}
    mask = None
    if layout.get("mask"):
        mask = z[f"{prefix}/__mask__"].astype(np.float32) * np.float32(layout["mask"]["scale"])
    return FieldData(extent=tuple(float(v) for v in layout["extent"]), fields=fields, mask=mask,  # type: ignore[arg-type]
                     label=layout.get("label", "mean"))


# ---------------------------------------------------------------------------
# 3D encodings: byte-shuffled arrays described in the JSON layout
# ---------------------------------------------------------------------------

def _put(arrays: dict[str, np.ndarray], key: str, a: Any, dtype: Any, *, scale: bool = False,
         offset: bool = False) -> dict:
    """Store ``a`` as ``dtype`` byte-shuffled under ``key``; return its layout entry.

    ``scale=True`` (float16 storage) divides by the largest finite magnitude
    first, exactly as :func:`_to_f16`, and records the factor; with ``offset=True``
    a field far from zero is stored about its mid-range (:data:`OFFSET_RATIO`).
    """
    a = np.asarray(a)
    meta: dict[str, Any] = {"enc": "shuffle", "shape": list(a.shape)}
    if scale and offset:
        a, meta["scale"], off = _to_f16_offset(a)
        if off:
            meta["offset"] = off
    elif scale:
        a, meta["scale"] = _to_f16(a)
    a = np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder("<"))
    meta["dtype"] = a.dtype.str
    arrays[key] = a.reshape(-1).view(np.uint8).reshape(-1, a.itemsize).T.copy()
    return meta


def _get(z: Mapping[str, np.ndarray], key: str, meta: Mapping[str, Any]) -> np.ndarray:
    """Inverse of :func:`_put`; scaled float16 comes back as float32."""
    raw = np.asarray(z[key])
    enc = meta.get("enc", "shuffle")
    if enc != "shuffle":
        raise ValueError(f"unknown array encoding {enc!r} for {key!r}")
    dt = np.dtype(meta["dtype"])
    if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[0] != dt.itemsize:
        raise ValueError(f"array {key!r} does not match its layout")
    a = raw.T.copy().view(dt).reshape(tuple(int(n) for n in meta["shape"]))
    a = a.astype(dt.newbyteorder("="), copy=False)
    if "scale" in meta:
        return _decode(a, meta)
    return a


def _index_dtype(faces: np.ndarray) -> Any:
    return np.int32 if faces.size == 0 or int(np.max(faces)) < 2 ** 31 - 1 else np.int64


def _check_name(name: str) -> None:
    if "/" in name:
        raise ValueError(f"field name {name!r} may not contain '/'")


def _pack_planar(fields: Mapping[str, np.ndarray], mask: np.ndarray | None, prefix: str,
                 arrays: dict[str, np.ndarray]) -> dict:
    out: dict[str, Any] = {"fields": {}, "mask": None}
    for name, a in fields.items():
        _check_name(name)
        out["fields"][name] = _put(arrays, f"{prefix}/{name}", a, np.float16, scale=True,
                                   offset=name in OFFSET_FIELDS)
    if mask is not None:
        out["mask"] = _put(arrays, f"{prefix}/__mask__", mask, np.float16, scale=True)
    return out


def _unpack_planar(layout: Mapping[str, Any], prefix: str,
                   z: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    fields = {name: _get(z, f"{prefix}/{name}", meta) for name, meta in layout.get("fields", {}).items()}
    mask = _get(z, f"{prefix}/__mask__", layout["mask"]) if layout.get("mask") else None
    return fields, mask


def _pack_cfd3d(result: CFD3DResult, arrays: dict[str, np.ndarray]) -> dict:
    layout: dict[str, Any] = {}
    hist_keys = []
    for k, v in result.history.items():
        arrays[f"history/{k}"] = np.asarray(v, dtype=np.float64)
        hist_keys.append(k)
    layout["history"] = hist_keys
    surf = []
    for i, sd in enumerate(result.surface):
        keys: dict[str, Any] = {}
        for k in _SURFACE3D_KEYS:
            v = getattr(sd, k)
            if v is None:
                continue
            dt = _index_dtype(np.asarray(v)) if k == "faces" else np.float32
            keys[k] = _put(arrays, f"surface3d/{i}/{k}", v, dt)
        surf.append({"body": sd.body, "arrays": keys})
    layout["surface"] = surf
    slices = []
    for i, sl in enumerate(result.slices):
        entry = {"axis": sl.axis, "position": float(sl.position), "extent": [float(v) for v in sl.extent],
                 "label": sl.label}
        entry.update(_pack_planar(sl.fields, sl.mask, f"slices/{i}", arrays))
        slices.append(entry)
    layout["slices"] = slices
    layout["volume"] = _pack_volume(result.volume, "volume", arrays) if result.volume is not None else None
    # extra blocks (DESIGN.md 7.13, e.g. the "instant" smoke snapshot) under their own key and prefix:
    # a reader that predates them ignores both, so it still loads the rest of the result
    if result.extra_volumes:
        layout["extra_volumes"] = [_pack_volume(vol, f"volumes/{i}", arrays)
                                   for i, vol in enumerate(result.extra_volumes)]
    return layout


def _pack_volume(vol: VolumeData, prefix: str, arrays: dict[str, np.ndarray]) -> dict:
    entry: dict[str, Any] = {"origin": [float(v) for v in vol.origin], "spacing": [float(v) for v in vol.spacing],
                             "label": vol.label, "mirrored": bool(vol.mirrored)}
    entry.update(_pack_planar(vol.fields, vol.mask, prefix, arrays))
    return entry


def _unpack_volume(entry: Mapping[str, Any], prefix: str, z: Mapping[str, np.ndarray]) -> VolumeData:
    fields, mask = _unpack_planar(entry, prefix, z)
    return VolumeData(origin=tuple(float(c) for c in entry["origin"]),  # type: ignore[arg-type]
                      spacing=tuple(float(c) for c in entry["spacing"]),  # type: ignore[arg-type]
                      fields=fields, mask=mask, label=entry.get("label", "mean"),
                      mirrored=bool(entry.get("mirrored", False)))


def _unpack_cfd3d(res: CFD3DResult, layout: Mapping[str, Any], z: Mapping[str, np.ndarray]) -> None:
    res.history = {k: np.asarray(z[f"history/{k}"], dtype=np.float64) for k in layout.get("history", [])}
    surfaces = []
    for i, s in enumerate(layout.get("surface", [])):
        got = {k: _get(z, f"surface3d/{i}/{k}", meta) for k, meta in s.get("arrays", {}).items()
               if k in _SURFACE3D_KEYS}
        if "faces" in got:
            got["faces"] = got["faces"].astype(np.int64)
        surfaces.append(SurfaceData3D(body=s["body"], **got))
    res.surface = surfaces
    slices = []
    for i, s in enumerate(layout.get("slices", [])):
        fields, mask = _unpack_planar(s, f"slices/{i}", z)
        slices.append(SliceData(axis=s["axis"], position=float(s["position"]),
                                extent=tuple(float(v) for v in s["extent"]),  # type: ignore[arg-type]
                                fields=fields, mask=mask, label=s.get("label", "mean")))
    res.slices = slices
    v = layout.get("volume")
    if v:
        res.volume = _unpack_volume(v, "volume", z)
    res.extra_volumes = [_unpack_volume(e, f"volumes/{i}", z) for i, e in enumerate(layout.get("extra_volumes") or [])]


def _pack_points(points: list, dicts: list[dict], names: tuple[str, ...], arrays: dict[str, np.ndarray]) -> list:
    """Move per-point distributions out of the JSON into ``points/<i>/<name>`` (float64)."""
    out = []
    for i, (p, pd) in enumerate(zip(points, dicts, strict=True)):
        keys = []
        for k in names:
            v = getattr(p, k)
            pd[k] = None
            if v is not None:
                arrays[f"points/{i}/{k}"] = np.asarray(v, dtype=np.float64)
                keys.append(k)
        out.append(keys)
    return out


def _unpack_points(points: list, layout: list, z: Mapping[str, np.ndarray]) -> None:
    for i, keys in enumerate(layout):
        for k in keys:
            setattr(points[i], k, np.asarray(z[f"points/{i}/{k}"], dtype=np.float64))


def _pack_diagnostics(result: Any, layout: dict, arrays: dict[str, np.ndarray]) -> None:
    """The solver's diagnostics table (``result.diagnostics_history``, DESIGN.md 10.4), float64.

    It is sampled every few steps, not every step, so it cannot share the force history's
    time column; it gets its own key, which readers that predate it ignore."""
    dh = getattr(result, "diagnostics_history", None) or {}
    if not isinstance(dh, Mapping) or not dh:
        return
    keys = []
    for k, v in dh.items():
        _check_name(str(k))
        arrays[f"diagnostics_history/{k}"] = np.asarray(v, dtype=np.float64)
        keys.append(str(k))
    layout["diagnostics_history"] = keys


def _unpack_diagnostics(res: Any, layout: Mapping[str, Any], z: Mapping[str, np.ndarray]) -> None:
    keys = layout.get("diagnostics_history")
    if keys:
        res.diagnostics_history = {k: np.asarray(z[f"diagnostics_history/{k}"], dtype=np.float64) for k in keys}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def pack_result(result: AnyResult) -> tuple[dict, dict[str, np.ndarray]]:
    """Split a result into the ``result.json`` document and the ``arrays.npz`` content."""
    token = uuid.uuid4().hex
    arrays: dict[str, np.ndarray] = {"__token__": np.array(token)}
    layout: dict[str, Any] = {}
    kind = result_type(result)
    scalars = result.to_dict()
    # the provenance record is the document's own top-level "provenance" key (the store
    # writes it), never part of the result's scalars, should the model grow such a field
    scalars.pop("provenance", None)
    scalars.pop("diagnostics_history", None)
    if isinstance(result, CFDResult):
        hist_keys = []
        for k, v in result.history.items():
            arrays[f"history/{k}"] = np.asarray(v, dtype=np.float64)
            hist_keys.append(k)
        layout["history"] = hist_keys
        surf = []
        for i, sd in enumerate(result.surface):
            keys = []
            for k in _SURFACE_KEYS:
                v = getattr(sd, k)
                if v is not None:
                    arrays[f"surface/{i}/{k}"] = np.asarray(v, dtype=np.float64)
                    keys.append(k)
            surf.append({"body": sd.body, "keys": keys})
        layout["surface"] = surf
        for slot in _FIELD_SLOTS:
            fd = getattr(result, slot)
            layout[slot] = _pack_fields(fd, slot, arrays) if fd is not None else None
    elif isinstance(result, PolarResult):
        layout["points"] = _pack_points(result.points, scalars["points"], _POINT_ARRAYS, arrays)
    elif isinstance(result, CFD3DResult):
        layout = _pack_cfd3d(result, arrays)
    else:  # WingResult
        layout["points"] = _pack_points(result.points, scalars["points"], _WING_ARRAYS, arrays)
    if isinstance(result, (CFDResult, CFD3DResult)):
        _pack_diagnostics(result, layout, arrays)
    doc = {"format": RESULT_FORMAT, "schema": RESULT_SCHEMA, "type": kind, "token": token,
           "result": scalars, "arrays": layout}
    return doc, arrays


def unpack_result(doc: dict, z: Mapping[str, np.ndarray] | None, where: str = "") -> AnyResult:
    """Rebuild a result from its JSON document and (optionally) its arrays.

    ``z=None`` returns the scalar part only (histories, surfaces, slices, volume
    and fields empty; polar / wing points without their distributions). The
    document's provenance record, when it has one, is put back as
    ``result.provenance`` (a dict), so a report or a script holding only the result
    can still say what produced it.
    """
    res = _unpack(doc, z, where)
    prov = doc.get("provenance")
    if isinstance(prov, dict):
        try:
            res.provenance = prov  # type: ignore[union-attr]
        except AttributeError:
            pass
    return res


def _unpack(doc: dict, z: Mapping[str, np.ndarray] | None, where: str) -> AnyResult:
    if not isinstance(doc, dict) or doc.get("format") != RESULT_FORMAT:
        raise ProjectCorruptError(f"not an Aerolume result file: {where}")
    kind = doc.get("type")
    layout = doc.get("arrays") or {}
    if kind not in RESULT_TYPES:
        raise ProjectCorruptError(
            f"unknown result type {kind!r} (saved by a newer Aerolume?): {where}")
    if z is not None:
        tok = str(z["__token__"]) if "__token__" in z else None
        if tok != doc.get("token"):
            raise ProjectCorruptError(
                f"result arrays do not belong to this result (interrupted save?): {where}")
    try:
        if kind == "cfd":
            res = CFDResult.from_dict(doc["result"])
            if z is not None:
                _unpack_diagnostics(res, layout, z)
                res.history = {k: np.asarray(z[f"history/{k}"], dtype=np.float64)
                               for k in layout.get("history", [])}
                res.surface = [SurfaceData(body=s["body"], **{k: np.asarray(z[f"surface/{i}/{k}"])
                                                              for k in s["keys"]})
                               for i, s in enumerate(layout.get("surface", []))]
                for slot in _FIELD_SLOTS:
                    if layout.get(slot):
                        setattr(res, slot, _unpack_fields(layout[slot], slot, z))
            return res
        if kind == "polar":
            res = PolarResult.from_dict(doc["result"])
            if z is not None:
                _unpack_points(res.points, layout.get("points", []), z)
            return res
        if kind == "cfd3d":
            res3 = CFD3DResult.from_dict(doc["result"])
            if z is not None:
                _unpack_cfd3d(res3, layout, z)
                _unpack_diagnostics(res3, layout, z)
            return res3
        resw = WingResult.from_dict(doc["result"])
        if z is not None:
            _unpack_points(resw.points, layout.get("points", []), z)
        return resw
    except KeyError as exc:
        raise ProjectCorruptError(f"result is missing {exc.args[0]!r}: {where}") from None
    except (TypeError, ValueError, IndexError, AttributeError) as exc:
        raise ProjectCorruptError(f"result is malformed ({exc}): {where}") from None
