"""DXF import via ezdxf.

CAD side profiles come in every possible state: one tidy closed LWPOLYLINE with
bulge arcs; a heap of LINE and ARC entities in random order and direction; old
2D POLYLINEs; SPLINEs from a surfacing tool; outlines buried in block INSERTs.
This reader turns all of them into closed polygons:

* LWPOLYLINE (incl. bulges), POLYLINE, LINE, ARC, CIRCLE, ELLIPSE, SPLINE and
  HATCH boundaries are converted with ``ezdxf.path.make_path`` and flattened to a
  distance tolerance -- curves are approximated, never just sampled at the
  definition points.
* INSERTs are exploded recursively (their block transforms applied).
* Open pieces are chained end-to-end where endpoints coincide within ``join_tol``
  (CAD "closed" outlines made of separate entities), then closed.

Coordinates are returned in drawing units (usually mm or m; ``$INSUNITS`` is
reported in ``notes``); the z coordinate is ignored, which is right for a 2D
drawing in the XY plane.
"""
from __future__ import annotations

from pathlib import Path

import ezdxf
import numpy as np

from .polygon import close_gaps, orient_ccw, remove_duplicates, signed_area

__all__ = ["read_dxf"]

_SUPPORTED = {"LWPOLYLINE", "POLYLINE", "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "HATCH"}
_INSUNITS = {0: "unitless", 1: "inches", 2: "feet", 4: "millimetres", 5: "centimetres", 6: "metres"}


def _entities(layout, depth: int = 0):
    """Yield drawable entities, exploding block references (max 8 levels deep)."""
    for e in layout:
        t = e.dxftype()
        if t == "INSERT" and depth < 8:
            # a malformed block reference is skipped rather than failing the import
            try:
                yield from _entities(e.virtual_entities(), depth + 1)
            except (ezdxf.DXFError, ValueError, TypeError, AttributeError):   # pragma: no cover
                continue
        elif t in _SUPPORTED:
            yield e


def read_dxf(path, *, tolerance: float | None = None, join_tol: float | None = None,
             layers: list[str] | None = None, notes: list[str] | None = None) -> list[np.ndarray]:
    """Read the model space of a DXF file as CCW, implicitly closed polygons.

    ``tolerance``: maximum chord deviation when flattening curves (drawing units);
    default 2e-4 of the drawing extent. ``join_tol``: endpoint distance below which
    open pieces are joined; default 1e-4 of the extent (covers the round-off of
    typical exports without bridging intentional gaps). ``layers`` restricts import
    to the named layers. Messages about closed gaps and units go to ``notes``.
    """
    from ezdxf import path as ezpath

    doc = ezdxf.readfile(str(Path(path)))
    msp = doc.modelspace()
    ents = [e for e in _entities(msp)
            if layers is None or e.dxf.get("layer", "0") in layers]
    if notes is not None:
        u = int(doc.header.get("$INSUNITS", 0) or 0)
        if u in _INSUNITS and u != 0:
            notes.append(f"DXF drawing units: {_INSUNITS[u]}.")

    paths = []
    for e in ents:
        try:
            if e.dxftype() == "HATCH":
                paths.extend(ezpath.from_hatch(e))   # boundary paths
            else:
                # segments: Bezier pieces per full circle; 32 keeps arcs exact to ~1e-8
                paths.append(ezpath.make_path(e, segments=32))
        except (ezdxf.DXFError, ValueError, TypeError, AttributeError, ArithmeticError, NotImplementedError):
            # unsupported entity variant: skip it, but say so
            if notes is not None:
                notes.append(f"Skipped a {e.dxftype()} entity that could not be converted.")
    paths = [p for p in paths if len(p) > 0 or p.start != p.end]
    if not paths:
        return []

    # extent from control vertices (cheap, and bounds the curves well enough)
    ctrl = np.array([[v.x, v.y] for p in paths for v in p.control_vertices()], dtype=float)
    size = max(float(np.ptp(ctrl, axis=0).max()), 1e-300)
    tol = tolerance if tolerance and tolerance > 0 else 2e-4 * size
    jtol = join_tol if join_tol and join_tol > 0 else 1e-4 * size

    closed, open_ = [], []
    for p in paths:
        # sub-paths (e.g. a HATCH with islands, multi-part paths) are handled one by one
        for sp in (p.sub_paths() if p.has_sub_paths else [p]):
            pts = np.array([[v.x, v.y] for v in sp.flattening(distance=tol, segments=8)], dtype=float)
            if len(pts) < 2:
                continue
            if np.hypot(*(pts[-1] - pts[0])) <= jtol and len(pts) > 3:
                closed.append(pts[:-1])
            else:
                open_.append(pts)
    polys = closed + close_gaps(open_, tol=jtol, notes=notes)
    out = []
    for P in polys:
        P = remove_duplicates(P, tol=1e-9 * size)
        if len(P) >= 3 and abs(signed_area(P)) > 1e-12 * size * size:
            out.append(orient_ccw(P))
    return out
