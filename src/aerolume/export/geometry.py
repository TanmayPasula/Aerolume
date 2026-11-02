"""Geometry exports: SVG (drawings, web), DXF (CAD) and Selig ``.dat`` (airfoil tools).

All three write the design frame (x downstream, y up, metres), not the rotated
wind frame, so an exported shape re-imports exactly where it came from.

* **SVG** is in millimetres (``width``/``height`` in mm and a viewBox in mm, so one
  user unit is one millimetre). SVG's y axis points down; it is flipped here.
* **DXF** (via ezdxf) holds one closed LWPOLYLINE per body on a layer named after the
  body role (BODY, WHEEL, ELEMENT); ``$INSUNITS`` records the unit so CAD packages
  scale it correctly. Metres by default, millimetres on request.
* **Selig .dat** is the de-facto airfoil exchange format (XFOIL, XFLR5, UIUC
  database): a name line, then x y pairs from the upper trailing edge forward over
  the upper surface, round the leading edge and back along the lower surface.
  That order is counter-clockwise, the same as Aerolume's outlines, so writing it is
  a rotation of the point list, not a re-ordering.
"""
from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from ..model import Design
from ..project.fileio import atomic_writer, write_text


def _ccw(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return p if area2 >= 0 else p[::-1].copy()


def design_svg(design: Design, path: str | os.PathLike, *, margin: float = 0.05,
               fill: str = "#3fb8af", stroke: str = "#1d5f5a") -> Path:
    """Write the design's outlines as an SVG drawing in millimetres.

    ``margin`` is the border around the shape as a fraction of its largest size.
    Each body is a closed ``<path>`` with ``id`` = body name and ``data-role``.
    """
    if not design.bodies:
        raise ValueError("design has no bodies")
    xmin, xmax, ymin, ymax = design.bounds()
    size = max(xmax - xmin, ymax - ymin) or 1.0
    m = margin * size
    w_mm, h_mm = (xmax - xmin + 2 * m) * 1e3, (ymax - ymin + 2 * m) * 1e3
    sw = 0.002 * max(w_mm, h_mm)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.4f}mm" height="{h_mm:.4f}mm" '
              f'viewBox="0 0 {w_mm:.4f} {h_mm:.4f}">'),
             f"  <title>{escape(design.name)}</title>",
             (f"  <desc>Exported by Aerolume. Units: mm. Design frame, y flipped to SVG's "
              f"downward axis. length_ref = {design.length_ref:g} m.</desc>"),
             (f'  <g fill="{fill}" fill-opacity="0.35" stroke="{stroke}" stroke-width="{sw:.4f}" '
              f'stroke-linejoin="round">')]
    for b in design.bodies:
        p = np.asarray(b.points, dtype=np.float64)
        sx = (p[:, 0] - xmin + m) * 1e3
        sy = (ymax - p[:, 1] + m) * 1e3
        d = "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in zip(sx.tolist(), sy.tolist(), strict=True)) + " Z"
        lines.append(f"    <path id={quoteattr(b.name)} data-role={quoteattr(b.role)} d=\"{d}\"/>")
    lines += ["  </g>", "</svg>", ""]
    return write_text(path, "\n".join(lines))


_DXF_UNITS = {"m": (6, 1.0), "mm": (4, 1e3)}


def design_dxf(design: Design, path: str | os.PathLike, *, units: str = "m") -> Path:
    """Write the design as a DXF (R2010) with one closed polyline per body.

    ``units`` is "m" (default, lossless round trip) or "mm" (what most mechanical
    CAD expects); the choice is recorded in ``$INSUNITS``.
    """
    import ezdxf  # imported lazily: only this export needs it

    if units not in _DXF_UNITS:
        raise ValueError(f"units must be one of {tuple(_DXF_UNITS)}")
    code, scale = _DXF_UNITS[units]
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = code
    doc.header["$MEASUREMENT"] = 1          # metric
    colours = {"body": 4, "wheel": 8, "element": 3}   # ACI cyan, grey, green
    msp = doc.modelspace()
    for b in design.bodies:
        layer = b.role.upper()
        if layer not in doc.layers:
            doc.layers.add(layer, color=colours.get(b.role, 7))
        pts = (np.asarray(b.points, dtype=np.float64) * scale).tolist()
        msp.add_lwpolyline(pts, format="xy", close=True, dxfattribs={"layer": layer})
    path = Path(path)
    with atomic_writer(path, "w", encoding=doc.encoding) as fh:
        doc.write(fh)
    return path


def selig_order(points: np.ndarray) -> np.ndarray:
    """Rotate a CCW airfoil outline to Selig order: upper TE -> LE -> lower TE.

    For a sharp trailing edge (one aftmost point) that point starts the list and is
    repeated at the end, as Selig files do. For a blunt one the upper corner starts
    and the lower corner ends the list; the base is the implied closing segment.
    """
    p = _ccw(points)
    x = p[:, 0]
    tol = 1e-7 * max(float(np.ptp(x)), 1e-12)
    aft = np.nonzero(x >= x.max() - tol)[0]
    start = int(aft[np.argmax(p[aft, 1])])
    q = np.roll(p, -start, axis=0)
    if aft.size == 1:
        q = np.vstack([q, q[:1]])
    return q


def design_dat(design: Design, path: str | os.PathLike, *, body: int = 0, normalize: bool = True) -> Path:
    """Write one body of the design as a Selig-format airfoil ``.dat`` file.

    ``normalize`` (default) shifts the leading edge (the foremost point) to the
    origin and scales the chord (x extent) to 1, the convention XFOIL and the UIUC
    database expect; otherwise coordinates are written in metres as stored.
    """
    if not design.bodies:
        raise ValueError("design has no bodies")
    q = selig_order(design.bodies[body].points)
    if normalize:
        le = q[int(np.argmin(q[:, 0]))]
        chord = float(np.ptp(q[:, 0])) or 1.0
        q = (q - le) / chord
    name = " ".join(design.name.split()) or "Aerolume airfoil"
    lines = [name] + [f" {x: .7f} {y: .7f}" for x, y in q.tolist()]
    return write_text(path, "\n".join(lines) + "\n")
