"""Glyphs of the placed physics in the 3D view (DESIGN.md 10): fans, heaters, emitters, jets and vents.

A fan in a 3D flow is an actuator disc a few cells thick; nothing in the
velocity field marks where it is until its jet has formed, so the viewer
draws the devices themselves, in the same visual language as the 2D field
view (:func:`aerolume.app.widgets.draw_tools.paint_placements`):

* **fan / propeller** -- a translucent disc of its true diameter and thickness
  in the accent colour, a rim, and an arrow along its blowing direction
  (length 0.8 diameters): the thing users look for is "which way does it blow";
* **heater** -- its true box or sphere, translucent, in the warm (warn)
  colour, since a heat source is a region, not a point;
* **emitter** -- where the smoke or the particles start: a small sphere for a
  point, a thin tube with nozzle beads for a rake / line, a translucent plate
  for a rect / plane / disc, in the smoke's own colour (that is how a viewer
  matches a filament to its source, as in the 2D legend);
* **jet / vent** -- a disc of the port's size with an arrow out of it (a jet)
  or into it (a vent).

Everything here is numpy only (no VTK, no Qt): :func:`placed_glyphs` turns a
:class:`~aerolume.model.Simulation` into :class:`Glyph` meshes in the wind
frame, which ``Scene.add_mesh`` draws. Body-frame items turn with the geometry
by the same attitude rotation as the meshes (DESIGN.md 7.2,
:func:`attitude_matrix`, identical to ``solvers.ns3d.geometry.rotation_matrix``);
wind-frame items stay where the tunnel has them. Glyph sizes that would be
invisible (a 2 mm nozzle beside a 4 m car) are floored at a fraction of the
scene's length scale, ``scale``; true sizes win whenever they are larger.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GLYPH_KINDS", "Glyph", "arrow_mesh", "attitude_matrix", "box_mesh", "cylinder_mesh", "disc_basis",
    "merge_meshes", "placed_glyphs", "sphere_mesh",
]

#: Glyph kinds, in drawing order (translucent volumes first, so arrows read over them).
GLYPH_KINDS = ("heater", "fan", "port", "vent", "emitter")

#: Smallest drawn size as a fraction of the scene length scale (bodies' largest extent).
MIN_FRACTION = 0.012
#: Arrow length per fan / port diameter.
ARROW_PER_DIAMETER = 0.8


@dataclass
class Glyph:
    """One device as a triangle mesh in the wind frame (metres).

    ``key`` names the item in its simulation list (``"fans:0"``, the 2D
    :class:`~aerolume.app.widgets.draw_tools.Placement` convention), ``role``
    the theme colour it is drawn in ("accent", "warn" or "own" for the smoke's
    ``color``), ``opacity`` its translucency. ``parts`` keeps the pieces apart
    (disc, rim, arrow), each with its own opacity, so a translucent disc does
    not hide its own arrow.
    """
    key: str
    kind: str
    name: str
    role: str
    color: str = ""
    enabled: bool = True
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    parts: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)

    @property
    def vertices(self) -> np.ndarray:
        return merge_meshes([(v, f) for v, f, _a in self.parts])[0]

    @property
    def faces(self) -> np.ndarray:
        return merge_meshes([(v, f) for v, f, _a in self.parts])[1]

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        v = self.vertices
        if not len(v):
            c = self.centre
            return (c[0], c[0], c[1], c[1], c[2], c[2])
        lo, hi = v.min(axis=0), v.max(axis=0)
        return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def attitude_matrix(alpha_deg: float, yaw_deg: float) -> np.ndarray:
    """Design -> wind rotation of DESIGN.md 7.2: yaw about +z by -yaw, then pitch about +y by +alpha.

    The same matrix as ``solvers.ns3d.geometry.rotation_matrix`` and
    ``simulation.plan3d.rotation3`` (a test pins them together); repeated here
    so drawing a glyph never imports torch.
    """
    a, b = math.radians(float(alpha_deg)), math.radians(-float(yaw_deg))
    ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    return ry @ rz


def _vec3(v, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if v is None:
        return np.asarray(default, dtype=np.float64)
    a = [float(c) for c in tuple(v)[:3]]
    while len(a) < 3:
        a.append(0.0)
    return np.asarray(a, dtype=np.float64)


def _unit(v, default=(1.0, 0.0, 0.0)) -> np.ndarray:
    a = _vec3(v, default)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-15 else np.asarray(default, dtype=np.float64)


def disc_basis(normal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, e1, e2): the unit normal and two unit vectors spanning the disc's plane (right-handed)."""
    n = _unit(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


# ---------------------------------------------------------------------------
# primitive meshes (closed, outward-facing triangles)
# ---------------------------------------------------------------------------

def merge_meshes(parts: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """One (vertices, faces) from several, faces re-indexed."""
    vs, fs, off = [], [], 0
    for v, f in parts:
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(vs), np.vstack(fs)


def cylinder_mesh(a, b, radius: float, k: int = 24, *, caps: bool = True,
                  radius_b: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A closed cylinder (a cone frustum with ``radius_b``) from point ``a`` to ``b``."""
    a, b = _vec3(a), _vec3(b)
    axis = b - a
    length = float(np.linalg.norm(axis))
    _n, e1, e2 = disc_basis(axis if length > 1e-15 else (1.0, 0.0, 0.0))
    rb = float(radius if radius_b is None else radius_b)
    t = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
    ring = np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2
    v = np.vstack([a + radius * ring, b + rb * ring])
    faces = []
    for i in range(k):
        j = (i + 1) % k
        faces += [(i, j, k + j), (i, k + j, k + i)]
    if caps:
        ca, cb = len(v), len(v) + 1
        v = np.vstack([v, a, b])
        for i in range(k):
            j = (i + 1) % k
            faces += [(ca, j, i), (cb, k + i, k + j)]
    return v, np.asarray(faces, dtype=np.int64)


def arrow_mesh(start, direction, length: float, radius: float, k: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """A shaft and a cone head of total ``length`` from ``start`` along ``direction``."""
    d = _unit(direction)
    s = _vec3(start)
    head = min(0.35 * length, 6.0 * radius)
    neck = s + d * (length - head)
    shaft = cylinder_mesh(s, neck, radius, k)
    cone = cylinder_mesh(neck, s + d * length, 2.6 * radius, k, radius_b=1e-6 * radius)
    return merge_meshes([shaft, cone])


def box_mesh(centre, size) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box of ``size`` (sx, sy, sz) centred at ``centre``."""
    c = _vec3(centre)
    s = _vec3(size, (1.0, 1.0, 1.0))
    h = 0.5 * np.abs(s)
    corners = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=np.float64)
    v = c + corners * h
    q = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    f = []
    for a, b, cc, d in q:
        f += [(a, b, cc), (a, cc, d)]
    return v, np.asarray(f, dtype=np.int64)


def sphere_mesh(centre, radius: float, n_lat: int = 12, n_lon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A UV sphere."""
    c = _vec3(centre)
    v = [c + (0.0, 0.0, radius)]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2.0 * math.pi * j / n_lon
            v.append(c + radius * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)]))
    v.append(c - (0.0, 0.0, radius))
    f = []
    last = len(v) - 1
    for j in range(n_lon):
        f.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 2):
        r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            jj = (j + 1) % n_lon
            f += [(r0 + j, r1 + j, r1 + jj), (r0 + j, r1 + jj, r0 + jj)]
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        f.append((last, base + (j + 1) % n_lon, base + j))
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def _plate(centre, normal, e1_half: float, e2_half: float, thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """A thin rectangular plate (a box in the disc basis of ``normal``)."""
    n, e1, e2 = disc_basis(normal)
    c = _vec3(centre)
    corners = []
    for z in (-0.5, 0.5):
        for y in (-1, 1):
            for x in (-1, 1):
                corners.append(c + x * e1_half * e1 + y * e2_half * e2 + z * thickness * n)
    v = np.asarray(corners)
    q = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    f = []
    for a, b, cc, d in q:
        f += [(a, b, cc), (a, cc, d)]
    return v, np.asarray(f, dtype=np.int64)


# ---------------------------------------------------------------------------
# a simulation's devices
# ---------------------------------------------------------------------------

#: (Simulation attribute, glyph kind) of every placed list.
PLACED_LISTS = (("heat_sources", "heater"), ("fans", "fan"), ("ports", "port"), ("emitters", "emitter"))


def placed_glyphs(sim, design=None, *, alpha_deg: float | None = None, yaw_deg: float | None = None,
                  scale: float = 1.0, include_disabled: bool = True) -> list[Glyph]:
    """Every heater, fan, port and emitter of a 3D simulation as wind-frame :class:`Glyph` meshes.

    ``alpha_deg`` / ``yaw_deg`` default to the simulation's conditions (explore
    passes the slider values); body-frame items turn about the design's
    ``moment_ref3``. ``scale`` is the scene's length scale (the bodies' largest
    extent), which floors glyph sizes (:data:`MIN_FRACTION`). Disabled items
    are included (drawn faint by the viewer) unless ``include_disabled`` is off.
    """
    if sim is None:
        return []
    cond = getattr(sim, "conditions", None)
    alpha = float(getattr(cond, "alpha_deg", 0.0) or 0.0) if alpha_deg is None else float(alpha_deg)
    yaw = float(getattr(cond, "yaw_deg", 0.0) or 0.0) if yaw_deg is None else float(yaw_deg)
    rot = attitude_matrix(alpha, yaw)
    ref = _vec3(getattr(design, "moment_ref3", None) if design is not None else None)
    floor = MIN_FRACTION * max(float(scale), 1e-9)

    def point(p, frame: str) -> np.ndarray:
        q = _vec3(p)
        return q if frame == "wind" else (q - ref) @ rot.T + ref

    def turn(d, frame: str) -> np.ndarray:
        q = _unit(d)
        return q if frame == "wind" else q @ rot.T

    out: list[Glyph] = []
    for attr, kind in PLACED_LISTS:
        for i, item in enumerate(getattr(sim, attr, None) or []):
            enabled = bool(getattr(item, "enabled", True))
            if not enabled and not include_disabled:
                continue
            frame = str(getattr(item, "frame", "wind"))
            c = point(getattr(item, "position", (0.0, 0.0, 0.0)), frame)
            g = Glyph(key=f"{attr}:{i}", kind=kind, name=str(getattr(item, "name", kind.capitalize())),
                      role="accent", enabled=enabled, centre=tuple(float(x) for x in c))
            if kind == "fan":
                _fan_parts(g, item, c, turn(getattr(item, "direction", (1.0, 0.0, 0.0)), frame), floor)
            elif kind == "heater":
                g.role = "warn"
                _heater_parts(g, item, c, rot if frame != "wind" else None, floor)
            elif kind == "port":
                if str(getattr(item, "kind", "jet")) == "vent":
                    g.kind = "vent"
                _port_parts(g, item, c, turn(getattr(item, "direction", (1.0, 0.0, 0.0)), frame), floor)
            else:
                g.role = "own"
                g.color = str(getattr(item, "color", "") or "")
                _emitter_parts(g, item, c, frame, point, turn, floor)
            if g.parts:
                out.append(g)
    out.sort(key=lambda g: GLYPH_KINDS.index(g.kind) if g.kind in GLYPH_KINDS else len(GLYPH_KINDS))
    return out


def _fan_parts(g: Glyph, fan, c: np.ndarray, d: np.ndarray, floor: float) -> None:
    dia = max(float(getattr(fan, "diameter", 0.0) or 0.0), 4.0 * floor)
    thick = getattr(fan, "thickness", None)
    thick = max(float(thick) if thick else dia / 8.0, floor)
    r = 0.5 * dia
    a, b = c - 0.5 * thick * d, c + 0.5 * thick * d
    g.parts.append((*cylinder_mesh(a, b, r, 40), 0.28))                         # the disc, true size
    rim = 0.04 * dia
    ring = []
    _n, e1, e2 = disc_basis(d)
    k = 40
    for j in range(k):                                                        # a rim of short tubes
        t0, t1 = 2 * math.pi * j / k, 2 * math.pi * (j + 1) / k
        p0 = c + r * (math.cos(t0) * e1 + math.sin(t0) * e2)
        p1 = c + r * (math.cos(t1) * e1 + math.sin(t1) * e2)
        ring.append(cylinder_mesh(p0, p1, 0.5 * rim, 8, caps=False))
    g.parts.append((*merge_meshes(ring), 0.9))
    g.parts.append((*arrow_mesh(c, d, ARROW_PER_DIAMETER * dia, max(0.035 * dia, 0.5 * floor)), 1.0))


def _heater_parts(g: Glyph, src, c: np.ndarray, rot: np.ndarray | None, floor: float) -> None:
    kind = str(getattr(src, "kind", "box"))
    size = tuple(float(s) for s in (getattr(src, "size", ()) or ()))
    if kind in ("sphere", "disc"):
        r = max(size[0] if size else 0.0, 2.0 * floor)
        g.parts.append((*sphere_mesh(c, r), 0.38))
        return
    s = np.array([size[k] if len(size) > k else (size[0] if size else 0.0) for k in range(3)])
    s = np.maximum(s, 2.0 * floor)
    v, f = box_mesh((0.0, 0.0, 0.0), s)
    v = v @ rot.T + c if rot is not None else v + c                         # a body-frame box pitches with the car
    g.parts.append((v, f, 0.38))


def _port_parts(g: Glyph, port, c: np.ndarray, d: np.ndarray, floor: float) -> None:
    dia = max(float(getattr(port, "size", 0.0) or 0.0), 3.0 * floor)
    g.parts.append((*cylinder_mesh(c - 0.03 * dia * d, c + 0.03 * dia * d, 0.5 * dia, 32), 0.45))
    length = ARROW_PER_DIAMETER * dia * 1.5
    rad = max(0.05 * dia, 0.5 * floor)
    if g.kind == "vent":                                                      # the flow goes into a vent
        g.parts.append((*arrow_mesh(c + d * length, -d, length, rad), 1.0))
    else:
        g.parts.append((*arrow_mesh(c, d, length, rad), 1.0))


def _emitter_parts(g: Glyph, em, c: np.ndarray, frame: str, point, turn, floor: float) -> None:
    kind = str(getattr(em, "kind", "point"))
    nozzle = max(float(getattr(em, "radius", 0.0) or 0.0), floor)
    end = getattr(em, "end", None)
    if kind in ("line", "rake") and end is not None:
        e = point(end, frame)
        g.parts.append((*cylinder_mesh(c, e, 0.45 * nozzle, 10), 0.95))
        n = int(getattr(em, "count", 0) or 0) if kind == "rake" else 0
        beads = [sphere_mesh(c + (e - c) * (k / (n - 1)), 1.1 * nozzle, 6, 10) for k in range(n)] if n > 1 else []
        beads += [sphere_mesh(c, 1.3 * nozzle, 6, 10), sphere_mesh(e, 1.3 * nozzle, 6, 10)]
        g.parts.append((*merge_meshes(beads), 1.0))
        return
    size = tuple(float(s) for s in (getattr(em, "size", ()) or ()))
    if kind in ("rect", "plane", "disc"):
        normal = turn(getattr(em, "normal", None) or (1.0, 0.0, 0.0), frame)
        if kind == "disc":
            r = max(size[0] if size else 0.0, 2.0 * floor)
            g.parts.append((*cylinder_mesh(c - 0.5 * floor * normal, c + 0.5 * floor * normal, r, 32), 0.35))
        elif kind == "plane":
            a = max(size[0] if size else 0.0, 2.0 * floor)
            b = max(size[1] if len(size) > 1 else a, 2.0 * floor)
            g.parts.append((*_plate(c, normal, 0.5 * a, 0.5 * b, floor), 0.35))
        else:                                                                # an axis-aligned box of size
            s = np.array([size[k] if len(size) > k else 0.0 for k in range(3)])
            g.parts.append((*box_mesh(c, np.maximum(s, 2.0 * floor)), 0.35))
        return
    g.parts.append((*sphere_mesh(c, 1.6 * nozzle, 8, 14), 1.0))
