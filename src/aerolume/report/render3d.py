"""Off-screen 3D pictures for report figures, rendered through :mod:`aerolume.viz3d`.

A report figure must look like the 3D viewer in the app (same colour maps, same
neutral body grey, same camera presets), so the pictures come from the very
:class:`aerolume.viz3d.Scene` the viewer paints, rendered off-screen. What this
module adds is what a *printed* figure needs and a live viewport does not:

* **Several views of one scene** (iso, front, side, top) with one shared colour
  range, so a reader can compare the front and the top of a car on one scale.
  The orthographic views use a parallel projection, as engineering drawings do.
* **No colour bar in the pixels.** The range and the colour map are handed back
  as a :class:`ColorSpec`; :func:`aerolume.viz.figures.fig_views3d` draws the
  colour bar with matplotlib, in the report's font and at print resolution.
* **Tight crops.** Each view is cropped to its content plus a small margin, so
  a long car does not sit in a frame of empty background.
* **Plane sections of a mesh** (:func:`plane_section`): the exact outline of a
  body in a slice plane (for the slice plots) and the surface values along it
  (the centre-line pressure distribution). Pure numpy.

Rendering needs an OpenGL context. Where there is none (a headless server
without OSMesa), :class:`RenderUnavailable` is raised with the reason; the PDF
and HTML writers turn it into a short note in place of the figure.

``vtk`` is imported only inside :mod:`aerolume.viz3d` (DESIGN.md 7.3), and only
when a picture is actually drawn.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["VIEW_TITLES", "ColorSpec", "RenderUnavailable", "Section", "Views", "plane_section", "render_views"]

#: Captions of the camera presets in a multi-view figure. "Side" and "Top" put the flow left
#: to right; "Front" looks downstream at the frontal area (viz3d.camera.PRESETS).
VIEW_TITLES = {
    "iso": "Isometric",
    "front": "Front (looking downstream)",
    "back": "Rear (looking upstream)",
    "side": "Side (flow left to right)",
    "top": "Top (flow left to right)",
    "bottom": "Underside",
}

#: Views drawn in parallel projection (engineering-drawing views); "iso" keeps perspective.
_ORTHO = frozenset({"front", "back", "side", "top", "bottom"})


class RenderUnavailable(RuntimeError):
    """3D rendering is not possible here (no OpenGL context); the message says why."""


@dataclass
class ColorSpec:
    """The colour mapping of a rendered layer, for a colour bar drawn outside the picture."""
    cmap: str
    vmin: float
    vmax: float
    label: str
    unit: str = ""
    extend: str = "neither"         # matplotlib colorbar ``extend``: values beyond the range

    @property
    def title(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label


@dataclass
class Views:
    """Rendered views of one scene.

    ``images`` are RGB arrays (row 0 = top), one per name in ``names`` (camera presets),
    with display ``titles``. ``extents`` holds, per view, the world extent of the image as
    an imshow extent (left, right, bottom, top) in metres along the view's horizontal and
    vertical axes, for the orthographic views (None for perspective ones): orthographic
    views are cropped to the bodies plus a fixed margin, so a figure can draw them at one
    common scale, aligned like an engineering drawing. ``axes`` names those world axes
    per view, e.g. ("x", "z") for the side view.
    """
    images: list[np.ndarray]
    titles: list[str]
    names: list[str]
    extents: list[tuple[float, float, float, float] | None]
    axes: list[tuple[str, str] | None]
    color: ColorSpec | None = None
    bounds: tuple[float, float, float, float, float, float] | None = None


def _crop(img: np.ndarray, background: Sequence[float], pad: float = 0.04) -> np.ndarray:
    """Crop to the pixels that differ from the background, plus a margin of ``pad`` x size."""
    bg = np.round(np.asarray(background, float) * 255.0)
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 6
    rows, cols = np.flatnonzero(diff.any(axis=1)), np.flatnonzero(diff.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return img
    h, w = img.shape[:2]
    m = round(pad * max(rows[-1] - rows[0], cols[-1] - cols[0]))
    r0, r1 = max(0, rows[0] - m), min(h, rows[-1] + 1 + m)
    c0, c1 = max(0, cols[0] - m), min(w, cols[-1] + 1 + m)
    return np.ascontiguousarray(img[r0:r1, c0:c1])


_NAMES = "xyz"


def _ortho_crop(scene: Any, img: np.ndarray, box: Sequence[float], margin: float
                ) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[str, str]]:
    """Crop an orthographic render to ``box`` grown by ``margin`` metres; return the world extent.

    The world axis along the image's columns (and rows) is found by projecting unit steps
    from the box centre, so the mapping is exact whatever the preset's handedness (the front
    view, for one, has +y to the left).
    """
    x0, x1, y0, y1, z0, z1 = (float(b) for b in box)
    c = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)])
    lo = np.array([x0, y0, z0]) - margin
    hi = np.array([x1, y1, z1]) + margin
    pc = scene.project(c[None])[0]
    steps = scene.project(c[None] + np.eye(3)) - pc                     # pixels per metre along x, y, z
    kh = int(np.argmax(np.abs(steps[:, 0])))
    kv = int(np.argmax(np.abs(steps[:, 1])))
    sh, sv = float(steps[kh, 0]), float(steps[kv, 1])
    # pixel columns / rows of the grown box edges
    cols = sorted(pc[0] + sh * (np.array([lo[kh], hi[kh]]) - c[kh]))
    rows = sorted(pc[1] + sv * (np.array([lo[kv], hi[kv]]) - c[kv]))
    h, w = img.shape[:2]
    c0, c1 = max(0, math.floor(cols[0])), min(w, math.ceil(cols[1]))
    r0, r1 = max(0, math.floor(rows[0])), min(h, math.ceil(rows[1]))
    out = np.ascontiguousarray(img[r0:r1, c0:c1])

    def world_h(px: float) -> float:
        return float(c[kh] + (px - pc[0]) / sh)

    def world_v(py: float) -> float:
        return float(c[kv] + (py - pc[1]) / sv)
    extent = (world_h(c0), world_h(c1), world_v(r1), world_v(r0))       # imshow: left, right, bottom, top
    return out, extent, (_NAMES[kh], _NAMES[kv])


def render_views(build: Callable[[Any], str | None], views: Sequence[str] = ("iso", "front", "side", "top"), *,
                 size: int = 1700, theme: str = "light", scale: float = 2.0, triad: Sequence[str] = (),
                 ground: float | None = None, focus: Sequence[float] | None = None,
                 margin: float = 0.05, orbit: tuple[float, float] | None = None) -> Views:
    """Render one scene from several camera presets.

    ``build(scene)`` adds the layers and returns the handle of the colour-mapped layer whose
    range the colour bar describes (or None). Every view is rendered about ``size`` pixels on
    its longer side. Perspective views ("iso") are cropped to their content; orthographic
    views are cropped to the bodies (or ``focus``, a box (x0, x1, y0, y1, z0, z1)) grown by
    ``margin`` x their largest dimension and come with their world extent, see :class:`Views`.
    ``ground`` (a height z) adds a floor patch around the bodies, drawn in the perspective
    views only: in an orthographic view the floor is a line, which the figure draws itself.
    ``scale`` grows the overlay line widths with the resolution (as for a high-DPI screen).
    The axis triad is shown in the ``triad`` views only (none by default: the view titles name
    the directions, and a corner triad would widen the content crop). ``orbit`` = (azimuth,
    elevation) degrees turns the perspective views away from their preset before the final
    fit (a long wing reads better from nearer the front). Raises :class:`RenderUnavailable`
    without an OpenGL context.
    """
    from ..viz3d import Scene
    from ..viz3d.colors import field_info

    scene = Scene(theme, axes=False)
    try:
        if not scene.available:
            raise RenderUnavailable(scene.error or "3D rendering is not available on this machine")
        handle = build(scene)
        if focus is not None:
            scene.set_focus(tuple(float(b) for b in focus))
        box = scene.focus_bounds()
        lx, ly, lz = box[1] - box[0], box[3] - box[2], box[5] - box[4]
        big = max(lx, ly, lz, 1e-9)
        g = None
        iso_focus = tuple(focus) if focus is not None else None
        if ground is not None:
            # a floor plate a little larger than the bodies, framed whole in the perspective
            # views: a plate with a visible edge reads as deliberate, a floor cut off by the
            # image border as an accident
            pad = 0.12 * big
            plate = (box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad)
            g = scene.add_ground(float(ground), plate)
            iso_focus = (*plate, min(box[4], float(ground)), box[5])
        color = None
        if handle is not None and scene.scalar_info(handle) is not None:
            info = scene.scalar_info(handle)
            scene.add_colorbar(layer=handle)
            spec = scene.colorbar_spec()          # tells whether values run past the range
            scene.remove_colorbar()
            fi = field_info(info["field"])
            lo, hi = spec.extend if spec is not None else (False, False)
            extend = {(False, False): "neither", (True, False): "min", (False, True): "max",
                      (True, True): "both"}[(bool(lo), bool(hi))]
            color = ColorSpec(str(info["cmap"]), float(info["vmin"]), float(info["vmax"]), fi.label, fi.unit,
                              extend)
        images, titles, extents, axes = [], [], [], []
        dims = {"x": lx, "y": ly, "z": lz}
        plane = {"front": ("y", "z"), "back": ("y", "z"), "side": ("x", "z"), "top": ("x", "y"),
                 "bottom": ("x", "y")}
        obox = list(box)
        if ground is not None:
            obox[4] = min(obox[4], float(ground))
        m = margin * big
        grown = (obox[0] - m, obox[1] + m, obox[2] - m, obox[3] + m, obox[4] - m, obox[5] + m)
        for v in views:
            ortho = v in _ORTHO
            if g is not None:
                scene.set_visible(g, not ortho)
            # an orthographic view frames the whole grown box, so the crop below never runs off the image
            scene.set_focus(grown if ortho else iso_focus)
            scene.set_parallel(ortho)
            scene.set_camera(v, fit=True)
            if orbit is not None and not ortho:
                scene.orbit(*orbit)
                scene.fit()
            scene.add_axes(v in triad)
            if ortho:
                a, b = plane[v]
                aspect = min(max((dims[a] + 0.1 * big) / max(dims[b] + 0.1 * big, 1e-9), 0.25), 6.0)
                w, h = (size, round(size / aspect)) if aspect >= 1 else (round(size * aspect), size)
            else:
                w, h = size, round(size * 0.68)
            img = scene.render(int(w), int(h), scale=scale)
            if ortho:
                img, ext, ax = _ortho_crop(scene, img, obox, margin * big)
                extents.append(ext)
                axes.append(ax)
            else:
                images_bg = scene.theme.background
                img = _crop(img, images_bg)
                extents.append(None)
                axes.append(None)
            images.append(img)
            titles.append(VIEW_TITLES.get(v, v))
        return Views(images, titles, list(views), extents, axes, color, tuple(float(b) for b in obox))
    finally:
        scene.close()


# ---------------------------------------------------------------------------
# plane sections of a triangle mesh
# ---------------------------------------------------------------------------

#: For a plane normal: the two in-plane axes (column, row) in (x, y, z), as in ``SliceData``.
PLANE_AXES = {"y": (0, 2), "z": (0, 1), "x": (1, 2)}
_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class Section:
    """One polyline of a plane section: in-plane points (k, 2) [m], optional values (k, ...)."""
    points: np.ndarray
    values: np.ndarray | None
    closed: bool


def plane_section(vertices: np.ndarray, faces: np.ndarray, axis: str, position: float,
                  values: np.ndarray | None = None) -> list[Section]:
    """Intersect a triangle mesh with the plane ``axis = position``.

    Returns polylines in the plane's (column, row) coordinates (``PLANE_AXES``; e.g. (x, z)
    for a "y" plane), with per-vertex ``values`` ((N,) or (N, k)) interpolated linearly along
    the cut edges. Segments are chained through the mesh edges they cross, and endpoints that
    coincide to 1e-7 of the mesh size are merged as well, so a triangle soup (an STL without
    shared vertices) still gives closed loops. Vertices lying exactly on the plane are nudged
    by a negligible amount instead of creating zero-length segments.
    """
    v = np.asarray(vertices, float).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        return []
    k = _AXIS[axis]
    size = float(np.ptp(v, axis=0).max()) or 1.0
    d = v[:, k] - float(position)
    d = np.where(d == 0.0, 1e-9 * size, d)
    side = d > 0
    s = side[f]
    cut = s.any(axis=1) & ~s.all(axis=1)
    fc = f[cut]
    if len(fc) == 0:
        return []
    # the two crossing edges of each cut face (exactly two, since no vertex is on the plane)
    e = np.stack([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]], axis=1)            # (m, 3, 2)
    crosses = side[e[..., 0]] != side[e[..., 1]]                                    # (m, 3)
    ends = e[crosses].reshape(-1, 2, 2)                                             # (m, 2 edges, 2 verts)
    a, b = ends[..., 0], ends[..., 1]
    t = d[a] / (d[a] - d[b])
    pts = v[a] + t[..., None] * (v[b] - v[a])                                       # (m, 2, 3)
    ci, ri = PLANE_AXES[axis]
    p2 = pts[..., [ci, ri]].reshape(-1, 2)
    vals = None
    if values is not None:
        val = np.asarray(values, float)
        val = val.reshape(len(v), -1)
        vals = (val[a] + t[..., None] * (val[b] - val[a])).reshape(len(p2), -1)
    # node ids: merge coincident points (shared edges give identical points; a soup gives near ones)
    q = np.round(p2 / (1e-7 * size)).astype(np.int64)
    _, node = np.unique(q, axis=0, return_inverse=True)
    node = node.reshape(-1, 2)
    keep = node[:, 0] != node[:, 1]
    node = node[keep]
    seg_pts = p2.reshape(-1, 2, 2)[keep]
    seg_vals = None if vals is None else vals.reshape(-1, 2, vals.shape[1])[keep]
    n_nodes = int(node.max()) + 1 if len(node) else 0
    coord = np.zeros((n_nodes, 2))
    coord[node.ravel()] = seg_pts.reshape(-1, 2)
    nval = None
    if seg_vals is not None:
        nval = np.zeros((n_nodes, seg_vals.shape[2]))
        nval[node.ravel()] = seg_vals.reshape(-1, seg_vals.shape[2])
    adj: dict[int, list[int]] = {}
    for i, j in node.tolist():
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    used: set[tuple[int, int]] = set()

    def walk(start: int) -> tuple[list[int], bool]:
        path, prev, cur = [start], -1, start
        while True:
            nxt = [n for n in adj[cur] if n != prev and (min(cur, n), max(cur, n)) not in used]
            if not nxt:
                return path, False
            n = nxt[0]
            used.add((min(cur, n), max(cur, n)))
            if n == start:
                return path, True
            path.append(n)
            prev, cur = cur, n

    out: list[Section] = []
    # open chains first, from their free ends, so they are not split in the middle
    order = sorted(adj, key=lambda n: len(adj[n]) != 1)
    for n0 in order:
        if all((min(n0, m), max(n0, m)) in used for m in adj[n0]):
            continue
        path, closed = walk(n0)
        if len(path) < 2:
            continue
        idx = np.asarray(path)
        vv = None
        if nval is not None:
            vv = nval[idx]
            if values is not None and np.asarray(values).ndim == 1:
                vv = vv[:, 0]
        out.append(Section(coord[idx], vv, closed))
    return out


def polygon_area(p: np.ndarray) -> float:
    """Signed area of a closed polyline (counter-clockwise positive)."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def nice_round(x: float) -> float:
    """A 1/2/5 step of the order of ``x`` (> 0)."""
    if not (x > 0 and math.isfinite(x)):
        return 1.0
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    return (1.0 if m < 1.5 else 2.0 if m < 3.5 else 5.0) * 10 ** e
