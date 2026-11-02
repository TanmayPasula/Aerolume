"""Closed reference shapes: sphere, box, cylinder, and a generic sedan (watertight, outward winding).

These are the 3D validation and template geometries (the sphere is the solver's
main benchmark, DESIGN.md 7.12), so they are built to be exactly closed and
consistently wound: every edge is shared by two faces, normals point out, and
the signed volume is positive. Vertices are shared (no duplicates).

:func:`sedan` is the car of the bundled 3D example project
(``examples/make_example_projects.py``). It is generated rather than shipped as
a file so that the example contains no third-party model and can be rebuilt
from source.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..model import Mesh3D

__all__ = ["box", "cylinder", "sedan", "sphere"]

_AXES = {"x": 0, "y": 1, "z": 2}


def sphere(d: float = 1.0, n: int = 48, center: Sequence[float] = (0.0, 0.0, 0.0), name: str = "Sphere"
           ) -> Mesh3D:
    """UV sphere of diameter ``d`` with ``n`` segments around and ``n // 2`` rings (poles on z).

    ``n`` should be a multiple of 4 so a meridian lies in each coordinate
    plane: the outline seen along x or y is then exactly the n-gon inscribed in
    the circle (frontal area ``d² n sin(2 pi / n) / 8``, 0.3 % below pi d² / 4 at n = 48).
    """
    n = max(int(n), 4)
    m = max(n // 2, 2)
    r = 0.5 * float(d)
    th = np.linspace(0.0, math.pi, m + 1)[1:-1]                 # polar angle of the inner rings
    ph = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    st, ct = np.sin(th)[:, None], np.cos(th)[:, None]
    ring = np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct * np.ones_like(ph)], -1).reshape(-1, 3)
    verts = np.vstack([[0.0, 0.0, r], ring, [0.0, 0.0, -r]])
    top, bot = 0, len(verts) - 1
    faces = []
    j = np.arange(n)
    jn = (j + 1) % n
    faces.append(np.stack([np.full(n, top), 1 + j, 1 + jn], 1))     # cap: outward (ccw from above)
    for i in range(m - 2):
        a, b = 1 + i * n + j, 1 + i * n + jn
        c, e = 1 + (i + 1) * n + j, 1 + (i + 1) * n + jn
        faces.append(np.stack([a, c, e], 1))
        faces.append(np.stack([a, e, b], 1))
    last = 1 + (m - 2) * n
    faces.append(np.stack([np.full(n, bot), last + jn, last + j], 1))
    verts = verts + np.asarray(center, dtype=float)
    verts[np.abs(verts) < 1e-15 * max(r, 1e-300)] = 0.0
    return Mesh3D(name=name, vertices=verts, faces=np.concatenate(faces))


def box(lx: float = 1.0, ly: float = 1.0, lz: float = 1.0, center: Sequence[float] = (0.0, 0.0, 0.0),
        name: str = "Box") -> Mesh3D:
    """Axis-aligned box of size lx x ly x lz (12 triangles, outward winding)."""
    hx, hy, hz = 0.5 * float(lx), 0.5 * float(ly), 0.5 * float(lz)
    v = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                  [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]) + np.asarray(center, float)
    f = np.array([[0, 2, 1], [0, 3, 2],      # bottom (-z)
                  [4, 5, 6], [4, 6, 7],      # top (+z)
                  [0, 1, 5], [0, 5, 4],      # -y
                  [2, 3, 7], [2, 7, 6],      # +y
                  [1, 2, 6], [1, 6, 5],      # +x
                  [3, 0, 4], [3, 4, 7]])     # -x
    return Mesh3D(name=name, vertices=v, faces=f)


def cylinder(d: float = 1.0, length: float = 1.0, axis: str = "z", n: int = 48,
             center: Sequence[float] = (0.0, 0.0, 0.0), capped: bool = True, name: str = "Cylinder"
             ) -> Mesh3D:
    """Circular cylinder of diameter ``d`` and ``length`` along ``axis``, centred on ``center``.

    ``n`` segments around; flat end caps are fans around a centre vertex
    (``capped=False`` leaves an open tube, which is not watertight).
    """
    n = max(int(n), 3)
    r, hl = 0.5 * float(d), 0.5 * float(length)
    ph = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    c, s = r * np.cos(ph), r * np.sin(ph)
    # build along z, then permute so z becomes the requested axis (a proper rotation)
    v = np.vstack([np.stack([c, s, np.full(n, -hl)], 1), np.stack([c, s, np.full(n, hl)], 1)])
    j = np.arange(n)
    jn = (j + 1) % n
    faces = [np.stack([j, jn, n + jn], 1), np.stack([j, n + jn, n + j], 1)]
    if capped:
        v = np.vstack([v, [[0.0, 0.0, -hl], [0.0, 0.0, hl]]])
        cb, ctp = 2 * n, 2 * n + 1
        faces.append(np.stack([np.full(n, cb), jn, j], 1))
        faces.append(np.stack([np.full(n, ctp), n + j, n + jn], 1))
    if axis not in _AXES:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    # cyclic permutations (proper rotations, winding stays outward): z -> requested axis
    cols = {"z": [0, 1, 2], "x": [2, 0, 1], "y": [1, 2, 0]}[axis]
    out = v[:, cols] + np.asarray(center, dtype=float)
    out[np.abs(out) < 1e-15 * max(r, hl, 1e-300)] = 0.0
    return Mesh3D(name=name, vertices=out, faces=np.concatenate(faces))


def _smooth_step(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """0 below ``a``, 1 above ``b``, a C1 cubic in between (the body lines blend with it)."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _sedan_body(length: float, width: float, height: float, clearance: float, nx: int, nt: int) -> Mesh3D:
    """The lofted body of :func:`sedan`: superelliptic rings along x, closed by two fans."""
    s = length / 4.5                                    # the profile below is drawn for a 4.5 m car
    xs = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, nx)))    # 0..1, rings dense at nose and tail
    # half-width: a rounded nose, a tail that narrows by 18 %
    nose = np.sqrt(np.clip(1.0 - (1.0 - xs / 0.12) ** 2, 0.0, 1.0))
    half_w = 0.5 * width * np.where(xs < 0.12, nose, 1.0) * (1.0 - 0.18 * _smooth_step(xs, 0.80, 1.0))
    half_w = np.maximum(half_w, 0.02 * s)
    # roof line: bumper -> bonnet -> windscreen -> roof -> fastback
    top = s * (0.55 + 0.25 * _smooth_step(xs, 0.0, 0.08) + 0.06 * _smooth_step(xs, 0.08, 0.30)
               + 0.55 * _smooth_step(xs, 0.30, 0.48) - 0.18 * _smooth_step(xs, 0.66, 0.97))
    top = np.minimum(top, height)
    bottom = clearance + s * (0.12 * (1.0 - _smooth_step(xs, 0.0, 0.10)) + 0.06 * _smooth_step(xs, 0.90, 1.0))
    cabin = _smooth_step(xs, 0.30, 0.48) * (1.0 - _smooth_step(xs, 0.85, 1.0))   # narrower greenhouse
    t = np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)
    e = 0.28                                            # superellipse exponent: a boxy, rounded section
    cy = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** e
    cz = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** e
    rings = []
    for i in range(nx):
        zc, hz = 0.5 * (top[i] + bottom[i]), 0.5 * (top[i] - bottom[i])
        ww = half_w[i] * (1.0 - 0.16 * cabin[i] * np.clip(cz, 0.0, 1.0) ** 3)
        rings.append(np.stack([np.full(nt, xs[i] * length), ww * cy, zc + hz * cz], 1))
    v = np.concatenate(rings + [rings[0].mean(0, keepdims=True), rings[-1].mean(0, keepdims=True)])
    j, jn = np.arange(nt), (np.arange(nt) + 1) % nt
    faces = []
    for i in range(nx - 1):
        a, b = i * nt + j, i * nt + jn
        faces += [np.stack([a, b, a + nt], 1), np.stack([b, b + nt, a + nt], 1)]
    n0, n1 = nx * nt, nx * nt + 1
    last = (nx - 1) * nt
    faces += [np.stack([np.full(nt, n0), jn, j], 1), np.stack([np.full(nt, n1), last + j, last + jn], 1)]
    f = np.concatenate(faces)
    # the ring parameter runs y -> z (counter-clockwise seen from +x), so with x increasing
    # along the rings the quads above face outward; check it rather than trust the algebra
    p = v[f]
    if np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() < 0.0:
        f = f[:, ::-1]
    return Mesh3D(name="body", vertices=v, faces=f)


def sedan(length: float = 4.5, *, wheels: bool = True, nx: int = 72, nt: int = 64) -> list[Mesh3D]:
    """A generic fastback saloon: a lofted body and (``wheels``) four cylinder wheels.

    Nose at x = 0, flow along +x, standing on the road z = 0, mirror-symmetric
    about y = 0 (so a 3D run can use the symmetry plane). At the default
    ``length`` of 4.5 m it is 1.82 m wide and 1.41 m high with 0.16 m ground
    clearance, the proportions of a mid-size car; other lengths scale every
    dimension. ``nx`` rings of ``nt`` points make the body (about 9,000 triangles
    by default: enough that the Draft grid, not the facets, limits the shape).

    The wheels are named "wheel_fl" ... "wheel_rr", so :func:`~aerolume.geometry3d.find_wheels3d`
    recognises them and a run on a moving road spins them; they reach into the
    wheel arches as on a real car. Every part is closed with outward winding.
    """
    if not (isinstance(length, (int, float)) and math.isfinite(length) and length > 0):
        raise ValueError("the sedan length must be a positive number of metres")
    s = float(length) / 4.5
    parts = [_sedan_body(float(length), 1.82 * s, 1.42 * s, 0.16 * s, max(int(nx), 8), max(int(nt), 8))]
    if wheels:
        r, width = 0.33 * s, 0.22 * s
        for name, x, y in (("wheel_fl", 0.95, -0.74), ("wheel_fr", 0.95, 0.74),
                           ("wheel_rl", 3.55, -0.74), ("wheel_rr", 3.55, 0.74)):
            parts.append(cylinder(2.0 * r, width, axis="y", n=32, center=(x * s, y * s, r), name=name))
    return parts
