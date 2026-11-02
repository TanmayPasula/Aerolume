"""Biot–Savart kernels for straight vortex filaments (vectorised numpy, chunked).

Two element types are enough for both finite-wing methods:

* a finite segment from ``a`` to ``b`` (bound vortices, the chordwise legs of
  vortex rings);
* a semi-infinite filament that starts at ``a`` and runs to infinity along a unit
  direction ``d`` (the steady wake, aligned with the free stream).

Formulas (unit circulation, right-hand rule along the filament direction):

    finite:  v = (r1 x r2) (|r1| + |r2|) / (4 pi |r1| |r2| (|r1||r2| + r1.r2))
    semi:    v = (d x r1) / (4 pi |r1| (|r1| - d.r1))

with ``r1 = p - a`` and ``r2 = p - b``. Both are the forms used by AVL and Katz &
Plotkin (Low-Speed Aerodynamics, 2nd ed., section 10.4.5): they stay finite on the
extension of the filament line (where the induced velocity is zero) and only blow
up on the filament itself, where a relative cut-off returns zero. That cut-off is
also what removes the self-induction of a segment at its own midpoint, which the
Kutta–Joukowski force evaluation relies on.

Prandtl–Glauert: the caller passes points already stretched by ``1/beta`` in x and
gets velocities in the stretched space; :func:`to_physical` converts them
(u / beta, v, w). Memory is bounded by evaluating the field points in chunks so
that a chunk never holds more than ~``CHUNK_ELEMENTS`` point-element pairs.
"""
from __future__ import annotations

import math

import numpy as np

FOUR_PI = 4.0 * math.pi
CHUNK_ELEMENTS = 2_000_000      # point x element pairs per chunk (~50 MB of float64 temporaries)
CUTOFF = 1e-6                   # relative distance below which a point counts as "on" a filament


def _chunks(n_points: int, n_elements: int) -> range:
    step = max(1, CHUNK_ELEMENTS // max(n_elements, 1))
    return range(0, n_points, step)


def segment_velocity(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Velocity at ``points`` (P, 3) of unit-strength segments ``a -> b`` (S, 3); returns (P, S, 3).

    Points closer to a segment's line than ``CUTOFF`` times its length (including
    its own midpoint) get zero from that segment.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)
    out = np.zeros((len(p), len(a), 3))
    if len(p) == 0 or len(a) == 0:
        return out
    r0sq = np.einsum("ij,ij->i", b - a, b - a)
    for s in _chunks(len(p), len(a)):
        q = p[s:s + max(1, CHUNK_ELEMENTS // len(a))]
        x1 = q[:, 0, None] - a[None, :, 0]
        y1 = q[:, 1, None] - a[None, :, 1]
        z1 = q[:, 2, None] - a[None, :, 2]
        x2 = q[:, 0, None] - b[None, :, 0]
        y2 = q[:, 1, None] - b[None, :, 1]
        z2 = q[:, 2, None] - b[None, :, 2]
        cx = y1 * z2 - z1 * y2
        cy = z1 * x2 - x1 * z2
        cz = x1 * y2 - y1 * x2
        n1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        n2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        n12 = n1 * n2
        den = n12 * (n12 + x1 * x2 + y1 * y2 + z1 * z2)
        csq = cx * cx + cy * cy + cz * cz
        # |r1 x r2| = |b - a| * distance to the line: zero inside the cut-off
        ok = (csq > (CUTOFF ** 2) * r0sq[None, :] ** 2) & (den > 0)
        f = np.zeros_like(den)
        np.divide(n1 + n2, FOUR_PI * den, out=f, where=ok)
        blk = out[s:s + len(q)]
        blk[..., 0] = cx * f
        blk[..., 1] = cy * f
        blk[..., 2] = cz * f
    return out


def semi_infinite_velocity(points: np.ndarray, a: np.ndarray, d: np.ndarray, *,
                           length_scale: float) -> np.ndarray:
    """Velocity at ``points`` (P, 3) of unit semi-infinite filaments from ``a`` (S, 3) along ``d``.

    ``d`` is one unit direction (3,) shared by all filaments or one per filament
    (S, 3). Points within ``CUTOFF * length_scale`` of a filament's line get zero
    from it. Returns (P, S, 3).
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    d = np.broadcast_to(np.asarray(d, dtype=float).reshape(-1, 3), a.shape)
    out = np.zeros((len(p), len(a), 3))
    if len(p) == 0 or len(a) == 0:
        return out
    lim = (CUTOFF * length_scale) ** 2
    for s in _chunks(len(p), len(a)):
        q = p[s:s + max(1, CHUNK_ELEMENTS // len(a))]
        r1 = q[:, None, :] - a[None, :, :]
        n1 = np.sqrt(np.einsum("ijk,ijk->ij", r1, r1))
        proj = np.einsum("ijk,jk->ij", r1, d)
        c = np.cross(np.broadcast_to(d[None, :, :], r1.shape), r1)
        csq = np.einsum("ijk,ijk->ij", c, c)
        den = n1 * (n1 - proj)
        ok = (csq > lim) & (den > 0)
        f = np.zeros_like(den)
        np.divide(1.0, FOUR_PI * den, out=f, where=ok)
        out[s:s + len(q)] = c * f[:, :, None]
    return out


def segment_velocity_sum(points: np.ndarray, a: np.ndarray, b: np.ndarray,
                         strength: np.ndarray) -> np.ndarray:
    """Total velocity (P, 3) of segments ``a -> b`` with circulations ``strength`` (S,).

    Same as ``segment_velocity(...) @ strength`` but contracted chunk by chunk, so the
    (P, S, 3) array never exists in full.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    g = np.asarray(strength, dtype=float).reshape(-1)
    out = np.zeros((len(p), 3))
    step = max(1, CHUNK_ELEMENTS // max(len(g), 1))
    for s in range(0, len(p), step):
        out[s:s + step] = np.einsum("psk,s->pk", segment_velocity(p[s:s + step], a, b), g)
    return out


def semi_infinite_velocity_sum(points: np.ndarray, a: np.ndarray, d: np.ndarray, strength: np.ndarray, *,
                               length_scale: float) -> np.ndarray:
    """Total velocity (P, 3) of semi-infinite filaments with circulations ``strength`` (S,)."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    g = np.asarray(strength, dtype=float).reshape(-1)
    out = np.zeros((len(p), 3))
    step = max(1, CHUNK_ELEMENTS // max(len(g), 1))
    for s in range(0, len(p), step):
        v = semi_infinite_velocity(p[s:s + step], a, d, length_scale=length_scale)
        out[s:s + step] = np.einsum("psk,s->pk", v, g)
    return out


def stretch(points: np.ndarray, beta: float) -> np.ndarray:
    """Prandtl–Glauert (Göthert) stretch: x -> x / beta, y and z unchanged."""
    q = np.array(points, dtype=float, copy=True)
    q[..., 0] /= beta
    return q


def to_physical(v: np.ndarray, beta: float) -> np.ndarray:
    """Velocity from the stretched space back to the physical one: (u / beta, v, w)."""
    out = np.array(v, dtype=float, copy=True)
    out[..., 0] /= beta
    return out


__all__ = [
    "segment_velocity",
    "segment_velocity_sum",
    "semi_infinite_velocity",
    "semi_infinite_velocity_sum",
    "stretch",
    "to_physical",
]
