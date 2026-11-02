"""Polygon operations on closed 2D outlines.

Every function here works on the Aerolume polygon convention (DESIGN.md section 3):
an ``(N, 2)`` float64 array, **implicitly closed** (the last point is not a copy of
the first) and **counter-clockwise** for a solid outline. The functions accept
anything array-like and are tolerant of the usual import debris -- a repeated
closing point, consecutive duplicates, clockwise input -- because outlines arrive
from SVG editors, CAD files, image tracing and hand-typed CSVs.

Only numpy is used, so the module is cheap to import from the GUI. The
differentiable (torch) counterparts of the few operations the optimiser needs live
in :mod:`aerolume.geometry.sdf` and :mod:`aerolume.geometry.params`.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "arc_length",
    "as_polygon",
    "bbox",
    "centroid",
    "close_gaps",
    "distance_to_polygon",
    "edge_lengths",
    "edges_cross",
    "frontal_height",
    "is_ccw",
    "is_self_intersecting",
    "orient_ccw",
    "perimeter",
    "point_in_polygon",
    "polygon_distance",
    "polygons_overlap",
    "remove_duplicates",
    "repair_self_intersections",
    "resample",
    "self_intersections",
    "signed_area",
    "simplify_rdp",
    "smooth",
    "transform",
    "turning_angles",
]


# ---------------------------------------------------------------------------
# basic measures
# ---------------------------------------------------------------------------

def as_polygon(poly: Iterable, *, drop_closing: bool = True) -> np.ndarray:
    """Coerce to a contiguous ``(N, 2)`` float64 array.

    When ``drop_closing`` is set and the last point repeats the first (to round-off
    relative to the outline size), it is dropped, so explicitly closed input from
    files becomes the implicitly closed form every other function expects.
    """
    p = np.ascontiguousarray(np.asarray(poly, dtype=np.float64).reshape(-1, 2))
    if drop_closing and len(p) > 1:
        scale = max(float(np.ptp(p, axis=0).max()), 1e-300)
        if np.all(np.abs(p[-1] - p[0]) <= 1e-12 * scale):
            p = p[:-1]
    return p


def signed_area(poly: Iterable) -> float:
    """Shoelace area; positive for counter-clockwise outlines."""
    p = as_polygon(poly)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    # subtracting the mean first keeps the products small (far-from-origin CAD data)
    x = x - x.mean()
    y = y - y.mean()
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def is_ccw(poly: Iterable) -> bool:
    """True when the outline is counter-clockwise (positive area)."""
    return signed_area(poly) > 0.0


def orient_ccw(poly: Iterable) -> np.ndarray:
    """Return the outline counter-clockwise, reversing it if needed.

    Reversal keeps point 0 in place (``[p0, pN-1, ..., p1]``) so that an airfoil in
    Selig order still starts at its trailing edge after re-orientation.
    """
    p = as_polygon(poly)
    if signed_area(p) < 0.0:
        p = np.concatenate([p[:1], p[:0:-1]])
    return p


def centroid(poly: Iterable) -> np.ndarray:
    """Area centroid ``(x, y)``; falls back to the vertex mean for degenerate outlines."""
    p = as_polygon(poly)
    if len(p) < 3:
        return p.mean(axis=0) if len(p) else np.zeros(2)
    o = p.mean(axis=0)
    q = p - o
    x, y = q[:, 0], q[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    a = 0.5 * c.sum()
    if abs(a) <= 1e-14 * max(float(np.ptp(q, axis=0).max()), 1e-300) ** 2:
        return o
    cx = ((x + xn) * c).sum() / (6.0 * a)
    cy = ((y + yn) * c).sum() / (6.0 * a)
    return o + np.array([cx, cy])


def bbox(poly_or_polys: Iterable | Sequence[Iterable]) -> tuple[float, float, float, float]:
    """``(xmin, xmax, ymin, ymax)`` of one outline or a list of outlines.

    The order matches :meth:`aerolume.model.Design.bounds`.
    """
    pts = _stack(poly_or_polys)
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def edge_lengths(poly: Iterable) -> np.ndarray:
    """Length of each edge ``i -> i+1`` including the implicit closing edge."""
    p = as_polygon(poly)
    return np.hypot(*(np.roll(p, -1, axis=0) - p).T)


def perimeter(poly: Iterable) -> float:
    """Total length of the closed outline."""
    return float(edge_lengths(poly).sum())


def arc_length(poly: Iterable) -> np.ndarray:
    """Cumulative arc length at each vertex, starting at 0 at vertex 0 (length N)."""
    e = edge_lengths(poly)
    return np.concatenate([[0.0], np.cumsum(e[:-1])])


def turning_angles(poly: Iterable) -> np.ndarray:
    """Signed exterior (turning) angle at each vertex, radians in (-pi, pi].

    Positive = left turn. For a CCW convex polygon the angles are positive and sum
    to 2 pi; a sharp corner such as a blunt trailing edge shows up as a value near
    pi/2, a cusp (sharp trailing edge) as a value near pi.
    """
    p = as_polygon(poly)
    d_in = p - np.roll(p, 1, axis=0)
    d_out = np.roll(p, -1, axis=0) - p
    cross = d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0]
    dot = (d_in * d_out).sum(axis=1)
    return np.arctan2(cross, dot)


def _stack(poly_or_polys) -> np.ndarray:
    if isinstance(poly_or_polys, np.ndarray) and poly_or_polys.ndim == 2:
        return poly_or_polys.reshape(-1, 2).astype(np.float64)
    items = list(poly_or_polys)
    if items and np.ndim(items[0]) == 1:          # a list of (x, y) pairs
        return np.asarray(items, dtype=np.float64).reshape(-1, 2)
    return np.vstack([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in items])


# ---------------------------------------------------------------------------
# clean-up
# ---------------------------------------------------------------------------

def remove_duplicates(poly: Iterable, tol: float | None = None) -> np.ndarray:
    """Drop consecutive points closer than ``tol`` (and a repeated closing point).

    ``tol`` defaults to 1e-9 of the outline size, i.e. only true duplicates from
    file round-off. Also removes zero-width spikes (a point whose neighbours
    coincide), which otherwise appear as self-touching outlines.
    """
    p = as_polygon(poly)
    if len(p) < 2:
        return p
    if tol is None:
        tol = 1e-9 * max(float(np.ptp(p, axis=0).max()), 1e-300)
    keep = [p[0]]
    for q in p[1:]:
        if math.hypot(q[0] - keep[-1][0], q[1] - keep[-1][1]) > tol:
            keep.append(q)
    while len(keep) > 1 and math.hypot(keep[-1][0] - keep[0][0], keep[-1][1] - keep[0][1]) <= tol:
        keep.pop()
    out = np.array(keep)
    # remove spikes p[i-1] == p[i+1] repeatedly (each removal can expose another)
    changed = True
    while changed and len(out) > 3:
        changed = False
        nxt = np.roll(out, -1, axis=0)
        prv = np.roll(out, 1, axis=0)
        spike = np.hypot(*(nxt - prv).T) <= tol
        if spike.any():
            i = int(np.flatnonzero(spike)[0])
            out = np.delete(out, [i, (i + 1) % len(out)], axis=0) if len(out) > 4 else np.delete(out, i, axis=0)
            changed = True
    return out


def simplify_rdp(poly: Iterable, epsilon: float, *, closed: bool = True) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification with tolerance ``epsilon``.

    Used on traced image outlines, where every pixel step is a vertex. For a closed
    outline the loop is split at vertex 0 and the vertex farthest from it, so both
    anchors are genuine extreme points. Iterative (no recursion limit on 100k-point
    contours).
    """
    p = as_polygon(poly) if closed else np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n < 4 or epsilon <= 0:
        return p.copy()
    if closed:
        far = int(np.argmax(np.hypot(*(p - p[0]).T)))
        if far == 0:
            return p.copy()
        ring = np.vstack([p, p[:1]])
        keep = np.zeros(n + 1, dtype=bool)
        keep[[0, far, n]] = True
        stack = [(0, far), (far, n)]
    else:
        ring = p
        keep = np.zeros(n, dtype=bool)
        keep[[0, n - 1]] = True
        stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = ring[i], ring[j]
        seg = ring[i + 1:j]
        ab = b - a
        L = math.hypot(*ab)
        if L == 0:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > epsilon:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    out = ring[keep]
    return out[:-1] if closed else out


def close_gaps(paths: Sequence[Iterable], tol: float | None = None, *,
               close_open: bool = True, notes: list[str] | None = None) -> list[np.ndarray]:
    """Chain open polylines into closed loops.

    CAD exports often describe one outline as dozens of separate LINE / ARC pieces in
    arbitrary order and direction, with endpoints that only match to a few microns.
    Pieces whose ends are within ``tol`` (default 1e-6 of the overall size: CAD
    round-off is joined, a deliberate gap in a hand-made drawing is not -- pass a
    larger value to heal those) are joined greedily, reversing as needed. A chain
    whose end returns to its start closes. Remaining open chains are closed with a
    straight segment when ``close_open`` is set (otherwise dropped), and a
    human-readable message is appended to ``notes`` (shown later by
    :func:`aerolume.geometry.check_design`).

    Pass only genuinely open pieces: an implicitly closed polygon cannot be told
    apart from an open polyline, and would be chained or closed again.
    Returns implicitly closed polygons (orientation unchanged).
    """
    items = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in paths]
    items = [p for p in items if len(p) >= 2]
    if not items:
        return []
    allpts = np.vstack(items)
    size = max(float(np.ptp(allpts, axis=0).max()), 1e-300)
    if tol is None:
        tol = 1e-6 * size
    pending = list(items)
    loops: list[np.ndarray] = []
    while pending:
        cur = pending.pop(0)
        chain = [cur]
        start, end = cur[0], cur[-1]
        while True:
            if len(np.vstack(chain)) > 2 and math.hypot(*(end - start)) <= tol:
                break
            if not pending:
                break
            heads = np.array([p[0] for p in pending])
            tails = np.array([p[-1] for p in pending])
            # extend at the end first, then at the start
            dh = np.hypot(*(heads - end).T)
            dt = np.hypot(*(tails - end).T)
            j_h, j_t = int(np.argmin(dh)), int(np.argmin(dt))
            if min(dh[j_h], dt[j_t]) <= tol:
                if dh[j_h] <= dt[j_t]:
                    nxt = pending.pop(j_h)
                else:
                    nxt = pending.pop(j_t)[::-1]
                chain.append(nxt[1:])
                end = nxt[-1]
                continue
            dh = np.hypot(*(heads - start).T)
            dt = np.hypot(*(tails - start).T)
            j_h, j_t = int(np.argmin(dh)), int(np.argmin(dt))
            if min(dh[j_h], dt[j_t]) <= tol:
                if dt[j_t] <= dh[j_h]:
                    prv = pending.pop(j_t)
                else:
                    prv = pending.pop(j_h)[::-1]
                chain.insert(0, prv[:-1])
                start = prv[0]
                continue
            break
        pts = np.vstack(chain)
        gap = math.hypot(*(pts[-1] - pts[0]))
        if gap <= tol:
            pts = pts[:-1]
        elif not close_open:
            continue
        elif notes is not None and len(pts) >= 3:
            notes.append(
                f"An open outline ({len(pts)} points) was closed with a straight {gap:.4g}-unit segment "
                f"({100 * gap / size:.2g}% of the drawing size); check the shape there.")
        if len(pts) >= 3:
            loops.append(pts)
    return loops


# ---------------------------------------------------------------------------
# resampling and smoothing
# ---------------------------------------------------------------------------

def _corner_mask(p: np.ndarray, corner_angle_deg: float) -> np.ndarray:
    return np.abs(turning_angles(p)) >= math.radians(corner_angle_deg)


def resample(poly: Iterable, n: int | None = None, *, spacing: float | None = None,
             curvature: float = 0.0, keep_corners: bool = True, corner_angle: float = 40.0,
             corners: Sequence[int] | None = None) -> np.ndarray:
    """Redistribute the vertices of a closed outline.

    Give either ``n`` (number of output points) or ``spacing`` (target edge length,
    same units as the outline). Points are placed by linear interpolation along the
    original outline, so the shape is preserved to the input's own resolution.

    ``keep_corners``: vertices whose turning angle exceeds ``corner_angle`` degrees
    (a blunt trailing edge, the rear edges of a car, the corners of a square) are
    kept exactly and the remaining points are distributed between them; without
    this a uniform resample would cut every sharp corner.

    ``curvature``: 0 gives uniform arc-length spacing. Larger values cluster points
    where the outline bends (``ds ~ 1 / (1 + curvature * L * |kappa|)``, with ``L``
    the outline size), the usual choice for a leading edge.

    ``corners``: explicit vertex indices to keep (overrides the angle detection);
    used by the templates, which know exactly where their sharp edges are. The
    input must then be free of duplicate points so the indices stay valid.
    """
    p = remove_duplicates(poly) if corners is None else as_polygon(poly)
    m = len(p)
    if m < 3:
        raise ValueError("resample needs a polygon with at least 3 distinct points")
    per = perimeter(p)
    if n is None:
        if spacing is None or spacing <= 0:
            raise ValueError("give n or a positive spacing")
        n = max(round(per / spacing), 3)
    n = int(n)
    if n < 3:
        raise ValueError("n must be at least 3")

    # weight per edge: 1 (uniform) plus a curvature term from the turning angles
    e = edge_lengths(p)
    w = np.ones(m)
    if curvature > 0:
        size = float(np.ptp(p, axis=0).max())
        ang = np.abs(turning_angles(p))
        if corners is not None:
            ang[np.asarray(corners, dtype=int) % m] = 0.0               # corners handled separately
        elif keep_corners:
            ang = np.where(_corner_mask(p, corner_angle), 0.0, ang)
        # discrete curvature at vertices -> per edge average
        kv = ang / np.maximum(0.5 * (e + np.roll(e, 1)), 1e-300)
        ke = 0.5 * (kv + np.roll(kv, -1))
        w = 1.0 + curvature * size * ke
    we = e * w                                     # "weighted length" of each edge

    if corners is not None:
        corners = np.unique(np.asarray(corners, dtype=int) % m)
    elif keep_corners:
        corners = np.flatnonzero(_corner_mask(p, corner_angle))
    else:
        corners = np.array([], int)
    if len(corners) >= n:
        # cannot place more points than corners: keep the sharpest n
        ang = np.abs(turning_angles(p))
        corners = np.sort(corners[np.argsort(-ang[corners])][:n])
    if len(corners) == 0:
        corners = np.array([0])          # no corners: one run starting at vertex 0

    # walk corner-to-corner runs; allocate points proportionally to weighted length
    runs = []
    for k in range(len(corners)):
        i0 = corners[k]
        i1 = corners[(k + 1) % len(corners)]
        idx = np.arange(i0, i1 if i1 > i0 else i1 + m) % m
        runs.append(idx)
    run_w = np.array([we[r].sum() for r in runs])
    free = n - len(corners)                       # points not pinned to corners
    alloc = run_w / run_w.sum() * free
    cnt = np.floor(alloc).astype(int)
    rem = free - cnt.sum()
    if rem > 0:
        cnt[np.argsort(-(alloc - cnt))[:rem]] += 1

    out = []
    for r, c in zip(runs, cnt):
        verts = np.vstack([p[r], p[(r[-1] + 1) % m]])          # run incl. end corner
        s_w = np.concatenate([[0.0], np.cumsum(we[r])])       # weighted parameter
        targets = s_w[-1] * np.arange(c + 1) / (c + 1)        # c interior + start
        seg = np.clip(np.searchsorted(s_w, targets, side="right") - 1, 0, len(r) - 1)
        frac = (targets - s_w[seg]) / np.maximum(we[r][seg], 1e-300)
        pts = verts[seg] + frac[:, None] * (verts[seg + 1] - verts[seg])
        out.append(pts)
    return np.vstack(out)


def smooth(poly: Iterable, iterations: int = 1, factor: float = 0.5, *,
           keep_corners: bool = True, corner_angle: float = 40.0) -> np.ndarray:
    """Taubin (lambda | mu) smoothing that removes jaggies without shrinking.

    Plain Laplacian smoothing shrinks an outline a little every pass (a traced
    silhouette would lose area); Taubin's alternating positive / negative pass is a
    low-pass filter with almost no shrinkage. Corners sharper than ``corner_angle``
    stay pinned when ``keep_corners`` is set. Vertex count is unchanged.
    """
    p = as_polygon(poly).copy()
    if len(p) < 4 or iterations <= 0:
        return p
    pinned = _corner_mask(p, corner_angle) if keep_corners else np.zeros(len(p), bool)
    lam, mu = factor, -factor / (1.0 - 0.1 * factor)    # k_pb = 0.1 pass-band
    for _ in range(int(iterations)):
        for f in (lam, mu):
            lap = 0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) - p
            lap[pinned] = 0.0
            p = p + f * lap
    return p


# ---------------------------------------------------------------------------
# intersections and repair
# ---------------------------------------------------------------------------

def _seg_intersections(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    """Proper intersections of segments a->b with c->d (broadcast). Returns (hit, t, u)."""
    r = b - a
    s = d - c
    den = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = c - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / den
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
    hit = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return hit, t, u


def self_intersections(poly: Iterable) -> list[tuple[int, int, np.ndarray]]:
    """All crossings between non-adjacent edges: ``[(i, j, point)]`` with ``i < j``.

    Edge ``i`` runs from vertex ``i`` to ``i+1`` (the last edge closes the loop).
    Uses a sweep over edges sorted by their x extent, so typical outlines of a few
    thousand points take milliseconds rather than an N^2 all-pairs test.
    """
    p = remove_duplicates(poly)
    m = len(p)
    if m < 4:
        return []
    a = p
    b = np.roll(p, -1, axis=0)
    xmin = np.minimum(a[:, 0], b[:, 0])
    xmax = np.maximum(a[:, 0], b[:, 0])
    ymin = np.minimum(a[:, 1], b[:, 1])
    ymax = np.maximum(a[:, 1], b[:, 1])
    order = np.argsort(xmin, kind="stable")
    xs = xmin[order]
    out = []
    for k, i in enumerate(order):
        hi = np.searchsorted(xs, xmax[i], side="right")
        cand = order[k + 1:hi]
        if len(cand) == 0:
            continue
        cand = cand[(ymin[cand] <= ymax[i]) & (ymax[cand] >= ymin[i])]
        # skip edges sharing a vertex with i
        cand = cand[(cand != (i + 1) % m) & (cand != (i - 1) % m) & (cand != i)]
        if len(cand) == 0:
            continue
        hit, t, _u = _seg_intersections(a[i], b[i], a[cand], b[cand])
        for j, tt in zip(cand[hit], t[hit]):
            ii, jj = (int(i), int(j)) if i < j else (int(j), int(i))
            out.append((ii, jj, a[i] + tt * (b[i] - a[i])))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def is_self_intersecting(poly: Iterable) -> bool:
    """True when any two non-adjacent edges of the outline cross or touch."""
    return len(self_intersections(poly)) > 0


def repair_self_intersections(poly: Iterable) -> np.ndarray:
    """Replace a self-intersecting outline by its outer boundary.

    The outline is split at every crossing into a planar graph, and the outer face is
    traced from the lowest-left vertex by always taking the most clockwise turn
    (keeping the exterior on the right). Small loops, overshooting edges and
    star-shaped tangles collapse to the region a viewer would see as the solid.
    Where the boundary only touches itself at a point (a bow-tie or "figure-8"),
    it is split there and the largest lobe is kept, since a pinch point cannot be
    represented as one simple outline. Returns a CCW outline; unchanged (CCW) input when there
    is nothing to repair. Raises ValueError if the walk does not close (degenerate
    input) so callers can report it rather than silently simulating garbage.
    """
    p = orient_ccw(remove_duplicates(poly))
    hits = self_intersections(p)
    if not hits:
        return p
    m = len(p)
    # parameter positions of intersections along each edge
    inserts: dict[int, list[tuple[float, np.ndarray]]] = {}
    for i, j, q in hits:
        for e in (i, j):
            a, b = p[e], p[(e + 1) % m]
            L2 = float(((b - a) ** 2).sum())
            t = float(((q - a) * (b - a)).sum() / L2) if L2 > 0 else 0.0
            inserts.setdefault(e, []).append((t, q))
    # build the split vertex sequence, merging coincident points into one node
    size = max(float(np.ptp(p, axis=0).max()), 1e-300)
    key_tol = 1e-9 * size
    nodes: list[np.ndarray] = []
    lookup: dict[tuple[int, int], int] = {}

    def node_id(q: np.ndarray) -> int:
        k = (round(float(q[0]) / key_tol), round(float(q[1]) / key_tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                kk = (k[0] + dx, k[1] + dy)
                if kk in lookup:
                    return lookup[kk]
        lookup[k] = len(nodes)
        nodes.append(np.asarray(q, dtype=float))
        return lookup[k]

    seq: list[int] = []
    for e in range(m):
        seq.append(node_id(p[e]))
        for _t, q in sorted(inserts.get(e, []), key=lambda r: r[0]):
            seq.append(node_id(q))
    adj: dict[int, set[int]] = {}
    for k in range(len(seq)):
        u, v = seq[k], seq[(k + 1) % len(seq)]
        if u != v:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    pts = np.array(nodes)
    start = min(adj, key=lambda k: (pts[k][0], pts[k][1]))
    heading = np.array([0.0, -1.0])                 # arrive at the leftmost node heading down
    cur, prev = start, -1
    path = [start]
    for _ in range(4 * len(pts) + 8):
        best, best_ang = None, None
        for nb in adj[cur]:
            if nb == prev and len(adj[cur]) > 1:
                continue
            d = pts[nb] - pts[cur]
            ang = math.atan2(heading[0] * d[1] - heading[1] * d[0], heading[0] * d[0] + heading[1] * d[1])
            if best_ang is None or ang < best_ang:
                best, best_ang = nb, ang
        if best is None:
            break
        heading = pts[best] - pts[cur]
        heading = heading / max(math.hypot(*heading), 1e-300)
        prev, cur = cur, best
        if cur == start:
            return remove_duplicates(orient_ccw(pts[_largest_simple_loop(path, pts)]))
        path.append(cur)
    raise ValueError("could not trace the outer boundary of a self-intersecting outline")


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def _largest_simple_loop(path: list[int], pts: np.ndarray) -> list[int]:
    """Split a closed node walk at repeated nodes (pinch points); largest loop wins."""
    loops: list[list[int]] = []
    stack: list[int] = []
    pos: dict[int, int] = {}
    for v in path:
        if v in pos:
            k = pos[v]
            loops.append(stack[k:])
            for u in stack[k + 1:]:
                pos.pop(u, None)
            stack = stack[:k + 1]
        else:
            pos[v] = len(stack)
            stack.append(v)
    loops.append(stack)
    loops = [lp for lp in loops if len(lp) >= 3]
    return max(loops, key=lambda lp: abs(signed_area(pts[lp])))


def point_in_polygon(poly: Iterable, points: Iterable, *, chunk: int = 200_000) -> np.ndarray | bool:
    """Non-zero winding test. ``points`` may be one ``(x, y)`` or an ``(M, 2)`` array.

    Points exactly on an edge are classified consistently but arbitrarily (half-open
    rule), which is what a rasteriser needs. Vectorised over points in chunks so a
    full grid can be classified without an M x N temporary blowing up.
    """
    p = as_polygon(poly)
    q = np.asarray(points, dtype=np.float64)
    single = q.ndim == 1
    q = q.reshape(-1, 2)
    a = p
    b = np.roll(p, -1, axis=0)
    e = b - a
    sgn = np.sign(e[:, 1])
    out = np.empty(len(q), dtype=bool)
    step = max(1, chunk // max(len(p), 1))
    for s in range(0, len(q), step):
        x = q[s:s + step, 0:1]
        y = q[s:s + step, 1:2]
        span = (a[:, 1] > y) != (b[:, 1] > y)
        cross = e[:, 0] * (y - a[:, 1]) - e[:, 1] * (x - a[:, 0])
        wn = (span & (cross * sgn > 0)) * sgn
        out[s:s + step] = wn.sum(axis=1) != 0
    return bool(out[0]) if single else out


def distance_to_polygon(poly: Iterable, points: Iterable) -> np.ndarray:
    """Unsigned distance from points ``(M, 2)`` to the outline (numpy, exact)."""
    p = as_polygon(poly)
    q = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = p
    e = np.roll(p, -1, axis=0) - p
    ee = np.maximum((e ** 2).sum(axis=1), 1e-300)
    out = np.empty(len(q))
    step = max(1, 2_000_000 // max(len(p), 1))
    for s in range(0, len(q), step):
        w = q[s:s + step, None, :] - a[None]
        t = np.clip((w * e).sum(-1) / ee, 0.0, 1.0)
        d = w - t[..., None] * e
        out[s:s + step] = np.sqrt((d ** 2).sum(-1).min(axis=1))
    return out


def polygon_distance(p1: Iterable, p2: Iterable) -> float:
    """Smallest distance between two outlines (0 when they touch or cross)."""
    a = as_polygon(p1)
    b = as_polygon(p2)
    if polygons_overlap(a, b):
        return 0.0
    return float(min(distance_to_polygon(a, b).min(), distance_to_polygon(b, a).min()))


def edges_cross(p1: Iterable, p2: Iterable) -> bool:
    """True when any edge of outline ``p1`` crosses or touches any edge of ``p2``."""
    a = as_polygon(p1)
    b = as_polygon(p2)
    a1, a2 = a, np.roll(a, -1, axis=0)
    b1, b2 = b, np.roll(b, -1, axis=0)
    # prune to edges inside the other outline's bounding box first
    bx0, bx1, by0, by1 = bbox(b)
    ka = ((np.maximum(a1[:, 0], a2[:, 0]) >= bx0) & (np.minimum(a1[:, 0], a2[:, 0]) <= bx1)
          & (np.maximum(a1[:, 1], a2[:, 1]) >= by0) & (np.minimum(a1[:, 1], a2[:, 1]) <= by1))
    a1, a2 = a1[ka], a2[ka]
    if not len(a1):
        return False
    step = max(1, 1_000_000 // len(b))
    for s in range(0, len(a1), step):
        hit, _, _ = _seg_intersections(a1[s:s + step, None], a2[s:s + step, None], b1[None], b2[None])
        if hit.any():
            return True
    return False


def polygons_overlap(p1: Iterable, p2: Iterable) -> bool:
    """True when the two solid regions share area or touch (edges meet, or one contains the other)."""
    a = as_polygon(p1)
    b = as_polygon(p2)
    ax0, ax1, ay0, ay1 = bbox(a)
    bx0, bx1, by0, by1 = bbox(b)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return False
    if point_in_polygon(b, a[0]) or point_in_polygon(a, b[0]):
        return True
    return edges_cross(a, b)


def frontal_height(poly_or_polys: Iterable | Sequence[Iterable]) -> float:
    """Height of the shape projected onto a plane normal to the flow (+x).

    The 2D analogue of frontal area. For several bodies the projected y-intervals are
    merged, so a gap between a car and a separate wing above it is not counted but
    overlapping bodies are not double-counted either.
    """
    if isinstance(poly_or_polys, np.ndarray) and poly_or_polys.ndim == 2:
        polys = [poly_or_polys]
    else:
        items = list(poly_or_polys)
        polys = [items] if items and np.ndim(items[0]) == 1 else items
    iv = sorted((float(np.min(np.asarray(q)[:, 1])), float(np.max(np.asarray(q)[:, 1])))
                for q in (np.asarray(x, dtype=float).reshape(-1, 2) for x in polys))
    total, cur0, cur1 = 0.0, None, None
    for lo, hi in iv:
        if cur0 is None:
            cur0, cur1 = lo, hi
        elif lo <= cur1:
            cur1 = max(cur1, hi)
        else:
            total += cur1 - cur0
            cur0, cur1 = lo, hi
    if cur0 is not None:
        total += cur1 - cur0
    return total


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def transform(poly: Iterable, scale: float | tuple[float, float] = 1.0, rotate_deg: float = 0.0,
              about: tuple[float, float] = (0.0, 0.0), translate: tuple[float, float] = (0.0, 0.0),
              mirror: str | None = None) -> np.ndarray:
    """Mirror, scale and rotate about ``about``, then translate.

    ``scale`` may be a scalar or ``(sx, sy)``; ``rotate_deg`` is counter-clockwise
    (the usual math sense; note that a nose-up pitch of a body facing -x is a
    *clockwise* rotation, which is why the solver rotates by -alpha); ``mirror`` is ``"x"`` (flip left-right, about the vertical line through
    ``about``) or ``"y"`` (flip upside down). A reflection (mirror, or an odd number
    of negative scale factors) reverses the vertex order so a CCW outline stays CCW.
    """
    p = as_polygon(poly)
    o = np.asarray(about, dtype=float)
    q = p - o
    flip = False
    if mirror is not None:
        if mirror not in ("x", "y"):
            raise ValueError("mirror must be 'x' (left-right) or 'y' (upside down)")
        if mirror == "x":
            q[:, 0] = -q[:, 0]
        else:
            q[:, 1] = -q[:, 1]
        flip = not flip
    sx, sy = (scale, scale) if np.isscalar(scale) else scale
    q = q * np.array([sx, sy], dtype=float)
    if (sx < 0) != (sy < 0):
        flip = not flip
    if rotate_deg:
        c, s = math.cos(math.radians(rotate_deg)), math.sin(math.radians(rotate_deg))
        q = q @ np.array([[c, s], [-s, c]])
    out = q + o + np.asarray(translate, dtype=float)
    if flip and len(out) > 2:
        out = np.concatenate([out[:1], out[:0:-1]])
    return out
