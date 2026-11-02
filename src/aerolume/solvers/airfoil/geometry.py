"""Turn an arbitrary airfoil outline into the coordinates the VII solver expects.

mfoil (like XFOIL) wants one closed airfoil contour in Selig order — trailing
edge, upper surface, leading edge, lower surface, back to the trailing edge —
with a well-defined trailing edge: sharp (first point == last point) or blunt
(first and last points are the two TE corners and the base between them is
modelled by a source panel). Users bring shapes from .dat files, SVG/DXF
drawings, image tracing or templates, in any scale, position, orientation and
point order, so this module:

1. extracts one outline (Design, Body, array or list) and rejects shapes that
   are not a single airfoil-like body with a clear message;
2. cleans it (duplicates, orientation) and finds the trailing edge: the end of
   the shape's longest dimension where the outline turns sharply over a short
   distance (a sharp wedge or the two corners of a blunt base);
3. takes the leading edge as the contour point farthest from the TE (XFOIL's
   definition), refined on a local spline so it need not be a data point;
4. de-rotates and scales so LE = (0, 0) and TE = (1, 0), mirrors shapes that
   face the wrong way, and returns Selig order plus a geometry report.

Gaps below ``SHARP_GAP`` (1e-5 chord, i.e. coordinate round-off) are reported
as a sharp trailing edge. The solvers then adapt the TE to their own
numerics (see ``solver.MIN_TE_GAP`` and docs/theory_airfoil.md); this module
reports the geometry as given.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

SHARP_GAP = 1e-5       # TE gaps below this fraction of chord are round-off: reported as sharp
MAX_TE_GAP = 0.10      # beyond this the "airfoil" is a bluff body
MAX_THICKNESS = 0.50
MAX_CAMBER = 0.25
MIN_POINTS = 10

_NOT_AIRFOIL = "Polar analysis needs a single airfoil-like body"


class AirfoilGeometryError(ValueError):
    """The shape cannot be analysed as an airfoil; the message says why, in user terms."""


# ---------------------------------------------------------------------------
# small polygon helpers (numpy only; the solver must not depend on GUI code)
# ---------------------------------------------------------------------------

def _signed_area(p: np.ndarray) -> float:
    x = p[:, 0] - p[:, 0].mean()
    y = p[:, 1] - p[:, 1].mean()
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _dedupe(p: np.ndarray) -> np.ndarray:
    """Drop consecutive (and wrap-around) duplicate points, to 1e-9 of the size."""
    tol = 1e-9 * max(float(np.ptp(p, axis=0).max()), 1e-300)
    keep = np.ones(len(p), dtype=bool)
    last = p[0]
    for i in range(1, len(p)):
        if math.hypot(*(p[i] - last)) <= tol:
            keep[i] = False
        else:
            last = p[i]
    p = p[keep]
    while len(p) > 1 and math.hypot(*(p[-1] - p[0])) <= tol:
        p = p[:-1]
    return p


def _turning(p: np.ndarray) -> np.ndarray:
    """Signed exterior angle at each vertex of a closed polygon (left turn positive)."""
    d_in = p - np.roll(p, 1, axis=0)
    d_out = np.roll(p, -1, axis=0) - p
    cross = d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0]
    return np.arctan2(cross, (d_in * d_out).sum(axis=1))


def _self_intersecting(p: np.ndarray) -> bool:
    try:
        from ...geometry.polygon import self_intersections  # contract of geometry/
        return len(self_intersections(p)) > 0
    except ImportError:                                        # pragma: no cover - fallback
        a, b = p, np.roll(p, -1, axis=0)
        m = len(p)
        for i in range(m):
            j = np.arange(i + 2, m)
            if i == 0:
                j = j[j != m - 1]
            if len(j) == 0:
                continue
            r, s = b[i] - a[i], b[j] - a[j]
            den = r[0] * s[:, 1] - r[1] * s[:, 0]
            q = a[j] - a[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / den
                u = (q[:, 0] * r[1] - q[:, 1] * r[0]) / den
            if np.any((den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
                return True
        return False


def _farthest_pair(p: np.ndarray) -> tuple[int, int]:
    """Indices of the two vertices farthest apart (via the convex hull)."""
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(p).vertices
    except Exception:          # noqa: BLE001 - degenerate (collinear) input
        h = np.arange(len(p))
    q = p[h]
    d2 = ((q[:, None, :] - q[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    return int(h[i]), int(h[j])


# ---------------------------------------------------------------------------
# outline extraction
# ---------------------------------------------------------------------------

def _extract(obj: Any) -> tuple[np.ndarray, str, list[str]]:
    """(points, name, warnings) from a Design, Body, dict, array or list of points."""
    warnings: list[str] = []
    name = ""
    bodies = getattr(obj, "bodies", None)
    if bodies is not None:                                  # Design
        kind = getattr(obj, "kind", "airfoil")
        name = getattr(obj, "name", "")
        if kind == "vehicle":
            raise AirfoilGeometryError(
                f"{_NOT_AIRFOIL}; '{name}' is a vehicle design. Use the CFD analysis for vehicles.")
        if len(bodies) != 1:
            raise AirfoilGeometryError(
                f"{_NOT_AIRFOIL}; '{name}' has {len(bodies)} bodies. Multi-element sections "
                "(slats, flaps, wheels) need the CFD analysis.")
        obj = bodies[0]
    if hasattr(obj, "points"):                              # Body
        name = name or getattr(obj, "name", "")
        obj = obj.points
    if isinstance(obj, (list, tuple)) and obj and np.ndim(obj[0]) == 2:
        if len(obj) != 1:
            raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; got {len(obj)} separate outlines.")
        obj = obj[0]
    try:
        p = np.asarray(obj, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; could not read the coordinates ({exc}).") from exc
    if p.ndim != 2 or 2 not in p.shape:
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; expected an (N, 2) array of points, got shape {p.shape}.")
    if p.shape[1] != 2:
        p = p.T
    if not np.all(np.isfinite(p)):
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; the coordinates contain NaN or infinite values.")
    return p, name, warnings


# ---------------------------------------------------------------------------
# trailing / leading edge
# ---------------------------------------------------------------------------

def _windowed_turn(p: np.ndarray, turn: np.ndarray, k: int, width: float) -> float:
    """Total |turning| of vertices within arc length ``width`` of vertex k (both sides)."""
    n = len(p)
    edge = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
    total = abs(turn[k])
    for step in (1, -1):
        s, j = 0.0, k
        while True:
            s += edge[j] if step == 1 else edge[(j - 1) % n]
            j = (j + step) % n
            if s > width or j == k:
                break
            total += abs(turn[j])
    return total


def _find_te(p: np.ndarray, warnings: list[str]) -> tuple[int, int | None]:
    """Trailing-edge vertex index, plus the other base corner for a blunt TE (else None).

    The TE sits at one end of the shape's longest dimension, and is the end
    where the outline turns through most of 180 deg within ~1 % of the length
    (a round nose spreads that turn over a much longer arc).
    """
    i, j = _farthest_pair(p)
    span = float(math.hypot(*(p[i] - p[j])))
    turn = _turning(p)
    wi = _windowed_turn(p, turn, i, 0.01 * span)
    wj = _windowed_turn(p, turn, j, 0.01 * span)
    if abs(wi - wj) < math.radians(10.0):
        k = i if p[i, 0] >= p[j, 0] else j          # ambiguous (e.g. a diamond): TE downstream
    else:
        k = i if wi > wj else j
    t = abs(turn)
    if t[k] >= math.radians(120.0):
        return k, None
    # blunt trailing edge: a second sharp corner a short way along the outline.
    # Search up to half the length so that a bluff base (a car's tail) is
    # recognised as such and rejected, rather than mistaken for a sharp TE.
    n = len(p)
    edge = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
    best: tuple[float, int] | None = None
    for step in (1, -1):
        s, acc, j = 0.0, t[k], k
        for _ in range(n // 2):
            s += edge[j] if step == 1 else edge[(j - 1) % n]
            j = (j + step) % n
            acc += t[j]
            if s > 0.5 * span:
                break
            if t[j] >= math.radians(45.0) and acc >= math.radians(150.0):
                if best is None or s < best[0]:
                    best = (s, j)
                break
    if best is not None:
        if best[0] > MAX_TE_GAP * span:
            raise AirfoilGeometryError(
                f"{_NOT_AIRFOIL}; its blunt base is {best[0] / span * 100:.0f}% of the length, which makes "
                "it a bluff body. Use the CFD analysis instead.")
        return k, best[1]
    if _windowed_turn(p, turn, k, 0.01 * span) >= math.radians(120.0):
        warnings.append("The trailing edge is not crisply defined; the sharpest rear point was used.")
        return k, None
    raise AirfoilGeometryError(
        f"{_NOT_AIRFOIL} with a trailing edge (sharp or blunt). The rear of this shape is rounded, "
        "so the Kutta condition that sets the lift is undefined; use the CFD analysis instead.")


def _refine_le(p: np.ndarray, te: np.ndarray, k: int) -> np.ndarray:
    """Point of the local cubic through vertices near k that is farthest from te."""
    n = len(p)
    idx = [(k + d) % n for d in range(-3, 4)]
    q = p[idx]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    if np.any(np.diff(s) <= 0):
        return p[k].copy()
    from scipy.interpolate import CubicSpline
    cx, cy = CubicSpline(s, q[:, 0]), CubicSpline(s, q[:, 1])
    ss = np.linspace(s[2], s[4], 401)
    d = np.hypot(cx(ss) - te[0], cy(ss) - te[1])
    m = int(np.argmax(d))
    out = np.array([cx(ss[m]), cy(ss[m])])
    return out if d[m] >= math.hypot(*(p[k] - te)) else p[k].copy()


def _surface_check(q: np.ndarray, ile: int, warnings: list[str]) -> tuple[float, float, float, float]:
    """Thickness / camber from the normalised Selig contour; rejects folded or crossing surfaces."""
    upper = q[:ile + 1][::-1]          # LE -> TE
    lower = q[ile:]                    # LE -> TE
    for name, s in (("upper", upper), ("lower", lower)):
        dx = np.diff(s[:, 0])
        back = -dx[dx < 0].sum()
        if back > 0.02:
            raise AirfoilGeometryError(
                f"{_NOT_AIRFOIL}; its {name} surface doubles back on itself "
                f"({back * 100:.1f}% of chord), which an airfoil surface never does.")
    xs = 0.5 * (1 - np.cos(np.linspace(0.02, np.pi - 0.02, 120)))

    def interp(s: np.ndarray) -> np.ndarray:
        order = np.argsort(s[:, 0], kind="stable")
        return np.interp(xs, s[order, 0], s[order, 1])

    yu, yl = interp(upper), interp(lower)
    th = yu - yl
    if np.min(th) < -1e-3:
        raise AirfoilGeometryError(
            f"{_NOT_AIRFOIL}; its upper and lower surfaces cross (at x/c = {xs[int(np.argmin(th))]:.2f}).")
    camber = 0.5 * (yu + yl)
    it, ic = int(np.argmax(th)), int(np.argmax(np.abs(camber)))
    return float(th[it]), float(xs[it]), float(camber[ic]), float(xs[ic])


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def prepare_coords(poly_or_design: Any) -> tuple[np.ndarray, float, dict]:
    """Normalise an airfoil outline for the VII solver: ``(coords, chord, info)``.

    ``poly_or_design`` may be a :class:`~aerolume.model.Design` (exactly one
    body; vehicle designs are rejected), a :class:`~aerolume.model.Body`, or
    an (N, 2) array / list of points in any order, orientation, scale and
    position (explicitly or implicitly closed, sharp or blunt trailing edge).

    Returns ``coords`` in Selig order (upper TE -> LE -> lower TE) with the
    leading edge at (0, 0) and the trailing edge at (1, 0) — a sharp TE point
    appears first and last, a blunt TE's corners are the first (upper) and last
    (lower) points — plus the ``chord`` in input units and an ``info`` dict:
    ``name, chord, le, te`` (input frame), ``incidence_deg`` (nose-up angle of
    the input chord line; the solver measures alpha from the chord line),
    ``mirrored``, ``te_type`` ("sharp" | "blunt"), ``te_gap`` (x/c), ``thickness``,
    ``thickness_x``, ``camber``, ``camber_x``, ``n_points``, ``warnings``.

    Raises :class:`AirfoilGeometryError` (a ValueError) with a user-readable
    message, starting "Polar analysis needs a single airfoil-like body", when
    the shape is not something an airfoil analysis can handle.
    """
    p, name, warnings = _extract(poly_or_design)
    p = _dedupe(p)
    if len(p) < MIN_POINTS:
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; the outline has only {len(p)} distinct points "
                                   f"(at least {MIN_POINTS} are needed).")
    size = float(np.ptp(p, axis=0).max())
    if abs(_signed_area(p)) < 1e-6 * size ** 2:
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; the outline encloses no area (a line or a single "
                                   "curve). Give a closed outline with thickness.")
    if _signed_area(p) < 0:
        p = p[::-1].copy()                       # CCW
    if _self_intersecting(p):
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; the outline crosses itself. Repair it in the "
                                   "Geometry page first.")

    k, k2 = _find_te(p, warnings)
    if k2 is None:
        te = p[k].copy()
    else:
        te = 0.5 * (p[k] + p[k2])
    dist = np.hypot(*(p - te).T)
    ile = int(np.argmax(dist))
    at_origin = np.flatnonzero(np.hypot(p[:, 0], p[:, 1]) <= 1e-6 * size)
    if len(at_origin) and te[0] > 0 and abs(te[1]) <= 1e-6 * size and dist[at_origin[0]] > 0.98 * dist[ile]:
        # already in the design frame (DESIGN.md: LE at the origin, TE on +x): keep that
        # chord line so polar angles agree with the CFD analysis of the same Design
        le = p[at_origin[0]].copy()
    else:
        le = _refine_le(p, te, ile)
    chord = float(math.hypot(*(te - le)))
    if chord <= 0 or chord < 0.5 * size:
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; no clear leading-to-trailing-edge chord was found.")

    # chord-line frame: LE at the origin, TE on +x at 1
    ang = math.atan2(te[1] - le[1], te[0] - le[0])
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    q = ((p - le) @ rot.T) / chord
    incidence = -math.degrees(ang)
    mirrored = False
    if abs(ang) > math.pi / 2:
        # the shape faces downstream-first: mirror it (keeps camber "up") rather than
        # turning it upside down, and report the incidence of the mirrored section
        mirrored = True
        warnings.append("The leading edge pointed downstream; the section was mirrored left-right so "
                        "that it faces the flow.")
        pm = p.copy()
        pm[:, 0] = -pm[:, 0]
        lem, tem = le.copy(), te.copy()
        lem[0], tem[0] = -lem[0], -tem[0]
        ang = math.atan2(tem[1] - lem[1], tem[0] - lem[0])
        c, s = math.cos(-ang), math.sin(-ang)
        rot = np.array([[c, -s], [s, c]])
        q = ((pm - lem) @ rot.T) / chord
        incidence = -math.degrees(ang)
        # mirroring reverses the orientation: restore CCW and re-index the TE vertices
        q = q[::-1].copy()
        k = len(q) - 1 - k
        k2 = None if k2 is None else len(q) - 1 - k2
    q[np.abs(q) < 1e-14] = 0.0

    # Selig order starting at the TE (CCW from the rear goes over the upper surface first)
    n = len(q)
    if k2 is None:
        te_type, gap = "sharp", 0.0
        q = np.roll(q, -k, axis=0)
        q = np.vstack([q, q[:1]])
        q[0] = q[-1] = (1.0, 0.0)
    else:
        # In CCW order the base runs from the lower corner up to the upper corner, and
        # it is the short way round between the two corners.
        up, lo = (k2, k) if (k2 - k) % n < (k - k2) % n else (k, k2)
        gap = float(math.hypot(*(q[up] - q[lo])))
        q = np.roll(q, -up, axis=0)
        lo_i = (lo - up) % n
        # keep upper corner .. LE .. lower corner; points on the base itself are dropped
        q = q[:lo_i + 1]
        te_type = "blunt"
        if gap < SHARP_GAP:
            mid = np.array([1.0, 0.0])
            q[0] = q[-1] = mid
            te_type, gap = "sharp", 0.0
    if gap > MAX_TE_GAP:
        raise AirfoilGeometryError(
            f"{_NOT_AIRFOIL}; its blunt base is {gap * 100:.0f}% of the length, which makes it a bluff "
            "body. Use the CFD analysis instead.")
    if gap > 0.03:
        warnings.append(f"Very thick trailing edge ({gap * 100:.1f}% of chord): base drag is only "
                        "approximately modelled.")

    ile = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
    t, tx, cmax, cx = _surface_check(q, ile, warnings)
    if t > MAX_THICKNESS:
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; it is {t * 100:.0f}% thick, far beyond any airfoil. "
                                   "Use the CFD analysis for bluff bodies.")
    if t > 0.30:
        warnings.append(f"Very thick section ({t * 100:.0f}% of chord): expect early separation and "
                        "reduced accuracy.")
    if t < 0.005:
        warnings.append(f"Very thin section ({t * 100:.2f}% of chord): leading-edge separation is likely "
                        "and the integral boundary layer may not converge.")
    if abs(cmax) > MAX_CAMBER:
        raise AirfoilGeometryError(f"{_NOT_AIRFOIL}; its camber ({cmax * 100:.0f}% of chord) is beyond "
                                   "what a single-element airfoil analysis can handle.")
    if len(q) > 1500:
        warnings.append(f"The outline has {len(q)} points; noisy outlines (e.g. traced from images) "
                        "give wavy pressure distributions. Smoothing or resampling it helps.")

    info = {
        "name": name,
        "chord": chord,
        "le": (float(le[0]), float(le[1])),
        "te": (float(te[0]), float(te[1])),
        "incidence_deg": float(incidence),
        "mirrored": mirrored,
        "te_type": te_type,
        "te_gap": float(gap),
        "thickness": t,
        "thickness_x": tx,
        "camber": cmax,
        "camber_x": cx,
        "n_points": len(q),
        "warnings": warnings,
    }
    return np.ascontiguousarray(q), chord, info
