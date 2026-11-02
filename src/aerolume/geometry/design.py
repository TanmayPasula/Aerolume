"""Turn raw outlines into a :class:`~aerolume.model.Design`, and check designs.

This is where imported geometry gets its *meaning*: which point is the leading
edge, where the road is, what the reference length and height are. Getting those
right is what makes a drag coefficient comparable with a published one, so the
rules are explicit (DESIGN.md section 3):

* Airfoils: leading edge at (0, 0), trailing edge at (chord, 0), Selig order,
  ``length_ref = force_ref = chord``, moments about the quarter chord.
* Vehicles: nose at the left (x = 0), road at y = 0 with the requested ground
  clearance, ``length_ref`` = overall length (for Re), ``force_ref`` = frontal
  height (2D analogue of frontal area), moments about (x-centre, road).
* Generic shapes: ``length_ref`` = bounding-box width, ``force_ref`` = height.

:func:`check_design` then looks for the things that silently ruin a 2D immersed
boundary simulation -- features thinner than the grid, slots that the grid closes,
a car touching the road -- and explains them in plain words for the Geometry page.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..model import Body, Design
from . import polygon as P

__all__ = [
    "check_design",
    "design_from_airfoil",
    "design_from_polygons",
    "find_wheels",
    "normalize_airfoil",
]


# ---------------------------------------------------------------------------
# airfoils
# ---------------------------------------------------------------------------

def _trailing_edge(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(TE point, polygon rolled so index 0 is the upper TE point), p CCW.

    Blunt TE in Selig order: the first and last points are both at the rear and the
    closing edge is roughly perpendicular to the chord -> TE = their midpoint.
    Otherwise the TE is the sharpest vertex among those near maximum x (largest
    turning angle), and the outline is rolled to start there.
    """
    xr = float(np.ptp(p[:, 0]))
    xmax = float(p[:, 0].max())
    d = p[0] - p[-1]
    if (p[0, 0] > xmax - 0.02 * xr and p[-1, 0] > xmax - 0.02 * xr and abs(d[1]) > abs(d[0])
            and np.hypot(*d) < 0.1 * xr):
        return 0.5 * (p[0] + p[-1]), p
    near = np.flatnonzero(p[:, 0] > xmax - 0.01 * xr)
    ang = np.abs(P.turning_angles(p))
    i = int(near[np.argmax(ang[near])])
    return p[i].copy(), np.roll(p, -i, axis=0)


def _refine_le(p: np.ndarray, te: np.ndarray, half: int = 4) -> tuple[np.ndarray, int]:
    """Locate the leading edge between vertices and make it a vertex.

    The LE is the point of the *continuous* outline farthest from the TE (there the
    tangent is perpendicular to the chord, which is what a sqrt-type nose model --
    CST, thin-airfoil theory, panel codes -- assumes). The discrete farthest vertex
    can be half a panel off, which on a cambered section shifts the nose by ~1e-3
    chord, so a parametric cubic spline through the neighbouring vertices is
    maximised instead and the result inserted as a vertex (unless it coincides
    with one). Returns (outline, LE index).
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize_scalar

    n = len(p)
    i = int(np.argmax(np.hypot(*(p - te).T)))
    if n < 2 * half + 3:
        return p, i
    idx = (np.arange(i - half, i + half + 1)) % n
    w = p[idx]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(w, axis=0).T))])
    if np.any(np.diff(s) <= 0):
        return p, i
    sx, sy = CubicSpline(s, w[:, 0]), CubicSpline(s, w[:, 1])

    def neg_d2(t: float) -> float:
        return -((sx(t) - te[0]) ** 2 + (sy(t) - te[1]) ** 2)

    r = minimize_scalar(neg_d2, bounds=(s[half - 1], s[half + 1]), method="bounded",
                        options={"xatol": 1e-12 * max(s[-1], 1e-300)})
    t = float(r.x)
    q = np.array([float(sx(t)), float(sy(t))])
    if -r.fun <= np.hypot(*(p[i] - te)) ** 2:
        return p, i
    k = int(np.searchsorted(s, t))                      # segment (k-1, k) of the window
    gap = min(t - s[k - 1], s[k] - t)
    if gap <= 1e-9 * s[-1]:                             # coincides with a vertex
        j = int(idx[k - 1] if t - s[k - 1] <= s[k] - t else idx[k % len(idx)])
        return p, j
    ins = int(idx[k - 1]) + 1                           # insert after vertex idx[k-1]
    return np.insert(p, ins, q, axis=0), ins


def normalize_airfoil(coords: np.ndarray, chord: float = 1.0) -> np.ndarray:
    """Scale/rotate/translate an airfoil to LE (0, 0), TE (chord, 0), Selig order, CCW.

    The leading edge is the point farthest from the trailing edge (XFOIL's
    definition), so a section that was drawn at an incidence is de-rotated. The
    returned outline starts at the upper trailing edge.
    """
    p = P.orient_ccw(P.remove_duplicates(coords))
    if len(p) < 3:
        raise ValueError("an airfoil needs at least 3 points")
    te, p = _trailing_edge(p)
    p, ile = _refine_le(p, te)
    le = p[ile]
    c0 = float(np.hypot(*(te - le)))
    if c0 <= 0:
        raise ValueError("degenerate airfoil (zero chord)")
    ang = math.atan2(te[1] - le[1], te[0] - le[0])
    q = P.transform(p - le, scale=chord / c0, rotate_deg=-math.degrees(ang))
    q[np.abs(q) < 1e-14 * chord] = 0.0
    return q


def design_from_airfoil(coords: np.ndarray, name: str = "Airfoil", chord: float = 1.0, *,
                        source: dict | None = None, notes: str = "") -> Design:
    """An airfoil :class:`Design` from coordinates in any position, scale or order.

    The outline is normalised with :func:`normalize_airfoil`; ``chord`` is in
    metres. ``source`` records provenance (e.g. ``{"type": "naca", "code": "2412"}``)
    and may carry ``"import_notes"`` that :func:`check_design` reports.
    """
    if not chord > 0:
        raise ValueError("chord must be positive")
    q = normalize_airfoil(coords, chord)
    if P.is_self_intersecting(q):
        q = P.repair_self_intersections(q)
        q = normalize_airfoil(q, chord)
        src = dict(source or {})
        src.setdefault("import_notes", []).append(
            "The airfoil outline crossed itself; it was replaced by its outer boundary.")
        source = src
    return Design(name=name, kind="airfoil", bodies=[Body(name=name, points=q, role="body")],
                  length_ref=float(chord), force_ref=float(chord), moment_ref=(0.25 * chord, 0.0),
                  source=dict(source or {"type": "airfoil"}), notes=notes)


# ---------------------------------------------------------------------------
# general outlines
# ---------------------------------------------------------------------------

def _clean(polys: Sequence[np.ndarray], notes: list[str]) -> list[np.ndarray]:
    """Deduplicate, orient, repair crossings, drop degenerate and nested outlines."""
    good = []
    for i, raw in enumerate(polys):
        q = P.remove_duplicates(raw)
        if len(q) < 3 or abs(P.signed_area(q)) <= 1e-14 * max(float(np.ptp(q, axis=0).max()), 1e-300) ** 2:
            notes.append(f"Ignored outline {i + 1}: it has no area.")
            continue
        q = P.orient_ccw(q)
        if P.is_self_intersecting(q):
            try:
                q = P.repair_self_intersections(q)
                notes.append(f"Outline {i + 1} crossed itself; it was replaced by its outer boundary.")
            except ValueError:
                notes.append(f"Outline {i + 1} crosses itself and could not be repaired automatically.")
        good.append(q)
    # outlines entirely inside another one are holes / interior detail (windows, logos)
    keep = []
    areas = [abs(P.signed_area(q)) for q in good]
    order = np.argsort(areas)[::-1]
    for k in order:
        q = good[k]
        inside = any(
            areas[j] > areas[k] and P.point_in_polygon(good[j], q).all()
            and not P.edges_cross(good[j], q)
            for j in keep)
        if inside:
            continue
        keep.append(k)
    dropped = len(good) - len(keep)
    if dropped:
        notes.append(f"Ignored {dropped} outline(s) lying inside another one (holes, windows or "
                     "interior details do not affect the external flow).")
    return [good[k] for k in sorted(keep)]


def find_wheels(polys: Sequence[np.ndarray]) -> list[tuple[float, float, float] | None]:
    """For each outline, ``(cx, cy, r)`` if it looks like a wheel, else None.

    A wheel is a near-circular outline (isoperimetric ratio 4 pi A / P^2 > 0.97)
    in the lower half of the vehicle and smaller than 40 % of the largest body.
    Detected only when there are several outlines, so a lone cylinder stays a body.
    """
    if len(polys) < 2:
        return [None] * len(polys)
    _, _, y0, y1 = P.bbox(polys)
    big = max(float(np.ptp(q[:, 0])) for q in polys)
    out: list[tuple[float, float, float] | None] = []
    for q in polys:
        a = abs(P.signed_area(q))
        per = P.perimeter(q)
        roundness = 4 * math.pi * a / per**2 if per > 0 else 0.0
        c = P.centroid(q)
        w = float(np.ptp(q[:, 0]))
        if roundness > 0.97 and w < 0.4 * big and c[1] < 0.5 * (y0 + y1):
            out.append((float(c[0]), float(c[1]), math.sqrt(a / math.pi)))
        else:
            out.append(None)
    return out


def design_from_polygons(polys: Sequence[np.ndarray], name: str = "Shape", kind: str = "generic", *,
                         length: float | None = None, ground_clearance: float | None = None,
                         source: dict | None = None, names: Sequence[str] | None = None,
                         notes: list[str] | None = None) -> Design:
    """A :class:`Design` from one or more outlines (e.g. straight from an importer).

    ``kind="vehicle"``: outlines are scaled uniformly so the overall length equals
    ``length`` [m] (if given), the nose is put at x = 0 and the lowest point at
    ``y = ground_clearance`` above the road (default: 10 % of the overall height,
    a typical car's ratio). Wheel-like circles become ``role="wheel"`` bodies with
    their centre set. References: ``length_ref`` = overall length, ``force_ref`` =
    frontal height, ``moment_ref`` = (x-centre, 0) on the road.

    ``kind="generic"``: optional scaling to ``length``; placed with its bounding box
    centred on the origin, or on the road at ``ground_clearance`` when that is given.
    ``length_ref`` = bounding-box width, ``force_ref`` = bounding-box height,
    ``moment_ref`` = bounding-box centre.

    ``kind="airfoil"``: one outline is normalised with :func:`design_from_airfoil`
    (chord = ``length`` or 1 m); several outlines are treated as a multi-element
    section scaled to overall length ``length`` with ``role="element"``.

    Messages about what was changed (repairs, ignored holes, import notes passed in
    ``notes``) are stored in ``source["import_notes"]`` for :func:`check_design`.
    """
    msgs: list[str] = list(notes or [])
    src = dict(source or {"type": "import"})
    msgs = list(src.get("import_notes", [])) + msgs
    polys = [np.asarray(q, dtype=float).reshape(-1, 2) for q in polys]
    if kind == "airfoil" and len(polys) == 1:
        if msgs:
            src["import_notes"] = msgs
        return design_from_airfoil(polys[0], name, chord=length or 1.0, source=src)
    clean = _clean(polys, msgs)
    if not clean:
        raise ValueError("no usable closed outline in the input")
    if names is not None and len(names) == len(polys) and len(clean) == len(polys):
        labels = list(names)
    else:
        labels = [name] if len(clean) == 1 else [f"{name} {i + 1}" for i in range(len(clean))]

    x0, x1, y0, y1 = P.bbox(clean)
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("outline has zero width or height")
    s = (length / width) if length else 1.0
    if length is not None and not length > 0:
        raise ValueError("length must be positive")

    if kind == "vehicle":
        clr = ground_clearance if ground_clearance is not None else 0.10 * height * s
        if clr < 0:
            raise ValueError("ground clearance cannot be negative")
        placed = [(q - [x0, y0]) * s + [0.0, clr] for q in clean]
        wheels = find_wheels(placed)
        bodies = []
        for q, lab, wh in zip(placed, labels, wheels):
            if wh is not None:
                bodies.append(Body(name=lab if names else f"Wheel {sum(b.role == 'wheel' for b in bodies) + 1}",
                                   points=q, role="wheel", center=(wh[0], wh[1])))
            else:
                bodies.append(Body(name=lab, points=q, role="body"))
        L = width * s
        d = Design(name=name, kind="vehicle", bodies=bodies, length_ref=L,
                   force_ref=P.frontal_height(placed), moment_ref=(0.5 * L, 0.0), source=src)
    elif kind == "airfoil":
        placed = [(q - [x0, 0.0]) * s for q in clean]
        main = int(np.argmax([abs(P.signed_area(q)) for q in placed]))
        le_y = float(placed[main][np.argmin(placed[main][:, 0]), 1])
        placed = [q - [0.0, le_y] for q in placed]
        L = width * s
        bodies = [Body(name=lab, points=q, role="element") for q, lab in zip(placed, labels)]
        d = Design(name=name, kind="airfoil", bodies=bodies, length_ref=L, force_ref=L,
                   moment_ref=(0.25 * L, 0.0), source=src)
    elif kind == "generic":
        if ground_clearance is not None:
            placed = [(q - [0.5 * (x0 + x1), y0]) * s + [0.0, ground_clearance] for q in clean]
        else:
            placed = [(q - [0.5 * (x0 + x1), 0.5 * (y0 + y1)]) * s for q in clean]
        bx0, bx1, by0, by1 = P.bbox(placed)
        bodies = [Body(name=lab, points=q, role="body") for q, lab in zip(placed, labels)]
        d = Design(name=name, kind="generic", bodies=bodies, length_ref=bx1 - bx0, force_ref=by1 - by0,
                   moment_ref=(0.5 * (bx0 + bx1), 0.5 * (by0 + by1)), source=src)
    else:
        raise ValueError(f"unknown design kind {kind!r}")
    if msgs:
        d.source["import_notes"] = msgs
    return d


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def _fmt_len(v: float) -> str:
    if v >= 1:
        return f"{v:.3g} m"
    if v >= 0.01:
        return f"{v * 100:.3g} cm"
    return f"{v * 1000:.3g} mm"


def _thin_points(q: np.ndarray, limit: float, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Points of outline ``q`` whose local solid thickness is below ``limit``.

    Thickness at a surface point = distance to the nearest point of the outline that
    is *across the solid*: far away along the outline (arc separation > 3x the
    straight distance, which excludes neighbours on the same surface and rounded
    edges) with the connecting segment inside the solid. Returns (points, thickness).
    """
    r = P.resample(q, spacing=ds, keep_corners=True)
    if len(r) > 4000:
        r = P.resample(q, 4000)
    s = P.arc_length(r)
    per = P.perimeter(r)
    best = np.full(len(r), np.inf)
    step = max(1, 2_000_000 // len(r))
    for a in range(0, len(r), step):
        d = np.hypot(r[a:a + step, None, 0] - r[None, :, 0], r[a:a + step, None, 1] - r[None, :, 1])
        sep = np.abs(s[a:a + step, None] - s[None, :])
        sep = np.minimum(sep, per - sep)
        cand = (d < limit) & (sep > 3.0 * d + 1e-12)
        ii, jj = np.nonzero(cand)
        if len(ii) == 0:
            continue
        mid = 0.5 * (r[a + ii] + r[jj])
        ok = P.point_in_polygon(q, mid)
        ii, jj = ii[ok], jj[ok]
        np.minimum.at(best, a + ii, d[ii, jj])
    sel = np.isfinite(best)
    return r[sel], best[sel]


def check_design(design: Design, cells_per_ref: float | None = None) -> list[str]:
    """Plain-language problems and caveats of a design (empty list = all clear).

    Always checked: notes left by the importer (auto-closed gaps, repairs), point
    count, orientation, self-intersections, overlapping bodies, and for vehicles
    contact with or penetration of the road. With ``cells_per_ref`` (grid cells per
    ``length_ref``) the resolution checks run too: solid features thinner than ~2
    cells, gaps narrower than ~2 cells (between bodies, or under a car) which the
    grid will close, and trailing edges too thin to resolve.
    """
    out: list[str] = list(design.source.get("import_notes", []) if isinstance(design.source, dict) else [])
    bodies = design.bodies
    if not bodies:
        return out + ["The design has no bodies."]
    L = float(design.length_ref)
    h = L / cells_per_ref if cells_per_ref else None
    polys = []
    for b in bodies:
        q = P.as_polygon(b.points)
        polys.append(q)
        if len(q) < 3:
            out.append(f"Body '{b.name}' has fewer than 3 points.")
            continue
        if P.signed_area(q) < 0:
            out.append(f"Body '{b.name}' is clockwise; it should be counter-clockwise "
                       "(re-import or use orient_ccw).")
        hits = P.self_intersections(q)
        if hits:
            out.append(f"Body '{b.name}' crosses itself at {len(hits)} place(s) "
                       f"(first near x = {hits[0][2][0]:.4g} m, y = {hits[0][2][1]:.4g} m); "
                       "the solver would see a wrong shape there.")
        if len(q) < 20 and b.role != "wheel":
            out.append(f"Body '{b.name}' has only {len(q)} points; curved parts will look faceted.")

    # body pairs
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if len(polys[i]) < 3 or len(polys[j]) < 3:
                continue
            if P.polygons_overlap(polys[i], polys[j]):
                extra = (" A wheel overlapping the body is fine for a fixed wheel, but a rotating wheel "
                         "spins only where it is exposed." if "wheel" in (bodies[i].role, bodies[j].role)
                         else "")
                out.append(f"Bodies '{bodies[i].name}' and '{bodies[j].name}' overlap; the solver treats "
                           f"them as one solid.{extra}")
            elif h is not None:
                gap = P.polygon_distance(polys[i], polys[j])
                if gap < 2 * h:
                    out.append(f"The gap between '{bodies[i].name}' and '{bodies[j].name}' is "
                               f"{_fmt_len(gap)}, under 2 grid cells ({_fmt_len(2 * h)}); the solver will "
                               "close it. Use a finer grid or widen the gap.")

    # road
    if design.kind == "vehicle":
        ymin = min(float(q[:, 1].min()) for q in polys if len(q))
        tol = 1e-6 * L
        if ymin < -tol:
            out.append(f"The vehicle extends {_fmt_len(-ymin)} below the road (y < 0); raise it or reduce "
                       "the ground clearance.")
        elif ymin <= tol:
            wheel_touch = any(b.role == "wheel" and float(q[:, 1].min()) <= tol for b, q in zip(bodies, polys))
            what = "The wheels touch" if wheel_touch else "The vehicle touches"
            out.append(f"{what} the road. In 2D this seals the gap under the car completely, so no air "
                       "flows underneath -- usually not what a real car (with air passing between the "
                       "wheels) experiences. Consider removing the wheels or adding clearance.")
        elif h is not None and ymin < 2 * h:
            out.append(f"Ground clearance {_fmt_len(ymin)} is under 2 grid cells ({_fmt_len(2 * h)}); the "
                       "underbody flow will be blocked. Use a finer grid.")

    # resolution: thin features
    if h is not None:
        for b, q in zip(bodies, polys):
            if len(q) < 3:
                continue
            pts, th = _thin_points(q, 2 * h, ds=min(0.5 * h, P.perimeter(q) / 200))
            if not len(pts):
                continue
            x0, x1, _, _ = P.bbox(q)
            span = x1 - x0
            if design.kind == "airfoil" or b.role == "element":
                rear = pts[:, 0] > x1 - 0.15 * span
                if rear.any():
                    xs = float(pts[rear, 0].min())
                    frac = (x1 - xs) / span
                    te_gap = _te_thickness(q)
                    msg = (f"Trailing edge of '{b.name}': the last {100 * frac:.2g}% of the chord is thinner "
                           f"than 2 grid cells ({_fmt_len(2 * h)}), so the solver sees a slightly blunter, "
                           "shorter trailing edge.")
                    if 0 < te_gap < 2 * h:
                        msg += f" The {_fmt_len(te_gap)} blunt trailing-edge base is not resolved."
                    # every sharp TE is under-resolved somewhere; only say so when it matters
                    if frac >= 0.01 or 0 < te_gap < 2 * h:
                        out.append(msg)
                if (~rear).any():
                    k = int(np.argmin(np.where(rear, np.inf, th)))
                    out.append(f"'{b.name}' has a feature only {_fmt_len(float(th[k]))} thick near "
                               f"x = {pts[k, 0]:.4g} m, under 2 grid cells; it will be poorly resolved.")
            else:
                k = int(np.argmin(th))
                out.append(f"'{b.name}' has features thinner than 2 grid cells ({_fmt_len(2 * h)}), the "
                           f"thinnest {_fmt_len(float(th[k]))} near x = {pts[k, 0]:.4g} m, y = {pts[k, 1]:.4g} m; "
                           "they will be smeared or lost. Use a finer grid or thicken them.")
    return out


def _te_thickness(q: np.ndarray) -> float:
    """Blunt TE base height for a Selig-ordered outline (0 for a sharp TE)."""
    d = q[0] - q[-1]
    xr = float(np.ptp(q[:, 0]))
    xmax = float(q[:, 0].max())
    if q[0, 0] > xmax - 0.02 * xr and q[-1, 0] > xmax - 0.02 * xr and abs(d[1]) > abs(d[0]):
        return float(np.hypot(*d))
    return 0.0
