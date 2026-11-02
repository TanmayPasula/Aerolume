"""Turn imported meshes into a 3D :class:`~aerolume.model.Design`, and check 3D designs.

As in 2D (:mod:`aerolume.geometry.design`), this is where an imported shape gets
its meaning -- units, where the road is, which reference area and length the
coefficients use -- because that is what makes a drag coefficient comparable
with a published one (DESIGN.md 7.2):

* ``kind="vehicle"``: nose at x = 0, centred on y = 0 (the symmetry plane), lowest
  point on the road (z = ``clearance``, default 0: wheels on the road). Moments
  about the road point under the middle of the vehicle.
* ``kind="generic"``: bounding box centred on the origin (or standing on the road
  when ``on_road``). Moments about the bounding-box centre.
* References: ``ref_area`` = frontal area (projection along x, the union of all
  parts), ``length_ref`` = overall length -- or, for a compact bluff body whose
  length is at most 1.5 times its smaller cross-stream size (a sphere, a cube, a
  cylinder across the flow, a plate facing it), that cross-stream size, i.e. the
  diameter or side the published coefficients use; ``span_ref`` = width.
  ``force_ref`` (the 2D field) is set to the frontal height.
* ``symmetric`` when the mirror image across y = 0 is within 0.5 % of the largest
  dimension (RMS, :func:`~aerolume.geometry3d.mesh_ops.symmetry_error`): the
  planner may then simulate half the domain.

:func:`check_design3d` then explains in plain words what will go wrong or is
being assumed: holes and how they are closed, flipped faces, units that look
wrong, a body below the road, parts thinner than the grid, gaps the grid closes.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..model import Design, Mesh3D
from ..solvers.fpmode import ieee_subnormals
from . import mesh_ops as O
from .meshio import guess_units, unit_scale

__all__ = ["check_design3d", "design_from_meshes", "find_wheels3d", "part_gap"]

SYMMETRY_TOL = 0.005          # of the largest dimension
COMPACT_RATIO = 1.5           # length / smaller cross-stream size at or below which a body is "compact"
_WHEEL_WORDS = ("wheel", "tyre", "tire", "rim")


def _fmt_len(v: float) -> str:
    if v >= 1:
        return f"{v:.3g} m"
    if v >= 0.01:
        return f"{v * 100:.3g} cm"
    return f"{v * 1000:.3g} mm"


def find_wheels3d(meshes: Sequence[Mesh3D]) -> list[tuple[tuple, tuple, float] | None]:
    """For each mesh, ``(centre, axis, radius)`` if it looks like a wheel, else None.

    A wheel is named like one ("wheel", "tyre", "tire", "rim") or shaped like
    one: round across a roughly spanwise axle (the two extents across it within
    12 %, the width under 80 % of the diameter), in the lower half of the
    vehicle and smaller than 40 % of its length. Only when there are several
    meshes, so a lone cylinder stays a body.
    """
    ms = list(meshes)
    if len(ms) < 2:
        return [None] * len(ms)
    x0, x1, _, _, z0, z1 = O.bbox(ms)
    L = x1 - x0
    out: list[tuple[tuple, tuple, float] | None] = []
    for m in ms:
        if len(m.faces) < 8:
            out.append(None)
            continue
        named = any(w in m.name.lower() for w in _WHEEL_WORDS)
        try:
            c, ax, r = O.wheel_params(m)
        except ValueError:
            out.append(None)
            continue
        ok = named
        if not ok:
            a = np.asarray(ax)
            e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(a, e1)
            v = m.vertices[np.unique(m.faces)]
            w_ax = float(np.ptp(v @ a))
            d1, d2 = float(np.ptp(v @ e1)), float(np.ptp(v @ e2))
            round_ = min(d1, d2) > 0.88 * max(d1, d2)
            ok = (round_ and w_ax < 0.8 * max(d1, d2) and abs(a[1]) > 0.9
                  and c[2] < 0.5 * (z0 + z1) and 2 * r < 0.4 * L)
        out.append((c, ax, r) if ok else None)
    return out


def design_from_meshes(meshes: Sequence[Mesh3D] | Mesh3D, name: str = "Shape", kind: str = "vehicle", *,
                       length: float | None = None, units: str | float | None = None,
                       on_road: bool | None = None, clearance: float | None = None,
                       fix_winding: bool = True, source: dict | None = None,
                       notes: list[str] | None = None) -> Design:
    """A 3D :class:`Design` (``dimension=3``) from meshes, with references filled in.

    ``units``: None = already metres; a unit name ("mm", "cm", "m", "in", "ft") or
    metres per unit; "auto" applies :func:`~aerolume.geometry3d.meshio.guess_units`.
    ``length`` [m]: scale uniformly so the overall length along x equals it.
    ``on_road`` (default: True for vehicles, False otherwise) puts the lowest
    point at ``clearance`` [m] (default 0) above the road plane z = 0.
    ``fix_winding`` makes face winding consistent and outward
    (:func:`~aerolume.geometry3d.mesh_ops.orient_outward`; vertices untouched,
    so per-vertex results still map onto the user's mesh). Messages about what
    was assumed or repaired go to ``source["import_notes"]`` for
    :func:`check_design3d`; the measured symmetry error, frontal area and scale
    are recorded in ``source`` too.
    """
    if kind not in ("vehicle", "generic"):
        raise ValueError("3D mesh designs are 'vehicle' or 'generic' (wings: design_from_wing)")
    ms = [meshes] if isinstance(meshes, Mesh3D) else list(meshes)
    src = dict(source or {"type": "import"})
    msgs: list[str] = list(src.get("import_notes", [])) + list(notes or [])
    ms = [m for m in ms if len(m.faces)]
    if not ms:
        raise ValueError("no triangles in the input")
    # units, then length
    if isinstance(units, str) and units.strip().lower() == "auto":
        s, why = guess_units(ms)
        msgs.append(why)
    else:
        s = unit_scale(units)
    if s != 1.0:
        ms = O.transform(ms, scale=s)
    x0, x1, y0, y1, z0, z1 = O.bbox(ms)
    if length is not None:
        if not length > 0:
            raise ValueError("length must be positive")
        if not x1 > x0:
            raise ValueError("the shape has no length along x")
        k = length / (x1 - x0)
        ms = O.transform(ms, scale=k)
        s *= k
    if fix_winding:
        fixed = []
        for m in ms:
            before = O.inconsistent_edges(m)
            m2 = O.orient_outward(m)
            flipped = int(np.sum(np.any(m2.faces != m.faces, axis=1)))
            if flipped:
                msgs.append(f"'{m.name}': {flipped} face(s) re-wound so the normals point outward"
                            + (f" ({before} edges had inconsistent winding)." if before else "."))
            fixed.append(m2)
        ms = fixed
    # placement
    road = (kind == "vehicle") if on_road is None else bool(on_road)
    clr = 0.0 if clearance is None else float(clearance)
    if clr < 0:
        raise ValueError("clearance cannot be negative")
    x0, x1, y0, y1, z0, z1 = O.bbox(ms)
    if kind == "vehicle":
        shift = (-x0, -0.5 * (y0 + y1), (clr - z0) if road else -0.5 * (z0 + z1))
    else:
        shift = (-0.5 * (x0 + x1), -0.5 * (y0 + y1), (clr - z0) if road else -0.5 * (z0 + z1))
    ms = O.transform(ms, translate=shift)
    x0, x1, y0, y1, z0, z1 = O.bbox(ms)
    L, W, H = x1 - x0, y1 - y0, z1 - z0
    if kind == "vehicle":
        wheels = find_wheels3d(ms)
        for i, wh in enumerate(wheels):
            if wh is not None and ms[i].role == "body":
                c, ax, _ = wh
                ms[i] = Mesh3D(name=ms[i].name, vertices=ms[i].vertices, faces=ms[i].faces, role="wheel",
                               omega=0.0, center=c, axis=ax)
        nw = sum(m.role == "wheel" for m in ms)
        if nw:
            msgs.append(f"{nw} mesh(es) recognised as wheels (they can spin with the moving road).")
    # references
    A = O.frontal_area(ms, axis="x")
    # compact: not much longer than its smaller cross-stream size (sphere, cube, a
    # cylinder or plate across the flow). A wing (thin and wide) or a car is not.
    cross = min(W, H)
    compact = cross > 0 and L <= COMPACT_RATIO * cross
    length_ref = cross if compact else L
    if not length_ref > 0:
        raise ValueError("the shape has no extent")
    if kind == "vehicle":
        mref = (x0 + 0.5 * L, 0.0, 0.0 if road else z0)
    else:
        mref = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))
    size = max(L, W, H)
    nf = sum(len(m.faces) for m in ms)
    sym_err = O.symmetry_error(ms, plane_y=0.0, n_samples=20000 if nf < 2_000_000 else 8000)
    symmetric = sym_err <= SYMMETRY_TOL * size
    open_parts = [m.name for m in ms if not O.is_watertight(m)]
    if open_parts:
        msgs.append(f"Not closed (holes or open edges): {', '.join(open_parts[:5])}"
                    f"{', ...' if len(open_parts) > 5 else ''}. That is fine: small holes are closed "
                    "automatically when the shape is voxelised.")
    src.update({
        "scale_to_m": float(s),
        "frontal_area": float(A),
        "symmetry_error": float(sym_err),
        "compact": bool(compact),
        "on_road": bool(road),
    })
    if compact:
        msgs.append(f"Compact bluff body: the reference length is its cross-stream size "
                    f"{_fmt_len(length_ref)} (not the length {_fmt_len(L)}).")
    if msgs:
        src["import_notes"] = msgs
    return Design(name=name, kind=kind, bodies=[], length_ref=float(length_ref), force_ref=float(H),
                  dimension=3, meshes=ms, ref_area=float(A), span_ref=float(W),
                  moment_ref3=tuple(float(v) for v in mref), symmetric=bool(symmetric), source=src)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def _thin_parts(meshes: list[Mesh3D], limit: float, max_samples: int = 60000
                ) -> tuple[float, np.ndarray | None, float]:
    """(thinnest thickness, where, thin area fraction) of solid regions thinner than ``limit``.

    Surface samples with outward normals (winding made consistent first) are
    paired with samples across the solid: closer than ``limit``, facing nearly
    the opposite way (normals more than ~140 degrees apart, so plates, fins and
    trailing edges count but an ordinary 50-90 degree edge of a body does not),
    and behind the sample along its inward normal. Returns
    ``(inf, None, 0)`` when nothing is that thin.
    """
    from scipy.spatial import cKDTree

    oriented = [O.orient_outward(m) for m in meshes]
    a = O.area(oriented)
    if a <= 0:
        return math.inf, None, 0.0
    n = int(min(max_samples, max(2000, a / (0.4 * limit) ** 2)))
    pts, mid, fid = O.sample_surface(oriented, n, seed=3)
    nrm = np.concatenate([O.face_normals(m) for m in oriented])
    offs = np.cumsum([0] + [len(m.faces) for m in oriented])[:-1]
    nn = nrm[offs[mid] + fid]
    with ieee_subnormals():                 # k-d trees crash under flush-to-zero (fpmode)
        pairs = cKDTree(pts).query_pairs(limit, output_type="ndarray")
    if len(pairs) == 0:
        return math.inf, None, 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    d = pts[j] - pts[i]
    dist = np.linalg.norm(d, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        across = (np.einsum("ij,ij->i", nn[i], nn[j]) < -0.75) & \
                 (np.einsum("ij,ij->i", d, nn[i]) < -0.5 * dist) & (dist > 0)
    if not across.any():
        return math.inf, None, 0.0
    k = int(np.argmin(np.where(across, dist, np.inf)))
    thin = np.zeros(n, dtype=bool)
    thin[i[across]] = True
    thin[j[across]] = True
    return float(dist[k]), 0.5 * (pts[i[k]] + pts[j[k]]), float(thin.mean())


def _inside_closed(points: np.ndarray, m: Mesh3D, k: int = 16) -> np.ndarray:
    """Whether points lie inside a closed, outward-wound part: (p - q) points against the face
    normal at the closest point q (among the k faces with the nearest centroids). Points whose
    closest point is an edge can be misjudged (a sharp trailing edge), so callers use the
    *fraction* of inside points, never a single one."""
    from scipy.spatial import cKDTree

    tri = m.vertices[m.faces]
    kk = min(k, len(tri))
    with ieee_subnormals():                 # k-d trees crash under flush-to-zero (solvers/fpmode.py)
        _, nn = cKDTree(tri.mean(1)).query(points, k=kk)
    nn = nn.reshape(len(points), kk)
    best = np.full(len(points), np.inf)
    side = np.zeros(len(points))
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    for j in range(kk):
        f = nn[:, j]
        q = O.closest_points_on_triangles(points, tri[f, 0], tri[f, 1], tri[f, 2])
        dist = np.linalg.norm(points - q, axis=1)
        closer = dist < best
        best = np.where(closer, dist, best)
        cos = np.einsum("ij,ij->i", points - q, fn[f]) / np.maximum(dist, 1e-300)
        side = np.where(closer, cos, side)
    return side < -0.3


def _one_way_gap(a: Mesh3D, b: Mesh3D, margin: float, size: float, samples: int) -> float | None:
    """Smallest distance from samples of ``a`` near ``b`` to ``b``'s surface: inf when no sample
    of ``a`` lies within ``margin`` of ``b``'s bounding box, None when they touch or ``a``
    pushes into ``b`` (not a gap)."""
    lo = np.asarray(b.bounds()[0::2]) - margin
    hi = np.asarray(b.bounds()[1::2]) + margin
    pts, _, _ = O.sample_surface(a, samples, seed=7)
    near = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if not len(near):
        return math.inf
    d = O._surface_distance(near, [b], n_samples=2 * samples)
    if not len(d) or float(d.min()) <= 1e-6 * size:
        return None                                       # touching
    if O.is_watertight(b) and _inside_closed(near, b).sum() >= max(5, 0.02 * len(near)):
        return None                                       # interpenetrating
    return float(d.min())


def part_gap(a: Mesh3D, b: Mesh3D, margin: float, *, size: float | None = None, samples: int = 3000) -> float:
    """Smallest clearance between the surfaces of two parts, or inf.

    Inf when their bounding boxes are farther apart than ``margin`` (nothing is
    sampled then), when they touch, and when one pushes into the other (a mirror
    pushed into the body is not a gap). Otherwise the smaller of the two one-way
    distances, each measured only from the samples of one part inside the other's
    bounding box grown by ``margin``: the small part's samples resolve the gap
    where the big part's would be sparse, and distant parts cost nothing.
    ``size`` sets the touching tolerance (1e-6 of it; default: the larger part).
    """
    ba, bb = a.bounds(), b.bounds()
    apart = max(ba[0] - bb[1], bb[0] - ba[1], ba[2] - bb[3], bb[2] - ba[3], ba[4] - bb[5], bb[4] - ba[5])
    if apart > margin:
        return math.inf
    if size is None:
        size = max(ba[1] - ba[0], ba[3] - ba[2], ba[5] - ba[4], bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
    g1 = _one_way_gap(a, b, margin, size, samples)
    if g1 is None:
        return math.inf
    g2 = _one_way_gap(b, a, margin, size, samples)
    if g2 is None:
        return math.inf
    g = min(g1, g2)
    return g if g <= margin else math.inf


def _check_wing(design: Design, cells_per_ref: float | None) -> list[str]:
    from .wing import naca_code, planform

    out: list[str] = []
    w = design.wing
    if w is None:
        return ["The wing design has no planform (sections)."]
    try:
        pf = planform(w)
    except ValueError as e:
        return [f"The wing planform is invalid: {e}."]
    for s in w.sections:
        a = str(s.airfoil)
        if a.startswith("design:"):
            continue
        if naca_code(a) is None:
            out.append(f"Section at y = {s.y:.4g} m: airfoil {a!r} is neither a NACA code nor "
                       "'design:<id>'.")
        if abs(s.twist_deg) > 15:
            out.append(f"Section at y = {s.y:.4g} m is twisted {s.twist_deg:.3g} degrees; the "
                       "linear (VLM) analysis loses accuracy for twists this large.")
    ar = pf["aspect_ratio"]
    if ar < 1.5:
        out.append(f"Aspect ratio {ar:.3g} is very low: vortex-lattice and lifting-line results "
                   "are rough for such wings (strong tip vortices, vortex lift); prefer 3D CFD.")
    elif ar > 40:
        out.append(f"Aspect ratio {ar:.3g} is unusually high; check the span and chords.")
    if abs(pf["sweep_c4_deg"]) > 55:
        out.append(f"Quarter-chord sweep {pf['sweep_c4_deg']:.3g} degrees: lifting-line theory is "
                   "not meant for strongly swept wings; use the vortex lattice.")
    if pf["taper"] < 0.1:
        out.append(f"Taper ratio {pf['taper']:.2g}: a nearly pointed tip; expect early tip stall.")
    if w.n_span < 8 or w.n_chord < 4:
        out.append(f"Coarse lattice ({w.n_chord} chordwise x {w.n_span} spanwise panels per half): "
                   "use at least 4 x 8 for usable results, 8 x 24 for good ones.")
    if cells_per_ref:
        h = design.length_ref / cells_per_ref
        tip = w.sections[-1]
        code = naca_code(tip.airfoil)
        tc = int(code[-2:]) / 100.0 if code else 0.12
        th = tc * tip.chord
        if th < 2 * h:
            out.append(f"The tip section is only {_fmt_len(th)} thick, under 2 grid cells "
                       f"({_fmt_len(2 * h)}) of a 3D CFD grid at this resolution; it will be "
                       "thickened to about a cell and its flow poorly resolved.")
    return out


def check_design3d(design: Design, cells_per_ref: float | None = None) -> list[str]:
    """Plain-language problems and caveats of a 3D design (empty list = all clear).

    Always checked: importer notes, missing meshes, units that look wrong, holes
    / non-manifold edges / flipped faces per mesh, references, a vehicle below
    the road, a ``symmetric`` flag that the geometry contradicts. With
    ``cells_per_ref`` (cells per ``length_ref``) also the resolution: too few
    cells across the body, parts and local features thinner than 2 cells, gaps
    between parts or under the vehicle narrower than 2 cells. Wings get planform
    checks instead (sections, airfoils, aspect ratio, sweep, lattice size).
    """
    out: list[str] = list(design.source.get("import_notes", []) if isinstance(design.source, dict) else [])
    if design.dimension != 3:
        return out + ["This is a 2D design; use check_design."]
    if design.kind == "wing" and design.wing is not None and not design.meshes:
        return out + _check_wing(design, cells_per_ref)
    meshes = [m for m in design.meshes if len(m.faces)]
    if not meshes:
        if design.meshes:
            return out + ["The meshes have no triangles loaded (their arrays are missing)."]
        return out + ["The design has no meshes."]
    x0, x1, y0, y1, z0, z1 = O.bbox(meshes)
    ext = np.array([x1 - x0, y1 - y0, z1 - z0])
    size = float(ext.max())
    if size < 0.01:
        out.append(f"The model is only {_fmt_len(size)} across: check the units (a model drawn in "
                   "millimetres but read as metres would be 1000 times too big, the reverse this small).")
    elif size > 300:
        out.append(f"The model is {size:.4g} m across: check the units (millimetres read as metres?).")
    for m in meshes:
        info = O.mesh_info(m)
        if not info["watertight"]:
            if info["boundary_edges"]:
                out.append(f"Mesh '{m.name}' is not closed: {info['holes']} hole(s), {info['boundary_edges']} "
                           "open edges. Small holes are closed automatically; a large opening (a missing "
                           "panel) is closed flat across its rim, so check the solid preview.")
            if info["nonmanifold_edges"]:
                out.append(f"Mesh '{m.name}' has {info['nonmanifold_edges']} edges shared by more than two "
                           "faces (internal walls or parts touching along an edge); the voxeliser copes, "
                           "but the model may not be what you expect.")
        if info["inconsistent_edges"]:
            out.append(f"Mesh '{m.name}': {info['inconsistent_edges']} edges join faces with opposite "
                       "winding (flipped normals). The solid is found without normals, but surface "
                       "plots need them: re-import with winding repair (orient_outward).")
        if info["degenerate_faces"]:
            out.append(f"Mesh '{m.name}' has {info['degenerate_faces']} zero-area triangle(s); they are ignored.")
    if not design.ref_area > 0:
        out.append("The reference area is not set (Auto uses the frontal area).")
    if not design.length_ref > 0:
        out.append("The reference length is not set.")
    if not design.span_ref > 0:
        out.append("The span / width reference is not set (used for roll and yaw moments).")
    L = float(design.length_ref) if design.length_ref > 0 else size
    h = L / cells_per_ref if cells_per_ref else None
    tol = 1e-6 * size
    bodies = [m for m in meshes if m.role != "wheel"]
    if design.kind == "vehicle":
        if z0 < -tol:
            out.append(f"The vehicle extends {_fmt_len(-z0)} below the road (z < 0); raise it.")
        else:
            body_low = min(m.bounds()[4] for m in bodies) if bodies else z1
            wheels_touch = any(m.role == "wheel" and m.bounds()[4] <= tol for m in meshes)
            if body_low <= tol and not wheels_touch:
                out.append("The body touches the road. In 3D air still flows around the contact, but "
                           "the contact patch is solid; add clearance if that is not intended.")
            elif h is not None and tol < body_low < 2 * h:
                out.append(f"Ground clearance {_fmt_len(body_low)} is under 2 grid cells ({_fmt_len(2 * h)}); "
                           "the underbody flow will be blocked. Use a finer grid or raise the body.")
    if design.symmetric:
        nf = sum(len(m.faces) for m in meshes)
        err = O.symmetry_error(meshes, plane_y=0.0, n_samples=20000 if nf < 2_000_000 else 6000)
        if err > SYMMETRY_TOL * size:
            out.append(f"The design is marked symmetric, but its mirror image across y = 0 is off by "
                       f"{_fmt_len(err)} (RMS); a half-domain simulation would model a different shape. "
                       "Centre it on y = 0 or clear the symmetric flag.")
    if h is not None:
        H = float(ext[2])
        if H / h < 12:
            out.append(f"Only {H / h:.3g} grid cells across the height of the body; forces need at "
                       "least ~16 (Draft) to be indicative.")
        for m in meshes:
            b = m.bounds()
            t = min(b[1] - b[0], b[3] - b[2], b[5] - b[4])
            if t < 2 * h:
                out.append(f"'{m.name}' is only {_fmt_len(t)} thick, under 2 grid cells ({_fmt_len(2 * h)}); "
                           "it will be thickened to about one cell and poorly resolved.")
        thin, where, frac = _thin_parts(meshes, 2 * h)
        if math.isfinite(thin) and frac > 0.003:
            out.append(f"Parts of the surface ({100 * frac:.2g}% of its area) enclose solid thinner than "
                       f"2 grid cells ({_fmt_len(2 * h)}); the thinnest is {_fmt_len(thin)} near "
                       f"x = {where[0]:.4g}, y = {where[1]:.4g}, z = {where[2]:.4g} m. Thin edges, "
                       "fins and wing trailing edges there will be smeared.")
        # gaps between parts: only pairs whose bounding boxes come within 2 cells are sampled,
        # and only near each other (part_gap), so the cost no longer grows with every pair
        parts = [m for m in meshes if m.role != "wheel"]
        if 1 < len(parts) <= 64:
            for a in range(len(parts)):
                for b in range(a + 1, len(parts)):
                    gap = part_gap(parts[a], parts[b], 2 * h, size=size, samples=4000)
                    if tol < gap < 2 * h:
                        out.append(f"The gap between '{parts[a].name}' and '{parts[b].name}' is "
                                   f"{_fmt_len(gap)}, under 2 grid cells ({_fmt_len(2 * h)}); the grid "
                                   "will close it.")
    return out
