"""Finite wings: planform metrics, lofting a :class:`~aerolume.model.WingSpec` to a mesh.

Wing frame (DESIGN.md 7.2): x downstream, y spanwise (right wing at +y), z up,
root leading edge at the origin. A :class:`~aerolume.model.WingSection` places a
unit-chord airfoil: scaled by ``chord``, rotated nose-up by ``twist_deg`` about
its leading edge (positive twist lowers the trailing edge), moved to
``(x_le, y, z_le)``. Sweep and dihedral come from how ``x_le`` and ``z_le``
change with ``y``. ``symmetric`` mirrors the half wing about y = 0.

Between stations the surface is *ruled*: every outline point moves on a straight
line from one station to the next, as in XFLR5 and a CAD straight loft. That
keeps the leading and trailing edges straight within a panel, so the lofted
mesh projects exactly onto the trapezoids :func:`planform` integrates, and the
vortex-lattice planform (solvers.wing) and the CFD geometry agree.

:func:`planform` reports the reference quantities a wing analysis is quoted on
(area, span, aspect ratio, mean aerodynamic chord and its position, taper,
quarter-chord sweep), computed exactly for piecewise-linear chord and leading
edge: per panel ``int c^2 dy = dy (c1^2 + c1 c2 + c2^2) / 3`` and so on.
Areas are projected on the x-y plane (dihedral does not add area), the usual
reference-area convention. The shared quantities come from the wing solver's
:func:`aerolume.solvers.wing.planform_metrics` itself, so a design's references
(``design_from_wing``) and the solver's defaults are the same numbers by
construction; this module adds the geometry-only extras.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..model import Design, Mesh3D, WingSection, WingSpec

__all__ = ["design_from_wing", "loft", "naca_code", "planform", "section_outline", "tapered_wing"]

AirfoilProvider = Callable[[str], np.ndarray]


def naca_code(airfoil: str) -> str | None:
    """The 4- or 5-digit NACA designation in ``airfoil`` ("NACA 2412", "2412", "naca-23012"), else None."""
    s = str(airfoil).strip().upper().replace("NACA", "").replace(" ", "").replace("-", "")
    return s if s.isdigit() and len(s) in (4, 5) else None


def _sections(wing: WingSpec):
    secs = list(wing.sections)
    if len(secs) < 2:
        raise ValueError("a wing needs at least two sections (root and tip)")
    y = np.array([s.y for s in secs], dtype=float)
    if np.any(np.diff(y) <= 0):
        raise ValueError("wing sections must be ordered root to tip with strictly increasing y")
    if any(not s.chord > 0 for s in secs):
        raise ValueError("every wing section needs a positive chord")
    if wing.symmetric and y[0] < 0:
        raise ValueError("a symmetric wing is described by its right half (y >= 0)")
    return secs


def planform(wing: WingSpec) -> dict[str, float]:
    """Planform metrics of the whole wing (both halves when symmetric).

    Keys: ``area`` [m²] (projected on the x-y plane), ``span`` [m] (tip to tip in
    y), ``aspect_ratio``, ``mac`` (mean aerodynamic chord, int c² dy / int c dy),
    ``mac_le_x`` (leading edge of the MAC: chord-weighted mean of x_le), ``mac_y``
    (its spanwise station on the right half, the centroid of the half planform),
    ``taper`` (tip / root chord), ``sweep_c4_deg`` (root-to-tip quarter-chord
    line), ``mean_chord`` (area / span) -- all from
    :func:`aerolume.solvers.wing.planform_metrics`, the definitions the wing solver
    uses -- plus ``mac_le_z`` (chord-weighted mean of z_le), ``sweep_le_deg`` and
    ``dihedral_deg`` (root-to-tip leading edge), ``root_chord`` and ``tip_chord``.
    Raises ``ValueError`` (the solver's ``WingGeometryError``) for a wing it cannot
    analyse.
    """
    # imported here: the wing solver pulls in scipy and the airfoil solver (~0.2 s), which
    # the GUI should not pay at start-up for importing geometry3d
    from ..solvers.wing.geometry import planform_metrics

    out = dict(planform_metrics(wing))
    secs = list(wing.sections)
    y = np.array([s.y for s in secs], dtype=float)
    c = np.array([s.chord for s in secs], dtype=float)
    x = np.array([s.x_le for s in secs], dtype=float)
    z = np.array([s.z_le for s in secs], dtype=float)
    dy = np.diff(y)
    c1, c2 = c[:-1], c[1:]
    s_half = float(np.sum(0.5 * (c1 + c2) * dy))
    # int c z dy for c, z linear on each panel (exact), over the half area: as mac_le_x
    cz = float(np.sum(dy * (2 * c1 * z[:-1] + c1 * z[1:] + c2 * z[:-1] + 2 * c2 * z[1:]) / 6.0))
    run = float(y[-1] - y[0])
    out.update({
        "mac_le_z": cz / s_half if s_half > 0 else float("nan"),
        "sweep_le_deg": math.degrees(math.atan2(x[-1] - x[0], run)),
        "dihedral_deg": math.degrees(math.atan2(z[-1] - z[0], run)),
        "root_chord": float(c[0]),
        "tip_chord": float(c[-1]),
    })
    return out


def section_outline(airfoil: str, n_chord: int = 40, *, airfoil_coords: AirfoilProvider | None = None
                    ) -> np.ndarray:
    """Unit-chord outline ``(2 n + 1, 2)``: upper TE -> LE (index n) -> lower TE.

    NACA codes come from the :mod:`aerolume.geometry` generators with ``n_chord``
    cosine-spaced panels per side (closed trailing edge: the first and last
    points coincide). Any other name ("design:<id>", a file stem, ...) is
    resolved by ``airfoil_coords(name)``, normalised (LE at the origin, TE at
    (1, 0)) and resampled at the same cosine stations per surface, so every
    section of a wing has the same point count and can be lofted point to point.
    A blunt trailing edge keeps its two distinct end points.
    """
    from ..geometry.airfoils import cosine_spacing, naca

    n = max(int(n_chord), 4)
    code = naca_code(airfoil)
    if code is not None:
        p = naca(code, n=2 * n, closed_te=True)
        return np.vstack([p, p[:1]])
    if airfoil_coords is None:
        raise ValueError(f"airfoil {airfoil!r} is not a NACA code; pass airfoil_coords to resolve it")
    from ..geometry.design import normalize_airfoil

    q = normalize_airfoil(np.asarray(airfoil_coords(airfoil), dtype=float), 1.0)
    ile = int(np.argmin(q[:, 0] ** 2 + q[:, 1] ** 2))
    upper = q[:ile + 1][::-1]                       # LE -> upper TE
    lower = q[ile:]                                 # LE -> lower TE
    if len(upper) < 2 or len(lower) < 2:
        raise ValueError(f"airfoil {airfoil!r}: could not split the outline at the leading edge")
    xs = cosine_spacing(n)

    def interp(s: np.ndarray) -> np.ndarray:
        xx = np.maximum.accumulate(s[:, 0])
        return np.interp(xs, xx, s[:, 1])

    yu, yl = interp(upper), interp(lower)
    yu[0] = yl[0] = 0.0
    up = np.column_stack([xs, yu])[::-1]            # TE -> LE
    lo = np.column_stack([xs, yl])[1:]              # after LE -> TE
    out = np.vstack([up, lo])
    if abs(out[0, 1] - out[-1, 1]) < 1e-9:
        out[-1] = out[0]
    return out


def _place(outline: np.ndarray, sec) -> np.ndarray:
    """Unit outline (K, 2) -> 3D points of a section (wing frame)."""
    th = math.radians(sec.twist_deg)
    c, s = math.cos(th), math.sin(th)
    x = outline[:, 0] * sec.chord
    z = outline[:, 1] * sec.chord
    xr = x * c + z * s                  # nose-up rotation about the LE (+theta about +y)
    zr = -x * s + z * c
    return np.column_stack([xr + sec.x_le, np.full(len(x), float(sec.y)), zr + sec.z_le])


def _piece(stations: list[np.ndarray], sharp: bool, cap_start: bool, cap_end: bool
           ) -> tuple[np.ndarray, np.ndarray, int]:
    """Mesh a ruled surface through station outlines (each (2n+1, 3), y ascending).

    Returns (vertices, faces, ring size K); station s owns vertices s*K .. s*K+K-1.
    Caps are strip-triangulated between matching upper and lower stations, which
    stays inside any section shape (a fan from the nose would not on a cambered,
    concave lower surface).
    """
    K = stations[0].shape[0] - (1 if sharp else 0)          # distinct ring points per station
    n = (stations[0].shape[0] - 1) // 2
    S = len(stations)
    V = np.concatenate([st[:K] for st in stations])
    i = np.arange(K)
    ip = (i + 1) % K
    faces = []
    for s in range(S - 1):
        A, B = s * K + i, s * K + ip
        C, D = (s + 1) * K + ip, (s + 1) * K + i
        faces.append(np.stack([A, C, B], 1))
        faces.append(np.stack([A, D, C], 1))
    k = np.arange(n)
    up, up1 = (n - k) % K, (n - k - 1) % K
    lo, lo1 = (n + k) % K, (n + k + 1) % K
    cap = np.concatenate([np.stack([up, up1, lo1], 1), np.stack([up, lo1, lo], 1)])
    bad = (cap[:, 0] == cap[:, 1]) | (cap[:, 1] == cap[:, 2]) | (cap[:, 0] == cap[:, 2])
    cap = cap[~bad]
    for s, want, on in ((0, -1.0, cap_start), (S - 1, 1.0, cap_end)):
        if not on:
            continue
        cf = cap + s * K
        t = V[cf]
        nsum = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]).sum(0)
        if nsum[1] * want < 0:
            cf = cf[:, [0, 2, 1]]
        faces.append(cf)
    return V, np.concatenate(faces), K


def loft(wing: WingSpec, *, n_chord: int = 40, n_span_per_panel: int | None = None, closed_tips: bool = True,
         airfoil_coords: AirfoilProvider | None = None, name: str = "Wing") -> Mesh3D:
    """Triangle mesh of a wing: ruled surface through its sections, closed tips.

    ``n_chord``: panels per airfoil side; ``n_span_per_panel``: spanwise
    subdivisions between two sections (default: about one per half mean chord
    of the panel, 1-32). A symmetric wing whose root is at y = 0 is one
    continuous surface across the centre line; with a root gap (y > 0) it is two
    closed halves. Watertight for sharp and blunt trailing edges, outward
    winding. ``airfoil_coords`` resolves non-NACA airfoil names (see
    :func:`section_outline`).
    """
    secs = _sections(wing)
    outl = [section_outline(s.airfoil, n_chord, airfoil_coords=airfoil_coords) for s in secs]
    # sharp everywhere: the trailing edge is one shared vertex row; otherwise every
    # outline keeps both TE points (a sharp one then gets a zero-width base strip)
    sharp = all(np.allclose(o[0], o[-1], atol=1e-9) for o in outl)
    pts = [_place(o, s) for o, s in zip(outl, secs)]
    stations = [pts[0]]
    for a in range(len(secs) - 1):
        dy = secs[a + 1].y - secs[a].y
        cm = 0.5 * (secs[a].chord + secs[a + 1].chord)
        ns = n_span_per_panel or int(min(32, max(1, math.ceil(dy / (0.5 * cm)))))
        for t in np.linspace(0.0, 1.0, ns + 1)[1:]:
            stations.append((1.0 - t) * pts[a] + t * pts[a + 1])
    root_on_axis = wing.symmetric and abs(secs[0].y) <= 1e-12 * max(1.0, secs[-1].y)
    V, F, K = _piece(stations, sharp, cap_start=closed_tips and not root_on_axis, cap_end=closed_tips)
    if wing.symmetric:
        # the left half is the mirror image of the right half's *mesh* (not a second
        # loft): its non-planar quads must be split along mirrored diagonals, or the
        # wing would not be exactly symmetric for a half-domain simulation
        Vm = V * np.array([1.0, -1.0, 1.0]) + 0.0
        Fm = F[:, [0, 2, 1]]                                  # a reflection reverses the winding
        if root_on_axis:                                      # share the root ring (y = 0)
            remap = np.concatenate([np.arange(K), len(V) + np.arange(len(V) - K)])
            F = np.concatenate([F, remap[Fm]])
            V = np.concatenate([V, Vm[K:]])
        else:
            F = np.concatenate([F, Fm + len(V)])
            V = np.concatenate([V, Vm])
    # the side the outlines are wound to is fixed by the airfoil convention; check it anyway
    t = V[F]
    c0 = V.mean(0)
    vol = np.einsum("ij,ij->i", t[:, 0] - c0, np.cross(t[:, 1] - c0, t[:, 2] - c0)).sum()
    if closed_tips and vol < 0:
        F = F[:, [0, 2, 1]]
    return Mesh3D(name=name, vertices=V, faces=F)


def tapered_wing(span: float, root_chord: float, tip_chord: float | None = None, *, sweep_deg: float = 0.0,
                 dihedral_deg: float = 0.0, twist_deg: float = 0.0, airfoil: str = "2412",
                 tip_airfoil: str | None = None, symmetric: bool = True, n_chord: int = 8,
                 n_span: int = 24) -> WingSpec:
    """A straight-tapered wing from the numbers a person quotes: span, chords, sweep.

    One panel from the root (y = 0) to the tip: ``span`` [m] tip to tip (the
    half span when ``symmetric`` is False), ``root_chord`` / ``tip_chord`` [m]
    (``None`` = untapered). ``sweep_deg`` is the **quarter-chord** sweep, the
    aerodynamic one, so ``planform(w)["sweep_c4_deg"]`` returns it;
    ``dihedral_deg`` raises the tip leading edge by ``tan(dihedral)`` per metre
    of span; ``twist_deg`` is the tip twist relative to the root (negative =
    washout). ``tip_airfoil`` defaults to ``airfoil`` (NACA codes or
    "design:<id>"). The command line, the scripting API and the planform
    editor build their wings here, so "sweep 10°" is the same wing in each.
    Raises ``ValueError`` for non-positive sizes or a sweep / dihedral of 80° or
    more (a planform no lattice resolves).
    """
    vals = {"span": span, "root_chord": root_chord, "tip_chord": root_chord if tip_chord is None else tip_chord}
    for key, v in vals.items():
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{key} must be a positive length in metres, got {v!r}")
    for key, v in (("sweep_deg", sweep_deg), ("dihedral_deg", dihedral_deg), ("twist_deg", twist_deg)):
        if not math.isfinite(float(v)):
            raise ValueError(f"{key} must be a finite angle, got {v!r}")
    if abs(sweep_deg) >= 80.0 or abs(dihedral_deg) >= 80.0:
        raise ValueError("sweep and dihedral must stay below 80 degrees")
    c_root, c_tip = float(vals["root_chord"]), float(vals["tip_chord"])
    semi = 0.5 * float(span) if symmetric else float(span)
    # quarter-chord line swept by sweep_deg: x_le(tip) + c_tip/4 = c_root/4 + semi tan(sweep)
    x_tip = semi * math.tan(math.radians(sweep_deg)) + 0.25 * (c_root - c_tip)
    z_tip = semi * math.tan(math.radians(dihedral_deg))
    root = WingSection(0.0, c_root, airfoil=str(airfoil))
    tip = WingSection(semi, c_tip, x_le=x_tip, z_le=z_tip, twist_deg=float(twist_deg),
                      airfoil=str(tip_airfoil if tip_airfoil is not None else airfoil))
    return WingSpec(sections=[root, tip], symmetric=bool(symmetric), n_chord=int(n_chord), n_span=int(n_span))


def design_from_wing(wing: WingSpec, name: str = "Wing", *, moment_ref: tuple[float, float, float] | None = None,
                     source: dict | None = None) -> Design:
    """A ``kind="wing"`` 3D :class:`Design` carrying the planform (lofted on demand).

    References from :func:`planform`: ``ref_area`` = planform area, ``length_ref``
    = MAC (also ``force_ref``, the 2D field), ``span_ref`` = span, moments about
    the quarter-MAC point ``(mac_le_x + mac / 4, 0, mac_le_z)`` unless
    ``moment_ref`` is given. ``symmetric`` follows the planform's mirroring.
    """
    pf = planform(wing)
    mref = moment_ref if moment_ref is not None else (pf["mac_le_x"] + 0.25 * pf["mac"], 0.0, pf["mac_le_z"])
    src = {"type": "wing"}
    src.update(source or {})
    return Design(name=name, kind="wing", bodies=[], length_ref=pf["mac"], force_ref=pf["mac"],
                  dimension=3, meshes=[], wing=wing, ref_area=pf["area"], span_ref=pf["span"],
                  moment_ref3=tuple(float(v) for v in mref), symmetric=bool(wing.symmetric), source=src)
