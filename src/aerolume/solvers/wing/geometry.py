"""Wing geometry for the finite-wing solvers: sections, planform metrics, panelling.

Frame (DESIGN.md 7.2): x downstream, y spanwise (``WingSection.y`` >= 0 is the
right half of a symmetric wing), z up; the root leading edge sits at the origin.
A wing is a list of :class:`~aerolume.model.WingSection` stations joined by
straight lines (linear lofting): leading edge, chord and twist vary linearly in
y between two stations, so the leading and trailing edges are straight and the
planform is exactly the one the user typed. Between two stations with different
airfoils the camber line (and, in the viscous coupling, the section polar) is
blended linearly with the same spanwise weight, as XFLR5 does.

Each section is placed by its leading edge ``(x_le, y, z_le)``, rotated nose-up
by ``twist_deg`` about an axis parallel to y through the leading edge, and its
camber offsets are applied along the local surface normal (tilted by the local
dihedral, the slope of z_le in y). The vortex-lattice and lifting-line solvers
only need the *mean surface* (thin-wing theory); thickness enters through the
section polars only.

Why our own panelling instead of :func:`aerolume.geometry3d.loft`: the lattice
needs the camber surface, spanwise nodes exactly at the section stations and a
spacing tuned for the vortex methods, none of which a display/CFD mesh provides.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ...model import WingSection, WingSpec

#: chordwise stations at which camber lines are tabulated (cosine spacing, x/c)
_CAMBER_STATIONS = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, 121)))


class WingGeometryError(ValueError):
    """The wing definition cannot be analysed (message is user-readable)."""


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

@dataclass
class SectionShape:
    """Mean line of one airfoil in its chord-line frame (chord 1, LE at 0, TE at 1).

    ``alpha_l0`` [rad] and ``cm0`` (about c/4) are the thin-airfoil-theory values of
    this camber line: they drive the inviscid lifting line and the first estimate
    of the angles the section polars must cover.
    """
    key: str
    x: np.ndarray
    z: np.ndarray
    alpha_l0: float
    cm0: float
    thickness: float = float("nan")

    def camber(self, xc: np.ndarray) -> np.ndarray:
        """Camber z/c at the chord fractions ``xc``."""
        return np.interp(xc, self.x, self.z)


def _thin_airfoil(x: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Zero-lift angle [rad] and cm about c/4 of a piecewise-linear camber line.

    With x = (1 - cos t) / 2 and a constant slope s_k on each segment, the
    thin-airfoil integrals are exact sums:
    alpha_L0 = -(1/pi) sum s_k [sin t - t],  A_n = (2/pi) sum s_k [sin(n t)/n],
    cm_c/4 = (pi/4)(A2 - A1).
    """
    xs = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    t = np.arccos(1.0 - 2.0 * xs)
    dx = np.diff(xs)
    ok = dx > 1e-12
    s = np.zeros_like(dx)
    s[ok] = np.diff(z)[ok] / dx[ok]
    t0, t1 = t[:-1], t[1:]
    a_l0 = -(1.0 / math.pi) * float(np.sum(s * ((np.sin(t1) - t1) - (np.sin(t0) - t0))))
    a1 = (2.0 / math.pi) * float(np.sum(s * (np.sin(t1) - np.sin(t0))))
    a2 = (2.0 / math.pi) * float(np.sum(s * (np.sin(2 * t1) - np.sin(2 * t0)) / 2.0))
    return a_l0, (math.pi / 4.0) * (a2 - a1)


def section_shape(key: str, coords: np.ndarray) -> SectionShape:
    """Camber line and thin-airfoil properties of an airfoil outline.

    ``coords`` is anything :func:`aerolume.solvers.airfoil.prepare_coords` accepts.
    The camber line is the mean of the upper and lower surfaces at equal x/c in
    the chord-line frame the section polars use, so camber, incidence and the
    polar angles share one reference line.
    """
    from ..airfoil import prepare_coords

    q, _, info = prepare_coords(coords)
    ile = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
    upper = q[:ile + 1][::-1]                 # LE -> TE
    lower = q[ile:]                           # LE -> TE

    def surf(p: np.ndarray) -> np.ndarray:
        xs = np.maximum.accumulate(np.clip(p[:, 0], 0.0, 1.0))
        return np.interp(_CAMBER_STATIONS, xs, p[:, 1])

    zc = 0.5 * (surf(upper) + surf(lower))
    zc[0] = zc[-1] = 0.0                      # chord line ends: LE (0, 0), TE (1, 0)
    a_l0, cm0 = _thin_airfoil(_CAMBER_STATIONS, zc)
    return SectionShape(key=key, x=_CAMBER_STATIONS.copy(), z=zc, alpha_l0=a_l0, cm0=cm0,
                        thickness=float(info.get("thickness", float("nan"))))


def flat_plate(key: str = "flat") -> SectionShape:
    """A zero-camber mean line (used for validation and for symmetric airfoils' geometry)."""
    return SectionShape(key=key, x=_CAMBER_STATIONS.copy(), z=np.zeros_like(_CAMBER_STATIONS),
                        alpha_l0=0.0, cm0=0.0, thickness=0.0)


# ---------------------------------------------------------------------------
# validation and planform metrics
# ---------------------------------------------------------------------------

def check_wing(wing: WingSpec) -> None:
    """Raise :class:`WingGeometryError` with a readable reason when the wing is not analysable."""
    secs = list(getattr(wing, "sections", None) or [])
    if len(secs) < 2:
        raise WingGeometryError("A wing needs at least two sections (root and tip).")
    ys = np.array([float(s.y) for s in secs])
    cs = np.array([float(s.chord) for s in secs])
    vals = np.array([[s.y, s.chord, s.x_le, s.z_le, s.twist_deg] for s in secs], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise WingGeometryError("Every section needs finite y, chord, x_le, z_le and twist values.")
    if np.any(np.diff(ys) <= 0):
        raise WingGeometryError("Section y positions must increase strictly from root to tip.")
    if wing.symmetric and ys[0] < -1e-12:
        raise WingGeometryError("A symmetric wing is defined by its right half: the first section "
                                "must be at y >= 0.")
    if np.any(cs < 0):
        raise WingGeometryError("Section chords must not be negative.")
    if np.any(0.5 * (cs[:-1] + cs[1:]) <= 0):
        raise WingGeometryError("Two neighbouring sections both have zero chord: that part of the wing "
                                "has no area.")
    if np.any(np.abs(vals[:, 4]) >= 45.0):
        raise WingGeometryError("Section twist must be within +/-45 deg.")
    if int(wing.n_chord) < 1 or int(wing.n_span) < 1:
        raise WingGeometryError("n_chord and n_span must be at least 1.")
    if wing.spacing not in ("cosine", "uniform"):
        raise WingGeometryError(f"Unknown spanwise spacing {wing.spacing!r} (use 'cosine' or 'uniform').")


def planform_metrics(wing: WingSpec) -> dict[str, float]:
    """Area, span, aspect ratio, mean aerodynamic chord and friends of a wing.

    ``area`` is the planform area projected on the x-y plane and ``span`` the
    tip-to-tip distance in y (the textbook S and b; with dihedral both are the
    projected values). For a symmetric wing they cover both halves. ``mac`` is
    the mean aerodynamic chord int(c^2 dy) / int(c dy), ``mac_le_x`` / ``mac_y``
    its leading-edge x and its spanwise station (on one half), ``taper`` the tip
    over root chord, ``sweep_c4_deg`` the sweep of the line joining the root and
    tip quarter-chord points. Exact for the piecewise-linear planform.
    """
    check_wing(wing)
    s = wing.sections
    y = np.array([t.y for t in s], float)
    c = np.array([t.chord for t in s], float)
    x = np.array([t.x_le for t in s], float)
    dy = np.diff(y)
    c0, c1, x0, x1, y0, y1 = c[:-1], c[1:], x[:-1], x[1:], y[:-1], y[1:]
    area_h = float(np.sum(0.5 * (c0 + c1) * dy))
    c2 = float(np.sum(dy * (c0 * c0 + c0 * c1 + c1 * c1) / 3.0))
    cx = float(np.sum(dy * (2 * c0 * x0 + c0 * x1 + c1 * x0 + 2 * c1 * x1) / 6.0))
    cy = float(np.sum(dy * (2 * c0 * y0 + c0 * y1 + c1 * y0 + 2 * c1 * y1) / 6.0))
    k = 2.0 if wing.symmetric else 1.0
    area = k * area_h
    span = 2.0 * float(y[-1]) if wing.symmetric else float(y[-1] - y[0])
    mac = c2 / area_h if area_h > 0 else float("nan")
    xc4 = x + 0.25 * c
    sweep = math.degrees(math.atan2(xc4[-1] - xc4[0], y[-1] - y[0]))
    return {
        "area": area,
        "span": span,
        "aspect_ratio": span * span / area if area > 0 else float("nan"),
        "mac": mac,
        "mac_le_x": cx / area_h if area_h > 0 else float("nan"),
        "mac_y": cy / area_h if area_h > 0 else float("nan"),
        "taper": float(c[-1] / c[0]) if c[0] > 0 else float("nan"),
        "sweep_c4_deg": sweep,
        "mean_chord": area / span if span > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# spanwise distribution
# ---------------------------------------------------------------------------

def _spacing_map(kind: str, root_is_edge: bool) -> tuple[Callable[[np.ndarray], np.ndarray],
                                                         Callable[[np.ndarray], np.ndarray], float]:
    """(eta -> theta, theta -> eta, theta_max) for the spanwise node distribution on [0, 1].

    "cosine" clusters nodes where the loading changes fastest: at the tip only
    (eta = sin theta, theta in [0, pi/2]) when the root joins the mirrored half,
    which makes the full wing the classical full-cosine distribution
    y = (b/2) cos theta; at both ends (eta = (1 - cos theta) / 2) when the root is
    a free edge too (a single half wing, or a wing with a gap at the centre).
    """
    if kind == "uniform":
        return (lambda e: e), (lambda t: t), 1.0
    if root_is_edge:
        return (lambda e: np.arccos(np.clip(1.0 - 2.0 * e, -1.0, 1.0)),
                lambda t: 0.5 * (1.0 - np.cos(t)), math.pi)
    return (lambda e: np.arcsin(np.clip(e, 0.0, 1.0))), (lambda t: np.sin(t)), 0.5 * math.pi


def spanwise_nodes(y_sections: Sequence[float], n_span: int, spacing: str, *,
                   root_is_edge: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spanwise nodes on one half: (y nodes, segment index of each panel, control-point fraction).

    The global distribution (cosine or uniform over the whole half span) is
    split at the section stations so every station is a node (kinks in the
    planform fall on panel edges); each segment gets panels in proportion to its
    share of the mapped coordinate, at least one, ``n_span`` in total (more when
    there are more segments than ``n_span``).

    The third array places each panel's control point between its two edges
    (0..1): at the midpoint of the *mapped* coordinate theta, not of y. With
    cosine spacing this is the "semicircle" rule of Multhopp / Weissinger: the
    loading of a tip falls like sqrt(distance), and collocating at the
    theta-midpoint makes the discrete solution of an elliptic wing exactly
    elliptic (span efficiency 1 for any panel count) instead of converging like
    1/n (measured: e = 1.025 at 24 panels per half with y-midpoints).
    """
    ys = np.asarray(y_sections, dtype=float)
    to_t, to_e, _ = _spacing_map(spacing, root_is_edge)
    eta = (ys - ys[0]) / (ys[-1] - ys[0])
    t = to_t(eta)
    share = np.diff(t) / (t[-1] - t[0])
    nseg = len(share)
    n_total = max(int(n_span), nseg)
    raw = share * n_total
    counts = np.maximum(1, np.floor(raw).astype(int))
    while counts.sum() < n_total:                    # largest remainder
        counts[int(np.argmax(raw - counts))] += 1
    while counts.sum() > n_total:
        cand = np.where(counts > 1)[0]
        counts[cand[int(np.argmin((raw - counts)[cand]))]] -= 1
    nodes = [ys[0]]
    seg_of = []
    tn = [t[0]]
    for k in range(nseg):
        tk = np.linspace(t[k], t[k + 1], counts[k] + 1)[1:]
        yk = ys[0] + (ys[-1] - ys[0]) * to_e(tk)
        yk[-1] = ys[k + 1]                          # exact station
        nodes.extend(yk.tolist())
        tn.extend(tk.tolist())
        seg_of.extend([k] * counts[k])
    yn = np.array(nodes)
    tn = np.array(tn)
    ym = ys[0] + (ys[-1] - ys[0]) * to_e(0.5 * (tn[:-1] + tn[1:]))
    frac = np.clip((ym - yn[:-1]) / np.diff(yn), 0.0, 1.0)
    return yn, np.array(seg_of, dtype=int), frac


# ---------------------------------------------------------------------------
# lofted mean surface
# ---------------------------------------------------------------------------

@dataclass
class Strip:
    """Per-strip data (one spanwise column of panels), whole wing, ordered in +y."""
    y: np.ndarray            # strip-centre y [m]
    chord: np.ndarray        # strip-centre chord [m]
    ds: np.ndarray           # strip width along the surface (y-z length) [m]
    dy: np.ndarray           # strip width projected on y [m]
    area: np.ndarray         # chord * ds [m^2]
    twist: np.ndarray        # strip-centre twist [rad]
    sec_a: np.ndarray        # section index blended from ...
    sec_b: np.ndarray        # ... to
    weight: np.ndarray       # weight of sec_b (0..1)
    c4: np.ndarray           # (n, 3) strip-centre quarter-chord point on the chord line
    u_a: np.ndarray          # (n, 3) chord direction (twist included)
    u_n: np.ndarray          # (n, 3) section normal
    cp_frac: np.ndarray      # control-point position between the strip's -y and +y edges (0..1)
    surface: np.ndarray      # index of the lifting surface the strip belongs to


@dataclass
class WingGeometry:
    """Mean surface of the whole wing, ready for the vortex methods.

    ``surfaces`` holds one node array (n_chord + 1, n_nodes, 3) per contiguous
    lifting surface (one for a wing whose halves meet at y = 0, two when there is
    a gap at the centre), nodes ordered in +y. ``c4_nodes`` gives the matching
    quarter-chord-line nodes (n_nodes, 3) for the lifting line; ``cp_points`` /
    ``cp_normals`` (n_chord, n_nodes - 1, 3) the lattice collocation points (3/4
    of each panel, at the semicircle spanwise fraction) on the true camber surface
    and the surface normal there. ``strips`` lists the spanwise columns of all
    surfaces in the same order.
    """
    surfaces: list[np.ndarray]
    c4_nodes: list[np.ndarray]
    cp_points: list[np.ndarray]
    cp_normals: list[np.ndarray]
    strips: Strip
    sections: list[WingSection]
    shapes: list[SectionShape]
    metrics: dict[str, float]
    n_chord: int
    notes: list[str] = field(default_factory=list)

    @property
    def n_strips(self) -> int:
        return len(self.strips.y)


def _interp_sections(wing: WingSpec, y: np.ndarray) -> dict[str, np.ndarray]:
    """Linear lofting quantities at spanwise stations ``y`` of the defined (right) half."""
    s = wing.sections
    ys = np.array([t.y for t in s], float)
    out = {k: np.interp(y, ys, np.array([getattr(t, k) for t in s], float))
           for k in ("chord", "x_le", "z_le", "twist_deg")}
    seg = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
    w = (y - ys[seg]) / (ys[seg + 1] - ys[seg])
    out["seg"] = seg
    out["w"] = np.clip(w, 0.0, 1.0)
    return out


def build_geometry(wing: WingSpec, shapes: Sequence[SectionShape], *, n_chord: int | None = None,
                   n_span: int | None = None, chordwise: str = "uniform") -> WingGeometry:
    """Panel the mean surface of ``wing`` (both halves when symmetric).

    ``shapes`` holds one :class:`SectionShape` per section (same order). The
    chordwise distribution is uniform by default: with the 1/4-3/4 rule it gives
    the exact 2D lift and moment of flat and parabolic mean lines for any panel
    count, which cosine spacing does not.
    """
    check_wing(wing)
    if len(shapes) != len(wing.sections):
        raise ValueError("one section shape per wing section is needed")
    nc = int(n_chord if n_chord is not None else wing.n_chord)
    ns = int(n_span if n_span is not None else wing.n_span)
    secs = list(wing.sections)
    ys = np.array([t.y for t in secs], float)
    root_edge = (not wing.symmetric) or ys[0] > 1e-9 * max(abs(ys[-1]), 1.0)
    yn, seg_of_panel, cp_frac = spanwise_nodes(ys, ns, wing.spacing, root_is_edge=root_edge)
    notes = []
    if len(seg_of_panel) > ns:
        notes.append(f"The wing has {len(ys) - 1} spanwise segments, more than n_span = {ns}: "
                     f"used one panel per segment ({len(seg_of_panel)} per half).")

    # node properties (right / defined half)
    q = _interp_sections(wing, yn)
    q["seg"] = np.concatenate([seg_of_panel[:1], seg_of_panel])   # node j belongs to panel j-1 (or 0)
    seg_node = q["seg"]
    # node weight inside its segment, recomputed with the segment the node is assigned to
    w_node = (yn - ys[seg_node]) / (ys[seg_node + 1] - ys[seg_node])
    dz = np.diff([t.z_le for t in secs])
    dyy = np.diff(ys)
    gam_seg = np.arctan2(dz, dyy)                                    # dihedral per segment
    gam = np.empty(len(yn))
    gam[1:-1] = 0.5 * (gam_seg[seg_of_panel[:-1]] + gam_seg[seg_of_panel[1:]])
    gam[0], gam[-1] = gam_seg[seg_of_panel[0]], gam_seg[seg_of_panel[-1]]
    if wing.symmetric and not root_edge:
        gam[0] = 0.0                                                 # the halves meet: flat at the root
    tau = np.radians(q["twist_deg"])
    xc = (np.arange(nc + 1) / nc) if chordwise == "uniform" else 0.5 * (1 - np.cos(np.linspace(0, math.pi, nc + 1)))

    def camber_at(j: int, xs: np.ndarray) -> np.ndarray:
        k = int(seg_node[j])
        t = float(np.clip(w_node[j], 0.0, 1.0))
        return (1 - t) * shapes[k].camber(xs) + t * shapes[k + 1].camber(xs)

    le = np.column_stack([q["x_le"], yn, q["z_le"]])
    ec = np.column_stack([np.cos(tau), np.zeros_like(tau), -np.sin(tau)])
    nrm0 = np.column_stack([np.sin(tau), np.zeros_like(tau), np.cos(tau)])
    nrm = np.column_stack([nrm0[:, 0], -nrm0[:, 2] * np.sin(gam), nrm0[:, 2] * np.cos(gam)])
    nodes_r = np.empty((nc + 1, len(yn), 3))
    for j in range(len(yn)):
        zc = camber_at(j, xc)
        nodes_r[:, j, :] = le[j] + q["chord"][j] * (xc[:, None] * ec[j] + zc[:, None] * nrm[j])
    c4_r = le + 0.25 * q["chord"][:, None] * ec

    # Collocation points and normals on the true camber surface at 3/4 of each panel, with the
    # local camber slope there. The panel's own corner points would give the secant slope, i.e.
    # the slope at the panel *midpoint*: an O(dx) error in the zero-lift angle (NACA 2412 with
    # 8 chordwise panels: -1.87 deg instead of the thin-airfoil -2.12 deg).
    ycp = yn[:-1] + cp_frac * np.diff(yn)
    qc = _interp_sections(wing, ycp)
    w_c = np.clip((ycp - ys[seg_of_panel]) / (ys[seg_of_panel + 1] - ys[seg_of_panel]), 0.0, 1.0)
    tau_c = np.radians(qc["twist_deg"])
    gam_c = gam[:-1] + cp_frac * np.diff(gam)
    ec_c = np.column_stack([np.cos(tau_c), np.zeros_like(tau_c), -np.sin(tau_c)])
    n0c = np.column_stack([np.sin(tau_c), np.zeros_like(tau_c), np.cos(tau_c)])
    nc_c = np.column_stack([n0c[:, 0], -n0c[:, 2] * np.sin(gam_c), n0c[:, 2] * np.cos(gam_c)])
    le_c = np.column_stack([qc["x_le"], ycp, qc["z_le"]])
    xcp = xc[:-1] + 0.75 * np.diff(xc)
    h = 1e-5
    cp_r = np.empty((nc, len(ycp), 3))
    cpn_r = np.empty((nc, len(ycp), 3))
    for j in range(len(ycp)):
        k, t = int(seg_of_panel[j]), float(w_c[j])

        def cam(xs, _k=k, _t=t):
            return (1 - _t) * shapes[_k].camber(xs) + _t * shapes[_k + 1].camber(xs)
        zc = cam(xcp)
        slope = (cam(np.minimum(xcp + h, 1.0)) - cam(np.maximum(xcp - h, 0.0))) / (
            np.minimum(xcp + h, 1.0) - np.maximum(xcp - h, 0.0))
        cp_r[:, j, :] = le_c[j] + qc["chord"][j] * (xcp[:, None] * ec_c[j] + zc[:, None] * nc_c[j])
        tc = ec_c[j][None, :] + slope[:, None] * nc_c[j][None, :]
        ts = 0.5 * (nodes_r[:-1, j + 1] - nodes_r[:-1, j] + nodes_r[1:, j + 1] - nodes_r[1:, j])
        nv = np.cross(tc, ts)
        cpn_r[:, j, :] = nv / np.linalg.norm(nv, axis=1, keepdims=True)

    # strip (panel column) properties of the right half, before mirroring
    ym = 0.5 * (yn[:-1] + yn[1:])
    qm = _interp_sections(wing, ym)
    seg_m = seg_of_panel
    w_m = (ym - ys[seg_m]) / (ys[seg_m + 1] - ys[seg_m])
    tau_m = np.radians(qm["twist_deg"])
    lem = np.column_stack([qm["x_le"], ym, qm["z_le"]])
    ec_m = np.column_stack([np.cos(tau_m), np.zeros_like(tau_m), -np.sin(tau_m)])
    gm = gam_seg[seg_m]
    n0m = np.column_stack([np.sin(tau_m), np.zeros_like(tau_m), np.cos(tau_m)])
    nm = np.column_stack([n0m[:, 0], -n0m[:, 2] * np.sin(gm), n0m[:, 2] * np.cos(gm)])
    dyz = np.diff(le[:, 1:], axis=0)
    right = {
        "y": ym, "chord": 0.5 * (q["chord"][:-1] + q["chord"][1:]), "ds": np.hypot(dyz[:, 0], dyz[:, 1]),
        "dy": np.abs(np.diff(yn)), "twist": tau_m, "sec_a": seg_m, "sec_b": seg_m + 1, "weight": w_m,
        "c4": lem + 0.25 * qm["chord"][:, None] * ec_m, "u_a": ec_m, "u_n": nm, "cp_frac": cp_frac,
    }

    def mirror(p: np.ndarray) -> np.ndarray:
        m = np.array(p, copy=True)
        m[..., 1] = -m[..., 1]
        return m

    surfaces, c4s, cps, cpns, parts = [], [], [], [], []
    if wing.symmetric:
        left_nodes = mirror(nodes_r[:, ::-1, :])
        left_c4 = mirror(c4_r[::-1])
        left_cp = mirror(cp_r[:, ::-1, :])
        left_cpn = mirror(cpn_r[:, ::-1, :])
        left = {k: (v[::-1].copy() if np.ndim(v) else v) for k, v in right.items()}
        left["y"] = -left["y"]
        left["cp_frac"] = 1.0 - left["cp_frac"]          # edges swap when the order is reversed
        for k in ("c4", "u_a", "u_n"):
            left[k] = mirror(left[k])
        if root_edge:
            surfaces += [left_nodes, nodes_r]
            c4s += [left_c4, c4_r]
            cps += [left_cp, cp_r]
            cpns += [left_cpn, cpn_r]
            parts += [(left, 0), (right, 1)]
        else:
            surfaces.append(np.concatenate([left_nodes[:, :-1, :], nodes_r], axis=1))
            c4s.append(np.concatenate([left_c4[:-1], c4_r], axis=0))
            cps.append(np.concatenate([left_cp, cp_r], axis=1))
            cpns.append(np.concatenate([left_cpn, cpn_r], axis=1))
            parts += [(left, 0), (right, 0)]
    else:
        surfaces.append(nodes_r)
        c4s.append(c4_r)
        cps.append(cp_r)
        cpns.append(cpn_r)
        parts.append((right, 0))
    keys = list(right.keys())
    merged = {k: np.concatenate([p[k] for p, _ in parts], axis=0) for k in keys}
    merged["surface"] = np.concatenate([np.full(len(p["y"]), sid, dtype=int) for p, sid in parts])
    merged["area"] = merged["chord"] * merged["ds"]
    strips = Strip(**merged)
    return WingGeometry(surfaces=surfaces, c4_nodes=c4s, cp_points=cps, cp_normals=cpns, strips=strips, sections=secs, shapes=list(shapes),
                        metrics=planform_metrics(wing), n_chord=nc, notes=notes)


__all__ = [
    "SectionShape",
    "Strip",
    "WingGeometry",
    "WingGeometryError",
    "build_geometry",
    "check_wing",
    "flat_plate",
    "planform_metrics",
    "section_shape",
    "spanwise_nodes",
]
