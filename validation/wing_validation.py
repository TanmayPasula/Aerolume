"""Validation of the finite-wing solvers (vortex lattice, nonlinear lifting line, viscous strips).

Run ``python validation/wing_validation.py`` to recompute every case, print the
comparison tables of docs/validation_wing.md and render docs/img/wing_*.png
(matplotlib Figure + Agg canvas, no pyplot). ``tests/test_wing_validation.py``
imports the case functions and the reference data from here and asserts on the
same numbers, so the document and the tests cannot drift apart.

Reference data (only the values that are compared are transcribed):

* Lifting-line theory, elliptic planform: CL_alpha = 2 pi AR / (AR + 2), e = 1
  (Prandtl; any aerodynamics text, e.g. Anderson, Fundamentals of Aerodynamics,
  ch. 5).
* Circular wing (AR 4/pi), lifting-surface theory: dCL/dalpha = 1.799 per radian
  (H. Multhopp, "Methods for calculating the lift distribution of wings
  (subsonic lifting-surface theory)", ARC R&M 2884, 1950, section 8; Kinner's
  series solution recomputed there converges to 1.804).
* Rectangular and swept constant-chord wings, lifting-surface theory: V. M.
  Falkner, "Calculated loadings due to incidence of a number of straight and
  swept-back wings", ARC R&M 2596, 1948/1952, Tables 5-34 (dCL/dalpha per radian;
  rectangular AR 6: 4.270 / 4.247 (84-vortex 9- and 6-point), 4.231 (Blenk,
  revised by Argyris), 4.527 (4-term lifting line); AR 4: 3.639; AR 2: 2.524;
  AR 1: 1.489; AR 6 with 30 deg sweep: 3.955 (standard) / 3.877 (centre
  "rounded off"); AR 6, 45 deg: 3.464 (standard), 3.367 / 3.365 (modified), 3.304
  (8-point); AR 4, 30 deg: 3.467 / 3.400; AR 4, 45 deg: 3.121 (standard), 3.039
  (modified), 2.986 (8-point)). Falkner notes that his 126-vortex values
  may need a factor 1 + 0.029 tan(sweep) "to bring them to the full potential
  value"; they are uncorrected here, so they carry ~1-3 % uncertainty.
* 45 deg swept wing, AR 5, constant chord, RAE 101 12 % section, Re 1.7e6:
  J. Weber and G. G. Brebner, "Low-speed tests on 45-deg swept-back wings,
  Part I", ARC R&M 2882, 1951/1958: wing CL from pressure integration (Table 4B)
  at alpha = 2.1, 4.2, 6.3, 8.4, 10.5 deg and the spanwise section lift at
  4.2 deg (Table 3), as transcribed in the FLOWPanel.jl example
  examples/liftingline_weber.jl (MIT licence, BYU FLOW Lab).
* Same planform with 4 horseshoe vortices per semispan: CL_alpha = 3.443 per
  radian (J. J. Bertin and M. L. Smith, Aerodynamics for Engineers, VLM worked
  example; value as quoted by T. Melin, "A vortex lattice MATLAB implementation
  for linear aerodynamic wing applications", KTH 2000). A code-to-code check of
  the identical discretisation.
* Viscous wings: R. H. Neely, T. V. Bollech, G. C. Westrick and R. R. Graham,
  "Experimental and calculated characteristics of several NACA 44-series wings
  with aspect ratios of 8, 10, and 12 and taper ratios of 2.5 and 3.5", NACA
  TN 1270, 1947 (Langley 19-ft pressure tunnel, smooth models): Table I
  (geometry) and Table II (experimental CD_min, (L/D)_max, alpha_L0, dCL/dalpha,
  CL_max, dCm/dCL and Cm at zero lift). XFLR5's own guidelines (v6.02, section
  4.3.15) validate against the same report.

Modelling choices for the NACA TN 1270 wings (not all stated in the report):
straight taper to the construction tip with the parabolic tip over the outer
5 % of the semispan modelled as a chord falling parabolically to zero (tangent to
the straight edges); unswept quarter-chord line; **linear** washout from root to
construction tip (the distribution is not given); thickness interpolated from
the root section to NACA 4412 at 95 % semispan; Re on the MAC from the table;
Mach 0.13, estimated from the stated tunnel density (0.0055 slug/ft^3) and Re;
free transition with Ncrit 9.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerolume.model import Fluid, WingSection, WingSpec
from aerolume.solvers.wing import SectionPolars, planform_metrics, wing_polar

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
FT = 0.3048

# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------

#: Multhopp R&M 2884: circular wing, dCL/dalpha [1/rad]
MULTHOPP_CIRCLE = 1.799

#: Falkner R&M 2596: rectangular wings, dCL/dalpha [1/rad] (all lifting-plane solutions listed)
FALKNER_RECT = {6: (4.270, 4.247, 4.231), 4: (3.639,), 2: (2.524,), 1: (1.489,)}
#: Falkner R&M 2596: 4-term lifting-line solution, rectangular AR 6
FALKNER_LL_AR6 = 4.527
#: Falkner R&M 2596: constant-chord swept wings (AR, sweep deg) -> dCL/dalpha [1/rad]
FALKNER_SWEPT = {(6, 30): (3.955, 3.877), (6, 45): (3.464, 3.367, 3.304, 3.365), (4, 30): (3.467, 3.400),
                 (4, 45): (3.121, 3.039, 2.986)}

#: Weber & Brebner R&M 2882 (via FLOWPanel.jl): integrated CL (Table 4B)
WEBER_ALPHA = np.array([2.1, 4.2, 6.3, 8.4, 10.5])
WEBER_CL = np.array([0.121, 0.238, 0.350, 0.456, 0.559])
#: Weber & Brebner Table 3: section lift at alpha = 4.2 deg against 2y/b
WEBER_Y2B = np.array([0.0, 0.041, 0.082, 0.163, 0.245, 0.367, 0.510, 0.653, 0.898, 0.949])
WEBER_CL_42 = np.array([0.235, 0.241, 0.248, 0.253, 0.251, 0.251, 0.251, 0.246, 0.192, 0.171])

#: Bertin & Smith VLM example (4 horseshoes per semispan): CL_alpha [1/rad]
BERTIN_SMITH_CLA = 3.443


@dataclass(frozen=True)
class Tn1270Wing:
    """One wing of NACA TN 1270: Table I geometry and Table II experimental values (smooth)."""
    name: str
    taper: float
    aspect_ratio: float
    root: str
    tip: str
    span_ft: float
    area_ft2: float
    mac_ft: float
    washout_deg: float
    re: float
    cd_min: float
    ld_max: float
    alpha_l0: float
    cl_alpha: float                 # per degree
    cl_max: float
    dcm_dcl: float
    cm0: float
    calc: dict = field(default_factory=dict)   # NACA's own "generalized method" calculation


TN1270 = {
    w.name: w for w in (
        Tn1270Wing("2.5-8-44,16", 2.5, 8.04, "4416", "4412", 15, 27.994, 1.990, 4.5, 4.32e6,
                   0.0090, 27.8, -2.9, 0.0820, 1.54, 0.012, -0.099,
                   {"cd_min": 0.0080, "ld_max": 29.4, "alpha_l0": -2.9, "cl_alpha": 0.0823, "cl_max": 1.48,
                    "dcm_dcl": 0.007, "cm0": -0.093}),
        Tn1270Wing("2.5-10-44,20", 2.5, 10.05, "4420", "4412", 15, 22.393, 1.592, 3.5, 3.49e6,
                   0.0083, 31.6, -3.2, 0.0860, 1.43, 0.0, -0.095,
                   {"cd_min": 0.0083, "ld_max": 32.0, "alpha_l0": -3.0, "cl_alpha": 0.0827, "cl_max": 1.41,
                    "dcm_dcl": 0.006, "cm0": -0.087}),
        Tn1270Wing("2.5-12-44,24", 2.5, 12.06, "4424", "4412", 15, 18.661, 1.328, 3.0, 2.87e6,
                   0.0091, 33.4, -3.2, 0.0870, 1.27, 0.021, -0.097,
                   {"cd_min": 0.0088, "ld_max": 32.6, "alpha_l0": -3.0, "cl_alpha": 0.0848, "cl_max": 1.31,
                    "dcm_dcl": 0.016, "cm0": -0.085}),
        Tn1270Wing("2.5-8-44,24", 2.5, 8.04, "4424", "4412", 15, 27.994, 1.990, 2.4, 4.32e6,
                   0.0081, 26.1, -3.2, 0.0795, 1.37, 0.021, -0.084,
                   {"cd_min": 0.0084, "ld_max": 28.0, "alpha_l0": -3.1, "cl_alpha": 0.0795, "cl_max": 1.35,
                    "dcm_dcl": 0.014, "cm0": -0.083}),
    )
}
TN1270_MACH = 0.13
TN1270_NCRIT = 9.0
TN1270_ALPHAS = np.arange(-6.0, 21.0, 1.0)      # to 20 deg: the critical-section stall lies at 13-17 deg

#: tolerances asserted by tests/test_wing_validation.py (see docs/validation_wing.md for why)
TOL = {
    "elliptic_llt_cla": 0.02,          # relative, vs 2 pi AR / (AR + 2)
    "elliptic_e": 0.98,                # minimum span efficiency (VLM and LLT)
    "lifting_surface": 0.025,          # relative, VLM vs Falkner / Multhopp (AR >= 4, circle)
    "lifting_surface_low_ar": 0.035,   # relative, VLM vs Falkner (AR 1 and 2)
    "swept": 0.03,                     # relative, VLM vs the range of Falkner's swept solutions
    "weber_cl": 0.05,                  # relative, VLM CL vs Weber & Brebner at each angle
    "weber_span": 0.035,               # absolute section cl, 0.04 <= 2y/b <= 0.95, alpha 4.2 deg
    "bertin_smith": 0.005,             # relative, identical discretisation
    # NACA TN 1270 wings: error bands of the method as measured (docs/validation_wing.md explains each);
    # for scale, NACA's own lifting-line calculation with *measured* 2D section data was within
    # 4 % (slope), 0.2 deg, 11 % (CD_min), 7 % ((L/D)max), 0.012 (Cm0) and 0.08 (CL_max)
    "tn_cla_vlm": 0.06,                # relative; thin-wing lift ignores the viscous lift-slope loss of thick roots
    "tn_cla_llt": 0.08,                # relative; a lifting line has no chordwise loading (AR 8: +7 %)
    "tn_alpha_l0": 0.6,                # deg; the twist distribution is not given (linear assumed)
    "tn_cd_min": 0.25,                 # relative; measured: 13-22 % low (free transition at Ncrit 9)
    "tn_ld_max": 0.20,                 # relative; follows from the drag bias
    "tn_cm0_vlm": 0.02,                # absolute; inviscid camber moment (no viscous decambering)
    "tn_cm0_llt": 0.012,               # absolute; section cm from the viscous polars
    "tn_cl_max": (-0.15, 0.4),         # VLM critical-section CL_max minus experiment; measured +0.17..+0.31
                                       # (section polars stall late): CL_max is NOT validated
}

# ---------------------------------------------------------------------------
# geometry builders
# ---------------------------------------------------------------------------


def elliptic_wing(ar: float, *, n_span: int = 24, n_chord: int = 8, span: float = 1.0) -> WingSpec:
    """Elliptic planform with a straight quarter-chord line, sections at the lattice nodes.

    The stations are the half-cosine (sine) distribution the solver itself uses, so
    every lattice node lies on the exact ellipse and each segment holds one panel.
    """
    c0 = 4.0 * span / (math.pi * ar)
    ys = 0.5 * span * np.sin(np.linspace(0.0, 0.5 * math.pi, n_span + 1))
    secs = []
    for y in ys:
        c = c0 * math.sqrt(max(0.0, 1.0 - (2.0 * y / span) ** 2))
        secs.append(WingSection(y=float(y), chord=c, x_le=0.25 * (c0 - c)))
    return WingSpec(sections=secs, symmetric=True, n_chord=n_chord, n_span=n_span, spacing="cosine")


def constant_chord_wing(ar: float, sweep_deg: float = 0.0, *, chord: float = 1.0, n_span: int = 24,
                        n_chord: int = 8, spacing: str = "cosine", airfoil: str = "0012") -> WingSpec:
    """Rectangular (or swept constant-chord, sweep of every chord line) wing."""
    s = 0.5 * ar * chord
    return WingSpec(sections=[WingSection(y=0.0, chord=chord, airfoil=airfoil),
                              WingSection(y=s, chord=chord, x_le=s * math.tan(math.radians(sweep_deg)),
                                          airfoil=airfoil)],
                    symmetric=True, n_chord=n_chord, n_span=n_span, spacing=spacing)


def circular_wing(*, n_span: int = 40, n_chord: int = 12, radius: float = 1.0) -> WingSpec:
    """Circular planform (AR = 4 / pi), sections on the half-cosine lattice nodes."""
    ys = radius * np.sin(np.linspace(0.0, 0.5 * math.pi, n_span + 1))
    secs = []
    for y in ys:
        h = math.sqrt(max(radius * radius - y * y, 0.0))
        secs.append(WingSection(y=float(y), chord=2.0 * h, x_le=radius - h))
    return WingSpec(sections=secs, symmetric=True, n_chord=n_chord, n_span=n_span)


def tn1270_wing(w: Tn1270Wing, *, n_span: int = 24, n_chord: int = 8) -> WingSpec:
    """NACA TN 1270 planform (see the module docstring for the modelling choices).

    The root chord is solved so that the planform area equals Table I's.
    """
    span = w.span_ft * FT
    target = w.area_ft2 * FT ** 2

    def build(cr: float) -> WingSpec:
        s = 0.5 * span
        k = 1.0 - 1.0 / w.taper
        c95 = cr * (1.0 - k * 0.95)
        slope = -cr * k
        q = -(c95 + slope * 0.05) / 0.05 ** 2           # parabolic tip, tangent at 95 %, zero at 100 %
        secs = []
        for eta in (0.0, 0.95, 0.97, 0.985, 0.995, 1.0):
            if eta <= 0.95:
                c = cr * (1.0 - k * eta)
            else:
                d = eta - 0.95
                c = max(c95 + slope * d + q * d * d, 0.0)
            secs.append(WingSection(y=eta * s, chord=c, x_le=0.25 * (cr - c), twist_deg=-w.washout_deg * eta,
                                    airfoil=w.root if eta == 0.0 else w.tip))
        return WingSpec(sections=secs, symmetric=True, n_chord=n_chord, n_span=n_span)

    lo, hi = 0.05, 3.0
    for _ in range(60):
        cr = 0.5 * (lo + hi)
        if planform_metrics(build(cr))["area"] < target:
            lo = cr
        else:
            hi = cr
    return build(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def range_deviation(value: float, refs) -> float:
    """Relative distance of ``value`` from the interval spanned by ``refs`` (0 inside it)."""
    lo, hi = min(refs), max(refs)
    if value < lo:
        return value / lo - 1.0
    if value > hi:
        return value / hi - 1.0
    return 0.0


def lift_slope(wing: WingSpec, method: str = "vlm", alpha: float = 2.0) -> tuple[float, float]:
    """(CL_alpha [1/rad], e) of an uncambered wing from one inviscid solution at ``alpha``."""
    r = wing_polar(wing, [alpha], method=method, viscous=False)
    p = r.points[0]
    return p.cl / math.radians(alpha), p.e


def case_elliptic(ars=(4, 8, 12), n_span: int = 24) -> list[dict]:
    """Elliptic wings: VLM and LLT lift slope and span efficiency against lifting-line theory."""
    rows = []
    for ar in ars:
        w = elliptic_wing(ar, n_span=n_span)
        v, ev = lift_slope(w, "vlm")
        line, el = lift_slope(w, "llt")
        rows.append({"ar": ar, "formula": 2 * math.pi * ar / (ar + 2), "vlm": v, "e_vlm": ev, "llt": line,
                     "e_llt": el})
    return rows


def case_lifting_surface() -> dict:
    """Circular and rectangular wings (VLM) and the rectangular AR 6 lifting line vs R&M 2884 / 2596."""
    out = {"circle": lift_slope(circular_wing())[0]}
    for ar in FALKNER_RECT:
        out[f"rect{ar}"] = lift_slope(constant_chord_wing(ar))[0]
    out["rect6_llt"] = lift_slope(constant_chord_wing(6), "llt")[0]
    for (ar, sw) in FALKNER_SWEPT:
        out[f"swept{ar}_{sw}"] = lift_slope(constant_chord_wing(ar, sw))[0]
    return out


def weber_wing(**kw) -> WingSpec:
    """Weber & Brebner's wing: AR 5, constant chord, 45 deg sweep (span 98 in)."""
    b = 98 * 0.0254
    return constant_chord_wing(5, 45.0, chord=b / 5, **kw)


def case_weber(n_span: int = 24, n_chord: int = 8) -> dict:
    """VLM (inviscid; its lift is not changed by the strip drag) against Weber & Brebner."""
    w = weber_wing(n_span=n_span, n_chord=n_chord)
    r = wing_polar(w, WEBER_ALPHA, viscous=False)
    cl = np.array([p.cl for p in r.points])
    p42 = r.points[1]
    b = r.span
    y2b = 2 * p42.y / b
    right = y2b > 0
    span_cl = np.interp(WEBER_Y2B, y2b[right], p42.cl_local[right])
    return {"cl": cl, "y2b": y2b[right], "cl_local": p42.cl_local[right], "span_cl": span_cl,
            "cla": float(np.polyfit(np.radians(WEBER_ALPHA), cl, 1)[0])}


def case_bertin_smith() -> float:
    """Bertin & Smith's lattice: 4 horseshoes per semispan, 1 chordwise, uniform spacing."""
    return lift_slope(weber_wing(n_span=4, n_chord=1, spacing="uniform"), alpha=1.0)[0]


@dataclass
class WingCase:
    """Viscous VLM and LLT polars of one NACA TN 1270 wing, plus derived characteristics."""
    wing: Tn1270Wing
    spec: WingSpec
    vlm: object
    llt: object
    stall_vlm: dict | None
    stall_llt: dict | None
    seconds: float
    metrics: dict = field(default_factory=dict)

    def summary(self, method: str) -> dict[str, float]:
        """dCL/dalpha (-4..4 deg), alpha_L0, CD_min, (L/D)_max, Cm0, dCm/dCL, CL_max (stall onset)."""
        r = self.vlm if method == "vlm" else self.llt
        t = r.table()
        a, cl, cd, cm = t["alpha"], t["cl"], t["cd"], t["cm"]
        sel = (a >= -4) & (a <= 4)
        s, i = np.polyfit(a[sel], cl[sel], 1)
        pos = cl > 0.05
        stall = self.stall_vlm if method == "vlm" else self.stall_llt
        return {
            "cl_alpha": float(s),
            "alpha_l0": float(-i / s),
            "cd_min": float(cd.min()),
            "ld_max": float(np.max(cl[pos] / cd[pos])),
            "cm0": float(np.interp(0.0, cl, cm)),
            "dcm_dcl": float(np.polyfit(cl[sel], cm[sel], 1)[0]),
            "cl_max": float(stall["cl"]) if stall else float("nan"),
            "alpha_stall": float(stall["alpha"]) if stall else float("nan"),
            "converged": int(np.sum([p.converged for p in r.points])),
            "total": len(r.points),
        }


def case_tn1270(name: str, *, alphas=TN1270_ALPHAS, cache_dir: Path | None = None,
                ncrit: float = TN1270_NCRIT) -> WingCase:
    """Viscous VLM and LLT polars of a NACA TN 1270 wing (one shared section-polar cache).

    ``cache_dir`` keeps the computed section points in a JSON file per wing
    (``SectionPolars.state``), so a rerun only recomputes the wing solvers.
    """
    w = TN1270[name]
    spec = tn1270_wing(w)
    m = planform_metrics(spec)
    fluid = Fluid()
    speed = w.re * fluid.viscosity / (fluid.density * m["mac"])
    sp = SectionPolars(ncrit=ncrit, mach=TN1270_MACH)
    cache = None
    if cache_dir is not None:
        tag = "" if ncrit == TN1270_NCRIT else f"_ncrit{ncrit:g}"
        cache = Path(cache_dir) / ("tn1270_" + "".join(ch if ch.isalnum() else "_" for ch in name) + tag + ".json")
        if cache.exists():
            sp.load_state(json.loads(cache.read_text(encoding="utf-8")))
    ref = (0.25 * spec.sections[0].chord, 0.0, 0.0)          # on the (unswept) quarter-chord line
    t0 = time.perf_counter()
    dv, dl = {}, {}
    kw = {"speed": speed, "fluid": fluid, "mach": TN1270_MACH, "ncrit": ncrit, "polar_provider": sp,
          "moment_ref": ref}
    vlm = wing_polar(spec, alphas, method="vlm", details=dv, **kw)
    llt = wing_polar(spec, alphas, method="llt", details=dl, **kw)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sp.state()), encoding="utf-8")
    c = WingCase(wing=w, spec=spec, vlm=vlm, llt=llt, stall_vlm=dv["stall"], stall_llt=dl["stall"],
                 seconds=time.perf_counter() - t0)
    c.metrics = {"area_ft2": m["area"] / FT ** 2, "mac_ft": m["mac"] / FT, "ar": m["aspect_ratio"],
                 "speed": speed}
    return c


def performance_wing(root: str = "2412", tip: str = "2412") -> WingSpec:
    """Two-section tapered wing used for the timing target (span 2 m, taper 0.5)."""
    return WingSpec(sections=[WingSection(y=0.0, chord=0.30, airfoil=root),
                              WingSection(y=1.0, chord=0.15, x_le=0.0375, airfoil=tip)],
                    n_chord=8, n_span=24)


def case_performance(root: str = "2412", tip: str = "2412", speed: float = 20.0) -> dict:
    """17-point viscous VLM polar (-4..12 deg), cold (empty polar cache) and warm."""
    w = performance_wing(root, tip)
    alphas = np.arange(-4.0, 13.0, 1.0)
    sp = SectionPolars()
    t0 = time.perf_counter()
    r = wing_polar(w, alphas, speed=speed, polar_provider=sp)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    r2 = wing_polar(w, alphas, speed=speed, polar_provider=sp)
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    r3 = wing_polar(w, alphas, speed=speed, viscous=False)
    inviscid = time.perf_counter() - t0
    return {"cold": cold, "warm": warm, "inviscid": inviscid, "points": sp.points_computed,
            "converged": sum(p.converged for p in r.points), "n": len(alphas),
            "same": all(abs(p.cd - q.cd) < 1e-12 for p, q in zip(r.points, r2.points)),
            "note": next((n for n in r.notes if n.startswith("Viscous strip")), ""), "r": r, "r_inv": r3}


# ---------------------------------------------------------------------------
# figures (matplotlib Figure + Agg canvas only; never pyplot)
# ---------------------------------------------------------------------------

CALC = "#0f7c8a"     # Aerolume teal (vortex lattice)
CALC2 = "#7a4fb5"    # lifting line
EXP = "#2b2b2b"
EXP2 = "#c0602b"


def _figure(ncols: int, width: float = 10.0, height: float = 4.0):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols)
    for ax in np.atleast_1d(axes):
        ax.grid(True, color="#d8dde3", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig, axes


def fig_lifting_surface(ell: list[dict], ls: dict, path: Path) -> None:
    """Lift slope vs aspect ratio: elliptic (theory), rectangular (Falkner), circle (Multhopp); sweep."""
    fig, (a1, a2) = _figure(2, 11.0, 4.2)
    ar = np.linspace(0.8, 14, 200)
    a1.plot(ar, 2 * np.pi * ar / (ar + 2), "-", color=EXP, lw=1, label="lifting-line theory, elliptic")
    a1.plot([r["ar"] for r in ell], [r["llt"] for r in ell], "s", color=CALC2, ms=7, mfc="none",
            label="Aerolume LLT, elliptic")
    a1.plot([r["ar"] for r in ell], [r["vlm"] for r in ell], "o", color=CALC, ms=6, label="Aerolume VLM, elliptic")
    rect = sorted(FALKNER_RECT)
    a1.plot(rect, [ls[f"rect{a}"] for a in rect], "D", color=CALC, ms=5, label="Aerolume VLM, rectangular")
    for a in rect:
        for k, v in enumerate(FALKNER_RECT[a]):
            a1.plot(a, v, "_", color=EXP2, ms=14, mew=2, label="Falkner R&M 2596, rectangular" if (a, k) == (6, 0)
                    else None)
    a1.plot(4 / math.pi, ls["circle"], "*", color=CALC, ms=10, label="Aerolume VLM, circular")
    a1.plot(4 / math.pi, MULTHOPP_CIRCLE, "_", color=EXP, ms=16, mew=2, label="Multhopp R&M 2884, circular")
    a1.set_xlabel("aspect ratio")
    a1.set_ylabel("dCL/dα [1/rad]")
    a1.set_title("Unswept wings", fontsize=10)
    a1.legend(fontsize=7, frameon=False, loc="lower right")
    sw = [0, 30, 45]
    a2.plot(sw, [ls["rect6"], ls["swept6_30"], ls["swept6_45"]], "o-", color=CALC, label="Aerolume VLM, AR 6")
    a2.plot(sw, [ls["rect4"], ls["swept4_30"], ls["swept4_45"]], "s--", color=CALC, mfc="none",
            label="Aerolume VLM, AR 4")
    for (a, s), vals in FALKNER_SWEPT.items():
        for v in vals:
            a2.plot(s, v, "_", color=EXP2, ms=14, mew=2)
    for v in FALKNER_RECT[6] + FALKNER_RECT[4]:
        a2.plot(0, v, "_", color=EXP2, ms=14, mew=2)
    a2.plot([], [], "_", color=EXP2, ms=14, mew=2, label="Falkner R&M 2596")
    a2.set_xlabel("sweep [deg]")
    a2.set_ylabel("dCL/dα [1/rad]")
    a2.set_title("Constant-chord swept wings", fontsize=10)
    a2.legend(fontsize=8, frameon=False)
    fig.savefig(path)


def fig_weber(wb: dict, path: Path) -> None:
    """Weber & Brebner 45 deg wing: CL(alpha) and the section lift at 4.2 deg."""
    fig, (a1, a2) = _figure(2)
    a = np.linspace(0, 11, 50)
    a1.plot(a, wb["cla"] * np.radians(a), "-", color=CALC, lw=2, label="Aerolume VLM")
    a1.plot(WEBER_ALPHA, WEBER_CL, "o", color=EXP, ms=6, label="Weber & Brebner (pressure plotting)")
    a1.set_xlabel("α [deg]")
    a1.set_ylabel("CL")
    a1.set_title("45° swept wing, AR 5", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    a2.plot(wb["y2b"], wb["cl_local"], "-", color=CALC, lw=2, label="Aerolume VLM")
    a2.plot(WEBER_Y2B, WEBER_CL_42, "o", color=EXP, ms=6, label="Weber & Brebner")
    a2.set_xlabel("2y/b")
    a2.set_ylabel("section lift coefficient")
    a2.set_ylim(0, 0.32)
    a2.set_title("Span loading at α = 4.2°", fontsize=10)
    a2.legend(fontsize=8, frameon=False, loc="lower left")
    fig.savefig(path)


def fig_tn1270(cases: dict[str, WingCase], path: Path) -> None:
    """Viscous polars of the first wing with Table II's characteristics, and the summary of all wings."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12.0, 7.6), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 3)
    for ax in axs.flat:
        ax.grid(True, color="#d8dde3", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    name = next(iter(cases))
    c = cases[name]
    w = c.wing
    tv, tl = c.vlm.table(), c.llt.table()
    a1, a2, a3 = axs[0]
    a1.plot(tv["alpha"], tv["cl"], "-", color=CALC, lw=2, label="Aerolume VLM + strips")
    a1.plot(tl["alpha"], tl["cl"], "--", color=CALC2, lw=2, label="Aerolume LLT (nonlinear)")
    aa = np.linspace(w.alpha_l0, w.alpha_l0 + w.cl_max / w.cl_alpha, 10)
    a1.plot(aa, w.cl_alpha * (aa - w.alpha_l0), ":", color=EXP, lw=1.5, label="NACA TN 1270: α_L0, dCL/dα")
    a1.axhline(w.cl_max, color=EXP, lw=0.8, ls="-.", label="NACA TN 1270: CL_max")
    if c.stall_vlm:
        a1.plot(c.stall_vlm["alpha"], c.stall_vlm["cl"], "v", color=CALC, ms=8, label="VLM stall onset (crit. section)")
    if c.stall_llt:
        a1.plot(c.stall_llt["alpha"], c.stall_llt["cl"], "v", color=CALC2, ms=8, label="LLT stall onset")
    a1.set_xlabel("α [deg]")
    a1.set_ylabel("CL")
    a1.set_title(f"Wing {name}, R {w.re:.3g}", fontsize=10)
    a1.legend(fontsize=7, frameon=False, loc="lower right")
    a2.plot(tv["cd"], tv["cl"], "-", color=CALC, lw=2, label="VLM")
    a2.plot(tl["cd"], tl["cl"], "--", color=CALC2, lw=2, label="LLT")
    a2.axvline(w.cd_min, color=EXP, lw=0.8, ls=":", label="exp. CD_min")
    cc = np.linspace(0, 1.4, 20)
    a2.plot(cc / w.ld_max, cc, "-.", color=EXP, lw=0.8, label="exp. (L/D)_max line")
    a2.set_xlim(0, 0.06)
    a2.set_xlabel("CD")
    a2.set_ylabel("CL")
    a2.legend(fontsize=7, frameon=False, loc="lower right")
    a3.plot(tv["cl"], tv["cm"], "-", color=CALC, lw=2, label="VLM")
    a3.plot(tl["cl"], tl["cm"], "--", color=CALC2, lw=2, label="LLT")
    cc = np.linspace(-0.3, 1.2, 10)
    a3.plot(cc, w.cm0 + w.dcm_dcl * cc, ":", color=EXP, lw=1.5, label="exp. Cm0, dCm/dCL")
    a3.set_xlabel("CL")
    a3.set_ylabel("Cm (quarter-chord line)")
    a3.legend(fontsize=7, frameon=False, loc="lower left")
    keys = [("cl_alpha", "dCL/dα [1/deg]"), ("cd_min", "CD_min"), ("ld_max", "(L/D)_max")]
    names = list(cases)
    x = np.arange(len(names))
    for ax, (k, lab) in zip(axs[1], keys):
        exp = [getattr(cases[n].wing, k) for n in names]
        calc = [cases[n].wing.calc[k] for n in names]
        v = [cases[n].summary("vlm")[k] for n in names]
        line = [cases[n].summary("llt")[k] for n in names]
        ax.plot(x - 0.15, exp, "o", color=EXP, ms=7, label="experiment")
        ax.plot(x - 0.05, calc, "^", color=EXP2, ms=6, mfc="none", label="NACA calc. (2D test data)")
        ax.plot(x + 0.05, v, "s", color=CALC, ms=6, label="Aerolume VLM")
        ax.plot(x + 0.15, line, "D", color=CALC2, ms=5, mfc="none", label="Aerolume LLT")
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_xticks(x, names, fontsize=7)
        ax.set_ylabel(lab)
    axs[1, 0].legend(fontsize=7, frameon=False)
    fig.savefig(path)


def fig_span_loading(path: Path) -> None:
    """Span loading of an elliptic and a tapered twisted wing: VLM vs LLT."""
    fig, (a1, a2) = _figure(2)
    w = elliptic_wing(8)
    for method, col, ls in (("vlm", CALC, "-"), ("llt", CALC2, "--")):
        p = wing_polar(w, [4.0], method=method, viscous=False).points[0]
        a1.plot(2 * p.y, p.c_cl / p.cl, ls, color=col, lw=2, label=f"Aerolume {method.upper()}")
    eta = np.linspace(-1, 1, 200)
    a1.plot(eta, 4 / np.pi * np.sqrt(1 - eta ** 2), ":", color=EXP, lw=1.5, label="elliptic, (4/π)√(1−η²)")
    a1.set_xlabel("2y/b")
    a1.set_ylabel("c·cl / (c̄·CL)")
    a1.set_title("Elliptic wing, AR 8", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="lower center")
    w2 = tn1270_wing(TN1270["2.5-8-44,16"])
    for method, col, ls in (("vlm", CALC, "-"), ("llt", CALC2, "--")):
        p = wing_polar(w2, [6.0], method=method, viscous=False).points[0]
        a2.plot(2 * p.y / (TN1270["2.5-8-44,16"].span_ft * FT), p.cl_local, ls, color=col, lw=2,
                label=f"{method.upper()} cl")
    a2.set_xlabel("2y/b")
    a2.set_ylabel("section cl")
    a2.set_title("NACA TN 1270 wing 2.5-8-44,16, α = 6° (inviscid)", fontsize=10)
    a2.legend(fontsize=8, frameon=False, loc="lower center")
    fig.savefig(path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f} %"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-figures", action="store_true", help="skip rendering docs/img/wing_*.png")
    ap.add_argument("--quick", action="store_true", help="skip the viscous NACA TN 1270 wings and the timing")
    ap.add_argument("--cache", metavar="DIR", help="keep the section polars of the NACA TN 1270 wings in DIR "
                    "(JSON) so reruns skip them")
    ap.add_argument("--ncrit-sensitivity", type=float, metavar="N",
                    help="also run the first NACA TN 1270 wing with this Ncrit (transition sensitivity)")
    ap.add_argument("--no-perf", action="store_true", help="skip the timing target")
    ap.add_argument("--wings", default=";".join(TN1270),
                    help="NACA TN 1270 wings to run, separated by ';' (the names contain commas)")
    args = ap.parse_args(argv)
    out = sys.stdout
    if not args.no_figures:
        IMG.mkdir(parents=True, exist_ok=True)

    ell = case_elliptic()
    print("## Elliptic wings (inviscid, 24 panels per half, 8 chordwise)\n", file=out)
    print("| AR | 2πAR/(AR+2) | LLT | Δ | e (LLT) | VLM | Δ | e (VLM) |", file=out)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|", file=out)
    for r in ell:
        print(f"| {r['ar']} | {r['formula']:.4f} | {r['llt']:.4f} | {_pct(r['llt'], r['formula'])} | "
              f"{r['e_llt']:.4f} | {r['vlm']:.4f} | {_pct(r['vlm'], r['formula'])} | {r['e_vlm']:.4f} |", file=out)

    ls = case_lifting_surface()
    print("\n## Lifting-surface references (VLM, inviscid, dCL/dα per radian)\n", file=out)
    print("| wing | reference | Aerolume | Δ |", file=out)
    print("|---|---:|---:|---:|", file=out)
    print(f"| circular (AR 1.27) | {MULTHOPP_CIRCLE} (Multhopp) | {ls['circle']:.4f} | "
          f"{_pct(ls['circle'], MULTHOPP_CIRCLE)} |", file=out)
    for ar, vals in FALKNER_RECT.items():
        ref = float(np.mean(vals))
        print(f"| rectangular AR {ar} | {' / '.join(f'{v:g}' for v in vals)} | {ls[f'rect{ar}']:.4f} | "
              f"{_pct(ls[f'rect{ar}'], ref)} |", file=out)
    print(f"| rectangular AR 6, LLT | {FALKNER_LL_AR6} (lifting line) | {ls['rect6_llt']:.4f} | "
          f"{_pct(ls['rect6_llt'], FALKNER_LL_AR6)} |", file=out)
    for (ar, sw), vals in FALKNER_SWEPT.items():
        v = ls[f"swept{ar}_{sw}"]
        print(f"| AR {ar}, {sw}° sweep | {' / '.join(f'{x:g}' for x in vals)} | {v:.4f} | "
              f"{range_deviation(v, vals) * 100:+.1f} % from range |", file=out)

    wb = case_weber()
    bs = case_bertin_smith()
    print("\n## 45° swept wing, AR 5 — Weber & Brebner (R&M 2882)\n", file=out)
    print("| α [deg] | CL exp | CL VLM | Δ |", file=out)
    print("|---:|---:|---:|---:|", file=out)
    for a, ce, cc in zip(WEBER_ALPHA, WEBER_CL, wb["cl"]):
        print(f"| {a:g} | {ce:.3f} | {cc:.3f} | {_pct(cc, ce)} |", file=out)
    print("\n| 2y/b | cl exp (4.2°) | cl VLM | Δ |", file=out)
    print("|---:|---:|---:|---:|", file=out)
    for y, ce, cc in zip(WEBER_Y2B, WEBER_CL_42, wb["span_cl"]):
        print(f"| {y:.3f} | {ce:.3f} | {cc:.3f} | {cc - ce:+.3f} |", file=out)
    print(f"\nCL_α (VLM) {wb['cla']:.4f}/rad; Bertin & Smith lattice (4 × 1 horseshoes): {bs:.4f} vs "
          f"{BERTIN_SMITH_CLA} ({_pct(bs, BERTIN_SMITH_CLA)})", file=out)
    if not args.no_figures:
        fig_lifting_surface(ell, ls, IMG / "wing_lifting_surface.png")
        fig_weber(wb, IMG / "wing_swept_weber.png")
        fig_span_loading(IMG / "wing_span_loading.png")
    if args.quick:
        return 0

    cases = {}
    for name in [n.strip() for n in args.wings.split(";") if n.strip()]:
        c = case_tn1270(name, cache_dir=Path(args.cache) if args.cache else None)
        cases[name] = c
        print(f"\n## NACA TN 1270 wing {name} (R {c.wing.re:.3g}, M {TN1270_MACH}, Ncrit {TN1270_NCRIT:g})\n",
              file=out)
        print(f"planform: area {c.metrics['area_ft2']:.3f} ft² (table {c.wing.area_ft2}), MAC "
              f"{c.metrics['mac_ft']:.3f} ft (table {c.wing.mac_ft}), AR {c.metrics['ar']:.2f} "
              f"(table {c.wing.aspect_ratio}); {c.seconds:.0f} s for both polars\n", file=out)
        sv, sl = c.summary("vlm"), c.summary("llt")
        print("| quantity | experiment | NACA calc. | VLM | LLT |", file=out)
        print("|---|---:|---:|---:|---:|", file=out)
        for k, lab, fmt in (("cl_alpha", "dCL/dα [1/deg]", ".4f"), ("alpha_l0", "α_L0 [deg]", ".2f"),
                            ("cd_min", "CD_min", ".4f"), ("ld_max", "(L/D)_max", ".1f"),
                            ("cm0", "Cm at CL = 0", ".3f"), ("dcm_dcl", "dCm/dCL", ".3f"),
                            ("cl_max", "CL_max (stall onset)", ".3f")):
            print(f"| {lab} | {getattr(c.wing, k):{fmt}} | {c.wing.calc[k]:{fmt}} | {sv[k]:{fmt}} | "
                  f"{sl[k]:{fmt}} |", file=out)
        print(f"\nconverged: VLM {sv['converged']}/{sv['total']}, LLT {sl['converged']}/{sl['total']}; "
              f"stall onset VLM α {sv['alpha_stall']:.1f}°, LLT α {sl['alpha_stall']:.1f}°", file=out)
        for wmsg in c.llt.warnings:
            print(f"- LLT warning: {wmsg}", file=out)
    if cases and not args.no_figures:
        fig_tn1270(cases, IMG / "wing_tn1270.png")
    if args.ncrit_sensitivity and cases:
        name = next(iter(cases))
        c = case_tn1270(name, ncrit=args.ncrit_sensitivity, cache_dir=Path(args.cache) if args.cache else None)
        sv, sl = c.summary("vlm"), c.summary("llt")
        print(f"\n## Transition sensitivity: wing {name} with Ncrit {args.ncrit_sensitivity:g}\n", file=out)
        print("| quantity | experiment | VLM | LLT |", file=out)
        print("|---|---:|---:|---:|", file=out)
        for k, lab, fmt in (("cd_min", "CD_min", ".4f"), ("ld_max", "(L/D)_max", ".1f"),
                            ("cl_alpha", "dCL/dα [1/deg]", ".4f"), ("cm0", "Cm at CL = 0", ".3f"),
                            ("cl_max", "CL_max (stall onset)", ".3f")):
            print(f"| {lab} | {getattr(c.wing, k):{fmt}} | {sv[k]:{fmt}} | {sl[k]:{fmt}} |", file=out)
    if args.no_perf:
        return 0

    perf = case_performance()
    perf2 = case_performance("2412", "0012")
    print("\n## Performance (17-point viscous VLM polar, 2-section wing, 8 × 48 panels)\n", file=out)
    for lab, p in (("2412 root and tip", perf), ("2412 root, 0012 tip", perf2)):
        print(f"- {lab}: cold {p['cold']:.1f} s ({p['points']} section points), warm {p['warm']:.2f} s, "
              f"inviscid {p['inviscid']:.2f} s; {p['converged']}/{p['n']} converged; {p['note']}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
