"""Results page: the numbers of one simulation, with their validity context, and every view of it.

Layout follows how an engineer reads a result (docs/ui_guidelines.md section 4):

* one header line of facts - convergence status, Reynolds number simulated vs
  real, grid, turbulence treatment, wall-clock time - and at most one warning
  line under it (DESIGN.md section 6: a number is never shown without whether it
  converged, at what Reynolds number and on what grid);
* tabs: **Summary** (a table: quantity, mean, ± standard error, RMS, min, max,
  unit, next to the flow conditions), **Forces** (histories with the averaging
  window), **Surface** (Cp, Cf), **Fields**, **Sensitivity**, **Polar**,
  **Data** (every scalar plus the exports).

The same page serves every analysis. A 2D Navier-Stokes result fills Forces,
Surface, Fields and (if computed) Sensitivity; a polar sweep fills Polar and
Surface (Cp at a chosen angle). Tabs without data are hidden rather than shown
empty. 3D results keep the header line and swap the tabs for the views of
:mod:`aerolume.app.widgets.results3d`: a 3D CFD run (six coefficients, forces
in N, the surface and the flow in 3D) and a finite-wing sweep (polars, span
loading, the wing coloured by its local lift). Those views are built the first
time a 3D result is shown, so a 2D study never loads the 3D viewer.

Data helpers at the top (:func:`split_loop`, :func:`surface_curves`,
:func:`summary_rows`, ...) are shared with the Compare page and tested directly.
"""
from __future__ import annotations

import contextlib
import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.fluid import display_name as fluid_name
from ...core.units import sig
from ...model import CFD3DResult, CFDResult, PolarResult, WingResult
from ..jobs import JobManager, airfoil_designs, require_api, run_in_background
from ..widgets.common import ElidedLabel, KeyValueList, SegmentedControl, ToolStrip
from ..widgets.field_view import (
    CP_LIMITS,
    SOLID_FILL,
    ColorBar,
    FieldView,
    ViewportToolbar,
    body_window,
    colormap_lut,
    colormap_names,
    display_range,
    field_info,
    near_field_window,
    robust_range,
)
from ..widgets.kpi import (
    MINUS,
    STEADY_TEXT,
    STEADY_TIP,
    WITHIN_ERROR_TEXT,
    Cell,
    DataTable,
    EmptyState,
    StatusLabel,
    WarningLine,
    button,
    decimals_from_error,
    fmt_duration,
    fmt_err,
    fmt_fixed,
    fmt_num,
    fmt_re,
    fmt_sig,
    fmt_speed,
    is_steady,
    label,
    stat_texts,
    value_error_texts,
    within_error,
)
from ..widgets.plots import (
    CpPlot,
    GeometryPlot,
    HistoryPlot,
    LinePlot,
    PolarGrid,
    decade_scale,
    import_error,
    lazy_import,
    qcolor,
    tokens,
)

log = logging.getLogger(__name__)

POLAR_METRIC_LABELS = {
    "cl_max": "Maximum lift Cl,max", "alpha_cl_max": "Angle of Cl,max",
    "cl_max_reached": "Stall reached in the sweep (1 = yes)", "cd_min": "Minimum drag Cd,min",
    "cl_at_cd_min": "Cl at Cd,min", "alpha_cd_min": "Angle of Cd,min", "ld_max": "Best lift-to-drag (L/D)max",
    "alpha_ld_max": "Angle of best L/D", "cl_at_ld_max": "Cl at best L/D", "lift_slope_per_deg": "Lift slope",
    "lift_slope_per_rad": "Lift slope", "alpha_zero_lift": "Zero-lift angle α0",
    "cm0": "Zero-lift moment Cm0", "alpha_min": "Lowest angle", "alpha_max": "Highest angle",
    "n_points": "Points computed", "n_converged": "Points converged",
}
FIELD_ORDER = ("speed", "cp", "p", "vorticity", "u", "v", "temperature", "theta", "dye", "dye1", "dye2", "dye3",
               "nut")
#: Pseudo-field of the Fields menu: all smoke channels composited in their emitters' colours.
SMOKE_COMPOSITE = "smoke_all"
SUMMARY_HEADERS = ["Quantity", "Mean", "± Std. error", "RMS", "Min", "Max", "Unit"]
POLAR_SUMMARY_HEADERS = ["Quantity", "Value", "Condition", "Unit"]
TAB_KEYS = ("summary", "forces", "surface", "fields", "heat", "fans", "sensitivity", "polar", "diagnostics", "data")
#: Polar table: (header, decimals). One decimal count per column, so the monospace digits
#: line up on the decimal point; Cd gets five because XFOIL-class drag is quoted to 0.00001.
POLAR_COLUMNS = (("α (°)", 2), ("Cl", 4), ("Cd", 5), ("Cdp", 5), ("Cdf", 5), ("Cm", 4), ("L/D", 1),
                 ("xtr upper", 3), ("xtr lower", 3), ("Converged", None))
#: Decimals of the coefficient rows of the Summary table (1 drag count = 0.0001) and of
#: their standard errors, shared by Cd, Cl and Cm so the block of numbers aligns.
COEF_DECIMALS, COEF_ERR_DECIMALS = 4, 5


# ---------------------------------------------------------------------------
# data helpers (also used by the Compare page)
# ---------------------------------------------------------------------------

def finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def rotate(x: np.ndarray, y: np.ndarray, deg: float, about=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Rotate points counter-clockwise by ``deg`` about ``about``."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = np.asarray(x) - about[0], np.asarray(y) - about[1]
    return about[0] + c * dx - s * dy, about[1] + s * dx + c * dy


def split_loop(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays of the upper and lower branch of a closed outline.

    The outline is cut at its leading edge (min x) and trailing edge (max x).
    For a counter-clockwise loop the walk from the trailing edge forwards passes
    over the top; the signed area decides, so clockwise input works as well.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        idx = np.arange(n)
        return idx, idx[:0]
    i_le, i_te = int(np.argmin(x)), int(np.argmax(x))
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def walk(a: int, b: int) -> np.ndarray:
        return (a + np.arange((b - a) % n + 1)) % n

    first, second = walk(i_te, i_le), walk(i_le, i_te)
    return (first, second) if area >= 0 else (second, first)


def surface_curves(sd, *, alpha_deg: float = 0.0, about=(0.0, 0.0), length: float = 1.0) -> dict:
    """Cp / Cf along the body-frame chord for one SurfaceData: {"upper": (xc, cp, cf), "lower": ...}.

    Surface data are stored in the wind frame (rotated geometry); plots read
    against the design's own x axis, so the points are rotated back by +alpha.
    x is normalised by the reference length and measured from the leading edge.
    """
    xb, yb = rotate(sd.x, sd.y, alpha_deg, about)
    up, lo = split_loop(xb, yb)
    x0 = float(np.min(xb)) if len(xb) else 0.0
    xc = (xb - x0) / (length or 1.0)
    cf = np.asarray(sd.cf if sd.cf is not None else np.full(len(xb), np.nan), dtype=float)
    cp = np.asarray(sd.cp, dtype=float)
    out = {}
    for name, idx in (("upper", up), ("lower", lo)):
        order = np.argsort(xc[idx], kind="stable")
        sel = idx[order]
        out[name] = (xc[sel], cp[sel], np.abs(cf[sel]))
    return out


def heat_curves(sd, values, *, alpha_deg: float = 0.0, about=(0.0, 0.0),
                length: float = 1.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A wall distribution (heat flux, local Nusselt number) on one body as {"upper": (x/L, v),
    "lower": ...}, split and ordered like :func:`surface_curves` so both tabs read alike."""
    xb, yb = rotate(sd.x, sd.y, alpha_deg, about)
    up, lo = split_loop(xb, yb)
    x0 = float(np.min(xb)) if len(xb) else 0.0
    xc = (xb - x0) / (length or 1.0)
    v = np.asarray(values, dtype=float)
    out = {}
    for name, idx in (("upper", up), ("lower", lo)):
        sel = idx[np.argsort(xc[idx], kind="stable")]
        out[name] = (xc[sel], v[sel])
    return out


def point_curves(pt) -> dict | None:
    """Cp / Cf branches of a PolarPoint with distributions (x/c from upper TE -> LE -> lower TE)."""
    if pt is None or pt.x is None or pt.cp is None or len(pt.x) < 3:
        return None
    x = np.asarray(pt.x, dtype=float)
    i_le = int(np.argmin(x))
    out = {}
    for name, sl in (("upper", slice(0, i_le + 1)), ("lower", slice(i_le, None))):
        xs = x[sl]
        order = np.argsort(xs, kind="stable")
        cp = np.asarray(pt.cp, dtype=float)[sl][order]
        cpi = np.asarray(pt.cp_inviscid, dtype=float)[sl][order] if pt.cp_inviscid is not None else None
        cf = np.abs(np.asarray(pt.cf, dtype=float)[sl][order]) if pt.cf is not None else None
        out[name] = (xs[order], cp, cf, cpi)
    return out


def lift_word(result, design) -> str:
    """'Downforce' for a vehicle producing negative lift, else 'Lift'."""
    if design is not None and design.kind == "vehicle" and finite(getattr(result, "lift", None)) \
            and result.lift < 0:
        return "Downforce"
    return "Lift"


def _re_matched(r: CFDResult) -> bool:
    return finite(r.re_real) and finite(r.re_sim) and abs(r.re_sim / r.re_real - 1.0) < 0.01


def _err(v) -> str:
    return fmt_err(v, "") or "—"


def _ratio_se(num: float, den: float, se_num: float, se_den: float) -> float:
    """First-order standard error of num / den for independent errors."""
    if not den or not all(finite(v) for v in (num, den, se_num, se_den)):
        return float("nan")
    r = num / den
    return abs(r) * math.hypot(se_num / num, se_den / den) if num else abs(se_num / den)


def is_lifting(design) -> bool:
    """Whether lift-to-drag is a meaningful figure for this design (airfoils and wings).

    For a road car or a bluff body L/D is a number without a use (4.02 for a
    sedan, −0.87 for a cylinder): it is neither optimised nor compared, so the
    pages leave it out and show lift with its sign instead.
    """
    return design is None or getattr(design, "kind", "airfoil") in ("airfoil", "wing")


def sim_kind(sim, design=None) -> str:
    """What a simulation produces: "polar", "wing" (finite-wing sweep), "cfd3d" or "cfd" (2D CFD).

    Results of different kinds are not compared like with like: a 2D Cd is per
    unit span, a 3D CD is on a reference area, and a wing sweep has no single
    operating point. The Compare page groups by this; "cfd" and "polar" form
    the 2D group, which the 2D comparison handles together.
    """
    if sim is None:
        return "cfd"
    if sim.analysis in ("polar", "wing"):
        return sim.analysis
    return "cfd3d" if design is not None and getattr(design, "dimension", 2) == 3 else "cfd"


def lift_label(design) -> str:
    """'Lift' for airfoils; 'Lift (downforce if negative)' for vehicles and other bodies."""
    return "Lift" if is_lifting(design) else "Lift (downforce if negative)"


def coef_reference(design, sim=None) -> str:
    """What the coefficients are referenced to: 'chord 1.000 m', 'frontal height 1.30 m'.

    A vehicle Cl of 2.54 means little until the reader knows it is referenced
    to the 2D frontal height rather than the length; the Summary table names
    the reference in a footnote and in the coefficient tooltips.
    """
    if design is None:
        return ""
    if design.kind == "airfoil":
        ref = f"chord {fmt_num(design.length_ref, 4)} m"
    elif design.kind == "vehicle":
        ref = f"frontal height {fmt_num(design.force_ref, 3)} m"
    else:
        ref = f"reference height {fmt_num(design.force_ref, 3)} m"
    if sim is not None:
        try:
            ref += f" and dynamic pressure {fmt_num(sim.conditions.dynamic_pressure(), 4)} Pa"
        except (AttributeError, TypeError, ValueError):
            pass
    return ref


#: The rounding sentence of every Summary footnote.
ROUNDING_NOTE = ("Each value is rounded to its standard error (one or two significant figures, the Particle Data "
                 "Group rule).")
#: Added when a row of the table is steady (and only then: on an unconverged room it promised what no row did).
STEADY_NOTE = ("“steady”: the averaging error is below 10⁻⁵ of the mean, far below the grid's error; such values "
               "have four significant figures.")


def rounding_note(rows=None) -> str:
    """The rounding sentences for a table of ``rows``: the rule, the steady case only when a row is steady,
    the warning-colour legend only when an error is drawn in it."""
    parts = [ROUNDING_NOTE]
    cells = [row[2] for row in rows or [] if len(row) > 2]
    if any(_cell_text(c) == STEADY_TEXT for c in cells):
        parts.append(STEADY_NOTE)
    if any(isinstance(c, Cell) and c.tone == "warn" for c in cells):
        parts.append(ROUGH_LEGEND)
    if any(_cell_text(c) == SET_TEXT for c in cells):
        parts.append("“0 (set)”: an input of the run, not a measured value.")
    if any(len(row) > 1 and _cell_text(row[1]) == WITHIN_ERROR_TEXT for row in rows or []):
        parts.append("“≈0”: zero within its standard error (the mean is smaller than the error).")
    return " ".join(parts)


def summary_note(design, sim, result=None, rows=None) -> str:
    """The footnote under the Summary table: the coefficients' reference, how values are rounded.

    A closed domain (a heated cavity, a room) shows no force coefficients: its
    footnote does not talk about coefficients it does not show. The steady
    sentence and the warning-colour legend appear only when ``rows`` (the
    table's rows) have such a cell (:func:`rounding_note`).
    """
    if rows is None and result is not None and isinstance(result, CFDResult):
        rows = summary_rows(result, design, sim)
    rounding = rounding_note(rows)
    if result is not None and (not has_forces(result) or (sim is not None and is_closed(sim))):
        return rounding
    ref = coef_reference(design, sim)
    first = f"Coefficients are referenced to the {ref}; forces are per metre of span." if ref else \
        "Forces are per metre of span."
    return " ".join(t for t in (first, buoyant_scale_caption(design, sim), rounding) if t)


def _cell_text(c) -> str:
    return c.text if isinstance(c, Cell) else str(c)


def blank_row(row) -> bool:
    """True when a Summary row has no value in any column (all "—"): a force row of a run without bodies."""
    return all(_cell_text(c).strip() in ("", "—") for c in row[1:6])


def has_forces(r) -> bool:
    """Whether a CFD result measured any force (a closed room or cavity without bodies has none)."""
    return any(finite(getattr(getattr(r, k, None), "mean", float("nan"))) for k in ("cd", "cl", "cm"))


def is_closed(sim) -> bool:
    """A closed domain (DESIGN.md 10.6: a room, a cavity), not a tunnel with a free stream."""
    return getattr(getattr(sim, "cfd", None), "domain_mode", "tunnel") == "closed"


def run_scales(design, sim):
    """The run's :class:`~aerolume.simulation.physics.PhysicsScales` (Ra, Pr, Gr, ΔT ...), or None.

    Recomputed from the design and simulation exactly as the planner does, so
    the numbers next to the result are the ones the run was set up with."""
    P = lazy_import("aerolume.simulation.physics")
    if P is None or design is None or sim is None or not hasattr(P, "physics_scales"):
        return None
    dim = int(getattr(design, "dimension", 2) or 2)
    try:
        box = getattr(sim.cfd, "domain_box", None) if is_closed(sim) else None
        return P.physics_scales(design, sim, dim, box=box)
    except Exception:  # noqa: BLE001 - a setting the planner refuses: no derived groups, not a crash
        return None


class BuoyantScale(NamedTuple):
    """The buoyant reference of a buoyancy-dominated tunnel run (:func:`buoyant_scale`)."""

    factor: float      #: U²/u_b²: a free-stream coefficient times this is the coefficient on ½ρu_b²
    u_b: float         #: the buoyant velocity [m/s] (:func:`~aerolume.simulation.physics.buoyant_speed`)
    q_b: float         #: ½ρu_b² [Pa]
    ri: float          #: Gr/Re² of the flow, above :data:`~aerolume.simulation.physics.RI_BUOYANT`


def buoyant_scale(design, sim) -> BuoyantScale | None:
    """The buoyant scale of a tunnel run whose buoyancy outruns its free stream, else None.

    Exactly when the planner's :func:`~aerolume.simulation.physics.buoyancy_dominated_warning`
    applies: the free stream's q = ½ρU² is then tiny and the coefficients on it are mostly the
    buoyant force over a small q (the hot sphere at Ri 627 read CD 6.14 and CL 119). Divided by
    ½ρu_b² instead they read as coefficients again (CD 0.0098, CL 0.19), so the Summary shows both.
    A closed box needs none: its velocity scale already is the buoyant one."""
    th = getattr(sim, "thermal", None) if sim is not None else None
    if design is None or th is None or is_closed(sim) or not (getattr(th, "enabled", False)
                                                              and getattr(th, "buoyancy", False)):
        return None
    P = lazy_import("aerolume.simulation.physics")
    if P is None or not hasattr(P, "buoyancy_dominated_warning") or not hasattr(P, "buoyant_speed"):
        return None
    sc = run_scales(design, sim)
    try:
        if sc is None or not P.buoyancy_dominated_warning(sc, closed=False):
            return None
        ub, U = float(P.buoyant_speed(sc)), float(sc.speed)
        rho = float(sc.props.density) if sc.props is not None else float("nan")
    except Exception:  # noqa: BLE001 - scales the planner could not finish: the free-stream rows only
        return None
    if not (finite(ub) and ub > 0 and finite(U) and U > 0):
        return None
    return BuoyantScale((U / ub) ** 2, ub, 0.5 * rho * ub * ub, float(sc.ri_flow))


def buoyant_scale_rows(r, design, sim, window: str, coefficients: Sequence[tuple[str, str]]) -> list[list]:
    """Summary rows of ``coefficients`` ((result attribute, symbol) pairs) on the buoyant scale ½ρu_b²,
    for a buoyancy-dominated tunnel run (:func:`buoyant_scale`); [] otherwise. Each is the free-stream
    coefficient's statistics times U²/u_b², rounded to its own standard error like every row."""
    bs = buoyant_scale(design, sim)
    if bs is None:
        return []
    tip = (f"On q_b = ½ρu_b² = {sig(bs.q_b, 3)} Pa, u_b = {sig(bs.u_b, 3)} m/s the buoyant velocity, instead of "
           f"the free stream's q: buoyancy dominates (Ri = Gr/Re² = {fmt_sig(bs.ri, 3)}). The free-stream "
           f"coefficient × U²/u_b² = {fmt_sig(bs.factor, 3)}.")
    out = []
    for key, symbol in coefficients:
        st = getattr(r, key, None)
        if st is not None and finite(getattr(st, "mean", float("nan"))):
            out.append(stat_cells(f"{symbol} on the buoyant scale", st, tip=tip, window=window, scale=bs.factor))
    return out


def buoyant_scale_caption(design, sim) -> str:
    """The one line under the Summary that names u_b, when :func:`buoyant_scale_rows` added rows."""
    bs = buoyant_scale(design, sim)
    if bs is None:
        return ""
    return (f"Buoyancy dominates (Ri {fmt_sig(bs.ri, 3)}): the free-stream coefficients are mostly buoyant force "
            f"over a small q; “on the buoyant scale” divides by ½ρu_b² = {sig(bs.q_b, 3)} Pa with the buoyant "
            f"velocity u_b = {sig(bs.u_b, 3)} m/s.")


def natural_convection(design, sim) -> bool:
    """True when the run's speed scale is the free-fall velocity (no forced flow): Re says nothing then."""
    sc = run_scales(design, sim) if is_closed(sim) else None
    return bool(sc is not None and getattr(sc, "natural", False))


def reference_value(r, design, sim) -> tuple[float, str, str] | None:
    """(value, quantity, source) of the published reference a reference-case run reproduces, or None.

    From :func:`aerolume.api.reference_case`: the de Vahl Davis cavity, the
    Lange heated cylinder ... and only while the run's settings still match
    the published conditions."""
    api = lazy_import("aerolume.api")
    fn = getattr(api, "reference_case", None) if api is not None else None
    if fn is None or design is None or sim is None:
        return None
    try:
        case, _why = fn(design, sim, re=getattr(r, "re_sim", None))
    except Exception:  # noqa: BLE001 - no reference rather than a broken Summary
        return None
    ref = getattr(case, "reference", None) if case is not None else None
    if not ref or not finite(ref.get("value")) or ref.get("kind") == "consistency":
        return None      # a consistency check (a fan's imposed thrust) is no published reference
    return float(ref["value"]), str(ref.get("quantity", "")), str(ref.get("source", ""))


#: Published mean Nusselt numbers of an isothermal cylinder, Pr 0.71, by Reynolds number: the spread a
#: run is judged by (docs/validation.md, the Heated cylinder template). The Lange et al. (1998)
#: correlation the Summary quotes is one number 0.8-2.5 % above this spread at Re 40 -- its own
#: scatter -- so quoting it alone overstated how precisely Nu is known. The values are those of
#: validation/thermal2d_cases.py ``CYLINDER_NU_REF`` (Singhal, Dutta & Kalita 2023, Table 2).
CYLINDER_NU_PUBLISHED_RANGE = {
    20.0: (2.433, 2.4483, "Jafroudi & Yang 1986; Biswas & Sarkar 2009"),
    40.0: (3.200, 3.255, "Jafroudi & Yang 1986; Biswas & Sarkar 2009; Apelt & Ledwich 1979"),
}
#: Acceptance band around a published range in docs/validation.md for the heated cylinder: ± 4 %
#: (its 5 % blockage raises Nu); beyond it the range row reads amber.
PUBLISHED_RANGE_TOL = 0.04


def published_range(r, design, sim, ref_source: str = "") -> tuple[float, float, str] | None:
    """(low, high, sources) of the published values of a reference case, or None.

    From the case's own ``reference["range"]`` when the API gives one, else the
    tabulated cylinder values (:data:`CYLINDER_NU_PUBLISHED_RANGE`) for a Lange
    cylinder run at Re 20 or 40 (within 1 %)."""
    api = lazy_import("aerolume.api")
    fn = getattr(api, "reference_case", None) if api is not None else None
    if fn is not None and design is not None and sim is not None:
        try:
            case, _why = fn(design, sim, re=getattr(r, "re_sim", None))
            rng = (getattr(case, "reference", None) or {}).get("range") if case is not None else None
            if rng and len(rng) >= 2 and finite(rng[0]) and finite(rng[1]):
                return float(rng[0]), float(rng[1]), str(rng[2]) if len(rng) > 2 else ""
        except Exception:  # no range rather than a broken Summary
            log.debug("no published range", exc_info=True)
    if not ref_source.startswith("Lange"):
        return None
    re_ = float(getattr(r, "re_sim", float("nan")))
    for re_ref, rng in CYLINDER_NU_PUBLISHED_RANGE.items():
        if finite(re_) and abs(re_ / re_ref - 1.0) <= 0.01:
            return rng
    return None


def range_text(measured: float, lo: float, hi: float) -> tuple[str, str | None]:
    """("+2.5 % above it" / "inside" / "−1.0 % below it", tone) of a measured value against a range."""
    if not finite(measured):
        return "", None
    if lo <= measured <= hi:
        return "measured inside", "good"
    edge = hi if measured > hi else lo
    d = 100.0 * (measured / edge - 1.0)
    word = "above" if d > 0 else "below"
    tone = None if abs(d) <= 100 * PUBLISHED_RANGE_TOL else "warn"
    return f"measured {d:+.1f} % {word}".replace("-", MINUS), tone


def reference_rows(r, design, sim) -> list[list]:
    """Summary rows quoting the published reference and how far the measured value is from it: the
    reference value (a correlation, a benchmark) and, where the literature gives a spread, the
    published range with the distance from its nearer edge."""
    ref = reference_value(r, design, sim)
    if ref is None:
        return []
    value, quantity, source = ref
    measured, se = float("nan"), float("nan")
    if quantity in ("nusselt", "nu"):
        P = _physics()
        st = P.nusselt_stat(r) if P is not None else None
        if st is not None:
            measured, se = float(st.mean), float(st.stderr)
        name = "Nu"
    elif quantity in ("cd", "drag_coefficient"):
        measured, se = float(r.cd.mean), float(r.cd.stderr)
        name = "Cd"
    else:
        return []        # a quantity the Summary does not measure: nothing to compare against
    who = source.split(",")[0].split(":")[0].strip() or "published value"
    diff = 100.0 * (measured / value - 1.0) if finite(measured) and value else float("nan")
    tone = None
    if finite(diff):
        tone = "good" if abs(diff) <= 1.0 else "warn" if abs(diff) > 3.0 else None
    tip = (f"{source}. Measured {fmt_sig(measured, 4)}" + (f" ± {fmt_sig(se, 2)}" if finite(se) and se > 0 else "")
           + f", {diff:+.1f} % from the reference." if finite(diff) else source)
    rows = [[Cell(f"Reference {name}: {who}", tooltip=tip), Cell(fmt_sig(value, 4), tooltip=source), "—", "—", "—",
             "—", Cell(f"measured {diff:+.1f} %".replace("-", MINUS) if finite(diff) else "", tone=tone,
                       tooltip=tip)]]
    rng = published_range(r, design, sim, source) if name == "Nu" else None
    if rng is not None:
        lo, hi, who_r = rng
        text, rtone = range_text(measured, lo, hi)
        rtip = (f"Published values {fmt_sig(lo, 4)}–{fmt_sig(hi, 4)} ({who_r}). docs/validation.md accepts "
                f"the range ± {100 * PUBLISHED_RANGE_TOL:.0f} % for this case (its 5 % blockage raises Nu).")
        rows.append([Cell(f"Published range {name}", tooltip=rtip),
                     Cell(f"{fmt_sig(lo, 4)}–{fmt_sig(hi, 4)}", tooltip=rtip), "—", "—", "—", "—",
                     Cell(text, tone=rtone, tooltip=rtip)])
    return rows


def summary_rows(r: CFDResult, design, sim) -> list[list]:
    """Rows of the Summary table: quantity, mean, ± standard error, RMS, min, max, unit.

    Dimensional rows are the coefficient statistics scaled by the same factor
    that turned the mean coefficient into the stored mean force, so the mean
    column equals ``r.drag`` / ``r.lift`` exactly and the uncertainty scales with it.

    Each row is rounded to its own standard error: the error by the Particle
    Data Group rule (one or two significant figures, :func:`round_error`), the
    mean, RMS, min and max to the same decimal place; rows without an error get
    four significant figures. The table then pads the numbers so the decimal
    points line up. L/D appears for lifting sections only (:func:`is_lifting`).
    A closed domain (:func:`is_closed`) leads with temperatures and heat paths
    instead of force coefficients (:func:`closed_thermal_rows`).
    """
    window = (f"Mean over t = {sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U"
              if finite(r.t_avg_start) and finite(r.t_end) else "Time average")

    def k_from(value: float, coef: float) -> float:
        return value / coef if finite(value) and finite(coef) and coef else float("nan")

    k_force = k_from(r.drag, r.cd.mean)
    if not finite(k_force):
        k_force = k_from(r.lift, r.cl.mean)
    k_moment = k_from(r.moment, r.cm.mean)
    force_ref = design.force_ref if design is not None else k_from(r.drag_area, r.cd.mean)
    speed = sim.conditions.speed if sim is not None else float("nan")
    ref = coef_reference(design)
    ref_tip = f"Referenced to the {ref}" if ref else ""

    def stat_row(name: str, st, k: float = 1.0, unit: str = "", mean=None, tip: str = "") -> list:
        """One quantity at one decimal count across mean / error / RMS / min / max."""
        return stat_cells(name, st, unit, tip=tip, window=window, scale=k, mean=mean)

    def coef_tip(text: str) -> str:
        return "; ".join(t for t in (text, ref_tip) if t)

    if sim is not None and is_closed(sim):
        # a room, an enclosure, a cavity: the coefficients of the furniture inside it (the circuit board's
        # "Cd −0.0302") mean nothing here; they stay in Forces. The rows lead with temperatures and heat paths.
        rows = closed_thermal_rows(r, design, sim, window) + closed_fan_rows(r, sim, window)
        if not natural_convection(design, sim):
            rows += [[Cell("Reynolds number", tooltip="From the run's velocity scale (Conditions) and the "
                                                      "reference length"), fmt_re(r.re_sim), "—", "—", "—", "—", ""]]
        rows += reference_rows(r, design, sim)
        return [row for row in rows if not blank_row(row)]

    rows = [
        stat_row("Drag coefficient Cd", r.cd, tip=coef_tip("")),
        stat_row("Lift coefficient Cl", r.cl, tip=coef_tip("Positive up" if not is_lifting(design) else "")),
        stat_row("Moment coefficient Cm", r.cm,
                 tip=coef_tip("Positive nose-up, about the moment reference point")),
    ]
    # a buoyancy-dominated tunnel run: the coefficients on ½ρu_b², right under the free-stream ones they correct
    rows += buoyant_scale_rows(r, design, sim, window, (("cd", "Cd"), ("cl", "Cl")))
    if is_lifting(design):
        ld_se = _ratio_se(r.cl.mean, r.cd.mean, r.cl.stderr, r.cd.stderr)
        ld_t, ld_se_t = value_error_texts(r.lift_to_drag, ld_se)
        rows.append([Cell("Lift-to-drag ratio L/D"),
                     Cell(ld_t, tooltip="Ratio of the means"),
                     Cell(ld_se_t, tooltip="First-order propagation of the Cd and Cl standard errors"),
                     "—", "—", "—", ""])
    rows += [
        stat_row("Drag", r.cd, k_force, "N/m", mean=r.drag, tip="Per metre of span"),
        stat_row(lift_label(design), r.cl, k_force, "N/m", mean=r.lift, tip="Per metre of span"),
        stat_row("Pitching moment", r.cm, k_moment, "N·m/m", mean=r.moment),
        stat_row("Drag area CdA", r.cd, force_ref, "m²/m", mean=r.drag_area,
                 tip="Cd × reference height (2D analogue of the drag area)"),
        stat_row("Drag power", r.cd, k_force * speed if finite(speed) else float("nan"), "W/m", mean=r.power,
                 tip="Power needed to overcome the mean drag, per metre of span"),
        [Cell("Strouhal number St", tooltip="Dominant shedding frequency × reference length / speed"),
         fmt_num(r.strouhal, 3) if r.strouhal is not None else "—", "—", "—", "—", "—", ""],
        ["Reynolds number (real)", fmt_re(r.re_real), "—", "—", "—", "—", ""],
        [Cell("Reynolds number (simulated)", tooltip="Equal to the real value unless it was capped"),
         Cell(fmt_re(r.re_sim), tone=None if _re_matched(r) else "warn"), "—", "—", "—", "—", ""],
    ]
    if natural_convection(design, sim):
        # natural convection: the "speed" is the free-fall velocity, so a Reynolds number of 119 is a
        # derived scale, not a condition; Ra, Pr and Gr in Conditions describe the flow instead
        rows = [row for row in rows if not _cell_text(row[0]).startswith("Reynolds number")]
    rows += thermal_summary_rows(r, window, sim, design)
    rows += reference_rows(r, design, sim)
    # rows with no value in any column (nine force rows of a cavity without bodies) say nothing
    return [row for row in rows if not blank_row(row)]


# ---------------------------------------------------------------------------
# placed physics (DESIGN.md 10): heat transfer, fans, solver diagnostics
# ---------------------------------------------------------------------------

def _physics():
    """``aerolume.analysis.physics`` (the one reader Results, Compare and the report share), or None."""
    return lazy_import("aerolume.analysis.physics")


def has_heat(result) -> bool:
    """Whether a Navier-Stokes result carries heat-transfer output."""
    P = _physics()
    if P is None or not isinstance(result, (CFDResult, CFD3DResult)):
        return False
    try:
        return bool(P.has_thermal(result))
    except Exception:  # noqa: BLE001 - a malformed result shows no heat tab, never a crash
        return False


def run_is_thermal(result, sim) -> bool:
    """Whether a run solved for temperature: its simulation's heat-transfer switch (a room heated by a
    heat source has a temperature field but no Nusselt number), else what the result carries."""
    th = getattr(sim, "thermal", None) if sim is not None else None
    if th is not None:
        return bool(getattr(th, "enabled", False))
    return has_heat(result)


#: Share of the heat input still being stored in the fluid at the end of a run above which the
#: Results header says so next to the convergence verdict. The verdict judges the monitored
#: quantities (forces, Nu, fan flow), not the temperature field as a whole: the heated-cylinder
#: example read "Converged" while 9 % of its heat was still warming the wake (critic round 3.4).
STORED_HEAT_FACT = 0.05


def stored_heat_fact(result) -> tuple[str, bool] | None:
    """("12 % of the heat still being stored", unsettled) for the Results header, or None.

    The share comes from the heat balance (:func:`aerolume.simulation.physics_run.storage_share`);
    it is stated from :data:`STORED_HEAT_FACT` up, and ``unsettled`` is True above the runner's own
    threshold (``STORAGE_UNSETTLED``, where the run also warns and marks its Nusselt error unreliable),
    which the header colours as a warning even when the monitors converged."""
    try:
        from ...simulation.physics_run import STORAGE_UNSETTLED, storage_share
        share = storage_share(getattr(result, "diagnostics", None) or {})
    except Exception:  # noqa: BLE001 - an advisory fact: a result without a heat balance shows none
        return None
    if share is None or not math.isfinite(share) or share < STORED_HEAT_FACT:
        return None
    return f"{100 * share:.0f} % of the heat still being stored", share > STORAGE_UNSETTLED


def has_fan_report(result) -> bool:
    P = _physics()
    return bool(P is not None and isinstance(result, (CFDResult, CFD3DResult)) and P.has_fans(result))


def _units2d(result) -> tuple[str, str, str]:
    """(power, force, volume-flow) units: per metre of span in 2D."""
    return ("W", "N", "m³/s") if isinstance(result, CFD3DResult) else ("W/m", "N/m", "m²/s")


#: Tooltip of an error cell drawn in the warning tone, and the Summary footnote's legend for that colour.
ROUGH_TIP = ("Rough estimate: the averaging window holds few independent batches, so this standard error "
             "may be off by a factor of two. Run longer for a firm one.")
ROUGH_LEGEND = "Errors in the warning colour are rough (a short averaging window; see their tooltips)."
#: Error column of a quantity the user set rather than measured (a heater's power, a fan's thrust).
SET_TEXT = "0 (set)"
SET_TIP = "Set in Setup, not measured: it carries no averaging error"


def error_cell(text: str, *, stderr=float("nan"), reliable: bool = True) -> Cell:
    """The ± column of a Summary row: the steady tooltip, or the warning tone with :data:`ROUGH_TIP` for a
    rough batch-means error, so every coloured error explains itself the same way."""
    if text == STEADY_TEXT:
        extra = f"; batch-means standard error {fmt_sig(stderr, 2)}" if finite(stderr) and stderr > 0 else ""
        return Cell(text, tooltip=STEADY_TIP + extra)
    if text == SET_TEXT:
        return Cell(text, tooltip=SET_TIP)
    if text not in ("", "—") and not reliable:
        return Cell(text, tone="warn", tooltip=ROUGH_TIP)
    return Cell(text)


def mean_cell(text: str, *, mean=float("nan"), stderr=float("nan"), window: str = "") -> Cell:
    """The Mean column: the averaging window as tooltip, or why a value reads "≈0" (zero within its error)."""
    if text == WITHIN_ERROR_TEXT:
        return Cell(text, tone="muted",
                    tooltip=f"The mean {fmt_sig(mean, 3)} is smaller than its standard error {fmt_sig(stderr, 2)}: "
                            "it cannot be told from zero")
    return Cell(text, tooltip=window)


def stat_cells(name: str, st, unit: str = "", *, tip: str = "", window: str = "", scale: float = 1.0,
               mean=None) -> list:
    """A full Summary row (quantity, mean, ± error, RMS, min, max, unit) of a :class:`~aerolume.model.Stat`,
    every number through :func:`stat_texts` (one rounding rule for every row of every table)."""
    k = float(scale) if finite(scale) else float("nan")
    lo, hi = st.min * k, st.max * k
    if k < 0:
        lo, hi = hi, lo
    m = st.mean * k if mean is None else float(mean)
    se = st.stderr * abs(k)
    mean_t, se_t, rms_t, lo_t, hi_t = stat_texts(m, se, st.rms * abs(k), lo, hi)
    return [Cell(name, tooltip=tip), mean_cell(mean_t, mean=m, stderr=se, window=window),
            error_cell(se_t, stderr=se, reliable=bool(getattr(st, "stderr_reliable", True))),
            rms_t, lo_t, hi_t, unit]


def _value_err_row(name: str, value: float, se: float, unit: str, *, reliable: bool = True, tip: str = "",
                   window: str = "") -> list:
    """A Summary-style row of a value with a standard error but no RMS / min / max."""
    mean_t, se_t = value_error_texts(value, se)
    return [Cell(name, tooltip=tip), mean_cell(mean_t, mean=value, stderr=se, window=window),
            error_cell(se_t, stderr=se, reliable=reliable), "—", "—", "—", unit]


def _plain_row(name: str, text: str, unit: str = "", *, tip: str = "", err: str = "—") -> list:
    """A row with a single value (a temperature read from the field, a set power): no statistics."""
    return [Cell(name, tooltip=tip), Cell(text, tooltip=tip), error_cell(err), "—", "—", "—", unit]


def body_heat_signs(r) -> list[float]:
    """Signs (+1 heated, −1 cooled) of the bodies' heat rates into the fluid, those that report one."""
    out = []
    for d in (getattr(r, "per_body", None) or {}).values():
        q = d.get("heat_rate", d.get("q"))
        if finite(q) and q != 0:
            out.append(math.copysign(1.0, float(q)))
    return out


def mean_nu_meaningful(r, sim=None) -> bool:
    """Whether one mean Nusselt number over the run says something.

    Not when bodies of opposite sign (a 200 W radiator and a 12 °C window) are
    summed into one signed mean (the 3D room read "Nu −53.6"), not when a
    closed room is driven by heat sources (its Nu is over walls that take heat
    out, not over a heated surface), and not when it is negative (heat flowing
    into the surfaces it is defined on). A differentially heated cavity (walls
    only) and a single heated body in a tunnel keep theirs.
    """
    P = _physics()
    nu = P.nusselt_stat(r) if P is not None else None
    if nu is None or not finite(nu.mean) or nu.mean <= 0:
        return False
    if len(set(body_heat_signs(r))) > 1:
        return False
    heated_room = sim is not None and is_closed(sim) and any(getattr(h, "enabled", True)
                                                             for h in getattr(sim, "heat_sources", None) or [])
    return not heated_room


def _thermal_bodies(r, design) -> dict:
    """{name: body} of the design's bodies (2D) or meshes (3D) that the result reports heat for."""
    items = list(getattr(design, "meshes", []) or []) if isinstance(r, CFD3DResult) else \
        list(getattr(design, "bodies", []) or []) if design is not None else []
    pb = getattr(r, "per_body", None) or {}
    return {getattr(b, "name", ""): b for b in items if getattr(b, "name", "") in pb}


def heat_rate_is_set(r, design) -> bool:
    """True when every heated body has a set wall heat flux: the heat rate is then q″ × area, an input of
    the run (the heated box's "156.0 steady"), and its wall temperature is the result."""
    bodies = _thermal_bodies(r, design)
    modes = [getattr(b, "thermal", "") for b in bodies.values() if getattr(b, "thermal", "") != "adiabatic"]
    return bool(modes) and all(m == "heat_flux" for m in modes)


def thermal_summary_rows(r, window: str = "", sim=None, design=None) -> list[list]:
    """Summary rows of a heat-transfer run: the mean Nusselt number (with its statistics) and the heat rate.

    The mean Nu only where :func:`mean_nu_meaningful` holds; the heat rate is
    the one into the fluid from the bodies (a tunnel run) or the heated wall (a
    cavity), marked as set when the bodies have a set heat flux
    (:func:`heat_rate_is_set`), whose wall temperatures are then listed. A
    closed room has its own rows (:func:`closed_thermal_rows`).
    """
    if not has_heat(r):
        return []
    P = _physics()
    rows = []
    nu = P.nusselt_stat(r)
    if nu is not None and mean_nu_meaningful(r, sim):
        rows.append(stat_cells("Mean Nusselt number Nu", nu, window=window,
                               tip="Nu = Q L / (k ΔT A) over the heated surfaces"))
    q, se, ok = P.heat_rate_stat(r)
    if finite(q) and (len(set(body_heat_signs(r))) <= 1):
        if heat_rate_is_set(r, design):
            rows.append(_plain_row("Heat input Q (set heat flux)", fmt_sig(q, 4), _units2d(r)[0], err=SET_TEXT,
                                   tip="The wall heat flux set in Setup times the wall area: an input of the run"))
            for name, b in _thermal_bodies(r, design).items():
                tw = (r.per_body or {}).get(name, {}).get("t_wall")
                if getattr(b, "thermal", "") == "heat_flux" and finite(tw):
                    rows.append(_plain_row(f"Wall temperature: {name}", fmt_fixed(float(tw), 1), _temp_unit(),
                                           tip="The mean temperature of the wall at its set heat flux"))
        else:
            rows.append(_value_err_row("Heat transfer rate Q", q, se, _units2d(r)[0], reliable=ok, window=window,
                                       tip="From the heated surfaces into the fluid"))
    return rows


# -- closed domains (rooms, enclosures, cavities): what the user came for is temperatures and heat paths

def _temp_unit() -> str:
    return "°C"


def fluid_temperatures(r) -> dict | None:
    """{"mean", "max", "min", "source"} of the fluid's temperature [°C], or None without a temperature.

    From the time-averaged field when the result stores one (the mean over
    the fluid cells, solids masked out), else the snapshot, else the solver's
    end-of-run diagnostics (``t_fluid_mean`` / ``_max`` / ``_min``). ``source``
    names which, for the tooltip.
    """
    blocks = []
    if isinstance(r, CFD3DResult):
        from ..widgets.view3d_panel import heat_volume_of, temperature_field_of
        vol = heat_volume_of(r)
        if vol is not None:
            name = temperature_field_of(vol)
            blocks.append((vol.fields.get(name) if name == "temperature" else None, vol.mask,
                           "the time-averaged temperature field" if vol.label == "mean" else "the last snapshot"))
    else:
        for fd in (getattr(r, "mean_fields", None), getattr(r, "instant_fields", None)):
            if fd is not None and "temperature" in fd.fields:
                blocks.append((fd.fields["temperature"], fd.mask,
                               "the time-averaged temperature field" if fd.label == "mean" else "the last snapshot"))
    for vals, mask, source in blocks:
        if vals is None:
            continue
        v = np.asarray(vals, dtype=float)
        if mask is not None and np.shape(mask) == v.shape:
            v = v[np.asarray(mask) < 0.5]
        v = v[np.isfinite(v)]
        if v.size:
            return {"mean": float(v.mean()), "max": float(v.max()), "min": float(v.min()), "source": source}
    d = getattr(r, "diagnostics", None) or {}
    if all(finite(d.get(k)) for k in ("t_fluid_mean", "t_fluid_max")):
        return {"mean": float(d["t_fluid_mean"]), "max": float(d["t_fluid_max"]),
                "min": float(d.get("t_fluid_min", float("nan"))), "source": "the end of the run"}
    return None


def _footprint_max(r, src, alpha_deg: float = 0.0, about=(0.0, 0.0)) -> float:
    """Highest fluid temperature [°C] inside a heat source's footprint (its rectangle / box, disc /
    sphere), nan when the result has no temperature field there. The component temperature a user
    asks about ("how hot does the processor get") is this, not the room's mean."""
    pos = tuple(float(c) for c in getattr(src, "position", ()) or ())
    size = tuple(float(c) for c in getattr(src, "size", ()) or ())
    if not pos or not size:
        return float("nan")
    kind = str(getattr(src, "kind", "rect"))
    if isinstance(r, CFD3DResult):
        from ..widgets.view3d_panel import heat_volume_of
        vol = heat_volume_of(r)
        if vol is None or "temperature" not in vol.fields or len(pos) < 3:
            return float("nan")
        t = np.asarray(vol.fields["temperature"], dtype=float)
        nz, ny, nx = t.shape
        (ox, oy, oz), (dx, dy, dz) = vol.origin, vol.spacing
        zc = oz + (np.arange(nz) + 0.5) * dz
        yc = oy + (np.arange(ny) + 0.5) * dy
        xc = ox + (np.arange(nx) + 0.5) * dx
        Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
        if kind == "sphere":
            rad = max(size[0], 0.5 * max(dx, dy, dz))
            inside = (X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2 <= rad * rad
        else:
            half = [max(0.5 * (size[i] if i < len(size) else size[0]), 0.5 * h) for i, h in enumerate((dx, dy, dz))]
            inside = (np.abs(X - pos[0]) <= half[0]) & (np.abs(Y - pos[1]) <= half[1]) & (np.abs(Z - pos[2]) <= half[2])
        if vol.mask is not None and np.shape(vol.mask) == t.shape:
            inside &= np.asarray(vol.mask) < 0.5
    else:
        fd = r.mean_fields if r.mean_fields is not None and "temperature" in r.mean_fields.fields else r.instant_fields
        if fd is None or "temperature" not in fd.fields:
            return float("nan")
        t = np.asarray(fd.fields["temperature"], dtype=float)
        ny, nx = t.shape
        x0, x1, y0, y1 = fd.extent
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * dx, y0 + (np.arange(ny) + 0.5) * dy)
        cx, cy = pos[0], pos[1]
        if getattr(src, "frame", "body") == "body" and alpha_deg:
            cxa, cya = rotate(np.array([cx]), np.array([cy]), -alpha_deg, about)
            cx, cy = float(cxa[0]), float(cya[0])
        if kind == "disc":
            rad = max(size[0], 0.5 * max(dx, dy))
            inside = (X - cx) ** 2 + (Y - cy) ** 2 <= rad * rad
        else:
            hx = max(0.5 * size[0], 0.5 * dx)
            hy = max(0.5 * (size[1] if len(size) > 1 else size[0]), 0.5 * dy)
            inside = (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)
        if fd.mask is not None and np.shape(fd.mask) == t.shape:
            inside &= np.asarray(fd.mask) < 0.5
    vals = t[inside]
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else float("nan")


def hottest_source(r, sim, design=None):
    """(heat source, its peak temperature) of the hottest enabled heater, or (None, nan)."""
    alpha = float(getattr(getattr(sim, "conditions", None), "alpha_deg", 0.0) or 0.0)
    about = getattr(design, "moment_ref", (0.0, 0.0)) if design is not None else (0.0, 0.0)
    best, best_t = None, float("nan")
    for src in getattr(sim, "heat_sources", None) or []:
        if not getattr(src, "enabled", True) or not finite(getattr(src, "power", None)) or src.power <= 0:
            continue
        t = _footprint_max(r, src, alpha, about)
        if finite(t) and (best is None or not finite(best_t) or t > best_t):
            best, best_t = src, t
    return best, best_t


def closed_thermal_rows(r, design, sim, window: str = "") -> list[list]:
    """The heat rows of a closed domain (a room, an enclosure, a cavity), the quantities a user asks for.

    In order: the hottest the fluid gets and its mean; each heater's set
    power ("Heater power (set)", never presented as a result) and the peak
    temperature in its footprint; each body's heat rate, split into heated
    and cooled, with its own Nusselt number (never one signed mean over
    both); the heat that leaves through the walls and the vents, and what is
    still being stored (a room still warming up). A wall-heated cavity, which
    has no heaters or bodies, keeps its mean Nusselt number and heat rate.
    """
    if sim is None or not run_is_thermal(r, sim):
        return []
    P = _physics()
    unit_q = _units2d(r)[0]
    rows: list[list] = []
    temps = fluid_temperatures(r)
    if temps is not None:
        tip = f"From {temps['source']}"
        rows.append(_plain_row("Maximum fluid temperature", fmt_fixed(temps["max"], 1), _temp_unit(), tip=tip))
        rows.append(_plain_row("Mean fluid temperature", fmt_fixed(temps["mean"], 1), _temp_unit(),
                               tip=tip + ", averaged over the fluid cells (solids left out)"))
    alpha = float(getattr(sim.conditions, "alpha_deg", 0.0) or 0.0)
    about = design.moment_ref if design is not None and hasattr(design, "moment_ref") else (0.0, 0.0)
    for src in getattr(sim, "heat_sources", None) or []:
        if not getattr(src, "enabled", True):
            continue
        name = getattr(src, "name", "") or "Heater"
        rows.append(_plain_row(f"Heater power (set): {name}", f"{float(src.power):.6g}".replace("-", MINUS), unit_q,
                               tip="The power set in Setup: an input of the run, not a result", err=SET_TEXT))
        peak = _footprint_max(r, src, alpha, about)
        if finite(peak):
            rows.append(_plain_row(f"Peak temperature at {name}", fmt_fixed(peak, 1), _temp_unit(),
                                   tip=f"The highest fluid temperature inside {name}'s footprint "
                                       f"({'time-averaged' if not isinstance(r, CFD3DResult) else 'stored'} field)"))
    items = list(getattr(design, "meshes", []) or []) if isinstance(r, CFD3DResult) else \
        list(getattr(design, "bodies", []) or []) if design is not None else []
    conds = {getattr(b, "name", ""): b for b in items}
    for name, d in (getattr(r, "per_body", None) or {}).items():
        q = d.get("heat_rate", d.get("q"))
        if not finite(q):
            continue
        heated = float(q) >= 0
        word = "heated" if heated else "cooled"
        rows.append(_value_err_row(f"Heat rate: {name} ({word})", float(q), d.get("heat_rate_stderr", float("nan")),
                                   unit_q, window=window,
                                   tip="Positive: from the body into the fluid; negative: the fluid loses heat to it"))
        nu = d.get("nu", d.get("nusselt"))
        if finite(nu):
            rows.append(_value_err_row(f"Nusselt number: {name}", float(nu), d.get("nu_stderr", float("nan")), "",
                                       window=window, tip="Nu = Q L / (k ΔT A) of this body alone"))
        b = conds.get(name)
        if b is not None and getattr(b, "thermal", "") == "heat_flux" and finite(d.get("t_wall")):
            rows.append(_plain_row(f"Wall temperature: {name}", fmt_fixed(float(d["t_wall"]), 1), _temp_unit(),
                                   tip="The mean temperature of the body's wall at its set heat flux"))
    hb = P.heat_balance(r) if P is not None else None
    diag = getattr(r, "diagnostics", None) or {}
    if hb:
        # what counts as "nothing": a cavity's walls net to 4×10⁻⁹ W/m against 0.10 W/m through its hot wall
        scale = max([abs(float(v)) for v in (hb.get("total_in"), hb.get("total_out"), getattr(r, "heat_rate", None),
                                             *[getattr(h, "power", None) for h in getattr(sim, "heat_sources", [])])
                     if finite(v)] + [1e-300])
        walls = diag.get("heat_in_walls")
        if finite(walls) and abs(float(walls)) > 1e-3 * scale:
            w = float(walls)
            rows.append(_plain_row("Heat lost through the walls" if w < 0 else "Heat gained through the walls",
                                   fmt_sig(abs(w), 4), unit_q, tip="Net heat through the domain walls with a set "
                                   "temperature or heat flux, at the end of the run"))
        ports = [p for p in getattr(sim, "ports", None) or [] if getattr(p, "enabled", True)]
        out_ports = float(diag.get("heat_out_ports", 0.0) or 0.0) - float(diag.get("heat_in_ports", 0.0) or 0.0)
        out_ports += float(diag.get("heat_out_outlet", 0.0) or 0.0) - float(diag.get("heat_in_inlet", 0.0) or 0.0)
        if ports and finite(out_ports):
            rows.append(_plain_row("Heat removed by the vents (net)", fmt_sig(out_ports, 4), unit_q,
                                   tip="Enthalpy carried out through the vents minus that brought in by the jets, "
                                       "at the end of the run"))
        st = hb.get("storage")
        if finite(st) and abs(float(st)) > 0.01 * scale:
            rows.append(_plain_row("Still being stored (warming)" if st > 0 else "Still being released (cooling)",
                                   fmt_sig(abs(float(st)), 4), unit_q,
                                   tip="The rate the fluid's heat content still changes at the end of the run: the "
                                       "room has not reached a thermal steady state while this is a sizeable share "
                                       "of the heat input"))
    # a wall-heated cavity: no heaters, no heated bodies -> its mean Nu and heat rate describe it
    if not any(getattr(h, "enabled", True) for h in getattr(sim, "heat_sources", None) or []) \
            and not body_heat_signs(r):
        rows += thermal_summary_rows(r, window, sim, design)
    return rows


def heat_tab_rows(r, design, sim, window: str = "") -> list[list]:
    """The table at the top of the Heat transfer tab: the closed-domain rows in a room or an enclosure,
    the mean Nusselt number and heat rate (where they mean something) elsewhere."""
    if sim is not None and is_closed(sim):
        return closed_thermal_rows(r, design, sim, window)
    return thermal_summary_rows(r, window, sim, design)


def heat_tab_title(sim) -> str:
    """Heading of the Heat transfer tab's table: a room is read by its temperatures and heat paths."""
    return "Temperatures and heat paths" if sim is not None and is_closed(sim) else "Nusselt number and heat rate"


def closed_fan_rows(r, sim, window: str = "") -> list[list]:
    """One volume-flow row per fan of a closed domain (with its statistics when the history has it)."""
    P = _physics()
    if P is None or not has_fan_report(r):
        return []
    unit = _units2d(r)[2]
    fans = list(getattr(sim, "fans", None) or []) if sim is not None else []
    rows = []
    for i, e in enumerate(P.fan_entries(r)):
        st = P.series_stat(r, f"fan{i}_flow")
        name = e["name"] or (getattr(fans[i], "name", "") if i < len(fans) else f"Fan {i + 1}")
        if st is not None and finite(st.mean):
            rows.append(stat_cells(f"Volume flow: {name}", st, unit, window=window,
                                   tip="Through the fan's disc, over the averaging window"))
        elif finite(e.get("volume_flow")):
            rows.append(_plain_row(f"Volume flow: {name}", fmt_sig(e["volume_flow"], 4), unit,
                                   tip="Through the fan's disc at the end of the run"))
    return rows


#: Footnote of the per-part / per-body force tables.
PER_PART_NOTE = ("Per-body values are rounded to the decimals of the total's error: the solver records no "
                 "per-body history, so their own errors are not known.")
PER_PART_TIP = "Means over the averaging window; " + PER_PART_NOTE[0].lower() + PER_PART_NOTE[1:]


def per_part_rows(r, keys: Sequence[str]) -> list[list]:
    """Rows (name, one value per key) of the per-body / per-part coefficient table.

    The solver keeps only each part's mean, not its history, so a part has no
    standard error of its own. Its values are given to the decimals of the
    total's error (a part's error is of the same order or smaller): one
    rounding rule with the Summary instead of a fixed four decimals.
    """
    decs = {}
    for k in keys:
        st = getattr(r, k, None)
        d = decimals_from_error(getattr(st, "stderr", float("nan")))
        decs[k] = d if d is not None else COEF_DECIMALS
    out = []
    for name, d in (getattr(r, "per_body", None) or {}).items():
        row = [name]
        for k in keys:
            v = d.get(k)
            st = getattr(r, k, None)
            se = getattr(st, "stderr", float("nan"))
            # "≈0" alone in the narrow part columns; the tooltip says why
            row.append(Cell("≈0", tone="muted", tooltip=f"{fmt_sig(v, 3)}: smaller than the total's standard "
                                                        f"error {fmt_sig(se, 2)}, so it cannot be told from zero")
                       if finite(v) and within_error(v, se) else fmt_fixed(v, decs[k]))
        out.append(row)
    return out


def heat_body_rows(r, design) -> list[list]:
    """Per-body rows of the heat-transfer tab: body, wall condition, Nu, ± error, heat rate (one rounding
    rule with the Summary: :func:`value_error_texts`)."""
    items = list(getattr(design, "meshes", []) or []) if isinstance(r, CFD3DResult) else \
        list(getattr(design, "bodies", []) or []) if design is not None else []
    cond = {getattr(b, "name", ""): b for b in items}
    unit_q = _units2d(r)[0]
    rows = []
    per_body = getattr(r, "per_body", None) or {}
    # One heat-exchanging body carries the whole heat rate, so its Nu *is* the run's mean Nu: show the
    # Summary's statistic (time mean of Nu(t) with its standard error), not the per-body value built
    # from time-averaged temperatures. Those two differ for a set-flux wall (the mean of 1/ΔT is not
    # 1/mean ΔT): the heated box read "Mean Nu 183 ± 12" above "Box 181.8 —" for the same body.
    exchanging = [n for n, d in per_body.items()
                  if finite(d.get("nu", d.get("nusselt"))) and getattr(cond.get(n), "thermal", "") != "adiabatic"]
    P = _physics()
    single = P.nusselt_stat(r) if P is not None and len(exchanging) == 1 else None
    if single is not None and not finite(getattr(single, "mean", float("nan"))):
        single = None
    for name, d in per_body.items():
        nu = d.get("nu", d.get("nusselt"))
        nu_se = d.get("nu_stderr", float("nan"))
        reliable = bool(d.get("nu_stderr_reliable", True))
        if single is not None and name == exchanging[0]:
            nu, nu_se = float(single.mean), float(getattr(single, "stderr", float("nan")))
            reliable = bool(getattr(single, "stderr_reliable", True))
        q = d.get("heat_rate", d.get("q"))
        if nu is None and q is None:
            continue
        b = cond.get(name)
        mode = str(getattr(b, "thermal", "")) if b is not None else ""
        val = getattr(b, "thermal_value", None) if b is not None else None
        typed = f"{float(val):.6g}".replace("-", MINUS) if finite(val) else "—"     # as the user set it
        wall = {"temperature": f"{typed} °C", "heat_flux": f"{typed} W/m²", "adiabatic": "Adiabatic"}.get(mode, "—")
        nu_t, nu_se_t = value_error_texts(nu, nu_se) if finite(nu) else ("—", "—")
        q_se = d.get("heat_rate_stderr", float("nan"))
        q_t = value_error_texts(q, q_se)[0] if finite(q) and finite(q_se) else fmt_sig(q, 4) if finite(q) else "—"
        rows.append([name, wall, nu_t, error_cell(nu_se_t, stderr=nu_se, reliable=reliable), q_t, unit_q])
    return rows


#: Heat-balance residual, as a share of the heat through the domain, above which the run is flagged:
#: "bad" (danger colour) above 1 %, "warn" above 0.2 %. A steady run's residual is round-off (the
#: de Vahl Davis cavity: 5×10⁻⁸ of its 0.102 W/m); a transient run's is O(dt), well under these.
BALANCE_BAD, BALANCE_WARN = 0.01, 0.002


def _wall_word(sim, side: str) -> str:
    """"Hot wall (left)", "Cold wall (right)", "Heated wall (top)" ... from the side's set condition."""
    th = getattr(sim, "thermal", None)
    walls = {str(getattr(w, "side", "")): w for w in getattr(th, "walls", None) or []}
    temps = [float(getattr(w, "value", float("nan"))) for w in walls.values()
             if getattr(w, "mode", "") == "temperature" and finite(getattr(w, "value", None))]
    w = walls.get(side)
    mode = getattr(w, "mode", "adiabatic") if w is not None else "adiabatic"
    if mode == "temperature" and len(temps) >= 2:
        v = float(w.value)
        if v >= max(temps):
            return f"Hot wall ({side})"
        if v <= min(temps):
            return f"Cold wall ({side})"
    if mode == "heat_flux":
        return f"Heated wall ({side})" if float(getattr(w, "value", 0.0) or 0.0) >= 0 else f"Cooled wall ({side})"
    return f"{side.capitalize()} wall"


def _balance_label(key: str, sim) -> str:
    """A heat-balance term's label for the table: the domain walls named by their set condition
    ("Hot wall (left)" rather than "Left wall (the hottest)"), the rest as the analysis names them."""
    m = re.fullmatch(r"(Left|Right|Bottom|Top|Front|Back) wall(?: \(the hottest\))?", key)
    if m and sim is not None:
        return _wall_word(sim, m.group(1).lower())
    if key == "Other domain walls (net)" and sim is not None:
        # the hottest wall is listed on its own: when one other wall has a set temperature and no wall
        # has a heat flux, the rest of the heat is that wall's (the others are adiabatic)
        th = getattr(sim, "thermal", None)
        walls = [w for w in getattr(th, "walls", None) or [] if getattr(w, "mode", "adiabatic") != "adiabatic"]
        hot = hot_side = None
        P = _physics()
        if P is not None:
            hot_side = P.hot_wall_side(sim)
        others = [w for w in walls if str(getattr(w, "side", "")) != hot_side]
        if len(others) == 1 and getattr(others[0], "mode", "") == "temperature":
            hot = _wall_word(sim, str(others[0].side))
        return hot or "Other domain walls (net)"
    return key


def heat_balance_rows(r, sim=None, design=None) -> tuple[list[list], str, str | None]:
    """(rows, note, tone of the residual) of the energy balance of a thermal run.

    The residual is judged against the heat *through* the domain, not against the
    net of its terms: a wall-heated cavity takes 0.102 W/m in through its hot wall
    and gives it back through its cold one, so its walls net to ~0 and a round-off
    residual of 5×10⁻⁹ W/m read as 100 % of "the input" (shown in the danger
    colour on the flagship validation case). The throughput is the larger of the
    gross heat in and the gross heat out over every term (walls, bodies, heaters,
    jets, vents, inlet, outlet) and the run's measured heat rate. ``tone`` is
    "bad" above :data:`BALANCE_BAD` of it, "warn" above :data:`BALANCE_WARN`,
    None otherwise.

    With ``sim`` the walls are listed one by one (heat in through the hot wall,
    out through the cold one) instead of as one net term; terms that are
    exactly zero (no jets, no inlet in a closed box) are left out and named in
    the note.
    """
    P = _physics()
    hb = P.heat_balance(r, sim, design) if P is not None else None
    if not hb:
        return [], "This run reports no heat-balance terms.", None
    unit = _units2d(r)[0]
    rows, zero = [], []
    gross_in = gross_out = 0.0
    for direction, terms in (("In", hb["in"]), ("Out", hb["out"])):
        for k, v in terms.items():
            if not finite(v):
                continue
            v = float(v)
            if v == 0.0:
                zero.append(_balance_label(k, sim).split(" (")[0].lower())
                continue
            if (v > 0) == (direction == "In"):
                gross_in += abs(v)
            else:
                gross_out += abs(v)
            rows.append([f"{direction}: {_balance_label(k, sim)}", fmt_sig(v, 4), unit])
    st = hb.get("storage")
    if finite(st):
        rows.append(["Stored in the domain", fmt_sig(st, 4), unit])
    q = getattr(r, "heat_rate", float("nan"))
    through = max(gross_in, gross_out, abs(float(q)) if finite(q) else 0.0)
    resid = hb.get("residual", float("nan"))
    rel = abs(float(resid)) / through if finite(resid) and through > 0 else float("nan")
    tone = None
    if finite(rel):
        tone = "bad" if rel > BALANCE_BAD else "warn" if rel > BALANCE_WARN else None
    # a round-off share reads as a fraction ("5.2×10⁻⁸ of"), not as "5.2×10⁻⁶ %"
    part = (f"{fmt_sig(100 * rel, 2)} %" if rel >= 1e-4 else fmt_sig(rel, 2)) if finite(rel) else ""
    share = f"{part} of the {fmt_sig(through, 4)} {unit} through the domain" if part else ""
    rows.append([Cell("Residual (in − out − stored)", bold=True),
                 Cell(fmt_sig(resid, 3), tone=tone, bold=True, tooltip=share), unit])
    note = ("Residual computed by the solver" if hb.get("reported") else
            "Residual computed here from the reported terms")
    if finite(rel):
        verdict = ("round-off" if rel < 1e-6 else "within research tolerance" if tone is None else
                   "above 0.2 %: check the time step and that the run reached a steady state" if tone == "warn"
                   else "above 1 %: the run does not conserve energy to research grade")
        note += f": {share} ({verdict})."
    else:
        note += "."
    if zero:
        note += " Zero in this run, so not listed: " + ", ".join(dict.fromkeys(zero)) + "."
    if hb.get("missing"):
        note += " Not reported: " + ", ".join(hb["missing"]) + "."
    return rows, note, tone


def fan_rows(r, sim) -> list[list]:
    """Rows of the Fans tab: what each actuator disc delivered, next to Froude momentum theory."""
    P = _physics()
    if P is None:
        return []
    fans = list(getattr(sim, "fans", []) or []) if sim is not None else []
    rho = float(sim.conditions.fluid.density) if sim is not None else float("nan")
    U = float(sim.conditions.speed) if sim is not None else float("nan")
    # Froude's actuator disc needs an unbounded stream. A closed room has none: its "speed" is the
    # planner's reference scale (for a fan room the fan's own static induced velocity), and the disc
    # sees the room's recirculation, so the theory's number (0.56 m/s beside 0.22 measured) compared
    # nothing. The column says why it is empty instead.
    closed = is_closed(sim)
    rows = []
    for i, e in enumerate(P.fan_entries(r)):
        fan = fans[i] if i < len(fans) else None
        area = float(getattr(fan, "diameter", float("nan"))) if fan is not None else float("nan")
        if isinstance(r, CFD3DResult) and finite(area):
            area = math.pi * area * area / 4.0
        usable = finite(e["thrust"]) and finite(area) and not closed
        mt = P.momentum_theory(e["thrust"], area, rho, U) if usable else {}
        mode = e.get("mode") or (getattr(fan, "mode", "") if fan is not None else "")
        target = e.get("target")
        imposed = mode == "thrust" and finite(e["thrust"]) and (
            is_steady(e["thrust"], e.get("thrust_stderr", float("nan")))
            or (finite(target) and abs(e["thrust"] - target) <= 1e-6 * max(abs(target), 1e-300)))
        if imposed:
            # imposed, not measured: "1.0000 ± 1.1×10⁻¹⁶" quoted round-off as if it were an uncertainty
            thrust = Cell(f"{fmt_sig(e['thrust'], 4)} (set)", tooltip=SET_TIP)
        else:
            thrust = value_error_cell(e["thrust"], e.get("thrust_stderr", float("nan")),
                                      reliable=e.get("thrust_reliable", True))
        if closed:
            theory = Cell("—", tone="muted", tooltip=CLOSED_FAN_TIP)
        else:
            theory = fmt_sig(mt.get("induced_velocity", float("nan")), 4)
        # the volume flow and the disc velocity carry the averaging error of their recorded histories
        # (``fan<i>_flow``, ``fan<i>_velocity``): the Summary quoted "0.079 ± 0.028" where this tab
        # showed "0.07915" with no error -- one rounding rule for every table
        flow_st = P.series_stat(r, f"fan{i}_flow")
        vel_st = P.series_stat(r, f"fan{i}_velocity")
        flow = (value_error_cell(flow_st.mean, flow_st.stderr, reliable=flow_st.stderr_reliable)
                if flow_st is not None and finite(flow_st.mean) else fmt_sig(e["volume_flow"], 4))
        vel_se = vel_st.stderr if vel_st is not None else float("nan")
        vel_ok = bool(getattr(vel_st, "stderr_reliable", True))
        induced = value_error_cell(e["induced_velocity"], vel_se, reliable=vel_ok)
        # P = T u_disc: with the thrust imposed its error is the disc velocity's times T
        power = (value_error_cell(e["ideal_power"], abs(float(e["thrust"])) * vel_se, reliable=vel_ok)
                 if imposed and finite(vel_se) else fmt_sig(e["ideal_power"], 4))
        rows.append([e["name"], mode, fmt_sig(e.get("target"), 4), thrust, flow, induced, theory, power,
                     fmt_sig(e["pressure_jump"], 4)])
    return rows


def value_error_cell(value, stderr, *, reliable: bool = True) -> Cell | str:
    """"0.079 ± 0.028" in one cell, rounded to its error (:func:`value_error_texts`); the value alone
    when the error is unknown or the series is steady; amber with the rough-estimate tooltip when the
    window held few independent batches."""
    if not finite(value):
        return "—"
    v_t, se_t = value_error_texts(value, stderr)
    if se_t in ("—", STEADY_TEXT):
        return v_t if se_t == "—" else Cell(v_t, tooltip="Steady over the averaging window: its averaging "
                                                         "error is far below the grid's")
    return Cell(f"{v_t} ± {se_t}", tone=None if reliable else "warn", tooltip="" if reliable else ROUGH_TIP)


#: Why the Fans tab's momentum-theory column is empty in a closed domain (tooltip and note).
CLOSED_FAN_TIP = ("Not defined in a closed domain: Froude's actuator-disc theory needs an unbounded stream, and in "
                  "a room the fan drives a recirculation whose speed the walls and friction set")


def fan_note_text(sim, *, three_d: bool = False) -> str:
    """The line under the Fans table: what the momentum-theory column compares with, or why it is empty."""
    if is_closed(sim):
        text = ("Momentum theory: not shown in a closed domain. Froude's actuator disc needs an unbounded stream; "
                "here the fan drives the room's recirculation, whose speed the walls and friction set.")
        if not three_d:
            # a 2D room's speeds have no grid-converged answer (theory_ns2d.md §18.4): the volume flow,
            # disc velocity and ideal power above are a picture of the circulation, not numbers to quote
            text += (" In 2D the volume flow, disc velocity and ideal power are qualitative and grid-dependent "
                     "(no vortex stretching; theory_ns2d.md §18.4); the thrust is exact. Use 3D for a room.")
        return text
    size = "diameter" if three_d else "width"
    return (f"Momentum theory: the induced velocity a Froude actuator disc of the fan's {size} needs for the "
            "delivered thrust at the run's speed and density.")


FAN_HEADERS = ["Fan", "Mode", "Target", "Thrust", "Volume flow", "Induced velocity", "Momentum theory",
               "Ideal power", "Pressure jump"]


def fan_units(r) -> list[str]:
    """Units row under the fan table's headers (per metre of span in 2D)."""
    unit_p, unit_f, unit_q = _units2d(r)
    return ["", "", "", unit_f, unit_q, "m/s", "m/s", unit_p, "Pa"]


def result_diagnostics_rows(r) -> list[list]:
    """(Quantity, value, unit) rows of the solver diagnostics stored with a result."""
    P = _physics()
    if P is None:
        return []
    from ..widgets.diagnostics import describe
    known = getattr(P, "DIAGNOSTIC_LABELS", {})
    rows = []
    for k, lab, v, unit in P.diagnostics_rows(r):
        if k not in known:
            # a key the report's table does not name ("du_l2" read as "du l2"): the live panel's words for it
            # ("L2 change of u", in U), the same the Simulate dock shows while the run goes
            q = describe(k)
            if q.group != "Other":
                lab, unit = q.label, q.unit or unit
        rows.append([lab, fmt_sig(v, 3), unit])
    return rows


def history_diagnostics(r):
    """A :class:`~aerolume.app.widgets.diagnostics.DiagnosticsLog` of the diagnostics a result stores over
    time, or None when it stores none.

    First the run's own diagnostics table (``result.diagnostics_history``: step, t and one column per
    quantity, about 200 rows the job manager decimated from the live log, or the solver's table); else
    the diagnostic series of the force history (keys such as ``div_rms``, ``du_rel``, ``cfl``)."""
    from ..widgets.diagnostics import DiagnosticsLog, describe
    dh = getattr(r, "diagnostics_history", None)
    if isinstance(dh, dict) and dh and len(np.asarray(dh.get("step", dh.get("t", []))).ravel()) >= 2:
        log_ = DiagnosticsLog()
        log_.add_block({str(k): np.asarray(v, float).ravel() for k, v in dh.items()})
        if len(log_):
            log_.source = "Stored with the result"
            return log_
    h = getattr(r, "history", None) or {}
    keys = [k for k in h if k not in ("t", "step", "cd", "cl", "cm", "cs", "c_roll", "cn", "nu", "heat_rate")
            and not k.startswith("fan") and describe(k).group != "Other"]
    if not keys:
        return None
    n = len(np.asarray(h[keys[0]]).ravel())
    table = {k: np.asarray(h[k], float).ravel() for k in keys if len(np.asarray(h[k]).ravel()) == n}
    table["step"] = np.asarray(h.get("step", np.arange(n)), float).ravel()[:n]
    if "t" in h and len(np.asarray(h["t"]).ravel()) == n:
        table["t"] = np.asarray(h["t"], float).ravel()
    log_ = DiagnosticsLog()
    log_.add_block(table)
    log_.source = "Stored with the result"
    return log_


#: SurfaceData attributes a solver may store a wall heat-flux / local Nusselt distribution in.
HEAT_SURFACE_KEYS = (("heat_flux", "Wall heat flux q″", "W/m²"), ("q_wall", "Wall heat flux q″", "W/m²"),
                     ("nu_local", "Local Nusselt number", ""), ("nu", "Local Nusselt number", ""))


def surface_heat(sd) -> tuple[np.ndarray, str, str] | None:
    """(values, label, unit) of a body's wall heat-flux distribution, None when the result has none."""
    for key, lab, unit in HEAT_SURFACE_KEYS:
        v = getattr(sd, key, None)
        if v is not None and len(np.asarray(v)) == len(np.asarray(sd.x)):
            return np.asarray(v, dtype=float), lab, unit
    return None


def temperature_field(fd) -> tuple[str, np.ndarray] | None:
    """("temperature" | "theta", values) of a stored field set, None without one."""
    if fd is None:
        return None
    for name in ("temperature", "theta"):
        if name in fd.fields:
            return name, fd.fields[name]
    return None


def polar_summary_rows(polar: PolarResult) -> list[list]:
    """Rows of the polar Summary table: quantity, value, condition, unit."""
    m = polar_metrics_safe(polar)
    n_ok, n_all = int(m.get("n_converged", 0) or 0), int(m.get("n_points", 0) or 0)
    if m.get("cl_max_reached") == 0.0:
        # the lift is still rising at the last converged angle: that is not Cl,max
        cl_max_row = [Cell("Maximum lift Cl,max", tooltip="The sweep ended before the stall; extend the last "
                           "angle in Setup to find Cl,max"),
                      Cell(f"≥ {fmt_fixed(m.get('cl_max'), 4)}", tone="warn"),
                      f"still rising at α = {fmt_fixed(m.get('alpha_cl_max'), 2)}°", ""]
    else:
        cl_max_row = ["Maximum lift Cl,max", fmt_fixed(m.get("cl_max"), 4),
                      f"α = {fmt_fixed(m.get('alpha_cl_max'), 2)}°", ""]
    # fixed decimals per kind of quantity (lift 4, drag 5, angles 2, L/D 1), as in the polar table
    return [
        cl_max_row,
        ["Minimum drag Cd,min", fmt_fixed(m.get("cd_min"), 5), f"Cl = {fmt_fixed(m.get('cl_at_cd_min'), 4)}", ""],
        ["Best lift-to-drag (L/D)max", fmt_fixed(m.get("ld_max"), 1),
         f"α = {fmt_fixed(m.get('alpha_ld_max'), 2)}°", ""],
        [Cell("Lift-curve slope Cl,α", tooltip="Thin-airfoil theory: 2π = 6.283 per radian"),
         fmt_fixed(m.get("lift_slope_per_rad"), 3), "linear range", "1/rad"],
        ["Zero-lift angle α0", fmt_fixed(m.get("alpha_zero_lift"), 2), "Cl = 0", "°"],
        ["Zero-lift moment Cm0", fmt_fixed(m.get("cm0"), 4), "about c/4", ""],
        ["Reynolds number", fmt_re(polar.re), f"Mach {fmt_num(polar.mach, 2)}", ""],
        ["Points converged", f"{n_ok} of {n_all}", f"Ncrit {float(polar.ncrit):g}",
         ""],
    ]


def polar_metrics_safe(polar: PolarResult) -> dict:
    """``analysis.polar_metrics`` with a minimal local fallback."""
    an = lazy_import("aerolume.analysis")
    if an is not None and hasattr(an, "polar_metrics"):
        try:
            return dict(an.polar_metrics(polar))
        except Exception:
            log.exception("analysis.polar_metrics failed")
    tab = polar.table()
    out: dict = {"n_points": len(polar.points), "n_converged": len(tab["alpha"])}
    if len(tab["alpha"]):
        i = int(np.nanargmax(tab["cl"]))
        # without the analysis module the stall test is unavailable: a peak at the last angle
        # is reported as not reached (a lower bound), never as Cl,max
        out.update(cl_max=tab["cl"][i], alpha_cl_max=tab["alpha"][i],
                   cl_max_reached=0.0 if tab["alpha"][i] >= np.nanmax(tab["alpha"]) else 1.0)
        j = int(np.nanargmin(tab["cd"]))
        out.update(cd_min=tab["cd"][j], cl_at_cd_min=tab["cl"][j])
        best = polar.best_ld()
        if best is not None:
            out.update(ld_max=best.cl / best.cd, alpha_ld_max=best.alpha)
    return out


#: Smoke fields of a 2D result (the composite of all channels and each channel).
SMOKE_NAMES = (SMOKE_COMPOSITE, "dye", "dye1", "dye2", "dye3")


def smoke_channels_of(fd) -> list[str]:
    """Smoke channels a stored field set carries with any smoke in them, in channel order (the solver
    stores an empty "dye" for every run: a "Smoke" entry that shows black says the run failed)."""
    out = []
    for n in ("dye", "dye1", "dye2", "dye3"):
        if fd is None or n not in fd.fields:
            continue
        a = np.asarray(fd.fields[n], dtype=float)
        if a.size and np.nanmax(a) > 1e-6:
            out.append(n)
    return out


def smoke_stretch(values, mask=None):
    """:func:`aerolume.viz.colormaps.dye_stretch`, or None when it cannot be computed."""
    try:
        from ...viz.colormaps import dye_stretch
        return dye_stretch(values, mask)
    except Exception:  # a picture without the stretch is still a picture
        log.debug("smoke stretch failed", exc_info=True)
        return None


def smoke_stretch_note(ranges, *, closed: bool = True) -> str:
    """The Fields note under a contrast-stretched smoke picture: what was stretched and what it means."""
    rs = [r for r in ranges if r]
    if not rs:
        return ""
    where = "the room" if closed else "the domain"
    span = "; ".join(f"{fmt_num(lo, 2)}–{fmt_num(hi, 2)}" for lo, hi in rs)
    return (f"The smoke has filled {where} (more than half the source concentration in over a quarter of it), so "
            f"each channel is contrast-stretched between the 5th and 99.5th percentiles of its own field "
            f"({span} of the source concentration, faint to full colour). The picture shows where the smoke is "
            f"thinner, not its absolute concentration.")


def field_menu(fd, result=None) -> list[tuple[str, str]]:
    """(field key, menu text) of the Results Fields menu for one stored field set.

    The fields of ``fd`` in :data:`FIELD_ORDER`, smoke first among the
    channels as their composite when there are several. Smoke is kept only as
    the last snapshot, so the time-averaged set offers it too, from the
    snapshot ("Smoke, last snapshot"): choosing it switches to Instantaneous.
    """
    names = [n for n in FIELD_ORDER if n in fd.fields] + \
            [n for n in fd.fields if n not in FIELD_ORDER and n not in ("mask", "divergence")]
    chans = smoke_channels_of(fd)
    inst = getattr(result, "instant_fields", None) if result is not None else None
    borrowed = not chans and inst is not None and inst is not fd and bool(smoke_channels_of(inst))
    if borrowed:
        chans = smoke_channels_of(inst)
    out: list[tuple[str, str]] = []
    if len(chans) >= 2:
        out.append((SMOKE_COMPOSITE, "Smoke (all channels, last snapshot)" if borrowed else "Smoke (all channels)"))
    elif chans:
        out.append(("dye", "Smoke (last snapshot)" if borrowed else "Smoke"))
    for n in names:
        if n in ("dye", "dye1", "dye2", "dye3"):
            if n not in chans or len(chans) == 1:
                continue                   # empty, or the one channel already first as "Smoke"
            out.append((n, f"Smoke, channel {chans.index(n) + 1}"))
            continue
        out.append((n, field_info(n)[0]))
    return out


def default_field(result, sim) -> str:
    """The field the Fields tab opens on: smoke when the run released any, else speed."""
    em = [e for e in (getattr(sim, "emitters", None) or []) if getattr(e, "enabled", True)
          and getattr(e, "species", "dye") == "dye"]
    inst = getattr(result, "instant_fields", None)
    chans = smoke_channels_of(inst) or smoke_channels_of(getattr(result, "mean_fields", None))
    if not em or not chans:
        return "speed"
    return SMOKE_COMPOSITE if len(chans) >= 2 else "dye"


def closed_rect(sim, extent=None) -> tuple[float, float, float, float] | None:
    """The 2D framing of a closed domain: its ``domain_box``, else the stored field's extent; None (the
    default near-field framing) for a tunnel."""
    if sim is None or not is_closed(sim):
        return None
    box = getattr(getattr(sim, "cfd", None), "domain_box", None)
    try:
        b = tuple(float(v) for v in box)[:4] if box is not None else ()
    except (TypeError, ValueError):
        b = ()
    if len(b) == 4 and b[1] > b[0] and b[3] > b[2]:
        return b
    return tuple(float(v) for v in extent) if extent is not None else None


def heat_focus_rect(r, design, sim, extent=None) -> tuple[float, float, float, float] | None:
    """The Heat-transfer view's framing of a closed domain: the hottest heater's footprint (else the
    body that gives off the most heat) with a margin of its own size and at least a fifth of the box,
    clipped to the box. None for a tunnel (the bodies' near field, the default)."""
    box = closed_rect(sim, extent)
    if box is None:
        return None
    src, _t = hottest_source(r, sim, design)
    rect = None
    if src is not None:
        pos, size = tuple(src.position), tuple(src.size) or (0.0,)
        if src.kind == "disc":
            hx = hy = float(size[0])
        else:
            hx, hy = 0.5 * float(size[0]), 0.5 * float(size[1] if len(size) > 1 else size[0])
        rect = (pos[0] - hx, pos[0] + hx, pos[1] - hy, pos[1] + hy)
    else:
        pb = getattr(r, "per_body", None) or {}
        hot = max(((float(d.get("heat_rate")), n) for n, d in pb.items() if finite(d.get("heat_rate"))), default=None)
        if hot is not None and hot[0] > 0 and design is not None:
            polys = {getattr(b, "name", ""): p for b, p in zip(design.bodies, body_polygons(r, design, sim))}
            pts = polys.get(hot[1])
            if pts is not None and len(pts):
                rect = (float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max()))
    if rect is None:
        return box
    bw, bh = box[1] - box[0], box[3] - box[2]
    mx = max(rect[1] - rect[0], 0.2 * bw)
    my = max(rect[3] - rect[2], 0.2 * bh)
    out = (max(box[0], rect[0] - mx), min(box[1], rect[1] + mx), max(box[2], rect[2] - my), min(box[3], rect[3] + my))
    return out if out[1] > out[0] and out[3] > out[2] else box


def body_polygons(result, design, sim) -> list[np.ndarray]:
    """Solved (wind-frame) outlines: from the result if stored, else the design rotated by -alpha."""
    bw = getattr(result, "bodies_wind", None) or []
    if bw:
        return [np.asarray(b.points, dtype=float) for b in bw]
    if design is None:
        return []
    alpha = sim.conditions.alpha_deg if sim is not None else 0.0
    return [np.stack(rotate(b.points[:, 0], b.points[:, 1], -alpha, design.moment_ref), axis=1)
            for b in design.bodies]


def focus_rect(polys: list[np.ndarray], *, ground: float | None = None) -> tuple[float, float, float, float] | None:
    """Near-field window around the bodies: the one default framing of every field view.

    Delegates to :func:`near_field_window` (1 length upstream, 3 downstream,
    1 to each side) so Results, Compare and the Simulate viewport frame the
    same flow identically. Double-click / ``F`` in the view returns here.
    """
    return near_field_window(polys, ground=ground)


def ground_of(sim) -> float | None:
    """Road height (0 m, DESIGN.md 3) of a ground run, None without a road."""
    g = getattr(sim, "ground", None)
    return 0.0 if g is not None and getattr(g, "enabled", False) else None


def short_device(device: str) -> str:
    """'GPU · NVIDIA GeForce RTX 4060 Ti (8.0 GB)' -> 'RTX 4060 Ti'; 'cuda' -> 'CUDA'.

    The header line is a list of facts, so the device gets the short name a
    user recognises; the full description stays in the Data tab.
    """
    d = re.sub(r"^(GPU|CPU)\s*[·:-]\s*", "", (device or "").strip())
    d = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", d).strip()
    d = re.sub(r"\s*\([^)]*\)\s*$", "", d)
    return {"cuda": "CUDA", "cpu": "CPU", "mps": "MPS"}.get(d.lower(), d)


def turbulence_text(turb: str) -> str:
    """'les-wale' -> 'LES-WALE', 'laminar' -> 'Laminar'."""
    t = (turb or "").strip()
    return t.upper() if t.lower().startswith("les") else t.capitalize()


def cp_range(vmin: float) -> tuple[float, float]:
    """Symmetric Cp colour range: at least +/-1 (stagnation is exactly Cp = 1), at most +/-3.

    Centred on zero so the neutral middle of the diverging map is free-stream
    pressure (docs/ui_guidelines.md: Cp coolwarm centred on 0); capped because
    a suction peak of -6 at a sharp leading edge would otherwise wash every
    other pressure into the same pale band.
    """
    m = min(max(abs(vmin), CP_LIMITS[0]), CP_LIMITS[1])
    return -m, m


def _ra_fact(design, sim) -> str | None:
    """"Ra 1.0×10⁴" for a natural-convection run (its Reynolds number is a derived scale), else None."""
    if not natural_convection(design, sim):
        return None
    sc = run_scales(design, sim)
    P = lazy_import("aerolume.simulation.physics")
    if sc is None or not finite(sc.ra) or not sc.ra:
        return None
    return f"Ra {P.fmt_number(sc.ra) if P is not None and hasattr(P, 'fmt_number') else fmt_sig(sc.ra, 2)}"


def header_facts(r, sim, design=None) -> tuple[str, str | None, str]:
    """(status word, status kind, facts line) for the Results header."""
    if isinstance(r, CFD3DResult):
        from ..widgets.results3d import header_facts3d
        return header_facts3d(r, sim, design)
    if isinstance(r, WingResult):
        from ..widgets.results3d import wing_header_facts
        return wing_header_facts(r, sim)
    if isinstance(r, CFDResult):
        if sim is not None and sim.status == "cancelled":
            status, kind = "Stopped early", "warn"
        elif r.converged:
            status, kind = "Converged", "good"
        else:
            status, kind = "Not converged", "warn"
        parts = []
        ra = _ra_fact(design, sim)
        if ra is not None:
            parts.append(ra)
        elif _re_matched(r):
            parts.append(f"Re {fmt_re(r.re_real)}")
        else:
            parts.append(f"Re {fmt_re(r.re_sim)} simulated ({fmt_re(r.re_real)} real)")
        nx, ny = r.grid
        if nx and ny:
            cells = f"{nx}×{ny} cells"
            if finite(r.cells_per_ref):
                per = "chord" if design is None or design.kind == "airfoil" else "length"
                cells += f", {r.cells_per_ref:.0f} per {per}"
            parts.append(cells)
        if r.turbulence:
            parts.append(turbulence_text(r.turbulence))
        stored = stored_heat_fact(r)
        if stored is not None:
            # the verdict judges the monitors, so a converged run can still be warming its domain
            parts.append(stored[0])
            if stored[1] and kind == "good":
                kind = "warn"
        if finite(r.wall_seconds):
            parts.append(fmt_duration(r.wall_seconds))
        if r.device:
            parts.append(short_device(r.device))
        return status, kind, " · ".join(parts)
    m = polar_metrics_safe(r)
    n_ok, n_all = int(m.get("n_converged", 0) or 0), int(m.get("n_points", 0) or 0)
    status = f"{n_ok} of {n_all} points converged"
    kind = "good" if n_ok == n_all else "warn"
    parts = [f"Re {fmt_re(r.re)}", f"Ncrit {float(r.ncrit):g}", f"Mach {fmt_num(r.mach, 2)}",
             "Viscous–inviscid (mfoil)"]
    if finite(r.wall_seconds):
        parts.append(fmt_duration(r.wall_seconds))
    return status, kind, " · ".join(parts)


def scalar_rows(result, design, sim) -> list[tuple[str, str, str]]:
    """(quantity, value, unit) rows for the Data tab, in reading order."""
    rows: list[tuple[str, str, str]] = []
    if isinstance(result, CFDResult):
        r = result
        for key, st in (("Cd", r.cd), ("Cl", r.cl), ("Cm", r.cm)):
            rows += [(f"{key} mean", fmt_num(st.mean, 6), ""), (f"{key} standard error", _err(st.stderr), ""),
                     (f"{key} rms fluctuation", fmt_num(st.rms, 4), ""),
                     (f"{key} min / max", f"{fmt_num(st.min, 4)} / {fmt_num(st.max, 4)}", "")]
        if is_lifting(design):
            rows.append(("L/D", fmt_num(r.lift_to_drag, 5), ""))
        rows += [("Drag", fmt_num(r.drag, 6), "N/m"),
                 (lift_label(design), fmt_num(r.lift, 6), "N/m"), ("Pitching moment", fmt_num(r.moment, 6), "N·m/m"),
                 ("Drag area CdA", fmt_num(r.drag_area, 5), "m²/m"), ("Drag power", fmt_num(r.power, 5), "W/m"),
                 ("Strouhal number", fmt_num(r.strouhal, 4) if r.strouhal is not None else "—", ""),
                 ("Reynolds number (real)", fmt_re(r.re_real), ""),
                 ("Reynolds number (simulated)", fmt_re(r.re_sim), ""),
                 ("Grid", f"{r.grid[0]} × {r.grid[1]}", "cells"),
                 ("Cells per reference length", fmt_num(r.cells_per_ref, 4), ""),
                 ("Simulated time", fmt_num(r.t_end, 4), "L/U"), ("Averaging from", fmt_num(r.t_avg_start, 4), "L/U"),
                 ("Statistically converged", "yes" if r.converged else "no", ""),
                 ("Turbulence treatment", turbulence_text(r.turbulence) or "—", ""), ("Device", r.device or "—", ""),
                 ("Precision", r.precision or "—", ""), ("Wall-clock time", fmt_num(r.wall_seconds, 4), "s")]
        for name, d in (r.per_body or {}).items():
            rows.append((f"Body “{name}”: Cd / Cl / Cm",
                         f"{fmt_num(d.get('cd'), 4)} / {fmt_num(d.get('cl'), 4)} / {fmt_num(d.get('cm'), 4)}", ""))
        if has_heat(r):
            P = _physics()
            nu = P.nusselt_stat(r)
            if nu is not None and finite(nu.mean):
                rows += [("Mean Nusselt number", fmt_num(nu.mean, 6), ""),
                         ("Nusselt number standard error", _err(nu.stderr), "")]
            q, se, _ok = P.heat_rate_stat(r)
            rows += [("Heat transfer rate", fmt_num(q, 6), "W/m"), ("Heat rate standard error", _err(se), "W/m")]
        for k, lab, unit in (("div_rms", "RMS divergence at the end", ""),):
            v = (r.diagnostics or {}).get(k)
            if v is not None:
                rows.append((lab, fmt_sig(v, 3), unit))
    elif isinstance(result, PolarResult):
        m = polar_metrics_safe(result)
        rows += [("Reynolds number", fmt_re(result.re), ""), ("Mach number", fmt_num(result.mach, 3), ""),
                 ("Ncrit", fmt_num(result.ncrit, 3), "")]
        units = {"alpha": "°", "per_deg": "1/°", "per_rad": "1/rad"}
        for k, v in m.items():
            unit = next((u for frag, u in units.items() if frag in k), "")
            text = str(int(v)) if k.startswith("n_") and finite(v) else fmt_num(v, 5)
            rows.append((POLAR_METRIC_LABELS.get(k, k.replace("_", " ")), text, unit))
        rows.append(("Wall-clock time", fmt_num(result.wall_seconds, 4), "s"))
    if sim is not None:
        c = sim.conditions
        rows.append(("Speed", fmt_num(c.speed, 5), "m/s"))
        if isinstance(result, PolarResult):
            rows += [(k, v, "") for k, v in polar_angle_items(result, sim) if k != "Ncrit"]
        else:
            rows.append(("Angle", fmt_num(c.alpha_deg, 4), "°"))
        rows += [("Fluid", fluid_name(c.fluid.name), ""), ("Density", fmt_num(c.fluid.density, 5), "kg/m³"),
                 ("Dynamic viscosity", fmt_num(c.fluid.viscosity, 5), "Pa·s")]
    if design is not None:
        rows += [("Design", design.name, ""), ("Reference length", fmt_num(design.length_ref, 5), "m"),
                 ("Reference height / chord (forces)", fmt_num(design.force_ref, 5), "m")]
    return rows


def deg_text(v: float) -> str:
    """An angle as the user typed it in Setup: '−4°', '12°', '0.5°' (no padded decimals)."""
    return f"{float(v):.6g}°".replace("-", MINUS)


def polar_angle_items(result: PolarResult, sim) -> list[tuple[str, str]]:
    """'Angle range' (what was swept, from the result's own points) and 'Ncrit' of a polar."""
    alphas = sorted(float(p.alpha) for p in getattr(result, "points", []) if finite(p.alpha))
    a = getattr(sim, "airfoil", None)
    if alphas:
        lo, hi = alphas[0], alphas[-1]
    elif a is not None:
        lo, hi = a.alpha_start, a.alpha_end
    else:
        return []
    step = getattr(a, "alpha_step", None) if a is not None else None
    if not finite(step) and len(alphas) > 1:
        step = float(np.median(np.diff(alphas)))
    text = f"{deg_text(lo)} to {deg_text(hi)}" + (f", step {deg_text(abs(step))}" if finite(step) and step else "")
    ncrit = getattr(result, "ncrit", None)
    if not finite(ncrit) and a is not None:
        ncrit = a.ncrit
    out = [("Angle range", text)]
    if finite(ncrit):
        out.append(("Ncrit", f"{float(ncrit):g}"))
    return out


def closed_conditions(result, design, sim) -> list[tuple]:
    """Conditions of a closed domain (a heated cavity, a room): what drives its flow.

    Speed, angle of attack and dynamic pressure describe a free stream a room
    does not have. A natural-convection cavity is set by Ra, Pr, Gr and the wall
    temperatures; a fan room by its velocity scale. The groups are recomputed
    with the planner's own :func:`~aerolume.simulation.physics.physics_scales`.
    """
    items: list[tuple] = []
    if design is not None:
        items.append(("Design", design.name))
    sc = run_scales(design, sim)
    P = lazy_import("aerolume.simulation.physics")
    num = getattr(P, "fmt_number", None) if P is not None else None

    def group(v) -> str:
        return num(v) if num is not None else fmt_sig(v, 2)

    c = sim.conditions
    natural = bool(sc is not None and getattr(sc, "natural", False))
    if not natural:
        basis = getattr(sc, "speed_basis", "") if sc is not None else ""
        items.append(("Velocity scale", fmt_speed(c.speed),
                      f"The velocity the run is scaled by: {basis}" if basis else None))
    items += [("Fluid", fluid_name(c.fluid.name)), ("Density", f"{fmt_num(c.fluid.density, 4)} kg/m³")]
    th = getattr(sim, "thermal", None)
    if sc is not None and getattr(sc, "thermal", False):
        # a heater's or a heat flux's ΔT is the planner's estimate (q L / (k Nu)): the fan room was planned
        # at 44.1 K and measured 3.31 K, so the plan's ΔT and Ra are labelled as such and the measured
        # groups (simulation.physics.result_scales) stand beside them. A ΔT set by the walls is exact.
        ms, how = (P.result_scales(sc, result) if P is not None and hasattr(P, "result_scales")
                   else (sc, ""))
        planned = how in ("measured", "planned estimate")
        tag = " (planned estimate)" if planned else ""
        if planned:
            # the run's own record of its plan wins over today's planner: the fan room's note reads 44.1 K
            # where the planner of a later version derives 4.10 K for the same setup. Ra and Gr scale with ΔT.
            dt_run = planned_delta_t(result)
            if finite(dt_run) and finite(sc.delta_t) and sc.delta_t > 0 and abs(dt_run / sc.delta_t - 1) > 0.01:
                import dataclasses
                k = dt_run / sc.delta_t
                with contextlib.suppress(TypeError, ValueError):
                    sc = dataclasses.replace(sc, delta_t=dt_run, ra=sc.ra * k, gr=sc.gr * k)
        tip = ("The planner's estimate of the temperature rise, the scale the run was set up with; the "
               "measured value is below it") if how == "measured" else None

        def planned_item(key: str, text: str) -> tuple:
            return (key, text, tip) if tip else (key, text)

        if finite(sc.delta_t):
            items.append(planned_item(f"ΔT{tag}" if planned else "Temperature difference ΔT", f"{sig(sc.delta_t, 3)} K"))
        if how == "measured" and ms is not None and finite(ms.delta_t):
            items.append(("ΔT measured", f"{sig(ms.delta_t, 3)} K", str(ms.delta_t_basis).split(": ", 1)[-1]))
        if sc.ra and finite(sc.ra):
            items.append(planned_item(f"Ra{tag}" if planned else "Rayleigh number Ra", group(sc.ra)))
        if how == "measured" and ms is not None and ms.ra and finite(ms.ra):
            items.append(("Ra measured", group(ms.ra), "On the measured ΔT"))
        if sc.gr and finite(sc.gr):
            items.append(planned_item(f"Gr{tag}" if planned else "Grashof number Gr", group(sc.gr)))
        if how == "measured" and ms is not None and ms.gr and finite(ms.gr):
            items.append(("Gr measured", group(ms.gr), "On the measured ΔT"))
        if finite(sc.pr):
            items.append(("Prandtl number Pr", fmt_sig(sc.pr, 3)))
        if natural and finite(sc.speed):
            items.append(("Free-fall velocity", fmt_speed(sc.speed)))
    if th is not None and getattr(th, "enabled", False):
        items.append(("Ambient temperature", f"{sig(th.t_ambient, 4)} °C"))
        for w in getattr(th, "walls", []) or []:
            if w.mode == "temperature":
                val = f"{sig(w.value, 4)} °C"
            elif w.mode == "heat_flux":
                val = f"{sig(w.value, 4)} W/m²"
            else:
                continue
            items.append((f"{str(w.side).capitalize()} wall", val))
    if design is not None:
        items.append(("Reference length", f"{fmt_num(design.length_ref, 4)} m"))
    if isinstance(result, CFDResult):
        if finite(result.t_avg_start) and finite(result.t_end):
            items.append(("Averaging window", f"{sig(result.t_avg_start, 3)}–{sig(result.t_end, 3)} L/U"))
        if result.precision:
            items.append(("Precision", result.precision))
    return items


_PLANNED_DT = re.compile(r"Temperature scale ΔT = ([0-9][0-9,]*\.?[0-9]*) K")


def planned_delta_t(result) -> float:
    """The temperature scale ΔT [K] a run was planned with, from its own notes ("Planned estimate:
    Temperature scale ΔT = 44.1 K ..."), or nan when the notes do not state it."""
    for n in getattr(result, "notes", None) or []:
        m = _PLANNED_DT.search(str(n))
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return float("nan")
    return float("nan")


def conditions_items(result, design, sim) -> list[tuple]:
    """Key-value facts shown next to the Summary table (what the numbers are valid for)."""
    if sim is not None and isinstance(result, CFDResult) and is_closed(sim):
        return closed_conditions(result, design, sim)
    items: list[tuple] = []
    if design is not None:
        items.append(("Design", design.name))
    if sim is not None:
        c = sim.conditions
        items.append(("Speed", fmt_speed(c.speed)))
        if isinstance(result, PolarResult):
            # a sweep has no single angle: the simulation's own angle (often 0) would be wrong here
            items += polar_angle_items(result, sim)
        else:
            items.append(("Pitch angle" if design is not None and design.kind == "vehicle" else "Angle of attack",
                          f"{fmt_num(c.alpha_deg, 3)}°"))
        items += [("Fluid", fluid_name(c.fluid.name)), ("Density", f"{fmt_num(c.fluid.density, 4)} kg/m³"),
                  ("Dynamic pressure", f"{fmt_num(c.dynamic_pressure(), 4)} Pa")]
    if design is not None:
        ref = "Chord" if design.kind == "airfoil" else "Reference length"
        items.append((ref, f"{fmt_num(design.length_ref, 4)} m"))
        if design.kind != "airfoil":
            items.append(("Reference height", f"{fmt_num(design.force_ref, 4)} m"))
    if isinstance(result, CFDResult):
        if finite(result.t_avg_start) and finite(result.t_end):
            items.append(("Averaging window", f"{sig(result.t_avg_start, 3)}–{sig(result.t_end, 3)} L/U"))
        if result.precision:
            items.append(("Precision", result.precision))
    return items


def make_table(headers: list[str]) -> DataTable:
    """A read-only results table (see :class:`DataTable`)."""
    return DataTable(headers)


def fill_table(t: DataTable, rows: list[list[str]], *, right: tuple[int, ...] | None = None) -> None:
    """Fill a read-only table; ``right`` = numeric columns (default: all but the first)."""
    headers = [t.horizontalHeaderItem(j).text() if t.horizontalHeaderItem(j) else "" for j in range(t.columnCount())]
    t.set_data(headers, rows, numeric=right)


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

class ResultsPage(QWidget):
    """Result of the current simulation (DESIGN.md 5.8). Actions: "summary", "forces", "fields", ..."""

    #: The Project panel selects the simulation; a selector here would compete with it.
    SHOW_SELECTOR = False

    TABS = TAB_KEYS

    def __init__(self, state, jobs: JobManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        require_api(jobs, ("result", "result_error"), "The Results page")
        self.state = state
        self.jobs = jobs
        self.result = None
        self.design = None
        self._sim_id = ""
        self._stream_call = None
        self._stream_cache: dict[tuple, list] = {}
        self._focus_for = ""
        self._fields_for = ""          # the result whose field menu was opened on its default field
        self._build()
        self._connect()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.empty = EmptyState()
        self.stack.addWidget(self.empty)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.stack.addWidget(content)

        head = QFrame()
        head.setProperty("chrome", "toolbar")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(12, 6, 8, 6)
        hl.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.sim_combo = QComboBox()
        self.sim_combo.setMinimumWidth(240)
        self.sim_combo.setVisible(self.SHOW_SELECTOR)
        row.addWidget(self.sim_combo)
        self.status_text = StatusLabel()
        row.addWidget(self.status_text)
        # one line, cut on the right with an ellipsis (full text in the tooltip) when the window is
        # narrow: with the Assistant dock open at 1280 px it was cut mid-word ("half domain (symmetr")
        self.facts = ElidedLabel("", "secondary", mode=Qt.TextElideMode.ElideRight)
        self.facts.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.facts, 1)
        self.compare_btn = button("Add to comparison", None, icon="compare",
                                  tooltip="Add this simulation to the Compare step")
        self.report_btn = button("Report…", None, icon="report", tooltip="Open the Report step")
        row.addWidget(self.compare_btn)
        row.addWidget(self.report_btn)
        hl.addLayout(row)
        self.warning_line = WarningLine()
        self.warning_line.hide()
        hl.addWidget(self.warning_line)
        lay.addWidget(head)

        # the 2D tabs, or a 3D view (widgets.results3d) built on first use
        self.body_stack = QStackedWidget()
        lay.addWidget(self.body_stack, 1)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # a Tab stop of its own: without a name a screen reader announces "tab bar" and nothing else
        self.tabs.tabBar().setAccessibleName("Result views")
        self.body_stack.addWidget(self.tabs)
        self.view3d = None          # CFD3DResultView
        self.wing_view = None       # WingResultView
        self._build_summary()
        self._build_forces()
        self._build_surface()
        self._build_fields()
        self._build_heat()
        self._build_fans()
        self._build_sensitivity()
        self._build_polar()
        self._build_diagnostics()
        self._build_data()

    @staticmethod
    def _tab_page(margins: int = 8) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(margins, margins, margins, margins)
        v.setSpacing(8)
        return w, v

    def _build_summary(self) -> None:
        # in a scroll area that reflows (results3d.summary_layout): a short or narrow window (or an
        # open dock) scrolls the fixed-height tables or puts Conditions below them, instead of
        # painting the note and the per-body table over the table
        from ..widgets.results3d import summary_layout
        self.summary_table = DataTable(SUMMARY_HEADERS)
        self.summary_table.setAccessibleName("Summary of the result")
        # the table is as tall as its rows; the page below it stays panel-coloured instead of
        # showing an empty white well, which reads as "something failed to load"
        tcol = QVBoxLayout()
        tcol.setSpacing(8)
        tcol.addWidget(self.summary_table)
        # what the coefficients mean: a vehicle's Cl of 2.5 is only readable with its reference
        self.summary_note = label("", "muted", wrap=True)
        tcol.addWidget(self.summary_note)
        self.body_caption = label("Per body", "h3")
        self.body_table = DataTable(["Body", "Cd", "Cl", "Cm"], stretch_last=False)
        self.body_table.setAccessibleName("Coefficients per body")
        tcol.addWidget(self.body_caption)
        tcol.addWidget(self.body_table)
        tcol.addStretch(1)
        side = QWidget()
        side.setFixedWidth(300)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        sl.addWidget(label("Conditions", "h3"))
        self.conditions = KeyValueList()
        sl.addWidget(self.conditions)
        sl.addStretch(1)
        w, self._summary_reflow = summary_layout(self.summary_table, tcol, side)
        self.tabs.addTab(w, "Summary")

    def _build_forces(self) -> None:
        w, v = self._tab_page()
        self.history_plot = HistoryPlot(keys=("cd", "cl", "cm"), show_mean=True)
        self.forces_caption = label("", "muted")
        self.forces_caption.setToolTip("The ± values in Summary are standard errors from batch means "
                                       "over this window. Time is in convective units (reference length / speed).")
        v.addWidget(self.forces_caption)
        v.addWidget(self.history_plot, 1)
        self.tabs.addTab(w, "Forces")

    def _build_surface(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = ToolStrip()
        # the view switch leads the strip as a segmented control (the checked half is filled), like
        # Time-averaged / Instantaneous on the Fields tab; a lone text toggle showed no state
        self.surf_view = SegmentedControl([("plots", "Distributions"), ("arrows", "Arrows on shape")], "plots")
        self.surf_view.setToolTip("Arrows on shape: Cp as arrows normal to the surface, outward where it is "
                                  "sucked (Cp < 0), inward where it is pushed")
        bar.add_widget(self.surf_view)
        self.surf_sep = bar.add_separator()
        self.surf_body_label = bar.add_widget(label("Body", "secondary"))
        self.surf_body = QComboBox()
        bar.add_widget(self.surf_body)
        self.surf_alpha_label = bar.add_widget(label("Angle of attack", "secondary"))
        self.surf_alpha = QComboBox()
        bar.add_widget(self.surf_alpha)
        bar.add_stretch()
        v.addWidget(bar)
        self.surf_stack = QStackedWidget()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        self.cp_plot = CpPlot()
        self.cf_plot = LinePlot(xlabel="x / c", ylabel="|C<sub>f</sub>|")
        split.addWidget(self.cp_plot)
        split.addWidget(self.cf_plot)
        split.setSizes([600, 400])
        self.surf_stack.addWidget(split)
        self.arrow_plot = GeometryPlot()
        self.surf_stack.addWidget(self.arrow_plot)
        v.addWidget(self.surf_stack, 1)
        self.surface_tab = w
        self.tabs.addTab(w, "Surface")

    def _build_fields(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.fld_bar = ViewportToolbar(fields=(), particles=False, streamlines=True)
        self.fld_slot = SegmentedControl([("mean", "Time-averaged"), ("instant", "Instantaneous")], "mean")
        self.fld_bar.row.insertWidget(0, self.fld_slot)
        self.fld_bar.row.insertSpacing(1, 8)
        self.fld_combo = self.fld_bar.field_combo
        self.fld_cmap = self.fld_bar.cmap_combo
        self.fld_stream = self.fld_bar.streamlines_btn
        self.field_view = FieldView()
        self.fld_bar.finish(self.field_view)
        self.fld_replay = self.fld_bar.add_button("play", "Particles", "Replay tracer particles from the emitters "
                                                  "through the stored velocity field (a frozen field: the "
                                                  "particles trace its streamlines)", checkable=True)
        self.fld_replay.setProperty("variant", "toggle")
        self.fld_devices = self.fld_bar.add_button("layers", "Devices", "Draw the emitters, fans, heaters, jets and "
                                                   "vents placed in Setup over the field", checkable=True)
        self.fld_devices.setProperty("variant", "toggle")
        self.fld_devices.setChecked(True)
        self.fld_png = self.fld_bar.add_button("download", "Save image…", "Save the view as a PNG image")
        # a narrow pane (1280 px with the Assistant docked) squeezed "Time-averaged" to "e-avera": the
        # segments keep their text, and the icon buttons give theirs up first
        from ..widgets.view3d_panel import keep_whole
        keep_whole(self.fld_slot)
        # at 1280 px with two docks the strip lacked ~40 px and squeezed the segments to "Time-averag": the
        # Streamlines and Fit toggles give up their text too (their icons and tooltips stay)
        from .. import theme as _theme
        self.fld_stream.setIcon(_theme.icon("wind", "text2", active="accent"))
        self.fld_stream.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        compact = [(self.fld_stream, "Streamlines")]
        if self.fld_bar.fit_btn is not None:
            compact.append((self.fld_bar.fit_btn, "Fit"))
        self.fld_bar.compactable = [*getattr(self.fld_bar, "compactable", []), *compact, (self.fld_replay, "Particles"),
                                    (self.fld_devices, "Devices"), (self.fld_png, "Save image…")]
        v.addWidget(self.fld_bar)
        v.addWidget(self.field_view, 1)
        # what the picture gave up to be readable (a contrast-stretched smoke that filled a room); hidden
        # when there is nothing to say
        self.fld_note = label("", "muted", wrap=True)
        self.fld_note.setContentsMargins(12, 4, 12, 6)
        self.fld_note.hide()
        v.addWidget(self.fld_note)
        self.fields_tab = w
        self.tabs.addTab(w, "Fields")
        self._replay = None                     # EmitterParticles of the running replay
        self._replay_timer = QTimer(self)
        self._replay_timer.setInterval(33)
        self._replay_timer.timeout.connect(self._replay_tick)

    def _build_heat(self) -> None:
        """Heat transfer: Nu and Q with their uncertainty, per body, the heat balance; the temperature field
        and the wall heat-flux distribution beside them."""
        from ..widgets.results3d import scroll_page
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        split.setChildrenCollapsible(False)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)
        self.heat_title = label("Nusselt number and heat rate", "h3")
        ll.addWidget(self.heat_title)
        self.heat_table = DataTable(SUMMARY_HEADERS)
        self.heat_table.setAccessibleName("Nusselt number and heat rate")
        ll.addWidget(self.heat_table)
        self.heat_note = label("", "muted", wrap=True)
        ll.addWidget(self.heat_note)
        self.heat_body_caption = label("Per body", "h3")
        ll.addWidget(self.heat_body_caption)
        self.heat_body_table = DataTable(["Body", "Wall", "Nu", "± Std. error", "Q", "Unit"], stretch_last=False)
        self.heat_body_table.setAccessibleName("Heat transfer per body")
        ll.addWidget(self.heat_body_table)
        ll.addWidget(label("Heat balance", "h3"))
        self.balance_table = DataTable(["Term", "Rate", "Unit"], stretch_last=False)
        self.balance_table.setAccessibleName("Heat balance")
        ll.addWidget(self.balance_table)
        self.balance_note = label("", "muted", wrap=True)
        ll.addWidget(self.balance_note)
        ll.addStretch(1)
        left.setMinimumWidth(420)
        self._heat_left = left
        self._heat_split = split
        split.addWidget(scroll_page(left))
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(1)
        right.setChildrenCollapsible(False)
        self.heat_view = FieldView()
        self.heat_view.setMinimumHeight(220)
        right.addWidget(self.heat_view)
        self.heat_flux_plot = LinePlot(xlabel="x / c", ylabel="q″ (W/m²)")
        right.addWidget(self.heat_flux_plot)
        right.setSizes([520, 260])
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 900])
        lay.addWidget(split)
        self.heat_tab = w
        self.tabs.addTab(w, "Heat transfer")

    def _build_fans(self) -> None:
        w, v = self._tab_page()
        v.addWidget(label("Fans", "h3"))
        self.fan_table = DataTable(FAN_HEADERS, stretch_last=False)
        self.fan_table.setAccessibleName("Fans")
        v.addWidget(self.fan_table)
        self.fan_note = label("Momentum theory: the induced velocity a Froude actuator disc of the fan's width "
                              "needs for the delivered thrust at the run's speed and density.", "muted", wrap=True)
        v.addWidget(self.fan_note)
        v.addStretch(1)
        self.fans_tab = w
        self.tabs.addTab(w, "Fans")

    def _build_diagnostics(self) -> None:
        from ..widgets.diagnostics import DiagnosticsPanel
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(8, 8, 8, 8)
        sl.setSpacing(6)
        sl.addWidget(label("At the end of the run", "h3"))
        self.diag_table = DataTable(["Quantity", "Value", "Unit"], stretch_last=False)
        self.diag_table.setAccessibleName("Diagnostics at the end of the run")
        sl.addWidget(self.diag_table)
        self.diag_note = label("", "muted", wrap=True)
        sl.addWidget(self.diag_note)
        sl.addStretch(1)
        side.setMinimumWidth(320)
        self._diag_side = side
        # twenty-odd end-of-run rows are taller than the tab at 1280 x 800 with a dock open: the column
        # scrolls (its fixed-height table otherwise ran under the note below it)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setWidget(side)
        side_scroll.setMinimumWidth(320)
        self._diag_side_scroll = side_scroll
        split.addWidget(side_scroll)
        self.diag_history = DiagnosticsPanel()
        # a stored result, not a running solver: the live panel's empty text ("appear here while a simulation
        # runs") and its Record switch say nothing true here
        self.diag_history.empty.set_content(
            "No diagnostic history stored",
            "This result keeps its end-of-run values (left). A run started from the app with Setup › Diagnostics "
            "› Record on stores about 200 samples of its step-by-step history with the result.")
        self.diag_history.record_btn.hide()
        split.addWidget(self.diag_history)
        split.setStretchFactor(1, 1)
        lay.addWidget(split)
        self.diag_tab = w
        self.tabs.addTab(w, "Diagnostics")

    def _build_sensitivity(self) -> None:
        w, v = self._tab_page()
        self.sens_stack = QStackedWidget()
        host = QWidget()
        hl = QVBoxLayout(host)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        self.sens_caption = label("", "muted", wrap=True)
        hl.addWidget(self.sens_caption)
        row = QHBoxLayout()
        row.setSpacing(0)
        self.sens_plot = GeometryPlot()
        self.sens_bar = ColorBar()
        row.addWidget(self.sens_plot, 1)
        row.addWidget(self.sens_bar)
        hl.addLayout(row, 1)
        self.sens_panel = host
        self.sens_stack.addWidget(host)
        self.sens_empty = EmptyState()
        self.sens_empty.set_content(
            "No sensitivity map for this run",
            "Turn on Compute sensitivity in Setup and run again.",
            "Open Setup", lambda: self.state.navigate("setup"))
        self.sens_stack.addWidget(self.sens_empty)
        v.addWidget(self.sens_stack, 1)
        self.sens_tab = w
        self.tabs.addTab(w, "Sensitivity")

    def _build_polar(self) -> None:
        w, v = self._tab_page()
        split = QSplitter(Qt.Orientation.Vertical)
        split.setHandleWidth(1)
        self.polar_grid = PolarGrid()
        split.addWidget(self.polar_grid)
        self.polar_table = DataTable([h for h, _ in POLAR_COLUMNS])
        self.polar_table.setAccessibleName("Polar")
        split.addWidget(self.polar_table)
        split.setSizes([560, 220])
        v.addWidget(split, 1)
        self.polar_tab = w
        self.tabs.addTab(w, "Polar")

    def _build_data(self) -> None:
        w, v = self._tab_page()
        row = QHBoxLayout()
        row.setSpacing(12)
        self.data_table = DataTable(["Quantity", "Value", "Unit"])
        self.data_table.setAccessibleName("Result data")
        row.addWidget(self.data_table, 1)
        side = QWidget()
        side.setFixedWidth(260)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(label("Export", "h3"))
        self.export_btns: dict[str, QPushButton] = {}
        for key, text in (("result_csv", "Summary (CSV)…"), ("history_csv", "Force history (CSV)…"),
                          ("surface_csv", "Surface data (CSV)…"), ("polar_csv", "Polar table (CSV)…"),
                          ("fields_vtk", "Flow fields (VTK .vti)…"), ("fields_npz", "Flow fields (NumPy .npz)…"),
                          ("figures", "All figures (PNG)…")):
            b = button(text, None, icon="export")
            b.setStyleSheet("text-align: left; padding-left: 8px;")
            b.clicked.connect(lambda *_, k=key: self.export(k))
            sl.addWidget(b)
            self.export_btns[key] = b
        sl.addSpacing(4)
        self.export_status = label("", "muted", wrap=True)
        sl.addWidget(self.export_status)
        sl.addStretch(1)
        row.addWidget(side)
        v.addLayout(row, 1)
        self.tabs.addTab(w, "Data")

    def _connect(self) -> None:
        st = self.state
        st.projectChanged.connect(self.refresh)
        st.currentSimulationChanged.connect(lambda *_: self.refresh())
        st.simulationsChanged.connect(self._fill_sim_combo)
        st.resultChanged.connect(self._on_result_changed)
        self.sim_combo.activated.connect(self._on_sim_chosen)
        self.compare_btn.clicked.connect(self._add_to_compare)
        self.report_btn.clicked.connect(lambda: self.state.navigate("report"))
        self.surf_body.currentIndexChanged.connect(self._draw_surface)
        self.surf_alpha.currentIndexChanged.connect(self._draw_surface)
        self.surf_view.changed.connect(self._draw_surface)
        self.fld_slot.changed.connect(self._on_field_slot_changed)
        self.fld_combo.currentIndexChanged.connect(self._draw_field)
        self.fld_cmap.currentIndexChanged.connect(self._draw_field)
        self.fld_stream.toggled.connect(self._draw_field)
        self.fld_png.clicked.connect(self._save_field_png)
        self.fld_devices.toggled.connect(self._show_devices)
        self.fld_replay.toggled.connect(self._toggle_replay)

    # ------------------------------------------------------------- page API
    def on_shown(self) -> None:
        self.refresh()

    def handle_action(self, action: str) -> None:
        """Select a tab by key ("summary", "forces", "surface", "fields", "sensitivity", "polar", "data")."""
        current = self.body_stack.currentWidget()
        if current is not self.tabs and current is not None:
            current.select_tab(action)
            return
        if action in TAB_KEYS:
            i = TAB_KEYS.index(action)
            if self.tabs.isTabVisible(i):
                self.tabs.setCurrentIndex(i)

    # ------------------------------------------------------------ refresh
    def _fill_sim_combo(self) -> None:
        project = self.state.project
        self.sim_combo.blockSignals(True)
        self.sim_combo.clear()
        if project is not None:
            for sid, sim in project.simulations.items():
                self.sim_combo.addItem(sim.name, sid)
            self.sim_combo.setCurrentIndex(self.sim_combo.findData(self.state.simulation_id))
        self.sim_combo.setEnabled(self.sim_combo.count() > 0)
        self.sim_combo.blockSignals(False)

    def _on_sim_chosen(self, index: int) -> None:
        sid = self.sim_combo.itemData(index)
        if sid and sid != self.state.simulation_id:
            self.state.select_simulation(sid)

    def _on_result_changed(self, sim_id: str) -> None:
        if sim_id == self.state.simulation_id:
            self._sim_id = ""
            self.refresh()

    def _show_empty(self, title: str, msg: str = "", action: str = "", cb=None, **kw) -> None:
        self.empty.set_content(title, msg, action, cb, **kw)
        self.stack.setCurrentIndex(0)
        self.result = None
        self.compare_btn.setEnabled(False)

    def refresh(self) -> None:
        self._fill_sim_combo()
        project, sim = self.state.project, self.state.simulation
        if project is None:
            return self._show_empty("No project open", "", "Open or create a project",
                                    lambda: self.state.navigate("start"))
        if sim is None:
            return self._show_empty("No simulation selected", "Select a simulation in the Project panel.",
                                    "Open Setup", lambda: self.state.navigate("setup"))
        res = self.jobs.result(sim.id)
        if res is None:
            err = self.jobs.result_error(sim.id)
            if err:
                return self._show_empty("The result cannot be read", err)
            if sim.status == "failed":
                return self._show_empty("The last run failed", sim.error or "Details are in Messages.",
                                        "Run again", lambda: self._run(sim.id),
                                        secondary_text="Open Setup",
                                        secondary_callback=lambda: self.state.navigate("setup"))
            if sim.status in ("running", "queued"):
                return self._show_empty("Running" if sim.status == "running" else "Queued",
                                        "Results appear here when the run ends.",
                                        "Open Simulate", lambda: self.state.navigate("simulate"))
            return self._show_empty(f"“{sim.name}” has not been run yet", "", "Run",
                                    lambda: self._run(sim.id))
        self.stack.setCurrentIndex(1)
        self.compare_btn.setEnabled(True)
        if res is self.result and sim.id == self._sim_id:
            return
        self.result, self._sim_id = res, sim.id
        self._populate()

    def _run(self, sim_id: str) -> None:
        self.state.select_simulation(sim_id)
        self.state.navigate("simulate:run")

    # ------------------------------------------------------------ populate
    def _apply_tab_visibility(self) -> None:
        """Show the tabs this result has and make the tab bar re-lay itself out.

        ``setTabVisible`` on a hidden tab bar can leave its layout stale: after a
        3D result, a polar and then a heated-cylinder result, ``isTabVisible``
        was True for all eight tabs while the bar still held the polar's layout
        (Forces, Fields, Heat transfer ... at zero width), so only Summary,
        Surface and Data were drawn (restyle critic round 2, reproduced by
        ``test_results_tabs_after_refill_while_hidden``). ``updateGeometry`` and
        ``update`` do not re-lay it out; re-setting the icon size marks the
        layout dirty, which the bar's size hint and ``tabRect`` then rebuild. It
        runs here and again when the page is shown (:meth:`showEvent`).
        """
        vis = getattr(self, "_tab_visibility", None)
        if not vis:
            return
        for key, visible in vis.items():
            i = TAB_KEYS.index(key)
            if self.tabs.isTabVisible(i) != visible:
                self.tabs.setTabVisible(i, visible)
        bar = self.tabs.tabBar()
        bar.setIconSize(bar.iconSize())         # marks the tab layout dirty (QTabBar has no public relayout)
        bar.updateGeometry()
        bar.update()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self.body_stack.currentWidget() is self.tabs:
            self._apply_tab_visibility()

    def _populate(self) -> None:
        r, sim = self.result, self.state.simulation
        design = self.state.project.designs.get(sim.design_id) if sim is not None else None
        self.design = design
        if isinstance(r, (CFD3DResult, WingResult)):
            self._fill_header(r, sim)
            self._populate_3d(r, sim, design)
            return
        self.body_stack.setCurrentWidget(self.tabs)
        is_cfd = isinstance(r, CFDResult)
        self._stop_replay()
        heat, fans = is_cfd and has_heat(r), is_cfd and has_fan_report(r)
        diag = is_cfd and bool(result_diagnostics_rows(r) or history_diagnostics(r) is not None
                               or len(self._session_diagnostics(sim)))
        # a closed room or cavity has no free stream: the Summary hides its coefficients
        # (has_forces / is_closed), so Forces and Sensitivity are hidden too (Data keeps the histories)
        closed = is_cfd and is_closed(sim)
        vis = {"forces": is_cfd and not closed, "fields": is_cfd, "sensitivity": is_cfd and not closed,
               "polar": not is_cfd, "heat": heat, "fans": fans, "diagnostics": diag}
        self._tab_visibility = vis
        self._apply_tab_visibility()
        self._fill_header(r, sim)
        self._fill_summary(r, sim, design)
        if is_cfd:
            self._fill_forces(r)
            self._fill_fields_controls(r)
            self._fill_sensitivity(r, sim)
            if heat:
                self._fill_heat(r, sim, design)
            if fans:
                self._fill_fans(r, sim)
            if diag:
                self._fill_diagnostics(r, sim)
        else:
            self._fill_polar(r)
        self._fill_surface_controls(r)
        self.data_table.set_data(["Quantity", "Value", "Unit"],
                                 [list(row) for row in scalar_rows(r, design, sim)], numeric=[1])
        for k, b in self.export_btns.items():
            b.setVisible((k == "polar_csv") != is_cfd or k in ("result_csv", "figures"))
        self.export_status.setText("")
        if not self.tabs.isTabVisible(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(0)

    def _populate_3d(self, r, sim, design) -> None:
        """Show a 3D CFD or wing result in its own view (built the first time one is needed)."""
        from ..widgets import results3d

        def log_fn(text: str, level: str = "info") -> None:
            self.state.log(text, level, "Results")

        if isinstance(r, CFD3DResult):
            if self.view3d is None:
                self.view3d = results3d.CFD3DResultView()
                self.body_stack.addWidget(self.view3d)
            self.body_stack.setCurrentWidget(self.view3d)
            rake_of = getattr(self.jobs, "seed_rake_for", None)
            self.view3d.set_result(r, sim, design, export_dir=self._export_dir(), log=log_fn,
                                   rake=rake_of(sim) if callable(rake_of) and sim is not None else None)
        else:
            if self.wing_view is None:
                self.wing_view = results3d.WingResultView()
                self.body_stack.addWidget(self.wing_view)
            self.body_stack.setCurrentWidget(self.wing_view)
            self.wing_view.set_result(r, sim, design, airfoils=airfoil_designs(self.state.project, design),
                                      export_dir=self._export_dir(), log=log_fn)

    def _fill_header(self, r, sim) -> None:
        status, kind, facts = header_facts(r, sim, self.design)
        tip = ("The run was stopped early; the averages cover a shorter window." if status == "Stopped early"
               else "Statistical convergence of the time-averaged coefficients"
               if isinstance(r, (CFDResult, CFD3DResult))
               else "Angles where the solver did not converge are left out of the plots")
        self.status_text.set_status(status, kind, tip)
        self.facts.setText(facts)
        # one line (the first clause of the first warning), everything else in its tooltip
        self.warning_line.set_warnings(list(getattr(r, "warnings", []) or []))

    def _fill_summary(self, r, sim, design) -> None:
        if isinstance(r, CFDResult):
            rows = summary_rows(r, design, sim)
            self.summary_table.set_data(SUMMARY_HEADERS, rows)
            self.summary_table.fit_height()
            pb = r.per_body or {}
            # per-body forces mean nothing in a closed room either (they are in Forces)
            multi = len(pb) > 1 and not is_closed(sim)
            self.body_table.set_data(["Body", "Cd", "Cl", "Cm"], per_part_rows(r, ("cd", "cl", "cm")))
            from ..widgets.results3d import compact_columns
            compact_columns(self.body_table, (1, 2, 3))
            self.body_table.fit_height()
            self.body_table.setVisible(multi)
            self.body_caption.setVisible(multi)
            self.body_caption.setToolTip(PER_PART_TIP)
            note = summary_note(design, sim, r, rows)
            if multi:
                note += " " + PER_PART_NOTE
            self.summary_note.setText(note)
        else:
            self.summary_table.set_data(POLAR_SUMMARY_HEADERS, polar_summary_rows(r), numeric=[1])
            self.summary_table.fit_height()
            self.body_table.hide()
            self.body_caption.hide()
            self.summary_note.setText("")
        self.summary_note.setVisible(bool(self.summary_note.text()))
        self.conditions.set_items(conditions_items(r, design, sim))
        from ..widgets.results3d import fit_table_width
        fit_table_width(self.summary_table)
        self._summary_reflow.apply()

    # -- heat transfer, fans, diagnostics (DESIGN.md 10)
    def _fill_heat(self, r: CFDResult, sim, design) -> None:
        from ..widgets.results3d import compact_columns, fit_table_width
        window = (f"Mean over t = {sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U"
                  if finite(r.t_avg_start) and finite(r.t_end) else "Time average")
        heat_rows = heat_tab_rows(r, design, sim, window)
        self.heat_title.setText(heat_tab_title(sim))
        self.heat_table.set_data(SUMMARY_HEADERS, heat_rows)
        self.heat_table.fit_height()
        need = fit_table_width(self.heat_table) + 40          # the tables never scroll sideways in their column
        if self._heat_split.sizes()[0] < need:
            total = max(sum(self._heat_split.sizes()), need + 400)
            self._heat_split.setSizes([need, total - need])
        self._heat_left.setMinimumWidth(min(need, 640))
        th = getattr(sim, "thermal", None)
        parts = [("Nu = Q L / (k ΔT A): the heat rate from the wall penalisation, the reference length L, the "
                  "fluid's conductivity k and the reference temperature difference ΔT."),
                 "Standard errors from batch means over the averaging window.", rounding_note(heat_rows)]
        if not any("Nusselt" in _cell_text(row[0]) for row in heat_rows):
            parts = parts[1:]              # a room without a Nusselt row: the definition explains nothing shown
        if th is not None and getattr(th, "enabled", False):
            amb = f"{float(th.t_ambient):.6g}".replace("-", MINUS)
            parts.insert(0, f"Ambient {amb} °C"
                         + ("; buoyancy on (Boussinesq)." if getattr(th, "buoyancy", False) else "; no buoyancy."))
        self.heat_note.setText(" ".join(parts))
        rows = heat_body_rows(r, design)
        self.heat_body_table.set_data(["Body", "Wall", "Nu", "± Std. error", "Q", "Unit"], rows, numeric=[2, 3, 4])
        compact_columns(self.heat_body_table, (2, 3, 4))
        self.heat_body_table.fit_height()
        for wdg in (self.heat_body_table, self.heat_body_caption):
            wdg.setVisible(bool(rows))
        brows, bnote, _tone = heat_balance_rows(r, sim, design)
        self.balance_table.set_data(["Term", "Rate", "Unit"], brows, numeric=[1])
        compact_columns(self.balance_table, (1,))
        self.balance_table.fit_height()
        self.balance_table.setVisible(bool(brows))
        self.balance_note.setText(bnote)
        self._draw_heat_field(r)
        self._draw_heat_flux(r, design)

    def _draw_heat_field(self, r: CFDResult) -> None:
        fd = r.mean_fields if temperature_field(r.mean_fields) else r.instant_fields
        tf = temperature_field(fd)
        polys = body_polygons(r, self.design, self.state.simulation)
        if tf is None:
            self.heat_view.clear()
            self.heat_view.set_bodies(polys)
            self.heat_view.set_placeholder("No temperature field stored", "")
            return
        name, vals = tf
        vmin, vmax = display_range(name, vals, extent=fd.extent, mask=fd.mask)
        self.heat_view.set_field(vals, fd.extent, field=name, cmap="thermal" if "thermal" in colormap_names()
                                 else None, vmin=vmin, vmax=vmax, mask=fd.mask)
        self.heat_view.set_bodies(polys)
        self.heat_view.set_ground(ground_of(self.state.simulation))
        self.heat_view.set_placements(self._placements() if self.fld_devices.isChecked() else [])
        # a room: the hottest heater (or heated body) with its surroundings, not the box's corner
        focus = heat_focus_rect(r, self.design, self.state.simulation, fd.extent)
        self.heat_view.set_focus_rect(focus, contain=focus is not None)
        self.heat_view.set_caption("Time-averaged" if fd.label == "mean" else "Final snapshot")

    def _draw_heat_flux(self, r: CFDResult, design) -> None:
        """The wall heat flux (or local Nusselt number) along each body, upper and lower branch against
        x / L from the leading edge, in the body frame -- the Surface tab's convention for Cp. Plotted in
        outline order against x, a closed body drew one loop folded back over itself."""
        p = self.heat_flux_plot
        p.clear_series()
        found = False
        length = design.length_ref if design is not None else 1.0
        sim = self.state.simulation
        alpha = sim.conditions.alpha_deg if sim is not None else 0.0
        about = design.moment_ref if design is not None else (0.0, 0.0)
        lab, unit = "Wall heat flux q″", "W/m²"
        several = sum(surface_heat(sd) is not None for sd in r.surface) > 1
        for i, sd in enumerate(r.surface):
            sh = surface_heat(sd)
            if sh is None:
                continue
            vals, lab, unit = sh
            found = True
            curves = heat_curves(sd, vals, alpha_deg=alpha, about=about, length=length)
            pre = f"{sd.body or f'Body {i + 1}'} " if several else ""
            for k, branch in enumerate(("upper", "lower")):
                x, y = curves[branch]
                p.set_series(f"{pre}{branch}", x, y, color=2 * i + k)
        p.set_labels(xlabel="x / c" if design is None or design.kind == "airfoil" else "x / L",
                     ylabel=f"{lab}" + (f" ({unit})" if unit else ""))
        p.setVisible(found)
        if found:
            p.autoRange()

    def _fill_fans(self, r: CFDResult, sim) -> None:
        units = fan_units(r)
        headers = [f"{h} ({u})" if u else h for h, u in zip(FAN_HEADERS, units)]
        self.fan_table.set_data(headers, fan_rows(r, sim), numeric=[2, 3, 4, 5, 6, 7, 8])
        self.fan_table.fit_height()
        self.fan_note.setText(fan_note_text(sim))
        from ..widgets.results3d import fit_table_width
        fit_table_width(self.fan_table)

    def _session_diagnostics(self, sim):
        """The diagnostics this session recorded for the simulation's last run (empty log if none)."""
        fn = getattr(self.jobs, "diagnostics_for", None)
        if sim is None or not callable(fn):
            from ..widgets.diagnostics import DiagnosticsLog
            return DiagnosticsLog()
        return fn(sim.id)

    def _fill_diagnostics(self, r: CFDResult, sim) -> None:
        rows = result_diagnostics_rows(r)
        self.diag_table.set_data(["Quantity", "Value", "Unit"], rows, numeric=[1])
        self.diag_table.fit_height()
        self.diag_table.setVisible(bool(rows))
        from ..widgets.results3d import fit_table_width
        need = max(320, fit_table_width(self.diag_table) + 20)
        self._diag_side.setMinimumWidth(need)
        self._diag_side_scroll.setMinimumWidth(need + 14)            # room for the vertical scroll bar
        log_ = history_diagnostics(r)
        if log_ is None:
            log_ = self._session_diagnostics(sim)
            note = ("The history plots are what this session recorded during the run (not saved with the result)."
                    if len(log_) else "No diagnostic history was recorded for this result.")
        else:
            note = "The history plots are stored with the result."
        self.diag_note.setText(("" if rows else "This result stores no end-of-run diagnostics. ") + note)
        self.diag_history.set_thermal(run_is_thermal(r, sim))     # no θ / heat columns without heat transfer
        self.diag_history.set_log(log_)
        self.diag_history.set_export_dir(self._export_dir())

    # -- particle replay in the stored velocity (frozen field)
    def _replay_sources(self, fd, sim) -> list:
        """Particle sources of the replay: the run's particle emitters, else its smoke emitters (released
        as particles), else a rake across the inlet."""
        from ...model import Emitter
        em = [e for e in (getattr(sim, "emitters", None) or []) if getattr(e, "enabled", True)]
        parts = [e for e in em if e.species == "particles"]
        x0, x1 = fd.extent[0], fd.extent[1]
        L = max(x1 - x0, 1e-9)
        u = np.asarray(fd.fields["u"], float)
        U = float(np.nanmedian(np.abs(u))) or 1.0
        rate = 3000.0 / (L / U)                    # about 3000 particles in flight once the domain is filled
        if not parts:
            parts = [Emitter(name=e.name, species="particles", kind=e.kind, position=e.position, end=e.end,
                             size=e.size, normal=e.normal, count=e.count, radius=e.radius,
                             rate=rate / max(len(em), 1), color=e.color, frame=e.frame) for e in em]
        if not parts:
            win = body_window(body_polygons(self.result, self.design, sim), fd.extent) or fd.extent
            parts = [Emitter(name="Inlet rake", species="particles", kind="line",
                             position=(x0 + 0.02 * L, win[2]), end=(x0 + 0.02 * L, win[3]), rate=rate)]
        return parts

    def _toggle_replay(self, on: bool) -> None:
        if not on:
            self._stop_replay()
            self._draw_field()
            return
        fd = self._field_slot()
        if fd is None or "u" not in fd.fields or "v" not in fd.fields:
            self.fld_replay.setChecked(False)
            return
        self.field_view.set_caption("Loading the particle tracer…")

        def build():
            from ...viz.particles import EmitterParticles
            return EmitterParticles

        run_in_background(build, on_done=lambda cls: self._start_replay(cls, fd),
                          on_error=lambda m: (self.fld_replay.setChecked(False),
                                              self.state.log(f"Particle replay unavailable: {m}", "error", "Results")))

    def _start_replay(self, cls, fd) -> None:
        if not self.fld_replay.isChecked() or fd is not self._field_slot():
            return
        sim = self.state.simulation
        emitters = self._replay_sources(fd, sim)
        x0, x1 = fd.extent[0], fd.extent[1]
        nx = np.shape(fd.fields["u"])[1]
        h = (x1 - x0) / nx
        U = float(np.nanpercentile(np.hypot(fd.fields["u"], fd.fields["v"]), 99)) or 1.0
        # playback: a free-stream particle crosses the domain in about 8 s of screen time
        self._replay_dt = (x1 - x0) / U / (8.0 * 30.0)
        self._replay_speed = U
        self._replay = cls(fd.extent, emitters, n_max=8000, lifetime=1.5 * (x1 - x0) / U, trail=6, cell=h)
        self._replay_fd = fd
        from ...viz.colormaps import hex_rgb
        self._replay_palette = np.array([hex_rgb(getattr(e, "color", "#f5f5f5") or "#f5f5f5") for e in emitters],
                                        dtype=np.uint8)
        what = "time-averaged flow" if fd.label == "mean" else "final snapshot"
        self.field_view.set_caption(f"Particles in the {what} (a frozen field: they trace its streamlines)")
        self._replay_timer.start()

    def _replay_tick(self) -> None:
        p, fd = self._replay, getattr(self, "_replay_fd", None)
        if p is None or fd is None or not self.field_view.isVisible():
            return
        n_sub = p.suggest_substeps(self._replay_dt, self._replay_speed, p.cell)
        p.step(fd.fields["u"], fd.fields["v"], dt=self._replay_dt, mask=fd.mask, substeps=n_sub)
        ids = p.emitter_ids()
        cols = self._replay_palette[np.clip(ids, 0, len(self._replay_palette) - 1)] if len(ids) else None
        self.field_view.set_particles(p.positions(), colors=cols, trails=p.trails())

    def _stop_replay(self) -> None:
        self._replay_timer.stop()
        self._replay = None
        if getattr(self, "fld_replay", None) is not None and self.fld_replay.isChecked():
            self.fld_replay.blockSignals(True)
            self.fld_replay.setChecked(False)
            self.fld_replay.blockSignals(False)
        if getattr(self, "field_view", None) is not None:
            self.field_view.set_particles(None)

    def _fill_forces(self, r: CFDResult) -> None:
        h = r.history or {}
        self.history_plot.set_history(h, (r.t_avg_start, r.t_end) if finite(r.t_avg_start) else None)
        self.history_plot.autoRange()
        self.forces_caption.setText(
            (f"Averaging window t = {sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U (grey band); "
             "dashed lines are the means. " if finite(r.t_avg_start) else "No averaging window recorded. ")
            + "Vertical ranges leave out the start-up transient.")

    # -- surface
    def _fill_surface_controls(self, r) -> None:
        for w in (self.surf_body, self.surf_alpha):
            w.blockSignals(True)
            w.clear()
        if isinstance(r, CFDResult):
            for i, sd in enumerate(r.surface):
                self.surf_body.addItem(sd.body or f"Body {i + 1}", i)
            if len(r.surface) > 1:
                self.surf_body.insertItem(0, "All bodies", -1)
            self.surf_body.setCurrentIndex(0)
        else:
            best = r.best_ld()
            for i, p in enumerate(r.points):
                if p.converged and p.cp is not None:
                    self.surf_alpha.addItem(f"{p.alpha:g}°", i)
            j = next((k for k in range(self.surf_alpha.count())
                      if best is not None and r.points[self.surf_alpha.itemData(k)] is best), 0)
            self.surf_alpha.setCurrentIndex(j)
        is_cfd = isinstance(r, CFDResult)
        self.surf_body.setVisible(is_cfd and self.surf_body.count() > 1)
        self.surf_body_label.setVisible(self.surf_body.isVisibleTo(self))
        self.surf_alpha.setVisible(not is_cfd)
        self.surf_alpha_label.setVisible(not is_cfd)
        self.surf_view.setVisible(is_cfd)
        if not is_cfd:
            self.surf_view.set_current("plots")
        # a separator only separates something: the view switch from a selector after it
        self.surf_sep.setVisible(is_cfd and self.surf_body.isVisibleTo(self))
        xlab = "x / c" if self.design is None or self.design.kind == "airfoil" else "x / L"
        self.cp_plot.set_labels(xlabel=xlab)
        self.cf_plot.set_labels(xlabel=xlab)
        for w in (self.surf_body, self.surf_alpha):
            w.blockSignals(False)
        has = (is_cfd and bool(r.surface)) or (not is_cfd and self.surf_alpha.count() > 0)
        self.tabs.setTabVisible(TAB_KEYS.index("surface"), has)
        self._draw_surface()

    def _draw_surface(self, *_) -> None:
        r, sim, design = self.result, self.state.simulation, self.design
        self.cp_plot.clear_series()
        self.cf_plot.clear_series()
        if r is None:
            return
        tk = tokens()
        cf_series: list[tuple[str, np.ndarray, np.ndarray, int]] = []
        if isinstance(r, CFDResult):
            sel = self.surf_body.currentData()
            surfaces = list(enumerate(r.surface)) if sel in (None, -1) else [(sel, r.surface[sel])]
            alpha = sim.conditions.alpha_deg if sim else 0.0
            about = design.moment_ref if design else (0.0, 0.0)
            length = design.length_ref if design else 1.0
            for i, sd in surfaces:
                cur = surface_curves(sd, alpha_deg=alpha, about=about, length=length)
                name = sd.body or f"Body {i + 1}"
                pre = f"{name} " if len(surfaces) > 1 else ""
                self.cp_plot.set_series(f"{pre}upper", cur["upper"][0], cur["upper"][1], color=2 * i)
                self.cp_plot.set_series(f"{pre}lower", cur["lower"][0], cur["lower"][1], color=2 * i + 1)
                cf_series += [(f"{pre}upper", cur["upper"][0], cur["upper"][2], 2 * i),
                              (f"{pre}lower", cur["lower"][0], cur["lower"][2], 2 * i + 1)]
            if self.surf_view.current() == "arrows":
                self.surf_stack.setCurrentIndex(1)
                self._draw_arrows([sd for _, sd in surfaces], tk)
            else:
                self.surf_stack.setCurrentIndex(0)
        else:
            self.surf_stack.setCurrentIndex(0)
            i = self.surf_alpha.currentData()
            pt = r.points[i] if i is not None and 0 <= i < len(r.points) else None
            cur = point_curves(pt)
            if cur is None:
                return
            for k, (name, col) in enumerate((("upper", 0), ("lower", 1))):
                x, cp, cf, cpi = cur[name]
                self.cp_plot.set_series(name, x, cp, color=col)
                if cpi is not None:
                    self.cp_plot.set_series(f"{name} (inviscid)" if k == 0 else f"_{name}_inv", x, cpi,
                                            color=qcolor(tk["muted"]), width=1.0,
                                            style=Qt.PenStyle.DashLine, legend=k == 0)
                if cf is not None:
                    cf_series.append((name, x, cf, col))
        # |Cf| is 1e-5..1e-2: plot it scaled, with the power of ten in the axis label, so the
        # ticks read 2, 4, 6 instead of pyqtgraph's "2e-05", "4e-05"
        factor, suffix = decade_scale(np.concatenate([s[2] for s in cf_series]) if cf_series else [])
        self.cf_plot.set_labels(ylabel=f"|C<sub>f</sub>|{suffix}")
        for name, x, cf, col in cf_series:
            self.cf_plot.set_series(name, x, np.asarray(cf, dtype=float) * factor, color=col)
        for p in (self.cp_plot, self.cf_plot):
            p.autoRange()

    def _draw_arrows(self, surfaces, tk) -> None:
        self.arrow_plot.clear_all()
        if not surfaces:
            return
        cmax = max((float(np.nanmax(np.abs(sd.cp))) for sd in surfaces if len(sd.cp)), default=1.0) or 1.0
        span = max(float(np.ptp(np.concatenate([sd.x for sd in surfaces]))), 1e-9)
        k = 0.22 * span / cmax
        series = tk["series"]
        for sd in surfaces:
            self.arrow_plot.add_outline(np.stack([sd.x, sd.y], axis=1), tk["text"], width=1.0, fill=True)
            n = len(sd.x)
            step = max(1, n // 140)
            idx = np.arange(0, n, step)
            cp = np.asarray(sd.cp)[idx]
            dx, dy = -cp * np.asarray(sd.nx)[idx] * k, -cp * np.asarray(sd.ny)[idx] * k
            suction = cp < 0
            for mask, col, outward in ((suction, series[0], True), (~suction, series[1], False)):
                if mask.any():
                    # pressure arrows point at the wall: start them outside so the head touches it
                    x0 = np.asarray(sd.x)[idx][mask] - (0 if outward else dx[mask])
                    y0 = np.asarray(sd.y)[idx][mask] - (0 if outward else dy[mask])
                    self.arrow_plot.add_arrows(x0, y0, dx[mask], dy[mask], col)
        self.arrow_plot.autoRange()

    # -- fields
    def _field_slot(self):
        r = self.result
        if not isinstance(r, CFDResult):
            return None
        fd = r.mean_fields if self.fld_slot.current() == "mean" else r.instant_fields
        return fd or r.mean_fields or r.instant_fields

    def _fill_fields_controls(self, r: CFDResult) -> None:
        btns = self.fld_slot._buttons
        btns["mean"].setEnabled(r.mean_fields is not None)
        btns["instant"].setEnabled(r.instant_fields is not None)
        self.fld_slot.blockSignals(True)
        if r.mean_fields is None and r.instant_fields is not None:
            self.fld_slot.set_current("instant")
        elif r.mean_fields is not None and self.fld_slot.current() == "instant" and r.instant_fields is None:
            self.fld_slot.set_current("mean")
        self.fld_slot.blockSignals(False)
        has = r.mean_fields is not None or r.instant_fields is not None
        self.tabs.setTabVisible(TAB_KEYS.index("fields"), has)
        self._on_field_slot_changed()

    def _on_field_slot_changed(self, *_) -> None:
        self._stop_replay()
        fd = self._field_slot()
        current = self.fld_combo.currentData()
        r = self.result
        if self._fields_for != self._sim_id:
            # a new result: open on its smoke when the run released any (the user placed emitters to see it);
            # smoke is stored as the last snapshot, so that is the slot it opens in
            self._fields_for = self._sim_id
            current = default_field(r, self.state.simulation)
            if current in SMOKE_NAMES and r is not None and r.instant_fields is not None \
                    and fd is not r.instant_fields:
                self.fld_slot.blockSignals(True)
                self.fld_slot.set_current("instant")
                self.fld_slot.blockSignals(False)
                fd = self._field_slot()
            elif current not in SMOKE_NAMES and r is not None and r.mean_fields is not None \
                    and fd is not r.mean_fields:
                # the previous result's smoke left the page on the snapshot: a new result opens on its averages
                self.fld_slot.blockSignals(True)
                self.fld_slot.set_current("mean")
                self.fld_slot.blockSignals(False)
                fd = self._field_slot()
        elif fd is not None and current in SMOKE_NAMES and not smoke_channels_of(fd):
            # the user chose Time-averaged while looking at smoke: the averaged flow, not a jump back
            current = "speed"
        self.fld_combo.blockSignals(True)
        self.fld_combo.clear()
        if fd is not None:
            for n, text in field_menu(fd, self.result):
                self.fld_combo.addItem(text, n)
            i = self.fld_combo.findData(current if current else "speed")
            if i < 0:
                i = self.fld_combo.findData("speed")
            self.fld_combo.setCurrentIndex(max(i, 0))
            has_uv = "u" in fd.fields and "v" in fd.fields
            self.fld_stream.setEnabled(has_uv)
            self.fld_stream.setToolTip("Evenly spaced streamlines" if has_uv
                                       else "This result has no velocity components stored.")
        self.fld_combo.blockSignals(False)
        self._draw_field()

    def _placements(self) -> list:
        """The run's emitters, fans, heaters and ports as glyphs (wind frame), [] when there are none."""
        sim = self.state.simulation
        if sim is None:
            return []
        try:
            from ..widgets.draw_tools import placements_of
            return placements_of(sim, self.design, sim.conditions.alpha_deg)
        except Exception:
            log.debug("no placements for the Results view", exc_info=True)
            return []

    def _show_devices(self, on: bool) -> None:
        pls = self._placements() if on else []
        self.field_view.set_placements(pls)
        self.heat_view.set_placements(pls)

    def _frame_once(self) -> None:
        """Frame the result once (then the user's zoom stays): a closed room's whole box, else the bodies'
        near field. A room framed on the circuit board cut off its walls, fan, intake and exhaust."""
        if self._focus_for == self._sim_id:
            return
        self._focus_for = self._sim_id
        rect = closed_rect(self.state.simulation, self.field_view.extent())
        self.field_view.set_focus_rect(rect, contain=rect is not None)

    def _set_field_note(self, text: str) -> None:
        self.fld_note.setText(text)
        self.fld_note.setVisible(bool(text))

    def _draw_smoke_composite(self, fd) -> None:
        """Every smoke channel in its emitters' colour on the dark smoke background, with a channel legend."""
        from ...viz import colormaps as CM
        sim = self.state.simulation
        chans = CM.dye_channels(getattr(sim, "emitters", None) or [])
        colour = {c["field"]: c["color"] for c in chans}
        from ..widgets.view3d_panel import smoke_channel_labels
        labels = smoke_channel_labels(sim) or {c["field"]: c["names"] for c in chans}
        names = {f: (", ".join(v[:2]) + (" …" if len(v) > 2 else "")) for f, v in labels.items()}
        fields = [n for n in CM.DYE_FIELDS if n in fd.fields]
        cols = [colour.get(n, CM.DYE_COLORS[i]) for i, n in enumerate(fields)]
        # a closed room keeps its smoke: once it has filled the room, each channel is contrast-stretched
        # over its own percentiles, or the whole room is one saturated wash with no flow in it
        ranges = [CM.dye_stretch(fd.fields[n], fd.mask) for n in fields]
        rgba = CM.composite_dye([fd.fields[n] for n in fields], cols, ranges=ranges)
        if fd.mask is not None:
            solid = np.asarray(fd.mask)[::-1] >= 0.5
            rgba[solid] = (*SOLID_FILL, 255)
        self.field_view.set_image(rgba, fd.extent, lut=colormap_lut("grayscale"), vmin=0.0, vmax=1.0,
                                  label="Smoke", units="")
        self.field_view.set_legend([(c, (names.get(n) or f"Channel {i + 1}")
                                     + (f" · {fmt_num(rg[0], 2)}–{fmt_num(rg[1], 2)}" if rg else ""))
                                    for i, (n, c, rg) in enumerate(zip(fields, cols, ranges))])
        polys = body_polygons(self.result, self.design, sim)
        self.field_view.set_bodies(polys)
        self.field_view.set_ground(ground_of(sim))
        self.field_view.set_placements(self._placements() if self.fld_devices.isChecked() else [])
        self._frame_once()
        stretched = [rg for rg in ranges if rg]
        self.field_view.set_caption("Smoke at the end of the run, one colour per channel"
                                    + (" · contrast-stretched" if stretched else ""))
        self._set_field_note(smoke_stretch_note(ranges, closed=is_closed(sim)) if stretched else "")

    def _draw_field(self, *_) -> None:
        fd = self._field_slot()
        name = self.fld_combo.currentData()
        self.field_view.set_legend(None)
        # a composite of smoke channels is coloured by its emitters: the colour-map choice does not apply
        self.fld_cmap.setEnabled(name != SMOKE_COMPOSITE)
        self.fld_cmap.setToolTip("Each smoke channel is drawn in its emitters' colour" if name == SMOKE_COMPOSITE
                                 else "Colour map; (default) is the recommended map for the field")
        if fd is not None and name in SMOKE_NAMES and name not in fd.fields and not (
                name == SMOKE_COMPOSITE and any(c in fd.fields for c in SMOKE_NAMES)):
            # smoke is kept as the last snapshot (a time-averaged smoke cloud is a blur): show that one
            r = self.result
            if r is not None and r.instant_fields is not None and fd is not r.instant_fields:
                self.fld_slot.set_current("instant")          # rebuilds the menu and draws again
                return
        if fd is not None and name == SMOKE_COMPOSITE:
            self._draw_smoke_composite(fd)
            return
        if fd is None or name is None or name not in fd.fields:
            self._set_field_note("")
            self.field_view.clear()
            self.field_view.set_placeholder("No flow field stored", "")
            return
        vals = fd.fields[name]
        mask = fd.mask
        polys = body_polygons(self.result, self.design, self.state.simulation)
        # the range is sampled around the body (bbox +- 1 reference length): over the whole
        # domain the free stream dominates and the boundary layer and wake saturate
        # (a closed room: over the whole box -- its fan jet and heater plume are the point, not the board)
        win = None if is_closed(self.state.simulation) else body_window(polys, fd.extent)
        vmin, vmax = display_range(name, vals, extent=fd.extent, mask=mask, window=win)
        rng = smoke_stretch(vals, mask) if name in SMOKE_NAMES else None
        if rng is not None:
            vmin, vmax = rng                 # the colour bar's ticks carry the stretched range
        self._set_field_note(smoke_stretch_note([rng], closed=is_closed(self.state.simulation)) if rng else "")
        self.fld_bar.fill_cmaps(name)
        self.field_view.set_field(vals, fd.extent, field=name, cmap=self.fld_cmap.currentData(),
                                  vmin=vmin, vmax=vmax, mask=mask)
        self.field_view.set_bodies(polys)
        self.field_view.set_ground(ground_of(self.state.simulation))
        self.field_view.set_placements(self._placements() if self.fld_devices.isChecked() else [])
        self._frame_once()           # frame once per result, then respect the user's view
        self.field_view.set_caption("Time-averaged" if fd.label == "mean" else "Instantaneous")
        self.field_view.set_streamlines(None)
        if self.fld_stream.isChecked() and "u" in fd.fields and "v" in fd.fields:
            self._request_streamlines(fd)

    def _request_streamlines(self, fd) -> None:
        key = (self._sim_id, fd.label)
        if key in self._stream_cache:
            self.field_view.set_streamlines(self._stream_cache[key])
            return
        sl = lazy_import("aerolume.viz.streamlines")
        if sl is None:
            self.state.log("Streamlines are unavailable: " + import_error("aerolume.viz.streamlines"),
                           "warning", "Results")
            return
        if self._stream_call is not None:
            self._stream_call.cancelled = True

        def done(lines) -> None:
            self._stream_cache[key] = lines
            if (self._sim_id, getattr(self._field_slot(), "label", None)) == key and self.fld_stream.isChecked():
                self.field_view.set_streamlines(lines)

        self.field_view.set_caption("Computing streamlines…")
        self._stream_call = run_in_background(
            sl.streamlines, fd.fields["u"], fd.fields["v"], fd.extent, density=1.0, mask=fd.mask,
            on_done=done, on_error=lambda m: self.state.log(f"Streamlines failed: {m}", "error", "Results"))

    def _save_field_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save field image", str(self._export_dir() / "field.png"),
                                              "PNG image (*.png)")
        if path:
            self.field_view.grab().save(path, "PNG")
            self.state.log(f"Saved {path}", "info", "Results")

    # -- sensitivity
    def _fill_sensitivity(self, r: CFDResult, sim) -> None:
        surfaces = [sd for sd in r.surface if sd.sensitivity is not None and len(sd.sensitivity)]
        self.sens_plot.clear_all()
        if not surfaces:
            self.sens_stack.setCurrentIndex(1)
            return
        self.sens_stack.setCurrentIndex(0)
        # Shown relative to its 98th-percentile magnitude, not in 1/m: the map's pattern is
        # right, but its size can be about twice the physical dJ/dn (the finite averaging
        # window, +23 % at the default window on a Re 40 cylinder, and the immersed
        # boundary's response to sub-cell shape changes; docs/theory_sensitivity.md section
        # 5). The 1/m values stay in the exports (surface CSV) for those who want them.
        allv = np.concatenate([np.abs(np.asarray(sd.sensitivity, float)) for sd in surfaces])
        allv = allv[np.isfinite(allv)]
        m = float(np.percentile(allv, 98)) if allv.size else 0.0
        if not m > 0:
            m = float(allv.max()) if allv.size and allv.max() > 0 else 1.0
        names = colormap_names()
        name = "RdBu_r" if "RdBu_r" in names else "coolwarm"
        lut = colormap_lut(name, "sensitivity")
        for sd in surfaces:
            rel = np.clip(np.asarray(sd.sensitivity, float) / m, -1.0, 1.0)
            self.sens_plot.add_outline(np.stack([sd.x, sd.y], axis=1), tokens()["border_strong"], width=1.0)
            self.sens_plot.add_value_outline(sd.x, sd.y, rel, lut, -1.0, 1.0, width=6)
        obj = {"cd": "Cd", "cl": "Cl", "ld": "L/D"}.get(sim.cfd.sensitivity_objective, sim.cfd.sensitivity_objective)
        self.sens_bar.set_scale(lut, -1.0, 1.0, f"d{obj} / dn, relative", "", extend=(True, True))
        self.sens_caption.setText(
            f"Where moving the surface outward changes {obj}: red increases it, blue decreases it. Scaled to "
            "the strongest 2 % of the outline; read the pattern, not the size (the magnitude can be about twice "
            "too large). Values in 1/m are in the surface CSV export.")
        self.sens_caption.setToolTip(f"1.0 on the colour bar is {sig(m, 3)} per metre of outward displacement "
                                     f"(98th percentile of |d{obj}/dn|).")
        self.sens_plot.autoRange()

    # -- polar
    def _fill_polar(self, r: PolarResult) -> None:
        self.polar_grid.set_polars([(self.state.simulation.name if self.state.simulation else "polar",
                                     r.table(), 0)])
        for p in self.polar_grid.plots():
            p.autoRange()
        rows = []
        dec = [d for _, d in POLAR_COLUMNS]
        for p in r.points:
            ld = p.cl / p.cd if finite(p.cd) and p.cd > 0 and finite(p.cl) else float("nan")
            vals = (p.alpha, p.cl, p.cd, p.cdp, p.cdf, p.cm, ld, p.xtr_upper, p.xtr_lower)
            rows.append([fmt_fixed(v, d) for v, d in zip(vals, dec)]
                        + [Cell("yes") if p.converged else Cell("no", tone="warn")])
        self.polar_table.set_data([h for h, _ in POLAR_COLUMNS], rows, numeric=list(range(9)))

    # ------------------------------------------------------------- exports
    def _export_dir(self) -> Path:
        p = self.state.project
        base = Path(getattr(p, "path", Path.home())) if p is not None else Path.home()
        d = base / "exports"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            return base
        return d

    def _stem(self) -> str:
        sim = self.state.simulation
        name = sim.name if sim else "result"
        # ASCII only: file names travel through e-mail, zip files and other operating systems
        return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "result"

    def export(self, key: str) -> None:
        """Export the current result: CSV / VTK / NPZ through aerolume.export, PNG through viz.figures."""
        r = self.result
        if r is None:
            return
        if key == "figures":
            return self._export_figures()
        exp = lazy_import("aerolume.export")
        fn = getattr(exp, key, None) if exp is not None else None
        if fn is None:
            msg = import_error("aerolume.export") or f"export.{key} is not available"
            self.export_status.setText(f"Export unavailable: {msg}")
            return
        ext, filt = {"fields_vtk": (".vti", "VTK image data (*.vti)"),
                     "fields_npz": (".npz", "NumPy archive (*.npz)")}.get(key, (".csv", "CSV (*.csv)"))
        suffix = key.replace("_csv", "").replace("fields_vtk", "fields").replace("fields_npz", "fields")
        default = self._export_dir() / f"{self._stem()}_{suffix}{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "Export", str(default), filt)
        if not path:
            return
        arg = r
        if key in ("fields_vtk", "fields_npz"):
            arg = self._field_slot()
            if arg is None:
                self.export_status.setText("This result has no flow fields to export.")
                return
        self.export_status.setText("Exporting…")
        run_in_background(fn, arg, path,
                          on_done=lambda *_: self._exported(path),
                          on_error=lambda m: self._export_failed(m))

    def _export_failed(self, message: str) -> None:
        self.export_status.setText(f"Export failed: {message}")
        self.state.log(f"Export failed: {message}", "error", "Results")

    def _exported(self, path: str) -> None:
        self.export_status.setText(f"Saved {Path(path).name}")
        self.state.log(f"Exported {path}", "info", "Results")

    def _export_figures(self) -> None:
        figs = lazy_import("aerolume.viz.figures")
        if figs is None:
            self.export_status.setText("Figures unavailable: " + import_error("aerolume.viz.figures"))
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the figures", str(self._export_dir()))
        if not folder:
            return
        r, sim, design, stem = self.result, self.state.simulation, self.design, self._stem()

        def work() -> list[str]:
            out: list[str] = []
            jobs: list[tuple[str, callable]] = []
            if design is not None:
                jobs.append(("geometry", lambda: figs.fig_geometry(design, alpha_deg=sim.conditions.alpha_deg)))
            if isinstance(r, CFDResult):
                if r.history:
                    jobs.append(("history", lambda: figs.fig_history(r.history, r.t_avg_start, r)))
                if r.surface:
                    jobs.append(("cp", lambda: figs.fig_cp(r.surface, kind=design.kind if design else "airfoil")))
                fd = r.mean_fields or r.instant_fields
                if fd is not None:
                    for name in ("speed", "cp", "vorticity"):
                        if name in fd.fields:
                            jobs.append((f"field_{name}",
                                         lambda n=name: figs.fig_field(fd, n, r.bodies_wind or None)))
                if any(sd.sensitivity is not None for sd in r.surface):
                    jobs.append(("sensitivity", lambda: figs.fig_sensitivity(
                        r.surface, objective=sim.cfd.sensitivity_objective)))
            else:
                jobs.append(("polar", lambda: figs.fig_polar(r)))
                best = r.best_ld()
                if best is not None and best.cp is not None:
                    jobs.append(("cp_best_ld", lambda: figs.fig_cp(best)))
            errors = []
            for name, make in jobs:
                try:
                    out.append(str(figs.save_figure(make(), Path(folder) / f"{stem}_{name}.png")))
                except Exception as exc:  # noqa: BLE001 - one broken figure must not lose the others
                    errors.append(f"{name}: {exc}")
            if errors and not out:
                raise RuntimeError("; ".join(errors))
            return out

        self.export_status.setText("Rendering figures…")
        run_in_background(work, on_done=lambda paths: self._exported_many(folder, paths),
                          on_error=lambda m: self._export_failed(m))

    def _exported_many(self, folder: str, paths: list[str]) -> None:
        self.export_status.setText(f"Saved {len(paths)} figures")
        self.state.log(f"Saved {len(paths)} figures to {folder}", "info", "Results")

    def _add_to_compare(self) -> None:
        """Add this run to the comparison, keeping it like with like.

        Runs of the other analysis are dropped from the set first: appending a
        NACA 4412 CFD run to the [polar, CFD] set a new study starts with kept
        the polar as the baseline, and the 2412-vs-4412 comparison had no
        deltas at all. With a single run left, Compare pairs it with the most
        similar other one.
        """
        sid = self.state.simulation_id
        p = self.state.project
        if sid and p is not None and sid in p.simulations:

            def kind_of(s: str) -> str:
                sim = p.simulations[s]
                return sim_kind(sim, p.designs.get(sim.design_id))

            kind = kind_of(sid)
            ids = self.state.compare_ids
            ids[:] = [s for s in ids if s in p.simulations and kind_of(s) == kind]
            if sid not in ids:
                ids.append(sid)
        self.state.navigate("compare")


__all__ = [
    "FAN_HEADERS",
    "SMOKE_COMPOSITE",
    "ResultsPage",
    "body_polygons",
    "coef_reference",
    "conditions_items",
    "cp_range",
    "deg_text",
    "fan_rows",
    "fan_units",
    "fill_table",
    "finite",
    "focus_rect",
    "ground_of",
    "has_fan_report",
    "has_heat",
    "header_facts",
    "heat_balance_rows",
    "heat_body_rows",
    "history_diagnostics",
    "is_lifting",
    "lift_label",
    "lift_word",
    "make_table",
    "point_curves",
    "polar_angle_items",
    "polar_metrics_safe",
    "polar_summary_rows",
    "result_diagnostics_rows",
    "robust_range",
    "rotate",
    "scalar_rows",
    "short_device",
    "sim_kind",
    "split_loop",
    "summary_note",
    "summary_rows",
    "surface_curves",
    "surface_heat",
    "temperature_field",
    "thermal_summary_rows",
]
