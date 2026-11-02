"""Views of 3D CFD and finite-wing results (Results page) and the 3D comparison tiles (Compare page).

The 2D Results page reads a result as a header line of facts, then tabs. A 3D
result is read the same way -- the engineer's questions do not change with the
dimension -- so these views keep the 2D layout and add what only 3D has:

* :class:`CFD3DResultView` -- **Summary** (the six coefficients CD, CS, CL,
  C_roll, Cm, Cn with mean, standard error, RMS, min, max, then forces in N,
  moments in N·m, drag area, power, Strouhal, Reynolds numbers), **Forces**
  (coefficient histories with the averaging window), **Surface** (the body in
  3D coloured by Cp or Cf, with camera presets), **Fields** (the flow in the
  volume around the body, DESIGN.md 7.13: streamlines of the mean flow from a
  movable seed rake, the stored smoke snapshot, the slice plane, Q-criterion
  vortex cores and surface Cp as toggles over one 3D view, and the slice as a
  2D field view beside or instead of it), **Sensitivity** (the dJ/dn map on the
  surface, relative, with its honesty note) and **Data** (every scalar and the
  exports: surface .vtp, volume and smoke .vti, slice CSV, histories CSV, mesh
  STL, a PNG of the 3D flow view).
* :class:`WingResultView` -- **Summary** (lift slope, zero-lift angle, CL,max or
  its lower bound, the stall-onset estimate, best L/D, CD0, Oswald e, span
  efficiency, Cm0 and dCm/dα from :func:`aerolume.analysis.wing_metrics`),
  **Polars**, **Span loading** (with the wing in 3D coloured by the local cl at
  the chosen angle), **Points** and **Data**.
* :class:`SurfaceCompareGrid` -- side-by-side 3D renders of several runs on one
  colour range with *linked cameras*: orbiting one tile turns all of them, so
  the same detail is always compared; the surfaces by Cp or Cf, optionally
  with each run's mean-flow streamlines from the same rake.

Design notes (the WHY):

* **3D views are built when their tab is first opened.** Every
  :class:`~aerolume.app.widgets.view3d.View3D` owns an off-screen OpenGL
  context and builds VTK pipelines; opening a result must not pay for four of
  them when the user reads the Summary. Tabs fill on first show and refill only
  when the result changes.
* **One colour rule.** Surface Cp, slice Cp and the 2D views share the range
  rule of the 2D pages (centred, at least ±1, at most ±3), and slices use
  :func:`~aerolume.app.widgets.field_view.display_range3d`, so a value has one
  colour on every view of the application.
* **Symmetry-plane runs are reported as full bodies** (DESIGN.md 7.2): side
  force, rolling and yawing moments are exactly zero by symmetry and are shown
  greyed with that reason, never as a measured zero.
* **Cross-sections without the solver stack.** The 2D slice view outlines the
  body with :func:`mesh_sections` (numpy; the solver's own ``mesh_section``
  lives next to PyTorch, which the GUI thread must not import).
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.fluid import display_name as fluid_name
from ...core.units import sig
from ...model import CFD3DResult, WingResult
from .common import KeyValueList, PropertyGrid, SegmentedControl, ToolStrip
from .field_view import (
    CP_LIMITS,
    FieldView,
    cmap_title,
    colormap_names,
    default_cmap3d,
    display_range3d,
    field_info3d,
    split_orientation,
)
from .kpi import (
    MINUS,
    Cell,
    DataTable,
    EmptyState,
    button,
    fmt_duration,
    fmt_fixed,
    fmt_num,
    fmt_re,
    fmt_speed,
    label,
    stat_texts,
    value_error_texts,
)
from .plots import HistoryPlot, lazy_import, tokens
from .wing_plots import SpanLoadingPlots, WingPolarGrid, nearest_point, span_order

log = logging.getLogger(__name__)

__all__ = [
    "COEFFS_3D",
    "SUMMARY_HEADERS",
    "VIEW_MODES",
    "CFD3DResultView",
    "Fields3DPanel",
    "Heat3DPanel",
    "Surface3DPanel",
    "SurfaceCompareGrid",
    "WingResultView",
    "compact_columns",
    "conditions_items3d",
    "crop_slice",
    "fit_table_width",
    "focus_box",
    "force_spectrum_peak",
    "frame_to_slice",
    "header_facts3d",
    "history_keys3d",
    "mesh_sections",
    "neutral_plane",
    "result_kind",
    "scalar_rows3d",
    "scroll_page",
    "stall_onset",
    "strouhal_cell",
    "summary_layout",
    "summary_rows3d",
    "surface_cp_range",
    "view_mode_control",
    "volume_bounds",
    "wing_conditions_items",
    "wing_header_facts",
    "wing_scalar_rows",
    "wing_summary_rows",
]

SUMMARY_HEADERS = ["Quantity", "Mean", "± Std. error", "RMS", "Min", "Max", "Unit"]
WING_SUMMARY_HEADERS = ["Quantity", "Value", "Condition", "Unit"]
#: (key, label) of the six 3D coefficients in reading order (DESIGN.md 7.2)
COEFFS_3D = (("cd", "Drag coefficient CD"), ("cs", "Side-force coefficient CS"), ("cl", "Lift coefficient CL"),
             ("c_roll", "Rolling-moment coefficient C_roll"), ("cm", "Pitching-moment coefficient Cm"),
             ("cn", "Yawing-moment coefficient Cn"))
#: coefficients that vanish by symmetry in a half-domain (symmetry-plane) run
ANTISYMMETRIC = frozenset({"cs", "c_roll", "cn"})
#: Surface fields offered on the Surface tab, in menu order: (key, menu text)
SURFACE_FIELDS = (("cp", "Pressure coefficient Cp"), ("cf", "Skin friction Cf"))
#: Fields a stored volume / slice may hold, in menu order
SLICE_FIELDS = ("speed", "cp", "vorticity", "q", "u", "v", "w", "p", "nut", "temperature", "temperature_rms")
_STALL_RE = re.compile(r"alpha\s*≈\s*(-?[0-9]+(?:\.[0-9]+)?)\s*deg,\s*CL\s*≈\s*(-?[0-9]+(?:\.[0-9]+)?)")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def result_kind(result) -> str:
    """'cfd3d', 'wing', or '2d' (2D CFD and airfoil polars): what can be compared with what."""
    if isinstance(result, CFD3DResult):
        return "cfd3d"
    if isinstance(result, WingResult):
        return "wing"
    return "2d"


def _deg(v: float) -> str:
    return f"{float(v):.6g}°".replace("-", MINUS) if _finite(v) else "—"


def surface_cp_range(values) -> tuple[float, float]:
    """The Cp colour range of every view: centred on 0, at least ±1, at most ±3 (the 2D rule)."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    a, b = np.percentile(v, [0.5, 99.5])
    m = min(max(abs(a), abs(b), CP_LIMITS[0]), CP_LIMITS[1])
    return -m, m


def _q_ref(r: CFD3DResult, design, sim) -> float:
    """q · A_ref [N]: the factor that turned the mean coefficients into the stored forces."""
    for force, st in ((r.drag, r.cd), (r.lift, r.cl), (r.side, r.cs)):
        if _finite(force) and _finite(st.mean) and abs(st.mean) > 1e-12:
            return float(force) / float(st.mean)
    try:
        return float(sim.conditions.dynamic_pressure()) * float(design.ref_area)
    except (AttributeError, TypeError, ValueError):
        return float("nan")


#: A Strouhal number is quoted only with a clear line in the force spectrum: its highest bin at least
#: this many times the median power of the band ...
STROUHAL_PEAK_RATIO = 5.0
#: ... and its main lobe (the peak bin and two on each side, the Hann window's width) holding at least
#: this share of the fluctuation energy in the band. The second test is needed because the highest of
#: N bins of pure noise is ~log2(N) x the median (5.6 to 12.6 x for 100 bins in eight draws), so a
#: ratio alone passes noise; noise spreads its energy (8-17 % in the lobe in those draws), a shedding
#: line does not (the Draft sedan's lift line: 60 %; a clean sine: 100 %).
STROUHAL_PEAK_SHARE = 0.3
#: The band searched [1 / convective time unit]: shedding of bluff bodies and cars lies far below it;
#: above it a smooth LES force record has only its decaying tail, whose tiny median would make any
#: peak look "clear"
SPECTRUM_BAND = 5.0


def force_spectrum_peak(r) -> tuple[float, float, float, str] | None:
    """(peak / median power, main-lobe energy share, peak frequency [U/L], "cl" or "cs") of a 3D run.

    The fluctuating lift or side force in the averaging window, whichever
    fluctuates more (the channel the solver took its Strouhal number from,
    ``analysis.summarize_history3d``); resampled to uniform steps, detrended and
    Hann-windowed; the band 0 < f <= SPECTRUM_BAND (time in convective units
    L/U, so f is a Strouhal number on the reference length). None when there
    is no usable record.
    """
    h = getattr(r, "history", None) or {}
    t = np.asarray(h.get("t", []), dtype=float)
    best: tuple[float, tuple[float, float, float, str]] | None = None
    for key in ("cl", "cs"):
        x = np.asarray(h.get(key, []), dtype=float)
        if x.size != t.size or t.size < 32:
            continue
        ok = np.isfinite(t) & np.isfinite(x)
        if _finite(getattr(r, "t_avg_start", None)):
            ok &= t >= float(r.t_avg_start)
        tt, xx = t[ok], x[ok]
        if xx.size < 32 or tt[-1] <= tt[0] or float(np.std(xx)) <= 0.0:
            continue
        tu = np.linspace(tt[0], tt[-1], xx.size)
        xu = np.interp(tu, tt, xx)
        xu = xu - np.polyval(np.polyfit(tu, xu, 1), tu)
        power = np.abs(np.fft.rfft(xu * np.hanning(xu.size))) ** 2
        freqs = np.fft.rfftfreq(xu.size, tu[1] - tu[0])
        band = (freqs > 0) & (freqs <= SPECTRUM_BAND)
        pb, fb = power[band], freqs[band]
        if pb.size < 8 or float(pb.sum()) <= 0.0:
            continue
        i = int(np.argmax(pb))
        med = float(np.median(pb))
        ratio = float(pb[i] / med) if med > 0 else float("inf")
        share = float(pb[max(i - 2, 0):i + 3].sum() / pb.sum())
        rms = float(np.std(xx))
        if best is None or rms > best[0]:
            best = (rms, (ratio, share, float(fb[i]), key))
    return None if best is None else best[1]


def strouhal_cell(r) -> tuple[str, str]:
    """(value text, tooltip) of the Strouhal number: shown only with a clear spectral line."""
    st = getattr(r, "strouhal", None)
    if st is None or not _finite(st):
        return "—", "No periodic force fluctuation was found"
    peak = force_spectrum_peak(r)
    if peak is None:
        return "—", "Too short a force record in the averaging window to find a shedding frequency"
    ratio, share, _f, key = peak
    name = "lift" if key == "cl" else "side-force"
    if ratio < STROUHAL_PEAK_RATIO or share < STROUHAL_PEAK_SHARE:
        return "—", (f"No clear shedding frequency: the {name} spectrum has no dominant line (highest peak "
                     f"{ratio:.1f} × the median power, {100 * share:.0f} % of the fluctuation energy; quoted "
                     f"above {STROUHAL_PEAK_RATIO:g} × and {100 * STROUHAL_PEAK_SHARE:.0f} %)")
    length = getattr(r, "strouhal_length", float("nan"))
    basis = f" on a length of {sig(float(length), 3)} m" if _finite(length) else ""
    return fmt_num(st, 3), (f"From the {name} spectrum: frequency of its dominant line × length / speed{basis} "
                            f"(peak {sig(ratio, 2)} × the median power, {100 * share:.0f} % of the fluctuation "
                            "energy)")


def history_keys3d(r: CFD3DResult) -> tuple[str, ...]:
    """Panels of the Forces tab: CD, CL, Cm; plus CS, C_roll, Cn unless they vanish by symmetry."""
    h = r.history or {}
    keys = ["cd", "cl", "cm"]
    if not r.symmetry_plane:
        for k in ("cs", "c_roll", "cn"):
            a = np.asarray(h.get(k, []), dtype=float)
            if a.size and np.nanmax(np.abs(a)) > 0:
                keys.append(k)
    order = ("cd", "cs", "cl", "c_roll", "cm", "cn")
    return tuple(k for k in order if k in keys and (k in h or not h))


# ---------------------------------------------------------------------------
# 3D CFD: header, summary and data rows (pure, tested directly)
# ---------------------------------------------------------------------------

def header_facts3d(r: CFD3DResult, sim, design=None) -> tuple[str, str | None, str]:
    """(status word, status kind, facts line) for the Results header of a 3D run."""
    if sim is not None and sim.status == "cancelled":
        status, kind = "Stopped early", "warn"
    elif r.converged:
        status, kind = "Converged", "good"
    else:
        status, kind = "Not converged", "warn"
    parts = []
    matched = _finite(r.re_real) and _finite(r.re_sim) and abs(r.re_sim / r.re_real - 1.0) < 0.01
    parts.append(f"Re {fmt_re(r.re_real)}" if matched else f"Re {fmt_re(r.re_sim)} simulated ({fmt_re(r.re_real)} real)")
    nx, ny, nz = (tuple(r.grid) + (0, 0, 0))[:3]
    if nx and ny and nz:
        cells = r.cells or nx * ny * nz
        text = f"{nx}×{ny}×{nz} grid, {cells / 1e6:.2f} M cells"
        if _finite(r.cells_per_ref):
            per = "MAC" if design is not None and design.kind == "wing" else "length"
            text += f", {r.cells_per_ref:.0f} per {per}"
        parts.append(text)
    parts.append("half domain (symmetry plane y = 0)" if r.symmetry_plane else "full domain")
    if r.turbulence:
        t = r.turbulence.strip()
        parts.append(t.upper() if t.lower().startswith("les") else t.capitalize())
    from ..pages.results import short_device, stored_heat_fact  # lazily: pages import this module
    stored = stored_heat_fact(r)
    if stored is not None:
        # the verdict judges the monitors, so a converged run can still be warming its domain
        parts.append(stored[0])
        if stored[1] and kind == "good":
            kind = "warn"
    if _finite(r.wall_seconds):
        parts.append(fmt_duration(r.wall_seconds))
    if r.device:
        parts.append(short_device(r.device))
    return status, kind, " · ".join(parts)


def summary_rows3d(r: CFD3DResult, design, sim) -> list[list]:
    """Rows of the 3D Summary table: quantity, mean, ± standard error, RMS, min, max, unit.

    Coefficients first (rounded to their standard errors, as in 2D), then the
    dimensional values: each is the coefficient's statistics scaled by the
    factor that turned its mean into the stored force or moment, so the mean
    column equals the result's own number exactly. In a symmetry-plane run the
    side force, rolling and yawing moments are zero by construction: those rows
    are greyed and say why.
    """
    window = (f"Mean over t = {sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U"
              if _finite(r.t_avg_start) and _finite(r.t_end) else "Time average")
    sym_tip = "Zero by symmetry: a half-domain run with a symmetry plane at y = 0 has no side force, " \
              "rolling or yawing moment"

    def stat_row(name: str, st, k: float = 1.0, unit: str = "", mean=None, tip: str = "",
                 zero_by_symmetry: bool = False) -> list:
        kk = k if _finite(k) else float("nan")
        lo, hi = st.min * kk, st.max * kk
        if kk < 0:
            lo, hi = hi, lo
        m = st.mean * kk if mean is None else mean
        se = st.stderr * abs(kk)
        if zero_by_symmetry:
            return [Cell(name, tooltip=sym_tip, tone="muted"), Cell("0", tone="muted", tooltip=sym_tip),
                    "—", "—", "—", "—", unit]
        if R is not None:
            return R.stat_cells(name, st, unit, tip=tip, window=window, scale=k, mean=mean)
        mean_t, se_t, rms_t, lo_t, hi_t = stat_texts(m, se, st.rms * abs(kk), lo, hi)
        return [Cell(name, tooltip=tip), Cell(mean_t, tooltip=window), se_t, rms_t, lo_t, hi_t, unit]

    R = _results_page()
    if R is not None and sim is not None and R.is_closed(sim):
        # a room: the cabinet's "CD −0.48" and its side force and rolling moment mean nothing here (they
        # stay in Forces); temperatures, heater powers and heat paths lead instead
        rows = R.closed_thermal_rows(r, design, sim, window) + R.closed_fan_rows(r, sim, window)
        if not R.natural_convection(design, sim):
            rows.append([Cell("Reynolds number", tooltip="From the run's velocity scale (Conditions) and the "
                                                         "reference length"), fmt_re(r.re_sim), "—", "—", "—", "—", ""])
        return rows

    sym = bool(r.symmetry_plane)
    lifting = design is not None and design.kind in ("wing",)
    qa = _q_ref(r, design, sim)
    lref = float(getattr(design, "length_ref", float("nan")) or float("nan")) if design is not None else float("nan")
    sref = float(getattr(design, "span_ref", 0.0) or float("nan")) if design is not None else float("nan")
    area = float(getattr(design, "ref_area", 0.0) or float("nan")) if design is not None else float("nan")
    ref_tip = (f"On q × {fmt_num(area, 4)} m² (reference area)" if _finite(area) else "On q × reference area")
    tips = {"cd": ref_tip, "cs": ref_tip + "; positive to the left (+y)", "cl": ref_tip + "; negative is downforce",
            "c_roll": ref_tip + " × span reference; about +x", "cm": ref_tip + " × length reference; positive nose-up",
            "cn": ref_tip + " × span reference; about +z"}
    rows = [stat_row(lab, getattr(r, key), tip=tips[key], zero_by_symmetry=sym and key in ANTISYMMETRIC)
            for key, lab in COEFFS_3D]
    if R is not None and hasattr(R, "buoyant_scale_rows"):
        # a buoyancy-dominated tunnel run (the hot sphere, Ri 627, read CD 6.14 and CL 119 on the free stream's
        # tiny q): the coefficients on ½ρu_b², right under the free-stream ones
        rows += R.buoyant_scale_rows(r, design, sim, window, (("cd", "CD"), ("cl", "CL")))
    if lifting:
        se = float("nan")
        if all(_finite(v) for v in (r.cl.mean, r.cd.mean, r.cl.stderr, r.cd.stderr)) and r.cd.mean:
            ld = r.cl.mean / r.cd.mean
            se = abs(ld) * math.hypot(r.cl.stderr / r.cl.mean if r.cl.mean else 0.0, r.cd.stderr / r.cd.mean)
        ld_t, se_t = value_error_texts(r.lift_to_drag, se)
        rows.append([Cell("Lift-to-drag ratio L/D"), Cell(ld_t, tooltip="Ratio of the means"), se_t,
                     "—", "—", "—", ""])
    moments = tuple(r.moments) + (float("nan"),) * 3
    st_text, st_tip = strouhal_cell(r)
    lift_name = "Lift" if lifting or design is None or design.kind != "vehicle" else "Lift (downforce if negative)"
    rows += [
        stat_row("Drag", r.cd, qa, "N", mean=r.drag),
        stat_row("Side force", r.cs, qa, "N", mean=r.side, zero_by_symmetry=sym),
        stat_row(lift_name, r.cl, qa, "N", mean=r.lift),
        stat_row("Rolling moment", r.c_roll, qa * sref, "N·m", mean=moments[0], zero_by_symmetry=sym),
        stat_row("Pitching moment", r.cm, qa * lref, "N·m", mean=moments[1],
                 tip="About the moment reference point; positive nose-up"),
        stat_row("Yawing moment", r.cn, qa * sref, "N·m", mean=moments[2], zero_by_symmetry=sym),
        stat_row("Drag area CDA", r.cd, area, "m²", mean=r.drag_area, tip="CD × reference area"),
        stat_row("Drag power", r.cd, qa * (sim.conditions.speed if sim is not None else float("nan")), "W",
                 mean=r.power, tip="Power needed to overcome the mean drag"),
        [Cell("Strouhal number St", tooltip=st_tip), Cell(st_text, tooltip=st_tip), "—", "—", "—", "—", ""],
        ["Reynolds number (real)", fmt_re(r.re_real), "—", "—", "—", "—", ""],
        [Cell("Reynolds number (simulated)", tooltip="Equal to the real value unless it was capped"),
         Cell(fmt_re(r.re_sim), tone=None if (_finite(r.re_sim) and _finite(r.re_real)
                                              and abs(r.re_sim / r.re_real - 1) < 0.01) else "warn"),
         "—", "—", "—", "—", ""],
    ]
    return rows


def summary_note3d(r: CFD3DResult, design, sim, rows=None) -> str:
    """Footnote of the 3D Summary: references and rounding (the steady sentence and the colour legend only
    when ``rows`` have such a cell; no coefficient reference in a closed room, which shows none)."""
    R = _results_page()
    rounding = R.rounding_note(rows) if R is not None else \
        "Each value is rounded to its standard error (one or two significant figures)."
    if R is not None and sim is not None and R.is_closed(sim):
        return rounding
    parts = []
    if design is not None:
        if design.ref_area:
            what = "planform area" if design.kind == "wing" else "frontal area"
            parts.append(f"reference area {fmt_num(design.ref_area, 4)} m² ({what})")
        if design.length_ref:
            parts.append(f"length {fmt_num(design.length_ref, 4)} m")
        if design.span_ref:
            parts.append(f"span {fmt_num(design.span_ref, 4)} m")
    try:
        q = sim.conditions.dynamic_pressure()
        parts.append(f"dynamic pressure {fmt_num(q, 4)} Pa")
    except AttributeError:
        pass
    ref = ("Coefficients are referenced to the " + ", ".join(parts) + "; moments are about "
           f"({', '.join(fmt_num(c, 3) for c in design.moment_ref3)}) m. ") if parts and design is not None else ""
    sym = ("Forces are for the whole body although half of it was simulated. " if r.symmetry_plane else "")
    buoyant = R.buoyant_scale_caption(design, sim) if R is not None and hasattr(R, "buoyant_scale_caption") else ""
    return ref + sym + (buoyant + " " if buoyant else "") + rounding


def conditions_items3d(r: CFD3DResult, design, sim) -> list[tuple]:
    """Key-value facts next to the 3D Summary table (what the numbers are valid for).

    A closed room has no free stream: the speed, angle of attack, yaw and
    dynamic pressure of a tunnel make way for its velocity scale, Rayleigh
    and Prandtl numbers and wall temperatures (the 2D page's
    :func:`~aerolume.app.pages.results.closed_conditions`)."""
    R = _results_page()
    if R is not None and sim is not None and R.is_closed(sim):
        items = list(R.closed_conditions(r, design, sim))
        box = getattr(getattr(sim, "cfd", None), "domain_box", None)
        if box is not None and len(box) == 6:
            items.insert(1, ("Room", " × ".join(fmt_num(box[2 * i + 1] - box[2 * i], 3) for i in range(3)) + " m"))
        if _finite(r.t_avg_start) and _finite(r.t_end) and not any(it[0] == "Averaging window" for it in items):
            items.append(("Averaging window", f"{sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U"))
        return items
    items: list[tuple] = []
    if design is not None:
        items.append(("Design", design.name))
    if sim is not None:
        c = sim.conditions
        items.append(("Speed", fmt_speed(c.speed)))
        items.append(("Pitch angle" if design is not None and design.kind == "vehicle" else "Angle of attack",
                      _deg(c.alpha_deg)))
        items.append(("Yaw angle", _deg(getattr(c, "yaw_deg", 0.0))))
        items += [("Fluid", fluid_name(c.fluid.name)), ("Density", f"{fmt_num(c.fluid.density, 4)} kg/m³"),
                  ("Dynamic pressure", f"{fmt_num(c.dynamic_pressure(), 4)} Pa")]
        if getattr(sim.ground, "enabled", False):
            items.append(("Road", "moving belt" if sim.ground.moving else "fixed floor"))
    if design is not None:
        if design.ref_area:
            items.append(("Reference area", f"{fmt_num(design.ref_area, 4)} m²"))
        items.append(("Reference length", f"{fmt_num(design.length_ref, 4)} m"))
        if design.span_ref:
            items.append(("Span reference", f"{fmt_num(design.span_ref, 4)} m"))
        items.append(("Moment reference", "(" + ", ".join(fmt_num(c, 3) for c in design.moment_ref3) + ") m"))
    items.append(("Domain", "half, symmetry plane y = 0" if r.symmetry_plane else "full"))
    if _finite(r.t_avg_start) and _finite(r.t_end):
        items.append(("Averaging window", f"{sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U"))
    if r.precision:
        items.append(("Precision", r.precision))
    return items


def scalar_rows3d(r: CFD3DResult, design, sim) -> list[tuple[str, str, str]]:
    """(quantity, value, unit) rows for the Data tab of a 3D run."""
    rows: list[tuple[str, str, str]] = []
    for key, lab in COEFFS_3D:
        st = getattr(r, key)
        short = lab.split()[-1]
        rows += [(f"{short} mean", fmt_num(st.mean, 6), ""), (f"{short} standard error", fmt_num(st.stderr, 3), ""),
                 (f"{short} rms fluctuation", fmt_num(st.rms, 4), "")]
    m = tuple(r.moments) + (float("nan"),) * 3
    rows += [("Drag", fmt_num(r.drag, 6), "N"), ("Side force", fmt_num(r.side, 6), "N"),
             ("Lift", fmt_num(r.lift, 6), "N"), ("Rolling moment", fmt_num(m[0], 6), "N·m"),
             ("Pitching moment", fmt_num(m[1], 6), "N·m"), ("Yawing moment", fmt_num(m[2], 6), "N·m"),
             ("Drag area CDA", fmt_num(r.drag_area, 5), "m²"), ("Drag power", fmt_num(r.power, 5), "W"),
             ("Strouhal number" + ("" if strouhal_cell(r)[0] != "—" or r.strouhal is None
                                   else " (no clear spectral peak)"),
              fmt_num(r.strouhal, 4) if r.strouhal is not None else "—", ""),
             ("Reynolds number (real)", fmt_re(r.re_real), ""), ("Reynolds number (simulated)", fmt_re(r.re_sim), ""),
             ("Grid", " × ".join(str(n) for n in r.grid), "cells"), ("Cells", f"{r.cells:,}", ""),
             ("Cells per reference length", fmt_num(r.cells_per_ref, 4), ""),
             ("Symmetry plane", "yes (y = 0)" if r.symmetry_plane else "no", ""),
             ("Simulated time", fmt_num(r.t_end, 4), "L/U"), ("Averaging from", fmt_num(r.t_avg_start, 4), "L/U"),
             ("Statistically converged", "yes" if r.converged else "no", ""),
             ("Turbulence treatment", r.turbulence or "—", ""), ("Device", r.device or "—", ""),
             ("Precision", r.precision or "—", ""), ("Wall-clock time", fmt_num(r.wall_seconds, 4), "s")]
    for name, d in (r.per_body or {}).items():
        rows.append((f"Part “{name}”: CD / CL", f"{fmt_num(d.get('cd'), 4)} / {fmt_num(d.get('cl'), 4)}", ""))
    if sim is not None:
        c = sim.conditions
        rows += [("Speed", fmt_num(c.speed, 5), "m/s"), ("Angle", fmt_num(c.alpha_deg, 4), "°"),
                 ("Yaw", fmt_num(getattr(c, "yaw_deg", 0.0), 4), "°"), ("Fluid", fluid_name(c.fluid.name), ""),
                 ("Density", fmt_num(c.fluid.density, 5), "kg/m³"),
                 ("Dynamic viscosity", fmt_num(c.fluid.viscosity, 5), "Pa·s")]
    if design is not None:
        rows += [("Design", design.name, ""), ("Reference area", fmt_num(design.ref_area, 5), "m²"),
                 ("Reference length", fmt_num(design.length_ref, 5), "m"),
                 ("Span reference", fmt_num(design.span_ref, 5), "m")]
    return rows


# ---------------------------------------------------------------------------
# wings: header, summary and data rows
# ---------------------------------------------------------------------------

def wing_metrics_safe(w: WingResult) -> dict:
    """``analysis.wing_metrics`` or an empty dict (with the failure logged)."""
    an = lazy_import("aerolume.analysis")
    fn = getattr(an, "wing_metrics", None) if an is not None else None
    if fn is None:
        return {}
    try:
        return dict(fn(w))
    except Exception:
        log.exception("analysis.wing_metrics failed")
        return {}


def stall_onset(w: WingResult) -> tuple[float, float] | None:
    """(alpha, CL) of the critical-section stall-onset estimate (None when the sweep has none).

    Read from ``WingResult.stall_alpha`` / ``stall_cl``; results saved before
    those fields existed carry it only in their notes, which are parsed then.
    """
    a, c = getattr(w, "stall_alpha", float("nan")), getattr(w, "stall_cl", float("nan"))
    if _finite(a) and _finite(c):
        return float(a), float(c)
    for n in getattr(w, "notes", []) or []:
        if "Stall onset" in str(n):
            m = _STALL_RE.search(str(n))
            if m:
                return float(m.group(1)), float(m.group(2))
    return None


def method_text(w: WingResult) -> str:
    """'Vortex lattice · viscous strips' / 'Lifting line · inviscid'."""
    m = "Vortex lattice" if w.method == "vlm" else "Lifting line" if w.method == "llt" else str(w.method)
    return f"{m} · {'viscous strips' if w.viscous else 'inviscid'}"


def wing_header_facts(w: WingResult, sim=None) -> tuple[str, str | None, str]:
    """(status, kind, facts) for the Results header of a wing sweep."""
    n_ok = sum(1 for p in w.points if p.converged)
    n = len(w.points)
    stopped = sim is not None and sim.status == "cancelled"
    status = f"{n_ok} of {n} angles converged" + (" (stopped)" if stopped else "")
    kind = "good" if n_ok == n and n and not stopped else "warn"
    parts = [method_text(w), f"Re {fmt_re(w.re_mac)} on the MAC ({fmt_num(w.mac, 3)} m)"]
    if _finite(w.aspect_ratio):
        parts.append(f"AR {w.aspect_ratio:.2f}")
    ws = getattr(sim, "wing", None)
    if ws is not None and w.viscous:
        parts.append(f"Ncrit {float(ws.ncrit):g}")
    if _finite(w.wall_seconds):
        parts.append(fmt_duration(w.wall_seconds))
    return status, kind, " · ".join(parts)


def wing_summary_rows(w: WingResult) -> list[list]:
    """Rows of the wing Summary table: quantity, value, condition, unit (wing_metrics)."""
    m = wing_metrics_safe(w)
    nan = float("nan")

    def g(k):
        return m.get(k, nan)

    inviscid = not w.viscous
    na = Cell("—", tone="muted", tooltip="Needs profile drag: this sweep is inviscid")
    if g("cl_max_reached") == 0.0:
        clmax = [Cell("Maximum lift CL,max", tooltip="The lift was still rising at the last angle: extend the sweep "
                      "to find CL,max"), Cell(f"≥ {fmt_fixed(g('cl_max'), 4)}", tone="warn"),
                 f"still rising at α = {fmt_fixed(g('alpha_cl_max'), 2)}°", ""]
    else:
        clmax = ["Maximum lift CL,max", fmt_fixed(g("cl_max"), 4), f"α = {fmt_fixed(g('alpha_cl_max'), 2)}°", ""]
    rows = [
        [Cell("Lift-curve slope CL,α", tooltip="Over the attached range; lifting-line theory for an elliptic wing: "
                                              "2π·AR/(AR + 2)"),
         fmt_fixed(g("cl_alpha_per_rad"), 3), f"{fmt_fixed(g('cl_alpha_per_deg'), 4)} per degree", "1/rad"],
        ["Zero-lift angle α0", fmt_fixed(g("alpha_zero_lift"), 2), "CL = 0", "°"],
        clmax,
    ]
    so = stall_onset(w)
    if so is not None:
        rows.append([Cell("Stall onset (estimate)", tooltip="The first strip reaching its section cl,max: the "
                          "critical-section estimate of the wing's CL,max. Not validated against stall data."),
                     fmt_fixed(so[1], 3), f"α ≈ {fmt_fixed(so[0], 1)}°", ""])
    if inviscid:
        rows += [["Best lift-to-drag (L/D)max", na, "", ""], ["Zero-lift drag CD0", na, "", ""],
                 ["Minimum drag CD,min", na, "", ""]]
    else:
        rows += [
            ["Best lift-to-drag (L/D)max", fmt_fixed(g("ld_max"), 1),
             f"α = {fmt_fixed(g('alpha_ld_max'), 2)}°, CL = {fmt_fixed(g('cl_at_ld_max'), 3)}", ""],
            [Cell("Zero-lift drag CD0", tooltip="Parabolic fit CD = CD0 + b·CL + k·CL² over the attached range"),
             fmt_fixed(g("cd0"), 5), "parabolic fit", ""],
            ["Minimum drag CD,min", fmt_fixed(g("cd_min"), 5), f"CL = {fmt_fixed(g('cl_at_cd_min'), 3)}", ""],
        ]
    rows += [
        [Cell("Oswald efficiency e", tooltip="1 / (π·AR·k) from the parabolic drag polar; includes the lift "
                                             "dependence of profile drag"),
         fmt_fixed(g("oswald"), 3), "parabolic fit", ""],
        [Cell("Span efficiency e", tooltip="CL² / (π·AR·CDi) at the cruise point (1 = elliptic loading)"),
         fmt_fixed(g("e_cruise"), 3), f"at CL = {fmt_fixed(g('cl_cruise'), 3)}", ""],
        ["Zero-lift moment Cm0", fmt_fixed(g("cm0"), 4), "about the moment reference", ""],
        [Cell("Moment slope dCm/dα", tooltip="Negative: statically stable about the moment reference point"),
         fmt_fixed(g("cm_alpha_per_deg"), 4), "stable" if _finite(g("cm_alpha_per_deg"))
         and g("cm_alpha_per_deg") < 0 else "", "1/°"],
        ["Reynolds number (MAC)", fmt_re(w.re_mac), "", ""],
        ["Aspect ratio", fmt_fixed(w.aspect_ratio, 2), f"span {fmt_num(w.span, 4)} m", ""],
        ["Points converged", f"{sum(1 for p in w.points if p.converged)} of {len(w.points)}", method_text(w), ""],
    ]
    return rows


def wing_conditions_items(w: WingResult, design, sim) -> list[tuple]:
    items: list[tuple] = []
    if design is not None:
        items.append(("Design", design.name))
    if sim is not None:
        c = sim.conditions
        items += [("Speed", fmt_speed(c.speed)), ("Yaw (sideslip)", _deg(getattr(c, "yaw_deg", 0.0))),
                  ("Fluid", fluid_name(c.fluid.name)), ("Density", f"{fmt_num(c.fluid.density, 4)} kg/m³"),
                  ("Dynamic pressure", f"{fmt_num(c.dynamic_pressure(), 4)} Pa")]
    items += [("Planform area", f"{fmt_num(w.area, 4)} m²"), ("Span", f"{fmt_num(w.span, 4)} m"),
              ("Aspect ratio", fmt_num(w.aspect_ratio, 4)), ("Mean aerodynamic chord", f"{fmt_num(w.mac, 4)} m"),
              ("Method", method_text(w))]
    ws = getattr(sim, "wing", None) if sim is not None else None
    if ws is not None:
        items += [("Angles", f"{_deg(ws.alpha_start)} to {_deg(ws.alpha_end)}, step {_deg(ws.alpha_step)}")]
        if w.viscous:
            items += [("Ncrit", f"{float(ws.ncrit):g}"), ("Mach", f"{float(ws.mach):g}")]
    if design is not None and any(design.moment_ref3):
        items.append(("Moment reference", "(" + ", ".join(fmt_num(c, 3) for c in design.moment_ref3) + ") m"))
    return items


def wing_scalar_rows(w: WingResult, design, sim) -> list[tuple[str, str, str]]:
    m = wing_metrics_safe(w)
    units = {"alpha": "°", "per_deg": "1/°", "per_rad": "1/rad"}
    labels = {"cl_alpha_per_deg": "Lift slope", "cl_alpha_per_rad": "Lift slope", "alpha_zero_lift": "Zero-lift angle",
              "cl_max": "CL,max", "alpha_cl_max": "Angle of CL,max", "cl_max_reached": "Stall reached (1 = yes)",
              "cd_min": "CD,min", "cl_at_cd_min": "CL at CD,min", "alpha_cd_min": "Angle of CD,min",
              "ld_max": "(L/D)max", "alpha_ld_max": "Angle of (L/D)max", "cl_at_ld_max": "CL at (L/D)max",
              "cd0": "CD0 (parabolic fit)", "k_parabolic": "k (parabolic fit)", "oswald": "Oswald efficiency",
              "e_cruise": "Span efficiency at cruise", "cl_cruise": "Cruise CL", "alpha_cruise": "Cruise angle",
              "cm0": "Cm0", "cm_alpha_per_deg": "dCm/dα", "aspect_ratio": "Aspect ratio", "alpha_min": "Lowest angle",
              "alpha_max": "Highest angle", "n_points": "Points computed", "n_converged": "Points converged"}
    rows: list[tuple[str, str, str]] = []
    for k, v in m.items():
        unit = next((u for frag, u in units.items() if frag in k), "")
        text = str(int(v)) if k.startswith("n_") and _finite(v) else fmt_num(v, 5)
        rows.append((labels.get(k, k.replace("_", " ")), text, unit))
    rows += [("Planform area", fmt_num(w.area, 5), "m²"), ("Span", fmt_num(w.span, 5), "m"),
             ("Mean aerodynamic chord", fmt_num(w.mac, 5), "m"), ("Reynolds number (MAC)", fmt_re(w.re_mac), ""),
             ("Method", method_text(w), ""), ("Wall-clock time", fmt_num(w.wall_seconds, 4), "s")]
    if sim is not None:
        c = sim.conditions
        rows += [("Speed", fmt_num(c.speed, 5), "m/s"), ("Fluid", fluid_name(c.fluid.name), ""),
                 ("Density", fmt_num(c.fluid.density, 5), "kg/m³")]
    if design is not None:
        rows.append(("Design", design.name, ""))
    return rows


# ---------------------------------------------------------------------------
# cross-sections of meshes (numpy; the slice outlines of the 2D view)
# ---------------------------------------------------------------------------

_AXIS = {"x": 0, "y": 1, "z": 2}
_SLICE_AXES = {"y": (0, 2), "z": (0, 1), "x": (1, 2)}


def mesh_sections(meshes: Sequence, axis: str, position: float) -> list[np.ndarray]:
    """Cross-section polylines of triangle meshes with the plane ``axis = position``.

    ``meshes`` are objects with ``vertices`` / ``faces`` (Mesh3D, SurfaceData3D)
    or (vertices, faces) pairs. Returns (N, 2) arrays in slice coordinates
    (SliceData column, row): closed loops repeat their first point, open sheets
    give open chains -- the convention of ``simulation.live3d.mesh_section``,
    which the field view's :meth:`~FieldView.set_sections` reads. Each triangle
    that straddles the plane gives one segment between the crossings of its two
    cut edges; segments are chained through the cut edges, which two triangles
    share. Vertices on the plane are nudged to one side first, so every
    crossing is a proper one.
    """
    k = _AXIS[axis]
    ci, ri = _SLICE_AXES[axis]
    out: list[np.ndarray] = []
    for m in meshes or []:
        if isinstance(m, (tuple, list)):
            v, f = np.asarray(m[0], dtype=float), np.asarray(m[1], dtype=np.int64)
        else:
            v, f = np.asarray(m.vertices, dtype=float), np.asarray(m.faces, dtype=np.int64)
        if len(v) == 0 or len(f) == 0:
            continue
        lo, hi = float(v[:, k].min()), float(v[:, k].max())
        if not (lo < position < hi):
            continue
        d = v[:, k] - position
        tiny = 1e-9 * max(hi - lo, 1e-12)
        d = np.where(np.abs(d) < tiny, tiny, d)              # nudge on-plane vertices to the + side
        s = d[f] > 0                                          # (M, 3)
        cut = s.any(axis=1) & ~s.all(axis=1)
        fc = f[cut]
        if len(fc) == 0:
            continue
        sc = s[cut]
        # the lone vertex (on its own side) and the two edges from it are the cut edges
        lone = np.where(sc.sum(axis=1) == 1, np.argmax(sc, axis=1), np.argmin(sc, axis=1))
        a = fc[np.arange(len(fc)), lone]
        b = fc[np.arange(len(fc)), (lone + 1) % 3]
        c = fc[np.arange(len(fc)), (lone + 2) % 3]

        p1, p2 = _crossings(v, d, a, b), _crossings(v, d, a, c)
        e1 = np.sort(np.stack([a, b], 1), axis=1)
        e2 = np.sort(np.stack([a, c], 1), axis=1)
        n_v = len(v)
        k1 = (e1[:, 0] * n_v + e1[:, 1]).tolist()
        k2 = (e2[:, 0] * n_v + e2[:, 1]).tolist()
        pts2: dict[int, np.ndarray] = {}
        for kk, pp in ((k1, p1), (k2, p2)):
            for key, pt in zip(kk, pp[:, (ci, ri)]):
                pts2[key] = pt
        out.extend(np.array([pts2[q] for q in chain]) for chain in _chain_segments(list(zip(k1, k2))))
    return out


def _crossings(v: np.ndarray, d: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Points where the edges p-q (vertex indices) cross the plane d = 0 (signed distances ``d``)."""
    t = d[p] / (d[p] - d[q])
    return v[p] + t[:, None] * (v[q] - v[p])


def _chain_segments(ends: list[tuple[int, int]]) -> list[list[int]]:
    """Chain segments, given by the keys of their two end edges, into polylines of keys.

    A key shared by two segments joins them. Open chains start at keys used by
    one segment only (a sheet's border); the rest are closed loops, whose key
    list returns to its first key.
    """
    adj: dict[int, list[int]] = {}
    for i, (x1, x2) in enumerate(ends):
        adj.setdefault(x1, []).append(i)
        adj.setdefault(x2, []).append(i)
    used = [False] * len(ends)

    def walk(seg: int, key: int) -> list[int]:
        chain = [key]
        while True:
            used[seg] = True
            a_, b_ = ends[seg]
            key = b_ if key == a_ else a_
            chain.append(key)
            nxt = [s_ for s_ in adj.get(key, []) if not used[s_]]
            if not nxt:
                return chain
            seg = nxt[0]

    chains = [walk(segs[0], key) for key, segs in adj.items() if len(segs) == 1 and not used[segs[0]]]
    chains += [walk(i, ends[i][0]) for i in range(len(ends)) if not used[i]]
    return chains


def focus_box(surfaces: Sequence, *, up: float = 0.6, down: float = 2.0, side: float = 0.6,
               ground: float | None = None) -> tuple[float, ...] | None:
    """The bodies' box grown by multiples of their length: the 3D views' default framing."""
    pts = [np.asarray(s.vertices, dtype=float) for s in surfaces if len(getattr(s, "vertices", []))]
    if not pts:
        return None
    a = np.vstack(pts)
    x0, y0, z0 = a.min(axis=0)
    x1, y1, z1 = a.max(axis=0)
    L = max(x1 - x0, y1 - y0, z1 - z0, 1e-9)
    box = [x0 - up * L, x1 + down * L, y0 - side * L, y1 + side * L, z0 - side * L, z1 + side * L]
    if ground is not None:
        box[4] = max(box[4], float(ground))
    return tuple(float(b) for b in box)


def crop_slice(sd, window: Sequence[float] | None):
    """A SliceData cut down to ``window`` (col0, col1, row0, row1) [m]; unchanged without one."""
    from ...model import SliceData
    if window is None:
        return sd
    first = next(iter(sd.fields.values()))
    nr, nc = np.shape(first)
    c0, c1, r0, r1 = (float(e) for e in sd.extent)
    w0, w1, w2, w3 = (float(e) for e in window)
    i0 = int(np.clip(math.floor((w0 - c0) / (c1 - c0) * nc), 0, nc - 1))
    i1 = int(np.clip(math.ceil((w1 - c0) / (c1 - c0) * nc), i0 + 1, nc))
    j0 = int(np.clip(math.floor((w2 - r0) / (r1 - r0) * nr), 0, nr - 1))
    j1 = int(np.clip(math.ceil((w3 - r0) / (r1 - r0) * nr), j0 + 1, nr))
    hc, hr = (c1 - c0) / nc, (r1 - r0) / nr
    ext = (c0 + i0 * hc, c0 + i1 * hc, r0 + j0 * hr, r0 + j1 * hr)
    fields = {k: np.asarray(v)[j0:j1, i0:i1] for k, v in sd.fields.items()}
    mask = np.asarray(sd.mask)[j0:j1, i0:i1] if sd.mask is not None else None
    return SliceData(axis=sd.axis, position=sd.position, extent=ext, fields=fields, mask=mask, label=sd.label)


def volume_bounds(volume) -> tuple[float, ...]:
    """(x0, x1, y0, y1, z0, z1) of a VolumeData, mirrored to the full width for symmetry-plane runs."""
    first = next(iter(volume.fields.values()))
    nz, ny, nx = np.shape(first)
    ox, oy, oz = volume.origin
    dx, dy, dz = volume.spacing
    y0, y1 = oy, oy + ny * dy
    if volume.mirrored:
        y0 = -y1
    return (ox, ox + nx * dx, y0, y1, oz, oz + nz * dz)


def _safe(name: str) -> str:
    return _SAFE_NAME.sub("_", name).strip("_") or "result"


def closed_room_box(sim) -> tuple[float, ...] | None:
    """The (x0, x1, y0, y1, z0, z1) box of a closed 3D domain (``cfd.domain_box``), None for a tunnel."""
    cfd = getattr(sim, "cfd", None)
    if getattr(cfd, "domain_mode", "tunnel") != "closed":
        return None
    try:
        b = tuple(float(v) for v in getattr(cfd, "domain_box", None) or ())
    except (TypeError, ValueError):
        return None
    ok = len(b) == 6 and all(math.isfinite(v) for v in b) and b[1] > b[0] and b[3] > b[2] and b[5] > b[4]
    return b if ok else None


def _view3d(**kw):
    """A View3D (imported here: the module loads VTK only when its first scene is made)."""
    from .view3d import View3D
    return View3D(**kw)


def _save_view_png(parent: QWidget, view, default: Path) -> str | None:
    """Ask for a file name and save the 3D view at 1600 x 1000 (independent of the window size)."""
    path, _ = QFileDialog.getSaveFileName(parent, "Save image", str(default), "PNG image (*.png)")
    if not path:
        return None
    from PySide6.QtGui import QImage
    img = np.ascontiguousarray(view.render_image(1600, 1000))
    h, w = img.shape[:2]
    QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy().save(path, "PNG")
    return path


def fixed_hint_combo(min_chars: int = 8) -> QComboBox:
    """A combo box whose minimum width does not follow its longest item.

    A QComboBox sizes itself to its contents when first shown, and again when
    a style sheet is applied: after a theme change the Field combo of the
    Fields panel asked for 167 px instead of 90, the panel's rows grew wider
    than the panel and their right edge was cut off. This one asks for
    ``min_chars`` characters and takes whatever width its row gives it.
    """
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(min_chars)
    return c


def _cmap_combo() -> QComboBox:
    c = fixed_hint_combo()
    c.setMinimumWidth(140)
    c.setToolTip("Colour map; (default) is the recommended map for the field")
    return c


def fit_cmap_combo(combo: QComboBox) -> int:
    """Make a colour-map combo wide enough for its longest possible item, "<name> (default)".

    For a combo in a tool strip, where nothing stretches it: the 8-character
    hint of :func:`fixed_hint_combo` clipped "Cool–warm (default)" to "Cool–warm
    (defau". The width covers every map with the mark, so the strip does not
    shift when the mark moves to another map with the field. Returns the width.
    """
    fm = combo.fontMetrics()
    widest = max((fm.horizontalAdvance(f"{cmap_title(n)} (default)") for n in colormap_names()), default=0)
    w = max(140, widest + 52)          # + padding (6 + 20 px), frame and the 18 px arrow of the themed combo
    combo.setMinimumWidth(w)
    return w


def _fill_cmap_combo(combo: QComboBox, field: str) -> None:
    """List the colour maps with the field's 3D default marked "(default)" (data None), keeping a pick."""
    from .field_view import fill_cmap_combo
    fill_cmap_combo(combo, default_cmap3d(field), combo.currentData())


# ---------------------------------------------------------------------------
# surface panel (Surface and Sensitivity tabs)
# ---------------------------------------------------------------------------

class Surface3DPanel(QWidget):
    """The body in 3D coloured by a per-vertex surface field, with camera presets and a colour bar.

    ``mode="fields"`` offers Cp and Cf, and **friction lines**: streamlines of
    the wall shear on the surface (the picture an oil-flow test gives:
    attachment, separation and reattachment lines), drawn by the viewer's
    ``add_surface_streaks`` from the stored ``tau``. ``mode="sensitivity"``
    shows dJ/dn relative to its 98th-percentile magnitude on a fixed ±1 scale
    (the map's pattern is reliable, its absolute size is not; see the caption).
    """

    def __init__(self, parent: QWidget | None = None, *, mode: str = "fields") -> None:
        super().__init__(parent)
        self.mode = mode
        self._surfaces: list = []
        self._ground: float | None = None
        self._export_dir = Path.home()
        self._stem = "surface"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.view = _view3d()
        self.view.set_placeholder("No surface data", "")
        self.bar = self.view.toolbar()
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(170)
        self.cmap_combo = _cmap_combo()
        if mode == "fields":
            fit_cmap_combo(self.cmap_combo)
            self.bar.row.insertWidget(0, label("Field", "secondary"))
            self.bar.row.insertWidget(1, self.field_combo)
            self.bar.row.insertSpacing(2, 8)
            self.bar.row.insertWidget(3, label("Colormap", "secondary"))
            self.bar.row.insertWidget(4, self.cmap_combo)
            self.bar.row.insertSpacing(5, 8)
        else:
            self.field_combo.hide()
            self.cmap_combo.hide()
        self.streaks_btn = QToolButton()
        self.streaks_btn.setText("Friction lines")
        self.streaks_btn.setCheckable(True)
        self.streaks_btn.setProperty("variant", "toggle")
        self.streaks_btn.setFixedHeight(24)
        self._streaks_tip = ("Streamlines of the wall shear on the surface, as an oil-flow test shows them: "
                             "where the flow attaches, separates and reattaches")
        self.streaks_btn.setToolTip(self._streaks_tip)
        if mode == "fields":
            self.bar.row.insertWidget(6, self.streaks_btn)
            self.bar.row.insertSpacing(7, 8)
        else:
            self.streaks_btn.hide()
        self.streaks_btn.toggled.connect(lambda *_: self.draw())
        self.save_btn = self.bar.add_button("download", "Save image…", "Save the 3D view as a PNG image")
        self.save_btn.clicked.connect(self._save)
        lay.addWidget(self.bar)
        self.caption = label("", "muted", wrap=True)
        self.caption.setContentsMargins(8, 4, 8, 4)
        self.caption.hide()
        lay.addWidget(self.caption)
        lay.addWidget(self.view, 1)
        self.field_combo.currentIndexChanged.connect(self._on_field)
        self.cmap_combo.currentIndexChanged.connect(lambda *_: self.draw())

    def showEvent(self, event) -> None:
        if self.mode == "fields":
            fit_cmap_combo(self.cmap_combo)          # at the font the shown, styled combo really has
        super().showEvent(event)

    def set_export(self, folder: Path, stem: str) -> None:
        self._export_dir, self._stem = folder, stem

    def set_surfaces(self, surfaces: Sequence, *, ground: float | None = None) -> None:
        self._surfaces = [s for s in surfaces if len(getattr(s, "faces", []))]
        self._ground = ground
        has_tau = any(getattr(s, "tau", None) is not None for s in self._surfaces)
        can = has_tau and hasattr(self.view.scene, "add_surface_streaks")
        self.streaks_btn.setEnabled(can)
        self.streaks_btn.setToolTip(self._streaks_tip if can else
                                    f"{self._streaks_tip}.\nUnavailable: this run stored no wall shear")
        if self.mode == "fields":
            cur = self.field_combo.currentData()
            self.field_combo.blockSignals(True)
            self.field_combo.clear()
            for key, text in SURFACE_FIELDS:
                if any(getattr(s, key, None) is not None for s in self._surfaces):
                    self.field_combo.addItem(text, key)
            i = self.field_combo.findData(cur if cur else "cp")
            self.field_combo.setCurrentIndex(max(i, 0))
            self.field_combo.blockSignals(False)
            _fill_cmap_combo(self.cmap_combo, self.field() or "cp")
        self.view.set_view("iso")
        self.draw()

    def field(self) -> str:
        return "sensitivity" if self.mode == "sensitivity" else (self.field_combo.currentData() or "cp")

    def _on_field(self, *_) -> None:
        _fill_cmap_combo(self.cmap_combo, self.field())
        self.draw()

    def draw(self) -> None:
        sc = self.view.scene
        sc.clear()
        if not self._surfaces:
            self.view.refresh()
            return
        field = self.field()
        if self.mode == "sensitivity":
            vals = [np.asarray(s.sensitivity, float) if s.sensitivity is not None else None for s in self._surfaces]
            allv = np.concatenate([np.abs(v) for v in vals if v is not None]) if any(
                v is not None for v in vals) else np.zeros(0)
            allv = allv[np.isfinite(allv)]
            m = float(np.percentile(allv, 98)) if allv.size else 0.0
            if not m > 0:
                m = float(allv.max()) if allv.size and allv.max() > 0 else 1.0
            self.scale98 = m
            for s, v in zip(self._surfaces, vals):
                if v is None:
                    sc.add_mesh(s)
                else:
                    sc.add_mesh(s, np.clip(v / m, -1.0, 1.0), field="sensitivity", cmap="RdBu_r", range=(-1.0, 1.0))
            sc.add_colorbar("dJ/dn, relative", "")
        else:
            have = [s for s in self._surfaces if getattr(s, field, None) is not None]
            rest = [s for s in self._surfaces if getattr(s, field, None) is None]
            cmap = self.cmap_combo.currentData() or default_cmap3d(field)
            if have:
                vals = np.concatenate([np.asarray(getattr(s, field), float) for s in have])
                rng = surface_cp_range(vals) if field == "cp" else display_range3d(field, vals)
                sc.add_meshes(have, field, cmap=cmap, range=rng)
            for s in rest:
                sc.add_mesh(s)
            sc.add_colorbar()
            if self.streaks_btn.isChecked() and self.streaks_btn.isEnabled():
                try:
                    sc.add_surface_streaks([s for s in self._surfaces if getattr(s, "tau", None) is not None])
                except Exception:
                    log.exception("friction lines failed")
        if self._ground is not None:
            sc.add_ground(float(self._ground))
        self.view.refresh()

    def _save(self) -> None:
        _save_view_png(self, self.view, self._export_dir / f"{self._stem}_{self.field()}.png")


# ---------------------------------------------------------------------------
# fields panel: the flow in the volume (smoke, streamlines, slices, vortex cores) and the slice in 2D
# ---------------------------------------------------------------------------

#: View modes of the flow tabs: (key, button text, tooltip)
VIEW_MODES = (("3d", "3D", "The body and the flow around it in 3D"),
              ("slice", "Slice", "The slice plane as a 2D field view"),
              ("split", "Split", "The 3D view and the slice side by side"))


def view_mode_control(current: str = "3d") -> SegmentedControl:
    seg = SegmentedControl([(k, t) for k, t, _tip in VIEW_MODES], current)
    seg.setToolTip("\n".join(f"{t}: {tip}" for _k, t, tip in VIEW_MODES))
    return seg


class Fields3DPanel(QWidget):
    """The flow of a 3D result in the volume around the body (DESIGN.md 7.13).

    Opens on a 3D view of the body with **streamlines of the mean flow** from a
    movable seed rake and the **smoke** the solver carried (the instantaneous
    dye snapshot stored with the run, when there is one); the **slice plane**,
    the **Q-criterion vortex cores** and the **surface Cp** are toggles in the
    strip above. The same slice is available as a 2D field view (View: Slice
    or Split). The slice comes from the stored mean volume when there is one
    (any normal, any position, cut with ``viz3d.data.slice_from_volume``),
    otherwise from the three stored mean slices (fixed positions). The vortex
    core level is relative to the 99.9th percentile of positive Q, so the
    slider means the same for any speed.
    """

    #: Q level slider: 10^(-2.5 .. 0) times the 99.9th percentile of positive Q
    Q_LOG_RANGE = (-2.5, 0.0)
    #: Width of the property panel on the right [px]; widened when its rows need more (see _fit_side)
    SIDE_W = 288

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from .view3d_panel import DEFAULT_SHOWN_RESULT, FlowScene, FlowToggles, RakeEditor
        self._r: CFD3DResult | None = None
        self._defaults_for: CFD3DResult | None = None   # the result the default layers were chosen for
        self._ground: float | None = None
        self._sim = self._design = None
        self._heat_range: tuple[float, float] | None = None
        self._q_ref = float("nan")
        self._export_dir = Path.home()
        self._stem = "fields"
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)                # slider drags: rebuild once they pause
        self._timer.timeout.connect(self._apply_pending)
        self._pending: set[str] = set()
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        self.view = _view3d(overlay_toolbar=True)
        self.view.set_placeholder("No flow field stored", "")
        self.flow = FlowScene(self.view, shown=dict(DEFAULT_SHOWN_RESULT))
        # a stored run's smoke is a snapshot: the rake drawn here seeds the streamlines only, so it
        # shows with them (drawn with smoke alone it read as the 7-nozzle smoke rake, which it is not)
        self.flow.rake_with_smoke = False
        # the View control and the layer toggles: one row when they fit, else two (bar_host reflows them);
        # seven layer toggles beside View, 3D / Slice / Split and Save image need about 1150 px
        self.bar_host = QWidget()
        self._bar_box = QBoxLayout(QBoxLayout.Direction.LeftToRight, self.bar_host)
        self._bar_box.setContentsMargins(0, 0, 0, 0)
        self._bar_box.setSpacing(0)
        self.bar = ToolStrip()
        self.bar.add_widget(label("View", "secondary"))
        from .view3d_panel import keep_whole
        self.mode = keep_whole(view_mode_control("3d"))
        self.bar.add_widget(self.mode)
        self._bar_sep = self.bar.add_separator()
        self.bar.add_stretch()                 # on its own row, View stays left (not pushed to the far edge)
        self.layer_bar = ToolStrip()
        self.toggles = FlowToggles(("smoke", "streamlines", "slice", "q", "heat", "particles", "surface"),
                                   shown={**DEFAULT_SHOWN_RESULT, "particles": False}, parent=self)
        self.toggles.add_to(self.layer_bar)
        # toggles of layers a run cannot have (a thermal plume without heat transfer) are not shown at
        # all, rather than greyed: a disabled "Thermal plume" on every isothermal result is noise
        self._absent: set[str] = {"heat"}
        self.toggles.set_shown_keys([k for k in self.toggles.buttons if k != "heat"])
        self.layer_bar.add_stretch()
        self.save_btn = self.layer_bar.add_button("download", "Save image…", "Save the 3D view as a PNG image")
        self.save_btn.clicked.connect(lambda: _save_view_png(self, self.view, self._export_dir / f"{self._stem}_3d.png"))
        self._bar_box.addWidget(self.bar)
        self._bar_box.addWidget(self.layer_bar, 1)
        self.bar_host.setMinimumWidth(1)
        self.bar_host.installEventFilter(self)
        ll.addWidget(self.bar_host)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        split.setChildrenCollapsible(False)
        split.addWidget(self.view)
        self.slice_view = FieldView()
        self.slice_view.keep_body = True      # a flat domain slice in a narrow pane: keep the body in view
        self.slice_view.setMinimumWidth(240)
        split.addWidget(self.slice_view)
        split.setSizes([620, 460])
        split.installEventFilter(self)         # Split re-orients with the viewport's shape
        self.split = split
        ll.addWidget(split, 1)
        root.addWidget(left, 1)

        side = self.side = QFrame()
        side.setProperty("panel", "true")
        side.setFixedWidth(self.SIDE_W)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        grid = self.grid = PropertyGrid(label_width=84)
        g = grid.add_group("Field")
        self.field_combo = fixed_hint_combo()
        self.field_combo.setToolTip("Colours the slice, the streamlines and (optionally) the vortex cores")
        g.add_row("Field", self.field_combo)
        self.cmap_combo = _cmap_combo()
        g.add_row("Colormap", self.cmap_combo)
        g = grid.add_group("Slice plane")
        self.axis_seg = SegmentedControl([("x", "x"), ("y", "y"), ("z", "z")], "y")
        self.axis_seg.setToolTip("Normal of the slice plane: y = side view (x–z), z = plan view (x–y), "
                                 "x = cross-flow view (y–z, seen from behind)")
        g.add_row("Normal", self.axis_seg)
        self.pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_slider.setRange(0, 1000)
        # the normal is named by the control above, so the readout is only the distance ("0.027 m")
        self.pos_value = label("", "secondary")
        self.pos_value.setMinimumWidth(52)
        self.pos_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        g.add_row("Position", _row(self.pos_slider, self.pos_value))
        # the smoke channels: colour chip, emitters, on / off (three emitters' smoke mixes into one haze
        # unless each can be told apart and switched off)
        from .view3d_panel import SmokeLegend
        g = self.smoke_group = grid.add_group("Smoke emitters")
        self.smoke_legend = SmokeLegend()
        g.add_widget(self.smoke_legend)
        self.smoke_legend.bind(self.flow)
        # "Streamline seeds", not "Seed rake": next to the smoke emitters the old name read as if it
        # moved the smoke, which a stored run's snapshot cannot do
        g = grid.add_group("Streamline seeds")
        self.rake_editor = RakeEditor(g, parent=self)
        g = grid.add_group("Vortex cores")
        self.q_slider = QSlider(Qt.Orientation.Horizontal)
        self.q_slider.setRange(0, 100)
        self.q_slider.setValue(60)
        self.q_value = label("", "secondary")
        self.q_value.setMinimumWidth(52)
        self.q_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        g.add_row("Level", _row(self.q_slider, self.q_value),
                  tooltip="Q-criterion level of the surface: lower shows weaker vortices")
        self.q_colour = fixed_hint_combo()
        self.q_colour.addItem("Uniform", None)
        self.q_colour.addItem("By field", "field")
        self.q_colour.setToolTip("Uniform grey-blue, or coloured by the field chosen above")
        g.add_row("Colour", self.q_colour)
        # the thermal plume (DESIGN.md 10.2): shown only for results with a temperature field
        g = self.heat_group = grid.add_group("Thermal plume")
        from .view3d_panel import HEAT_STYLES
        self.heat_style = SegmentedControl([(k, t) for k, t, _tip in HEAT_STYLES], "plume")
        self.heat_style.setToolTip("\n".join(f"{t}: {tip}" for _k, t, tip in HEAT_STYLES))
        g.add_row("Show as", self.heat_style)
        self.heat_slider = QSlider(Qt.Orientation.Horizontal)
        self.heat_slider.setRange(0, 100)
        self.heat_slider.setValue(12)
        self.heat_value = label("", "secondary")
        self.heat_value.setMinimumWidth(60)
        self.heat_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        g.add_row("Threshold", _row(self.heat_slider, self.heat_value),
                  tooltip="Plume: fluid cooler than this is clear. Iso-surface: the temperature of the surface. "
                          "Measured from the ambient (left) to the hottest fluid (right).")
        self.heat_group.hide()
        self.note = label("", "muted", wrap=True)
        grid.add_widget(self.note)
        grid.add_stretch()
        self.side_scroll = grid.scrollable()
        # never clip: if the rows ever need more than the panel (a large system font), scroll instead
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sl.addWidget(self.side_scroll)
        root.addWidget(side)
        self._fit_side()
        # aliases kept for callers and tests written against the first version of this panel
        self.stream_on = self.toggles.buttons["streamlines"]
        self.slice_3d = self.toggles.buttons["slice"]
        self.q_on = self.toggles.buttons["q"]
        self.smoke_on = self.toggles.buttons["smoke"]
        self.surface_on = self.toggles.buttons["surface"]
        self.stream_n = self.rake_editor.n

        self.mode.changed.connect(self._on_mode)
        self.toggles.changed.connect(self._on_toggle)
        theme_mod = lazy_import("aerolume.app.theme")
        if theme_mod is not None and hasattr(theme_mod, "changed"):
            # after the new style sheet has re-polished the rows (queued behind it)
            theme_mod.changed.connect(self._on_theme)
        self.field_combo.currentIndexChanged.connect(self._on_field)
        self.cmap_combo.currentIndexChanged.connect(lambda *_: self._request("slice"))
        self.axis_seg.changed.connect(self._on_axis)
        self.pos_slider.valueChanged.connect(self._on_pos)
        self.rake_editor.changed.connect(lambda: self._request("rake"))
        # a rake dragged in the view: the rows follow while it moves; the scene re-traced on release
        self.flow.rakeMoving.connect(self.rake_editor.show_rake)
        self.flow.rakeEdited.connect(self.rake_editor.show_rake)
        self.q_slider.valueChanged.connect(self._on_q)
        self.q_colour.currentIndexChanged.connect(lambda *_: self._request("q"))
        self.heat_style.changed.connect(lambda *_: self._request("heat"))
        self.heat_slider.valueChanged.connect(self._on_heat_level)
        # particle replay through the stored mean velocity (a frozen field: the particles trace its streamlines)
        self._replay = None
        self._replay_timer = QTimer(self)
        self._replay_timer.setInterval(33)
        self._replay_timer.timeout.connect(self._replay_tick)
        self.toggles.buttons["particles"].setToolTip(
            "Replay tracer particles from the run's particle emitters (else its smoke emitters, else the seed "
            "rake) through the time-averaged flow: a frozen field, so their paths are its streamlines")
        self._on_mode("3d")

    def _fit_side(self) -> None:
        """Make the property panel wide enough for its rows plus a vertical scroll bar.

        The rows sit in a scroll area that resizes them to its viewport; when
        their minimum width is larger than the viewport Qt keeps the minimum and
        cuts the right edge off (combos, unit labels and the note were clipped
        after a 1280 -> 1600 resize, when the vertical bar came and went). With
        the bar's width reserved, the rows always fit.
        """
        bar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        need = self.grid.minimumSizeHint().width() + bar + 2
        self.side.setFixedWidth(max(self.SIDE_W, need))

    def _on_theme(self, *_args) -> None:
        QTimer.singleShot(0, self, self._fit_side)     # once the new style sheet has re-polished the rows

    def showEvent(self, event) -> None:
        self._fit_side()                 # fonts and styles are final once the panel is shown
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._stop_replay()              # a replay nobody sees costs a timer and the CPU
        super().hideEvent(event)

    # -- particle replay (DESIGN.md 10.1) ------------------------------------
    #: Particles in flight once the domain is filled, and the screen time a free-stream particle takes
    #: to cross the stored volume [s].
    REPLAY_PARTICLES = 4000
    REPLAY_CROSSING_S = 8.0
    #: Flow-throughs of the stored volume the replay is advanced before its first frame, in steps
    #: :data:`REPLAY_WARM_STRIDE` times the drawn step: the streaklines then already reach past the body
    #: when the picture appears (started empty, the car's tracers were still within 0.3 m of their
    #: emitters, in front of the bumper, when the view was looked at).
    REPLAY_WARM_CROSSINGS = 1.0
    REPLAY_WARM_STRIDE = 4.0

    def _replay_sources(self, extent) -> list:
        """The replay's particle emitters (3D, wind frame): the run's particle emitters, else its smoke
        emitters released as particles, else one nozzle per seed of the rake (at most 40)."""
        from ...model import Emitter
        sim = self._sim
        em = [e for e in (getattr(sim, "emitters", None) or []) if getattr(e, "enabled", True)]
        parts = [e for e in em if e.species == "particles"]
        if not parts and em:
            parts = [Emitter(name=e.name, species="particles", kind=e.kind, position=e.position, end=e.end,
                             size=e.size, normal=e.normal, count=e.count, radius=e.radius, color=e.color,
                             frame=e.frame) for e in em]
        if not parts:
            rake = self.rake_editor.rake()
            seeds = rake.seeds() if rake is not None else np.zeros((0, 3))
            seeds = seeds[:: max(1, len(seeds) // 40)]
            parts = [Emitter(name=f"Rake {i + 1}", species="particles", kind="point",
                             position=tuple(float(c) for c in pt)) for i, pt in enumerate(seeds)]
        placed = lazy_import("aerolume.viz3d.placed")
        if placed is not None and sim is not None:
            # body-frame emitters turn with the geometry, as the solver placed them
            rot = placed.attitude_matrix(float(sim.conditions.alpha_deg), float(sim.conditions.yaw_deg))
            ref = np.asarray(getattr(self._design, "moment_ref3", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0), float)
            for e in parts:
                if e.frame == "body":
                    e.position = tuple(float(c) for c in (np.asarray(e.position[:3], float) - ref) @ rot.T + ref)
                    if e.end is not None:
                        e.end = tuple(float(c) for c in (np.asarray(e.end[:3], float) - ref) @ rot.T + ref)
                    e.frame = "wind"
        return parts

    def _start_replay(self) -> None:
        """Build the replay (torch is imported on a worker thread the first time) and start its timer."""
        r = self._r
        from .view3d_panel import has_velocity
        if r is None or not has_velocity(r.volume):
            self.toggles.set_on("particles", False)
            return
        data = lazy_import("aerolume.viz3d.data")
        origin, spacing, fields, mask = data.full_volume(r.volume)
        nz, ny, nx = np.shape(fields["u"])
        ext = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1],
               origin[2], origin[2] + nz * spacing[2])
        u = np.asarray(fields["u"], np.float32)
        speed = float(np.nanpercentile(np.abs(u[np.isfinite(u)]), 99)) if np.isfinite(u).any() else 1.0
        speed = speed or 1.0
        span = ext[1] - ext[0]
        sources = self._replay_sources(ext)
        life = 1.5 * span / speed
        for e in sources:
            e.rate = self.REPLAY_PARTICLES / life / max(len(sources), 1)
        replay_fields = (u, np.asarray(fields["v"], np.float32), np.asarray(fields["w"], np.float32),
                         None if mask is None else np.asarray(mask, np.float32))
        tick_dt = span / speed / (self.REPLAY_CROSSING_S * 30.0)
        warm = (self.REPLAY_WARM_CROSSINGS, self.REPLAY_WARM_STRIDE)

        def build():
            from ...viz.particles import EmitterParticles
            p = EmitterParticles(ext, sources, n_max=2 * self.REPLAY_PARTICLES, lifetime=life, trail=6,
                                 cell=float(min(spacing)))
            # advance the replay one flow-through before it is shown (on this worker thread), so the first
            # frame already has streaklines past the body: the lifetime (1.5 crossings) lets them get there
            uu, vv, ww, mm = replay_fields
            dt = tick_dt * warm[1]
            for _ in range(math.ceil(warm[0] * span / speed / dt)):
                p.step(uu, vv, ww, dt=dt, mask=mm, substeps=p.suggest_substeps(dt, speed, p.cell))
            return p

        def ready(p) -> None:
            if not self.toggles.wanted("particles") or self._r is not r:
                return
            from ...viz.colormaps import hex_rgb
            self._replay = p
            self._replay_fields = replay_fields
            self._replay_dt = tick_dt
            self._replay_speed = speed
            self._replay_palette = np.array([hex_rgb(e.color or "#f5f5f5") for e in sources], dtype=np.uint8)
            self._replay_timer.start()

        from ..jobs import run_in_background
        run_in_background(build, on_done=ready,
                          on_error=lambda m: (self.toggles.set_on("particles", False),
                                              log.warning("3D particle replay unavailable: %s", m)))

    def _replay_tick(self) -> None:
        p = self._replay
        if p is None or not self.isVisible():
            return
        u, v, w, m = self._replay_fields
        n_sub = p.suggest_substeps(self._replay_dt, self._replay_speed, p.cell)
        p.step(u, v, w, dt=self._replay_dt, mask=m, substeps=n_sub)
        ids = p.emitter_ids()
        cols = self._replay_palette[np.clip(ids, 0, len(self._replay_palette) - 1)] if len(ids) else None
        # an empty pool keeps the layer (and its actors): the next tick fills it again
        self.flow.set_particles(p.positions(), colors=cols, trails=p.trails())

    def _stop_replay(self) -> None:
        self._replay_timer.stop()
        self._replay = None
        if getattr(self, "flow", None) is not None and self.flow.particles is not None:
            self.flow.set_particles(None)

    # -- data -------------------------------------------------------------
    def set_export(self, folder: Path, stem: str) -> None:
        self._export_dir, self._stem = folder, stem

    def set_result(self, r: CFD3DResult, *, ground: float | None = None, rake: dict | None = None,
                   sim=None, design=None) -> None:
        """Show ``r``. ``rake`` is the rake dict its run released its smoke from (the simulation's rake,
        ``JobManager.seed_rake_for``); None: the solver's default rake. Streamlines start there too, so
        they and the stored smoke come from the one rake drawn. ``sim`` / ``design`` (when given) colour
        the smoke channels by their emitters and draw the run's fans, heaters, emitters and jets."""
        from .view3d_panel import (
            RakeSpec,
            dye_colours_of,
            has_velocity,
            heat_volume_of,
            smoke_volume_of,
            volume_box,
        )
        self._stop_replay()
        self.toggles.set_on("particles", False)
        self._r = r
        self._ground = ground
        self._sim, self._design = sim, design
        vol = r.volume
        names = set(vol.fields) if vol is not None else set()
        for s in r.slices:
            names |= set(s.fields)
        if has_velocity(vol):
            names.add("speed")
        cur = self.field_combo.currentData()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for n in SLICE_FIELDS:
            if n in names:
                self.field_combo.addItem(field_info3d(n)[0], n)
        i = self.field_combo.findData(cur if cur else "speed")
        self.field_combo.setCurrentIndex(max(i, 0))
        self.field_combo.blockSignals(False)
        _fill_cmap_combo(self.cmap_combo, self.field())
        self._q_ref = float("nan")
        if vol is not None and "q" in vol.fields:
            qv = np.asarray(vol.fields["q"], dtype=float)
            pos = qv[np.isfinite(qv) & (qv > 0)]
            self._q_ref = float(np.percentile(pos, 99.9)) if pos.size else float("nan")
        self.axis_seg.blockSignals(True)
        self.axis_seg.set_current("y")
        self.axis_seg.blockSignals(False)
        self._reset_position()
        self._update_q_label()
        flow = self.flow
        from .view3d_panel import bounds_of, flow_focus_box
        room = closed_room_box(sim)
        # a room is framed by its walls (fans, heaters, vents and smoke fill it), not the block inside
        flow.set_bodies(r.surface, ground=ground, key=("result", id(r)),
                        focus=room or flow_focus_box(bounds_of(r.surface), ground=ground))
        flow.set_room(room, rebuild=False)
        flow.set_volume(vol if has_velocity(vol) or (vol is not None and "q" in vol.fields) else None, rebuild=False)
        smoke = smoke_volume_of(r)
        flow.set_smoke(smoke, rebuild=False)
        flow.set_dye_colors(dye_colours_of(sim), rebuild=False)
        from .view3d_panel import smoke_channel_labels
        flow.set_channel_names(smoke_channel_labels(sim))
        flow.hidden_channels.clear()           # a new result shows all its smoke
        heat = heat_volume_of(r)
        self._heat_range = None
        if heat is not None:
            from .view3d_panel import heat_display_range, temperature_field_of
            vals = np.asarray(heat.fields[temperature_field_of(heat)], dtype=float)
            if heat.mask is not None:
                vals = vals[np.asarray(heat.mask) <= 0.5]
            lo, hi, _mode = heat_display_range(vals)
            self._heat_range = (lo, hi)
        flow.set_heat(heat, style=self.heat_style.current(), level=self._heat_level(), rng=self._heat_range,
                      rebuild=False)
        self.heat_group.setVisible(heat is not None)
        self._set_absent("heat", heat is None)
        self._update_heat_label()
        flow.set_devices(self._glyphs(), rebuild=False)
        if r is not self._defaults_for:
            # a new result opens on its smoke with thin speed-coloured mean-flow streamlines over it
            # (DESIGN 7.13); one without a smoke snapshot on its streamlines alone (shaded tubes).
            # Smoke alone showed the streamline seed rake as if it were the smoke source. Later
            # toggles are the user's.
            self._defaults_for = r
            self.toggles.set_on("smoke", smoke is not None)
            self.toggles.set_on("streamlines", has_velocity(vol))
            if room is not None and heat is not None and self.field_combo.findData("temperature") >= 0:
                # a room has no plume against still ambient air: every cell is warmer or cooler than the
                # ambient, and the plume volume filled the box with near-black (light theme). Its
                # temperature reads on a slice through the room instead.
                self.toggles.set_on("heat", False)
                self.field_combo.setCurrentIndex(self.field_combo.findData("temperature"))
                self.toggles.set_on("slice", True)
            else:
                self.toggles.set_on("heat", heat is not None)   # a heat-transfer run is about its plume
                self.toggles.set_on("slice", False)             # (a room before it left its slice on)
        box = volume_box(vol)
        bb = flow.body_bounds()
        rake = RakeSpec.resolve(rake, bb, domain=box) if bb is not None and has_velocity(vol) else None
        self.rake_editor.set_rake(rake, default=rake, box=box)
        flow.set_rake(rake, rebuild=False)
        flow.set_q(self._q_level() if _finite(self._q_ref) else None,
                   color_by_slice=self.q_colour.currentData() == "field", rebuild=False)
        self._update_slice(rebuild=False)
        self._sync_toggles()
        flow.rebuild()
        self.smoke_legend.refresh()
        self.smoke_group.setVisible(smoke is not None and bool(self.smoke_legend.rows()))
        self._update_note(smoke)
        self.slice_view.clear()
        self._draw_slice2d()
        self.view.set_view("iso")
        self._fit_side()

    def _sync_toggles(self) -> None:
        """Enable the toggles that have data, each disabled one saying why."""
        reasons = {"smoke": "this run stored no smoke (dye) snapshot",
                   "streamlines": "this run stored no 3D velocity volume",
                   "q": "this run stored no Q-criterion volume",
                   "slice": "no slice is stored for this normal",
                   "surface": "this run stored no surface pressure",
                   "heat": "this run stored no temperature field",
                   "particles": "this run stored no 3D velocity volume to carry them"}
        for key in self.toggles.buttons:
            ok, _why = self.flow.available(key)
            if key == "particles":                   # a replay needs only the stored velocity
                from .view3d_panel import has_velocity
                ok = self._r is not None and has_velocity(self._r.volume)
            self.toggles.set_available(key, ok, reasons.get(key, ""))
            self.flow.shown[key] = self.toggles.wanted(key)

    def _set_absent(self, key: str, absent: bool) -> None:
        """Hide (absent) or show the toggle of a layer the result cannot / can have."""
        if absent:
            self._absent.add(key)
        else:
            self._absent.discard(key)
        self._sync_layer_buttons()

    def _sync_layer_buttons(self) -> None:
        """The layer toggles the result offers in this view mode (FlowToggles hides the rest, and moves
        those that do not fit the row into its "⋯" menu)."""
        on3d = self.mode.current() != "slice"
        self.toggles.set_shown_keys([k for k in self.toggles.buttons if on3d and k not in self._absent])

    def _glyphs(self) -> list:
        """The run's fans, heaters, emitters and jets as 3D glyphs (none without the simulation)."""
        if self._sim is None:
            return []
        placed = lazy_import("aerolume.viz3d.placed")
        if placed is None:
            return []
        from .view3d_panel import bounds_of
        b = bounds_of(self._r.surface) if self._r is not None else None
        scale = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) if b is not None else 1.0
        try:
            return placed.placed_glyphs(self._sim, self._design, scale=scale)
        except Exception:
            log.exception("building the device glyphs failed")
            return []

    def _heat_level(self) -> float | None:
        """The plume threshold the slider stands for (field units), None without a temperature range."""
        rng = getattr(self, "_heat_range", None)
        if rng is None:
            return None
        return rng[0] + (rng[1] - rng[0]) * self.heat_slider.value() / 100.0

    def _update_heat_label(self) -> None:
        t = self._heat_level()
        from .view3d_panel import heat_volume_of, temperature_field_of
        f = temperature_field_of(heat_volume_of(self._r)) if self._r is not None else None
        unit = " °C" if f == "temperature" else ""
        self.heat_value.setText(f"{sig(t, 3)}{unit}".replace("-", MINUS) if _finite(t) else "—")

    def _on_heat_level(self, *_) -> None:
        self._update_heat_label()
        self._request("heat")

    def _update_note(self, smoke) -> None:
        r = self._r
        lines = []
        if r is not None and r.volume is None:
            lines.append("This result stores three mean slices but no volume: slices sit at fixed positions and "
                         "there are no streamlines or vortex cores.")
        if smoke is not None:
            t = getattr(smoke, "time", None) or getattr(smoke, "t", None)
            when = f" at t = {sig(float(t), 3)} L/U" if _finite(t) else " at the end of the run"
            lines.append(f"Smoke: the instantaneous dye{when}, released by the rake the run used. Streamlines are "
                         "of the time-averaged flow.")
        elif r is not None and r.volume is not None:
            lines.append("No smoke snapshot is stored with this run; the streamlines show the time-averaged flow "
                         "from the rake.")
        from .view3d_panel import heat_volume_of
        heat = heat_volume_of(r) if r is not None else None
        if heat is not None:
            what = "time-averaged" if getattr(heat, "label", "") == "mean" else "final"
            lines.append(f"Thermal plume: the {what} temperature, clear below the threshold. The colour range runs "
                         "from the coolest to the hottest fluid around the ambient; Heat transfer lists the "
                         "Nusselt numbers.")
        self.note.setText("\n\n".join(lines))
        self.note.setVisible(bool(lines))

    def field(self) -> str:
        return self.field_combo.currentData() or "speed"

    def axis(self) -> str:
        return self.axis_seg.current()

    def _axis_range(self) -> tuple[float, float]:
        r = self._r
        if r is not None and r.volume is not None:
            b = volume_bounds(r.volume)
        elif r is not None and r.slices:
            s = next((s for s in r.slices if s.axis == self.axis()), r.slices[0])
            return float(s.position), float(s.position)
        else:
            return 0.0, 1.0
        k = _AXIS[self.axis()]
        return float(b[2 * k]), float(b[2 * k + 1])

    def _default_position(self) -> float:
        """Through the body: the stored slice's position (y = 0, mid-height, behind the body)."""
        r = self._r
        for s in (r.slices if r is not None else []):
            if s.axis == self.axis():
                return float(s.position)
        lo, hi = self._axis_range()
        return 0.5 * (lo + hi)

    def _reset_position(self) -> None:
        lo, hi = self._axis_range()
        pos = self._default_position()
        self.pos_slider.blockSignals(True)
        self.pos_slider.setValue(round(1000 * (pos - lo) / (hi - lo)) if hi > lo else 0)
        self.pos_slider.setEnabled(hi > lo and self._r is not None and self._r.volume is not None)
        self.pos_slider.blockSignals(False)
        self._pos_label()

    def position(self) -> float:
        lo, hi = self._axis_range()
        return lo + (hi - lo) * self.pos_slider.value() / 1000.0 if hi > lo else lo

    def _pos_label(self) -> None:
        pos = self.position()
        self.pos_value.setText(f"{pos:.3f} m".replace("-", MINUS) if _finite(pos) else "—")
        self.pos_value.setToolTip(f"Slice plane at {self.axis()} = {sig(pos, 4)} m".replace("-", MINUS))

    def current_slice(self):
        """The SliceData shown: cut from the volume at the slider position, else the stored one."""
        r = self._r
        if r is None:
            return None
        field = self.field()
        if r.volume is not None and (field in r.volume.fields or field == "speed"):
            data = lazy_import("aerolume.viz3d.data")
            if data is not None:
                try:
                    return data.slice_from_volume(r.volume, self.axis(), self.position(), field)
                except (KeyError, ValueError):
                    log.debug("slice from the volume failed", exc_info=True)
        return next((s for s in r.slices if s.axis == self.axis() and field in s.fields), None)

    # -- events -----------------------------------------------------------
    def _on_mode(self, key: str) -> None:
        self.view.setVisible(key in ("3d", "split"))
        self.slice_view.setVisible(key in ("slice", "split"))
        self._sync_layer_buttons()
        self.save_btn.setVisible(key != "slice")
        self.layer_bar.setVisible(key != "slice")          # no layers on a flat slice: no empty second row
        if key == "split":
            self._orient_split(force=True)

    def eventFilter(self, obj, event) -> bool:
        if obj is getattr(self, "split", None) and event.type() == QEvent.Type.Resize:
            self._orient_split()
        elif obj is getattr(self, "bar_host", None) and event.type() in (QEvent.Type.Resize, QEvent.Type.Show,
                                                                         QEvent.Type.LayoutRequest):
            self._reflow_bar()
        return super().eventFilter(obj, event)

    def _reflow_bar(self) -> None:
        """View and the layer toggles in one row when both fit at their natural width, else in two."""
        a = self.bar.minimumSizeHint().width()
        if self.bar_host.width() > 1:
            # what does not fit beside "Save image…" goes into the toggles' "⋯" menu (it read "Surfa" at 1280)
            save = self.save_btn.sizeHint().width() if not self.save_btn.isHidden() else 0
            m = self.layer_bar.layout().contentsMargins() if self.layer_bar.layout() is not None else None
            pad = (m.left() + m.right()) if m is not None else 16
            self.toggles.overflow_to(self.bar_host.width() - save - pad - 16)
        b = self.layer_bar.minimumSizeHint().width()
        self.bar_host.setMinimumWidth(max(a, b, 1))
        one_row = self.bar_host.width() >= a + b
        want = QBoxLayout.Direction.LeftToRight if one_row else QBoxLayout.Direction.TopToBottom
        if self._bar_box.direction() != want:
            self._bar_box.setDirection(want)
        self._bar_sep.setVisible(one_row)

    def bar_rows(self) -> int:
        """1 or 2: how many rows the View control and the layer toggles take now."""
        return 1 if self._bar_box.direction() == QBoxLayout.Direction.LeftToRight else 2

    def _orient_split(self, *, force: bool = False) -> None:
        """Split: the slice beside the 3D view, or under it when its shape fits that better.

        A car's side slice (about 3:1) beside the 3D view filled a 90-140 px
        band of a portrait pane; :func:`split_orientation` stacks it under the
        3D view instead, and keeps a cross-flow slice (about 1:1) beside it.
        """
        if self.mode.current() != "split":
            return
        w, h = max(self.split.width(), 2), max(self.split.height(), 2)
        want = split_orientation(w, h, self.slice_view.framing_aspect())
        if want != self.split.orientation() or force:
            self.split.setOrientation(want)
            n = h if want == Qt.Orientation.Vertical else w
            self.split.setSizes([int(0.56 * n), n - int(0.56 * n)])

    def _on_toggle(self, key: str, on: bool) -> None:
        if key == "particles":
            self.flow.shown["particles"] = bool(on)
            if on:
                self._start_replay()
            else:
                self._stop_replay()
            return
        self.flow.show(key, on)

    def _on_field(self, *_) -> None:
        _fill_cmap_combo(self.cmap_combo, self.field())
        self._request("slice")

    def _on_axis(self, *_) -> None:
        self._reset_position()
        self._request("slice")

    def _on_pos(self, *_) -> None:
        self._pos_label()
        self._request("slice")

    def _on_q(self, *_) -> None:
        self._update_q_label()
        self._request("q")

    def _q_level(self) -> float:
        lo, hi = self.Q_LOG_RANGE
        return self._q_ref * 10.0 ** (lo + (hi - lo) * self.q_slider.value() / 100.0)

    def _update_q_label(self) -> None:
        q = self._q_level()
        self.q_value.setText(f"{sig(q, 2)} s⁻²" if _finite(q) else "—")
        self.q_value.setToolTip(f"Q = {sig(q, 3)} 1/s²" if _finite(q) else "No Q-criterion stored")

    def _request(self, what: str) -> None:
        self._pending.add(what)
        if what == "slice":
            self._draw_slice2d()             # the 2D view is cheap: follow the slider at once
        self._timer.start()

    def _apply_pending(self) -> None:
        todo, self._pending = self._pending, set()
        if self._r is None:
            return
        flow = self.flow
        if "slice" in todo:
            self._update_slice(rebuild=False)
        if "rake" in todo:
            flow.set_rake(self.rake_editor.rake(), rebuild=False)
        if "q" in todo or "slice" in todo:
            flow.set_q(self._q_level() if _finite(self._q_ref) else None,
                       color_by_slice=self.q_colour.currentData() == "field", rebuild=False)
        if "heat" in todo:
            flow.set_heat(flow.heat, style=self.heat_style.current(), level=self._heat_level(),
                          rng=getattr(self, "_heat_range", None), rebuild=False)
        layers = set()
        if "slice" in todo:
            layers |= {"slice", "streamlines", "q"}
        if "rake" in todo:
            layers |= {"streamlines", "rake"}
        if "q" in todo:
            layers.add("q")
        if "heat" in todo:
            layers.add("heat")
        flow.rebuild(*sorted(layers))
        self._sync_toggles()

    # -- drawing ----------------------------------------------------------
    def _range_and_cmap(self, sd) -> tuple[tuple[float, float], str]:
        field = self.field()
        cmap = self.cmap_combo.currentData() or default_cmap3d(field)
        vals = np.asarray(sd.fields[field], dtype=float)
        win = None
        room = closed_room_box(self._sim)
        # a room: its colours over the whole room (the cabinet's surroundings held 20.000-20.015 degC)
        fb = room or (focus_box(self._r.surface, ground=self._ground) if self._r is not None else None)
        if fb is not None:
            ci, ri = _SLICE_AXES[sd.axis]
            win = (fb[2 * ci], fb[2 * ci + 1], fb[2 * ri], fb[2 * ri + 1])
        rng = display_range3d(field, vals, extent=sd.extent, mask=sd.mask, window=win)
        return rng, cmap

    def _update_slice(self, *, rebuild: bool = True) -> None:
        """Hand the current slice (cropped to the near field for the 3D view) and its colours to the scene."""
        sd = self.current_slice()
        field = self.field()
        if sd is None or field not in sd.fields:
            self.flow.set_slice(None, field=field, rebuild=rebuild)
            return
        (vmin, vmax), cmap = self._range_and_cmap(sd)
        # the plane covers the framed region (body and near wake) and a margin, not the domain
        from .view3d_panel import bounds_of, near_field_box, plane_window
        room = closed_room_box(self._sim)
        crop = plane_window(room or near_field_box(bounds_of(self._r.surface), ground=self._ground), sd.axis)
        self.flow.set_slice(crop_slice(sd, crop), field=field, cmap=cmap, rng=(vmin, vmax), rebuild=rebuild)

    def _draw_slice2d(self) -> None:
        sd = self.current_slice()
        fv = self.slice_view
        if sd is None:
            fv.clear()
            fv.set_placeholder("No slice stored for this normal", "")
            return
        field = self.field()
        (vmin, vmax), cmap = self._range_and_cmap(sd)
        lab, unit, _div, _c = field_info3d(field)
        sections = mesh_sections(self._r.surface, sd.axis, float(sd.position)) if self._r is not None else []
        first = not fv.has_image()
        fv.set_axis_names({"y": ("x", "z"), "z": ("x", "y"), "x": ("y", "z")}[sd.axis])
        fv.set_sections(sections)
        fv.set_ground(self._ground if sd.axis in ("y", "x") else None)
        fv.set_field(np.asarray(sd.fields[field], dtype=np.float32), sd.extent, field=field, cmap=cmap,
                     vmin=vmin, vmax=vmax, mask=sd.mask, label=lab, units=unit)
        fv.set_caption(f"{sd.axis} = {sig(float(sd.position), 3)} m · time-averaged".replace("-", MINUS))
        if first:
            fv.fit()
        self._orient_split()                  # another normal can have another shape

    def handles(self) -> dict[str, str]:
        """First layer handle of each flow layer (for tests): "slice", "stream", "iso", "smoke", "rake", "body"."""
        names = {"streamlines": "stream", "q": "iso"}
        return {names.get(k, k): v[0] for k, v in self.flow.layer_handles().items() if v}


def _row(w: QWidget, value: QWidget) -> QWidget:
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addWidget(w, 1)
    row.addWidget(value)
    return host


# ---------------------------------------------------------------------------
# data tab (shared by both views)
# ---------------------------------------------------------------------------

class _DataTab(QWidget):
    """Every scalar of a result plus its export buttons (run on a worker thread)."""

    exported = Signal(str)

    def __init__(self, exports: Sequence[tuple[str, str, str, str]], parent: QWidget | None = None) -> None:
        """``exports``: (key, button text, file suffix, file-dialog filter)."""
        super().__init__(parent)
        self._exports = {k: (text, suffix, filt) for k, text, suffix, filt in exports}
        self._export_dir = Path.home()
        self._stem = "result"
        self._make = None
        self.log = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(12)
        self.table = DataTable(["Quantity", "Value", "Unit"])
        lay.addWidget(self.table, 1)
        side = QWidget()
        side.setFixedWidth(260)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(label("Export", "h3"))
        self.buttons: dict[str, QPushButton] = {}
        for key, (text, _suffix, _filt) in self._exports.items():
            b = button(text, None, icon="export")
            b.setStyleSheet("text-align: left; padding-left: 8px;")
            b.clicked.connect(lambda *_, k=key: self.export(k))
            sl.addWidget(b)
            self.buttons[key] = b
        sl.addSpacing(4)
        self.status = label("", "muted", wrap=True)
        sl.addWidget(self.status)
        sl.addStretch(1)
        lay.addWidget(side)

    def set_rows(self, rows: Sequence[tuple[str, str, str]]) -> None:
        self.table.set_data(["Quantity", "Value", "Unit"], [list(r) for r in rows], numeric=[1])
        self.status.setText("")

    def configure(self, folder: Path, stem: str, make, log_fn=None) -> None:
        """``make(key, path)`` returns the zero-argument export call for ``key`` (None: unavailable)."""
        self._export_dir, self._stem, self._make, self.log = folder, stem, make, log_fn

    def export(self, key: str) -> None:
        from ..jobs import run_in_background
        _text, suffix, filt = self._exports[key]
        default = self._export_dir / f"{self._stem}_{key.split('_')[0]}{suffix}"
        path, _ = QFileDialog.getSaveFileName(self, "Export", str(default), filt)
        if not path or self._make is None:
            return
        try:
            call = self._make(key, path)
        except Exception as exc:  # noqa: BLE001 - a missing piece of the result becomes a message
            self.status.setText(f"Export unavailable: {exc}")
            return
        if call is None:
            self.status.setText("This result has no data for that export.")
            return
        self.status.setText("Exporting…")
        run_in_background(call, on_done=lambda *_: self._done(path), on_error=self._failed)

    def _done(self, path: str) -> None:
        self.status.setText(f"Saved {Path(path).name}")
        self.exported.emit(path)
        if self.log is not None:
            self.log(f"Exported {path}", "info")

    def _failed(self, message: str) -> None:
        self.status.setText(f"Export failed: {message}")
        if self.log is not None:
            self.log(f"Export failed: {message}", "error")


def _tab_page(margins: int = 8) -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(margins, margins, margins, margins)
    v.setSpacing(8)
    return w, v


#: Width of each numeric column of a compact per-part / per-body table [px]
PART_COL_W = 90


def scroll_page(content: QWidget) -> QScrollArea:
    """``content`` in a frameless scroll area on the page colour.

    Summary tabs hold tables fixed to the height of their rows. In a layout
    that is shorter than their sum (1280 x 800, or 1600 x 1000 with the
    Assistant or Script dock open) Qt cannot shrink a fixed-height widget, so
    it lays the next ones over it: the caption and the per-part table were
    painted across the Drag ... Reynolds rows. Inside a scroll area the page
    keeps its natural height and scrolls instead.
    """
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.Shape.NoFrame)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    sc.setWidget(content)
    return sc


def compact_columns(table: DataTable, numeric: Sequence[int], width: int = PART_COL_W) -> None:
    """Right-aligned numeric columns of a fixed width, the table no wider than its columns.

    A small table with a stretched last column put C_D at the left edge and C_L
    at the far right with ~900 px of nothing between them; read side by side,
    the numbers belong next to each other.
    """
    hh = table.horizontalHeader()
    hh.setStretchLastSection(False)
    for j in numeric:
        if j < table.columnCount():
            hh.setSectionResizeMode(j, QHeaderView.ResizeMode.Fixed)
            # at least the widest cell: a bold residual "−2.40×10⁻⁴" was elided to "−2.40×10…" at the fixed width
            hh.resizeSection(j, max(width, table.sizeHintForColumn(j) + 4, hh.sectionSizeHint(j)))
    table.resizeColumnToContents(0)
    total = sum(table.columnWidth(j) for j in range(table.columnCount())) + 2 * table.frameWidth() + 2
    table.setMaximumWidth(total)


def fit_table_width(table: DataTable) -> int:
    """Give a results table the width of its columns as its minimum, and return it.

    A narrow page (1280 x 800 with a dock open) otherwise squeezed the
    Summary table until it scrolled inside itself: a horizontal bar covered
    its last row and the Max column was out of sight. With this minimum the
    table is always whole; the page around it reflows or scrolls instead.
    """
    hh = table.horizontalHeader()
    width = sum(max(table.sizeHintForColumn(j), hh.sectionSizeHint(j)) for j in range(table.columnCount()))
    width += 2 * table.frameWidth() + 4
    table.setMinimumWidth(width)
    # whole by construction (fit_height and this minimum): a scroll bar could only appear by a pixel
    # of rounding, and one bar's width or height then brings the other one with it
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return width


class SummaryReflow(QObject):
    """Conditions beside the Summary table when the page is wide enough, below it otherwise.

    Watches the scroll area's viewport; ``apply`` switches the row's direction
    (QBoxLayout) so neither the table nor the conditions are ever cut or
    pushed out of sight sideways.
    """

    def __init__(self, scroll: QScrollArea, row, table: DataTable, side: QWidget) -> None:
        super().__init__(scroll)
        self.scroll, self.row, self.table, self.side = scroll, row, table, side
        scroll.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Resize:
            self.apply()
        return False

    def apply(self) -> None:
        m = self.row.contentsMargins()
        need = (self.table.minimumWidth() + self.row.spacing() + self.side.width() + m.left() + m.right())
        wide = self.scroll.viewport().width() >= need
        want = self.row.Direction.LeftToRight if wide else self.row.Direction.TopToBottom
        if self.row.direction() != want:
            self.row.setDirection(want)


def summary_layout(table: DataTable, column, side: QWidget) -> tuple[QWidget, SummaryReflow]:
    """The Summary tab's page: ``column`` (the table and what follows it) and ``side`` (Conditions) in a
    scroll area (:func:`scroll_page`) that reflows with :class:`SummaryReflow`."""
    w = QWidget()
    outer = QVBoxLayout(w)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    content = QWidget()
    row = QHBoxLayout(content)
    row.setContentsMargins(8, 8, 8, 8)
    row.setSpacing(12)
    row.addLayout(column, 1)
    row.addWidget(side, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    sc = scroll_page(content)
    outer.addWidget(sc, 1)
    return w, SummaryReflow(sc, row, table, side)


def _summary_tab(headers: Sequence[str]) -> tuple[QWidget, DataTable, QWidget, DataTable, QWidget, KeyValueList]:
    """Summary layout shared with the 2D page: table (+ note, + per-part table) beside the conditions,
    all in one scroll area (:func:`scroll_page`)."""
    col = QVBoxLayout()
    col.setSpacing(8)
    table = DataTable(list(headers))
    col.addWidget(table)
    note = label("", "muted", wrap=True)
    col.addWidget(note)
    parts_caption = label("Per part", "h3")
    parts = DataTable(["Part", "CD", "CL"], stretch_last=False)
    col.addWidget(parts_caption)
    col.addWidget(parts)
    col.addStretch(1)
    side = QWidget()
    side.setFixedWidth(300)
    sl = QVBoxLayout(side)
    sl.setContentsMargins(0, 0, 0, 0)
    sl.setSpacing(6)
    sl.addWidget(label("Conditions", "h3"))
    cond = KeyValueList()
    sl.addWidget(cond)
    sl.addStretch(1)
    w, reflow = summary_layout(table, col, side)
    w.reflow = reflow
    return w, table, note, parts, parts_caption, cond


# ---------------------------------------------------------------------------
# 3D CFD result view
# ---------------------------------------------------------------------------

def _results_page():
    """``aerolume.app.pages.results`` (its heat, fan and diagnostics readers), imported on first use: the
    page imports this module, so a module-level import would be circular."""
    return lazy_import("aerolume.app.pages.results")


class Heat3DPanel(QWidget):
    """The Heat transfer tab of a 3D result (DESIGN.md 10.2).

    Left, the numbers: the mean Nusselt number and the heat rate with their
    standard errors, the per-body Nusselt numbers and heat rates with each
    body's wall condition, and the energy balance (residual toned when it is
    above 0.5 % / 2 % of the throughput) -- the same readers as the 2D tab
    (:mod:`aerolume.app.pages.results`). Right, where the heat goes: a
    temperature slice through the body (any normal, any position) in the
    thermal map over the plume's colour range
    (:func:`~aerolume.app.widgets.view3d_panel.heat_display_range`, so the
    slice and the 3D plume share their colours), with the body's section.
    "Show the plume in 3D" (``plume_requested``) opens the Fields tab with the
    thermal plume on.
    """

    plume_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._r: CFD3DResult | None = None
        self._ground: float | None = None
        self._range: tuple[float, float] | None = None
        self._hot = None                          # the hottest heater of a room (the slice frames it)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = self.split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        split.setChildrenCollapsible(False)
        left = self.left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)
        self.title = label("Nusselt number and heat rate", "h3")
        ll.addWidget(self.title)
        self.table = DataTable(SUMMARY_HEADERS)
        ll.addWidget(self.table)
        self.note = label("", "muted", wrap=True)
        ll.addWidget(self.note)
        self.body_caption = label("Per body", "h3")
        ll.addWidget(self.body_caption)
        self.body_table = DataTable(["Body", "Wall", "Nu", "± Std. error", "Q", "Unit"], stretch_last=False)
        ll.addWidget(self.body_table)
        ll.addWidget(label("Heat balance", "h3"))
        self.balance_table = DataTable(["Term", "Rate", "Unit"], stretch_last=False)
        ll.addWidget(self.balance_table)
        self.balance_note = label("", "muted", wrap=True)
        ll.addWidget(self.balance_note)
        self.plume_btn = button("Show the plume in 3D", "link")
        self.plume_btn.setToolTip("Open Fields with the thermal plume shown around the body")
        self.plume_btn.clicked.connect(self.plume_requested.emit)
        ll.addWidget(self.plume_btn, 0, Qt.AlignmentFlag.AlignLeft)
        ll.addStretch(1)
        left.setMinimumWidth(420)
        split.addWidget(scroll_page(left))
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        bar = self.bar = ToolStrip()
        bar.add_widget(label("Slice", "secondary"))
        self.axis_seg = SegmentedControl([("y", "Side"), ("z", "Plan"), ("x", "Cross-flow")], "y")
        self.axis_seg.setToolTip("Normal of the slice: side view (y), plan view (z), cross-flow view (x)")
        bar.add_widget(self.axis_seg)
        bar.row.addSpacing(8)
        self.pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_slider.setRange(0, 1000)
        self.pos_slider.setMaximumWidth(220)
        self.pos_slider.setToolTip("Where the slice cuts the flow, in metres along its normal")
        bar.add_widget(self.pos_slider)
        self.pos_value = label("", "secondary")
        self.pos_value.setMinimumWidth(56)
        bar.add_widget(self.pos_value)
        bar.add_stretch()
        rl.addWidget(bar)
        self.view = FieldView()
        self.view.keep_body = True
        self.view.setMinimumSize(360, 240)
        rl.addWidget(self.view, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 900])
        root.addWidget(split)
        self.axis_seg.changed.connect(self._on_axis)
        self.pos_slider.valueChanged.connect(lambda *_: self._draw())

    # -- data -------------------------------------------------------------
    def set_result(self, r: CFD3DResult, sim, design, *, ground: float | None = None) -> None:
        R = _results_page()
        self._r, self._ground = r, ground
        if R is None:
            return
        window = (f"Mean over t = {sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U"
                  if _finite(r.t_avg_start) and _finite(r.t_end) else "Time average")
        heat_rows = R.heat_tab_rows(r, design, sim, window)
        self.title.setText(R.heat_tab_title(sim))
        self.table.set_data(SUMMARY_HEADERS, heat_rows)
        self.table.fit_height()
        need = fit_table_width(self.table) + 40
        if self.split.sizes()[0] < need:
            total = max(sum(self.split.sizes()), need + 400)
            self.split.setSizes([need, total - need])
        self.left.setMinimumWidth(min(need, 640))
        th = getattr(sim, "thermal", None)
        parts = [("Nu = Q L / (k ΔT A): the heat rate from the wall penalisation, the reference length L, the "
                  "fluid's conductivity k and the reference temperature difference ΔT."),
                 "Standard errors from batch means over the averaging window.", R.rounding_note(heat_rows)]
        if not any("Nusselt" in R._cell_text(row[0]) for row in heat_rows):
            parts = parts[1:]              # a room without a Nusselt row: the definition explains nothing shown
        if th is not None and getattr(th, "enabled", False):
            amb = f"{float(th.t_ambient):.6g}".replace("-", MINUS)
            parts.insert(0, f"Ambient {amb} °C" + ("; buoyancy on (Boussinesq, gravity along −z)."
                                                   if getattr(th, "buoyancy", False) else "; no buoyancy."))
        self.note.setText(" ".join(parts))
        rows = R.heat_body_rows(r, design)
        self.body_table.set_data(["Body", "Wall", "Nu", "± Std. error", "Q", "Unit"], rows, numeric=[2, 3, 4])
        compact_columns(self.body_table, (2, 3, 4))
        self.body_table.fit_height()
        for w in (self.body_table, self.body_caption):
            w.setVisible(bool(rows))
        brows, bnote, _tone = R.heat_balance_rows(r, sim, design)
        self.balance_table.set_data(["Term", "Rate", "Unit"], brows, numeric=[1])
        compact_columns(self.balance_table, (1,))
        self.balance_table.fit_height()
        self.balance_table.setVisible(bool(brows))
        self.balance_note.setText(bnote)
        from .view3d_panel import heat_display_range, heat_volume_of, temperature_field_of
        vol = heat_volume_of(r)
        self._range = None
        if vol is not None:
            vals = np.asarray(vol.fields[temperature_field_of(vol)], dtype=float)
            if vol.mask is not None:
                vals = vals[np.asarray(vol.mask) <= 0.5]
            lo, hi, _mode = heat_display_range(vals)
            self._range = (lo, hi)
        self.plume_btn.setVisible(vol is not None)
        # a room: the slice goes through the hottest heater and frames it (the room's corner said nothing)
        self._hot = R.hottest_source(r, sim, design)[0] if sim is not None and R.is_closed(sim) else None
        self.axis_seg.blockSignals(True)
        self.axis_seg.set_current("y")
        self.axis_seg.blockSignals(False)
        self._reset_position()
        self.view.clear()
        self._draw()

    def _volume(self):
        from .view3d_panel import heat_volume_of
        return heat_volume_of(self._r) if self._r is not None else None

    def _axis_range(self) -> tuple[float, float]:
        vol = self._volume()
        if vol is None:
            return 0.0, 0.0
        b = volume_bounds(vol)
        k = _AXIS[self.axis_seg.current()]
        return float(b[2 * k]), float(b[2 * k + 1])

    def _reset_position(self) -> None:
        """Through the body's centre (y = 0 for a side slice, mid-height, the body's mid-length)."""
        lo, hi = self._axis_range()
        from .view3d_panel import bounds_of
        bb = bounds_of(self._r.surface) if self._r is not None else None
        k = _AXIS[self.axis_seg.current()]
        pos = 0.5 * (bb[2 * k] + bb[2 * k + 1]) if bb is not None else 0.5 * (lo + hi)
        if self.axis_seg.current() == "y":
            pos = 0.0 if lo <= 0.0 <= hi else pos
        hot = getattr(self, "_hot", None)
        if hot is not None and len(tuple(hot.position)) > k:
            pos = min(max(float(hot.position[k]), lo), hi)
        self.pos_slider.blockSignals(True)
        self.pos_slider.setValue(round(1000 * (pos - lo) / (hi - lo)) if hi > lo else 0)
        self.pos_slider.setEnabled(hi > lo)
        self.pos_slider.blockSignals(False)

    def position(self) -> float:
        lo, hi = self._axis_range()
        return lo + (hi - lo) * self.pos_slider.value() / 1000.0 if hi > lo else lo

    def _on_axis(self, *_) -> None:
        self._reset_position()
        self.view.clear()
        self._draw()

    def current_slice(self):
        vol = self._volume()
        if vol is None:
            return None
        from .view3d_panel import temperature_field_of
        data = lazy_import("aerolume.viz3d.data")
        if data is None:
            return None
        try:
            return data.slice_from_volume(vol, self.axis_seg.current(), self.position(), temperature_field_of(vol),
                                          label=getattr(vol, "label", "mean"))
        except (KeyError, ValueError):
            log.debug("temperature slice failed", exc_info=True)
            return None

    def _draw(self) -> None:
        fv = self.view
        pos = self.position()
        self.pos_value.setText(f"{pos:.3f} m".replace("-", MINUS) if _finite(pos) else "—")
        sd = self.current_slice()
        if sd is None or self._range is None:
            fv.clear()
            fv.set_placeholder("No temperature field stored", "This run stored no 3D temperature volume.")
            return
        from .view3d_panel import temperature_field_of
        f = temperature_field_of(sd)
        lab, unit, _div, _c = field_info3d(f)
        first = not fv.has_image()
        fv.set_axis_names({"y": ("x", "z"), "z": ("x", "y"), "x": ("y", "z")}[sd.axis])
        fv.set_sections(mesh_sections(self._r.surface, sd.axis, float(sd.position)))
        fv.set_ground(self._ground if sd.axis in ("y", "x") else None)
        fv.set_field(np.asarray(sd.fields[f], dtype=np.float32), sd.extent, field=f, cmap=default_cmap3d(f),
                     vmin=self._range[0], vmax=self._range[1], mask=sd.mask, label=lab, units=unit)
        when = "time-averaged" if getattr(sd, "label", "mean") == "mean" else "final snapshot"
        fv.set_caption(f"{sd.axis} = {sig(float(sd.position), 3)} m · {when}".replace("-", MINUS))
        if first:
            rect = self._hot_rect(sd)
            if rect is not None:
                fv.set_focus_rect(rect, contain=True)
            else:
                fv.fit()

    def _hot_rect(self, sd) -> tuple[float, float, float, float] | None:
        """The hottest heater's footprint in the slice's plane with a margin of its own size (at least a
        fifth of the room), clipped to the slice; None without a heater."""
        hot = getattr(self, "_hot", None)
        if hot is None:
            return None
        ci, ri = _SLICE_AXES[sd.axis]
        pos = tuple(float(c) for c in hot.position)
        size = tuple(float(c) for c in hot.size) or (0.0,)
        half = [size[0]] * 3 if hot.kind == "sphere" else [0.5 * size[min(i, len(size) - 1)] for i in range(3)]
        x0, x1, y0, y1 = (float(v) for v in sd.extent)
        mx = max(2 * half[ci], 0.2 * (x1 - x0))
        my = max(2 * half[ri], 0.2 * (y1 - y0))
        rect = (max(x0, pos[ci] - half[ci] - mx), min(x1, pos[ci] + half[ci] + mx),
                max(y0, pos[ri] - half[ri] - my), min(y1, pos[ri] + half[ri] + my))
        return rect if rect[1] > rect[0] and rect[3] > rect[2] else None


class CFD3DResultView(QWidget):
    """Tabs of a :class:`~aerolume.model.CFD3DResult` (see the module docstring).

    ``set_result(result, sim, design, export_dir=..., log=...)`` fills it;
    the 3D tabs build their scenes the first time they are shown.
    """

    TABS = ("summary", "forces", "surface", "fields", "heat", "fans", "sensitivity", "diagnostics", "data")
    EXPORTS = (("result_csv", "Summary (CSV)…", ".csv", "CSV (*.csv)"),
               ("history_csv", "Force history (CSV)…", ".csv", "CSV (*.csv)"),
               ("surface_vtp", "Surface data (VTK .vtp)…", ".vtp", "VTK PolyData (*.vtp)"),
               ("volume_vti", "Flow volume (VTK .vti)…", ".vti", "VTK image data (*.vti)"),
               ("smoke_vti", "Smoke volume (VTK .vti)…", ".vti", "VTK image data (*.vti)"),
               ("slice_csv", "Mean slice (CSV)…", ".csv", "CSV (*.csv)"),
               ("mesh_stl", "Surface mesh (STL)…", ".stl", "STL mesh (*.stl)"),
               ("view_png", "3D flow view (PNG)…", ".png", "PNG image (*.png)"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result: CFD3DResult | None = None
        self.sim = None
        self.design = None
        self._rake: dict | None = None        # where the run released its smoke (a viz3d.flow rake dict)
        self._filled: set[str] = set()
        self._export_dir = Path.home()
        self._log = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        lay.addWidget(self.tabs, 1)
        (w, self.summary_table, self.summary_note, self.part_table, self.part_caption,
         self.conditions) = _summary_tab(SUMMARY_HEADERS)
        self.tabs.addTab(w, "Summary")
        w, v = _tab_page()
        self.forces_caption = label("", "muted")
        self.forces_caption.setToolTip("The ± values in Summary are standard errors from batch means over the "
                                       "averaging window. Time is in convective units (reference length / speed).")
        self.history_plot = HistoryPlot(keys=("cd", "cl", "cm"), labels=HistoryPlot.LABELS_3D, show_mean=True)
        v.addWidget(self.forces_caption)
        v.addWidget(self.history_plot, 1)
        self.tabs.addTab(w, "Forces")
        # heavy tabs: placeholders until first shown (in TABS order)
        self._hosts: dict[str, QWidget] = {}
        titles = {"surface": "Surface", "fields": "Fields", "heat": "Heat transfer", "fans": "Fans",
                  "sensitivity": "Sensitivity", "diagnostics": "Diagnostics"}
        for key in self.TABS[2:-1]:
            host = QWidget()
            hl = QVBoxLayout(host)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(0)
            self._hosts[key] = host
            self.tabs.addTab(host, titles[key])
        self.surface_panel: Surface3DPanel | None = None
        self.fields_panel: Fields3DPanel | None = None
        self.heat_panel: Heat3DPanel | None = None
        self.fan_table: DataTable | None = None
        self.diag_panel = None
        self.sens_panel: Surface3DPanel | None = None
        self.sens_stack: QStackedWidget | None = None
        self.data_tab = _DataTab(self.EXPORTS)
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.currentChanged.connect(self._on_tab)

    # -- API --------------------------------------------------------------
    def set_result(self, r: CFD3DResult, sim, design, *, export_dir: Path | None = None, log=None,
                   rake: dict | None = None) -> None:
        """Show a 3D result; ``rake`` is where its run released its smoke (see :meth:`Fields3DPanel.set_result`)."""
        self.result, self.sim, self.design = r, sim, design
        self._rake = rake
        self._export_dir = export_dir or Path.home()
        self._log = log
        self._filled = set()
        self._sens_available = any(s.sensitivity is not None and len(s.sensitivity) for s in r.surface)
        # hiding the current tab moves the selection: fill nothing until the result is in place
        self.tabs.blockSignals(True)
        self.tabs.setTabVisible(self.TABS.index("surface"), bool(r.surface))
        self.tabs.setTabVisible(self.TABS.index("fields"), r.volume is not None or bool(r.slices))
        R = _results_page()
        # a closed room has no free stream: the Summary hides its coefficients, so the Forces and
        # Sensitivity tabs would only invite misreading (the histories stay in Data / CSV)
        closed = bool(R is not None and R.is_closed(sim))
        self.tabs.setTabVisible(self.TABS.index("forces"), not closed)
        self.tabs.setTabVisible(self.TABS.index("sensitivity"), not closed)
        self.tabs.setTabVisible(self.TABS.index("heat"), bool(R is not None and R.has_heat(r)))
        self.tabs.setTabVisible(self.TABS.index("fans"), bool(R is not None and R.has_fan_report(r)))
        self.tabs.setTabVisible(self.TABS.index("diagnostics"), bool(
            R is not None and (R.result_diagnostics_rows(r) or R.history_diagnostics(r) is not None)))
        if not self.tabs.isTabVisible(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(0)
        self.tabs.blockSignals(False)
        self._fill_summary()
        self._fill_forces()
        self._fill_data()
        self._on_tab(self.tabs.currentIndex())

    def select_tab(self, key: str) -> bool:
        if key in self.TABS:
            i = self.TABS.index(key)
            if self.tabs.isTabVisible(i):
                self.tabs.setCurrentIndex(i)
                return True
        return False

    def current_tab(self) -> str:
        return self.TABS[self.tabs.currentIndex()]

    def _stem(self) -> str:
        return _safe(self.sim.name if self.sim is not None else "result")

    def _ground(self) -> float | None:
        g = getattr(self.sim, "ground", None)
        return 0.0 if g is not None and getattr(g, "enabled", False) else None

    # -- tabs -------------------------------------------------------------
    def _on_tab(self, index: int) -> None:
        if self.result is None or not 0 <= index < len(self.TABS):
            return
        key = self.TABS[index]
        if key in self._filled:
            return
        self._filled.add(key)
        try:
            if key == "surface":
                self._fill_surface()
            elif key == "fields":
                self._fill_fields()
            elif key == "heat":
                self._fill_heat()
            elif key == "fans":
                self._fill_fans()
            elif key == "diagnostics":
                self._fill_diagnostics()
            elif key == "sensitivity":
                self._fill_sensitivity()
        except Exception:
            log.exception("filling the %s tab failed", key)

    def _fill_summary(self) -> None:
        r, sim, design = self.result, self.sim, self.design
        rows = summary_rows3d(r, design, sim)
        R = _results_page()
        closed = R is not None and sim is not None and R.is_closed(sim)
        if R is not None and not closed:
            window = (f"Mean over t = {sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U"
                      if _finite(r.t_avg_start) and _finite(r.t_end) else "Time average")
            rows = rows + R.thermal_summary_rows(r, window, sim, design)
        self.summary_table.set_data(SUMMARY_HEADERS, rows)
        self.summary_table.fit_height()
        fit_table_width(self.summary_table)
        self.tabs.widget(0).reflow.apply()
        note = summary_note3d(r, design, sim, rows)
        pb = r.per_body or {}
        multi = len(pb) > 1 and not closed
        if R is not None:
            self.part_table.set_data(["Part", "CD", "CL"], R.per_part_rows(r, ("cd", "cl")))
            self.part_caption.setToolTip(R.PER_PART_TIP)
            if multi:
                note += " " + R.PER_PART_NOTE
        else:
            self.part_table.set_data(["Part", "CD", "CL"], [[n, fmt_fixed(d.get("cd"), 4), fmt_fixed(d.get("cl"), 4)]
                                                          for n, d in pb.items()])
        self.summary_note.setText(note)
        compact_columns(self.part_table, (1, 2))
        self.part_table.fit_height()
        self.part_table.setVisible(multi)
        self.part_caption.setVisible(multi)
        self.conditions.set_items(conditions_items3d(r, design, sim))

    def _fill_forces(self) -> None:
        r = self.result
        self.history_plot.set_keys(history_keys3d(r), HistoryPlot.LABELS_3D)
        self.history_plot.set_history(r.history or {}, (r.t_avg_start, r.t_end) if _finite(r.t_avg_start) else None)
        self.history_plot.autoRange()
        sym = " Side force, rolling and yawing moment are zero by symmetry (half domain)." if r.symmetry_plane else ""
        self.forces_caption.setText(
            (f"Averaging window t = {sig(r.t_avg_start, 3)}–{sig(r.t_end, 3)} L/U (grey band); dashed lines are "
             "the means." if _finite(r.t_avg_start) else "No averaging window recorded.") + sym)

    def _fill_surface(self) -> None:
        if self.surface_panel is None:
            self.surface_panel = Surface3DPanel(mode="fields")
            self._hosts["surface"].layout().addWidget(self.surface_panel)
        self.surface_panel.set_export(self._export_dir, self._stem())
        self.surface_panel.set_surfaces(self.result.surface, ground=self._ground())

    def _view_png_writer(self, path: str):
        """Render the Fields tab's 3D view at 1600 x 1000 now (OpenGL lives on this thread); write it later.

        The flow view is built first if its tab was never opened, so the export
        shows the flow (smoke, streamlines) and not only the surface.
        """
        if self.fields_panel is None:
            self._filled.add("fields")
            self._fill_fields()
        img = np.ascontiguousarray(self.fields_panel.view.render_image(1600, 1000))

        def write() -> str:
            from PySide6.QtGui import QImage
            h, w = img.shape[:2]
            if not QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy().save(path, "PNG"):
                raise OSError(f"could not write {path}")
            return path
        return write

    def _fill_fields(self) -> None:
        if self.fields_panel is None:
            self.fields_panel = Fields3DPanel()
            self._hosts["fields"].layout().addWidget(self.fields_panel)
        self.fields_panel.set_export(self._export_dir, self._stem())
        self.fields_panel.set_result(self.result, ground=self._ground(), rake=self._rake, sim=self.sim,
                                     design=self.design)

    def _fill_heat(self) -> None:
        if self.heat_panel is None:
            self.heat_panel = Heat3DPanel()
            self.heat_panel.plume_requested.connect(self._show_plume)
            self._hosts["heat"].layout().addWidget(self.heat_panel)
        self.heat_panel.set_result(self.result, self.sim, self.design, ground=self._ground())

    def _show_plume(self) -> None:
        """The Heat tab's "Show the plume in 3D": the Fields tab with the thermal plume on."""
        if self.select_tab("fields") and self.fields_panel is not None:
            b = self.fields_panel.toggles.buttons.get("heat")
            if b is not None and b.isEnabled() and not b.isChecked():
                b.setChecked(True)

    def _fill_fans(self) -> None:
        R = _results_page()
        if self.fan_table is None:
            w, v = _tab_page()
            v.addWidget(label("Fans", "h3"))
            self.fan_table = DataTable(R.FAN_HEADERS if R is not None else [], stretch_last=False)
            v.addWidget(self.fan_table)
            self.fan_note = label("Momentum theory: the induced velocity a Froude actuator disc of the fan's "
                                  "diameter needs for the delivered thrust at the run's speed and density.",
                                  "muted", wrap=True)
            v.addWidget(self.fan_note)
            v.addStretch(1)
            self._hosts["fans"].layout().addWidget(w)
        if R is None:
            return
        units = R.fan_units(self.result)
        headers = [f"{h} ({u})" if u else h for h, u in zip(R.FAN_HEADERS, units)]
        self.fan_table.set_data(headers, R.fan_rows(self.result, self.sim), numeric=[2, 3, 4, 5, 6, 7, 8])
        self.fan_table.fit_height()
        if hasattr(R, "fan_note_text"):
            self.fan_note.setText(R.fan_note_text(self.sim, three_d=True))
        fit_table_width(self.fan_table)

    def _fill_diagnostics(self) -> None:
        R = _results_page()
        if R is None:
            return
        if self.diag_panel is None:
            from .diagnostics import DiagnosticsPanel
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
            sl.addWidget(self.diag_table)
            self.diag_note = label("", "muted", wrap=True)
            sl.addWidget(self.diag_note)
            sl.addStretch(1)
            side.setMinimumWidth(320)
            split.addWidget(side)
            self.diag_panel = DiagnosticsPanel()
            split.addWidget(self.diag_panel)
            split.setStretchFactor(1, 1)
            lay.addWidget(split)
            self._hosts["diagnostics"].layout().addWidget(w)
        rows = R.result_diagnostics_rows(self.result)
        self.diag_table.set_data(["Quantity", "Value", "Unit"], rows, numeric=[1])
        self.diag_table.fit_height()
        self.diag_table.setVisible(bool(rows))
        log_ = R.history_diagnostics(self.result)
        if log_ is None:
            from .diagnostics import DiagnosticsLog
            log_ = DiagnosticsLog()
        self.diag_note.setText(("" if rows else "This result stores no end-of-run diagnostics. ")
                               + ("The history plots are stored with the result." if len(log_) else
                                  "No diagnostic history was stored with this result."))
        self.diag_panel.set_thermal(R.run_is_thermal(self.result, self.sim))   # no θ / heat columns without heat
        self.diag_panel.set_log(log_)
        self.diag_panel.set_export_dir(self._export_dir)

    def _fill_sensitivity(self) -> None:
        if self.sens_stack is None:
            self.sens_stack = QStackedWidget()
            host = QWidget()
            hl = QVBoxLayout(host)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(0)
            self.sens_panel = Surface3DPanel(mode="sensitivity")
            hl.addWidget(self.sens_panel)
            self.sens_stack.addWidget(host)
            self.sens_empty = EmptyState()
            self.sens_empty.set_content("No sensitivity map for this run",
                                        "Turn on Compute sensitivity in Setup and run again (3D maps need a "
                                        "grid of at most about 2 million cells).")
            self.sens_stack.addWidget(self.sens_empty)
            self._hosts["sensitivity"].layout().addWidget(self.sens_stack)
        if not self._sens_available:
            self.sens_stack.setCurrentIndex(1)
            return
        self.sens_stack.setCurrentIndex(0)
        p = self.sens_panel
        p.set_export(self._export_dir, self._stem())
        p.set_surfaces(self.result.surface, ground=self._ground())
        obj = {"cd": "CD", "cl": "CL", "ld": "L/D", "cm": "Cm"}.get(
            getattr(getattr(self.sim, "cfd", None), "sensitivity_objective", "cd"), "CD")
        p.caption.setText(f"Where pushing the surface outward changes {obj}: red increases it, blue decreases it. "
                          "Scaled to the strongest 2 % of the surface: read the pattern, not the size (in 3D the "
                          "magnitude was within 6 % of the physical slope on a sphere at Re 30 but 28 % above it at "
                          "Re 100). Values in 1/m are in the surface .vtp export.")
        p.caption.setToolTip(f"1.0 on the colour bar is {sig(getattr(p, 'scale98', float('nan')), 3)} per metre "
                             f"of outward displacement (98th percentile of |d{obj}/dn|).")
        p.caption.show()

    def _fill_data(self) -> None:
        from .view3d_panel import smoke_volume_of
        r, sim, design = self.result, self.sim, self.design
        self.data_tab.set_rows(scalar_rows3d(r, design, sim))
        self.data_tab.buttons["volume_vti"].setVisible(r.volume is not None)
        self.data_tab.buttons["smoke_vti"].setVisible(smoke_volume_of(r) is not None)
        self.data_tab.buttons["view_png"].setVisible(bool(r.surface))
        self.data_tab.buttons["slice_csv"].setVisible(bool(r.slices) or r.volume is not None)
        self.data_tab.buttons["surface_vtp"].setVisible(bool(r.surface))
        self.data_tab.buttons["mesh_stl"].setVisible(bool(r.surface))
        self.data_tab.configure(self._export_dir, self._stem(), self._make_export, self._log)

    def _make_export(self, key: str, path: str):
        exp = lazy_import("aerolume.export")
        if exp is None:
            raise RuntimeError("the export module is unavailable")
        r = self.result
        if key == "volume_vti":
            return None if r.volume is None else (lambda: exp.volume_vti(r.volume, path))
        if key == "smoke_vti":
            from .view3d_panel import smoke_volume_of
            smoke = smoke_volume_of(r)
            return None if smoke is None else (lambda: exp.volume_vti(smoke, path))
        if key == "view_png":
            return self._view_png_writer(path)
        if key == "slice_csv":
            sd = self.fields_panel.current_slice() if self.fields_panel is not None else None
            if sd is not None:
                return lambda: exp.slice_csv(sd, path)
            return (lambda: exp.slice_csv(r, path, index=0)) if r.slices else None
        if key == "mesh_stl":
            from ...model import Mesh3D
            meshes = [Mesh3D(s.body or f"part{i + 1}", s.vertices, s.faces) for i, s in enumerate(r.surface)]
            return lambda: exp.mesh_stl(meshes, path)
        fn = getattr(exp, key)
        return lambda: fn(r, path)


# ---------------------------------------------------------------------------
# wing result view
# ---------------------------------------------------------------------------

def _airfoil_coords_for(airfoils: dict):
    """Resolver for "design:<id>" section names from a snapshot of 2D designs (numpy + the airfoil prep)."""
    if not airfoils:
        return None
    cache: dict[str, np.ndarray] = {}

    def resolve(name: str) -> np.ndarray:
        key = str(name).strip()
        if key not in cache:
            did = key.split(":", 1)[1].strip() if key.lower().startswith("design:") else key
            d = airfoils.get(did)
            if d is None:
                raise ValueError(f"airfoil {name!r} is not in this project")
            from ...solvers.airfoil import prepare_coords
            cache[key] = np.asarray(prepare_coords(d)[0], dtype=float)
        return cache[key]
    return resolve


def _loft_wing(wing, airfoils: dict):
    """The wing's surface mesh (worker thread): lofted through its sections."""
    from ...geometry3d.wing import loft
    return loft(wing, n_chord=32, airfoil_coords=_airfoil_coords_for(airfoils), name="Wing")


def vertex_cl(mesh, point) -> np.ndarray | None:
    """Per-vertex local cl of a lofted wing: the strip value at each vertex's span station."""
    if mesh is None or point is None or point.y is None or point.cl_local is None:
        return None
    o = span_order(point)
    y = np.asarray(point.y, dtype=float)[o]
    cl = np.asarray(point.cl_local, dtype=float)[o]
    ok = np.isfinite(cl)
    if ok.sum() < 2:
        return None
    vy = np.asarray(mesh.vertices, dtype=float)[:, 1]
    if np.nanmin(y) >= -1e-9:                 # a half-wing sweep: mirror for the left half
        vy = np.abs(vy)
    return np.interp(vy, y[ok], cl[ok])


class WingResultView(QWidget):
    """Tabs of a :class:`~aerolume.model.WingResult` (see the module docstring)."""

    TABS = ("summary", "polar", "span", "points", "data")
    EXPORTS = (("result_csv", "Summary (CSV)…", ".csv", "CSV (*.csv)"),
               ("wing_csv", "Polar table (CSV)…", ".csv", "CSV (*.csv)"),
               ("wing_span_csv", "Span loading (CSV)…", ".csv", "CSV (*.csv)"),
               ("mesh_stl", "Wing mesh (STL)…", ".stl", "STL mesh (*.stl)"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result: WingResult | None = None
        self.sim = None
        self.design = None
        self._airfoils: dict = {}
        self._mesh = None
        self._loft_call = None
        self._export_dir = Path.home()
        self._log = None
        self._filled: set[str] = set()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        lay.addWidget(self.tabs, 1)
        (w, self.summary_table, self.summary_note, _parts, _cap, self.conditions) = _summary_tab(WING_SUMMARY_HEADERS)
        _parts.hide()
        _cap.hide()
        self.tabs.addTab(w, "Summary")
        self.polars = WingPolarGrid()
        self.tabs.addTab(self.polars, "Polars")
        span = QWidget()
        sv = QVBoxLayout(span)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)
        bar = ToolStrip()
        bar.add_widget(label("Angle of attack", "secondary"))
        self.alpha_combo = QComboBox()
        self.alpha_combo.setMinimumWidth(90)
        bar.add_widget(self.alpha_combo)
        bar.add_separator()
        self.point_facts = label("", "secondary")
        bar.add_widget(self.point_facts)
        bar.add_stretch()
        sv.addWidget(bar)
        self.span_split = QSplitter(Qt.Orientation.Horizontal)
        self.span_split.setHandleWidth(1)
        self.span_split.setChildrenCollapsible(False)
        self.span_plots = SpanLoadingPlots()
        self.span_split.addWidget(self.span_plots)
        self._view_host = QWidget()
        vl = QVBoxLayout(self._view_host)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        self.span_split.addWidget(self._view_host)
        self.span_split.setSizes([620, 520])
        sv.addWidget(self.span_split, 1)
        self.tabs.addTab(span, "Span loading")
        self.wing_view = None
        w, v = _tab_page()
        self.points_table = DataTable(["α (°)", "CL", "CD", "CDi", "CDv", "Cm", "L/D", "e", "Converged"])
        v.addWidget(self.points_table, 1)
        self.tabs.addTab(w, "Points")
        self.data_tab = _DataTab(self.EXPORTS)
        self.tabs.addTab(self.data_tab, "Data")
        self.alpha_combo.currentIndexChanged.connect(self._draw_span)
        self.tabs.currentChanged.connect(self._on_tab)

    def set_result(self, w: WingResult, sim, design, *, airfoils: dict | None = None,
                   export_dir: Path | None = None, log=None) -> None:
        self.result, self.sim, self.design = w, sim, design
        self._airfoils = dict(airfoils or {})
        self._export_dir = export_dir or Path.home()
        self._log = log
        self._filled = set()
        self._mesh = None
        if self._loft_call is not None:
            self._loft_call.cancelled = True
            self._loft_call = None
        self.summary_table.set_data(WING_SUMMARY_HEADERS, wing_summary_rows(w), numeric=[1])
        self.summary_table.fit_height()
        fit_table_width(self.summary_table)
        self.tabs.widget(0).reflow.apply()
        note = ("Coefficients on the planform area "
                f"{fmt_num(w.area, 4)} m²; moments about the moment reference point. CL,max and the stall onset "
                "come from 2D section polars and are not validated against stall data.")
        if not w.viscous:
            note += " Inviscid sweep: CD is induced drag only."
        self.summary_note.setText(note)
        self.conditions.set_items(wing_conditions_items(w, design, sim))
        self.polars.set_wings([(sim.name if sim is not None else "Wing", w, 0)])
        rows = []
        for p in sorted(w.points, key=lambda q: q.alpha):
            ld = p.cl / p.cd if p.converged and w.viscous and _finite(p.cd) and p.cd > 0 else float("nan")
            rows.append([fmt_fixed(p.alpha, 2), fmt_fixed(p.cl, 4), fmt_fixed(p.cd, 5), fmt_fixed(p.cdi, 5),
                         fmt_fixed(p.cdv, 5), fmt_fixed(p.cm, 4), fmt_fixed(ld, 1), fmt_fixed(p.e, 3),
                         Cell("yes") if p.converged else Cell("no", tone="warn")])
        self.points_table.set_data(["α (°)", "CL", "CD", "CDi", "CDv", "Cm", "L/D", "e", "Converged"], rows,
                                   numeric=list(range(8)))
        self._fill_alpha_combo()
        self.data_tab.set_rows(wing_scalar_rows(w, design, sim))
        self.data_tab.buttons["mesh_stl"].setVisible(design is not None and getattr(design, "wing", None) is not None)
        self.data_tab.configure(self._export_dir, _safe(sim.name if sim is not None else "wing"),
                                self._make_export, self._log)
        self._on_tab(self.tabs.currentIndex())

    def select_tab(self, key: str) -> bool:
        key = {"polars": "polar", "forces": "polar", "surface": "span"}.get(key, key)
        if key in self.TABS:
            self.tabs.setCurrentIndex(self.TABS.index(key))
            return True
        return False

    def _fill_alpha_combo(self) -> None:
        w = self.result
        self.alpha_combo.blockSignals(True)
        self.alpha_combo.clear()
        pts = sorted((p for p in w.points if p.converged and p.y is not None), key=lambda p: p.alpha)
        for p in pts:
            self.alpha_combo.addItem(_deg(p.alpha), float(p.alpha))
        m = wing_metrics_safe(w)
        want = m.get("alpha_ld_max", float("nan"))
        if not _finite(want):
            want = m.get("alpha_cruise", float("nan"))
        if not _finite(want) and pts:
            want = pts[len(pts) // 2].alpha
        if pts:
            best = min(range(len(pts)), key=lambda i: abs(pts[i].alpha - want)) if _finite(want) else 0
            self.alpha_combo.setCurrentIndex(best)
        self.alpha_combo.blockSignals(False)

    def current_point(self):
        a = self.alpha_combo.currentData()
        return nearest_point(self.result, a) if self.result is not None and a is not None else None

    def _on_tab(self, index: int) -> None:
        if self.result is None or not 0 <= index < len(self.TABS):
            return
        key = self.TABS[index]
        if key == "span" and "span" not in self._filled:
            self._filled.add("span")
            self._draw_span()
            self._start_loft()

    def _draw_span(self, *_) -> None:
        pt = self.current_point()
        w = self.result
        if pt is None or w is None:
            self.span_plots.set_points([])
            self.point_facts.setText("No span data")
            return
        name = self.sim.name if self.sim is not None else "Wing"
        self.span_plots.set_points([(name, pt, float(w.span), 0)])
        ld = pt.cl / pt.cd if w.viscous and _finite(pt.cd) and pt.cd > 0 else float("nan")
        self.point_facts.setText(f"CL {fmt_num(pt.cl, 4)} · CD {fmt_num(pt.cd, 4)} · CDi {fmt_num(pt.cdi, 4)}"
                                 + (f" · L/D {fmt_num(ld, 3)}" if _finite(ld) else "")
                                 + f" · e {fmt_num(pt.e, 3)}")
        self.polars.mark_alpha(pt.alpha)
        self._colour_wing()

    def _start_loft(self) -> None:
        design = self.design
        wing = getattr(design, "wing", None) if design is not None else None
        if self.wing_view is None:
            self.wing_view = _view3d(overlay_toolbar=True)
            self.wing_view.set_placeholder("Building the wing surface…", "")
            self._view_host.layout().addWidget(self.wing_view)
        if wing is None:
            self.wing_view.set_placeholder("The design has no wing planform", "")
            return
        from ..jobs import run_in_background

        def done(mesh) -> None:
            self._loft_call = None
            self._mesh = mesh
            self._colour_wing(first=True)

        def failed(msg: str) -> None:
            self._loft_call = None
            self.wing_view.set_placeholder("The wing surface could not be built", msg)

        self._loft_call = run_in_background(_loft_wing, wing, self._airfoils, on_done=done, on_error=failed)

    def _colour_wing(self, first: bool = False) -> None:
        if self.wing_view is None or self._mesh is None:
            return
        sc = self.wing_view.scene
        sc.clear()
        pt = self.current_point()
        cl = vertex_cl(self._mesh, pt)
        if cl is None:
            sc.add_mesh(self._mesh)
        else:
            lo, hi = float(np.nanmin(cl)), float(np.nanmax(cl))
            pad = 0.05 * max(hi - lo, 1e-3)
            sc.add_mesh(self._mesh, cl, field="cl_local", cmap="viridis", range=(min(lo - pad, 0.0), hi + pad))
            sc.add_colorbar("Local lift coefficient cl", "")
            sc.set_caption(f"α = {pt.alpha:g}°".replace("-", MINUS))
        if first:
            self.wing_view.set_view("iso")
        self.wing_view.refresh()

    def _make_export(self, key: str, path: str):
        exp = lazy_import("aerolume.export")
        if exp is None:
            raise RuntimeError("the export module is unavailable")
        w = self.result
        if key == "mesh_stl":
            wing = getattr(self.design, "wing", None)
            if wing is None:
                return None
            mesh, airfoils = self._mesh, dict(self._airfoils)

            def write():
                return exp.mesh_stl([mesh if mesh is not None else _loft_wing(wing, airfoils)], path)
            return write
        fn = getattr(exp, key)
        return lambda: fn(w, path)


# ---------------------------------------------------------------------------
# live slices in the 3D view (Simulate page)
# ---------------------------------------------------------------------------

def frame_to_slice(frame, *, max_cols: int = 240):
    """A live slice frame as SliceData, values recovered from its colours (its plane in the 3D view).

    Live frames carry an RGBA image, not values. Inverting the colour map (the
    frame's 256-entry LUT, exact for fluid pixels; solids and overlays are not
    map colours and become the mask) gives values accurate to 1/255 of the
    colour range, finer than a plane in a 3D view can show; down-sampled to
    ``max_cols`` columns. None for frames that are not 3D slices.
    """
    from ...model import SliceData
    from .field_view import colormap_lut, lut_indices
    axis = getattr(frame, "slice_axis", None)
    if not axis:
        return None
    rgba = np.asarray(frame.rgba)
    step = max(1, math.ceil(rgba.shape[1] / max_cols))
    img = rgba[::step, ::step]
    field = getattr(frame, "field", "speed") or "speed"
    lut = colormap_lut(getattr(frame, "cmap", None) or default_cmap3d(field), field)
    idx = lut_indices(img, lut)
    vmin, vmax = float(frame.vmin), float(frame.vmax)
    vals = vmin + (vmax - vmin) * np.clip(idx, 0, None).astype(np.float32) / (len(lut) - 1)
    mask = (idx < 0).astype(np.float32)
    # image order (row 0 = top) -> SliceData order (row 0 = row minimum)
    return SliceData(axis=str(axis), position=float(getattr(frame, "slice_pos", 0.0) or 0.0),
                     extent=tuple(float(e) for e in frame.extent), fields={field: vals[::-1].copy()},
                     mask=mask[::-1].copy(), label="instant")


def neutral_plane(axis: str, position: float, window: Sequence[float]):
    """A plain plane (every cell 'solid', drawn in the theme's neutral grey) marking where a slice will be."""
    from ...model import SliceData
    shape = (2, 2)
    return SliceData(axis=axis, position=float(position), extent=tuple(float(w) for w in window),
                     fields={"u": np.zeros(shape, np.float32)}, mask=np.ones(shape, np.float32))


# ---------------------------------------------------------------------------
# side-by-side 3D surfaces with linked cameras (Compare page)
# ---------------------------------------------------------------------------

class SurfaceCompareGrid(QWidget):
    """Tiles of 3D renders (one per run) on one colour range, with linked cameras.

    ``set_runs([(name, surfaces[, result]), ...], ground=...)``. Every tile
    frames the union of all bodies, so equal shapes appear equally large;
    orbiting, panning or zooming one tile moves the others the same way. The
    surfaces are coloured by Cp or Cf; **Streamlines** adds each run's
    mean-flow streamlines from one rake placed for all runs alike (in one
    neutral colour: the colour bar belongs to the surface field), and
    **Smoke** each run's stored smoke snapshot. A toggle is enabled only when
    every run has the data, so the tiles always show the same things.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from .view3d_panel import FlowToggles
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.bar = ToolStrip()
        self.bar.add_widget(label("Field", "secondary"))
        self.field_combo = QComboBox()
        for key, text in SURFACE_FIELDS:
            self.field_combo.addItem(text, key)
        self.bar.add_widget(self.field_combo)
        self.bar.add_separator()
        self.toggles = FlowToggles(("streamlines", "smoke"), shown={"streamlines": False, "smoke": False}, parent=self)
        self.toggles.add_to(self.bar)
        self.bar.add_separator()
        from .view3d import CAMERA_PRESETS
        for key, text, tip in CAMERA_PRESETS:
            self.bar.add_button(None, text, tip, on_click=lambda k=key: self._preset(k))
        self.bar.add_stretch()
        self.link_note = label("Cameras are linked", "muted")
        self.link_note.setToolTip("Orbiting, panning or zooming one view moves all of them")
        self.bar.add_widget(self.link_note)
        self.bar.row.addSpacing(4)
        lay.addWidget(self.bar)
        self.host = QWidget()
        self.grid = QGridLayout(self.host)
        self.grid.setContentsMargins(1, 1, 1, 1)
        self.grid.setSpacing(1)
        lay.addWidget(self.host, 1)
        self.tiles: list = []
        self._runs: list = []
        self._ground: float | None = None
        self._syncing = False
        self.field_combo.currentIndexChanged.connect(lambda *_: self._draw())
        self.toggles.changed.connect(lambda *_: self._draw())

    def set_runs(self, runs: Sequence[tuple], *, ground: float | None = None) -> None:
        """``runs``: (name, surfaces) or (name, surfaces, result) per run, in tile order."""
        from .view3d_panel import has_velocity, smoke_volume_of
        self._runs = []
        for item in runs:
            name, surf = item[0], item[1]
            res = item[2] if len(item) > 2 else None
            self._runs.append((name, [s for s in surf if len(getattr(s, "faces", []))], res))
        self._ground = ground
        results = [r for _n, _s, r in self._runs]
        has_vel = bool(results) and all(r is not None and has_velocity(getattr(r, "volume", None)) for r in results)
        has_smoke = bool(results) and all(r is not None and smoke_volume_of(r) is not None for r in results)
        self.toggles.set_available("streamlines", has_vel, "a run without a stored 3D velocity volume")
        self.toggles.set_available("smoke", has_smoke, "a run without a stored smoke snapshot")
        self._draw()

    def close_tiles(self) -> None:
        for v in self.tiles:
            self.grid.removeWidget(v)
            try:
                v.scene.close()
            except Exception:
                log.debug("closing a compare tile's scene failed", exc_info=True)
            v.setParent(None)
            v.deleteLater()
        self.tiles = []

    def _draw(self) -> None:
        from .view3d_panel import (
            RakeSpec,
            add_smoke_layers,
            flow_focus_box,
            rake_mesh,
            smoke_volume_of,
            volume_box,
        )
        self.close_tiles()
        runs = self._runs
        if not runs:
            return
        field = self.field_combo.currentData() or "cp"
        vals = [np.asarray(getattr(s, field), float) for _n, surf, _r in runs for s in surf
                if getattr(s, field, None) is not None]
        rng = None
        if vals:
            allv = np.concatenate(vals)
            rng = surface_cp_range(allv) if field == "cp" else display_range3d(field, allv)
        allsurf = [s for _n, surf, _r in runs for s in surf]
        box = rake = None
        if allsurf:
            pts = np.vstack([np.asarray(s.vertices, float) for s in allsurf])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            box = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            rake = RakeSpec.around(box, n=20)
        streams = self.toggles.is_on("streamlines") and rake is not None
        smoke = self.toggles.is_on("smoke")
        flow_box = flow_focus_box(box, ground=self._ground) if (streams or smoke) else box
        cols = 1 if len(runs) == 1 else 2 if len(runs) <= 4 else 3
        for i, (name, surf, res) in enumerate(runs):
            v = _view3d()
            v.setMinimumHeight(220)
            sc = v.scene
            have = [s for s in surf if getattr(s, field, None) is not None]
            if have:
                sc.add_meshes(have, field, cmap=default_cmap3d(field), range=rng)
            for s in surf:
                if getattr(s, field, None) is None:
                    sc.add_mesh(s)
            if self._ground is not None:
                sc.add_ground(float(self._ground))
            try:
                if streams:
                    seeds = rake.clamped(volume_box(res.volume)).seeds()
                    sc.add_streamlines(res.volume, seeds, direction="forward", color_by=None)
                    sc.add_mesh(rake_mesh(rake, max(box[1] - box[0], 1e-6)), color=tokens()["accent"], smooth=False)
                if smoke:
                    add_smoke_layers(sc, smoke_volume_of(res))
            except Exception:
                log.exception("compare tile flow layers failed")
            if flow_box is not None:
                sc.set_focus(flow_box)
            sc.add_colorbar()
            sc.set_caption(name)
            v.set_view("iso")
            v.cameraChanged.connect(lambda src=v: self._sync(src))
            self.grid.addWidget(v, i // cols, i % cols)
            self.tiles.append(v)

    def _preset(self, key: str) -> None:
        for v in self.tiles:
            v.scene.set_camera(key, fit=True)
            v.refresh()

    def _sync(self, src) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            state = src.scene.camera
            for v in self.tiles:
                if v is not src:
                    v.scene.set_camera_state(state)
                    v.refresh()
        finally:
            self._syncing = False
