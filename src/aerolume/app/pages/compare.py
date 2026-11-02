"""Compare page: several simulations side by side against a baseline.

This is where design decisions are made ("is variant B really better?"), so the
page leads with a table (docs/ui_guidelines.md section 4: "table first"): one
row per quantity, one value column per simulation and, for every variant, a
change column in percent of the baseline. Each change answers three questions
at a glance: how big it is (the number), whether it is an improvement (green /
red text) and whether it is larger than the statistical noise of the two runs
(changes within noise are grey, with the reason in the tooltip). The rows come
from :func:`aerolume.analysis.compare_rows`, the same function the report uses,
so the screen and the PDF never disagree.

Plots sit below the table in tabs (Chart, Geometry, Pressure, Polars, Fields).
Each simulation keeps one series colour across every tab, so "the orange one"
means the same variant in the bar chart, the outline overlay and the Cp plot.

**Like with like.** Results come in three kinds that share no reference: 2D
runs (Navier-Stokes per unit span, and airfoil polars, compared together as
before), 3D CFD runs (coefficients on a reference area, forces in N) and
finite-wing sweeps (no single operating point). A selection that mixes kinds is
refused with a plain statement of why and one-click choices of a single kind;
it is never shown as a table of dashes. 3D CFD runs get Chart, Forces (the six
coefficient histories overlaid) and Surface (the bodies coloured by Cp side by
side, on one colour range, with linked cameras, and on request each run's
mean-flow streamlines from one rake or its stored smoke); wing sweeps get Chart,
Polars (overlaid CL-α, drag polars, Cm-α, L/D-α) and Span loading (at one
angle, with the elliptic reference).

"Better" is not always obvious: lower drag is always good, but whether more
lift is good depends on the user (a wing wants lift, a race car downforce, a
road car neither). The *lift objective* selector makes that explicit instead
of guessing, and defaults from the design kind.

**Placed physics** (DESIGN.md 10). Heat-transfer and fan results add their
rows to the table (mean Nusselt number with its standard error, heat rate,
fan thrust, volume flow and ideal power, from :mod:`aerolume.analysis.physics`
through ``compare_rows``), the same quantities become bar-chart choices, and
the Fields tab offers Temperature (and its difference to the baseline) when a
compared result stores a temperature field. A result without them shows a dash
in those rows, never a zero.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.units import sig
from ...model import CFD3DResult, CFDResult, PolarResult, WingResult
from .. import theme
from ..jobs import JobManager, require_api
from ..widgets.common import ToolStrip
from ..widgets.field_view import FieldView, body_window, colormap_names, display_range, field_info
from ..widgets.kpi import (
    MINUS,
    Cell,
    DataTable,
    EmptyState,
    WarningLine,
    button,
    fmt_re,
    fmt_sig,
    fmt_speed,
    label,
    subscripts,
    value_error_texts,
)
from ..widgets.plots import (
    BarChart,
    CpPlot,
    GeometryPlot,
    HistoryPlot,
    MultiHistoryPlot,
    PolarGrid,
    lazy_import,
    series_color,
)
from ..widgets.wing_plots import SpanLoadingPlots, WingPolarGrid, nearest_point
from .results import (
    body_polygons,
    cp_range,
    finite,
    ground_of,
    is_lifting,
    point_curves,
    sim_kind,
    surface_curves,
)

log = logging.getLogger(__name__)

LIFT_GOALS = (("Neutral", None), ("More lift is better", "higher"), ("More downforce is better", "lower"))
BAR_METRICS = ("cd", "cl", "ld", "drag", "lift", "drag_area", "power", "cl_max", "cd_min", "ld_max",
               "cs", "side", "cm", "cd0", "oswald", "e_cruise",
               # placed physics (DESIGN.md 10): offered only when a compared result has them
               "nusselt", "heat_rate", "t_fluid_max", "t_fluid_mean", "vent_heat_out", "fan_thrust", "fan_flow",
               "fan_power")
#: Fields every 2D Navier-Stokes result can be compared on, and those only some carry (heat transfer).
COMPARE_FIELDS = ("speed", "cp", "vorticity", "p")
OPTIONAL_COMPARE_FIELDS = ("temperature",)
TAB_KEYS = ("chart", "geometry", "pressure", "polars", "fields", "forces", "surface", "wing_polars", "span")
#: legacy action names (older menus / tests) -> tab key
TAB_ALIASES = {"overview": "chart"}
CONTEXT_KEYS = ("re_real", "re_sim", "converged")
#: Tabs that apply to each comparison group (see the module docstring)
GROUP_TABS = {"2d": ("chart", "geometry", "pressure", "polars", "fields"),
              "cfd3d": ("chart", "forces", "surface"),
              "wing": ("chart", "wing_polars", "span")}
#: What a group is called in messages and in the simulation list
GROUP_NAMES = {"2d": "2D", "cfd3d": "3D CFD", "wing": "wing"}
#: Width of the Simulations list [px]: wide enough for "CFD 3D · 30 m/s · α 0° — Sphere 0.2 m"
SIM_LIST_W = 260
#: The metrics table gets at least this share of the page height when its rows do not all fit
#: ("table first", ui_guidelines 4) ...
TABLE_MIN_FRACTION = 0.45
#: ... while the plots below keep at least this many pixels (tab bar, tool strip, a readable chart)
PLOTS_MIN_H = 260
#: settings key of the user's own table / plots split (a fraction of the page height)
SPLIT_KEY = "compare/table_fraction"
KIND_TEXT = {"polar": "airfoil polar", "cfd": "Navier–Stokes 2D", "cfd3d": "Navier–Stokes 3D", "wing": "wing sweep"}


def compare_group(kind: str) -> str:
    """The comparison group of a simulation kind (:func:`.results.sim_kind`): "2d", "cfd3d" or "wing"."""
    return "2d" if kind in ("cfd", "polar") else kind


def result_group(result) -> str:
    """The comparison group of a loaded result."""
    if isinstance(result, CFD3DResult):
        return "cfd3d"
    if isinstance(result, WingResult):
        return "wing"
    return "2d"


def mixed_message(groups: dict[str, list]) -> str:
    """Why a selection that mixes kinds is not compared (one sentence per kind present)."""
    parts = []
    if "2d" in groups:
        parts.append("2D coefficients are per unit span")
    if "cfd3d" in groups:
        parts.append("3D CFD coefficients are on the body's reference area")
    if "wing" in groups:
        parts.append("a wing sweep has no single operating point")
    return ("These results are of different kinds: " + ", ".join(parts) + ". Tick results of one kind to "
            "compare them.")


def _fallback_rows(results: dict, baseline_id: str) -> list[dict]:
    """Minimal compare_rows for when aerolume.analysis is unavailable (CFD Cd / Cl / L/D only)."""
    specs = [("cd", "Drag coefficient Cd", "lower", lambda r: r.cd.mean),
             ("cl", "Lift coefficient Cl", None, lambda r: r.cl.mean),
             ("ld", "Lift-to-drag ratio L/D", "higher", lambda r: r.lift_to_drag),
             ("drag", "Drag", "lower", lambda r: r.drag)]
    rows = []
    for key, lab, better, get in specs:
        vals = {sid: (get(r) if isinstance(r, CFDResult) else None) for sid, r in results.items()}
        b = vals.get(baseline_id)
        cells = []
        for sid, v in vals.items():
            delta = v - b if finite(v) and finite(b) else None
            pct = 100 * delta / abs(b) if delta is not None and b else None
            cells.append({"sim_id": sid, "is_baseline": sid == baseline_id, "value": v, "stderr": None,
                          "delta": delta, "pct": pct, "improved": None, "significant": None})
        rows.append({"key": key, "label": lab, "unit": "N/m" if key == "drag" else "", "better": better,
                     "kind": "cfd", "baseline": b, "cells": cells})
    return rows


def comparison_rows(results: dict, baseline_id: str) -> list[dict]:
    """``analysis.compare_rows`` (shared with the report), or a small local fallback."""
    an = lazy_import("aerolume.analysis")
    if an is not None and hasattr(an, "compare_rows"):
        try:
            return an.compare_rows(results, baseline_id)
        except Exception:
            log.exception("analysis.compare_rows failed; using the minimal table")
    return _fallback_rows(results, baseline_id)


def comparison_warnings(results: dict, conditions: dict) -> list[str]:
    """``analysis.comparison_warnings`` (the report prints the same sentences), or none."""
    an = lazy_import("aerolume.analysis")
    fn = getattr(an, "comparison_warnings", None) if an is not None else None
    if fn is None:
        return []
    try:
        return list(fn(results, conditions))
    except Exception:
        log.exception("analysis.comparison_warnings failed")
        return []


def default_baseline(ids: list[str], analysis_of: dict[str, str]) -> str | None:
    """The baseline a comparison starts with: the first run of the kind most of them are.

    A polar and a CFD run share almost no quantities (a polar has no single Cd),
    so a polar baseline against CFD variants gives a table of dashes; with a
    tie, CFD wins (its table has error bars and the flow fields).
    """
    if not ids:
        return None
    n_cfd = sum(1 for s in ids if analysis_of.get(s) != "polar")
    kind = "cfd" if n_cfd >= len(ids) - n_cfd else "polar"
    return next((s for s in ids if (analysis_of.get(s) == "polar") == (kind == "polar")), ids[0])


def adapt_rows(rows: list[dict], lifting: dict[str, bool]) -> list[dict]:
    """Fit the shared comparison rows to the compared designs (``lifting``: sim id -> airfoil?).

    L/D is a figure of merit for lifting sections only: for a car or a bluff
    body it is a number nobody optimises (the cylinder's −0.87), so its row is
    dropped when no compared design is an airfoil, and its cells are blanked
    for the non-lifting ones in a mixed comparison. Without an airfoil the
    lift row says how to read its sign.
    """
    if not lifting or all(lifting.values()):
        return rows
    out = []
    for row in rows:
        if row.get("key") == "ld":
            if not any(lifting.values()):
                continue
            row = dict(row)
            row["cells"] = [c if lifting.get(c.get("sim_id"), True) else
                            {**c, "value": None, "delta": None, "pct": None, "stderr": None, "significant": None}
                            for c in row["cells"]]
        elif row.get("key") == "lift" and not any(lifting.values()):
            row = {**row, "label": "Lift (downforce if negative)"}
        out.append(row)
    return out


def judge(row: dict, cell: dict, lift_goal: str | None) -> bool | None:
    """Is this cell an improvement on the baseline? None = no preferred direction / no change."""
    better = lift_goal if row["key"] in ("cl", "lift") else row.get("better")
    delta, pct = cell.get("delta"), cell.get("pct")
    if not better or delta is None or delta == 0 or cell.get("is_baseline") or row["key"] in CONTEXT_KEYS:
        return None
    if pct is not None and abs(pct) < 0.05:  # displays as +0.0 %: colouring it would be noise
        return None
    return delta < 0 if better == "lower" else delta > 0


def fmt_value(row: dict, v, *, lower_bound: bool = False, stderr=None) -> str:
    """A table value without its unit (the unit has its own column).

    ``lower_bound`` (a polar whose lift was still rising at the last angle)
    prints "≥ 1.547": the number is the end of the sweep, not Cl,max. With a
    known ``stderr`` the value is rounded to it (:func:`value_error_texts`, the
    Summary's rule): the fan room's flow read "0.07915" here and "0.079 ± 0.028"
    on its Summary.
    """
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if row["key"].startswith("re_"):
        return fmt_re(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "—"
    se = _f(stderr)
    if math.isfinite(se) and se > 0 and row["key"] not in CONTEXT_KEYS:
        text = value_error_texts(f, se)[0]
        return f"≥ {text}" if lower_bound else text
    # four significant figures, × 10ⁿ below 10⁻³ (the Summary's and the API's rounding), not "3.110e-07"
    text = fmt_sig(f, 4)
    return f"≥ {text}" if lower_bound else text


#: A difference is coloured (better / worse) only when it exceeds this many combined standard errors.
SIGNIFICANCE_SIGMAS = 2.0


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def thrust_is_set(sim) -> bool:
    """Whether every enabled fan of ``sim`` imposes its thrust (mode "thrust"): the run's total fan
    thrust is then an input, not a measurement."""
    fans = [f for f in getattr(sim, "fans", None) or [] if getattr(f, "enabled", True)]
    return bool(fans) and all(getattr(f, "mode", "") == "thrust" for f in fans)


def is_set_value(row: dict, sim) -> bool:
    """Whether a compare row's value is a setting of ``sim`` rather than a result: an input row
    (``row["input"]``, the heater power) or the total thrust of fans that impose theirs."""
    if row.get("input"):
        return True
    return row.get("key") == "fan_thrust" and sim is not None and thrust_is_set(sim)


#: Why a Δ cell of a setting shows "—".
SET_DELTA_WHY = "a setting of the runs, not a result: a change there is a change of the case, not an effect"


def delta_state(row: dict, cell: dict, base_cell: dict, *, comparable: bool = True,
                is_set: bool = False) -> tuple[str, str]:
    """("pct" | "none" | "noise" | "significant", reason) of one Δ cell.

    * "none": no percentage is shown ("—"), because the designs are not
      variants of one object (``comparable`` False: a car against a sphere
      gave "+37206.2 %"), or the baseline is zero within its own standard
      error (a percentage of noise: "+100.0 %" of a zero side force);
    * "noise": the difference is within :data:`SIGNIFICANCE_SIGMAS` combined
      standard errors, shown but never coloured;
    * "significant": larger than that, coloured better / worse;
    * "pct": no error is known on either side (a polar), coloured as before.
    """
    if row.get("key") in CONTEXT_KEYS or cell.get("is_baseline"):
        return "none", ""
    if is_set or row.get("input"):
        return "none", SET_DELTA_WHY
    if not comparable:
        return "none", "different designs: a percentage of one object's value says nothing about another"
    b, se_b = _f(base_cell.get("value")), _f(base_cell.get("stderr"))
    d, se_c = _f(cell.get("delta")), _f(cell.get("stderr"))
    if not math.isfinite(b) or not math.isfinite(d):
        return "none", ""
    # a temperature in °C near zero is a temperature, not a zero baseline: its Δ is a difference in K
    if not is_interval_row(row) and (b == 0 or (math.isfinite(se_b) and se_b > 0 and abs(b) <= se_b)):
        return "none", "the baseline is zero within its standard error: a percentage of it is noise"
    errs = [e for e in (se_b, se_c) if math.isfinite(e) and e > 0]
    if not errs:
        return "pct", ""
    sigma = math.sqrt(sum(e * e for e in errs))
    if abs(d) <= SIGNIFICANCE_SIGMAS * sigma:
        return "noise", f"within {SIGNIFICANCE_SIGMAS:g} combined standard errors (± {sig(sigma, 2)}) of the two runs"
    return "significant", f"more than {SIGNIFICANCE_SIGMAS:g} combined standard errors (± {sig(sigma, 2)})"


def comparable_designs(a, b) -> bool:
    """Whether two designs are variants of one object, so a percentage between them means something.

    Variants share their kind (vehicle, airfoil, wing, body) and their size
    (reference lengths within a factor of two): three car shapes are compared
    in percent, a car and a 10 cm sphere are not.
    """
    if a is None or b is None or a is b:
        return True
    if getattr(a, "kind", None) != getattr(b, "kind", None):
        return False
    la, lb = _f(getattr(a, "length_ref", None)), _f(getattr(b, "length_ref", None))
    if math.isfinite(la) and math.isfinite(lb) and la > 0 and lb > 0:
        return max(la, lb) / min(la, lb) <= 2.0
    return True


def default_selection(avail: list[str], kind: dict[str, str], design_of: dict[str, str],
                      current: str | None, comparable=None) -> list[str]:
    """The simulations Compare ticks when nothing is ticked yet, at most four, or none.

    Runs of the same kind on one design (the current simulation's first, else the design with the most
    runs); failing that, runs of the same kind on designs that are variants of one object
    (``comparable(a, b)``, :func:`comparable_designs`: three airfoils, three car shapes); none
    otherwise. Ticking unrelated 3D runs (a car, a hot sphere, a heated box) filled the first view with
    "+37206.2 %" in red and green.
    """
    def pick_for(sid: str, same_design: bool) -> list[str]:
        out = [s for s in avail if kind[s] == kind[sid]
               and (design_of.get(s) == design_of.get(sid) if same_design
                    else comparable is None or comparable(s, sid))]
        return [sid] + [s for s in out if s != sid]

    for same_design in (True, False):
        if current in avail:
            pick = pick_for(current, same_design)
            if len(pick) >= 2:
                return pick[:4]         # the current run first: it becomes the baseline
        best: list[str] = []
        for sid in avail:
            pick = pick_for(sid, same_design)
            if len(pick) > len(best):
                best = pick
        if len(best) >= 2:
            return best[:4]
    return []


def fmt_pct(pct: float | None) -> str:
    """'+4.2 %' / '−10.0 %' with a typographic minus; an em dash when there is no change to show."""
    if pct is None or not math.isfinite(pct):
        return "—"
    return f"{pct:+.1f} %".replace("-", MINUS)


#: Rows on an interval scale (temperatures in °C): a percentage of 22 °C means nothing, so their
#: Δ column shows the difference itself, in kelvin (``analysis.compare.INTERVAL_KEYS``).
try:
    from ...analysis.compare import INTERVAL_KEYS
except Exception:  # noqa: BLE001 - an older analysis package: the two temperature rows it would have
    INTERVAL_KEYS = frozenset({"t_fluid_max", "t_fluid_mean"})


def is_interval_row(row: dict) -> bool:
    """Whether a compare row is on an interval scale (its Δ is a difference, never a percentage)."""
    return row.get("key") in INTERVAL_KEYS


def fmt_delta_kelvin(delta: float | None) -> str:
    """'+1.80 K' / '−0.42 K': the difference of two temperatures (a temperature difference is in K)."""
    if delta is None or not math.isfinite(delta):
        return "—"
    text = sig(abs(delta), 3)
    return f"{'+' if delta >= 0 else MINUS}{text} K"


class ComparePage(QWidget):
    """Side-by-side comparison of simulations with results (DESIGN.md 5.8).

    ``handle_action`` accepts the tab keys "chart", "geometry", "pressure",
    "polars" and "fields" (and "overview" for the chart).
    """

    def __init__(self, state, jobs: JobManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        require_api(jobs, ("result",), "The Compare page")
        self.state = state
        self.jobs = jobs
        self._results: dict[str, object] = {}
        self._rows: list[dict] = []
        self._colour: dict[str, int] = {}
        self._thumb_views: list[FieldView] = []
        self._goal_auto = True
        self._defaulted = False              # the default selection was made for this project
        self._field_chosen = False          # the user picked a field in the Fields tab
        self._group = "2d"                  # comparison group of the results shown (or "mixed")
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
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self.stack.addWidget(content)

        strip = ToolStrip()
        strip.add_widget(label("Baseline", "secondary"))
        self.baseline_combo = QComboBox()
        self.baseline_combo.setMinimumWidth(200)
        self.baseline_combo.setToolTip("Every change in the table is measured against this simulation")
        strip.add_widget(self.baseline_combo)
        strip.row.addSpacing(12)
        strip.add_widget(label("Lift objective", "secondary"))
        self.goal_combo = QComboBox()
        for text, val in LIFT_GOALS:
            self.goal_combo.addItem(text, val)
        self.goal_combo.setToolTip("Decides whether a change in lift is shown as an improvement")
        strip.add_widget(self.goal_combo)
        strip.add_stretch()
        self.noise_note = label("", "muted")
        strip.add_widget(self.noise_note)
        strip.row.addSpacing(4)
        cl.addWidget(strip)
        # why the compared runs may differ for reasons other than the design (different
        # speeds, angles, fluids or simulated Reynolds numbers): the report's callout, one line
        self.condition_note = WarningLine(details=False)
        self.condition_note.setContentsMargins(8, 4, 8, 4)
        self.condition_note.hide()
        cl.addWidget(self.condition_note)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        cl.addWidget(split, 1)

        # left: the simulations to compare
        left = QFrame()
        left.setProperty("panel", "true")
        left.setMinimumWidth(200)
        left.setMaximumWidth(400)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(4)
        ll.addWidget(label("Simulations", "h3"))
        self.sim_list = QListWidget()
        self.sim_list.setProperty("pane", "true")
        self.sim_list.setToolTip("Simulations with a result; ticked ones are compared")
        # long names lose their middle ("CFD 3D · 30 m/s … Sphere 0.2 m"), never their end, and the
        # list never scrolls sideways (the full name is in each item's tooltip)
        self.sim_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.sim_list.setWordWrap(False)
        self.sim_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ll.addWidget(self.sim_list, 1)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.all_btn = button("Select all", "link")
        self.none_btn = button("Clear", "link")
        row.addWidget(self.all_btn)
        row.addWidget(self.none_btn)
        row.addStretch(1)
        ll.addLayout(row)
        split.addWidget(left)

        # right: table above, plots below
        self.inner = QStackedWidget()
        self.inner_empty = EmptyState()
        self.inner.addWidget(self.inner_empty)
        body = QSplitter(Qt.Orientation.Vertical)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(1)
        self.body_split = body
        self.inner.addWidget(body)
        split.addWidget(self.inner)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([SIM_LIST_W, 1300])

        self.table = DataTable(["Quantity"])
        self.table.setProperty("flat", "true")
        self.table.setMinimumHeight(120)
        body.addWidget(self.table)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        body.addWidget(self.tabs)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([340, 520])
        # the user's split, once they drag it (splitterMoved is only emitted for user moves)
        self._table_fraction = self._stored_fraction()
        body.splitterMoved.connect(self._on_split_moved)
        self._split_timer = QTimer(self)
        self._split_timer.setSingleShot(True)
        self._split_timer.setInterval(0)
        self._split_timer.timeout.connect(self.fit_table_split)
        body.installEventFilter(self)
        self._build_chart()
        self._build_geometry()
        self._build_pressure()
        self._build_polars()
        self._build_fields()
        self._build_forces3d()
        self._build_surface3d()
        self._build_wing_polars()
        self._build_span()
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ---------------------------------------------------------- table split
    @staticmethod
    def _stored_fraction() -> float | None:
        try:
            from .. import settings as app_settings
            v = float(app_settings.get().value(SPLIT_KEY, 0.0) or 0.0)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return None
        return v if 0.05 < v < 0.95 else None

    def _on_split_moved(self, *_args) -> None:
        """Remember the user's table / plots split (as a fraction, so it survives a resize)."""
        sizes = self.body_split.sizes()
        total = sum(sizes)
        if total <= 0:
            return
        self._table_fraction = sizes[0] / total
        try:
            from .. import settings as app_settings
            app_settings.get().set_value(SPLIT_KEY, round(self._table_fraction, 4))
        except (AttributeError, RuntimeError):
            log.debug("could not store the Compare split", exc_info=True)

    def table_rows_height(self) -> int:
        """Pixels the metrics table needs to show every row without scrolling."""
        t = self.table
        return t.horizontalHeader().sizeHint().height() + t.rowCount() * theme.ROW_H + 2 * t.frameWidth() + 2

    def fit_table_split(self) -> None:
        """Table first: give the metrics table its rows, or at least 45 % of the page.

        The table had about 110 px (four rows, the pitching moment already cut)
        under a 350 px bar chart. Now it gets every row when they fit beside a
        usable plot area (PLOTS_MIN_H), otherwise at least TABLE_MIN_FRACTION of
        the height and a scroll bar; no empty well below a short table. A split
        the user dragged is kept, as a fraction of the height.
        """
        body = self.body_split
        total = body.height() - body.handleWidth()
        if total <= 0 or self.inner.currentWidget() is not body or not body.isVisible():
            return
        if self._table_fraction is not None:
            want = round(self._table_fraction * total)
        else:
            rows = self.table_rows_height()
            want = rows if rows <= total - PLOTS_MIN_H else max(total - PLOTS_MIN_H,
                                                                 round(TABLE_MIN_FRACTION * total))
        want = max(min(want, total - 120), min(self.table.minimumHeight(), total // 2))
        if body.sizes()[0] != want:
            body.setSizes([want, total - want])

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # the page split follows the splitter's own size (a dock opening changes it without
        # resizing the page): once its layout has settled, hence the deferred call
        if obj is self.body_split and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._split_timer.start()
        return super().eventFilter(obj, event)

    @staticmethod
    def _tab(strip: ToolStrip | None, w: QWidget) -> QWidget:
        """A tab page: the tool strip over the content, the content in a scroll area.

        The plots keep their minimum heights inside the scroll area instead of
        imposing them on the page: the Forces tab's three panels asked for
        513 px, and the splitter honoured that by squeezing the metrics table
        to its 120 px minimum (four rows) whatever the user wanted.
        """
        from ..widgets.results3d import scroll_page
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if strip is not None:
            lay.addWidget(strip)
        lay.addWidget(w if isinstance(w, QScrollArea) else scroll_page(w), 1)
        return host

    def _build_chart(self) -> None:
        strip = ToolStrip()
        strip.add_widget(label("Quantity", "secondary"))
        self.bar_metric = QComboBox()
        self.bar_metric.setMinimumWidth(200)
        strip.add_widget(self.bar_metric)
        strip.add_stretch()
        self.bars = BarChart()
        self.bars.setMinimumHeight(180)
        self.tabs.addTab(self._tab(strip, self.bars), "Chart")

    def _build_geometry(self) -> None:
        self.geo_plot = GeometryPlot(legend=True)
        self.geo_plot.setMinimumHeight(180)
        self.geo_plot.setToolTip("Outlines as solved: wind frame, rotated by the angle of each simulation")
        self.tabs.addTab(self._tab(None, self.geo_plot), "Geometry")

    def _build_pressure(self) -> None:
        strip = ToolStrip()
        self.cp_alpha_label = strip.add_widget(label("Polar angle", "secondary"))
        self.cp_alpha = QComboBox()
        strip.add_widget(self.cp_alpha)
        strip.add_stretch()
        strip.add_widget(label("Upper surface solid, lower surface dashed", "muted"))
        strip.row.addSpacing(4)
        self.cp_plot = CpPlot()
        self.cp_plot.setMinimumHeight(180)
        self.tabs.addTab(self._tab(strip, self.cp_plot), "Pressure")

    def _build_polars(self) -> None:
        self.polar_grid = PolarGrid()
        self.tabs.addTab(self._tab(None, self.polar_grid), "Polars")

    def _build_fields(self) -> None:
        strip = ToolStrip()
        strip.add_widget(label("Field", "secondary"))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(170)
        for key in COMPARE_FIELDS:
            self.field_combo.addItem(field_info(key)[0], key)
        strip.add_widget(self.field_combo)
        strip.add_separator()
        self.same_range = self._toggle(strip, "Same range", "One colour range for every view, so colours "
                                       "compare directly between simulations")
        self.same_range.setChecked(True)
        self.diff_check = self._toggle(strip, "Difference to baseline",
                                       "Each variant minus the baseline, on the baseline's grid: "
                                       "shows where the flow changed")
        strip.add_stretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.thumbs_host = QWidget()
        self.thumbs = QGridLayout(self.thumbs_host)
        self.thumbs.setContentsMargins(4, 4, 4, 4)
        self.thumbs.setSpacing(4)
        scroll.setWidget(self.thumbs_host)
        self.tabs.addTab(self._tab(strip, scroll), "Fields")

    def _build_forces3d(self) -> None:
        strip = ToolStrip()
        self.forces_caption = label("", "muted")
        strip.add_widget(self.forces_caption)
        strip.add_stretch()
        self.forces3d = MultiHistoryPlot(keys=("cd", "cl", "cm"), labels=HistoryPlot.LABELS_3D)
        self.tabs.addTab(self._tab(strip, self.forces3d), "Forces")

    def _build_surface3d(self) -> None:
        # the tiles own OpenGL contexts: built the first time the tab is opened
        self.surface_host = QWidget()
        lay = QVBoxLayout(self.surface_host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.surface_grid = None
        self._surface_dirty = True
        self.tabs.addTab(self.surface_host, "Surface")

    def _build_wing_polars(self) -> None:
        self.wing_polars = WingPolarGrid()
        self.tabs.addTab(self._tab(None, self.wing_polars), "Polars")

    def _build_span(self) -> None:
        strip = ToolStrip()
        strip.add_widget(label("Angle of attack", "secondary"))
        self.span_alpha = QComboBox()
        self.span_alpha.setMinimumWidth(90)
        strip.add_widget(self.span_alpha)
        strip.add_stretch()
        strip.add_widget(label("Dashed: elliptic loading of the baseline's CL", "muted"))
        strip.row.addSpacing(4)
        self.span_plots = SpanLoadingPlots()
        self.tabs.addTab(self._tab(strip, self.span_plots), "Span loading")

    @staticmethod
    def _toggle(strip: ToolStrip, text: str, tip: str) -> QToolButton:
        b = strip.add_button(None, text, tip, checkable=True)
        b.setProperty("variant", "toggle")
        return b

    def _connect(self) -> None:
        st = self.state
        st.projectChanged.connect(self._on_project)
        st.simulationsChanged.connect(self.refresh)
        st.resultChanged.connect(lambda *_: self.refresh())
        self.sim_list.itemChanged.connect(self._on_item_changed)
        self.baseline_combo.activated.connect(lambda *_: self._update_views())
        self.goal_combo.activated.connect(self._on_goal_chosen)
        self.all_btn.clicked.connect(lambda: self._set_all(True))
        self.none_btn.clicked.connect(lambda: self._set_all(False))
        self.bar_metric.currentIndexChanged.connect(self._fill_bars)
        self.cp_alpha.currentIndexChanged.connect(self._fill_cp)
        self.field_combo.currentIndexChanged.connect(self._fill_fields)
        self.field_combo.activated.connect(lambda *_: setattr(self, "_field_chosen", True))
        self.same_range.toggled.connect(self._fill_fields)
        self.diff_check.toggled.connect(self._fill_fields)
        self.span_alpha.currentIndexChanged.connect(self._fill_span)

    # ------------------------------------------------------------- page API
    def _on_project(self, *_) -> None:
        self._defaulted = False
        self.refresh()

    def on_shown(self) -> None:
        self.refresh()

    def handle_action(self, action: str) -> None:
        key = TAB_ALIASES.get(action, action)
        if key == "polars" and not self.tabs.isTabVisible(TAB_KEYS.index("polars")):
            key = "wing_polars"
        if key == "fields" and not self.tabs.isTabVisible(TAB_KEYS.index("fields")):
            key = "surface"
        if key in TAB_KEYS and self.tabs.isTabVisible(TAB_KEYS.index(key)):
            self.tabs.setCurrentIndex(TAB_KEYS.index(key))

    # -------------------------------------------------------------- refresh
    def _available(self) -> list[str]:
        p = self.state.project
        if p is None:
            return []
        has = getattr(p, "has_result", None)
        out = []
        for sid, sim in p.simulations.items():
            ok = has(sid) if callable(has) else sim.status in ("done", "cancelled")
            if ok and sim.status not in ("running", "queued"):
                out.append(sid)
        return out

    def refresh(self) -> None:
        p = self.state.project
        if p is None:
            self.empty.set_content("No project open", "", "Open or create a project",
                                   lambda: self.state.navigate("start"))
            self.stack.setCurrentIndex(0)
            return
        avail = self._available()
        if len(avail) < 2:
            self.empty.set_content(
                "Nothing to compare",
                "A comparison needs two simulations with results." + (" One is ready." if avail else ""),
                "New simulation…" if self.state.design is not None else "Open Setup",
                lambda: self.state.navigate("setup:new" if self.state.design is not None else "setup"),
                secondary_text="Open Simulate" if p.simulations else "",
                secondary_callback=lambda: self.state.navigate("simulate"))
            self.stack.setCurrentIndex(0)
            return
        self.stack.setCurrentIndex(1)
        ids = self.state.compare_ids
        ids[:] = [s for s in ids if s in avail]
        kind = {s: self._kind(s) for s in avail}
        if not ids and not self._defaulted:
            # Like with like: runs of one design and one kind (default_selection), or none. Once per
            # project: a user who unticked everything is not re-ticked on the next visit.
            self._defaulted = True
            cur = self.state.simulation
            ids.extend(default_selection(avail, kind, {s: p.simulations[s].design_id for s in avail},
                                         cur.id if cur is not None else None, comparable=self._comparable))
        elif len(ids) == 1:
            # "Add to comparison" from Results selects one run: pair it with the most similar
            # other result (same kind first, then same design) so the page opens on a
            # comparison, not a hint
            s0 = p.simulations[ids[0]]
            others = [s for s in avail if s != ids[0]]

            def similarity(sid: str) -> tuple[bool, bool, bool]:
                o = p.simulations[sid]
                return (compare_group(kind[sid]) == compare_group(kind[ids[0]]), kind[sid] == kind[ids[0]],
                        o.design_id == s0.design_id)
            ids.append(max(others, key=similarity))     # max keeps the first of equals
        many_designs = len({p.simulations[s].design_id for s in avail}) > 1
        self.sim_list.blockSignals(True)
        self.sim_list.clear()
        for sid in avail:
            sim = p.simulations[sid]
            d = p.designs.get(sim.design_id)
            text = f"{sim.name} — {d.name}" if many_designs and d is not None else sim.name
            item = QListWidgetItem(text)
            item.setToolTip(f"{text}\n{d.name if d else '?'} · {fmt_speed(sim.conditions.speed, both=False)} · "
                            f"α {sig(sim.conditions.alpha_deg, 3)}° · {KIND_TEXT.get(kind[sid], sim.analysis)}")
            item.setData(Qt.ItemDataRole.UserRole, sid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if sid in ids else Qt.CheckState.Unchecked)
            self.sim_list.addItem(item)
        self.sim_list.blockSignals(False)
        if self._goal_auto:
            kinds = {p.designs[p.simulations[s].design_id].kind for s in ids
                     if p.simulations[s].design_id in p.designs}
            goal = "higher" if kinds and kinds <= {"airfoil", "wing"} else None
            self.goal_combo.setCurrentIndex(self.goal_combo.findData(goal))
        self._update_views()

    def _on_goal_chosen(self, *_) -> None:
        self._goal_auto = False          # the user decided: stop deriving it from the design kind
        self._fill_table()
        self._fill_bars()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.ItemDataRole.UserRole)
        ids = self.state.compare_ids
        if item.checkState() == Qt.CheckState.Checked and sid not in ids:
            ids.append(sid)
        elif item.checkState() != Qt.CheckState.Checked and sid in ids:
            ids.remove(sid)
        self._update_views()

    def _set_all(self, on: bool) -> None:
        self.sim_list.blockSignals(True)
        ids = self.state.compare_ids
        ids.clear()
        for i in range(self.sim_list.count()):
            it = self.sim_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            if on:
                ids.append(it.data(Qt.ItemDataRole.UserRole))
        self.sim_list.blockSignals(False)
        self._update_views()

    def _name(self, sid: str) -> str:
        """The simulation's name, with its design's name when that alone would be ambiguous.

        New studies name runs after their conditions ("CFD · 30 m/s"), so three car variants
        compared at the same speed would otherwise give three identical column headers and
        bar labels.
        """
        p = self.state.project
        if p is None or sid not in p.simulations:
            return sid
        sim = p.simulations[sid]
        clash = any(o != sid and o in p.simulations and p.simulations[o].name == sim.name
                    for o in self.state.compare_ids)
        design = p.designs.get(sim.design_id)
        if clash and design is not None:
            return f"{design.name} · {sim.name}"
        return sim.name

    def _analysis_of(self, ids) -> dict[str, str]:
        p = self.state.project
        return {s: p.simulations[s].analysis for s in ids if p is not None and s in p.simulations}

    def _kind(self, sid: str) -> str:
        """"cfd", "polar", "cfd3d" or "wing" (:func:`.results.sim_kind`)."""
        p = self.state.project
        sim = p.simulations.get(sid) if p is not None else None
        return sim_kind(sim, p.designs.get(sim.design_id) if sim is not None else None)

    def _select_group(self, group: str, ids: list[str]) -> None:
        """Replace the ticked simulations by ``ids`` (one kind) and redraw."""
        self.state.compare_ids[:] = list(ids)
        self.refresh()

    def _update_views(self) -> None:
        ids = list(self.state.compare_ids)
        base = self.baseline_combo.currentData()
        if base not in ids:
            # no choice yet, or the chosen one was unticked: a baseline that can produce
            # deltas for most rows (never a polar against CFD variants)
            base = default_baseline(ids, self._analysis_of(ids))
        self.baseline_combo.blockSignals(True)
        self.baseline_combo.clear()
        for sid in ids:
            self.baseline_combo.addItem(self._name(sid), sid)
        self.baseline_combo.setCurrentIndex(max(self.baseline_combo.findData(base), 0))
        self.baseline_combo.blockSignals(False)
        self._results = {}
        for sid in ids:
            r = self.jobs.result(sid)
            if r is not None:
                self._results[sid] = r
        if len(self._results) < 2:
            self.inner_empty.set_content("Select two or more simulations", "Tick them in the list on the left.")
            self.inner.setCurrentIndex(0)
            self.noise_note.setText("")
            self.condition_note.set_warnings([])
            return
        groups: dict[str, list[str]] = {}
        for sid, r in self._results.items():
            groups.setdefault(result_group(r), []).append(sid)
        if len(groups) > 1:
            # never a table of dashes: say why, and offer each kind on its own
            # each choice takes every available result of its kind (a single one compares with nothing)
            avail: dict[str, list[str]] = {}
            for sid in self._available():
                avail.setdefault(compare_group(self._kind(sid)), []).append(sid)
            ranked = sorted((g for g in groups if len(avail.get(g, [])) >= 2), key=lambda g: -len(avail[g]))
            choices = [(g, avail[g][:4]) for g in ranked[:2]]
            first = choices[0] if choices else None
            second = choices[1] if len(choices) > 1 else None
            self.inner_empty.set_content(
                "These results cannot be compared", mixed_message(groups),
                f"Compare the {GROUP_NAMES[first[0]]} results ({len(first[1])})" if first else "",
                (lambda: self._select_group(*first)) if first else None,
                secondary_text=f"Compare the {GROUP_NAMES[second[0]]} results ({len(second[1])})" if second else "",
                secondary_callback=(lambda: self._select_group(*second)) if second else None)
            self.inner.setCurrentIndex(0)
            self.noise_note.setText("")
            self.condition_note.set_warnings([])
            self._group = "mixed"
            return
        self._group = next(iter(groups))
        self.inner.setCurrentIndex(1)
        # colours follow the order of the list, not the baseline choice, so a variant keeps its colour
        self._colour = {sid: i for i, sid in enumerate(self._results)}
        base = self.baseline_combo.currentData()
        if base not in self._results:
            base = (default_baseline(list(self._results), self._analysis_of(self._results))
                    if self._group == "2d" else None) or next(iter(self._results))
        ordered = {base: self._results[base], **{k: v for k, v in self._results.items() if k != base}}
        self._results = ordered
        lifting = {sid: is_lifting(self._context(sid)[1]) for sid in ordered}
        self._rows = adapt_rows(comparison_rows(ordered, base), lifting)
        sims = {sid: self._context(sid)[0] for sid in ordered if self._context(sid)[0] is not None}
        conds = {sid: s.conditions for sid, s in sims.items()}
        # the Simulation objects, not only their flow conditions: two closed rooms are then judged by
        # their driving inputs (heater power, fans, ports) instead of a "different speeds" warning
        # about their reference speeds (analysis.comparison_warnings; the report does the same)
        self.condition_note.set_warnings(comparison_warnings(ordered, sims))
        if not self._field_chosen and self.field_combo.currentData() == "speed":
            # speeds differ: with one colour range the slower run is uniformly dark, so the
            # Fields tab opens on the dimensionless Cp instead (until the user picks a field)
            speeds = {round(c.speed, 9) for sid, c in conds.items() if isinstance(ordered.get(sid), CFDResult)}
            if len(speeds) > 1:
                self.field_combo.blockSignals(True)
                self.field_combo.setCurrentIndex(max(self.field_combo.findData("cp"), 0))
                self.field_combo.blockSignals(False)
        any_polar = any(isinstance(r, PolarResult) for r in ordered.values())
        any_cfd = any(isinstance(r, CFDResult) for r in ordered.values())
        has_fields = any_cfd and any((r.mean_fields or r.instant_fields) is not None
                                     for r in ordered.values() if isinstance(r, CFDResult))
        allowed = set(GROUP_TABS[self._group])
        self.tabs.blockSignals(True)
        for i, key in enumerate(TAB_KEYS):
            ok = key in allowed
            if key == "polars":
                ok = ok and any_polar
            elif key == "fields":
                ok = ok and has_fields
            elif key == "surface":
                ok = ok and any(isinstance(r, CFD3DResult) and r.surface for r in ordered.values())
            self.tabs.setTabVisible(i, ok)
        if not self.tabs.isTabVisible(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(0)
        self.tabs.blockSignals(False)
        self._fill_table()
        self._fill_bar_choices()
        if self._group == "2d":
            self._fill_geometry()
            self._fill_cp_choices()
            self._fill_polars()
            self._fill_fields()
        elif self._group == "cfd3d":
            self._fill_forces3d()
            self._surface_dirty = True
            self._on_tab_changed(self.tabs.currentIndex())
        else:
            self._fill_wing_polars()
            self._fill_span_choices()

    # ----------------------------------------------------------------- table
    def table_columns(self) -> list[tuple[str, str]]:
        """(kind, sim id) per table column after "Quantity": ("value", id), ("delta", id), ..., ("unit", "")."""
        cols: list[tuple[str, str]] = []
        for i, sid in enumerate(self._results):
            cols.append(("value", sid))
            if i > 0:
                cols.append(("delta", sid))
        cols.append(("unit", ""))
        return cols

    def _fill_table(self) -> None:
        rows, ids = self._rows, list(self._results)
        if not ids:
            return
        goal = self.goal_combo.currentData()
        cols = self.table_columns()
        unconverged = self.unconverged_ids()
        headers = ["Quantity"]
        for kind, sid in cols:
            if kind == "value":
                headers.append(f"{self._name(sid)} (baseline)" if sid == ids[0] else self._name(sid))
            elif kind == "delta":
                headers.append("Δ")
            else:
                headers.append("Unit")
        out_rows = []
        n_noise = 0
        sims = self.state.project.simulations if self.state.project is not None else {}
        for row in rows:
            cells = {c["sim_id"]: c for c in row["cells"]}
            set_ids = {sid for sid in ids if is_set_value(row, sims.get(sid))}
            line: list = [Cell(row["label"], tooltip=self._row_tip(row))]
            for kind, sid in cols:
                c = cells.get(sid, {})
                if kind == "unit":
                    line.append(row.get("unit") or "")
                elif kind == "value":
                    se = c.get("stderr")
                    tip = f"Standard error ± {sig(se, 2)}" if se is not None and finite(se) else ""
                    if tip and c.get("stderr_reliable") is False:
                        tip += " (an estimate: too few independent samples to measure it)"
                    tone = None
                    if sid in unconverged and row["key"] not in CONTEXT_KEYS:
                        tone, tip = "muted", (tip + "; " if tip else "") + "not statistically converged"
                    bound = bool(c.get("lower_bound"))
                    if bound:
                        tone, tip = "muted", ("The lift was still rising at the last angle of the sweep: this is "
                                              "the last angle's value, a lower bound, not the stall")
                    text = fmt_value(row, c.get("value"), lower_bound=bound, stderr=se)
                    if sid in set_ids and not row.get("input") and text != "—":
                        text, tip = f"{text} (set)", "Set in Setup, not measured: it carries no averaging error"
                    line.append(Cell(text, tone=tone, tooltip=tip, bold=bool(c.get("is_baseline"))))
                else:
                    if row["key"] in CONTEXT_KEYS:
                        line.append(Cell("", tone="muted"))
                        continue
                    imp = judge(row, c, goal)
                    tone, tip = None, f"{self._name(sid)} relative to the baseline"
                    base_cell = next((x for x in row["cells"] if x.get("is_baseline")), {})
                    state, why = delta_state(row, c, base_cell, comparable=self._comparable(sid, ids[0]),
                                             is_set=bool(set_ids & {sid, ids[0]}))
                    if state == "none":
                        line.append(Cell("—", tone="muted", tooltip=tip + (f"; no percentage: {why}" if why else "")))
                        continue
                    if c.get("lower_bound") or base_cell.get("lower_bound"):
                        tone, tip = "muted", tip + "; not comparable: a stall was not reached inside the sweep"
                        imp = None
                    elif state == "noise" and c.get("delta"):
                        tone, tip = "muted", tip + f"; {why}"
                        n_noise += 1
                    elif imp is True:
                        tone, tip = "good", tip + "; an improvement" + (f", {why}" if why else "")
                    elif imp is False:
                        tone, tip = "bad", tip + "; worse than the baseline" + (f", {why}" if why else "")
                    if (state == "significant" and c.get("delta")
                            and (c.get("stderr_reliable") is False or base_cell.get("stderr_reliable") is False)):
                        tip += " (one of the error bars is only an estimate: a short averaging window)"
                    if is_interval_row(row):
                        line.append(Cell(fmt_delta_kelvin(_f(c.get("delta"))), tone=tone,
                                         tooltip=tip + "; a temperature difference in kelvin (a percentage of a "
                                                       "°C value means nothing)"))
                        continue
                    line.append(Cell(fmt_pct(c.get("pct")), tone=tone, tooltip=tip))
            out_rows.append(line)
        numeric = [j + 1 for j, (kind, _) in enumerate(cols) if kind != "unit"]
        self.table.set_data(headers, out_rows, numeric=numeric)
        warn_icon = theme.icon("warning", "warn")
        for j, (kind, sid) in enumerate(cols):
            item = self.table.horizontalHeaderItem(j + 1)
            if item is None:
                continue
            if kind == "delta":
                item.setToolTip(f"Change of “{self._name(sid)}” in percent of the baseline")
            elif kind == "value" and sid in unconverged:
                # a number that is not converged must not look as final as one that is
                item.setIcon(warn_icon)
                item.setToolTip(f"“{self._name(sid)}” is not statistically converged (or was stopped early): "
                                "its averages are less certain than the standard errors suggest. "
                                "Run it longer before relying on small differences.")
        self.noise_note.setText(f"{n_noise} change{'s' if n_noise != 1 else ''} within noise (grey)"
                                if n_noise else "")
        self.noise_note.setToolTip(
            f"A change is coloured only when it exceeds {SIGNIFICANCE_SIGMAS:g} combined standard errors of the two "
            "runs; smaller ones are grey. “—” in a Δ column: no meaningful percentage (a baseline that is zero "
            "within its error, or designs that are not variants of one object); its tooltip says which.")
        self.fit_table_split()

    def _comparable(self, sid: str, base: str) -> bool:
        """Whether ``sid``'s design is a variant of the baseline's (:func:`comparable_designs`)."""
        return comparable_designs(self._context(sid)[1], self._context(base)[1])

    def unconverged_ids(self) -> set[str]:
        """Compared Navier-Stokes runs whose averages are not converged, or that were stopped early."""
        p = self.state.project
        out = set()
        for sid, r in self._results.items():
            sim = p.simulations.get(sid) if p is not None else None
            if isinstance(r, (CFDResult, CFD3DResult)) and (not r.converged
                                                            or (sim is not None and sim.status == "cancelled")):
                out.add(sid)
        return out

    @staticmethod
    def _row_tip(row: dict) -> str:
        better = row.get("better")
        if row["key"] in ("cl", "lift"):
            return "Whether more is better is set by the lift objective"
        return {"lower": "Lower is better", "higher": "Higher is better"}.get(better or "", "")

    def delta_column(self, sim_id: str) -> int:
        """Table column of the change of ``sim_id`` against the baseline (-1 for the baseline)."""
        for j, (kind, sid) in enumerate(self.table_columns()):
            if kind == "delta" and sid == sim_id:
                return j + 1
        return -1

    # ----------------------------------------------------------------- chart
    def _fill_bar_choices(self) -> None:
        keys = [r["key"] for r in self._rows if r["key"] in BAR_METRICS]
        cur = self.bar_metric.currentData()
        self.bar_metric.blockSignals(True)
        self.bar_metric.clear()
        for r in self._rows:
            if r["key"] in keys:
                unit = f" [{r['unit']}]" if r.get("unit") else ""
                self.bar_metric.addItem(r["label"] + unit, r["key"])
        # open on a measured quantity: the first in table order that is no setting of any compared run (the
        # fan room opened on "Fan thrust (all fans)", the 1.000 N/m it was given)
        sims = self.state.project.simulations if self.state.project is not None else {}
        measured = [r["key"] for r in self._rows if r["key"] in keys
                    and not any(is_set_value(r, sims.get(c["sim_id"])) for c in r["cells"])]
        first = "cd" if "cd" in measured else (measured or keys or [None])[0]
        i = self.bar_metric.findData(cur if cur in keys else first)
        self.bar_metric.setCurrentIndex(max(i, 0))
        self.bar_metric.blockSignals(False)
        self._fill_bars()

    def _fill_bars(self, *_) -> None:
        key = self.bar_metric.currentData()
        row = next((r for r in self._rows if r["key"] == key), None)
        if row is None:
            self.bars.set_bars([], [])
            self.bars.set_labels(ylabel="")
            return
        unit = row.get("unit") or ""
        self.bars.set_labels(ylabel=subscripts(row["label"]) + (f" ({unit})" if unit else ""))
        labels, vals, errs, cols = [], [], [], []
        for cell in row["cells"]:
            sid = cell["sim_id"]
            labels.append(self._name(sid))
            v = cell.get("value")
            vals.append(v if v is not None else math.nan)
            errs.append(cell.get("stderr") or 0.0)
            # bars are data, so they use the series colour of the simulation, never a status colour
            cols.append(series_color(self._colour.get(sid, 0)).name())
        self.bars.set_bars(labels, vals, cols, errors=errs, fmt=lambda v: fmt_sig(v, 3))

    # -------------------------------------------------------------- overlays
    def _context(self, sid: str):
        p = self.state.project
        sim = p.simulations.get(sid) if p is not None else None
        design = p.designs.get(sim.design_id) if sim is not None else None
        return sim, design

    def _fill_geometry(self) -> None:
        self.geo_plot.clear_all()
        for sid, r in self._results.items():
            sim, design = self._context(sid)
            for k, poly in enumerate(body_polygons(r, design, sim)):
                self.geo_plot.add_outline(poly, series_color(self._colour.get(sid, 0)), width=1.5,
                                          name=self._name(sid) if k == 0 else None)
        self.geo_plot.autoRange()

    def _fill_cp_choices(self) -> None:
        alphas = sorted({p.alpha for r in self._results.values() if isinstance(r, PolarResult)
                         for p in r.points if p.converged and p.cp is not None})
        cur = self.cp_alpha.currentData()
        self.cp_alpha.blockSignals(True)
        self.cp_alpha.clear()
        for a in alphas:
            self.cp_alpha.addItem(f"{a:g}°", a)
        if alphas:
            want = cur if cur in alphas else min(alphas, key=lambda a: abs(a - 4.0))
            self.cp_alpha.setCurrentIndex(self.cp_alpha.findData(want))
        self.cp_alpha.blockSignals(False)
        self.cp_alpha.setVisible(bool(alphas))
        self.cp_alpha_label.setVisible(bool(alphas))
        self._fill_cp()

    def _fill_cp(self, *_) -> None:
        self.cp_plot.clear_series()
        want = self.cp_alpha.currentData()
        for sid, r in self._results.items():
            name = self._name(sid)
            col = series_color(self._colour.get(sid, 0))
            if isinstance(r, CFDResult) and r.surface:
                sim, design = self._context(sid)
                cur = surface_curves(r.surface[0], alpha_deg=sim.conditions.alpha_deg if sim else 0.0,
                                     about=design.moment_ref if design else (0, 0),
                                     length=design.length_ref if design else 1.0)
                up, lo = cur["upper"], cur["lower"]
            elif isinstance(r, PolarResult) and want is not None:
                pt = min((p for p in r.points if p.converged and p.cp is not None),
                         key=lambda p: abs(p.alpha - want), default=None)
                cur = point_curves(pt)
                if cur is None:
                    continue
                up, lo = cur["upper"], cur["lower"]
            else:
                continue
            self.cp_plot.set_series(name, up[0], up[1], color=col)
            self.cp_plot.set_series(f"{name} lower", lo[0], lo[1], color=col, style=Qt.PenStyle.DashLine,
                                    legend=False)
        self.cp_plot.autoRange()

    def _fill_polars(self) -> None:
        pol = [(self._name(sid), r.table(), self._colour.get(sid, 0)) for sid, r in self._results.items()
               if isinstance(r, PolarResult)]
        self.polar_grid.set_polars(pol)
        for p in self.polar_grid.plots():
            p.autoRange()

    # ----------------------------------------------------------- 3D and wing
    def _on_tab_changed(self, index: int) -> None:
        """Build the 3D surface tiles only when their tab is opened (each owns a 3D view)."""
        if (0 <= index < len(TAB_KEYS) and TAB_KEYS[index] == "surface" and self._surface_dirty
                and getattr(self, "_group", "") == "cfd3d"):
            self._surface_dirty = False
            self._fill_surface3d()

    def _fill_forces3d(self) -> None:
        items = [(sid, r) for sid, r in self._results.items() if isinstance(r, CFD3DResult)]
        keys = ["cd", "cl", "cm"]
        for k in ("cs", "c_roll", "cn"):          # zero by symmetry in half-domain runs: shown when not
            for _sid, r in items:
                a = np.asarray((r.history or {}).get(k, []), dtype=float)
                if not r.symmetry_plane and a.size and np.nanmax(np.abs(a)) > 0:
                    keys.append(k)
                    break
        order = ("cd", "cs", "cl", "c_roll", "cm", "cn")
        self.forces3d.set_keys(tuple(k for k in order if k in keys), HistoryPlot.LABELS_3D)
        self.forces3d.clear_runs()
        for sid, r in items:
            win = (r.t_avg_start, r.t_end) if finite(r.t_avg_start) and finite(r.t_end) else None
            self.forces3d.add_run(self._name(sid), r.history or {}, win, self._colour.get(sid, 0))
        self.forces_caption.setText("Time in convective units of each run; the thick tick under each trace "
                                    "is its averaging window.")

    def _fill_surface3d(self) -> None:
        if self.surface_grid is None:
            from ..widgets.results3d import SurfaceCompareGrid
            self.surface_grid = SurfaceCompareGrid()
            self.surface_host.layout().addWidget(self.surface_grid)
        # each run's result goes along: the tiles can add its mean-flow streamlines or smoke snapshot
        runs = [(self._name(sid) + ("  (baseline)" if i == 0 else ""), r.surface, r)
                for i, (sid, r) in enumerate(self._results.items()) if isinstance(r, CFD3DResult)]
        sim0, _d0 = self._context(next(iter(self._results)))
        self.surface_grid.set_runs(runs, ground=ground_of(sim0))

    def _fill_wing_polars(self) -> None:
        wings = [(self._name(sid), r, self._colour.get(sid, 0)) for sid, r in self._results.items()
                 if isinstance(r, WingResult)]
        self.wing_polars.set_wings(wings, split=False)

    def _fill_span_choices(self) -> None:
        alphas = sorted({round(float(p.alpha), 6) for r in self._results.values() if isinstance(r, WingResult)
                         for p in r.points if p.converged and p.y is not None})
        cur = self.span_alpha.currentData()
        self.span_alpha.blockSignals(True)
        self.span_alpha.clear()
        for a in alphas:
            self.span_alpha.addItem(f"{a:g}°".replace("-", MINUS), a)
        if alphas:
            want = cur if cur in alphas else min(alphas, key=lambda a: abs(a - 4.0))
            self.span_alpha.setCurrentIndex(self.span_alpha.findData(want))
        self.span_alpha.blockSignals(False)
        self._fill_span()

    def _fill_span(self, *_) -> None:
        a = self.span_alpha.currentData()
        pts = []
        for sid, r in self._results.items():
            if isinstance(r, WingResult) and a is not None:
                pt = nearest_point(r, a)
                if pt is not None and abs(pt.alpha - a) < 1e-6:
                    pts.append((self._name(sid), pt, float(r.span), self._colour.get(sid, 0)))
        self.span_plots.set_points(pts)

    # ----------------------------------------------------------------- fields
    def _fill_fields(self, *_) -> None:
        for v in self._thumb_views:
            self.thumbs.removeWidget(v)
            v.setParent(None)
            v.deleteLater()
        self._thumb_views = []
        items = [(sid, r) for sid, r in self._results.items() if isinstance(r, CFDResult)]
        self._sync_field_choices([(r.mean_fields or r.instant_fields) for _sid, r in items])
        name = self.field_combo.currentData()
        # the table's baseline is the first entry of _results; only a CFD baseline has a
        # field, so with a polar baseline no tile is "the baseline" and there is no difference
        base_id = next(iter(self._results), None)
        base_is_cfd = bool(items) and items[0][0] == base_id
        self.diff_check.setEnabled(base_is_cfd)
        self.diff_check.setToolTip("Each variant minus the baseline, on the baseline's grid: shows where the flow "
                                   "changed" if base_is_cfd else
                                   "Needs a Navier–Stokes baseline (the baseline is a polar, which has no field)")
        fds = {sid: (r.mean_fields or r.instant_fields) for sid, r in items}
        ranges = []
        _, _, div = field_info(name)
        for sid, r in items:
            fd = fds[sid]
            if fd is not None and name in fd.fields:
                sim, design = self._context(sid)
                # the Results page's rule: sampled around the body, magnitudes from 0
                win = body_window(body_polygons(r, design, sim), fd.extent)
                ranges.append(display_range(name, fd.fields[name], extent=fd.extent, mask=fd.mask, window=win))
        common = None
        if ranges and self.same_range.isChecked():
            lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
            common = (-max(abs(lo), abs(hi)), max(abs(lo), abs(hi))) if div else (lo, hi)
            if name == "cp":
                common = cp_range(common[0])
        cols = 2 if len(items) <= 4 else 3
        base_fd = fds.get(items[0][0]) if base_is_cfd else None
        diffs = self._differences(name, fds, base_fd) if (self.diff_check.isChecked() and base_is_cfd) else {}
        dmax = max((float(np.nanpercentile(np.abs(d), 99.5)) for d in diffs.values() if np.isfinite(d).any()),
                   default=0.0) or 1.0
        label_, unit, _ = field_info(name)
        for i, (sid, r) in enumerate(items):
            fd = fds[sid]
            view = FieldView(compact=True)
            view.setMinimumHeight(260 if cols == 2 else 220)
            view.set_title(self._name(sid) + ("  −  baseline" if sid in diffs else
                                              ("  (baseline)" if sid == base_id else "")))
            sim, design = self._context(sid)
            if fd is None or name not in fd.fields:
                view.set_placeholder("No field stored", "")
            else:
                polys = body_polygons(r, design, sim)
                if sid in diffs:
                    view.set_field(diffs[sid], base_fd.extent, field="v", cmap=self._diff_cmap(),
                                   vmin=-dmax, vmax=dmax, label=f"Δ {label_}", units=unit)
                    # both shapes are solid somewhere in a difference map: draw both
                    sim0, design0 = self._context(items[0][0])
                    view.set_bodies(polys + body_polygons(items[0][1], design0, sim0))
                else:
                    vmin, vmax = common if common else (None, None)
                    view.set_field(fd.fields[name], fd.extent, field=name, vmin=vmin, vmax=vmax, mask=fd.mask,
                                   window=body_window(polys, fd.extent))
                    view.set_bodies(polys)
                view.set_ground(ground_of(sim))
                view.fit()       # the default near-field framing, as on Simulate and Results
            self.thumbs.addWidget(view, i // cols, i % cols)
            self._thumb_views.append(view)
        for r in range(self.thumbs.rowCount()):
            self.thumbs.setRowStretch(r, 0)
        self.thumbs.setRowStretch(self.thumbs.rowCount(), 1)

    def _sync_field_choices(self, fds) -> None:
        """Offer Temperature (and any other optional field) when a compared result stores it; keep the choice."""
        have = [k for k in OPTIONAL_COMPARE_FIELDS if any(fd is not None and k in fd.fields for fd in fds)]
        want = list(COMPARE_FIELDS) + have
        now = [self.field_combo.itemData(i) for i in range(self.field_combo.count())]
        if now == want:
            return
        cur = self.field_combo.currentData()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for key in want:
            self.field_combo.addItem(field_info(key)[0], key)
        self.field_combo.setCurrentIndex(max(self.field_combo.findData(cur), 0))
        self.field_combo.blockSignals(False)

    @staticmethod
    def _diff_cmap() -> str | None:
        """The diverging map of the guidelines (vorticity's RdBu_r): white is "no change"."""
        names = colormap_names()
        return next((n for n in ("RdBu_r", "coolwarm") if n in names), None)

    @staticmethod
    def _differences(name: str, fds: dict, base_fd) -> dict[str, np.ndarray]:
        """sim id -> field minus the baseline's, on the baseline grid (NaN inside either solid).

        Variants of one study usually share the grid, and then this is a plain
        subtraction. Otherwise the variant is sampled bilinearly at the baseline's
        cell centres (outside its domain: NaN), so a longer car or a larger domain
        still compares point by point in metres.
        """
        out: dict[str, np.ndarray] = {}
        if base_fd is None or name not in base_fd.fields:
            return out
        base = np.asarray(base_fd.fields[name], dtype=np.float64)
        bmask = base_fd.mask if base_fd.mask is not None else np.zeros_like(base)
        ny, nx = base.shape
        x0, x1, y0, y1 = base_fd.extent
        xc = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        yc = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        first = True
        for sid, fd in fds.items():
            if first:            # the baseline itself (dicts keep the baseline first)
                first = False
                continue
            if fd is None or name not in fd.fields:
                continue
            a = np.asarray(fd.fields[name], dtype=np.float64)
            m = fd.mask if fd.mask is not None else np.zeros_like(a)
            if a.shape == base.shape and np.allclose(fd.extent, base_fd.extent):
                va, vm = a, m
            else:
                va, vm = _resample(a, fd.extent, xc, yc), _resample(m, fd.extent, xc, yc)
            d = va - base
            d[(np.asarray(vm) >= 0.5) | (np.asarray(bmask) >= 0.5)] = np.nan
            out[sid] = d
        return out


def _resample(a: np.ndarray, extent, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Bilinear sample of a cell-centred (ny, nx) array at the grid xc x yc (NaN outside)."""
    ny, nx = a.shape
    x0, x1, y0, y1 = extent
    fx = (xc - x0) / (x1 - x0) * nx - 0.5
    fy = (yc - y0) / (y1 - y0) * ny - 0.5
    i0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx = np.clip(fx - i0, 0.0, 1.0)[None, :]
    ty = np.clip(fy - j0, 0.0, 1.0)[:, None]
    J, I = np.meshgrid(j0, i0, indexing="ij")
    v = ((1 - ty) * ((1 - tx) * a[J, I] + tx * a[J, I + 1]) + ty * ((1 - tx) * a[J + 1, I] + tx * a[J + 1, I + 1]))
    outside = (fx < -0.5) | (fx > nx - 0.5)
    v[:, outside] = np.nan
    v[(fy < -0.5) | (fy > ny - 0.5), :] = np.nan
    return v


__all__ = ["ComparePage", "adapt_rows", "compare_group", "comparison_rows", "fmt_pct", "fmt_value", "judge", "mixed_message",
           "result_group"]
