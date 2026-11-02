"""Solver diagnostics: the numbers that say whether a running simulation can be trusted (DESIGN.md 10.4).

AeroJAX shows, next to the flow, how well the solver is doing its job: how
far the velocity is from divergence-free after the projection, how much each
field still changes per step, the CFL number and time step, the pressure-solve
residual and the throughput. A converged-looking picture with a divergence of
1e-2 is a broken run; a run whose change norms stopped falling has reached its
steady state (or its limit cycle). This module records and shows them.

* :class:`DiagnosticsLog` (no Qt) -- a bounded, thread-safe table of samples
  (step, t, one column per quantity). The job manager and the explore worker
  feed it from the *worker* thread, before frames are coalesced, so a slow GUI
  never drops samples; the panel reads snapshots. :meth:`DiagnosticsLog.to_csv`
  writes it out, every column, full precision.
* :data:`QUANTITIES` and :func:`describe` -- what each key means, its unit and
  its plot group. The solver decides which keys it reports; names follow the
  pattern ``div_rms``, ``du_rel`` (relative change of u per step block),
  ``cfl``, ``dt``, ``residual`` ..., and unknown keys are still recorded and
  plotted (group "Other"), so the panel never hides a number the solver sent.
* :class:`DiagnosticsPanel` -- a header strip with a Record switch (Lumen's
  panel-header switch: off keeps the values but stops adding), the x axis
  (step or time), Clear and Export CSV; the latest values as a key / value
  list; stacked plots sharing the x axis, logarithmic where a quantity spans
  decades (divergence, change norms, residuals).

Repaint cost is bounded the Lumen way: the plots are redrawn on a timer only
when the log's revision changed, at most :data:`REFRESH_MS` apart, and each
series is decimated to :data:`PLOT_POINTS`.
"""
from __future__ import annotations

import csv
import math
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ALIASES", "MIN_PLOT_W", "QUANTITIES", "ROW_KEYS", "DiagnosticsLog", "DiagnosticsPanel", "Quantity",
           "canonical", "compact_ticks", "constant_value", "describe", "frame_diagnostics", "latest_values_list",
           "si_ticks"]

#: Rows kept per log (a long explore session: 20 samples/s for 40 min).
MAX_ROWS = 50_000
#: Rows of the diagnostics table a finished run stores with its result (DiagnosticsLog.decimated): enough
#: for the history plots' shape (the decay of the change norms, a drifting CFL), small next to the fields.
STORED_ROWS = 200
#: Points drawn per series (the plot cannot show more on a dock's width).
PLOT_POINTS = 2000
#: Minimum interval between plot redraws [ms]: about 2 Hz. At 5 Hz the open dock took explore from 549 to
#: 252 steps/s (shared GPU, so indicative): a redraw of seven pyqtgraph panels holds the GUI thread the
#: solver's frames are delivered on, and a health plot does not need more than two updates a second.
REFRESH_MS = 500
#: Smallest height of one plot group [px], its title line included: below it a dock's plots are flowed
#: into columns instead (a wide, short bottom dock stacked seven 15 px strips that showed nothing). At
#: 88 px the rotated axis names did not fit beside the plot ("Δ per step" read "er step").
MIN_PLOT_H = 120
#: Extra height of a plot that carries the x axis (tick labels and "step") [px]: without it the axis was
#: taken out of the plot's own 88 px and the "steps/s" plot kept a 40 px sliver.
BOTTOM_AXIS_H = 36
#: Smallest width of one plot group [px], left axis included (about 160 px of plot): narrower, the
#: groups wrap into further rows and the plots scroll. Eight groups side by side in a 975 px dock
#: were 20 px wide and their tick labels ran together ("0 20004000").
MIN_PLOT_W = 220
#: Most tick labels on a linear axis of a diagnostics plot.
MAX_TICKS = 3
#: A linear series whose spread is below this share of its mean is drawn as a constant (an adaptive
#: time step holds the CFL number at 0.5: its axis read 0.49999998 .. 0.50000006, float noise).
CONSTANT_REL = 1e-6


@dataclass(frozen=True)
class Quantity:
    """How one diagnostic is labelled and plotted: label, unit, plot group, logarithmic axis."""
    label: str
    unit: str
    group: str
    log: bool


#: Known keys. Velocities are in units of U, lengths of L, times of L/U (the solver's units).
QUANTITIES: dict[str, Quantity] = {
    "div_rms": Quantity("RMS divergence", "U/L", "Divergence", True),
    "div_max": Quantity("Max |divergence|", "U/L", "Divergence", True),
    "div_l2": Quantity("L2 divergence", "U/L", "Divergence", True),
    "du_rel": Quantity("Relative change of u", "", "Change per step", True),
    "dv_rel": Quantity("Relative change of v", "", "Change per step", True),
    "dw_rel": Quantity("Relative change of w", "", "Change per step", True),
    "dp_rel": Quantity("Relative change of p", "", "Change per step", True),
    "dtheta_rel": Quantity("Relative change of θ", "", "Change per step", True),
    # the L2 norms are domain integrals, (∫ Δφ² dV)^½: the field's unit times L in 2D and L^(3/2) in 3D
    # (solvers.ns2d.diagnostics, ns3d DIAG_KEYS), so no one unit fits; the report leaves it blank too
    "du_l2": Quantity("L2 change of u", "", "Change per step", True),
    "dv_l2": Quantity("L2 change of v", "", "Change per step", True),
    "dp_l2": Quantity("L2 change of p", "", "Change per step", True),
    "du_rms": Quantity("RMS change of u", "U", "Change per step", True),
    "dv_rms": Quantity("RMS change of v", "U", "Change per step", True),
    "du_max": Quantity("Max change of u", "U", "Change per step", True),
    "dv_max": Quantity("Max change of v", "U", "Change per step", True),
    "cfl": Quantity("CFL number", "", "CFL", False),
    "cfl_max": Quantity("Largest CFL number", "", "CFL", False),
    "dt": Quantity("Time step Δt", "L/U", "Time step", False),
    "residual": Quantity("Pressure-solve residual", "", "Residual", True),
    "p_residual": Quantity("Pressure-solve residual", "", "Residual", True),
    "poisson_residual": Quantity("Pressure-solve residual", "", "Residual", True),
    "pressure_residual": Quantity("Pressure-solve residual", "", "Residual", True),
    "steps_per_s": Quantity("Solver rate", "steps/s", "Rate", False),
    # the 3D solver's extra columns (solvers.ns3d.solver.DIAG_KEYS). Labels stay short: the Latest list
    # is 300 px wide and a longer key pushed every value past its right edge ("6.52×10" for 6.52×10⁻⁷ U/L)
    "umax": Quantity("Max |u|+|v|+|w|", "U", "Velocity", False),
    "heat_storage": Quantity("Heat storage rate", "", "Heat", False),       # solver units (k ΔT L in 3D)
}
#: Other names of the same quantity -> the name the log keeps. The 3D solver reports its pressure-solve
#: residual twice ("poisson_residual" and the 2D name "p_residual"): one quantity, one curve, one column.
ALIASES = {"poisson_residual": "p_residual", "pressure_residual": "p_residual", "divergence_rms": "div_rms",
           "divergence_max": "div_max"}
#: Plot order of the groups.
GROUPS = ("Divergence", "Change per step", "Residual", "CFL", "Time step", "Velocity", "Heat", "Rate", "Other")
#: Short axis labels of the groups (the group name is in the legend and the tooltip).
GROUP_AXIS = {"Divergence": "∇·u", "Change per step": "Δ per step", "Residual": "Residual", "CFL": "CFL",
              "Time step": "Δt (L/U)", "Velocity": "max (U)", "Heat": "heat rate", "Rate": "steps/s", "Other": ""}
#: Keys shown in the latest-values list first (others follow alphabetically).
PRIMARY = ("div_rms", "div_max", "du_rel", "dv_rel", "dw_rel", "dp_rel", "dtheta_rel", "cfl", "dt", "p_residual",
           "residual", "steps_per_s")


#: How a field of a change-norm key is written ("dtheta_l2" -> "L2 change of θ").
_FIELD_SYMBOL = {"theta": "θ", "temperature": "θ"}


def describe(key: str) -> Quantity:
    """The :class:`Quantity` of a key, guessed from its name when it is not in :data:`QUANTITIES`."""
    if key in QUANTITIES:
        return QUANTITIES[key]
    k = key.lower()
    if k.startswith("div"):
        return Quantity(key.replace("_", " "), "", "Divergence", True)
    m = re.fullmatch(r"d([a-z]+)_(rel|l2|rms|max)", k)
    if m:
        what = {"rel": "Relative change", "l2": "L2 change", "rms": "RMS change", "max": "Max change"}[m.group(2)]
        fld = m.group(1)
        # relative: a ratio; L2: a domain integral whose unit depends on the dimension (see QUANTITIES)
        unit = "" if m.group(2) in ("rel", "l2") else {"u": "U", "v": "U", "w": "U", "p": "ρU²",
                                                         "theta": "ΔT"}.get(fld, "")
        return Quantity(f"{what} of {_FIELD_SYMBOL.get(fld, fld)}", unit, "Change per step", True)
    if "resid" in k:
        return Quantity(key.replace("_", " ").capitalize(), "", "Residual", True)
    if "cfl" in k:
        return Quantity(key.replace("_", " "), "", "CFL", False)
    if k in ("dt", "dt_s"):
        return Quantity("Time step", "", "Time step", False)
    return Quantity(key.replace("_", " "), "", "Other", False)


def canonical(values: Mapping[str, float]) -> dict[str, float]:
    """``values`` with :data:`ALIASES` folded into their canonical names (the canonical one wins when a
    row carries both, which is the 3D solver's case: they are the same number)."""
    out: dict[str, float] = {}
    for k, v in values.items():
        name = ALIASES.get(str(k), str(k))
        if name in out and name != k:
            continue
        out[name] = v
    return out


def _scalar(v) -> float | None:
    """``float(v)``, or None for something that is not a number (NaN and inf are kept: they are news)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


#: Keys of a diagnostics row that index it rather than measure anything (and the solver's own count of
#: the steps an advance call ran, which is bookkeeping, not a diagnostic).
ROW_KEYS = frozenset({"step", "t", "perf_steps"})


def frame_diagnostics(frame) -> dict[str, float]:
    """The scalar diagnostics a frame carries (``frame.diagnostics``), plus its measured steps/s.

    Array values (a solver that sends whole histories) are left to
    :meth:`DiagnosticsLog.add_frame`, which takes them as a block.
    """
    out: dict[str, float] = {}
    d = getattr(frame, "diagnostics", None)
    if isinstance(d, Mapping):
        for k, v in d.items():
            if np.ndim(v) == 0 and k not in ROW_KEYS:
                f = _scalar(v)
                if f is not None:
                    out[str(k)] = f
    sps = _scalar(getattr(frame, "steps_per_s", None))
    if sps is not None and math.isfinite(sps) and sps > 0 and "steps_per_s" not in out:
        out["steps_per_s"] = sps
    return out


class DiagnosticsLog:
    """A bounded table of diagnostic samples (thread-safe; see the module docstring).

    Rows are (step, t, values...); a key first seen later gets NaN for the
    earlier rows, so every column has the same length. A step that goes
    backwards (a solver reset, a new run) starts a new table: plotting two
    runs on one step axis would draw a line back to zero.
    """

    def __init__(self, max_rows: int = MAX_ROWS) -> None:
        self.max_rows = int(max_rows)
        self._lock = threading.Lock()
        self._cols: dict[str, list[float]] = {"step": [], "t": []}
        self.revision = 0
        self.source = ""                 # what is being recorded ("Explore · NACA 0012", ...)
        self.recording = True

    # -- writing (any thread) ----------------------------------------------
    def clear(self, source: str | None = None) -> None:
        with self._lock:
            self._cols = {"step": [], "t": []}
            if source is not None:
                self.source = source
            self.revision += 1

    def add(self, step: float, t: float, values: Mapping[str, float]) -> bool:
        """Append one sample; returns False when nothing was recorded (paused, no values).

        ``values`` may repeat the row's own "step" and "t" (a solver's latest
        diagnostics row carries them); they are the row's index here, never a
        quantity: appended as values too, they made those two columns grow by
        two per row and every later plot failed on mismatched lengths.
        """
        values = canonical({k: v for k, v in values.items() if k not in ROW_KEYS})
        if not self.recording or not values:
            return False
        with self._lock:
            c = self._cols
            steps = c["step"]
            if steps and float(step) < steps[-1]:
                self._cols = c = {"step": [], "t": []}
                steps = c["step"]
            if steps and float(step) == steps[-1]:
                # the same step again (a paused session re-renders): update that row instead
                for k, v in values.items():
                    if k not in c:
                        c[k] = [math.nan] * len(steps)
                    c[k][-1] = float(v)
                self.revision += 1
                return True
            n = len(steps)
            steps.append(float(step))
            c["t"].append(float(t))
            for k, v in values.items():
                if k not in c:
                    c[k] = [math.nan] * n
                c[k].append(float(v))
            for k, col in c.items():
                if len(col) < n + 1:
                    col.append(math.nan)
            if n + 1 > self.max_rows:
                drop = n + 1 - self.max_rows
                for k in c:
                    del c[k][:drop]
            self.revision += 1
            return True

    def add_block(self, table: Mapping[str, np.ndarray]) -> None:
        """Replace the log with a whole history the solver kept itself ({"step": ..., "t": ..., key: ...})."""
        if not self.recording:
            return
        n = len(np.asarray(table.get("step", table.get("t", [])))) if table else 0
        with self._lock:
            cols = {}
            for k, v in canonical(table).items():
                a = np.asarray(v, dtype=float).ravel()
                if len(a) == n:
                    cols[str(k)] = a[-self.max_rows:].tolist()
            cols.setdefault("step", list(range(len(next(iter(cols.values()), [])))))
            cols.setdefault("t", [math.nan] * len(cols["step"]))
            self._cols = cols
            self.revision += 1

    def add_rows(self, table: Mapping[str, np.ndarray]) -> int:
        """Append the rows of a column table ({"step": [...], key: [...]}) that are newer than the log.

        A solver that records diagnostics on the device and copies its ring to
        the host once per block of steps hands over several rows at a time;
        rows already recorded (same or older step) are skipped, so a frame that
        repeats the ring costs nothing. Returns the number of rows added.
        """
        if not self.recording or not table or "step" not in table:
            return 0
        steps = np.asarray(table["step"], dtype=float).ravel()
        cols = canonical({str(k): np.asarray(v, dtype=float).ravel() for k, v in table.items()
                          if k not in ("step", "t") and np.ndim(v) == 1 and len(np.ravel(v)) == len(steps)})
        t = np.asarray(table.get("t", np.full(len(steps), math.nan)), dtype=float).ravel()
        with self._lock:
            last = self._cols["step"][-1] if self._cols["step"] else -math.inf
        # (a solver reset is not guessed from old step numbers: whoever resets the flow clears the log)
        added = 0
        for i in np.argsort(steps, kind="stable"):
            if steps[i] > last:
                if self.add(steps[i], t[i] if i < len(t) else math.nan, {k: v[i] for k, v in cols.items()}):
                    added += 1
                last = steps[i]
        return added

    def add_frame(self, frame) -> bool:
        """Record what a ``LiveFrame`` reports (worker thread): its scalar diagnostics, a block of rows
        (``diagnostics`` holding arrays with a "step" column), or a whole history block."""
        hist = getattr(frame, "diagnostics_history", None)
        if isinstance(hist, Mapping) and hist:
            self.add_block(hist)
            return True
        d = getattr(frame, "diagnostics", None)
        if isinstance(d, Mapping) and "step" in d and np.ndim(d["step"]) == 1:
            n = self.add_rows(d)
            sps = _scalar(getattr(frame, "steps_per_s", None))
            if n and sps is not None and math.isfinite(sps) and sps > 0:
                with self._lock:
                    c = self._cols
                    if "steps_per_s" not in c:
                        c["steps_per_s"] = [math.nan] * len(c["step"])
                    c["steps_per_s"][-1] = sps
            return n > 0
        step = getattr(frame, "step", None)
        t = getattr(frame, "t", math.nan)
        vals = frame_diagnostics(frame)
        if step is None or not vals:
            return False
        return self.add(float(step), float(t) if t is not None else math.nan, vals)

    # -- reading -------------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._cols["step"])

    def keys(self) -> list[str]:
        """Recorded quantities (not step and t), in :data:`PRIMARY` order, then alphabetically."""
        with self._lock:
            ks = [k for k in self._cols if k not in ("step", "t")]
        first = [k for k in PRIMARY if k in ks]
        return first + sorted(k for k in ks if k not in first)

    def snapshot(self) -> dict[str, np.ndarray]:
        """Every column as a float64 array (copies; safe to use while the worker writes)."""
        with self._lock:
            return {k: np.asarray(v, dtype=float) for k, v in self._cols.items()}

    def decimated(self, n: int = STORED_ROWS) -> dict[str, np.ndarray]:
        """At most ``n`` rows of the log, evenly spaced over its samples, first and last kept: the table a
        finished run stores with its result (``result.diagnostics_history``), so the history plots of a
        reopened result are not one end-of-run sample. Empty when the log is."""
        snap = self.snapshot()
        m = len(snap.get("step", ()))
        if m == 0:
            return {}
        idx = np.unique(np.linspace(0, m - 1, min(int(n), m)).round().astype(int))
        return {k: np.asarray(v, dtype=np.float64)[idx] for k, v in snap.items() if len(v) == m}

    def latest(self) -> dict[str, float]:
        """The newest finite value of every quantity."""
        snap = self.snapshot()
        out = {}
        for k, a in snap.items():
            ok = np.isfinite(a)
            if ok.any():
                out[k] = float(a[ok][-1])
        return out

    def to_csv(self, path: str | Path) -> Path:
        """Write every column (step, t, then the quantities) with 10 significant figures; returns the path."""
        snap = self.snapshot()
        keys = ["step", "t"] + self.keys()
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            n = len(snap.get("step", []))
            for i in range(n):
                row = []
                for k in keys:
                    v = snap[k][i] if k in snap and i < len(snap[k]) else math.nan
                    row.append("" if not math.isfinite(v) else (str(int(v)) if k == "step" else f"{v:.10g}"))
                w.writerow(row)
        return path


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

_LATEST_CLS = None


def _latest_values_class():
    """:class:`~aerolume.app.widgets.common.KeyValueList` that updates its values in place.

    The panel refreshes five times a second while a run goes. Rebuilding thirty label pairs each
    time flickered, and a repaint between the rebuild and the new labels' (queued) show drew an
    empty "Latest" list -- caught in a screenshot of the live 3D dock. Built on first use: the log
    half of this module stays importable without the widget kit.
    """
    global _LATEST_CLS
    if _LATEST_CLS is None:
        from .common import KeyValueList

        class LatestValues(KeyValueList):
            def update_items(self, items) -> None:
                """New values for the same keys in place; a different key list rebuilds the rows."""
                keys = [str(it[0]) for it in items]
                if keys != [k.text() for k, _v in self._rows]:
                    self.set_items(items)
                    return
                for (_k, v), it in zip(self._rows, items, strict=True):
                    text = str(it[1])
                    if v.text() != text:
                        v.setText(text)

        _LATEST_CLS = LatestValues
    return _LATEST_CLS


def latest_values_list(**kw) -> QWidget:
    """A key / value list whose values update in place (no rebuild, no flicker) -- the panel's Latest
    list, also used by Simulate's closed-domain monitors. ``update_items([(key, value), ...])``."""
    return _latest_values_class()(**kw)

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def decade_ticks(lo: float, hi: float, max_labels: int = 5) -> list[list[tuple[float, str]]]:
    """Ticks of a logarithmic axis spanning 10^lo .. 10^hi: labelled decades ("10⁻⁷"), every other one
    when there are many, and unlabelled 2..9 minor ticks (pyqtgraph's own labels every minor tick of a
    short log axis, which piles "2·10⁻⁷ 3·10⁻⁷ ..." into an unreadable column)."""
    k0, k1 = math.floor(lo), math.ceil(hi)
    decades = [k for k in range(k0, k1 + 1) if lo - 1e-9 <= k <= hi + 1e-9]
    step = max(1, math.ceil(len(decades) / max_labels))
    major = [(float(k), f"10{str(k).translate(_SUPERSCRIPT)}") for k in decades if k % step == 0]
    minor = [(math.log10(m) + k, "") for k in range(k0 - 1, k1 + 1) for m in range(2, 10)
             if lo <= math.log10(m) + k <= hi]
    return [major, minor]


def compact_ticks(lo: float, hi: float, n: int = MAX_TICKS) -> list[float]:
    """At most ``n`` round tick values (1, 2, 2.5, 5 x 10^k steps) inside [lo, hi]."""
    from .plots import nice_ticks
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else []
    single: list[float] = []
    # two or three ticks: a 1-2-5 step of span / 2 can leave only one ("0.01" on a 0.004-0.016 axis),
    # so finer requests are tried until two fit
    for k in range(n, n + 4):
        ticks = [t for t in nice_ticks(lo, hi, k) if lo - 1e-9 * (hi - lo) <= t <= hi + 1e-9 * (hi - lo)]
        if 1 < len(ticks) <= n:
            return ticks
        if len(ticks) == 1 and not single:
            single = ticks
    return single or [0.5 * (lo + hi)]


_SI = ((1e9, "G"), (1e6, "M"), (1e3, "k"))


def si_ticks(values, spacing: float) -> list[str]:
    """Tick labels with SI prefixes for large numbers ("0", "2k", "4k"; "1.5M") and as many decimals as the
    spacing needs below that ("0.004", "0.005"); a real minus sign. Long labels on a 160 px axis ran into
    each other ("0 20004000")."""
    vals = [float(v) for v in values]
    if not vals:
        return []
    top = max(abs(v) for v in vals)
    div, suf = next(((d, sfx) for d, sfx in _SI if top >= d), (1.0, ""))
    step = abs(float(spacing)) / div if spacing else 0.0
    if len(vals) == 1 or step <= 1e-9 * max(top / div, 1e-300):
        # one tick (a single sample, a constant): three significant figures, not the spacing's decimals
        # ("4.3300k" for the one stored row of a result)
        from .kpi import fmt_sig
        return [(fmt_sig(v / div, 3) + suf if v else "0").replace("-", "−") for v in vals]
    if 0 < top / div < 1e-3 and not suf:
        from .kpi import fmt_sig
        return [fmt_sig(v, 2) if v else "0" for v in vals]
    dec = max(0, -math.floor(math.log10(step) + 1e-9)) if step > 0 else 0
    dec = min(dec, 6)
    out = []
    for v in vals:
        txt = f"{v / div:.{dec}f}"
        if float(txt) == 0.0:
            txt = "0"
        out.append((txt + suf if txt != "0" else "0").replace("-", "−"))
    return out


def constant_value(ys) -> float | None:
    """The value of a set of series that are constant to :data:`CONSTANT_REL` (None when they vary)."""
    vals = np.concatenate([np.asarray(y, dtype=float).ravel() for y in ys]) if len(ys) else np.zeros(0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    mean = float(vals.mean())
    if hi - lo <= CONSTANT_REL * abs(mean) or hi == lo:
        return mean
    return None


_AXIS_CLS = None


def compact_axis_class():
    """``pyqtgraph.AxisItem`` with at most :data:`MAX_TICKS` labelled ticks and SI labels on linear axes
    (log axes keep the decade ticks the panel sets). Built on first use (pyqtgraph is imported lazily)."""
    global _AXIS_CLS
    if _AXIS_CLS is None:
        import pyqtgraph as pg

        class CompactAxis(pg.AxisItem):
            def tickValues(self, minVal, maxVal, size):
                if self.logMode or self._tickLevels is not None:
                    return super().tickValues(minVal, maxVal, size)
                lo, hi = sorted((float(minVal), float(maxVal)))
                if hi - lo <= 1e-3 * max(abs(lo), abs(hi), 1e-300):     # one sample, a constant: one tick
                    mid = 0.5 * (lo + hi)
                    return [(max(abs(mid), 1.0), [mid]), (max(abs(mid), 1.0), [])]
                # as many labels as the axis has room for: two on a 30 px axis ran "100" into "50"
                ticks = compact_ticks(lo, hi, max(1, min(MAX_TICKS, int(float(size) // 26) + 1)))
                if not ticks:
                    return super().tickValues(minVal, maxVal, size)
                spacing = ticks[1] - ticks[0] if len(ticks) > 1 else max(hi - lo, 1e-300)
                return [(spacing, ticks), (spacing, [])]

            def tickStrings(self, values, scale, spacing):
                if self.logMode:
                    return super().tickStrings(values, scale, spacing)
                return si_ticks([v * scale for v in values], spacing * scale)

        _AXIS_CLS = CompactAxis
    return _AXIS_CLS


def _decimate(x: np.ndarray, ys: list[np.ndarray], n: int = PLOT_POINTS):
    if len(x) <= n:
        return x, ys
    idx = np.unique(np.linspace(0, len(x) - 1, n).round().astype(np.int64))
    return x[idx], [y[idx] for y in ys]


def fmt_value(key: str, v: float) -> str:
    """A diagnostic value as text: steps/s as an integer, the rest to 3 significant figures."""
    from .kpi import fmt_num, fmt_sig
    if not math.isfinite(v):
        return "—"
    if key == "steps_per_s":
        return f"{v:,.0f}"
    if key in ("cfl",):
        return fmt_num(v, 3)
    return fmt_sig(v, 3)


class DiagnosticsPanel(QWidget):
    """Diagnostics view of a :class:`DiagnosticsLog` (see the module docstring).

    ``set_log(log)`` binds a log (Simulate binds the explore worker's or the
    job manager's); the panel polls its revision, so the worker never calls
    into Qt. ``exported(path)`` fires after a CSV export.
    """

    exported = Signal(str)

    def __init__(self, parent: QWidget | None = None, *, log: DiagnosticsLog | None = None) -> None:
        super().__init__(parent)
        from .common import SegmentedControl, ToolStrip, label
        from .kpi import EmptyState
        self.log = log if log is not None else DiagnosticsLog()
        self._seen = -1
        #: whether the run has heat transfer (None: unknown, keep whatever the solver reports)
        self._thermal: bool | None = None
        self._constant: dict[str, str] = {}      # plot group -> "constant 0.5" while its series are
        self._export_dir = Path.home()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        bar = self.bar = ToolStrip()
        self.record_btn = bar.add_button(None, "Record", "Record the solver's diagnostics; off keeps what was "
                                         "recorded and stops adding (the solver keeps computing them)",
                                         checkable=True)
        self.record_btn.setProperty("variant", "toggle")
        self.record_btn.setChecked(True)
        bar.add_separator()
        bar.add_widget(label("Axis", "secondary"))
        self.axis = SegmentedControl([("step", "Step"), ("t", "Time")], "step")
        self.axis.setToolTip("Plot against the solver step or the simulated time (L/U)")
        bar.add_widget(self.axis)
        self.source_label = bar.add_widget(label("", "muted"))
        bar.add_stretch()
        self.count_label = bar.add_widget(label("", "muted"))
        self.clear_btn = bar.add_button("trash", "", "Clear the recorded diagnostics")
        self.csv_btn = bar.add_button("export", "Export CSV…", "Save every recorded diagnostic as a CSV table")
        root.addWidget(bar)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.empty = EmptyState()
        self.empty.set_content("No diagnostics yet",
                               "Divergence, change per step, CFL number, time step and the pressure-solve residual "
                               "appear here while a simulation or explore runs.")
        self.stack.addWidget(self.empty)
        body = QSplitter()
        body.setHandleWidth(1)
        body.setChildrenCollapsible(False)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(10, 8, 10, 8)
        sl.setSpacing(6)
        sl.addWidget(label("Latest", "h3"))
        self.values = _latest_values_class()(mono_values=True)
        sl.addWidget(self.values)
        sl.addStretch(1)
        # twenty-odd quantities do not fit a short dock: the list scrolls (it used to be squeezed to nothing)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setWidget(side)
        side_scroll.setMinimumWidth(300)          # "Relative change of θ  1.23×10⁻⁴" whole
        side_scroll.setMaximumWidth(360)
        self.side_scroll = side_scroll
        body.addWidget(side_scroll)
        self.plots = self._make_plots()
        # the groups wrap into rows (each at least MIN_PLOT_W wide); rows that do not fit the dock's
        # height scroll rather than squeeze to strips
        plot_scroll = self._plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setFrameShape(QFrame.Shape.NoFrame)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        plot_scroll.setWidget(self.plots)
        plot_scroll.viewport().installEventFilter(self)
        # one sample (a result that kept only its end-of-run row) has no history to plot: flat
        # one-point plots with a "4.3300k" axis said nothing, so the Latest list stands alone
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(plot_scroll)
        self.one_note = label("One sample: the history plots need two or more. The values are listed on the "
                              "left and in Export CSV.", "muted", wrap=True)
        self.one_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_stack.addWidget(self.one_note)
        body.addWidget(self.plot_stack)
        body.setStretchFactor(1, 1)
        self.stack.addWidget(body)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.record_btn.toggled.connect(self._on_record)
        self.axis.changed.connect(lambda *_: self.refresh(force=True))
        self.clear_btn.clicked.connect(self._clear)
        self.csv_btn.clicked.connect(self.export_csv)
        self.refresh(force=True)

    # -- plots ---------------------------------------------------------------
    def _make_plots(self):
        import pyqtgraph as pg

        from .plots import connect_theme
        w = pg.GraphicsLayoutWidget()
        w.ci.setContentsMargins(4, 6, 10, 4)
        w.ci.setSpacing(2)
        self._glw = w
        self._panels: dict[str, object] = {}
        self._curves: dict[str, object] = {}
        self._grid_key: tuple | None = None
        self._bottom: set[str] = set()
        w.viewport().installEventFilter(self)
        connect_theme(self._apply_theme)
        return w

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        sc = getattr(self, "_plot_scroll", None)
        if sc is not None and obj is sc.viewport() and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self, self._layout_panels)
        elif sc is not None and obj is self._glw.viewport() and event.type() == QEvent.Type.Wheel:
            # the plots follow the data (no zoom): the wheel scrolls the rows instead
            from PySide6.QtWidgets import QApplication
            QApplication.sendEvent(sc.verticalScrollBar(), event)
            return True
        return super().eventFilter(obj, event)

    def grid_shape(self) -> tuple[int, int]:
        """(rows, columns) the plot groups are laid out in now."""
        return (self._grid_key[0], self._grid_key[1]) if self._grid_key else (0, 0)

    def _layout_panels(self) -> None:
        """Place the group plots on a grid: as many rows as fit at :data:`MIN_PLOT_H`, but never so few
        that a plot is narrower than :data:`MIN_PLOT_W` -- the groups then wrap into further rows, which
        scroll.

        Row-major (the first row holds the first groups: divergence, change per step, residual ...), so a
        short dock shows the most telling plots without scrolling. While the rows of a grid scroll, every plot
        carries its own x axis (a row seen alone must still read), each at least :data:`MIN_PLOT_H` plus
        :data:`BOTTOM_AXIS_H` tall; when they all fit, only the last plot of each column does."""
        groups = list(self._panels)
        if not groups:
            self._grid_key = None
            return
        vp = self._plot_scroll.viewport()
        w, h = max(vp.width(), 1), max(vp.height(), 1)
        n = len(groups)
        cols_max = max(1, w // MIN_PLOT_W)
        rows = max(math.ceil(n / cols_max), min(n, max(1, (h - BOTTOM_AXIS_H) // MIN_PLOT_H)))
        cols = math.ceil(n / rows)
        rows = math.ceil(n / cols)                      # balanced: 7 groups in 4 columns are 4 + 3
        scrolls = rows * MIN_PLOT_H + BOTTOM_AXIS_H > h
        # a wide grid that scrolls shows one row at a time: each row needs its own x axis. A single column
        # (a tall side dock) reads top to bottom over one shared axis at its foot, as a stack of plots does
        own_axes = scrolls and cols > 1
        bottom = set(groups) if own_axes else {g for i, g in enumerate(groups) if i + cols >= n}
        if scrolls:
            need_h = rows * MIN_PLOT_H + (rows if own_axes else 1) * BOTTOM_AXIS_H
        else:
            need_h = 0
        self._glw.setMinimumHeight(need_h)
        key = (rows, cols, tuple(groups), scrolls)
        if key == self._grid_key:
            return
        self._grid_key = key
        ci = self._glw.ci
        for pi in self._panels.values():
            if pi.scene() is not None:
                ci.removeItem(pi)
        for i, g in enumerate(groups):
            pi = self._panels[g]
            pi.setMinimumHeight(MIN_PLOT_H + (BOTTOM_AXIS_H if g in bottom else 0) if scrolls else 0)
            ci.addItem(pi, row=i // cols, col=i % cols)
        self._bottom = bottom
        self._apply_theme()

    def _apply_theme(self) -> None:
        from .plots import make_pen, series_color, style_plot_item, tokens
        tk = tokens()
        self._glw.setBackground(tk["surface"])
        groups = list(self._panels)
        for i, g in enumerate(groups):
            bottom = g in self._bottom if self._bottom else i == len(groups) - 1
            style_plot_item(self._panels[g], tk, xlabel=self._xlabel() if bottom else "",
                            ylabel=GROUP_AXIS.get(g, g))
            title = self._title_html(g, tk) or self._constant_html(g, tk)
            self._panels[g].setTitle(title or None, size="8pt", justify="left")
            self._panels[g].getAxis("bottom").setStyle(showValues=bottom)
            self._panels[g].getAxis("left").setToolTip(g)
        for key, c in self._curves.items():
            c.setPen(make_pen(series_color(self._series_index(key), tk), 1.4))

    def _constant_html(self, group: str, tk: dict) -> str:
        """"constant 0.500" over a group whose series do not vary (in the secondary text colour)."""
        text = self._constant.get(group, "")
        return f"<span style='color:{tk['text2']}'>{text}</span>" if text else ""

    def _series_index(self, key: str) -> int:
        """A stable colour per quantity within its group (u blue, v orange, p green, θ red ...)."""
        order = ("u", "v", "w", "p", "theta")
        m = re.fullmatch(r"d([a-z]+)_\w+", key)
        if m and m.group(1) in order:
            return order.index(m.group(1))
        return {"div_rms": 0, "div_max": 1, "div_l2": 2, "cfl": 0, "dt": 1}.get(key, abs(hash(key)) % 6)

    def _xlabel(self) -> str:
        return "step" if self.axis.current() == "step" else "t · U/L"

    @staticmethod
    def plotted_keys(keys: list[str]) -> list[str]:
        """The quantities drawn as curves: every key, except that "Change per step" shows only the
        relative changes when the log has them (u, v, p, θ: four curves an eye can follow, the AeroJAX
        convention). Their L2 / RMS / max companions, sixteen curves on one small plot, stay in the
        Latest list and the CSV."""
        rel = [k for k in keys if describe(k).group == "Change per step" and k.endswith("_rel")]
        if not rel:
            return list(keys)
        return [k for k in keys if describe(k).group != "Change per step" or k in rel]

    def _rebuild(self, keys: list[str]) -> None:
        """(Re)create one panel per group present (laid out by :meth:`_layout_panels`), sharing the x axis."""
        import pyqtgraph as pg
        keys = self.plotted_keys(keys)
        groups = [g for g in GROUPS if any(describe(k).group == g for k in keys)]
        if groups == list(self._panels) and set(self._curves) == set(keys):
            return
        from .plots import adopt_plot_menu, release_plot_menus
        release_plot_menus(self._panels.values())     # discarded panels free their menus with them
        self._glw.ci.clear()
        self._panels, self._curves = {}, {}
        self._grid_key, self._bottom = None, set()
        self._constant = {}
        first = None
        axis = compact_axis_class()
        for g in groups:
            pi = pg.PlotItem(axisItems={"left": axis("left"), "bottom": axis("bottom")})
            pi.setMenuEnabled(False)
            adopt_plot_menu(pi, self._glw)
            pi.hideButtons()
            pi.setMouseEnabled(x=False, y=False)       # each refresh re-ranges anyway; the wheel scrolls
            pi.getAxis("left").setWidth(self.AXIS_MIN_W)
            log = any(describe(k).log for k in keys if describe(k).group == g)
            pi.setLogMode(y=log)
            if log:
                # label decades only: minor-tick labels (2·10⁻⁷, 3·10⁻⁷ ...) crowd a short axis into a smear
                pi.getAxis("left").setStyle(maxTextLevel=0)
            if first is None:
                first = pi
            else:
                pi.setXLink(first)
            # several curves: their names in their colours as the plot's title line (a legend box
            # inside a 90 px plot covered the curves and clipped its own labels to "iv")
            self._panels[g] = pi
            for k in keys:
                if describe(k).group == g:
                    c = pg.PlotDataItem([], [], name=self._legend_name(k))
                    pi.addItem(c)
                    self._curves[k] = c
        self._layout_panels()

    def _title_html(self, group: str, tk: dict) -> str:
        """The coloured key of a multi-curve group: "rms  max", "u  v  p  θ", each in its curve's colour."""
        from .plots import series_color
        ks = [k for k in self._curves if describe(k).group == group]
        if len(ks) < 2:
            return ""
        parts = [f"<span style='color:{series_color(self._series_index(k), tk).name()}'>{self._legend_name(k)}</span>"
                 for k in ks]
        return "&nbsp;&nbsp;".join(parts)

    @staticmethod
    def _legend_name(key: str) -> str:
        # the divergence names first: "div_rms" also matches d<field>_<norm> (as "iv"), which is how
        # the Divergence legend read "iv iv"
        short = {"div_rms": "rms", "div_max": "max", "div_l2": "L2"}
        if key in short:
            return short[key]
        m = re.fullmatch(r"d([a-z]+)_(rel|l2|rms|max)", key)
        if m:
            return {"theta": "θ"}.get(m.group(1), m.group(1))
        return describe(key).label

    # -- updates -------------------------------------------------------------
    def set_log(self, log: DiagnosticsLog) -> None:
        """Show another log (explore and a run each have their own)."""
        if log is not self.log:
            log.recording = self.record_btn.isChecked()
            self.log = log
            self.refresh(force=True)

    def set_export_dir(self, folder: str | Path) -> None:
        self._export_dir = Path(folder)

    def set_thermal(self, on: bool | None) -> None:
        """Whether the run shown has heat transfer: without, θ and heat columns are left out even if the
        solver reports them (the 3D solver keeps their columns, NaN or not). None: show what is there."""
        on = None if on is None else bool(on)
        if on != self._thermal:
            self._thermal = on
            self.refresh(force=True)

    @staticmethod
    def is_thermal_key(key: str) -> bool:
        return "theta" in key or describe(key).group == "Heat" or key.startswith("heat")

    def shown_keys(self, snap: dict[str, np.ndarray], keys: list[str]) -> list[str]:
        """The quantities worth showing: those with at least one finite value, and no temperature or heat
        quantities for a run without heat transfer (a car's diagnostics listed θ and plotted an empty
        "heat rate" axis)."""
        out = []
        for k in keys:
            col = snap.get(k)
            if col is None or not np.isfinite(col).any():
                continue
            if self._thermal is False and self.is_thermal_key(k):
                continue
            out.append(k)
        return out

    def _set_count_text(self, text: str) -> None:
        """Set the "940 samples · steps 10–9,400" line at a width that changes only with its number of
        digits. The label was resized on every refresh (proportional digits), the bar re-laid out a frame
        later, and in between the line ran under the Clear button ("steps 10–9,40(")."""
        lab = self.count_label
        if lab.text() == text:
            return
        template = re.sub(r"\d", "8", text)              # 8 is at least as wide as every digit here
        lab.setMinimumWidth(lab.fontMetrics().horizontalAdvance(template) + 2 if text else 0)   # int px
        lab.setText(text)

    def refresh(self, force: bool = False) -> None:
        """Redraw when the log changed since the last redraw (called by the panel's timer)."""
        log = self.log
        if not force and log.revision == self._seen:
            return
        if not self.isVisible() and not force:
            return                       # hidden: nothing to draw; the next show redraws
        self._seen = log.revision
        snap = log.snapshot()
        keys = self.shown_keys(snap, log.keys())
        n = len(snap.get("step", []))
        self.source_label.setText(log.source)
        if n == 0 or not keys:
            self.stack.setCurrentIndex(0)
            self._set_count_text("")
            self.csv_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            return
        self.stack.setCurrentIndex(1)
        self.csv_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        steps = snap["step"]
        span = f"steps {int(steps[0]):,}–{int(steps[-1]):,}" if n > 1 else f"step {int(steps[0]):,}"
        self._set_count_text(f"{n:,} sample{'s' if n != 1 else ''} · {span}")
        latest = log.latest()
        items = []
        for k in keys:
            q = describe(k)
            unit = f" {q.unit}" if q.unit else ""
            items.append((q.label, fmt_value(k, latest.get(k, math.nan)) + unit))
        self.values.update_items(items)
        if n < 2:
            self.plot_stack.setCurrentWidget(self.one_note)
            return
        self.plot_stack.setCurrentIndex(0)
        self._rebuild(keys)
        x = snap["step"] if self.axis.current() == "step" else snap["t"]
        # every column has the log's length by construction; a block handed over by a producer that
        # broke that rule is cut to the common length rather than failing every redraw
        ys = [snap.get(k, np.full(n, np.nan))[:len(x)] for k in self._curves]
        m = min([len(x)] + [len(y) for y in ys])
        x, ys = x[:m], [y[:m] for y in ys]
        xd, yd = _decimate(x, ys)
        spans: dict[str, list[float]] = {}
        for (k, c), y in zip(self._curves.items(), yd):
            q = describe(k)
            yy = np.where(y > 0, y, np.nan) if q.log else y
            ok = np.isfinite(xd) & np.isfinite(yy)
            c.setData(xd[ok], yy[ok])
            if q.log and ok.any():
                spans.setdefault(q.group, []).extend([float(np.nanmin(yy[ok])), float(np.nanmax(yy[ok]))])
        constant: dict[str, str] = {}
        by_group: dict[str, list[np.ndarray]] = {}
        for (k, _c), y in zip(self._curves.items(), yd):
            if not describe(k).log:
                by_group.setdefault(describe(k).group, []).append(y)
        for g, pi in self._panels.items():
            pi.enableAutoRange(x=True)
            cv = constant_value(by_group[g]) if g in by_group else None
            if cv is not None:
                # constant to float noise (CFL held at 0.5 by the adaptive step): one tick at the value
                # and "constant 0.500" in the title, not eight digits of noise on the axis
                from .kpi import fmt_sig
                half = max(abs(cv) * 0.05, 1e-12)
                pi.enableAutoRange(y=False)
                pi.setYRange(cv - half, cv + half, padding=0)
                pi.getAxis("left").setTicks([[(cv, fmt_sig(cv, 3))], []])
                constant[g] = f"constant {fmt_sig(cv, 3)}"
                continue
            if g in spans:
                # at least two decades around the data, so a decade tick is always labelled
                lo, hi = math.log10(min(spans[g])), math.log10(max(spans[g]))
                mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo) + 0.3, 1.0)
                pi.enableAutoRange(y=False)
                pi.setYRange(mid - half, mid + half, padding=0)
                pi.getAxis("left").setTicks(decade_ticks(mid - half, mid + half))
            else:
                pi.getAxis("left").setTicks(None)
                pi.enableAutoRange(y=True)
        if constant != self._constant:
            self._constant = constant
            self._apply_theme()
        self._fit_axes()
        # the tick labels are measured when the axes paint: fit again once they have
        QTimer.singleShot(80, self, self._fit_axes)

    #: Narrowest left axis [px]: the rotated group name plus "10⁻⁶".
    AXIS_MIN_W = 54

    def needed_axis_width(self) -> int:
        """The left-axis width every panel needs for its rotated name and its widest tick label.

        The labels' widths are known only once an axis has drawn them (pyqtgraph measures them while
        painting, ``AxisItem.textWidth``), so this follows the plots one refresh behind. A fixed
        54 px let "0.0070" on the time-step axis run into its rotated "Δt (L/U)" name."""
        need = self.AXIS_MIN_W
        for pi in self._panels.values():
            ax = pi.getAxis("left")
            st = ax.style
            w = float(ax.textWidth) + (st["tickTextOffset"][0] if st["showValues"] else 0) + max(0, st["tickLength"])
            if ax.label.isVisible():
                w += ax.label.boundingRect().height() * 0.8
            need = max(need, math.ceil(w) + 2)
        return int(need)

    def _fit_axes(self) -> None:
        """One left-axis width for all panels, wide enough for each (plot frames stay aligned)."""
        if not self._panels:
            return
        need = self.needed_axis_width()
        for pi in self._panels.values():
            ax = pi.getAxis("left")
            if ax.fixedWidth != need:
                ax.setWidth(need)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.refresh(force=True)

    def _on_record(self, on: bool) -> None:
        self.log.recording = bool(on)
        self.record_btn.setText("Record" if on else "Paused")

    def _clear(self) -> None:
        self.log.clear()
        self.refresh(force=True)

    def export_csv(self, path: str | None = None) -> str | None:
        """Write the log as CSV (asks for a file name without ``path``); returns the path written."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export diagnostics", str(self._export_dir / "diagnostics.csv"),
                                                  "CSV (*.csv)")
            if not path:
                return None
        out = str(self.log.to_csv(path))
        self.exported.emit(out)
        return out
