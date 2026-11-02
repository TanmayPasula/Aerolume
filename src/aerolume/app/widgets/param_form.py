"""Unit-aware input widgets and the auto-generated parameter form.

Every number a user types into Aerolume is a physical quantity, and users think
in whatever unit is on their drawing: "4.5 m", "180 mm", "120 km/h". The widgets
here therefore *display* a value in a chosen unit but *accept* any unit of the
same kind (parsed by :mod:`aerolume.core.units`) and always report SI. Pages never
convert units themselves, which is what keeps the model strictly SI.

* :class:`QuantitySpinBox` - a spin box for one quantity kind ("length", "speed",
  "angle", ...). Typing "180 mm" into a box displaying metres converts on commit.
  Angles and temperatures show proper symbols (``4.0°``, ``15.0 °C``).
* :class:`ParamForm` - builds property-grid rows from a list of ``ParamSpec``
  records (the geometry templates publish theirs; see DESIGN.md 5.1) and emits
  the whole value dict, in SI, whenever anything changes. Its label column has a
  fixed width so it lines up with the surrounding
  :class:`~aerolume.app.widgets.common.PropertyGroup` rows.

Also a few small helpers the Geometry and Setup pages share: the wheel guard
(scrolling an inspector must never change a value), :func:`svg_icon` for the
handful of glyphs the bundled icon set does not have, and the segmented control.
Everything is styled by the global stylesheet; colours only ever come from
:func:`aerolume.app.theme.tokens`.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any

from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics, QIcon, QIconEngine, QPainter, QPixmap, QTextLayout, QValidator
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QWidget,
)

from ...core import units

# Unit kinds that are counts rather than continuous quantities. Templates use
# these for e.g. the number of points or spokes; they get an integer spin box.
_INTEGER_KINDS = {"count", "int", "integer", "n", "points"}
# Dimensionless kinds that read better with a suffix than bare.
_SUFFIX_KINDS = {"percent": " %", "%": " %", "ratio": "", "fraction": "", "": "", "none": ""}
# Display suffixes that differ from the unit's ASCII name (guidelines section 2:
# proper symbols). Parsing accepts both spellings.
_DISPLAY_SUFFIX = {("angle", "deg"): "°", ("temperature", "C"): " °C"}

LABEL_WIDTH = 132       # property-grid label column used by the Geometry and Setup inspectors

# One decimal convention per quantity and display unit, so every length in an
# inspector reads the same way ("1.000 m" for a chord and for the Reynolds
# chord, not "1.00 m" beside "1.000 m"). Millimetre resolution for metres is
# what a 2D side profile or a model-scale section needs; a field whose step is
# finer than the convention still gets the decimals to show its step.
DECIMALS: dict[tuple[str, str], int] = {
    ("length", "m"): 3, ("length", "cm"): 1, ("length", "mm"): 1, ("length", "km"): 4,
    ("length", "in"): 2, ("length", "ft"): 3,
    ("angle", "deg"): 1, ("angle", "rad"): 4,
    ("speed", "m/s"): 2, ("speed", "km/h"): 1, ("speed", "mph"): 1, ("speed", "kn"): 1, ("speed", "ft/s"): 1,
}


def _decimals_for(step: float, unit_factor: float = 1.0) -> int:
    """Decimal places so that ``step`` (SI) is representable in the display unit.

    Showing four decimals for a 4.5 m car is noise, showing zero for a 0.12 m
    chord loses information; the step the template author chose is the best
    hint for the resolution that matters.
    """
    s = abs(step / unit_factor) if step else 0.0
    if s <= 0 or not math.isfinite(s):
        return 3
    # fewest decimals that represent the step exactly (0.25 -> 2, 0.005 -> 3, 1 -> 0)
    for d in range(7):
        if abs(round(s * 10**d) - s * 10**d) < 1e-6 * max(s * 10**d, 1.0):
            return d
    return 6


class QuantitySpinBox(QDoubleSpinBox):
    """A spin box holding one physical quantity, SI in and out.

    ``value()`` / ``setValue()`` keep their Qt meaning (the number in the display
    unit); use :meth:`si_value` / :meth:`set_si_value` from application code.
    ``siValueChanged(float)`` fires with the SI value whenever the user commits a
    change. The display unit can be switched at run time with :meth:`set_unit`
    without changing the SI value - this is what the speed unit selector does.

    Typed text may carry any unit of the same kind ("180 mm" in a metre box);
    the text is parsed with :func:`aerolume.core.units.parse`.

    ``significant`` (opt-in) keeps at least that many significant figures for
    small values: the decimals grow with the value's magnitude. A spin box rounds
    what it holds to its decimals, so a fixed two-decimal speed box showed the
    0.0301 m/s of a Re = 40 cylinder as 0.03 and could not take 0.0301 typed in.
    """

    siValueChanged = Signal(float)

    def __init__(self, kind: str, unit: str | None = None, *, minimum: float = -1e12,
                 maximum: float = 1e12, step: float | None = None, decimals: int | None = None,
                 show_suffix: bool = True, significant: int | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if kind not in units.UNITS:
            raise ValueError(f"unknown quantity kind {kind!r}")
        self.kind = kind
        self.unit = unit or units.DEFAULT_UNIT[kind]
        self._si_min, self._si_max = float(minimum), float(maximum)
        self._si_step = step
        self._fixed_decimals = decimals
        self._significant = significant
        self._exact_si: float | None = None    # the value set_si_value loaded (see set_unit)
        # False when a unit selector sits beside the box: "30.00 m/s [m/s]" says it twice
        self._show_suffix = bool(show_suffix)
        self.setKeyboardTracking(False)       # commit on Enter / focus-out, not per keystroke
        self.setAccelerated(True)
        self.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # The default minimum width is the width of the widest representable value
        # ("-1000000000.000 m"), which would force inspector panels wider than the
        # screen; real values are short, so allow the box to be narrow.
        self.setMinimumWidth(64)
        self._apply_unit()
        self.valueChanged.connect(self._value_changed)

    def _value_changed(self, _v: float) -> None:
        self._exact_si = None                    # the user's value is the one shown
        self.siValueChanged.emit(self.si_value())

    # -- unit handling ----------------------------------------------------
    def _factor(self) -> float:
        return units.UNITS[self.kind][self.unit]

    def display_suffix(self) -> str:
        """The suffix shown after the number (" m", "°", " km/h"); empty when hidden."""
        if not self._show_suffix:
            return ""
        return _DISPLAY_SUFFIX.get((self.kind, self.unit), f" {self.unit}")

    def decimals_for_unit(self) -> int:
        """Decimals for the current unit: fixed if given, else the per-quantity
        convention (:data:`DECIMALS`), widened when the step needs more."""
        if self._fixed_decimals is not None:
            return self._fixed_decimals
        f = self._factor()
        conv = DECIMALS.get((self.kind, self.unit))
        if conv is None:
            return _decimals_for(self._si_step, f) if self._si_step else 3
        if not self._si_step:
            return conv
        # widen only for a step finer than the convention resolves; an exact
        # decimal count of a converted step (1 m/s = 2.2369... mph) would be noise
        shown = abs(self._si_step / f)
        need = _decimals_for(shown)                  # exact for round steps (0.25 -> 2)
        if need > 4:                                 # a converted, non-round step
            need = math.ceil(-math.log10(shown) - 1e-9) if 0 < shown < 1 else 0
        return max(conv, min(need, 6))

    def decimals_for_value(self, shown: float) -> int:
        """Decimals to show ``shown`` (display unit): :meth:`decimals_for_unit`, widened to keep
        ``significant`` figures of a small value (at most 6 decimals)."""
        base = self.decimals_for_unit()
        if not self._significant or not math.isfinite(shown) or shown == 0:
            return base
        need = self._significant - 1 - math.floor(math.log10(abs(shown)))
        return max(base, min(need, 6))

    def _fit_decimals(self, shown: float) -> None:
        """Set the decimals for ``shown`` before it is stored (the spin box rounds to them)."""
        dec = self.decimals_for_value(shown)
        if dec != self.decimals():
            blocked = self.blockSignals(True)
            self.setDecimals(dec)
            self.blockSignals(blocked)

    def _apply_unit(self, keep_si: float | None = None) -> None:
        """Re-derive decimals, range, step and suffix for the current display unit."""
        f = self._factor()
        self.blockSignals(True)
        dec = self.decimals_for_unit() if keep_si is None else self.decimals_for_value(keep_si / f)
        self.setDecimals(dec)
        lo, hi = sorted((self._si_min / f, self._si_max / f))
        self.setRange(lo, hi)
        self.setSingleStep((self._si_step / f) if self._si_step else 10.0 ** (-max(dec - 1, 0)))
        self.setSuffix(self.display_suffix())
        if keep_si is not None:
            self.setValue(keep_si / f)
        self.blockSignals(False)

    def set_unit(self, unit: str) -> None:
        """Change the display unit, keeping the SI value.

        The value loaded with :meth:`set_si_value` is kept exactly while the box still shows it:
        converting the rounded display instead drifted on every switch (0.0301 m/s -> 0.108 km/h
        -> 0.0300 m/s)."""
        if unit not in units.UNITS[self.kind]:
            raise ValueError(f"unknown {self.kind} unit {unit!r}")
        if unit != self.unit:
            si = self.exact_si_value()
            self.unit = unit
            self._apply_unit(keep_si=si)
            self._exact_si = si

    def stepBy(self, steps: int) -> None:
        """Arrow / wheel steps; a significant-figure box sizes its decimals for the value it steps
        to first, so the stored value is the one shown (Qt rounds to the decimals as it sets)."""
        if self._significant:
            self._fit_decimals(self.value() + steps * self.singleStep())
        super().stepBy(steps)

    def set_si_range(self, minimum: float, maximum: float) -> None:
        si = self.si_value()
        self._si_min, self._si_max = float(minimum), float(maximum)
        self._apply_unit(keep_si=si)

    # -- SI accessors -----------------------------------------------------
    def si_value(self) -> float:
        return self.value() * self._factor()

    def exact_si_value(self) -> float:
        """The SI value loaded with :meth:`set_si_value`, unrounded, while the box still shows it;
        else :meth:`si_value`. Editors that write several boxes back at once (a vector, a domain
        box) read this, so committing one component does not round the others to the display:
        a 45.7 mm cavity shown as "0.046 m" once became 46 mm (+2 % in the Rayleigh number) when
        another edge was typed."""
        si = self.si_value()
        exact = self._exact_si
        if exact is not None and abs(exact - si) <= 0.5001 * 10.0 ** -self.decimals() * self._factor():
            return exact
        return si

    def set_si_value(self, value: float, *, emit: bool = False) -> None:
        """Set from SI. Silent by default: programmatic loads must not echo back as edits."""
        if self._significant:
            self._fit_decimals(float(value) / self._factor())
        if not emit:
            self.blockSignals(True)
        self.setValue(float(value) / self._factor())
        if not emit:
            self.blockSignals(False)
        self._exact_si = float(value)

    # -- text <-> value (accepts any unit of the kind) --------------------
    def _parse(self, text: str) -> float | None:
        t = text.strip()
        # Qt passes the text including our suffix; parse handles it, but a bare
        # suffix (user deleted the number) must be "incomplete", not an error.
        if self.suffix().strip() and t == self.suffix().strip():
            return None
        t = t.replace("°C", "C").replace("° C", "C")
        try:
            return units.parse(t, self.kind, default_unit=self.unit) / self._factor()
        except (ValueError, KeyError):
            return None

    def validate(self, text: str, pos: int):
        v = self._parse(text)
        if v is None:
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Acceptable, text, pos

    def valueFromText(self, text: str) -> float:
        v = self._parse(text)
        if v is not None and self._significant:
            self._fit_decimals(v)              # before Qt rounds the typed value to the decimals
        return self.value() if v is None else v

    def textFromValue(self, value: float) -> str:
        # No thousands separators: they would make the text ambiguous to re-parse
        # in locales that use ',' as decimal mark.
        return f"{value:.{self.decimals()}f}"


def _spec_get(spec: Any, key: str, default: Any = None) -> Any:
    """Read a ParamSpec field from a dataclass, namedtuple or dict alike."""
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


class _ElidedLabel(QLabel):
    """A property-grid label of fixed width that elides instead of pushing the editor."""

    def __init__(self, text: str, width: int) -> None:
        super().__init__()
        self._full = text
        self.setProperty("role", "pg-label")
        self.setFixedWidth(width)
        self.setMinimumHeight(24)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self) -> None:
        self.setText(QFontMetrics(self.font()).elidedText(self._full, Qt.TextElideMode.ElideRight,
                                                           max(self.width() - 2, 20)))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._elide()

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._elide()


class ParamForm(QWidget):
    """Property-grid rows generated from a list of ``ParamSpec`` records.

    A spec carries ``name, label, unit_kind, default, min, max, step`` and
    optionally ``choices`` and ``help`` (DESIGN.md 5.1; the geometry templates
    publish theirs). ``unit_kind`` picks the editor:

    * a kind known to :mod:`aerolume.core.units` ("length", "angle", ...) -> a
      :class:`QuantitySpinBox` (SI in and out, any unit accepted);
    * "count" (and integer defaults) -> integer spin box;
    * "bool" -> check box; "text" -> line edit; "choice" -> combo of ``choices``;
    * anything else ("ratio", unknown kinds) -> plain decimal box, so a kind
      this form does not know yet still gets a usable field.

    Rows are label | editor with a fixed ``label_width`` so the form lines up
    with the property group it sits in. The spec's ``help`` goes in the tooltip.

    Switch parameters gate the parameters they own: while "Wheels" is off,
    "Wheel radius" and "Wheel gap to road" are disabled (they change nothing,
    and an enabled field that does nothing reads as a bug). A spec may name
    its switch explicitly (``enabled_by``); otherwise a bool ``wheels`` / ``flap``
    owns the parameters named ``wheel_*`` / ``flap_*`` (see :func:`switch_dependents`).

    Emits ``valuesChanged(dict)`` with all current values after any user edit
    and ``paramChanged(str, object)`` with the field that changed (used to merge
    rapid edits of one field into a single undo step). :meth:`set_values`
    loads values without emitting.
    """

    valuesChanged = Signal(dict)
    paramChanged = Signal(str, object)

    def __init__(self, specs: list[Any] | None = None, parent: QWidget | None = None,
                 *, preferred_units: dict[str, str] | None = None, label_width: int = LABEL_WIDTH) -> None:
        super().__init__(parent)
        self._preferred = preferred_units or {}
        self._label_width = label_width
        self._fields: dict[str, QWidget] = {}
        self._labels: dict[str, QLabel] = {}
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(8)
        self._layout.setVerticalSpacing(4)
        self._layout.setColumnStretch(1, 1)
        self.set_specs(specs or [])

    # -- construction -----------------------------------------------------
    def set_specs(self, specs: list[Any]) -> None:
        """Rebuild the form for a new set of parameters."""
        clear_layout(self._layout)
        self._fields.clear()
        self._labels.clear()
        self._specs = list(specs)
        self._dependents = switch_dependents(self._specs)
        for r, spec in enumerate(self._specs):
            name = str(_spec_get(spec, "name"))
            text = str(_spec_get(spec, "label") or name.replace("_", " ").capitalize())
            w = self._make_field(spec)
            w.setObjectName(f"param_{name}")
            tip = str(_spec_get(spec, "help") or _spec_get(spec, "description") or "")
            full_tip = f"{text}\n{tip}" if tip else text
            w.setToolTip(full_tip)
            lab = _ElidedLabel(text, self._label_width)
            lab.setToolTip(full_tip)
            lab.setBuddy(w)
            self._fields[name] = w
            self._labels[name] = lab
            self._layout.addWidget(lab, r, 0)
            self._layout.addWidget(w, r, 1)
        if not self._specs:
            note = QLabel("No parameters")
            note.setProperty("role", "muted")
            self._layout.addWidget(note, 0, 0, 1, 2)
        self._update_enabled()
        guard_wheel(self)

    def _update_enabled(self) -> None:
        """Enable each switch's dependent rows only while the switch is on."""
        for switch, names in self._dependents.items():
            box = self._fields.get(switch)
            on = not isinstance(box, QCheckBox) or box.isChecked()
            for n in names:
                for w in (self._fields.get(n), self._labels.get(n)):
                    if w is not None:
                        w.setEnabled(on)

    def is_gated_off(self, name: str) -> bool:
        """True while ``name`` is disabled by its switch parameter."""
        w = self._fields.get(name)
        return w is not None and not w.isEnabled()

    def _make_field(self, spec: Any) -> QWidget:
        kind = str(_spec_get(spec, "unit_kind") or "").lower()
        default = _spec_get(spec, "default", 0.0)
        name = str(_spec_get(spec, "name"))
        if kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(default))
            cb.setMinimumHeight(24)
            cb.toggled.connect(lambda v, n=name: self._edited(n, bool(v)))
            return cb
        if kind == "text":
            le = keep_start_visible(QLineEdit(str(default if default is not None else "")))
            le.editingFinished.connect(lambda n=name, w=le: self._edited(n, w.text().strip()))
            return le
        if kind == "choice":
            combo = QComboBox()
            for c in _spec_get(spec, "choices") or ():
                combo.addItem(str(c).replace("_", " ").capitalize(), c)
            combo.setCurrentIndex(max(combo.findData(default), 0))
            combo.currentIndexChanged.connect(lambda _i, n=name, w=combo: self._edited(n, w.currentData()))
            return combo
        lo = _spec_get(spec, "min")
        hi = _spec_get(spec, "max")
        step = _spec_get(spec, "step")
        lo = -1e9 if lo is None else float(lo)
        hi = 1e9 if hi is None else float(hi)
        if kind in units.UNITS:
            unit = self._preferred.get(kind) or display_unit(kind)
            w = QuantitySpinBox(kind, unit, minimum=lo, maximum=hi, step=float(step) if step else None)
            w.set_si_value(float(default))
            w.siValueChanged.connect(lambda v, n=name: self._edited(n, v))
            return w
        is_int = kind in _INTEGER_KINDS or (
            isinstance(default, int) and not isinstance(default, bool)
            and (step is None or float(step).is_integer()))
        if is_int:
            s = QSpinBox()
            s.setRange(math.ceil(lo) if lo > -2**31 else -2**31 + 1,
                       math.floor(hi) if hi < 2**31 else 2**31 - 1)
            s.setSingleStep(int(step) if step else 1)
            s.setValue(int(default))
            s.setKeyboardTracking(False)
            s.setMinimumWidth(64)
            s.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            s.valueChanged.connect(lambda v, n=name: self._edited(n, int(v)))
            return s
        d = QDoubleSpinBox()
        d.setDecimals(_decimals_for(float(step)) if step else 3)
        d.setRange(lo, hi)
        d.setSingleStep(float(step) if step else 0.01)
        d.setValue(float(default))
        d.setSuffix(_SUFFIX_KINDS.get(kind, f" {kind}" if kind else ""))
        d.setKeyboardTracking(False)
        d.setMinimumWidth(64)
        d.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        d.valueChanged.connect(lambda v, n=name: self._edited(n, float(v)))
        return d

    # -- values -----------------------------------------------------------
    def _edited(self, name: str, value: Any) -> None:
        if name in self._dependents:
            self._update_enabled()
        self.paramChanged.emit(name, value)
        self.valuesChanged.emit(self.values())

    def field(self, name: str) -> QWidget:
        return self._fields[name]

    def label_for(self, name: str) -> QLabel:
        return self._labels[name]

    def names(self) -> list[str]:
        return list(self._fields)

    def values(self) -> dict[str, Any]:
        """Current values: SI floats, ints for counts, bools, strings for text/choice."""
        out: dict[str, Any] = {}
        for n, w in self._fields.items():
            if isinstance(w, QuantitySpinBox):
                out[n] = w.si_value()
            elif isinstance(w, QSpinBox):
                out[n] = int(w.value())
            elif isinstance(w, QDoubleSpinBox):
                out[n] = float(w.value())
            elif isinstance(w, QCheckBox):
                out[n] = w.isChecked()
            elif isinstance(w, QLineEdit):
                out[n] = w.text().strip()
            elif isinstance(w, QComboBox):
                out[n] = w.currentData()
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        """Load values (SI) without emitting change signals."""
        for n, v in values.items():
            w = self._fields.get(n)
            if w is None or v is None:
                continue
            w.blockSignals(True)
            try:
                if isinstance(w, QuantitySpinBox):
                    w.set_si_value(float(v))
                elif isinstance(w, QSpinBox):
                    w.setValue(round(float(v)))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(v))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QLineEdit):
                    if not w.hasFocus():            # never overwrite text being typed
                        set_line_text(w, str(v))
                elif isinstance(w, QComboBox):
                    w.setCurrentIndex(max(w.findData(v), 0))
            finally:
                w.blockSignals(False)
        self._update_enabled()

    def defaults(self) -> dict[str, Any]:
        return {str(_spec_get(s, "name")): _spec_get(s, "default") for s in self._specs}


def switch_dependents(specs: list[Any]) -> dict[str, list[str]]:
    """Map each on/off parameter to the parameters it switches.

    An explicit ``enabled_by`` on a spec wins. Otherwise the naming convention
    of the templates is used: a bool ``wheels`` (or ``wheel``) owns every
    parameter whose name starts with ``wheel_``, a bool ``flap`` owns
    ``flap_*``. Bools whose name is not such a stem (``closed_te``) own nothing.
    """
    names = [str(_spec_get(s, "name")) for s in specs]
    bools = [str(_spec_get(s, "name")) for s in specs if str(_spec_get(s, "unit_kind") or "").lower() == "bool"]
    out: dict[str, list[str]] = {}
    explicit = set()
    for s in specs:
        owner = _spec_get(s, "enabled_by")
        if owner and str(owner) in names:
            out.setdefault(str(owner), []).append(str(_spec_get(s, "name")))
            explicit.add(str(_spec_get(s, "name")))
    for b in bools:
        stems = {b + "_", (b[:-1] + "_") if b.endswith("s") else b + "_"}
        deps = [n for n in names if n != b and n not in explicit and any(n.startswith(st) for st in stems)]
        if deps:
            out.setdefault(b, []).extend(deps)
    return out


def wrap_elided(text: str, font, width: int, max_lines: int) -> list[str]:
    """``text`` word-wrapped to ``width`` px in at most ``max_lines`` lines, the
    last one elided with "…" if the text does not fit.

    QLabel can either wrap (unbounded height) or be elided by the caller (one
    line); a planner note wants both: two readable lines, then an ellipsis.
    QTextLayout does the line breaking so the result matches what Qt itself
    would wrap. (Its positions are UTF-16 units; the texts shown here are
    BMP-only, where those equal Python string indices.)
    """
    fm = QFontMetrics(font)
    width = max(int(width), 20)
    layout = QTextLayout(text, font)
    layout.beginLayout()
    spans: list[tuple[int, int]] = []
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        spans.append((line.textStart(), line.textLength()))
    layout.endLayout()
    if len(spans) <= max_lines:
        return [text[s:s + n].rstrip() for s, n in spans] or [""]
    out = [text[s:s + n].rstrip() for s, n in spans[:max_lines - 1]]
    rest = text[spans[max_lines - 1][0]:]
    out.append(fm.elidedText(rest, Qt.TextElideMode.ElideRight, width))
    return out


class ElidedLabel(QLabel):
    """A label that elides to its width and keeps the full text in the tooltip.

    For values that must not grow a panel or wrap into a paragraph (an
    Estimate value, a planner headline): the layout decides the width, the
    tooltip carries everything. With ``max_lines`` > 1 the text first wraps
    (up to that many lines) and only the last line is elided, so a sentence
    is readable without hovering yet still cannot grow into a paragraph.
    """

    def __init__(self, text: str = "", tooltip: str | None = None, *,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight, max_lines: int = 1,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self._tip: str | None = None
        self._mode = mode
        self._max_lines = max(1, int(max_lines))
        self._n_lines = 1
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(self._max_lines > 1)
        self.setSizePolicy(policy)
        self.set_full_text(text, tooltip)

    def set_full_text(self, text: str, tooltip: str | None = None) -> None:
        self._full = str(text)
        self._tip = tooltip
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self) -> None:
        if self._max_lines > 1:
            lines = wrap_elided(self._full, self.font(), self.width(), self._max_lines)
            shown = "\n".join(lines)
            elided = " ".join(lines) != " ".join(self._full.split())
            if len(lines) != self._n_lines:
                self._n_lines = len(lines)
                self.updateGeometry()
        else:
            shown = QFontMetrics(self.font()).elidedText(self._full, self._mode, max(self.width(), 20))
            elided = shown != self._full
        super().setText(shown)
        tip = self._tip if self._tip is not None else (self._full if elided else "")
        self.setToolTip(tip)

    def _height_for_lines(self, n: int) -> int:
        m = self.contentsMargins()
        return n * QFontMetrics(self.font()).lineSpacing() + m.top() + m.bottom()

    def hasHeightForWidth(self) -> bool:
        return self._max_lines > 1

    def heightForWidth(self, width: int) -> int:
        if self._max_lines <= 1:
            return super().heightForWidth(width)
        return self._height_for_lines(len(wrap_elided(self._full, self.font(), width, self._max_lines)))

    def minimumSizeHint(self) -> QSize:
        h = self._height_for_lines(self._n_lines) if self._max_lines > 1 else super().minimumSizeHint().height()
        return QSize(20, h)

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        h = self._height_for_lines(self._n_lines) if self._max_lines > 1 else super().sizeHint().height()
        return QSize(fm.horizontalAdvance(self._full) + 4, h)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._elide()

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._elide()


# ---------------------------------------------------------------------------
# small helpers shared by the Geometry and Setup pages
# ---------------------------------------------------------------------------

def display_unit(kind: str) -> str:
    """The user's preferred display unit for a quantity kind (settings), else SI."""
    unit = None
    with contextlib.suppress(Exception):          # preferences are optional
        from .. import settings
        unit = settings.get().display_unit(kind)
    if unit in units.UNITS.get(kind, {}):
        return str(unit)
    return units.DEFAULT_UNIT.get(kind, "")


def icon(name: str, colour: str | None = None, active: str | None = None) -> QIcon:
    """A themed line icon (active colour when checked), or an empty icon
    when the theme module is unavailable."""
    try:
        from .. import theme
        return theme.icon(name, colour, active)
    except Exception:  # noqa: BLE001
        return QIcon()


def _tok(key: str) -> str:
    """A theme colour by token name (a hex string passes through). ``plots.tokens()`` merges the
    theme's table over the v4 fallback table, so a theme that fails to load still gives v4 colours
    (this returned one v3 blue-grey for every key then)."""
    from .plots import tokens
    return str(tokens().get(key, key))


def _active_token(key: str) -> str:
    """The token a checked glyph is drawn in, through the theme's own active-icon mapping.

    Since theme v4 a checked chrome icon is ink, not the blue ``accent`` its call site asks for
    (``theme.ACTIVE_ICON_TOKEN``). The bundled icons get that mapping in the theme's icon engine;
    these hand-drawn glyphs have their own engine, so without this the checked "Dimensions" and
    "Points" toggles stayed blue beside an ink "Grid" and "Physics" in the same toolbar.
    """
    try:
        from .. import theme
        return str(theme.ACTIVE_ICON_TOKEN.get(key, key))
    except Exception:  # noqa: BLE001 - the theme module is optional for this widget's tests
        return key


# Glyphs the bundled icon set lacks, in its style: 16 px grid, 1.25 px round
# strokes, ``currentColor`` replaced at paint time.
SVG_GLYPHS = {
    "undo": "M6 3.5 3 6.5l3 3 M3.5 6.5H10a3 3 0 0 1 0 6H7",
    "redo": "M10 3.5l3 3-3 3 M12.5 6.5H6a3 3 0 0 0 0 6h3",
    "dimension": "M2.5 4.5v7 M13.5 4.5v7 M3.5 8h9 M5.5 6l-2 2 2 2 M10.5 6l2 2-2 2",
    "points": "M2.5 12.5 6 5l4.5 3 3-4.5",
    "points-dots": "M1.6 11.6h1.8v1.8H1.6z M5.1 4.1h1.8v1.8H5.1z M9.6 7.1h1.8v1.8H9.6z M12.6 2.6h1.8v1.8h-1.8z",
    # flow domain: frame corners around a small body
    "domain": "M2.5 5.5v-2h2.5 M11 3.5h2.5v2 M13.5 10.5v2H11 M5 12.5H2.5v-2 M5.5 8.5c1-1.4 3.6-1.4 5 0 "
              "c-1.4 0.8-3.6 0.8-5 0z",
    # 3D: an isometric box (meshes), a wing planform, rotate, mirror plane, a mesh triangle pair
    "cube": "M8 2.5 13.5 5.5v5.5L8 14 2.5 11V5.5z M2.5 5.5 8 8.5l5.5-3 M8 8.5V14",
    "wing": "M8 3 1.5 10l1 1.5L8 9l5.5 2.5 1-1.5z M8 9v3.5 M6.5 13h3",
    "rotate": "M12.8 8.5A4.8 4.8 0 1 1 11.2 4.4 M12.5 2.2v3h-3",
    "mirror": "M8 1.8v12.4 M6 4.5 2.5 11.5H6z M10 4.5l3.5 7H10z",
    "mesh": "M2.5 12.5 8 3l5.5 9.5z M5.2 7.8h5.6 M8 12.5 5.2 7.8 M8 12.5l2.8-4.7",
}


def _glyph_svg(name: str, colour: str) -> bytes:
    d = SVG_GLYPHS[name]
    fill = ""
    if name == "points":
        fill = f'<path d="{SVG_GLYPHS["points-dots"]}" fill="{colour}" stroke="none"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="{colour}" '
            f'stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"><path d="{d}"/>{fill}</svg>'
            ).encode()


class _GlyphEngine(QIconEngine):
    """Paints one of :data:`SVG_GLYPHS` at the requested size in a theme colour
    resolved at paint time (so the icon follows a theme switch)."""

    def __init__(self, name: str, colour: str, active: str | None) -> None:
        super().__init__()
        self._name, self._colour, self._active = name, colour, active

    def paint(self, painter: QPainter, rect, mode, state) -> None:
        if mode == QIcon.Mode.Disabled:
            c = _tok("faint")
        elif self._active and state == QIcon.State.On:
            c = _tok(_active_token(self._active))
        else:
            c = _tok(self._colour)
        side = min(rect.width(), rect.height())
        r = QRectF(rect.x() + (rect.width() - side) / 2, rect.y() + (rect.height() - side) / 2, side, side)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        QSvgRenderer(QByteArray(_glyph_svg(self._name, c))).render(painter, r)
        painter.restore()

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        pm = QPixmap(size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        self.paint(p, pm.rect(), mode, state)
        p.end()
        return pm

    def clone(self) -> QIconEngine:
        return _GlyphEngine(self._name, self._colour, self._active)


def svg_icon(name: str, colour: str = "text2", active: str | None = None) -> QIcon:
    """A line icon for a glyph the bundled set does not have ("undo", "redo",
    "dimension", "points", "domain"); falls back to the theme set for any other name."""
    if name in SVG_GLYPHS:
        return QIcon(_GlyphEngine(name, colour, active))
    return icon(name, colour, active)


def set_prop(widget: QWidget, name: str, value: str) -> None:
    """Set a style-hook property and re-polish so the stylesheet re-applies."""
    if widget.property(name) != value:
        widget.setProperty(name, value)
        st = widget.style()
        st.unpolish(widget)
        st.polish(widget)
        widget.update()


def set_line_text(edit: QLineEdit, text: str) -> None:
    """Set ``edit``'s text with its *start* in view.

    ``QLineEdit.setText`` leaves the cursor at the end and the field scrolls to
    keep the cursor visible, so a name longer than the field showed its tail with
    the first letters clipped ("eated cylinder · d 10 mm · ΔT 10 K"). A name is
    recognised by how it begins; the cursor goes back to position 0. A field the
    user is typing in is left alone (the caller checks focus, as before)."""
    edit.setText(text)
    edit.setCursorPosition(0)


class _StartOnLeave(QObject):
    """Event filter: when a line edit loses focus, scroll it back to its first character.

    After an edit the cursor sits where the user stopped typing, usually at the
    end, so a long name would stay clipped at the front once focus moves on.
    A focus loss to a popup (the field's own context menu) keeps the cursor."""

    def eventFilter(self, obj, event) -> bool:
        if (event.type() == QEvent.Type.FocusOut and isinstance(obj, QLineEdit)
                and event.reason() != Qt.FocusReason.PopupFocusReason):
            obj.setCursorPosition(0)
        return False


_START_ON_LEAVE: _StartOnLeave | None = None


def keep_start_visible(edit: QLineEdit) -> QLineEdit:
    """Make ``edit`` show its first character whenever it is not being edited (see :func:`set_line_text`)."""
    global _START_ON_LEAVE
    if _START_ON_LEAVE is None:
        _START_ON_LEAVE = _StartOnLeave()
    if not edit.property("_start_on_leave"):
        edit.installEventFilter(_START_ON_LEAVE)
        edit.setProperty("_start_on_leave", True)
    edit.setCursorPosition(0)
    return edit


class _WheelGuard(QObject):
    """Event filter: number fields and combos ignore the mouse wheel unless focused.

    Inspector panels scroll; without this, scrolling the panel with the cursor
    over a spin box silently changes a dimension or the flow speed.
    """

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and not obj.hasFocus():
            # hand the wheel to the enclosing scroll area so the panel still scrolls
            p = obj.parentWidget()
            while p is not None and not isinstance(p, QAbstractScrollArea):
                p = p.parentWidget()
            if p is not None:
                QCoreApplication.sendEvent(p.verticalScrollBar(), event)
            return True
        return False


_WHEEL_GUARD: _WheelGuard | None = None


def guard_wheel(root: QWidget) -> None:
    """Install the wheel guard on every spin box, combo box and slider under ``root``."""
    global _WHEEL_GUARD
    if _WHEEL_GUARD is None:
        _WHEEL_GUARD = _WheelGuard()
    for cls in (QAbstractSpinBox, QComboBox, QSlider):
        for w in root.findChildren(cls):
            if not w.property("_wheel_guarded"):
                w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                w.installEventFilter(_WHEEL_GUARD)
                w.setProperty("_wheel_guarded", True)


def clear_layout(layout) -> None:
    """Remove and delete every widget of a layout *now*.

    ``deleteLater`` alone keeps the old widgets painted until the event loop
    runs, so a panel rebuilt in place would briefly show both generations
    overlapping; detaching them first makes the rebuild atomic.
    """
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif it.layout() is not None:
            clear_layout(it.layout())


def label(text: str = "", role: str | None = None, *, wrap: bool = False) -> QLabel:
    """A QLabel carrying a theme ``role`` ("h3", "secondary", "muted", "mono", ...)."""
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    lab.setWordWrap(wrap)
    if wrap:
        lab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return lab


def hbox(*widgets: QWidget, stretch: int | None = 0, spacing: int = 4) -> QWidget:
    """Widgets side by side in one editor cell; ``stretch`` is the index that takes
    spare room (None: a trailing spacer does)."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(spacing)
    for i, x in enumerate(widgets):
        h.addWidget(x, 1 if stretch is not None and i == stretch else 0)
    if stretch is None:
        h.addStretch(1)
    return w


def small_button(text: str, tooltip: str = "", width: int | None = None) -> QPushButton:
    """A normal 24 px push button for an inspector row ("Rotate", "Resample")."""
    b = QPushButton(text)
    if tooltip:
        b.setToolTip(tooltip)
    if width:
        b.setFixedWidth(width)
    return b


class FormGrid(QWidget):
    """Label | editor rows with a fixed-width text2 label column.

    The fixed width is what lets several of these (and a :class:`ParamForm`)
    stacked inside one property group share a single editor edge. Long labels
    are elided; the full text goes in the tooltip.
    """

    def __init__(self, label_width: int = LABEL_WIDTH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label_width = label_width
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.grid.setColumnStretch(1, 1)
        self._row = 0
        self.labels: dict[str, QLabel] = {}

    def add_row(self, text: str, editor: QWidget, tooltip: str = "", key: str | None = None) -> QLabel:
        lab = _ElidedLabel(text, self._label_width)
        tip = f"{text}\n{tooltip}" if tooltip else (text if lab.text() != text else "")
        if tip:
            lab.setToolTip(tip)
        if tooltip and not editor.toolTip():
            editor.setToolTip(tooltip)
        self.grid.addWidget(lab, self._row, 0)
        self.grid.addWidget(editor, self._row, 1)
        self._row += 1
        self.labels[key or text] = lab
        return lab

    def add_widget(self, w: QWidget, *, indent: bool = False) -> QWidget:
        """A full-width widget, or (``indent``) one in the editor column only."""
        if indent:
            self.grid.addWidget(w, self._row, 1)
        else:
            self.grid.addWidget(w, self._row, 0, 1, 2)
        self._row += 1
        return w

    @staticmethod
    def set_label_text(lab: QLabel, text: str) -> None:
        """Relabel a row made by :meth:`add_row` (keeps the elision)."""
        if isinstance(lab, _ElidedLabel):
            lab._full = text
            lab._elide()
        else:
            lab.setText(text)


_LEVEL_GLYPH = {"error": ("error", "bad"), "warning": ("warning", "warn"), "ok": ("check-circle", "good"),
                "info": ("info", "text2")}


class StatusLine(QWidget):
    """One factual line with a small leading glyph: a dialog's or a field's status.

    ``set(level, text)`` with level "info" | "ok" | "warning" | "error"; the
    text is plain (text colour for problems, text2 otherwise) and the glyph
    carries the colour, as the guidelines ask for status (section 2).
    ``text()`` / ``level()`` read it back (tests, accessibility).
    """

    def __init__(self, text: str = "", level: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._glyph = QLabel()
        self._glyph.setFixedSize(16, 16)
        lay.addWidget(self._glyph, 0, Qt.AlignmentFlag.AlignTop)
        self._label = label("", None, wrap=True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._label, 1)
        self._level = level
        self.set(level, text)
        with contextlib.suppress(Exception):
            from .. import theme
            theme.changed.connect(self._retint)

    def _retint(self, *_a) -> None:
        self.set(self._level, self._label.text())

    def set(self, level: str, text: str) -> None:
        self._level = level if level in _LEVEL_GLYPH else "info"
        name, colour = _LEVEL_GLYPH[self._level]
        try:
            from .. import theme
            self._glyph.setPixmap(theme.render_icon(name, 14, colour, self.devicePixelRatioF()))
        except Exception:  # noqa: BLE001 - glyph is decoration
            self._glyph.clear()
        set_prop(self._label, "role", "" if self._level in ("error", "warning") else "secondary")
        self._label.setText(text)
        self._glyph.setVisible(bool(text))

    def text(self) -> str:
        return self._label.text()

    def level(self) -> str:
        return self._level


class SegmentedButtons(QFrame):
    """A row of mutually exclusive buttons (the theme's ``segmented="true"`` hook).

    ``options`` is a list of (key, text); ``currentChanged(str)`` fires with the
    key when the user picks another segment (not on :meth:`set_current`).
    """

    currentChanged = Signal(str)

    def __init__(self, options: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("segmented", "true")
        lay = QHBoxLayout(self)
        # the 24 px segments fill the 24 px frame, whose outline theme.paint_segmented_frame draws
        # (with a QSS border and 1 px margins they had 20 px and were clipped at the bottom)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.setFixedHeight(24)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        self._dividers: dict[str, QFrame] = {}       # the divider left of each button but the first
        for i, (key, text) in enumerate(options):
            if i:
                sep = QFrame()
                sep.setProperty("divider", "vertical")
                lay.addWidget(sep)
                self._dividers[key] = sep
            b = QPushButton(text)
            b.setCheckable(True)
            b.setObjectName(f"segment_{key}")
            from .. import theme
            b.setFixedHeight(theme.MIN_TARGET)       # WCAG 2.5.8: at least 24 px to hit
            self._group.addButton(b)
            lay.addWidget(b, 1)
            self.buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self.currentChanged.emit(k))

    def paintEvent(self, _e) -> None:
        from .. import theme
        theme.paint_segmented_frame(self)

    def set_visible_keys(self, keys) -> None:
        """Show only the segments named in ``keys``, in that order.

        The Setup page offers different analyses per design (CFD / Polar for a
        2D airfoil, Wing / 3D CFD for a wing); hiding a segment also hides the
        divider that would otherwise dangle at an edge of the control.
        """
        order = [k for k in keys if k in self.buttons]
        order += [k for k in self.buttons if k not in order]
        keep = set(keys)
        self._order = order
        lay = self.layout()
        while lay.count():                       # re-lay the segments in the order asked for
            lay.takeAt(0)
        first = True
        for k in order:
            b = self.buttons[k]
            on = k in keep
            sep = self._dividers.get(k)
            if sep is None:                      # the first segment of the constructor has none
                sep = QFrame()
                sep.setProperty("divider", "vertical")
                self._dividers[k] = sep
            lay.addWidget(sep)
            lay.addWidget(b, 1)
            b.setVisible(on)
            sep.setVisible(on and not first)
            if on:
                first = False

    def visible_keys(self) -> list[str]:
        """The shown segments, left to right."""
        return [k for k in getattr(self, "_order", list(self.buttons)) if not self.buttons[k].isHidden()]

    def current(self) -> str | None:
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return None

    def set_current(self, key: str) -> None:
        b = self.buttons.get(key)
        if b is not None:
            b.setChecked(True)
