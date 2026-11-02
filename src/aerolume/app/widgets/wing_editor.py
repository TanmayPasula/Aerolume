"""The finite-wing planform editor (DESIGN.md 7.11, XFLR5's wing editor in Aerolume's look).

A wing design (``kind == "wing"``) is a :class:`~aerolume.model.WingSpec`: a
list of spanwise sections (y, chord, leading-edge x and z, twist, airfoil) plus
the symmetry flag and the lattice size. The editor is two widgets that share it:

* :class:`WingEditor` - the workspace in the centre of the Geometry page: a
  live 3D preview of the lofted wing (the exact surface a 3D CFD run would
  get), a top-view planform drawing (outline, section chords, quarter-chord
  line, mean aerodynamic chord) and the section table. The table edits the
  leading edge either as positions (x LE, z LE) or as panel angles (sweep,
  dihedral): an angle edit moves that section *and everything outboard*, so
  the outer panels keep their own angles - what "change the sweep of the
  inner panel" means to a designer.
* :class:`WingInspector` - the right-hand property grid: quick generators
  (rectangular, tapered, elliptic, swept, with span / chords / sweep /
  dihedral / washout fields), the symmetry and discretisation options, the
  planform metrics (area, span, aspect ratio, MAC, taper, sweep) and the
  reference quantities with Auto switches, and the planform checks.

Neither widget changes the project: both emit ``specEdited(spec, text,
merge_key)`` / ``refEdited`` and the Geometry page turns that into one undo
step (:func:`update_wing`), as for every other design edit. The pure helpers
(:func:`generate_planform`, :func:`set_panel_angle`, :func:`insert_section`,
...) carry the geometry rules and are unit-tested on their own.
"""
from __future__ import annotations

import contextlib
import copy
import itertools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core import units
from ...model import Design, WingSection, WingSpec
from .. import theme
from .common import KeyValueList, PropertyGrid, ToolStrip
from .geometry_canvas import GeometryView3D, fmt_len, mix, qcolor, theme_tokens
from .param_form import (
    QuantitySpinBox,
    SegmentedButtons,
    StatusLine,
    clear_layout,
    display_unit,
    guard_wheel,
    hbox,
    label,
    small_button,
)

LABEL_W = 116
ROW_BUTTON_W = 64

#: Quick generators: (key, label).
SHAPES = (
    ("rectangular", "Rectangular"),
    ("tapered", "Tapered"),
    ("elliptic", "Elliptic"),
    ("swept", "Swept"),
)
#: NACA sections offered in the airfoil column (any 4- or 5-digit code can be typed).
COMMON_AIRFOILS = ("0009", "0012", "0015", "2412", "2415", "4412", "4415", "23012", "23015", "6409")
#: Reference fields of a wing design that can be fixed by typing (the rest follow the planform).
WING_REF_FIELDS = ("ref_area", "length_ref", "span_ref", "moment_ref3")
SPACINGS = (("cosine", "Cosine (denser at the tips)"), ("uniform", "Uniform"))

COL_Y, COL_CHORD, COL_X, COL_Z, COL_TWIST, COL_AIRFOIL = range(6)


def _g3():
    from ... import geometry3d
    return geometry3d


# ---------------------------------------------------------------------------
# pure helpers (no Qt)
# ---------------------------------------------------------------------------

def airfoil_label(name: str, names: dict[str, str] | None = None) -> str:
    """How a section's airfoil reads in the table: "NACA 2412" or a project design's name."""
    s = str(name).strip()
    if s.lower().startswith("design:"):
        did = s.split(":", 1)[1]
        return (names or {}).get(did, f"Missing design ({did})")
    code = _g3().naca_code(s)
    return f"NACA {code}" if code else s


def parse_airfoil(text: str, names: dict[str, str] | None = None) -> str | None:
    """The stored airfoil key for what the user typed or picked: a NACA code
    ("2412"), "design:<id>" for a project airfoil (by id or by its name), or
    None if it is neither."""
    t = str(text).strip()
    if not t:
        return None
    if t.lower().startswith("design:"):
        return t if t.split(":", 1)[1] in (names or {}) else None
    for did, n in (names or {}).items():
        if n == t:
            return f"design:{did}"
    code = _g3().naca_code(t)
    if code is None:
        return None
    from ..dialogs.import_dialogs import validate_naca
    return None if validate_naca(code) else code


def generate_planform(shape: str, *, span: float, root_chord: float, tip_chord: float,
                      sweep_c4_deg: float = 0.0, dihedral_deg: float = 0.0, washout_deg: float = 0.0,
                      airfoil: str = "2412", n_sections: int = 9, symmetric: bool = True) -> list[WingSection]:
    """Sections of a quick-generated planform (right half when ``symmetric``).

    * ``rectangular`` - constant chord ``root_chord``;
    * ``tapered`` / ``swept`` - linear chord from root to tip (a "swept" wing is a
      tapered one whose sweep the caller sets);
    * ``elliptic`` - chord ``root_chord * sqrt(1 - (y / b_half)^2)`` at
      ``n_sections`` stations clustered towards the tip (sin spacing), with the
      tip closed to ``tip_chord`` (a lattice needs a finite chord there).

    The quarter-chord line is straight and swept by ``sweep_c4_deg`` (so an
    unswept tapered or elliptic wing has an unswept quarter-chord line, the
    aerodynamically meaningful one); ``dihedral_deg`` raises the leading edge
    linearly; ``washout_deg`` twists the tip nose-down relative to the root.
    """
    if not (span > 0 and root_chord > 0):
        raise ValueError("span and root chord must be positive")
    if shape != "rectangular" and not tip_chord > 0:
        raise ValueError("the tip chord must be positive")
    half = 0.5 * span if symmetric else span
    if shape == "elliptic":
        n = max(3, int(n_sections))
        ys = half * np.sin(0.5 * np.pi * np.arange(n) / (n - 1))
        cs = root_chord * np.sqrt(np.clip(1.0 - (ys / half) ** 2, 0.0, 1.0))
        cs = np.maximum(cs, min(tip_chord, root_chord))
    else:
        ys = np.array([0.0, half])
        tip = root_chord if shape == "rectangular" else tip_chord
        cs = np.array([root_chord, tip])
    t = np.tan(np.radians(sweep_c4_deg))
    td = np.tan(np.radians(dihedral_deg))
    out = []
    for y, c in zip(ys, cs, strict=True):
        frac = float(y / half) if half > 0 else 0.0
        out.append(WingSection(y=float(y), chord=float(c), x_le=float(0.25 * root_chord + y * t - 0.25 * c),
                               z_le=float(y * td), twist_deg=float(-washout_deg * frac) + 0.0,
                               airfoil=str(airfoil)))
    out[0].x_le = 0.0                      # the root leading edge at the origin (DESIGN.md 7.2)
    return out


def generator_values(spec: WingSpec) -> dict[str, Any]:
    """The generator fields that describe an existing planform (to pre-fill them)."""
    secs = spec.sections
    try:
        pf = _g3().planform(spec)
    except ValueError:
        pf = {"span": 1.0, "sweep_c4_deg": 0.0, "dihedral_deg": 0.0}
    root, tip = secs[0], secs[-1]
    if len(secs) > 3:
        shape = "elliptic"
    elif math.isclose(root.chord, tip.chord, rel_tol=1e-9):
        shape = "rectangular"
    elif abs(pf["sweep_c4_deg"]) > 5:
        shape = "swept"
    else:
        shape = "tapered"
    return {"shape": shape, "span": float(pf["span"]), "root_chord": float(root.chord),
            "tip_chord": float(tip.chord), "sweep_c4_deg": float(pf["sweep_c4_deg"]),
            "dihedral_deg": float(pf["dihedral_deg"]), "washout_deg": float(root.twist_deg - tip.twist_deg),
            "airfoil": str(root.airfoil), "n_sections": max(len(secs), 9) if shape == "elliptic" else 9}


def panel_angles(sections: list[WingSection]) -> list[tuple[float, float]]:
    """(leading-edge sweep, dihedral) in degrees of the panel *inboard* of each
    section; (0, 0) for the root."""
    out = [(0.0, 0.0)]
    for a, b in itertools.pairwise(sections):
        dy = b.y - a.y
        if dy <= 0:
            out.append((float("nan"), float("nan")))
            continue
        out.append((math.degrees(math.atan2(b.x_le - a.x_le, dy)), math.degrees(math.atan2(b.z_le - a.z_le, dy))))
    return out


def set_panel_angle(sections: list[WingSection], index: int, *, sweep_deg: float | None = None,
                    dihedral_deg: float | None = None) -> list[WingSection]:
    """Sections with the panel inboard of ``index`` swept / raised to the given
    angle. The section and every section outboard of it move together, so the
    outer panels keep their own angles."""
    out = [copy.copy(s) for s in sections]
    if not 1 <= index < len(out):
        raise ValueError("the root section has no inboard panel")
    a, b = out[index - 1], out[index]
    dy = b.y - a.y
    if dy <= 0:
        raise ValueError("sections must be ordered by y")
    if sweep_deg is not None:
        if abs(sweep_deg) >= 89:
            raise ValueError("the sweep must be between -89 and 89 degrees")
        dx = a.x_le + dy * math.tan(math.radians(sweep_deg)) - b.x_le
        for s in out[index:]:
            s.x_le += dx
    if dihedral_deg is not None:
        if abs(dihedral_deg) >= 89:
            raise ValueError("the dihedral must be between -89 and 89 degrees")
        dz = a.z_le + dy * math.tan(math.radians(dihedral_deg)) - b.z_le
        for s in out[index:]:
            s.z_le += dz
    return out


def insert_section(sections: list[WingSection], index: int) -> list[WingSection]:
    """Sections with a new one after ``index``: halfway to the next section
    (every value interpolated, the inboard airfoil), or beyond the tip when
    ``index`` is the last section (continuing the last panel's taper, sweep and
    dihedral over half its width)."""
    out = [copy.copy(s) for s in sections]
    i = max(0, min(int(index), len(out) - 1))
    a = out[i]
    if i + 1 < len(out):
        b = out[i + 1]
        new = WingSection(y=0.5 * (a.y + b.y), chord=0.5 * (a.chord + b.chord), x_le=0.5 * (a.x_le + b.x_le),
                          z_le=0.5 * (a.z_le + b.z_le), twist_deg=0.5 * (a.twist_deg + b.twist_deg),
                          airfoil=a.airfoil)
    else:
        p = out[i - 1] if i > 0 else WingSection(y=a.y - max(a.chord, 1e-3), chord=a.chord, x_le=a.x_le,
                                                 z_le=a.z_le, twist_deg=a.twist_deg, airfoil=a.airfoil)
        t = 0.5
        dy = a.y - p.y
        new = WingSection(y=a.y + t * dy, chord=max(a.chord + t * (a.chord - p.chord), 0.25 * a.chord),
                          x_le=a.x_le + t * (a.x_le - p.x_le), z_le=a.z_le + t * (a.z_le - p.z_le),
                          twist_deg=a.twist_deg + t * (a.twist_deg - p.twist_deg), airfoil=a.airfoil)
    out.insert(i + 1, new)
    return out


def remove_section(sections: list[WingSection], index: int) -> list[WingSection]:
    """Sections without ``index`` (a wing keeps at least its root and tip)."""
    if len(sections) <= 2:
        raise ValueError("a wing needs at least two sections (root and tip)")
    out = [copy.copy(s) for s in sections]
    del out[int(index)]
    return out


def validate_spec(spec: WingSpec) -> str | None:
    """A plain-language problem with a planform, or None when it can be lofted and analysed."""
    try:
        _g3().planform(spec)
    except ValueError as exc:
        return str(exc).capitalize()
    if not (1 <= spec.n_chord <= 64 and 2 <= spec.n_span <= 400):
        return "Panel counts out of range."
    return None


def auto_wing_references(spec: WingSpec) -> dict[str, Any]:
    """References that follow the planform (geometry3d.design_from_wing): planform
    area, MAC, span, and the quarter-MAC point on the centre line."""
    pf = _g3().planform(spec)
    return {"ref_area": pf["area"], "length_ref": pf["mac"], "span_ref": pf["span"],
            "moment_ref3": (pf["mac_le_x"] + 0.25 * pf["mac"], 0.0, pf["mac_le_z"]), "force_ref": pf["mac"]}


def update_wing(design: Design, spec: WingSpec) -> Design:
    """A copy of a wing design with a new planform; references follow it except
    the ones the user typed (``source["manual_refs"]``)."""
    after = Design.from_dict(copy.deepcopy(design.to_dict()))
    after.wing = WingSpec.from_dict(copy.deepcopy(spec.to_dict()))
    after.symmetric = bool(spec.symmetric)
    refresh_wing_references(after)
    return after


def refresh_wing_references(design: Design) -> None:
    """Recompute the automatic references of a wing design in place."""
    if design.wing is None:
        return
    try:
        auto = auto_wing_references(design.wing)
    except ValueError:
        return                                   # an invalid planform keeps its old references
    manual = set(design.source.get("manual_refs", []))
    for f in WING_REF_FIELDS:
        if f not in manual:
            setattr(design, f, tuple(float(v) for v in auto[f]) if f == "moment_ref3" else float(auto[f]))
    design.force_ref = float(auto["force_ref"])


def metrics_rows(spec: WingSpec) -> list[tuple[str, str, str]]:
    """Planform metrics as (key, value, tooltip) rows for a key-value table."""
    try:
        pf = _g3().planform(spec)
    except ValueError as exc:
        return [("Planform", "Invalid", str(exc))]
    u = display_unit("length")

    def ln(v: float) -> str:
        return units.fmt(v, "length", u, 4)

    def deg(v: float) -> str:
        return f"{v:.1f}°".replace("-", "−")

    from .mesh_inspector import area_text
    return [
        ("Area", area_text(pf["area"]), "Planform area projected on the x–y plane (both halves)"),
        ("Span", ln(pf["span"]), "Tip to tip"),
        ("Aspect ratio", f"{pf['aspect_ratio']:.2f}", "b² / S"),
        ("Mean aerodynamic chord", ln(pf["mac"]), "MAC = (2/S) ∫ c² dy: the pitching-moment reference length"),
        ("MAC leading edge", f"x {pf['mac_le_x']:.3f}, y {pf['mac_y']:.3f} m",
         "Leading edge of the MAC and its spanwise station"),
        ("Taper ratio", f"{pf['taper']:.3f}", "Tip chord / root chord"),
        ("Sweep, quarter chord", deg(pf["sweep_c4_deg"]), "Root-to-tip quarter-chord line"),
        ("Sweep, leading edge", deg(pf["sweep_le_deg"]), "Root-to-tip leading edge"),
        ("Dihedral", deg(pf["dihedral_deg"]), "Root-to-tip leading edge"),
        ("Root / tip chord", f"{ln(pf['root_chord'])} / {ln(pf['tip_chord'])}", ""),
    ]


def project_airfoils(project) -> dict[str, str]:
    """{design id: name} of the project's 2D single-element airfoil designs (usable as sections)."""
    if project is None:
        return {}
    return {did: d.name for did, d in project.designs.items()
            if getattr(d, "dimension", 2) == 2 and d.kind == "airfoil" and len(d.bodies) == 1}


def airfoil_resolver(project) -> Callable[[str], np.ndarray] | None:
    """``airfoil_coords(name)`` for "design:<id>" sections: the outline of a project
    airfoil (the loft and the planner normalise it). None without a project."""
    if project is None:
        return None

    def resolve(name: str) -> np.ndarray:
        did = str(name).split(":", 1)[1] if ":" in str(name) else str(name)
        d = project.designs.get(did)
        if d is None or getattr(d, "dimension", 2) != 2 or not d.bodies:
            raise ValueError(f"airfoil {name!r} is not a 2D airfoil design of this project")
        return np.asarray(d.bodies[0].points, dtype=float)
    return resolve


# ---------------------------------------------------------------------------
# the top-view planform drawing
# ---------------------------------------------------------------------------

class PlanformSketch(QWidget):
    """Top view of the planform: span across, flow downwards, true shape (1:1).

    Draws the outline (both halves of a symmetric wing, the left one as its
    mirror image), each section's chord, the quarter-chord line (dashed), the
    mean aerodynamic chord and the span dimension. The selected section is
    drawn in the accent colour; clicking near a section selects it
    (``sectionClicked``).
    """

    sectionClicked = Signal(int)

    MARGIN = (28, 40, 28, 30)          # left, top, right, bottom (px): room for the dimensions

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._spec: WingSpec | None = None
        self._selected = -1
        self._map: tuple[float, float, float] | None = None       # (scale, ox, oy): y, x -> px
        with contextlib.suppress(Exception):
            theme.changed.connect(self.update)

    def sizeHint(self) -> QSize:
        return QSize(420, 260)

    def set_spec(self, spec: WingSpec | None) -> None:
        self._spec = spec
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = int(index)
        self.update()

    def _layout(self) -> tuple[float, float, float] | None:
        s = self._spec
        if s is None or len(s.sections) < 2:
            return None
        ys = np.array([q.y for q in s.sections])
        xl = np.array([q.x_le for q in s.sections])
        xt = xl + np.array([q.chord for q in s.sections])
        ylo, yhi = (-ys.max(), ys.max()) if s.symmetric else (ys.min(), ys.max())
        xlo, xhi = float(min(xl.min(), 0.0)), float(xt.max())
        ml, mt, mr, mb = self.MARGIN
        w, h = max(self.width() - ml - mr, 40), max(self.height() - mt - mb, 40)
        span, depth = max(yhi - ylo, 1e-9), max(xhi - xlo, 1e-9)
        k = min(w / span, h / depth)
        ox = ml + (w - k * span) / 2 - k * ylo
        oy = mt + (h - k * depth) / 2 - k * xlo
        return k, ox, oy

    def _pt(self, y: float, x: float) -> QPointF:
        k, ox, oy = self._map
        return QPointF(ox + k * y, oy + k * x)

    def paintEvent(self, _ev) -> None:
        tok = theme_tokens()
        p = QPainter(self)
        p.fillRect(self.rect(), qcolor(tok.get("surface", "#ffffff")))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = QFont(self.font())
        font.setPointSizeF(8.0)
        p.setFont(font)
        fm = QFontMetricsF(font)
        p.setPen(qcolor(tok["muted"]))
        p.drawText(QPointF(10, 6 + fm.ascent()), "Planform (top view, flow downwards)")
        self._map = self._layout()
        s = self._spec
        if self._map is None or s is None:
            p.end()
            return
        secs = s.sections
        halves = (1.0, -1.0) if s.symmetric else (1.0,)
        text, text2, muted, accent = (qcolor(tok["text"]), qcolor(tok["text2"]), qcolor(tok["muted"]),
                                      qcolor(tok["accent"]))

        def pen(c, w=1.0, style=Qt.PenStyle.SolidLine):
            q = QPen(c, w, style)
            q.setCosmetic(True)
            return q

        # centre line
        if s.symmetric:
            a, b = self._pt(0, min(q.x_le for q in secs) - 0.08 * secs[0].chord), \
                self._pt(0, max(q.x_le + q.chord for q in secs) + 0.08 * secs[0].chord)
            p.setPen(pen(muted, 1.0, Qt.PenStyle.DashDotLine))
            p.drawLine(a, b)
        for sgn in halves:
            path = QPainterPath()
            le = [self._pt(sgn * q.y, q.x_le) for q in secs]
            te = [self._pt(sgn * q.y, q.x_le + q.chord) for q in secs]
            path.addPolygon(QPolygonF(le + te[::-1] + [le[0]]))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor(tok["text"], 18))
            p.drawPath(path)
            # section chords
            for i, q in enumerate(secs):
                sel = i == self._selected
                p.setPen(pen(accent if sel else muted, 2.0 if sel else 1.0))
                p.drawLine(le[i], te[i])
            # quarter-chord line
            c4 = [self._pt(sgn * q.y, q.x_le + 0.25 * q.chord) for q in secs]
            p.setPen(pen(text2, 1.0, Qt.PenStyle.DashLine))
            p.drawPolyline(QPolygonF(c4))
            # outline
            p.setPen(pen(text, 1.25))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(le))
            p.drawPolyline(QPolygonF(te))
            p.drawLine(le[-1], te[-1])
            if not s.symmetric:
                p.drawLine(le[0], te[0])
        # mean aerodynamic chord
        with contextlib.suppress(ValueError):
            pf = _g3().planform(s)
            for sgn in halves:
                a = self._pt(sgn * pf["mac_y"], pf["mac_le_x"])
                b = self._pt(sgn * pf["mac_y"], pf["mac_le_x"] + pf["mac"])
                p.setPen(pen(mix(tok["accent"], tok.get("surface", "#ffffff"), 0.35), 2.0))
                p.drawLine(a, b)
            a = self._pt(pf["mac_y"], pf["mac_le_x"] + pf["mac"])
            p.setPen(text2)
            p.drawText(QPointF(a.x() + 4, a.y() + fm.ascent()), f"MAC {fmt_len(pf['mac'])} m")
            # span dimension above the wing
            y0 = -max(q.y for q in secs) if s.symmetric else secs[0].y
            y1 = secs[-1].y
            top = min(self._pt(0, min(q.x_le for q in secs)).y(), self._pt(y1, secs[-1].x_le).y()) - 14
            pa, pb = self._pt(y0, 0), self._pt(y1, 0)
            pa.setY(top)
            pb.setY(top)
            p.setPen(pen(text2))
            p.drawLine(pa, pb)
            for tip in (pa, pb):
                p.drawLine(QPointF(tip.x(), tip.y() - 4), QPointF(tip.x(), tip.y() + 4))
            lab = f"b = {fmt_len(pf['span'])} m"
            tw = fm.horizontalAdvance(lab) + 8
            box = QRectF((pa.x() + pb.x()) / 2 - tw / 2, top - fm.height() / 2 - 1, tw, fm.height() + 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor(tok.get("surface", "#ffffff")))
            p.drawRect(box)
            p.setPen(text2)
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, lab)
        p.end()

    def mousePressEvent(self, ev) -> None:
        s, m = self._spec, self._map
        if s is None or m is None or ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)
        k, ox, _oy = m
        px = ev.position()
        y = abs((px.x() - ox) / k) if s.symmetric else (px.x() - ox) / k
        d = [abs(y - q.y) * k for q in s.sections]
        i = int(np.argmin(d))
        if d[i] < 12:
            self.sectionClicked.emit(i)


# ---------------------------------------------------------------------------
# the section table
# ---------------------------------------------------------------------------

class _AirfoilDelegate(QStyledItemDelegate):
    """An editable combo for the airfoil column: common NACA sections and the
    project's airfoil designs; any NACA code can be typed."""

    def __init__(self, table: SectionTable) -> None:
        super().__init__(table)
        self._table = table

    def createEditor(self, parent, option, index):
        if index.column() != COL_AIRFOIL:
            return super().createEditor(parent, option, index)
        c = QComboBox(parent)
        c.setEditable(True)
        c.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for code in COMMON_AIRFOILS:
            c.addItem(f"NACA {code}")
        names = self._table.airfoil_names
        if names:
            c.insertSeparator(c.count())
            for n in names.values():
                c.addItem(n)
        return c

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            editor.setCurrentText(str(index.data()))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText())
        else:
            super().setModelData(editor, model, index)


class SectionTable(QTableWidget):
    """The wing's sections, root first: y, chord, leading edge (as positions or
    as panel angles), twist and airfoil. Edits are parsed with units ("120 mm",
    "4 deg") and reported as a new section list (``sectionsEdited``); a value that
    makes the planform invalid is refused with a reason (``problem``)."""

    sectionsEdited = Signal(list, str, str)        # sections, undo text, merge key
    problem = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 6, parent)
        self.setObjectName("section_table")
        self.verticalHeader().setVisible(True)
        self.verticalHeader().setDefaultSectionSize(theme.ROW_H + 2)
        self.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                             | QAbstractItemView.EditTrigger.EditKeyPressed
                             | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setAlternatingRowColors(False)
        hh = self.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setItemDelegateForColumn(COL_AIRFOIL, _AirfoilDelegate(self))
        self.angles = False                 # leading edge as panel angles instead of positions
        self.airfoil_names: dict[str, str] = {}
        self._sections: list[WingSection] = []
        self._loading = False
        self.itemChanged.connect(self._item_changed)

    def sections(self) -> list[WingSection]:
        return [copy.copy(s) for s in self._sections]

    def set_angles(self, on: bool) -> None:
        self.angles = bool(on)
        self.set_sections(self._sections)

    def set_sections(self, sections: list[WingSection], names: dict[str, str] | None = None) -> None:
        if names is not None:
            self.airfoil_names = dict(names)
        self._sections = [copy.copy(s) for s in sections]
        lu = display_unit("length")
        heads = [f"y ({lu})", f"Chord ({lu})",
                 "Sweep LE (°)" if self.angles else f"x LE ({lu})",
                 "Dihedral (°)" if self.angles else f"z LE ({lu})", "Twist (°)", "Airfoil"]
        self.setHorizontalHeaderLabels(heads)
        tips = ["Spanwise station; the root is at y = 0 for a symmetric wing",
                "Chord length", "Leading-edge sweep of the panel inboard of this section (moves this section "
                "and all outboard ones)" if self.angles else "Leading-edge position along the flow",
                "Dihedral of the panel inboard of this section (moves this section and all outboard ones)"
                if self.angles else "Leading-edge height", "Nose-up twist about the leading edge",
                "A NACA 4- or 5-digit code, or an airfoil design of this project"]
        for c, tip in enumerate(tips):
            it = self.horizontalHeaderItem(c)
            if it is not None:
                it.setToolTip(tip)
                it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c == COL_AIRFOIL
                                     else Qt.AlignmentFlag.AlignRight) | Qt.AlignmentFlag.AlignVCenter)
        sel = self.currentRow()
        self._loading = True
        try:
            self.setRowCount(len(self._sections))
            ang = panel_angles(self._sections)
            factor = units.UNITS["length"][lu]
            for r, s in enumerate(self._sections):
                vals = [s.y / factor, s.chord / factor,
                        ang[r][0] if self.angles else s.x_le / factor,
                        ang[r][1] if self.angles else s.z_le / factor, s.twist_deg]
                for c, v in enumerate(vals):
                    root_angle = self.angles and r == 0 and c in (COL_X, COL_Z)
                    dec = 1 if c in (COL_TWIST,) or (self.angles and c in (COL_X, COL_Z)) else \
                        (3 if lu == "m" else 1)
                    if abs(v) < 0.5 * 10.0 ** -dec:
                        v = 0.0                    # never "−0.000"
                    text = "—" if root_angle else f"{v:.{dec}f}".replace("-", "−")
                    it = QTableWidgetItem(text)
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    if root_angle:
                        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        it.setToolTip("The root has no inboard panel")
                    self.setItem(r, c, it)
                af = QTableWidgetItem(airfoil_label(s.airfoil, self.airfoil_names))
                af.setData(Qt.ItemDataRole.UserRole, s.airfoil)
                self.setItem(r, COL_AIRFOIL, af)
            self.setVerticalHeaderLabels(["Root" if r == 0 else ("Tip" if r == len(self._sections) - 1 else str(r + 1))
                                          for r in range(len(self._sections))])
        finally:
            self._loading = False
        if 0 <= sel < self.rowCount():
            self.selectRow(sel)

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        r, c = item.row(), item.column()
        if not 0 <= r < len(self._sections):
            return
        text = item.text().strip().replace("−", "-")
        secs = [copy.copy(s) for s in self._sections]
        s = secs[r]
        what = ""
        try:
            if c == COL_AIRFOIL:
                key = parse_airfoil(text, self.airfoil_names)
                if key is None:
                    raise ValueError(f"“{text}” is not a NACA 4- or 5-digit code or a project airfoil")
                if key == s.airfoil:
                    QTimer.singleShot(0, lambda: self.set_sections(self._sections))   # show the label again
                    return
                s.airfoil = key
                what = "airfoil"
            elif c == COL_TWIST or (self.angles and c in (COL_X, COL_Z)):
                v = units.parse(text, "angle", default_unit="deg")
                if c == COL_TWIST:
                    s.twist_deg = float(v)
                    what = "twist"
                elif c == COL_X:
                    secs = set_panel_angle(secs, r, sweep_deg=float(v))
                    what = "sweep"
                else:
                    secs = set_panel_angle(secs, r, dihedral_deg=float(v))
                    what = "dihedral"
            else:
                v = units.parse(text, "length", default_unit=display_unit("length"))
                if c == COL_Y:
                    s.y = float(v)
                    what = "y"
                elif c == COL_CHORD:
                    if not v > 0:
                        raise ValueError("the chord must be positive")
                    s.chord = float(v)
                    what = "chord"
                elif c == COL_X:
                    s.x_le = float(v)
                    what = "x LE"
                else:
                    s.z_le = float(v)
                    what = "z LE"
            if c == COL_Y:
                secs.sort(key=lambda q: q.y)
            err = validate_spec(WingSpec(sections=secs))
            if err:
                raise ValueError(err)
        except (ValueError, KeyError) as exc:
            msg = str(exc)[:1].upper() + str(exc)[1:]
            # rebuilding the rows inside itemChanged would replace the item Qt is still
            # committing from its editor: refuse on the next event-loop turn instead
            QTimer.singleShot(0, lambda: (self.problem.emit(msg), self.set_sections(self._sections)))
            return
        label_ = "Root" if r == 0 else ("Tip" if r == len(self._sections) - 1 else f"Section {r + 1}")
        self._sections = secs
        QTimer.singleShot(0, lambda: self.sectionsEdited.emit(secs, f"{label_}: {what}", f"sec:{r}:{c}"))


# ---------------------------------------------------------------------------
# the workspace
# ---------------------------------------------------------------------------

class WingEditor(QWidget):
    """The centre of the Geometry page for a wing design (see the module docstring).

    ``specEdited(WingSpec, str, str)``: the planform as edited, the undo text and
    a merge key (consecutive edits of one cell merge into one undo step).
    """

    specEdited = Signal(object, str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._design: Design | None = None
        self._resolver: Callable | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # one column: the 3D preview, the planform as a full-width strip (a wing is far
        # wider than deep, so beside the 3D view it was drawn tiny), the section table
        split = QSplitter(Qt.Orientation.Vertical)
        split.setObjectName("wing_split")
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        self.view3d = GeometryView3D()
        self.view3d.setObjectName("wing_view3d")
        self.view3d.setAcceptDrops(False)
        self.view3d.set_placeholder("No planform", "The sections do not describe a valid wing.")
        self.view3d.show_reference = True             # the moment point (quarter MAC) lies on the wing
        self.sketch = PlanformSketch()
        self.sketch.setObjectName("planform_sketch")
        self.sketch.setMinimumHeight(130)
        split.addWidget(self.view3d)
        split.addWidget(self.sketch)
        split.addWidget(self._build_table_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setStretchFactor(2, 0)
        split.setSizes([440, 190, 220])
        self._split = split
        # the table is sized to its rows (fit_panes) until the user drags a splitter handle;
        # setSizes does not emit splitterMoved, so this only hears the user
        self._user_sized = False
        split.splitterMoved.connect(self._split_moved)
        lay.addWidget(split)
        self.sketch.sectionClicked.connect(self.select_section)
        self.table.itemSelectionChanged.connect(lambda: self.sketch.set_selected(self.table.currentRow()))
        self.table.sectionsEdited.connect(self._sections_edited)
        self.table.problem.connect(lambda t: self.status.set("error", t))

    def _build_table_panel(self) -> QWidget:
        w = QFrame()
        w.setProperty("panel", "true")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = ToolStrip()
        bar.setObjectName("section_tools")
        bar.add_widget(label("Sections", "h3"))
        bar.row.addSpacing(8)
        self.add_button = bar.add_button("plus", "Add tip", "Add a section beyond the tip")
        self.insert_button = bar.add_button(None, "Insert", "Insert a section halfway to the next one")
        self.remove_button = bar.add_button("minus", "Remove", "Remove the selected section")
        self.add_button.clicked.connect(lambda: self._insert(len(self.table.sections()) - 1, "Add tip section"))
        self.insert_button.clicked.connect(lambda: self._insert(max(self.table.currentRow(), 0), "Insert section"))
        self.remove_button.clicked.connect(self._remove)
        bar.add_stretch()
        bar.add_widget(label("Leading edge", "secondary"))
        bar.row.addSpacing(4)
        self.le_mode = SegmentedButtons([("position", "Position"), ("angles", "Angles")])
        self.le_mode.setObjectName("le_mode")
        self.le_mode.setFixedWidth(150)
        self.le_mode.set_current("position")
        self.le_mode.buttons["position"].setToolTip("Edit the leading edge as x and z positions")
        self.le_mode.buttons["angles"].setToolTip("Edit the leading edge as panel sweep and dihedral angles")
        self.le_mode.currentChanged.connect(lambda k: self.table.set_angles(k == "angles"))
        bar.add_widget(self.le_mode)
        v.addWidget(bar)
        self.table = SectionTable()
        self.table.setProperty("flat", "true")
        v.addWidget(self.table, 1)
        foot = QWidget()
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(12, 4, 12, 6)
        self.status = StatusLine()
        self.status.setObjectName("wing_status")
        fl.addWidget(self.status, 1)
        v.addWidget(foot)
        return w

    # -- pane sizes -----------------------------------------------------------------
    #: The table never takes more than this share of the height, nor the sketch.
    TABLE_MAX_SHARE = 0.34
    SKETCH_SHARE = 0.24

    def _split_moved(self, *_a) -> None:
        self._user_sized = True

    def fit_panes(self) -> None:
        """Give the table the height of its rows (plus its tool strip and status line),
        the planform sketch about a quarter, and the 3D preview the rest.

        A two-section wing showed eight empty table rows under its two real ones while
        the 3D preview, the view people look at, was squeezed to a strip at 1280 x 800.
        Once the user drags a handle their sizes are kept.
        """
        if self._user_sized:
            return
        total = sum(self._split.sizes())
        if total <= 0:
            return
        t = self.table
        rows = max(t.rowCount(), 2) + 1                       # one free row says "more fit"
        chrome = self._table_chrome_h()
        want = chrome + t.horizontalHeader().height() + rows * t.verticalHeader().defaultSectionSize() + 4
        table_h = int(min(max(want, 120), self.TABLE_MAX_SHARE * total))
        sketch_h = int(max(self.sketch.minimumHeight(), self.SKETCH_SHARE * total))
        self._split.setSizes([max(total - table_h - sketch_h, 120), sketch_h, table_h])

    def _table_chrome_h(self) -> int:
        panel = self.table.parentWidget()
        lay = panel.layout() if panel is not None else None
        h = 0
        for i in range(lay.count() if lay is not None else 0):
            w = lay.itemAt(i).widget()
            if w is not None and w is not self.table:
                h += w.sizeHint().height()
        return h

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.fit_panes()

    # -- state -> widgets ---------------------------------------------------------
    def design(self) -> Design | None:
        return self._design

    def set_design(self, design: Design | None, *, airfoil_names: dict[str, str] | None = None,
                   airfoil_coords: Callable | None = None) -> None:
        """Show a wing design: table, planform drawing and the lofted 3D preview."""
        self._design = design
        self._resolver = airfoil_coords
        spec = design.wing if design is not None else None
        self.table.set_sections(spec.sections if spec is not None else [], airfoil_names)
        self.sketch.set_spec(spec)
        self.sketch.set_selected(self.table.currentRow())
        self.remove_button.setEnabled(spec is not None and len(spec.sections) > 2)
        meshes = []
        if spec is not None:
            try:
                meshes = [_g3().loft(spec, airfoil_coords=airfoil_coords, name=design.name or "Wing")]
                self.status.set("info", "")
            except ValueError as exc:
                self.status.set("error", f"The wing cannot be lofted: {exc}")
        self.view3d.set_design(design, meshes=meshes, ground=False)
        self.fit_panes()                             # the row count may have changed

    def select_section(self, index: int) -> None:
        if 0 <= index < self.table.rowCount():
            self.table.selectRow(index)
            self.sketch.set_selected(index)

    # -- edits ----------------------------------------------------------------------
    def _spec_with(self, sections: list[WingSection]) -> WingSpec | None:
        d = self._design
        if d is None or d.wing is None:
            return None
        w = d.wing
        return WingSpec(sections=sections, symmetric=w.symmetric, n_chord=w.n_chord, n_span=w.n_span,
                        spacing=w.spacing)

    def _sections_edited(self, sections: list, text: str, key: str) -> None:
        spec = self._spec_with(sections)
        if spec is not None:
            self.status.set("info", "")
            self.specEdited.emit(spec, text, key)

    def _insert(self, index: int, text: str) -> None:
        secs = self.table.sections()
        if not secs:
            return
        new = insert_section(secs, index)
        spec = self._spec_with(new)
        err = validate_spec(spec) if spec is not None else "no wing"
        if err:
            self.status.set("error", err)
            return
        self.specEdited.emit(spec, text, "")
        self.select_section(index + 1)

    def _remove(self) -> None:
        secs = self.table.sections()
        r = self.table.currentRow()
        if not 0 <= r < len(secs):
            self.status.set("error", "Select the section to remove.")
            return
        try:
            new = remove_section(secs, r)
        except ValueError as exc:
            self.status.set("error", str(exc).capitalize())
            return
        spec = self._spec_with(new)
        err = validate_spec(spec) if spec is not None else "no wing"
        if err:
            self.status.set("error", err)
            return
        self.specEdited.emit(spec, "Remove section", "")


# ---------------------------------------------------------------------------
# the inspector
# ---------------------------------------------------------------------------

def _toggle(text: str, tooltip: str) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setCheckable(True)
    b.setProperty("variant", "toggle")
    b.setToolTip(tooltip)
    b.setFixedSize(ROW_BUTTON_W, 24)
    return b


def _spacer(width: int) -> QWidget:
    w = QLabel()
    w.setFixedWidth(width)
    return w


class WingInspector(QWidget):
    """Right-hand property grid of a wing design: Generate, Wing, Planform,
    References, Diagnostics (see the module docstring).

    Signals: ``specEdited(WingSpec, str, str)``, ``refEdited(str, object)``,
    ``autoRefToggled(str, bool)``.
    """

    specEdited = Signal(object, str, str)
    refEdited = Signal(str, object)
    autoRefToggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._design: Design | None = None
        self._loading = False
        self._airfoil_names: dict[str, str] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.grid = PropertyGrid(label_width=LABEL_W)
        self.grid.setObjectName("wing_properties")
        self._build_generate()
        self._build_wing()
        self._build_metrics()
        self._build_refs()
        self._build_diagnostics()
        self.grid.add_stretch()
        lay.addWidget(self.grid.scrollable())
        guard_wheel(self)

    def _build_generate(self) -> None:
        g = self.grid.add_group("Generate", key="generate", expanded=False)
        self.generate_group = g
        lu = display_unit("length")
        self.shape = QComboBox()
        self.shape.setObjectName("wing_shape")
        for k, t in SHAPES:
            self.shape.addItem(t, k)
        self.shape.currentIndexChanged.connect(self._shape_changed)
        g.add_row("Shape", self.shape)
        self.gen_span = QuantitySpinBox("length", lu, minimum=1e-3, maximum=1e3, step=0.05)
        self.gen_root = QuantitySpinBox("length", lu, minimum=1e-4, maximum=1e2, step=0.01)
        self.gen_tip = QuantitySpinBox("length", lu, minimum=1e-4, maximum=1e2, step=0.01)
        self.gen_sweep = QuantitySpinBox("angle", "deg", minimum=-60, maximum=60, step=1, decimals=1)
        self.gen_dihedral = QuantitySpinBox("angle", "deg", minimum=-30, maximum=30, step=0.5, decimals=1)
        self.gen_washout = QuantitySpinBox("angle", "deg", minimum=-10, maximum=10, step=0.5, decimals=1)
        for w, n in ((self.gen_span, "gen_span"), (self.gen_root, "gen_root"), (self.gen_tip, "gen_tip"),
                     (self.gen_sweep, "gen_sweep"), (self.gen_dihedral, "gen_dihedral"),
                     (self.gen_washout, "gen_washout")):
            w.setObjectName(n)
        g.add_row("Span", self.gen_span, tooltip="Tip to tip (both halves of a symmetric wing)")
        g.add_row("Root chord", self.gen_root)
        self._tip_row = g.add_row("Tip chord", self.gen_tip, tooltip="Elliptic wings: the closing chord at the tip")
        g.add_row("Sweep (c/4)", self.gen_sweep, tooltip="Sweep of the quarter-chord line")
        g.add_row("Dihedral", self.gen_dihedral)
        g.add_row("Washout", self.gen_washout, tooltip="Tip twisted nose-down by this angle relative to the root")
        self.gen_airfoil = QComboBox()
        self.gen_airfoil.setObjectName("gen_airfoil")
        self.gen_airfoil.setEditable(True)
        g.add_row("Airfoil", self.gen_airfoil, tooltip="Section of every station; edit single sections in the table")
        self.generate_button = small_button("Replace sections", "Replace the sections with the generated "
                                                                "planform (one undo step)")
        self.generate_button.setObjectName("generate_button")
        self.generate_button.clicked.connect(self._generate)
        g.add_row("", hbox(self.generate_button, stretch=None))
        self.gen_status = StatusLine()
        g.add_widget(self.gen_status)

    def _build_wing(self) -> None:
        g = self.grid.add_group("Wing", key="wing")
        self.wing_group = g
        self.symmetric = QCheckBox("Mirror about y = 0")
        self.symmetric.setObjectName("wing_symmetric")
        self.symmetric.toggled.connect(lambda on: self._option("symmetric", bool(on), "Symmetric"))
        g.add_row("Symmetric", self.symmetric,
                  tooltip="The sections describe the right half; the left half is its mirror image")
        self.n_chord = QSpinBox()
        self.n_chord.setRange(1, 64)
        self.n_chord.setObjectName("n_chord")
        self.n_span = QSpinBox()
        self.n_span.setRange(2, 400)
        self.n_span.setObjectName("n_span")
        for w in (self.n_chord, self.n_span):
            w.setKeyboardTracking(False)
            w.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.n_chord.valueChanged.connect(lambda v: self._option("n_chord", int(v), "Chordwise panels"))
        self.n_span.valueChanged.connect(lambda v: self._option("n_span", int(v), "Spanwise panels"))
        g.add_row("Chordwise panels", self.n_chord, tooltip="Vortex-lattice panels along the chord")
        g.add_row("Spanwise panels", self.n_span,
                  tooltip="Vortex-lattice panels (lifting-line strips) per half wing; 24 is a good default")
        self.spacing = QComboBox()
        self.spacing.setObjectName("spacing")
        for k, t in SPACINGS:
            self.spacing.addItem(t, k)
        self.spacing.currentIndexChanged.connect(
            lambda _i: self._option("spacing", str(self.spacing.currentData()), "Spacing"))
        g.add_row("Spanwise spacing", self.spacing)

    def _build_metrics(self) -> None:
        g = self.grid.add_group("Planform", key="planform")
        self.metrics_group = g
        self.metrics = KeyValueList()
        self.metrics.setObjectName("wing_metrics")
        g.add_widget(self.metrics)

    def _build_refs(self) -> None:
        g = self.grid.add_group("References", key="references")
        self.refs_group = g
        lu = display_unit("length")
        self.ref_area = QDoubleSpinBox()
        self.ref_area.setRange(1e-8, 1e4)
        self.ref_area.setDecimals(4)
        self.ref_area.setSuffix(" m²")
        self.ref_area.setKeyboardTracking(False)
        self.ref_area.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.ref_area.setObjectName("wing_ref_area")
        self.ref_area.valueChanged.connect(lambda v: self._ref("ref_area", float(v)))
        self.length_ref = QuantitySpinBox("length", lu, minimum=1e-6, maximum=1e4, step=0.001)
        self.length_ref.setObjectName("wing_length_ref")
        self.length_ref.siValueChanged.connect(lambda v: self._ref("length_ref", float(v)))
        self.span_ref = QuantitySpinBox("length", lu, minimum=1e-6, maximum=1e4, step=0.001)
        self.span_ref.setObjectName("wing_span_ref")
        self.span_ref.siValueChanged.connect(lambda v: self._ref("span_ref", float(v)))
        self.mref = [QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.001) for _ in range(3)]
        for i, w in enumerate(self.mref):
            w.setObjectName(f"wing_mref_{'xyz'[i]}")
            w.siValueChanged.connect(lambda _v: self._ref("moment_ref3", tuple(b.si_value() for b in self.mref)))
        self.auto: dict[str, QToolButton] = {}
        for f, text, w, tip in (
                ("ref_area", "Reference area", self.ref_area, "Coefficients use q × S. Auto: the planform area."),
                ("length_ref", "Reference chord", self.length_ref,
                 "Reynolds number and pitching-moment arm. Auto: the mean aerodynamic chord."),
                ("span_ref", "Reference span", self.span_ref, "Rolling and yawing moment arm. Auto: the span."),
                ("moment_ref3", "Moment point x", self.mref[0], "Auto: the quarter-chord point of the MAC.")):
            b = _toggle("Auto", "Follow the planform; typing a value fixes it")
            b.setObjectName(f"wing_auto_{f}")
            b.clicked.connect(lambda on, f=f: self.autoRefToggled.emit(f, bool(on)))
            self.auto[f] = b
            g.add_row(text, hbox(w, b), tooltip=tip, key=f)
        g.add_row("Moment point y", hbox(self.mref[1], _spacer(ROW_BUTTON_W)))
        g.add_row("Moment point z", hbox(self.mref[2], _spacer(ROW_BUTTON_W)))

    def _build_diagnostics(self) -> None:
        g = self.grid.add_group("Diagnostics", key="diagnostics")
        self.diag_group = g
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 2, 0, 0)
        v.setSpacing(6)
        self.diag_summary = StatusLine()
        self.diag_summary.setObjectName("wing_diag_summary")
        v.addWidget(self.diag_summary)
        self._diag_items = QVBoxLayout()
        self._diag_items.setSpacing(6)
        v.addLayout(self._diag_items)
        g.add_widget(box)

    # -- loading --------------------------------------------------------------------
    def set_design(self, design: Design | None, *, airfoil_names: dict[str, str] | None = None) -> None:
        self._design = design
        self._airfoil_names = dict(airfoil_names or {})
        if design is None or design.wing is None:
            return
        spec = design.wing
        self._loading = True
        try:
            gv = generator_values(spec)
            self.shape.setCurrentIndex(max(self.shape.findData(gv["shape"]), 0))
            self.gen_span.set_si_value(gv["span"])
            self.gen_root.set_si_value(gv["root_chord"])
            self.gen_tip.set_si_value(gv["tip_chord"])
            self.gen_sweep.set_si_value(gv["sweep_c4_deg"])
            self.gen_dihedral.set_si_value(gv["dihedral_deg"])
            self.gen_washout.set_si_value(gv["washout_deg"])
            self.gen_airfoil.clear()
            for code in COMMON_AIRFOILS:
                self.gen_airfoil.addItem(f"NACA {code}")
            for n in self._airfoil_names.values():
                self.gen_airfoil.addItem(n)
            self.gen_airfoil.setCurrentText(airfoil_label(gv["airfoil"], self._airfoil_names))
            self._tip_row.set_label("Tip chord")
            self.symmetric.setChecked(bool(spec.symmetric))
            self.n_chord.setValue(int(spec.n_chord))
            self.n_span.setValue(int(spec.n_span))
            self.spacing.setCurrentIndex(max(self.spacing.findData(spec.spacing), 0))
            self.metrics.set_items(metrics_rows(spec))
            a = float(design.ref_area)
            dec = 3 if not a > 0 else int(min(8, max(3, 3 - math.floor(math.log10(a)))))
            self.ref_area.setDecimals(dec)
            self.ref_area.setValue(a)
            self.length_ref.set_si_value(float(design.length_ref))
            self.span_ref.set_si_value(float(design.span_ref))
            for w, v in zip(self.mref, design.moment_ref3, strict=True):
                w.set_si_value(float(v))
            manual = set(design.source.get("manual_refs", []))
            for f, b in self.auto.items():
                b.setChecked(f not in manual)
            self._load_diagnostics(design)
        finally:
            self._loading = False

    def _load_diagnostics(self, d: Design) -> None:
        clear_layout(self._diag_items)
        try:
            msgs = list(_g3().check_design3d(d))
        except Exception as exc:  # noqa: BLE001 - a checker failure is itself a diagnostic
            msgs = [f"The planform check failed: {exc}"]
        if msgs:
            self.diag_summary.set("warning", f"{len(msgs)} warning{'s' if len(msgs) != 1 else ''}")
            self.diag_group.set_expanded(True)
        else:
            self.diag_summary.set("ok", "No problems found")
        for m in msgs:
            line = StatusLine(m, "warning")
            line.setProperty("diag_level", "warning")
            self._diag_items.addWidget(line)

    def diagnostics(self) -> list[str]:
        out = []
        for i in range(self._diag_items.count()):
            w = self._diag_items.itemAt(i).widget()
            if isinstance(w, StatusLine):
                out.append(w.text())
        return out

    # -- edits -------------------------------------------------------------------------
    def _spec(self) -> WingSpec | None:
        d = self._design
        return None if d is None or d.wing is None else WingSpec.from_dict(copy.deepcopy(d.wing.to_dict()))

    def _option(self, field: str, value: Any, text: str) -> None:
        if self._loading:
            return
        spec = self._spec()
        if spec is None or getattr(spec, field) == value:
            return
        setattr(spec, field, value)
        if field == "symmetric" and value and spec.sections[0].y < 0:
            self.gen_status.set("error", "A symmetric wing is described by its right half (y ≥ 0).")
            self.set_design(self._design, airfoil_names=self._airfoil_names)
            return
        self.specEdited.emit(spec, text, f"opt:{field}")

    def _shape_changed(self, _i: int) -> None:
        if self._loading:
            return
        shape = str(self.shape.currentData())
        if shape == "swept" and abs(self.gen_sweep.si_value()) < 1e-9:
            self.gen_sweep.set_si_value(25.0)
        if shape == "rectangular":
            self.gen_tip.set_si_value(self.gen_root.si_value())
        if shape == "elliptic" and self.gen_tip.si_value() >= self.gen_root.si_value():
            self.gen_tip.set_si_value(0.15 * self.gen_root.si_value())
        self.gen_tip.setEnabled(shape != "rectangular")

    def _generate(self) -> None:
        spec = self._spec()
        if spec is None:
            return
        key = parse_airfoil(self.gen_airfoil.currentText(), self._airfoil_names)
        if key is None:
            self.gen_status.set("error", "The airfoil is not a NACA 4- or 5-digit code or a project airfoil.")
            return
        try:
            secs = generate_planform(str(self.shape.currentData()), span=self.gen_span.si_value(),
                                     root_chord=self.gen_root.si_value(), tip_chord=self.gen_tip.si_value(),
                                     sweep_c4_deg=self.gen_sweep.si_value(),
                                     dihedral_deg=self.gen_dihedral.si_value(),
                                     washout_deg=self.gen_washout.si_value(), airfoil=key,
                                     symmetric=spec.symmetric)
        except ValueError as exc:
            self.gen_status.set("error", str(exc).capitalize())
            return
        spec.sections = secs
        err = validate_spec(spec)
        if err:
            self.gen_status.set("error", err)
            return
        self.gen_status.set("info", "")
        self.specEdited.emit(spec, f"Generate {self.shape.currentText().lower()} planform", "")

    def _ref(self, field: str, value: Any) -> None:
        if not self._loading:
            self.refEdited.emit(field, value)


__all__ = [
    "COMMON_AIRFOILS",
    "SHAPES",
    "WING_REF_FIELDS",
    "PlanformSketch",
    "SectionTable",
    "WingEditor",
    "WingInspector",
    "airfoil_label",
    "airfoil_resolver",
    "auto_wing_references",
    "generate_planform",
    "generator_values",
    "insert_section",
    "metrics_rows",
    "panel_angles",
    "parse_airfoil",
    "project_airfoils",
    "refresh_wing_references",
    "remove_section",
    "set_panel_angle",
    "update_wing",
    "validate_spec",
]
