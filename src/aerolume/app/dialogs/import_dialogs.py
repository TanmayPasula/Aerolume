"""Dialogs that create a design: NACA generator, file importers, template browser.

Every dialog previews the exact :class:`~aerolume.model.Design` it will return,
with the same canvas the Geometry page uses, because the commonest import
mistake is scale (an SVG drawn in pixels, a DXF in millimetres) and the cure is
to *see* the dimensions before accepting. Each dialog therefore asks for the
real-world length and shows it on the preview.

Layout follows ``docs/ui_guidelines.md``: a narrow form column on the left
(label column in text2, 24 px editors), the preview in a bordered surface well
on the right, one status line with a small glyph, and OK / Cancel at the
bottom right. Explanations live in tooltips.

The dialogs never touch the project: the Geometry page calls :meth:`design` after
``exec()`` and adds the result through ``AppState``. All file parsing goes
through :mod:`aerolume.geometry` (DESIGN.md 5.1); it is imported lazily so the
application starts even while that package is unavailable, and every reader
error is shown in the dialog instead of escaping as an exception.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QRegularExpression, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QImage, QPainter, QPen, QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...model import KINDS, KINDS_2D, Design
from ..widgets.geometry_canvas import (
    GeometryCanvas,
    fmt_len,
    frontal_extent,
    qcolor,
    render_thumbnail,
    theme_tokens,
)
from ..widgets.param_form import QuantitySpinBox, StatusLine, keep_start_visible, label, set_line_text

# extension -> reader kind
_EXT = {
    ".dat": "dat", ".txt": "dat", ".cor": "dat",
    ".svg": "svg", ".dxf": "dxf", ".csv": "csv",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".bmp": "image",
    ".gif": "image", ".tif": "image", ".tiff": "image", ".webp": "image",
}
FORMAT_FILTERS = {
    "dat": "Airfoil coordinates (*.dat *.txt *.cor)",
    "svg": "SVG drawings (*.svg)",
    "dxf": "DXF drawings (*.dxf)",
    "image": "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)",
    "csv": "CSV point lists (*.csv)",
}
#: 3D meshes go to :mod:`.mesh_import`; listed here so one Import dialog offers every format
MESH_FILTER_ENTRY = "3D meshes (*.stl *.obj *.ply)"
FILE_FILTER = ";;".join(
    ["All geometry (" + " ".join([*(f"*{e}" for e in _EXT), "*.stl", "*.obj", "*.ply"]) + ")",
     *FORMAT_FILTERS.values(), MESH_FILTER_ENTRY, "All files (*)"])
FORMAT_NAMES = {"dat": "Airfoil .dat", "svg": "SVG", "dxf": "DXF", "image": "Image silhouette",
                "csv": "CSV points", "naca": "NACA", "template": "Template", "stl": "STL mesh",
                "obj": "Wavefront OBJ mesh", "ply": "PLY mesh"}
KIND_LABELS = {"airfoil": "Airfoil / wing section", "vehicle": "Vehicle (side profile)",
               "generic": "Generic body", "wing": "Finite wing (3D)"}

FORM_LABEL_WIDTH = 104


def classify_path(path: str | os.PathLike) -> str | None:
    """Reader kind for a file name ("dat", "svg", "dxf", "image", "csv") or None."""
    return _EXT.get(Path(path).suffix.lower())


def geometry_module():
    """The :mod:`aerolume.geometry` package, with a readable error if it is missing."""
    try:
        from ... import geometry
    except Exception as exc:
        raise RuntimeError(f"The geometry module is not available ({exc}).") from exc
    return geometry


def templates_module():
    try:
        from ...geometry import templates
    except Exception as exc:
        raise RuntimeError(f"Shape templates are not available ({exc}).") from exc
    return templates


# ---------------------------------------------------------------------------
# shared layout pieces
# ---------------------------------------------------------------------------

class _Form(QWidget):
    """A two-column form: text2 label column of fixed width, 24 px editors."""

    def __init__(self, label_width: int = FORM_LABEL_WIDTH) -> None:
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.grid.setColumnMinimumWidth(0, label_width)
        self.grid.setColumnStretch(1, 1)
        self._row = 0

    def add_row(self, text: str | QLabel, editor: QWidget, tooltip: str = "") -> QLabel:
        lab = text if isinstance(text, QLabel) else label(text, "pg-label")
        lab.setMinimumHeight(24)
        if tooltip:
            lab.setToolTip(tooltip)
            editor.setToolTip(tooltip)
        self.grid.addWidget(lab, self._row, 0)
        self.grid.addWidget(editor, self._row, 1)
        self._row += 1
        return lab

    def add_widget(self, w: QWidget) -> None:
        self.grid.addWidget(w, self._row, 0, 1, 2)
        self._row += 1


def _preview_canvas() -> GeometryCanvas:
    c = GeometryCanvas()
    c.setMinimumSize(480, 260)
    c.empty_text = "Nothing to preview"
    c.setAcceptDrops(False)
    return c


def _well(w: QWidget) -> QFrame:
    """Put a viewport in a 1 px bordered surface frame (``surface="true"``)."""
    f = QFrame()
    f.setProperty("surface", "true")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(1, 1, 1, 1)
    lay.addWidget(w)
    return f


def _button_box(ok_text: str) -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    ok = box.button(QDialogButtonBox.StandardButton.Ok)
    ok.setText(ok_text)
    ok.setProperty("variant", "primary")
    ok.setDefault(True)
    return box


def _footer(status: QWidget, buttons: QDialogButtonBox) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 4, 0, 0)
    h.setSpacing(12)
    h.addWidget(status, 1)
    h.addWidget(buttons, 0, Qt.AlignmentFlag.AlignBottom)
    return w


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


def _dialog_layout(dlg: QDialog) -> QVBoxLayout:
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(8)
    return lay


# ---------------------------------------------------------------------------
# NACA
# ---------------------------------------------------------------------------

def validate_naca(code: str) -> str | None:
    """Plain-language problem with a NACA designation, or None if it is usable.

    Checks the families the geometry module generates: 4-digit "MPTT" and
    5-digit "LPSTT". The thresholds are the physically sensible ones (thickness
    1-40 %, camber position required whenever there is camber), so users get a
    precise hint instead of a generic "invalid".
    """
    c = code.strip().upper().removeprefix("NACA").strip()
    if not c:
        return "Enter a 4- or 5-digit NACA code, e.g. 2412 or 23012."
    if not c.isdigit():
        return "NACA codes contain digits only."
    if len(c) == 4:
        m, p, t = int(c[0]), int(c[1]), int(c[2:])
        if t < 1 or t > 40:
            return "Thickness (last two digits) must be 01-40 % of the chord."
        if m > 0 and p == 0:
            return "A cambered section needs a camber position (2nd digit 1-9)."
        if m == 0 and p != 0:
            return "Camber position given without camber: use 00xx for a symmetric section."
        return None
    if len(c) == 5:
        cl, p, s, t = int(c[0]), int(c[1]), int(c[2]), int(c[3:])
        if cl == 0:
            return "First digit (design lift) must be 1-9."
        if p < 1 or p > 5:
            return "Second digit (camber position) must be 1-5."
        if s not in (0, 1):
            return "Third digit must be 0 (normal) or 1 (reflex camber)."
        if t < 1 or t > 40:
            return "Thickness (last two digits) must be 01-40 % of the chord."
        return None
    return "NACA codes have 4 or 5 digits."


def describe_naca(code: str) -> str:
    """Factual one-line reading of a valid NACA code ("2 % camber at 40 % chord, 12 % thick")."""
    c = code.strip().upper().removeprefix("NACA").strip()
    if validate_naca(c):
        return ""
    if len(c) == 4:
        m, p, t = int(c[0]), int(c[1]), int(c[2:])
        if m == 0:
            return f"Symmetric, {t} % thick"
        return f"{m} % camber at {10 * p} % chord, {t} % thick"
    cl, p, s, t = int(c[0]), int(c[1]), int(c[2]), int(c[3:])
    reflex = ", reflex camber line" if s == 1 else ""
    return f"Design Cl {0.15 * cl:.2f}, camber peak at {5 * p} % chord, {t} % thick{reflex}"


NACA_HELP = ("4-digit MPTT: M = maximum camber (% chord), P = its position (tenths of chord), "
             "TT = thickness (% chord).\n5-digit LPSTT: L × 0.15 = design lift coefficient, "
             "P = camber position (1/20 chord), S = 1 for a reflex camber line, TT = thickness.")


def naca_design(code: str, chord: float = 1.0, n: int = 160, name: str | None = None) -> Design:
    """Build an airfoil design from a NACA code at a real chord length [m].

    The section is generated at unit chord and scaled here, so the design does
    not depend on whether the geometry module's ``design_from_airfoil`` expects
    unit or dimensional coordinates.
    """
    geo = geometry_module()
    code = code.strip().upper().removeprefix("NACA").strip()
    coords = np.asarray(geo.naca(code, n=int(n), chord=1.0), dtype=float)
    d = _airfoil_design(geo, coords, name or f"NACA {code}", chord)
    d.source = {**(d.source or {}), "type": "naca", "code": code, "n": int(n), "chord": float(chord)}
    return d


def _airfoil_design(geo, unit_coords: np.ndarray, name: str, chord: float) -> Design:
    """design_from_airfoil with a guaranteed real chord, whatever its scaling rule."""
    d = geo.design_from_airfoil(unit_coords, name, chord)
    x0, x1, _, _ = d.bounds()
    span = x1 - x0
    if span > 0 and abs(span - chord) > 1e-6 * max(chord, 1e-9):
        # it kept the unit chord: scale everything (refs included) ourselves
        k = chord / span
        for b in d.bodies:
            b.points = b.points * k
            if b.center is not None:
                b.center = (b.center[0] * k, b.center[1] * k)
        d.length_ref, d.force_ref = d.length_ref * k, d.force_ref * k
        d.moment_ref = (d.moment_ref[0] * k, d.moment_ref[1] * k)
    return d


class NacaDialog(QDialog):
    """Generate a NACA 4- or 5-digit airfoil with live validation and preview."""

    QUICK = ("0009", "0012", "0015", "2412", "2415", "4412", "4415", "6409", "23012", "23015")

    def __init__(self, parent: QWidget | None = None, *, code: str = "2412", chord: float = 1.0,
                 length_unit: str = "m") -> None:
        super().__init__(parent)
        self.setWindowTitle("New NACA airfoil")
        self.resize(820, 440)
        self._design: Design | None = None
        lay = _dialog_layout(self)
        body = QHBoxLayout()
        body.setSpacing(12)
        lay.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(8)
        form = _Form()
        self.code_edit = QLineEdit(code)
        self.code_edit.setObjectName("naca_code")
        self.code_edit.setMaxLength(9)
        self.code_edit.setPlaceholderText("2412")
        self.code_edit.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"^\s*(?:[Nn][Aa][Cc][Aa])?\s*\d{0,5}\s*$")))
        self.chord = QuantitySpinBox("length", length_unit, minimum=1e-4, maximum=1e3, step=0.01)
        self.chord.set_si_value(chord)
        self.points = QSpinBox()
        self.points.setRange(60, 600)
        self.points.setSingleStep(20)
        self.points.setValue(160)
        self.points.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.add_row("NACA code", self.code_edit, NACA_HELP)
        form.add_row("Chord", self.chord, "Chord of the real section; sets the Reynolds number.")
        form.add_row("Outline points", self.points, "Points along the outline; 160 resolves the leading "
                                                    "edge well.")
        left.addWidget(form)
        left.addSpacing(4)
        left.addWidget(label("Common sections", "secondary"))
        self.quick_list = QListWidget()
        self.quick_list.setObjectName("naca_quick")
        for q in self.QUICK:
            it = QListWidgetItem(f"NACA {q}")
            it.setData(Qt.ItemDataRole.UserRole, q)
            it.setToolTip(describe_naca(q))
            self.quick_list.addItem(it)
        self.quick_list.itemClicked.connect(lambda it: self.code_edit.setText(it.data(Qt.ItemDataRole.UserRole)))
        left.addWidget(self.quick_list, 1)
        lw = _wrap(left)
        lw.setFixedWidth(260)
        body.addWidget(lw)
        self.preview = _preview_canvas()
        body.addWidget(_well(self.preview), 1)

        self.status = StatusLine()
        self.buttons = _button_box("Create")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(_footer(self.status, self.buttons))
        self.code_edit.textChanged.connect(self._refresh)
        self.chord.siValueChanged.connect(self._refresh)
        self.points.valueChanged.connect(self._refresh)
        self._refresh()

    def code(self) -> str:
        return self.code_edit.text().strip().upper().removeprefix("NACA").strip()

    def _refresh(self, *_: Any) -> None:
        err = validate_naca(self.code())
        self._design = None
        if err is None:
            try:
                self._design = naca_design(self.code(), self.chord.si_value(), self.points.value())
            except Exception as exc:  # noqa: BLE001 - shown to the user
                err = str(exc) or exc.__class__.__name__
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._design is not None)
        self.code_edit.setProperty("invalid", "true" if err else "false")
        self.code_edit.style().unpolish(self.code_edit)
        self.code_edit.style().polish(self.code_edit)
        if err:
            self.status.set("error", err)
        else:
            self.status.set("info", f"NACA {self.code()}: {describe_naca(self.code())}")
        for i in range(self.quick_list.count()):
            it = self.quick_list.item(i)
            it.setSelected(it.data(Qt.ItemDataRole.UserRole) == self.code())
        self.preview.set_design(self._design, fit=True)

    def design(self) -> Design | None:
        """The generated design (None while the code is invalid)."""
        return self._design


# ---------------------------------------------------------------------------
# file import (dat / svg / dxf / csv; images subclass below)
# ---------------------------------------------------------------------------

def _guess_kind(polys: list[np.ndarray], fmt: str) -> str:
    """Best-guess design kind from outline proportions.

    Airfoil sections are slender (length/height > ~5); road-vehicle side
    profiles sit around 2-5. The user can always change it - this just makes
    the common case one click.
    """
    if fmt == "dat":
        return "airfoil"
    if not polys:
        return "generic"
    pts = np.vstack(polys)
    w = float(np.ptp(pts[:, 0]))
    h = float(np.ptp(pts[:, 1])) or 1e-9
    ar = w / h
    if ar >= 5.0:
        return "airfoil"
    if ar >= 1.6:
        return "vehicle"
    return "generic"


_LENGTH_TIPS = {
    "airfoil": "Chord of the real section. Sets the Reynolds number.",
    "vehicle": "Overall length, bumper to bumper. The drawing is scaled to it.",
    "generic": "Overall length (x extent). The drawing is scaled to it.",
}
CLEARANCE_TIP = ("Gap between the road and the lowest point of the body, not counting the wheels. "
                 "Wheels stand on the road (plus the wheel gap) whatever the clearance, as a car's "
                 "suspension would put them. Follows the drawing until you type a value.")
WHEEL_GAP_TIP = ("Lift the wheels off the road. In 2D a wheel is an infinitely wide cylinder, so a wheel "
                 "touching the road seals the underbody; a small gap keeps a channel open.")


# ---------------------------------------------------------------------------
# outline classification and road placement (pure numpy; unit-tested)
# ---------------------------------------------------------------------------

@dataclass
class OutlineInfo:
    """What one outline of an imported file appears to be, for the outline list.

    ``role`` is "main" (largest area), "wheel", "part" (another outline that
    overlaps the main body's extent), "inner" (lies inside a larger outline:
    a window or hole, which the design builder drops anyway) or "separate"
    (its bounding box does not touch the main body's: a stray mark, a title
    block, a dimension). ``include`` is the default check state; ``enabled``
    is False for inner outlines, which can never affect the external flow.
    """
    index: int
    role: str
    text: str
    tooltip: str
    include: bool
    enabled: bool = True


def _bbox(p: np.ndarray) -> tuple[float, float, float, float]:
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())


def _boxes_touch(a: tuple, b: tuple) -> bool:
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]


def classify_outlines(polys: list[np.ndarray]) -> list[OutlineInfo]:
    """Classify the outlines of a drawing so the user can pick what to import.

    Why: drawings carry more than the shape. A DXF exported from CAD often
    holds a title-block mark or a construction circle; an image trace keeps
    specks. Before this list existed every outline was imported, and a stray
    circle above the roof both counted as part of the car (its height set the
    dimensions) and could not be removed. Outlines that cannot touch the main
    body are therefore unchecked by default, but stay one click away, because
    multi-element wings legitimately have separate elements (the dialog
    re-includes them when the drawing is guessed to be an airfoil).
    """
    if not polys:
        return []
    areas = [abs(_signed_area(p)) for p in polys]
    main = int(np.argmax(areas))
    try:
        from ...geometry.design import find_wheels
        wheels = find_wheels(polys)
    except Exception:  # noqa: BLE001 - classification is a convenience, never a blocker
        wheels = [None] * len(polys)
    try:
        from ...geometry.polygon import point_in_polygon
    except Exception:  # noqa: BLE001
        point_in_polygon = None
    main_box = _bbox(polys[main])
    out: list[OutlineInfo] = []
    n_wheel = 0
    for i, p in enumerate(polys):
        w, h = float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))
        size = f"{len(p):,} points, {w:.4g} × {h:.4g} in file units"
        if i == main:
            out.append(OutlineInfo(i, "main", f"Outline {i + 1} · main body", f"The largest outline. {size}.",
                                   True))
            continue
        container = None
        if point_in_polygon is not None:
            for j in np.argsort(areas)[::-1]:
                if areas[j] > areas[i] and bool(np.all(point_in_polygon(polys[j], p))):
                    container = int(j)
                    break
        if container is not None:
            out.append(OutlineInfo(i, "inner", f"Outline {i + 1} · inside outline {container + 1}",
                                   f"Lies inside outline {container + 1} (a window, hole or detail); it does not "
                                   f"affect the external flow and is ignored. {size}.", False, enabled=False))
        elif wheels[i] is not None:
            n_wheel += 1
            out.append(OutlineInfo(i, "wheel", f"Outline {i + 1} · wheel",
                                   f"Round outline in the lower half: imported as a wheel. {size}.", True))
        elif not _boxes_touch(_bbox(p), main_box):
            out.append(OutlineInfo(i, "separate", f"Outline {i + 1} · separate",
                                   "Does not reach the main body's extent: often a stray mark, a title block or "
                                   f"a dimension, so it is left out. Check it to include it. {size}.", False))
        else:
            out.append(OutlineInfo(i, "part", f"Outline {i + 1}", f"{size}.", True))
    return out


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def drawing_clearance(polys: list[np.ndarray], scale: float) -> float | None:
    """Body-to-road gap the drawing itself shows, in metres, or None without wheels.

    A side view with wheels has its road where the wheels stand: the gap
    between the lowest wheel point and the lowest point of the body, scaled.
    It is the natural default for the clearance field, because then the
    preview reproduces the drawing.
    """
    try:
        from ...geometry.design import find_wheels
        wheels = find_wheels(polys)
    except Exception:  # noqa: BLE001
        return None
    w_lo = [float(p[:, 1].min()) for p, wh in zip(polys, wheels) if wh is not None]
    b_lo = [float(p[:, 1].min()) for p, wh in zip(polys, wheels) if wh is None]
    if not w_lo or not b_lo:
        return None
    return max((min(b_lo) - min(w_lo)) * scale, 0.0)


def place_on_road(design: Design, clearance: float, wheel_gap: float = 0.0) -> None:
    """Put a vehicle on the road with the clearance measured to the *body*.

    ``design_from_polygons`` puts the lowest point of all outlines - wheels
    included - at the clearance, which left the wheels floating at the
    clearance height and the body above them, contradicting what the field
    says. Here the non-wheel bodies move together so their lowest point sits at
    ``clearance``, and each wheel moves vertically so its lowest point sits at
    ``wheel_gap`` (the template convention: wheels on the road unless lifted).
    The frontal height is recomputed because wheels and body move by different
    amounts. Works in place; a design without wheels just gets its body placed.
    """
    body = [b for b in design.bodies if b.role != "wheel"]
    if not body:
        return
    dy = float(clearance) - min(float(np.min(b.points[:, 1])) for b in body)
    for b in body:
        b.points = b.points + np.array([0.0, dy])
        if b.center is not None:
            b.center = (b.center[0], b.center[1] + dy)
    for b in design.bodies:
        if b.role != "wheel":
            continue
        dw = float(wheel_gap) - float(np.min(b.points[:, 1]))
        b.points = b.points + np.array([0.0, dw])
        if b.center is not None:
            b.center = (b.center[0], b.center[1] + dw)
    design.force_ref = frontal_extent([b.points for b in design.bodies])


def vehicle_height(design: Design) -> float:
    """Overall height as the canvas dimension shows it: from the road for a
    vehicle standing on it, else the bounding-box height."""
    _, _, y0, y1 = design.bounds()
    y_lo = 0.0 if design.kind == "vehicle" and y0 >= -1e-9 else y0
    return y1 - y_lo


class _ImportPreview(GeometryCanvas):
    """The import preview plus the outlines left out, dashed, so the user can
    see what an unchecked entry in the outline list refers to."""

    def __init__(self) -> None:
        super().__init__()
        self.ghosts: list[np.ndarray] = []

    def set_ghosts(self, ghosts: list[np.ndarray]) -> None:
        self.ghosts = list(ghosts)
        self.update()

    def _content_bounds(self):
        b = super()._content_bounds()
        if b is None or not self.ghosts:
            return b
        pts = np.vstack(self.ghosts)
        return (min(b[0], float(pts[:, 0].min())), max(b[1], float(pts[:, 0].max())),
                min(b[2], float(pts[:, 1].min())), max(b[3], float(pts[:, 1].max())))

    def _paint_overlay(self, p: QPainter, tok: dict) -> None:
        if not self.ghosts or self._design is None:
            return
        pen = QPen(qcolor(tok["muted"]), 1.0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for g in self.ghosts:
            p.drawPolygon(self._poly(g))


class PolygonImportDialog(QDialog):
    """Import outlines from a file and size them to the real world.

    Shows what was read (outlines, points, drawing size), lets the user choose
    which outlines to import, asks for the design kind, the real overall
    length and - for vehicles - the ground clearance of the body and the wheel
    gap, and previews the resulting design with its dimensions. The footer
    reports the same length and height as the preview's dimension lines.
    """

    LEFT_WIDTH = 300

    def __init__(self, path: str | os.PathLike, fmt: str | None = None, parent: QWidget | None = None,
                 *, length_unit: str = "m") -> None:
        super().__init__(parent)
        self.path = str(path)
        self.fmt = fmt or classify_path(path) or "csv"
        self.setWindowTitle(f"Import {FORMAT_NAMES.get(self.fmt, self.fmt)}: {Path(self.path).name}")
        self.resize(900, 520)
        self._polys: list[np.ndarray] = []
        self._outlines: list[OutlineInfo] = []
        self._file_name: str | None = None      # airfoil name from a .dat header
        self._design: Design | None = None
        self._error: str | None = None
        self._length_touched = False
        self._clearance_touched = False
        self._has_wheels = False

        lay = _dialog_layout(self)
        self.info = label("", "secondary")
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.info)
        self._body = QHBoxLayout()
        self._body.setSpacing(12)
        lay.addLayout(self._body, 1)
        self._extra = QVBoxLayout()          # subclasses put source-specific controls here
        self._extra.setSpacing(6)
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addLayout(self._extra)
        form = _Form()
        self.name_edit = keep_start_visible(QLineEdit(Path(self.path).stem))   # a long file name: its start
        self.kind_combo = QComboBox()
        for k in KINDS_2D:
            self.kind_combo.addItem(KIND_LABELS[k], k)
        self.length = QuantitySpinBox("length", length_unit, minimum=1e-4, maximum=1e4, step=0.01)
        self.length.set_si_value(1.0)
        self.clearance = QuantitySpinBox("length", length_unit, minimum=0.0, maximum=100.0, step=0.005)
        self.clearance.setObjectName("import_clearance")
        self.clearance.set_si_value(0.15)
        self.wheel_gap = QuantitySpinBox("length", length_unit, minimum=0.0, maximum=1.0, step=0.002)
        self.wheel_gap.setObjectName("import_wheel_gap")
        self.wheel_gap.set_si_value(0.0)
        form.add_row("Name", self.name_edit)
        form.add_row("Kind", self.kind_combo, "What the outline is; sets the reference conventions.")
        self._length_label = form.add_row("Length", self.length)
        self._clear_label = form.add_row("Ground clearance", self.clearance, CLEARANCE_TIP)
        self._gap_label = form.add_row("Wheel gap to road", self.wheel_gap, WHEEL_GAP_TIP)
        left.addWidget(form)
        # which outlines to import; hidden for single-outline files
        self.outline_head = label("Outlines", "secondary")
        self.outline_head.setToolTip("Unchecked outlines are not imported; they are drawn dashed in the preview.")
        self.outline_list = QListWidget()
        self.outline_list.setObjectName("outline_list")
        self.outline_list.setMinimumHeight(4 * 22 + 4)
        left.addWidget(self.outline_head)
        left.addWidget(self.outline_list, 1)
        self._left_stretch = QWidget()          # takes the room when the list is hidden
        left.addWidget(self._left_stretch, 1)
        lw = _wrap(left)
        lw.setFixedWidth(self.LEFT_WIDTH)
        self._body.addWidget(lw)
        self.preview = _ImportPreview()
        self.preview.setMinimumSize(480, 260)
        self.preview.empty_text = "Nothing to preview"
        self.preview.setAcceptDrops(False)
        self._body.addWidget(_well(self.preview), 1)
        self.status = StatusLine()
        self.buttons = _button_box("Import")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(_footer(self.status, self.buttons))

        self.kind_combo.currentIndexChanged.connect(self._kind_changed)
        self.length.siValueChanged.connect(self._length_edited)
        self.clearance.siValueChanged.connect(self._clearance_edited)
        self.wheel_gap.siValueChanged.connect(lambda _v: self._rebuild())
        self.name_edit.textChanged.connect(lambda _t: self._rebuild())
        self.outline_list.itemChanged.connect(lambda _it: self._rebuild())
        self._build_extra()
        self._reload()

    # -- hooks ------------------------------------------------------------
    def _build_extra(self) -> None:
        """Subclasses add their controls to ``self._extra`` here."""

    def _read(self) -> list[np.ndarray]:
        geo = geometry_module()
        if self.fmt == "dat":
            name, coords = geo.read_airfoil_dat(self.path)
            self._file_name = (name or "").strip() or None
            return [np.asarray(coords, dtype=float)]
        reader = {"svg": geo.read_svg, "dxf": geo.read_dxf, "csv": geo.read_points_csv}.get(self.fmt)
        if reader is None:
            raise RuntimeError(f"No reader for {self.fmt!r} files.")
        return [np.asarray(p, dtype=float) for p in reader(self.path)]

    def source_info(self) -> dict:
        from .mesh_import import file_provenance
        return file_provenance(self.path, self.fmt)

    # -- logic ------------------------------------------------------------
    def _reload(self) -> None:
        """(Re)read the file, classify its outlines, then guess kind and length once."""
        first = not self._polys
        try:
            self._polys = [p for p in self._read() if len(p) >= 3]
            self._error = None if self._polys else "No closed outline with 3 or more points in the file."
        except Exception as exc:  # noqa: BLE001 - reader errors become a message
            self._polys = []
            self._error = f"Could not read the file: {exc}"
        self._fill_outline_list()
        if self._polys:
            pts = np.vstack(self._polys)
            w, h = float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))
            npts = sum(len(p) for p in self._polys)
            n = len(self._polys)
            self.info.setText(
                f"{Path(self.path).name}  ·  {FORMAT_NAMES.get(self.fmt, self.fmt)}  ·  {n} outline"
                f"{'s' if n != 1 else ''}  ·  {npts:,} points  ·  drawing {w:.4g} × {h:.4g} (file units)")
            if first:
                if self._file_name and self.fmt == "dat":
                    set_line_text(self.name_edit, self._file_name)
                sel = self.selected_polygons() or self._polys
                kind = _guess_kind(sel, self.fmt)
                if kind == "airfoil":
                    # multi-element wings have separate elements: keep them in
                    self._set_checks(lambda o: o.enabled and o.role != "inner")
                    sel = self.selected_polygons()
                self.kind_combo.blockSignals(True)
                self.kind_combo.setCurrentIndex(max(self.kind_combo.findData(kind), 0))
                self.kind_combo.blockSignals(False)
                if not self._length_touched:
                    self.length.set_si_value(self._default_length(kind, float(np.ptp(np.vstack(sel)[:, 0]))))
        else:
            self.info.setText(f"{Path(self.path).name}  ·  {FORMAT_NAMES.get(self.fmt, self.fmt)}")
        self._kind_changed(rebuild=False)
        self._rebuild()

    def _fill_outline_list(self) -> None:
        self._outlines = classify_outlines(self._polys) if len(self._polys) > 1 else []
        self.outline_list.blockSignals(True)
        self.outline_list.clear()
        for o in self._outlines:
            it = QListWidgetItem(o.text)
            it.setData(Qt.ItemDataRole.UserRole, o.index)
            it.setToolTip(o.tooltip)
            flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
            if o.enabled:
                flags |= Qt.ItemFlag.ItemIsEnabled
            it.setFlags(flags)
            it.setCheckState(Qt.CheckState.Checked if o.include else Qt.CheckState.Unchecked)
            self.outline_list.addItem(it)
        self.outline_list.blockSignals(False)
        many = bool(self._outlines)
        self.outline_head.setVisible(many)
        self.outline_list.setVisible(many)
        self._left_stretch.setVisible(not many)

    def _set_checks(self, rule) -> None:
        self.outline_list.blockSignals(True)
        for o, i in zip(self._outlines, range(self.outline_list.count())):
            self.outline_list.item(i).setCheckState(Qt.CheckState.Checked if rule(o) else Qt.CheckState.Unchecked)
        self.outline_list.blockSignals(False)

    def outline_infos(self) -> list[OutlineInfo]:
        return list(self._outlines)

    def set_outline_included(self, index: int, on: bool) -> None:
        """Check or uncheck the list entry of outline ``index`` (as a click would)."""
        for i in range(self.outline_list.count()):
            it = self.outline_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == index:
                it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def included_indices(self) -> list[int]:
        if not self._outlines:
            return list(range(len(self._polys)))
        out = []
        for i in range(self.outline_list.count()):
            it = self.outline_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(int(it.data(Qt.ItemDataRole.UserRole)))
        return out

    def selected_polygons(self) -> list[np.ndarray]:
        """The outlines that will be imported (all of them for a single-outline file)."""
        return [self._polys[i] for i in self.included_indices() if i < len(self._polys)]

    @staticmethod
    def _default_length(kind: str, file_width: float) -> float:
        """A plausible real size: airfoils 1 m, vehicles 4.5 m; but trust files
        that already look metric (an airfoil .dat is unit chord by definition)."""
        if kind == "vehicle":
            return file_width if 1.0 <= file_width <= 30.0 else 4.5
        if kind == "airfoil":
            return file_width if 0.01 <= file_width <= 20.0 else 1.0
        return file_width if 0.01 <= file_width <= 50.0 else 1.0

    def _length_edited(self, _v: float) -> None:
        self._length_touched = True
        self._rebuild()

    def _clearance_edited(self, _v: float) -> None:
        self._clearance_touched = True
        self._rebuild()

    def kind(self) -> str:
        return str(self.kind_combo.currentData())

    def _kind_changed(self, *_: Any, rebuild: bool = True) -> None:
        k = self.kind()
        self._length_label.setText("Chord" if k == "airfoil" else "Length")
        self._length_label.setToolTip(_LENGTH_TIPS[k])
        self.length.setToolTip(_LENGTH_TIPS[k])
        self._update_vehicle_rows()
        if rebuild:
            self._rebuild()

    def _update_vehicle_rows(self) -> None:
        vehicle = self.kind() == "vehicle"
        for w in (self.clearance, self._clear_label):
            w.setVisible(vehicle)
        for w in (self.wheel_gap, self._gap_label):
            w.setVisible(vehicle and self._has_wheels)

    def _rebuild(self) -> None:
        self._design = None
        err = self._error
        sel = self.selected_polygons()
        if not err and not sel:
            err = "No outline is checked. Check at least one outline to import."
        if not err:
            self._sync_clearance(sel)
            try:
                self._design = self._make_design(sel)
            except Exception as exc:  # noqa: BLE001
                err = f"Could not build a design: {exc}"
        self._update_vehicle_rows()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._design is not None)
        if err:
            self.status.set("error", err)
        else:
            d = self._design
            x0, x1, _, _ = d.bounds()
            # the same numbers, rounded the same way, as the preview's dimension lines
            size = f"{fmt_len(x1 - x0)} × {fmt_len(vehicle_height(d))} m"
            what = "length × height above the road" if d.kind == "vehicle" else "after scaling"
            n_in, n_all = len(sel), len(self._polys)
            left_out = f" · {n_all - n_in} of {n_all} outlines left out" if n_in < n_all else ""
            self.status.set("info", f"{size} ({what}){left_out}")
        self.preview.set_design(self._design, fit=True)
        self.preview.set_ghosts(self._ghosts(sel))

    def _sync_clearance(self, sel: list[np.ndarray]) -> None:
        """Let the clearance follow the drawing until the user types one."""
        width = float(np.ptp(np.vstack(sel)[:, 0])) or 1.0
        drawn = drawing_clearance(sel, self.length.si_value() / width)
        self._has_wheels = drawn is not None
        if drawn is not None and not self._clearance_touched:
            self.clearance.set_si_value(drawn)

    def _make_design(self, sel: list[np.ndarray] | None = None) -> Design:
        geo = geometry_module()
        sel = self.selected_polygons() if sel is None else sel
        name = self.name_edit.text().strip() or Path(self.path).stem
        kind = self.kind()
        length = self.length.si_value()
        if self.fmt == "dat" and kind == "airfoil" and len(sel) == 1:
            coords = sel[0]
            span = float(np.ptp(coords[:, 0])) or 1.0
            d = _airfoil_design(geo, coords / span, name, length)
        else:
            vehicle = kind == "vehicle"
            d = geo.design_from_polygons(
                [p.copy() for p in sel], name, kind, length=length,
                ground_clearance=self.clearance.si_value() if vehicle else None)
            if vehicle:
                place_on_road(d, self.clearance.si_value(), self.wheel_gap.si_value())
            # the builder numbers bodies by cleaned outline ("car 3"); a lone body
            # beside its wheels is simply the design
            solid = [b for b in d.bodies if b.role != "wheel"]
            if len(solid) == 1:
                solid[0].name = name
        d.name = name
        extra: dict[str, Any] = {}
        if kind == "vehicle":
            extra["ground_clearance"] = self.clearance.si_value()
            if any(b.role == "wheel" for b in d.bodies):
                extra["wheel_gap"] = self.wheel_gap.si_value()
        left_out = sorted(set(range(len(self._polys))) - set(self.included_indices()))
        if self._outlines and left_out:
            extra["excluded_outlines"] = [i + 1 for i in left_out]          # 1-based, as listed
        d.source = {**(d.source or {}), **self.source_info(), **extra}
        return d

    def _ghosts(self, sel: list[np.ndarray]) -> list[np.ndarray]:
        """Unchecked outlines mapped into design coordinates for the preview.

        The design builder scales uniformly and translates, so the map follows
        from the largest selected outline and its placed counterpart (the
        largest non-wheel body). An airfoil built from one outline is also
        de-rotated; there is nothing sensible to overlay then.
        """
        d = self._design
        excluded = sorted(set(range(len(self._polys))) - set(self.included_indices()))
        if d is None or not excluded or not sel:
            return []
        if d.kind == "airfoil" and len(d.bodies) == 1:
            return []
        bodies = [b for b in d.bodies if b.role != "wheel"] or d.bodies
        placed = max(bodies, key=lambda b: abs(_signed_area(b.points))).points
        raw = max(sel, key=lambda p: abs(_signed_area(p)))
        width = float(np.ptp(np.vstack(sel)[:, 0])) or 1.0
        s = self.length.si_value() / width
        t = placed.min(axis=0) - s * raw.min(axis=0)
        return [self._polys[i] * s + t for i in excluded]

    def design(self) -> Design | None:
        return self._design

    def polygons(self) -> list[np.ndarray]:
        """Every outline read from the file (see :meth:`selected_polygons`)."""
        return list(self._polys)


class ImageSilhouetteDialog(PolygonImportDialog):
    """Trace a silhouette from a photo or drawing with a live threshold preview.

    The original image is shown beside the traced outline, so the effect of
    the threshold and *invert* (dark object on light background vs. the
    reverse) is visible immediately. Tracing runs 150 ms after the last change
    to keep the slider fluid on large images.
    """

    def _build_extra(self) -> None:
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedHeight(150)
        img = QImage(self.path)
        if not img.isNull():
            self.image_label.setPixmap(QPixmap.fromImage(img).scaled(
                QSize(self.LEFT_WIDTH - 6, 144), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:
            self.image_label.setText("The image could not be loaded.")
            self.image_label.setProperty("role", "muted")
        self._extra.addWidget(_well(self.image_label))
        form = _Form()
        self.auto_threshold = QCheckBox("Automatic")
        self.auto_threshold.setChecked(True)
        self.auto_threshold.setToolTip("Otsu's method picks the grey level that best separates object "
                                       "and background.")
        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(1, 254)
        self.threshold.setValue(128)
        self.threshold.setEnabled(False)
        self.threshold_value = label("auto", "mono")
        self.threshold_value.setMinimumWidth(30)
        self.threshold_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        trow = QHBoxLayout()
        trow.setSpacing(6)
        trow.addWidget(self.threshold, 1)
        trow.addWidget(self.threshold_value)
        self.invert = QCheckBox("Object lighter than background")
        self.invert.setToolTip("Trace light shapes on a dark background instead.")
        form.add_row("Threshold", self.auto_threshold)
        form.add_row("", _wrap(trow))
        form.add_row("Invert", self.invert)
        self._extra.addWidget(form)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._reload)
        self.auto_threshold.toggled.connect(self._threshold_mode)
        self.threshold.valueChanged.connect(self._threshold_moved)
        self.invert.toggled.connect(lambda _on: self._timer.start())

    def _threshold_mode(self, auto: bool) -> None:
        self.threshold.setEnabled(not auto)
        self.threshold_value.setText("auto" if auto else str(self.threshold.value()))
        self._timer.start()

    def _threshold_moved(self, v: int) -> None:
        self.threshold_value.setText(str(v))
        self._timer.start()

    def threshold_value_arg(self) -> float | None:
        """Threshold passed to the reader: None (automatic, Otsu) or a 0..255 grey level."""
        return None if self.auto_threshold.isChecked() else float(self.threshold.value())

    def _read(self) -> list[np.ndarray]:
        geo = geometry_module()
        polys = geo.read_image_silhouette(self.path, threshold=self.threshold_value_arg(),
                                          invert=self.invert.isChecked())
        return [np.asarray(p, dtype=float) for p in polys]

    def source_info(self) -> dict:
        return {**super().source_info(), "threshold": self.threshold_value_arg(),
                "invert": self.invert.isChecked()}


# ---------------------------------------------------------------------------
# template browser
# ---------------------------------------------------------------------------

_THUMB_CACHE: dict[tuple[str, str], QPixmap] = {}


def template_label(info: Any) -> str:
    """Display name of a TemplateInfo: its ``label`` if any, else a tidied ``name``."""
    lab = getattr(info, "label", None)
    if lab:
        return str(lab)
    name = str(getattr(info, "name", info))
    return name.replace("_", " ").strip().capitalize() if name.islower() else name


def template_thumbnail(name: str, w: int = 56, h: int = 28) -> QPixmap:
    """Thumbnail of a template at its default parameters (cached per theme and size)."""
    tok = theme_tokens()
    key = (name, f"{tok['text']}{tok.get('surface', tok['plot_bg'])}{w}x{h}")
    pm = _THUMB_CACHE.get(key)
    if pm is None:
        try:
            design = templates_module().make(name)
        except Exception:  # noqa: BLE001 - a broken template still gets an entry
            design = None
        pm = render_thumbnail(design, w, h, tokens=tok, margin=0.06)
        _THUMB_CACHE[key] = pm
    return pm


_GROUP_TITLES = {"airfoil": "Airfoils and wing sections", "vehicle": "Vehicles", "generic": "Basic shapes"}
NAME_ROLE = int(Qt.ItemDataRole.UserRole)


def _param_text(spec: Any) -> str:
    """Default value of a template parameter, formatted for the details list."""
    v = getattr(spec, "default", "")
    kind = str(getattr(spec, "unit_kind", "") or "")
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if kind == "length" and isinstance(v, (int, float)):
        return f"{v:g} m"
    if kind == "angle" and isinstance(v, (int, float)):
        return f"{v:g}°"
    if kind == "choice":
        return str(v).replace("_", " ").capitalize()
    return f"{v:g}" if isinstance(v, float) else str(v)


class TemplateGalleryDialog(QDialog):
    """Pick a parametric starting shape: a searchable list grouped by category on
    the left, the selected template's preview, description and default
    parameters on the right. Double-click (or Enter) uses the template."""

    def __init__(self, parent: QWidget | None = None, *, kind: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New design from template")
        self.resize(960, 600)
        self._selected: str | None = None
        self._infos: dict[str, Any] = {}
        lay = _dialog_layout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        lay.addWidget(split, 1)

        left = QVBoxLayout()
        left.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter")
        self.search.setClearButtonEnabled(True)
        left.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setObjectName("template_list")
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(0)
        self.tree.setIconSize(QSize(56, 28))
        self.tree.setUniformRowHeights(False)
        left.addWidget(self.tree, 1)
        lw = _wrap(left)
        lw.setMinimumWidth(260)
        split.addWidget(lw)

        right = QVBoxLayout()
        right.setContentsMargins(12, 0, 0, 0)
        right.setSpacing(6)
        self.title = label("", "h3")
        right.addWidget(self.title)
        self.detail = label("Select a template.", "secondary", wrap=True)
        right.addWidget(self.detail)
        self.preview = _preview_canvas()
        self.preview.setMinimumSize(420, 240)
        right.addWidget(_well(self.preview), 1)
        right.addWidget(label("Default parameters", "secondary"))
        self.params_view = QTreeWidget()
        self.params_view.setObjectName("template_params")
        self.params_view.setColumnCount(2)
        self.params_view.setHeaderLabels(["Parameter", "Default"])
        self.params_view.setRootIsDecorated(False)
        self.params_view.setFixedHeight(150)
        self.params_view.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self.params_view.header().setStretchLastSection(True)
        self.params_view.setColumnWidth(0, 220)
        right.addWidget(self.params_view)
        split.addWidget(_wrap(right))
        split.setSizes([300, 640])

        self.status = StatusLine()
        self.buttons = _button_box("Create")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(_footer(self.status, self.buttons))

        self.items: dict[str, QTreeWidgetItem] = {}
        self._groups: list[tuple[QTreeWidgetItem, list[QTreeWidgetItem]]] = []
        try:
            infos = list(templates_module().list_templates())
            err = None
        except Exception as exc:  # noqa: BLE001
            infos, err = [], str(exc)
        order = {k: i for i, k in enumerate(KINDS)}
        infos.sort(key=lambda i: (0 if kind and getattr(i, "kind", "") == kind else 1,
                                  order.get(getattr(i, "kind", "generic"), 9)))
        by_group: dict[str, list[Any]] = {}
        for info in infos:
            group = str(getattr(info, "category", "") or "") or _GROUP_TITLES.get(
                str(getattr(info, "kind", "generic")), "Other")
            by_group.setdefault(group, []).append(info)
        for group, items in by_group.items():
            head = QTreeWidgetItem([group])
            head.setFlags(Qt.ItemFlag.ItemIsEnabled)
            f = head.font(0)
            f.setBold(True)
            head.setFont(0, f)
            head.setSizeHint(0, QSize(10, 26))
            self.tree.addTopLevelItem(head)
            children = []
            for info in items:
                name = str(info.name)
                it = QTreeWidgetItem([template_label(info)])
                it.setData(0, NAME_ROLE, name)
                it.setIcon(0, QIcon(template_thumbnail(name)))
                it.setToolTip(0, str(getattr(info, "description", "") or ""))
                it.setSizeHint(0, QSize(10, 34))
                self.tree.addTopLevelItem(it)
                self.items[name] = it
                self._infos[name] = info
                children.append(it)
            self._groups.append((head, children))
        if err or not infos:
            self.status.set("error" if err else "info", err or "No templates available.")
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemDoubleClicked.connect(self._activate_item)
        self.tree.itemActivated.connect(self._activate_item)
        self.search.textChanged.connect(self._filter)
        if self.items:
            self.tree.setCurrentItem(next(iter(self.items.values())))

    # -- selection ----------------------------------------------------------
    def _current_changed(self, cur: QTreeWidgetItem | None, _prev) -> None:
        name = cur.data(0, NAME_ROLE) if cur is not None else None
        if not name:
            return
        self._selected = str(name)
        info = self._infos[self._selected]
        self.title.setText(template_label(info))
        desc = str(getattr(info, "description", "") or "")
        self.detail.setText(desc)
        self.detail.setVisible(bool(desc))
        self.params_view.clear()
        for spec in getattr(info, "params", []) or []:
            row = QTreeWidgetItem([str(getattr(spec, "label", "") or spec.name), _param_text(spec)])
            row.setTextAlignment(1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.params_view.addTopLevelItem(row)
        try:
            design = templates_module().make(self._selected)
        except Exception as exc:  # noqa: BLE001
            design = None
            self.status.set("error", f"This template could not be built: {exc}")
        else:
            n = len(getattr(info, "params", []) or [])
            self.status.set("info", f"{n} parameter{'s' if n != 1 else ''}, editable after creation.")
        self.preview.set_design(design, fit=True)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(design is not None)

    def _activate_item(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        name = item.data(0, NAME_ROLE)
        if name:
            self._selected = str(name)
            self.accept()

    def _filter(self, text: str) -> None:
        words = [w for w in re.split(r"\s+", text.lower()) if w]
        for head, children in self._groups:
            any_visible = False
            for it in children:
                name = str(it.data(0, NAME_ROLE))
                info = self._infos.get(name)
                hay = (f"{name} {it.text(0)} {getattr(info, 'description', '')} "
                       f"{getattr(info, 'kind', '')} {head.text(0)}").lower()
                vis = all(w in hay for w in words)
                it.setHidden(not vis)
                any_visible |= vis
            head.setHidden(not any_visible)

    def template_names(self) -> list[str]:
        return list(self.items)

    def is_listed(self, name: str) -> bool:
        """Whether a template is currently shown (not filtered out)."""
        it = self.items.get(name)
        return it is not None and not it.isHidden()

    def select(self, name: str) -> bool:
        it = self.items.get(name)
        if it is None:
            return False
        self.tree.setCurrentItem(it)
        return True

    def selected_name(self) -> str | None:
        return self._selected

    def design(self) -> Design | None:
        """Design built from the selected template at its defaults."""
        if not self._selected:
            return None
        return templates_module().make(self._selected)
