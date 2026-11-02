"""Import a 3D mesh file (STL, OBJ, PLY) as one or more 3D designs (DESIGN.md 7.4, 7.11).

A mesh file carries no units and no convention for "up" or "forward", and
getting either wrong is the commonest way to waste a 3D run: a car exported in
millimetres read as metres is 4.5 km long, a Y-up game model lies on its side.
So, as the 2D importers do, the dialog shows the design it will create - in a
live 3D preview with the bounding-box dimensions and, for vehicles, the road -
and lets the user settle the four things a file cannot say:

* **Units**, pre-selected from :func:`aerolume.geometry3d.guess_units`, whose
  reason (and the plausible alternatives) is shown under the combo;
* **Orientation**: as in the file, automatic (longest axis along the flow,
  flat side down, mirror symmetry decides the span), or a preset such as
  "Y up -> Z up";
* **Kind**: a vehicle (nose at x = 0, wheels recognised, standing on the road
  with moments about the road) or a generic body (centred, optionally on the
  road), with the height of the model's **lowest point** above the road (the
  tyres of a vehicle: 0 stands it on its wheels). For a vehicle the dialog also
  shows the body's **ground clearance** (wheels excluded), with the definition
  the Geometry page uses, and draws the parts it recognises as wheels in the
  wheel grey;
* for multi-part files, **one design or one design per part**.

On OK the design is built by :func:`aerolume.geometry3d.design_from_meshes`
(references, symmetry and wheel detection; about a second for a 200k-triangle
mesh) and added through :class:`~aerolume.app.state.AppState` by
:func:`import_mesh_dialog`. Reading errors stay in the dialog's status line.
"""
from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core import units
from ...model import Design, Mesh3D
from ..widgets.geometry_canvas import GeometryView3D
from ..widgets.mesh_inspector import (
    CLEARANCE_TIP,
    KIND3_ITEMS,
    ORIENTATION_TIPS,
    ORIENTATIONS,
    UNIT_NAMES,
    body_clearance,
)
from ..widgets.param_form import (
    ElidedLabel,
    QuantitySpinBox,
    StatusLine,
    keep_start_visible,
    label,
    set_line_text,
)
from .import_dialogs import _button_box, _dialog_layout, _footer, _Form, _well

MESH_EXTENSIONS = (".stl", ".obj", ".ply")
MESH_FILTER = ("3D meshes (*.stl *.obj *.ply);;STL (*.stl);;Wavefront OBJ (*.obj);;PLY (*.ply);;"
               "All files (*)")
UNIT_ITEMS = (("m", "Metres"), ("cm", "Centimetres"), ("mm", "Millimetres"), ("in", "Inches"), ("ft", "Feet"))
_UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
_VEHICLE_WORDS = ("car", "vehicle", "truck", "van", "bus", "kart", "sedan", "suv", "lorry", "bike")
ORIENT_ITEMS = (("none", "As in the file"), *ORIENTATIONS)


def is_mesh_path(path) -> bool:
    """Whether a file name is a 3D mesh the importer reads (.stl, .obj, .ply)."""
    return Path(str(path)).suffix.lower() in MESH_EXTENSIONS


def _g3():
    from ... import geometry3d
    return geometry3d


def guess_kind(meshes: list[Mesh3D], name: str = "") -> str:
    """"vehicle" when the parts include wheels or the name says so, else "generic"."""
    low = name.lower()
    if any(w in low for w in _VEHICLE_WORDS):
        return "vehicle"
    with contextlib.suppress(Exception):
        if any(w is not None for w in _g3().find_wheels3d(meshes)):
            return "vehicle"
    return "generic"


def place_preview(meshes: list[Mesh3D], kind: str, on_road: bool, clearance: float) -> list[Mesh3D]:
    """Where :func:`design_from_meshes` will put the meshes (translation only, cheap):
    a vehicle's nose at x = 0, centred on y = 0; on the road at ``clearance``, or
    centred on z = 0."""
    g = _g3()
    x0, x1, y0, y1, z0, z1 = g.bbox(meshes)
    dx = -x0 if kind == "vehicle" else -0.5 * (x0 + x1)
    dz = (clearance - z0) if on_road else -0.5 * (z0 + z1)
    return g.transform(meshes, translate=(dx, -0.5 * (y0 + y1), dz))


def file_provenance(path, fmt: str | None = None) -> dict:
    """``aerolume.api.file_source(path)``: type, file, path, format, bytes and SHA-256 of the file read.

    The dialogs record what a script's ``import_mesh`` records, so a GUI import can be traced to the
    exact bytes too. If the file cannot be read again (moved since it was opened) the name, format
    and path are kept without the checksum."""
    p = Path(str(path or ""))
    try:
        from ...api import file_source
        return file_source(p, fmt)
    except (OSError, ImportError):
        return {"type": "import", "file": p.name, "path": str(p), "format": fmt or p.suffix.lower().lstrip(".")}


class MeshImportDialog(QDialog):
    """Units, orientation, kind and placement of a mesh file, with a live 3D preview."""

    def __init__(self, parent: QWidget | None = None, *, path: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import 3D mesh")
        self.setObjectName("mesh_import_dialog")
        self.resize(1040, 620)
        self.path: str | None = None
        self.load_error = ""
        self._raw: list[Mesh3D] = []
        self._notes: list[str] = []
        self._guess: tuple[float, str] = (1.0, "")
        self._oriented: dict[tuple, tuple[list[Mesh3D], list[str]]] = {}
        self._wheels: dict[str, list[bool]] = {}      # orientation -> which parts are wheels
        self._designs: list[Design] | None = None
        self._designs_key: tuple | None = None
        lay = _dialog_layout(self)
        body = QHBoxLayout()
        body.setSpacing(12)
        lay.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(8)
        form = _Form(label_width=96)
        self.file_label = ElidedLabel("", None, mode=Qt.TextElideMode.ElideMiddle)
        self.file_label.setObjectName("mesh_file")
        browse = QPushButton("Browse…")
        browse.setObjectName("mesh_browse")
        browse.clicked.connect(self._browse)
        frow = QWidget()
        fl = QHBoxLayout(frow)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        fl.addWidget(self.file_label, 1)
        fl.addWidget(browse)
        form.add_row("File", frow)
        self.contents = label("", "secondary")
        self.contents.setObjectName("mesh_contents")
        form.add_row("Contents", self.contents)
        self.units = QComboBox()
        self.units.setObjectName("mesh_units")
        for k, t in UNIT_ITEMS:
            self.units.addItem(t, k)
        form.add_row("Units", self.units, "The unit the file's numbers are in; they are scaled to metres")
        self.units_reason = label("", "muted", wrap=True)
        self.units_reason.setObjectName("units_reason")
        form.add_widget(self.units_reason)
        self.size_label = label("", None)
        self.size_label.setObjectName("mesh_size")
        form.add_row("Size", self.size_label, "Length × width × height in metres, after units and orientation")
        self.orientation = QComboBox()
        self.orientation.setObjectName("mesh_orientation")
        for k, t in ORIENT_ITEMS:
            self.orientation.addItem(t, k)
            self.orientation.setItemData(self.orientation.count() - 1, ORIENTATION_TIPS.get(k, ""),
                                         Qt.ItemDataRole.ToolTipRole)
        form.add_row("Orientation", self.orientation,
                     "Aerolume's frame: x downstream (the nose points to −x, into the flow), y across, z up")
        self.kind = QComboBox()
        self.kind.setObjectName("mesh_kind")
        for k, t in KIND3_ITEMS:
            self.kind.addItem(t, k)
        form.add_row("Kind", self.kind, "Vehicles stand on the road with their wheels recognised; generic "
                                        "bodies are centred")
        self.on_road = QCheckBox("Stands on the road")
        self.on_road.setObjectName("mesh_on_road")
        form.add_row("Road", self.on_road)
        # the model's lowest point, which for a vehicle is the tyres (0: on its wheels); not
        # called "Clearance", which the Geometry page uses for the body (wheels excluded)
        self.clearance = QuantitySpinBox("length", "m", minimum=0.0, maximum=100.0, step=0.005)
        self.clearance.setObjectName("mesh_clearance")
        form.add_row("Lowest point", self.clearance,
                     "Height of the model's lowest point above the road (z = 0): the tyres of a vehicle, "
                     "so 0 stands it on its wheels")
        self.ground_clearance = label("", "secondary")
        self.ground_clearance.setObjectName("mesh_body_clearance")
        self._ground_clearance_label = form.add_row("Ground clearance", self.ground_clearance, CLEARANCE_TIP)
        self.parts = QComboBox()
        self.parts.setObjectName("mesh_parts")
        self.parts.addItem("One design", "one")
        self.parts.addItem("One design per part", "each")
        form.add_row("Parts", self.parts, "A multi-part file (OBJ groups, STL solids) can become one design "
                                          "or one design per part")
        self.name = keep_start_visible(QLineEdit())        # a long file name shows its start
        self.name.setObjectName("mesh_name")
        form.add_row("Name", self.name)
        left.addWidget(form)
        left.addStretch(1)
        lw = QWidget()
        left.setContentsMargins(0, 0, 0, 0)
        lw.setLayout(left)
        lw.setFixedWidth(340)
        body.addWidget(lw)
        self.preview = GeometryView3D()
        self.preview.setObjectName("mesh_preview")
        self.preview.setAcceptDrops(False)
        self.preview.setMinimumSize(520, 380)
        self.preview.set_placeholder("No mesh", "Choose an STL, OBJ or PLY file.")
        body.addWidget(_well(self.preview), 1)

        self.status = StatusLine()
        self.status.setObjectName("mesh_status")
        self.buttons = _button_box("Import")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(_footer(self.status, self.buttons))

        for w in (self.units, self.orientation, self.kind, self.parts):
            w.currentIndexChanged.connect(self._changed)
        self.on_road.toggled.connect(self._changed)
        self.clearance.siValueChanged.connect(self._changed)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        if path:
            self.load(path)
        else:
            self._refresh()

    # -- file ------------------------------------------------------------------------
    def _ask_file(self) -> str:
        start = ""
        with contextlib.suppress(Exception):
            from .. import settings
            start = settings.get().last_dir
        path, _ = QFileDialog.getOpenFileName(self, "Import 3D mesh", str(start or ""), MESH_FILTER)
        if path:
            with contextlib.suppress(Exception):
                from .. import settings
                settings.get().last_dir = str(Path(path).parent)
        return path

    def _browse(self) -> None:
        path = self._ask_file()
        if path:
            self.load(path)

    def load(self, path: str) -> bool:
        """Read ``path`` (in its own units) and pre-select units, kind and name."""
        g = _g3()
        self.path = str(path)
        self.file_label.set_full_text(Path(path).name, str(path))
        self._oriented.clear()
        self._wheels.clear()
        self._designs = None
        notes: list[str] = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._raw = [m for m in g.read_mesh(path, notes=notes) if len(m.faces)]
            if not self._raw:
                raise ValueError("the file has no triangles")
            self.load_error = ""
        except Exception as exc:  # noqa: BLE001 - every reader problem is shown, never raised
            self._raw, self.load_error = [], f"Could not read {Path(path).name}: {exc}"
        finally:
            QApplication.restoreOverrideCursor()
        self._notes = notes
        if self._raw:
            self._guess = g.guess_units(self._raw)
            key = next((k for k, s in _UNIT_SCALE.items() if np.isclose(s, self._guess[0])), "m")
            kind = guess_kind(self._raw, Path(path).stem)
            for w, v in ((self.units, key), (self.kind, kind)):
                w.blockSignals(True)
                w.setCurrentIndex(max(w.findData(v), 0))
                w.blockSignals(False)
            self.on_road.blockSignals(True)
            self.on_road.setChecked(kind == "vehicle")
            self.on_road.blockSignals(False)
            set_line_text(self.name, Path(path).stem.replace("_", " ").strip() or "Mesh")
            n = len(self._raw)
            tri = sum(len(m.faces) for m in self._raw)
            self.contents.setText(f"{tri:,} triangles in {n} part{'s' if n != 1 else ''}")
            self.units_reason.setText(self._guess[1])
        else:
            self.contents.setText("—")
            self.units_reason.setText("")
        self._kind_changed()
        self._refresh()
        return not self.load_error

    # -- options -> preview -----------------------------------------------------------
    def unit_scale(self) -> float:
        return _UNIT_SCALE[str(self.units.currentData())]

    def _kind_changed(self, *_a) -> None:
        vehicle = self.kind.currentData() == "vehicle"
        self.on_road.setEnabled(not vehicle)
        if vehicle and not self.on_road.isChecked():
            self.on_road.blockSignals(True)
            self.on_road.setChecked(True)
            self.on_road.blockSignals(False)
        self._changed()

    def _changed(self, *_a) -> None:
        self._designs = None
        self._refresh()

    def _oriented_raw(self) -> tuple[list[Mesh3D], list[str]]:
        """The file's meshes, in file units, turned to the chosen orientation.

        Cached per orientation: a rotation commutes with the unit scale (and
        ``auto_orient``'s tests are relative to the size), so changing the units
        never re-runs the automatic orientation, which samples the surface."""
        g = _g3()
        key = str(self.orientation.currentData())
        if key not in self._oriented:
            notes: list[str] = []
            if key == "auto":
                ms = list(g.auto_orient(self._raw, notes))
            elif key != "none":
                ms = list(g.orient_preset(self._raw, key))
            else:
                ms = list(self._raw)
            self._oriented[key] = (ms, notes)
        return self._oriented[key]

    def _wheel_mask(self) -> list[bool]:
        """Which oriented parts :func:`~aerolume.geometry3d.find_wheels3d` takes for wheels,
        as the import will. Cached per orientation: the test is relative to the model's
        size, so the units do not change it, and it fits a circle to every part."""
        key = str(self.orientation.currentData())
        if key not in self._wheels:
            ms, _ = self._oriented_raw()
            try:
                self._wheels[key] = [w is not None for w in _g3().find_wheels3d(ms)]
            except Exception:  # noqa: BLE001 - no wheels found is a valid answer for a preview
                self._wheels[key] = [False] * len(ms)
        return self._wheels[key]

    def _oriented_meshes(self) -> tuple[list[Mesh3D], list[str]]:
        """The oriented meshes scaled to metres (for the preview)."""
        ms, notes = self._oriented_raw()
        s = self.unit_scale()
        return (_g3().transform(ms, scale=s) if s != 1.0 else ms), notes

    def _refresh(self) -> None:
        ok = bool(self._raw)
        self.parts.setEnabled(ok and len(self._raw) > 1)
        on_road = self.on_road.isChecked()
        self.clearance.setEnabled(on_road)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
        kind = str(self.kind.currentData())
        # the body's ground clearance, as the Geometry page will show it: vehicles only (for
        # a generic body it equals the lowest point) and only when the parts stay together
        one = self.parts.currentData() != "each" or len(self._raw) < 2
        show_body = ok and kind == "vehicle" and on_road and one
        self.ground_clearance.setVisible(show_body)
        self._ground_clearance_label.setVisible(show_body)
        if not ok:
            self.size_label.setText("—")
            self.preview.set_design(None, meshes=[])
            self.status.set("error" if self.load_error else "info", self.load_error or "Choose a mesh file.")
            return
        ms, notes = self._oriented_meshes()
        placed = place_preview(ms, kind, on_road, self.clearance.si_value())
        if kind == "vehicle":
            # the parts the import will take for wheels: drawn in the wheel grey, as on the
            # Geometry page, and left out of the ground clearance
            mask = self._wheel_mask()
            placed = [dataclasses.replace(m, role="wheel") if w and m.role == "body" else m
                      for m, w in zip(placed, mask, strict=True)]
        if show_body:
            nw = sum(m.role == "wheel" for m in placed)
            self.ground_clearance.setText(f"{body_clearance(placed):.3f} m" + (
                f" · {nw} wheel{'s' if nw != 1 else ''} recognised" if nw else " · no wheels recognised"))
        x0, x1, y0, y1, z0, z1 = _g3().bbox(placed)
        self.size_label.setText(f"{x1 - x0:.3f} × {y1 - y0:.3f} × {z1 - z0:.3f} m")
        preview = Design(name=self.name.text() or "Mesh", kind=kind, bodies=[], length_ref=1.0, force_ref=1.0,
                         dimension=3, meshes=placed, id="__mesh_import_preview__")
        self.preview.set_design(preview, meshes=placed, ground=on_road, fit=True)
        size = max(x1 - x0, y1 - y0, z1 - z0)
        if size > 50 or size < 0.005:       # a bus is 12 m, a ship is not wind-tunnel work
            self.status.set("warning", f"The model is {units.sig(size, 3)} m across: check the units.")
        else:
            n = len(self._raw)
            what = (f"{n} designs" if self.parts.currentData() == "each" and n > 1 else "1 design")
            unit = UNIT_NAMES.get(self.unit_scale(), "").lower()
            self.status.set("info", f"Imports as {what}; read in {unit}"
                            + (f"; {notes[0][0].lower()}{notes[0][1:]}" if notes else "."))

    # -- result ------------------------------------------------------------------------
    def _settings_key(self) -> tuple:
        return (self.path, self.unit_scale(), str(self.orientation.currentData()), str(self.kind.currentData()),
                self.on_road.isChecked(), round(self.clearance.si_value(), 9), str(self.parts.currentData()),
                self.name.text().strip())

    def designs(self) -> list[Design]:
        """The designs the import creates (built once per set of options)."""
        if not self._raw:
            raise ValueError(self.load_error or "no mesh loaded")
        key = self._settings_key()
        if self._designs is not None and self._designs_key == key:
            return self._designs
        g = _g3()
        s = self.unit_scale()
        # design_from_meshes scales to metres itself and records the scale
        ms, orient_notes = self._oriented_raw()
        kind = str(self.kind.currentData())
        name = self.name.text().strip() or Path(self.path or "Mesh").stem
        unit = str(self.units.currentData())
        notes = list(self._notes) + list(orient_notes)
        if unit != "m" or not np.isclose(self._guess[0], s):
            notes.append(f"Read in {UNIT_NAMES.get(s, unit).lower()} (scale {s:g} to metres).")
        groups = [[m] for m in ms] if self.parts.currentData() == "each" and len(ms) > 1 else [ms]
        out = []
        for grp in groups:
            src = {**file_provenance(self.path),
                   "orientation": str(self.orientation.currentData()), "import_notes": list(notes)}
            nm = name if len(groups) == 1 else f"{name} – {grp[0].name}"
            out.append(g.design_from_meshes(grp, nm, kind, units=s, on_road=self.on_road.isChecked(),
                                            clearance=self.clearance.si_value() if self.on_road.isChecked() else None,
                                            source=src))
        self._designs, self._designs_key = out, key
        return out

    def accept(self) -> None:
        """Build the designs before closing; a failure stays in the dialog."""
        self.status.set("info", "Measuring the frontal area and the symmetry…")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.designs()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.status.set("error", f"Could not create the design: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        super().accept()

    def done(self, r: int) -> None:
        super().done(r)
        with contextlib.suppress(Exception):     # release the preview's OpenGL context now
            self.preview.scene.close()


def import_mesh_dialog(parent: QWidget | None, state, path: str | None = None, *, exec_dialog=None,
                       add=None) -> list[Design] | None:
    """Ask for a mesh file (unless ``path`` is given), show the import dialog and
    add the resulting design(s) to the open project through ``state``.

    Returns the designs added, or None when cancelled (or without a project).
    ``exec_dialog(dialog) -> bool`` replaces the modal ``exec`` (tests, the
    Geometry page's funnel); ``add(design)`` replaces ``state.add_design`` (the
    Geometry page passes its own, which makes names unique and logs).
    """
    if state is None or state.project is None:
        return None
    dlg = MeshImportDialog(parent, path=None)
    if path is None:
        path = dlg._ask_file()
        if not path:
            dlg.done(QDialog.DialogCode.Rejected)
            dlg.deleteLater()
            return None
    dlg.load(path)
    run = exec_dialog or (lambda d: d.exec() == QDialog.DialogCode.Accepted)
    try:
        if not run(dlg):
            return None
        designs = dlg.designs()
    except Exception as exc:  # noqa: BLE001 - reported through the state's log
        state.log(f"Import failed: {exc}", "error", "Geometry")
        state.message(f"Could not import {Path(str(path)).name}: {exc}", 8000)
        return None
    finally:
        # the dialog is a child of ``parent``: drop it and the file's arrays it caches now,
        # or every import would keep a copy of its mesh alive for the life of the page
        with contextlib.suppress(Exception):
            dlg.preview.scene.close()
        dlg._raw, dlg._designs = [], None
        dlg._oriented.clear()
        dlg.deleteLater()
    adder = add or (lambda d: state.add_design(d, select=True))
    for d in designs:
        adder(d)
    return designs


__all__ = ["MESH_EXTENSIONS", "MESH_FILTER", "MeshImportDialog", "guess_kind", "import_mesh_dialog",
           "is_mesh_path", "place_preview"]
