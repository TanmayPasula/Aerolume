"""Geometry page: view, create, import and edit the current design.

Layout (docs/ui_guidelines.md sections 3-4): the viewport takes the space, with
a compact tool strip above it (Import, Templates, Undo/Redo, Fit, and the
Points / Dimensions / Grid toggles) and a one-line status strip below it
(design summary, cursor position). On the right sits the inspector, a property
grid with five groups: Source (how the shape is made: template parameters, NACA
code or the imported file), Transform, Cleanup, References and Diagnostics.
Design selection, renaming, duplication and deletion belong to the Project
tree and the Edit menu; this page edits whichever design is current.

Design edits are *recipes*, not destructive changes. A design keeps in
``Design.source`` how it was made (template + parameters, NACA code, file) and
the ordered list of edits applied since (``source["edits"]``: scale, rotate,
translate, mirror, lift, resample, smooth). Changing a template parameter
therefore regenerates the shape *and replays the edits*, so a car scaled to
4.2 m stays 4.2 m when its roof is lowered. Reference quantities follow the
outline automatically unless the user typed one (``source["manual_refs"]``).

Every edit is one ``QUndoCommand`` holding the design before and after (as
JSON-safe dicts: a design is at most a few thousand points, so snapshots are
cheaper and far more robust than inverse operations). There is one undo stack
per design, grouped in a ``QUndoGroup`` that follows the current design.

All reads and writes go through :class:`~aerolume.app.state.AppState`. Modal
interactions are funnelled through small ``_ask_*`` / ``_exec_dialog`` methods
so tests can drive the page without blocking dialogs.

**3D designs** (``Design.dimension == 3``, DESIGN.md 7.11) switch the page to
a 3D workspace with its own tool strip (Import, New, undo, the Iso / Front /
Side / Top / Back camera presets, Fit, Edges, Dimensions):

* a meshed design (an imported STL / OBJ / PLY, or a primitive) is shown in a
  :class:`~aerolume.app.widgets.geometry_canvas.GeometryView3D` with the road
  under vehicles and the bounding-box dimensions, and edited in the
  :class:`~aerolume.app.widgets.mesh_inspector.MeshInspector` (transform,
  references, parts, diagnostics);
* a wing design (``kind == "wing"``) gets the planform editor
  (:class:`~aerolume.app.widgets.wing_editor.WingEditor`: 3D preview of the
  loft, top-view planform, section table) and its inspector.

Undo works the same for both, with one difference for meshes: a 3D snapshot
is a :func:`~aerolume.app.widgets.mesh_inspector.clone_design` that *shares*
the vertex and face arrays (``to_dict`` leaves them out, and copying a large
mesh per step would be wasteful), so a step costs only the arrays the edit
actually replaced. 2D designs keep the canvas and inspector described above.
"""
from __future__ import annotations

import contextlib
import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QUndoCommand, QUndoGroup, QUndoStack
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core import units
from ...model import KINDS_2D, Design, WingSpec
from ..dialogs.import_dialogs import (
    FILE_FILTER,
    FORMAT_FILTERS,
    FORMAT_NAMES,
    KIND_LABELS,
    ImageSilhouetteDialog,
    NacaDialog,
    PolygonImportDialog,
    TemplateGalleryDialog,
    classify_path,
    describe_naca,
    geometry_module,
    naca_design,
    template_label,
    templates_module,
    validate_naca,
)
from ..dialogs.mesh_import import MESH_FILTER, import_mesh_dialog, is_mesh_path
from ..state import AppState
from ..widgets import physics_editor as phys
from ..widgets.common import Divider, EmptyState, PropertyGrid, ToolStrip
from ..widgets.geometry_canvas import GeometryCanvas, GeometryView3D, airfoil_chord, frontal_extent
from ..widgets.mesh_inspector import (
    MeshInspector,
    area_text,
    clone_design,
    design_summary3d,
    edit_design3d,
    measure_symmetry,
    refresh_references3d,
    symmetric_from_error,
)
from ..widgets.param_form import (
    FormGrid,
    ParamForm,
    QuantitySpinBox,
    StatusLine,
    clear_layout,
    display_unit,
    guard_wheel,
    hbox,
    icon,
    label,
    set_prop,
    small_button,
    svg_icon,
)
from ..widgets.wing_editor import (
    WingEditor,
    WingInspector,
    airfoil_resolver,
    generate_planform,
    project_airfoils,
    refresh_wing_references,
    update_wing,
)

INSPECTOR_WIDTH = 380
LABEL_W = 116            # inspector label column (shared by every row and sub-form)
ROW_BUTTON_W = 64        # "Rotate", "Move", "Auto" ... one width so editors line up

# ---------------------------------------------------------------------------
# pure helpers (no Qt) - unit-tested directly
# ---------------------------------------------------------------------------

REF_FIELDS = ("length_ref", "force_ref", "moment_ref")


def auto_references(design: Design) -> tuple[float, float, tuple[float, float]]:
    """Reference length, force length and moment point that follow the outline.

    Conventions (DESIGN.md section 3): airfoils use the chord for both lengths
    and the quarter chord as moment point; vehicles use the overall length for
    Re, the frontal height (union of the bodies' vertical extents, wheels
    included) for coefficients, and the mid-length point on the road; generic
    bodies the overall length, frontal height and bounding-box centre.
    """
    if not design.bodies:
        return design.length_ref, design.force_ref, tuple(design.moment_ref)  # type: ignore[return-value]
    x0, x1, y0, y1 = design.bounds()
    length = max(x1 - x0, 1e-12)
    polys = [b.points for b in design.bodies]
    if design.kind == "airfoil":
        le, te = airfoil_chord(design)
        c = float(np.hypot(*(te - le))) or length
        q = le + 0.25 * (te - le)
        return c, c, (float(q[0]), float(q[1]))
    height = max(frontal_extent(polys), 1e-12)
    if design.kind == "vehicle":
        return length, height, (0.5 * (x0 + x1), 0.0)
    return length, height, (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _map_bodies(design: Design, fn) -> None:
    """Apply a point map ``fn((N,2)) -> (N,2)`` to every body (and wheel centre)."""
    for b in design.bodies:
        b.points = np.asarray(fn(np.asarray(b.points, float)), float).reshape(-1, 2)
        if b.center is not None:
            c = fn(np.asarray([b.center], float))[0]
            b.center = (float(c[0]), float(c[1]))


def apply_edit(design: Design, edit: dict) -> None:
    """Apply one recorded edit to a design in place.

    Edits are semantic ("scale to 4.2 m", "rotate 2 deg about the centre") rather
    than absolute matrices so that they still mean the same thing when replayed
    on a regenerated shape whose size has changed.

    * ``scale_to`` - uniform scale about the origin so the chord (airfoils) or
      overall length becomes ``length``. The origin is the leading edge for
      airfoils and on the road for vehicles, so neither drifts.
    * ``scale`` - uniform ``factor`` about the origin.
    * ``rotate`` - ``deg`` counter-clockwise about the bounding-box centre.
    * ``translate`` - by ``dx``, ``dy`` metres.
    * ``mirror`` - ``axis`` "x" flips front/back, "y" flips top/bottom (about
      the bounding-box centre); point order is reversed to stay CCW.
    * ``lift`` - translate vertically so the lowest point sits ``height`` above
      the road (vehicles).
    * ``resample`` - ``n`` points per body; ``smooth`` - ``iterations`` passes.
    """
    op = edit.get("op")
    if not design.bodies:
        return
    x0, x1, y0, y1 = design.bounds()
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if op in ("scale_to", "scale"):
        if op == "scale_to":
            if design.kind == "airfoil":
                le, te = airfoil_chord(design)
                current = float(np.hypot(*(te - le)))
            else:
                current = x1 - x0
            k = float(edit["length"]) / current if current > 0 else 1.0
        else:
            k = float(edit["factor"])
        _map_bodies(design, lambda p: p * k)
    elif op == "rotate":
        a = math.radians(float(edit["deg"]))
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, -s], [s, c]])
        ctr = np.array([cx, cy])
        _map_bodies(design, lambda p: (p - ctr) @ rot.T + ctr)
    elif op == "translate":
        d = np.array([float(edit.get("dx", 0.0)), float(edit.get("dy", 0.0))])
        _map_bodies(design, lambda p: p + d)
    elif op == "mirror":
        axis = edit.get("axis", "x")
        if axis == "x":
            _map_bodies(design, lambda p: np.column_stack([2 * cx - p[:, 0], p[:, 1]]))
        else:
            _map_bodies(design, lambda p: np.column_stack([p[:, 0], 2 * cy - p[:, 1]]))
        for b in design.bodies:          # a reflection turns CCW into CW
            b.points = b.points[::-1].copy()
            if b.role == "wheel":
                b.omega = -b.omega
    elif op == "lift":
        dy = float(edit["height"]) - float(np.vstack([b.points for b in design.bodies])[:, 1].min())
        _map_bodies(design, lambda p: p + np.array([0.0, dy]))
    elif op == "resample":
        geo = geometry_module()
        n = int(edit["n"])
        for b in design.bodies:
            # wheels are circles: a fixed modest count keeps them round and cheap
            b.points = np.asarray(geo.resample(b.points, n if b.role != "wheel" else min(n, 96)), float)
    elif op == "smooth":
        geo = geometry_module()
        for b in design.bodies:
            b.points = np.asarray(geo.smooth(b.points, int(edit.get("iterations", 1))), float)
    else:
        raise ValueError(f"unknown edit {op!r}")


def refresh_references(design: Design, *, scale: float | None = None) -> None:
    """After a geometry change: recompute the auto references, keep manual ones.

    ``source["manual_refs"]`` lists the fields the user typed; those keep their
    value, except that a uniform ``scale`` (from a scale edit) scales manual
    lengths and the manual moment point too - a chord typed for a 1 m section
    is wrong for a 2 m one.
    """
    manual = set(design.source.get("manual_refs", []))
    length, force, mref = auto_references(design)
    k = 1.0 if scale is None else float(scale)
    design.length_ref = design.length_ref * k if "length_ref" in manual else length
    design.force_ref = design.force_ref * k if "force_ref" in manual else force
    if "moment_ref" in manual:
        design.moment_ref = (design.moment_ref[0] * k, design.moment_ref[1] * k)
    else:
        design.moment_ref = mref


def template_source(design: Design) -> tuple[str, dict] | None:
    """(template name, params) if the design was made from a template."""
    src = design.source or {}
    if src.get("type") != "template":
        return None
    name = src.get("template") or src.get("name")
    if not name:
        return None
    return str(name), dict(src.get("params") or {})


# Recipe parameters that own a quantity a Transform edit can also set. A
# template or NACA design hides those Transform rows (one editor per quantity),
# and typing one of these parameters drops older edits that would override it
# on replay - otherwise "Length 5 m" followed by a recorded "scale to 4.7 m"
# would silently rebuild a 4.7 m car.
SIZE_PARAMS = frozenset({"length", "chord", "diameter", "side"})
HEIGHT_PARAMS = frozenset({"ground_clearance", "ride_height", "clearance"})


def supersede_edits(edits: list[dict], changed: set[str]) -> list[dict]:
    """``edits`` without the ones a just-changed recipe parameter now owns:
    scale edits when a size parameter changed, road-height edits when a
    clearance parameter changed. Other edits (rotate, move, mirror ...) stay."""
    drop: set[str] = set()
    if changed & SIZE_PARAMS:
        drop |= {"scale_to", "scale"}
    if changed & HEIGHT_PARAMS:
        drop.add("lift")
    return [e for e in edits if e.get("op") not in drop]


def regenerate(design: Design, *, params: dict | None = None, naca: dict | None = None) -> Design:
    """Rebuild a template / NACA design from its recipe and replay its edits.

    Returns a new design with the same id, name, notes and creation time.
    ``params`` replaces the template parameters; ``naca`` updates any of
    ``code``, ``chord``, ``n`` of a NACA design. Manual reference values survive.
    Edits that a changed parameter now owns are dropped (:func:`supersede_edits`).
    """
    src = copy.deepcopy(design.source or {})
    edits = list(src.get("edits", []))
    manual = list(src.get("manual_refs", []))
    if src.get("type") == "template":
        name, old = template_source(design) or ("", {})
        new_params = {**old, **(params or {})}
        edits = supersede_edits(edits, {k for k, v in (params or {}).items() if old.get(k) != v})
        fresh = templates_module().make(name, **new_params)
        fresh.source = {**(fresh.source or {}), "type": "template", "template": name,
                        "params": new_params}
    elif src.get("type") == "naca":
        cfg = {"code": src.get("code", "0012"), "chord": float(src.get("chord", design.length_ref)),
               "n": int(src.get("n", 160)), **(naca or {})}
        if "chord" in (naca or {}) and float(cfg["chord"]) != float(src.get("chord", design.length_ref)):
            edits = supersede_edits(edits, {"chord"})
        fresh = naca_design(str(cfg["code"]), float(cfg["chord"]), int(cfg["n"]))
    else:
        raise ValueError("only template and NACA designs can be regenerated")
    fresh.id, fresh.name, fresh.created, fresh.notes = design.id, design.name, design.created, design.notes
    for e in edits:
        apply_edit(fresh, e)
    fresh.source["edits"] = edits
    if manual:
        fresh.source["manual_refs"] = manual
        for f in manual:
            setattr(fresh, f, copy.deepcopy(getattr(design, f)))
    if edits:
        refresh_references(fresh)
    return fresh


_KIND_NOUN = {"airfoil": "Airfoil", "vehicle": "Vehicle", "generic": "Body", "wing": "Wing"}

#: The name of a new wing (made unique): the main window's fallback uses the same, and
#: it is not the kind word, which the step bar and the tree already show before it.
NEW_WING_NAME = "Tapered wing"

#: Undo depth per design. A 3D step holds the arrays its edit replaced (a rotated
#: 200k-triangle mesh: ~5 MB), so meshed designs keep fewer steps than outlines.
UNDO_LIMIT_2D = 200
UNDO_LIMIT_3D = 60

#: Shapes the New 3D menu creates (geometry3d.primitives): key -> (label, factory kwargs)
PRIMITIVES = {
    "sphere": ("Sphere", {"d": 0.2}),
    "box": ("Box", {"lx": 0.5, "ly": 0.3, "lz": 0.3}),
    "cylinder": ("Cylinder", {"d": 0.2, "length": 0.6, "axis": "y"}),
}


def default_wing_spec() -> WingSpec:
    """The planform a new wing starts from: a 2 m tapered wing (taper 0.6, 2°
    dihedral, NACA 2412), aspect ratio 8.3 - a typical model-glider wing that
    every analysis handles; the generators and the table change it from there."""
    return WingSpec(sections=generate_planform("tapered", span=2.0, root_chord=0.3, tip_chord=0.18,
                                               dihedral_deg=2.0, airfoil="2412"))


def design_summary(design: Design) -> str:
    """One factual line for the status strip: kind, size, bodies, points."""
    if design.dimension == 3:
        return design_summary3d(design)
    if not design.bodies:
        return f"{_KIND_NOUN.get(design.kind, design.kind)} · no bodies"
    x0, x1, _, _ = design.bounds()
    npts = sum(len(b.points) for b in design.bodies)
    unit = display_unit("length")
    if design.kind == "airfoil":
        size = f"chord {units.fmt(design.length_ref, 'length', unit, 3)}"
    else:
        size = f"length {units.fmt(x1 - x0, 'length', unit, 3)}"
    nb = len(design.bodies)
    return (f"{_KIND_NOUN.get(design.kind, design.kind)} · {size} · {nb} bod{'ies' if nb != 1 else 'y'} · "
            f"{npts:,} points")


def source_type_text(design: Design) -> str:
    """How a design was made, as a noun phrase for the Source group."""
    src = design.source or {}
    typ = str(src.get("type", ""))
    if typ == "template":
        return "Template"
    if typ == "physics_template":
        return "Starter case"                  # Setup ▸ New case from template (physics_templates)
    if typ == "naca":
        return "NACA section"
    if typ == "import":
        return f"Imported {FORMAT_NAMES.get(str(src.get('format', '')), 'file')}"
    if typ == "airfoil":
        return "Airfoil coordinates"
    return typ.capitalize() or "Custom"


def starter_case_info(design: Design):
    """The :class:`~aerolume.app.dialogs.physics_templates.PhysicsTemplate` a starter-case design was
    made from (title and description for the Source group), or None."""
    src = design.source or {}
    if src.get("type") != "physics_template" or not src.get("template"):
        return None
    try:
        from ..dialogs.physics_templates import template_info
        return template_info(str(src["template"]))
    except Exception:  # noqa: BLE001 - an unknown or newer template key: the plain "Starter case"
        return None


def check_messages(design: Design) -> list[tuple[str, str]]:
    """Diagnostics as (level, text) with level in "error" | "warning" | "info".

    Uses ``geometry.check_design`` when available, accepting strings, (level,
    text) pairs, dicts or objects with ``level``/``message``; adds a few cheap
    checks of its own so the group is useful even without it.
    """
    out: list[tuple[str, str]] = []
    if not design.bodies and design.dimension == 2 and (design.source or {}).get("physics"):
        # a natural-convection cavity: no body is the case, the closed box's walls drive the flow
        return [("info", ("No bodies: the walls of the simulation's closed box drive the flow (Setup ▸ Domain "
                          "and Heat transfer)."))]
    try:
        geo = geometry_module()
        check = getattr(geo, "check_design", None)
    except RuntimeError:
        geo, check = None, None
    if check is not None:
        try:
            raw = check(design) or []
        except Exception as exc:  # noqa: BLE001 - a checker failure is itself a diagnostic
            raw = [("warning", f"Geometry check failed: {exc}")]
        for m in raw:
            if isinstance(m, str):
                # geometry.check_design returns plain sentences, each a problem or caveat
                low = m.lower()
                if low.startswith("error"):
                    lvl = "error"
                elif low.startswith(("note", "info")):
                    lvl = "info"
                else:
                    lvl = "warning"
                out.append((lvl, m))
            elif isinstance(m, dict):
                out.append((str(m.get("level", "warning")), str(m.get("message", m.get("text", m)))))
            elif isinstance(m, (tuple, list)) and len(m) >= 2:
                out.append((str(m[0]), str(m[1])))
            else:
                out.append((str(getattr(m, "level", "warning")),
                            str(getattr(m, "message", getattr(m, "text", m)))))
        return out
    # fallback checks
    for b in design.bodies:
        if len(b.points) < 8:
            msg = f"{b.name}: only {len(b.points)} points; resample for a smoother surface."
            out.append(("warning", msg))
        if geo is not None and hasattr(geo, "is_self_intersecting") and geo.is_self_intersecting(b.points):
            out.append(("error", f"{b.name}: the outline crosses itself."))
    if design.kind == "vehicle" and design.bodies:
        ymin = design.bounds()[2]
        if ymin < -1e-9:
            depth = units.fmt(-ymin, "length", "mm", 3)
            out.append(("error", f"The vehicle extends {depth} below the road."))
    return out


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------

class DesignEditCommand(QUndoCommand):
    """Snapshot undo step for one design: apply ``after`` on redo, ``before`` on undo.

    A snapshot is the design's JSON dict (2D outlines, wings) or, for a meshed
    3D design, a :func:`clone_design` sharing the mesh arrays.

    ``merge_key`` lets consecutive edits of the same control (holding a spin
    box arrow, typing a few values in a row) collapse into one undo step - but
    only within ``MERGE_WINDOW_S`` of the previous edit, so coming back to the
    same field later is a separate, separately undoable step.
    """

    MERGE_ID = 0x4145524F
    MERGE_WINDOW_S = 1.5

    def __init__(self, page: GeometryPage, design_id: str, before: dict | Design, after: dict | Design, text: str,
                 merge_key: str | None = None) -> None:
        super().__init__(text)
        self.page, self.design_id = page, design_id
        self.before, self.after = before, after
        self.merge_key = merge_key
        self.stamp = time.monotonic()

    def id(self) -> int:
        return self.MERGE_ID if self.merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if (isinstance(other, DesignEditCommand) and other.merge_key == self.merge_key
                and other.design_id == self.design_id
                and other.stamp - self.stamp <= self.MERGE_WINDOW_S):
            self.after = other.after
            self.stamp = other.stamp
            return True
        return False

    def redo(self) -> None:
        self.page._apply_snapshot(self.design_id, self.after)

    def undo(self) -> None:
        self.page._apply_snapshot(self.design_id, self.before)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def _toggle(text: str, tooltip: str) -> QToolButton:
    """A small checkable "Auto" switch for a reference row."""
    b = QToolButton()
    b.setText(text)
    b.setCheckable(True)
    b.setProperty("variant", "toggle")
    b.setToolTip(tooltip)
    b.setFixedSize(ROW_BUTTON_W, 24)
    return b


class GeometryPage(QWidget):
    """The Geometry step: canvas with tool strip, and the design inspector (DESIGN.md 5.8)."""

    REGEN_DELAY_MS = 120

    def __init__(self, state: AppState, jobs: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.jobs = jobs
        self._loading = False                 # True while widgets are filled from the model
        self._stacks: dict[str, QUndoStack] = {}
        self._undo_group = QUndoGroup(self)
        self._source_sig: tuple | None = None
        self._pending_params: dict | None = None
        self._pending_key: str | None = None
        self._pending_design_id = ""
        self._naca_fields: set[str] = set()
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(self.REGEN_DELAY_MS)
        self._regen_timer.timeout.connect(self._flush_regeneration)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._pages = QStackedWidget()
        root.addWidget(self._pages)
        self._pages.addWidget(EmptyState(title="No project open",
                                         text="Designs belong to a project.",
                                         action_text="Show welcome page",
                                         on_action=lambda: self.state.navigate("start")))
        self._pages.addWidget(self._build_main())
        self._build_actions()
        guard_wheel(self)
        # files dropped anywhere on the page (the inspector, the wing editor) are imported;
        # the viewports have their own drop highlight
        self.setAcceptDrops(True)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.release_3d)

        self._physics_sel: tuple | None = None
        state.projectChanged.connect(self._on_project_changed)
        state.designsChanged.connect(self._on_designs_changed)
        state.currentDesignChanged.connect(self._on_current_design_changed)
        state.currentSimulationChanged.connect(lambda _sid: self.update_physics_layer())
        state.simulationsChanged.connect(self.update_physics_layer)
        self._on_project_changed()

    def release_3d(self) -> None:
        """Free the OpenGL contexts of the 3D views now (window close), not at interpreter exit."""
        for v in (self.view3d, self.wing_editor.view3d):
            with contextlib.suppress(Exception):
                v.scene.close()

    # ===================================================================
    # construction
    # ===================================================================
    def _build_main(self) -> QWidget:
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setObjectName("geometry_split")
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        split.addWidget(self._build_viewport())
        split.addWidget(self._build_inspector())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1000, INSPECTOR_WIDTH])
        self._split = split
        return split

    def _build_viewport(self) -> QWidget:
        """The centre: the 2D canvas, or the 3D workspace for 3D designs."""
        self._viewport_stack = QStackedWidget()
        self._viewport_stack.setObjectName("geometry_viewports")
        self._viewport_stack.addWidget(self._build_viewport2d())
        self._viewport_stack.addWidget(self._build_viewport3d())
        return self._viewport_stack

    def _build_viewport3d(self) -> QWidget:
        """Tool strip, the mesh view or the wing editor, and a status strip."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = ToolStrip()
        bar.setObjectName("geometry_tools3d")
        self.import_button3d = bar.add_button("import", "Import", "Import geometry from a file (Ctrl+I)")
        self.import_button3d.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.new3d_button = bar.add_button(None, "New", "New 3D design: a finite wing or a basic shape")
        self.new3d_button.setIcon(svg_icon("cube"))
        self.new3d_button.setIconSize(QSize(16, 16))
        self.new3d_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.new3d_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bar.add_separator()
        self.undo_button3d = bar.add_button(None, "", "Undo (Ctrl+Z)", on_click=self._undo_group.undo)
        self.redo_button3d = bar.add_button(None, "", "Redo (Ctrl+Y)", on_click=self._undo_group.redo)
        self.undo_button3d.setIcon(svg_icon("undo"))
        self.redo_button3d.setIcon(svg_icon("redo"))
        for b in (self.undo_button3d, self.redo_button3d):
            b.setIconSize(QSize(16, 16))
            b.setFixedWidth(26)
            b.setEnabled(False)
        self._undo_group.canUndoChanged.connect(self.undo_button3d.setEnabled)
        self._undo_group.canRedoChanged.connect(self.redo_button3d.setEnabled)
        self._undo_group.undoTextChanged.connect(
            lambda t: self.undo_button3d.setToolTip(f"Undo {t} (Ctrl+Z)" if t else "Undo (Ctrl+Z)"))
        self._undo_group.redoTextChanged.connect(
            lambda t: self.redo_button3d.setToolTip(f"Redo {t} (Ctrl+Y)" if t else "Redo (Ctrl+Y)"))
        bar.add_separator()
        self.view_buttons: dict[str, QToolButton] = {}
        for key, text, tip in (("iso", "Iso", "Isometric view from upstream, left and above"),
                               ("front", "Front", "Look downstream at the frontal area (along +x)"),
                               ("side", "Side", "Side view, flow from left to right"),
                               ("top", "Top", "Plan view from above"),
                               ("back", "Back", "Look upstream at the base and the wake side (along −x)")):
            self.view_buttons[key] = bar.add_button(None, text, tip, on_click=lambda k=key: self._set_view3d(k))
        self.fit_button3d = bar.add_button("fullscreen", "Fit", "Frame the model (F, or double-click)",
                                           on_click=lambda: self._active_view3d().fit())
        bar.add_stretch()
        self.edges_toggle = bar.add_button(None, "Edges", "Show the triangle edges of the mesh", checkable=True)
        self.edges_toggle.setIcon(svg_icon("mesh", active="accent"))
        self.dims_toggle3d = bar.add_button(None, "Dimensions", "Show the bounding-box dimensions and the "
                                                                "moment reference point", checkable=True)
        self.dims_toggle3d.setIcon(svg_icon("dimension", active="accent"))
        for b in (self.edges_toggle, self.dims_toggle3d):
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setIconSize(QSize(16, 16))
        self.dims_toggle3d.setChecked(True)
        self.physics_toggle3d = bar.add_button("layers", "Physics", PHYSICS_LAYER_TIP, checkable=True)
        self.physics_toggle3d.setObjectName("geometry_physics3d")
        self.physics_toggle3d.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.physics_toggle3d.setChecked(True)
        lay.addWidget(bar)
        self._content3d = QStackedWidget()
        self.view3d = GeometryView3D()
        self.view3d.setObjectName("geometry_view3d")
        self.wing_editor = WingEditor()
        self.wing_editor.setObjectName("wing_editor")
        self._content3d.addWidget(self.view3d)
        self._content3d.addWidget(self.wing_editor)
        lay.addWidget(self._content3d, 1)
        lay.addWidget(Divider())
        status = QFrame()
        status.setProperty("panel", "true")
        status.setFixedHeight(24)
        sl = QHBoxLayout(status)
        sl.setContentsMargins(8, 0, 8, 0)
        sl.setSpacing(12)
        self.summary_label3d = label("", "secondary")
        self.summary_label3d.setObjectName("summary3d")
        self.refs_label3d = label("", "muted")
        self.refs_label3d.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        sl.addWidget(self.summary_label3d, 1)
        sl.addWidget(self.refs_label3d)
        lay.addWidget(status)
        self.edges_toggle.toggled.connect(self._set_edges3d)
        self.dims_toggle3d.toggled.connect(self._set_dims3d)
        self.view3d.filesDropped.connect(self.import_files)
        self.wing_editor.specEdited.connect(self._wing_spec_edited)
        self.physics_toggle3d.toggled.connect(self.set_physics_layer)
        self._connect_physics_view(self.view3d)
        return w

    def _build_viewport2d(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = ToolStrip()
        bar.setObjectName("geometry_tools")
        self.import_button = bar.add_button("import", "Import", "Import geometry from a file (Ctrl+I)")
        self.import_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.gallery_button = bar.add_button("layers", "Templates…", "New design from a parametric template",
                                             on_click=self.open_template_gallery)
        bar.add_separator()
        self.undo_button = bar.add_button(None, "", "Undo (Ctrl+Z)", on_click=self._undo_group.undo)
        self.redo_button = bar.add_button(None, "", "Redo (Ctrl+Y)", on_click=self._undo_group.redo)
        self.undo_button.setIcon(svg_icon("undo"))
        self.redo_button.setIcon(svg_icon("redo"))
        for b in (self.undo_button, self.redo_button):
            b.setIconSize(QSize(16, 16))
            b.setFixedWidth(26)
            b.setEnabled(False)
        self._undo_group.canUndoChanged.connect(self.undo_button.setEnabled)
        self._undo_group.canRedoChanged.connect(self.redo_button.setEnabled)
        self._undo_group.undoTextChanged.connect(
            lambda t: self.undo_button.setToolTip(f"Undo {t} (Ctrl+Z)" if t else "Undo (Ctrl+Z)"))
        self._undo_group.redoTextChanged.connect(
            lambda t: self.redo_button.setToolTip(f"Redo {t} (Ctrl+Y)" if t else "Redo (Ctrl+Y)"))
        bar.add_separator()
        self.fit_button = bar.add_button("fullscreen", "Fit", "Fit the design to the view (F, or double-click)")
        bar.add_stretch()
        self.vertices_toggle = bar.add_button(None, "Points", "Show outline vertices; the first vertex is "
                                                              "highlighted", checkable=True)
        self.vertices_toggle.setIcon(svg_icon("points", active="accent"))
        self.dims_toggle = bar.add_button(None, "Dimensions", "Show dimensions", checkable=True)
        self.dims_toggle.setIcon(svg_icon("dimension", active="accent"))
        self.grid_toggle = bar.add_button("grid", "Grid", "Show the grid", checkable=True)
        for b in (self.vertices_toggle, self.dims_toggle, self.grid_toggle):
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setIconSize(QSize(16, 16))
        self.dims_toggle.setChecked(True)
        self.grid_toggle.setChecked(True)
        self.physics_toggle = bar.add_button("layers", "Physics", PHYSICS_LAYER_TIP, checkable=True)
        self.physics_toggle.setObjectName("geometry_physics")
        self.physics_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.physics_toggle.setIconSize(QSize(16, 16))
        self.physics_toggle.setChecked(True)
        lay.addWidget(bar)

        self.canvas = GeometryCanvas()
        self.canvas.setObjectName("geometry_canvas")
        self.canvas.setAccessibleName("Geometry view")
        lay.addWidget(self.canvas, 1)

        lay.addWidget(Divider())
        status = QFrame()
        status.setProperty("panel", "true")
        status.setFixedHeight(24)
        sl = QHBoxLayout(status)
        sl.setContentsMargins(8, 0, 8, 0)
        sl.setSpacing(12)
        self.summary_label = label("", "secondary")
        self.hover_label = label("", "muted")
        self.hover_label.setMinimumWidth(220)
        self.hover_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        sl.addWidget(self.summary_label, 1)
        sl.addWidget(self.hover_label)
        lay.addWidget(status)

        self.fit_button.clicked.connect(lambda: self.canvas.fit_view())
        self.vertices_toggle.toggled.connect(self.canvas.set_show_vertices)
        self.dims_toggle.toggled.connect(self.canvas.set_show_dimensions)
        self.grid_toggle.toggled.connect(self.canvas.set_show_grid)
        self.canvas.hoverChanged.connect(self._hover)
        self.canvas.filesDropped.connect(self.import_files)
        self.physics_toggle.toggled.connect(self.set_physics_layer)
        self._connect_physics_view(self.canvas)
        return w

    def _build_inspector(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("panel", "true")
        panel.setObjectName("geometry_inspector")
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(560)
        panel.resize(INSPECTOR_WIDTH, panel.height())
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QWidget()
        head.setFixedHeight(30)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(12, 0, 12, 0)
        hl.setSpacing(8)
        self.title_label = label("Properties", "h3")
        self.kind_label = label("", "muted")
        hl.addWidget(self.title_label, 1)
        hl.addWidget(self.kind_label)
        lay.addWidget(head)
        lay.addWidget(Divider())

        self._inspector_stack = QStackedWidget()
        self._inspector_stack.addWidget(self._build_add_actions())
        self.grid = PropertyGrid(label_width=LABEL_W)
        self.grid.setObjectName("geometry_properties")
        self._build_source_group()
        self._build_transform_group()
        self._build_cleanup_group()
        self._build_refs_group()
        self._build_diag_group()
        self.grid.add_stretch()
        self._inspector_stack.addWidget(self.grid.scrollable())
        self.mesh_inspector = MeshInspector()
        self.mesh_inspector.setObjectName("mesh_inspector")
        self.mesh_inspector.editRequested.connect(self.apply_edit3d)
        self.mesh_inspector.refEdited.connect(self._ref_edited3)
        self.mesh_inspector.autoRefToggled.connect(self._auto_ref3)
        self._inspector_stack.addWidget(self.mesh_inspector)
        self.wing_inspector = WingInspector()
        self.wing_inspector.setObjectName("wing_inspector")
        self.wing_inspector.specEdited.connect(self._wing_spec_edited)
        self.wing_inspector.refEdited.connect(self._wing_ref_edited)
        self.wing_inspector.autoRefToggled.connect(self._wing_auto_ref)
        self._inspector_stack.addWidget(self.wing_inspector)
        lay.addWidget(self._inspector_stack, 1)

        lay.addWidget(Divider())
        foot = QWidget()
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(12, 8, 12, 8)
        fl.addStretch(1)
        self.next_button = QPushButton("Set up simulation")
        self.next_button.setObjectName("next_setup")
        self.next_button.setProperty("variant", "primary")
        self.next_button.setToolTip("Go to the Setup step (Ctrl+2)")
        self.next_button.clicked.connect(self._go_setup)
        fl.addWidget(self.next_button)
        lay.addWidget(foot)
        return panel

    def _build_add_actions(self) -> QWidget:
        """Inspector content when the project has no current design: a plain action list."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(2)
        lay.addWidget(label("No design selected", "secondary"))
        lay.addSpacing(6)
        for text, icon_name, slot in (("New from template…", "layers", self.open_template_gallery),
                                      ("New NACA airfoil…", "airfoil", self.new_naca),
                                      ("New finite wing", "wing", self.new_wing),
                                      ("Import file…", "import", self.import_dialog),
                                      ("Import 3D mesh…", "cube", lambda: self.import_dialog("mesh"))):
            b = QPushButton(text)
            b.setProperty("variant", "link")
            b.setIcon(svg_icon(icon_name, "accent"))
            b.setIconSize(QSize(16, 16))
            b.clicked.connect(lambda _=False, s=slot: s())
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addSpacing(8)
        lay.addWidget(label("Files can also be dropped on the viewport: outlines (.dat, SVG, DXF, CSV, "
                            "images) and 3D meshes (STL, OBJ, PLY).", "muted", wrap=True))
        lay.addStretch(1)
        return w

    # -- inspector groups ---------------------------------------------------
    def _build_source_group(self) -> None:
        g = self.grid.add_group("Source", key="source")
        self.source_group = g
        # wraps: "Starter case · Natural-convection cavity (2D, validation)" on one unbreakable line set
        # the inspector's minimum width and pushed every editor past the window's edge
        self.source_type = label("", None, wrap=True)
        self.source_type_row = g.add_row("Type", self.source_type)
        # template parameters
        self.template_box = FormGrid(LABEL_W)
        self.template_name_label = label("", None)
        self.template_box.add_row("Template", self.template_name_label)
        self.param_form = ParamForm(label_width=LABEL_W)
        self.param_form.setObjectName("param_form")
        self.param_form.paramChanged.connect(self._param_edited)
        self.template_box.add_widget(self.param_form)
        self.reset_params_button = QPushButton("Reset to defaults")
        self.reset_params_button.setToolTip("Set every parameter back to the template default "
                                            "(one undo step)")
        self.reset_params_button.clicked.connect(self._reset_params)
        self.template_box.add_widget(hbox(self.reset_params_button, stretch=None), indent=True)
        g.add_widget(self.template_box)
        # NACA
        self.naca_box = FormGrid(LABEL_W)
        self.naca_code = QLineEdit()
        self.naca_code.setObjectName("naca_code_field")
        self.naca_code.setMaxLength(9)
        self.naca_code.setPlaceholderText("2412")
        self.naca_code.editingFinished.connect(self._naca_code_edited)
        self.naca_code.textChanged.connect(self._naca_code_typing)
        self.naca_status = StatusLine()
        self.naca_chord = QuantitySpinBox("length", display_unit("length"), minimum=1e-4, maximum=1e3,
                                          step=0.01)
        self.naca_chord.siValueChanged.connect(lambda v: self._naca_changed({"chord": v}, "naca_chord"))
        self.naca_points = QSpinBox()
        self.naca_points.setRange(60, 600)
        self.naca_points.setSingleStep(20)
        self.naca_points.setKeyboardTracking(False)
        self.naca_points.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.naca_points.valueChanged.connect(lambda v: self._naca_changed({"n": int(v)}, "naca_n"))
        self.naca_box.add_row("NACA code", self.naca_code, "4 or 5 digits; press Enter to apply.")
        self.naca_box.add_widget(self.naca_status, indent=True)
        self.naca_box.add_row("Chord", self.naca_chord)
        self.naca_box.add_row("Outline points", self.naca_points, "Points along the outline")
        g.add_widget(self.naca_box)
        # imported / other
        self.import_box = FormGrid(LABEL_W)
        self.import_file = label("", None)
        self.import_file.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.import_file.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.import_format = label("", None)
        self.import_threshold = label("", None)
        self.import_edits = label("", None)
        self._file_label = self.import_box.add_row("File", self.import_file)
        self._format_label = self.import_box.add_row("Format", self.import_format)
        self._threshold_label = self.import_box.add_row("Threshold", self.import_threshold)
        self.import_box.add_row("Edits", self.import_edits,
                                "Imported outlines are fixed; Transform and Cleanup edits are recorded.")
        g.add_widget(self.import_box)
        self.import_info = self.import_box

    def _build_transform_group(self) -> None:
        g = self.grid.add_group("Transform", key="transform")
        self.transform_group = g
        lu = display_unit("length")
        self.scale_length = QuantitySpinBox("length", lu, minimum=1e-4, maximum=1e4, step=0.01)
        self.scale_length.setObjectName("scale_length")
        self.scale_length.siValueChanged.connect(lambda _v: self.apply_scale())
        self._scale_row = g.add_row("Length", self.scale_length,
                                    tooltip="Type a new size and press Enter: the shape is scaled "
                                            "uniformly. Templates and NACA sections set their size "
                                            "in Source instead.")
        self.rotate_deg = QuantitySpinBox("angle", "deg", minimum=-180, maximum=180, step=0.5)
        rot = small_button("Rotate", "Rotate about the bounding-box centre (counter-clockwise positive)",
                           ROW_BUTTON_W)
        rot.clicked.connect(lambda: self.apply_edit({"op": "rotate", "deg": self.rotate_deg.si_value()},
                                                    f"Rotate {self.rotate_deg.text()}"))
        g.add_row("Rotate by", hbox(self.rotate_deg, rot))
        self.move_dx = QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.01)
        self.move_dy = QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.01)
        move = small_button("Move", "Translate by Δx, Δy", ROW_BUTTON_W)
        move.clicked.connect(lambda: self.apply_edit(
            {"op": "translate", "dx": self.move_dx.si_value(), "dy": self.move_dy.si_value()}, "Move"))
        g.add_row("Move Δx", hbox(self.move_dx, _spacer(ROW_BUTTON_W)))
        g.add_row("Move Δy", hbox(self.move_dy, move))
        self.lift_height = QuantitySpinBox("length", lu, minimum=0.0, maximum=100.0, step=0.005)
        lift = small_button("Place", "Move the shape vertically so its lowest point is at this height",
                            ROW_BUTTON_W)
        lift.clicked.connect(lambda: self.apply_edit(
            {"op": "lift", "height": self.lift_height.si_value()}, "Place above road"))
        self._lift_row = g.add_row("Height above road", hbox(self.lift_height, lift))
        mx = small_button("Front/back", "Mirror left-right: the nose ends up at the other end")
        my = small_button("Top/bottom", "Mirror top-bottom")
        mx.clicked.connect(lambda: self.apply_edit({"op": "mirror", "axis": "x"}, "Mirror front/back"))
        my.clicked.connect(lambda: self.apply_edit({"op": "mirror", "axis": "y"}, "Mirror top/bottom"))
        g.add_row("Mirror", hbox(mx, my, stretch=None))

    def _build_cleanup_group(self) -> None:
        g = self.grid.add_group("Cleanup", expanded=False, key="cleanup")
        self.cleanup_group = g
        self.resample_n = QSpinBox()
        self.resample_n.setRange(24, 4000)
        self.resample_n.setSingleStep(20)
        self.resample_n.setValue(300)
        self.resample_n.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        rs = small_button("Apply", "Redistribute the outline points evenly (sharp corners are kept)",
                          ROW_BUTTON_W)
        rs.clicked.connect(lambda: self.apply_edit({"op": "resample", "n": self.resample_n.value()},
                                                   f"Resample to {self.resample_n.value()} points"))
        g.add_row("Resample to", hbox(self.resample_n, rs), tooltip="Points per body")
        self.smooth_iter = QSpinBox()
        self.smooth_iter.setRange(1, 50)
        self.smooth_iter.setValue(3)
        self.smooth_iter.setSuffix(" passes")
        self.smooth_iter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        sm = small_button("Apply", "Gentle smoothing of noisy imported outlines", ROW_BUTTON_W)
        sm.clicked.connect(lambda: self.apply_edit({"op": "smooth", "iterations": self.smooth_iter.value()},
                                                   "Smooth"))
        g.add_row("Smooth", hbox(self.smooth_iter, sm))

    def _build_refs_group(self) -> None:
        g = self.grid.add_group("References", key="references")
        self.refs_group = g
        self.kind_combo = QComboBox()
        self.kind_combo.setObjectName("kind_combo")
        for k in KINDS_2D:
            self.kind_combo.addItem(KIND_LABELS[k], k)
        self.kind_combo.currentIndexChanged.connect(self._kind_edited)
        g.add_row("Kind", self.kind_combo, tooltip="Sets the reference conventions (chord for airfoils; "
                                                   "length and frontal height for vehicles)")
        lu = display_unit("length")
        self.length_ref = QuantitySpinBox("length", lu, minimum=1e-5, maximum=1e4, step=0.001)
        self.force_ref = QuantitySpinBox("length", lu, minimum=1e-5, maximum=1e4, step=0.001)
        self.mref_x = QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.001)
        self.mref_y = QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.001)
        self.length_ref.setObjectName("length_ref")
        self.force_ref.setObjectName("force_ref")
        self.length_ref.siValueChanged.connect(lambda v: self._ref_edited("length_ref", v))
        self.force_ref.siValueChanged.connect(lambda v: self._ref_edited("force_ref", v))
        self.mref_x.siValueChanged.connect(
            lambda v: self._ref_edited("moment_ref", (v, self.mref_y.si_value())))
        self.mref_y.siValueChanged.connect(
            lambda v: self._ref_edited("moment_ref", (self.mref_x.si_value(), v)))
        self._auto_buttons: dict[str, QToolButton] = {}
        self._ref_rows = {}
        for f, w in (("length_ref", self.length_ref), ("force_ref", self.force_ref),
                     ("moment_ref", self.mref_x)):
            b = _toggle("Auto", "Follow the shape automatically; typing a value fixes it")
            b.clicked.connect(lambda on, f=f: self._auto_ref_clicked(f, on))
            self._auto_buttons[f] = b
            self._ref_rows[f] = g.add_row(f, hbox(w, b), key=f)
        g.add_row("Moment point y", hbox(self.mref_y, _spacer(ROW_BUTTON_W)))

    def _build_diag_group(self) -> None:
        g = self.grid.add_group("Diagnostics", key="diagnostics")
        self.diag_group = g
        box = QWidget()
        self.diag_list = QVBoxLayout(box)
        self.diag_list.setContentsMargins(0, 2, 0, 0)
        self.diag_list.setSpacing(6)
        self.diag_badge = StatusLine()
        self.diag_badge.setObjectName("diag_summary")
        self.diag_list.addWidget(self.diag_badge)
        self._diag_items = QVBoxLayout()
        self._diag_items.setSpacing(6)
        self.diag_list.addLayout(self._diag_items)
        g.add_widget(box)

    def _build_actions(self) -> None:
        menu = QMenu(self)
        entries = [
            ("NACA airfoil…", "airfoil", self.new_naca),
            ("Airfoil coordinates (.dat)…", None, lambda: self.import_dialog("dat")),
            ("SVG drawing…", None, lambda: self.import_dialog("svg")),
            ("DXF drawing…", None, lambda: self.import_dialog("dxf")),
            ("Image silhouette…", None, lambda: self.import_dialog("image")),
            ("CSV points…", None, lambda: self.import_dialog("csv")),
        ]
        self.import_actions: dict[str, QAction] = {}
        for text, icon_name, slot in entries:
            a = menu.addAction(text)
            if icon_name:
                a.setIcon(icon(icon_name))
            a.triggered.connect(lambda _=False, s=slot: s())
            self.import_actions[text] = a
        menu.addSeparator()
        a = menu.addAction(svg_icon("cube"), "3D mesh (STL, OBJ, PLY)…")
        a.triggered.connect(lambda _=False: self.import_dialog("mesh"))
        self.import_actions["3D mesh (STL, OBJ, PLY)…"] = a
        menu.addSeparator()
        a = menu.addAction(icon("layers"), "From template…")
        a.triggered.connect(lambda _=False: self.open_template_gallery())
        self.import_actions["From template…"] = a
        self.import_button.setMenu(menu)
        self.import_button3d.setMenu(menu)
        new3d = QMenu(self)
        a = new3d.addAction(svg_icon("wing"), "Finite wing")
        a.setToolTip("A planform of sections for the wing analysis (VLM / lifting line) and 3D CFD")
        a.triggered.connect(lambda _=False: self.new_wing())
        self.new3d_actions: dict[str, QAction] = {"wing": a}
        new3d.addSeparator()
        for key, (text, _kw) in PRIMITIVES.items():
            a = new3d.addAction(text)
            a.triggered.connect(lambda _=False, k=key: self.new_primitive(k))
            self.new3d_actions[key] = a
        self.new3d_button.setMenu(new3d)
        # the 2D strip keeps its look; its Import menu now also offers the 3D entries
        menu.addSeparator()
        sub = menu.addMenu(svg_icon("cube"), "New 3D design")
        for a in new3d.actions():
            sub.addAction(a)

        def shortcut(seq, slot) -> QAction:
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            act.triggered.connect(slot)
            self.addAction(act)
            return act

        # Ctrl+I (import) and F2 (rename) are window actions; only undo lives here
        self.undo_action = shortcut(QKeySequence.StandardKey.Undo, self._undo_group.undo)
        self.redo_action = shortcut(QKeySequence.StandardKey.Redo, self._undo_group.redo)
        if QKeySequence(QKeySequence.StandardKey.Redo) != QKeySequence("Ctrl+Y"):
            shortcut("Ctrl+Y", self._undo_group.redo)

    # ===================================================================
    # page protocol
    # ===================================================================
    def on_shown(self) -> None:
        self._load_design(force_rebuild=False)

    def hideEvent(self, event) -> None:
        # apply a parameter edit still waiting for its debounce before the user leaves
        self._flush_regeneration()
        super().hideEvent(event)

    def handle_action(self, action: str) -> None:
        """Navigation actions: "import", "import3d", "naca", "wing", "templates", "import:<path>"."""
        if action == "import":
            self.import_dialog()
        elif action in ("import3d", "mesh"):
            self.import_dialog("mesh")
        elif action == "naca":
            self.new_naca()
        elif action == "wing":
            self.new_wing()
        elif action in ("templates", "gallery"):
            self.open_template_gallery()
        elif action.startswith("import:"):
            self.import_files([action.split(":", 1)[1]])

    # -- drops anywhere on the page ----------------------------------------------
    @staticmethod
    def _local_files(event) -> list[str]:
        md = event.mimeData()
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile()] if md.hasUrls() else []

    def dragEnterEvent(self, event) -> None:
        if self.state.project is not None and self._local_files(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = self._local_files(event)
        if not paths or self.state.project is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.import_files(paths)

    # -- modal funnels (patched by tests) ------------------------------------
    def _exec_dialog(self, dlg: QDialog) -> bool:
        return dlg.exec() == QDialog.DialogCode.Accepted

    def _ask_open_file(self, caption: str, file_filter: str) -> str:
        start = ""
        with contextlib.suppress(Exception):        # preferences are optional
            from .. import settings
            start = settings.get().last_dir
        path, _ = QFileDialog.getOpenFileName(self, caption, str(start or ""), file_filter)
        if path:
            with contextlib.suppress(Exception):
                from .. import settings
                settings.get().last_dir = str(Path(path).parent)
        return path

    def _ask_text(self, title: str, prompt: str, text: str) -> str | None:
        value, ok = QInputDialog.getText(self, title, prompt, QLineEdit.EchoMode.Normal, text)
        return value if ok else None

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _error(self, title: str, text: str) -> None:
        self.state.log(f"{title}: {text}", "error", "Geometry")
        QMessageBox.warning(self, title, text)

    # ===================================================================
    # state -> widgets
    # ===================================================================
    @property
    def design(self) -> Design | None:
        return self.state.design

    def _on_project_changed(self) -> None:
        self._regen_timer.stop()
        self._pending_params = None
        for st in self._stacks.values():
            self._undo_group.removeStack(st)
            st.deleteLater()
        self._stacks.clear()
        self._pages.setCurrentIndex(0 if self.state.project is None else 1)
        if self.state.project is not None and not self.state.design_id and self.state.project.designs:
            self.state.select_design(next(iter(self.state.project.designs)))
        self._load_design(force_rebuild=True)

    def _on_designs_changed(self) -> None:
        if self.state.project is None:
            return
        live = set(self.state.project.designs)
        for did in [d for d in self._stacks if d not in live]:
            st = self._stacks.pop(did)
            self._undo_group.removeStack(st)
            st.deleteLater()
        self._load_design(force_rebuild=False)

    def _on_current_design_changed(self, _design_id: str) -> None:
        self._flush_regeneration()
        self._load_design(force_rebuild=True)

    def _stack_for(self, design_id: str) -> QUndoStack:
        st = self._stacks.get(design_id)
        if st is None:
            st = QUndoStack(self)
            p = self.state.project
            d = p.designs.get(design_id) if p is not None else None
            meshed = d is not None and d.dimension == 3 and bool(d.meshes)
            st.setUndoLimit(UNDO_LIMIT_3D if meshed else UNDO_LIMIT_2D)
            self._stacks[design_id] = st
            self._undo_group.addStack(st)
        return st

    # -- the physics layer: the selected simulation's placed items over the geometry -------------
    def _connect_physics_view(self, view) -> None:
        view.physicsSelected.connect(self._physics_selected)
        view.physicsMoved.connect(self._physics_moved)
        view.physicsDropped.connect(self._physics_dropped)

    def physics_simulation(self):
        """The CFD simulation whose placed items the Physics layer shows: the selected one, when it
        belongs to the design on screen."""
        sim, d = self.state.simulation, self.design
        if sim is None or d is None or sim.design_id != d.id or sim.analysis != "cfd":
            return None
        return sim

    def set_physics_layer(self, on: bool) -> None:
        """The Physics toggles (2D and 3D tool strips stay in step)."""
        for b in (self.physics_toggle, self.physics_toggle3d):
            if b.isChecked() != bool(on):
                b.blockSignals(True)
                b.setChecked(bool(on))
                b.blockSignals(False)
        self.update_physics_layer()

    def update_physics_layer(self) -> None:
        """Hand the selected simulation's items (design frame) to the viewport on screen."""
        d, sim = self.design, self.physics_simulation()
        on = self.physics_toggle.isChecked()
        specs = phys.design_frame_specs(d, sim, selected=self._physics_sel) if (on and sim is not None) else []
        three = d is not None and d.dimension == 3
        self.canvas.set_show_physics(on)
        self.view3d.set_show_physics(on)
        self.canvas.set_physics([] if three else specs)
        self.view3d.set_physics(specs if three else [])
        if d is not None and not three and not d.bodies:
            # a design without bodies (a natural-convection cavity): its closed box is the case
            self.canvas.empty_text = ("No bodies: the case is the closed box of its simulation. Add a shape from "
                                      "Templates… or import one." if specs else
                                      "No bodies. Add a shape from Templates… or import one.")

    def _physics_selected(self, key) -> None:
        self._physics_sel = tuple(key) if key is not None else None
        self.update_physics_layer()

    def _physics_moved(self, key, handle: str, point) -> None:
        """A drag in the design frame: back to the wind frame the items are stored against, then moved."""
        d, sim = self.design, self.physics_simulation()
        if d is None or sim is None:
            return
        p_wind = phys.to_wind(point, d, sim, "body")[0] if key and key[0] != "domain" else point
        if phys.apply_drag(d, sim, tuple(key), handle, p_wind):
            self._physics_sel = tuple(key)
            self.update_physics_layer()

    def _physics_dropped(self, key, handle: str, point) -> None:
        self._physics_moved(key, handle, point)
        sim = self.physics_simulation()
        if sim is not None and self.state.project is not None and sim.id in self.state.project.simulations:
            self.state.update_simulation(sim)

    def _sim_count(self, design_id: str) -> int:
        p = self.state.project
        return 0 if p is None else sum(1 for s in p.simulations.values() if s.design_id == design_id)

    def _load_design(self, force_rebuild: bool) -> None:
        """Fill canvas and inspector from the current design.

        ``force_rebuild`` re-creates the source form; otherwise it is only
        rebuilt when the recipe's structure changed, so a live parameter edit
        never yanks focus out of the spin box being typed in.
        """
        d = self.design
        self._undo_group.setActiveStack(self._stack_for(d.id) if d is not None else None)
        has = d is not None
        self.next_button.setEnabled(has)
        if d is not None and d.dimension == 3:
            self._load_design3d(d)
            return
        self._viewport_stack.setCurrentIndex(0)
        self._inspector_stack.setCurrentIndex(1 if has else 0)
        self.canvas.set_design(d)
        self.update_physics_layer()
        if d is None:
            self.summary_label.setText("")
            self.title_label.setText("Properties")
            self.kind_label.setText("")
            return
        self.summary_label.setText(design_summary(d))
        self.title_label.setText(d.name)
        self.title_label.setToolTip(d.name)
        self.kind_label.setText(_KIND_NOUN.get(d.kind, d.kind))
        # a design without bodies (a cavity) has nothing to transform or clean up
        self.transform_group.setVisible(bool(d.bodies))
        self.cleanup_group.setVisible(bool(d.bodies))
        self._loading = True
        try:
            self._load_source(d, force_rebuild)
            self._load_transform(d)
            self._load_refs(d)
            self._load_diagnostics(d)
        finally:
            self._loading = False

    def _load_source(self, d: Design, force_rebuild: bool) -> None:
        src = d.source or {}
        typ = str(src.get("type", ""))
        tmpl = template_source(d)
        sig = (d.id, typ, tmpl[0] if tmpl else None)
        rebuild = force_rebuild or sig != self._source_sig
        self._source_sig = sig
        self.template_box.setVisible(tmpl is not None)
        self.naca_box.setVisible(typ == "naca")
        self.import_box.setVisible(tmpl is None and typ != "naca")
        self.source_type.setText(source_type_text(d))
        if tmpl is not None:
            name, params = tmpl
            info = self._template_info(name)
            if rebuild:
                specs = list(getattr(info, "params", []) or [])
                self.param_form.set_specs(specs)
                self._attach_naca_validation(specs)
            self.template_name_label.setText(template_label(info) if info else name)
            self.template_name_label.setToolTip(str(getattr(info, "description", "") or ""))
            self.param_form.set_values(params)
        elif typ == "naca":
            if not self.naca_code.hasFocus() or rebuild:
                self.naca_code.setText(str(src.get("code", "")))
            self.naca_chord.set_si_value(float(src.get("chord", d.length_ref)))
            self.naca_points.blockSignals(True)
            self.naca_points.setValue(int(src.get("n", 160)))
            self.naca_points.blockSignals(False)
            self._show_naca_status(None)
        else:
            # a starter case (Setup ▸ New case from template) has no file: its rows would only say "—";
            # the case's own title and description go on the Type line instead
            starter = typ == "physics_template"
            for w in (self._file_label, self.import_file, self._format_label, self.import_format):
                w.setVisible(not starter)
            if starter:
                info = starter_case_info(d)
                if info is not None:
                    self.source_type.setText(f"Starter case · {info.title}")
                    self.source_type.setToolTip(info.description)
            else:
                self.source_type.setToolTip("")
            fmt = str(src.get("format", ""))
            self.import_file.setText(str(src.get("file") or "—"))
            self.import_file.setToolTip(str(src.get("path") or src.get("file") or ""))
            self.import_format.setText(FORMAT_NAMES.get(fmt, fmt) if fmt else
                                       ("Coordinates" if typ == "airfoil" else "—"))
            image = fmt == "image"
            self._threshold_label.setVisible(image)
            self.import_threshold.setVisible(image)
            if image:
                thr = src.get("threshold")
                self.import_threshold.setText(("Automatic" if thr is None else f"{float(thr):.0f} / 255")
                                              + (", inverted" if src.get("invert") else ""))
            n_edits = len(src.get("edits", []))
            self.import_edits.setText(f"{n_edits} since import" if typ == "import" else f"{n_edits}")

    def _template_info(self, name: str):
        try:
            for info in templates_module().list_templates():
                if info.name == name:
                    return info
        except RuntimeError:
            pass
        return None

    def _load_transform(self, d: Design) -> None:
        if not d.bodies:
            return
        if d.kind == "airfoil":
            le, te = airfoil_chord(d)
            self._scale_row.set_label("Chord")
            self.scale_length.set_si_value(float(np.hypot(*(te - le))))
        else:
            x0, x1, _, _ = d.bounds()
            self._scale_row.set_label("Length")
            self.scale_length.set_si_value(x1 - x0)
        # One editor per quantity: a NACA section has its chord in Source, and a
        # template owns its size and its height above the road (Length, Chord,
        # Diameter, Ground clearance, Ride height parameters). A second editable
        # "Length" here showed the same number with a different meaning
        # (rescale vs regenerate) and the two drifted apart after either edit.
        recipe = (d.source or {}).get("type") == "naca" or template_source(d) is not None
        self._scale_row.set_visible(not recipe)
        vehicle = d.kind == "vehicle"
        self._lift_row.set_visible(vehicle and not recipe)
        if vehicle:
            self.lift_height.set_si_value(max(d.bounds()[2], 0.0))

    def _load_refs(self, d: Design) -> None:
        self.kind_combo.blockSignals(True)
        self.kind_combo.setCurrentIndex(max(self.kind_combo.findData(d.kind), 0))
        self.kind_combo.blockSignals(False)
        self.length_ref.set_si_value(d.length_ref)
        self.force_ref.set_si_value(d.force_ref)
        self.mref_x.set_si_value(d.moment_ref[0])
        self.mref_y.set_si_value(d.moment_ref[1])
        manual = set(d.source.get("manual_refs", []))
        for f, b in self._auto_buttons.items():
            b.setChecked(f not in manual)
        kind = d.kind if d.kind in ("airfoil", "vehicle", "generic") else "generic"
        names = {"airfoil": ("Chord (Re)", "Chord (coeff.)", "Moment point x"),
                 "vehicle": ("Length (Re)", "Frontal height", "Moment point x"),
                 "generic": ("Reference length", "Frontal height", "Moment point x")}[kind]
        tips = {"airfoil": ("Re = ρ U c / μ", "Coefficients are per unit chord", "Quarter chord by default"),
                "vehicle": ("Re uses the overall length", ("Coefficients use the frontal height "
                            "(2D frontal area); drag area CdA = Cd × height"), "Mid-length, on the road"),
                "generic": ("Re uses this length", "Coefficients use the frontal height",
                            "Bounding-box centre by default")}[kind]
        for f, t, tip in zip(REF_FIELDS, names, tips, strict=True):
            row = self._ref_rows[f]
            row.set_label(t)
            row.label.setToolTip(tip)

    def _load_diagnostics(self, d: Design) -> None:
        clear_layout(self._diag_items)
        msgs = check_messages(d)
        errors = sum(1 for lv, _ in msgs if lv == "error")
        warns = sum(1 for lv, _ in msgs if lv == "warning")
        if errors:
            self.diag_badge.set("error", f"{errors} error{'s' if errors != 1 else ''}"
                                + (f", {warns} warning{'s' if warns != 1 else ''}" if warns else ""))
        elif warns:
            self.diag_badge.set("warning", f"{warns} warning{'s' if warns != 1 else ''}")
        else:
            self.diag_badge.set("ok", "No problems found")
            self.diag_badge.setToolTip("Closed, counter-clockwise outlines that do not cross themselves")
        for lv, text in msgs:
            line = StatusLine(text, lv if lv in ("error", "warning") else "info")
            line.setProperty("diag_level", lv)
            self._diag_items.addWidget(line)
        if errors or warns:
            self.diag_group.set_expanded(True)

    # -- 3D ----------------------------------------------------------------------
    def _load_design3d(self, d: Design) -> None:
        """Fill the 3D workspace and inspector from a 3D design."""
        self._viewport_stack.setCurrentIndex(1)
        self.canvas.set_design(None)
        self.update_physics_layer()
        self.title_label.setText(d.name)
        self.title_label.setToolTip(d.name)
        self.kind_label.setText(_KIND_NOUN.get(d.kind, d.kind) + " · 3D")
        self.summary_label3d.setText(design_summary(d))
        lu = display_unit("length")
        self.refs_label3d.setText(f"A ref {area_text(d.ref_area)} · L ref {units.fmt(d.length_ref, 'length', lu, 4)}"
                                  if d.ref_area > 0 else "")
        self.refs_label3d.setToolTip("Reference area and length of the coefficients (References group)")
        wing = d.kind == "wing" and not d.meshes
        p = self.state.project
        names = project_airfoils(p)
        if wing:
            self._content3d.setCurrentIndex(1)
            self._inspector_stack.setCurrentIndex(3)
            self.wing_editor.set_design(d, airfoil_names=names, airfoil_coords=airfoil_resolver(p))
            self.wing_inspector.set_design(d, airfoil_names=names)
        else:
            self._content3d.setCurrentIndex(0)
            self._inspector_stack.setCurrentIndex(2)
            self.view3d.set_design(d)
            self.mesh_inspector.set_design(d)
        self.edges_toggle.setEnabled(True)

    def _active_view3d(self) -> GeometryView3D:
        return self.wing_editor.view3d if self._content3d.currentIndex() == 1 else self.view3d

    def _set_view3d(self, preset: str) -> None:
        self._active_view3d().set_view(preset)

    def _set_edges3d(self, on: bool) -> None:
        for v in (self.view3d, self.wing_editor.view3d):
            v.set_edges(on)

    def _set_dims3d(self, on: bool) -> None:
        self.view3d.set_show_dimensions(on)
        self.wing_editor.view3d.set_show_dimensions(on)
        self.wing_editor.view3d.show_reference = bool(on)      # the quarter-MAC point lies on the wing

    def _push3d(self, before: Design, after: Design, text: str, merge_key: str | None = None) -> None:
        """One undo step for a meshed 3D design (snapshots share the mesh arrays)."""
        cmd = DesignEditCommand(self, before.id, clone_design(before), clone_design(after), text, merge_key)
        self._stack_for(before.id).push(cmd)

    def apply_edit3d(self, edit: dict, text: str) -> bool:
        """Apply a geometry edit (``mesh_inspector.apply_edit3d``) to the current 3D design."""
        d = self.design
        if d is None or d.dimension != 3 or not d.meshes:
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            after, info = edit_design3d(d, edit)
        except Exception as exc:  # noqa: BLE001 - shown to the user, design untouched
            QApplication.restoreOverrideCursor()
            self._error("Edit failed", str(exc))
            return False
        QApplication.restoreOverrideCursor()
        for note in info.get("notes", []):
            self.state.log(note, "info", "Geometry")
        self._push3d(d, after, text)
        return True

    def _ref_edited3(self, field_name: str, value) -> None:
        d = self.design
        if d is None or d.dimension != 3:
            return
        after = clone_design(d)
        if field_name == "moment_ref3":
            after.moment_ref3 = tuple(float(v) for v in value)
        elif field_name == "symmetric":
            after.symmetric = bool(value)
        else:
            setattr(after, field_name, float(value))
        manual = list(dict.fromkeys([*after.source.get("manual_refs", []), field_name]))
        after.source = {**after.source, "manual_refs": manual}
        label_ = {"ref_area": "reference area", "length_ref": "reference length", "span_ref": "span reference",
                  "moment_ref3": "moment point", "symmetric": "symmetry"}.get(field_name, field_name)
        self._push3d(d, after, f"Set {label_}", merge_key=f"{d.id}:{field_name}")

    def _auto_ref3(self, field_name: str, on: bool) -> None:
        d = self.design
        if d is None or d.dimension != 3:
            return
        after = clone_design(d)
        manual = [f for f in after.source.get("manual_refs", []) if f != field_name]
        if not on:
            manual.append(field_name)
        after.source = {**after.source, "manual_refs": manual}
        if not manual:
            after.source.pop("manual_refs", None)
        err = None
        if field_name == "symmetric" and on:
            err = after.source.get("symmetry_error")
            if err is None:
                err = measure_symmetry(after)
            after.symmetric = symmetric_from_error(after, float(err))
        refresh_references3d(after, symmetry_error=err)
        self._push3d(d, after, f"{'Auto' if on else 'Fix'} {field_name.replace('_', ' ')}")

    def _wing_spec_edited(self, spec, text: str, key: str = "") -> None:
        d = self.design
        if d is None or d.kind != "wing":
            return
        after = update_wing(d, spec)
        self._push(d, after, text or "Edit planform", merge_key=f"{d.id}:{key}" if key else None)

    def _wing_ref_edited(self, field_name: str, value) -> None:
        d = self.design
        if d is None or d.kind != "wing":
            return
        after = Design.from_dict(copy.deepcopy(d.to_dict()))
        if field_name == "moment_ref3":
            after.moment_ref3 = tuple(float(v) for v in value)
        else:
            setattr(after, field_name, float(value))
        manual = list(dict.fromkeys([*after.source.get("manual_refs", []), field_name]))
        after.source = {**after.source, "manual_refs": manual}
        self._push(d, after, f"Set {field_name.replace('_', ' ')}", merge_key=f"{d.id}:{field_name}")

    def _wing_auto_ref(self, field_name: str, on: bool) -> None:
        d = self.design
        if d is None or d.kind != "wing":
            return
        after = Design.from_dict(copy.deepcopy(d.to_dict()))
        manual = [f for f in after.source.get("manual_refs", []) if f != field_name]
        if not on:
            manual.append(field_name)
        after.source = {**after.source, "manual_refs": manual}
        if not manual:
            after.source.pop("manual_refs", None)
        refresh_wing_references(after)
        self._push(d, after, f"{'Auto' if on else 'Fix'} {field_name.replace('_', ' ')}")

    def _hover(self, x: float, y: float, inside: bool) -> None:
        if not inside:
            self.hover_label.setText("")
            return
        s = max(self.canvas.scale(), 1e-9)
        dec = max(2, min(6, 1 - math.floor(math.log10(1.0 / s))))
        self.hover_label.setText(f"x {x:.{dec}f}   y {y:.{dec}f}   m")

    # ===================================================================
    # design actions (also reachable from the Project tree and Edit menu)
    # ===================================================================
    def _unique_name(self, base: str) -> str:
        """``base``, or ``base 2``, ``base 3``...: the AppState rule, so a second wing is
        "Tapered wing 2" whether it came from this page, the menu or an import."""
        return self.state.unique_design_name(base)

    def add_design(self, design: Design) -> None:
        """Add a new design to the project and select it."""
        if self.state.project is None:
            return
        design.name = self._unique_name(design.name)
        self.state.add_design(design, select=True)
        self.state.log(f"Added design “{design.name}” ({design_summary(design)})", "info", "Geometry")
        self.state.message(f"Added {design.name}")

    def duplicate_design(self) -> Design | None:
        d = self.design
        if d is None:
            return None
        # a meshed design's arrays are not in to_dict: share them (they are never modified in place)
        dup = clone_design(d) if d.dimension == 3 and d.meshes else Design.from_dict(copy.deepcopy(d.to_dict()))
        from ...model import new_id, now_iso
        dup.id, dup.created = new_id(), now_iso()
        dup.name = self._unique_name(f"{d.name} copy")
        self.state.add_design(dup, select=True)
        return dup

    def rename_design(self) -> None:
        d = self.design
        if d is None:
            return
        name = self._ask_text("Rename design", "Name", d.name)
        if name and name.strip() and name.strip() != d.name:
            d.name = name.strip()
            self.state.update_design(d)

    def delete_design(self) -> None:
        d = self.design
        if d is None:
            return
        n = self._sim_count(d.id)
        extra = (f" Its {n} simulation{'s' if n != 1 else ''} and their results are deleted too."
                 if n else "")
        if not self._confirm("Delete design", f"Delete “{d.name}”?{extra} This cannot be undone."):
            return
        self.state.remove_design(d.id)
        p = self.state.project
        if p is not None and p.designs and not self.state.design_id:
            self.state.select_design(next(iter(p.designs)))

    # ===================================================================
    # creation and import
    # ===================================================================
    def new_naca(self) -> None:
        if self.state.project is None:
            return
        dlg = NacaDialog(self, length_unit=display_unit("length"))
        if self._exec_dialog(dlg) and dlg.design() is not None:
            self.add_design(dlg.design())

    def open_template_gallery(self) -> None:
        if self.state.project is None:
            return
        d = self.design
        dlg = TemplateGalleryDialog(self, kind=d.kind if d is not None else None)
        if self._exec_dialog(dlg):
            try:
                design = dlg.design()
            except Exception as exc:  # noqa: BLE001
                self._error("Template failed", str(exc))
                return
            if design is not None:
                self.add_design(design)

    def import_dialog(self, fmt: str | None = None) -> None:
        if self.state.project is None:
            return
        if fmt == "mesh":
            flt = MESH_FILTER
        else:
            flt = (FORMAT_FILTERS[fmt] + ";;All files (*)") if fmt in FORMAT_FILTERS else FILE_FILTER
        path = self._ask_open_file("Import 3D mesh" if fmt == "mesh" else "Import geometry", flt)
        if path:
            self.import_files([path], fmt=None if fmt == "mesh" else fmt)

    def new_wing(self) -> Design | None:
        """Add a finite-wing design (a default tapered planform) and select it."""
        if self.state.project is None:
            return None
        from ...geometry3d import design_from_wing
        # not "Wing": the step bar and the Project tree put the kind tag before the name,
        # and "Wing | Wing" reads as a glitch; add_design makes the name unique
        d = design_from_wing(default_wing_spec(), NEW_WING_NAME)
        self.add_design(d)
        return d

    def new_primitive(self, shape: str) -> Design | None:
        """Add a basic 3D body (sphere, box, cylinder across the flow) as a generic design."""
        if self.state.project is None or shape not in PRIMITIVES:
            return None
        from ... import geometry3d as g3
        text, kw = PRIMITIVES[shape]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mesh = getattr(g3, shape)(name=text, **kw)
            d = g3.design_from_meshes([mesh], text, "generic", source={"type": "primitive", "shape": shape})
        finally:
            QApplication.restoreOverrideCursor()
        self.add_design(d)
        return d

    def _add_imported(self, design: Design) -> None:
        self.add_design(design)

    def import_files(self, paths: list[str], fmt: str | None = None) -> list[Design]:
        """Import files (from the menu or a drop). Returns the designs added."""
        added: list[Design] = []
        if self.state.project is None:
            self.state.message("Open or create a project before importing geometry.", 6000)
            return added
        for path in paths:
            if fmt is None and is_mesh_path(path):
                got = import_mesh_dialog(self, self.state, str(path), exec_dialog=self._exec_dialog,
                                         add=self._add_imported)
                added.extend(got or [])
                continue
            kind = fmt or classify_path(path)
            if kind is None:
                self._error("Unsupported file", f"{Path(path).name}: use .dat, .svg, .dxf, .csv, an image, or a "
                                                "3D mesh (.stl, .obj, .ply).")
                continue
            cls = ImageSilhouetteDialog if kind == "image" else PolygonImportDialog
            dlg = cls(path, kind, self, length_unit=display_unit("length"))
            if self._exec_dialog(dlg):
                d = dlg.design()
                if d is not None:
                    self.add_design(d)
                    added.append(d)
        return added

    # ===================================================================
    # edits (all undoable)
    # ===================================================================
    def _apply_snapshot(self, design_id: str, snapshot: dict | Design) -> None:
        """Make the project's design equal to ``snapshot`` (undo/redo target)."""
        p = self.state.project
        if p is None or design_id not in p.designs:
            return
        if isinstance(snapshot, Design):              # a meshed 3D design: arrays shared, not copied
            self.state.update_design(clone_design(snapshot))
        else:
            self.state.update_design(Design.from_dict(copy.deepcopy(snapshot)))

    def _push(self, before: Design, after: Design, text: str, merge_key: str | None = None) -> None:
        cmd = DesignEditCommand(self, before.id, copy.deepcopy(before.to_dict()),
                                copy.deepcopy(after.to_dict()), text, merge_key)
        self._stack_for(before.id).push(cmd)          # push() runs redo() -> update_design

    def apply_edit(self, edit: dict, text: str) -> bool:
        """Apply a recorded geometry edit to the current design (one undo step)."""
        d = self.design
        if d is None:
            return False
        after = Design.from_dict(copy.deepcopy(d.to_dict()))
        try:
            apply_edit(after, edit)
        except Exception as exc:  # noqa: BLE001 - shown to the user, design untouched
            self._error("Edit failed", str(exc))
            return False
        after.source = {**after.source, "edits": [*after.source.get("edits", []), edit]}
        scale = None
        if edit.get("op") in ("scale", "scale_to") and d.bodies:
            w0 = d.bounds()[1] - d.bounds()[0]
            scale = (after.bounds()[1] - after.bounds()[0]) / w0 if w0 > 0 else None
        refresh_references(after, scale=scale)
        self._push(d, after, text)
        return True

    def apply_scale(self) -> None:
        """Scale the current design so its chord / length equals the Transform field."""
        if self._loading:
            return
        d = self.design
        if d is None or not d.bodies:
            return
        if d.kind == "airfoil":
            le, te = airfoil_chord(d)
            current = float(np.hypot(*(te - le)))
        else:
            current = d.bounds()[1] - d.bounds()[0]
        target = self.scale_length.si_value()
        if math.isclose(current, target, rel_tol=1e-9):
            return
        self.apply_edit({"op": "scale_to", "length": target}, f"Scale to {self.scale_length.text()}")

    # -- template parameters (live, debounced) ------------------------------
    def _attach_naca_validation(self, specs: list) -> None:
        """Live NACA-code validation for template text fields that hold one
        (the NACA and front-wing templates), same rules as the NACA dialog."""
        self._naca_fields = set()
        for spec in specs:
            text = f"{getattr(spec, 'name', '')} {getattr(spec, 'label', '')}".lower()
            if getattr(spec, "unit_kind", "") == "text" and "naca" in text:
                w = self.param_form.field(spec.name)
                self._naca_fields.add(spec.name)
                w.textChanged.connect(lambda t, w=w: self._mark_naca_field(w, t))

    def _mark_naca_field(self, w: QWidget, text: str) -> None:
        err = validate_naca(text)
        set_prop(w, "invalid", "true" if err else "false")
        w.setToolTip(err or (describe_naca(text) + "\nPress Enter to apply."))

    def _param_edited(self, name: str, value: Any) -> None:
        if self._loading:
            return
        if name in self._naca_fields and validate_naca(str(value)):
            self.state.message(validate_naca(str(value)) or "", 6000)
            return
        self._pending_params = self.param_form.values()
        self._pending_key = f"param:{name}"
        self._pending_design_id = self.state.design_id
        self._regen_timer.start()

    def _flush_regeneration(self) -> None:
        self._regen_timer.stop()
        params, key = self._pending_params, self._pending_key
        self._pending_params = None
        p = self.state.project
        d = p.designs.get(self._pending_design_id) if p is not None else None
        if params is None or d is None or template_source(d) is None:
            return
        try:
            after = regenerate(d, params=params)
        except Exception as exc:  # noqa: BLE001
            self.state.log(f"Could not rebuild “{d.name}”: {exc}", "error", "Geometry")
            self.state.message(f"Could not rebuild the shape: {exc}", 8000)
            return
        self._push(d, after, "Change " + (key or "parameter").split(":", 1)[-1].replace("_", " "),
                   merge_key=f"{d.id}:{key}")

    def _reset_params(self) -> None:
        d = self.design
        tmpl = template_source(d) if d is not None else None
        if d is None or tmpl is None:
            return
        self._pending_params = dict(self.param_form.defaults())
        self._pending_key = "reset"
        self._pending_design_id = d.id
        self._flush_regeneration()

    # -- NACA ---------------------------------------------------------------
    def _show_naca_status(self, err: str | None) -> None:
        if err:
            self.naca_status.set("error", err)
            set_prop(self.naca_code, "invalid", "true")
        else:
            self.naca_status.set("info", describe_naca(self.naca_code.text()))
            set_prop(self.naca_code, "invalid", "false")

    def _naca_code_typing(self, text: str) -> None:
        if not self._loading:
            self._show_naca_status(validate_naca(text))

    def _naca_code_edited(self) -> None:
        if self._loading:
            return
        code = self.naca_code.text().strip().upper().removeprefix("NACA").strip()
        err = validate_naca(code)
        self._show_naca_status(err)
        d = self.design
        if err is None and d is not None and code != str(d.source.get("code", "")):
            self._naca_changed({"code": code}, "naca_code", rename=True)

    def _naca_changed(self, change: dict, key: str, rename: bool = False) -> None:
        d = self.design
        if self._loading or d is None or d.source.get("type") != "naca":
            return
        try:
            after = regenerate(d, naca=change)
        except Exception as exc:  # noqa: BLE001
            self._show_naca_status(str(exc))
            return
        if rename and d.name.upper().startswith("NACA"):
            after.name = f"NACA {after.source.get('code', '')}"
        self._push(d, after, "Change NACA " + key.split("_", 1)[-1], merge_key=f"{d.id}:{key}")

    # -- references ---------------------------------------------------------
    def _kind_edited(self, _index: int) -> None:
        d = self.design
        if self._loading or d is None:
            return
        after = Design.from_dict(copy.deepcopy(d.to_dict()))
        after.kind = str(self.kind_combo.currentData())
        refresh_references(after)
        self._push(d, after, f"Kind: {after.kind}")

    def _ref_edited(self, field_name: str, value) -> None:
        d = self.design
        if self._loading or d is None:
            return
        after = Design.from_dict(copy.deepcopy(d.to_dict()))
        setattr(after, field_name, tuple(float(v) for v in value) if field_name == "moment_ref"
                else float(value))
        manual = list(dict.fromkeys([*after.source.get("manual_refs", []), field_name]))
        after.source = {**after.source, "manual_refs": manual}
        self._push(d, after, f"Set {field_name.replace('_', ' ')}", merge_key=f"{d.id}:{field_name}")

    def _auto_ref_clicked(self, field_name: str, on: bool) -> None:
        d = self.design
        if d is None:
            return
        after = Design.from_dict(copy.deepcopy(d.to_dict()))
        manual = [f for f in after.source.get("manual_refs", []) if f != field_name]
        if not on:
            manual.append(field_name)        # "un-auto": freeze the current value
        after.source = {**after.source, "manual_refs": manual}
        if not manual:
            after.source.pop("manual_refs", None)
        refresh_references(after)
        self._push(d, after, f"{'Auto' if on else 'Fix'} {field_name.replace('_', ' ')}")

    # -- navigation ---------------------------------------------------------
    def _go_setup(self) -> None:
        self._flush_regeneration()
        self.state.navigate("setup")


PHYSICS_LAYER_TIP = ("Show the selected simulation's emitters, heat sources, fans, jets, vents and closed box "
                     "over the geometry, in the design frame; drag one to move it")


def _spacer(width: int) -> QWidget:
    """An empty cell as wide as a row button, so editors in rows without one line up."""
    w = QLabel()
    w.setFixedWidth(width)
    return w
