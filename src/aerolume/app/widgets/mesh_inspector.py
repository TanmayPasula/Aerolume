"""The Geometry page's 3D inspector, and the 3D design edits it drives.

A 3D design (``Design.dimension == 3``) made from imported meshes is edited
with the same model as a 2D one (DESIGN.md 7.11): every change is one undoable
step that replaces the design, and reference quantities follow the geometry
unless the user typed one (``source["manual_refs"]``). What differs is the
data. A mesh can hold a few hundred thousand triangles, so:

* **Edits are transforms, applied once.** A 2D design replays a recipe; a
  meshed design has no recipe, so :func:`apply_edit3d` transforms the arrays
  (``geometry3d.transform`` returns new arrays and never modifies its input)
  and appends the edit to ``source["edits"]`` only as a record.
* **Arrays are shared, never copied, by undo.** :func:`clone_design` copies the
  small JSON part of a design and *shares* the vertex and face arrays, which
  every edit treats as read-only; an undo step therefore costs the arrays that
  really changed (a rotation: the vertices; faces are re-shared when the
  winding did not flip), not a copy of the whole model per step.
* **Slow measurements stay off the GUI thread.** Mesh statistics (holes, open
  edges per part) and :func:`aerolume.geometry3d.check_design3d` take up to a
  second on a large mesh; the inspector runs them on a worker thread and shows
  "Checking..." meanwhile. The two quantities an edit itself must update - the
  frontal area and, after a rotation or a move, the mirror error that decides
  the ``symmetric`` flag - are measured directly but cheaply (rasterised area;
  6000 surface samples instead of the importer's 20 000).

The inspector (:class:`MeshInspector`) only *requests* changes through signals;
the page turns them into undo commands, so the widget never touches the
project.
"""
from __future__ import annotations

import contextlib
import copy
import math
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core import units
from ...model import Design, Mesh3D
from .. import theme
from .common import PropertyGrid
from .geometry_canvas import meshes_bounds
from .param_form import (
    ElidedLabel,
    QuantitySpinBox,
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

#: Fields a user can fix by typing a value; the rest follow the geometry ("Auto").
REF_FIELDS3 = ("ref_area", "length_ref", "span_ref", "moment_ref3", "symmetric")
#: The importer's thresholds (geometry3d.design3d), repeated so a re-measure agrees with it.
SYMMETRY_TOL = 0.005          # RMS mirror error over the largest dimension
COMPACT_RATIO = 1.5
SYMMETRY_SAMPLES = 6000       # an edit re-measures cheaply; the importer uses 20 000

#: Orientation presets offered in the inspector and the import dialog: (key, label).
#: Keys are ``geometry3d.ORIENT_PRESETS`` names, plus "auto" (``auto_orient``).
ORIENTATIONS = (
    ("auto", "Automatic"),
    ("y-up->z-up", "Y up → Z up"),
    ("z-up->y-up", "Z up → Y up"),
    ("x-up->z-up", "X up → Z up"),
    ("upside-down", "Upside down"),
    ("nose+x->nose-x", "Turn to face the flow (180° about z)"),
    ("nose+y->nose-x", "Nose +y → −x"),
    ("nose-y->nose-x", "Nose −y → −x"),
)
ORIENTATION_TIPS = {
    "auto": "The longest dimension along x (the flow), the flat side down; mirror symmetry decides the span",
    "y-up->z-up": "Most OBJ, glTF and game exports have y up",
    "nose+x->nose-x": "The model points downstream: turn it to face the flow",
}
KIND3_ITEMS = (("vehicle", "Vehicle (on a road)"), ("generic", "Generic body"))
FORMAT3 = {".stl": "STL", ".obj": "Wavefront OBJ", ".ply": "PLY"}
UNIT_NAMES = {1.0: "Metres", 0.01: "Centimetres", 0.001: "Millimetres", 0.0254: "Inches", 0.3048: "Feet"}


def _g3():
    from ... import geometry3d
    return geometry3d


# ---------------------------------------------------------------------------
# pure helpers (no Qt) - unit-tested directly
# ---------------------------------------------------------------------------

def clone_design(design: Design) -> Design:
    """A copy of ``design`` whose meshes *share* the vertex and face arrays.

    ``Design.to_dict`` leaves mesh arrays out (they live in the project's
    ``.npz``); this copies everything else and re-attaches the same array
    objects. Safe because nothing in the application modifies mesh arrays in
    place: every geometry operation returns new arrays.
    """
    d = Design.from_dict(copy.deepcopy(design.to_dict()))
    for new, old in zip(d.meshes, design.meshes, strict=True):
        new.vertices, new.faces = old.vertices, old.faces
    return d


def _bbox(meshes: list[Mesh3D]) -> np.ndarray:
    """(2, 3) array: min and max corner over the vertices the faces use (cached per array)."""
    b = meshes_bounds(list(meshes))
    if b is None:
        raise ValueError("no triangles")
    return np.array([[b[0], b[2], b[4]], [b[1], b[3], b[5]]])


def _solid(design: Design) -> list[Mesh3D]:
    return [m for m in design.meshes if len(m.faces) and len(m.vertices)]


def on_road(design: Design) -> bool:
    """A 3D design standing on the road plane z = 0 (vehicles always; generic bodies
    when imported or placed so)."""
    src = design.source if isinstance(design.source, dict) else {}
    return design.kind == "vehicle" or bool(src.get("on_road"))


#: The one definition of "Ground clearance" the app shows (the inspector and the mesh
#: import dialog): measured to the body, wheels excluded, as a car's ride height is. The
#: import dialog's placement field is the model's lowest point instead (the tyres for a
#: vehicle) and is named so; calling both "clearance" showed 0.000 m and 0.160 m for one car.
CLEARANCE_TIP = ("Height of the body's lowest point above the road (z = 0), wheels excluded: "
                 "the gap under the floor")


def body_clearance(meshes: list[Mesh3D]) -> float:
    """Height [m] of the lowest point of the non-wheel parts above the road (all the
    parts when none is a wheel): :data:`CLEARANCE_TIP`."""
    ms = [m for m in meshes if m.role != "wheel"] or list(meshes)
    return float(_bbox(ms)[0, 2]) if ms else 0.0


def clearance(design: Design) -> float:
    """Height of the lowest point of the non-wheel parts above the road [m]."""
    return body_clearance(_solid(design))


def auto_references3d(design: Design) -> dict[str, Any]:
    """Reference quantities that follow a meshed 3D design (DESIGN.md 7.2).

    The importer's conventions (:func:`aerolume.geometry3d.design_from_meshes`):
    reference area = frontal area (union of the parts projected along x);
    reference length = overall length, or for a compact bluff body (length at
    most 1.5 times its smaller cross-stream size) that cross-stream size, the
    diameter or side published coefficients use; span reference = width;
    moment point at mid-length on the road for vehicles (at the lowest point
    for a vehicle lifted off the road), the bounding-box centre otherwise;
    ``force_ref`` (the 2D field) = frontal height.
    """
    ms = _solid(design)
    if not ms:
        return {}
    (x0, y0, z0), (x1, y1, z1) = _bbox(ms)
    L, W, H = x1 - x0, y1 - y0, z1 - z0
    area = float(_g3().frontal_area(ms, axis="x"))
    cross = min(W, H)
    compact = cross > 0 and L <= COMPACT_RATIO * cross
    if design.kind == "vehicle":
        mref = (x0 + 0.5 * L, 0.0, 0.0 if on_road(design) else z0)
    else:
        mref = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))
    return {"ref_area": area, "length_ref": float(cross if compact else L), "span_ref": float(W),
            "moment_ref3": tuple(float(v) for v in mref), "force_ref": float(H), "compact": bool(compact)}


def measure_symmetry(design: Design, n_samples: int = SYMMETRY_SAMPLES) -> float:
    """RMS distance [m] between the surface mirrored across y = 0 and the surface."""
    ms = _solid(design)
    return float(_g3().symmetry_error(ms, plane_y=0.0, n_samples=n_samples)) if ms else 0.0


def symmetric_from_error(design: Design, error: float) -> bool:
    """The importer's rule: symmetric when the mirror error is within 0.5 % of the
    largest dimension."""
    ms = _solid(design)
    if not ms:
        return False
    size = float(np.ptp(_bbox(ms), axis=0).max())
    return bool(error <= SYMMETRY_TOL * size)


def refresh_references3d(design: Design, *, scale: float | None = None, symmetry_error: float | None = None) -> None:
    """After a geometry change: recompute the automatic references, keep typed ones.

    ``scale`` is the uniform factor of a scale edit: typed lengths and the typed
    moment point scale with it and a typed area with its square (a reference
    typed for a 1:5 model is wrong for the full-size car). ``symmetry_error``
    [m], when the edit re-measured it, updates ``source["symmetry_error"]`` and,
    unless the user set the flag by hand, ``symmetric``.
    """
    manual = set(design.source.get("manual_refs", [])) if isinstance(design.source, dict) else set()
    auto = auto_references3d(design)
    if not auto:
        return
    k = 1.0 if scale is None else float(scale)
    design.ref_area = design.ref_area * k * k if "ref_area" in manual else auto["ref_area"]
    design.length_ref = design.length_ref * k if "length_ref" in manual else auto["length_ref"]
    design.span_ref = design.span_ref * k if "span_ref" in manual else auto["span_ref"]
    if "moment_ref3" in manual:
        design.moment_ref3 = tuple(float(c) * k for c in design.moment_ref3)
    else:
        design.moment_ref3 = auto["moment_ref3"]
    design.force_ref = auto["force_ref"]
    src = dict(design.source)
    src["frontal_area"] = auto["ref_area"]
    src["compact"] = auto["compact"]
    if symmetry_error is not None:
        src["symmetry_error"] = float(symmetry_error)
        if "symmetric" not in manual:
            design.symmetric = symmetric_from_error(design, symmetry_error)
    elif scale is not None and src.get("symmetry_error") is not None:
        src["symmetry_error"] = float(src["symmetry_error"]) * k
    design.source = src


def _reshare(new: list[Mesh3D], old: list[Mesh3D]) -> list[Mesh3D]:
    """Give each new mesh its old face array back when the faces did not change
    (a transform without a reflection copies them unchanged)."""
    for n, o in zip(new, old, strict=True):
        if n.faces is not o.faces and n.faces.shape == o.faces.shape and np.array_equal(n.faces, o.faces):
            n.faces = o.faces
    return new


def _settle(design: Design, meshes: list[Mesh3D], *, road_clearance: float | None) -> list[Mesh3D]:
    """Put re-oriented meshes back where the importer puts them: a vehicle's nose
    at x = 0, centred on y = 0, on the road at its clearance; a generic body
    centred on the origin (or standing on the road)."""
    g = _g3()
    (x0, y0, z0), (x1, y1, z1) = _bbox(meshes)
    dx = -x0 if design.kind == "vehicle" else -0.5 * (x0 + x1)
    dy = -0.5 * (y0 + y1)
    if road_clearance is not None:
        road_clearance = max(float(road_clearance), 0.0)
        body = [m for m in meshes if m.role != "wheel"] or meshes
        dz = road_clearance - float(_bbox(body)[0, 2]) if design.kind == "vehicle" else road_clearance - z0
        if design.kind == "vehicle":                      # the wheels, not the body, stand on the road
            dz = max(dz, -z0)
    else:
        dz = -0.5 * (z0 + z1)
    return g.transform(meshes, translate=(dx, dy, dz))


#: Edits that can change the left-right mirror symmetry (and so must re-measure it).
_SYMMETRY_OPS = frozenset({"rotate", "orient", "mirror", "centre_y", "translate", "kind"})


def apply_edit3d(design: Design, edit: dict) -> dict:
    """Apply one edit to a meshed 3D design, in place (new mesh objects, new arrays).

    Returns ``{"scale": k}`` for uniform scale edits and ``{"symmetry": True}``
    when the edit can change the mirror symmetry, so the caller knows what to
    re-measure (:func:`refresh_references3d`). Edits:

    * ``scale_to`` - ``length`` [m]: uniform scale about the origin (the nose on
      the road for vehicles, the centre for generic bodies) to that overall length;
      ``scale`` - ``factor``.
    * ``rotate`` - ``axis`` "x" | "y" | "z", ``deg`` (right-hand rule) about the
      bounding-box centre; a design on the road is set back on it at its
      clearance (and a vehicle's nose back at x = 0).
    * ``orient`` - ``preset``: an orientation preset (``geometry3d.ORIENT_PRESETS``)
      or "auto" (``auto_orient``), then placed as the importer places it.
    * ``place_on_road`` - ``clearance`` [m] (the design then stands on the road);
      ``free`` - a generic body taken off the road, centred on z = 0.
    * ``centre_y`` - centre the bounding box on y = 0 (the symmetry plane).
    * ``mirror`` - ``axis``: reflect across the plane through the box centre.
    * ``translate`` - ``dx``, ``dy``, ``dz`` [m].
    * ``kind`` - ``kind`` "vehicle" | "generic"; a new vehicle is set on the road
      with its nose at x = 0 and its wheel-shaped parts become wheels.
    * ``role`` - ``index``, ``role`` ("body" | "wheel"): a wheel gets its centre
      and axle from :func:`aerolume.geometry3d.wheel_params` (it spins with a
      moving road).
    """
    g = _g3()
    op = edit.get("op")
    old = list(design.meshes)
    ms = _solid(design)
    if not ms:
        raise ValueError("the design has no triangles to edit")
    if len(ms) != len(old):
        raise ValueError("some meshes of this design have no triangles loaded")
    out: dict = {}
    (x0, y0, z0), (x1, y1, z1) = _bbox(ms)
    centre = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))
    road = on_road(design)
    clr = clearance(design) if road else None
    src = dict(design.source)
    if op in ("scale_to", "scale"):
        if op == "scale_to":
            length = float(edit["length"])
            if not (length > 0 and x1 > x0):
                raise ValueError("the length must be positive")
            k = length / (x1 - x0)
        else:
            k = float(edit["factor"])
            if not k > 0:
                raise ValueError("the scale factor must be positive")
        new = g.transform(ms, scale=k)
        out["scale"] = k
    elif op == "rotate":
        axis = str(edit.get("axis", "z"))
        if axis not in "xyz" or len(axis) != 1:
            raise ValueError("the rotation axis must be x, y or z")
        rot = [0.0, 0.0, 0.0]
        rot["xyz".index(axis)] = float(edit["deg"])
        new = g.transform(ms, rotate_deg=rot, about=centre)
        if road:
            new = _settle(design, new, road_clearance=clr)
    elif op == "orient":
        preset = str(edit.get("preset", "auto"))
        notes: list[str] = []
        new = g.auto_orient(ms, notes) if preset == "auto" else g.orient_preset(ms, preset)
        new = list(new) if not isinstance(new, Mesh3D) else [new]
        new = _settle(design, new, road_clearance=clr if road else None)
        if notes:
            out["notes"] = notes
    elif op == "place_on_road":
        c = float(edit.get("clearance", 0.0))
        if c < 0:
            raise ValueError("the clearance cannot be negative")
        body = [m for m in ms if m.role != "wheel"] or ms
        dz = c - float(_bbox(body)[0, 2])
        if design.kind == "vehicle" and any(m.role == "wheel" for m in ms):
            dz = max(dz, -z0)                            # the tyres may not sink into the road
        new = g.transform(ms, translate=(0.0, 0.0, dz))
        src["on_road"] = True
    elif op == "free":
        if design.kind == "vehicle":
            raise ValueError("a vehicle stands on the road; change the kind to Generic body first")
        new = g.transform(ms, translate=(0.0, 0.0, -centre[2]))
        src["on_road"] = False
    elif op == "centre_y":
        new = g.centre_y(ms)
    elif op == "mirror":
        axis = str(edit.get("axis", "y"))
        new = g.transform(ms, about=centre, mirror_axis=axis)
    elif op == "translate":
        new = g.transform(ms, translate=(float(edit.get("dx", 0.0)), float(edit.get("dy", 0.0)),
                                         float(edit.get("dz", 0.0))))
    elif op == "kind":
        kind = str(edit["kind"])
        if kind not in ("vehicle", "generic"):
            raise ValueError("a meshed 3D design is a vehicle or a generic body")
        design.kind = kind
        new = list(ms)
        if kind == "vehicle":
            # a vehicle stands on the road with its nose at x = 0, and its wheels spin
            with contextlib.suppress(Exception):
                for i, wh in enumerate(g.find_wheels3d(new)):
                    if wh is not None and new[i].role == "body":
                        m = new[i]
                        new[i] = Mesh3D(name=m.name, vertices=m.vertices, faces=m.faces, role="wheel", omega=0.0,
                                        center=wh[0], axis=wh[1])
                        new[i].vertices, new[i].faces = m.vertices, m.faces
            src["on_road"] = True
            new = _settle(design, new, road_clearance=clr if clr is not None else 0.0)
    elif op == "role":
        i = int(edit["index"])
        role = str(edit["role"])
        if role not in ("body", "wheel"):
            raise ValueError("a mesh is a body or a wheel")
        new = list(ms)
        m = ms[i]
        if role == "wheel":
            c, ax, _r = g.wheel_params(m)
            new[i] = Mesh3D(name=m.name, vertices=m.vertices, faces=m.faces, role="wheel", omega=0.0,
                            center=c, axis=ax)
        else:
            new[i] = Mesh3D(name=m.name, vertices=m.vertices, faces=m.faces, role="body")
        new[i].vertices, new[i].faces = m.vertices, m.faces        # shared, not copied
    else:
        raise ValueError(f"unknown 3D edit {op!r}")
    design.meshes = _reshare(list(new), ms)
    design.source = {**src, "edits": [*src.get("edits", []), dict(edit)]}
    if op in _SYMMETRY_OPS:
        out["symmetry"] = True
    return out


def edit_design3d(design: Design, edit: dict) -> tuple[Design, dict]:
    """``(new design, info)``: a clone of ``design`` with ``edit`` applied and its
    references refreshed - one undo step's "after" state."""
    after = clone_design(design)
    info = apply_edit3d(after, edit)
    err = measure_symmetry(after) if info.get("symmetry") else None
    refresh_references3d(after, scale=info.get("scale"), symmetry_error=err)
    return after, info


def size3(design: Design) -> tuple[float, float, float] | None:
    ms = _solid(design)
    if not ms:
        return None
    b = _bbox(ms)
    return tuple(float(v) for v in (b[1] - b[0]))  # type: ignore[return-value]


def _len(v: float) -> str:
    return units.fmt(v, "length", display_unit("length"), 4)


def design_summary3d(design: Design) -> str:
    """One factual line for the status strip of a 3D design."""
    noun = {"vehicle": "Vehicle", "generic": "Body", "wing": "Wing"}.get(design.kind, design.kind)
    if design.kind == "wing" and design.wing is not None and not design.meshes:
        try:
            pf = _g3().planform(design.wing)
        except ValueError as exc:
            return f"Wing · invalid planform ({exc})"
        n = len(design.wing.sections)
        return (f"Wing · span {_len(pf['span'])} · area {area_text(pf['area'])} · AR {pf['aspect_ratio']:.2f} · "
                f"{n} section{'s' if n != 1 else ''}")
    s = size3(design)
    if s is None:
        return f"{noun} · no triangles loaded"
    nparts = len(design.meshes)
    ntri = sum(len(m.faces) for m in design.meshes)
    u = display_unit("length")
    dims = " × ".join(units.fmt(v, "length", u, 4).rsplit(" ", 1)[0] for v in s) + f" {u}"
    return f"{noun} · {dims} · {nparts} part{'s' if nparts != 1 else ''} · {ntri:,} triangles"


def mm_text(v: float) -> str:
    """A small length in millimetres with 2-3 significant figures ("0.048 mm", "22.5 mm")."""
    return f"{units.sig(v * 1000.0, 3)} mm" if v < 1.0 else f"{units.sig(v, 3)} m"


def area_text(a: float) -> str:
    """An area with enough digits for both a car (2.13 m²) and a model (78.5 cm²)."""
    if not math.isfinite(a):
        return "—"
    if a >= 0.01:
        return f"{units.sig(a, 4)} m²"
    return f"{units.sig(a * 1e4, 4)} cm²"


def unit_text(scale_to_m: float | None) -> str:
    """"Millimetres (× 0.001)" for the importer's recorded scale."""
    if scale_to_m is None:
        return "—"
    s = float(scale_to_m)
    for k, name in UNIT_NAMES.items():
        if math.isclose(s, k, rel_tol=1e-9):
            return f"{name} (× {s:g})" if s != 1.0 else "Metres"
    return f"Custom (× {s:.6g})"


def surface_text(info: dict) -> str:
    """"Closed" or "Open: 2 holes, 64 open edges" for :func:`mesh_info` results."""
    if info.get("watertight"):
        return "Closed"
    holes, edges = int(info.get("holes", 0)), int(info.get("boundary_edges", 0))
    if edges == 0:
        return "Non-manifold edges"
    return f"Open: {holes} hole{'s' if holes != 1 else ''}, {edges:,} open edges"


def classify_check(text: str) -> str:
    """Level of a :func:`check_design3d` sentence: importer notes are information,
    problems are warnings, a body below the road is an error."""
    low = text.lower()
    if "below the road" in low or "no meshes" in low or "no triangles" in low:
        return "error"
    if low.startswith(("re-oriented", "orientation kept", "scaled by", "largest dimension", "compact bluff body",
                       "not closed (holes")) \
            or "recognised as wheels" in low or "re-wound" in low or "plausible in metres" in low:
        return "info"
    return "warning"


_REWOUND = re.compile(r"^'(?P<part>.+)': (?P<n>\d+) face\(s\) re-wound so the normals point outward\.?$")


def merge_checks(checks: list[str]) -> list[str]:
    """The checks with the importer's per-part winding notes merged into one line.

    The importer reports each re-wound part ("'wheel_fl': 192 face(s) re-wound
    ..."); on a car of a dozen parts that buried the one line that mattered.
    The information is kept, as one sentence with the total.
    """
    out: list[str] = []
    parts, faces, at = [], 0, None
    for c in checks:
        m = _REWOUND.match(str(c))
        if m:
            parts.append(m["part"])
            faces += int(m["n"])
            at = len(out) if at is None else at
        else:
            out.append(str(c))
    if parts:
        if len(parts) == 1:
            text = f"'{parts[0]}': {faces:,} face(s) re-wound so the normals point outward."
        else:
            text = f"Face winding made consistent on {len(parts)} parts ({faces:,} faces re-wound outward)."
        out.insert(at, text)
    return out


def analyse_design3d(design: Design) -> dict:
    """The slow part of the inspector (run on a worker thread): per-part mesh
    statistics and the plain-language checks of :func:`check_design3d`."""
    g = _g3()
    parts = []
    for m in design.meshes:
        try:
            parts.append(g.mesh_info(m) if len(m.faces) else {"triangles": 0})
        except Exception as exc:  # noqa: BLE001 - one odd part must not hide the others
            parts.append({"triangles": len(m.faces), "error": str(exc)})
    try:
        checks = list(g.check_design3d(design))
    except Exception as exc:  # noqa: BLE001 - a checker failure is itself a diagnostic
        checks = [f"The geometry check failed: {exc}"]
    return {"parts": parts, "checks": checks}


# ---------------------------------------------------------------------------
# a tiny worker for the slow analysis (one at a time, newest wins)
# ---------------------------------------------------------------------------

class _Analyser(QObject):
    """Runs :func:`analyse_design3d` on a thread; ``done(token, result)`` in the GUI thread."""

    done = Signal(int, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._token = 0
        self._busy = threading.Event()

    def start(self, design: Design) -> int:
        self._token += 1
        token = self._token
        self._busy.set()

        def run() -> None:
            try:
                result = analyse_design3d(design)
            except Exception as exc:  # noqa: BLE001 - reported, never raised on a thread
                result = {"parts": [], "checks": [f"The geometry check failed: {exc}"]}
            with contextlib.suppress(RuntimeError):     # the inspector was deleted meanwhile
                self.done.emit(token, result)

        threading.Thread(target=run, name="aerolume-mesh-check", daemon=True).start()
        return token

    @property
    def token(self) -> int:
        return self._token


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


def _area_box() -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(1e-8, 1e4)
    s.setDecimals(4)
    s.setSuffix(" m²")
    s.setKeyboardTracking(False)
    s.setMinimumWidth(64)
    s.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return s


def _set_area(box: QDoubleSpinBox, a: float) -> None:
    """Load an area with enough decimals for its size (2.130 m², 0.007854 m²)."""
    dec = 3 if not a > 0 else int(min(8, max(3, 3 - math.floor(math.log10(a)))))
    box.blockSignals(True)
    box.setDecimals(dec)
    box.setSingleStep(10.0 ** (-dec + 1))
    box.setValue(a)
    box.blockSignals(False)


# ---------------------------------------------------------------------------
# the inspector
# ---------------------------------------------------------------------------

class MeshInspector(QWidget):
    """Property grid for a meshed 3D design: Source, Transform, References, Parts, Diagnostics.

    Signals (the page makes undo steps of them):

    * ``editRequested(dict, str)`` - a geometry edit for :func:`apply_edit3d` and
      its undo text;
    * ``refEdited(str, object)`` - a typed reference value (``ref_area``,
      ``length_ref``, ``span_ref`` as floats, ``moment_ref3`` as a tuple,
      ``symmetric`` as a bool);
    * ``autoRefToggled(str, bool)`` - a reference switched to / from Auto.

    A change of kind is an edit too (``{"op": "kind"}``): a new vehicle is set
    on the road and its wheels are recognised.
    """

    editRequested = Signal(dict, str)
    refEdited = Signal(str, object)
    autoRefToggled = Signal(str, bool)

    CHECK_DELAY_MS = 150

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._design: Design | None = None
        self._loading = False
        self._geom_sig: tuple | None = None
        self._analysis: dict | None = None
        self._analysis_sig: tuple | None = None
        self._pending_token = 0
        self._analyser = _Analyser(self)
        self._analyser.done.connect(self._analysis_done)
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(self.CHECK_DELAY_MS)
        self._check_timer.timeout.connect(self._start_analysis)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.grid = PropertyGrid(label_width=LABEL_W)
        self.grid.setObjectName("mesh_properties")
        self._build_source()
        self._build_transform()
        self._build_refs()
        self._build_parts()
        self._build_diagnostics()
        self.grid.add_stretch()
        lay.addWidget(self.grid.scrollable())
        guard_wheel(self)

    # -- construction -----------------------------------------------------------
    def _build_source(self) -> None:
        g = self.grid.add_group("Source", key="source")
        self.source_group = g
        self.src_type = label("", None)
        self.src_file = ElidedLabel("", None, mode=Qt.TextElideMode.ElideMiddle)
        self.src_format = label("", None)
        self.src_units = label("", None)
        self.src_size = label("", None)
        self.src_triangles = label("", None)
        self.src_surface = label("", None)
        self.src_edits = label("", None)
        g.add_row("Type", self.src_type)
        self._file_row = g.add_row("File", self.src_file)
        self._format_row = g.add_row("Format", self.src_format)
        self._units_row = g.add_row("Units", self.src_units, tooltip="Unit the file was read in, scaled to metres")
        g.add_row("Size", self.src_size, tooltip="Length × width × height of the bounding box")
        g.add_row("Triangles", self.src_triangles)
        g.add_row("Surface", self.src_surface,
                  tooltip="Whether every part is a closed surface. Open meshes are fine: small holes are closed "
                          "when the shape is voxelised.")
        g.add_row("Edits", self.src_edits, tooltip="Transforms applied since import (undo with Ctrl+Z)")

    def _build_transform(self) -> None:
        g = self.grid.add_group("Transform", key="transform")
        self.transform_group = g
        lu = display_unit("length")
        self.scale_length = QuantitySpinBox("length", lu, minimum=1e-4, maximum=1e4, step=0.01)
        self.scale_length.setObjectName("scale_length3")
        self.scale_length.siValueChanged.connect(self._scale_edited)
        g.add_row("Length", self.scale_length,
                  tooltip="Overall length along x. Type the real length and press Enter: the model is scaled "
                          "uniformly (references follow).")
        self.rot_axis = QComboBox()
        self.rot_axis.setObjectName("rotate_axis")
        for a in "xyz":
            self.rot_axis.addItem(a, a)
        self.rot_axis.setCurrentIndex(2)
        self.rot_axis.setToolTip("Axis of the rotation (through the bounding-box centre)")
        self.rot_deg = QuantitySpinBox("angle", "deg", minimum=-180, maximum=180, step=5, decimals=1)
        self.rot_deg.setObjectName("rotate_deg3")
        self.rot_deg.set_si_value(90.0)
        rot = small_button("Rotate", "Rotate about the bounding-box centre (right-hand rule); a model on "
                                     "the road stays on it", ROW_BUTTON_W)
        rot.setObjectName("rotate_button3")
        rot.clicked.connect(self._rotate)
        self.rot_axis.setFixedWidth(52)
        g.add_row("Rotate about", hbox(self.rot_axis, self.rot_deg, rot, stretch=1))
        self.orient = QComboBox()
        self.orient.setObjectName("orient_preset")
        for k, t in ORIENTATIONS:
            self.orient.addItem(t, k)
            self.orient.setItemData(self.orient.count() - 1, ORIENTATION_TIPS.get(k, ""),
                                    Qt.ItemDataRole.ToolTipRole)
        self.orient.setToolTip("Re-orient by a whole rotation; the model is then placed as on import "
                               "(nose at x = 0, centred on y = 0, on the road)")
        apply = small_button("Apply", "Apply the orientation", ROW_BUTTON_W)
        apply.setObjectName("orient_apply")
        apply.clicked.connect(self._orient)
        g.add_row("Orientation", hbox(self.orient, apply))
        self.on_road_box = QCheckBox("Stands on the road")
        self.on_road_box.setObjectName("on_road")
        self.on_road_box.toggled.connect(self._road_toggled)
        self._road_row = g.add_row("Road", self.on_road_box,
                                   tooltip="A generic body may stand on the road plane z = 0 (a rider, a "
                                           "trailer) or fly free (a drone, a sphere)")
        self.clearance = QuantitySpinBox("length", lu, minimum=0.0, maximum=100.0, step=0.005)
        self.clearance.setObjectName("clearance3")
        self.clearance.setToolTip(CLEARANCE_TIP)
        place = small_button("Place", "Move the model vertically so the lowest point of the body is this "
                                      "high above the road; wheels stay on the road", ROW_BUTTON_W)
        place.setObjectName("place_button3")
        place.clicked.connect(lambda: self.editRequested.emit(
            {"op": "place_on_road", "clearance": self.clearance.si_value()},
            f"Place {self.clearance.text()} above the road"))
        self._clearance_row = g.add_row("Ground clearance", hbox(self.clearance, place), tooltip=CLEARANCE_TIP)
        centre = small_button("Centre on y = 0", "Centre the bounding box on the symmetry plane y = 0")
        centre.setObjectName("centre_y_button")
        centre.clicked.connect(lambda: self.editRequested.emit({"op": "centre_y"}, "Centre on y = 0"))
        g.add_row("Lateral position", hbox(centre, stretch=None))
        self.mirror_axis = QComboBox()
        self.mirror_axis.setObjectName("mirror_axis")
        for ax, text in (("x", "Front / back"), ("y", "Left / right"), ("z", "Top / bottom")):
            self.mirror_axis.addItem(text, ax)
        self.mirror_axis.setToolTip("Reflect across the mid-plane of the bounding box normal to this direction")
        mirror = small_button("Mirror", "Mirror the model (one undo step)", ROW_BUTTON_W)
        mirror.setObjectName("mirror_button3")
        mirror.clicked.connect(lambda: self.editRequested.emit(
            {"op": "mirror", "axis": str(self.mirror_axis.currentData())},
            f"Mirror {self.mirror_axis.currentText().lower()}"))
        g.add_row("Mirror", hbox(self.mirror_axis, mirror))
        for c in (self.orient, self.mirror_axis):
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(6)

    def _build_refs(self) -> None:
        g = self.grid.add_group("References", key="references")
        self.refs_group = g
        self.kind_combo = QComboBox()
        self.kind_combo.setObjectName("kind_combo3")
        for k, t in KIND3_ITEMS:
            self.kind_combo.addItem(t, k)
        self.kind_combo.currentIndexChanged.connect(self._kind_changed)
        g.add_row("Kind", self.kind_combo,
                  tooltip="Vehicles stand on the road with their nose at x = 0 and moments about the road "
                          "under mid-length; generic bodies use their bounding-box centre")
        lu = display_unit("length")
        self.ref_area = _area_box()
        self.ref_area.setObjectName("ref_area")
        self.ref_area.valueChanged.connect(lambda v: self._ref("ref_area", float(v)))
        self.length_ref = QuantitySpinBox("length", lu, minimum=1e-6, maximum=1e4, step=0.001)
        self.length_ref.setObjectName("length_ref3")
        self.length_ref.siValueChanged.connect(lambda v: self._ref("length_ref", float(v)))
        self.span_ref = QuantitySpinBox("length", lu, minimum=1e-6, maximum=1e4, step=0.001)
        self.span_ref.setObjectName("span_ref")
        self.span_ref.siValueChanged.connect(lambda v: self._ref("span_ref", float(v)))
        self.mref = [QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.001) for _ in range(3)]
        for i, w in enumerate(self.mref):
            w.setObjectName(f"mref3_{'xyz'[i]}")
            w.siValueChanged.connect(lambda _v: self._ref("moment_ref3", tuple(b.si_value() for b in self.mref)))
        self.auto: dict[str, QToolButton] = {}
        rows = (("ref_area", "Reference area", self.ref_area,
                 "Force coefficients use q × A. Auto: the frontal area (all parts projected along x)."),
                ("length_ref", "Reference length", self.length_ref,
                 ("Reynolds number and the pitching-moment arm. Auto: the length, or the diameter / side "
                  "of a compact bluff body.")),
                ("span_ref", "Span / width", self.span_ref,
                 "Arm of the rolling and yawing moments. Auto: the width."),
                ("moment_ref3", "Moment point x", self.mref[0],
                 ("Moments are taken about this point. Auto: mid-length on the road (vehicles) or the "
                  "bounding-box centre.")))
        self._ref_rows = {}
        for f, text, w, tip in rows:
            b = _toggle("Auto", "Follow the geometry; typing a value fixes it")
            b.setObjectName(f"auto_{f}")
            b.clicked.connect(lambda on, f=f: self.autoRefToggled.emit(f, bool(on)))
            self.auto[f] = b
            self._ref_rows[f] = g.add_row(text, hbox(w, b), tooltip=tip, key=f)
        g.add_row("Moment point y", hbox(self.mref[1], _spacer(ROW_BUTTON_W)))
        g.add_row("Moment point z", hbox(self.mref[2], _spacer(ROW_BUTTON_W)))
        # "Symmetry: [x] About y = 0": the row label says what; the longer "Symmetric about y = 0"
        # made this the widest row and pushed the inspector 4 px past its column at 1280 px
        self.symmetric = QCheckBox("About y = 0")
        self.symmetric.setAccessibleName("Symmetric about y = 0")
        self.symmetric.setObjectName("symmetric3")
        self.symmetric.clicked.connect(lambda on: self.refEdited.emit("symmetric", bool(on)))
        b = _toggle("Auto", "Decide from the measured mirror error (within 0.5 % of the largest dimension)")
        b.setObjectName("auto_symmetric")
        b.clicked.connect(lambda on: self.autoRefToggled.emit("symmetric", bool(on)))
        self.auto["symmetric"] = b
        g.add_row("Symmetry", hbox(self.symmetric, b),
                  tooltip="A symmetric design at zero yaw is simulated as half a domain with a symmetry "
                          "plane at y = 0, which halves the cells")
        self.symmetry_note = label("", "muted", wrap=True)
        self.symmetry_note.setObjectName("symmetry_note")
        g.add_row("Mirror error", self.symmetry_note,
                  tooltip="RMS distance between the surface mirrored across y = 0 and the surface")

    def _build_parts(self) -> None:
        g = self.grid.add_group("Parts", key="parts")
        self.parts_group = g
        t = QTableWidget(0, 4)
        t.setObjectName("parts_table")
        t.setHorizontalHeaderLabels(["Part", "Role", "Triangles", "Surface"])
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(theme.ROW_H + 4)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setShowGrid(False)
        t.setWordWrap(False)
        t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hh = t.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        t.setColumnWidth(1, 84)                  # room for "Wheel" in the role combo
        for c in (2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        t.itemSelectionChanged.connect(self._part_selected)
        self.parts = t
        g.add_widget(t)
        self.part_detail = label("", "muted", wrap=True)
        self.part_detail.setObjectName("part_detail")
        g.add_widget(self.part_detail)

    def _build_diagnostics(self) -> None:
        g = self.grid.add_group("Diagnostics", key="diagnostics")
        self.diag_group = g
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 2, 0, 0)
        v.setSpacing(6)
        self.diag_summary = StatusLine()
        self.diag_summary.setObjectName("diag_summary3")
        v.addWidget(self.diag_summary)
        self._diag_items = QVBoxLayout()
        self._diag_items.setSpacing(6)
        v.addLayout(self._diag_items)
        g.add_widget(box)

    # -- loading ----------------------------------------------------------------------
    def design(self) -> Design | None:
        return self._design

    def set_design(self, design: Design | None) -> None:
        """Show ``design`` (a meshed 3D design); slow statistics follow on a thread."""
        self._design = design
        if design is None:
            return
        self._loading = True
        try:
            self._load_source(design)
            self._load_transform(design)
            self._load_refs(design)
            self._load_parts(design)
        finally:
            self._loading = False
        sig = self._analysis_key(design)
        if sig != self._analysis_sig:
            self._analysis = None
            self._show_analysis(None)
            self._check_timer.start()
        else:
            self._show_analysis(self._analysis)

    @staticmethod
    def _analysis_key(design: Design) -> tuple:
        """What the checks depend on: the arrays and roles, the references and flags."""
        return (design.id, tuple((id(m.vertices), id(m.faces), m.role) for m in design.meshes), design.kind,
                round(design.ref_area, 12), round(design.length_ref, 12), round(design.span_ref, 12),
                bool(design.symmetric), tuple(design.source.get("import_notes", [])))

    def _load_source(self, d: Design) -> None:
        src = d.source if isinstance(d.source, dict) else {}
        typ = str(src.get("type", ""))
        # a starter case reads like the 2D Source group's (geometry.source_type_text), not "Physics_template"
        self.src_type.setText({"import": "Imported mesh", "primitive": "Primitive shape",
                               "physics_template": "Starter case"}.get(typ, typ.replace("_", " ").capitalize()
                                                                      or "Mesh"))
        file = str(src.get("file") or "")
        tip = str(src.get("path") or file)
        if src.get("sha256"):
            # the exact bytes read (api.file_source): a result traces to the file after it moved or changed
            tip = f"{tip}\nSHA-256 {src['sha256']}" if tip else f"SHA-256 {src['sha256']}"
        self.src_file.set_full_text(file or "—", tip or None)
        imported = typ == "import"
        self._file_row.set_visible(imported)
        self._format_row.set_visible(imported)
        self._units_row.set_visible(imported)
        fmt = str(src.get("format") or "").lower().lstrip(".")
        self.src_format.setText(FORMAT3.get(f".{fmt}" if fmt else Path(file).suffix.lower(), "—")
                                if (fmt or file) else "—")
        self.src_units.setText(unit_text(src.get("scale_to_m")))
        s = size3(d)
        u = display_unit("length")
        self.src_size.setText(" × ".join(units.fmt(v, "length", u, 4).rsplit(" ", 1)[0] for v in s) + f" {u}"
                              if s else "—")
        ntri = sum(len(m.faces) for m in d.meshes)
        n = len(d.meshes)
        self.src_triangles.setText(f"{ntri:,} in {n} part{'s' if n != 1 else ''}")
        edits = len(src.get("edits", []))
        self.src_edits.setText(f"{edits} since import" if imported else str(edits))

    def _load_transform(self, d: Design) -> None:
        s = size3(d)
        if s:
            self.scale_length.set_si_value(s[0])
        road = on_road(d)
        vehicle = d.kind == "vehicle"
        self._road_row.set_visible(not vehicle)
        self.on_road_box.blockSignals(True)
        self.on_road_box.setChecked(road)
        self.on_road_box.blockSignals(False)
        self._clearance_row.set_visible(road)
        if road and _solid(d):
            self.clearance.set_si_value(max(clearance(d), 0.0))

    def _load_refs(self, d: Design) -> None:
        self.kind_combo.blockSignals(True)
        self.kind_combo.setCurrentIndex(max(self.kind_combo.findData(d.kind), 0))
        self.kind_combo.blockSignals(False)
        _set_area(self.ref_area, float(d.ref_area))
        self.length_ref.set_si_value(float(d.length_ref))
        self.span_ref.set_si_value(float(d.span_ref))
        for w, v in zip(self.mref, d.moment_ref3, strict=True):
            w.set_si_value(float(v))
        manual = set(d.source.get("manual_refs", []))
        for f, b in self.auto.items():
            b.setChecked(f not in manual)
        self.symmetric.setChecked(bool(d.symmetric))
        err = d.source.get("symmetry_error")
        s = size3(d)
        if err is None or not s:
            self.symmetry_note.setText("Not measured")
        else:
            limit = SYMMETRY_TOL * max(s)
            shown = "below 0.001 mm" if float(err) < 1e-6 else f"{mm_text(float(err))} RMS"
            self.symmetry_note.setText(f"{shown} (limit {mm_text(limit)})")
            self.symmetry_note.setToolTip("RMS distance between the surface mirrored across y = 0 and the "
                                          "surface; the design counts as symmetric below 0.5 % of its "
                                          "largest dimension")
        compact = bool(d.source.get("compact"))
        self._ref_rows["length_ref"].label.setToolTip(
            "Reynolds number and the pitching-moment arm. Auto: "
            + ("the cross-stream size of this compact bluff body (diameter or side)." if compact
               else "the overall length."))

    def _load_parts(self, d: Design) -> None:
        t = self.parts
        sel = t.currentRow()
        t.setRowCount(len(d.meshes))
        info = (self._analysis or {}).get("parts") if self._analysis_sig == self._analysis_key(d) else None
        for i, m in enumerate(d.meshes):
            name = QTableWidgetItem(m.name or f"Part {i + 1}")
            name.setToolTip(m.name)
            t.setItem(i, 0, name)
            combo = t.cellWidget(i, 1)
            if not isinstance(combo, QComboBox):
                combo = QComboBox()
                combo.addItem("Body", "body")
                combo.addItem("Wheel", "wheel")
                combo.setProperty("part_index", i)
                combo.currentIndexChanged.connect(lambda _i, c=combo: self._role_changed(c))
                t.setCellWidget(i, 1, combo)
                # the style's padding and drop-down button decide the width "Wheel" needs, and the
                # cell keeps a few pixels round its widget (a fixed 84 px column showed "Whee")
                combo.setMinimumWidth(combo.sizeHint().width())
                need = combo.minimumWidth() + 12
                if t.columnWidth(1) < need:
                    t.setColumnWidth(1, need)
            combo.blockSignals(True)
            combo.setCurrentIndex(1 if m.role == "wheel" else 0)
            combo.setProperty("part_index", i)
            combo.blockSignals(False)
            tri = QTableWidgetItem(f"{len(m.faces):,}")
            tri.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t.setItem(i, 2, tri)
            pi = info[i] if info and i < len(info) else None
            surf = QTableWidgetItem("…" if pi is None else ("Closed" if pi.get("watertight") else "Open"))
            if pi is not None and not pi.get("watertight"):
                surf.setToolTip(surface_text(pi))
            t.setItem(i, 3, surf)
        rows = min(len(d.meshes), 8)
        t.setFixedHeight(t.horizontalHeader().sizeHint().height() + rows * t.verticalHeader().defaultSectionSize()
                         + 2 * t.frameWidth() + 2)
        if 0 <= sel < len(d.meshes):
            t.selectRow(sel)
        self._part_selected()

    def _part_selected(self) -> None:
        d = self._design
        i = self.parts.currentRow()
        if d is None or not (0 <= i < len(d.meshes)):
            self.part_detail.setText("Select a part to see its details." if d is not None and d.meshes else "")
            return
        m = d.meshes[i]
        if m.role == "wheel":
            r_text = ""
            with contextlib.suppress(Exception):
                r_text = f" · radius {_g3().wheel_params(m)[2]:.3f} m"
            c = m.center or (0.0, 0.0, 0.0)
            ax = m.axis or (0.0, 1.0, 0.0)
            axis = next((f"{'+' if v > 0 else '−'}{'xyz'[k]}" for k, v in enumerate(ax) if abs(v) > 0.999), None)
            axis = axis or "(" + ", ".join(f"{v:.2f}" for v in ax) + ")"
            self.part_detail.setText(f"Wheel: centre x {c[0]:.3f}, y {c[1]:.3f}, z {c[2]:.3f} m · axle {axis}"
                                     f"{r_text}. Spins with a moving road.")
        else:
            info = (self._analysis or {}).get("parts") or []
            pi = info[i] if i < len(info) else None
            extra = f" · {surface_text(pi)}" if pi is not None else ""
            b = m.bounds()
            self.part_detail.setText(f"{m.name}: {len(m.vertices):,} vertices, {len(m.faces):,} triangles{extra}; "
                                     f"x {b[0]:.3f} to {b[1]:.3f} m.")

    # -- the slow analysis -------------------------------------------------------
    def _start_analysis(self) -> None:
        d = self._design
        if d is None:
            return
        self._pending_sig = self._analysis_key(d)
        self._pending_token = self._analyser.start(clone_design(d))

    def _analysis_done(self, token: int, result: dict) -> None:
        if token != self._pending_token or self._design is None:
            return                                         # a newer request is on its way
        self._analysis = result
        self._analysis_sig = getattr(self, "_pending_sig", None)
        if self._analysis_sig == self._analysis_key(self._design):
            self._show_analysis(result)
            self._load_parts(self._design)

    def analysis_pending(self) -> bool:
        return self._check_timer.isActive() or (self._analysis is None and self._design is not None)

    def wait_analysis(self, timeout: float = 30.0) -> bool:
        """Block (processing events) until the checks of the shown design arrived (tests)."""
        import time

        from PySide6.QtWidgets import QApplication
        if self._check_timer.isActive():
            self._check_timer.stop()
            self._start_analysis()
        t0 = time.monotonic()
        while self._analysis is None and time.monotonic() - t0 < timeout:
            QApplication.processEvents()
            time.sleep(0.005)
        return self._analysis is not None

    def _show_analysis(self, result: dict | None) -> None:
        clear_layout(self._diag_items)
        d = self._design
        if result is None:
            self.diag_summary.set("info", "Checking…")
            self.src_surface.setText("…")
            return
        parts = result.get("parts") or []
        open_parts = [p for p in parts if not p.get("watertight")]
        if not parts:
            self.src_surface.setText("—")
        elif not open_parts:
            self.src_surface.setText("Closed")
        else:
            holes = sum(int(p.get("holes", 0)) for p in open_parts)
            edges = sum(int(p.get("boundary_edges", 0)) for p in open_parts)
            self.src_surface.setText(f"Open: {len(open_parts)} part{'s' if len(open_parts) != 1 else ''}, "
                                     f"{holes} hole{'s' if holes != 1 else ''}, {edges:,} open edges")
        msgs = [(classify_check(t), t) for t in merge_checks(result.get("checks", []))]
        errors = sum(1 for lv, _ in msgs if lv == "error")
        warns = sum(1 for lv, _ in msgs if lv == "warning")
        if errors:
            self.diag_summary.set("error", f"{errors} error{'s' if errors != 1 else ''}"
                                  + (f", {warns} warning{'s' if warns != 1 else ''}" if warns else ""))
        elif warns:
            self.diag_summary.set("warning", f"{warns} warning{'s' if warns != 1 else ''}")
        else:
            self.diag_summary.set("ok", "No problems found")
        for lv, text in msgs:
            line = StatusLine(text, lv)
            line.setProperty("diag_level", lv)
            self._diag_items.addWidget(line)
        if d is not None and (errors or warns):
            self.diag_group.set_expanded(True)

    def diagnostics(self) -> list[tuple[str, str]]:
        """(level, text) of the shown checks (empty while checking)."""
        if self._analysis is None:
            return []
        return [(classify_check(t), t) for t in merge_checks(self._analysis.get("checks", []))]

    # -- widget edits -> requests ------------------------------------------------------
    def _scale_edited(self, value: float) -> None:
        d = self._design
        if self._loading or d is None:
            return
        s = size3(d)
        if s and not math.isclose(s[0], value, rel_tol=1e-9):
            self.editRequested.emit({"op": "scale_to", "length": float(value)},
                                    f"Scale to {self.scale_length.text()}")

    def _rotate(self) -> None:
        axis = str(self.rot_axis.currentData())
        deg = self.rot_deg.si_value()
        if deg:
            self.editRequested.emit({"op": "rotate", "axis": axis, "deg": deg}, f"Rotate {deg:g}° about {axis}")

    def _orient(self) -> None:
        key = str(self.orient.currentData())
        self.editRequested.emit({"op": "orient", "preset": key}, f"Orientation: {self.orient.currentText()}")

    def _road_toggled(self, on: bool) -> None:
        if self._loading or self._design is None:
            return
        if on:
            self.editRequested.emit({"op": "place_on_road", "clearance": 0.0}, "Place on the road")
        else:
            self.editRequested.emit({"op": "free"}, "Take off the road")

    def _kind_changed(self, _i: int) -> None:
        if not self._loading:
            kind = str(self.kind_combo.currentData())
            self.editRequested.emit({"op": "kind", "kind": kind}, f"Kind: {self.kind_combo.currentText()}")

    def _ref(self, field: str, value: Any) -> None:
        if not self._loading:
            self.refEdited.emit(field, value)

    def _role_changed(self, combo: QComboBox) -> None:
        if self._loading:
            return
        i = int(combo.property("part_index"))
        role = str(combo.currentData())
        d = self._design
        name = d.meshes[i].name if d is not None and i < len(d.meshes) else f"part {i + 1}"
        self.editRequested.emit({"op": "role", "index": i, "role": role}, f"{name}: {role}")


__all__ = [
    "KIND3_ITEMS",
    "ORIENTATIONS",
    "REF_FIELDS3",
    "MeshInspector",
    "analyse_design3d",
    "apply_edit3d",
    "area_text",
    "auto_references3d",
    "classify_check",
    "clearance",
    "clone_design",
    "design_summary3d",
    "edit_design3d",
    "measure_symmetry",
    "merge_checks",
    "on_road",
    "refresh_references3d",
    "size3",
    "surface_text",
    "symmetric_from_error",
    "unit_text",
]
