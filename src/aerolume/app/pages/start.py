"""Start page: the IDE-style welcome shown when no project is open.

Plain and quick to scan (docs/ui_guidelines.md section 4): the product line,
then the kinds of study as one row of tiles - airfoil, vehicle, 3D body from
a mesh, wing, heated room and electronics enclosure - each with a small
schematic of what it sets up, its dimension and one sentence of what it starts
with (the row wraps to two when a narrow window or an open panel leaves too
little width for every tile's title); below them the
recent projects (type, folder, last modified) and the bundled examples, 3D
first, beside a short column of the other ways in (open, example, more study
types) and the guides; a footer with Documentation, Validation and Licence.
Each help link appears once. The "Open example" action is hidden when no
examples are installed, rather than greyed out with no reason.

The tiles are a choice list, not feature marketing: their schematics show
the setup (the body, the free stream or the closed box, the heated part in
the highlight colour), and the 3D kinds are studies in their own right rather
than hidden behind an import command, because "can it do 3D, can it do
heat?" is what the first screen has to answer. The 2D enclosure has a tile of
its own because the 3D room takes tens of minutes on a Draft GPU run, and a
first heat-transfer study should finish while the user watches (the enclosure
took 204 s on an RTX 4060 Ti, v3.3 audit). An empty project is in the New
study dialog (More study types…).

The project-level actions live here as plain functions so the main window's
menus and the tests share one code path with the page:

* :func:`create_study` - new project folder + starter design + default simulations
  (:data:`STUDY_TYPES` lists the kinds; a "mesh" study starts empty and the window
  runs the mesh import into it, see ``MainWindow.new_study``).
* :func:`open_project` - open a ``.aerolume`` folder and remember it as recent.
* :func:`example_projects` / :func:`open_example` - bundled examples, opened as a copy.
* :func:`recent_projects` / :func:`remove_recent` / :func:`clear_recent` - the recent list
  (project store first, app settings as fallback).
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ... import __version__
from ...model import AirfoilSettings, FlowConditions, GroundConfig, Simulation, WingSection, WingSpec
from ...simulation.defaults import default_cfd_settings, default_simulation
from .. import settings as app_settings
from .. import theme
from ..widgets.common import (
    SPACING,
    Divider,
    ElidedLabel,
    button,
    label,
)

PROJECT_SUFFIX = ".aerolume"
#: extensions of the 3D surface meshes a "mesh" study / File ▸ Import geometry ▸ 3D mesh read
MESH_EXTENSIONS = (".stl", ".obj", ".ply")


@dataclass(frozen=True)
class StudyType:
    """One kind of new study: key, list title, icon, one-line description, default name."""
    key: str
    title: str
    icon: str
    description: str
    default_name: str


#: The kinds of new study, in the order the Start page and the New study dialog list them.
STUDY_TYPES: tuple[StudyType, ...] = (
    StudyType("airfoil", "Airfoil (2D)", "airfoil",
              "NACA 2412 section: a viscous polar at Re = 1.0×10⁶ and a CFD run at α = 4°.", "Airfoil study"),
    StudyType("vehicle", "Vehicle (2D)", "car",
              "Sedan side profile: CFD at 30 m/s over a moving road with rotating wheels.", "Vehicle study"),
    StudyType("mesh", "3D body from a mesh", "cube",
              "An STL, OBJ or PLY surface (car, drone, rider) in a 3D wind tunnel: "
              "a Draft CFD run at 30 m/s, a few minutes on a GPU.", "3D study"),
    StudyType("wing", "Wing (3D)", "wing",
              "Tapered wing, 2 m span, NACA 2412: a vortex-lattice polar from −4° to 12° at 30 m/s, "
              "in seconds.", "Wing study"),
    StudyType("room", "Heated room (3D)", "thermometer",
              "A closed 4 × 3 × 2.5 m room: a 200 W radiator under a 12 °C window, a ceiling fan, "
              "supply and extract air. 3D CFD.",
              "Room study"),
    StudyType("enclosure", "Electronics enclosure (2D)", "chip",
              "A closed box: a 40 W/m chip on a circuit board, an intake fan and an exhaust vent. "
              "2D CFD, about 3½ minutes on a GPU.",
              "Enclosure study"),
    StudyType("blank", "Empty project", "new",
              "No geometry. Import a 2D outline or a 3D mesh, or start from a template.", "New study"),
)
_TYPES = {t.key: t for t in STUDY_TYPES}
#: kind -> short name (kept for callers that only need the keys)
KINDS = {"airfoil": "Airfoil", "vehicle": "Vehicle", "mesh": "3D body", "wing": "Wing", "room": "Room",
         "enclosure": "Enclosure", "blank": "Blank"}
#: study kinds built by the physics templates (app/dialogs/physics_templates.py, DESIGN.md section 10)
PHYSICS_KINDS = {"room": "room", "enclosure": "enclosure"}


def physics_templates_available() -> bool:
    """Whether the heat-transfer study kinds can be created (their template module imports)."""
    try:
        from ..dialogs import physics_templates  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - the 2D/3D aerodynamics studies still work
        return False


def available_study_types() -> tuple[StudyType, ...]:
    """:data:`STUDY_TYPES` minus the kinds this build cannot create (no physics templates)."""
    if physics_templates_available():
        return STUDY_TYPES
    return tuple(t for t in STUDY_TYPES if t.key not in PHYSICS_KINDS)
# Quality of the starter CFD simulation; see _starter_simulations for why it is not "standard".
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# project actions (no widgets: shared by the start page, menus and tests)
# ---------------------------------------------------------------------------


def safe_name(name: str) -> str:
    """A project name made safe for a folder name on every OS (keeps spaces)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().rstrip(".")
    return s or "Untitled"


def project_folder(folder: str | Path, name: str) -> Path:
    """The ``<folder>/<Name>.aerolume`` directory a new project of that name would use."""
    return Path(folder) / f"{safe_name(name)}{PROJECT_SUFFIX}"


def unique_project_name(folder: str | Path, base: str) -> str:
    """``base``, or ``base 2``, ``base 3``... whichever does not exist in ``folder`` yet."""
    name, k = base, 2
    while project_folder(folder, name).exists():
        name, k = f"{base} {k}", k + 1
    return name


def _starter_design(kind: str):
    """The design a new study starts from, or (None, reason) when geometry is unavailable.

    Imported lazily: the geometry package is optional at shell level, and a
    missing or broken module must degrade to an empty project, not a crash.
    """
    try:
        if kind == "airfoil":
            try:
                from ...geometry import design_from_airfoil, naca  # DESIGN.md 5.1 package API
            except ImportError:
                # same functions straight from their modules (package re-exports not in place)
                from ...geometry.airfoils import naca
                from ...geometry.design import design_from_airfoil
            return design_from_airfoil(naca("2412"), name="NACA 2412", chord=1.0), None
        if kind == "vehicle":
            from ...geometry import templates
            return templates.make("sedan"), None
        if kind == "wing":
            from ...geometry3d import design_from_wing
            return design_from_wing(default_wing_spec(), "Tapered wing"), None
    except Exception as exc:          # noqa: BLE001 - any failure -> empty project + message
        return None, f"{type(exc).__name__}: {exc}"
    return None, None


def default_wing_spec() -> WingSpec:
    """The planform a new wing starts from.

    The Geometry page's planform editor owns the definition
    (``pages.geometry.default_wing_spec``: the wing its "New wing" adds), so a
    wing study and a wing added later in the same project are the same wing.
    When that page cannot be imported, :func:`_fallback_wing_spec` stands in.
    """
    try:
        from .geometry import default_wing_spec as editor_default
        return editor_default()
    except Exception as exc:  # noqa: BLE001 - a broken Geometry page must not stop a new study
        _log.debug("Geometry page default wing unavailable: %s", exc)
        return _fallback_wing_spec()


def _fallback_wing_spec() -> WingSpec:
    """A tapered wing with an unswept quarter-chord line and no twist.

    2 m span, 0.25 m root and 0.15 m tip chord (taper 0.6, aspect ratio 10),
    NACA 2412 throughout: a sailplane-model wing whose chord Reynolds numbers
    at 30 m/s (3–5×10⁵) sit where the section polars are reliable. The
    leading edge sweeps back by a quarter of the chord change, so the
    quarter-chord line is straight and the lattice and lifting-line methods
    agree (lifting line assumes an unswept quarter-chord line).
    """
    root, tip, half = 0.25, 0.15, 1.0
    return WingSpec(sections=[WingSection(y=0.0, chord=root, airfoil="2412"),
                              WingSection(y=half, chord=tip, x_le=0.25 * (root - tip), airfoil="2412")],
                    symmetric=True)


def _starter_simulations(kind: str, design, device: str, precision: str) -> list[Simulation]:
    """Sensible first simulations, so "press run" works straight after creating a study.

    Airfoil: a viscous polar at Re = 1e6 (fast, XFOIL-class) plus a CFD run at
    4 degrees at the same speed. Vehicle: CFD at 30 m/s with a moving ground and
    spinning wheels - the configuration a rolling-road wind tunnel would use.

    CFD settings come from :func:`aerolume.simulation.defaults.default_cfd_settings`,
    like every other new simulation: Draft quality, and for the airfoil at Re 1e6
    the capped Reynolds mode. The first run is how a new user decides whether the
    program works at all: Draft finishes the NACA 2412 case in about 10 s on a
    mid-range GPU, while Standard (1250 x 640 cells) took about 3 minutes there
    and far longer on a CPU, and minutes of a run that has not yet produced
    numbers read as "it does not run". The Quality combo on Setup steps up once
    the case looks right.

    The CFD run comes first, because the first simulation of a project is the
    one that is selected when it opens, and so the one the first F5 runs. With
    the polar first, that press showed four empty plots and "0 of 19 angles"
    for the ~9 s the first (cold-started) angle takes: read as "it does not
    run". The CFD run shows a live flow within a second.
    """
    if kind == "airfoil":
        cond = FlowConditions()
        chord = float(getattr(design, "length_ref", 1.0) or 1.0)
        # speed giving Re = 1e6 on this chord in ISA sea-level air (14.6 m/s for 1 m)
        cond.speed = 1.0e6 * cond.fluid.viscosity / (cond.fluid.density * chord)
        polar = Simulation(name="Polar · Re 1.0×10⁶", design_id=design.id, analysis="polar",
                           conditions=cond, airfoil=AirfoilSettings())
        cfd_cond = FlowConditions(fluid=cond.fluid, speed=cond.speed, alpha_deg=4.0)
        # Draft and, at Re 1e6, the capped Reynolds mode: the same start as every other
        # new simulation (simulation.defaults), so the first Run gives a resolved flow
        # in seconds instead of an unresolved LES that disagrees with the polar.
        cfd_sim = Simulation(name="CFD · α 4°", design_id=design.id, analysis="cfd", conditions=cfd_cond,
                             cfd=default_cfd_settings(design, cfd_cond, device=device, precision=precision))
        return [cfd_sim, polar]
    if kind == "vehicle":
        cond = FlowConditions(speed=30.0, alpha_deg=0.0)
        sim = Simulation(name="CFD · 30 m/s", design_id=design.id, analysis="cfd", conditions=cond,
                         ground=GroundConfig(enabled=True, moving=True, rotate_wheels=True),
                         cfd=default_cfd_settings(design, cond, device=device, precision=precision))
        return [sim]
    if kind == "wing":
        # the lattice polar (seconds, flight Reynolds number); a 3D CFD run of the lofted
        # wing is one context-menu entry away, but it is not what a first Run should start
        return [default_simulation(design, "Wing polar · 30 m/s", FlowConditions(speed=30.0),
                                   device=device, precision=precision)]
    if kind == "mesh":
        # simulation.defaults puts vehicles on a moving road with spinning wheels, starts at
        # Draft (a few minutes for a car on a mid-range GPU) and lets the planner use the
        # symmetry plane when the body is mirror-symmetric
        return [default_simulation(design, "CFD 3D · 30 m/s", FlowConditions(speed=30.0),
                                   device=device, precision=precision)]
    return []


def starter_simulations(kind: str, design, *, device: str | None = None,
                        precision: str | None = None) -> list[Simulation]:
    """The simulations a new study of ``kind`` gives ``design`` (public for the window's
    mesh import: designs imported into a new "mesh" study get the same start)."""
    s = app_settings.get()
    return _starter_simulations(kind, design, device or s.device, precision or s.precision)


def create_study(kind: str, name: str, folder: str | Path, *, device: str | None = None,
                 precision: str | None = None, designs: list | None = None) -> tuple[object, list[str]]:
    """Create a project with a starter design and default simulations.

    ``kind`` is a :data:`STUDY_TYPES` key. A "mesh" study has no built-in
    shape: it holds ``designs`` (3D designs already imported, each with a
    starter CFD run) or starts empty for the window's mesh import to fill.

    Returns ``(project, messages)``: ``messages`` explains anything that could
    not be added (e.g. the geometry package failed to import), for the status
    bar. Raises ``FileExistsError`` if the project folder already exists and
    ``ImportError`` if the project store itself is unavailable.
    """
    from ...project import Project

    if kind not in KINDS:
        raise ValueError(f"unknown study kind {kind!r}")
    s = app_settings.get()
    device = device or s.device
    precision = precision or s.precision
    path = project_folder(folder, name)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    project = Project.create(path, name.strip() or "Untitled")
    messages: list[str] = []
    if kind == "mesh":
        for design in designs or []:
            project.add_design(design)
            for sim in _starter_simulations(kind, design, device, precision):
                project.add_simulation(sim)
    elif kind in PHYSICS_KINDS:
        try:
            from ..dialogs.physics_templates import build_template
            design, sims = build_template(PHYSICS_KINDS[kind], device=device, precision=precision)
        except Exception as exc:  # noqa: BLE001 - an empty project + the reason, never a crash
            messages.append(f"Starter case unavailable ({type(exc).__name__}: {exc}); "
                            "add a shape on the Geometry page.")
        else:
            project.add_design(design)
            for sim in sims:
                sim.design_id = design.id
                project.add_simulation(sim)
    elif kind != "blank":
        design, err = _starter_design(kind)
        if design is None:
            messages.append(f"Starter geometry unavailable ({err}); add a shape on the Geometry page.")
        else:
            project.add_design(design)
            for sim in _starter_simulations(kind, design, device, precision):
                project.add_simulation(sim)
    project.save()
    _remember(project_path(project) or path)
    s.last_dir = Path(folder)
    return project, messages


def project_path(project) -> Path | None:
    """The folder a Project lives in, whatever the store calls the attribute."""
    for attr in ("path", "root", "folder", "directory"):
        v = getattr(project, attr, None)
        if v:
            return Path(v)
    return None


def project_name(project) -> str:
    name = getattr(project, "name", None)
    if name:
        return str(name)
    p = project_path(project)
    return p.stem if p else "Untitled"


def open_project(path: str | Path):
    """Open a project folder (or its ``project.json``) and record it as recent."""
    from ...project import Project

    p = Path(path)
    if p.is_file():
        p = p.parent
    project = Project.open(p)
    _remember(project_path(project) or p)
    return project


def _remember(path: Path) -> None:
    """Record a project that was just created or opened: the recent lists and the desktop.

    The desktop part (``platform.note_project_opened``) writes the
    ``project.aerolume`` marker the installer's file association opens, and
    puts it in the taskbar jump list of an installed build (a no-op from
    source). Every create and open goes through here, so this is the one call.
    A folder inside the installed program (the bundled examples, opened in
    place by a script) is left untouched: the program files are not the user's.
    """
    app_settings.get().add_recent(path)
    try:
        from .. import platform as plat
        if not _inside_program(Path(path), plat):
            plat.note_project_opened(path)
    except Exception as exc:  # noqa: BLE001 - desktop sugar must never stop a project opening
        _log.debug("note_project_opened failed: %s", exc)
    try:
        from ... import project as store
        add = getattr(store, "add_recent", None)
        if callable(add):
            add(path)
    except Exception as exc:  # noqa: BLE001 - the store's recent list is a nicety
        _log.debug("project store add_recent failed: %s", exc)


def _inside_program(path: Path, plat) -> bool:
    """Whether ``path`` lies in the package or the installed bundle (never written to)."""
    try:
        target = path.resolve()
        roots = {Path(__file__).resolve().parents[2], Path(plat.package_dir()).resolve(),
                 Path(plat.bundle_dir()).resolve()}
    except (OSError, AttributeError, RuntimeError):
        return True
    return any(target == r or r in target.parents for r in roots)


def recent_projects(limit: int = 8) -> list[Path]:
    """Recent project folders, newest first: the project store's list when it has
    one (``aerolume.project.recent_projects``), merged with the app settings list."""
    out: list[Path] = []
    try:
        from ... import project as store
        fn = getattr(store, "recent_projects", None)
        if callable(fn):
            for item in fn() or []:
                p = Path(getattr(item, "path", item))
                if p.exists() and p not in out:
                    out.append(p)
    except Exception as exc:  # noqa: BLE001
        _log.debug("project store recent_projects failed: %s", exc)
    for s in app_settings.get().recent_projects():
        p = Path(s)
        if p not in out:
            out.append(p)
    return out[:limit]


def clear_recent() -> None:
    """Empty both recent lists (the project store's and the app settings')."""
    app_settings.get().clear_recent()
    try:
        from ... import project as store
        fn = getattr(store, "clear_recent", None)
        if callable(fn):
            fn()
    except Exception as exc:  # noqa: BLE001
        _log.debug("project store clear_recent failed: %s", exc)


def examples_dirs() -> list[Path]:
    """Folders that may hold bundled example projects (package data, then the repo)."""
    here = Path(__file__).resolve()
    pkg = here.parents[2]                       # src/aerolume
    return [pkg / "examples" / "projects", pkg.parents[1] / "examples" / "projects"]


def example_projects() -> list[Path]:
    """Every example project folder (``examples/projects/*``) that exists, by name."""
    seen: dict[str, Path] = {}
    for d in examples_dirs():
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.is_dir() and (p / "project.json").exists() and p.name not in seen:
                    seen[p.name] = p
    return list(seen.values())


def open_example(example: str | Path, dest_root: str | Path | None = None):
    """Copy an example into ``~/Documents/Aerolume/Examples`` and open the copy.

    Examples ship read-only inside the install; opening the original would let a
    first experiment silently overwrite the reference. Re-opening the same
    example reuses the earlier copy (with the user's edits).
    """
    src = Path(example)
    root = Path(dest_root) if dest_root else app_settings.default_projects_dir() / "Examples"
    dst = root / src.name
    if not dst.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    return open_project(dst)


@dataclass
class ProjectSummary:
    """What the recent / examples lists show without opening a project."""
    path: Path
    name: str
    modified: float
    designs: int
    simulations: int
    description: str
    outlines: list[np.ndarray]     # first design's bodies, for the thumbnail
    kind: str
    #: "2D", "3D" and/or "Wing" for the designs read (empty for an empty project)
    tags: frozenset[str] = field(default_factory=frozenset)
    #: every design kind read ("airfoil", "vehicle", ...)
    kinds: frozenset[str] = field(default_factory=frozenset)
    #: the physics templates the designs came from ("room", "enclosure", ...; DESIGN.md section 10)
    templates: frozenset[str] = field(default_factory=frozenset)

    @property
    def type_text(self) -> str:
        """What the recent list says the project is: "Airfoil (2D)", "Vehicle (3D)", "Wing", "2D and 3D".

        The kind word is used only when every design shares it: a car next to
        a cylinder is a "2D" study, not a "Vehicle (2D)" one.
        """
        one = next(iter(self.kinds)) if len(self.kinds) == 1 else (self.kind if not self.kinds else "")
        kind = {"airfoil": "Airfoil", "vehicle": "Vehicle", "generic": "Body", "wing": "Wing"}.get(one, "")
        if len(self.templates) == 1 and self.templates <= set(_TEMPLATE_WORDS):
            kind = _TEMPLATE_WORDS[next(iter(self.templates))]
        if not self.tags:
            return "Empty" if self.designs == 0 else (kind or "")
        if self.tags == {"Wing"}:
            return "Wing"
        if self.tags == {"2D"}:
            return f"{kind} (2D)" if kind else "2D"
        if self.tags == {"3D"}:
            return f"{kind} (3D)" if kind else "3D"
        if self.tags == {"3D", "Wing"}:
            return "3D"                         # a wing is 3D too: "2D and 3D" would be wrong
        return "2D and 3D"

    @property
    def icon(self) -> str:
        """Icon name: the 2D kind glyph, ``cube`` when there is any 3D mesh, ``wing`` for wing-only,
        ``thermometer`` for a study built from a physics template (heated body, cavity, room)."""
        if self.tags == {"Wing"}:
            return "wing"
        if self.tags & {"3D", "Wing"}:
            return "cube"
        if self.templates:
            return "thermometer"            # heated cylinder, cavity, room: a heat-transfer study
        return {"vehicle": "car", "airfoil": "airfoil"}.get(self.kind, "folder")


#: what a project built from one physics template is called in the recent list
_TEMPLATE_WORDS = {"room": "Room", "enclosure": "Enclosure", "cavity": "Cavity", "heated_cylinder": "Cylinder"}
#: scripting-API reference cases (``aerolume.api.*_case``) that are one of the templates above
_CASE_TEMPLATES = {"fan_room": "room"}


def _design_tag(d: dict) -> str:
    """"2D" / "3D" / "Wing" from a design's JSON (old files have no "dimension": 2D)."""
    if d.get("kind") == "wing":
        return "Wing"
    try:
        return "3D" if int(d.get("dimension", 2) or 2) == 3 else "2D"
    except (TypeError, ValueError):
        return "2D"


@lru_cache(maxsize=64)
def _summary_cached(path: str, mtime: float) -> ProjectSummary:
    p = Path(path)
    name, desc, n_d, n_s, kind = p.name.removesuffix(PROJECT_SUFFIX), "", 0, 0, ""
    meta = {}
    try:
        meta = json.loads((p / "project.json").read_text(encoding="utf-8"))
        name = meta.get("name") or name
        desc = meta.get("description") or meta.get("notes") or ""
    except Exception as exc:  # noqa: BLE001 - a damaged file still lists, just with less detail
        _log.debug("unreadable project.json in %s: %s", p, exc)
    designs = sorted((p / "designs").glob("*.json")) if (p / "designs").is_dir() else []
    n_d = len(designs) or len(meta.get("designs", []) or [])
    sims_dir = p / "sims"
    n_s = len([d for d in sims_dir.iterdir() if d.is_dir()]) if sims_dir.is_dir() else len(
        meta.get("simulations", []) or [])
    outlines: list[np.ndarray] = []
    tags: set[str] = set()
    kinds: set[str] = set()
    templates: set[str] = set()
    # the type column reads every design's kind and dimension (small JSON files; the
    # meshes live in .npz files that are never opened here); the cap keeps a project
    # with hundreds of variants from slowing the start page down
    for i, f in enumerate(designs[:40]):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _log.debug("unreadable design in %s: %s", p, exc)
            continue
        tags.add(_design_tag(d))
        if d.get("kind"):
            kinds.add(str(d["kind"]))
        src = d.get("source")
        if isinstance(src, dict):
            # a physics template records "template"; a reference case built through the scripting
            # API (the bundled cavity, cylinder and room examples) records "case" instead
            case = src.get("case")
            for key in (src.get("template"), _CASE_TEMPLATES.get(case, case)):
                if key in _TEMPLATE_WORDS:
                    templates.add(str(key))
                    break
        if i == 0:
            kind = d.get("kind", "")
            for b in d.get("bodies", []):
                pts = np.asarray(b.get("points", []), dtype=float).reshape(-1, 2)
                if len(pts) >= 3:
                    outlines.append(pts)
    return ProjectSummary(p, name, mtime, n_d, n_s, desc, outlines, kind, frozenset(tags), frozenset(kinds),
                          frozenset(templates))


def summarize(path: str | Path) -> ProjectSummary:
    """Light-weight summary of a project folder (cached until project.json changes)."""
    p = Path(path)
    try:
        mtime = (p / "project.json").stat().st_mtime
    except OSError:
        mtime = 0.0
    return _summary_cached(str(p), mtime)



def remove_recent(path: str | Path) -> None:
    """Drop one project from both recent lists (the folder itself is untouched)."""
    app_settings.get().remove_recent(path)
    try:
        from ... import project as store
        fn = getattr(store, "remove_recent", None)
        if callable(fn):
            fn(path)
    except Exception as exc:  # noqa: BLE001
        _log.debug("project store remove_recent failed: %s", exc)


def display_folder(path: str | Path) -> str:
    """A folder as shown in lists: the home directory abbreviated to "~" (full path in the tooltip)."""
    p = Path(path)
    try:
        return str(Path("~") / p.resolve().relative_to(Path.home().resolve()))
    except (ValueError, OSError):
        return str(p)


def format_modified(ts: float) -> str:
    """Last-modified time as "2026-10-05 14:32" (empty when unknown)."""
    if ts <= 0:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


# ---------------------------------------------------------------------------
# dialogs
# ---------------------------------------------------------------------------

class _TypeList(QListWidget):
    """The study types as a plain list (icon + title), Qt Creator's "New project" style.

    Exposes the small interface the old segmented control had (``current``,
    ``set_current``, ``keys``, ``changed``) so callers did not have to change.
    """

    changed = Signal(str)

    ROW_H = 26

    def __init__(self, types: tuple[StudyType, ...], current: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIconSize(QSize(16, 16))
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAccessibleName("Study type")
        for t in types:
            it = QListWidgetItem(theme.icon(t.icon, "text2", active="accent"), t.title)
            it.setData(Qt.ItemDataRole.UserRole, t.key)
            it.setSizeHint(QSize(0, self.ROW_H))
            it.setToolTip(t.description)
            self.addItem(it)
        self.setFixedHeight(self.ROW_H * len(types) + 2 * self.frameWidth() + 2)
        self.currentItemChanged.connect(self._item_changed)
        self.set_current(current if current in self.keys() else self.keys()[0])

    def _item_changed(self, cur: QListWidgetItem | None, _prev) -> None:
        if cur is not None:
            self.changed.emit(cur.data(Qt.ItemDataRole.UserRole))

    def keys(self) -> list[str]:
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def current(self) -> str:
        it = self.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else ""

    def set_current(self, key: str) -> None:
        for i in range(self.count()):
            if self.item(i).data(Qt.ItemDataRole.UserRole) == key:
                self.setCurrentRow(i)
                return


class NewStudyDialog(QDialog):
    """Type, name and folder for a new project (plus the mesh file of a 3D study).

    The type list states the 2D / 3D split in its titles and the right-hand
    text says in one line what the study starts with, so the choice is made
    knowing what Run will do. A 3D study from a mesh asks for the file here:
    the name defaults to the file's, and the import dialog (units,
    orientation) follows once the project exists. Validates before accepting.
    """

    def __init__(self, kind: str = "airfoil", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New study")
        self.setModal(True)
        self.setMinimumWidth(620)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4 * SPACING, 4 * SPACING, 4 * SPACING, 3 * SPACING)
        lay.setSpacing(3 * SPACING)

        # -- type: list on the left, what it starts with on the right
        top = QHBoxLayout()
        top.setSpacing(4 * SPACING)
        self.kind = _TypeList(available_study_types(), kind if kind in _TYPES else "airfoil")
        self.kind.setFixedWidth(200)
        top.addWidget(self.kind, 0, Qt.AlignmentFlag.AlignTop)
        about = QVBoxLayout()
        about.setSpacing(SPACING)
        self.type_title = label("", "h3")
        about.addWidget(self.type_title)
        self.hint = label("", "secondary", wrap=True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        about.addWidget(self.hint)
        about.addStretch(1)
        top.addLayout(about, 1)
        lay.addLayout(top)
        lay.addWidget(Divider())

        form = QFormLayout()
        form.setHorizontalSpacing(3 * SPACING)
        form.setVerticalSpacing(2 * SPACING)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        s = app_settings.get()
        folder = Path(s.last_dir)
        # the mesh file comes first for a 3D study: the name follows from it
        mrow = QHBoxLayout()
        mrow.setSpacing(2 * SPACING)
        mrow.setContentsMargins(0, 0, 0, 0)
        self.mesh = QLineEdit()
        self.mesh.setPlaceholderText("STL, OBJ or PLY file")
        mrow.addWidget(self.mesh, 1)
        mrow.addWidget(button("Browse…", None, None, self._browse_mesh))
        self.mesh_label = label("Mesh file", "pg-label")
        self._mesh_row = QWidget()
        self._mesh_row.setLayout(mrow)
        form.addRow(self.mesh_label, self._mesh_row)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Wing section trade study")
        form.addRow(label("Name", "pg-label"), self.name)
        frow = QHBoxLayout()
        frow.setSpacing(2 * SPACING)
        self.folder = QLineEdit(str(folder))
        frow.addWidget(self.folder, 1)
        frow.addWidget(button("Browse…", None, None, self._browse))
        form.addRow(label("Location", "pg-label"), frow)
        self.where = ElidedLabel("", "muted")
        form.addRow("", self.where)
        lay.addLayout(form)

        self.error = label("", "status-bad", wrap=True)
        self.error.setVisible(False)
        lay.addWidget(self.error)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = ok = box.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("Create")
        ok.setProperty("variant", "primary")
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        self.kind.changed.connect(self._kind_changed)
        self.name.textChanged.connect(self._update_where)
        self.folder.textChanged.connect(self._update_where)
        self.mesh.textChanged.connect(self._mesh_changed)
        self._kind_changed(self.kind.current())
        self.name.setFocus()

    _HINTS: ClassVar[dict[str, str]] = {t.key: t.description for t in STUDY_TYPES}
    _DEFAULT_NAMES: ClassVar[dict[str, str]] = {t.key: t.default_name for t in STUDY_TYPES}

    def _kind_changed(self, kind: str) -> None:
        t = _TYPES.get(kind)
        self.type_title.setText(t.title if t else "")
        self.hint.setText(self._HINTS.get(kind, ""))
        is_mesh = kind == "mesh"
        self.mesh_label.setVisible(is_mesh)
        self._mesh_row.setVisible(is_mesh)
        # the import dialog (units, orientation) follows Create: the button says so
        self.ok.setText("Create and import…" if is_mesh else "Create")
        if not self.name.text().strip() or self.name.text() in self._suggested():
            self.name.setText(unique_project_name(self.folder.text() or ".", self._default_name(kind)))
            self.name.selectAll()
        self.error.setVisible(False)

    def _default_name(self, kind: str) -> str:
        if kind == "mesh" and self.mesh.text().strip():
            stem = Path(self.mesh.text().strip()).stem.replace("_", " ").strip()
            if stem:
                return stem[:1].upper() + stem[1:]
        return self._DEFAULT_NAMES.get(kind, "New study")

    def _suggested(self) -> set[str]:
        out = set()
        bases = set(self._DEFAULT_NAMES.values())
        if self.mesh.text().strip():
            bases.add(self._default_name("mesh"))
        bases |= set(getattr(self, "_mesh_names", set()))
        for base in bases:
            out.add(base)
            out.update(f"{base} {k}" for k in range(2, 50))
        return out

    def _mesh_changed(self) -> None:
        # remember every name a mesh file suggested, so picking another file renames again
        self._mesh_names = getattr(self, "_mesh_names", set()) | {self._default_name("mesh")}
        suggested = not self.name.text().strip() or self.name.text() in self._suggested()
        if self.kind.current() == "mesh" and suggested:
            self.name.setText(unique_project_name(self.folder.text() or ".", self._default_name("mesh")))
        self.error.setVisible(False)

    def _update_where(self) -> None:
        name = self.name.text().strip()
        if name and self.folder.text().strip():
            self.where.setText(f"Folder: {project_folder(self.folder.text().strip(), name)}")
        else:
            self.where.setText("")
        self.error.setVisible(False)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose a folder for the project", self.folder.text())
        if d:
            self.folder.setText(d)

    def _browse_mesh(self) -> None:
        start = str(Path(self.mesh.text()).parent) if self.mesh.text().strip() else self.folder.text()
        pattern = " ".join(f"*{e}" for e in MESH_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Choose a 3D mesh", start,
                                              f"3D meshes ({pattern});;All files (*)")
        if path:
            self.mesh.setText(path)

    def values(self) -> tuple[str, str, Path]:
        """(kind, name, parent folder)."""
        return self.kind.current(), self.name.text().strip(), Path(self.folder.text().strip())

    def mesh_path(self) -> str:
        """The mesh file of a "mesh" study ("" for the other kinds)."""
        return self.mesh.text().strip() if self.kind.current() == "mesh" else ""

    def _accept(self) -> None:
        kind, name, folder = self.values()
        problem = None
        mesh = self.mesh_path()
        if not name:
            problem = "Enter a name for the study."
        elif not str(folder) or str(folder) == ".":
            problem = "Choose a folder."
        elif project_folder(folder, name).exists():
            problem = f"A project named “{safe_name(name)}” already exists in that folder."
        elif kind == "mesh" and not mesh:
            problem = "Choose the mesh file (STL, OBJ or PLY)."
        elif kind == "mesh" and not Path(mesh).is_file():
            problem = f"“{mesh}” does not exist."
        elif kind == "mesh" and Path(mesh).suffix.lower() not in MESH_EXTENSIONS:
            problem = "The mesh must be an STL, OBJ or PLY file."
        if problem:
            self.error.setText(problem)
            self.error.setVisible(True)
            return
        self.accept()


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def _action_link(text: str, icon: str, tooltip: str = "") -> QPushButton:
    """An action-list entry: 16 px icon + underlined ink text (a website text link), no button chrome."""
    b = button(text, "link", icon)
    b.setToolTip(tooltip)
    b.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


# ---------------------------------------------------------------------------
# study tiles: what each kind of study sets up, drawn as a small schematic
# ---------------------------------------------------------------------------

def _arrow(p: QPainter, x0: float, y: float, x1: float) -> None:
    """A free-stream arrow from x0 to x1 at height y (a line and a small open head)."""
    p.drawLine(QPointF(x0, y), QPointF(x1, y))
    p.drawLine(QPointF(x1, y), QPointF(x1 - 4, y - 3))
    p.drawLine(QPointF(x1, y), QPointF(x1 - 4, y + 3))


def _flow(p: QPainter, r: QRectF, ys: Sequence[float]) -> None:
    pen = QPen(theme.qcolor("muted"), 1.0)
    p.setPen(pen)
    for f in ys:
        _arrow(p, r.left() + 2, r.top() + f * r.height(), r.left() + 0.22 * r.width())


def paint_schematic(p: QPainter, kind: str, r: QRectF) -> None:
    """A line schematic of what a study of ``kind`` starts with, inside ``r``.

    Not decoration: each drawing shows the setup the study creates (the body,
    the free stream or the closed box, the heated part in the ``highlight``
    colour), in the 1.25 px line style of the icon set, so a user can tell the
    kinds apart before reading. Strokes are ``fg2`` (>= 4.5:1 on the surface).
    """
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    line = QPen(theme.qcolor("fg2"), 1.25)
    line.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    line.setCapStyle(Qt.PenCapStyle.RoundCap)
    heat = theme.qcolor("highlight")
    fill = theme.qcolor("field_solid")
    fill.setAlphaF(0.22)
    cx, cy, w, h = r.center().x(), r.center().y(), r.width(), r.height()
    if kind == "airfoil":
        _flow(p, r, (0.3, 0.5, 0.7))
        x0, x1 = r.left() + 0.3 * w, r.right() - 0.06 * w
        c = x1 - x0
        n = 40
        pts_up, pts_lo = [], []
        for i in range(n + 1):                  # NACA 2412, the section the airfoil study starts with
            x = (1 - math.cos(math.pi * i / n)) / 2
            t = 5 * 0.12 * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
            m, pk = 0.02, 0.4
            camber = m / pk ** 2 * (2 * pk * x - x * x) if x < pk else m / (1 - pk) ** 2 * (1 - 2 * pk + 2 * pk * x - x * x)
            pts_up.append(QPointF(x0 + x * c, cy - (camber + t) * c))
            pts_lo.append(QPointF(x0 + x * c, cy - (camber - t) * c))
        path = QPainterPath(pts_up[-1])
        for q in reversed(pts_up[:-1]):
            path.lineTo(q)
        for q in pts_lo[1:]:
            path.lineTo(q)
        path.closeSubpath()
        p.setPen(line)
        p.setBrush(fill)
        p.drawPath(path)
    elif kind == "vehicle":
        _flow(p, r, (0.3, 0.55))
        road = r.bottom() - 0.16 * h
        x0, x1 = r.left() + 0.3 * w, r.right() - 0.06 * w
        L = x1 - x0
        body = QPainterPath(QPointF(x0, road - 0.12 * h))
        body.cubicTo(QPointF(x0, road - 0.36 * h), QPointF(x0 + 0.18 * L, road - 0.38 * h),
                     QPointF(x0 + 0.3 * L, road - 0.4 * h))
        body.cubicTo(QPointF(x0 + 0.42 * L, road - 0.66 * h), QPointF(x0 + 0.68 * L, road - 0.68 * h),
                     QPointF(x0 + 0.8 * L, road - 0.44 * h))
        body.lineTo(QPointF(x1, road - 0.38 * h))
        body.lineTo(QPointF(x1, road - 0.12 * h))
        body.closeSubpath()
        p.setPen(line)
        p.setBrush(fill)
        p.drawPath(body)
        p.setBrush(theme.qcolor("surface"))
        rw = 0.09 * h
        for fx in (0.2, 0.78):
            p.drawEllipse(QPointF(x0 + fx * L, road - rw - 0.5), rw, rw)
        p.drawLine(QPointF(r.left() + 0.24 * w, road + 0.5), QPointF(r.right() - 2, road + 0.5))
    elif kind == "mesh":
        _flow(p, r, (0.35, 0.55, 0.75))
        s = 0.42 * h
        ox, oy = cx + 0.06 * w, cy + 0.08 * h
        dx, dy = 0.5 * s, 0.3 * s
        front = QPolygonF([QPointF(ox - s / 2, oy - s / 2), QPointF(ox + s / 2, oy - s / 2),
                           QPointF(ox + s / 2, oy + s / 2), QPointF(ox - s / 2, oy + s / 2)])
        p.setPen(line)
        p.setBrush(fill)
        p.drawPolygon(front)
        p.setBrush(Qt.BrushStyle.NoBrush)
        top = QPolygonF([QPointF(ox - s / 2, oy - s / 2), QPointF(ox - s / 2 + dx, oy - s / 2 - dy),
                         QPointF(ox + s / 2 + dx, oy - s / 2 - dy), QPointF(ox + s / 2, oy - s / 2)])
        p.drawPolygon(top)
        p.drawLine(QPointF(ox + s / 2 + dx, oy - s / 2 - dy), QPointF(ox + s / 2 + dx, oy + s / 2 - dy))
        p.drawLine(QPointF(ox + s / 2 + dx, oy + s / 2 - dy), QPointF(ox + s / 2, oy + s / 2))
        thin = QPen(theme.qcolor("muted"), 0.8)
        p.setPen(thin)
        p.drawLine(QPointF(ox - s / 2, oy + s / 2), QPointF(ox + s / 2, oy - s / 2))
        p.drawLine(QPointF(ox - s / 2, oy), QPointF(ox, oy - s / 2))
        p.drawLine(QPointF(ox, oy + s / 2), QPointF(ox + s / 2, oy))
    elif kind == "wing":
        _flow(p, r, (0.25, 0.75))
        # the span runs up the tile: bounded by the height too, or a wide tile (1600 px window) cut the tips
        root, tip, span = 0.5 * h, 0.28 * h, min(0.82 * w * 0.5, 0.88 * h)
        rx = cx - 0.02 * w
        x_le_tip = rx + 0.25 * (root - tip) + 0.06 * h
        wing = QPolygonF([QPointF(rx, cy - 0.04 * h), QPointF(x_le_tip, cy - span * 0.5),
                          QPointF(x_le_tip + tip, cy - span * 0.5), QPointF(rx + root, cy - 0.04 * h),
                          QPointF(rx + root, cy + 0.04 * h), QPointF(x_le_tip + tip, cy + span * 0.5),
                          QPointF(x_le_tip, cy + span * 0.5), QPointF(rx, cy + 0.04 * h)])
        p.setPen(line)
        p.setBrush(fill)
        p.drawPolygon(wing)
        dash = QPen(theme.qcolor("muted"), 0.8, Qt.PenStyle.DashLine)
        p.setPen(dash)
        p.drawLine(QPointF(rx + 0.25 * root, cy), QPointF(x_le_tip + 0.25 * tip, cy - span * 0.5))
        p.drawLine(QPointF(rx + 0.25 * root, cy), QPointF(x_le_tip + 0.25 * tip, cy + span * 0.5))
    elif kind in ("room", "enclosure"):
        box = QRectF(r.left() + 0.14 * w, r.top() + 0.08 * h, 0.72 * w, 0.84 * h)
        p.setPen(QPen(theme.qcolor("fg2"), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(box)
        p.setPen(line)
        if kind == "room":
            # the window (cold wall) above the radiator, both on the left wall
            win_top, win_bot = box.top() + 0.14 * box.height(), box.top() + 0.42 * box.height()
            p.setPen(QPen(theme.qcolor("fg2"), 1.0))
            p.drawLine(QPointF(box.left() + 3, win_top), QPointF(box.left() + 3, win_bot))
            rad = QRectF(box.left() + 4, box.bottom() - 0.42 * box.height(), 5, 0.26 * box.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(heat)
            p.drawRect(rad)
            # the plume rising off the radiator, along the ceiling, down the far wall, back on the floor
            p.setPen(QPen(theme.qcolor("muted"), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            loop = QRectF(rad.right() + 6, box.top() + 7, box.width() - (rad.right() - box.left()) - 16,
                          box.height() - 14)
            p.drawRoundedRect(loop, 0.38 * loop.height(), 0.38 * loop.height())
            tip = QPointF(loop.center().x(), loop.top())
            p.drawLine(tip, tip + QPointF(-4, -3))
            p.drawLine(tip, tip + QPointF(-4, 3))
            tip2 = QPointF(loop.center().x(), loop.bottom())
            p.drawLine(tip2, tip2 + QPointF(4, -3))
            p.drawLine(tip2, tip2 + QPointF(4, 3))
        else:
            board_y = box.bottom() - 0.32 * box.height()
            p.drawLine(QPointF(box.left() + 0.16 * box.width(), board_y),
                       QPointF(box.right() - 0.16 * box.width(), board_y))
            chip = QRectF(box.center().x() - 6, board_y - 5, 12, 5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(heat)
            p.drawRect(chip)
            p.setPen(QPen(theme.qcolor("muted"), 1.0))
            _arrow(p, box.left() - 10, box.top() + 0.4 * box.height(), box.left() + 14)
            _arrow(p, box.right() - 6, box.top() + 0.55 * box.height(), box.right() + 10)
            p.setPen(line)
            fx = box.left() + 20
            p.drawLine(QPointF(fx, box.top() + 0.4 * box.height() - 7), QPointF(fx, box.top() + 0.4 * box.height() + 7))
    else:                                   # empty project
        p.setPen(QPen(theme.qcolor("muted"), 1.0, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(cx - 0.3 * w, cy - 0.32 * h, 0.6 * w, 0.64 * h), 3, 3)
    p.restore()


class StudyTile(QAbstractButton):
    """One kind of study: its schematic, title, dimension and what it starts with.

    A button (click, Space, Enter) with a 1 px ``line`` border that turns
    ``control`` on hover and gets the focus ring on keyboard focus. Painted by
    hand so the schematic, the title and the wrapped description share one
    hit target; the accessible name and description carry the same words.
    """

    SCHEMATIC_H = 64
    #: the title weight: Medium, as the website sets its headings (v4; it was SemiBold)
    TITLE_WEIGHT = QFont.Weight.Medium
    MIN_W, MAX_W = 150, 340
    PAD = 12
    BODY_PT = 8.5
    #: the width changed (the page re-fits every tile's height to the new widths)
    widthChanged = Signal(int)

    def __init__(self, study: StudyType, tag: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.study = study
        self.tag = tag
        self.setText(study.title.split(" (")[0])
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # never narrower than the title: beside an open Assistant panel at 1280 px the tiles shrank to
        # 155 px and "3D body from a mesh" was cut mid-letter
        title_fm = QFontMetrics(theme.ui_font(weight=self.TITLE_WEIGHT, pt=theme.FONT_PT["h2"]))
        self.setMinimumWidth(max(self.MIN_W, title_fm.horizontalAdvance(self.text()) + 2 * self.PAD + 2))
        self.setMaximumWidth(self.MAX_W)
        self.setFixedHeight(self.required_height(220))
        self.setToolTip(study.description)
        self.setAccessibleName(f"New {study.title.lower()} study")
        self.setAccessibleDescription(study.description)
        theme.changed.connect(self._on_theme)     # bound method: auto-disconnects when deleted

    def _on_theme(self, _name: str = "") -> None:
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(220, self.height())

    _TEXT_FLAGS = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop) | int(Qt.TextFlag.TextWordWrap)

    def _text_width(self, width: float) -> float:
        """Width of the description box in a tile ``width`` wide: the one place both
        :meth:`required_height` and :meth:`paintEvent` take it from. They used to differ by
        the 1 px of the half-pixel border inset, enough to wrap a line one word earlier when
        painted than when measured, and cut the last line (v3.4 audit)."""
        return max(40.0, float(width) - 1.0 - 2 * self.PAD)

    def _body_top(self) -> float:
        """Top of the description box: schematic, title row and tag row above it (as painted)."""
        title = QFontMetrics(theme.ui_font(weight=self.TITLE_WEIGHT, pt=theme.FONT_PT["h2"]))
        small = QFontMetrics(theme.ui_font(pt=theme.FONT_PT["small"]))
        return float(self.PAD + self.SCHEMATIC_H + 10 + title.height() + 1 + small.height() + 5)

    def required_height(self, width: int) -> int:
        """The height that shows the whole description at ``width`` (titles and schematic included)."""
        body = QFontMetricsF(theme.ui_font(pt=self.BODY_PT))
        box = body.boundingRect(QRectF(0, 0, self._text_width(width), 4000), self._TEXT_FLAGS,
                                self.study.description)
        return math.ceil(self._body_top() + box.height() + self.PAD + 1)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self.required_height(width)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if e.size().width() != e.oldSize().width():
            self.widthChanged.emit(e.size().width())

    #: focus reasons that mean the keyboard brought focus here: the website's ``:focus-visible``
    _KEYBOARD_FOCUS = (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason,
                       Qt.FocusReason.ShortcutFocusReason)
    #: focus reasons that only restore an earlier focus (window or popup came back): keep its look
    _RESTORED_FOCUS = (Qt.FocusReason.ActiveWindowFocusReason, Qt.FocusReason.PopupFocusReason)

    def focus_ring_visible(self) -> bool:
        """True when the tile has focus that the keyboard gave it (the ring is painted).

        The first tile takes focus when the window opens; painting the ring then made it look
        selected in every screenshot and in the real launch, though nobody had pressed a key.
        Like the website's ``:focus-visible``, the ring shows for Tab, Shift+Tab, F6 and any key
        pressed while the tile has focus, not for a click or the window's initial focus.
        """
        return self.hasFocus() and self._focus_visible

    _focus_visible = False

    def focusInEvent(self, e) -> None:
        reason = e.reason()
        if reason in self._KEYBOARD_FOCUS:
            self._focus_visible = True
        elif reason not in self._RESTORED_FOCUS:
            self._focus_visible = False
        super().focusInEvent(e)

    def keyPressEvent(self, e) -> None:
        if not self._focus_visible:
            self._focus_visible = True
            self.update()
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
            return
        super().keyPressEvent(e)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hover = self.underMouse() or self.isDown()
        p.setPen(QPen(theme.qcolor("control" if hover else "line"), 1.0))
        p.setBrush(theme.qcolor("hover" if self.isDown() else "surface"))
        p.drawRoundedRect(r, theme.RADIUS_LG, theme.RADIUS_LG)      # the website's 8 px frames
        pad = float(self.PAD)
        sch = QRectF(pad, pad, r.width() - 2 * pad, self.SCHEMATIC_H)
        paint_schematic(p, self.study.key, sch)
        y = sch.bottom() + 10
        body_top = self._body_top()
        title = theme.ui_font(weight=self.TITLE_WEIGHT, pt=theme.FONT_PT["h2"])
        p.setFont(title)
        p.setPen(theme.qcolor("fg"))
        fm = QFontMetrics(title)
        p.drawText(QRectF(pad, y, r.width() - 2 * pad, fm.height()), Qt.AlignmentFlag.AlignLeft,
                   fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, int(r.width() - 2 * pad)))
        y += fm.height() + 1
        small = theme.ui_font(pt=theme.FONT_PT["small"])
        p.setFont(small)
        p.setPen(theme.qcolor("muted"))
        fs = QFontMetrics(small)
        p.drawText(QRectF(pad, y, r.width() - 2 * pad, fs.height()), Qt.AlignmentFlag.AlignLeft, self.tag)
        body = theme.ui_font(pt=self.BODY_PT)
        p.setFont(body)
        p.setPen(theme.qcolor("fg2"))
        # the box runs to the bottom edge: :meth:`required_height` made the tile tall enough, and a
        # box clipped above the text's own height would cut its last line rather than show it
        box = QRectF(pad, body_top, self._text_width(self.width()), max(0.0, r.bottom() - body_top))
        p.drawText(box, self._TEXT_FLAGS, self.study.description)
        if self.focus_ring_visible():
            theme.paint_focus_ring(p, QRectF(self.rect()).adjusted(1, 1, -1, -1), theme.RADIUS_LG - 1)
            theme.paint_focus_ring(p, QRectF(self.rect()).adjusted(2, 2, -2, -2), theme.RADIUS_LG - 2)
        p.end()


@lru_cache(maxsize=1)
def joukowski_streamlines() -> tuple[tuple[tuple[float, float], ...], ...]:
    """Streamlines of inviscid flow past a cambered Joukowski airfoil at 4 degrees (Kutta condition).

    The same flow the website draws below its hero (``website/build.py: streakline_svg``), so the
    one decorative drawing in the app is also a correct flow: in the circle plane a uniform stream
    at angle alpha passes a circle through zeta = 1 (centre mu) with circulation
    Gamma = 4 pi U R sin(alpha + beta), which puts the rear stagnation point on the cusp, and
    z = zeta + 1/zeta maps it to the airfoil. Each line is traced in the circle plane (RK2 along the
    velocity direction) and mapped, then turned by -alpha so the stream runs level. Returned in
    airfoil units (chord about 4, nose near x = -2), each line as (x, y) points left to right, with
    the airfoil outline as the last entry. Pure Python, about 30 ms once, then cached.
    """
    alpha = math.radians(4.0)
    mu = complex(-0.08, 0.05)
    radius = abs(1 - mu)
    beta = math.asin(mu.imag / radius)
    gamma = 4 * math.pi * radius * math.sin(alpha + beta)
    ea = complex(math.cos(alpha), -math.sin(alpha))

    def vel(zeta: complex) -> complex:
        s = zeta - mu
        return (ea - radius * radius / (ea * s * s) + 1j * gamma / (2 * math.pi * s)).conjugate()

    def jac(zeta: complex) -> float:
        return abs(1 - 1 / (zeta * zeta)) or 1e-9

    x0, x1 = -14.0, 6.0
    lines: list[tuple[tuple[float, float], ...]] = []
    # seeds unevenly spaced, as a real smoke rake is (and as the website's divider is)
    for y0 in (-1.62, -1.3, -0.97, -0.69, -0.45, -0.19, 0.05, 0.3, 0.56, 0.86, 1.22, 1.6):
        zeta = complex(x0, y0) / ea
        pts = [zeta + 1 / zeta]
        for _ in range(6000):
            z = pts[-1]
            if (z * ea).real > x1:
                break
            h = min(0.5, 0.025 + 0.05 * abs(z))
            v1 = vel(zeta)
            if abs(v1) < 1e-6:
                break
            mid = zeta + 0.5 * h * v1 / abs(v1) / jac(zeta)
            v2 = vel(mid)
            zeta = zeta + h * v2 / abs(v2) / jac(mid)
            if abs(zeta - mu) < radius:
                break
            pts.append(zeta + 1 / zeta)
        if (pts[-1] * ea).real > x1:            # a line that stopped at the body is dropped
            lines.append(tuple(((p * ea).real, (p * ea).imag) for p in pts))
    ring = [mu + radius * complex(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 96 for i in range(96))]
    lines.append(tuple((((r + 1 / r) * ea).real, ((r + 1 / r) * ea).imag) for r in ring))
    return tuple(lines)


class FlowMotif(QWidget):
    """A quiet, static field of streak dots beside the product name on the Start page.

    The website opens on a live flow drawn as short ink streaks; the app keeps that motif, still
    and faint, in the one place that introduces the product (docs/ui_guidelines.md, v4: the app
    carries over the site's language). The streaks lie on the streamlines of
    :func:`joukowski_streamlines`, evenly spaced along each line with a fixed pseudo-random phase,
    so the picture is a real flow past an airfoil seen as smoke, not decoration drawn by hand.

    Cost: nothing at run time beyond one ``drawPixmap``. The streaks are rendered once into a
    pixmap per size, device pixel ratio and theme (``theme.changed`` drops it); there is no timer
    or animation. Colour: ``line_strong``, the streaks fading in from the left, so the motif sits below every text contrast on the page and never competes with the
    tiles. It takes no focus and has no accessible name (it carries no information).
    """

    HEIGHT = 40
    #: dash length and the gap after it along a streamline (px)
    DASH, GAP = 3.0, 8.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._cache: tuple[tuple, object] | None = None
        theme.changed.connect(self._on_theme)      # bound method: auto-disconnects when deleted

    def _on_theme(self, _name: str = "") -> None:
        self._cache = None
        self.update()

    def render_pixmap(self, w: int, h: int, dpr: float):
        """The motif at ``w`` x ``h`` logical pixels (a QPixmap); also used by the tests."""
        from PySide6.QtGui import QColor, QPixmap

        pm = QPixmap(max(1, round(w * dpr)), max(1, round(h * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        if w < 40 or h < 10:
            return pm
        lines = joukowski_streamlines()
        # the five lines nearest the airfoil, every other one: at 40 px all twelve were 3 px apart and
        # read as a dotted band, and the airfoil came out 40 px long with no visible bend around it
        near = [ln for ln in lines[:-1] if -1.1 <= ln[0][1] <= 1.3]
        streams, foil = near[::2], lines[-1]
        ys = [y for ln in streams for _x, y in ln]
        mid_y = 0.5 * (min(ys) + max(ys))
        scale = (h - 6) / max(1e-6, max(ys) - min(ys))         # every line inside the strip
        cx = w - 2.2 * scale - 10                             # the airfoil's tail 10 px from the right end
        tk = theme.tokens()
        ink, foil_fill = QColor(tk["line_strong"]), QColor(tk["line_strong"])
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        period = self.DASH + self.GAP
        fade0, fade1 = 0.15 * w, 0.65 * w                    # streaks fade in from the left
        for k, ln in enumerate(streams):
            pts = [QPointF(cx + scale * x, 0.5 * h - scale * (y - mid_y)) for x, y in ln]
            # far upstream the stream is uniform: extend each line level to the strip's left edge
            if pts[0].x() > 0:
                pts.insert(0, QPointF(-period, pts[0].y()))
            # one dash every DASH + GAP px of arc length, its ends interpolated along the line (so a
            # dash that spans several of the short steps near the airfoil keeps its length); the
            # phase is a fixed function of the line index, so neighbouring lines do not line up
            xs = np.array([q.x() for q in pts])
            yv = np.array([q.y() for q in pts])
            arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(yv)))))
            starts = np.arange((k * 3.7) % period, arc[-1] - self.DASH, period)
            for xa, ya, xb, yb in zip(np.interp(starts, arc, xs), np.interp(starts, arc, yv),
                                      np.interp(starts + self.DASH, arc, xs),
                                      np.interp(starts + self.DASH, arc, yv), strict=True):
                if not 0 <= xa <= w:
                    continue
                c = QColor(ink)
                c.setAlphaF(max(0.0, min(1.0, (xa - fade0) / (fade1 - fade0))))
                pen = QPen(c, 1.2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(xa, ya), QPointF(xb, yb))
        outline = QPolygonF([QPointF(cx + scale * x, 0.5 * h - scale * (y - mid_y)) for x, y in foil])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(foil_fill)
        p.drawPolygon(outline)
        p.end()
        return pm

    def paintEvent(self, _e) -> None:
        w, h, dpr = self.width(), self.height(), self.devicePixelRatioF()
        key = (w, h, dpr, theme.current())
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self.render_pixmap(w, h, dpr))
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache[1])
        p.end()


#: the tiles the Start page shows (the New study dialog lists every kind, the empty project too)
START_TILES: tuple[tuple[str, str], ...] = (
    ("airfoil", "2D section"), ("vehicle", "2D side profile"), ("mesh", "3D, from STL / OBJ / PLY"),
    ("wing", "3D planform"), ("room", "3D, heat transfer"), ("enclosure", "2D, heat transfer"),
)


class _ProjectList(QTreeWidget):
    """Recent / example projects as a flat table: name, type, then a stretching column.

    ``widths`` gives the fixed starting width of each column (``0`` = the one that
    stretches: the folder of a recent project, the description of an example).
    """

    def __init__(self, headers: list[str], widths: list[int] | None = None, *,
                 elide: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(len(headers))
        self.setHeaderLabels(headers)
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setIconSize(QSize(16, 16))
        self.setMouseTracking(True)
        self.setTextElideMode(elide)        # folders keep both ends; sentences keep their start
        h = self.header()
        h.setStretchLastSection(False)
        h.setSectionsMovable(False)
        widths = widths or [240] + [0] + [140] * (len(headers) - 2)
        for c, w in enumerate(widths):
            if w:
                h.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
                h.resizeSection(c, w)
            else:
                h.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)

    def fit_rows(self, max_rows: int | None = None) -> None:
        """Height for the rows present (at most ``max_rows``), so short lists stay short.

        The header and rows are measured *after* polishing: on a window's first
        refresh the style sheet's header padding and the bundled font were not
        applied yet, so the list came out 12 px short and the last example row
        was clipped behind a scroll bar (measured at 1600 x 1000, light theme).
        The last ``max_rows`` is kept, and :meth:`StartPage._recolour` /
        ``showEvent`` measure again after a theme switch or the first show.
        """
        if max_rows is not None:
            self._max_rows = max_rows
        limit = getattr(self, "_max_rows", 10)
        self.ensurePolished()
        self.header().ensurePolished()
        rows = max(1, min(self.topLevelItemCount(), limit))
        row_h = self.sizeHintForRow(0) if self.topLevelItemCount() else theme.ROW_H
        frame = 2 * self.frameWidth()
        self.setFixedHeight(self.header().sizeHint().height() + rows * max(row_h, theme.ROW_H) + frame)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.fit_rows()


class StartPage(QWidget):
    """Welcome page. Emits requests; the main window owns dialogs and project switching.

    Layout (docs/ui_guidelines.md section 4): the product line at the top, then
    the kinds of study as a row of tiles - each with a schematic of what it
    sets up, its dimension and one sentence of what it starts with - then the
    recent projects and the bundled examples beside a short column of the
    other ways in (open, examples, empty project) and the guides. No hero, no
    feature marketing: a returning user's last project is one click away, a
    new user's first study is one click away, and both are above the fold at
    1280 x 800.

    Signals: ``newStudyRequested(kind)`` (a :data:`STUDY_TYPES` key),
    ``openRequested()`` (folder dialog), ``openPathRequested(path)``,
    ``openExampleRequested(path)``, ``helpRequested(topic)`` ("guide",
    "scripting", "assistant", "mcp", "tour", "shortcuts", "validation",
    "licence"). ``handle_action`` accepts "new-<kind>", "new", "open".
    """

    newStudyRequested = Signal(str)
    openRequested = Signal()
    openPathRequested = Signal(str)
    openExampleRequested = Signal(str)
    helpRequested = Signal(str)

    def __init__(self, state, jobs=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state, self.jobs = state, jobs
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)    # the tiles and links take focus, not the area
        outer.addWidget(scroll)
        self._scroll = scroll
        host = QWidget()
        scroll.setWidget(host)
        page = QVBoxLayout(host)
        page.setContentsMargins(10 * SPACING, 6 * SPACING, 10 * SPACING, 4 * SPACING)
        page.setSpacing(0)

        # -- the product line
        head = QHBoxLayout()
        head.setSpacing(3 * SPACING)
        self.mark = QLabel()
        self.mark.setFixedSize(36, 36)
        self.mark.setPixmap(theme.app_icon_pixmap(36, self.devicePixelRatioF()))
        head.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        names = QVBoxLayout()
        names.setSpacing(0)
        self.title = label("Aerolume", "title")
        names.addWidget(self.title)
        self.version = label(f"Version {__version__}. Aerodynamics and heat transfer in 2D and 3D, "
                             "on the GPU or the CPU.", "secondary")
        names.addWidget(self.version)
        head.addLayout(names, 0)
        # the website's flow motif, still and faint, in the room the product line leaves (v4)
        self.motif = FlowMotif()
        head.addWidget(self.motif, 1, Qt.AlignmentFlag.AlignVCenter)
        page.addLayout(head)
        page.addSpacing(5 * SPACING)

        # -- the kinds of study
        page.addWidget(label("New study", "h3"))
        page.addSpacing(2 * SPACING)
        self._tile_grid = tiles = QGridLayout()
        tiles.setContentsMargins(0, 0, 0, 0)
        tiles.setHorizontalSpacing(3 * SPACING)
        tiles.setVerticalSpacing(3 * SPACING)
        self.tiles: dict[str, StudyTile] = {}
        self._tile_columns = 0
        # The page's resize comes before the grid hands the tiles their new widths, so a height
        # fitted then used the old (wider) width: on first show at 1600 x 1000 the descriptions lost
        # their third line until something else relaid the page (v3.4 audit). Each tile reports its
        # own width change; one refit runs after the layout has settled.
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(0)
        self._refit_timer.timeout.connect(self._fit_tiles)
        available = {t.key for t in available_study_types()}
        for key, tag in START_TILES:
            if key not in available:
                continue
            t = StudyTile(_TYPES[key], tag)
            t.clicked.connect(lambda _=False, k=key: self.newStudyRequested.emit(k))
            t.widthChanged.connect(lambda _w: self._refit_timer.start())
            self.tiles[key] = t
        self.tiles_widget = QWidget()
        self.tiles_widget.setLayout(tiles)
        self._reflow_tiles(len(self.tiles))
        page.addWidget(self.tiles_widget)
        # names the window, the tour and the tests use for the study entries
        self.link_airfoil = self.tiles.get("airfoil")
        self.link_vehicle = self.tiles.get("vehicle")
        self.link_mesh = self.tiles.get("mesh")
        self.link_wing = self.tiles.get("wing")
        self.link_room = self.tiles.get("room")
        self.link_enclosure = self.tiles.get("enclosure")
        self.actions_widget = self.tiles_widget
        page.addSpacing(6 * SPACING)

        cols = QHBoxLayout()
        cols.setSpacing(10 * SPACING)
        page.addLayout(cols, 1)

        # -- left: recent projects and examples
        lists = QVBoxLayout()
        lists.setContentsMargins(0, 0, 0, 0)
        lists.setSpacing(2 * SPACING)
        head = QHBoxLayout()
        head.addWidget(label("Recent projects", "h3"))
        head.addStretch(1)
        self.clear_link = button("Clear list", "link")
        self.clear_link.setToolTip("Forget the recent projects (the folders are not touched)")
        self.clear_link.clicked.connect(self._clear_recent)
        head.addWidget(self.clear_link)
        lists.addLayout(head)
        self.recent = _ProjectList(["Name", "Type", "Folder", "Modified"], [240, 130, 0, 130])
        self.recent.setAccessibleName("Recent projects")
        self.recent.itemClicked.connect(self._recent_clicked)
        self.recent.itemActivated.connect(self._recent_clicked)
        self.recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self._recent_menu)
        lists.addWidget(self.recent)
        self.recent_empty = label("No recent projects yet. A study you create or open is listed here.", "muted")
        lists.addWidget(self.recent_empty)
        lists.addSpacing(5 * SPACING)
        self.examples_header = label("Examples", "h3")
        lists.addWidget(self.examples_header)
        self.examples = _ProjectList(["Name", "Type", "Description", "Designs"], [240, 130, 0, 64],
                                     elide=Qt.TextElideMode.ElideRight)
        self.examples.setAccessibleName("Example projects")
        self.examples.itemClicked.connect(self._example_clicked)
        self.examples.itemActivated.connect(self._example_clicked)
        lists.addWidget(self.examples)
        self.examples_note = label("An example opens as a copy in Documents/Aerolume/Examples; the original "
                                   "is never changed.", "small", wrap=True)
        lists.addWidget(self.examples_note)
        lists.addStretch(1)
        lw = QWidget()
        lw.setLayout(lists)
        cols.addWidget(lw, 1)

        # -- right: the other ways in, then the guides
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(SPACING)
        side.addWidget(label("Open", "h3"))
        side.addSpacing(SPACING)
        self.link_open = _action_link("Open project…", "open", "Open a .aerolume project folder (Ctrl+O)")
        self.link_open.clicked.connect(self.openRequested.emit)
        side.addWidget(self.link_open)
        self.link_examples = _action_link("Open example", "folder", "Copy a bundled example and open it")
        self.examples_menu = QMenu(self)
        self.examples_menu.aboutToShow.connect(self._fill_examples_menu)
        self.link_examples.clicked.connect(self._show_examples_menu)
        side.addWidget(self.link_examples)
        self.link_more = _action_link("More study types…", "new",
                                      "Every kind of study and an empty project (Ctrl+N)")
        self.link_more.clicked.connect(lambda: self.newStudyRequested.emit("blank"))
        side.addWidget(self.link_more)
        side.addSpacing(6 * SPACING)
        self.learn_header = label("Learn", "h3")         # the tour outlines it with the links below
        side.addWidget(self.learn_header)
        side.addSpacing(SPACING)
        # The user guide and the validation record are the footer's Documentation and Validation links
        # (ui_guidelines section 4); this group lists the other guides, each once.
        self.help_links: dict[str, QPushButton] = {}
        for text, icon, topic, tip in (("Welcome tour", "compass", "tour",
                                        "Three steps over this page; the rest when a project is open"),
                                       ("Scripting guide", "code", "scripting",
                                        "The Python API: scripts, the Script panel and the command line"),
                                       ("Experiment assistant", "assistant", "assistant",
                                        "Plan and run experiments with Claude or a local model"),
                                       ("Driving Aerolume from Claude", "external", "mcp",
                                        ("Claude Desktop or Claude Code can run experiments through the "
                                         "aerolume-mcp server")),
                                       ("Keyboard shortcuts", "help", "shortcuts", "")):
            b = _action_link(text, icon, tip)
            b.clicked.connect(lambda _=False, t=topic: self.helpRequested.emit(t))
            side.addWidget(b)
            self.help_links[topic] = b
        side.addStretch(1)
        sw = QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(260)
        cols.addWidget(sw, 0, Qt.AlignmentFlag.AlignTop)

        # -- footer
        page.addSpacing(6 * SPACING)
        page.addWidget(Divider())
        page.addSpacing(2 * SPACING)
        foot = QHBoxLayout()
        foot.setSpacing(5 * SPACING)
        self.footer_links: dict[str, QPushButton] = {}
        for text, topic in (("Documentation", "guide"), ("Validation", "validation"), ("Licence", "licence")):
            b = button(text, "link")
            b.clicked.connect(lambda _=False, t=topic: self.helpRequested.emit(t))
            foot.addWidget(b)
            self.footer_links[topic] = b
        foot.addStretch(1)
        foot.addWidget(label(f"Aerolume {__version__}, Apache License 2.0", "small"))
        page.addLayout(foot)

        self.refresh()
        app_settings.signals.changed.connect(self._settings_changed)
        theme.changed.connect(self._recolour)       # bound method: disconnected when the page is deleted

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit_tiles()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._fit_tiles()

    def tile_columns(self) -> int:
        """How many tiles one row holds now (all of them, unless the width is short)."""
        return self._tile_columns

    def _columns_for(self, width: int) -> int:
        """Tiles per row: the fewest rows whose rows all fit ``width`` with every title whole.

        Rows are balanced (six tiles wrap as three and three, not five and one),
        so the tiles of every row keep one width.
        """
        tiles = list(self.tiles.values())
        n = len(tiles)
        gap = self._tile_grid.horizontalSpacing()
        for rows in range(1, n + 1):
            cols = -(-n // rows)
            split = [tiles[i:i + cols] for i in range(0, n, cols)]
            if all(sum(t.minimumWidth() for t in row) + gap * (len(row) - 1) <= width for row in split):
                return cols
        return 1

    def _reflow_tiles(self, cols: int) -> None:
        """Lay the tiles out ``cols`` to a row (equal columns; a short last row keeps the widths)."""
        cols = max(1, min(cols, len(self.tiles)))
        if cols == self._tile_columns:
            return
        grid = self._tile_grid
        for t in self.tiles.values():
            grid.removeWidget(t)
        for c in range(grid.columnCount()):
            grid.setColumnStretch(c, 0)
        for i, t in enumerate(self.tiles.values()):
            grid.addWidget(t, i // cols, i % cols)
        for c in range(cols):
            grid.setColumnStretch(c, 1)
        self._tile_columns = cols

    def _fit_tiles(self) -> None:
        """Tiles per row from the width, then one height for all: the tallest description, never clipped.

        Six tiles never narrower than their titles need about 1000 px; beside the
        Assistant panel at 1280 px the page has about 880, so the row wraps to
        two rows of three instead of pushing the page wider than the window.
        """
        if not self.tiles:
            return
        host = self._scroll.widget()
        side = host.layout().contentsMargins() if host is not None and host.layout() is not None else None
        avail = self._scroll.viewport().width() - (side.left() + side.right() if side else 0)
        if avail > 0:
            self._reflow_tiles(self._columns_for(avail))
        h = self.tile_height()
        for t in self.tiles.values():
            if t.height() != h:
                t.setFixedHeight(h)

    def tile_height(self) -> int:
        """The one height every tile gets: the tallest description at the tiles' current width.

        Each tile is measured at its own width (the narrowest decides when a row
        is uneven); before the first layout the widths are unknown and 220 px
        stands in until the tiles report their real widths (``widthChanged``).
        """
        tiles = list(self.tiles.values())
        if not tiles:
            return 0
        widths = [t.width() for t in tiles if t.width() > 0 and t.isVisibleTo(self)]
        width = min(widths) if widths else 220
        return max(t.required_height(width) for t in tiles)

    # -- page protocol --------------------------------------------------------
    def on_shown(self) -> None:
        self.refresh()

    def handle_action(self, action: str) -> None:
        if action.startswith("new-") and action[4:] in _TYPES:
            self.newStudyRequested.emit(action[4:])
        elif action == "new":
            self.newStudyRequested.emit("airfoil")
        elif action == "open":
            self.openRequested.emit()

    # -- content --------------------------------------------------------------
    def _settings_changed(self, key: str) -> None:
        if key == "paths/recent" and self.isVisible():
            self.refresh()

    #: secondary columns of each list (they read in the theme's text2 colour)
    _SECONDARY: ClassVar[dict[str, tuple[int, ...]]] = {"recent": (1, 2, 3), "examples": (1, 2)}

    def _recolour(self, _name: str = "") -> None:
        """Re-read the secondary text colour and icons after a theme switch.

        The item colours are brushes fixed when the rows are made: built in the
        light theme and shown in the dark one, the Type and Description columns
        kept the light theme's grey on the dark background (about 1.6:1).
        """
        for key, lst in (("recent", self.recent), ("examples", self.examples)):
            colour = theme.qcolor("text2")
            for i in range(lst.topLevelItemCount()):
                it = lst.topLevelItem(i)
                for c in self._SECONDARY[key]:
                    it.setForeground(c, colour)
                icon = it.data(0, Qt.ItemDataRole.UserRole + 1)
                if icon:
                    it.setIcon(0, theme.icon(icon, "text2"))
            if lst.topLevelItemCount():
                lst.fit_rows()              # a font switch changes the row and header heights
        self.mark.setPixmap(theme.app_icon_pixmap(36, self.devicePixelRatioF()))

    @staticmethod
    def _example_order(path: Path) -> tuple[int, str]:
        """3D examples first, then the placed-physics ones (heated bodies, rooms with fans), then wings,
        then the other 2D ones: the 3D results and the heat / fan studies the Start tiles headline are what
        a first look should find."""
        s = summarize(path)
        dims = set(s.tags or ())
        rank = 0 if "3D" in dims else 1 if s.templates else 2 if "Wing" in dims else 3
        return rank, s.name.lower()

    def refresh(self) -> None:
        """Rebuild the recent and examples lists from disk."""
        self.recent.clear()
        for path in recent_projects(10):
            s = summarize(path)
            it = QTreeWidgetItem([s.name, s.type_text, display_folder(path.parent),
                                  format_modified(s.modified)])
            it.setIcon(0, theme.icon(s.icon, "text2"))
            it.setData(0, Qt.ItemDataRole.UserRole, str(path))
            it.setData(0, Qt.ItemDataRole.UserRole + 1, s.icon)
            n = f"{s.designs} design{'s' if s.designs != 1 else ''}, {s.simulations} simulation" \
                f"{'s' if s.simulations != 1 else ''}"
            for c in range(4):
                it.setToolTip(c, f"{path}\n{n}")
            for c in self._SECONDARY["recent"]:
                it.setForeground(c, theme.qcolor("text2"))
            self.recent.addTopLevelItem(it)
        has_recent = self.recent.topLevelItemCount() > 0
        self.recent.setVisible(has_recent)
        self.recent_empty.setVisible(not has_recent)
        self.clear_link.setVisible(has_recent)
        if has_recent:
            self.recent.fit_rows(8)

        self.examples.clear()
        examples = sorted(example_projects(), key=self._example_order)
        for path in examples[:12]:
            s = summarize(path)
            it = QTreeWidgetItem([s.name, s.type_text, s.description or "", str(s.designs)])
            it.setIcon(0, theme.icon(s.icon, "text2"))
            it.setData(0, Qt.ItemDataRole.UserRole, str(path))
            it.setData(0, Qt.ItemDataRole.UserRole + 1, s.icon)
            it.setTextAlignment(3, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for c in self._SECONDARY["examples"]:
                it.setForeground(c, theme.qcolor("text2"))
            it.setToolTip(0, str(path))
            if s.description:
                it.setToolTip(2, s.description)
            self.examples.addTopLevelItem(it)
        self.examples_header.setVisible(bool(examples))
        self.examples.setVisible(bool(examples))
        self.examples_note.setVisible(bool(examples))
        self.link_examples.setVisible(bool(examples))
        if examples:
            self.examples.fit_rows(8)

    def recent_paths(self) -> list[str]:
        return [self.recent.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
                for i in range(self.recent.topLevelItemCount())]

    def _recent_clicked(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        self.openPathRequested.emit(item.data(0, Qt.ItemDataRole.UserRole))

    def _example_clicked(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        self.openExampleRequested.emit(item.data(0, Qt.ItemDataRole.UserRole))

    def _recent_menu(self, pos: QPoint) -> None:
        item = self.recent.itemAt(pos)
        if item is None:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        m = QMenu(self)
        m.addAction("Open", lambda: self.openPathRequested.emit(path))
        m.addAction("Remove from list", lambda: (remove_recent(path), self.refresh()))
        m.exec(self.recent.viewport().mapToGlobal(pos))

    def _clear_recent(self) -> None:
        clear_recent()
        self.refresh()

    def _fill_examples_menu(self) -> None:
        self.examples_menu.clear()
        for p in example_projects():
            self.examples_menu.addAction(summarize(p).name,
                                         lambda path=str(p): self.openExampleRequested.emit(path))

    def _show_examples_menu(self) -> None:
        self.examples_menu.exec(self.link_examples.mapToGlobal(self.link_examples.rect().bottomLeft()))


__all__ = [
    "KINDS",
    "MESH_EXTENSIONS",
    "PROJECT_SUFFIX",
    "START_TILES",
    "STUDY_TYPES",
    "NewStudyDialog",
    "ProjectSummary",
    "StartPage",
    "StudyTile",
    "StudyType",
    "available_study_types",
    "clear_recent",
    "create_study",
    "default_wing_spec",
    "display_folder",
    "example_projects",
    "format_modified",
    "open_example",
    "open_project",
    "paint_schematic",
    "physics_templates_available",
    "project_folder",
    "project_name",
    "project_path",
    "recent_projects",
    "remove_recent",
    "safe_name",
    "starter_simulations",
    "summarize",
    "unique_project_name",
]
