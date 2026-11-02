"""Application state: the one place the GUI reads and changes "what is open".

Pages never hold their own copy of the project. They read through ``AppState``,
mutate through its methods, and redraw when its signals fire. That keeps the
Geometry, Setup, Results and Compare pages consistent without them knowing
about each other.

The module also owns the few words every part of the shell uses to say *what*
a design or a simulation is (:func:`design_tag`, :func:`design_icon`,
:func:`analysis_label`, ...). A project can mix 2D outlines, 3D meshes and
wings, and the Project tree, the step bar, the Start page and the status bar
must call the same thing by the same name ("CFD 3D", "Wing"), or a user cannot
tell which of two similar entries is the 3D one.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QObject, Signal

from ..model import Design, FlowConditions, Simulation
from ..project import Project

# ---------------------------------------------------------------------------
# what a design / simulation is, in the shell's words
# ---------------------------------------------------------------------------

#: Kind words for designs ("Airfoil", "Vehicle", ...); ``model.KINDS`` order.
KIND_TEXT = {"airfoil": "Airfoil", "vehicle": "Vehicle", "generic": "Body", "wing": "Wing"}


def is_wing(design: Design | None) -> bool:
    """A finite-wing planform (``kind == "wing"``, 3D, analysed by VLM / LLT or lofted for CFD)."""
    return design is not None and design.kind == "wing"


def is_3d(design: Design | None) -> bool:
    """A three-dimensional design: a wing or one or more surface meshes."""
    return design is not None and int(getattr(design, "dimension", 2) or 2) == 3


def design_tag(design: Design | None) -> str:
    """"2D", "3D" (a surface mesh) or "Wing" ("" for no design).

    Wings get their own tag rather than "3D": they open a planform editor and
    a seconds-long lattice analysis, not a mesh viewer and a minutes-long CFD
    run, and the shell says which of the two the user is about to get.
    """
    if design is None:
        return ""
    if is_wing(design):
        return "Wing"
    return "3D" if is_3d(design) else "2D"


def design_icon(design: Design | None) -> str:
    """Icon name for a design: the 2D kind glyph, ``cube`` for meshes, ``wing`` for wings."""
    if design is None:
        return "layers"
    if is_wing(design):
        return "wing"
    if is_3d(design):
        return "cube"
    return {"airfoil": "airfoil", "vehicle": "car"}.get(design.kind, "layers")


def design_description(design: Design | None) -> str:
    """One factual line for tooltips: "Vehicle · 3D mesh · 2 parts, 48,212 triangles"."""
    if design is None:
        return ""
    kind = KIND_TEXT.get(design.kind, design.kind.capitalize())
    if is_wing(design):
        wing = getattr(design, "wing", None)
        n = len(wing.sections) if wing is not None else 0
        span = float(getattr(design, "span_ref", 0.0) or 0.0)
        bits = [f"{n} section{'s' if n != 1 else ''}"] if n else []
        if span > 0:
            bits.append(f"span {span:.3g} m")
        return "Wing · planform" + (f" · {', '.join(bits)}" if bits else "")
    if is_3d(design):
        meshes = list(getattr(design, "meshes", []) or [])
        tris = sum(len(m.faces) for m in meshes)
        parts = f"{len(meshes)} part{'s' if len(meshes) != 1 else ''}"
        return f"{kind} · 3D mesh · {parts}, {tris:,} triangles"
    n = len(design.bodies)
    return f"{kind} · 2D outline" + (f" · {n} bodies" if n > 1 else "")


def analysis_label(sim: Simulation | None, design: Design | None = None) -> str:
    """Short analysis name: "CFD 2D", "CFD 3D", "Polar" or "Wing".

    CFD carries its dimension because the same word means a seconds-long 2D
    run or a minutes-long 3D one; the design decides which (a wing sent to CFD
    is lofted and runs in 3D). Without the design the label is plain "CFD".
    """
    if sim is None:
        return ""
    if sim.analysis == "polar":
        return "Polar"
    if sim.analysis == "wing":
        return "Wing"
    if design is None:
        return "CFD"
    return "CFD 3D" if is_3d(design) else "CFD 2D"


def analysis_description(sim: Simulation | None, design: Design | None = None) -> str:
    """The analysis in words for tooltips: "CFD (Navier–Stokes, 3D)", "Wing (vortex lattice, viscous)"."""
    if sim is None:
        return ""
    if sim.analysis == "polar":
        return "Polar (viscous–inviscid)"
    if sim.analysis == "wing":
        ws = getattr(sim, "wing", None)
        method = "lifting line" if ws is not None and ws.method == "llt" else "vortex lattice"
        visc = ", viscous" if ws is None or ws.viscous else ", inviscid"
        return f"Wing ({method}{visc})"
    dim = "" if design is None else (", 3D" if is_3d(design) else ", 2D")
    return f"CFD (Navier–Stokes{dim})"


def analyses_for(design: Design | None) -> list[str]:
    """The analyses a design can have, default first: wings "wing" then "cfd" (lofted),
    single-element airfoils "cfd" then "polar", everything else "cfd"."""
    if design is None:
        return []
    if is_wing(design):
        return ["wing", "cfd"]
    if not is_3d(design) and design.kind == "airfoil" and len(design.bodies) == 1:
        return ["cfd", "polar"]
    return ["cfd"]


def _speed_text(cond: FlowConditions) -> str:
    v = float(cond.speed)
    return f"{v:.3g} m/s" if math.isfinite(v) else "m/s"


class AppState(QObject):
    projectChanged = Signal()                 # a different project was opened/created/closed
    designsChanged = Signal()                 # a design was added, removed or edited
    simulationsChanged = Signal()             # a simulation was added, removed, renamed or its status changed
    currentDesignChanged = Signal(str)        # design id ("" for none)
    currentSimulationChanged = Signal(str)    # simulation id ("" for none)
    resultChanged = Signal(str)               # simulation id whose result was (re)written
    reportsChanged = Signal()                 # a report was written to (or removed from) reports/
    navigateRequested = Signal(str)           # page key: "start","geometry","setup","simulate","results","compare","report"
    statusMessage = Signal(str, int)          # text, timeout ms (0 = sticky)
    messageLogged = Signal(str, str, str)     # level ("info"|"warning"|"error"), source, text -> Messages dock
    #: a shell command that is not a page (see :meth:`request`); the main window carries it out
    commandRequested = Signal(str)

    #: commands :meth:`request` understands (the main window implements them)
    COMMANDS = ("import-mesh", "new-wing")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project: Project | None = None
        self.design_id: str = ""
        self.simulation_id: str = ""
        self.compare_ids: list[str] = []      # simulations selected for comparison

    # -- project ----------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.design_id = ""
        self.simulation_id = ""
        self.compare_ids = []
        self.projectChanged.emit()
        if project is not None:
            if project.designs:
                self.select_design(next(iter(project.designs)))
            if project.simulations:
                self.select_simulation(next(iter(project.simulations)))

    def save(self) -> None:
        if self.project is not None:
            self.project.save()

    # -- selection --------------------------------------------------------
    @property
    def design(self) -> Design | None:
        if self.project is None or not self.design_id:
            return None
        return self.project.designs.get(self.design_id)

    @property
    def simulation(self) -> Simulation | None:
        if self.project is None or not self.simulation_id:
            return None
        return self.project.simulations.get(self.simulation_id)

    def select_design(self, design_id: str) -> None:
        if design_id != self.design_id:
            self.design_id = design_id
            self.currentDesignChanged.emit(design_id)

    def select_simulation(self, sim_id: str) -> None:
        if sim_id != self.simulation_id:
            self.simulation_id = sim_id
            sim = self.simulation
            if sim is not None and sim.design_id != self.design_id:
                self.select_design(sim.design_id)
            self.currentSimulationChanged.emit(sim_id)

    # -- mutation (always via the project store, then signal) -------------
    def add_design(self, design: Design, select: bool = True) -> None:
        assert self.project is not None
        self.project.add_design(design)
        self.project.save()
        self.designsChanged.emit()
        if select:
            self.select_design(design.id)

    def update_design(self, design: Design) -> None:
        assert self.project is not None
        self.project.update_design(design)
        self.project.save()
        self.designsChanged.emit()

    def remove_design(self, design_id: str) -> None:
        assert self.project is not None
        self.project.remove_design(design_id)
        self.project.save()
        if self.design_id == design_id:
            self.design_id = ""
            self.currentDesignChanged.emit("")
        self.designsChanged.emit()
        self.simulationsChanged.emit()

    def add_simulation(self, sim: Simulation, select: bool = True) -> None:
        assert self.project is not None
        self.project.add_simulation(sim)
        self.project.save()
        self.simulationsChanged.emit()
        if select:
            self.select_simulation(sim.id)

    def update_simulation(self, sim: Simulation) -> None:
        assert self.project is not None
        self.project.update_simulation(sim)
        self.project.save()
        self.simulationsChanged.emit()

    def new_simulation(self, design_id: str, analysis: str | None = None, *, select: bool = True,
                       device: str | None = None, precision: str | None = None) -> Simulation:
        """Add a simulation of a design with every default applied, and select it.

        For the entries that name an analysis explicitly ("New wing analysis",
        "New 3D CFD" of a wing) and for new 3D / wing studies; the Setup page's
        own "New simulation" keeps its comparable-template rules for 2D CFD.
        Defaults come from :func:`aerolume.simulation.defaults.default_simulation`
        (the same as the CLI), at the conditions of the design's latest
        simulation when it has one, so a wing's CFD run is at the speed its
        polar used. ``device`` / ``precision`` default to the preferences.

        Raises ``ValueError`` for an analysis the design cannot have (a polar of
        a mesh, a wing analysis of a car) or an unknown design.
        """
        assert self.project is not None
        from ..simulation.defaults import default_cfd_settings, default_simulation

        design = self.project.designs.get(design_id)
        if design is None:
            raise ValueError(f"no design {design_id!r} in this project")
        allowed = analyses_for(design)
        analysis = analysis or allowed[0]
        if analysis not in allowed:
            raise ValueError(f"a {design_tag(design)} {design.kind} cannot have a {analysis!r} analysis")
        if device is None or precision is None:
            from . import settings as app_settings
            s = app_settings.get()
            device = device or s.device
            precision = precision or s.precision
        latest = [s for s in self.project.simulations.values() if s.design_id == design_id]
        cond = (FlowConditions.from_dict(latest[-1].conditions.to_dict()) if latest else FlowConditions())
        sim = default_simulation(design, "", cond, device=device, precision=precision)
        if sim.analysis != analysis:
            sim.analysis = analysis
            sim.cfd = default_cfd_settings(design, cond, device=device, precision=precision)
        word = {"wing": "Wing polar", "polar": "Polar"}.get(analysis, "CFD 3D" if is_3d(design) else "CFD")
        sim.name = self.unique_simulation_name(design_id, f"{word} · {_speed_text(cond)}")
        self.add_simulation(sim, select=select)
        return sim

    def unique_design_name(self, base: str) -> str:
        """``base``, or ``base 2``, ``base 3``... unused among the project's designs."""
        taken = {d.name for d in self.project.designs.values()} if self.project is not None else set()
        name, k = base, 2
        while name in taken:
            name, k = f"{base} {k}", k + 1
        return name

    def unique_simulation_name(self, design_id: str, base: str) -> str:
        """``base``, or ``base 2``, ``base 3``... unused among the design's simulations."""
        taken = set()
        if self.project is not None:
            taken = {s.name for s in self.project.simulations.values() if s.design_id == design_id}
        name, k = base, 2
        while name in taken:
            name, k = f"{base} {k}", k + 1
        return name

    def remove_simulation(self, sim_id: str) -> None:
        assert self.project is not None
        self.project.remove_simulation(sim_id)
        self.project.save()
        if sim_id in self.compare_ids:
            self.compare_ids.remove(sim_id)
        if self.simulation_id == sim_id:
            self.simulation_id = ""
            self.currentSimulationChanged.emit("")
        self.simulationsChanged.emit()

    def notify_result(self, sim_id: str) -> None:
        self.resultChanged.emit(sim_id)

    def navigate(self, page: str) -> None:
        self.navigateRequested.emit(page)

    def request(self, command: str) -> None:
        """Ask the window for a shell action that belongs to no page.

        ``"import-mesh"`` (or ``"import-mesh:<path>"``) runs the 3D mesh import
        (file, units and orientation dialog) and adds the designs it returns;
        ``"new-wing"`` adds a default tapered wing. Pages and the Project tree
        use this instead of reaching for the main window, exactly as they use
        :meth:`navigate` for pages.
        """
        self.commandRequested.emit(command)

    def message(self, text: str, timeout_ms: int = 4000) -> None:
        self.statusMessage.emit(text, timeout_ms)

    def log(self, text: str, level: str = "info", source: str = "") -> None:
        """Append a line to the Messages dock (planner notes, warnings, run log)."""
        self.messageLogged.emit(level, source, text)
