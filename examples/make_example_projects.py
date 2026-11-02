"""Build the example projects that ship with Aerolume (File > Open example, Start page).

Usage (from the repository root, a CUDA GPU makes this take a few minutes)::

    python examples/make_example_projects.py               # all examples
    python examples/make_example_projects.py --only 3d     # the 3D ones (car3d and wing3d)
    python examples/make_example_projects.py --only heat   # heated cylinder, cavity and the room
    python examples/make_example_projects.py --only wing   # one example (car3d, wing3d, wing, cars, cylinder,
                                                           # cavity, room) or a group (3d, 2d, heat)
    python examples/make_example_projects.py --out DIR     # somewhere else (default: package data)

The projects are written to ``src/aerolume/examples/projects/`` because that
folder is package data: it is installed with the package, so the examples work
from a wheel as well as from a source checkout (``app/pages/start.py`` looks
there first, then in ``examples/projects/`` of a checkout, and lists the 3D
examples first). The app copies an example to ``~/Documents/Aerolume/Examples``
before opening it, so the shipped copy is never modified.

Every result in them is real: the simulations are created with the defaults a
new simulation gets in the app (``default_simulation``) and run with the same
function the Run button uses (``run_simulation``: the planner and ``run_cfd`` /
``run_polar`` / ``run_wing``), at Draft quality so the folders stay small
(fields are stored as float16 and compressed). Re-run this script after a
change to the solver, the planner or the result format, and commit the new
folders.

The 3D car is the one example whose result is trimmed to ship: its Draft run
stores a 1.8 M-cell mean volume and an equally large smoke snapshot (40 MB of
float16, about 15 MB in the project file). :func:`fit_volumes` block-averages
both volumes -- the smoke and its instantaneous velocity included -- until the
stored result fits ``CAR3D_BUDGET_MB``, and the result says so in its notes.
Everything else (forces, history, surface pressure, slices) is exactly what the
run produced.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "src" / "aerolume" / "examples" / "projects"

#: Megabytes the 3D car's stored result (``arrays.npz``, as the project writes it) may
#: take. Package-data examples stay under 15 MB each; 12 leaves room for the design
#: and the JSON, and one halving along the flow brings the Draft car well inside it.
CAR3D_BUDGET_MB = 12.0


def _progress(label: str):
    """A progress callback printing at most every 10 % (the runner calls it per chunk)."""
    last = [-1]

    def cb(fraction: float, message: str = "") -> None:
        step = int(fraction * 10)
        if step != last[0]:
            last[0] = step
            print(f"    {label}: {100 * fraction:5.1f} %  {message}", flush=True)
    return cb


def run_and_store(project, sim_id: str, *, trim=None) -> None:
    """Run one saved simulation exactly as the app's job manager does, and store the result.

    ``trim(result)``, when given, may shrink the result before it is stored
    (:func:`fit_volumes`).
    """
    from aerolume.model import now_iso
    from aerolume.simulation.runner import run_simulation

    sim = project.simulations[sim_id]
    design = project.designs[sim.design_id]
    t0 = time.perf_counter()

    def planned(plan) -> None:
        cells = " x ".join(str(n) for n in plan.grid)
        print(f"    {sim.name}: {cells} cells, {plan.device_name}, estimated {plan.estimate_label}", flush=True)

    result = run_simulation(design, sim, project=project, progress=_progress(sim.name), planned=planned)
    if trim is not None:
        trim(result)
    project.save_result(sim.id, result)
    sim.status, sim.finished = "done", now_iso()
    project.update_simulation(sim)
    project.save()
    dt = time.perf_counter() - t0
    if sim.analysis in ("polar", "wing"):
        ok = sum(p.converged for p in result.points)
        print(f"  {sim.name}: {ok}/{len(result.points)} points converged in {dt:.0f} s")
    else:
        print(f"  {sim.name}: {headline(result)}, converged {result.converged}, {dt:.0f} s")


def headline(result) -> str:
    """The numbers a CFD example is about: Nu and the heat rate, fan thrust, then the forces when it has any
    (a closed cavity or room has none, and "Cd nan ± nan" said nothing)."""
    import math
    parts = []
    nu = getattr(result, "nusselt", None)
    if nu is not None and math.isfinite(nu.mean):
        parts.append(f"Nu {nu.mean:.4g} ± {nu.stderr:.2g}")
    q = getattr(result, "heat_rate", float("nan"))
    if math.isfinite(q):
        parts.append(f"Q {q:.4g}")
    for fan in getattr(result, "fans_report", None) or []:
        if isinstance(fan.get("thrust"), (int, float)):
            parts.append(f"{fan.get('name', 'fan')} thrust {fan['thrust']:.4g}")
    for name in ("cd", "cl"):
        st = getattr(result, name, None)
        if st is not None and math.isfinite(st.mean):
            parts.append(f"{name.capitalize()} {st.mean:.4f} ± {st.stderr:.4f}")
    return ", ".join(parts) or "no measured quantities"


# ---------------------------------------------------------------------------
# keeping the 3D result small
# ---------------------------------------------------------------------------

def _block_mean(a: np.ndarray, f: tuple[int, int, int]) -> np.ndarray:
    """Average ``a`` (nz, ny, nx) over whole blocks of ``f`` = (fz, fy, fx) cells (ragged edges dropped)."""
    fz, fy, fx = f
    nz, ny, nx = (a.shape[0] // fz) * fz, (a.shape[1] // fy) * fy, (a.shape[2] // fx) * fx
    b = np.asarray(a[:nz, :ny, :nx], dtype=np.float32)
    return b.reshape(nz // fz, fz, ny // fy, fy, nx // fx, fx).mean(axis=(1, 3, 5))


def coarsen_volume(vol, f: tuple[int, int, int]):
    """``vol`` block-averaged by ``f`` = (fz, fy, fx); the origin stays, the spacing scales.

    The same whole-cell block average the runner uses to fit the solver grid into
    ``VOLUME_MAX_CELLS``, so a coarsened volume is what the run would have stored
    with a smaller cap (velocity, cp and smoke are cell averages; Q is averaged,
    not recomputed, which lowers its peaks the same way).
    """
    fields = {k: _block_mean(v, f) for k, v in vol.fields.items()}
    mask = None if vol.mask is None else _block_mean(vol.mask, f)
    dx, dy, dz = vol.spacing
    return dataclasses.replace(vol, fields=fields, mask=mask, spacing=(dx * f[2], dy * f[1], dz * f[0]))


def stored_bytes(result) -> int:
    """Size of the result's ``arrays.npz`` exactly as the project store writes it
    (packed float16, byte-shuffled, deflated): what the example will weigh."""
    from aerolume.project.fileio import npz_bytes
    from aerolume.project.results import pack_result

    return len(npz_bytes(pack_result(result)[1], level=1))


def fit_volumes(budget_mb: float):
    """A ``trim`` for :func:`run_and_store`: coarsen every volume of the result until the
    stored result is at most ``budget_mb``.

    The flow direction is halved first: smoke filaments and wake structures are
    stretched along the flow, so x loses the least, and the cross-stream detail
    is what the pictures need most. Then y and z, then x again.
    """
    steps = [(1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 1, 2)]

    def trim(result) -> None:
        before = size = stored_bytes(result)
        total = (1, 1, 1)
        for f in steps:
            if size <= budget_mb * 1e6:
                break
            if result.volume is not None:
                result.volume = coarsen_volume(result.volume, f)
            result.extra_volumes = [coarsen_volume(v, f) for v in result.extra_volumes]
            total = (total[0] * f[0], total[1] * f[1], total[2] * f[2])
            size = stored_bytes(result)
        if total == (1, 1, 1):
            return
        vol = result.volumes()[0]
        nz, ny, nx = next(iter(vol.fields.values())).shape
        result.notes.append(
            f"Example project: the stored flow volume and smoke snapshot were block-averaged a further "
            f"{total[2]} × {total[1]} × {total[0]} (to {nx} × {ny} × {nz} cells) so that the example ships "
            "small; forces, surface pressure and slices are as run. Run it again for the full volume.")
        print(f"    stored result {before / 1e6:.1f} MB -> {size / 1e6:.1f} MB "
              f"(volumes block-averaged {total[2]} x {total[1]} x {total[0]}, now {nx} x {ny} x {nz})", flush=True)
    return trim


# ---------------------------------------------------------------------------
# the examples
# ---------------------------------------------------------------------------

def _fresh(out: Path, name: str):
    """Create an empty project folder ``out/<name>.aerolume`` (replacing an older build)."""
    from aerolume.project import Project
    path = out / f"{name}.aerolume"
    if path.exists():
        shutil.rmtree(path)
    out.mkdir(parents=True, exist_ok=True)
    return Project.create(path, name)


def make_car3d(out: Path) -> Path:
    """A generic sedan in 3D at 30 m/s on a moving road, Draft quality, wheels spinning.

    What a new user opens to see the 3D side without waiting for a GPU run:
    surface pressure, the mean flow volume (streamlines, vortex iso-surfaces,
    slices) and the smoke of the last time step. The car is
    :func:`aerolume.geometry3d.sedan` -- generated, so the example holds no
    third-party model -- and the simulation is what "New simulation" gives a 3D
    vehicle (``default_simulation``).
    """
    import aerolume.geometry3d as G
    from aerolume.model import FlowConditions
    from aerolume.simulation.defaults import default_simulation, simulation_name

    project = _fresh(out, "Sedan in 3D")
    project.description = "Draft 3D CFD of a generic sedan at 30 m/s: smoke, streamlines, surface pressure."
    design = G.design_from_meshes(G.sedan(), "Sedan", "vehicle", source={"type": "primitive", "shape": "sedan"})
    project.add_design(design)
    sim = default_simulation(design, conditions=FlowConditions(speed=30.0, alpha_deg=0.0))
    sim.name = simulation_name(design, sim)
    project.add_simulation(sim)
    project.save()
    run_and_store(project, sim.id, trim=fit_volumes(CAR3D_BUDGET_MB))
    return project.path


def make_wing3d(out: Path) -> Path:
    """A tapered NACA 2412 wing (span 3 m, taper 0.5, 3° dihedral, 2° washout) at 25 m/s.

    The finite-wing analysis a new wing study starts with -- a viscous
    vortex-lattice polar, seconds to run -- giving CL, CD split into induced and
    profile drag, span loading and the stall-onset estimate. Kept apart from the
    car so that each example opens on one subject.
    """
    import aerolume.geometry3d as G
    from aerolume.model import FlowConditions
    from aerolume.simulation.defaults import default_simulation, simulation_name

    project = _fresh(out, "Tapered wing")
    project.description = "Tapered NACA 2412 wing, span 3 m, at 25 m/s: polar, span loading, induced drag."
    wing = G.tapered_wing(3.0, 0.40, 0.20, dihedral_deg=3.0, twist_deg=-2.0, airfoil="2412")
    design = G.design_from_wing(wing, "Tapered wing", source={"type": "wing", "template": "tapered"})
    project.add_design(design)
    sim = default_simulation(design, conditions=FlowConditions(speed=25.0))     # -4 to 12 deg, as a new wing
    sim.name = simulation_name(design, sim)
    project.add_simulation(sim)
    project.save()
    run_and_store(project, sim.id)
    return project.path


def make_wing(out: Path) -> Path:
    """NACA 2412, chord 1 m, at Re = 1e6: a viscous polar and a Draft CFD run at 4 degrees.

    The same two simulations a new airfoil study starts with, so the example shows
    what pressing Run on them gives. The CFD run uses the default capped Reynolds
    number (a resolved low-Re flow, labelled as such); the polar is the answer at
    the real Re.
    """
    from aerolume.geometry import design_from_airfoil, naca
    from aerolume.model import AirfoilSettings, FlowConditions, Simulation
    from aerolume.simulation.plan import default_cfd_settings

    project = _fresh(out, "Wing section NACA 2412")
    project.description = "NACA 2412, chord 1 m, Re 1.0×10⁶: viscous polar and a Draft CFD run at α 4°."
    design = design_from_airfoil(naca("2412"), "NACA 2412", 1.0, source={"type": "naca", "code": "2412"})
    project.add_design(design)
    cond = FlowConditions()
    cond.speed = 1.0e6 * cond.fluid.viscosity / (cond.fluid.density * design.length_ref)   # Re = 1e6
    polar = Simulation(name="Polar · Re 1.0×10⁶", design_id=design.id, analysis="polar", conditions=cond,
                       airfoil=AirfoilSettings(alpha_start=-4.0, alpha_end=14.0, alpha_step=1.0))
    cfd_cond = FlowConditions(fluid=cond.fluid, speed=cond.speed, alpha_deg=4.0)
    cfd = Simulation(name="CFD · α 4°", design_id=design.id, analysis="cfd", conditions=cfd_cond,
                     cfd=default_cfd_settings(design, cfd_cond))
    for sim in (polar, cfd):
        project.add_simulation(sim)
    project.save()
    for sim in (polar, cfd):
        run_and_store(project, sim.id)
    return project.path


def make_cars(out: Path) -> Path:
    """A sedan and a fastback, both 30 m/s on a moving road, Draft quality.

    The templates' default geometry has no wheels: in 2D a wheel touching the road seals
    the underbody (see docs/user_guide_simulation.md, "Reading results honestly").

    Two designs with the same set-up, ready for the Compare page. The settings are
    the ones a new vehicle study starts with.
    """
    from aerolume.geometry import templates
    from aerolume.model import FlowConditions, GroundConfig, Simulation
    from aerolume.simulation.plan import default_cfd_settings

    project = _fresh(out, "Sedan vs fastback")
    project.description = "Two side profiles at 30 m/s on a moving road (Draft CFD), ready to compare."
    sims = []
    for tmpl, name in (("sedan", "Sedan"), ("fastback", "Fastback")):
        design = templates.make(tmpl)
        design.name = name
        project.add_design(design)
        cond = FlowConditions(speed=30.0, alpha_deg=0.0)
        sim = Simulation(name="CFD · 30 m/s", design_id=design.id, analysis="cfd", conditions=cond,
                         ground=GroundConfig(enabled=True, moving=True, rotate_wheels=True),
                         cfd=default_cfd_settings(design, cond))
        project.add_simulation(sim)
        sims.append(sim)
    project.save()
    for sim in sims:
        run_and_store(project, sim.id)
    return project.path


def _api_project(out: Path, name: str):
    """A fresh project opened through the scripting API (``aerolume.api.Project``): the
    reference cases (:func:`aerolume.api.cavity_case` and friends) build their design and
    simulation there exactly as a script or the Assistant would."""
    import aerolume as al
    path = out / f"{name}.aerolume"
    if path.exists():
        shutil.rmtree(path)
    out.mkdir(parents=True, exist_ok=True)
    return al.new_project(path, name=name)


def make_heated_cylinder(out: Path) -> Path:
    """Forced convection from a heated cylinder at Re 40 (``heated_cylinder_case``), Draft.

    A 10 mm cylinder held 10 K above a 15 °C air stream, buoyancy off: the run gives the
    mean Nusselt number with its standard error, the temperature field and the wake, and
    the Results page compares the Nusselt number with the Lange, Durst & Breuer (1998)
    correlation that the case carries as its reference.
    """
    import aerolume as al
    ap = _api_project(out, "Heated cylinder")
    ap.raw.description = ("Forced convection from a 10 mm cylinder 10 K above the air stream at Re 40 (Draft CFD): "
                          "Nusselt number against Lange et al. (1998), temperature field.")
    sim = ap.add_simulation(al.heated_cylinder_case(re=40), quality="draft")
    ap.save()
    run_and_store(ap.raw, sim.id)
    return ap.path


def make_cavity(out: Path) -> Path:
    """The differentially heated square cavity at Ra 10⁵ (``cavity_case``), Draft.

    The natural-convection benchmark: air in a closed 0.1 m box, the left wall hot, the
    right wall cold, the others adiabatic. Its reference is de Vahl Davis' (1983) mean
    Nusselt number of the hot wall, Nu₀ = 4.509; the Results page shows the measured value
    next to it and the run's energy balance in its diagnostics.
    """
    import aerolume as al
    ap = _api_project(out, "Heated cavity Ra 1e5")
    ap.raw.description = ("Natural convection in a differentially heated square cavity, Ra 10⁵, Pr 0.71 (Draft CFD): "
                          "a validation case against de Vahl Davis (1983).")
    sim = ap.add_simulation(al.cavity_case(ra=1e5), quality="draft")
    ap.save()
    run_and_store(ap.raw, sim.id)
    return ap.path


def make_room(out: Path) -> Path:
    """A 4 m × 3 m room (2D) with a thrust fan, a heater, a supply jet and a vent, Draft.

    Everything the placed-physics tools do in one study: the fan (1 N per metre of
    span, ``fan_room_case``) drives the room's circulation, a 60 W/m heater on the floor
    makes a buoyant plume, a slow supply jet under the ceiling is balanced by a vent on
    the opposite wall, and two smoke emitters of different colours mark the flow. The
    ambient temperature is the fluid's (15 °C), so the density and viscosity match it.
    It is an illustration, not a validation: a real room is turbulent (Ra ~ 10¹⁰ here),
    which the Draft grid does not resolve, and the run's warnings say so. The fan's
    delivered thrust is the case's consistency check.
    """
    import aerolume as al
    ap = _api_project(out, "Room with a fan and a heater")
    ap.raw.description = ("A 4 m × 3 m room (2D, Draft CFD): thrust fan, floor heater, supply jet and vent, smoke "
                          "emitters. An illustration of the placed-physics tools, not a validation.")
    smoke = [{"kind": "rake", "position": (0.4, 0.4), "end": (0.4, 2.6), "count": 8, "color": "#f0a040",
              "name": "Smoke rake"},
             {"kind": "point", "position": (3.0, 0.45), "color": "#e04040", "name": "Plume tracer"}]
    sim = ap.add_simulation(
        al.fan_room_case(thrust=1.0), quality="draft", emitters=smoke,
        thermal={"enabled": True, "t_ambient": 15.0, "buoyancy": True},
        heat_sources=[{"name": "Heater", "position": (3.0, 0.25), "size": (0.4, 0.2), "power": 60.0}],
        ports=[{"kind": "jet", "name": "Supply", "position": (0.05, 2.7), "direction": (1, 0), "size": 0.2,
                "speed": 0.5},
               {"kind": "vent", "name": "Vent", "position": (3.95, 2.7), "direction": (1, 0), "size": 0.2,
                "speed": 0.5}],
        name="Room · fan 1 N/m · heater 60 W/m")
    ap.save()
    run_and_store(ap.raw, sim.id)
    return ap.path


EXAMPLES = {"car3d": make_car3d, "wing3d": make_wing3d, "wing": make_wing, "cars": make_cars,
            "cylinder": make_heated_cylinder, "cavity": make_cavity, "room": make_room}
#: Names that build several examples at once.
GROUPS = {"3d": ("car3d", "wing3d"), "2d": ("wing", "cars"), "heat": ("cylinder", "cavity", "room")}


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    ap.add_argument("--only", choices=sorted(EXAMPLES) + sorted(GROUPS), action="append",
                    help="build only these examples (3d = car3d and wing3d, 2d = wing and cars, "
                         "heat = cylinder, cavity and room)")
    a = ap.parse_args(argv)
    keys = [k for name in (a.only or list(EXAMPLES)) for k in GROUPS.get(name, (name,))]
    for key in dict.fromkeys(keys):
        print(f"== {key}", flush=True)
        path = EXAMPLES[key](a.out)
        print(f"  wrote {path} ({folder_size(path) / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
