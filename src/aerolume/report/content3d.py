"""Report content for 3D Navier–Stokes results and finite-wing sweeps (DESIGN.md section 7).

:mod:`.content` owns the section order and the 2D material; for every 3D entry it
asks this module for the blocks. The rules are the 2D ones (every number with its
validity context), plus what 3D adds:

* **Whole-body forces.** 3D coefficients are on a reference *area* (frontal area of
  a vehicle, planform area of a wing) and forces are in newtons for the complete
  body, so the report never mixes them with 2D per-span numbers: a comparison table
  holds 3D CFD runs only, or wing sweeps only.
* **Symmetry planes.** A half-domain run reports side force, roll and yaw as zero
  by construction; the tables say "0 (symmetry)" instead of printing a number that
  looks measured.
* **What has been validated.** The 3D solver is validated on spheres (laminar,
  Re 20–300). On an Ahmed-type car body it over-predicts the wind-tunnel drag by a
  factor of about 2.1–2.2 at Draft and Standard quality (wall-unresolved LES on
  immersed walls). The vortex lattice and lifting line are validated against
  lifting-surface theory and NACA TN 1270 wind-tunnel wings, with a known
  under-prediction of profile drag. The methodology section quotes those numbers
  and the summary flags runs that are outside the validated range.

Figures are built lazily (the renderer calls the builder), so a 3D render that
cannot be produced on a machine without OpenGL becomes a note in the report
rather than a failed report.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.fluid import display_name as fluid_name
from ..model import FieldData
from ..viz import figures as F
from . import fmt
from . import render3d as R
from .common import (
    ROUGH,
    ROUGH_NOTE,
    STDERR_OK,
    STDERR_WARN,
    Entry,
    compare_rows,
    comparison_warnings,
    delta_cell,
    dyn_pressure,
    err_text,
    fig,
    pm_text,
    rel_err,
    rough,
    sci,
    turbulence_text,
)
from .document import KPI, Bullets, Callout, Cell, KeyValues, KPIRow, Paragraph, Table

# Validity thresholds (DESIGN.md 7.12, docs/theory_ns3d.md section 10, docs/validation_wing.md).
SPHERE_RE_MAX = 300.0           # highest Re of the laminar 3D validation cases (sphere)
MACH_LIMIT = 0.3
SPEED_OF_SOUND_AIR = 340.3
#: measured over-prediction of an Ahmed-type body's drag (simulation agent, Draft / Standard)
AHMED_FACTOR = "2.1–2.2"

KEYS3 = ("cd", "cs", "cl", "c_roll", "cm", "cn")
#: coefficients that a symmetry-plane run reports as zero by construction
ANTISYMMETRIC = ("cs", "c_roll", "cn")
NAMES3 = {"cd": "Drag CD", "cs": "Side force CS", "cl": "Lift CL", "c_roll": "Rolling moment Croll",
          "cm": "Pitching moment Cm", "cn": "Yawing moment Cn"}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _theme(style: Any) -> str:
    return "dark" if getattr(style, "name", style) == "dark" else "light"


def _surface_bounds(surfaces: Sequence[Any]) -> tuple[float, float, float, float, float, float] | None:
    pts = [np.asarray(s.vertices, float) for s in surfaces if np.size(getattr(s, "vertices", [])) >= 3]
    if not pts:
        return None
    v = np.vstack(pts)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


def _on_road(e: Entry) -> bool:
    g = getattr(e.sim, "ground", None)
    return bool(getattr(g, "enabled", False))


def _design_on_road(d: Any) -> bool:
    return bool((getattr(d, "source", {}) or {}).get("on_road")) or getattr(d, "kind", "") == "vehicle"


def _ref_name(e: Entry) -> tuple[str, str]:
    """(reference-area wording, reference-length wording) of a 3D design's coefficients."""
    if e.kind == "wing":
        return "planform area", "mean aerodynamic chord"
    if e.kind == "vehicle":
        return "frontal area", "vehicle length"
    return "frontal area", "reference length"


def _stall_missed(m: dict) -> bool:
    return m.get("cl_max_reached") == 0.0


def _cl_max_text(m: dict, digits: int = 3) -> str:
    t = fmt.num(m.get("cl_max"), digits)
    return f"≥ {t}" if _stall_missed(m) and t != fmt.DASH else t


def _oswald_fit_text(m: dict) -> str:
    """"parabolic fit, CL a–b": the Oswald factor depends on the CL range it was fitted over
    (analysis/wing.py), so it is never quoted without it."""
    lo, hi = m.get("oswald_cl_lo"), m.get("oswald_cl_hi")
    if fmt.finite(lo) and fmt.finite(hi):
        return f"parabolic fit, CL {fmt.fixed(lo, 2)}–{fmt.fixed(hi, 2)}"
    return "parabolic fit"


def wing_metrics(r: Any) -> dict[str, float]:
    """``analysis.wing_metrics`` of a sweep, or {} when it cannot be computed."""
    try:
        from ..analysis.wing import wing_metrics as wm
        return wm(r)
    except Exception:  # noqa: BLE001 - a damaged sweep must not sink the report
        return {}


def _airfoil_coords(project: Any) -> Any:
    """Resolver for "design:<id>" section airfoils (lofting a wing), or None."""
    if project is None:
        return None
    try:
        from ..simulation.wing import airfoil_resolver
        return airfoil_resolver(project)
    except Exception:  # noqa: BLE001 - optional; NACA sections loft without it
        return None


def _design_meshes(design: Any, airfoil_coords: Any = None) -> list[Any]:
    meshes = list(getattr(design, "meshes", []) or [])
    if meshes and all(len(m.faces) for m in meshes):
        return meshes
    wing = getattr(design, "wing", None)
    if wing is not None:
        from ..geometry3d import loft
        return [loft(wing, airfoil_coords=airfoil_coords, name=getattr(design, "name", "Wing"))]
    return meshes


# ---------------------------------------------------------------------------
# lazily built figures (called by the renderer with style= and figsize=)
# ---------------------------------------------------------------------------

#: Camera turn (azimuth, elevation in degrees) from the iso preset for a wing: seen from the
#: iso direction a high-aspect-ratio wing is a thin diagonal sliver; turned towards the front
#: and raised, the span runs across the picture and the planform reads.
WING_ORBIT = (30.0, 25.0)


def figure_geometry3d(design: Any, *, airfoil_coords: Any = None, style: Any = "print",
                      figsize: tuple[float, float] | None = None) -> Any:
    """Grey iso / side / front / top render of a 3D design (iso only for a wing)."""
    meshes = _design_meshes(design, airfoil_coords)
    if not meshes:
        raise ValueError("the design has no surface mesh")
    wing = getattr(design, "kind", "") == "wing"
    ground = 0.0 if _design_on_road(design) else None

    def build(s: Any) -> None:
        s.add_meshes(meshes)
    v = R.render_views(build, ("iso",) if wing else ("iso", "side", "front", "top", "back"), ground=ground,
                       theme=_theme(style), orbit=WING_ORBIT if wing else None)
    return F.fig_views3d(v.images, v.names, v.titles, v.extents, ground=ground, style=style, figsize=figsize)


def figure_cover3d(design: Any, *, airfoil_coords: Any = None, style: Any = "print",
                   figsize: tuple[float, float] | None = None) -> Any:
    """One iso render of the design for the title page."""
    meshes = _design_meshes(design, airfoil_coords)
    if not meshes:
        raise ValueError("the design has no surface mesh")
    ground = 0.0 if _design_on_road(design) else None

    def build(s: Any) -> None:
        s.add_meshes(meshes)
    wing = getattr(design, "kind", "") == "wing"
    v = R.render_views(build, ("iso",), ground=ground, theme=_theme(style), orbit=WING_ORBIT if wing else None)
    return F.fig_views3d(v.images, v.names, v.titles, v.extents, style=style, figsize=figsize)


def figure_surface3d(surfaces: Sequence[Any], field: str, *, ground: float | None = None,
                     views: Sequence[str] = ("iso", "side", "front", "top"), relative: bool = False,
                     label: str | None = None, style: Any = "print",
                     figsize: tuple[float, float] | None = None) -> Any:
    """Surface field (Cp, Cf or the sensitivity) on every mesh of a 3D result, several views.

    ``relative`` scales the field by its 98th-percentile magnitude onto [-1, 1] (the
    sensitivity map: its pattern is reliable, its size only indicative); ``label``
    replaces the colour-bar title.
    """
    surfs = [s for s in surfaces if getattr(s, field, None) is not None]
    if not surfs:
        raise ValueError(f"no surface carries {field!r}")

    def build(s: Any) -> str:
        if relative:
            allv = np.concatenate([np.abs(np.asarray(getattr(x, field), float)).ravel() for x in surfs])
            allv = allv[np.isfinite(allv)]
            smax = float(np.percentile(allv, 98)) if allv.size else 1.0
            smax = smax or 1.0
            handles = [s.add_mesh(x, np.clip(np.nan_to_num(np.asarray(getattr(x, field), float)) / smax, -1, 1),
                                  cmap="RdBu_r", range=(-1.0, 1.0), field=field) for x in surfs]
        else:
            handles = s.add_meshes(surfs, field)
        others = [x for x in surfaces if all(x is not y for y in surfs)]
        if others:
            s.add_meshes(others)
        return handles[0]
    v = R.render_views(build, views, ground=ground, theme=_theme(style))
    color = v.color
    if color is not None and label:
        color = R.ColorSpec(color.cmap, color.vmin, color.vmax, label, "", color.extend)
    return F.fig_views3d(v.images, v.names, v.titles, v.extents, color=color, ground=ground, style=style,
                         figsize=figsize)


def _volume_bounds(volume: Any) -> tuple[float, float, float, float, float, float] | None:
    """Outer bounds of a stored volume, mirrored to both sides for a symmetry-plane run."""
    fields = getattr(volume, "fields", {}) or {}
    if not fields:
        return None
    nz, ny, nx = next(iter(fields.values())).shape
    o, h = volume.origin, volume.spacing
    y0 = -(o[1] + ny * h[1]) if getattr(volume, "mirrored", False) else o[1]
    return (o[0], o[0] + nx * h[0], y0, o[1] + ny * h[1], o[2], o[2] + nz * h[2])


def _flow_focus(surfaces: Sequence[Any], volume: Any) -> tuple[float, ...] | None:
    """Body plus near wake (0.3 lengths ahead, 1.0 behind), clipped to the stored volume."""
    b = _surface_bounds(surfaces)
    if b is None:
        return None
    lx = max(b[1] - b[0], 1e-9)
    wy = max(b[3] - b[2], 1e-9)
    hz = max(b[5] - b[4], 1e-9)
    box = [b[0] - 0.3 * lx, b[1] + 1.0 * lx, b[2] - 0.35 * wy, b[3] + 0.35 * wy, b[4] - 0.3 * hz, b[5] + 0.45 * hz]
    vb = _volume_bounds(volume) if volume is not None else None
    if vb is not None:
        box = [max(box[0], vb[0]), min(box[1], vb[1]), max(box[2], vb[2]), min(box[3], vb[3]),
               max(box[4], vb[4]), min(box[5], vb[5])]
    return tuple(box)


def figure_flow3d(surfaces: Sequence[Any], volume: Any, kind: str, *, ground: float | None = None,
                  style: Any = "print", figsize: tuple[float, float] | None = None) -> Any:
    """Streamlines ("streamlines") or the Q-criterion iso-surface ("q") around the grey body, iso view.

    Streamlines start on a horizontal rake upstream at 35 % of the body height, 15 % wider than
    the body: low enough that the lines are deflected sideways and show the wake and its
    vortices, rather than passing over the roof and hiding the body.
    """
    if volume is None:
        raise ValueError("no volume data stored")
    focus = _flow_focus(surfaces, volume)
    b = _surface_bounds(surfaces)

    def build(s: Any) -> str:
        s.add_meshes(surfaces)
        if kind == "streamlines":
            from ..viz3d import rake_seeds
            vb = _volume_bounds(volume)
            seeds: Any = "rake"
            if b is not None and vb is not None:
                seeds = rake_seeds(b, vb, "rake", 24)
                seeds[:, 2] = b[4] + 0.35 * (b[5] - b[4])
            return s.add_streamlines(volume, seeds, color_by="speed", direction="forward")
        # Scene.add_isosurface trims the wall skin of a vortex field itself, smoothly, within
        # WALL_CLEARANCE_CELLS of the meshes just added; pre-zeroing a voxel band here used to
        # leave its staircase as a saw-tooth rim on the picture
        return s.add_isosurface(volume, "q", color_by="speed")
    v = R.render_views(build, ("iso",), ground=ground, focus=focus, theme=_theme(style))
    return F.fig_views3d(v.images, v.names, v.titles, v.extents, color=v.color, style=style, figsize=figsize)


#: in-plane velocity components of a slice: (horizontal, vertical) per plane normal
_IN_PLANE = {"y": ("u", "w"), "z": ("u", "v"), "x": ("v", "w")}
_AXIS_LABELS = {"y": ("x [m]", "z [m]"), "z": ("x [m]", "y [m]"), "x": ("y [m]", "z [m]")}


def _slice_bodies(surfaces: Sequence[Any], axis: str, position: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    loops: list[np.ndarray] = []
    lines: list[np.ndarray] = []
    for s in surfaces:
        for sec in R.plane_section(s.vertices, s.faces, axis, position):
            if sec.closed and len(sec.points) >= 3:
                loops.append(sec.points)
            else:
                lines.append(sec.points)
    return loops, lines


def figure_slice3d(result: Any, index: int, field: str, *, streamlines: bool = False, ground: bool = False,
                   q_inf: float | None = None, style: Any = "print",
                   figsize: tuple[float, float] | None = None) -> Any:
    """One stored plane cut of a 3D result as an engineering plot (metric axes, colour bar).

    The bodies are drawn with their exact plane section (not the staircase of the grid
    mask); in-plane streamlines follow the two velocity components lying in the plane.
    ``field`` "ux" is the streamwise velocity (for cross-flow cuts behind the body).
    """
    sd = result.slices[index]
    f = {k: np.asarray(a, float) for k, a in sd.fields.items()}
    if "speed" not in f and all(k in f for k in ("u", "v", "w")):
        f["speed"] = np.sqrt(f["u"] ** 2 + f["v"] ** 2 + f["w"] ** 2)
    if "cp" not in f and "p" in f and q_inf:
        f["cp"] = f["p"] / q_inf
    h, v = _IN_PLANE[sd.axis]
    fields = {}
    if h in f and v in f:
        fields["u"], fields["v"] = f[h], f[v]
    src = "u" if field == "ux" else field
    if src not in f:
        raise KeyError(f"slice has no field {src!r}")
    fields[field] = f[src]
    fd = FieldData(extent=tuple(sd.extent), fields=fields, mask=sd.mask, label=sd.label)
    surfaces = list(getattr(result, "surface", []) or [])
    loops, lines = _slice_bodies(surfaces, sd.axis, float(sd.position))
    b = _surface_bounds(surfaces)
    view: Any = "auto"
    if b is not None and (sd.axis == "x" or not loops):
        ext = sd.extent
        if sd.axis == "x":
            wy, hz = b[3] - b[2], b[5] - b[4]
            pad = 0.55 * max(wy, hz)
            view = (b[2] - pad, b[3] + pad, (0.0 if ground else b[4] - pad), b[5] + pad)
        else:
            lx = b[1] - b[0]
            r0, r1 = (b[4], b[5]) if sd.axis == "y" else (b[2], b[3])
            view = (b[0] - 0.6 * lx, b[1] + 1.6 * lx, r0 - 0.6 * (r1 - r0), r1 + 0.6 * (r1 - r0))
        view = (max(view[0], ext[0]), min(view[1], ext[1]), max(view[2], ext[2]), min(view[3], ext[3]))
    xl, yl = _AXIS_LABELS[sd.axis]
    return F.fig_field(fd, field, loops or None, streamlines=streamlines and "u" in fields, view=view,
                       ground=ground and sd.axis in ("y", "x"), q_inf=q_inf, xlabel=xl, ylabel=yl,
                       lines=lines, style=style, figsize=figsize, stream_density=1.0)


def _centre_cuts(surfaces: Sequence[Any], position: float) -> list[F.SurfaceCut]:
    """The largest closed loop of each surface in the plane y = ``position``, with Cp and streamwise Cf."""
    out = []
    for s in surfaces:
        cp = np.asarray(s.cp, float)
        tau = getattr(s, "tau", None)
        vals = cp[:, None] if tau is None else np.column_stack([cp, np.asarray(tau, float).reshape(-1, 3)[:, 0]])
        secs = [x for x in R.plane_section(s.vertices, s.faces, "y", position, vals) if x.closed and len(x.points) > 8]
        if not secs:
            continue
        best = max(secs, key=lambda x: abs(R.polygon_area(x.points)))
        val = np.asarray(best.values, float).reshape(len(best.points), -1)
        out.append(F.SurfaceCut(str(s.body), best.points[:, 0], best.points[:, 1], val[:, 0],
                                val[:, 1] if val.shape[1] > 1 else None))
    return out


def figure_centreline_cp(surfaces: Sequence[Any], position: float, *, kind: str = "body", style: Any = "print",
                         figsize: tuple[float, float] | None = None) -> Any:
    """Cp (and streamwise Cf) along the cut of the surface with the plane y = ``position``."""
    cuts = _centre_cuts(surfaces, position)
    if not cuts:
        raise ValueError("the centre plane does not cut a closed surface section")
    has_cf = any(c.cf is not None for c in cuts)
    return F.fig_cp(cuts, kind=kind, show_cf=has_cf, labels=[c.body for c in cuts] if len(cuts) > 1 else None,
                    style=style, figsize=figsize or (6.7, 4.3 if has_cf else 3.5))


def figure_wing_render(design: Any, *, airfoil_coords: Any = None, style: Any = "print",
                       figsize: tuple[float, float] | None = None) -> Any:
    return figure_geometry3d(design, airfoil_coords=airfoil_coords, style=style, figsize=figsize)


# ---------------------------------------------------------------------------
# validity, summary
# ---------------------------------------------------------------------------

def assess_cfd3d(e: Entry) -> list[tuple[str, str]]:
    """Validity badges of a 3D Navier–Stokes result."""
    r = e.result
    out: list[tuple[str, str]] = [("Statistically converged", "ok") if r.converged else ("Not converged", "warn")]
    rel = rel_err(r.cd)
    if fmt.finite(rel):
        tone = "ok" if rel < STDERR_OK else ("warn" if rel < STDERR_WARN else "bad")
        if rough(r.cd):              # an estimated error bar never earns the "good" colour
            tone = "warn" if tone == "ok" else tone
        out.append((f"CD uncertainty ±{fmt.num(rel * 100, 2)} %" + (" (rough estimate)" if rough(r.cd) else ""),
                    tone))
    else:
        from . import content_physics as CP
        out += CP.uncertainty_badge(e) if not CP.has_forces(e) else [("Uncertainty not estimated", "warn")]
    if fmt.finite(r.re_sim) and fmt.finite(r.re_real):
        if r.re_sim < 0.95 * r.re_real:
            out.append((f"Re scaled: {fmt.reynolds(r.re_sim)} of {fmt.reynolds(r.re_real)}", "warn"))
        else:
            out.append(("Full-scale Reynolds number", "ok"))
        if r.re_sim <= SPHERE_RE_MAX:
            out.append((f"Within the laminar benchmark range (Re ≤ {SPHERE_RE_MAX:.0f})", "ok"))
        else:
            out.append(("Absolute forces indicative (wall-unresolved LES)", "warn"))
    nx, ny, nz = (tuple(r.grid) + (0, 0, 0))[:3]
    out.append((f"Grid {nx}×{ny}×{nz} ({_cells_text(r.cells or nx * ny * nz)} cells)", "info"))
    if r.symmetry_plane:
        out.append(("Half domain (symmetry plane y = 0)", "info"))
    out.append(("3D · forces on the whole body" if fmt.finite(r.cd.mean) else "3D", "info"))
    speed = getattr(e.sim.conditions, "speed", 0.0)
    if speed / SPEED_OF_SOUND_AIR > MACH_LIMIT and "Water" not in e.sim.conditions.fluid.name:
        out.append((f"Mach ≈ {speed / SPEED_OF_SOUND_AIR:.2f} > {MACH_LIMIT}: incompressible model", "bad"))
    return out


def assess_wing(e: Entry) -> list[tuple[str, str]]:
    """Validity badges of a finite-wing sweep."""
    r = e.result
    n, nc = len(r.points), sum(1 for p in r.points if p.converged)
    out = [(f"{nc} of {n} angles converged", "ok" if nc == n else ("warn" if nc else "bad")),
           (f"Re (MAC) {fmt.reynolds(r.re_mac)}", "info")]
    method = "Vortex lattice" if r.method == "vlm" else "Nonlinear lifting line"
    out.append((f"{method} + viscous strips" if r.viscous else f"{method}, inviscid (no profile drag)",
                "info" if r.viscous else "warn"))
    m = wing_metrics(r)
    if m and _stall_missed(m):
        out.append(("Stall not reached in the sweep", "info"))
    mach = float(getattr(getattr(e.sim, "wing", None), "mach", 0.0) or 0.0)
    if mach >= 0.6:
        out.append((f"Mach {mach:.2f}: beyond Prandtl–Glauert", "bad"))
    return out


def _cells_text(n: int) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.0f} k"


def kpis_cfd3d(e: Entry) -> KPIRow:
    r = e.result

    def se_sub(st: Any, missing: str) -> str:
        if not fmt.finite(st.stderr) or st.stderr == 0:
            return missing
        return f"± {err_text(st)} " + ("(rough estimate)" if rough(st) else "(1 s.e.)")
    lift_label = "Lift (− = downforce)" if e.kind == "vehicle" else "Lift"
    return KPIRow([
        KPI("Drag coefficient CD", fmt.mean_only(r.cd.mean, r.cd.stderr), sub=se_sub(r.cd, "no uncertainty estimate")),
        KPI("Lift coefficient CL", fmt.mean_only(r.cl.mean, r.cl.stderr), sub=se_sub(r.cl, "")),
        KPI("Lift-to-drag L/D", _ld_text(r), sub=fmt.eq("Cm", fmt.mean_only(r.cm.mean, r.cm.stderr))),
        KPI("Drag D", _si(r.drag, "N").rsplit(" ", 1)[0], "N", sub=f"{lift_label} {_si(r.lift, 'N')}"),
    ])


def kpis_wing(e: Entry) -> KPIRow | None:
    m = wing_metrics(e.result)
    if not m:
        return None
    viscous = bool(e.result.viscous)
    items = []
    if viscous and fmt.finite(m.get("ld_max")):
        items.append(KPI("Best lift-to-drag (L/D)max", fmt.fixed(m["ld_max"], 1),
                         sub=f"at α = {fmt.angle(m.get('alpha_ld_max'))}, CL = {fmt.num(m.get('cl_at_ld_max'), 3)}"))
    items.append(KPI("Maximum lift CL,max", _cl_max_text(m),
                     sub=(f"still rising at α = {fmt.angle(m.get('alpha_cl_max'))}" if _stall_missed(m)
                          else f"at α = {fmt.angle(m.get('alpha_cl_max'))}")))
    if viscous and fmt.finite(m.get("oswald")):
        items.append(KPI("Oswald efficiency e", fmt.num(m["oswald"], 3),
                         sub=f"{_oswald_fit_text(m)}; CD0 = {fmt.num(m.get('cd0'), 3)}"))
    else:
        items.append(KPI("Span efficiency e", fmt.num(m.get("e_cruise"), 3),
                         sub=f"at CL = {fmt.num(m.get('cl_cruise'), 3)}"))
    items.append(KPI("Lift slope dCL/dα", fmt.num(m.get("cl_alpha_per_rad"), 3), "/rad",
                     sub=f"zero lift at α = {fmt.num(m.get('alpha_zero_lift'), 3)}°"))
    return KPIRow(items)


def findings(entries: Sequence[Entry]) -> list[str]:
    """Key-finding bullets for the 3D entries of the summary."""
    from . import content_physics as CP
    out: list[str] = []
    cfd = [e for e in entries if e.is_cfd3d]
    for e in cfd[:4]:
        r = e.result
        c = e.sim.conditions
        phys = CP.physics_finding(e)
        if not CP.has_forces(e):
            out.append(f"**{e.name}**: {phys}." if phys else f"**{e.name}**: no forces (no solid bodies).")
            continue
        out.append(f"**{e.name}**: {fmt.eq('CD', pm_text(r.cd))}, {fmt.eq('CL', pm_text(r.cl))}; drag {_si(r.drag, 'N')} "
                   f"on the whole body at {fmt.num(c.speed, 3)} m/s (drag area CD·A = {fmt.num(r.drag_area, 3)} m²), "
                   f"needing {_si(r.power, 'W')} to overcome" + (f"; {phys}." if phys else "."))
    cfd = [e for e in cfd if CP.has_forces(e)]          # drag rankings only among runs with forces
    if len(cfd) >= 2:
        base = cfd[0].result
        best = min(cfd, key=lambda e: e.result.cd.mean if fmt.finite(e.result.cd.mean) else math.inf)
        if best is not cfd[0] and fmt.finite(base.cd.mean) and base.cd.mean:
            d = (best.result.cd.mean - base.cd.mean) / abs(base.cd.mean) * 100
            se = math.hypot(best.result.cd.stderr, base.cd.stderr)
            if rough(best.result.cd) or rough(base.cd) or not fmt.finite(se):
                verdict = " (error bars too rough to say whether this exceeds the noise)."
            elif abs(best.result.cd.mean - base.cd.mean) > 2 * se:
                verdict = " (statistically significant)."
            else:
                verdict = " (within statistical noise)."
            out.append(f"Lowest drag (3D): **{best.name}**, {fmt.pct(d)} CD versus the baseline {cfd[0].name}"
                       + verdict)
        elif best is cfd[0]:
            out.append(f"The baseline **{cfd[0].name}** has the lowest drag of the compared 3D variants.")
    for e in [x for x in entries if x.is_wing][:4]:
        m = wing_metrics(e.result)
        if not m:
            continue
        parts = []
        if fmt.finite(m.get("ld_max")):
            parts.append(f"best L/D {fmt.fixed(m['ld_max'], 1)} at α = {fmt.angle(m['alpha_ld_max'])}")
        parts.append(f"CL,max {_cl_max_text(m)}")
        if fmt.finite(m.get("oswald")):
            parts.append(f"Oswald e = {fmt.num(m['oswald'], 3)} ({_oswald_fit_text(m)})")
        out.append(f"**{e.name}**: " + ", ".join(parts) + f" (AR {fmt.num(e.result.aspect_ratio, 3)}).")
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def geometry_blocks(d: Any, ctx: dict) -> list:
    """Geometry section content for a 3D design (imported mesh or wing planform)."""
    blocks: list = []
    coords = _airfoil_coords(ctx.get("project"))
    if getattr(d, "kind", "") == "wing" and getattr(d, "wing", None) is not None:
        w = d.wing
        try:
            from ..geometry3d import planform
            pf = planform(w)
        except Exception:  # noqa: BLE001 - metrics are a nicety; the drawing works without
            pf = {}
        blocks.append(fig(figure_wing_render, d, airfoil_coords=coords,
                          caption=f"{d.name}: the wing lofted from its sections (isometric view, no ground)."))
        blocks.append(fig(F.fig_wing_planform, w, metrics=pf, title="",
                          caption=f"{d.name}: planform with the quarter-chord line (dash-dot), the mean "
                          "aerodynamic chord (orange) and its quarter-chord point (+), the moment reference of "
                          "the coefficients. The free stream comes from the top of the page."))
        rows = [("Kind", "Finite wing (planform)"), ("Sections", f"{len(w.sections)}"
                                                     + (", mirrored about y = 0" if w.symmetric else "")),
                ("Span b", f"{fmt.num(pf.get('span', d.span_ref), 4)} m"),
                ("Planform area S", f"{fmt.num(pf.get('area', d.ref_area), 4)} m²"),
                ("Aspect ratio AR", fmt.num(pf.get("aspect_ratio"), 4)),
                ("Mean aerodynamic chord", f"{fmt.num(pf.get('mac', d.length_ref), 4)} m"),
                ("Taper ratio", fmt.num(pf.get("taper"), 3)),
                ("Quarter-chord sweep", fmt.angle(round(pf.get("sweep_c4_deg", float("nan")), 2))),
                ("Dihedral", fmt.angle(round(pf.get("dihedral_deg", float("nan")), 2))),
                ("Moment reference", _vec3(d.moment_ref3) + " (MAC quarter chord)")]
        blocks.append(KeyValues(rows, columns=2))
        blocks.append(_sections_table(d, ctx))
        return blocks
    meshes = list(getattr(d, "meshes", []) or [])
    if not meshes:
        blocks.append(Paragraph(f"{d.name} has no surface mesh stored.", tone="muted"))
        return blocks
    blocks.append(fig(figure_geometry3d, d, airfoil_coords=coords,
                      caption=f"{d.name}: isometric view; side, front, top and rear views at one common scale. "
                      + ("The road plane is hatched. " if _design_on_road(d) else "")
                      + "x points downstream, y to the left of the flow, z up."))
    bx = d.bounds3()
    src = getattr(d, "source", {}) or {}
    tri = sum(len(m.faces) for m in meshes)
    ver = sum(len(m.vertices) for m in meshes)
    kind = {"vehicle": "Vehicle (3D)", "generic": "Generic body (3D)"}.get(d.kind, d.kind)
    ref_a, ref_l = ("frontal area" if d.kind != "wing" else "planform area",
                    "vehicle length" if d.kind == "vehicle" else "reference length")
    rows = [("Kind", kind), ("Source", _source3d(src)),
            ("Parts", f"{len(meshes)}: " + ", ".join(m.name + (f" ({m.role})" if m.role != "body" else "")
                                                 for m in meshes[:6]) + (" …" if len(meshes) > 6 else "")),
            ("Surface mesh", f"{tri:,} triangles, {ver:,} vertices"),
            ("Overall size L × W × H", (f"{fmt.num(bx[1] - bx[0], 4)} × {fmt.num(bx[3] - bx[2], 4)} × "
                                      f"{fmt.num(bx[5] - bx[4], 4)} m")),
            (f"Reference area A ({ref_a})", f"{fmt.num(d.ref_area, 4)} m²"),
            (f"Reference length ({ref_l})", f"{fmt.num(d.length_ref, 4)} m"),
            ("Lateral reference (roll, yaw)", f"{fmt.num(d.span_ref, 4)} m"),
            ("Moment reference", _vec3(d.moment_ref3)),
            ("Mirror-symmetric", ("yes" if d.symmetric else "no") + _mirror_text(src.get("symmetry_error")))]
    if src.get("on_road") or d.kind == "vehicle":
        rows.append(("Ground clearance", f"{fmt.num(bx[4] * 1000, 3)} mm"))
    info = _mesh_quality(meshes)
    if info:
        rows.append(("Mesh quality", info))
    blocks.append(KeyValues(rows, columns=2))
    notes = src.get("import_notes")
    if notes:
        blocks.append(Bullets([fmt.solver_text(n) for n in (notes if isinstance(notes, list) else [notes])],
                              tone="muted"))
    if getattr(d, "notes", ""):
        blocks.append(Paragraph(d.notes, tone="muted"))
    return blocks


def _mirror_text(err: Any) -> str:
    """' (mirror error 0.4 mm)', or '< 0.01 mm' for an exactly symmetric mesh."""
    if not fmt.finite(err):
        return ""
    mm = float(err) * 1000.0
    return " (mirror error < 0.01 mm)" if mm < 0.01 else f" (mirror error {fmt.num(mm, 2)} mm)"


def _vec3(v: Sequence[float]) -> str:
    return "(" + ", ".join(fmt.num(c, 3) for c in v) + ") m"


def _source3d(src: dict) -> str:
    t = src.get("type", "")
    path = src.get("path") or src.get("file") or ""
    scale = src.get("scale_to_m")
    unit = {1.0: "m", 0.001: "mm", 0.01: "cm", 0.0254: "inch"}.get(scale, None) if fmt.finite(scale) else None
    text = "imported mesh" if t in ("import", "stl", "obj", "ply", "file") else (t or "unknown")
    if path:
        text += f" ({path})"
    if unit:
        text += f", units {unit}"
    elif fmt.finite(scale):
        text += f", scaled ×{fmt.num(scale, 4)} to metres"
    return text


def _mesh_quality(meshes: Sequence[Any]) -> str:
    """Watertightness and holes in words (imported meshes need not be closed: the solver copes)."""
    try:
        from ..geometry3d import mesh_info
        infos = [mesh_info(m) for m in meshes[:12]]
    except Exception:  # noqa: BLE001 - optional detail
        return ""
    open_ = [i for i in infos if not i["watertight"]]
    if not open_:
        return "closed (watertight)"
    holes = sum(i["holes"] for i in open_)
    return (f"{len(open_)} of {len(infos)} parts open ({holes} hole{'s' if holes != 1 else ''}); the "
            "voxeliser tolerates small gaps")


def _sections_table(d: Any, ctx: dict) -> Table:
    designs = getattr(ctx.get("project"), "designs", {}) or {}
    rows = []
    for s in sorted(d.wing.sections, key=lambda s: s.y):
        name = s.airfoil
        if name.startswith("design:"):
            ref = designs.get(name.split(":", 1)[1]) if hasattr(designs, "get") else None
            name = getattr(ref, "name", name)
        elif name.isdigit():
            name = f"NACA {name}"
        rows.append([fmt.fixed(s.y, 3), fmt.fixed(s.chord, 3), fmt.fixed(s.x_le, 3), fmt.fixed(s.z_le, 3),
                     fmt.angle(round(s.twist_deg, 3)), name])
    return Table(["y [m]", "Chord [m]", "x LE [m]", "z LE [m]", "Twist", "Airfoil"], rows,
                 align=["r", "r", "r", "r", "r", "l"], caption=f"{d.name}: wing sections (right half)",
                 widths=[1, 1.1, 1, 1, 0.9, 2.2])


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def _domain_text(r: Any) -> str | None:
    """Solved domain from the stored slices (the y cut spans x and z, the z cut spans y)."""
    sl = {s.axis: s for s in getattr(r, "slices", []) or []}
    if "y" not in sl or "z" not in sl:
        return None
    ey, ez = sl["y"].extent, sl["z"].extent
    lx, lz, ly = ey[1] - ey[0], ey[3] - ey[2], ez[3] - ez[2]
    return (f"{fmt.num(lx, 3)} × {fmt.num(ly, 3)} × {fmt.num(lz, 3)} m (x {fmt.num(ey[0], 3)} … "
            f"{fmt.num(ey[1], 3)} m)" + ("; half y ≥ 0 solved" if r.symmetry_plane else ""))


def setup_cfd3d(e: Entry) -> list:
    r = e.result if e.is_cfd3d else None
    c = e.sim.conditions
    fl = c.fluid
    cfd = e.sim.cfd
    d = e.design
    L = d.length_ref if d is not None else float("nan")
    ref_a, ref_l = _ref_name(e)
    yaw = float(getattr(c, "yaw_deg", 0.0) or 0.0)
    rows = [("Analysis", "3D incompressible Navier–Stokes (time-accurate)"), ("Fluid", fl.name),
            ("Density ρ", f"{fmt.num(fl.density, 4)} kg/m³"), ("Dynamic viscosity μ", f"{sci(fl.viscosity)} Pa·s"),
            ("Free-stream speed U∞", f"{fmt.num(c.speed, 4)} m/s ({fmt.num(c.speed * 3.6, 4)} km/h)"),
            ("Dynamic pressure q", f"{fmt.num(dyn_pressure(e), 4)} Pa"),
            ("Angle of attack / pitch α", fmt.angle(c.alpha_deg) + " (nose up +)"),
            ("Yaw (sideslip) β", fmt.angle(yaw) + (" (wind from the −y side)" if yaw else ""))]
    if d is not None:
        rows += [("Reference area A", f"{fmt.num(d.ref_area, 4)} m² ({ref_a})"),
                 ("Reference length", f"{fmt.num(d.length_ref, 4)} m ({ref_l}; Re and Cm)"),
                 ("Lateral reference", f"{fmt.num(d.span_ref, 4)} m (Croll and Cn)"),
                 ("Moment reference", _vec3(d.moment_ref3))]
    re_real = r.re_real if r is not None else c.reynolds(L)
    rows.append(("Reynolds number (real)", f"{fmt.reynolds(re_real)}  (on the {ref_l})"))
    if r is not None:
        capped = fmt.finite(r.re_sim) and r.re_sim < 0.95 * r.re_real
        rows.append(("Reynolds number (simulated)", fmt.reynolds(r.re_sim) + ("  (capped)" if capped else "")))
        nx, ny, nz = (tuple(r.grid) + (0, 0, 0))[:3]
        cells = r.cells or nx * ny * nz
        rows.append(("Grid", (f"{nx:,} × {ny:,} × {nz:,} = {cells:,} cells, {fmt.num(r.cells_per_ref, 3)} per "
                     "reference length (uniform)")))
        rows.append(("Symmetry plane", "yes: half domain y ≥ 0, mirrored for forces and pictures"
                     if r.symmetry_plane else "no: full domain"))
        dom = _domain_text(r)
        if dom:
            rows.append(("Domain", dom))
        tconv = L / c.speed if c.speed else float("nan")
        rows.append(("Simulated time", f"{fmt.num(r.t_end, 3)} convective units ({fmt.num(r.t_end * tconv, 3)} s)"))
        rows.append(("Averaging window", f"t·U/L = {fmt.num(r.t_avg_start, 3)} … {fmt.num(r.t_end, 3)}"))
        rows.append(("Turbulence treatment", turbulence_text(r.turbulence, cfd.turbulence)))
        rows.append(("Precision / device", f"{r.precision or cfd.precision} on {r.device or cfd.device}"))
    else:
        rows.append(("Turbulence treatment", turbulence_text("", cfd.turbulence)))
        rows.append(("Symmetry", "automatic (half domain when symmetric and β = 0)" if cfd.symmetry == "auto"
                     else "full domain"))
    rows.append(("Quality preset", str(cfd.quality)))
    g = e.sim.ground
    if g.enabled:
        rows.append(("Ground", ("moving belt at U∞" if g.moving else "fixed road")
                     + (", wheels rotating" if g.rotate_wheels else "")))
    elif e.kind == "vehicle":
        rows.append(("Ground", "none (free air)"))
    if getattr(cfd, "compute_sensitivity", False):
        rows.append(("Sensitivity objective", str(cfd.sensitivity_objective).upper()))
    blocks: list = [KeyValues(rows, columns=2)]
    notes = list(getattr(r, "notes", []) or []) if r is not None else []
    if notes:
        blocks.append(Bullets([fmt.solver_text(n) for n in notes], tone="muted"))
    return blocks


def setup_wing(e: Entry) -> list:
    r = e.result if e.is_wing else None
    c = e.sim.conditions
    ws = e.sim.wing
    d = e.design
    spec = getattr(d, "wing", None)
    method = ("Vortex lattice: ring vortices on the cambered mean surface, induced drag in the Trefftz plane"
              if ws.method == "vlm" else
              "Nonlinear lifting line (Phillips & Snyder) with the section lift curves of the polars")
    rows = [("Analysis", "Finite-wing sweep (potential flow + viscous strips)" if ws.viscous
             else "Finite-wing sweep (potential flow, inviscid)"),
            ("Method", method),
            ("Profile drag", f"strip theory from 2D viscous polars, e^N transition (Ncrit = {fmt.num(ws.ncrit, 3)})"
             if ws.viscous else "none (inviscid: CD = induced drag only)"),
            ("Fluid", fluid_name(c.fluid.name)), ("Free-stream speed U∞", f"{fmt.num(c.speed, 4)} m/s"),
            ("Dynamic pressure q", f"{fmt.num(dyn_pressure(e), 4)} Pa"),
            ("Angle range", f"{fmt.angle(ws.alpha_start)} … {fmt.angle(ws.alpha_end)}, step {fmt.angle(ws.alpha_step)}"),
            ("Sideslip β", fmt.angle(float(getattr(c, "yaw_deg", 0.0) or 0.0))),
            ("Mach number", fmt.num(ws.mach, 3) + (" (Prandtl–Glauert)" if ws.mach > 0 else ""))]
    if r is not None:
        rows.append(("Reynolds number (MAC)", fmt.reynolds(r.re_mac)))
    if spec is not None:
        if ws.method == "vlm":
            rows.append(("Lattice", (f"{spec.n_chord} chordwise × {spec.n_span} spanwise panels per half, "
                         f"{spec.spacing} spacing")))
        else:
            rows.append(("Lifting line", f"{spec.n_span} horseshoe vortices per half, {spec.spacing} spacing"))
    if d is not None:
        rows += [("Reference area S", f"{fmt.num(d.ref_area, 4)} m² (planform)"),
                 ("Reference chord", f"{fmt.num(d.length_ref, 4)} m (mean aerodynamic chord)"),
                 ("Reference span b", f"{fmt.num(d.span_ref, 4)} m"),
                 ("Moment reference", _vec3(d.moment_ref3))]
    blocks: list = [KeyValues(rows, columns=2)]
    notes = list(getattr(r, "notes", []) or []) if r is not None else []
    if notes:
        blocks.append(Bullets([fmt.solver_text(n) for n in notes], tone="muted"))
    return blocks


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def results_cfd3d(e: Entry) -> list:
    r = e.result
    rows = []
    for key in KEYS3:
        s = getattr(r, key)
        if r.symmetry_plane and key in ANTISYMMETRIC and not (s.mean or s.rms):
            rows.append([Cell(NAMES3[key], bold=True), "0 (symmetry)", fmt.DASH, fmt.DASH, fmt.DASH, fmt.DASH,
                         fmt.DASH])
            continue
        rel = rel_err(s)
        rel_txt = fmt.pct(rel * 100, 2, signed=False) if fmt.finite(rel) else fmt.DASH
        rows.append([Cell(NAMES3[key], bold=True), fmt.mean_only(s.mean, s.stderr), err_text(s),
                     (ROUGH + rel_txt) if rough(s) and rel_txt != fmt.DASH else rel_txt,
                     fmt.num(s.rms, 3), fmt.num(s.min, 4), fmt.num(s.max, 4)])
    blocks: list = [Table(["Coefficient", "Mean", "Std. error", "Rel. error", "RMS fluct.", "Min", "Max"], rows,
                          align=["l", "r", "r", "r", "r", "r", "r"],
                          caption="Time-averaged force and moment coefficients (wind axes, whole body)")]
    if any(rough(getattr(r, k)) for k in KEYS3):
        blocks.append(Paragraph(ROUGH_NOTE, tone="muted"))
    m = tuple(r.moments) if r.moments is not None else (float("nan"),) * 3
    d = e.design
    st_txt = (fmt.num(r.strouhal, 3) + " (dominant shedding frequency)") if r.strouhal else \
        "no periodic shedding measured"
    sym = r.symmetry_plane
    kv = [("Drag D", _si(r.drag, "N")),
          ("Side force Y", "0 (symmetry)" if sym and not r.side else _si(r.side, "N")),
          ("Lift L", _si(r.lift, "N") + ("  (negative = downforce)" if e.kind == "vehicle" else "")),
          ("Rolling moment", "0 (symmetry)" if sym and not m[0] else _si(m[0], "N·m")),
          ("Pitching moment", _si(m[1], "N·m") + " (nose-up +)"),
          ("Yawing moment", "0 (symmetry)" if sym and not m[2] else _si(m[2], "N·m")),
          ("Lift-to-drag L/D", _ld_text(r)),
          ("Drag area CD·A", f"{fmt.num(r.drag_area, 4)} m²"),
          ("Power to overcome drag", _si(r.power, "W")
           + (f" ({fmt.num(r.power / 1000, 3)} kW)" if fmt.finite(r.power) and abs(r.power) >= 1000 else "")),
          ("Strouhal number St", st_txt),
          ("References", f"q = {fmt.num(dyn_pressure(e), 4)} Pa, A = {fmt.num(d.ref_area, 4)} m², "
                         f"L = {fmt.num(d.length_ref, 4)} m" if d is not None else fmt.DASH)]
    blocks.append(KeyValues(kv, columns=2, title="Forces and moments on the whole body"))
    pb = getattr(r, "per_body", {}) or {}
    if len(pb) > 1:
        total = sum(v.get("cd", 0.0) for v in pb.values())
        prow = [[name, fmt.num(v.get("cd"), 4), fmt.num(v.get("cs"), 4), fmt.num(v.get("cl"), 4),
                 fmt.num(v.get("cm"), 4),
                 fmt.pct(100 * v.get("cd", float("nan")) / total, 1, signed=False) if total else fmt.DASH]
                for name, v in pb.items()]
        blocks.append(Table(["Part", "CD", "CS", "CL", "Cm", "Share of drag"], prow,
                            align=["l", "r", "r", "r", "r", "r"], caption="Contribution of each part"))
    return blocks


def _si(x: Any, unit: str) -> str:
    """A force, moment or power with its unit; very small or large values in powers of ten."""
    if not fmt.finite(x):
        return fmt.DASH
    v = float(x)
    if v != 0 and not (1e-3 <= abs(v) < 1e7):
        return f"{sci(v)} {unit}"
    return f"{fmt.num(v, 4)} {unit}"


def _ld_text(r: Any) -> str:
    """L/D, or "≈ 0" when the lift is zero up to round-off (a symmetric body)."""
    if abs(r.cl.mean) < fmt.ZERO_COEFF:
        return "≈ 0"
    ld = r.lift_to_drag
    return fmt.fixed(ld, 2) if fmt.finite(ld) and abs(ld) < 1e3 else fmt.num(ld, 3)


RESULTS_NOTE_3D = ("3D coefficients use q = ½ρU² and the design's reference area (frontal area for vehicles and "
                   "bluff bodies, planform area for wings); the pitching moment additionally uses the reference "
                   "length, roll and yaw the lateral reference, all about the moment reference point. Forces are "
                   "for the complete body. Two variants whose means differ by less than about twice their combined "
                   "standard error cannot be ranked at this simulation length.")


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------

def surface_cfd3d(e: Entry) -> list:
    r = e.result
    surfs = list(r.surface)
    ground = 0.0 if _on_road(e) else None
    wing = e.kind == "wing"
    views = ("iso", "top", "front", "side") if wing else ("iso", "side", "front", "top", "back")
    blocks: list = [fig(figure_surface3d, surfs, "cp", ground=ground, views=views,
                        caption=f"{e.name}: time-averaged surface pressure coefficient. Red: over-pressure "
                        "(stagnation); blue: suction. The side, front, top and rear views share one scale; the "
                        "colour range is clipped at robust percentiles (triangles mark values beyond it).")]
    les = str(getattr(r, "turbulence", "")).startswith("les")
    cf_note = (" This run is a wall-unresolved LES: the wall shear is estimated from the resolved velocity 1.5 to "
               "2.5 cells off the wall with the molecular viscosity, which under-estimates a turbulent boundary "
               "layer's friction many times over. Read the pattern (attached against separated regions), not "
               "the values." if les else "")
    centre = 0.0 if (e.design is not None and e.design.symmetric) else None
    if centre is None:
        b = _surface_bounds(surfs)
        centre = 0.5 * (b[2] + b[3]) if b else 0.0
    if not wing and _centre_cuts(surfs, centre):
        blocks.append(fig(figure_centreline_cp, surfs, centre, kind="vehicle" if e.kind == "vehicle" else "body",
                          caption=f"{e.name}: Cp (suction up) and streamwise skin friction along the centre "
                          f"plane y = {fmt.num(centre, 3)} m, upper and lower contour of the body. Cf < 0 marks "
                          "reversed (separated) flow." + cf_note))
    if any(getattr(s, "cf", None) is not None for s in surfs):
        blocks.append(fig(figure_surface3d, surfs, "cf", ground=ground, views=("iso", "top"),
                          caption=f"{e.name}: skin-friction coefficient magnitude. Low values on a lee side mark "
                          "separated flow." + cf_note))
    return blocks


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

def fields_cfd3d(e: Entry) -> list:
    r = e.result
    blocks: list = []
    q = dyn_pressure(e)
    g = _on_road(e)
    sl = {s.axis: i for i, s in enumerate(r.slices or [])}
    # a closed box (a heated cavity, a fan in a room) has no body: no "upstream", no wake
    body = bool(getattr(r, "surface", None))
    if "y" in sl:
        pos = r.slices[sl["y"]].position
        blocks.append(fig(figure_slice3d, r, sl["y"], "speed", streamlines=True, ground=g, q_inf=q,
                          caption=f"{e.name}: mean velocity magnitude in the vertical centre plane (y = "
                          f"{fmt.num(pos, 3)} m) with streamlines of the in-plane mean flow"
                          + ("; the body is drawn with its exact section." if body else ".")))
        blocks.append(fig(figure_slice3d, r, sl["y"], "cp", ground=g, q_inf=q,
                          caption="Mean pressure coefficient in the same plane. Red: over-pressure; blue: suction."))
    if "z" in sl:
        pos = r.slices[sl["z"]].position
        blocks.append(fig(figure_slice3d, r, sl["z"], "speed", streamlines=True, q_inf=q,
                          caption=f"{e.name}: mean velocity magnitude in the horizontal plane z = {fmt.num(pos, 3)} m "
                          + ("(flow left to right, seen from above)." if body else "(seen from above).")))
    if "x" in sl:
        pos = r.slices[sl["x"]].position
        blocks.append(fig(figure_slice3d, r, sl["x"], "ux", streamlines=True, ground=g, q_inf=q, size=(6.7, 3.6),
                          caption=(f"{e.name}: streamwise velocity in the cross-flow plane x = {fmt.num(pos, 3)} m "
                                   "behind the body, seen from behind (looking upstream), with streamlines of the "
                                   "in-plane (v, w) flow: the trailing vortices show as spirals, the velocity deficit "
                                   "as the wake.") if body else
                          (f"{e.name}: velocity u through the plane x = {fmt.num(pos, 3)} m, seen from +x, with "
                           "streamlines of the in-plane (v, w) flow.")))
    vol = getattr(r, "volume", None)
    if vol is not None and all(k in (vol.fields or {}) for k in ("u", "v", "w")):
        ground = 0.0 if g else None
        blocks.append(fig(figure_flow3d, list(r.surface), vol, "streamlines", ground=ground,
                          caption=f"{e.name}: streamlines of the mean flow released from a rake upstream of the body, "
                          "coloured by speed." if body else
                          f"{e.name}: streamlines of the mean flow traced from a line of seed points across the "
                          "domain, coloured by speed."))
        if "q" in vol.fields:
            blocks.append(fig(figure_flow3d, list(r.surface), vol, "q", ground=ground,
                              caption=f"{e.name}: vortex structures and shear layers of the mean flow, an "
                              "iso-surface of the Q-criterion (Q at 10 % of its 99.9th percentile; the thin layer of "
                              "large Q that every immersed wall carries is trimmed) coloured by speed. Mean-flow "
                              "structures only: instantaneous turbulent eddies average out."))
    smoke = r.find_volume("dye") if hasattr(r, "find_volume") else None
    has_emitters = any(getattr(em, "enabled", True) for em in getattr(e.sim, "emitters", None) or [])
    if smoke is not None and not has_emitters:
        # with emitters the smoke belongs to the flow-visualisation section, with its legend
        from .content_physics import figure_smoke3d
        blocks.append(fig(figure_smoke3d, list(r.surface), smoke, ground=0.0 if g else None,
                          caption=f"{e.name}: smoke released by the rake upstream of the body at the end of the run "
                          "(volume rendering of the instantaneous dye concentration): a snapshot, not an average."))
    return blocks


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

def convergence_cfd3d(e: Entry) -> list:
    r = e.result
    t = np.asarray(r.history.get("t", []), float)
    n_win = int(np.sum(t >= r.t_avg_start)) if t.size and fmt.finite(r.t_avg_start) else 0
    lateral_zero = bool(r.symmetry_plane) and all(
        not np.any(np.abs(np.asarray(r.history.get(k, []), float)) > 0) for k in ANTISYMMETRIC)
    what = ("drag, lift and pitching-moment coefficient histories" if lateral_zero else
            "force (left) and moment (right) coefficient histories")
    blocks: list = [fig(F.fig_history3d, r.history, r.t_avg_start, r, symmetry_plane=bool(r.symmetry_plane),
                        caption=f"{e.name}: {what}. The shaded region is the averaging window; the black line is "
                        "the mean and its grey band ±1 standard error.")]
    rows = [("Simulated time", f"{fmt.num(r.t_end, 3)} convective units"),
            ("Transient discarded", f"t·U/L < {fmt.num(r.t_avg_start, 3)}"),
            ("Averaging window", f"{fmt.num(r.t_end - r.t_avg_start, 3)} convective units, {n_win:,} samples"),
            ("Converged", fmt.yes_no(r.converged))]
    for key in ("cd", "cl", "cm"):
        rel = rel_err(getattr(r, key))
        rows.append((f"Relative s.e. of {NAMES3[key].split()[-1]}",
                     (ROUGH if rough(getattr(r, key)) else "") + fmt.pct(rel * 100, 2, signed=False)))
    if r.strouhal:
        st_len = getattr(r, "strouhal_length", float("nan"))
        ratio = 1.0
        if e.design is not None and fmt.finite(st_len) and st_len > 0:
            ratio = float(e.design.length_ref) / float(st_len)
        cycles = r.strouhal * (r.t_end - r.t_avg_start) * ratio
        rows.append(("Shedding cycles averaged", f"≈ {cycles:.0f} (St = {fmt.num(r.strouhal, 3)})"))
    blocks.append(KeyValues(rows, columns=2))
    return blocks


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------

def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals turned to point away from the body's centroid."""
    t = v[f]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    c = t.mean(axis=1)
    centre = v.mean(axis=0)
    flip = np.einsum("ij,ij->i", fn, c - centre) < 0
    fn[flip] *= -1.0
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)


def hotspots3d(surfaces: Sequence[Any], objective: str, n: int = 3) -> list[str]:
    """Up to ``n`` well-separated surface places where the sensitivity is strongest, in words."""
    cand = []
    allv = np.concatenate([np.abs(np.asarray(s.sensitivity, float)) for s in surfaces])
    allv = allv[np.isfinite(allv)]
    ref98 = float(np.percentile(allv, 98)) if allv.size else 0.0
    b = _surface_bounds(surfaces)
    if b is None:
        return []
    size = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-9)
    for s in surfaces:
        v = np.asarray(s.vertices, float)
        sens = np.nan_to_num(np.asarray(s.sensitivity, float))
        nrm = _vertex_normals(v, np.asarray(s.faces, np.int64))
        for i in np.argsort(-np.abs(sens))[:400]:
            cand.append((abs(sens[i]), sens[i], v[i], nrm[i], str(s.body)))
    cand.sort(key=lambda c: -c[0])
    picked: list = []
    for c in cand:
        if len(picked) >= n or c[0] == 0:
            break
        # distance in (x, |y|, z): the mirror image of a picked place on a symmetric body is the
        # same advice, not a second place
        q = np.array([c[2][0], abs(c[2][1]), c[2][2]])
        if all(np.linalg.norm(q - np.array([p[2][0], abs(p[2][1]), p[2][2]])) > 0.18 * size for p in picked):
            picked.append(c)
    sym, sense = F.OBJECTIVES.get(objective, (objective.upper(), "reduce"))
    sym = {"cd": "CD", "cl": "CL", "ld": "L/D"}.get(objective, sym)
    out = []
    for mag, val, p, nv, body in picked:
        where = ("front face" if nv[0] < -0.6 else "rear face" if nv[0] > 0.6 else
                 "upper surface" if nv[2] > 0.6 else "underside" if nv[2] < -0.6 else "side")
        grow_helps = (val < 0) == (sense == "reduce")
        move = "outward (add material)" if grow_helps else "inward (remove material)"
        xr = (p[0] - b[0]) / max(b[1] - b[0], 1e-9)
        strength = f" (relative strength {mag / ref98:.2f})" if ref98 > 0 else ""
        out.append(f"{body}, {where} at x/L ≈ {xr:.2f}, z = {fmt.num(p[2], 3)} m: moving the surface {move} "
                   f"{sense}s {sym}{strength}.")
    return out


def sensitivity_cfd3d(e: Entry) -> list:
    r = e.result
    surfs = [s for s in r.surface if getattr(s, "sensitivity", None) is not None]
    obj = str(getattr(e.sim.cfd, "sensitivity_objective", "cd"))
    sym = {"cd": "CD", "cl": "CL", "ld": "L/D"}.get(obj, obj)
    _, sense = F.OBJECTIVES.get(obj, (obj, "reduce"))
    allv = np.concatenate([np.abs(np.asarray(s.sensitivity, float)) for s in surfs])
    allv = allv[np.isfinite(allv)]
    smax = float(np.percentile(allv, 98)) if allv.size else float("nan")
    ground = 0.0 if _on_road(e) else None
    blocks: list = [fig(figure_surface3d, list(r.surface), "sensitivity", ground=ground, relative=True,
                        label=f"∂{sym}/∂n, relative (1 = {smax:.3g} per m)",
                        caption=f"{e.name}: sensitivity of {sym} to a small outward displacement of the surface, "
                        f"relative to its 98th-percentile magnitude. Red: pushing the surface outward raises {sym}; "
                        f"blue: it lowers {sym}. To {sense} {sym}, move red areas inward and blue areas outward.")]
    hs = hotspots3d(surfs, obj)
    if hs:
        blocks.append(Paragraph("Most effective places to modify the shape:", tone="body"))
        blocks.append(Bullets(hs))
    return blocks


SENSITIVITY_NOTE_3D = (
    "The 3D sensitivity is the derivative of the time-averaged objective with respect to a normal displacement "
    "of each surface vertex, obtained by automatic differentiation through the 3D solver over a short averaging "
    "window and smoothed over about 1.5 cells. It is a local, linear guide for small changes; re-simulate any "
    "modified shape to confirm the gain. Read the pattern, not the size: on a sphere the map was within 6 % of "
    "the physical slope at Re 30 and 28 % above it at Re 100 (16 cells per diameter), and the same method can "
    "be up to about twice too large in 2D. Signs and the places that matter most are reliable; the map is "
    "therefore shown relative to its strongest values.")


# ---------------------------------------------------------------------------
# wing polars
# ---------------------------------------------------------------------------

def wing_blocks(e: Entry) -> list:
    r = e.result
    blocks: list = [fig(F.fig_wing_polar, r, labels=[e.name], size=(6.7, 4.9),
                        caption=f"{e.name}: lift curve, drag polar (dashed: induced drag CDi), pitching moment about "
                        f"the moment reference and lift-to-drag ratio at Re (MAC) = {fmt.reynolds(r.re_mac)}. The "
                        "ringed point is the best L/D.")]
    if any(p.y is not None and p.c_cl is not None for p in r.points):
        blocks.append(fig(F.fig_span_loading, r, size=(6.7, 3.4),
                          caption=f"{e.name}: span loading c·cl/c̄ at selected angles against the elliptic loading of "
                          "the same CL (dotted), and the section lift coefficient across the span: the station whose "
                          "cl approaches the airfoil's maximum stalls first."))
    m = wing_metrics(r)
    if m:
        na = fmt.DASH + " (inviscid)"
        visc = bool(r.viscous)
        blocks.append(KeyValues([
            ("Lift slope dCL/dα", f"{fmt.num(m.get('cl_alpha_per_rad'), 4)} /rad ({fmt.num(m.get('cl_alpha_per_deg'), 4)} /°)"),
            ("Zero-lift angle α0", f"{fmt.num(m.get('alpha_zero_lift'), 3)}°"),
            ("Maximum lift CL,max", f"{_cl_max_text(m)} at α = {fmt.angle(m.get('alpha_cl_max'))}"
             + (" (still rising: stall not reached)" if _stall_missed(m) else "")),
            ("Minimum drag CD,min", f"{fmt.num(m.get('cd_min'), 3)} at CL = {fmt.num(m.get('cl_at_cd_min'), 3)}"
             if visc else na),
            ("Best L/D", f"{fmt.fixed(m.get('ld_max'), 1)} at α = {fmt.angle(m.get('alpha_ld_max'))}" if visc else na),
            ("Zero-lift drag CD0 (parabolic fit)", fmt.num(m.get("cd0"), 3) if visc else na),
            ("Oswald efficiency e (parabolic fit)",
             f"{fmt.num(m.get('oswald'), 3)} over CL {fmt.fixed(m['oswald_cl_lo'], 2)}–{fmt.fixed(m['oswald_cl_hi'], 2)}"
             if fmt.finite(m.get("oswald")) and fmt.finite(m.get("oswald_cl_lo")) and fmt.finite(m.get("oswald_cl_hi"))
             else fmt.num(m.get("oswald"), 3)),
            ("Span efficiency at cruise", f"{fmt.num(m.get('e_cruise'), 3)} at CL = {fmt.num(m.get('cl_cruise'), 3)}"),
            ("Zero-lift moment Cm0", fmt.num(m.get("cm0"), 3)),
            ("Moment slope dCm/dα", f"{fmt.num(m.get('cm_alpha_per_deg'), 3)} /° "
             + ("(statically stable about the reference)" if fmt.finite(m.get("cm_alpha_per_deg"))
                and m["cm_alpha_per_deg"] < 0 else "")),
            ("Aspect ratio AR", fmt.num(r.aspect_ratio, 4)),
            ("Planform", f"S = {fmt.num(r.area, 4)} m², b = {fmt.num(r.span, 4)} m, MAC = {fmt.num(r.mac, 4)} m"),
        ], columns=2, title="Wing figures of merit"))
    rows = []
    for p in sorted(r.points, key=lambda p: p.alpha):
        ld = p.cl / p.cd if (fmt.finite(p.cd) and p.cd > 0 and fmt.finite(p.cl)) else float("nan")
        tone = "neutral" if p.converged else "warn"
        rows.append([Cell(fmt.angle(p.alpha), tone), fmt.fixed(p.cl, 4), fmt.fixed(p.cd, 5), fmt.fixed(p.cdi, 5),
                     fmt.fixed(p.cdv, 5), fmt.fixed(p.cm, 4), fmt.fixed(ld, 1) if r.viscous else fmt.DASH,
                     fmt.fixed(p.e, 3), Cell("yes" if p.converged else "no", tone)])
    blocks.append(Table(["α", "CL", "CD", "CDi", "CDv", "Cm", "L/D", "e", "Conv."], rows,
                        align=["r"] * 8 + ["c"],
                        caption=f"{e.name}: operating points (CDi induced, CDv profile drag; e span efficiency "
                        "CL²/(π AR CDi))", font_scale=0.92))
    for w in getattr(r, "warnings", []) or []:
        blocks.append(Callout(fmt.solver_text(w), "warn"))
    return blocks


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare_cfd3d(cfd_all: Sequence[Entry]) -> list:
    """The 3D CFD comparison: forces over the runs that have them, then heat transfer and fans."""
    from . import content_physics as CP
    blocks: list = []
    cfd = [e for e in cfd_all if CP.has_forces(e)]
    res = {e.sim_id: e.result for e in cfd}
    by_key = {row["key"]: row for row in ((compare_rows(res) or []) if len(cfd) >= 2 else [])}
    blocks.append(Paragraph(
        f"Baseline: **{(cfd[0] if len(cfd) >= 2 else cfd_all[0]).name}**. Δ is the change relative to the baseline. "
        "Green/red mark an improvement/deterioration larger than twice the combined standard error (✱, "
        "statistically significant); ~ marks a difference within the statistical noise."))
    kinds = {e.kind for e in cfd}
    if len(kinds) > 1:
        blocks.append(Callout("This comparison mixes design kinds (" + ", ".join(sorted(kinds)) + "). Their "
                              "coefficients use different reference areas, so compare forces rather than "
                              "coefficients across kinds.", "warn"))
    for msg in comparison_warnings({e.sim_id: e.result for e in cfd_all},
                                   {e.sim_id: e.sim for e in cfd_all}):
        blocks.append(Callout(msg, "warn"))
    if len(cfd) < 2:
        return blocks + CP.physics_comparison(cfd_all, prefix="3D ")
    rows = []
    for i, e in enumerate(cfd):
        r = e.result

        def cell(key: str, i: int = i) -> dict | None:
            row = by_key.get(key)
            if not row:
                return None
            vals = [abs(c["value"]) for c in row["cells"] if c.get("value") is not None]
            b = row.get("baseline")
            if b is not None and vals and abs(b) <= 1e-3 * max(vals) and not row["cells"][i].get("is_baseline"):
                return None             # zero up to round-off (a symmetric body's lift): no percentage
            return row["cells"][i]
        c = e.sim.conditions
        rows.append([Cell(e.name, bold=(i == 0)), fmt.num(c.speed, 3),
                     f"{fmt.angle(c.alpha_deg)} / {fmt.angle(float(getattr(c, 'yaw_deg', 0.0) or 0.0))}",
                     pm_text(r.cd), delta_cell(cell("cd")), pm_text(r.cl), delta_cell(cell("cl")),
                     _si(r.drag, "").strip(), delta_cell(cell("drag")), fmt.num(r.drag_area, 3)])
    blocks.append(Table(["Simulation", "U [m/s]", "α / β", "CD", "ΔCD", "CL", "ΔCL", "D [N]", "ΔD", "CD·A [m²]"],
                        rows, align=["l"] + ["r"] * 9,
                        caption="3D CFD comparison (± one standard error; Δ relative to the baseline)",
                        widths=[2.6, 0.9, 1.1, 1.5, 1.1, 1.5, 1.1, 1.0, 1.0, 1.1], font_scale=0.84))
    if any(rough(e.result.cd) or rough(e.result.cl) for e in cfd):
        blocks.append(Paragraph(ROUGH_NOTE + " Differences against such a value are not marked significant.",
                                tone="muted"))
    bar_rows = [{"label": e.name, "cd": e.result.cd, "cl": e.result.cl} for e in cfd]
    blocks.append(fig(F.fig_compare_bars, bar_rows, ["cd", "cl"],
                      metric_labels={"cd": "Drag coefficient CD", "cl": "Lift coefficient CL"},
                      caption="3D drag and lift coefficients with ±1 standard-error bars; percentages are relative "
                      "to the baseline (grey)."))
    return blocks + CP.physics_comparison(cfd_all, prefix="3D ")


def compare_wings(wings: Sequence[Entry]) -> list:
    blocks: list = [fig(F.fig_wing_polar, [e.result for e in wings], labels=[e.name for e in wings], size=(6.7, 5.4),
                        caption="Wing polars overlaid. Rings mark each sweep's best L/D.")]
    ms = [wing_metrics(e.result) for e in wings]
    base = ms[0]

    def dcell(m: dict, key: str, better: str | None, i: int, digits: int = 3) -> Cell:
        v, b = m.get(key), base.get(key)
        if key == "cl_max" and (_stall_missed(m) or _stall_missed(base)):
            return Cell(_cl_max_text(m, digits))
        if i == 0 or better is None or not (fmt.finite(v) and fmt.finite(b) and b):
            return Cell(fmt.num(v, digits))
        d = (v - b) / abs(b) * 100
        if abs(d) < 0.05:
            return Cell(f"{fmt.num(v, digits)} (0.0 %)")
        good = d < 0 if better == "lower" else d > 0
        return Cell(f"{fmt.num(v, digits)} ({fmt.pct(d)})", "ok" if good else "bad")
    rows = []
    for i, (e, m) in enumerate(zip(wings, ms, strict=True)):
        rows.append([Cell(e.name, bold=(i == 0)), fmt.num(e.result.aspect_ratio, 3), fmt.reynolds(e.result.re_mac),
                     dcell(m, "cl_max", "higher", i), dcell(m, "ld_max", "higher", i, 4),
                     dcell(m, "cd0", "lower", i), dcell(m, "oswald", "higher", i),
                     fmt.num(m.get("cl_alpha_per_rad"), 3), fmt.num(m.get("cm0"), 3)])
    blocks.append(Table(["Wing sweep", "AR", "Re (MAC)", "CL,max", "(L/D)max", "CD0", "Oswald e", "CLα [/rad]", "Cm0"],
                        rows, align=["l"] + ["r"] * 8, caption="Wing comparison (Δ relative to the first sweep)",
                        widths=[2.4, 0.7, 1.4, 1.5, 1.6, 1.5, 1.5, 1.1, 0.9], font_scale=0.86))
    if len({(e.result.method, bool(e.result.viscous)) for e in wings}) > 1:
        blocks.append(Callout("These sweeps use different methods (vortex lattice / lifting line, viscous / "
                              "inviscid). Part of the difference is the method, not the wing.", "warn"))
    return blocks


# ---------------------------------------------------------------------------
# methodology
# ---------------------------------------------------------------------------

REFERENCES_3D = [
    ("Angot, P., Bruneau, C.-H. & Fabrie, P. (1999). A penalization method to take into account obstacles in "
     "incompressible viscous flows. Numerische Mathematik 81, 497–520."),
    ("Nicoud, F. & Ducros, F. (1999). Subgrid-scale stress modelling based on the square of the velocity "
     "gradient tensor. Flow, Turbulence and Combustion 62, 183–200."),
    ("Johnson, T. A. & Patel, V. C. (1999). Flow past a sphere up to a Reynolds number of 300. Journal of Fluid "
     "Mechanics 378, 19–70."),
    ("Ahmed, S. R., Ramm, G. & Faltin, G. (1984). Some salient features of the time-averaged ground vehicle wake. "
     "SAE Technical Paper 840300."),
    ("Jeong, J. & Hussain, F. (1995). On the identification of a vortex. Journal of Fluid Mechanics 285, 69–94."),
]
REFERENCES_WING = [
    "Katz, J. & Plotkin, A. (2001). Low-Speed Aerodynamics, 2nd ed. Cambridge University Press.",
    ("Phillips, W. F. & Snyder, D. O. (2000). Modern adaptation of Prandtl's classic lifting-line theory. "
     "Journal of Aircraft 37(4), 662–670."),
    ("Neely, R. H., Bollech, T. V., Westrick, G. C. & Graham, R. R. (1947). NACA TN 1270: wind-tunnel and "
     "calculated characteristics of tapered NACA 44-series wings of aspect ratio 8 to 12."),
]


def methodology_cfd3d(cfd3: Sequence[Entry], h: Any, blocks: list) -> None:
    """Append the method and validity text for 3D CFD entries to ``blocks``.

    ``h(title)`` appends a numbered subheading to the same list, so text and headings stay
    in order.
    """
    if cfd3:
        h("3D Navier–Stokes solver")
        blocks.append(Paragraph(
            "The 3D solver integrates the incompressible, time-dependent Navier–Stokes equations on a uniform "
            "Cartesian grid (staggered velocities, third-order Runge–Kutta time stepping with a pressure projection "
            "at every stage, an exact FFT/DCT pressure solve), in PyTorch on a GPU or a CPU. Bodies are imported "
            "triangle meshes, which need not be closed: they are voxelised robustly and imposed as immersed walls "
            "by Brinkman penalisation (Angot et al., 1999), so no body-fitted mesh is generated. A symmetric "
            "body in straight-on flow is simulated in half a domain with a symmetry plane at y = 0; forces are "
            "reported for the whole body. Vehicles run over a road that moves with the free stream, like a "
            "rolling-road wind tunnel. Forces and moments are integrated every step and averaged over the window "
            "after the start-up transient, with batch-means standard errors as in 2D."))
        les = sorted({str(e.result.turbulence) for e in cfd3 if str(e.result.turbulence).startswith("les")})
        if les:
            models = ", ".join(turbulence_text(x).split(", ", 1)[-1] for x in les)
            blocks.append(Paragraph(
                f"Turbulence: runs whose boundary layers are thinner than a cell are large-eddy simulations ({models}): "
                "the large eddies are resolved on the grid and a sub-grid model represents the rest. There is no "
                "wall model and no transition model: the near-wall turbulence on the immersed walls is not "
                "resolved."))
        h("Validity and limits (3D)")
        items = [
            ("**Validated range:** spheres in laminar flow, Re 20–300 (Johnson & Patel, 1999): Re 100 drag within the "
            "1.05–1.15 band at 24 cells per diameter, the half-domain run equal to the full domain within 1 %. "
            "Runs at higher Reynolds numbers are outside the validated range."),
            (f"**Vehicles and high-Re bodies:** on an Ahmed-type body (25° slant, moving road, Re 2.9 × 10⁶) the "
            f"solver gave CD 0.64 at Draft and 0.61 at Standard quality where the wind tunnel measures about 0.29 "
            f"(Ahmed et al., 1984): absolute drag is over-predicted by a factor of about {AHMED_FACTOR}. The likely "
            "cause is the wall-unresolved LES on immersed walls with no wall model, but that is not "
            "yet verified, and differences between designs have not been validated either. Treat absolute CD, CL "
            "and forces of such runs as indicative, and compare designs only when simulated at the same quality "
            "and conditions."),
            ("**Uniform grid:** the exact pressure solver needs equal cells everywhere, so the far field costs as "
            "much as the region next to the body and the grid around a body is coarser than a body-fitted CFD "
            "mesh would be (Draft 1–2 M cells, Standard 4–6 M, Fine 10–14 M). Thin parts (spoilers, mirrors, "
            "trailing edges) thinner than about two cells are not resolved."),
            ("**Skin friction:** Cf is estimated from the resolved velocity 1.5 to 2.5 cells off the immersed "
            "wall with the molecular viscosity. That is right for laminar boundary layers spanning several cells; "
            "in a wall-unresolved LES it under-estimates the friction many times over, so Cf there is a "
            "qualitative map (attached against separated flow), not a friction-drag prediction."),
            ("**Symmetry plane:** a half-domain run suppresses wake motions that break the symmetry (a sideways "
            "swinging wake). Mean drag and lift were unaffected in the cases measured; side force, roll and yaw "
            "are zero by construction."),
            "**Compressibility:** incompressible flow, valid below about Mach 0.3 (≈ 100 m/s in sea-level air).",
        ]
        blocks.append(Bullets(items))


def methodology_wing(wings: Sequence[Entry], h: Any, blocks: list) -> None:
    """Append the method and validity text for finite-wing sweeps to ``blocks`` (see methodology_cfd3d)."""
    if wings:
        h("Finite-wing analysis")
        blocks.append(Paragraph(
            "Wings are analysed with potential-flow methods on the planform built from the section table. The "
            "vortex lattice places ring vortices on the cambered mean surface (Katz & Plotkin, 2001) with the wake "
            "along the free stream; the nonlinear lifting line (Phillips & Snyder, 2000) places horseshoe vortices "
            "on the quarter-chord line and uses the section lift curves of the viscous polars. Induced drag is "
            "computed in the Trefftz plane. In a viscous sweep each spanwise strip adds the profile drag of its "
            "airfoil at its local Reynolds number and effective angle, from the same XFOIL-class 2D solver as the "
            "airfoil polars (strip theory); compressibility is a Prandtl–Glauert correction below Mach 0.6."))
        h("Validity and limits (wings)")
        blocks.append(Bullets([
            ("**Assumptions:** thin lifting surfaces in attached flow, small angles, a flat wake along the free "
            "stream, no spanwise boundary-layer flow; strips are independent 2D sections. Fuselages, nacelles "
            "and ground effect are not modelled."),
            ("**Validated:** the lifting line reproduces 2πAR/(AR+2) for an elliptic wing to 0.05 % with e = 1.000; "
            "the lattice matches lifting-surface references (Multhopp, Falkner) within 2 % and a 45° swept wind-tunnel "
            "wing within 4 %."),
            ("**Against wind-tunnel wings (NACA TN 1270):** lift slope 2–7 % high, zero-lift angle within 0.5°, "
            "minimum drag 13–22 % low and maximum L/D 6–18 % high (free transition, Ncrit 9). CL,max is **not** "
            "validated: the stall-onset estimate is 0.17–0.31 too high because the 2D polars stall late."),
            ("**Lifting line limits:** results are not reliable for quarter-chord sweep above about 20° or aspect "
            "ratio below about 4 (the solver warns); use the vortex lattice there."),
        ]))
