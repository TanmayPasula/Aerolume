"""CSV exports: result summary, force history, surface distributions, polar; and for 3D
(DESIGN.md 7.10) slices, wing polars and wing span loading.

Plain comma-separated files with one header row and no comment lines, so Excel,
LibreOffice, pandas and MATLAB all open them without options. Numbers are written
with ``repr`` (the shortest text that reads back to the identical float), so a
CSV round trip is lossless; missing values are written as ``nan``.
"""
from __future__ import annotations

import csv
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from ..model import CFD3DResult, CFDResult, PolarResult, SliceData, SurfaceData, WingResult
from ..project.fileio import atomic_writer


def _num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return repr(v) if math.isfinite(v) else ("nan" if math.isnan(v) else ("inf" if v > 0 else "-inf"))
    return str(v)


def _write_rows(path: str | os.PathLike, header: Sequence[str], rows: Iterable[Sequence]) -> Path:
    path = Path(path)
    with atomic_writer(path, "w") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow([_num(v) for v in row])
    return path


def _cfd_rows(r: CFDResult) -> list[tuple]:
    nan = float("nan")
    rows: list[tuple] = []
    for key, label in (("cd", "Drag coefficient"), ("cl", "Lift coefficient"), ("cm", "Moment coefficient")):
        s = getattr(r, key)
        rows += [(key, s.mean, s.stderr, "", f"{label}, time mean (stderr = standard error of the mean)"),
                 (f"{key}_rms", s.rms, nan, "", f"{label}, rms of fluctuation"),
                 (f"{key}_min", s.min, nan, "", f"{label}, minimum"),
                 (f"{key}_max", s.max, nan, "", f"{label}, maximum")]
    rows += [
        ("lift_to_drag", r.lift_to_drag, nan, "", "Lift-to-drag ratio"),
        ("drag", r.drag, nan, "N/m", "Mean drag per unit span"),
        ("lift", r.lift, nan, "N/m", "Mean lift per unit span (negative = downforce)"),
        ("moment", r.moment, nan, "N m/m", "Mean pitching moment per unit span, nose up positive"),
        ("drag_area", r.drag_area, nan, "m^2/m", "Drag area Cd x reference height"),
        ("power", r.power, nan, "W/m", "Power to overcome mean drag"),
        ("strouhal", r.strouhal if r.strouhal is not None else nan, nan, "", "Strouhal number of lift"),
        ("re_real", r.re_real, nan, "", "Reynolds number of the real flow"),
        ("re_sim", r.re_sim, nan, "", "Reynolds number simulated"),
        ("grid_nx", r.grid[0] if len(r.grid) > 0 else 0, nan, "cells", "Grid cells in x"),
        ("grid_ny", r.grid[1] if len(r.grid) > 1 else 0, nan, "cells", "Grid cells in y"),
        ("cells_per_ref", r.cells_per_ref, nan, "", "Cells per reference length"),
        ("t_end", r.t_end, nan, "L/U", "Simulated time in convective units"),
        ("t_avg_start", r.t_avg_start, nan, "L/U", "Start of averaging in convective units"),
        ("converged", r.converged, nan, "", "Statistically converged"),
        ("wall_seconds", r.wall_seconds, nan, "s", "Wall-clock time"),
        ("device", r.device, nan, "", "Compute device"),
        ("precision", r.precision, nan, "", "Floating-point precision"),
        ("turbulence", r.turbulence, nan, "", "Turbulence treatment"),
    ]
    for body, vals in r.per_body.items():
        for k, v in vals.items():
            rows.append((f"body:{body}:{k}", v, nan, "", f"{k} of body '{body}'"))
    return rows + _physics_rows(r, "W/m", "N/m", "m^2/s")


def _physics_rows(r: CFDResult | CFD3DResult, unit_w: str, unit_n: str, unit_q: str) -> list[tuple]:
    """Heat transfer, fans and solver diagnostics (DESIGN.md section 10), when the run has them.

    Read through :mod:`aerolume.analysis.physics`, the reader the report and the Compare
    page use, so the CSV quotes the same numbers and standard errors.
    """
    from ..analysis import physics as P  # numpy-only, cheap
    nan = float("nan")
    rows: list[tuple] = []
    nu = P.nusselt_stat(r)
    if nu is not None and math.isfinite(nu.mean):
        rows += [("nusselt", nu.mean, nu.stderr, "",
                  ("Mean Nusselt number of the heated bodies, or of the hottest wall of a closed box without "
                   "heated bodies (stderr = standard error of the mean)")),
                 ("nusselt_rms", nu.rms, nan, "", "Nusselt number, rms of fluctuation")]
    q, q_se, _ = P.heat_rate_stat(r)
    if math.isfinite(q) and P.heat_rate_is_input(r):
        # a room heated only by heaters: the "heat rate" is the power set, so it is not given an error bar
        rows.append(("heat_rate", q, nan, unit_w, ("Heat put into the fluid by the heaters; equals the heater power "
                                                   "set (an input, not a measured result)")))
    elif math.isfinite(q):
        rows.append(("heat_rate", q, q_se, unit_w, ("Heat-transfer rate into the fluid: from the heated bodies, "
                                                    "else through the hottest wall of a closed box, else from the "
                                                    "heat inputs")))
    q_set = P.heater_power_set(r)
    if math.isfinite(q_set):
        rows.append(("heater_power_set", q_set, nan, unit_w, "Total power of the enabled heaters, as set (an input)"))
    temps = P.fluid_temperatures(r)
    for key, desc, unit in (("t_fluid_max", "Highest fluid temperature", "degC"),
                            ("t_fluid_mean", "Mean fluid temperature (fluid cells only)", "degC"),
                            ("t_fluid_min", "Lowest fluid temperature", "degC"),
                            ("t_fluid_dev_max", "Largest departure of the fluid from ambient", "K")):
        if key in temps:
            rows.append((key, temps[key], nan, unit, desc))
    vent = P.vent_heat_out(r)
    if math.isfinite(vent):
        rows.append(("vent_heat_out", vent, nan, unit_w, "Heat carried out through the vents minus what the jets bring in"))
    bal = P.heat_balance(r)
    if bal is not None:
        for label, v in {**bal["in"], **{f"{k} (out)": v for k, v in bal["out"].items()}}.items():
            rows.append((f"heat_balance:{label}", v, nan, unit_w, "Heat-balance term (in: added to the fluid)"))
        rows += [("heat_balance:storage", bal["storage"], nan, unit_w, "Rate of change of heat stored in the domain"),
                 ("heat_balance:residual", bal["residual"], nan, unit_w,
                  "In - out - storage" + ("" if bal["reported"] else " (computed from the reported terms)")),
                 ("heat_balance:relative_residual", bal["relative"], nan, "",
                  "Residual / heat through the domain (largest of gross in, gross out, heat rate)")]
    for i, f in enumerate(P.fan_entries(r)):
        name = f["name"]
        rows += [(f"fan:{i}:thrust", f["thrust"], f["thrust_stderr"], unit_n, f"Thrust delivered by '{name}'"),
                 (f"fan:{i}:volume_flow", f["volume_flow"], nan, unit_q, f"Volume flow through '{name}'"),
                 (f"fan:{i}:induced_velocity", f["induced_velocity"], nan, "m/s",
                  f"Mean axial velocity through '{name}'"),
                 (f"fan:{i}:ideal_power", f["ideal_power"], nan, unit_w, f"Ideal (momentum-theory) power of '{name}'"),
                 (f"fan:{i}:pressure_jump", f["pressure_jump"], nan, "Pa", f"Pressure jump across '{name}'")]
    for key, label, v, unit in P.diagnostics_rows(r):
        rows.append((f"diag:{key}", v, nan, unit, f"Solver diagnostic: {label}"))
    return rows


def _polar_rows(p: PolarResult) -> list[tuple]:
    from ..analysis.polar import polar_metrics  # analysis is numpy-only, cheap to import
    nan = float("nan")
    units = {"alpha_cl_max": "deg", "alpha_cd_min": "deg", "alpha_ld_max": "deg", "alpha_zero_lift": "deg",
             "alpha_min": "deg", "alpha_max": "deg", "lift_slope_per_deg": "1/deg", "lift_slope_per_rad": "1/rad"}
    rows: list[tuple] = [("re", p.re, nan, "", "Reynolds number"), ("mach", p.mach, nan, "", "Mach number"),
                         ("ncrit", p.ncrit, nan, "", "Transition amplification factor")]
    for k, v in polar_metrics(p).items():
        rows.append((k, v, nan, units.get(k, ""), k.replace("_", " ")))
    rows.append(("wall_seconds", p.wall_seconds, nan, "s", "Wall-clock time"))
    return rows


_COEFF3D = (("cd", "Drag coefficient CD"), ("cs", "Side-force coefficient CS"), ("cl", "Lift coefficient CL"),
            ("c_roll", "Rolling-moment coefficient"), ("cm", "Pitching-moment coefficient Cm"),
            ("cn", "Yawing-moment coefficient Cn"))


def _cfd3d_rows(r: CFD3DResult) -> list[tuple]:
    nan = float("nan")
    rows: list[tuple] = []
    for key, label in _COEFF3D:
        st = getattr(r, key)
        rows += [(key, st.mean, st.stderr, "", f"{label}, time mean (stderr = standard error of the mean)"),
                 (f"{key}_rms", st.rms, nan, "", f"{label}, rms of fluctuation"),
                 (f"{key}_min", st.min, nan, "", f"{label}, minimum"),
                 (f"{key}_max", st.max, nan, "", f"{label}, maximum")]
    moments = tuple(r.moments) + (nan,) * (3 - len(tuple(r.moments)))
    grid = tuple(r.grid) + (0,) * (3 - len(tuple(r.grid)))
    rows += [
        ("lift_to_drag", r.lift_to_drag, nan, "", "Lift-to-drag ratio"),
        ("drag", r.drag, nan, "N", "Mean drag"),
        ("side", r.side, nan, "N", "Mean side force (+y, to the left)"),
        ("lift", r.lift, nan, "N", "Mean lift (negative = downforce)"),
        ("roll_moment", moments[0], nan, "N m", "Mean rolling moment about +x"),
        ("pitch_moment", moments[1], nan, "N m", "Mean pitching moment, nose up positive"),
        ("yaw_moment", moments[2], nan, "N m", "Mean yawing moment about +z"),
        ("drag_area", r.drag_area, nan, "m^2", "Drag area CD x reference area"),
        ("power", r.power, nan, "W", "Power to overcome mean drag"),
        ("strouhal", r.strouhal if r.strouhal is not None else nan, nan, "", "Strouhal number"),
        ("strouhal_length", r.strouhal_length, nan, "m", "Length the Strouhal number is based on"),
        ("re_real", r.re_real, nan, "", "Reynolds number of the real flow"),
        ("re_sim", r.re_sim, nan, "", "Reynolds number simulated"),
        ("grid_nx", grid[0], nan, "cells", "Grid cells in x"),
        ("grid_ny", grid[1], nan, "cells", "Grid cells in y (half the domain with a symmetry plane)"),
        ("grid_nz", grid[2], nan, "cells", "Grid cells in z"),
        ("cells", r.cells, nan, "cells", "Total grid cells"),
        ("cells_per_ref", r.cells_per_ref, nan, "", "Cells per reference length"),
        ("symmetry_plane", r.symmetry_plane, nan, "", "Half domain with a symmetry plane at y = 0"),
        ("t_end", r.t_end, nan, "L/U", "Simulated time in convective units"),
        ("t_avg_start", r.t_avg_start, nan, "L/U", "Start of averaging in convective units"),
        ("converged", r.converged, nan, "", "Statistically converged"),
        ("wall_seconds", r.wall_seconds, nan, "s", "Wall-clock time"),
        ("device", r.device, nan, "", "Compute device"),
        ("precision", r.precision, nan, "", "Floating-point precision"),
        ("turbulence", r.turbulence, nan, "", "Turbulence treatment"),
    ]
    for body, vals in r.per_body.items():
        for k, v in vals.items():
            rows.append((f"body:{body}:{k}", v, nan, "", f"{k} of body '{body}'"))
    return rows + _physics_rows(r, "W", "N", "m^3/s")


_WING_UNITS = {"cl_alpha_per_deg": "1/deg", "cl_alpha_per_rad": "1/rad", "alpha_zero_lift": "deg",
               "alpha_cl_max": "deg", "alpha_cd_min": "deg", "alpha_ld_max": "deg", "alpha_cruise": "deg",
               "cm_alpha_per_deg": "1/deg", "alpha_min": "deg", "alpha_max": "deg"}
#: Descriptions that the key alone does not convey (the Oswald factor depends on its fit range).
_WING_DESCRIPTIONS = {
    "cd0": "Zero-lift drag CD0 (parabolic fit over oswald_cl_lo..oswald_cl_hi)",
    "k_parabolic": "Induced-drag factor k of CD = CD0 + b CL + k CL^2 (same fit)",
    "oswald": "Oswald efficiency e = 1/(pi AR k) (same fit; nan when R^2 < 0.98)",
    "oswald_cl_lo": "Lowest CL of the parabolic drag-polar fit",
    "oswald_cl_hi": "Highest CL of the parabolic drag-polar fit",
    "oswald_r2": "R^2 of the parabolic drag-polar fit",
    "e_cruise": "Span efficiency CL^2/(pi AR CDi) at the cruise point",
}


def _wing_rows(w: WingResult) -> list[tuple]:
    from ..analysis.wing import wing_metrics  # analysis is numpy-only, cheap to import
    nan = float("nan")
    rows: list[tuple] = [
        ("method", w.method, nan, "", "Finite-wing method (vlm = vortex lattice, llt = lifting line)"),
        ("viscous", w.viscous, nan, "", "Profile drag from section polars"),
        ("area", w.area, nan, "m^2", "Planform (reference) area"),
        ("span", w.span, nan, "m", "Span"),
        ("aspect_ratio", w.aspect_ratio, nan, "", "Aspect ratio"),
        ("mac", w.mac, nan, "m", "Mean aerodynamic chord"),
        ("re_mac", w.re_mac, nan, "", "Reynolds number on the mean aerodynamic chord"),
        ("stall_alpha", w.stall_alpha, nan, "deg", "Stall onset, critical-section estimate (nan = not reached)"),
        ("stall_cl", w.stall_cl, nan, "", "Wing CL at the stall-onset estimate"),
    ]
    for k, v in wing_metrics(w).items():
        if k == "aspect_ratio":
            continue
        rows.append((k, v, nan, _WING_UNITS.get(k, ""), _WING_DESCRIPTIONS.get(k, k.replace("_", " "))))
    rows.append(("wall_seconds", w.wall_seconds, nan, "s", "Wall-clock time"))
    return rows


def result_csv(result: CFDResult | PolarResult | CFD3DResult | WingResult, path: str | os.PathLike) -> Path:
    """Scalar summary, one quantity per row: ``quantity,value,stderr,unit,description``.

    3D CFD results list all six coefficients with their statistics, forces in N
    and moments in N m; wing results the planform and :func:`wing_metrics`.
    """
    if isinstance(result, CFDResult):
        rows = _cfd_rows(result)
    elif isinstance(result, PolarResult):
        rows = _polar_rows(result)
    elif isinstance(result, CFD3DResult):
        rows = _cfd3d_rows(result)
    elif isinstance(result, WingResult):
        rows = _wing_rows(result)
    else:
        raise TypeError(f"not a result: {type(result).__name__}")
    rows += [("note", n, float("nan"), "", "") for n in result.notes]
    rows += [("warning", w, float("nan"), "", "") for w in result.warnings]
    return _write_rows(path, ("quantity", "value", "stderr", "unit", "description"), rows)


_STD_2D = ("cd", "cl", "cm")
_STD_3D = ("cd", "cs", "cl", "c_roll", "cm", "cn")


def history_csv(result: CFDResult | CFD3DResult | WingResult | Mapping[str, np.ndarray],
                path: str | os.PathLike) -> Path:
    """Force-coefficient history, one row per stored step: ``t,cd,cl,cm[,...]``.

    ``t`` is in convective times (length_ref / speed). Extra history keys follow
    the standard ones in their stored order (the Nusselt number "nu", a fan's
    "fan0_thrust", ...). A series sampled on another clock than ``t`` (a
    diagnostics column recorded every few steps) is left out rather than
    misaligned; :func:`aerolume.export.diagnostics_csv` writes those. A 3D CFD
    result writes ``t,cd,cs,cl,c_roll,cm,cn``. A wing analysis has no time history
    (it is a steady solution per angle); its sequence of results is the sweep, so a
    :class:`WingResult` writes :func:`wing_csv`.
    """
    if isinstance(result, WingResult):
        return wing_csv(result, path)
    std = _STD_3D if isinstance(result, CFD3DResult) else _STD_2D
    hist = result.history if isinstance(result, (CFDResult, CFD3DResult)) else result
    if "t" not in hist:
        raise ValueError("history has no time column 't'")
    if not isinstance(result, (CFDResult, CFD3DResult)) and any(k in hist for k in ("cs", "c_roll", "cn")):
        std = _STD_3D                                   # a bare 3D history mapping
    n = np.asarray(hist["t"]).size
    if any(np.asarray(hist[k]).size != n for k in std if k in hist):
        raise ValueError("history columns differ in length")
    keys = ["t"] + [k for k in std if k in hist] + [k for k in hist if k != "t" and k not in std
                                                    and np.asarray(hist[k]).size == n]
    cols = [np.asarray(hist[k], dtype=np.float64).ravel() for k in keys]
    return _write_rows(path, keys, zip(*(c.tolist() for c in cols), strict=True))


def surface_csv(result: CFDResult | Sequence[SurfaceData], path: str | os.PathLike) -> Path:
    """Surface distributions of all bodies stacked: ``body,s,x,y,nx,ny,cp,cf[,sensitivity]``.

    Wind-frame coordinates in metres; ``s`` is arc length from each body's first point.
    """
    surfaces = result.surface if isinstance(result, CFDResult) else list(result)
    has_sens = any(sd.sensitivity is not None for sd in surfaces)
    keys = ["s", "x", "y", "nx", "ny", "cp", "cf"] + (["sensitivity"] if has_sens else [])
    rows: list[list] = []
    for sd in surfaces:
        cols = []
        for k in keys:
            v = getattr(sd, k)
            cols.append(np.full(len(sd.s), np.nan) if v is None else np.asarray(v, dtype=np.float64))
        rows.extend([sd.body, *vals] for vals in zip(*(c.tolist() for c in cols), strict=True))
    return _write_rows(path, ["body", *keys], rows)


def polar_csv(polar: PolarResult, path: str | os.PathLike) -> Path:
    """Every polar point: ``alpha,cl,cd,cdp,cdf,cm,ld,xtr_upper,xtr_lower,converged``.

    Non-converged points are kept (flagged ``false``) so the file shows where the
    sweep failed; filter on ``converged`` for plotting.
    """
    rows = []
    for p in sorted(polar.points, key=lambda q: q.alpha):
        ld = p.cl / p.cd if p.cd and math.isfinite(p.cd) and p.cd > 0 else float("nan")
        rows.append((p.alpha, p.cl, p.cd, p.cdp, p.cdf, p.cm, ld, p.xtr_upper, p.xtr_lower, p.converged))
    return _write_rows(path, ("alpha", "cl", "cd", "cdp", "cdf", "cm", "ld", "xtr_upper", "xtr_lower",
                              "converged"), rows)


def read_csv(path: str | os.PathLike) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a CSV written by this module (strings; for tests and re-import)."""
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        return header, [row for row in r]


# ---------------------------------------------------------------------------
# 3D: slices and finite wings
# ---------------------------------------------------------------------------

#: in-plane (column, row) axes of a slice by its normal (SliceData docstring)
_SLICE_AXES = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}


def slice_csv(slice_data: SliceData | CFD3DResult, path: str | os.PathLike, *, index: int = 0,
              axis: str | None = None) -> Path:
    """One planar cut as a point table: ``x,y,z,<fields...>[,mask]``, one row per cell.

    Coordinates are the cell centres in metres (wind frame; the slice-normal
    coordinate is the plane position on every row), so the file imports into
    any plotting tool as scattered 3D points. Rows run column-fastest (row 0 =
    lowest row coordinate). Given a :class:`CFD3DResult`, the slice is chosen by
    ``axis`` (the first slice with that normal) or else by ``index``.
    """
    if isinstance(slice_data, CFD3DResult):
        slices = slice_data.slices
        if axis is not None:
            pick = [s for s in slices if s.axis == axis]
            if not pick:
                raise ValueError(f"the result has no slice normal to {axis!r}")
            sd = pick[0]
        else:
            if not slices:
                raise ValueError("the result has no slices")
            sd = slices[index]
    else:
        sd = slice_data
    if sd.axis not in _SLICE_AXES:
        raise ValueError(f"slice axis must be 'x', 'y' or 'z', got {sd.axis!r}")
    shapes = {np.shape(a) for a in sd.fields.values()} | ({np.shape(sd.mask)} if sd.mask is not None else set())
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(f"slice fields must share one 2D shape, got {sorted(shapes)}")
    nr, nc = next(iter(shapes))
    c0, c1, r0, r1 = (float(v) for v in sd.extent)
    cc = c0 + (c1 - c0) / nc * (np.arange(nc) + 0.5)
    rr = r0 + (r1 - r0) / nr * (np.arange(nr) + 0.5)
    col_ax, row_ax = _SLICE_AXES[sd.axis]
    cgrid, rgrid = np.meshgrid(cc, rr)              # (nr, nc), row-major = column-fastest
    coords = {sd.axis: np.full(nr * nc, float(sd.position)), col_ax: cgrid.ravel(), row_ax: rgrid.ravel()}
    names = list(sd.fields)
    for reserved in ("x", "y", "z", "mask"):
        if reserved in names:
            raise ValueError(f"field name {reserved!r} is reserved in the slice export")
    cols = [coords["x"], coords["y"], coords["z"]] + \
        [np.asarray(sd.fields[n], dtype=np.float64).ravel() for n in names]
    header = ["x", "y", "z", *names]
    if sd.mask is not None:
        cols.append(np.asarray(sd.mask, dtype=np.float64).ravel())
        header.append("mask")
    return _write_rows(path, header, zip(*(c.tolist() for c in cols), strict=True))


def wing_csv(wing: WingResult, path: str | os.PathLike) -> Path:
    """Every wing operating point: ``alpha,cl,cd,cdi,cdv,cm,cy,c_roll,cn,e,ld,converged``.

    Coefficients on the wing's reference area (wind axes, DESIGN.md 7.2).
    Non-converged points are kept (flagged ``false``), as in :func:`polar_csv`.
    """
    rows = []
    for p in sorted(wing.points, key=lambda q: q.alpha):
        ld = p.cl / p.cd if p.cd and math.isfinite(p.cd) and p.cd > 0 else float("nan")
        rows.append((p.alpha, p.cl, p.cd, p.cdi, p.cdv, p.cm, p.cy, p.c_roll, p.cn, p.e, ld, p.converged))
    return _write_rows(path, ("alpha", "cl", "cd", "cdi", "cdv", "cm", "cy", "c_roll", "cn", "e", "ld",
                              "converged"), rows)


_SPAN_KEYS = ("cl_local", "c_cl", "alpha_i", "cd_local")


def wing_span_csv(wing: WingResult, path: str | os.PathLike, *, converged_only: bool = False) -> Path:
    """Span loading of every angle in long format: ``alpha,y,cl_local,c_cl,alpha_i,cd_local,converged``.

    One row per (angle, strip), angles ascending, strips in the solver's order
    (y in metres across the whole wing). Missing distributions (``cd_local`` of
    an inviscid run) are written as ``nan``; points without span data are
    skipped. Long format reads directly into pandas / Excel pivot tables, and
    holds sweeps whose angles have different strip counts.
    """
    rows: list[tuple] = []
    for p in sorted(wing.points, key=lambda q: q.alpha):
        if p.y is None or (converged_only and not p.converged):
            continue
        y = np.asarray(p.y, dtype=np.float64).ravel()
        cols = []
        for k in _SPAN_KEYS:
            v = getattr(p, k)
            v = np.full(y.size, np.nan) if v is None else np.asarray(v, dtype=np.float64).ravel()
            if v.size != y.size:
                raise ValueError(f"alpha {p.alpha:g}: {k} has {v.size} values for {y.size} strips")
            cols.append(v)
        rows.extend((p.alpha, yy, *vals, p.converged)
                    for yy, *vals in zip(y.tolist(), *(c.tolist() for c in cols), strict=True))
    return _write_rows(path, ("alpha", "y", *_SPAN_KEYS, "converged"), rows)
