"""Report content: turns a project + simulation ids into a :class:`Document`.

This module decides *what* the report says; :mod:`.pdf` and :mod:`.html` decide how
it looks. Each section builder receives the loaded entries (simulation, design,
result) and returns blocks, or nothing when the section does not apply (no polar
in a CFD-only report, no comparison for a single run).

Principles the text follows (DESIGN.md section 6): every number carries its validity
context. Means come with their standard error, a scaled Reynolds number is stated
next to the real one, unconverged statistics and points outside the solver's
benchmark range are flagged with badges in the summary, and the methodology section
explains in plain words what the numbers can and cannot be used for.

3D Navier–Stokes results and finite-wing sweeps (DESIGN.md section 7) get their own
blocks from :mod:`.content3d` inside the same sections; 2D and 3D numbers are never
put in one comparison table (per-span and whole-body coefficients are different
quantities).

The project is duck-typed: ``project.name``, ``project.designs[id]``,
``project.simulations[id]`` and ``project.load_result(sim_id)``.
"""
from __future__ import annotations

import contextlib
import functools
import math
import platform
import time
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from typing import Any

import numpy as np

from .. import __version__
from ..analysis import physics as P
from ..viz import figures as F
from . import content3d as C3
from . import content_physics as CP
from . import fmt
from .common import (
    ROUGH,
    ROUGH_NOTE,
    STDERR_OK,
    STDERR_WARN,
    Entry,
)
from .common import compare_rows as _compare_rows
from .common import comparison_warnings as _comparison_warnings
from .common import delta_cell as _delta_cell
from .common import dyn_pressure as _q
from .common import err_text as _err_text
from .common import fig as _fig
from .common import fluid as _fluid
from .common import pm_text as _pm_text
from .common import rel_err as _rel_err
from .common import rough as _rough
from .common import sci as _sci
from .common import sub as _sub
from .common import turbulence_text as _turbulence_text
from .document import (
    KPI,
    Badges,
    Bullets,
    Callout,
    Cell,
    Document,
    FigureBlock,
    Heading,
    KeyValues,
    KPIRow,
    Paragraph,
    Section,
    Table,
)

SECTIONS = ("summary", "geometry", "setup", "results", "surface", "fields", "heat", "flowvis", "fans",
            "convergence", "grid", "sensitivity", "polar", "comparison", "methodology")

SECTION_TITLES = {
    "summary": "Summary",
    "geometry": "Geometry",
    "setup": "Simulation setup",
    "results": "Results",
    "surface": "Surface pressure and friction",
    "fields": "Flow fields",
    "heat": "Heat transfer",
    "flowvis": "Flow visualisation",
    "fans": "Fans and jets",
    "convergence": "Convergence",
    "grid": "Grid convergence",
    "sensitivity": "Shape sensitivity",
    "polar": "Polars",
    "comparison": "Comparison",
    "methodology": "Methodology and validity",
    "appendix": "Appendix: run details",
}

# Validity thresholds. They encode what the solver has actually been benchmarked
# against (DESIGN.md section 6: cylinder Re 20-100, Schaefer-Turek, NACA 0012 at
# Re 1000) and what statistical precision is good enough to rank variants.
BENCHMARK_RE_MAX = 1000.0       # highest Re among the laminar validation cases
GRID_FINE = 128.0               # cells per reference length
GRID_STANDARD = 64.0
MACH_LIMIT = 0.3                # incompressible assumption
SPEED_OF_SOUND_AIR = 340.3      # m/s at ISA sea level; only used for the Mach warning


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------

def _lookup(mapping: Any, key: str) -> Any:
    try:
        return mapping[key]
    except (KeyError, IndexError, TypeError):
        return None


def load_entries(project: Any, sim_ids: Sequence[str]) -> list[Entry]:
    """Resolve simulations, designs and results. Unknown ids raise ``KeyError``;
    a result that fails to load becomes ``Entry.error`` (reported, not fatal)."""
    out = []
    for sid in sim_ids:
        sim = _lookup(project.simulations, sid)
        if sim is None:
            raise KeyError(f"simulation {sid!r} is not in project {getattr(project, 'name', '?')!r}")
        design = _lookup(project.designs, getattr(sim, "design_id", ""))
        result, err = None, None
        try:
            result = project.load_result(sid)
        except Exception as exc:  # noqa: BLE001 - a damaged result must not kill the whole report
            err = f"{type(exc).__name__}: {exc}"
        out.append(Entry(sid, sim, design, result, err))
    _disambiguate(out)
    return out


def _disambiguate(entries: list[Entry]) -> None:
    """Give simulations that share a name a unique label: "<design> · <name>", then a counter.

    New studies name their runs after the conditions ("CFD · 30 m/s"), so a comparison
    of two car designs would otherwise list two identical names in every table.
    """
    def counts(names: list[str]) -> dict[str, int]:
        c: dict[str, int] = {}
        for n in names:
            c[n] = c.get(n, 0) + 1
        return c
    dup = counts([e.name for e in entries])
    for e in entries:
        if dup[e.name] > 1 and e.design is not None:
            e.label = f"{getattr(e.design, 'name', '?')} · {getattr(e.sim, 'name', e.sim_id)}"
    seen: dict[str, int] = {}
    dup = counts([e.name for e in entries])
    for e in entries:
        if dup[e.name] > 1:
            seen[e.name] = seen.get(e.name, 0) + 1
            e.label = f"{e.name} ({seen[e.name]})"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _ref_sym(e: Entry) -> str:
    return "c" if e.kind == "airfoil" else "L"


def _main_bodies(e: Entry) -> list[str] | None:
    """Names of non-wheel bodies (Cp plots of a spinning wheel are not meaningful)."""
    if e.design is None:
        return None
    names = [b.name for b in e.design.bodies if getattr(b, "role", "body") != "wheel"]
    return names or None


def _source_text(d: Any) -> str:
    src = getattr(d, "source", {}) or {}
    t = src.get("type", "")
    if t == "naca":
        return f"NACA {src.get('code', '')}".strip()
    if t == "template":
        return f"template “{src.get('name', '')}”"
    if t in ("dat", "svg", "dxf", "image", "csv", "file"):
        return f"imported {t.upper()} ({src.get('path', src.get('file', ''))})".replace(" ()", "")
    if t:
        return t
    return "drawn / unknown"


def _polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _thickness(p: np.ndarray) -> tuple[float, float]:
    """Max thickness / chord and its chordwise position for an airfoil outline (design frame)."""
    x, y = p[:, 0], p[:, 1]
    n = len(x)
    i_le, i_te = int(np.argmin(x)), int(np.argmax(x))
    a = np.array([(i_te + k) % n for k in range(((i_le - i_te) % n) + 1)])
    b = np.array([(i_le + k) % n for k in range(((i_te - i_le) % n) + 1)])
    chord = float(x[i_te] - x[i_le]) or 1.0
    xs = np.linspace(x[i_le], x[i_te], 400)
    oa, ob = np.argsort(x[a]), np.argsort(x[b])
    ya = np.interp(xs, x[a][oa], y[a][oa])
    yb = np.interp(xs, x[b][ob], y[b][ob])
    t = np.abs(ya - yb)
    i = int(np.argmax(t))
    return float(t[i] / chord), float((xs[i] - x[i_le]) / chord)


# ---------------------------------------------------------------------------
# validity assessment
# ---------------------------------------------------------------------------

def assess(e: Entry) -> list[tuple[str, str]]:
    """Validity badges (text, tone) for one simulation — the 'can I trust this?' line."""
    r = e.result
    if r is None:
        return [("No result", "warn")]
    if e.is_cfd3d:
        return C3.assess_cfd3d(e)
    if e.is_wing:
        return C3.assess_wing(e)
    out: list[tuple[str, str]] = []
    if e.is_cfd:
        out.append(("Statistically converged", "ok") if r.converged else ("Not converged", "warn"))
        rel = _rel_err(r.cd)
        if fmt.finite(rel):
            tone = "ok" if rel < STDERR_OK else ("warn" if rel < STDERR_WARN else "bad")
            # two significant figures: a converged laminar run is known to a few hundredths
            # of a per cent, which one decimal would print as a meaningless "±0.0 %"
            out.append((f"Cd uncertainty ±{fmt.num(rel * 100, 2)} %", tone))
        elif not CP.has_forces(e):
            out += CP.uncertainty_badge(e)
        else:
            out.append(("Uncertainty not estimated", "warn"))
        if fmt.finite(r.re_sim) and fmt.finite(r.re_real):
            if r.re_sim < 0.95 * r.re_real:
                out.append((f"Re scaled: {fmt.reynolds(r.re_sim)} of {fmt.reynolds(r.re_real)}", "warn"))
            else:
                out.append(("Full-scale Reynolds number", "ok"))
            if r.re_sim > BENCHMARK_RE_MAX:
                out.append((f"Beyond benchmark range (Re ≤ {BENCHMARK_RE_MAX:,.0f})", "warn"))
            else:
                out.append(("Within benchmark range", "ok"))
        cpr = r.cells_per_ref
        if fmt.finite(cpr):
            # resolution classes, deliberately not called Draft / Standard / Fine: those are the
            # planner's quality presets, and a Draft airfoil (64 cells per chord) must not read
            # as "Standard grid"
            if cpr >= GRID_FINE:
                out.append((f"High resolution ({cpr:.0f} cells/{_ref_sym(e)})", "ok"))
            elif cpr >= GRID_STANDARD:
                out.append((f"Moderate resolution ({cpr:.0f} cells/{_ref_sym(e)})", "info"))
            else:
                out.append((f"Low resolution ({cpr:.0f} cells/{_ref_sym(e)})", "warn"))
        out.append(("2D · per metre span", "info"))
    elif e.is_polar:
        n = len(r.points)
        nc = sum(1 for p in r.points if p.converged)
        out.append((f"{nc} of {n} points converged", "ok" if nc == n else ("warn" if nc else "bad")))
        out.append((f"Re {fmt.reynolds(r.re)}", "info"))
        if fmt.finite(r.mach) and r.mach > MACH_LIMIT:
            out.append((f"Mach {r.mach:.2f}: compressibility", "warn"))
        out.append(("Viscous–inviscid (mfoil)", "info"))
    speed = getattr(e.sim.conditions, "speed", 0.0)
    if e.is_cfd and speed / SPEED_OF_SOUND_AIR > MACH_LIMIT and "Water" not in _fluid(e).name:
        out.append((f"Mach ≈ {speed / SPEED_OF_SOUND_AIR:.2f} > {MACH_LIMIT}: incompressible model", "bad"))
    return out


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _kpis_cfd(e: Entry) -> KPIRow:
    r = e.result
    ld = r.lift_to_drag if hasattr(r, "lift_to_drag") else (r.cl.mean / r.cd.mean if r.cd.mean else float("nan"))
    lift_label = "Lift L′" if e.kind != "vehicle" else "Lift L′ (− = downforce)"
    def se_sub(st: Any, missing: str) -> str:
        if not fmt.finite(st.stderr):
            return missing
        return f"± {_err_text(st)} " + ("(rough estimate)" if _rough(st) else "(1 s.e.)")

    return KPIRow([
        KPI("Drag coefficient Cd", fmt.mean_only(r.cd.mean, r.cd.stderr),
            sub=se_sub(r.cd, "no uncertainty estimate")),
        KPI("Lift coefficient Cl", fmt.mean_only(r.cl.mean, r.cl.stderr), sub=se_sub(r.cl, "")),
        KPI("Lift-to-drag L/D", fmt.fixed(ld, 1) if abs(ld) < 1000 else fmt.num(ld, 3),
            sub=fmt.eq("Cm", fmt.mean_only(r.cm.mean, r.cm.stderr))),
        KPI("Drag D′", fmt.num(r.drag, 4), "N/m",
            sub=f"{lift_label} {fmt.num(r.lift, 3)} N/m"),
    ])


def _stall_missed(m: Mapping[str, Any]) -> bool:
    """The polar's lift is still rising at its last angle: its 'Cl,max' is a lower bound."""
    return m.get("cl_max_reached") == 0.0


def _cl_max_text(m: Mapping[str, Any], digits: int = 3) -> str:
    text = fmt.num(m.get("cl_max"), digits)
    return f"≥ {text}" if _stall_missed(m) else text


def _cl_max_where(m: Mapping[str, Any]) -> str:
    a = fmt.angle(m.get("alpha_cl_max"))
    return f"still rising at α = {a} (stall not reached)" if _stall_missed(m) else f"at α = {a}"


def _kpis_polar(e: Entry) -> KPIRow | None:
    m = _polar_metrics(e.result)
    if not m or not fmt.finite(m.get("cl_max")):
        return None
    return KPIRow([
        KPI("Best lift-to-drag (L/D)max", fmt.fixed(m["ld_max"], 1),
            sub=f"at α = {fmt.angle(m['alpha_ld_max'])}, Cl = {fmt.num(m['cl_at_ld_max'], 3)}"),
        KPI("Maximum lift Cl,max", _cl_max_text(m), sub=_cl_max_where(m)),
        KPI("Minimum drag Cd,min", fmt.num(m["cd_min"], 3), sub=f"at Cl = {fmt.num(m['cl_at_cd_min'], 3)}"),
        KPI("Lift slope dCl/dα", fmt.num(m["lift_slope_per_rad"], 3), "/rad",
            sub=f"zero lift at α = {fmt.num(m['alpha_zero_lift'], 3)}°"),
    ])


def _polar_metrics(polar: Any) -> dict[str, float]:
    try:
        from ..analysis.polar import polar_metrics
        return polar_metrics(polar)
    except Exception:  # noqa: BLE001 - analysis is another module; degrade to the basics
        tb = polar.table()
        if tb["alpha"].size == 0:
            return {}
        i = int(np.argmax(tb["cl"]))
        ld = tb["cl"] / np.where(tb["cd"] > 0, tb["cd"], np.nan)
        j = int(np.nanargmax(ld))
        k = int(np.argmin(tb["cd"]))
        nan = float("nan")
        # without the stall test a peak at the last angle is reported as not reached
        reached = 0.0 if tb["alpha"][i] >= np.max(tb["alpha"]) else 1.0
        return {"cl_max": tb["cl"][i], "alpha_cl_max": tb["alpha"][i], "cl_max_reached": reached, "ld_max": ld[j],
                "alpha_ld_max": tb["alpha"][j], "cl_at_ld_max": tb["cl"][j], "cd_min": tb["cd"][k],
                "cl_at_cd_min": tb["cl"][k], "lift_slope_per_rad": nan, "alpha_zero_lift": nan, "cm0": nan}


def _sim_overview_table(entries: Sequence[Entry]) -> Table:
    rows = []
    for e in entries:
        c = e.sim.conditions
        status = "result available" if e.result is not None else (
            "result unreadable" if e.error else str(getattr(e.sim, "status", "not run")))
        typ = e.type_label
        alpha = fmt.angle(c.alpha_deg) if e.analysis == "cfd" else "sweep"
        re_ = c.reynolds(e.design.length_ref) if e.design is not None else float("nan")
        rows.append([e.name, getattr(e.design, "name", "—"), typ, fmt.num(c.speed, 4), alpha,
                     fmt.reynolds(re_), Cell(status, "neutral" if e.result is not None else "warn")])
    return Table(["Simulation", "Design", "Type", "U [m/s]", "α", "Re (real)", "Status"], rows,
                 align=["l", "l", "l", "r", "r", "r", "l"], caption="Simulations covered by this report",
                 widths=[3.0, 2.2, 1.0, 1.0, 0.8, 1.4, 1.7])


def section_summary(entries: Sequence[Entry], ctx: dict) -> list:
    blocks: list = []
    designs = {id(e.design): e.design for e in entries if e.design is not None}
    counts = {"cfd2": 0, "polar": 0, "cfd3": 0, "wing": 0}
    for e in entries:
        counts["wing" if e.analysis == "wing" else "polar" if e.analysis == "polar" else
               "cfd3" if e.dimension == 3 else "cfd2"] += 1
    words = {"cfd2": ("2D Navier–Stokes CFD simulation", "s"), "cfd3": ("3D Navier–Stokes CFD simulation", "s"),
             "polar": ("airfoil polar", "s"), "wing": ("finite-wing sweep", "s")}
    kinds = [f"{'one' if n == 1 else n} {words[k][0]}{words[k][1] if n > 1 else ''}" for k, n in counts.items() if n]
    lead = (f"This report documents {', '.join(kinds[:-1]) + ' and ' + kinds[-1] if len(kinds) > 1 else kinds[0]} "
            f"of {'one' if len(designs) == 1 else len(designs)} design{'s' if len(designs) != 1 else ''} in project "
            f"**{ctx['project_name']}**. ")
    has2, has3 = counts["cfd2"] + counts["polar"] > 0, counts["cfd3"] + counts["wing"] > 0
    if has2 and has3:
        lead += ("2D forces are per metre of span; 3D forces are for the whole body, in newtons, with "
                 "coefficients on a reference area, so the two are never compared with each other. ")
    elif counts["cfd3"]:
        lead += "Forces are for the whole body, in newtons; coefficients use the design's reference area. "
    elif has3:
        lead += "Wing coefficients are for the whole wing, on its planform area. "
    else:
        lead += "All forces are two-dimensional, per metre of span. "
    timed = counts["cfd2"] + counts["cfd3"] > 0          # only Navier–Stokes runs carry ± values
    if "methodology" in ctx["numbers"]:
        lead += (("Values marked ± are one standard error of the time average; the" if timed else "The")
                 + " line under each result summarises how far the numbers can be trusted, and "
                 f"section {ctx['numbers'].get('methodology', '')} explains the method and its limits.")
    elif timed:
        lead += "± values are one standard error of the time average."
    blocks.append(Paragraph(lead, tone="lead"))
    if len(entries) > 1:
        blocks.append(_sim_overview_table(entries))
    findings = _key_findings(entries)
    if findings:
        blocks.append(Heading("Key findings", level=2))
        blocks.append(Bullets(findings))
    detailed = [e for e in entries if e.result is not None][:6]
    for e in detailed:
        blocks.append(Heading(e.name, level=2))
        # a run without forces (a heated cavity, a fan in a closed room) leads with its heat
        # transfer or its fans instead of four tiles of dashes; a heated body shows both rows
        forces = not (e.is_cfd or e.is_cfd3d) or CP.has_forces(e)
        row = None if not forces else (
            _kpis_cfd(e) if e.is_cfd else _kpis_polar(e) if e.is_polar else
            C3.kpis_cfd3d(e) if e.is_cfd3d else C3.kpis_wing(e) if e.is_wing else None)
        if row is not None:
            blocks.append(row)
        if e.is_cfd or e.is_cfd3d:
            phys = CP.physics_kpis(e)
            if phys is not None:
                blocks.append(phys)
        blocks.append(Badges(assess(e)))
        for w in getattr(e.result, "warnings", []) or []:
            blocks.append(Callout(fmt.solver_text(w), "warn"))
    for e in entries:
        if e.result is None:
            msg = (f"The result of **{e.name}** could not be read ({e.error})." if e.error else
                   f"**{e.name}** has no result yet (status: {getattr(e.sim, 'status', 'new')}). "
                   "Run it and regenerate the report to include it.")
            blocks.append(Callout(msg, "warn", title="Missing result"))
    return blocks


def _key_findings(entries: Sequence[Entry]) -> list[str]:
    out: list[str] = []
    cfd = [e for e in entries if e.is_cfd]
    pol = [e for e in entries if e.is_polar]
    for e in cfd[:4]:
        r = e.result
        c = e.sim.conditions
        phys = CP.physics_finding(e)
        if not CP.has_forces(e):
            # no solid body, no forces: the heat transfer or the fans are the finding
            out.append(f"**{e.name}**: {phys}." if phys else f"**{e.name}**: no forces (no solid bodies).")
            continue
        text = f"**{e.name}**: {fmt.eq('Cd', fmt.pm(r.cd.mean, r.cd.stderr))}, {fmt.eq('Cl', fmt.pm(r.cl.mean, r.cl.stderr))}"
        # the dimensional clause only when the result has the numbers: a sentence of dashes
        # ("drag — N ..., needing — W/m") reads as a broken report, not as a missing value
        if fmt.finite(r.drag):
            text += f"; drag {fmt.num(r.drag, 3)} N per metre span at {fmt.num(c.speed, 3)} m/s"
            if fmt.finite(r.power):
                text += f", needing {fmt.num(r.power, 3)} W/m to overcome"
        if phys:
            text += f"; {phys}"
        out.append(text + ".")
    cfd = [e for e in cfd if CP.has_forces(e)]          # drag rankings only among runs with forces
    if len(cfd) >= 2:
        base = cfd[0].result
        best = min(cfd, key=lambda e: e.result.cd.mean if fmt.finite(e.result.cd.mean) else math.inf)
        if best is not cfd[0] and fmt.finite(base.cd.mean) and base.cd.mean:
            d = (best.result.cd.mean - base.cd.mean) / abs(base.cd.mean) * 100
            se = math.hypot(best.result.cd.stderr, base.cd.stderr)
            if _rough(best.result.cd) or _rough(base.cd) or not fmt.finite(se):
                verdict = " (error bars too rough to say whether this exceeds the noise)."
            elif abs(best.result.cd.mean - base.cd.mean) > 2 * se:
                verdict = " (statistically significant)."
            else:
                verdict = " (within statistical noise)."
            out.append(f"Lowest drag: **{best.name}**, {fmt.pct(d)} Cd versus the baseline {cfd[0].name}"
                       + verdict)
        elif best is cfd[0]:
            out.append(f"The baseline **{cfd[0].name}** has the lowest drag of the compared variants.")
    for e in pol[:4]:
        m = _polar_metrics(e.result)
        if m and fmt.finite(m.get("ld_max")):
            out.append(f"**{e.name}**: best L/D {fmt.fixed(m['ld_max'], 1)} at α = {fmt.angle(m['alpha_ld_max'])}; "
                       f"Cl,max {_cl_max_text(m)} {_cl_max_where(m)}.")
    return out + C3.findings(entries)


def section_geometry(entries: Sequence[Entry], ctx: dict) -> list:
    blocks: list = []
    designs: list[Any] = []
    for e in entries:
        if e.design is not None and all(d is not e.design for d in designs):
            designs.append(e.design)
    if not designs:
        return []
    no = ctx["no"]
    for k, d in enumerate(designs, 1):
        if len(designs) > 1:
            blocks.append(Heading(d.name, level=2, number=f"{no}.{k}"))
        if int(getattr(d, "dimension", 2) or 2) == 3:
            blocks += C3.geometry_blocks(d, ctx)
            continue
        if not d.bodies:
            blocks.append(Paragraph(f"{d.name} has no outline stored.", tone="muted"))
            continue
        pts = np.vstack([b.points for b in d.bodies])
        w, h = float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))
        kind = {"airfoil": "Airfoil section", "vehicle": "Vehicle side profile"}.get(d.kind, "Generic shape")
        bodies = ", ".join(f"{b.name}" + (f" ({b.role})" if b.role != "body" else "") for b in d.bodies)
        sym = "chord c" if d.kind == "airfoil" else "length L"
        rows = [("Kind", kind), ("Source", _source_text(d)),
                ("Bodies", f"{len(d.bodies)}: {bodies}"),
                ("Overall size", f"{fmt.num(w, 4)} × {fmt.num(h, 4)} m"),
                (f"Reference {sym} (Re)", f"{fmt.num(d.length_ref, 4)} m"),
                ("Force reference", f"{fmt.num(d.force_ref, 4)} m ("
                 + ("chord" if d.kind == "airfoil" else "frontal height" if d.kind == "vehicle" else "reference")
                 + ")"),
                ("Moment reference", f"({fmt.num(d.moment_ref[0], 3)}, {fmt.num(d.moment_ref[1], 3)}) m"),
                ("Cross-section area", f"{fmt.num(sum(abs(_polygon_area(b.points)) for b in d.bodies), 4)} m²")]
        if d.kind == "airfoil" and d.bodies:
            t, xt = _thickness(d.bodies[0].points)
            rows.append(("Max thickness t/c", f"{t * 100:.2f} % at x/c = {xt:.3f}"))
        rows.append(("Outline points", f"{sum(len(b.points) for b in d.bodies):,}"))
        blocks.append(_fig(F.fig_geometry, d, title="", caption=f"{d.name}: outline in the design frame. "
                           "The blue cross marks the moment reference point; U∞ is the free-stream "
                           "direction. Angles of attack rotate the shape about the reference point."))
        blocks.append(KeyValues(rows, columns=2))
        if getattr(d, "notes", ""):
            blocks.append(Paragraph(d.notes, tone="muted"))
    return blocks


def section_setup(entries: Sequence[Entry], ctx: dict) -> list:
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(entries, 1):
        _sub(blocks, entries, e, no, k)
        if e.analysis == "wing":
            blocks += C3.setup_wing(e)
            continue
        if e.dimension == 3:
            blocks += C3.setup_cfd3d(e)
            continue
        c = e.sim.conditions
        fl = c.fluid
        L = e.design.length_ref if e.design is not None else float("nan")
        rows = [("Fluid", fl.name), ("Density ρ", f"{fmt.num(fl.density, 4)} kg/m³"),
                ("Dynamic viscosity μ", f"{_sci(fl.viscosity)} Pa·s"),
                ("Free-stream speed U∞", f"{fmt.num(c.speed, 4)} m/s ({fmt.num(c.speed * 3.6, 4)} km/h)"),
                ("Dynamic pressure q", f"{fmt.num(_q(e), 4)} Pa")]
        if e.analysis == "cfd":
            r = e.result if e.is_cfd else None
            cfd = e.sim.cfd
            rows.insert(0, ("Analysis", "2D incompressible Navier–Stokes (time-accurate)"))
            rows.append(("Angle of attack / pitch", fmt.angle(c.alpha_deg)))
            re_real = r.re_real if r is not None else c.reynolds(L)
            rows.append(("Reynolds number (real)", f"{fmt.reynolds(re_real)}  (based on {_ref_sym(e)})"))
            if r is not None:
                capped = fmt.finite(r.re_sim) and r.re_sim < 0.95 * r.re_real
                rows.append(("Reynolds number (simulated)", fmt.reynolds(r.re_sim) + ("  (capped)" if capped else "")))
                gx, gy = (tuple(r.grid) + (0, 0))[:2]
                rows.append(("Grid", f"{gx:,} × {gy:,} cells, {fmt.num(r.cells_per_ref, 3)} per {_ref_sym(e)}"))
                ext = (r.mean_fields.extent if getattr(r, "mean_fields", None) is not None else None)
                if ext is not None:
                    rows.append(("Domain", (f"{fmt.num(ext[1] - ext[0], 3)} × {fmt.num(ext[3] - ext[2], 3)} m "
                                 f"(x {fmt.num(ext[0], 3)} … {fmt.num(ext[1], 3)} m)")))
                tconv = L / c.speed if c.speed else float("nan")
                rows.append(("Simulated time", (f"{fmt.num(r.t_end, 3)} convective units "
                             f"({fmt.num(r.t_end * tconv, 3)} s)")))
                rows.append(("Averaging window", f"t·U/L = {fmt.num(r.t_avg_start, 3)} … {fmt.num(r.t_end, 3)}"))
                rows.append(("Turbulence treatment", _turbulence_text(r.turbulence, cfd.turbulence)))
                rows.append(("Precision / device", f"{r.precision or cfd.precision} on {r.device or cfd.device}"))
            else:
                rows.append(("Turbulence treatment", _turbulence_text("", cfd.turbulence)))
                rows.append(("Reynolds mode", cfd.reynolds_mode
                             + (f" (cap {fmt.reynolds(cfd.reynolds_cap)})" if cfd.reynolds_cap else "")))
            rows.append(("Quality preset", str(cfd.quality)))
            g = e.sim.ground
            if g.enabled:
                rows.append(("Ground", ("moving belt at U∞" if g.moving else "fixed road")
                             + (", wheels rotating" if g.rotate_wheels else "")))
            elif e.kind == "vehicle":
                rows.append(("Ground", "none (free air)"))
            if getattr(cfd, "compute_sensitivity", False):
                rows.append(("Sensitivity objective", str(cfd.sensitivity_objective)))
        else:
            a = e.sim.airfoil
            pr = e.result if e.is_polar else None
            rows.insert(0, ("Analysis", "Viscous–inviscid airfoil polar (mfoil)"))
            rows.append(("Reynolds number", fmt.reynolds(pr.re if pr is not None else c.reynolds(L))))
            rows.append(("Mach number", fmt.num(pr.mach if pr is not None else a.mach, 3)))
            rows.append(("Angle range", (f"{fmt.angle(a.alpha_start)} … {fmt.angle(a.alpha_end)}, "
                                        f"step {fmt.angle(a.alpha_step)}")))
            rows.append(("Transition", f"e^N, Ncrit = {fmt.num(pr.ncrit if pr is not None else a.ncrit, 3)}"))
            forced = [f"{s} x/c = {v:g}" for s, v in (("upper", a.xtr_upper), ("lower", a.xtr_lower)) if v < 1]
            rows.append(("Forced transition", ", ".join(forced) if forced else "none (free)"))
            rows.append(("Panels", str(a.panels)))
        blocks.append(KeyValues(rows, columns=2))
        notes = list(getattr(e.result, "notes", []) or []) if e.result is not None else []
        if notes:
            blocks.append(Bullets(notes, tone="muted"))
    return blocks


def _no_forces_text(e: Entry, ctx: dict) -> str:
    """Why a CFD run has no force table, and where its results are."""
    closed = getattr(getattr(e.sim, "cfd", None), "domain_mode", "tunnel") == "closed"
    where = [f"{title} (section {ctx['numbers'][key]})" for key, title in (("heat", "heat transfer"),
                                                                          ("fans", "fans and jets"))
             if key in ctx["numbers"] and ((key == "heat" and P.has_thermal(e.result))
                                           or (key == "fans" and P.has_fans(e.result)))]
    text = ("No forces: " + ("the run is a closed box whose walls are the domain, " if closed else "")
            + "there is no solid body to integrate them on.")
    if where:
        text += " Its results are the " + " and the ".join(where) + "."
    return text


def section_results(entries: Sequence[Entry], ctx: dict) -> list:
    cfd = [e for e in entries if e.is_cfd or e.is_cfd3d]
    if not cfd:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(cfd, 1):
        _sub(blocks, cfd, e, no, k)
        if e.result is not None and not CP.has_forces(e):
            blocks.append(Paragraph(_no_forces_text(e, ctx), tone="muted"))
            continue
        if e.is_cfd3d:
            blocks += C3.results_cfd3d(e)
            continue
        r = e.result
        rows = []
        for key, name in (("cd", "Drag Cd"), ("cl", "Lift Cl"), ("cm", "Moment Cm")):
            s = getattr(r, key)
            rel = _rel_err(s)
            rel_txt = fmt.pct(rel * 100, 2, signed=False) if fmt.finite(rel) else fmt.DASH
            rows.append([Cell(name, bold=True), fmt.mean_only(s.mean, s.stderr), _err_text(s),
                         (ROUGH + rel_txt) if _rough(s) and rel_txt != fmt.DASH else rel_txt,
                         fmt.num(s.rms, 3), fmt.num(s.min, 4), fmt.num(s.max, 4)])
        blocks.append(Table(["Coefficient", "Mean", "Std. error", "Rel. error", "RMS fluct.", "Min", "Max"], rows,
                            align=["l", "r", "r", "r", "r", "r", "r"],
                            caption="Time-averaged force coefficients (wind axes)"))
        if any(_rough(getattr(r, k)) for k in ("cd", "cl", "cm")):
            blocks.append(Paragraph(ROUGH_NOTE, tone="muted"))
        ld = r.lift_to_drag if hasattr(r, "lift_to_drag") else float("nan")
        st_txt = (fmt.num(r.strouhal, 3) + " (dominant shedding frequency)") if r.strouhal else \
            "no periodic shedding detected"
        kv = [("Drag D′", f"{fmt.num(r.drag, 4)} N/m"),
              ("Lift L′", f"{fmt.num(r.lift, 4)} N/m" + ("  (negative = downforce)" if e.kind == "vehicle" else "")),
              ("Pitching moment M′", f"{fmt.num(r.moment, 4)} N·m/m (nose-up +)"),
              ("Lift-to-drag L/D", fmt.fixed(ld, 2)),
              (f"Drag area Cd·{'c' if e.kind == 'airfoil' else 'h'}", f"{fmt.num(r.drag_area, 4)} m²/m"),
              ("Power to overcome drag", f"{fmt.num(r.power, 4)} W/m"),
              ("Strouhal number St", st_txt),
              ("Reference", f"q = {fmt.num(_q(e), 4)} Pa, {'c' if e.kind == 'airfoil' else 'h'} = "
                            f"{fmt.num(e.design.force_ref, 4)} m"
               if e.design is not None else "—")]
        blocks.append(KeyValues(kv, columns=2, title="Forces per metre of span"))
        pb = getattr(r, "per_body", {}) or {}
        if len(pb) > 1:
            total = sum(v.get("cd", 0.0) for v in pb.values())
            prow = [[name, fmt.num(v.get("cd"), 4), fmt.num(v.get("cl"), 4), fmt.num(v.get("cm"), 4),
                     fmt.pct(100 * v.get("cd", float("nan")) / total, 1, signed=False) if total else fmt.DASH]
                    for name, v in pb.items()]
            blocks.append(Table(["Body", "Cd", "Cl", "Cm", "Share of drag"], prow, align=["l", "r", "r", "r", "r"],
                                caption="Contribution of each body"))
    if any(e.is_cfd for e in cfd):
        blocks.append(Paragraph(
            ("2D coefficients" if any(e.is_cfd3d for e in cfd) else "Coefficients")
            + " use q = ½ρU² and the force reference length of the design (chord for airfoils, "
            "frontal height for vehicles); moments are about the moment reference point. Two variants whose "
            "means differ by less than about twice their combined standard error cannot be ranked at this "
            "simulation length.", tone="muted"))
    if any(e.is_cfd3d for e in cfd):
        blocks.append(Paragraph(C3.RESULTS_NOTE_3D, tone="muted"))
    return blocks


def section_surface(entries: Sequence[Entry], ctx: dict) -> list:
    cfd = [e for e in entries if (e.is_cfd or e.is_cfd3d) and getattr(e.result, "surface", None)]
    if not cfd:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(cfd, 1):
        _sub(blocks, cfd, e, no, k)
        if e.is_cfd3d:
            blocks += C3.surface_cfd3d(e)
            continue
        kind = "vehicle" if e.kind == "vehicle" else "airfoil"
        has_cf = any(getattr(s, "cf", None) is not None for s in e.result.surface)
        blocks.append(_fig(F.fig_cp, e.result.surface, kind=kind, body_names=_main_bodies(e), show_cf=has_cf,
                           size=(6.7, 4.3 if has_cf else 3.5),
                           caption=f"{e.name}: time-averaged surface pressure coefficient"
                           + (" and skin friction" if has_cf else "")
                           + ". Cp is plotted negative-up (suction up); Cf is positive where the near-wall "
                           "flow follows the free stream and negative in reversed (separated) flow."
                           + (" Wheels are omitted." if kind == "vehicle" else "")))
    return blocks


def section_fields(entries: Sequence[Entry], ctx: dict) -> list:
    cfd = [e for e in entries if (e.is_cfd and getattr(e.result, "mean_fields", None) is not None)
           or (e.is_cfd3d and (getattr(e.result, "slices", None) or getattr(e.result, "volume", None) is not None))]
    if not cfd:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(cfd, 1):
        _sub(blocks, cfd, e, no, k)
        if e.is_cfd3d:
            blocks += C3.fields_cfd3d(e)
            continue
        r = e.result
        bodies = r.bodies_wind or (e.design.bodies if e.design is not None else None)
        g = e.ground
        blocks.append(_fig(F.fig_field, r.mean_fields, "speed", bodies, streamlines=True, ground=g, q_inf=_q(e),
                           caption=f"{e.name}: time-averaged velocity magnitude with streamlines of the mean "
                           "flow (wind frame; the geometry is shown at its simulated attitude)."))
        blocks.append(_fig(F.fig_field, r.mean_fields, "cp", bodies, ground=g, q_inf=_q(e),
                           caption="Time-averaged pressure coefficient. Red: over-pressure (stagnation); "
                           "blue: suction."))
        inst = getattr(r, "instant_fields", None)
        src, when = (inst, "Instantaneous") if inst is not None else (r.mean_fields, "Time-averaged")
        blocks.append(_fig(F.fig_field, src, "vorticity", bodies, ground=g, q_inf=_q(e),
                           caption=f"{when} vorticity: red counter-clockwise, blue clockwise rotation. "
                           "Shear layers and the shed vortices of the wake show where energy is lost."))
    return blocks


def section_convergence(entries: Sequence[Entry], ctx: dict) -> list:
    cfd = [e for e in entries if (e.is_cfd or e.is_cfd3d) and getattr(e.result, "history", None)]
    if not cfd:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(cfd, 1):
        _sub(blocks, cfd, e, no, k)
        if not CP.has_forces(e):
            blocks += CP.monitor_block(e)
            blocks += CP.diagnostics_block(e)
            continue
        if e.is_cfd3d:
            blocks += C3.convergence_cfd3d(e)
            blocks += CP.diagnostics_block(e)
            continue
        r = e.result
        t = np.asarray(r.history.get("t", []), float)
        n_win = int(np.sum(t >= r.t_avg_start)) if t.size and fmt.finite(r.t_avg_start) else 0
        blocks.append(_fig(F.fig_history, r.history, r.t_avg_start, r,
                           caption=f"{e.name}: force-coefficient histories. The shaded region is the averaging "
                           "window; the black line is the mean and its grey band ±1 standard error."))
        rows = [("Simulated time", f"{fmt.num(r.t_end, 3)} convective units"),
                ("Transient discarded", f"t·U/L < {fmt.num(r.t_avg_start, 3)}"),
                ("Averaging window", f"{fmt.num(r.t_end - r.t_avg_start, 3)} convective units, {n_win:,} samples"),
                ("Converged", fmt.yes_no(r.converged)),
                ("Relative s.e. of Cd", fmt.pct(_rel_err(r.cd) * 100, 2, signed=False)),
                ("Relative s.e. of Cl", fmt.pct(_rel_err(r.cl) * 100, 2, signed=False))]
        if r.strouhal:
            # St is based on its own reference length (SimPlan.strouhal_length: the chord, a
            # car's force reference, or the frontal height of a pitched bluff body) while time
            # is in L/U with L the Reynolds length, so the window holds St * T * L / st_len
            # cycles. Results saved before st_len was stored fall back to the force reference,
            # which is right for airfoils and cars but under-counts a pitched flat plate 2.8x.
            ratio = 1.0
            st_len = getattr(r, "strouhal_length", float("nan"))
            if e.design is not None:
                with contextlib.suppress(TypeError, ValueError, ZeroDivisionError):
                    ref = float(st_len) if fmt.finite(st_len) and st_len > 0 else float(e.design.force_ref)
                    ratio = float(e.design.length_ref) / ref
            cycles = r.strouhal * (r.t_end - r.t_avg_start) * ratio
            rows.append(("Shedding cycles averaged", f"≈ {cycles:.0f} (St = {fmt.num(r.strouhal, 3)})"))
        blocks.append(KeyValues(rows, columns=2))
        blocks += CP.diagnostics_block(e)
    blocks.append(Paragraph(
        "The start-up transient is discarded and forces are averaged over the remaining window. Because "
        "successive samples are correlated, the standard error is estimated with batch means rather than "
        "the naive σ/√N; it shrinks roughly with the square root of the window length, so halving the "
        "uncertainty needs about four times the averaging time.", tone="muted"))
    return blocks


def _hotspots(surf: Any, objective: str, ref: str = "c", n: int = 3) -> list[str]:
    """Up to ``n`` well-separated surface locations with the strongest sensitivity, in words."""
    s = np.asarray(surf.sensitivity, float)
    x, y = np.asarray(surf.x, float), np.asarray(surf.y, float)
    nx, ny = np.asarray(surf.nx, float), np.asarray(surf.ny, float)
    ds = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
    arc = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    per = float(ds.sum()) or 1.0
    order = np.argsort(-np.abs(np.nan_to_num(s)))
    picked: list[int] = []
    for i in order:
        if len(picked) >= n or not np.isfinite(s[i]) or s[i] == 0:
            break
        if all(min(abs(arc[i] - arc[j]), per - abs(arc[i] - arc[j])) > 0.08 * per for j in picked):
            picked.append(int(i))
    sym, sense = F.OBJECTIVES.get(objective, (objective, "reduce"))
    xr = (x - x.min()) / (np.ptp(x) or 1.0)
    fin = np.abs(s[np.isfinite(s)])
    ref98 = float(np.percentile(fin, 98)) if fin.size else 0.0
    out = []
    for i in picked:
        where = ("front face" if nx[i] < -0.75 else "rear face" if nx[i] > 0.75 else
                 "upper surface" if ny[i] > 0 else "lower surface")
        grow_helps = (s[i] < 0) == (sense == "reduce")
        move = "outward (add material)" if grow_helps else "inward (remove material)"
        # relative strength, not 1/m: the map's size is only indicative (section_sensitivity)
        strength = f" (relative strength {abs(s[i]) / ref98:.2f})" if ref98 > 0 else ""
        out.append(f"{surf.body}, {where} at x/{ref} ≈ {xr[i]:.2f}: moving the surface {move} "
                   f"{sense}s {sym}{strength}.")
    return out


def section_sensitivity(entries: Sequence[Entry], ctx: dict) -> list:
    sel = [e for e in entries if (e.is_cfd or e.is_cfd3d) and any(getattr(s, "sensitivity", None) is not None
                                                                 for s in (e.result.surface or []))]
    if not sel:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(sel, 1):
        _sub(blocks, sel, e, no, k)
        if e.is_cfd3d:
            blocks += C3.sensitivity_cfd3d(e)
            continue
        surfs = [s for s in e.result.surface if s.sensitivity is not None]
        allsurf = list(e.result.surface)            # wheels etc. drawn for context, uncoloured
        obj = str(getattr(e.sim.cfd, "sensitivity_objective", "cd"))
        sym, sense = F.OBJECTIVES.get(obj, (obj, "reduce"))
        blocks.append(_fig(F.fig_sensitivity, allsurf, objective=obj, ground=e.ground,
                           caption=f"{e.name}: sensitivity of {sym} to a small outward displacement of the "
                           f"surface. Arrows show the shape change that would {sense} {sym}; their length "
                           "is proportional to the local sensitivity."))
        hs = [h for s in surfs for h in _hotspots(s, obj, _ref_sym(e))][:4]
        if hs:
            blocks.append(Paragraph("Most effective places to modify the shape:", tone="body"))
            blocks.append(Bullets(hs))
    if any(e.is_cfd for e in sel):
        blocks.append(Paragraph(
            ("For the 2D runs, the" if any(e.is_cfd3d for e in sel) else "The")
            + " sensitivity is the derivative of the objective with respect to a normal displacement of each "
            "surface point, obtained by automatic differentiation through the solver. It is a local, linear "
            "guide for small changes (a few per cent of the local thickness); re-simulate any modified shape "
            "to confirm the gain. Read the pattern, not the size: the magnitude can be about twice the "
            "physical derivative (the finite averaging window adds about 20 % at the default window, and the "
            "immersed boundary responds more strongly to sub-cell shape changes than to whole-cell ones), so "
            "the map is shown relative to its strongest values.", tone="muted"))
    if any(e.is_cfd3d for e in sel):
        blocks.append(Paragraph(C3.SENSITIVITY_NOTE_3D, tone="muted"))
    return blocks


def section_polar(entries: Sequence[Entry], ctx: dict) -> list:
    pol = [e for e in entries if e.is_polar or e.is_wing]
    if not pol:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(pol, 1):
        _sub(blocks, pol, e, no, k)
        if e.is_wing:
            blocks += C3.wing_blocks(e)
            continue
        r = e.result
        blocks.append(_fig(F.fig_polar, r, labels=[e.name], size=(6.7, 4.9),
                           caption=f"{e.name}: lift curve, drag polar, pitching moment about the quarter "
                           f"chord and lift-to-drag ratio at Re = {fmt.reynolds(r.re)}. The ringed point is "
                           "the best L/D."))
        m = _polar_metrics(r)
        if m:
            blocks.append(KeyValues([
                ("Maximum lift Cl,max", f"{_cl_max_text(m)} {_cl_max_where(m)}"),
                ("Best L/D", f"{fmt.fixed(m.get('ld_max'), 1)} at α = {fmt.angle(m.get('alpha_ld_max'))}"),
                ("Minimum drag Cd,min", f"{fmt.num(m.get('cd_min'), 3)} at Cl = {fmt.num(m.get('cl_at_cd_min'), 3)}"),
                ("Lift slope", f"{fmt.num(m.get('lift_slope_per_rad'), 3)} /rad (thin-airfoil theory: 2π)"),
                ("Zero-lift angle α0", f"{fmt.num(m.get('alpha_zero_lift'), 3)}°"),
                ("Zero-lift moment Cm0", fmt.num(m.get("cm0"), 3)),
            ], columns=2))
        rows = []
        for p in sorted(r.points, key=lambda p: p.alpha):
            ld = p.cl / p.cd if (fmt.finite(p.cd) and p.cd > 0 and fmt.finite(p.cl)) else float("nan")
            tone = "neutral" if p.converged else "warn"
            rows.append([Cell(fmt.angle(p.alpha), tone), fmt.fixed(p.cl, 4), fmt.fixed(p.cd, 5),
                         fmt.fixed(p.cdp, 5), fmt.fixed(p.cdf, 5), fmt.fixed(p.cm, 4), fmt.fixed(ld, 1),
                         fmt.fixed(p.xtr_upper, 3), fmt.fixed(p.xtr_lower, 3),
                         Cell("yes" if p.converged else "no", "neutral" if p.converged else "warn")])
        blocks.append(Table(["α", "Cl", "Cd", "Cdp", "Cdf", "Cm", "L/D", "xtr up", "xtr low", "Conv."], rows,
                            align=["r"] * 9 + ["c"], caption=f"{e.name}: polar points "
                            "(Cdp pressure drag, Cdf friction drag, xtr transition position x/c)",
                            font_scale=0.92))
        # Cp at a few characteristic angles
        dist = [p for p in r.points if p.converged and p.cp is not None and p.x is not None]
        if dist and m:
            want = [0.0, m.get("alpha_ld_max"), m.get("alpha_cl_max")]
            chosen: list[Any] = []
            for a in want:
                if not fmt.finite(a):
                    continue
                p = min(dist, key=lambda q: abs(q.alpha - a))
                if all(p is not c for c in chosen):
                    chosen.append(p)
            if chosen:
                blocks.append(_fig(F.fig_cp, chosen, labels=[f"α = {p.alpha:g}°" for p in chosen],
                                   size=(6.7, 3.9),
                                   caption=f"{e.name}: pressure distributions at α = "
                                   + ", ".join(f"{p.alpha:g}°" for p in chosen)
                                   + " (solid upper, dashed lower surface; dotted: inviscid)."))
        for w in getattr(r, "warnings", []) or []:
            blocks.append(Callout(fmt.solver_text(w), "warn"))
    return blocks


def section_comparison(entries: Sequence[Entry], ctx: dict) -> list:
    # a CFD group is compared only when two of its runs share something to compare: forces, heat
    # transfer or fans (a heated cavity beside a fan room has none in common)
    cfd = [e for e in entries if e.is_cfd]
    cfd = cfd if CP.comparable(cfd) else []
    pol = [e for e in entries if e.is_polar]
    cfd3 = [e for e in entries if e.is_cfd3d]
    cfd3 = cfd3 if CP.comparable(cfd3) else []
    wings = [e for e in entries if e.is_wing]
    groups = [g for g in (cfd, pol, cfd3, wings) if len(g) >= 2]
    if not groups:
        return []
    blocks: list = []
    no = ctx["no"]
    sub = 0
    multi = len(groups) > 1

    def head(text: str) -> None:
        nonlocal sub
        sub += 1
        blocks.append(Heading(text, level=2, number=f"{no}.{sub}"))
    dims = {e.dimension for e in entries if e.result is not None}
    if len(dims) > 1:
        blocks.append(Paragraph(
            "This report covers 2D and 3D results. 2D coefficients are per unit span on a reference length and "
            "3D coefficients are on a reference area, so results are compared only within their own group: "
            "2D CFD with 2D CFD, airfoil polars with polars, 3D CFD with 3D CFD and wing sweeps with wing "
            "sweeps.", tone="muted"))
    if len(cfd) >= 2:
        if multi:
            head("2D CFD results" if cfd3 else "CFD results")
        res_all = {e.sim_id: e.result for e in cfd}
        # the force comparison covers the runs that have forces (a heated cavity or a fan room
        # has none: a row of dashes, and a baseline without a drag would give no Δ at all)
        cfd_f = [e for e in cfd if CP.has_forces(e)]
        res = {e.sim_id: e.result for e in cfd_f}
        rows_cmp = (_compare_rows(res) or []) if len(cfd_f) >= 2 else []
        by_key = {r["key"]: r for r in rows_cmp}
        lead_name = (cfd_f[0] if len(cfd_f) >= 2 else cfd[0]).name
        blocks.append(Paragraph(
            f"Baseline: **{lead_name}**. Δ is the change relative to the baseline. Green/red mark an "
            "improvement/deterioration that is larger than twice the combined standard error (✱, "
            "statistically significant); ~ marks a difference within the statistical noise."))
        kinds = {e.kind for e in cfd_f}
        if len(kinds) > 1:
            blocks.append(Callout("This comparison mixes design kinds (" + ", ".join(sorted(kinds)) + "). "
                                  "Their coefficients use different reference lengths (chord vs frontal "
                                  "height), so compare forces rather than coefficients across kinds.", "warn"))
        # the Compare page shows the same sentences (analysis.comparison_warnings)
        for msg in _comparison_warnings(res_all, {e.sim_id: e.sim for e in cfd}):
            blocks.append(Callout(msg, "warn"))
        rows = []
        for i, e in enumerate(cfd_f if len(cfd_f) >= 2 else []):
            r = e.result

            def cell(key: str, i: int = i) -> dict | None:
                row = by_key.get(key)
                return row["cells"][i] if row else None
            rows.append([Cell(e.name, bold=(i == 0)),
                         fmt.num(e.sim.conditions.speed, 3), fmt.angle(e.sim.conditions.alpha_deg),
                         _pm_text(r.cd), _delta_cell(cell("cd"), "lower"),
                         _pm_text(r.cl), _delta_cell(cell("cl"), None),
                         fmt.fixed(r.lift_to_drag, 1), _delta_cell(cell("ld"), "higher"),
                         fmt.num(r.drag, 3)])
        if rows:
            blocks.append(Table(["Simulation", "U [m/s]", "α", "Cd", "ΔCd", "Cl", "ΔCl", "L/D", "ΔL/D", "D′ [N/m]"],
                                rows, align=["l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
                                caption="CFD comparison (± one standard error; Δ relative to the baseline)",
                                font_scale=0.86))
            if any(_rough(e.result.cd) or _rough(e.result.cl) for e in cfd_f):
                blocks.append(Paragraph(ROUGH_NOTE + " Differences against such a value are not marked "
                                        "significant.", tone="muted"))
            bar_rows = [{"label": e.name, "cd": e.result.cd, "cl": e.result.cl} for e in cfd_f]
            blocks.append(_fig(F.fig_compare_bars, bar_rows, ["cd", "cl"],
                               caption="Drag and lift coefficients with ±1 standard-error bars; percentages are "
                               "relative to the baseline (grey)."))
        blocks += CP.physics_comparison(cfd)
        same_kind = {e.kind for e in cfd_f}
        surf = [(e, e.result.surface) for e in cfd_f if getattr(e.result, "surface", None)]
        if len(surf) >= 2 and len(same_kind) == 1:
            items, labels = [], []
            for e, ss in surf:
                main = _main_bodies(e)
                pick = [s for s in ss if main is None or s.body in main]
                if pick:
                    items.append(pick[0])
                    labels.append(e.name)
            if len(items) >= 2:
                blocks.append(_fig(F.fig_cp, items, labels=labels, kind="vehicle" if "vehicle" in same_kind
                                   else "airfoil", size=(6.7, 3.9),
                                   caption="Surface pressure distributions overlaid (solid upper, dashed lower "
                                   "surface)."))
    if len(pol) >= 2:
        if multi:
            head("Airfoil polars")
        blocks.append(_fig(F.fig_polar, [e.result for e in pol], labels=[e.name for e in pol], size=(6.7, 5.4),
                           caption="Polars overlaid. Rings mark each polar's best L/D."))
        ms = [_polar_metrics(e.result) for e in pol]
        base = ms[0]

        def dcell(m: dict, key: str, better: str, i: int, digits: int = 3) -> Cell:
            v, b = m.get(key), base.get(key)
            if key == "cl_max" and (_stall_missed(m) or _stall_missed(base)):
                # an end-of-sweep value is a lower bound: no change, no colour
                return Cell(_cl_max_text(m, digits))
            if i == 0 or not (fmt.finite(v) and fmt.finite(b) and b):
                return Cell(fmt.num(v, digits))
            d = (v - b) / abs(b) * 100
            if abs(d) < 0.05:
                return Cell(f"{fmt.num(v, digits)} (0.0 %)")
            good = d < 0 if better == "lower" else d > 0
            return Cell(f"{fmt.num(v, digits)} ({fmt.pct(d)})", "ok" if good else "bad")
        rows = []
        for i, (e, m) in enumerate(zip(pol, ms, strict=True)):
            rows.append([Cell(e.name, bold=(i == 0)), fmt.reynolds(e.result.re),
                         dcell(m, "cl_max", "higher", i), fmt.angle(m.get("alpha_cl_max")),
                         dcell(m, "cd_min", "lower", i), dcell(m, "ld_max", "higher", i, 4),
                         fmt.angle(m.get("alpha_ld_max")), fmt.num(m.get("alpha_zero_lift"), 3),
                         fmt.num(m.get("cm0"), 3)])
        blocks.append(Table(["Polar", "Re", "Cl,max", "α", "Cd,min", "(L/D)max", "α", "α0 [°]", "Cm0"], rows,
                            align=["l", "r", "r", "r", "r", "r", "r", "r", "r"],
                            caption="Polar comparison (Δ relative to the first polar)",
                            widths=[2.4, 1.3, 1.6, 0.7, 1.6, 1.6, 0.7, 0.9, 0.9], font_scale=0.9))
    if len(cfd3) >= 2:
        if multi:
            head("3D CFD results")
        blocks += C3.compare_cfd3d(cfd3)
    if len(wings) >= 2:
        if multi:
            head("Wing polars")
        blocks += C3.compare_wings(wings)
    return blocks


REFERENCES = [
    ("Chorin, A. J. (1968). Numerical solution of the Navier–Stokes equations. Mathematics of Computation 22, "
    "745–762."),
    "Peskin, C. S. (2002). The immersed boundary method. Acta Numerica 11, 479–517.",
    "Mittal, R. & Iaccarino, G. (2005). Immersed boundary methods. Annual Review of Fluid Mechanics 37, 239–261.",
    ("Nicoud, F. & Ducros, F. (1999). Subgrid-scale stress modelling based on the square of the velocity "
    "gradient tensor. Flow, Turbulence and Combustion 62, 183–200."),
    ("Vreman, A. W. (2004). An eddy-viscosity subgrid-scale model for turbulent shear flow. Physics of "
    "Fluids 16, 3670–3681."),
    ("Schäfer, M. & Turek, S. (1996). Benchmark computations of laminar flow around a cylinder. Notes on "
    "Numerical Fluid Mechanics 52, 547–566."),
    ("Williamson, C. H. K. (1996). Vortex dynamics in the cylinder wake. Annual Review of Fluid Mechanics 28, "
    "477–539."),
    ("Flyvbjerg, H. & Petersen, H. G. (1989). Error estimates on averages of correlated data. Journal of "
    "Chemical Physics 91, 461–466."),
    ("Drela, M. & Giles, M. B. (1987). Viscous-inviscid analysis of transonic and low Reynolds number airfoils. "
    "AIAA Journal 25(10), 1347–1355."),
    ("Fidkowski, K. J. mfoil: a viscous-inviscid airfoil analysis code in Python (MIT licence), University "
    "of Michigan."),
]


def section_methodology(entries: Sequence[Entry], ctx: dict) -> list:
    blocks: list = []
    no = ctx["no"]
    sub = 0
    cfd = [e for e in entries if e.analysis == "cfd" and e.dimension == 2]
    cfd3 = [e for e in entries if e.analysis == "cfd" and e.dimension == 3]
    pol = [e for e in entries if e.analysis == "polar"]
    wings = [e for e in entries if e.analysis == "wing"]

    def h(text: str) -> None:
        nonlocal sub
        sub += 1
        blocks.append(Heading(text, level=2, number=f"{no}.{sub}"))
    if cfd:
        h("2D Navier–Stokes solver" if cfd3 else "Navier–Stokes solver")
        blocks.append(Paragraph(
            "Aerolume solves the two-dimensional, incompressible, time-dependent Navier–Stokes equations on a "
            "uniform Cartesian grid, written in PyTorch so the same code runs on a GPU or a CPU. Bodies are "
            "represented with an immersed-boundary (solid-fraction mask) method, so any outline can be "
            "simulated without generating a body-fitted mesh. Each time step advects the velocity with an "
            "upwind-biased scheme, adds viscous diffusion and then projects the velocity onto a divergence-free "
            "field by solving a pressure Poisson equation (Chorin's projection method). Forces on each body are "
            "integrated every step; the reported values are averages over the window after the start-up "
            "transient. Because the solver is differentiable, the same run can also return the sensitivity of "
            "a force to the shape (section on shape sensitivity)."))
        les = sorted({str(e.result.turbulence) for e in cfd if e.is_cfd and str(e.result.turbulence).startswith("les")})
        if les:
            blocks.append(Paragraph(
                "Turbulence: runs at simulated Reynolds numbers of a few thousand and above are large-eddy "
                f"simulations ({', '.join(_turbulence_text(x) for x in les)}): the large vortices are "
                "resolved and the effect of smaller scales is modelled. Turbulence is physically "
                "three-dimensional; a 2D simulation captures the large, mostly two-dimensional shedding that "
                "dominates bluff-body and stalled-flow forces, but not 3D turbulent mixing. Treat absolute "
                "numbers as indicative and the differences between variants as the primary result."))
    C3.methodology_cfd3d(cfd3, h, blocks)
    if cfd or cfd3:
        scaled = [e for e in cfd + cfd3 if (e.is_cfd or e.is_cfd3d) and fmt.finite(e.result.re_sim)
                  and e.result.re_sim < 0.95 * e.result.re_real]
        if scaled:
            h("Reynolds number")
            blocks.append(Paragraph(
                "Resolving the boundary layers of a full-size wing or vehicle at Re = 10⁶–10⁷ needs far more "
                "cells than an interactive simulation can afford. Aerolume therefore may cap the simulated "
                "Reynolds number (Re_sim) while keeping the real geometry, speed and fluid for converting "
                "coefficients to forces. Pressure-dominated quantities (bluff-body drag, lift before stall, "
                "pressure distributions) transfer reasonably well; viscous quantities (skin-friction drag, "
                "transition and separation points, maximum lift) depend on Re and are not full-scale "
                "predictions."))
            rows = [[e.name, fmt.reynolds(e.result.re_real), fmt.reynolds(e.result.re_sim),
                     f"1 : {e.result.re_real / e.result.re_sim:,.0f}"] for e in scaled]
            blocks.append(Table(["Simulation", "Re real", "Re simulated", "Scale"], rows, align=["l", "r", "r", "r"],
                                caption="Real and simulated Reynolds numbers"))
        h("Statistical uncertainty")
        blocks.append(Paragraph(
            "Unsteady flows never settle to a single value, so forces are time-averaged. The ± values in this "
            "report are one standard error of that average (about 68 % confidence; double it for about 95 %), "
            "estimated with the batch-means method, which accounts for the correlation between successive "
            "samples (Flyvbjerg & Petersen, 1989). The standard error measures only the sampling noise of the "
            "average; it does not include modelling or discretisation error."))
    if pol:
        h("Airfoil analysis")
        blocks.append(Paragraph(
            "Polars are computed with mfoil (K. Fidkowski), a viscous–inviscid interaction method of the XFOIL "
            "family (Drela & Giles, 1987): a panel method for the inviscid flow coupled to integral "
            "boundary-layer equations with e^N transition prediction. It is fast and accurate for attached and "
            "mildly separated flow at Reynolds numbers of roughly 10⁵–10⁷. Near and beyond stall the method "
            "loses accuracy and may not converge; such points are flagged in the tables and left out of the "
            "plots."))
    C3.methodology_wing(wings, h, blocks)
    extra_refs = CP.methodology_physics(entries, h, blocks)
    refs = list(dict.fromkeys([*_relevant_refs(cfd, pol, cfd3, wings), *extra_refs]))
    if not (cfd or pol):
        h("References")
        blocks.append(Bullets([f"[{i}] {r}" for i, r in enumerate(refs, 1)], tone="small"))
        return blocks
    h("Validity and limits (2D)" if (cfd3 or wings) else "Validity and limits")
    items = [("**Two-dimensional:** forces are per metre of span for an infinitely long section. Tip vortices, "
             "induced drag and three-dimensional separation are not represented.")]
    if any(e.kind == "vehicle" and e.dimension == 2 for e in entries):
        items.append("**Vehicles:** a side profile ignores the flow around the sides and the 3D wake. The drag "
                     "of a 2D profile is a comparative indicator for design changes, not the 3D Cd·A of the car.")
    if cfd:
        items.append("**Benchmarks:** the Navier–Stokes solver is validated against laminar reference cases "
                     "(circular cylinder Re 20–100, Schäfer–Turek, NACA 0012 at Re 1 000); see "
                     "docs/validation.md for the current comparison. Results at higher Re_sim rely on the LES "
                     "model and are outside that benchmark range.")
        items.append("**Compressibility:** the flow is incompressible, valid below about Mach 0.3 "
                     "(≈ 100 m/s in sea-level air).")
        items.append("**Resolution:** the grid resolves the boundary layer only at moderate Re_sim; thin "
                     "high-Re boundary layers are under-resolved and friction drag is approximate.")
    blocks.append(Bullets(items))
    h("References")
    blocks.append(Bullets([f"[{i}] {r}" for i, r in enumerate(refs, 1)], tone="small"))
    return blocks


def _relevant_refs(cfd: Sequence[Entry], pol: Sequence[Entry], cfd3: Sequence[Entry] = (),
                   wings: Sequence[Entry] = ()) -> list[str]:
    out: list[str] = []
    if cfd:
        out += REFERENCES[:8]
    if cfd3:
        out += [*C3.REFERENCES_3D, REFERENCES[7]]       # + batch-means standard errors
    if pol or wings:
        out += REFERENCES[8:]                           # the 2D section polars of a viscous wing sweep
    if wings:
        out += C3.REFERENCES_WING
    return list(dict.fromkeys(out))


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except Exception:  # noqa: BLE001 - not installed / unknown distribution name
        return "not installed"


def section_appendix(entries: Sequence[Entry], ctx: dict) -> list:
    blocks: list = [KeyValues([
        ("Aerolume", __version__), ("Python", platform.python_version()),
        ("NumPy", _version("numpy")), ("matplotlib", _version("matplotlib")),
        ("PyTorch", _version("torch")), ("VTK (3D rendering)", _version("vtk")),
        ("Operating system", f"{platform.system()} {platform.release()}"),
        ("Report generated", ctx["generated"]), ("Report file", ctx.get("filename", "—")),
    ], columns=2, title="Software")]
    rows = []
    for e in entries:
        r = e.result
        cfd_like = e.is_cfd or e.is_cfd3d
        dev = (getattr(r, "device", "") if cfd_like else "CPU (mfoil)" if e.is_polar else
               "CPU (lattice + mfoil)" if e.is_wing else "—")
        prec = getattr(r, "precision", "") if cfd_like else "float64" if (e.is_polar or e.is_wing) else "—"
        rows.append([e.name, e.sim_id, e.type_label,
                     str(getattr(e.sim, "status", "")), dev or "—", prec or "—",
                     fmt.seconds(getattr(r, "wall_seconds", float("nan"))) if r is not None else "—",
                     str(getattr(e.sim, "finished", "") or "—").replace("T", " ")])
    blocks.append(Table(["Simulation", "Id", "Type", "Status", "Device", "Precision", "Run time", "Finished"], rows,
                        align=["l", "l", "l", "l", "l", "l", "r", "l"], caption="Run details",
                        widths=[2.2, 1.3, 0.8, 0.8, 2.4, 0.95, 1.0, 1.6], font_scale=0.86))
    blocks.append(Table(["Simulation", "Configuration hash", "Solver-settings hash", "Random seed", "Computed with"],
                        [CP.provenance_rows(e) for e in entries], align=["l", "l", "l", "l", "l"],
                        caption="Provenance: the configuration hash identifies the design geometry and settings, "
                                "the solver-settings hash the planner's resolved grid and solver configuration "
                                "(SHA-256, first 12 hex digits)",
                        widths=[2.4, 1.7, 1.7, 1.0, 2.6], font_scale=0.86))
    blocks.append(Paragraph(
        "The full provenance record of each result (software versions, settings as run, hashes, device) is saved in "
        "its result file; Project.export_reproduction_script(sim_id) writes a Python script that rebuilds and re-runs "
        "the simulation and checks the numbers against this report.", tone="muted"))
    return blocks


BUILDERS = {
    "summary": section_summary, "geometry": section_geometry, "setup": section_setup,
    "results": section_results, "surface": section_surface, "fields": section_fields,
    "convergence": section_convergence, "sensitivity": section_sensitivity, "polar": section_polar,
    "comparison": section_comparison, "methodology": section_methodology,
    "heat": CP.section_heat, "flowvis": CP.section_flowvis, "fans": CP.section_fans, "grid": CP.section_grid,
}

_NOT_APPLICABLE = {
    "results": "None of the selected simulations has a CFD result (2D or 3D).",
    "geometry": "The designs of the selected simulations are not available.",
    "surface": "No surface distributions are available for the selected simulations.",
    "fields": "No flow fields were stored for the selected simulations.",
    "convergence": "No force histories are available for the selected simulations.",
    "sensitivity": "No shape sensitivity was computed for the selected simulations. Enable "
                   "“compute sensitivity” in the CFD settings to include it.",
    "polar": "None of the selected simulations is an airfoil polar or a wing sweep.",
    "heat": "None of the selected simulations has heat transfer switched on.",
    "flowvis": "None of the selected simulations has smoke or particle emitters.",
    "fans": "None of the selected simulations has fans, jets or vents.",
    "grid": ("No grid-convergence study among the selected simulations: it needs three or more runs of the same "
             "set-up at different resolutions (cells per reference length)."),
    "comparison": ("A comparison needs at least two finished simulations of the same type (2D CFD, airfoil "
                   "polar, 3D CFD or wing sweep)."),
}


def _has_geometry(design: Any) -> bool:
    """Whether a design has a body to draw: 2D outlines, 3D meshes or a wing planform."""
    if int(getattr(design, "dimension", 2) or 2) == 3:
        return bool(getattr(design, "meshes", None)) or getattr(design, "wing", None) is not None
    return bool(getattr(design, "bodies", None))


def _polar_title(entries: Sequence[Entry]) -> str:
    """The polar section's title for what it holds: airfoil polars, wing polars or both."""
    pol, wing = any(e.is_polar for e in entries), any(e.is_wing for e in entries)
    if pol and wing:
        return "Airfoil and wing polars"
    return "Wing polars" if wing else "Airfoil polars"


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

def build_document(project: Any, sim_ids: Sequence[str], *, sections: Iterable[str] | None = None,
                   title: str | None = None, author: str | None = None, filename: str = "") -> Document:
    """Assemble the report :class:`Document` for ``sim_ids`` of ``project``.

    ``sections=None`` includes every applicable section of :data:`SECTIONS` (sections without
    content are omitted); an explicit list keeps the requested ones in canonical order and shows
    a short "not applicable" note for those without content. The appendix (software versions,
    devices, run times) follows the methodology section whenever that is included.
    """
    ids = list(dict.fromkeys(sim_ids))
    if not ids:
        raise ValueError("a report needs at least one simulation id")
    explicit = sections is not None
    wanted = list(SECTIONS) if sections is None else list(dict.fromkeys(sections))
    bad = [s for s in wanted if s not in SECTIONS]
    if bad:
        raise ValueError(f"unknown report section(s) {bad}; valid: {', '.join(SECTIONS)}")
    order = [s for s in SECTIONS if s in wanted]
    entries = load_entries(project, ids)
    pname = str(getattr(project, "name", "Untitled project"))
    generated = time.strftime("%Y-%m-%d %H:%M")
    ctx: dict[str, Any] = {"project_name": pname, "generated": generated, "filename": filename, "numbers": {},
                           "project": project}

    # Build every section first, then number only the non-empty ones, then fix cross-references
    # (the summary points at the methodology section number).
    raw: list[tuple[str, list]] = []
    for key in order:
        ctx["no"] = "#"
        blocks = BUILDERS[key](entries, ctx)
        if not blocks and explicit:
            blocks = [Paragraph(_NOT_APPLICABLE.get(key, "Not applicable to the selected simulations."),
                                tone="muted")]
        if blocks:
            raw.append((key, blocks))
    numbers = {key: str(i) for i, (key, _) in enumerate(raw, 1)}
    ctx["numbers"] = numbers
    doc_sections = []
    for key, _ in raw:
        ctx["no"] = numbers[key]
        blocks = BUILDERS[key](entries, ctx)    # rebuild with final numbers (cheap: figures are lazy)
        if not blocks:
            blocks = [Paragraph(_NOT_APPLICABLE.get(key, "Not applicable."), tone="muted")]
        title_ = _polar_title(entries) if key == "polar" else SECTION_TITLES[key]
        doc_sections.append(Section(key, title_, numbers[key], blocks))
    if "methodology" in order:
        ctx["no"] = "A"
        doc_sections.append(Section("appendix", SECTION_TITLES["appendix"], "A", section_appendix(entries, ctx)))

    designs: list[Any] = []
    for e in entries:
        if e.design is not None and all(d is not e.design for d in designs):
            designs.append(e.design)
    sim_names = [e.name for e in entries]
    meta = [("Project", pname),
            ("Design" + ("s" if len(designs) != 1 else ""), ", ".join(d.name for d in designs) or "—"),
            ("Simulation" + ("s" if len(entries) != 1 else ""),
             ", ".join(sim_names[:6]) + (f" and {len(sim_names) - 6} more" if len(sim_names) > 6 else "")),
            ("Author", author or "—"), ("Date", generated), ("Software", f"Aerolume {__version__} (Apache-2.0)")]
    cover = None
    # the first design with something to draw: a heated cavity or a fan room has no body (its walls
    # are the domain), and a cover picture of nothing -- or an error in its place -- is worse than none
    drawable = [d for d in designs if _has_geometry(d)]
    if drawable and int(getattr(drawable[0], "dimension", 2) or 2) == 3:
        cover = FigureBlock(functools.partial(C3.figure_cover3d, drawable[0],
                                              airfoil_coords=C3._airfoil_coords(project)), caption="", width=1.0)
    elif drawable:
        cover = FigureBlock(functools.partial(F.fig_geometry, drawable[0], show_reference=False, title=""),
                            caption="", width=1.0)
    meth = numbers.get("methodology")
    dims = {e.dimension for e in entries}
    if dims == {3}:
        scope = ("Results are three-dimensional numerical predictions: forces on the whole body, coefficients "
                 "on the stated reference area. ")
    elif 3 in dims:
        scope = ("Results are numerical predictions; 2D results are per metre of span, 3D results are for the "
                 "whole body. ")
    else:
        scope = "Results are two-dimensional numerical predictions per metre of span. "
    doc = Document(
        title=title or "Aerodynamic analysis report",
        subtitle=pname,
        author=author or "",
        date=generated,
        software=f"Aerolume {__version__}",
        meta=meta,
        sections=doc_sections,
        cover_figure=cover,
        disclaimer=(scope
                    + (f"Section {meth} describes the method, its validated range and the uncertainty "
                       "of every number." if meth else
                       "Interpret them together with the stated Reynolds number and uncertainty.")),
    )
    doc.number_floats()
    return doc
