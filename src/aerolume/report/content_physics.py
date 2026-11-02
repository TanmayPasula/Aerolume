"""Report sections for the placed physics and the research-grade checks (DESIGN.md section 10).

Four sections, each written only when the selected simulations have something to show:

* **Heat transfer** ("heat"): the thermal set-up (ambient temperature, buoyancy, body,
  wall and heater conditions, the non-dimensional groups the planner chose), the Nusselt
  number and heat rate with their standard errors, the heat balance (energy conservation
  of the run) and the temperature field;
* **Flow visualisation** ("flowvis"): the emitters and the smoke they released, one
  colour per smoke channel (2D composite, 3D volume rendering);
* **Fans and jets** ("fans"): each actuator disc's delivered thrust, volume flow, induced
  velocity and ideal power next to Froude momentum theory, and the ports;
* **Grid convergence** ("grid"): every grid study among the selected runs (same set-up,
  three or more resolutions; ``analysis.gci``): observed order, Richardson extrapolation,
  GCI and the statistical band, with the figure.

Like the rest of the report, every number comes from one reader (``analysis.physics``,
``analysis.gci``), so the report, the Compare page and the CSV exports agree.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..analysis import gci as G
from ..analysis import physics as P
from ..viz import figures as F
from . import fmt
from .common import (
    ROUGH,
    ROUGH_NOTE,
    STDERR_OK,
    STDERR_WARN,
    Entry,
    compare_rows,
    delta_cell,
    fig,
    rough,
    sub,
)
from .document import KPI, Bullets, Callout, Cell, Heading, KeyValues, KPIRow, Paragraph, Table

#: smoke channels in the order emitters are assigned to them (DESIGN.md 10.1)
DYE_CHANNELS = ("dye", "dye1", "dye2", "dye3")


def _fin(x: Any) -> bool:
    return fmt.finite(x)


def _cfd(entries: Sequence[Entry]) -> list[Entry]:
    return [e for e in entries if e.is_cfd or e.is_cfd3d]


def _units(e: Entry) -> tuple[str, str, str]:
    """(power, force, volume-flow) units: per metre of span in 2D."""
    return ("W", "N", "m³/s") if e.dimension == 3 else ("W/m", "N/m", "m²/s")


def _vec(v: Any) -> str:
    try:
        return "(" + ", ".join(fmt.num(float(c), 3) for c in v) + ") m"
    except (TypeError, ValueError):
        return fmt.DASH


def has_forces(e: Entry) -> bool:
    """Whether the entry's CFD result has force coefficients (a closed room or cavity has none)."""
    return e.result is not None and P.has_forces(e.result)


def has_physics(e: Entry) -> bool:
    """Whether the entry's CFD result carries heat transfer or a fan report."""
    return e.result is not None and (P.has_thermal(e.result) or P.has_fans(e.result))


# ---------------------------------------------------------------------------
# headline numbers: summary tiles, key findings and the comparison (2D and 3D alike)
# ---------------------------------------------------------------------------

#: A standard error below this fraction of the mean is round-off of a steady run, printed as
#: "steady" (``analysis.physics.STEADY_REL``: the float32 resolution).
STEADY_REL = P.STEADY_REL


def steady(st: Any) -> bool:
    """Whether a Stat's standard error is zero up to round-off (a steady result)."""
    return P.steady(getattr(st, "mean", float("nan")), getattr(st, "stderr", float("nan")))


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _se_text(st: Any) -> str:
    """The sub-line of a tile: '± se (1 s.e.)', '(rough estimate)', 'steady' or ''."""
    if st is None or not _fin(getattr(st, "stderr", float("nan"))):
        return ""
    if steady(st):
        return "steady (no fluctuation in the window)"
    return f"± {fmt.err_only(st.stderr)} " + ("(rough estimate)" if rough(st) else "(1 s.e.)")


def physics_kpis(e: Entry) -> KPIRow | None:
    """Summary tiles of a thermal / fan run: Nusselt number, heat rate, fan thrust and flow (at
    most four; None when the run has neither). The heat rate says what it measures (bodies,
    the hottest wall of a closed box, or the heat inputs: ``analysis.physics.heat_rate_basis``)."""
    r = e.result
    if r is None or not has_physics(e):
        return None
    uw, un, uq = _units(e)
    items: list[KPI] = []
    nu = P.nusselt_stat(r)
    if nu is not None and _fin(nu.mean):
        items.append(KPI("Nusselt number Nu", fmt.num(nu.mean, 5) if steady(nu) else fmt.mean_only(nu.mean, nu.stderr),
                         sub=_se_text(nu)))
    q, q_se, _ = P.heat_rate_stat(r)
    if _fin(q) and P.heat_rate_is_input(r):
        # a room heated only by heaters: its "heat rate" is the power the user set, an input
        items.append(KPI("Heater power", fmt.num(P.heater_power_set(r), 4), uw, sub="set (an input, not a result)"))
    elif _fin(q):
        basis = P.heat_rate_basis(e.sim, e.design)
        items.append(KPI("Heat rate", fmt.mean_only(q, q_se) if _fin(q_se) and q_se > 0 else fmt.num(q, 4), uw,
                         sub=basis["short"]))
    temps = P.fluid_temperatures(r)
    if (nu is None or not _fin(nu.mean)) and "t_fluid_max" in temps:
        items.append(KPI("Hottest fluid", fmt.num(temps["t_fluid_max"], 4), "°C",
                         sub=f"mean {fmt.num(temps.get('t_fluid_mean'), 4)} °C" if "t_fluid_mean" in temps else ""))
    fans = P.fan_entries(r)
    if fans:
        thrust = [f["thrust"] for f in fans if _fin(f["thrust"])]
        flow = [f["volume_flow"] for f in fans if _fin(f["volume_flow"])]
        label = "Fan thrust" if len(fans) == 1 else f"Thrust of {len(fans)} fans"
        if thrust:
            set_to = [f["target"] for f in fans if _fin(f["target"]) and f["mode"] == "thrust"]
            items.append(KPI(label, fmt.num(sum(thrust), 4), un,
                             sub=f"set {fmt.num(sum(set_to), 4)} {un}" if len(set_to) == len(fans) else "delivered"))
        if flow:
            w = [f["induced_velocity"] for f in fans if _fin(f["induced_velocity"])]
            items.append(KPI("Volume flow" if len(fans) == 1 else "Volume flow (all fans)", fmt.num(sum(flow), 4),
                             uq, sub=f"disc velocity {fmt.num(w[0], 3)} m/s" if len(w) == 1 else ""))
    return KPIRow(items[:4]) if items else None


def uncertainty_badge(e: Entry) -> list[tuple[str, str]]:
    """The uncertainty badge of a run without forces: from its Nusselt number when it has one
    (the same thresholds as the drag's), else a plain statement that there are no forces."""
    nu = P.nusselt_stat(e.result) if e.result is not None else None
    if nu is not None and _fin(nu.mean) and _fin(nu.stderr) and nu.mean:
        if steady(nu):
            return [("Nu steady", "ok")]
        rel = abs(nu.stderr / nu.mean)
        tone = "ok" if rel < STDERR_OK else ("warn" if rel < STDERR_WARN else "bad")
        if rough(nu):
            tone = "warn" if tone == "ok" else tone
        return [(f"Nu uncertainty ±{fmt.num(rel * 100, 2)} %" + (" (rough estimate)" if rough(nu) else ""), tone)]
    return [("No forces: no solid bodies", "info")]


def physics_finding(e: Entry) -> str:
    """The heat / fan clause of a key-finding bullet ('' when the run has neither)."""
    r = e.result
    if r is None:
        return ""
    uw, un, uq = _units(e)
    parts = []
    nu = P.nusselt_stat(r)
    if nu is not None and _fin(nu.mean):
        if steady(nu):
            text = fmt.num(nu.mean, 4) + " (steady)"
        elif _fin(nu.stderr):
            text = fmt.pm(nu.mean, nu.stderr)
        else:
            text = fmt.num(nu.mean, 4)
        parts.append(fmt.eq("Nu", text))
    q, _, _ = P.heat_rate_stat(r)
    temps = P.fluid_temperatures(r)
    if _fin(q) and P.heat_rate_is_input(r):
        # the set heater power is not a finding; how hot the fluid gets is
        if "t_fluid_max" in temps:
            parts.append(f"hottest fluid {fmt.num(temps['t_fluid_max'], 4)} °C"
                         + (f" (mean {fmt.num(temps['t_fluid_mean'], 4)} °C)" if "t_fluid_mean" in temps else "")
                         + f" with {fmt.num(P.heater_power_set(r), 4)} {uw} of heaters (set)")
        else:
            parts.append(f"heaters set to {fmt.num(P.heater_power_set(r), 4)} {uw}")
    elif _fin(q):
        parts.append(f"heat rate {fmt.num(q, 4)} {uw} ({P.heat_rate_basis(e.sim, e.design)['short']})")
    for f in P.fan_entries(r)[:2]:
        if _fin(f["thrust"]):
            txt = f"{f['name']} delivers {fmt.num(f['thrust'], 4)} {un}"
            if _fin(f["volume_flow"]):
                txt += f" and moves {fmt.num(f['volume_flow'], 3)} {uq}"
            parts.append(txt)
    return ", ".join(parts)


#: compare_rows keys of the placed physics, their short column titles and their unit: a slot of
#: _units() (0 power, 1 force, 2 volume flow), a literal unit, or None
_PHYS_COLUMNS: dict[str, tuple[str, int | str | None]] = {
    "nusselt": ("Nu", None), "heat_rate": ("Heat rate", 0), "t_fluid_max": ("Hottest fluid", "°C"),
    "t_fluid_mean": ("Mean fluid", "°C"), "vent_heat_out": ("Heat out, vents", 0),
    "heater_power": ("Heater power (set)", 0), "fan_thrust": ("Fan thrust", 1), "fan_flow": ("Fan flow", 2),
    "fan_power": ("Ideal fan power", 0)}
#: the heat-transfer table's columns, results first and the set heater power last
_HEAT_KEYS = ("nusselt", "heat_rate", "t_fluid_max", "t_fluid_mean", "vent_heat_out", "heater_power")


def _column_title(key: str, units: tuple[str, str, str]) -> str:
    title, unit = _PHYS_COLUMNS[key]
    if isinstance(unit, int):
        unit = units[unit]
    return title + (f" [{unit}]" if unit else "")


def _phys_delta(row: dict, c: dict) -> Cell:
    """The Δ cell of a physics column: a difference in K for temperatures (a percentage of a
    Celsius value means nothing), none for a setting, else the usual Δ% cell."""
    if row.get("input"):
        return Cell("baseline", "neutral") if c.get("is_baseline") else Cell(fmt.DASH, "neutral")
    if row["key"] in ("t_fluid_max", "t_fluid_mean") and not c.get("is_baseline"):
        d = c.get("delta")
        return Cell(f"{d:+.1f} K".replace("-", "−") if d is not None and _fin(d) else fmt.DASH, "neutral")
    return delta_cell(c, row.get("better"))


def _value_text(c: dict) -> str:
    v, se = c.get("value"), c.get("stderr")
    if v is None or not _fin(v):
        return fmt.DASH
    if se is not None and _fin(se) and abs(se) > STEADY_REL * max(abs(v), 1e-300):
        text = fmt.pm(v, se)
        return text.replace("± ", "± " + ROUGH) if not c.get("stderr_reliable", True) else text
    return fmt.num(v, 4)


def comparable(entries: Sequence[Entry]) -> bool:
    """Whether at least two CFD runs share something to compare: forces, heat transfer or fans."""
    done = [e for e in entries if e.result is not None]
    return any(sum(1 for e in done if test(e)) >= 2 for test in
               (has_forces, lambda e: P.has_thermal(e.result), lambda e: P.has_fans(e.result)))


#: At most this many quantities (each a value and a Δ column) per comparison table: seven or more
#: columns overflow the text width of the HTML report and shrink the PDF's type below legibility.
PHYS_COLUMNS_PER_TABLE = 3


def _physics_table(sel: Sequence[Entry], keys: Sequence[str], caption: str) -> list:
    """Comparison tables over ``sel`` for the ``compare_rows`` keys among ``keys``, at most
    :data:`PHYS_COLUMNS_PER_TABLE` quantities per table (a room's temperatures, vents and heater
    power next to a body's Nu and heat rate made one 13-column table that ran off the page)."""
    rows_cmp = compare_rows({e.sim_id: e.result for e in sel}) or []
    phys = [row for row in rows_cmp if row["key"] in keys]
    if not phys:
        return []
    units = _units(sel[0])
    chunks = [phys[i:i + PHYS_COLUMNS_PER_TABLE] for i in range(0, len(phys), PHYS_COLUMNS_PER_TABLE)]
    blocks: list = []
    for k, part in enumerate(chunks, 1):
        header = ["Simulation"]
        for row in part:
            header += [_column_title(row["key"], units), "Δ"]
        body = []
        for i, e in enumerate(sel):
            line: list[Any] = [Cell(e.name, bold=(i == 0))]
            for row in part:
                c = row["cells"][i]
                line += [_value_text(c), _phys_delta(row, c)]
            body.append(line)
        # the name column gets the room a generated name needs ("Heated cavity · Ra 1,000 · 36 cells/ref")
        widths = [3.0] + [1.6, 1.0] * len(part)
        note = "Δ relative to the first run"
        if any(row["key"] in ("t_fluid_max", "t_fluid_mean") for row in part):
            note += "; temperatures as a difference in K"
        if any(row.get("input") for row in part):
            note += "; the heater power is a setting, shown for context"
        title = caption + (f", {k} of {len(chunks)}" if len(chunks) > 1 else "")
        blocks.append(Table(header, body, align=["l"] + ["r"] * (len(header) - 1), font_scale=0.84, widths=widths,
                            caption=title + f" (± one standard error; {note})"))
    if any(c.get("value") is not None and not c.get("stderr_reliable", True)
           and not _steady_cell(c) for row in phys for c in row["cells"]):
        blocks.append(Paragraph(ROUGH_NOTE, tone="muted"))
    return blocks


def _steady_cell(c: dict) -> bool:
    v, se = c.get("value"), c.get("stderr")
    return v is not None and se is not None and _fin(v) and _fin(se) and abs(se) <= STEADY_REL * max(abs(v), 1e-300)


def physics_comparison(entries: Sequence[Entry], *, prefix: str = "") -> list:
    """Tables (and Nusselt bars) of the heat-transfer and fan quantities of compared CFD runs.

    The numbers are ``analysis.compare_rows`` (the Compare page's): one table of the Nusselt
    number and heat rate over the runs with heat transfer, one of fan thrust, flow and ideal
    power over the runs with fans, each with its Δ against the first run of that table and
    with columns only for quantities a run has. Empty when fewer than two runs have either.
    ``prefix`` starts the captions ("3D ").
    """
    heat = [e for e in entries if e.result is not None and P.has_thermal(e.result)]
    fans = [e for e in entries if e.result is not None and P.has_fans(e.result)]
    blocks: list = []
    if len(heat) >= 2:
        blocks += _physics_table(heat, _HEAT_KEYS, f"{prefix}Heat transfer compared")
        nus = [(e, P.nusselt_stat(e.result)) for e in heat]
        nus = [(e, s) for e, s in nus if s is not None and _fin(s.mean)]
        if len(nus) >= 2:
            blocks.append(fig(F.fig_compare_bars, [{"label": e.name, "nu": s} for e, s in nus], ["nu"],
                              metric_labels={"nu": "Nusselt number Nu"},
                              caption="Nusselt numbers with ±1 standard-error bars; percentages are relative to "
                              "the first run (grey)."))
    if len(fans) >= 2:
        blocks += _physics_table(fans, ("fan_thrust", "fan_flow", "fan_power"), f"{prefix}Fans compared")
    return blocks


# ---------------------------------------------------------------------------
# heat transfer
# ---------------------------------------------------------------------------

_GROUP_WORDS = ("Rayleigh", "Grashof", "Prandtl", "Richardson", "Péclet", "Peclet", "Nusselt", "Boussinesq",
                "temperature", "Temperature", "ΔT", "buoyan", "Buoyan", "Heat", "heat", "thermal", "Thermal")


#: the plan rows whose value scales with the planned ΔT (a heater's or a heat flux's estimated rise)
_DT_ROW_WORDS = ("ΔT", "Temperature difference", "Rayleigh", "Grashof", "Richardson", "β·ΔT", "Boussinesq")
#: how the runner starts the note that states the measured rise (simulation.physics_run.apply_measured_thermal)
MEASURED_DT_NOTE = "Measured temperature difference ΔT = "


def _plan_thermal_rows(e: Entry) -> list[tuple[str, str]]:
    """The planner's own thermal lines (non-dimensional groups, reference temperature
    difference), from the plan recorded with the result.

    When the run measured its temperature rise (a heater or heat-flux room, where the plan's
    ΔT is an estimate), the rows that scale with ΔT are labelled "(planned estimate)" and the
    measured values follow: the fan room was planned at 44.1 K and measured 3.31 K, and the
    report quoted the plan's Ra 1.3×10¹¹ as the run's (critic round 3.3)."""
    prov = getattr(e.result, "provenance", None) or {}
    summ = ((prov.get("plan") or {}).get("summary") or {}) if isinstance(prov, dict) else {}
    rows = [(str(k), str(v)) for k, v in summ.items() if any(w in str(k) for w in _GROUP_WORDS)]
    measured = next((str(n) for n in (getattr(e.result, "notes", None) or [])
                     if str(n).startswith(MEASURED_DT_NOTE)), None)
    if measured is None:
        return rows
    rows = [(f"{k} (planned estimate)" if any(w in k for w in _DT_ROW_WORDS) else k, v) for k, v in rows]
    # "Measured temperature difference ΔT = 3.31 K: <basis> (the planned estimate ...). On the measured ΔT:
    # Gr ..., Ra ... on ... (planned ...); ..." -> the measured ΔT and the groups on it
    head = measured[len(MEASURED_DT_NOTE):]
    rows.append(("ΔT measured", head.split(":", 1)[0].strip()))
    if "On the measured ΔT:" in head:
        groups = head.split("On the measured ΔT:", 1)[1].split("(planned", 1)[0].strip(" .;")
        rows.append(("Groups on the measured ΔT", groups))
    return rows


def _body_conditions(e: Entry) -> list[list[Any]]:
    d = e.design
    if d is None:
        return []
    items = list(getattr(d, "meshes", []) or []) if e.dimension == 3 else list(getattr(d, "bodies", []) or [])
    rows = []
    for b in items:
        mode = str(getattr(b, "thermal", "adiabatic") or "adiabatic")
        val = getattr(b, "thermal_value", 0.0)
        text = {"temperature": f"{fmt.num(val, 4)} °C", "heat_flux": f"{fmt.num(val, 4)} W/m²"}.get(mode, "—")
        rows.append([b.name, mode.replace("_", " "), text])
    return rows


def _wall_conditions(th: Any) -> list[list[Any]]:
    rows = []
    for w in getattr(th, "walls", []) or []:
        text = {"temperature": f"{fmt.num(w.value, 4)} °C", "heat_flux": f"{fmt.num(w.value, 4)} W/m²"}.get(
            w.mode, "—")
        rows.append([w.side, w.mode.replace("_", " "), text])
    return rows


def _temperature_field(e: Entry) -> tuple[Any, str] | None:
    """(FieldData, label) of the temperature, mean preferred, for a 2D result."""
    r = e.result
    for slot in ("mean_fields", "instant_fields"):
        fd = getattr(r, slot, None)
        if fd is not None:
            for name in ("temperature", "theta"):
                if name in (fd.fields or {}):
                    return fd, name
    return None


def _figure_temperature2d(fd: Any, name: str, bodies: Any, t_amb: float, *, ground: bool = False,
                          style: Any = "print", figsize: Any = None) -> Any:
    arr = np.asarray(fd.fields[name], float)
    fin = arr[np.isfinite(arr)]
    ref = t_amb if name == "temperature" else 0.0
    center = None
    if fin.size and fin.min() < ref - 1e-9 * max(1.0, abs(ref)) and fin.max() > ref + 1e-9 * max(1.0, abs(ref)):
        center = ref                     # hot and cold regions: a diverging map about the ambient
    return F.fig_field(fd, name, bodies, center=center, ground=ground, streamlines="u" in fd.fields,
                       view="full" if not bodies else "auto", style=style, figsize=figsize)


def _temperature_slices(e: Entry) -> list[int]:
    return [i for i, s in enumerate(getattr(e.result, "slices", None) or [])
            if "temperature" in (s.fields or {}) or "theta" in (s.fields or {})]


def section_heat(entries: Sequence[Entry], ctx: dict) -> list:
    sel = [e for e in _cfd(entries) if P.has_thermal(e.result)
           or bool(getattr(getattr(e.sim, "thermal", None), "enabled", False))]
    sel = [e for e in sel if e.result is not None]
    if not sel:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(sel, 1):
        sub(blocks, sel, e, no, k)
        r = e.result
        th = getattr(e.sim, "thermal", None)
        uw, _, _ = _units(e)
        kv = []
        if th is not None:
            kv += [("Ambient / inlet temperature", f"{fmt.num(th.t_ambient, 4)} °C"),
                   ("Buoyancy (Boussinesq)", ("on, g = " + fmt.num(th.gravity, 4) + " m/s² along "
                                              + ("−z" if e.dimension == 3 else "−y")) if th.buoyancy else "off"),
                   ("Thermal expansion β", f"{fmt.num(th.beta, 4)} 1/K" if _fin(th.beta) else
                    "1/T_ambient (ideal gas)" if th.buoyancy else "—"),
                   ("Prandtl number", fmt.num(th.prandtl, 3) if _fin(th.prandtl) else "from the fluid")]
        domain = getattr(e.sim.cfd, "domain_mode", "tunnel")
        kv.append(("Domain", "closed (walls on every side, no free stream)" if domain == "closed"
                   else "wind tunnel (inflow at the ambient temperature, zero-gradient outlet)"))
        kv += _plan_thermal_rows(e)
        blocks.append(KeyValues(kv, columns=2, title="Thermal set-up"))
        bodies = _body_conditions(e)
        walls = _wall_conditions(th) if th is not None else []
        if bodies or walls:
            rows = [[Cell(n, bold=True), m, v] for n, m, v in bodies] + [[f"Domain wall: {s}", m, v]
                                                                         for s, m, v in walls]
            blocks.append(Table(["Surface", "Condition", "Value"], rows, align=["l", "l", "r"],
                                caption="Thermal boundary conditions (walls not listed are adiabatic)"))
        srcs = [h for h in getattr(e.sim, "heat_sources", []) or [] if getattr(h, "enabled", True)]
        if srcs:
            blocks.append(Table(["Heat source", "Shape", "Centre", "Size", f"Power [{uw}]"],
                                [[h.name, h.kind, _vec(h.position), _vec(h.size), fmt.num(h.power, 4)] for h in srcs],
                                align=["l", "l", "r", "r", "r"], caption="Volumetric heat sources"))
        nu = P.nusselt_stat(r)
        q, q_se, q_ok = P.heat_rate_stat(r)
        basis = P.heat_rate_basis(e.sim, e.design)
        rows = []
        if nu is not None and _fin(nu.mean):
            se = fmt.err_only(nu.stderr)
            if steady(nu):
                rows.append([Cell("Mean Nusselt number Nu", bold=True), fmt.num(nu.mean, 5), "0 (steady)",
                             "0" if abs(_f(nu.rms)) <= STEADY_REL * abs(nu.mean) else fmt.num(nu.rms, 3)])
            else:
                rows.append([Cell("Mean Nusselt number Nu", bold=True), fmt.mean_only(nu.mean, nu.stderr),
                             (ROUGH + se) if rough(nu) and se != fmt.DASH else se, fmt.num(nu.rms, 3)])
        if _fin(q) and P.heat_rate_is_input(r):
            # the heat put in by heaters alone equals their set power: an input, not a measurement
            rows.append([Cell(f"Heater power (set) [{uw}]", bold=True), fmt.num(P.heater_power_set(r), 4),
                         "set, not measured", fmt.DASH])
        elif _fin(q):
            se = fmt.err_only(q_se)
            if _fin(q_se) and abs(q_se) <= STEADY_REL * max(abs(q), 1e-300):
                rows.append([Cell(f"{basis['label']} [{uw}]", bold=True), fmt.num(q, 5), "0 (steady)", fmt.DASH])
            else:
                rows.append([Cell(f"{basis['label']} [{uw}]", bold=True), fmt.mean_only(q, q_se),
                             (ROUGH + se) if not q_ok and se != fmt.DASH else se, fmt.DASH])
        for name, vals in (getattr(r, "per_body", None) or {}).items():
            if "nu" in vals or "heat_rate" in vals:
                rows.append([f"  {name}", fmt.num(vals.get("nu"), 4),
                             fmt.DASH, f"Q = {fmt.num(vals.get('heat_rate'), 4)} {uw}" if "heat_rate" in vals
                             else fmt.DASH])
        if rows:
            blocks.append(Table(["Quantity", "Mean", "Std. error", "RMS fluct."], rows, align=["l", "r", "r", "r"],
                                caption="Heat transfer (time averages over the averaging window)"))
            if basis["kind"] == "wall":
                definition = (f"Nu = (Q/A) H / (k ΔT): Q the heat through the {basis['side']} wall (the hottest), A "
                              "its area (its length per metre of span in 2D), H the height of the box, k the fluid's "
                              "conductivity and ΔT the hot-to-cold wall temperature difference: the definition of "
                              "the differentially heated cavity benchmarks (de Vahl Davis 1983).")
            elif basis["kind"] == "bodies":
                definition = ("Nu = Q L / (k ΔT A): Q the heat rate from the body surfaces, L the reference length, "
                              "k the fluid's conductivity, ΔT the reference temperature difference and A the heated "
                              "area (perimeter per metre of span in 2D).")
            elif P.heat_rate_is_input(r):
                definition = ("Nothing in this run is held at a fixed temperature, and the heat the heaters put into "
                              "the fluid equals the power set for them (the solver adds it exactly; its own balance "
                              f"agrees to {fmt.num(100 * P.SET_REL, 2)} % or better), so it is listed as a setting. "
                              "The fluid temperatures and the heat balance below are the result.")
            else:
                definition = ("Nothing in this run is held at a fixed temperature, so the heat rate is the heat the "
                              "placed inputs (heaters, heat-flux walls or bodies) put into the fluid, measured by "
                              "the solver's own balance; the fluid temperatures below are the result.")
            if not P.heat_rate_is_input(r):
                definition += " The standard error is from batch means of the history, like the forces."
            blocks.append(Paragraph(definition, tone="muted"))
        temps = P.fluid_temperatures(r)
        if temps:
            kv_t = [(lab, f"{fmt.num(temps[k], 5)} °C") for k, lab in (("t_fluid_mean", "Mean fluid temperature"),
                                                                       ("t_fluid_min", "Lowest"),
                                                                       ("t_fluid_max", "Highest")) if k in temps]
            if "t_fluid_dev_max" in temps:
                kv_t.append(("Largest departure from ambient", f"{fmt.num(temps['t_fluid_dev_max'], 4)} K"))
            blocks.append(KeyValues(kv_t, columns=2, title="Fluid temperatures measured"))
        bal = P.heat_balance(r, e.sim, e.design)
        if bal is not None:
            # a term that is exactly zero is a boundary or input the run does not have (no outlet
            # in a closed box, no heaters): listed, the table would be mostly zeros
            brow = [[f"In: {k}", fmt.num(v, 4)] for k, v in bal["in"].items() if v != 0]
            brow += [[f"Out: {k}", fmt.num(v, 4)] for k, v in bal["out"].items() if v != 0]
            zero = [k for k, v in {**bal["in"], **bal["out"]}.items() if v == 0]
            if _fin(bal["storage"]):
                brow.append(["Stored in the domain (rate)", fmt.num(bal["storage"], 4)])
            rel = bal["relative"]
            tone = "ok" if _fin(rel) and abs(rel) < 0.01 else "warn" if _fin(rel) and abs(rel) < 0.05 else "bad"
            brow.append([Cell("Residual (in − out − stored)", bold=True),
                         Cell(f"{fmt.num(bal['residual'], 3)} ({fmt.pct(100 * rel, 2) if _fin(rel) else fmt.DASH})",
                              tone)])
            blocks.append(Table(["Heat balance term", f"Rate [{uw}]"], brow, align=["l", "r"],
                                caption="Energy balance of the run: a small residual shows the temperature equation "
                                        "conserves heat" + ("" if bal["reported"] else
                                                            " (residual computed from the reported terms)")))
            if zero:
                blocks.append(Paragraph("Terms that are exactly zero in this run (" + ", ".join(zero).lower()
                                        + ") are not listed.", tone="muted"))
            if bal["walls_split"] == "hot_wall":
                blocks.append(Paragraph(
                    f"The solver reports the domain walls as one net term; it is split here into the {basis['side']} "
                    "wall's heat (the run's heat rate) and the net of the other walls, so the table shows the heat "
                    "passing through the box. The split leaves the residual unchanged.", tone="muted"))
            elif bal["walls_split"] == "net" and abs(bal["total_in"]) + abs(bal["total_out"]) == 0 and _fin(q):
                blocks.append(Paragraph(
                    "Every term is zero because the domain walls are reported as one net term: the heat entering "
                    "through the heated walls leaves through the cooled ones, so in a steady state the net is zero "
                    "by the balance itself and the residual tests nothing more.", tone="muted"))
            if bal["missing"]:
                blocks.append(Paragraph("Not reported by the solver: " + ", ".join(bal["missing"])
                                        + "; the residual cannot include them.", tone="muted"))
        if e.is_cfd:
            tf = _temperature_field(e)
            if tf is not None:
                fd, name = tf
                bod = r.bodies_wind or (e.design.bodies if e.design is not None else None)
                t_amb = float(th.t_ambient) if th is not None else 0.0
                blocks.append(fig(_figure_temperature2d, fd, name, bod, t_amb, ground=e.ground,
                                  caption=f"{e.name}: {'time-averaged' if fd.label == 'mean' else 'instantaneous'} "
                                  "temperature" + (" with streamlines of the flow" if "u" in fd.fields else "")
                                  + ". A field with hot and cold regions is centred on the ambient temperature "
                                  "(white)."))
        else:
            from .content3d import figure_slice3d
            for i in _temperature_slices(e)[:2]:
                s = r.slices[i]
                name = "temperature" if "temperature" in s.fields else "theta"
                blocks.append(fig(figure_slice3d, r, i, name, streamlines=True, ground=e.ground,
                                  caption=f"{e.name}: {s.label} temperature in the plane {s.axis} = "
                                  f"{fmt.num(s.position, 3)} m with in-plane streamlines."))
        hist = getattr(r, "history", None) or {}
        if "nu" in hist and "t" in hist and np.size(hist["nu"]) == np.size(hist["t"]):
            stats = {"nu": _steady_stat(nu)} if nu is not None else None
            blocks.append(fig(F.fig_history, hist, r.t_avg_start, stats, keys=["nu"], size=(6.7, 2.3),
                              labels={"nu": "Nusselt Nu"},
                              caption=f"{e.name}: Nusselt-number history; the shaded window is the average quoted "
                              "above, the band its standard error."))
    return blocks


# ---------------------------------------------------------------------------
# emitters and smoke
# ---------------------------------------------------------------------------

def dye_channels(emitters: Sequence[Any]) -> list[tuple[str, str, list[str]]]:
    """Smoke channels of a simulation's dye emitters: ``(field, colour, emitter names)``.

    Emitters share a channel when they share a colour; channels are given out in order of
    first appearance, the first being the existing "dye" field, at most four (DESIGN.md
    10.1). Disabled emitters and particle emitters take no channel.
    """
    out: list[tuple[str, str, list[str]]] = []
    for em in emitters or []:
        if not getattr(em, "enabled", True) or getattr(em, "species", "dye") != "dye":
            continue
        color = str(getattr(em, "color", "#f5f5f5")).lower()
        for _name, c, names in out:
            if c == color:
                names.append(em.name)
                break
        else:
            if len(out) < len(DYE_CHANNELS):
                out.append((DYE_CHANNELS[len(out)], color, [em.name]))
    return out


#: names for describing an emitter colour in a caption (a reader sees "orange", not "#e98a2e")
_COLOUR_NAMES = {"white": (0.96, 0.96, 0.96), "grey": (0.55, 0.55, 0.55), "black": (0.08, 0.08, 0.08),
                 "red": (0.85, 0.2, 0.2), "orange": (0.91, 0.54, 0.18), "yellow": (0.95, 0.85, 0.2),
                 "green": (0.25, 0.7, 0.35), "cyan": (0.2, 0.75, 0.85), "blue": (0.2, 0.4, 0.9),
                 "purple": (0.6, 0.35, 0.85), "pink": (0.95, 0.5, 0.75)}


def colour_name(color: str) -> str:
    """The nearest plain colour name of a hex / named colour (for captions)."""
    from matplotlib.colors import to_rgb
    try:
        rgb = np.array(to_rgb(color))
    except ValueError:
        return str(color)
    return min(_COLOUR_NAMES, key=lambda k: float(np.sum((np.array(_COLOUR_NAMES[k]) - rgb) ** 2)))


def _emitter_row(em: Any) -> list[Any]:
    where = _vec(em.position)
    if em.kind in ("line", "rake") and em.end is not None:
        where += f" → {_vec(em.end)}"
    rate = f"{fmt.num(em.rate, 3)}" + (" /s" if em.species == "particles" else "")
    pulse = (f"{fmt.num(em.pulse_period, 3)} s, {fmt.pct(100 * em.pulse_duty, 0, signed=False)} on"
             if _fin(em.pulse_period) and em.pulse_period > 0 else "continuous")
    return [em.name, em.species, em.kind + (f" ×{em.count}" if em.kind == "rake" else ""), where, rate, pulse,
            "tunnel" if em.frame == "wind" else "body"]


def figure_smoke3d(surfaces: Sequence[Any], volume: Any, *, ground: float | None = None, style: Any = "print",
                   figsize: Any = None) -> Any:
    """The instantaneous smoke cloud of a 3D result around the grey body, iso view."""
    from . import content3d as C3
    from . import render3d as R
    focus = C3._flow_focus(surfaces, volume)

    def build(scene: Any) -> None:
        scene.add_meshes(surfaces)
        scene.add_smoke(volume)
    v = R.render_views(build, ("iso",), ground=ground, focus=focus, theme=C3._theme(style))
    return F.fig_views3d(v.images, v.names, v.titles, v.extents, style=style, figsize=figsize)


def section_flowvis(entries: Sequence[Entry], ctx: dict) -> list:
    def has_smoke(e: Entry) -> bool:
        r = e.result
        if e.is_cfd3d:
            return r.find_volume("dye") is not None
        fd = getattr(r, "instant_fields", None)
        return fd is not None and any(k in (fd.fields or {}) for k in DYE_CHANNELS)

    sel = [e for e in _cfd(entries) if e.result is not None
           and any(getattr(em, "enabled", True) for em in getattr(e.sim, "emitters", None) or [])]
    if not sel:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(sel, 1):
        sub(blocks, sel, e, no, k)
        ems = list(getattr(e.sim, "emitters", []) or [])
        on = [em for em in ems if getattr(em, "enabled", True)]
        if on:
            blocks.append(Table(["Emitter", "Releases", "Shape", "Position", "Rate", "Pulsing", "Fixed to"],
                                [_emitter_row(em) for em in on], align=["l", "l", "l", "l", "r", "l", "l"],
                                caption="Emitters (positions in the design frame; rate: released concentration "
                                        "0–1 for smoke, particles per second for tracers)", font_scale=0.9))
        chans = dye_channels(on)
        if not has_smoke(e) and chans:
            blocks.append(Paragraph("The result stores no smoke field (the run predates emitters, or the smoke was "
                                    "not kept); re-run it to include the smoke picture.", tone="muted"))
        if e.is_cfd:
            fd = getattr(e.result, "instant_fields", None)
            if fd is not None and any(c in (fd.fields or {}) for c, _, _ in chans):
                bod = e.result.bodies_wind or (e.design.bodies if e.design is not None else None)
                pairs = [(c, col) for c, col, _ in chans if c in fd.fields]
                names_ = [", ".join(n) for c, _, n in chans if c in fd.fields]
                legend = "; ".join(f"{', '.join(n)} in {colour_name(col)}" for c, col, n in chans if c in fd.fields)
                blocks.append(fig(F.fig_smoke, fd, pairs, bod, ground=e.ground, labels=names_,
                                  caption=f"{e.name}: smoke at the end of the run (instantaneous; {legend}). Smoke "
                                  "is a passive tracer released by the emitters and carried by the flow; it shows "
                                  "where fluid from each emitter went, not a time average."))
        elif e.is_cfd3d:
            vol = e.result.find_volume("dye")
            if vol is not None:
                ground = 0.0 if e.ground else None
                blocks.append(fig(figure_smoke3d, list(e.result.surface), vol, ground=ground,
                                  caption=f"{e.name}: the smoke cloud at the end of the run (volume rendering of the "
                                  "instantaneous dye concentration around the body)."))
        if any(em.species == "particles" for em in on):
            blocks.append(Paragraph("Tracer particles are drawn live in the Simulate view; results keep the smoke "
                                    "(dye) fields, not particle positions.", tone="muted"))
    return blocks


# ---------------------------------------------------------------------------
# fans and ports
# ---------------------------------------------------------------------------

def _fan_area(fan: Any, dim: int) -> float:
    d = float(getattr(fan, "diameter", float("nan")))
    return math.pi * d * d / 4.0 if dim == 3 else d


def section_fans(entries: Sequence[Entry], ctx: dict) -> list:
    sel = [e for e in _cfd(entries) if e.result is not None and (P.has_fans(e.result)
                                                                  or getattr(e.sim, "fans", None)
                                                                  or getattr(e.sim, "ports", None))]
    if not sel:
        return []
    blocks: list = []
    no = ctx["no"]
    for k, e in enumerate(sel, 1):
        sub(blocks, sel, e, no, k)
        uw, un, uq = _units(e)
        fans = [f for f in getattr(e.sim, "fans", []) or [] if getattr(f, "enabled", True)]
        reported = P.fan_entries(e.result)
        rho = float(e.sim.conditions.fluid.density)
        # Froude's actuator disc needs an unbounded stream. A closed room has none: the planner's
        # reference speed there is a scale (for a fan room, the fan's own static induced velocity),
        # so a "theory" column computed from it compares nothing and would read as a failed check.
        # Same rule as the Results page's Fans tab (app.pages.results.fan_rows).
        closed = P.is_closed_domain(e.sim)
        speed = float(e.sim.conditions.speed)
        rows = []
        for i, rep in enumerate(reported):
            spec = fans[i] if i < len(fans) else None
            thrust = rep["thrust"]
            # a thrust-mode fan delivers its set thrust exactly: its error bar is round-off
            se = fmt.DASH if P.steady(thrust, rep["thrust_stderr"]) else fmt.err_only(rep["thrust_stderr"])
            usable = spec is not None and _fin(thrust) and not closed
            theory = P.momentum_theory(thrust, _fan_area(spec, e.dimension), rho, speed) if usable else {}
            mode = rep["mode"] or (getattr(spec, "mode", "") if spec is not None else "")
            target = rep["target"] if _fin(rep["target"]) else getattr(spec, "value", float("nan"))
            unit_t = {"thrust": un, "pressure_jump": "Pa", "velocity": "m/s"}.get(mode, "")
            imposed = mode == "thrust" and _fin(thrust) and (
                P.steady(thrust, rep["thrust_stderr"])
                or (_fin(target) and abs(thrust - target) <= 1e-6 * max(abs(target), 1e-300)))
            if imposed:
                thrust_cell: Any = f"{fmt.num(thrust, 4)} (set)"     # imposed, so not a measurement
            elif se != fmt.DASH:
                thrust_cell = f"{fmt.mean_only(thrust, rep['thrust_stderr'])} ± {se}"
            else:
                thrust_cell = fmt.num(thrust, 4)
            theory_cell: Any = Cell(fmt.DASH, "muted") if closed else fmt.num(theory.get("induced_velocity"), 3)
            rows.append([Cell(rep["name"], bold=True), f"{mode.replace('_', ' ')} {fmt.num(target, 3)} {unit_t}".strip(),
                         thrust_cell, fmt.num(rep["volume_flow"], 4), fmt.num(rep["induced_velocity"], 3),
                         theory_cell, fmt.num(rep["ideal_power"], 4), fmt.num(rep["pressure_jump"], 4)])
        if rows:
            blocks.append(Table(["Fan", "Set to", f"Thrust [{un}]", f"Flow [{uq}]", "Induced w [m/s]",
                                 "w, momentum theory", f"Ideal power [{uw}]", "Δp [Pa]"], rows,
                                align=["l", "l", "r", "r", "r", "r", "r", "r"], font_scale=0.86,
                                caption=("Actuator-disc fans in a closed domain: delivered values (time averages); "
                                         "a thrust marked (set) is imposed" if closed else
                                         "Actuator-disc fans: delivered values (time averages) next to Froude "
                                         "momentum theory for the delivered thrust")))
            if closed:
                blocks.append(Paragraph(P.CLOSED_FAN_NOTE + " The validated fan checks (docs/validation.md "
                                        "section 9, fans and jets) are in open tunnels, where the theory applies.",
                                        tone="muted"))
            else:
                blocks.append(Paragraph(
                    "Momentum theory for an ideal actuator disc of area A in a stream U: T = 2ρA w (U + w), w the "
                    "induced velocity at the disc (half of the far-wake increase), ideal power T (U + w). A real "
                    "(simulated) disc in a confined domain, with viscous losses, differs; the comparison checks the "
                    "momentum delivered, and the validation (docs/validation.md) quantifies it.", tone="muted"))
        elif fans:
            blocks.append(Paragraph(f"{len(fans)} fan{'s' if len(fans) > 1 else ''} set up, but the result carries "
                                    "no fan report (a result of a solver without the fan model?).", tone="muted"))
        ports = [p for p in getattr(e.sim, "ports", []) or [] if getattr(p, "enabled", True)]
        if ports:
            blocks.append(Table(["Port", "Kind", "Position", "Direction", "Size [m]", "Speed [m/s]", "Carries"],
                                [[p.name, p.kind, _vec(p.position), _vec(p.direction).replace(" m", ""),
                                  fmt.num(p.size, 3), fmt.num(p.speed, 3),
                                  ", ".join(x for x in (f"{fmt.num(p.temperature, 3)} °C" if _fin(p.temperature)
                                                        else "", "smoke" if p.dye else "") if x) or "—"]
                                 for p in ports], align=["l", "l", "l", "l", "r", "r", "l"], font_scale=0.9,
                                caption="Jets and vents"))
        hist = getattr(e.result, "history", None) or {}
        keys = [f"fan{i}_thrust" for i in range(len(reported)) if f"fan{i}_thrust" in hist][:3]
        if keys and "t" in hist:
            stats = {k: _steady_stat(s) for k in keys if (s := P.series_stat(e.result, k)) is not None}
            blocks.append(fig(F.fig_history, hist, e.result.t_avg_start, stats, keys=keys,
                              size=(6.7, 1.0 + 1.3 * len(keys)),
                              labels={k: f"Thrust {reported[int(k[3:].split('_')[0])]['name']} [{un}]" for k in keys},
                              caption=f"{e.name}: fan thrust history (the regulated fans settle to their target)."))
    return blocks


# ---------------------------------------------------------------------------
# grid convergence
# ---------------------------------------------------------------------------

def studies_in(entries: Sequence[Entry]) -> list[G.GridStudy]:
    """The grid studies among the selected CFD results (``analysis.gci.find_grid_studies``)."""
    cfd = [e for e in _cfd(entries) if e.result is not None]
    groups = G.find_grid_studies({e.sim_id: e.sim for e in cfd}, {e.sim_id: e.result for e in cfd})
    out = []
    by_id = {e.sim_id: e for e in cfd}
    for ids in groups:
        try:
            out.append(G.grid_study({i: by_id[i].result for i in ids}))
        except ValueError:
            continue
    return out


def _band(x: Any) -> str:
    """A ± half-width in a narrow table column: two significant figures, powers of ten below
    10⁻⁴ ("2.0×10⁻⁶", not "2.0e-06", which a column of this width breaks after the "e-")."""
    if not _fin(x):
        return fmt.DASH
    v = float(x)
    return "0" if v == 0 else fmt.sci1(v) if abs(v) < 1e-4 else fmt.num(v, 2)


def _gci_row(q: G.GCIResult) -> list[Any]:
    tone = "ok" if q.usable and q.asymptotic else ("warn" if q.usable else "bad")
    verdict = ("asymptotic" if q.asymptotic else "not asymptotic" if q.asymptotic is False else "—")
    if q.noise_limited:
        verdict = "noise‑limited"            # a non-breaking hyphen: the column is narrow
    gci_txt = fmt.pct(100 * q.gci_fine, 2, signed=False) if _fin(q.gci_fine) else fmt.DASH
    return [Cell(q.quantity, bold=True), fmt.num(q.fine.value, 5), q.convergence or fmt.DASH,
            fmt.num(q.p, 3) if _fin(q.p) else fmt.DASH, fmt.num(q.phi_ext, 5),
            gci_txt, _band(q.u_num), _band(q.u_stat), _band(q.u_total), Cell(verdict, tone)]


def _insensitive(q: G.GCIResult) -> bool:
    """A quantity with the same value on every grid (the lift of a symmetric body, ≈ 0 on each):
    there is no discretisation error to estimate, and its "noise-limited" verdict would only
    distract from the quantities that do change."""
    return q.convergence == "insensitive" or all(abs(lv.value) < fmt.ZERO_COEFF for lv in q.levels)


def section_grid(entries: Sequence[Entry], ctx: dict) -> list:
    studies = studies_in(entries)
    if not studies:
        return []
    names = {e.sim_id: e.name for e in entries}
    blocks: list = []
    no = ctx["no"]
    for k, st in enumerate(studies, 1):
        if len(studies) > 1:
            blocks.append(Heading(f"Study {k}: {names.get(st.sim_ids[0], st.sim_ids[0])}", level=2,
                                  number=f"{no}.{k}"))
        any_q = next(iter(st.quantities.values()), None)
        rows = []
        for i, sid in enumerate(st.sim_ids):
            vals = []
            for q in st.quantities.values():
                lv = q.levels[i]
                vals.append(fmt.pm(lv.value, lv.stderr) if _fin(lv.stderr) else fmt.num(lv.value, 5))
            size = " × ".join(fmt.num(s, 3) for s in st.domain_sizes[i]) if i < len(st.domain_sizes) else fmt.DASH
            rows.append([names.get(sid, sid), fmt.num(1.0 / st.h[i], 4), f"{st.cells[i]:,}", size, *vals])
        blocks.append(Table(["Simulation (fine → coarse)", "Cells / ref.", "Cells", "Domain [ref.]",
                             *[q.quantity for q in st.quantities.values()]], rows,
                            align=["l", "r", "r", "r", *["r"] * len(st.quantities)], font_scale=0.86,
                            caption="Grids of the study (values ± one standard error)"))
        if any_q is not None:
            blocks.append(KeyValues([("Refinement ratios r₂₁, r₃₂", f"{any_q.r21:.3f}, {any_q.r32:.3f}"),
                                     ("Domain held fixed", fmt.yes_no(st.domain_fixed)),
                                     ("Formal order", fmt.num(any_q.formal_order, 2)),
                                     ("Safety factor Fs", fmt.num(any_q.fs, 3))], columns=2))
        studied = [q for q in st.quantities.values() if not _insensitive(q)]
        flat = [q for q in st.quantities.values() if _insensitive(q)]
        if studied:
            blocks.append(Table(["Quantity", "Fine grid", "Convergence", "Observed p", "Extrapolated",
                                 "GCI (fine)", "± numerical", "± statistical", "± total", "Verdict"],
                                [_gci_row(q) for q in studied],
                                align=["l", "r", "l", "r", "r", "r", "r", "r", "r", "l"], font_scale=0.8,
                                caption="Discretisation uncertainty (Celik et al. 2008): ± values are about 95 % "
                                        "bands on the fine-grid value; total = numerical and statistical in "
                                        "quadrature"))
            blocks.append(fig(F.fig_grid_convergence, st, caption="Values against cell size (h / h₁ = 1 is the "
                              "finest grid, 0 the grid-converged limit) with ±2 standard-error bars, the Richardson "
                              "curve of the observed order and the GCI band on the finest grid."))
        if flat:
            names_flat = ", ".join(q.quantity for q in flat)
            zero = all(abs(lv.value) < fmt.ZERO_COEFF for q in flat for lv in q.levels)
            blocks.append(Paragraph(
                f"{names_flat}: {'≈ 0' if zero else 'the same value'} on every grid"
                + (" (a symmetric flow)" if zero else "") + ", so there is no discretisation error to estimate.",
                tone="muted"))
        # a quantity's own warnings name it, so a noise warning about one coefficient is not read
        # as a verdict on the whole study
        many = len(studied) > 1
        msgs = list(dict.fromkeys([*st.warnings, *((f"{q.quantity}: {w}" if many else w) for q in studied
                                                   for w in q.warnings)]))
        for m in msgs[:6]:
            blocks.append(Callout(m, "warn"))
        notes = list(dict.fromkeys(n for q in studied for n in q.notes))
        if notes:
            blocks.append(Bullets(notes[:4], tone="small"))
    blocks.append(Paragraph(
        "Procedure: the observed order p comes from the three finest grids (Celik et al. 2008, eq. 3–5), the "
        "extrapolated value from Richardson extrapolation and the band from Roache's Grid Convergence Index with "
        "Fs = 1.25. Each grid's value is a time average with a standard error; when the grid differences are within "
        "two combined standard errors the study is noise-limited and its order is not meaningful. 'Asymptotic' means "
        "the observed order is within 10 % of the formal order of the scheme (or, with four grids, that successive "
        "triplets agree): with three grids alone Roache's ratio check is satisfied by construction. docs/methods.md "
        "section 9 gives the details and the validation of the procedure.", tone="muted"))
    return blocks


# ---------------------------------------------------------------------------
# methodology
# ---------------------------------------------------------------------------

REFERENCES_PHYSICS = {
    "boussinesq": ("Gray, D. D. & Giorgini, A. (1976). The validity of the Boussinesq approximation for liquids and "
                   "gases. International Journal of Heat and Mass Transfer 19, 545–551."),
    "penalisation": ("Kadoch, B., Kolomenskiy, D., Angot, P. & Schneider, K. (2012). A volume penalization method for "
                     "incompressible flows and scalar advection–diffusion with moving obstacles. Journal of "
                     "Computational Physics 231, 4365–4383."),
    "cavity": ("de Vahl Davis, G. (1983). Natural convection of air in a square cavity: a bench mark numerical "
               "solution. International Journal for Numerical Methods in Fluids 3, 249–264."),
    "glauert": ("Glauert, H. (1935). Airplane propellers. In W. F. Durand (ed.), Aerodynamic Theory, vol. IV, "
                "division L. Berlin: Springer."),
    "celik": ("Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H. & Raad, P. E. (2008). Procedure for "
              "estimation and reporting of uncertainty due to discretization in CFD applications. Journal of Fluids "
              "Engineering 130(7), 078001."),
    "roache": ("Roache, P. J. (1998). Verification and Validation in Computational Science and Engineering. "
               "Albuquerque: Hermosa Publishers."),
}


def methodology_physics(entries: Sequence[Entry], h: Any, blocks: list) -> list[str]:
    """Methodology subsections for heat transfer, fans and grid studies; returns their references."""
    refs: list[str] = []
    cfd = [e for e in _cfd(entries) if e.result is not None]
    thermal = [e for e in cfd if P.has_thermal(e.result) or getattr(getattr(e.sim, "thermal", None), "enabled", False)]
    if thermal:
        h("Heat transfer model")
        blocks.append(Paragraph(
            "Temperature is transported as θ = (T − T∞)/ΔT by the advection–diffusion equation "
            "∂θ/∂t + u·∇θ = ∇·((1/Pe + νt/Prt)∇θ) + q, with the Péclet number Pe = Re·Pr, a turbulent Prandtl "
            "number of 0.85 in large-eddy simulations, a bounded MUSCL advection (van Leer limiter, so no new "
            "temperature extrema) and flux-form diffusion that conserves heat exactly, integrated in the same "
            "Runge–Kutta stages as the flow. Buoyancy follows the Boussinesq approximation, a body force Ri·θ along −g with "
            "Ri = gβΔT L/U², valid while βΔT is small (about 0.1 or less for gases; Gray & Giorgini 1976). Solid "
            "bodies at a fixed temperature are imposed by the same volume penalisation as the no-slip wall (Kadoch "
            "et al. 2012), and the heat rate comes from the penalisation term, like the forces. In a closed domain "
            "driven by buoyancy alone the velocity scale is the free-fall velocity √(gβΔT L). The differentially "
            "heated cavity of de Vahl Davis (1983) is the main validation case (docs/validation.md)."))
        refs += [REFERENCES_PHYSICS[k] for k in ("boussinesq", "penalisation", "cavity")]
    fans = [e for e in cfd if P.has_fans(e.result) or getattr(e.sim, "fans", None)]
    if fans:
        h("Fans and jets")
        blocks.append(Paragraph(
            "A fan is an actuator disc: a thin region whose body force adds the pressure jump Δp = T/A (or, in "
            "velocity mode, a regulated jump that reaches the target axial speed through the disc), optionally with "
            "swirl. It represents the momentum and energy a fan adds, not its blades. The delivered thrust is the "
            "integral of the force actually applied; the report sets it beside Froude's momentum theory, "
            "T = 2ρA w (U + w) (Glauert 1935). Jets are mass sources that add their volume flow with the jet's "
            "momentum, temperature and smoke; vents are sinks that remove fluid with what it carries, and the "
            "pressure projection makes the flow satisfy both exactly."))
        refs.append(REFERENCES_PHYSICS["glauert"])
    if studies_in(entries):
        h("Discretisation uncertainty")
        blocks.append(Paragraph(
            "Grid studies follow Celik et al. (2008): the observed order of accuracy from three systematically "
            "refined grids, Richardson extrapolation to zero cell size and Roache's Grid Convergence Index with a "
            "safety factor of 1.25 (Roache 1998). The GCI is an approximately 95 % band on the fine-grid value for "
            "the discretisation error alone; the statistical band (two standard errors of the time average) is "
            "reported beside it and the two are combined in quadrature."))
        refs += [REFERENCES_PHYSICS["celik"], REFERENCES_PHYSICS["roache"]]
    return refs


# ---------------------------------------------------------------------------
# convergence-section and appendix additions
# ---------------------------------------------------------------------------

def diagnostics_block(e: Entry) -> list:
    """The solver's own health record (DESIGN.md 10.4) for the convergence section: the
    final values and, when the run kept its diagnostics table, their history."""
    blocks: list = []
    rows = P.diagnostics_rows(e.result)
    if rows:
        blocks.append(KeyValues([(lab + (f" [{unit}]" if unit else ""), fmt.num(v, 3) if abs(v) >= 1e-3 or v == 0
                                  else fmt.sci1(v) if v > 0 else fmt.num(v, 3)) for _, lab, v, unit in rows[:12]],
                                columns=2, title="Solver diagnostics (end of run)"))
    table = getattr(e.result, "diagnostics_history", None) or {}
    if table and "t" in table and np.size(table["t"]) > 1:
        blocks.append(fig(F.fig_diagnostics, table, caption=f"{e.name}: solver diagnostics over the run. Top: the "
                          "divergence the pressure projection leaves (round-off level: the projection is exact); "
                          "middle: the relative change of each field per step, which falls as a steady flow "
                          "converges and levels off for an unsteady one; bottom: CFL number and time step."))
    return blocks


def _steady_stat(st: Any) -> Any:
    """``st`` with a round-off standard error replaced by 0, so a figure prints the mean alone."""
    if st is not None and steady(st) and st.stderr != 0:
        import dataclasses
        return dataclasses.replace(st, stderr=0.0, stderr_reliable=True)
    return st


def monitor_block(e: Entry) -> list:
    """The convergence figure and numbers of a run without forces (a heated cavity, a fan in a
    room): the series its convergence verdict was taken on -- the kinetic energy of the box, the
    Nusselt number, the rise of the warmest and of the mean fluid above ambient (a heated closed box is
    judged on them; force coefficients never are), each fan's disc velocity -- instead of force
    histories that are all zero."""
    r = e.result
    hist = getattr(r, "history", None) or {}
    if "t" not in hist:
        return []
    fans = P.fan_entries(r)
    want = [("nu", "Nusselt Nu"), ("t_rise_max", "Warmest fluid above ambient [K]"),
            ("t_rise_mean", "Mean fluid above ambient [K]"), ("ke", "Mean kinetic energy [J/kg]")]
    want += [(f"fan{i}_velocity", f"Disc velocity {f['name']} [m/s]") for i, f in enumerate(fans[:2])]
    keys = [k for k, _ in want if k in hist and np.size(hist[k]) == np.size(hist["t"])][:4]
    blocks: list = []
    if keys:
        stats = {k: _steady_stat(s) for k in keys if (s := P.series_stat(r, k)) is not None}
        blocks.append(fig(F.fig_history, hist, r.t_avg_start, stats, keys=keys, size=(6.7, 1.0 + 1.3 * len(keys)),
                          labels=dict(want),
                          caption=f"{e.name}: the monitored quantities the run's convergence was judged on (it has "
                          "no forces). The shaded region is the averaging window; the black line is the mean and "
                          "its grey band ±1 standard error."))
    t = np.asarray(hist["t"], float)
    n_win = int(np.sum(t >= r.t_avg_start)) if t.size and _fin(r.t_avg_start) else 0
    rows = [("Simulated time", f"{fmt.num(r.t_end, 3)} convective units"),
            ("Transient discarded", f"t·U/L < {fmt.num(r.t_avg_start, 3)}"),
            ("Averaging window", f"{fmt.num(r.t_end - r.t_avg_start, 3)} convective units, {n_win:,} samples"),
            ("Converged", fmt.yes_no(r.converged))]
    nu = P.nusselt_stat(r)
    if nu is not None and _fin(nu.mean) and nu.mean:
        rows.append(("Relative s.e. of Nu", "0 (steady)" if steady(nu)
                     else fmt.pct(abs(nu.stderr / nu.mean) * 100, 2, signed=False) if _fin(nu.stderr) else fmt.DASH))
    blocks.append(KeyValues(rows, columns=2))
    return blocks


def provenance_rows(e: Entry) -> list[Any]:
    """One appendix row: simulation, configuration hash, solver-settings hash, seed, software."""
    rec = getattr(e.result, "provenance", None) if e.result is not None else None
    if not isinstance(rec, dict) or not rec:
        return [e.name, "not recorded", fmt.DASH, fmt.DASH, fmt.DASH]
    sw = rec.get("software") or {}
    pk = sw.get("packages") or {}
    soft = f"Aerolume {sw.get('aerolume', '?')}" + (f", torch {pk['torch']}" if pk.get("torch") else "")
    plan = rec.get("plan") or {}
    return [e.name, str(rec.get("config_hash", ""))[:12] or fmt.DASH, str(plan.get("solver_hash", ""))[:12] or "—",
            "none" if rec.get("seed") is None else str(rec["seed"]), soft]


__all__ = [
    "DYE_CHANNELS",
    "REFERENCES_PHYSICS",
    "comparable",
    "diagnostics_block",
    "dye_channels",
    "figure_smoke3d",
    "has_forces",
    "has_physics",
    "methodology_physics",
    "monitor_block",
    "physics_comparison",
    "physics_finding",
    "physics_kpis",
    "provenance_rows",
    "section_fans",
    "section_flowvis",
    "section_grid",
    "section_heat",
    "studies_in",
]
