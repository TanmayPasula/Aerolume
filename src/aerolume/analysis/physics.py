"""Statistics of the placed-physics outputs: heat transfer, fans and jets (DESIGN.md section 10).

The solvers report these quantities in three places, and this module is the one reader
the Compare page, the report and the exports share, so they never disagree:

* result fields: ``nusselt`` (a :class:`~aerolume.model.Stat` of the mean Nusselt
  number over the heated bodies), ``heat_rate`` (W in 3D, W/m in 2D, from the bodies
  into the fluid), ``fans_report`` (one dict per fan) and ``diagnostics``;
* history series recorded every step (:data:`PHYSICS_HISTORY_KEYS`, plus per-fan series
  ``fan<i>_thrust`` / ``fan<i>_flow``), from which standard errors are measured with
  the same batch-means rules as the forces;
* ``per_body`` entries (``nu``, ``heat_rate``) for several heated bodies.

Fan reports are read through :data:`FAN_KEYS`, which lists the accepted spellings of
each quantity, so a report written by an older or newer solver still reads. A missing
quantity is ``nan``: nothing here invents a number.

The heat balance (:func:`heat_balance`) checks energy conservation of a thermal run:
heat added by bodies, domain walls and volumetric sources against heat carried out by
the flow and stored in the domain. Its residual is the research-grade check that the
thermal solver conserves energy; it is only as complete as the terms the solver
reports, and the function says which terms were missing.

numpy and the standard library only (DESIGN.md section 2).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..model import Stat
from .stats import coefficient_stats

_NAN = float("nan")

#: A standard error below this fraction of its mean is round-off, not a measured uncertainty: the
#: float32 machine epsilon (1.19e-7), the resolution of a single-precision GPU run. A steady run's
#: batch means of identical samples give 1e-16 to 1e-9 relative, which a table would otherwise print
#: as a "rough" error bar of 10^-16; :func:`steady` names the rule for the report, GCI and exports.
STEADY_REL = 1.2e-7


def steady(value: float, stderr: float) -> bool:
    """Whether ``stderr`` is zero up to round-off of ``value`` (:data:`STEADY_REL`)."""
    v, se = _f(value), _f(stderr)
    return math.isfinite(v) and math.isfinite(se) and abs(se) <= STEADY_REL * max(abs(v), 1e-300)


#: History series of a thermal run (the runner records them per step when heat transfer is on).
PHYSICS_HISTORY_KEYS = ("nu", "heat_rate")
#: History series of fan ``i``: ``fan{i}_thrust`` [N or N/m], ``fan{i}_flow`` [m³/s or m²/s].
FAN_HISTORY = re.compile(r"^fan(\d+)_(thrust|flow|velocity|power|dp)$")

#: Accepted spellings of each fan-report quantity, first = the canonical one.
FAN_KEYS: dict[str, tuple[str, ...]] = {
    "thrust": ("thrust", "thrust_n", "force"),
    "thrust_stderr": ("thrust_stderr", "thrust_se"),
    "volume_flow": ("volume_flow", "flow", "flow_rate", "q"),
    "induced_velocity": ("induced_velocity", "mean_velocity", "velocity", "u_disc", "axial_velocity"),
    "ideal_power": ("ideal_power", "power"),
    "pressure_jump": ("pressure_jump", "dp", "delta_p"),
    "target": ("target", "value", "setpoint"),
}

#: Heat-balance terms (W in 3D, W/m in 2D) read from ``result.diagnostics``: positive adds heat to
#: the fluid. ``out`` terms are heat leaving through open boundaries (positive = leaving).
#: ``heat_in_walls`` is the NET of every domain wall (the solvers sum the sides), so in a steady
#: closed box the heated and the cooled walls cancel; :func:`heat_balance` splits it when it can.
HEAT_IN_TERMS = {"heat_in_bodies": "Bodies", "heat_in_walls": "Domain walls (net)", "heat_in_sources": "Heat sources",
                 "heat_in_ports": "Jets (enthalpy in)", "heat_in_inlet": "Inlet (enthalpy in)"}
#: (the ports' term also holds the fluid an unbalanced closed box loses evenly, which carries its heat out)
HEAT_OUT_TERMS = {"heat_out_outlet": "Outlet (enthalpy out)",
                  "heat_out_ports": "Vents and closed-box leak (enthalpy out)"}
HEAT_STORAGE = "heat_storage_rate"
HEAT_RESIDUAL = "heat_balance_residual"
#: Per-side wall heat rates, positive into the fluid (``heat_wall_left``, ... ``heat_wall_back``), when
#: a solver reports them; they replace the net ``heat_in_walls`` term in the balance.
HEAT_WALL = re.compile(r"^heat_wall_(left|right|bottom|top|front|back)$")
#: Domain sides in the order the balance lists them.
WALL_SIDES = ("left", "right", "bottom", "top", "front", "back")


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NAN
    return v


def _fin(x: Any) -> bool:
    return math.isfinite(_f(x))


def has_thermal(result: Any) -> bool:
    """Whether a CFD result carries heat-transfer output (a finite Nusselt number or heat rate)."""
    nu = getattr(result, "nusselt", None)
    return bool((nu is not None and _fin(getattr(nu, "mean", _NAN))) or _fin(getattr(result, "heat_rate", _NAN))
                or any(k in (getattr(result, "history", None) or {}) for k in PHYSICS_HISTORY_KEYS))


def has_fans(result: Any) -> bool:
    """Whether a CFD result carries a fan report."""
    return bool(getattr(result, "fans_report", None))


def has_forces(result: Any) -> bool:
    """Whether a CFD result has force coefficients: a finite mean drag coefficient.

    A closed room or a heated cavity has no solid body (the walls are the domain), so the
    solver reports no forces and every coefficient is nan; summaries and comparisons then
    lead with the heat transfer or the fans instead of a row of dashes.
    """
    cd = getattr(result, "cd", None)
    return cd is not None and _fin(getattr(cd, "mean", _NAN))


def series_stat(result: Any, key: str, t_start: float | None = None) -> Stat | None:
    """Batch-means statistics of a recorded history series, over the result's averaging window.

    ``None`` when the series was not recorded. The window starts at ``t_start`` or the
    result's ``t_avg_start`` (the same window as the forces).
    """
    h = getattr(result, "history", None) or {}
    if key not in h or "t" not in h:
        return None
    t = np.asarray(h["t"], float)
    x = np.asarray(h[key], float)
    if t.size != x.size or t.size == 0:
        return None
    t0 = t_start if t_start is not None else _f(getattr(result, "t_avg_start", _NAN))
    if not math.isfinite(t0):
        t0 = None
    return coefficient_stats(t, x, t0)


def nusselt_stat(result: Any) -> Stat | None:
    """The mean Nusselt number with its standard error: the result's own Stat, or one measured
    from the "nu" history when the result has none (results of an early solver)."""
    nu = getattr(result, "nusselt", None)
    if nu is not None and _fin(getattr(nu, "mean", _NAN)):
        if not _fin(nu.stderr):
            measured = series_stat(result, "nu")
            if measured is not None and _fin(measured.stderr):
                return Stat(mean=nu.mean, rms=nu.rms, min=nu.min, max=nu.max, stderr=measured.stderr,
                            stderr_reliable=measured.stderr_reliable)
        return nu
    return series_stat(result, "nu")


def heat_rate_stat(result: Any) -> tuple[float, float, bool]:
    """(value, standard error, reliable) of the heat rate from the bodies into the fluid.

    The standard error comes from the "heat_rate" history when recorded, else from the
    Nusselt number's relative error (the heat rate is proportional to Nu at fixed
    temperatures). nan when there is no heat rate.
    """
    q = _f(getattr(result, "heat_rate", _NAN))
    s = series_stat(result, "heat_rate")
    if s is not None:
        if not math.isfinite(q):
            q = s.mean
        return q, s.stderr, bool(s.stderr_reliable)
    nu = nusselt_stat(result)
    if math.isfinite(q) and nu is not None and _fin(nu.mean) and nu.mean and _fin(nu.stderr):
        return q, abs(q * nu.stderr / nu.mean), bool(nu.stderr_reliable)
    return q, _NAN, True


def _pick(d: Mapping[str, Any], names: Sequence[str]) -> float:
    for n in names:
        if n in d:
            return _f(d[n])
    return _NAN


def fan_entries(result: Any) -> list[dict[str, Any]]:
    """The fans of a result in canonical form, one dict per fan (input order).

    Keys: ``name``, ``mode``, the canonical names of :data:`FAN_KEYS` (nan when not
    reported) and ``thrust_reliable``. When the solver did not give a thrust standard
    error but recorded a ``fan<i>_thrust`` history, it is measured from that history.
    """
    out = []
    for i, raw in enumerate(getattr(result, "fans_report", None) or []):
        if not isinstance(raw, Mapping):
            continue
        e: dict[str, Any] = {"name": str(raw.get("name") or f"Fan {i + 1}"), "mode": str(raw.get("mode") or ""),
                             "thrust_reliable": True}
        for key, names in FAN_KEYS.items():
            e[key] = _pick(raw, names)
        if not math.isfinite(e["thrust_stderr"]):
            s = series_stat(result, f"fan{i}_thrust")
            if s is not None:
                e["thrust_stderr"] = s.stderr
                e["thrust_reliable"] = bool(s.stderr_reliable)
                if not math.isfinite(e["thrust"]):
                    e["thrust"] = s.mean
        out.append(e)
    return out


def momentum_theory(thrust: float, area: float, density: float, speed: float) -> dict[str, float]:
    """Froude actuator-disc momentum theory for a disc of ``area`` in a stream ``speed``.

    ``T = 2 ρ A U² a (1 + a)`` written with the axial induction a = w/U at the disc
    (w the induced velocity), solved for a; returns the induced velocity, the far-wake
    velocity increment (2w), the ideal power ``T (U + w)`` and the static (U = 0)
    values ``w = sqrt(T / (2 ρ A))``. Used by the validation of the fan model and as a
    reference column in the report's fan table. ``area`` is m² (3D) or the disc width in
    m (2D, per metre of span).
    """
    nan = {"induced_velocity": _NAN, "wake_increment": _NAN, "ideal_power": _NAN, "induction": _NAN}
    if not (thrust > 0 and area > 0 and density > 0 and speed >= 0):
        return nan
    if speed == 0:
        w = math.sqrt(thrust / (2.0 * density * area))
        return {"induced_velocity": w, "wake_increment": 2 * w, "ideal_power": thrust * w, "induction": math.inf}
    # 2 rho A (U + w) w = T  ->  w = (-U + sqrt(U^2 + 2T/(rho A))) / 2
    w = 0.5 * (-speed + math.sqrt(speed * speed + 2.0 * thrust / (density * area)))
    return {"induced_velocity": w, "wake_increment": 2 * w, "ideal_power": thrust * (speed + w),
            "induction": w / speed}


def _heated_items(design: Any) -> list[Any]:
    """The bodies (2D) or meshes (3D) of a design held at a temperature or given a heat flux."""
    if design is None:
        return []
    items = list(getattr(design, "meshes", None) or []) if int(getattr(design, "dimension", 2) or 2) == 3 \
        else list(getattr(design, "bodies", None) or [])
    return [b for b in items if str(getattr(b, "thermal", "adiabatic") or "adiabatic") != "adiabatic"]


def hot_wall_side(sim: Any, design: Any = None) -> str | None:
    """The domain side whose heat the run reports as ``heat_rate`` (and whose Nusselt number it
    quotes): the hottest fixed-temperature wall of a closed box without heated bodies, or None.

    The same rule as the runner's (``simulation.physics_run.hot_wall``), read from the model
    instead of the plan, so a report can label a stored result without re-planning it.
    """
    th = getattr(sim, "thermal", None)
    if th is None or not getattr(th, "enabled", False):
        return None
    if getattr(getattr(sim, "cfd", None), "domain_mode", "tunnel") != "closed" or _heated_items(design):
        return None
    temps = [(_f(getattr(w, "value", _NAN)), str(getattr(w, "side", ""))) for w in getattr(th, "walls", None) or []
             if getattr(w, "mode", "") == "temperature" and _fin(getattr(w, "value", _NAN))]
    return max(temps)[1] if temps else None


def heat_rate_basis(sim: Any, design: Any = None) -> dict[str, str]:
    """What a result's ``heat_rate`` measures, for labels: ``{"kind", "label", "short", "side"}``.

    ``kind`` is "bodies" (heated bodies into the fluid), "wall" (the hottest wall of a closed
    box, ``side`` names it), or "inputs" (the heat the placed inputs -- heaters, heat-flux walls
    -- put into the fluid, which the runner reports when nothing has a fixed temperature).
    """
    if _heated_items(design):
        return {"kind": "bodies", "label": "Heat rate, bodies → fluid", "short": "bodies → fluid", "side": ""}
    side = hot_wall_side(sim, design)
    if side:
        return {"kind": "wall", "label": f"Heat rate through the {side} wall (the hottest)",
                "short": f"{side} wall → fluid", "side": side}
    return {"kind": "inputs", "label": "Heat put into the fluid", "short": "inputs → fluid", "side": ""}


def heat_balance(result: Any, sim: Any = None, design: Any = None) -> dict[str, Any] | None:
    """Energy balance of a thermal run from ``result.diagnostics`` (W in 3D, W/m in 2D).

    Returns ``{"in": {label: W}, "out": {label: W}, "storage": W, "residual": W,
    "relative": residual / throughput, "missing": [labels], "reported": bool,
    "walls_split": str}`` or None when the result reports no heat-balance terms.
    ``reported`` says whether the residual is the solver's own (True) or computed here
    from the terms (False). The bodies' term falls back to ``result.heat_rate`` when the
    diagnostics do not repeat it.

    The walls' term: per-side ``heat_wall_<side>`` diagnostics, when the solver reports
    them, are listed one by one (a heated wall under "in", a cooled one under "out"); else
    the net ``heat_in_walls`` is split, when ``sim`` shows that ``result.heat_rate`` is the
    hottest wall's heat (:func:`hot_wall_side`), into that wall and the other walls' net.
    Without either, a steady closed box lists a net of ~0 and ``walls_split`` is "net" so
    the caller can say why. Splitting changes only the rows, never the residual.

    ``relative`` is the residual over the heat passing through the domain: the largest of
    the gross heat in, the gross heat out and ``|result.heat_rate|`` (``"throughput"``), so a
    net walls term of ~0 never makes round-off look like a failed balance.
    """
    diag = getattr(result, "diagnostics", None) or {}
    keys = set(HEAT_IN_TERMS) | set(HEAT_OUT_TERMS) | {HEAT_STORAGE, HEAT_RESIDUAL}
    if not any(k in diag for k in keys):
        return None
    terms_in = {label: _f(diag[k]) for k, label in HEAT_IN_TERMS.items() if k in diag}
    # without the design, assume (as before designs were passed) that the heat rate is the bodies'
    if "heat_in_bodies" not in diag and _fin(getattr(result, "heat_rate", _NAN)) and \
            (design is None or _heated_items(design)):
        terms_in[HEAT_IN_TERMS["heat_in_bodies"]] = _f(result.heat_rate)
    terms_out = {label: _f(diag[k]) for k, label in HEAT_OUT_TERMS.items() if k in diag}
    walls_label = HEAT_IN_TERMS["heat_in_walls"]
    split = "none" if walls_label not in terms_in else "net"
    per_side = {m.group(1): _f(v) for k, v in diag.items() if (m := HEAT_WALL.match(str(k))) and _fin(v)}
    if per_side:
        terms_in.pop(walls_label, None)
        for side in WALL_SIDES:
            if side in per_side and per_side[side] != 0:
                if per_side[side] > 0:
                    terms_in[f"{side.capitalize()} wall"] = per_side[side]
                else:
                    terms_out[f"{side.capitalize()} wall"] = -per_side[side]
        split = "sides"
    elif walls_label in terms_in and sim is not None:
        side = hot_wall_side(sim, design)
        q = _f(getattr(result, "heat_rate", _NAN))
        net = terms_in[walls_label]
        if side and math.isfinite(q) and math.isfinite(net):
            terms_in.pop(walls_label)
            terms_in[f"{side.capitalize()} wall (the hottest)"] = q
            rest = net - q                      # the other walls together: negative = heat leaving
            if rest > 0:
                terms_in["Other domain walls (net)"] = rest
            elif rest < 0:
                terms_out["Other domain walls (net)"] = -rest
            split = "hot_wall"
    storage = _f(diag.get(HEAT_STORAGE, _NAN))
    total_in = sum(v for v in terms_in.values() if math.isfinite(v))
    total_out = sum(v for v in terms_out.values() if math.isfinite(v))
    reported = HEAT_RESIDUAL in diag
    if reported:
        resid = _f(diag[HEAT_RESIDUAL])
    else:
        resid = total_in - total_out - (storage if math.isfinite(storage) else 0.0)
    # the heat passing through the domain, not the net of the terms: a closed cavity's walls term nets to
    # ~0 (heat in through the hot wall, out through the cold one), and against that "relative" read ~1 for
    # round-off (critic round 3.4). The gross in, the gross out and the measured heat rate bound it.
    gross_in = sum(abs(v) for v in terms_in.values() if math.isfinite(v))
    gross_out = sum(abs(v) for v in terms_out.values() if math.isfinite(v))
    q = _f(getattr(result, "heat_rate", _NAN))
    scale = max(gross_in, gross_out, abs(q) if math.isfinite(q) else 0.0, 1e-300)
    missing = []
    if not terms_out:
        missing.append("heat carried out by the flow")
    if not math.isfinite(storage):
        missing.append("storage in the domain")
    return {"in": terms_in, "out": terms_out, "storage": storage, "residual": resid,
            "relative": resid / scale if math.isfinite(resid) else _NAN, "missing": missing,
            "reported": reported, "total_in": total_in, "total_out": total_out, "walls_split": split,
            "throughput": scale if scale > 1e-300 else _NAN}


#: ``result.diagnostics`` keys and their labels / units, in the order the report lists them: the
#: columns of the solvers' diagnostics rows (``solvers.ns2d.diagnostics.DIAG_COLUMNS``, the last
#: row of a run), with the longer spellings a solver may also use.
DIAGNOSTIC_LABELS: dict[str, tuple[str, str]] = {
    "div_rms": ("RMS velocity divergence", "U/L"),
    "divergence_rms": ("RMS velocity divergence", "U/L"),
    "div_max": ("Max velocity divergence", "U/L"),
    "divergence_max": ("Max velocity divergence", "U/L"),
    "p_residual": ("Pressure-solve residual (relative)", ""),
    "pressure_residual": ("Pressure-solve residual (relative)", ""),
    "cfl": ("CFL number of the last step", ""),
    "cfl_max": ("Largest CFL number", ""),
    "dt": ("Time step", "L/U"),
    "steps_per_s": ("Throughput", "steps/s"),
    "du_rel": ("Relative change of u per step", ""),
    "dv_rel": ("Relative change of v per step", ""),
    "dw_rel": ("Relative change of w per step", ""),
    "dp_rel": ("Relative change of p per step", ""),
    "dtheta_rel": ("Relative change of θ per step", ""),
}
# the absolute change norms of each field (solvers.ns2d.diagnostics: L2 = the domain-integral norm),
# after the relative ones: in solver units, so they matter mostly for exports and trend plots
_FIELD_UNITS = {"u": "U", "v": "U", "w": "U", "p": "ρU²", "theta": "ΔT"}
for _fld, _unit in _FIELD_UNITS.items():
    _sym = "θ" if _fld == "theta" else _fld
    DIAGNOSTIC_LABELS[f"d{_fld}_max"] = (f"Max change of {_sym} per step", _unit)
    DIAGNOSTIC_LABELS[f"d{_fld}_rms"] = (f"RMS change of {_sym} per step", _unit)
    DIAGNOSTIC_LABELS[f"d{_fld}_l2"] = (f"L2 change of {_sym} per step (domain norm)", "")
del _fld, _unit, _sym
#: diagnostics that are bookkeeping, not health numbers (left out of the report's table)
DIAGNOSTIC_SKIP = ("step", "t", "perf_steps")
#: results, not solver health: the fluid temperatures and the heater power set (the heat section shows them)
FLUID_TEMPERATURE_KEYS = ("t_fluid_mean", "t_fluid_min", "t_fluid_max", "t_fluid_dev_max", "heat_sources_set")


def diagnostics_rows(result: Any) -> list[tuple[str, str, float, str]]:
    """``(key, label, value, unit)`` of a result's solver diagnostics: the known keys first in
    :data:`DIAGNOSTIC_LABELS` order, then any other numeric key (heat-balance terms excluded)."""
    diag = getattr(result, "diagnostics", None) or {}
    skip = (set(HEAT_IN_TERMS) | set(HEAT_OUT_TERMS) | {HEAT_STORAGE, HEAT_RESIDUAL} | set(DIAGNOSTIC_SKIP)
            | set(FLUID_TEMPERATURE_KEYS))
    rows = [(k, lab, _f(diag[k]), unit) for k, (lab, unit) in DIAGNOSTIC_LABELS.items() if k in diag]
    seen = {lab for _, lab, _, _ in rows}
    rows = [r for i, r in enumerate(rows) if r[1] not in {x[1] for x in rows[:i]}]   # one spelling each
    rows += [(k, k.replace("_", " "), _f(v), "") for k, v in diag.items()
             if k not in DIAGNOSTIC_LABELS and k not in skip and not HEAT_WALL.match(str(k)) and _fin(v)
             and k.replace("_", " ") not in seen]
    return rows


#: Relative agreement within which a run's heat rate *is* the heater power that was set: the solvers
#: put a heater's power into the fluid exactly (DESIGN.md 10.3, exact totals), so a measured value
#: this close to the setting carries no information beyond the input.
SET_REL = 1e-3


def heater_power_set(result: Any) -> float:
    """The total heater power set for the run (W in 3D, W/m in 2D), from ``diagnostics``
    (``heat_sources_set``, written by the runner); nan when the run had no heaters."""
    return _f((getattr(result, "diagnostics", None) or {}).get("heat_sources_set", _NAN))


def heat_rate_is_input(result: Any) -> bool:
    """Whether the result's ``heat_rate`` is just the heater power that was set.

    True when nothing in the run has a fixed temperature (no Nusselt number) and the heat rate
    equals the heaters' set power to :data:`SET_REL`: a room heated only by heaters, where the
    runner reports "the heat put into the fluid". Presenting that as a measured heat-transfer
    rate, with an error bar and a percentage change, would quote an input back as a result
    (two rooms with 300 W/m heaters "agree to −0.0 %"), so the Compare page, the report and the
    exports label it "Heater power (set)" instead.
    """
    q, q_set = _f(getattr(result, "heat_rate", _NAN)), heater_power_set(result)
    if not (math.isfinite(q) and math.isfinite(q_set)):
        return False
    nu = getattr(result, "nusselt", None)
    if nu is not None and _fin(getattr(nu, "mean", _NAN)):
        return False
    return abs(q - q_set) <= SET_REL * max(abs(q_set), 1e-300)


def vent_heat_out(result: Any) -> float:
    """Net heat carried out through the vents, minus what the jets bring in (W or W/m), from
    ``diagnostics``; nan when the run has no ports (both terms absent or exactly zero)."""
    diag = getattr(result, "diagnostics", None) or {}
    out_, in_ = _f(diag.get("heat_out_ports", _NAN)), _f(diag.get("heat_in_ports", _NAN))
    terms = [v for v in (out_, in_) if math.isfinite(v)]
    if not terms or all(v == 0 for v in terms):
        return _NAN
    return (out_ if math.isfinite(out_) else 0.0) - (in_ if math.isfinite(in_) else 0.0)


def is_closed_domain(sim: Any) -> bool:
    """Whether a simulation runs in a closed domain (a room, a cavity: DESIGN.md 10.6) rather than a
    wind tunnel. Its "speed" is then the planner's reference scale, not a free stream."""
    return getattr(getattr(sim, "cfd", None), "domain_mode", "tunnel") == "closed"


#: Why momentum theory is not quoted for a fan in a closed domain (the Results page's Fans tab says the same).
CLOSED_FAN_NOTE = ("Momentum theory: not shown in a closed domain. Froude's actuator disc needs an unbounded "
                   "stream; here the fan drives the room's recirculation, whose speed the walls and friction set.")


def fluid_temperatures(result: Any) -> dict[str, float]:
    """The fluid temperatures a thermal run measured (``t_fluid_mean`` / ``_min`` / ``_max`` in °C,
    ``t_fluid_dev_max`` the largest departure from ambient in K), from ``result.diagnostics``;
    only the finite ones."""
    diag = getattr(result, "diagnostics", None) or {}
    return {k: _f(diag[k]) for k in FLUID_TEMPERATURE_KEYS[:4] if k in diag and _fin(diag[k])}


__all__ = ["CLOSED_FAN_NOTE", "DIAGNOSTIC_LABELS", "FAN_HISTORY", "FAN_KEYS", "FLUID_TEMPERATURE_KEYS",
           "HEAT_IN_TERMS", "HEAT_OUT_TERMS", "HEAT_RESIDUAL", "HEAT_STORAGE", "HEAT_WALL", "PHYSICS_HISTORY_KEYS",
           "SET_REL", "STEADY_REL", "WALL_SIDES",
           "diagnostics_rows", "fan_entries", "fluid_temperatures", "has_fans", "has_forces", "has_thermal",
           "heat_balance", "heat_rate_basis", "heat_rate_is_input", "heat_rate_stat", "heater_power_set",
           "hot_wall_side", "is_closed_domain", "momentum_theory", "nusselt_stat", "series_stat", "steady",
           "vent_heat_out"]
