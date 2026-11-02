"""Side-by-side comparison of results against a baseline.

The Compare page and the report's comparison section both show the same table:
one row per metric, one cell per simulation, each cell with the value, its
change from the baseline (absolute and %), whether that change is an
improvement, and whether it is larger than the statistical noise of the two
runs. Building it once here keeps the two consistent.

A change counts as *significant* when it exceeds twice the combined standard
error of the two means (about 95 % confidence for independent runs). Without
that flag, a 0.3 % "improvement" in Cd from two LES runs would look as real as
a 10 % one; AirShaper-style users act on these deltas, so the noise floor has
to be in the table, not in a footnote. The flag is left undecided (None) when
either error bar is missing or is only an estimate (``Stat.stderr_reliable``
False: too few independent samples for batch means), because a significance
test on an error bar that is several times too small is worse than none.

A table cannot say whether the runs it compares are comparable at all, so
:func:`comparison_warnings` does: runs at different speeds, angles or fluids, or
simulating different Reynolds numbers, differ for reasons that are not the
design. The Compare page shows those warnings above the table and the report
prints them as callouts, from this one rule.

**2D and 3D never share a row set.** A 2D coefficient is per unit span on a
reference length, a 3D one is on a reference area: a 2D Cd of 0.30 and a 3D CD
of 0.30 are different quantities that happen to share a name, and a delta
between them is meaningless. Every result therefore belongs to a dimension
group (:func:`result_dimension`: 2D CFD and airfoil polars are 2; 3D CFD and
finite-wing sweeps are 3), the table is built from the metrics of the
*baseline's* group only, and results of the other group keep their column with
``comparable: False`` cells and no values. :func:`comparison_warnings` says so
in a sentence, and :func:`comparison_groups` lets a caller split a selection
into comparable sets. Inside the 3D group a 3D CFD run and a wing sweep share
the Reynolds number only (a CFD run is one operating point, a sweep a curve),
exactly like a 2D CFD run and an airfoil polar.

**Placed physics** (DESIGN.md section 10). Heat-transfer and fan results add rows for
the mean Nusselt number, the measured heat rate, the maximum and mean fluid temperature,
the heat the vents carry out, and the thrust, volume flow and ideal power summed over the
fans, read through :mod:`.physics` (the same reader the report uses). Like every other row
they appear only when a compared result has them, and carry measured standard errors where
the solver recorded the history. A heater's power is an *input*: a room heated only by
heaters reports the power it was given as its "heat rate", so that number goes to a
"Heater power (set)" row with the flow conditions (no error bar, no verdict, rows flagged
``"input": True``) instead of leading the table as a result that "agrees to 0.0 %".
Temperatures in °C get a difference but no percentage (an interval scale).

**Closed domains** (rooms, cavities) have no free stream: their speed and Reynolds number
are reference scales the planner chose. Given the simulations, :func:`comparison_warnings`
therefore never says that two rooms "ran at different speeds"; it compares what drives
them (heater power, fans, jets, heated walls) and says when a room is compared with a
wind-tunnel run.

A Reynolds-number difference is reported only when it is not the design's own: runs
that each simulate their real Reynolds number at the same speed and fluid differ in
Re because their reference lengths differ, which is part of the design comparison.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..model import CFD3DResult, CFDResult, PolarResult, WingResult
from . import physics
from .polar import polar_metrics
from .wing import wing_metrics

Result = CFDResult | PolarResult | CFD3DResult | WingResult

#: Two simulated Reynolds numbers count as different above this relative difference.
REYNOLDS_RTOL = 0.01


@dataclass(frozen=True)
class Metric:
    """How to read one comparable number from a result."""
    key: str
    label: str
    unit: str
    better: str | None          # "lower" | "higher" | None (no preferred direction)
    kind: str                   # "cfd" | "polar" | "cfd3d" | "wing"


def _f(v) -> float | None:
    """Finite float or None (tables and JSON want None, not nan)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ratio_se(num: float, den: float, se_num: float, se_den: float) -> float:
    """First-order standard error of num/den for independent num, den."""
    if not den or not all(math.isfinite(v) for v in (num, den, se_num, se_den)):
        return float("nan")
    r = num / den
    return abs(r) * math.sqrt((se_num / num) ** 2 + (se_den / den) ** 2) if num else abs(se_num / den)


def _ok(stat) -> bool:
    """Whether a Stat's standard error is a measured one (see the module docstring)."""
    return bool(getattr(stat, "stderr_reliable", True))


# --- placed physics (DESIGN.md section 10): heat transfer and fans, read through analysis.physics --------

def _nu(r) -> tuple[float, float, bool]:
    s = physics.nusselt_stat(r)
    return (s.mean, s.stderr, _ok(s)) if s is not None else (float("nan"), float("nan"), True)


def _fan_errors(r, i: int, fan: dict[str, Any], key: str) -> tuple[float, bool]:
    """(standard error, reliable) of one fan's ``key``: the thrust's own, the volume flow's from the
    ``fan<i>_flow`` history, the ideal power's from the disc velocity's (P = T·u_disc, T exact in
    thrust mode); (nan, True) when nothing was recorded. The Results page's Fans tab uses the same
    series, so Compare rounds "Fan volume flow" to its error as the Summary does (0.079 ± 0.028)."""
    if key == "thrust":
        return fan["thrust_stderr"], bool(fan["thrust_reliable"])
    if key == "volume_flow":
        st = physics.series_stat(r, f"fan{i}_flow")
        return (st.stderr, bool(st.stderr_reliable)) if st is not None else (float("nan"), True)
    if key == "ideal_power" and fan.get("mode") == "thrust" and math.isfinite(fan["thrust"]):
        st = physics.series_stat(r, f"fan{i}_velocity")
        if st is not None and math.isfinite(st.stderr):
            return abs(float(fan["thrust"])) * st.stderr, bool(st.stderr_reliable)
    return float("nan"), True


def _fan_total(r, key: str) -> tuple[float, float, bool]:
    """Sum of one fan quantity over a result's fans; the fans' standard errors add in quadrature
    (nan when any fan's is unknown)."""
    fans = physics.fan_entries(r)
    vals = [(i, f) for i, f in enumerate(fans) if math.isfinite(f[key])]
    if not vals:
        return float("nan"), float("nan"), True
    errs = [_fan_errors(r, i, f, key) for i, f in vals]
    ses = [e for e, _ok in errs]
    se = math.sqrt(sum(s * s for s in ses)) if all(math.isfinite(s) for s in ses) else float("nan")
    return float(sum(f[key] for _i, f in vals)), se, all(ok for _e, ok in errs)


_NAN3 = (float("nan"), float("nan"), True)


def _heat_rate(r) -> tuple[float, float, bool]:
    """The measured heat rate; nan when it is only the heater power that was set (that goes to the
    "heater_power" input row instead, see :func:`.physics.heat_rate_is_input`)."""
    return _NAN3 if physics.heat_rate_is_input(r) else physics.heat_rate_stat(r)


def _diag(key: str) -> Callable[[Any], tuple[float, float, bool]]:
    """Reader of one ``result.diagnostics`` value (no standard error: a single stored number)."""
    def get(r) -> tuple[float, float, bool]:
        v = _f((getattr(r, "diagnostics", None) or {}).get(key))
        return (float("nan") if v is None else v), float("nan"), True
    return get


def _physics_metrics(kind: str, unit_w: str, unit_n: str, unit_q: str
                     ) -> list[tuple[Metric, Callable[[Any], tuple[float, float, bool]]]]:
    """Heat-transfer and fan result rows (shown only when a compared result has them).

    A closed room heated only by heaters has no Nusselt number and its "heat rate" is the power
    the user set, so what describes it is how hot the fluid gets and where the heat goes: the
    maximum and mean fluid temperature and the heat the vents carry out. The heater power itself
    is an input row (:func:`_input_metrics`), never a result with an error bar or a verdict.
    """
    return [
        (Metric("nusselt", "Mean Nusselt number Nu", "", None, kind), _nu),
        (Metric("heat_rate", "Heat-transfer rate into the fluid", unit_w, None, kind), _heat_rate),
        (Metric("t_fluid_max", "Maximum fluid temperature", "°C", None, kind), _diag("t_fluid_max")),
        (Metric("t_fluid_mean", "Mean fluid temperature", "°C", None, kind), _diag("t_fluid_mean")),
        (Metric("vent_heat_out", "Heat removed by the vents (net of jets)", unit_w, None, kind),
         lambda r: (physics.vent_heat_out(r), float("nan"), True)),
        (Metric("fan_thrust", "Fan thrust (all fans)", unit_n, None, kind), lambda r: _fan_total(r, "thrust")),
        (Metric("fan_flow", "Fan volume flow (all fans)", unit_q, None, kind),
         lambda r: _fan_total(r, "volume_flow")),
        (Metric("fan_power", "Fan ideal power (all fans)", unit_w, "lower", kind),
         lambda r: _fan_total(r, "ideal_power")),
    ]


def _input_metrics(kind: str, unit_w: str) -> list[tuple[Metric, Callable[[Any], tuple[float, float, bool]]]]:
    """Rows that show what was set rather than what was measured (listed with the flow conditions)."""
    return [(Metric("heater_power", "Heater power (set)", unit_w, None, kind),
             lambda r: (physics.heater_power_set(r), float("nan"), True))]


# value, stderr and whether that stderr is reliable, per metric key, CFD results
_CFD: list[tuple[Metric, Callable[[CFDResult], tuple[float, float, bool]]]] = [
    (Metric("cd", "Drag coefficient Cd", "", "lower", "cfd"), lambda r: (r.cd.mean, r.cd.stderr, _ok(r.cd))),
    (Metric("cl", "Lift coefficient Cl", "", None, "cfd"), lambda r: (r.cl.mean, r.cl.stderr, _ok(r.cl))),
    (Metric("cm", "Moment coefficient Cm", "", None, "cfd"), lambda r: (r.cm.mean, r.cm.stderr, _ok(r.cm))),
    (Metric("ld", "Lift-to-drag ratio L/D", "", "higher", "cfd"),
     lambda r: (r.lift_to_drag, _ratio_se(r.cl.mean, r.cd.mean, r.cl.stderr, r.cd.stderr),
                _ok(r.cl) and _ok(r.cd))),
    (Metric("drag", "Drag", "N/m", "lower", "cfd"),
     lambda r: (r.drag, abs(r.drag * r.cd.stderr / r.cd.mean) if r.cd.mean else float("nan"), _ok(r.cd))),
    (Metric("lift", "Lift", "N/m", None, "cfd"),
     lambda r: (r.lift, abs(r.lift * r.cl.stderr / r.cl.mean) if r.cl.mean else float("nan"), _ok(r.cl))),
    (Metric("moment", "Pitching moment", "N·m/m", None, "cfd"), lambda r: (r.moment, float("nan"), True)),
    (Metric("drag_area", "Drag area CdA", "m²/m", "lower", "cfd"),
     lambda r: (r.drag_area, abs(r.drag_area * r.cd.stderr / r.cd.mean) if r.cd.mean else float("nan"),
                _ok(r.cd))),
    (Metric("power", "Power to overcome drag", "W/m", "lower", "cfd"),
     lambda r: (r.power, abs(r.power * r.cd.stderr / r.cd.mean) if r.cd.mean else float("nan"), _ok(r.cd))),
    (Metric("cl_rms", "Lift fluctuation Cl′ (rms)", "", "lower", "cfd"), lambda r: (r.cl.rms, float("nan"), True)),
    (Metric("strouhal", "Strouhal number St", "", None, "cfd"),
     lambda r: (r.strouhal if r.strouhal is not None else float("nan"), float("nan"), True)),
    *_physics_metrics("cfd", "W/m", "N/m", "m²/s"),
    *_input_metrics("cfd", "W/m"),
    (Metric("re_real", "Reynolds number (real)", "", None, "cfd"), lambda r: (r.re_real, float("nan"), True)),
    (Metric("re_sim", "Reynolds number (simulated)", "", None, "cfd"), lambda r: (r.re_sim, float("nan"), True)),
]

_POLAR: list[tuple[Metric, str]] = [
    (Metric("cl_max", "Maximum lift Cl,max", "", "higher", "polar"), "cl_max"),
    (Metric("alpha_cl_max", "Angle of Cl,max", "deg", None, "polar"), "alpha_cl_max"),
    (Metric("cd_min", "Minimum drag Cd,min", "", "lower", "polar"), "cd_min"),
    (Metric("cl_at_cd_min", "Cl at Cd,min", "", None, "polar"), "cl_at_cd_min"),
    (Metric("ld_max", "Best lift-to-drag (L/D)max", "", "higher", "polar"), "ld_max"),
    (Metric("alpha_ld_max", "Angle of best L/D", "deg", None, "polar"), "alpha_ld_max"),
    (Metric("lift_slope_per_rad", "Lift slope dCl/dα", "1/rad", None, "polar"), "lift_slope_per_rad"),
    (Metric("alpha_zero_lift", "Zero-lift angle α0", "deg", None, "polar"), "alpha_zero_lift"),
    (Metric("cm0", "Zero-lift moment Cm0", "", None, "polar"), "cm0"),
]

def _scaled_se(value: float, stat) -> float:
    """Standard error of a dimensional value proportional to a coefficient's mean."""
    return abs(value * stat.stderr / stat.mean) if stat.mean else float("nan")


def _moment(r: CFD3DResult, i: int) -> float:
    m = r.moments
    return float(m[i]) if m is not None and len(m) > i else float("nan")


# 3D CFD results (DESIGN.md 7.2: coefficients on q * ref_area; forces in N, moments in N m)
_CFD3D: list[tuple[Metric, Callable[[CFD3DResult], tuple[float, float, bool]]]] = [
    (Metric("cd", "Drag coefficient CD", "", "lower", "cfd3d"), lambda r: (r.cd.mean, r.cd.stderr, _ok(r.cd))),
    (Metric("cs", "Side-force coefficient CS", "", None, "cfd3d"), lambda r: (r.cs.mean, r.cs.stderr, _ok(r.cs))),
    (Metric("cl", "Lift coefficient CL", "", None, "cfd3d"), lambda r: (r.cl.mean, r.cl.stderr, _ok(r.cl))),
    (Metric("c_roll", "Rolling-moment coefficient", "", None, "cfd3d"),
     lambda r: (r.c_roll.mean, r.c_roll.stderr, _ok(r.c_roll))),
    (Metric("cm", "Pitching-moment coefficient Cm", "", None, "cfd3d"),
     lambda r: (r.cm.mean, r.cm.stderr, _ok(r.cm))),
    (Metric("cn", "Yawing-moment coefficient Cn", "", None, "cfd3d"),
     lambda r: (r.cn.mean, r.cn.stderr, _ok(r.cn))),
    (Metric("ld", "Lift-to-drag ratio L/D", "", "higher", "cfd3d"),
     lambda r: (r.lift_to_drag, _ratio_se(r.cl.mean, r.cd.mean, r.cl.stderr, r.cd.stderr),
                _ok(r.cl) and _ok(r.cd))),
    (Metric("drag", "Drag", "N", "lower", "cfd3d"), lambda r: (r.drag, _scaled_se(r.drag, r.cd), _ok(r.cd))),
    (Metric("side", "Side force", "N", None, "cfd3d"), lambda r: (r.side, _scaled_se(r.side, r.cs), _ok(r.cs))),
    (Metric("lift", "Lift", "N", None, "cfd3d"), lambda r: (r.lift, _scaled_se(r.lift, r.cl), _ok(r.cl))),
    (Metric("roll_moment", "Rolling moment", "N·m", None, "cfd3d"),
     lambda r: (_moment(r, 0), float("nan"), True)),
    (Metric("pitch_moment", "Pitching moment", "N·m", None, "cfd3d"),
     lambda r: (_moment(r, 1), float("nan"), True)),
    (Metric("yaw_moment", "Yawing moment", "N·m", None, "cfd3d"),
     lambda r: (_moment(r, 2), float("nan"), True)),
    (Metric("drag_area", "Drag area CDA", "m²", "lower", "cfd3d"),
     lambda r: (r.drag_area, _scaled_se(r.drag_area, r.cd), _ok(r.cd))),
    (Metric("power", "Power to overcome drag", "W", "lower", "cfd3d"),
     lambda r: (r.power, _scaled_se(r.power, r.cd), _ok(r.cd))),
    (Metric("cl_rms", "Lift fluctuation CL′ (rms)", "", "lower", "cfd3d"),
     lambda r: (r.cl.rms, float("nan"), True)),
    (Metric("strouhal", "Strouhal number St", "", None, "cfd3d"),
     lambda r: (r.strouhal if r.strouhal is not None else float("nan"), float("nan"), True)),
    *_physics_metrics("cfd3d", "W", "N", "m³/s"),
    *_input_metrics("cfd3d", "W"),
    (Metric("re_real", "Reynolds number (real)", "", None, "cfd3d"), lambda r: (r.re_real, float("nan"), True)),
    (Metric("re_sim", "Reynolds number (simulated)", "", None, "cfd3d"),
     lambda r: (r.re_sim, float("nan"), True)),
]

# finite-wing sweeps: (metric, wing_metrics key)
_WING: list[tuple[Metric, str]] = [
    (Metric("cl_max", "Maximum lift CL,max", "", "higher", "wing"), "cl_max"),
    (Metric("alpha_cl_max", "Angle of CL,max", "deg", None, "wing"), "alpha_cl_max"),
    (Metric("cd_min", "Minimum drag CD,min", "", "lower", "wing"), "cd_min"),
    (Metric("ld_max", "Best lift-to-drag (L/D)max", "", "higher", "wing"), "ld_max"),
    (Metric("alpha_ld_max", "Angle of best L/D", "deg", None, "wing"), "alpha_ld_max"),
    (Metric("cd0", "Zero-lift drag CD0 (parabolic fit)", "", "lower", "wing"), "cd0"),
    (Metric("oswald", "Oswald efficiency e (parabolic fit)", "", "higher", "wing"), "oswald"),
    (Metric("e_cruise", "Span efficiency e at cruise", "", "higher", "wing"), "e_cruise"),
    (Metric("cl_alpha_per_rad", "Lift slope dCL/dα", "1/rad", None, "wing"), "cl_alpha_per_rad"),
    (Metric("alpha_zero_lift", "Zero-lift angle α0", "deg", None, "wing"), "alpha_zero_lift"),
    (Metric("cm0", "Zero-lift moment Cm0", "", None, "wing"), "cm0"),
    (Metric("cm_alpha_per_deg", "Moment slope dCm/dα", "1/deg", None, "wing"), "cm_alpha_per_deg"),
]

#: Polar metrics that are only a lower bound (the end of the sweep) when the stall was not reached.
_STALL_KEYS = ("cl_max", "alpha_cl_max")

# Metric keys whose value is a flow condition or a setting rather than a result: shown, but a
# change there is never called an improvement and never tested for significance.
_CONTEXT = {"re_real", "re_sim", "heater_power"}
#: Settings (the row carries ``"input": True`` so a caller can mark them "(set)").
INPUT_KEYS = frozenset({"heater_power"})
#: Quantities on an interval scale (°C): a difference is meaningful, a percentage of it is not
#: (+10 % of 20 °C is not a physical statement), so their cells carry ``delta`` but ``pct`` None.
INTERVAL_KEYS = frozenset({"t_fluid_max", "t_fluid_mean"})


def _as_mapping(results: Mapping[str, Result] | Sequence[Result]) -> dict[str, Result]:
    if isinstance(results, Mapping):
        return dict(results)
    return {r.sim_id: r for r in results}


def result_dimension(result: Result) -> int:
    """2 for 2D results (CFD, airfoil polar), 3 for 3D results (3D CFD, finite wing).

    Results of different dimensions are never compared (see the module docstring).
    """
    if isinstance(result, (CFDResult, PolarResult)):
        return 2
    if isinstance(result, (CFD3DResult, WingResult)):
        return 3
    raise TypeError(f"cannot compare {type(result).__name__}")


def comparison_groups(results: Mapping[str, Result] | Sequence[Result]) -> dict[int, list[str]]:
    """Simulation ids by dimension group, ``{2: [...], 3: [...]}`` (non-empty groups, input order)."""
    out: dict[int, list[str]] = {}
    for sid, r in _as_mapping(results).items():
        out.setdefault(result_dimension(r), []).append(sid)
    return out


def _metrics(dim: int) -> list[Metric]:
    """The metric rows of one dimension group, deduplicated by key (CFD metrics first)."""
    if dim == 2:
        cfd, other = [m for m, _ in _CFD], [m for m, _ in _POLAR]
    else:
        cfd, other = [m for m, _ in _CFD3D], [m for m, _ in _WING]
    keys = {m.key for m in cfd}
    return cfd + [m for m in other if m.key not in keys]


def _values(res: Result) -> dict[str, dict[str, Any]]:
    """metric key -> {"value", "stderr", "reliable", "lower_bound"} for one result."""
    if isinstance(res, CFDResult):
        out = {}
        for m, g in _CFD:
            v, se, ok = g(res)
            out[m.key] = {"value": _f(v), "stderr": _f(se), "reliable": bool(ok), "lower_bound": False}
        return out
    if isinstance(res, PolarResult):
        pm = polar_metrics(res)
        open_top = pm.get("cl_max_reached") == 0.0          # Cl still rising at the last angle
        vals = {m.key: {"value": _f(pm[k]), "stderr": None, "reliable": True,
                        "lower_bound": open_top and m.key in _STALL_KEYS} for m, k in _POLAR}
        vals["re_real"] = {"value": _f(res.re), "stderr": None, "reliable": True, "lower_bound": False}
        return vals
    if isinstance(res, CFD3DResult):
        out = {}
        for m, g in _CFD3D:
            v, se, ok = g(res)
            out[m.key] = {"value": _f(v), "stderr": _f(se), "reliable": bool(ok), "lower_bound": False}
        return out
    if isinstance(res, WingResult):
        wm = wing_metrics(res)
        open_top = wm.get("cl_max_reached") == 0.0
        vals = {m.key: {"value": _f(wm[k]), "stderr": None, "reliable": True,
                        "lower_bound": open_top and m.key in _STALL_KEYS} for m, k in _WING}
        vals["re_real"] = {"value": _f(res.re_mac), "stderr": None, "reliable": True, "lower_bound": False}
        return vals
    raise TypeError(f"cannot compare {type(res).__name__}")


_EMPTY = {"value": None, "stderr": None, "reliable": True, "lower_bound": False}


def compare_rows(results: Mapping[str, Result] | Sequence[Result], baseline_id: str | None = None) -> list[dict]:
    """Comparison table: one dict per metric, cells in the order of ``results``.

    ``results`` is ``{sim_id: result}`` or a sequence of results (keyed by their
    ``sim_id``); ``baseline_id`` defaults to the first one. Rows appear only for
    metrics that at least one result has. Each row::

        {"key", "label", "unit", "better", "kind", "baseline": float | None,
         "cells": [{"sim_id", "is_baseline", "value", "stderr", "stderr_reliable",
                    "lower_bound", "delta", "pct", "improved": bool | None,
                    "significant": bool | None}, ...]}

    ``pct`` is the change relative to |baseline| in percent (None when the baseline
    is zero or missing, and for temperatures in °C, :data:`INTERVAL_KEYS`). ``input``
    is True for rows of settings (:data:`INPUT_KEYS`: the heater power), which are
    never judged or tested for significance. ``improved`` is None for metrics without a preferred
    direction, for unchanged values and for a lower bound. ``significant`` is None
    when either side has no finite standard error or only an estimated one
    (``stderr_reliable`` False). ``lower_bound`` marks a polar's or a wing's CL,max
    and its angle when the stall was not reached inside the sweep (the value is
    the last angle's). A final ``converged`` row carries booleans (CFD only).

    Rows come from the baseline's dimension group only (2D: CFD + polar metrics;
    3D: 3D CFD + wing metrics, see the module docstring). Every cell also carries
    ``comparable``: False for a result of the other dimension, whose cells then
    hold no value, delta or verdict; the column stays so callers that index cells
    by position keep working.
    """
    res = _as_mapping(results)
    if not res:
        return []
    ids = list(res)
    base = ids[0] if baseline_id is None else baseline_id
    if base not in res:
        raise KeyError(f"baseline {base!r} is not among the compared results")
    dims = {sid: result_dimension(r) for sid, r in res.items()}
    group = dims[base]
    comparable = {sid: dims[sid] == group for sid in ids}
    vals = {sid: (_values(r) if comparable[sid] else {}) for sid, r in res.items()}

    metrics = _metrics(group)
    rows: list[dict] = []
    for m in metrics:
        if not any(vals[s].get(m.key, _EMPTY)["value"] is not None for s in ids):
            continue
        bv = vals[base].get(m.key, _EMPTY)
        b, b_se = bv["value"], bv["stderr"]
        cells = []
        for sid in ids:
            cv = vals[sid].get(m.key, _EMPTY)
            v, se = cv["value"], cv["stderr"]
            bound = bool(cv["lower_bound"] or bv["lower_bound"])
            delta = pct = improved = significant = None
            if v is not None and b is not None:
                delta = v - b
                pct = 100.0 * delta / abs(b) if b != 0 and m.key not in INTERVAL_KEYS else None
                if m.better and m.key not in _CONTEXT and delta != 0 and not bound:
                    improved = (delta < 0) if m.better == "lower" else (delta > 0)
                measured = cv["reliable"] and bv["reliable"]
                if (sid != base and se is not None and b_se is not None and measured
                        and m.key not in _CONTEXT):
                    significant = abs(delta) > 2.0 * math.hypot(se, b_se)
            cells.append({"sim_id": sid, "is_baseline": sid == base, "value": v, "stderr": se,
                          "stderr_reliable": bool(cv["reliable"]), "lower_bound": bool(cv["lower_bound"]),
                          "delta": delta, "pct": pct, "improved": improved, "significant": significant,
                          "comparable": comparable[sid]})
        rows.append({"key": m.key, "label": m.label, "unit": m.unit, "better": m.better,
                     "kind": m.kind, "baseline": b, "cells": cells, "input": m.key in INPUT_KEYS})

    cfd_type = CFDResult if group == 2 else CFD3DResult
    if any(isinstance(r, cfd_type) for r in res.values()):
        rows.append({"key": "converged", "label": "Statistically converged", "unit": "", "better": None,
                     "kind": "cfd" if group == 2 else "cfd3d", "baseline": None, "input": False,
                     "cells": [{"sim_id": sid, "is_baseline": sid == base,
                                "value": bool(r.converged) if isinstance(r, cfd_type) else None,
                                "stderr": None, "stderr_reliable": True, "lower_bound": False, "delta": None,
                                "pct": None, "improved": None, "significant": None,
                                "comparable": comparable[sid]}
                               for sid, r in res.items()]})
    return rows


# ---------------------------------------------------------------------------
# are the compared runs comparable?
# ---------------------------------------------------------------------------

def reynolds_mismatch(results: Mapping[str, Result] | Sequence[Result],
                      rtol: float = REYNOLDS_RTOL) -> dict[str, float]:
    """``{sim_id: re_sim}`` of the compared CFD results when their simulated Reynolds
    numbers differ by more than ``rtol`` (relative to the smallest), else ``{}``.

    In the capped mode the simulated Reynolds number used to follow the grid, so
    a Draft and a Standard run of the same case simulated different flows (Re 6,400
    against 25,600 for NACA 2412) and the "grid effect" between them was mostly a
    Reynolds effect. Any comparison across different simulated Reynolds numbers
    needs saying so, whatever caused it. 2D and 3D CFD runs are judged
    separately (they are never compared with each other); the result holds the
    runs of every group that mismatches.
    """
    res = _as_mapping(results)
    out: dict[str, float] = {}
    for cls in (CFDResult, CFD3DResult):          # 2D and 3D runs are judged separately
        re_sim = {sid: float(r.re_sim) for sid, r in res.items()
                  if isinstance(r, cls) and _f(r.re_sim) is not None and r.re_sim > 0}
        if len(re_sim) < 2:
            continue
        lo, hi = min(re_sim.values()), max(re_sim.values())
        if hi > (1.0 + rtol) * lo:
            out.update(re_sim)
    return out


#: Relative difference in density or viscosity below which two fluids count as the same
#: (see :func:`conditions_differ`).
FLUID_REL_TOL = 1e-3


def conditions_differ(conditions: Mapping[str, Any] | Sequence[Any]) -> list[str]:
    """Which flow conditions differ between simulations: a subset of ["speed", "angle",
    "yaw", "fluid"], in that order. ``conditions`` holds ``FlowConditions`` (or anything
    with ``speed``, ``alpha_deg`` and ``fluid``; ``yaw_deg`` is optional and taken as 0
    when absent, so 2D conditions never differ in yaw), keyed by simulation id or in a
    sequence.

    Speeds and angles are compared to 1e-6 relative / absolute. Fluids are compared by
    density and viscosity to :data:`FLUID_REL_TOL` (1e-3) relative, so a renamed preset
    is still the same air, and so is the same air written with different rounding:
    projects saved before the preset had one definition hold 1.7894e-5 Pa s for "Air,
    ISA sea level" where the script API stores 1.7892976e-5 (5.7e-5 apart). No flow
    result can tell 0.1 % of density or viscosity apart; a real change of fluid (water,
    another altitude or temperature: 1,000 m of ISA is already 9 % less dense) is far
    larger.
    """
    items = list(conditions.values()) if isinstance(conditions, Mapping) else list(conditions)
    items = [c for c in items if c is not None]
    if len(items) < 2:
        return []

    def spread(values: list[float], rel: bool, tol: float = 1e-6) -> bool:
        lo, hi = min(values), max(values)
        return (hi - lo) > (tol * max(abs(lo), abs(hi)) if rel else tol)

    out = []
    if spread([float(c.speed) for c in items], True):
        out.append("speed")
    if spread([float(c.alpha_deg) for c in items], False):
        out.append("angle")
    if spread([float(getattr(c, "yaw_deg", 0.0) or 0.0) for c in items], False):
        out.append("yaw")
    if (spread([float(c.fluid.density) for c in items], True, FLUID_REL_TOL)
            or spread([float(c.fluid.viscosity) for c in items], True, FLUID_REL_TOL)):
        out.append("fluid")
    return out


_WHAT = {"speed": "speeds", "angle": "angles", "yaw": "yaw angles", "fluid": "fluids"}


def _simulates_real_re(r: Result) -> bool:
    """Whether a CFD result simulated its real Reynolds number (not capped), to 0.1 %."""
    re_real, re_sim = _f(getattr(r, "re_real", None)), _f(getattr(r, "re_sim", None))
    return re_real is not None and re_sim is not None and re_real > 0 and abs(re_sim - re_real) <= 1e-3 * re_real


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _split_context(c: Any) -> tuple[Any, Any]:
    """(FlowConditions, Simulation or None) from a ``conditions`` value, which may be either."""
    if c is not None and hasattr(c, "conditions") and hasattr(c, "cfd"):
        return c.conditions, c
    return c, None


def _round(x: Any) -> float:
    """``x`` to 9 significant digits (0.0 when missing), so settings typed the same compare equal."""
    v = _f(x)
    return 0.0 if v is None else float(f"{v:.9g}")


def driving_inputs(sim: Any) -> dict[str, Any]:
    """What drives a closed-domain run, as comparable values: ``{"heater power": W, "fans": (...),
    "ports (jets and vents)": (...), "heated walls": (...), "buoyancy": bool}``.

    A room has no free stream, so these settings take the place the speed has in a tunnel:
    two rooms differ "for reasons other than the design" exactly when one of these differs.
    Disabled items do not count; heat inputs count only with heat transfer on.
    """
    th = getattr(sim, "thermal", None)
    heat_on = bool(getattr(th, "enabled", False))
    heaters = [h for h in getattr(sim, "heat_sources", None) or [] if getattr(h, "enabled", True)] if heat_on else []
    walls = ([w for w in getattr(th, "walls", None) or [] if getattr(w, "mode", "adiabatic") != "adiabatic"]
             if heat_on else [])
    fans = [f for f in getattr(sim, "fans", None) or [] if getattr(f, "enabled", True)]
    ports = [q for q in getattr(sim, "ports", None) or [] if getattr(q, "enabled", True)]
    return {
        "heater power": _round(sum(_round(getattr(h, "power", None)) for h in heaters)),
        "fans": tuple(sorted((str(getattr(f, "mode", "")), _round(getattr(f, "value", None)),
                              _round(getattr(f, "diameter", None))) for f in fans)),
        "ports (jets and vents)": tuple(sorted((str(getattr(q, "kind", "")), _round(getattr(q, "speed", None)),
                                        _round(getattr(q, "size", None)), _round(getattr(q, "temperature", None)))
                                       for q in ports)),
        "heated walls": tuple(sorted((str(getattr(w, "side", "")), str(getattr(w, "mode", "")),
                                      _round(getattr(w, "value", None))) for w in walls)),
        "buoyancy": bool(heat_on and getattr(th, "buoyancy", False)),
    }


def _inputs_sentence(sims: Mapping[str, Any], unit_w: str) -> str | None:
    """The warning for closed-domain runs driven by different inputs, or None when they match."""
    sigs = [driving_inputs(s) for s in sims.values()]
    keys = [k for k in sigs[0] if len({repr(v[k]) for v in sigs}) > 1]
    if not keys:
        return None
    words = []
    for k in keys:
        if k == "heater power":
            vals = sorted({v[k] for v in sigs})
            words.append(f"heater power ({_join([f'{x:.4g}' for x in vals])} {unit_w})")
        elif k == "buoyancy":
            words.append("buoyancy (on in some runs, off in others)")
        else:
            words.append(k)
    return (f"These closed-domain runs are driven by different inputs: {_join(words)}. A room has no free stream, "
            "so these settings play the part the speed plays in a wind tunnel: the differences include their "
            "effect, not only the design's.")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("s" if n != 1 else "")


def comparison_warnings(results: Mapping[str, Result] | Sequence[Result],
                        conditions: Mapping[str, Any] | None = None) -> list[str]:
    """Sentences saying why compared runs differ for reasons other than the design.

    ``conditions`` maps sim id -> the ``Simulation`` behind each result, or only its
    ``FlowConditions`` (older callers). Only Navier-Stokes runs are judged (a polar or a
    wing sweep sweeps the angle by design), 2D and 3D runs each among themselves. A selection
    that mixes 2D and 3D results gets a sentence saying they are not comparable (the table
    compares only the baseline's dimension). The Compare page and the report show the same
    sentences.

    With the simulations, closed domains are judged as rooms: two rooms are never said to run
    at "different speeds" or Reynolds numbers (their speed is a reference scale, not a free
    stream); their driving inputs are compared instead (:func:`driving_inputs`), and a room
    compared with a wind-tunnel run gets its own sentence. With conditions only, the speed rule
    applies to every run as before.
    """
    res = _as_mapping(results)
    out: list[str] = []
    groups = comparison_groups(res)
    if len(groups) > 1:
        out.append(f"This selection mixes {len(groups[2])} 2D and {len(groups[3])} 3D results. 2D coefficients "
                   "are per unit span and 3D coefficients are on a reference area, so they are not comparable: "
                   "the table compares only the results of the baseline's dimension and shows the others as "
                   "not comparable.")
    conds: dict[str, Any] = {}
    sims: dict[str, Any] = {}
    for sid, c in (conditions or {}).items():
        conds[sid], sim = _split_context(c)
        if sim is not None:
            sims[sid] = sim
    for cls in (CFDResult, CFD3DResult):
        cfd_ids = [sid for sid, r in res.items() if isinstance(r, cls)]
        if len(cfd_ids) < 2:
            continue
        closed = [sid for sid in cfd_ids if sid in sims and physics.is_closed_domain(sims[sid])]
        tunnel = [sid for sid in cfd_ids if sid in sims and not physics.is_closed_domain(sims[sid])]
        if closed and tunnel:
            out.append(f"This selection compares {_plural(len(closed), 'closed-domain run')} (a room or a cavity) "
                       f"with {_plural(len(tunnel), 'wind-tunnel run')}. A room has no free stream, so its speed "
                       "and Reynolds number are reference scales, not flow conditions: compare rooms with rooms and "
                       "tunnel runs with tunnel runs.")
            continue
        if closed and len(closed) == len(cfd_ids):
            differ = [k for k in conditions_differ({sid: conds.get(sid) for sid in cfd_ids}) if k != "speed"]
            if differ:
                out.append(f"These simulations ran at different {_join([_WHAT[k] for k in differ])}, so the "
                           "differences include the effect of the flow conditions, not only of the geometry.")
            msg = _inputs_sentence({sid: sims[sid] for sid in cfd_ids}, "W/m" if cls is CFDResult else "W")
            if msg:
                out.append(msg)
            continue
        differ = []
        if conditions is not None:
            differ = conditions_differ({sid: conds.get(sid) for sid in cfd_ids})
            if differ:
                out.append(f"These simulations ran at different {_join([_WHAT[k] for k in differ])}, so the "
                           "differences include the effect of the flow conditions, not only of the geometry.")
        mism = reynolds_mismatch({sid: res[sid] for sid in cfd_ids})
        if mism and all(_simulates_real_re(res[sid]) for sid in mism):
            # Every run simulates its own real Reynolds number at the same conditions, so the
            # difference comes from the designs' reference lengths (a longer car at the same
            # speed): that is part of the design comparison, not a modelling artefact. The
            # sedan and fastback templates differ by 4.3 % this way and used to get the
            # warning on the first comparison most users make.
            mism = {}
        if mism and not ({"speed", "fluid"} & set(differ)):
            from ..core.units import fmt_reynolds
            lo, hi = min(mism.values()), max(mism.values())
            out.append(f"These runs simulate different Reynolds numbers ({fmt_reynolds(lo)} to "
                       f"{fmt_reynolds(hi)}): the differences between them are not design effects alone. "
                       "Run them with the same Reynolds cap to compare the designs.")
    return out


__all__ = ["FLUID_REL_TOL", "INPUT_KEYS", "INTERVAL_KEYS", "REYNOLDS_RTOL", "Metric", "compare_rows",
           "comparison_groups", "comparison_warnings", "conditions_differ", "driving_inputs", "result_dimension",
           "reynolds_mismatch"]
