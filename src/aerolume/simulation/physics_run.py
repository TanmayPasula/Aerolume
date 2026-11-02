"""Placed physics during and after a run: monitors, convergence, SI outputs (DESIGN.md 10).

The solvers report heat rates, fan thrust and diagnostics in their own non-dimensional
units (2D heat in rho c_p U ΔT L per metre of span, 3D heat in conduction units k ΔT L;
thrust in rho U^2 L^(d-1)). The runners and the explore sessions turn them into what a
result stores (:class:`~aerolume.model.CFDResult` / ``CFD3DResult``: ``nusselt``,
``heat_rate`` [W or W/m], ``fans_report`` in SI, ``diagnostics`` with the heat balance in
W or W/m, the history series ``nu``, ``heat_rate`` and ``fan<i>_thrust|flow|velocity`` the
analysis layer reads, ``analysis.physics``) through this module, so 2D and 3D runs,
production and explore, say the same thing.

**Nusselt number.** With heated bodies: the mean over them of the body's own number,
Nu = Q L / (k (T_wall − T_amb) A) on the reference length (DESIGN.md 10.2; the solvers'
``heat_transfer``). A closed box without heated bodies (a cavity, a room heated at its
walls): the number of its hottest fixed-temperature wall, Nu = (Q / A_wall) H / (k ΔT_ref)
with H the length of Ra and ΔT_ref the spread of the set temperatures -- the definition of
the de Vahl Davis (1983) and Rayleigh-Benard benchmarks (for the square cavity Nu = Pe Q in
2D solver units, as validated in theory_ns2d.md 21.1).

**Convergence.** :func:`monitor_verdict` judges the plan's monitors with the runner's rules
(:func:`.runner._judge`): the Nusselt number to ``NU_RTOL_FACTOR`` times the plan's
tolerance, fans and the kinetic energy to the tolerance itself. A *steady* monitor (a
cavity settling exponentially) must also stop drifting to a tenth of its tolerance between
the halves of the window, because a monotone approach can sit within the tolerance of its
current value for a long time while still far from the asymptote.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import units
from ..model import Stat

__all__ = ["PLANNED_PREFIX", "STORAGE_UNSETTLED", "MonitorLog", "apply_measured_thermal", "chunk_samples",
           "fans_report_si", "fluid_temperatures", "heat_outputs", "measured_boussinesq", "merge_series",
           "monitor_verdict", "physics_history_si", "storage_share", "trace_payload", "warnings_after_run"]

#: A monitored quantity counts as steady once its rms is below this share of its tolerance; it
#: must then also drift by less than STEADY_DRIFT of the tolerance between the window halves.
STEADY_DRIFT = 0.1


def _scales(plan):
    return getattr(plan, "physics", None)


def _placed(plan):
    return getattr(plan, "placed", None)


def _rho(plan) -> float:
    return float(plan.fluid.density)


# ---------------------------------------------------------------------------
# per-step series in SI
# ---------------------------------------------------------------------------

def _wall_geometry(plan) -> dict[str, float]:
    """Area [m² / m] of each domain side of a plan (solved domain)."""
    if getattr(plan, "dimension", 2) == 3:
        x0, x1, y0, y1, z0, z1 = plan.extent3
        lx, ly, lz = x1 - x0, y1 - y0, z1 - z0
        return {"left": ly * lz, "right": ly * lz, "front": lx * lz, "back": lx * lz, "bottom": lx * ly,
                "top": lx * ly}
    x0, x1, y0, y1 = plan.extent
    return {"left": y1 - y0, "right": y1 - y0, "bottom": x1 - x0, "top": x1 - x0}


def hot_wall(plan) -> str | None:
    """The hottest fixed-temperature side of a closed box without heated bodies (whose Nusselt
    number the run reports), or None."""
    s, pl = _scales(plan), _placed(plan)
    if s is None or not s.thermal or pl is None or pl.thermal is None or not plan.closed:
        return None
    if any(m != "adiabatic" for m, _ in pl.body_thermal):
        return None
    walls = pl.thermal.walls
    items = [(w.side, w.mode, w.value) for w in walls] if isinstance(walls, list) else \
        [(k, m, v) for k, (m, v) in dict(walls).items()]
    temps = [(v, side) for side, m, v in items if m == "temperature"]
    return max(temps)[1] if temps else None


def physics_history_si(solver, plan) -> dict[str, np.ndarray]:
    """Per-step placed-physics series in SI, keyed for ``CFDResult.history`` / analysis.physics:
    ``t``; ``nu`` (see the module docstring), ``heat_rate`` [W or W/m] from the bodies into the
    fluid (from the hot wall in a closed box without heated bodies); per fan ``fan<i>_thrust``
    [N or N/m], ``fan<i>_velocity`` [m/s] and ``fan<i>_flow`` [m³/s or m²/s]. Empty when the run
    has none. Series the solver keeps per step only; the 3D walls come from chunk samples
    (:class:`MonitorLog`)."""
    s, pl = _scales(plan), _placed(plan)
    if pl is None:
        return {}
    dim = 3 if getattr(plan, "dimension", 2) == 3 else 2
    ph = solver.physics_history()
    if not ph or "t" not in ph:
        return {}
    U, L, rho = plan.speed, plan.length_ref, _rho(plan)
    out: dict[str, np.ndarray] = {"t": np.asarray(ph["t"], dtype=np.float64)}
    n = len(out["t"])
    if s is not None and s.thermal and pl.thermal is not None:
        hu = s.heat_unit
        if dim == 2:
            ht = solver.heat_transfer()
            pe = float(ht.get("pe", s.pe))
            # conducted heat (bodies, walls) in the capped flow's own Pe: Q = Nu k dT P / L
            cu = s.conduction_unit(pe)
            nus, heat = [], np.zeros(n)
            for b in ht["bodies"]:
                key = b["name"] or ""
                if b["mode"] == "temperature":
                    q = np.asarray(ph.get(f"heat:{key}", np.full(n, np.nan)), dtype=np.float64)
                    heat = heat + q
                    dth = float(b["theta_wall"])
                    if abs(dth) > 1e-12 and b["perimeter"] > 0:
                        nus.append(pe * q / (dth * b["perimeter"]))
                elif b["mode"] == "heat_flux":
                    heat = heat + float(b["heat_rate"])
                    tw = np.asarray(ph.get(f"theta_wall:{key}", np.full(n, np.nan)), dtype=np.float64)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        g = float(b["heat_rate"]) * pe / max(float(b["perimeter"]), 1e-300)
                        nus.append(np.where(np.abs(tw) > 1e-12, g / tw, np.nan))
            if nus:
                out["nu"] = np.mean(np.vstack(nus), axis=0)
                out["heat_rate"] = heat * cu
            else:
                side = hot_wall(plan)
                if side is not None and f"heat:{side}" in ph:
                    q = np.asarray(ph[f"heat:{side}"], dtype=np.float64)
                    area = _wall_geometry(plan)[side]
                    out["nu"] = q * pe * s.length_ra / area
                    out["heat_rate"] = q * cu
        else:
            if "nu" in ph:
                out["nu"] = np.asarray(ph["nu"], dtype=np.float64)
            if "heat_rate" in ph and any(m != "adiabatic" for m, _ in pl.body_thermal):
                out["heat_rate"] = np.asarray(ph["heat_rate"], dtype=np.float64) * hu
    fans = list(pl.fans_si)
    for i, f in enumerate(fans):
        if dim == 2:
            name = pl.fans[i].name or f"fan{i}"
            thrust = ph.get(f"thrust:{name}")
            ud = ph.get(f"disc_velocity:{name}")
            if thrust is None or ud is None:
                continue
            width = float(f.diameter)
            out[f"fan{i}_thrust"] = np.asarray(thrust, dtype=np.float64) * rho * U * U * L
            out[f"fan{i}_velocity"] = np.asarray(ud, dtype=np.float64) * U
            out[f"fan{i}_flow"] = out[f"fan{i}_velocity"] * width
        else:
            if f"fan{i}_thrust" not in ph:
                continue
            out[f"fan{i}_thrust"] = np.asarray(ph[f"fan{i}_thrust"], dtype=np.float64) * rho * U * U * L * L
            out[f"fan{i}_velocity"] = np.asarray(ph[f"fan{i}_velocity"], dtype=np.float64) * U
            out[f"fan{i}_flow"] = np.asarray(ph[f"fan{i}_flow"], dtype=np.float64) * U * L * L
    return out if len(out) > 1 else {}


def chunk_samples(solver, plan) -> dict[str, float]:
    """Quantities the solver does not record per step, sampled once per chunk (one host sync):
    the mean kinetic energy of the fluid [J/kg = m²/s²] of a closed box ("ke"), the rise of the
    warmest and of the mean fluid above the ambient [K] of a heated one ("t_rise_max",
    "t_rise_mean": rises, not °C, so that a relative tolerance means something) and, in 3D, the
    hot wall's Nusselt number and heat rate of a closed box ("nu_wall", "heat_wall")."""
    out: dict[str, float] = {}
    mons = getattr(plan, "monitors", ())
    want_t = "t_rise_max" in mons or "t_rise_mean" in mons
    if "ke" in mons or want_t:
        import torch
        with torch.no_grad():
            m = solver.field("mask")
            fluid = (1.0 - m).clamp_min(0.0)
            n_fluid = fluid.sum().clamp_min(1.0)
            if "ke" in mons:
                ke = 0.5 * (solver.field("speed") ** 2 * fluid).sum() / n_fluid
                out["ke"] = float(ke) * plan.speed ** 2
            s = _scales(plan)
            if want_t and s is not None and s.thermal:
                theta = solver.field("temperature" if getattr(plan, "dimension", 2) == 3 else "theta")
                dt_ref = float(s.delta_t)
                out["t_rise_mean"] = float((theta * fluid).sum() / n_fluid) * dt_ref
                out["t_rise_max"] = float(theta.masked_fill(m >= 0.5, float("-inf")).max()) * dt_ref
    if getattr(plan, "dimension", 2) == 3:
        side = hot_wall(plan)
        s = _scales(plan)
        if side is not None and s is not None:
            w = solver.wall_heat_rates()
            if side in w:
                q = float(w[side])
                area = _wall_geometry(plan)[side]
                L = plan.length_ref
                out["nu_wall"] = q * L * s.length_ra / area
                out["heat_wall"] = q * s.heat_unit
    return out


def measured_speed_factor(solver, plan, t_first: float) -> float | None:
    """The run's CFL speed factor (max|u| + max|v| (+ max|w|)) / U from the solver diagnostics,
    as the harmonic mean over the rows since ``t_first`` (what ``cfl h / mean dt`` measures when
    the CFL bound sets every step), or None without diagnostics.

    With placed physics the time step is often set by another bound -- the temperature's
    positivity limit, a buoyant parcel, a jet's own speed -- and ``cfl h / mean dt`` then reads
    that bound as a fast flow. Measured on the Draft square cavity: 6.27 against the real 0.231
    at Ra 10^3 (diffusion-limited) and 2.32 against 0.372 at Ra 10^4, which inflated every later
    estimate of the class; on a CFL-limited fan room both read 12.57. The diagnostics record
    the real maxima (2D: cfl = dt (max|u| + max|v|) / h; 3D: the ``umax`` column).
    """
    if not hasattr(solver, "diagnostics"):
        return None
    try:
        d = solver.diagnostics()
    except Exception:  # noqa: BLE001 - a calibration must never fail a run
        return None
    t = np.asarray(d.get("t", ()), dtype=np.float64)
    if not t.size:
        return None
    if "umax" in d:
        u = np.asarray(d["umax"], dtype=np.float64)
    elif "cfl" in d and "dt" in d:
        u = np.asarray(d["cfl"], dtype=np.float64) * float(plan.solver_config.h) / np.asarray(d["dt"], dtype=np.float64)
    else:
        return None
    sel = (t >= t_first) & np.isfinite(u) & (u > 0)
    if sel.sum() < 3:
        return None
    return float(sel.sum() / np.sum(1.0 / u[sel]))


class MonitorLog:
    """Chunk samples (:func:`chunk_samples`) of a run, merged into its history at the end."""

    def __init__(self) -> None:
        self.t: list[float] = []
        self.rows: list[dict[str, float]] = []

    def add(self, t: float, values: dict[str, float]) -> None:
        if values:
            self.t.append(float(t))
            self.rows.append(dict(values))

    def series(self) -> dict[str, np.ndarray]:
        if not self.t:
            return {}
        keys = sorted({k for r in self.rows for k in r})
        out = {"t": np.asarray(self.t)}
        for k in keys:
            out[k] = np.asarray([r.get(k, np.nan) for r in self.rows], dtype=np.float64)
        return out

    def clear(self) -> None:
        self.t.clear()
        self.rows.clear()


def merge_series(h: dict[str, np.ndarray], phys: dict[str, np.ndarray], chunks: dict[str, np.ndarray]) -> dict:
    """The force history ``h`` with the physics series aligned to its times.

    Per-step series share the solver's time steps (copied where the times agree, linearly
    interpolated otherwise); chunk samples are interpolated onto the step times (inside their
    sampled span; NaN outside it, so nothing is extrapolated). The 3D hot-wall samples become
    the ``nu`` / ``heat_rate`` series when the solver has no per-step ones.
    """
    out = dict(h)
    t = np.asarray(h.get("t", np.zeros(0)), dtype=np.float64)
    if not t.size:
        return out

    def onto(ts: np.ndarray, x: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if ts.size == t.size and np.allclose(ts, t, rtol=0, atol=1e-12 * max(1.0, float(abs(t).max()))):
            return x.copy()
        if ts.size < 2:
            return np.full(t.size, x[-1] if x.size else np.nan)
        y = np.interp(t, ts, x)
        y[(t < ts[0]) | (t > ts[-1])] = np.nan
        return y

    for k, v in phys.items():
        if k != "t":
            out[k] = onto(phys["t"], v)
    if chunks:
        ct = chunks["t"]
        for k, v in chunks.items():
            if k == "t":
                continue
            name = {"nu_wall": "nu", "heat_wall": "heat_rate"}.get(k, k)
            if name in out and name not in ("ke",) and k in ("nu_wall", "heat_wall"):
                continue
            out[name] = onto(ct, v)
    return out


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

def monitor_verdict(h: dict[str, np.ndarray], plan, t_min: float, *, rtol: float, laminar: bool) -> dict:
    """The verdict on the plan's monitors over ``t >= t_min`` (the forces are judged by the runner's
    own ``_verdict``). Returns ``{"stop", "drifting", "judged": {name: judgement}, "worst": name}``."""
    from .plan import NU_RTOL_FACTOR
    from .runner import _judge
    out = {"stop": True, "drifting": False, "judged": {}, "worst": None}
    mons = [m for m in getattr(plan, "monitors", ()) if m in h]
    if not mons:
        out["stop"] = not getattr(plan, "monitors", ())
        return out
    t = np.asarray(h["t"], dtype=np.float64)
    for m in mons:
        x = np.asarray(h[m], dtype=np.float64)
        ok = np.isfinite(x)
        if ok.sum() < 3:
            out["stop"] = False
            out["judged"][m] = {"state": "insufficient", "reason": "No samples yet."}
            out["worst"] = out["worst"] or m
            continue
        r = rtol * (NU_RTOL_FACTOR if m == "nu" else 1.0)
        scale = float(np.nanmax(np.abs(x[ok]))) if ok.any() else 0.0
        j = _judge(t[ok], x[ok], t_min, rtol=r, atol=max(1e-12, 1e-6 * scale), laminar=laminar)
        if j["state"] == "converged" and math.isfinite(j.get("drift", math.nan)):
            steady = math.isfinite(j.get("amplitude_ratio", math.nan)) is False
            if steady and abs(j["drift"]) > STEADY_DRIFT * j["tol"]:
                j = dict(j, state="transient", drifting=True,
                         reason=(f"Still settling ({j['drift']:+.3g} between the halves of the window; steady values "
                                 f"must hold to ±{units.sig(STEADY_DRIFT * j['tol'], 2)})."))
        out["judged"][m] = j
        if j["state"] != "converged":
            out["stop"] = False
            out["worst"] = out["worst"] or m
        out["drifting"] = out["drifting"] or bool(j.get("drifting"))
    return out


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _window(h: dict, key: str, t0: float) -> np.ndarray:
    t = np.asarray(h["t"], dtype=np.float64)
    x = np.asarray(h[key], dtype=np.float64)
    sel = (t >= t0) & np.isfinite(x)
    return x[sel]


def fans_report_si(solver, plan, h: dict, t0: float, averaging: bool) -> list[dict]:
    """The fans of a run in SI (canonical keys of ``analysis.physics.FAN_KEYS``): name, mode, thrust
    [N or N/m] with its standard error from the thrust history, pressure_jump [Pa], volume_flow
    [m³/s or m²/s], disc_velocity and induced_velocity [m/s], ideal_power [W or W/m], target
    (the set value in its own unit) and, in 3D, in_domain (the share of the disc inside the
    solved domain)."""
    pl = _placed(plan)
    if pl is None or not pl.fans:
        return []
    from ..analysis import coefficient_stats
    dim = 3 if getattr(plan, "dimension", 2) == 3 else 2
    U, L, rho = plan.speed, plan.length_ref, _rho(plan)
    rep = solver.fans_report(mean=averaging)
    out = []
    lf = L if dim == 2 else L * L
    for i, (r, f) in enumerate(zip(rep, pl.fans_si)):
        e = {"name": f.name, "mode": f.mode,
             "thrust": float(r.get("thrust", math.nan)) * rho * U * U * lf,
             "pressure_jump": float(r.get("pressure_jump", math.nan)) * rho * U * U,
             "volume_flow": float(r.get("flow_rate", r.get("flow", math.nan))) * U * lf,
             "disc_velocity": float(r.get("disc_velocity", math.nan)) * U,
             "induced_velocity": float(r.get("induced_velocity", math.nan)) * U,
             "ideal_power": float(r.get("power", math.nan)) * rho * U ** 3 * lf,
             "target": float(f.value), "diameter": float(f.diameter)}
        if "in_domain" in r:
            e["in_domain"] = float(r["in_domain"])
        key = f"fan{i}_thrust"
        if key in h and "t" in h:
            st = coefficient_stats(np.asarray(h["t"]), np.asarray(h[key]), t0)
            if math.isfinite(st.mean):
                e["thrust"] = float(st.mean)
                e["thrust_stderr"] = float(st.stderr)
                e["thrust_reliable"] = bool(st.stderr_reliable)
        out.append(e)
    return out


def heat_outputs(solver, plan, h: dict, t0: float, averaging: bool) -> dict[str, Any]:
    """Heat transfer of a finished run: ``nusselt`` (Stat), ``heat_rate`` [W or W/m], ``per_body``
    ({name: {"nu", "heat_rate"}}), ``diagnostics`` (the heat-balance terms in W or W/m) and
    ``notes`` and ``warnings`` (a temperature field still warming up: :func:`storage_share`). Empty
    fields when the run has no heat transfer."""
    from ..analysis import coefficient_stats
    s, pl = _scales(plan), _placed(plan)
    out: dict[str, Any] = {"nusselt": Stat(), "heat_rate": math.nan, "per_body": {}, "diagnostics": {},
                           "notes": [], "warnings": []}
    if s is None or not s.thermal or pl is None or pl.thermal is None:
        return out
    dim = 3 if getattr(plan, "dimension", 2) == 3 else 2
    t = np.asarray(h.get("t", np.zeros(0)), dtype=np.float64)
    if "nu" in h and t.size:
        out["nusselt"] = coefficient_stats(t, np.asarray(h["nu"], dtype=np.float64), t0)
    if "heat_rate" in h and t.size:
        w = _window(h, "heat_rate", t0)
        out["heat_rate"] = float(w.mean()) if w.size else math.nan
    hu = s.heat_unit
    try:
        ht = solver.heat_transfer(mean=averaging)
    except Exception:  # noqa: BLE001 - a report must never cost the run its forces
        ht = {"bodies": [], "walls": {}}
    # conducted heat (bodies, walls) converts with the solver's own Pe (PhysicsScales.conduction_unit):
    # in a capped 2D run rho cp U dT L would report the more conductive simulated fluid's rate
    cu = s.conduction_unit(ht.get("pe") if dim == 2 else None)
    for b in ht.get("bodies", []):
        if b.get("mode", "adiabatic") == "adiabatic":
            continue
        out["per_body"][str(b["name"])] = {"nu": float(b.get("nusselt", math.nan)),
                                           "heat_rate": float(b.get("heat_rate", math.nan)) * cu,
                                           "t_wall": float(s.celsius(float(b.get("theta_wall", math.nan))))}
    unit = "W" if dim == 3 else "W/m"
    side = hot_wall(plan)
    if side is not None:
        out["notes"].append(f"Nusselt number of the {side} wall (the hottest), Nu = (Q/A) H / (k ΔT) with H = "
                            f"{units.sig(s.length_ra, 3)} m and ΔT = {units.sig(s.delta_t, 3)} K, the definition of "
                            "the cavity benchmarks; the heat rate is the heat through that wall.")
    walls = {str(k): float(v) * cu for k, v in (ht.get("walls", {}) or {}).items() if math.isfinite(float(v))}
    if walls:
        # in a tunnel the left and right sides are the inlet and the outlet (the heat the flow carries
        # across them); values far below the largest are round-off (a denormal "-5.7e-39" inlet)
        q = abs(out["heat_rate"]) if math.isfinite(out["heat_rate"]) else 0.0
        big = max(max(abs(v) for v in walls.values()), q)        # the bodies' heat sets the scale too
        names = {} if getattr(plan, "closed", False) else {"left": "inlet", "right": "outlet"}
        parts = [f"{names.get(k, k + ' wall')} {units.sig(v, 3)}" for k, v in walls.items()
                 if big > 0 and abs(v) > 1e-9 * big]
        if parts:
            out["notes"].append(f"Heat through the domain sides ({unit}, positive into the fluid): "
                                + ", ".join(parts) + ".")
    try:
        bal = solver.heat_balance(mean=averaging) if dim == 2 else solver.heat_balance()
    except Exception:  # noqa: BLE001
        bal = {}
    # The balance is the simulated flow's own and must close, so every term converts with one unit,
    # rho cp U dT L (2D; heaters and advected heat are right in it). In a capped run that is the
    # more conductive simulated fluid's balance: its body and wall terms exceed the heat rates above
    # (real conductivity) by Re_real / Re_sim, which the note below states.
    diag = {k: float(v) * hu for k, v in bal.items() if v is not None and math.isfinite(float(v))}
    ratio = (hu / cu) if (dim == 2 and cu > 0) else 1.0
    if diag and abs(ratio - 1.0) > 1e-6 and any(abs(diag.get(k, 0.0)) > 0 for k in ("heat_in_bodies", "heat_in_walls")):
        out["notes"].append(
            f"Reynolds number capped: the Nusselt number comes from the capped flow (Péclet number "
            f"{units.sig(float(ht.get('pe', math.nan)), 4)}) and the heat rates are Q = Nu k ΔT P / L with the real "
            f"conductivity. The heat balance is the simulated flow's own (it closes in its units); its body and "
            f"wall terms are {units.sig(ratio, 4)}× those heat rates, because the capped flow behaves as a fluid "
            f"{units.sig(ratio, 4)}× more conductive.")
    heaters = [q for q in getattr(pl, "heaters_si", []) or [] if getattr(q, "enabled", True)]
    if heaters:
        diag["heat_sources_set"] = float(sum(float(q.power) for q in heaters))
    if not math.isfinite(out["heat_rate"]) and not math.isfinite(out["nusselt"].mean):
        # Nothing has a fixed temperature to define a Nusselt number (a room warmed by a heater,
        # a tunnel with a heat source): the headline is the heat the placed heat inputs put into
        # the fluid, measured from the solver's own balance -- never NaN while heat is on, so the
        # result is recognisably a heat-transfer result (analysis.physics.has_thermal).
        put_in = [diag[k] for k in ("heat_in_bodies", "heat_in_walls", "heat_in_sources") if k in diag]
        if put_in:
            out["heat_rate"] = float(sum(put_in))
            what = []
            if "heat_sources_set" in diag:
                what.append(f"the heaters ({units.sig(diag.get('heat_in_sources', math.nan), 4)} {unit} measured, "
                            f"{units.sig(diag['heat_sources_set'], 4)} {unit} set)")
            if abs(diag.get("heat_in_walls", 0.0)) > 0:
                what.append(f"the domain walls ({units.sig(diag['heat_in_walls'], 4)} {unit})")
            out["notes"].append(
                f"Heat rate {units.sig(out['heat_rate'], 4)} {unit}: the heat put into the fluid by "
                + (" and ".join(what) if what else "the heated surfaces")
                + ". No Nusselt number: nothing in the run is held at a fixed temperature to define one; "
                "the heat balance and the fluid temperatures are the result.")
    temps = fluid_temperatures(solver, plan, averaging)
    if temps:
        averaged = bool(temps.pop("_averaged", False))
        diag.update(temps)
        out["notes"].append(
            f"Fluid temperature ({'time-averaged' if averaged else 'last step'}): mean "
            f"{units.sig(temps['t_fluid_mean'], 4)} °C, min {units.sig(temps['t_fluid_min'], 4)} °C, max "
            f"{units.sig(temps['t_fluid_max'], 4)} °C (ambient {units.sig(s.t_ambient, 4)} °C).")
    stored = storage_share(diag)
    if stored is not None and stored > STORAGE_UNSETTLED:
        # the run ended while the fluid was still warming up: the Draft hot sphere stored 69 % of its
        # heat and reported Nu 53.51 ± 0.05 (critic round 3.4), a precision its mean did not have
        nu = out["nusselt"]
        if math.isfinite(nu.mean):
            nu.stderr_reliable = False
        out["warnings"].append(
            f"The temperature field had not settled: {100 * stored:.0f} % of the heat put into the fluid was still "
            f"being stored in it at the end (heat balance, {unit}), so the "
            + ("Nusselt number and heat rate were" if math.isfinite(nu.mean) else "fluid temperatures were")
            + " still changing and their error bars understate the uncertainty. Run longer (Duration) before "
            "relying on them.")
    if diag:
        out["diagnostics"] = diag
    return out


#: Share of the heat put into the fluid that may still be going into storage (warming the fluid)
#: when a thermal run ends before its result counts as thermally settled (:func:`storage_share`).
STORAGE_UNSETTLED = 0.1


def storage_share(diag: dict | None) -> float | None:
    """Share of the heat put into the fluid (bodies, walls, heaters, jets, inlet) that the heat
    balance's storage term still absorbs, or None without a finite storage rate or heat input.
    Near 0 at a thermal steady state (the validated hot sphere: 1 − 0.9999), near 1 at the start."""
    d = diag or {}
    try:
        store = float(d.get("heat_storage_rate", math.nan))
    except (TypeError, ValueError):
        return None
    total = sum(float(v) for k, v in d.items() if k.startswith("heat_in") and math.isfinite(float(v)) and float(v) > 0)
    if not (math.isfinite(store) and total > 0):
        return None
    return store / total


def fluid_temperatures(solver, plan, averaging: bool) -> dict[str, float]:
    """Mean, min and max temperature [°C] of the fluid (solid cells excluded) of a thermal run, on
    the solver's own grid: from the time-averaged field while averaging, else the last step.

    Keys ``t_fluid_mean`` (volume-weighted by the fluid fraction 1 − mask), ``t_fluid_min``,
    ``t_fluid_max`` (over cells that are mostly fluid) and ``_averaged`` (bool, popped by the
    caller). Empty when the run carries no temperature or the fields cannot be read. A room
    heated by a radiator is judged by these, so they come from the full-resolution field, not
    from the down-sampled block the 3D viewer gets (whose pooling would flatten the maximum).
    """
    s = _scales(plan)
    if s is None or not s.thermal:
        return {}
    dim = 3 if getattr(plan, "dimension", 2) == 3 else 2
    name = "temperature" if dim == 3 else "theta"
    try:
        import torch
        with torch.no_grad():
            theta, averaged = None, False
            if averaging:
                try:
                    theta = (solver.mean_fields((name,)) if dim == 3 else solver.mean_fields())[name]
                    averaged = True
                except Exception:  # noqa: BLE001 - no averages yet: the last step
                    theta = None
            if theta is None:
                theta = solver.field(name)
            theta = theta.double()
            m = solver.field("mask").double().clamp(0.0, 1.0)
            w = 1.0 - m
            fluid = m < 0.5
            if float(w.sum()) <= 0 or not bool(fluid.any()):
                return {}
            mean = float((theta * w).sum() / w.sum())
            lo, hi = float(theta[fluid].min()), float(theta[fluid].max())
            # the largest excursion from the ambient at any recorded instant (mean field and last
            # step): what the Boussinesq check needs, since a mean field hides a hot plume's peaks
            dev = max(abs(lo), abs(hi))
            if averaged:
                last = solver.field(name).double()[fluid]
                dev = max(dev, float(last.abs().max()))
    except Exception:  # noqa: BLE001 - a report must never cost the run its other outputs
        return {}
    vals = [s.celsius(x) for x in (mean, lo, hi)]
    if not all(math.isfinite(v) for v in vals) or not math.isfinite(dev):
        return {}
    return {"t_fluid_mean": float(vals[0]), "t_fluid_min": float(vals[1]), "t_fluid_max": float(vals[2]),
            "t_fluid_dev_max": float(dev * s.delta_t), "_averaged": averaged}


def measured_boussinesq(result, plan) -> tuple[list[str], list[str], list[str]]:
    """(warnings to drop, notes to add, warnings to add) once a run has measured its temperatures.

    The planner's ΔT_ref for a heater or a heat flux is an order-of-magnitude rise (q L / (k Nu)
    from a laminar correlation) and can be far from what the run produces: the Draft room with a
    500 W/m heater was planned with ΔT 72.3 K (β·ΔT 0.25, a Boussinesq warning) and its hottest
    fluid was 5.6 K above the ambient (β·ΔT 0.019). The plan's warning is replaced by the measured
    value when that is inside the range; a run that exceeds the range without having been warned
    gets the warning. Both say which ΔT they use.
    """
    from .physics import BOUSSINESQ_LIMIT, boussinesq_warning
    s = _scales(plan)
    diag = getattr(result, "diagnostics", None) or {}
    dev = diag.get("t_fluid_dev_max")
    if s is None or not s.thermal or not s.buoyancy or s.props is None or dev is None or not math.isfinite(dev):
        return [], [], []
    beta = float(s.props.beta)
    measured = beta * float(dev)
    planned = boussinesq_warning(s)
    warned = planned in (getattr(result, "warnings", None) or [])
    reached = f"the fluid reached {units.sig(dev, 3)} K from the ambient"
    if warned and measured <= BOUSSINESQ_LIMIT:
        note = (f"β·ΔT = {measured:.2g} measured ({reached}), inside the Boussinesq range (below about "
                f"{BOUSSINESQ_LIMIT:g}); the plan's temperature scale ΔT {units.sig(s.delta_t, 3)} K was an estimate "
                "of the rise, not a measurement.")
        return [planned], [note], []
    if not warned and measured > BOUSSINESQ_LIMIT:
        warn = (f"β·ΔT = {measured:.2f} measured ({reached}): above about {BOUSSINESQ_LIMIT:g} the Boussinesq "
                "approximation (constant density except in the buoyancy force) loses accuracy, so the "
                "buoyancy-driven flow is approximate.")
        return [], [], [warn]
    return [], [], []


#: Plan warnings whose number depends on ΔT_ref, by :func:`physics.warning_key`: once a run has
#: measured the rise a heater or a heat flux really produced, they are re-derived from it.
_DT_WARNING_KEYS = ("rayleigh", "thermal-resolution", "buoyancy-dominated", "regime-not-validated")
#: Plan notes that state the planned ΔT_ref and the groups on it (physics.thermal_checks), which a
#: measured run keeps, labelled as the planned estimate.
_DT_NOTE_PREFIXES = ("Temperature scale ΔT = ", "Buoyancy: Boussinesq", "β·ΔT = ")
PLANNED_PREFIX = "Planned estimate: "


def apply_measured_thermal(result, plan):
    """Re-state a thermal result on the temperature rise it measured; returns the measured
    :class:`~aerolume.simulation.physics.PhysicsScales` (None when nothing changed).

    The planner's ΔT_ref for a heater or a heat flux is q L / (k Nu) from a laminar correlation,
    an order of magnitude: the Draft fan room (300 W/m heater) was planned at 43.4 K while its
    warmest fluid was 3.48 K above the ambient, so the plan's Ra 1.3×10¹¹ ("turbulent") and its
    other ΔT-based warnings described a different room (critic round 3.3). Here, with the measured
    rise (:func:`physics.measured_delta_t`):

    * the Boussinesq warning follows :func:`measured_boussinesq` (the largest excursion seen);
    * the Rayleigh, thermal-resolution and buoyancy-dominated warnings are re-derived with
      :func:`physics.thermal_checks` on the measured scales, on the plan's own grid
      (``plan.thermal_grid``), and replace the planned ones;
    * the plan's temperature-scale and buoyancy notes stay, prefixed "Planned estimate: ", and a
      note states ΔT, Gr, Ra (and Ri) measured next to the planned values.

    A ΔT_ref set by the case (fixed wall, body or jet temperatures: the cavity's T_hot − T_cold)
    is exact, and only the Boussinesq check applies to it.
    """
    from .physics import (
        delta_t_is_estimate,
        fmt_number,
        merge_warnings,
        result_scales,
        thermal_checks,
        warning_key,
    )
    drop, add_notes, add_warnings = measured_boussinesq(result, plan)
    result.warnings = [w for w in result.warnings if w not in drop] + add_warnings
    result.notes.extend(add_notes)
    s = _scales(plan)
    if not delta_t_is_estimate(s):
        return None
    ms, how = result_scales(s, result)
    if how != "measured" or ms is None:
        return None
    grid = getattr(plan, "thermal_grid", None)
    keys = set(_DT_WARNING_KEYS) if grid else set(_DT_WARNING_KEYS) - {"thermal-resolution"}
    h_m, n_bl, bl_target = grid if grid else (float("nan"), float("nan"), 4.0)
    dim = 3 if getattr(plan, "dimension", 2) == 3 else 2
    _, fresh = thermal_checks(ms, plan.sim, h_m=h_m, n_bl=n_bl, bl_target=bl_target,
                              closed=bool(getattr(plan, "closed", False) or getattr(plan, "domain_mode", "") == "closed"),
                              live=False, dim=dim)
    fresh = [w for w in fresh if warning_key(w) in keys]
    kept = [w for w in result.warnings if warning_key(w) not in keys]
    result.warnings = merge_warnings(kept + fresh)
    measured_notes = set(add_notes)                  # this run's own β·ΔT line is a measurement, not a plan
    result.notes = [(PLANNED_PREFIX + n) if (n not in measured_notes and any(n.startswith(p) for p in _DT_NOTE_PREFIXES))
                    else n for n in result.notes]
    lra = f"{units.sig(ms.length_ra, 3)} m, the {ms.length_ra_basis}"
    txt = (f"Measured temperature difference ΔT = {units.sig(ms.delta_t, 3)} K: {ms.delta_t_basis.split(': ', 1)[-1]} "
           f"(the planned estimate was {units.sig(s.delta_t, 3)} K, the {s.delta_t_basis})")
    if ms.gr > 0:
        txt += (f". On the measured ΔT: Gr {fmt_number(ms.gr)}, Ra {fmt_number(ms.ra)} on {lra}"
                f" (planned estimate Gr {fmt_number(s.gr)}, Ra {fmt_number(s.ra)})")
        if ms.ri_flow is not None and s.ri_flow is not None:
            txt += f"; Richardson number Gr/Re² {fmt_number(ms.ri_flow)} (planned {fmt_number(s.ri_flow)})"
        txt += f"; convection {ms.regime.lower()}"
    result.notes.append(txt + ". The run itself used the planned scale only as its unit of temperature.")
    return ms


def warnings_after_run(plan, result) -> list[str]:
    """The plan's warnings as they read once ``result`` (a finished run of this plan's case) has
    measured its temperatures: the ΔT-dependent ones (Boussinesq, Rayleigh, thermal resolution,
    buoyancy-dominated, unvalidated regime) re-derived on the measured rise, exactly as
    :func:`apply_measured_thermal` re-states the result. For the Setup page: a heater room's
    "β·ΔT = 0.15 … loses accuracy" stayed on Setup after its run had measured β·ΔT 0.014
    (critic round 3.4). The plan's own list when the run measured nothing (or ΔT was set)."""
    import types
    warns = list(getattr(plan, "warnings", None) or [])
    diag = getattr(result, "diagnostics", None) or {}
    if not diag:
        return warns
    shadow = types.SimpleNamespace(warnings=list(warns), notes=list(getattr(plan, "notes", None) or []),
                                   diagnostics=diag)
    try:
        apply_measured_thermal(shadow, plan)
    except Exception:  # noqa: BLE001 - a page's warning list must never fail on an old result
        return warns
    return list(shadow.warnings)


def trace_payload(trace: dict, plan, *, step: int, t: float, axis: str | None = None,
                  pos: float | None = None) -> dict:
    """A captured solver step for the GUI's trace viewer: ``{"stages": {key: (rows, cols) array},
    "extent": metres, "step", "t", "notes"}`` in solver orientation and solver units.

    2D traces are the solver's cell-centred fields. A 3D trace is a volume per stage; the viewer
    shows planes, so each is cut at the slice the run is showing (``axis`` / ``pos`` [m], default:
    the plan's side view)."""
    stages: dict[str, np.ndarray] = {}
    notes = [(f"Solver units: velocities in U = {units.sig(plan.speed, 3)} m/s, lengths in L = "
              f"{units.sig(plan.length_ref, 3)} m, pressure in ρU².")]
    if getattr(plan, "dimension", 2) == 3:
        from .live3d import SLICE_AXES, slice_extent
        axis = axis if axis in SLICE_AXES else "y"
        ext3 = plan.extent3
        k = {"x": 0, "y": 1, "z": 2}[axis]
        p = plan.slice_positions.get(axis, 0.5 * (ext3[2 * k] + ext3[2 * k + 1])) if pos is None else float(pos)
        lo, hi = ext3[2 * k], ext3[2 * k + 1]
        n = (plan.grid3[0], plan.grid3[1], plan.grid3[2])[k]
        idx = int(min(max(math.floor((p - lo) / (hi - lo) * n), 0), n - 1))
        dim_index = {"x": 2, "y": 1, "z": 0}[axis]
        for key, arr in trace.items():
            a = np.asarray(arr.float().cpu().numpy() if hasattr(arr, "cpu") else arr, dtype=np.float32)
            if a.ndim == 3:
                stages[key] = np.take(a, idx, axis=dim_index)
        extent = slice_extent(axis, ext3)
        notes.append(f"3D step shown on the {axis} = {units.sig(p, 3)} m plane.")
    else:
        for key, arr in trace.items():
            a = np.asarray(arr.float().cpu().numpy() if hasattr(arr, "cpu") else arr, dtype=np.float32)
            if a.ndim == 2:
                stages[key] = a
        extent = tuple(float(e) for e in plan.extent)
    return {"stages": stages, "extent": tuple(float(e) for e in extent), "step": int(step), "t": float(t),
            "notes": notes}
