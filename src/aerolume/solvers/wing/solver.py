"""Finite-wing analysis: public entry points ``analyze_wing`` and ``wing_polar``.

This module ties the pieces together (DESIGN.md 7.6):

* geometry — :mod:`.geometry` panels the wing's cambered mean surface from its
  :class:`~aerolume.model.WingSpec` (airfoils via :class:`.polars.SectionPolars`);
* inviscid loads — :mod:`.vlm` (vortex lattice, the default) or :mod:`.llt`
  (nonlinear lifting line, Phillips & Snyder);
* viscous coupling (XFLR5-style strip theory) — every spanwise strip takes its
  profile drag from the 2D viscous polar of its airfoil at its own Reynolds
  number. With the lattice, the strip's angle is the one at which the section
  polar produces the strip's lift (inverting cl(alpha)); with the lifting line the
  section lift curve itself is part of the solution, so the strip angle is the
  solved effective angle and the lift is nonlinear up to the section cl_max.

Frames and signs (DESIGN.md 7.2): the wing is defined in its own frame (x aft,
y spanwise, z up). Angle of attack ``alpha`` rotates the wing about +y through
the moment reference by +alpha (nose up); sideslip ``beta`` rotates it about +z
by -beta first (positive beta = the yaw_deg convention of FlowConditions: the
free stream gains a +y component in the wing frame). Forces and moments are
reported in wind axes: CD along the free stream, CY (side) along +y, CL along +z;
C_roll, Cm (+ nose up), Cn are moments about +x, +y, +z through ``moment_ref``.
Coefficients use q * ref_area; Cm also ref_chord (MAC), C_roll and Cn ref_span.

Drag: CD = CDi + CDv. CDi is the Trefftz-plane (far-field) induced drag, CDv the
strip-integrated profile drag. Span efficiency e = CL^2 / (pi AR CDi) with AR =
span^2 / ref_area of the geometric span.

Limits (docs/theory_wing.md): thin-wing potential flow plus 2D strip theory —
no stall prediction beyond what the section polars contain (the lattice's lift
is linear; the lifting line clips each section at its cl_max), no fuselage, no
spanwise boundary-layer flow, Prandtl–Glauert below M 0.6.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...model import WING_METHODS, Fluid, WingPoint, WingResult, WingSpec
from .geometry import WingGeometry, build_geometry, check_wing, section_shape
from .llt import LiftingLine
from .polars import (
    ALPHA_MAX,
    ALPHA_MIN,
    PolarCancelled,
    PolarTable,
    SectionPolarError,
    SectionPolars,
    airfoil_key,
    blend_tables,
)
from .vlm import VortexLattice

ProgressFn = Callable[[float, str], None]

MACH_WARN = 0.6              # Prandtl–Glauert is trusted below this (DESIGN.md 7.6)
MACH_MAX = 0.9
ALPHA_LIMIT = 45.0           # |alpha| beyond this is rejected (thin-wing theory is meaningless there)
RE_FLOOR_FRACTION = 0.3      # strips below this fraction of the largest strip Re use the polar at the floor
RE_FLOOR = 2.0e4             # ... and never below this Reynolds number
EXTEND_STEP = 3.0            # deg added to a section polar whose computed range ends before the needed lift
EXTEND_ROUNDS = 8           # enough to walk any polar from its first estimate to ALPHA_MAX
LLT_SWEEP_WARN = 20.0        # deg of quarter-chord sweep beyond which the lifting line is outside its range
LLT_AR_WARN = 4.0
E_MIN_CL = 0.05              # span efficiency is reported only for |CL| at least this
TIP_EXCLUDE = 0.05           # strips this close to a tip (fraction of the semispan) are left out of the
                             # stall-onset estimate and the clipping warnings (see _tip_core)


class Cancelled(Exception):
    """Raised internally when the caller's cancel event is set."""


@dataclass
class _Setup:
    """Everything one sweep needs, built once."""
    wing: WingSpec
    geom: WingGeometry
    method: str
    viscous: bool
    beta_pg: float
    mach: float
    speed: float
    fluid: Fluid
    ref: np.ndarray
    s_ref: float
    c_ref: float
    b_ref: float
    re_strip: np.ndarray          # true strip Reynolds numbers
    re_polar: np.ndarray          # Reynolds numbers used for the polars (floored)
    af_a: list[str]               # airfoil keys per strip (blend from a ...
    af_b: list[str]               # ... to b with weight geom.strips.weight)
    alpha_l0: np.ndarray          # thin-airfoil zero-lift angle per strip [rad]
    cm0: np.ndarray               # thin-airfoil cm_c/4 per strip
    provider: SectionPolars | None
    notes: list[str]
    warnings: list[str]
    core: np.ndarray | None = None  # strips away from the tips (stall onset, clipping warnings)
    margins: dict[float, tuple[float, int]] = field(default_factory=dict)   # alpha -> (max margin, strip)
    tip_clipped: set = field(default_factory=set)                         # alphas where only tip strips clipped


def _rotation(alpha_deg: float, beta_deg: float) -> np.ndarray:
    """Wing-frame -> wind-frame rotation: yaw by -beta about z, then pitch by +alpha about y."""
    a, b = math.radians(alpha_deg), math.radians(beta_deg)
    rz = np.array([[math.cos(b), math.sin(b), 0.0], [-math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    return ry @ rz


def _tip_core(wing: WingSpec, y: np.ndarray) -> np.ndarray:
    """Strips farther than ``TIP_EXCLUDE`` of the semispan from a wing tip.

    Strip theory (and a lifting line in particular) fails in the last few percent
    of the span, where the tip vortex forms: a rounded tip modelled by sections whose
    chord falls to zero gets strong upwash from the neighbouring trailing leg, and the
    lifting line then reports local cl far above the inboard values (measured on the
    NACA TN 1270 wings: 1.2 at the tip against 0.45 inboard). Those strips still carry
    their (small) loads, but they must not decide the wing's stall onset.
    """
    ys = [s.y for s in wing.sections]
    if wing.symmetric:
        half = max(abs(ys[0]), abs(ys[-1]))
        dist = (half - np.abs(y)) / half
    else:
        dist = (ys[-1] - y) / (ys[-1] - ys[0])
    return dist >= TIP_EXCLUDE


def freestream_direction(alpha_deg: float, beta_deg: float = 0.0) -> np.ndarray:
    """Unit free-stream direction in the wing frame: (cos a cos b, cos a sin b, sin a)."""
    return _rotation(alpha_deg, beta_deg)[0].copy()


def _fmt_list(vals: Iterable[float]) -> str:
    return ", ".join(f"{v:g}" for v in vals)


# ---------------------------------------------------------------------------
# set-up
# ---------------------------------------------------------------------------

def _setup(wing: WingSpec, *, speed: float, fluid: Fluid | None, method: str, viscous: bool, ncrit: float,
           mach: float, moment_ref, polar_provider, coords_provider, ref_area, ref_chord, ref_span) -> _Setup:
    if method not in WING_METHODS:
        raise ValueError(f"unknown wing method {method!r} (use one of {', '.join(WING_METHODS)})")
    fluid = fluid if fluid is not None else Fluid()
    if not (math.isfinite(speed) and speed > 0):
        raise ValueError(f"speed must be positive, got {speed!r}")
    if not (fluid.density > 0 and fluid.viscosity > 0):
        raise ValueError("the fluid needs a positive density and viscosity")
    if not (0.0 <= mach < MACH_MAX):
        raise ValueError(f"Mach number must be in [0, {MACH_MAX}), got {mach!r}")
    if not (0.0 < ncrit <= 20.0):
        raise ValueError(f"Ncrit must be in (0, 20], got {ncrit!r}")
    check_wing(wing)
    notes: list[str] = []
    warnings: list[str] = []

    provider = polar_provider
    if provider is not None and hasattr(provider, "matches") and not provider.matches(ncrit=ncrit, mach=mach):
        notes.append(f"The supplied section-polar cache was built for Ncrit {provider.ncrit:g}, Mach "
                     f"{provider.mach:g}; a separate cache was used for Ncrit {ncrit:g}, Mach {mach:g}.")
        provider = SectionPolars(ncrit=ncrit, mach=mach,
                                 coords_provider=getattr(provider, "coords_provider", None) or coords_provider)
    if provider is None:
        provider = SectionPolars(ncrit=ncrit, mach=mach, coords_provider=coords_provider)

    shapes = []
    by_key: dict[str, Any] = {}
    for i, s in enumerate(wing.sections):
        key = airfoil_key(s.airfoil)
        if key not in by_key:
            try:
                by_key[key] = section_shape(key, provider.coords(s.airfoil))
            except ValueError as exc:
                raise ValueError(f"Section {i + 1} (y = {s.y:g} m), airfoil {s.airfoil!r}: {exc}") from exc
        shapes.append(by_key[key])
    geom = build_geometry(wing, shapes)
    notes.extend(geom.notes)
    m = geom.metrics
    s_ref = float(ref_area) if ref_area else m["area"]
    c_ref = float(ref_chord) if ref_chord else m["mac"]
    b_ref = float(ref_span) if ref_span else m["span"]
    if moment_ref is None:
        ref = np.array([m["mac_le_x"] + 0.25 * m["mac"], 0.0, 0.0])
        notes.append(f"Moments about the quarter chord of the mean aerodynamic chord, x = {ref[0]:.4g} m.")
    else:
        ref = np.asarray(moment_ref, dtype=float).reshape(3)

    st = geom.strips
    keys = [airfoil_key(s.airfoil) for s in wing.sections]
    af_a = [keys[i] for i in st.sec_a]
    af_b = [keys[i] for i in st.sec_b]
    w = st.weight
    a_l0 = np.array([(1 - t) * shapes[i].alpha_l0 + t * shapes[j].alpha_l0 for i, j, t in zip(st.sec_a, st.sec_b, w)])
    cm0 = np.array([(1 - t) * shapes[i].cm0 + t * shapes[j].cm0 for i, j, t in zip(st.sec_a, st.sec_b, w)])
    re_strip = fluid.density * speed * st.chord / fluid.viscosity
    floor = max(RE_FLOOR_FRACTION * float(re_strip.max()), RE_FLOOR)
    re_polar = np.maximum(re_strip, floor)
    beta_pg = math.sqrt(1.0 - mach * mach)
    if mach >= MACH_WARN:
        warnings.append(f"Mach {mach:.2f} is beyond the Prandtl–Glauert range trusted here (M < {MACH_WARN}): "
                        "the results ignore transonic effects and are indicative only.")
    elif mach > 0:
        notes.append(f"Compressibility: Prandtl–Glauert, beta = {beta_pg:.4f} (M {mach:g}).")
    if viscous and np.any(re_strip < floor * 0.999):
        notes.append(f"Strips with a Reynolds number below {floor:.3g} (the tips) take their section data at "
                     f"Re {floor:.3g}: their area is small and polars at very low Re are unreliable.")
    if method == "llt":
        if abs(m["sweep_c4_deg"]) > LLT_SWEEP_WARN:
            warnings.append(f"The quarter-chord sweep is {m['sweep_c4_deg']:.0f} deg: a lifting line has no "
                            "chordwise loading and is outside its validity beyond ~20 deg of sweep. Use the "
                            "vortex-lattice method for this wing.")
        if m["aspect_ratio"] < LLT_AR_WARN:
            warnings.append(f"Aspect ratio {m['aspect_ratio']:.2f} is low for a lifting line (it over-predicts "
                            "the lift slope below AR ~4). Use the vortex-lattice method for this wing.")
    return _Setup(wing=wing, geom=geom, method=method, viscous=viscous, beta_pg=beta_pg, mach=mach, speed=speed,
                  fluid=fluid, ref=ref, s_ref=s_ref, c_ref=c_ref, b_ref=b_ref, re_strip=re_strip,
                  re_polar=re_polar, af_a=af_a, af_b=af_b, alpha_l0=a_l0, cm0=cm0, provider=provider,
                  notes=notes, warnings=warnings, core=_tip_core(wing, st.y))


# ---------------------------------------------------------------------------
# section tables per strip
# ---------------------------------------------------------------------------

class _StripTables:
    """Blended section tables per strip (airfoil blend and Reynolds number), with sharing."""

    def __init__(self, su: _Setup) -> None:
        self.su = su
        self._cache: dict[tuple, PolarTable] = {}

    def invalidate(self) -> None:
        self._cache.clear()

    def key(self, j: int) -> tuple:
        su = self.su
        w = float(su.geom.strips.weight[j])
        a, b = su.af_a[j], su.af_b[j]
        if a == b or w <= 1e-9:
            return (a, a, 0.0, float(f"{su.re_polar[j]:.6g}"))
        if w >= 1 - 1e-9:
            return (b, b, 0.0, float(f"{su.re_polar[j]:.6g}"))
        return (a, b, round(w, 6), float(f"{su.re_polar[j]:.6g}"))

    def table(self, j: int) -> PolarTable:
        k = self.key(j)
        tab = self._cache.get(k)
        if tab is None:
            a, b, w, re = k
            ta = self.su.provider.table(a, re)
            tab = ta if a == b else blend_tables([ta, self.su.provider.table(b, re)], [1.0 - w, w])
            self._cache[k] = tab
        return tab

    def groups(self) -> dict[tuple, list[int]]:
        out: dict[tuple, list[int]] = {}
        for j in range(len(self.su.re_polar)):
            out.setdefault(self.key(j), []).append(j)
        return out


def _demands(su: _Setup, alpha_need: np.ndarray, lo_margin: float, hi_margin: np.ndarray | float
             ) -> dict[str, list[tuple[float, float, float]]]:
    """Section-polar requests from estimated strip angles ``alpha_need`` (n_alpha, n_strips) [deg]."""
    lo = np.nanmin(alpha_need, axis=0) - lo_margin
    hi = np.nanmax(alpha_need + hi_margin, axis=0)
    st = su.geom.strips
    out: dict[str, list[tuple[float, float, float]]] = {}
    for j in range(len(st.y)):
        for af, wt in ((su.af_a[j], 1.0 - st.weight[j]), (su.af_b[j], st.weight[j])):
            if wt > 1e-9 or su.af_a[j] == su.af_b[j]:
                out.setdefault(af, []).append((float(su.re_polar[j]), float(max(lo[j], ALPHA_MIN)),
                                               float(min(hi[j], ALPHA_MAX))))
    return out


def _alpha_estimate(su: _Setup, cl_strip: np.ndarray) -> np.ndarray:
    """Section angle [deg] a thin airfoil would need for ``cl_strip`` (first guess for the polar range)."""
    a2d = 2.0 * math.pi / su.beta_pg
    return np.degrees(su.alpha_l0[None, :] + cl_strip / a2d)


def _extension_requests(su: _Setup, tabs: _StripTables, need_hi: dict[int, float], need_lo: dict[int, float]
                        ) -> dict[str, list[tuple[float, float, float]]]:
    """Requests that extend polars whose computed range ended before the lift (or angle) a strip needs."""
    st = su.geom.strips
    out: dict[str, list[tuple[float, float, float]]] = {}

    def add(j: int, a0: float, a1: float) -> None:
        for af, wt in ((su.af_a[j], 1.0 - st.weight[j]), (su.af_b[j], st.weight[j])):
            if wt > 1e-9 or su.af_a[j] == su.af_b[j]:
                out.setdefault(af, []).append((float(su.re_polar[j]), a0, a1))

    for j in need_hi:
        t = tabs.table(j)
        top = float(t.alpha[-1])
        if t.top_limit == "range" and top < ALPHA_MAX - 1e-9:
            add(j, top, min(top + EXTEND_STEP, ALPHA_MAX))
    for j in need_lo:
        t = tabs.table(j)
        bot = float(t.alpha[0])
        if t.bottom_limit == "range" and bot > ALPHA_MIN + 1e-9:
            add(j, max(bot - EXTEND_STEP, ALPHA_MIN), bot)
    return out


# ---------------------------------------------------------------------------
# assembling a WingPoint
# ---------------------------------------------------------------------------

def _point(su: _Setup, alpha: float, beta: float, force_b: np.ndarray, moment_b: np.ndarray, di: float,
           gamma_strip: np.ndarray, downwash: np.ndarray, cd_strip: np.ndarray | None,
           alpha_i: np.ndarray | None = None) -> WingPoint:
    """Coefficients in wind axes from body-frame loads (unit speed and density, q = 1/2)."""
    st = su.geom.strips
    R = _rotation(alpha, beta)
    vhat = R[0]
    f, mom = force_b.copy(), moment_b.copy()
    cdv = 0.0
    if cd_strip is not None:
        fv = 0.5 * (cd_strip * st.area)[:, None] * vhat[None, :]
        f = f + fv.sum(axis=0)
        mom = mom + np.cross(st.c4 - su.ref, fv).sum(axis=0)
        cdv = float(np.sum(cd_strip * st.area)) / su.s_ref
    fw, mw = R @ f, R @ mom
    q_s = 0.5 * su.s_ref
    cl = fw[2] / q_s
    cdi = di / q_s
    m = su.geom.metrics
    ar = m["span"] ** 2 / su.s_ref
    # e is a ratio of two vanishing quantities near zero lift (a twisted wing even has CDi > 0 at
    # CL = 0): only reported where it means something
    e = cl * cl / (math.pi * ar * cdi) if (cdi > 1e-9 and abs(cl) >= E_MIN_CL) else float("nan")
    cl_loc = 2.0 * gamma_strip / st.chord
    if alpha_i is None:
        alpha_i = np.degrees(np.arctan(-0.5 * downwash))
    return WingPoint(
        alpha=float(alpha), cl=float(cl), cd=float(cdi + cdv), cdi=float(cdi), cdv=float(cdv),
        cm=float(mw[1] / (q_s * su.c_ref)), cy=float(fw[1] / q_s), c_roll=float(mw[0] / (q_s * su.b_ref)),
        cn=float(mw[2] / (q_s * su.b_ref)), e=float(e), converged=True,
        y=st.y.copy(), cl_local=cl_loc, c_cl=cl_loc * st.chord / m["mean_chord"],
        alpha_i=np.asarray(alpha_i, dtype=float), cd_local=None if cd_strip is None else np.asarray(cd_strip, float))


def _failed(alpha: float) -> WingPoint:
    return WingPoint(alpha=float(alpha), converged=False)


# ---------------------------------------------------------------------------
# the sweeps
# ---------------------------------------------------------------------------

def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled


def _sweep_vlm(su: _Setup, alphas: list[float], beta: float, report, cancel,
               out: dict[float, WingPoint]) -> None:
    """Vortex-lattice sweep; fills ``out`` as points finish (so a cancel keeps them)."""
    lat = VortexLattice(su.geom, beta=su.beta_pg, moment_ref=su.ref)
    sols = {}
    for k, a in enumerate(alphas):
        _check_cancel(cancel)
        s = sols[a] = lat.solve(freestream_direction(a, beta))
        if not su.viscous:
            out[a] = _point(su, a, beta, s.force, s.moment, s.induced_drag, s.gamma_te, s.downwash, None)
            report("solve", (k + 1) / len(alphas), f"alpha = {a:g} deg")
    st = su.geom.strips
    if not su.viscous:
        return

    cl_strip = np.array([2.0 * sols[a].gamma_te / st.chord for a in alphas])
    est = _alpha_estimate(su, cl_strip)
    reqs = _demands(su, est, 2.0, 2.0 + 0.15 * np.abs(est - np.degrees(su.alpha_l0)[None, :]))
    report("polars", 0.0, "planning section polars")
    su.provider.ensure(reqs, progress=lambda f, m: report("polars", f, m), cancel=cancel)
    tabs = _StripTables(su)
    for _ in range(EXTEND_ROUNDS):
        need_hi: dict[int, float] = {}
        need_lo: dict[int, float] = {}
        try:
            for j in range(len(st.y)):
                t = tabs.table(j)
                if np.max(cl_strip[:, j]) > t.cl_max:
                    need_hi[j] = float(np.max(cl_strip[:, j]))
                if np.min(cl_strip[:, j]) < t.cl_min:
                    need_lo[j] = float(np.min(cl_strip[:, j]))
        except SectionPolarError:
            break
        ext = _extension_requests(su, tabs, need_hi, need_lo)
        if not ext:
            break
        su.provider.ensure(ext, progress=lambda f, m: report("polars", f, m), cancel=cancel)
        tabs.invalidate()

    clipped: dict[float, int] = {}
    groups = [np.asarray(idx) for idx in tabs.groups().values()]
    for k, a in enumerate(alphas):
        _check_cancel(cancel)
        s = sols[a]
        try:
            cd = np.empty(len(st.y))
            nclip = 0
            best = (-np.inf, -1)
            ntip = 0
            for idx in groups:
                t = tabs.table(idx[0])
                ae, clip = t.alpha_at_cl(cl_strip[k, idx])
                cd[idx] = t.drag(ae)
                core = su.core[idx]
                nclip += int(np.count_nonzero(clip[core]))
                ntip += int(np.count_nonzero(clip[~core]))
                if t.top_limit == "stall" and np.any(core):
                    mg = np.where(core, cl_strip[k, idx] - t.cl_max, -np.inf)
                    i = int(np.argmax(mg))
                    if mg[i] > best[0]:
                        best = (float(mg[i]), int(idx[i]))
            if best[1] >= 0:
                su.margins[a] = best
            if nclip:
                clipped[a] = nclip
            elif ntip:
                su.tip_clipped.add(a)
            out[a] = _point(su, a, beta, s.force, s.moment, s.induced_drag, s.gamma_te, s.downwash, cd)
        except SectionPolarError as exc:
            out[a] = _failed(a)
            msg = f"No viscous drag at alpha = {a:g} deg: {exc}."
            if msg not in su.warnings:
                su.warnings.append(msg)
        report("solve", (k + 1) / len(alphas), f"alpha = {a:g} deg")
    if clipped:
        su.warnings.append(
            "Strips asked for more lift than their section polar reaches (beyond cl_max) at alpha = "
            + _fmt_list(sorted(clipped)) + " deg: the vortex lattice's lift is linear and does not stall, so "
            "CL there is optimistic and those strips' profile drag is the polar's at cl_max. Use the lifting "
            "line (nonlinear section lift) or CFD near stall.")


def _lift_function(su: _Setup, tabs: _StripTables | None):
    """(cl, dcl/dalpha) per strip for angles in radians: section polars or thin-airfoil theory."""
    if tabs is None:
        a2d = 2.0 * math.pi / su.beta_pg

        def lift(al: np.ndarray):
            return a2d * (al - su.alpha_l0), np.full_like(al, a2d)
        return lift, np.full(len(su.alpha_l0), a2d), su.alpha_l0.copy()

    groups = [(tabs.table(idx[0]), np.array(idx)) for idx in tabs.groups().values()]
    n = len(su.alpha_l0)
    slope0 = np.empty(n)
    alpha0 = np.empty(n)
    for t, idx in groups:
        a0 = float(t.alpha_at_cl(np.array([0.0]))[0][0])
        _, d = t.lift(np.array([a0]))[:2]
        slope0[idx] = max(float(d[0]) * 180.0 / math.pi, 1.0)
        alpha0[idx] = math.radians(a0)

    def lift(al: np.ndarray):
        cl = np.empty_like(al)
        dcl = np.empty_like(al)
        for t, idx in groups:
            c, d, _ = t.lift(np.degrees(al[idx]))
            cl[idx] = c
            dcl[idx] = d * 180.0 / math.pi
        return cl, dcl
    return lift, slope0, alpha0


def _sweep_llt(su: _Setup, alphas: list[float], beta: float, report, cancel,
               out: dict[float, WingPoint]) -> None:
    """Nonlinear lifting-line sweep (warm-started outward from zero); fills ``out``."""
    line = LiftingLine(su.geom)
    st = su.geom.strips
    order = sorted(alphas, key=lambda a: (abs(a), a))        # warm starts outward from zero
    tabs = None
    if su.viscous:
        lift0, s0, a0 = _lift_function(su, None)
        cl_est = []
        for a in alphas:
            sol = line.solve(freestream_direction(a, beta), lift0, slope0=s0, alpha0=a0)
            cl_est.append(2.0 * sol.gamma / st.chord)
        est = _alpha_estimate(su, np.array(cl_est))
        reqs = _demands(su, est, 2.0, 2.0 + 0.15 * np.abs(est - np.degrees(su.alpha_l0)[None, :]))
        report("polars", 0.0, "planning section polars")
        su.provider.ensure(reqs, progress=lambda f, m: report("polars", f, m), cancel=cancel)
        tabs = _StripTables(su)

    clipped: dict[float, int] = {}
    sols: dict[float, Any] = {}
    for _round in range(EXTEND_ROUNDS + 1):
        try:
            lift, slope0, alpha0 = _lift_function(su, tabs)
        except SectionPolarError as exc:
            su.warnings.append(f"No section polar for the lifting line: {exc}.")
            out.update({a: _failed(a) for a in alphas})
            return
        prev = None
        need_hi: dict[int, float] = {}
        need_lo: dict[int, float] = {}
        for k, a in enumerate(order):
            _check_cancel(cancel)
            g0 = prev if (prev is not None and prev[0] * a >= 0) else None
            sol = line.solve(freestream_direction(a, beta), lift, slope0=slope0, alpha0=alpha0,
                             gamma0=None if g0 is None else g0[1])
            if not sol.converged and g0 is not None:
                sol = line.solve(freestream_direction(a, beta), lift, slope0=slope0, alpha0=alpha0)
            sols[a] = sol
            if sol.converged:
                prev = (a, sol.gamma)
            if tabs is not None:
                ad = np.degrees(sol.alpha)
                for j in range(len(st.y)):
                    t = tabs.table(j)
                    if ad[j] > t.alpha_cl_max + 1e-9 and t.top_limit == "range":
                        need_hi[j] = ad[j]
                    if ad[j] < t.alpha_cl_min - 1e-9 and t.bottom_limit == "range":
                        need_lo[j] = ad[j]
            report("solve", (k + 1) / len(order), f"alpha = {a:g} deg")
        if tabs is None:
            break
        ext = _extension_requests(su, tabs, need_hi, need_lo)
        if not ext or _round == EXTEND_ROUNDS:
            break
        su.provider.ensure(ext, progress=lambda f, m: report("polars", f, m), cancel=cancel)
        tabs.invalidate()

    for a in alphas:
        sol = sols[a]
        if not sol.converged:
            out[a] = _failed(a)
            continue
        V = sol.velocity
        V2 = np.einsum("ik,ik->i", V, V)
        ad = np.degrees(sol.alpha)
        cd = None
        if tabs is not None:
            cd = np.empty(len(st.y))
            cm = np.empty(len(st.y))
            nclip = 0
            best = (-np.inf, -1)
            ntip = 0
            for idx in tabs.groups().values():
                idx = np.asarray(idx)
                t = tabs.table(idx[0])
                cd[idx] = t.drag(ad[idx])
                cm[idx] = t.moment(np.clip(ad[idx], t.alpha[0], t.alpha[-1]))
                _, _, clip = t.lift(ad[idx])
                core = su.core[idx]
                nclip += int(np.count_nonzero(clip[core]))
                ntip += int(np.count_nonzero(clip[~core]))
                if t.top_limit == "stall" and np.any(core):
                    mg = np.where(core, ad[idx] - t.alpha_cl_max, -np.inf)
                    i = int(np.argmax(mg))
                    if mg[i] > best[0]:
                        best = (float(mg[i]), int(idx[i]))
            if best[1] >= 0:
                su.margins[a] = best
            if nclip:
                clipped[a] = nclip
            elif ntip:
                su.tip_clipped.add(a)
        else:
            cm = su.cm0 / su.beta_pg
        force = sol.force.sum(axis=0)
        u_s = line.u_s
        m_sec = (0.5 * V2 * cm * st.chord * st.area)[:, None] * u_s
        moment = np.cross(line.cp - su.ref, sol.force).sum(axis=0) + m_sec.sum(axis=0)
        alpha_i = np.degrees(sol.alpha_geo - sol.alpha)
        out[a] = _point(su, a, beta, force, moment, sol.induced_drag, sol.gamma, sol.downwash, cd, alpha_i)
    if clipped:
        su.warnings.append(
            "Section lift was clipped at the polar's cl_max (post-stall) on some strips at alpha = "
            + _fmt_list(sorted(clipped)) + " deg. The section polars do not model post-stall lift, so these "
            "points are not a stall prediction: CL there is an estimate and the real wing may stall earlier.")


def _stall_onset(su: _Setup, points: dict[float, WingPoint]) -> dict[str, float] | None:
    """Critical-section estimate of stall: where the first strip reaches its section cl_max.

    Walking up from the lowest positive-lift angle, the margin of the most loaded
    strip (lattice: its cl minus the section cl_max; lifting line: its effective
    angle minus the angle of cl_max) crosses zero between two computed angles; the
    crossing is interpolated linearly, with CL at the same place. This is the
    classical critical-section estimate of CL_max (a lower bound when the stall
    spreads gradually, as on a washed-out wing).
    """
    al = sorted(a for a in su.margins if a in points and points[a].converged and points[a].cl > 0)
    prev = None
    for a in al:
        m, j = su.margins[a]
        if m >= 0:
            if prev is None:
                return None                       # stalled from the first computed angle on
            a0, m0 = prev
            f = (0.0 - m0) / (m - m0) if m != m0 else 0.0
            cl = points[a0].cl + f * (points[a].cl - points[a0].cl)
            return {"alpha": a0 + f * (a - a0), "cl": float(cl), "y": float(su.geom.strips.y[j])}
        prev = (a, m)
    return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def wing_polar(wing: WingSpec, alphas: Iterable[float], *, speed: float = 30.0, fluid: Fluid | None = None,
               method: str = "vlm", viscous: bool = True, ncrit: float = 9.0, mach: float = 0.0,
               beta_deg: float = 0.0, moment_ref: Sequence[float] | None = None,
               polar_provider: SectionPolars | None = None, coords_provider: Callable[[str], Any] | None = None,
               ref_area: float | None = None, ref_chord: float | None = None, ref_span: float | None = None,
               sim_id: str = "", progress: ProgressFn | None = None, cancel: threading.Event | None = None,
               details: dict | None = None) -> WingResult:
    """Finite-wing polar: one :class:`~aerolume.model.WingPoint` per angle of attack.

    ``method`` is "vlm" (vortex lattice on the cambered mean surface; any
    planform, sweep, dihedral, sideslip) or "llt" (nonlinear lifting line;
    straight or moderately swept wings of AR >= ~4, nonlinear section lift up to
    cl_max). ``viscous`` adds strip profile drag (and, for "llt", the viscous
    section lift curves) from 2D polars of each section at its Reynolds number
    ``rho * speed * chord / mu``, computed with ``ncrit`` and ``mach`` and cached
    in ``polar_provider`` (a :class:`SectionPolars`; pass the same instance to
    later calls to reuse its polars). ``coords_provider(name)`` resolves
    "design:<id>" airfoils. ``moment_ref`` defaults to the quarter chord of the
    mean aerodynamic chord; ``ref_area`` / ``ref_chord`` / ``ref_span`` default to
    the planform area, MAC and span. ``beta_deg`` is the sideslip (FlowConditions
    ``yaw_deg`` convention).

    ``progress(fraction, message)`` is called during the section-polar warm-up
    and after each angle; ``cancel`` (a ``threading.Event``) stops the sweep and
    returns the points finished so far with a warning. A ``details`` dict, when
    given, is filled with diagnostics that have no place in the data model:
    ``stall`` (critical-section stall onset {"alpha", "cl", "y"} or None),
    ``strip_y`` / ``strip_re`` (per strip), ``polar_nodes`` {airfoil: [Re, ...]},
    ``reference`` {"area", "chord", "span", "moment_ref"}. Points that fail are
    marked ``converged=False`` (NaN coefficients) instead of raising. Bad inputs
    (no sections, non-increasing y, unknown airfoil, Mach >= 0.9, ...) raise
    ``ValueError`` before any work is done.
    """
    t0 = time.perf_counter()
    alist = [float(a) for a in alphas]
    if not alist:
        raise ValueError("wing_polar needs at least one angle of attack")
    if not all(math.isfinite(a) and abs(a) < ALPHA_LIMIT for a in alist):
        raise ValueError(f"angles of attack must be finite and within +/-{ALPHA_LIMIT:g} deg")
    if not (math.isfinite(beta_deg) and abs(beta_deg) < ALPHA_LIMIT):
        raise ValueError(f"sideslip must be finite and within +/-{ALPHA_LIMIT:g} deg")
    su = _setup(wing, speed=speed, fluid=fluid, method=method, viscous=viscous, ncrit=ncrit, mach=mach,
                moment_ref=moment_ref, polar_provider=polar_provider, coords_provider=coords_provider,
                ref_area=ref_area, ref_chord=ref_chord, ref_span=ref_span)
    uniq = sorted(set(alist))
    m = su.geom.metrics
    st = su.geom.strips
    result = WingResult(sim_id=sim_id, method=method, viscous=bool(viscous), area=m["area"], span=m["span"],
                        aspect_ratio=m["aspect_ratio"], mac=m["mac"],
                        re_mac=su.fluid.density * speed * m["mac"] / su.fluid.viscosity)

    # progress: section polars dominate a cold viscous run, the wing solves take the rest; the
    # reported fraction never goes backwards (a polar extension restarts its own count)
    phase_w = {"polars": 0.9, "solve": 0.1} if viscous else {"polars": 0.0, "solve": 1.0}
    last = [0.0, 0.0]                          # time, fraction

    def report(phase: str, frac: float, msg: str) -> None:
        if progress is None:
            return
        f = phase_w["polars"] * frac if phase == "polars" else phase_w["polars"] + phase_w["solve"] * frac
        f = min(max(f, last[1]), 1.0)
        now = time.perf_counter()
        if frac >= 1.0 or now - last[0] > 0.05 or phase == "solve":
            last[0], last[1] = now, f
            progress(f, msg)

    pts0 = getattr(su.provider, "points_computed", 0)
    sec0 = getattr(su.provider, "seconds", 0.0)
    cancelled = False
    done: dict[float, WingPoint] = {}
    try:
        if method == "vlm":
            _sweep_vlm(su, uniq, beta_deg, report, cancel, done)
        else:
            _sweep_llt(su, uniq, beta_deg, report, cancel, done)
    except (Cancelled, PolarCancelled):
        cancelled = True

    result.points = [done[a] for a in alist if a in done]
    stall = _stall_onset(su, done) if viscous else None
    if su.tip_clipped:
        result.notes.append(
            "Only strips within 5 % of the semispan from a tip exceeded their section cl_max at alpha = "
            + _fmt_list(sorted(su.tip_clipped)) + " deg (their lift was held at cl_max). Strip theory does "
            "not hold where the tip vortex forms, so this is not reported as wing stall.")
    if stall is not None:
        # structured as well as in the notes, so the GUI and report need not parse text
        result.stall_alpha, result.stall_cl = float(stall["alpha"]), float(stall["cl"])
        result.notes.append(
            f"Stall onset (critical-section estimate): the first strip, at y = {stall['y']:+.3g} m, reaches its "
            f"section cl_max at alpha ≈ {stall['alpha']:.1f} deg, CL ≈ {stall['cl']:.3f}. Treat this as the "
            "wing's CL_max estimate; the curves above it are not a stall prediction.")
    if details is not None:
        details["stall"] = stall
        details["strip_y"] = su.geom.strips.y.copy()
        details["strip_re"] = su.re_strip.copy()
        details["polar_nodes"] = ({af: su.provider.node_res(af) for af in sorted(set(su.af_a) | set(su.af_b))}
                                  if viscous and hasattr(su.provider, "node_res") else {})
        details["reference"] = {"area": su.s_ref, "chord": su.c_ref, "span": su.b_ref,
                                "moment_ref": tuple(float(v) for v in su.ref)}
    # notes
    nc, ns = su.geom.n_chord, su.geom.n_strips
    if method == "vlm":
        result.notes.append(f"Vortex lattice: {nc} chordwise x {ns} spanwise ring vortices on the cambered mean "
                            "surface, wake along the free stream; induced drag in the Trefftz plane.")
    else:
        result.notes.append(f"Nonlinear lifting line (Phillips & Snyder): {ns} horseshoe vortices on the "
                            "quarter-chord line, wake along the free stream; induced drag in the Trefftz plane.")
    if viscous:
        nodes = []
        for af in sorted(set(su.af_a) | set(su.af_b)):
            res = su.provider.node_res(af) if hasattr(su.provider, "node_res") else []
            if res:
                nodes.append(f"{af} at Re {_fmt_list(float(f'{r:.3g}') for r in res)}")
        n_new = getattr(su.provider, "points_computed", 0) - pts0
        sec = getattr(su.provider, "seconds", 0.0) - sec0
        result.notes.append(
            "Viscous strip theory: section polars (XFOIL-class, Ncrit "
            f"{ncrit:g}, Mach {mach:g}) of " + "; ".join(nodes) + ", interpolated in alpha and log(Re)"
            + (f"; {n_new} section points computed in {sec:.1f} s" if n_new else "; all section polars were cached")
            + ".")
        if method == "vlm":
            result.notes.append("Profile drag per strip at the angle where its section polar gives the strip's "
                                "lift; the lattice's lift itself is inviscid (no viscous lift loss).")
        else:
            result.notes.append("The section lift curves come from the viscous polars (nonlinear up to cl_max); "
                                "profile drag and cm at each strip's effective angle.")
        used = {f"Section {af} " for af in set(su.af_a) | set(su.af_b)}
        result.warnings.extend(w for w in getattr(su.provider, "warnings", [])
                               if any(w.startswith(u) for u in used) and w not in result.warnings)
    else:
        result.notes.append("Inviscid: CD is the induced drag only (no profile drag).")
    result.notes.append(
        f"Reference area {su.s_ref:.4g} m², chord {su.c_ref:.4g} m, span {su.b_ref:.4g} m; moments about "
        f"({su.ref[0]:.4g}, {su.ref[1]:.4g}, {su.ref[2]:.4g}) m. Strip Re {st.chord.min() * su.speed * su.fluid.density / su.fluid.viscosity:.3g}"
        f"–{su.re_strip.max():.3g}.")
    result.notes.extend(su.notes)
    result.warnings.extend(w for w in su.warnings if w not in result.warnings)
    bad = sorted({p.alpha for p in result.points if not p.converged})
    if bad:
        result.warnings.append("Not converged at alpha = " + _fmt_list(bad) + " deg; these points are excluded "
                               "from the polar curves.")
    if cancelled:
        result.warnings.append(f"Cancelled after {len(done)} of {len(uniq)} angles.")
    result.wall_seconds = time.perf_counter() - t0
    return result


def section_points_needed(wing: WingSpec, alphas: Iterable[float], *, speed: float = 30.0,
                          fluid: Fluid | None = None, ncrit: float = 9.0, mach: float = 0.0,
                          beta_deg: float = 0.0, polar_provider: SectionPolars | None = None,
                          coords_provider: Callable[[str], Any] | None = None) -> int:
    """Section points a viscous :func:`wing_polar` of these angles would still compute.

    The cost of a viscous wing run is almost all 2D section points (~0.5 s each, against
    milliseconds for the vortex solves), so this is what a run-time estimate needs. It
    plans the first pass exactly as the sweep does, from thin-airfoil strip lift solved
    on the lifting line (~1 ms per angle, the same for both methods to within the
    margins the planner adds), and asks ``polar_provider`` what is not cached yet;
    nothing is computed. Extensions towards cl_max that a sweep beyond the attached
    range adds later are not counted, so near stall this is a lower bound. Raises
    ``ValueError`` for the inputs :func:`wing_polar` rejects.
    """
    alist = sorted({float(a) for a in alphas})
    if not alist:
        return 0
    su = _setup(wing, speed=speed, fluid=fluid, method="llt", viscous=True, ncrit=ncrit, mach=mach,
                moment_ref=None, polar_provider=polar_provider, coords_provider=coords_provider,
                ref_area=None, ref_chord=None, ref_span=None)
    line = LiftingLine(su.geom)
    lift0, s0, a0 = _lift_function(su, None)
    chord = su.geom.strips.chord
    cl = np.array([2.0 * line.solve(freestream_direction(a, beta_deg), lift0, slope0=s0, alpha0=a0).gamma / chord
                   for a in alist])
    est = _alpha_estimate(su, cl)
    reqs = _demands(su, est, 2.0, 2.0 + 0.15 * np.abs(est - np.degrees(su.alpha_l0)[None, :]))
    return int(su.provider.cost(reqs))


def analyze_wing(wing: WingSpec, *, alpha_deg: float, speed: float = 30.0, fluid: Fluid | None = None,
                 method: str = "vlm", viscous: bool = True, ncrit: float = 9.0, mach: float = 0.0,
                 beta_deg: float = 0.0, moment_ref: Sequence[float] | None = None,
                 polar_provider: SectionPolars | None = None, coords_provider: Callable[[str], Any] | None = None,
                 ref_area: float | None = None, ref_chord: float | None = None, ref_span: float | None = None,
                 cancel: threading.Event | None = None) -> WingPoint:
    """One finite-wing operating point (see :func:`wing_polar` for every argument).

    Returns the :class:`~aerolume.model.WingPoint`; its notes and warnings are
    those of the one-angle :func:`wing_polar`, which callers that need them
    should use directly.
    """
    res = wing_polar(wing, [alpha_deg], speed=speed, fluid=fluid, method=method, viscous=viscous, ncrit=ncrit,
                     mach=mach, beta_deg=beta_deg, moment_ref=moment_ref, polar_provider=polar_provider,
                     coords_provider=coords_provider, ref_area=ref_area, ref_chord=ref_chord, ref_span=ref_span,
                     cancel=cancel)
    return res.points[0] if res.points else WingPoint(alpha=float(alpha_deg), converged=False)


__all__ = ["analyze_wing", "freestream_direction", "section_points_needed", "wing_polar"]
