"""Viscous–inviscid airfoil analysis: single points, polar sweeps, inviscid Cp.

This is the public face of the vendored mfoil (K. J. Fidkowski, MIT), an
XFOIL-class solver: a linear-vorticity panel method coupled to an integral
boundary layer (laminar / turbulent closures, e^N transition, lagged shear
stress, wake), all solved simultaneously by Newton's method.

What this module adds on top of mfoil:

* geometry hygiene — every entry point runs :func:`prepare_coords`, so any
  airfoil outline (Design, Body, raw polygon, any chord, any orientation) is
  analysed in the unit-chord, chord-line frame mfoil expects;
* speed — the hot loops run through :mod:`._fast` (vectorised, verified
  against mfoil to round-off), and a polar keeps the angle-independent panel
  matrices and warm-starts each point from its neighbour;
* robustness — a point that does not converge is retried with intermediate
  angles and then from a fresh boundary layer; if it still fails it is marked
  ``converged=False`` instead of aborting the sweep;
* integration — PolarPoint / PolarResult outputs, progress and cancel hooks,
  no stdout noise, no shared mutable state (each call owns its solver object,
  so concurrent calls from worker threads are safe).

Units: everything here is non-dimensional (chord = 1, V = 1); coefficients
follow DESIGN.md section 3 (Cl, Cd per unit span on the chord, Cm about the
quarter chord, positive nose-up). The angle of attack is measured from the
chord line (leading edge to trailing edge) of the prepared coordinates.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ...model import PolarPoint, PolarResult
from . import _fast as fa
from . import _mfoil as mf
from .geometry import prepare_coords

ProgressFn = Callable[[float, str], None]

DEFAULT_PANELS = 199
MIN_TE_GAP = 5e-4           # viscous runs open sharp / near-sharp trailing edges to this gap (x/c)
RTOL = 1e-8                 # Newton converged when ||R||_2 of the BL residuals is below this
MAX_ITER_COLD = 50          # Newton iterations from a freshly marched boundary layer
MAX_ITER_WARM = 25          # ... from a neighbouring converged point
EARLY_IT = 15               # warm starts: past this many iterations, give up unless ||R|| < LOOSE_R
LS_AFTER = 4                # line search on the Newton step from this iteration on ...
LS_BELOW = 0.1              # ... when ||R|| is below this but did not at least halve
LOOSE_R = 2e-2              # "engineering" convergence (see _stalled_but_settled): ||R|| bound,
LOOSE_N = 6                 # ... over this many final iterations,
LOOSE_CL, LOOSE_CD = 2e-3, 0.01   # ... with Cl varying less than this and Cd less than this fraction
SOLVER_NAME = "mfoil (XFOIL-class viscous–inviscid panel/integral-BL method, Fidkowski 2022)"


class Cancelled(Exception):
    """Raised internally when the caller's cancel event is set."""


def adapt_trailing_edge(coords: np.ndarray, gap: float, target: float) -> np.ndarray:
    """Set the TE gap of prepared (Selig, unit-chord) coords to ``target`` (0 = sharp).

    The upper surface is moved by +d/2 * x/c and the lower by -d/2 * x/c along
    the chord normal (XFOIL's TGAP with a full-chord blend), so the leading
    edge is untouched and the change is at most d/2 at the trailing edge.

    Why the solvers need this (measured, see docs/theory_airfoil.md):

    * viscous: with an exactly sharp TE mfoil's sharp-TE condition is exact in
      principle, but the coupled Newton iteration often fails or converges to a
      spurious branch (NACA 2412, Re 1e6: Cl 0.94-1.01 at 6-10 deg instead of
      0.91-1.24). With a 0.02-0.25 % gap it converges at every angle, and the
      results approach the converged sharp-TE ones as the gap shrinks (Cl 1.2437
      at 0.05 % vs 1.2432 sharp at 10 deg). 0.05 % is used.
    * inviscid: the blunt-TE source panel is ill-conditioned when the gap is
      much smaller than the neighbouring panels (the TE edge velocity drifts
      as the gap shrinks), so small gaps are closed to the exact sharp TE.
    """
    q = np.array(coords, dtype=float, copy=True)
    d = target - gap
    if abs(d) < 1e-15:
        return q
    ile = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
    x = np.clip(q[:, 0], 0.0, 1.0)
    if gap == 0.0 and target > 0.0:
        # sharp -> blunt: the shared TE point (first == last) splits in two
        q[:ile + 1, 1] += 0.5 * d * x[:ile + 1]
        q[ile + 1:, 1] -= 0.5 * d * x[ile + 1:]
    elif target == 0.0:
        # blunt -> sharp: move each corner to their midpoint, blended along the chord
        mid = 0.5 * (q[0] + q[-1])
        q[:ile + 1] -= np.outer(x[:ile + 1], q[0] - mid)
        q[ile + 1:] -= np.outer(x[ile + 1:], q[-1] - mid)
        q[-1] = q[0]
    else:
        q[:ile + 1, 1] += 0.5 * d * x[:ile + 1]
        q[ile + 1:, 1] -= 0.5 * d * x[ile + 1:]
    return q


class _Case:
    """One paneled airfoil at fixed Reynolds / Mach / Ncrit / trip settings.

    Holds the mfoil state object and the angle-independent matrices, so a
    sweep pays for paneling and the influence matrix once. Not shared between
    calls; one instance lives inside one analyze/polar call.
    """

    def __init__(self, coords: np.ndarray, *, re: float, mach: float, ncrit: float,
                 xtr_upper: float, xtr_lower: float, panels: int, viscous: bool, te_gap: float | None = None):
        self.te_note = ""
        if te_gap is not None and te_gap < MIN_TE_GAP:
            if viscous:
                coords = adapt_trailing_edge(coords, te_gap, MIN_TE_GAP)
                self.te_note = (f"The trailing edge was opened to {MIN_TE_GAP * 100:.2f}% of chord for the "
                                "viscous solver (numerical robustness; effect on Cl and Cd ~0.5%).")
            elif te_gap > 0.0:
                coords = adapt_trailing_edge(coords, te_gap, 0.0)
        with np.errstate(all="ignore"):
            M = mf.mfoil(coords=coords, npanel=int(panels))
        M.param.verb = 0
        M.param.doplot = False
        # prepare_coords puts the LE at (0, 0) and the TE at (1, 0): the chord is
        # exactly 1 even where the max-minus-min x of the spline differs by ~1e-6
        M.geom.chord = 1.0
        M.geom.xref = np.array([0.25, 0.0])
        M.param.ncrit = float(ncrit)
        M.oper.Re = float(re) if viscous else 1e6
        M.oper.Ma = float(mach)
        M.oper.xtr = np.array([float(xtr_lower), float(xtr_upper)])
        M.oper.viscous = bool(viscous)
        self.M = M
        self.viscous = bool(viscous)
        self.cache = fa.GeometryCache(M)
        self.has_bl = False          # a converged boundary-layer state exists to warm-start from
        self._carry = None           # (ue - ue_inviscid, node signs) of that state
        self.last_rnorm = float("nan")
        self.iterations = 0
        self.last_error = ""
        self.loose = False           # last solve accepted by the engineering criterion, not RTOL

    # -- inviscid ----------------------------------------------------------
    def solve_inviscid(self, alpha: float) -> None:
        """Panel solution at alpha (fills M.post with cl, cm, cp; mfoil solve_inviscid)."""
        M = self.M
        with np.errstate(all="ignore"):
            M.oper.alpha = float(alpha)
            M.oper.viscous = False
            mf.init_thermo(M)
            M.isol.sgnue = np.ones(M.foil.N)
            fa.build_gamma(M, self.cache)
            mf.calc_force(M)
            M.glob.conv = True

    # -- viscous -----------------------------------------------------------
    def snapshot(self) -> tuple:
        """Converged boundary-layer state to warm-start a later solve from.

        Besides the state and laminar/turbulent flags it keeps the viscous
        edge-velocity correction ``ue - ue_inviscid`` (and the surface sign of
        every node), see :meth:`_carry_viscous_ue`.
        """
        M = self.M
        with np.errstate(all="ignore"):
            delta = M.glob.U[3, :] - mf.get_ueinv(M)
        sg = np.concatenate((M.isol.sgnue, np.ones(M.wake.N)))
        sides = tuple(np.asarray(M.vsol.Is[si]).copy() for si in (0, 1))
        return (M.glob.U.copy(), np.array(M.vsol.turb, copy=True), delta, sg, M.isol.xi.copy(), sides)

    def restore(self, snap: tuple | None) -> None:
        """Make ``snap`` the initial guess of the next warm solve (None = start cold)."""
        if snap is None:
            self.has_bl = False
            self._carry = None
            return
        self.M.glob.U = snap[0].copy()
        self.M.vsol.turb = snap[1].copy()
        self._carry = snap
        self.has_bl = True

    def _carry_viscous_ue(self) -> None:
        """Warm start: keep the converged viscous ue and add only the inviscid change.

        mfoil restarts from the *inviscid* edge velocity at the new angle, which
        discards the displacement effect of the boundary layer and starts the
        Newton iteration at ||R|| ~ 1 even for a 1 deg step (~19 iterations
        measured). Like XFOIL, we instead use ue = ue_inv(new) + (ue - ue_inv)(old)
        on every node that stayed on the same side of the stagnation point; the
        few nodes that switched side keep the inviscid value. Measured on a NACA
        2412 sweep at Re 1e6 (1 deg steps, 1..14 deg): 7.4 Newton iterations per
        point on average instead of 10.9. The remaining start-up residual sits on
        the few nodes beside the stagnation point, whose states scale with the
        distance from it; re-interpolating them against that distance (XFOIL's
        STMOVE idea) was tried and made convergence slower, because the first
        stagnation-point estimate is not accurate enough.
        """
        M = self.M
        if self._carry is None:
            return
        delta, sg_old = self._carry[2], self._carry[3]
        sg = np.concatenate((M.isol.sgnue, np.ones(M.wake.N)))
        if delta.shape != sg.shape:
            return
        same = sg == sg_old
        ue = M.glob.U[3, :]
        new = ue + np.where(same, delta, 0.0)
        # never create a sign change inside a surface (stagnation handling assumes ue > 0)
        ok = new > 0.05 * np.abs(ue)
        M.glob.U[3, :] = np.where(ok, new, ue)

    @staticmethod
    def _stalled_but_settled(tail: list[tuple[float, float, float]], rnorm: float) -> bool:
        """Accept a Newton iteration that cycles at a small residual with settled coefficients.

        With free transition the transition location of a panel interval is the
        root of mfoil's trapezoidal amplification balance, and where the
        amplification rate switches on steeply inside the interval that root
        can jump with tiny state changes (it can even have two roots). The
        coupled iteration then cycles between neighbouring states with ||R||
        ~ 1e-3..1e-2, concentrated on that one station, and never reaches
        RTOL, although Cl and Cd no longer change (XFOIL's convergence test,
        on the size of the Newton update, behaves similarly). Such a point is
        accepted when the last LOOSE_N iterates all had ||R|| < LOOSE_R and
        Cl / Cd varied by less than LOOSE_CL / LOOSE_CD between them; the
        polar notes say which angles were accepted this way.
        """
        if len(tail) < LOOSE_N or rnorm >= LOOSE_R:
            return False
        last = tail[-LOOSE_N:]
        cl = [t[1] for t in last]
        cd = [t[2] for t in last]
        if not all(math.isfinite(v) for v in cl + cd) or min(cd) <= 0:
            return False
        return (max(cl) - min(cl) < LOOSE_CL) and (max(cd) - min(cd) < LOOSE_CD * float(np.mean(cd)))

    def _line_search(self, rnorm: float) -> None:
        """Apply the Newton update with the step length that most reduces ||R||.

        Near convergence the coupled iteration sometimes 2-cycles: the
        transition station's residual depends very steeply on the state where
        the amplification rate switches on inside the transition interval, so
        the full Newton step lands on the mirror image of the current state
        (||R|| ~ 1e-2 for ever; NACA 2412 at 4.5 deg, Re 6e6). Trying the
        half and quarter steps breaks the cycle. Only used when the plain
        iteration is not converging quadratically, so normal solves are
        unchanged. ``update_state`` only moves ``U``, hence the cheap restore.
        """
        M = self.M
        U0, dU = M.glob.U.copy(), M.glob.dU.copy()
        best_s, best_r = 1.0, math.inf
        for s in (1.0, 0.5, 0.25, 0.125):
            M.glob.U = U0.copy()
            M.glob.dU = dU * s
            mf.update_state(M)
            r = float(np.linalg.norm(fa.build_glob_sys(M)[0]))
            if r < best_r:
                best_s, best_r = s, r
            if r < 0.5 * rnorm:
                break
        M.glob.U = U0
        M.glob.dU = dU * best_s
        mf.update_state(M)

    def solve_viscous(self, alpha: float, *, warm: bool, cancel: threading.Event | None = None) -> bool:
        """Coupled solve at alpha; returns convergence. Mirrors mfoil solve_viscous/solve_coupled.

        ``warm`` keeps the current boundary layer (from :meth:`restore` or the
        previous solve) as the Newton initial guess, like XFOIL's polar
        accumulation; otherwise the boundary layer is marched afresh from the
        inviscid edge velocity.
        """
        M = self.M
        warm = warm and self.has_bl
        max_iter = MAX_ITER_WARM if warm else MAX_ITER_COLD
        self.iterations = 0
        self.last_rnorm = float("nan")
        self.last_error = ""
        self.loose = False
        tail: list[tuple[float, float, float]] = []      # (||R||, cl, cd) of the settled iterates
        try:
            with np.errstate(all="ignore"):
                M.oper.alpha = float(alpha)
                M.oper.viscous = False
                mf.init_thermo(M)
                M.isol.sgnue = np.ones(M.foil.N)
                fa.build_gamma(M, self.cache)
                M.oper.viscous = True
                mf.init_thermo(M)
                fa.build_wake(M, self.cache)
                mf.stagpoint_find(M)
                mf.identify_surfaces(M)
                mf.set_wake_gap(M)
                fa.calc_ue_m(M, self.cache)
                M.oper.initbl = not warm
                mf.init_boundary_layer(M)
                if warm:
                    self._carry_viscous_ue()
                self.has_bl = False          # only a converged state may seed a warm start
                mf.stagpoint_move(M)
                M.glob.conv = False
                best = prev = math.inf
                for it in range(max_iter):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled
                    R, R_U = fa.build_glob_sys(M)
                    rnorm = float(np.linalg.norm(R))
                    self.last_rnorm, self.iterations = rnorm, it
                    if not math.isfinite(rnorm):
                        return False
                    if rnorm < RTOL:
                        M.glob.conv = True
                        break
                    best = min(best, rnorm)
                    # Give up early on a diverging iteration, and on a warm start that
                    # stalls (the polar driver retries with smaller steps, which is
                    # cheaper). A cold start is never cut short: from a freshly marched
                    # boundary layer the residual typically plateaus at ||R|| ~ 1 for
                    # 10-25 iterations while the transition points and the stagnation
                    # point settle, then converges quadratically (NACA 2412 at -4 deg,
                    # Re 3e6: 31 iterations; NACA 4412 at 0 deg: 21).
                    if rnorm > 1e4 * max(best, 1e-3) or (warm and it >= EARLY_IT and rnorm > LOOSE_R):
                        return False
                    if rnorm < LOOSE_R and it >= LOOSE_N:
                        mf.calc_force(M)
                        tail.append((rnorm, float(M.post.cl), float(M.post.cd)))
                    else:
                        tail.clear()
                    fa.solve_glob(M, R_U)
                    if it >= LS_AFTER and rnorm < LS_BELOW and rnorm > 0.5 * prev:
                        self._line_search(rnorm)
                    else:
                        mf.update_state(M)
                    prev = rnorm
                    mf.stagpoint_move(M)
                    fa.update_transition(M)
                if not M.glob.conv:
                    R, _ = fa.build_glob_sys(M)
                    if not self._stalled_but_settled(tail, float(np.linalg.norm(R))):
                        return False
                    self.loose = True
                mf.calc_force(M)
                mf.get_distributions(M)
                U = M.glob.U
                ok = bool(np.all(np.isfinite(U)) and np.all(U[0] > 0) and np.all(U[1] > 0)
                          and math.isfinite(M.post.cl) and math.isfinite(M.post.cd) and M.post.cd > 0)
                if ok:
                    self._carry = self.snapshot()
                    self.has_bl = True
                return ok
        except Cancelled:
            raise
        except Exception as exc:     # noqa: BLE001 - any numerical failure inside mfoil = not converged
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    # -- results -----------------------------------------------------------
    def _transition(self) -> tuple[float, float]:
        """Transition x/c (upper, lower); 1.0 when a side stays laminar to the trailing edge."""
        M = self.M
        out = []
        for si in (1, 0):
            Is = np.asarray(M.vsol.Is[si])
            turb = np.asarray(M.vsol.turb)[Is]
            out.append(float(M.vsol.Xt[si, 1]) if np.any(turb) else 1.0)
        return out[0], out[1]

    def point(self, alpha: float, converged: bool, distributions: bool = True) -> PolarPoint:
        """PolarPoint for the current M state (NaN coefficients when not converged)."""
        M = self.M
        if not converged:
            return PolarPoint(alpha=float(alpha), converged=False)
        N = M.foil.N
        p = PolarPoint(alpha=float(alpha), cl=float(M.post.cl), cm=float(M.post.cm), converged=True)
        if self.viscous:
            p.cd, p.cdp, p.cdf = float(M.post.cd), float(M.post.cdp), float(M.post.cdf)
            p.xtr_upper, p.xtr_lower = self._transition()
        if distributions:
            # mfoil nodes run lower TE -> LE -> upper TE; PolarPoint wants upper TE first
            p.x = M.foil.x[0, ::-1].copy()
            p.y = M.foil.x[1, ::-1].copy()
            p.cp = np.asarray(M.post.cp, dtype=float)[:N][::-1].copy()
            p.cp_inviscid = np.asarray(M.post.cpi, dtype=float)[:N][::-1].copy()
            if self.viscous:
                p.cf = np.asarray(M.post.cf, dtype=float)[:N][::-1].copy()
        return p

    def sonic_cp(self) -> float | None:
        """Critical Cp at the free-stream Mach (None when incompressible)."""
        return float(self.M.param.cps) if self.M.oper.Ma > 0 else None


# ---------------------------------------------------------------------------
# argument checks shared by the public functions
# ---------------------------------------------------------------------------

def _check_common(re: float | None, mach: float, ncrit: float, xtr_upper: float, xtr_lower: float,
                  panels: int) -> None:
    if re is not None and not (math.isfinite(re) and re > 0):
        raise ValueError(f"Reynolds number must be positive, got {re!r}")
    if not (0.0 <= mach < 0.9):
        raise ValueError(f"Mach number must be in [0, 0.9), got {mach!r}")
    if not (0.0 < ncrit <= 20.0):
        raise ValueError(f"Ncrit must be in (0, 20], got {ncrit!r}")
    for name, v in (("xtr_upper", xtr_upper), ("xtr_lower", xtr_lower)):
        if not (math.isfinite(v) and v >= 0.0):
            raise ValueError(f"{name} must be a chord fraction in [0, 1] (1 = free), got {v!r}")
    if not (40 <= int(panels) <= 600):
        raise ValueError(f"panels must be between 40 and 600, got {panels!r}")


def _validity_warnings(re: float, mach: float, info: dict) -> list[str]:
    w = list(info.get("warnings", []))
    if re < 5e4:
        w.append(f"Re = {re:.3g} is very low: laminar separation dominates and integral boundary-layer "
                 "results are uncertain (treat as indicative).")
    elif re > 5e7:
        w.append(f"Re = {re:.3g} is beyond the range the transition/turbulence closures were tuned for.")
    if mach > 0.5:
        w.append(f"Mach {mach:.2f}: the Karman-Tsien correction is only reliable while the flow stays "
                 "subsonic everywhere.")
    return w


RUNOFF_NEAR = 0.95          # transition this close to the TE (x/c) counts as "running off" it
RUNOFF_OFF = 0.9995         # ... and at or beyond this the surface is laminar to the TE


def _transition_runoff_notes(points: Sequence[PolarPoint]) -> list[str]:
    """Explain the lift-curve kink where a surface's transition point leaves the trailing edge.

    As the angle of attack rises, the pressure-side transition point moves aft
    and finally off the trailing edge. Just before it does, that boundary
    layer separates laminar a few per cent of chord ahead of the trailing
    edge; its growing displacement thickness acts as extra aft camber, so over
    1-3 deg the lift slope rises by up to ~50 % and Cm turns nose-down, and
    both recover once the surface is laminar to the trailing edge (NACA 2412,
    Re 1e6: dCl/dalpha 0.11 -> 0.15 -> 0.09 /deg between 1.5 and 5 deg, Cm
    -0.046 -> -0.060 -> -0.056). XFOIL 6.99 run on the same coordinates gives
    the same curve to within 0.006 in Cl and 0.001 in Cm (see
    docs/validation_airfoil.md), so it is the e^N / integral boundary-layer
    model's prediction, not a convergence artefact. Users read it as a solver
    bug, hence the note naming the angles.
    """
    conv = sorted((p for p in points if p.converged), key=lambda p: p.alpha)
    notes = []
    for name, attr, sign in (("Lower", "xtr_lower", 1.0), ("Upper", "xtr_upper", -1.0)):
        # walk in the direction in which this surface becomes the pressure side
        seq = conv if sign > 0 else conv[::-1]
        xt = [getattr(p, attr) for p in seq]
        for i in range(len(seq) - 1):
            if not (math.isfinite(xt[i]) and math.isfinite(xt[i + 1])):
                continue
            if xt[i] < RUNOFF_OFF <= xt[i + 1]:
                j = i
                while j > 0 and math.isfinite(xt[j - 1]) and RUNOFF_NEAR <= xt[j - 1] < RUNOFF_OFF:
                    j -= 1
                a0, a1 = sorted((seq[j].alpha, seq[i + 1].alpha))
                notes.append(f"{name}-surface transition reaches the trailing edge between {a0:g}° and {a1:g}°: "
                             "the lift slope steepens and Cm turns nose-down there (laminar separation just "
                             "ahead of the trailing edge). XFOIL predicts the same; it is not a convergence "
                             "problem.")
                break
    return notes


def _solve_ramp(case: _Case, start: float, target: float, snap, cancel, *, step: float = 2.0) -> bool:
    """Reach ``target`` from a converged ``start`` state in steps of at most ``step`` degrees."""
    n = max(1, math.ceil(abs(target - start) / step - 1e-9))
    case.restore(snap)
    for k in range(1, n + 1):
        a = start + (target - start) * k / n
        if not case.solve_viscous(a, warm=case.has_bl, cancel=cancel):
            return False
    return True


def _solve_with_retries(case: _Case, alpha: float, base: tuple[float, object] | None,
                        cancel: threading.Event | None) -> bool:
    """Converge one angle, trying progressively more conservative strategies.

    1. warm start from the neighbouring converged point ``base = (alpha0, snapshot)``;
    2. the same step split in two, then in four (smaller jumps in the BL state);
    3. a fresh boundary layer marched at ``alpha`` itself;
    4. a fresh boundary layer at 0 deg ramped to ``alpha`` in 2-deg steps (cold
       starts at high angles often fail where a ramp succeeds).
    """
    if base is not None:
        a0, snap = base
        case.restore(snap)
        if case.solve_viscous(alpha, warm=True, cancel=cancel):
            return True
        for nsub in (2, 4):
            if abs(alpha - a0) / nsub < 0.05:
                break
            if _solve_ramp(case, a0, alpha, snap, cancel, step=abs(alpha - a0) / nsub + 1e-12):
                return True
    case.restore(None)
    if case.solve_viscous(alpha, warm=False, cancel=cancel):
        return True
    if abs(alpha) > 2.0:
        case.restore(None)
        if (case.solve_viscous(0.0, warm=False, cancel=cancel)
                and _solve_ramp(case, 0.0, alpha, case.snapshot(), cancel, step=2.0)):
            return True
    return False


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def inviscid(coords, alpha_deg: float, *, mach: float = 0.0, panels: int = DEFAULT_PANELS) -> PolarPoint:
    """Panel-method (potential flow) solution: Cl, Cm and the inviscid Cp distribution.

    Fast (tens of milliseconds) and always converged, so it suits live previews
    while a shape is edited. There is no boundary layer, hence no drag
    (``cd``/``cdp``/``cdf`` are NaN) and lift is somewhat over-predicted
    compared with :func:`analyze` (no decambering by the boundary layer, no
    stall). ``coords`` is anything :func:`prepare_coords` accepts.
    """
    _check_common(None, mach, 9.0, 1.0, 1.0, panels)
    xy, _, info = prepare_coords(coords)
    case = _Case(xy, re=1e6, mach=mach, ncrit=9.0, xtr_upper=1.0, xtr_lower=1.0, panels=panels, viscous=False,
                 te_gap=info["te_gap"])
    case.solve_inviscid(alpha_deg)
    return case.point(alpha_deg, True)


def analyze(coords, re: float, alpha_deg: float, *, mach: float = 0.0, ncrit: float = 9.0,
            xtr_upper: float = 1.0, xtr_lower: float = 1.0, viscous: bool = True,
            panels: int = DEFAULT_PANELS, cancel: threading.Event | None = None) -> PolarPoint:
    """Analyse one operating point; returns a :class:`~aerolume.model.PolarPoint`.

    ``re`` is the chord Reynolds number, ``alpha_deg`` the angle of attack from
    the chord line, ``ncrit`` the e^N critical amplification exponent (9 = an
    average wind tunnel / clean free flight, lower = more disturbed flow),
    ``xtr_upper``/``xtr_lower`` forced transition (trip) positions as x/c
    (1.0 = free transition). With ``viscous=False`` this is :func:`inviscid`.
    The result carries Cl, Cd (= Cdp + Cdf), Cm about c/4, transition
    locations, a ``converged`` flag and surface Cp / inviscid Cp / Cf
    distributions ordered upper TE -> LE -> lower TE. A point that cannot be
    converged is returned with ``converged=False`` and NaN coefficients.
    """
    if not viscous:
        return inviscid(coords, alpha_deg, mach=mach, panels=panels)
    res = polar(coords, re, [alpha_deg], mach=mach, ncrit=ncrit, xtr_upper=xtr_upper,
                xtr_lower=xtr_lower, panels=panels, cancel=cancel)
    if not res.points:
        return PolarPoint(alpha=float(alpha_deg), converged=False)
    return res.points[0]


def polar(coords, re: float, alphas: Iterable[float], *, mach: float = 0.0, ncrit: float = 9.0,
          xtr_upper: float = 1.0, xtr_lower: float = 1.0, panels: int = DEFAULT_PANELS,
          distributions: bool = True, sim_id: str = "", progress: ProgressFn | None = None,
          cancel: threading.Event | None = None) -> PolarResult:
    """Viscous polar over ``alphas`` at fixed Re / Mach / Ncrit; returns a PolarResult.

    The sweep starts at the angle closest to zero (cold start), then walks up
    and down from there, warm-starting each point from its converged neighbour
    — the XFOIL "ASEQ" strategy, which is both faster and far more robust than
    independent solves. Failed points are retried (sub-steps, fresh boundary
    layer) and finally marked ``converged=False`` rather than failing the sweep;
    after three consecutive failures in one direction (usually deep stall) the
    remaining angles in that direction get a single cold-start attempt each.

    ``progress(fraction, message)`` is called after each angle; ``cancel``
    (a ``threading.Event``) stops the sweep between Newton iterations and
    returns the points finished so far with a warning. Points are returned in
    the order of ``alphas``. ``distributions=False`` drops the Cp/Cf arrays
    (smaller results when only the coefficients are needed).
    """
    t0 = time.perf_counter()
    alist = [float(a) for a in alphas]
    if not alist:
        raise ValueError("polar needs at least one angle of attack")
    if not all(math.isfinite(a) and abs(a) < 90.0 for a in alist):
        raise ValueError("angles of attack must be finite and within +/-90 deg")
    _check_common(re, mach, ncrit, xtr_upper, xtr_lower, panels)
    xy, _, info = prepare_coords(coords)
    case = _Case(xy, re=re, mach=mach, ncrit=ncrit, xtr_upper=xtr_upper, xtr_lower=xtr_lower,
                 panels=panels, viscous=True, te_gap=info["te_gap"])
    result = PolarResult(sim_id=sim_id, re=float(re), mach=float(mach), ncrit=float(ncrit))
    result.warnings.extend(_validity_warnings(re, mach, info))
    trip = ("free transition" if xtr_upper >= 1 and xtr_lower >= 1
            else f"transition forced at x/c = {min(xtr_upper, 1):.3g} (upper), {min(xtr_lower, 1):.3g} (lower)")
    result.notes.append(f"Solver: {SOLVER_NAME}; {case.M.foil.N - 1} panels; Ncrit = {ncrit:g}; {trip}.")
    result.notes.append(f"Geometry: {info['te_type']} trailing edge"
                        + (f" (gap {info['te_gap'] * 100:.2f}% c)" if info['te_type'] == 'blunt' else "")
                        + f", t/c = {info['thickness'] * 100:.1f}%, camber = {info['camber'] * 100:.1f}%.")
    if case.te_note:
        result.notes.append(case.te_note)

    # unique angles, solved in sweep order; results mapped back to the input order
    uniq = sorted(set(alist))
    i0 = min(range(len(uniq)), key=lambda i: (abs(uniq[i]), uniq[i] < 0))
    up = uniq[i0:]
    down = uniq[:i0][::-1]
    done: dict[float, PolarPoint] = {}
    total = len(uniq)
    cancelled = False
    sonic = case.sonic_cp()
    supersonic: list[float] = []
    loose: list[float] = []

    def report(a: float, p: PolarPoint) -> None:
        if progress is None:
            return
        msg = (f"alpha = {a:g} deg: Cl = {p.cl:.3f}, Cd = {p.cd:.5f}" if p.converged
               else f"alpha = {a:g} deg: not converged")
        progress(len(done) / total, msg)

    try:
        start_snap = None
        for direction, seq in (("up", up), ("down", down)):
            base = None
            if direction == "down" and start_snap is not None:
                base = start_snap
            fails = 0
            for a in seq:
                if cancel is not None and cancel.is_set():
                    raise Cancelled
                if fails >= 3:
                    case.restore(None)
                    ok = case.solve_viscous(a, warm=False, cancel=cancel)
                else:
                    ok = _solve_with_retries(case, a, base, cancel)
                p = case.point(a, ok, distributions)
                if ok:
                    if case.loose:
                        loose.append(a)
                    base = (a, case.snapshot())
                    if direction == "up" and start_snap is None:
                        start_snap = base
                    fails = 0
                    if sonic is not None and p.cp is not None and np.nanmin(p.cp) < sonic:
                        supersonic.append(a)
                else:
                    fails += 1
                done[a] = p
                report(a, p)
    except Cancelled:
        cancelled = True

    result.points = [done[a] for a in alist if a in done]
    bad = [p.alpha for p in result.points if not p.converged]
    if bad:
        result.warnings.append("Not converged at alpha = " + ", ".join(f"{a:g} deg" for a in sorted(set(bad)))
                               + " (typically stall or a laminar separation bubble the model cannot "
                                 "resolve); these points are excluded from the polar curves.")
    if loose:
        result.notes.append("Converged to engineering tolerance at alpha = "
                            + ", ".join(f"{a:g} deg" for a in sorted(loose))
                            + f": the transition point flips between two panels, so the Newton residual "
                              f"settles at ~1e-2 instead of {RTOL:g}; Cl and Cd vary by less than "
                              f"{LOOSE_CL:g} and {LOOSE_CD * 100:g}% between iterates.")
    result.notes.extend(_transition_runoff_notes(result.points))
    if supersonic:
        result.warnings.append("Local flow is supersonic at alpha = " + ", ".join(f"{a:g} deg" for a in supersonic)
                               + "; results there are outside the model's validity (no shocks).")
    if cancelled:
        result.warnings.append(f"Cancelled after {len(done)} of {total} angles.")
    result.wall_seconds = time.perf_counter() - t0
    return result


def alpha_range(start: float, end: float, step: float) -> np.ndarray:
    """Inclusive angle sequence (like AirfoilSettings.alphas) with round-off-safe end point."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = math.floor((end - start) / step + 1e-9) + 1
    return start + step * np.arange(max(n, 1))


__all__: Sequence[str] = ("Cancelled", "alpha_range", "analyze", "inviscid", "polar")
