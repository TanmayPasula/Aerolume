"""Section polars for the finite-wing solvers: a cache of 2D viscous polars.

:class:`SectionPolars` runs :func:`aerolume.solvers.airfoil.polar` (the XFOIL-class
viscous–inviscid solver) for each airfoil of a wing at a few Reynolds numbers
("nodes") and serves interpolated section data to the wing solvers:

* in angle of attack, linearly between the computed angles (a 1 deg grid by
  default, aligned to multiples of the step so nodes can be extended and reused);
* in Reynolds number, linearly in log(Re) between the two nodes that bracket a
  strip; ``cd`` is interpolated as log(cd) (profile drag follows a power law in
  Re, so this is exact for cd ~ Re^-n), cl and cm linearly. Nodes are no further
  than a factor ``re_ratio`` (2 by default) apart, and existing nodes are reused
  whenever they already bracket a strip that closely, so a second analysis of the
  same wing costs nothing.
* Nodes sit on a fixed logarithmic grid (:data:`RE_GRID_PER_DECADE` = 48 values
  per decade, 4.9 % apart), the "Re buckets": any Reynolds number is within 2.4 %
  of a grid value, inside the 5 % at which a strip uses one node alone, so a
  one-chord wing still needs a single node. New nodes extend the existing set
  greedily (the next node as far up as ``re_ratio`` allows from the node below),
  so a study that changes the speed or the planform adds only the nodes its new
  Reynolds numbers need. The first version placed new nodes at both ends of each
  uncovered range: a 22 % speed change of a tapered wing added two nodes beside
  the two it already had (36 s, more than the cold run), where one is enough.

Why a cache: one 2D polar point costs ~0.5 s, and a wing polar of 17 angles
over 2 airfoils x 2-3 Reynolds numbers needs ~40-60 of them, while the vortex
methods themselves take milliseconds. Everything the wing needs is computed
once, in sweeps (warm-started like XFOIL's ASEQ), before the wing solves.

Post-stall: the section solver does not converge far beyond cl_max, and an
integral boundary-layer method does not predict post-stall lift anyway. A
:class:`PolarTable` therefore exposes the *attached branch* (from the angle of
cl_min to the angle of cl_max) and clips outside it: lift held at cl_max with a
zero slope, which the wing solvers report as a warning. ``top_limit`` says
whether cl_max is a real maximum ("stall": lift falls after it or the solver
failed above it) or just the end of the computed range ("range", the caller
should extend the polar).
"""
from __future__ import annotations

import math
import re as _re
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ProgressFn = Callable[[float, str], None]
CoordsProvider = Callable[[str], Any]

ALPHA_MIN = -20.0            # section polars are never computed outside this range (deg)
ALPHA_MAX = 25.0
SNAP = math.log(1.05)        # a strip within 5 % of a node's Re uses that node alone
DEFAULT_PANELS = 140         # vs 199: cl within 0.002, cd within 0.6 % at Re 2e5-4e5 (measured), ~1.3x faster
#: Reynolds-number nodes lie on 10^(k / RE_GRID_PER_DECADE): half a grid step (2.4 %) is
#: inside SNAP, so every Reynolds number has a grid node it may use alone.
RE_GRID_PER_DECADE = 48


def re_grid_value(k: int) -> float:
    """The k-th Reynolds-number grid node, 10^(k / RE_GRID_PER_DECADE), to 6 significant digits."""
    return float(f"{10.0 ** (k / RE_GRID_PER_DECADE):.6g}")


def _grid_pos(re: float) -> float:
    return math.log10(re) * RE_GRID_PER_DECADE


class SectionPolarError(RuntimeError):
    """A section polar could not be produced (no converged point at all)."""


class PolarCancelled(Exception):
    """The caller's cancel event stopped the polar warm-up."""


def airfoil_key(name: str) -> str:
    """Normalise an airfoil name: NACA designations to their digits, others stripped.

    "NACA 2412", "naca2412" and "2412" are the same airfoil and share a cache entry.
    """
    s = str(name).strip()
    digits = _re.sub(r"(?i)^naca[\s\-]*", "", s).replace(" ", "")
    if digits.isdigit() and len(digits) in (4, 5):
        return digits
    return s


@dataclass(frozen=True)
class PolarTable:
    """Converged section data at one Reynolds number (angles ascending, degrees).

    ``i_lo`` / ``i_hi`` delimit the attached branch (indices of cl_min and
    cl_max); ``top_limit`` / ``bottom_limit`` are "stall" when that extreme is a
    real maximum/minimum and "range" when the polar simply ends there.
    """
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    re: float
    i_lo: int
    i_hi: int
    top_limit: str
    bottom_limit: str

    @property
    def cl_max(self) -> float:
        return float(self.cl[self.i_hi])

    @property
    def cl_min(self) -> float:
        return float(self.cl[self.i_lo])

    @property
    def alpha_cl_max(self) -> float:
        return float(self.alpha[self.i_hi])

    @property
    def alpha_cl_min(self) -> float:
        return float(self.alpha[self.i_lo])

    def _branch(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.alpha[self.i_lo:self.i_hi + 1]
        c = self.cl[self.i_lo:self.i_hi + 1]
        return a, np.maximum.accumulate(c)

    def lift(self, alpha_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cl, dcl/dalpha [1/deg], clipped) on the attached branch, clipped outside it.

        The lift curve is the monotone (PCHIP-like, Fritsch–Carlson) cubic through the
        branch points, so the lifting-line Newton iteration sees a continuous slope;
        ``clipped`` is -1 / +1 below / above the branch (cl held at cl_min / cl_max).
        """
        a = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
        xa, ya = self._branch()
        if len(xa) < 2:
            cl = np.full_like(a, float(ya[0]) if len(ya) else float("nan"))
            return cl, np.zeros_like(a), np.where(a < (xa[0] if len(xa) else 0), -1, 1)
        cl, d = _pchip(xa, ya, np.clip(a, xa[0], xa[-1]))
        clip = np.where(a > xa[-1], 1, np.where(a < xa[0], -1, 0))
        d = np.where(clip != 0, 0.0, d)
        return cl, d, clip

    def alpha_at_cl(self, cl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Angle [deg] at which the attached branch reaches ``cl``; (alpha, clipped -1/0/+1)."""
        c = np.atleast_1d(np.asarray(cl, dtype=float))
        xa, ya = self._branch()
        if len(xa) < 2:
            return np.full_like(c, float(xa[0]) if len(xa) else float("nan")), np.zeros(c.shape, int)
        # strictly increasing for the inverse (flat stretches get a tiny slope)
        yy = ya + 1e-9 * np.arange(len(ya))
        a = np.interp(c, yy, xa)
        clip = np.where(c > yy[-1], 1, np.where(c < yy[0], -1, 0))
        return a, clip

    def drag(self, alpha_deg: np.ndarray) -> np.ndarray:
        """Profile drag at ``alpha_deg`` (linear in alpha, held at the ends of the data)."""
        return np.interp(alpha_deg, self.alpha, self.cd)

    def moment(self, alpha_deg: np.ndarray) -> np.ndarray:
        """cm about c/4 at ``alpha_deg`` (linear in alpha, held at the ends)."""
        return np.interp(alpha_deg, self.alpha, self.cm)


def _pchip(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone piecewise-cubic Hermite interpolation and its derivative (Fritsch–Carlson)."""
    h = np.diff(x)
    dlt = np.diff(y) / h
    m = np.zeros_like(y)
    if len(x) == 2:
        m[:] = dlt[0]
    else:
        w1 = 2 * h[1:] + h[:-1]
        w2 = h[1:] + 2 * h[:-1]
        same = (dlt[:-1] * dlt[1:]) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = (w1 + w2) / (w1 / dlt[:-1] + w2 / dlt[1:])
        m[1:-1] = np.where(same, hm, 0.0)

        def end(h0, h1, d0, d1):
            v = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            if np.sign(v) != np.sign(d0):
                return 0.0
            if np.sign(d0) != np.sign(d1) and abs(v) > abs(3 * d0):
                return 3 * d0
            return v
        m[0] = end(h[0], h[1], dlt[0], dlt[1])
        m[-1] = end(h[-1], h[-2], dlt[-1], dlt[-2])
    k = np.clip(np.searchsorted(x, xi, side="right") - 1, 0, len(x) - 2)
    t = (xi - x[k]) / h[k]
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    yi = h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]
    d00 = 6 * t * t - 6 * t
    d10 = 3 * t * t - 4 * t + 1
    d01 = -6 * t * t + 6 * t
    d11 = 3 * t * t - 2 * t
    di = (d00 * y[k] + d01 * y[k + 1]) / h[k] + d10 * m[k] + d11 * m[k + 1]
    return yi, di


def _make_table(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray, cm: np.ndarray, re: float, *,
                failed_above: bool = False, failed_below: bool = False) -> PolarTable:
    """Build a table and locate its attached branch around the minimum-|cl| region."""
    o = np.argsort(alpha)
    a, l, d, m = alpha[o], cl[o], cd[o], cm[o]
    n = len(a)
    if n == 0:
        raise SectionPolarError("no converged section points")
    # walk up / down from the point nearest zero lift while lift keeps rising / falling
    i0 = int(np.argmin(np.abs(l)))
    hi = i0
    best = hi
    for i in range(i0 + 1, n):
        if l[i] > l[best]:
            best = i
        elif l[best] - l[i] > 0.02:          # a clear drop: past cl_max
            break
    hi = best
    lo = i0
    best = lo
    for i in range(i0 - 1, -1, -1):
        if l[i] < l[best]:
            best = i
        elif l[i] - l[best] > 0.02:
            break
    lo = best
    top = "stall" if (hi < n - 1 or failed_above) else "range"
    bottom = "stall" if (lo > 0 or failed_below) else "range"
    return PolarTable(alpha=a, cl=l, cd=d, cm=m, re=float(re), i_lo=lo, i_hi=hi, top_limit=top, bottom_limit=bottom)


def blend_tables(tables: Sequence[PolarTable], weights: Sequence[float]) -> PolarTable:
    """Weighted blend of tables on their common angle range (cl, cm linear; cd in log).

    Used both for Reynolds interpolation (weights from log Re) and for strips
    between two sections with different airfoils (weights from the span
    position). The blend keeps the "stall" verdicts: an extreme of the blend
    that sits at the end of the common range is a real stall only when every
    source table reached stall there.
    """
    ts = [t for t, w in zip(tables, weights) if w > 1e-12]
    ws = np.array([w for w in weights if w > 1e-12], float)
    if len(ts) == 1:
        return ts[0]
    ws = ws / ws.sum()
    lo = max(float(t.alpha[0]) for t in ts)
    hi = min(float(t.alpha[-1]) for t in ts)
    grid = np.unique(np.concatenate([t.alpha[(t.alpha >= lo - 1e-9) & (t.alpha <= hi + 1e-9)] for t in ts]))
    if len(grid) == 0:
        grid = np.array([lo])
    cl = sum(w * np.interp(grid, t.alpha, t.cl) for t, w in zip(ts, ws))
    cm = sum(w * np.interp(grid, t.alpha, t.cm) for t, w in zip(ts, ws))
    lcd = sum(w * np.interp(grid, t.alpha, np.log(np.maximum(t.cd, 1e-6))) for t, w in zip(ts, ws))
    re = float(np.exp(sum(w * math.log(t.re) for t, w in zip(ts, ws))))
    tab = _make_table(grid, cl, np.exp(lcd), cm, re)
    top_all = all(t.top_limit == "stall" and t.alpha_cl_max <= hi + 1e-9 for t in ts)
    bot_all = all(t.bottom_limit == "stall" and t.alpha_cl_min >= lo - 1e-9 for t in ts)
    top = "stall" if (tab.i_hi < len(grid) - 1 or top_all) else "range"
    bottom = "stall" if (tab.i_lo > 0 or bot_all) else "range"
    return PolarTable(alpha=tab.alpha, cl=tab.cl, cd=tab.cd, cm=tab.cm, re=re, i_lo=tab.i_lo, i_hi=tab.i_hi,
                      top_limit=top, bottom_limit=bottom)


@dataclass
class _Node:
    """All section points computed for one airfoil at one Reynolds number."""
    re: float
    points: dict[float, tuple[float, float, float, bool]] = field(default_factory=dict)  # alpha -> cl, cd, cm, ok
    table: PolarTable | None = None

    def converged(self) -> list[float]:
        return sorted(a for a, p in self.points.items() if p[3])

    def build(self) -> PolarTable:
        ok = self.converged()
        if not ok:
            raise SectionPolarError(f"no converged section point at Re = {self.re:.3g}")
        a = np.array(ok)
        data = np.array([self.points[x][:3] for x in ok])
        failed_above = any((not p[3]) and x > a[-1] for x, p in self.points.items())
        failed_below = any((not p[3]) and x < a[0] for x, p in self.points.items())
        self.table = _make_table(a, data[:, 0], data[:, 1], data[:, 2], self.re,
                                 failed_above=failed_above, failed_below=failed_below)
        return self.table


class SectionPolars:
    """Cached 2D viscous section polars per (airfoil, Reynolds number) for wing analyses.

    ``coords_provider(name)`` resolves airfoils that are not NACA 4/5-digit codes
    (the project's "design:<id>" airfoils; ``simulation.wing`` supplies it) and
    may return anything :func:`~aerolume.solvers.airfoil.prepare_coords`
    accepts. ``ncrit``, ``mach`` and the trip positions are those of every polar
    in the cache (one cache per flow condition); ``panels`` and ``alpha_step``
    trade accuracy for speed; ``re_ratio`` is the largest Reynolds-number factor
    between two interpolation nodes. Thread-safe (one lock; polars run while it
    is held, so concurrent wing analyses share rather than duplicate work).

    Typical use::

        sp = SectionPolars(ncrit=9, mach=0.0, coords_provider=my_lookup)
        sp.ensure({"2412": [(3e5, -6, 12), (1.5e5, -6, 12)]}, progress=cb, cancel=ev)
        tab = sp.table("2412", 2.2e5)        # blended PolarTable
    """

    def __init__(self, *, ncrit: float = 9.0, mach: float = 0.0, xtr_upper: float = 1.0,
                 xtr_lower: float = 1.0, panels: int = DEFAULT_PANELS, alpha_step: float = 1.0,
                 re_ratio: float = 2.0, coords_provider: CoordsProvider | None = None) -> None:
        if not (alpha_step > 0 and re_ratio > 1.0):
            raise ValueError("alpha_step must be positive and re_ratio greater than 1")
        self.ncrit = float(ncrit)
        self.mach = float(mach)
        self.xtr_upper = float(xtr_upper)
        self.xtr_lower = float(xtr_lower)
        self.panels = int(panels)
        self.alpha_step = float(alpha_step)
        self.re_ratio = float(re_ratio)
        self.coords_provider = coords_provider
        self._coords: dict[str, np.ndarray] = {}
        self._nodes: dict[str, dict[float, _Node]] = {}
        self._tables: dict[tuple[str, float], PolarTable] = {}
        self._lock = threading.RLock()
        self.notes: list[str] = []
        self.warnings: list[str] = []
        self.points_computed = 0
        self.seconds = 0.0

    # -- airfoils --------------------------------------------------------------

    def coords(self, name: str) -> np.ndarray:
        """Outline of airfoil ``name`` (NACA 4/5-digit generated, others via the provider)."""
        key = airfoil_key(name)
        with self._lock:
            if key in self._coords:
                return self._coords[key]
            if key.isdigit():
                from ...geometry.airfoils import naca
                xy = naca(key, 200)
            elif self.coords_provider is not None:
                xy = self.coords_provider(str(name))
                if xy is None:
                    raise ValueError(f"Airfoil {name!r} could not be found.")
            elif key.startswith("design:"):
                raise ValueError(f"Airfoil {name!r} refers to a 2D design of the project; the caller must "
                                 "supply coords_provider to resolve it.")
            else:
                raise ValueError(f"Unknown airfoil {name!r}: use a NACA 4- or 5-digit code or 'design:<id>'.")
            self._coords[key] = xy
            return xy

    def matches(self, *, ncrit: float, mach: float) -> bool:
        """True when this cache was built for the given transition and Mach settings."""
        return math.isclose(self.ncrit, ncrit) and math.isclose(self.mach, mach)

    # -- planning ----------------------------------------------------------------

    def _grid(self, lo: float, hi: float) -> list[float]:
        st = self.alpha_step
        a0 = math.floor(max(lo, ALPHA_MIN) / st + 1e-9) * st
        a1 = math.ceil(min(hi, ALPHA_MAX) / st - 1e-9) * st
        n = round((a1 - a0) / st) + 1
        return [round(a0 + k * st, 6) for k in range(max(n, 1))]

    def _node_set(self, key: str, res: Iterable[float]) -> list[float]:
        """Reynolds nodes (existing plus new) that cover ``res`` within ``re_ratio``."""
        have = sorted(self._nodes.get(key, {}).keys())
        req = sorted({float(r) for r in res})
        lmax = math.log(self.re_ratio)

        def covered(r: float, nodes: list[float]) -> bool:
            lr = math.log(r)
            ln = [math.log(x) for x in nodes]
            if any(abs(lr - x) <= SNAP for x in ln):
                return True
            below = [x for x in ln if x <= lr]
            above = [x for x in ln if x >= lr]
            return bool(below and above and (min(above) - max(below)) <= lmax + 1e-12)

        # Greedy on the grid (module docstring): take the lowest uncovered Re, r. With a node
        # below it within re_ratio, the next node brackets everything up to it: when some
        # uncovered Re lie beyond what that node can reach, the highest grid node it reaches
        # (the most progress per node); otherwise the grid node at or just above the highest
        # uncovered Re (the closest bracket). Without such a node, the grid node nearest to r,
        # which is within SNAP of it. Each step covers r, so the loop ends; the bound is a guard.
        nodes = list(have)
        for _ in range(len(req) + 8):
            miss = [r for r in req if not covered(r, nodes)]
            if not miss:
                break
            r = miss[0]
            lr = math.log(r)
            below = [x for x in nodes if math.log(x) <= lr]
            g = re_grid_value(round(_grid_pos(r)))
            if below and lr - math.log(max(below)) <= lmax + 1e-12:
                reach = math.log(max(below)) + lmax
                if any(math.log(m) > reach + 1e-12 for m in miss):
                    far = re_grid_value(math.floor(reach / math.log(10.0) * RE_GRID_PER_DECADE + 1e-9))
                    if math.log(far) >= lr - SNAP:
                        g = far
                else:
                    target = miss[-1]
                    up = re_grid_value(math.ceil(_grid_pos(target) - 1e-9))
                    g = up if math.log(up) <= reach + 1e-12 else re_grid_value(round(_grid_pos(target)))
            if g in nodes:                         # cannot happen on the grid; never loop on it
                break
            nodes = sorted(set(nodes) | {g})
        for r in req:                              # last resort: a node exactly there
            if not covered(r, nodes):
                nodes = sorted(set(nodes) | {float(f"{r:.6g}")})
        return nodes

    def _bracket(self, key: str, re: float) -> list[tuple[float, float]]:
        """[(node Re, weight)] for one Reynolds number from the existing nodes of ``key``."""
        nodes = sorted(self._nodes.get(key, {}).keys())
        if not nodes:
            raise SectionPolarError(f"no section polar for {key!r} yet")
        lr = math.log(re)
        ln = np.log(nodes)
        k = int(np.argmin(np.abs(ln - lr)))
        if abs(ln[k] - lr) <= SNAP or len(nodes) == 1:
            return [(nodes[k], 1.0)]
        if lr <= ln[0]:
            return [(nodes[0], 1.0)]
        if lr >= ln[-1]:
            return [(nodes[-1], 1.0)]
        j = int(np.searchsorted(ln, lr))
        t = (lr - ln[j - 1]) / (ln[j] - ln[j - 1])
        return [(nodes[j - 1], 1.0 - t), (nodes[j], t)]

    def plan(self, demands: dict[str, Sequence[tuple[float, float, float]]]) -> dict[tuple[str, float], list[float]]:
        """Section angles still to compute: {(airfoil key, node Re): [alpha, ...]}.

        ``demands`` maps an airfoil name to (Re, alpha_lo, alpha_hi) requests: the
        section must be known over [alpha_lo, alpha_hi] at that Reynolds number.
        """
        todo: dict[tuple[str, float], list[float]] = {}
        with self._lock:
            for name, reqs in demands.items():
                key = airfoil_key(name)
                reqs = [(float(r), float(a), float(b)) for r, a, b in reqs if r > 0]
                if not reqs:
                    continue
                nodes = self._node_set(key, [r for r, _, _ in reqs])
                ranges: dict[float, list[float]] = {}
                for r, a, b in reqs:
                    lr = math.log(r)
                    ln = [math.log(x) for x in nodes]
                    near = [nodes[i] for i in range(len(nodes)) if abs(ln[i] - lr) <= SNAP]
                    if near:
                        used = near
                    else:
                        below = [x for x in nodes if x < r]
                        above = [x for x in nodes if x > r]
                        used = ([max(below)] if below else []) + ([min(above)] if above else [])
                    for x in used:
                        lo, hi = ranges.get(x, [a, b])
                        ranges[x] = [min(lo, a), max(hi, b)]
                for x, (lo, hi) in ranges.items():
                    have = self._nodes.get(key, {}).get(x)
                    grid = self._grid(lo, hi)
                    miss = [g for g in grid if have is None or g not in have.points]
                    if miss:
                        todo[(key, x)] = miss
        return todo

    def cost(self, demands: dict[str, Sequence[tuple[float, float, float]]]) -> int:
        """Number of section points :meth:`ensure` would compute for ``demands``."""
        return sum(len(v) for v in self.plan(demands).values())

    # -- computing ---------------------------------------------------------------

    def ensure(self, demands: dict[str, Sequence[tuple[float, float, float]]], *,
               progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> int:
        """Compute whatever ``demands`` needs that is not cached yet; returns the points computed.

        Each missing (airfoil, Re) block is one warm-started sweep. When it extends
        an existing node, the sweep is seeded with the node's nearest converged angle
        so it starts from a known state. Raises :class:`PolarCancelled` when
        ``cancel`` is set (points finished so far stay cached).
        """
        import time

        from ..airfoil import polar

        with self._lock:
            todo = self.plan(demands)
            runs: list[tuple[str, float, list[float]]] = []
            for (key, re), alphas in sorted(todo.items()):
                node = self._nodes.get(key, {}).get(re)
                runs.extend((key, re, r) for r in self._runs(alphas, node.converged() if node else []))
            total = sum(len(r) for _, _, r in runs)
            done = 0
            for key, re, run in runs:
                if cancel is not None and cancel.is_set():
                    raise PolarCancelled
                node = self._nodes.setdefault(key, {}).setdefault(re, _Node(re=re))
                t0 = time.perf_counter()

                n_run = len(run)

                def cb(frac: float, msg: str, _done=done, _n=n_run, _key=key, _re=re) -> None:
                    if progress is not None and total > 0:
                        progress(min(1.0, (_done + frac * _n) / total),
                                 f"Section polar {_key}, Re {_re:.3g}: {msg}")

                res = polar(self.coords(key), re, run, mach=self.mach, ncrit=self.ncrit,
                            xtr_upper=self.xtr_upper, xtr_lower=self.xtr_lower, panels=self.panels,
                            distributions=False, progress=cb, cancel=cancel)
                self.seconds += time.perf_counter() - t0
                cancelled = cancel is not None and cancel.is_set()
                for p in res.points:
                    if cancelled and not p.converged:
                        continue                  # not attempted to the end: leave it to a later call
                    node.points[round(float(p.alpha), 6)] = (p.cl, p.cd, p.cm, bool(p.converged))
                self.points_computed += len(res.points)
                for w in res.warnings:
                    if w.startswith(("Cancelled", "Not converged")):
                        continue
                    msg = f"Section {key} at Re {re:.3g}: {w}"
                    if msg not in self.warnings:
                        self.warnings.append(msg)
                node.table = None
                self._tables = {k: v for k, v in self._tables.items() if k[0] != key}
                done += len(run)
                if cancelled:
                    raise PolarCancelled
            return done

    @staticmethod
    def _runs(alphas: Sequence[float], conv: Sequence[float]) -> list[list[float]]:
        """Split missing angles into warm-startable sweeps around what a node already has.

        :func:`~aerolume.solvers.airfoil.polar` starts at the angle closest to zero and
        walks outward. Angles above the converged range are run together with its top
        angle as the seed (so the sweep starts from a state that converged before),
        angles below it with its bottom angle; gaps inside the range run on their own.
        """
        if not conv:
            return [sorted(alphas)]
        lo, hi = conv[0], conv[-1]
        above = sorted(a for a in alphas if a > hi)
        below = sorted(a for a in alphas if a < lo)
        inside = sorted(a for a in alphas if lo <= a <= hi)
        out = []
        if above:
            out.append([hi] + above)
        if below:
            out.append(below + [lo])
        if inside:
            out.append(inside)
        return out

    def node_res(self, name: str) -> list[float]:
        """Reynolds numbers at which ``name`` has cached polars."""
        with self._lock:
            return sorted(self._nodes.get(airfoil_key(name), {}).keys())

    def _settings(self) -> dict[str, float]:
        return {"ncrit": self.ncrit, "mach": self.mach, "xtr_upper": self.xtr_upper, "xtr_lower": self.xtr_lower,
                "panels": self.panels, "alpha_step": self.alpha_step}

    def state(self) -> dict:
        """The computed section points as a JSON-safe dict (to persist the cache between sessions).

        Airfoils are keyed by name: a caller that persists the state must drop it
        when a "design:<id>" airfoil's outline changes.
        """
        with self._lock:
            return {"settings": self._settings(),
                    "nodes": {key: {repr(re): [[a, *map(float, p[:3]), bool(p[3])] for a, p in sorted(n.points.items())]
                                    for re, n in nodes.items()} for key, nodes in self._nodes.items()}}

    def load_state(self, state: dict) -> None:
        """Merge points saved by :meth:`state`; raises ``ValueError`` if the settings differ."""
        saved = state.get("settings", {})
        mine = self._settings()
        if any(not math.isclose(float(saved.get(k, math.nan)), float(v)) for k, v in mine.items()):
            raise ValueError(f"section-polar state was computed with {saved}, this cache uses {mine}")
        with self._lock:
            for key, nodes in state.get("nodes", {}).items():
                for re_s, pts in nodes.items():
                    re = float(re_s)
                    node = self._nodes.setdefault(key, {}).setdefault(re, _Node(re=re))
                    for a, cl, cd, cm, ok in pts:
                        node.points.setdefault(round(float(a), 6), (cl, cd, cm, bool(ok)))
                    node.table = None
            self._tables.clear()

    def table(self, name: str, re: float) -> PolarTable:
        """Section table of ``name`` at Reynolds number ``re`` (log-Re blend of the bracketing nodes).

        Raises :class:`SectionPolarError` when nothing converged at a needed node.
        A Reynolds number outside the nodes uses the nearest node (no extrapolation);
        :meth:`ensure` normally places nodes so that this does not happen.
        """
        key = airfoil_key(name)
        cache_key = (key, float(f"{re:.6g}"))
        with self._lock:
            tab = self._tables.get(cache_key)
            if tab is not None:
                return tab
            parts = []
            for x, w in self._bracket(key, re):
                node = self._nodes[key][x]
                parts.append((node.table or node.build(), w))
            tab = blend_tables([p for p, _ in parts], [w for _, w in parts])
            self._tables[cache_key] = tab
            return tab


__all__ = [
    "ALPHA_MAX",
    "ALPHA_MIN",
    "RE_GRID_PER_DECADE",
    "PolarCancelled",
    "PolarTable",
    "SectionPolarError",
    "SectionPolars",
    "airfoil_key",
    "blend_tables",
    "re_grid_value",
]
