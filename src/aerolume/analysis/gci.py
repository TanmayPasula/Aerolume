"""Discretisation uncertainty from grid-convergence studies.

A single CFD run says nothing about how far its numbers are from the grid-converged
answer. Running the *same* simulation on three (or more) systematically refined grids
does: the change between grids gives the observed order of accuracy, Richardson
extrapolation gives an estimate of the grid-converged value, and Roache's Grid
Convergence Index (GCI) turns the fine-grid error estimate into an uncertainty band.
This module implements the procedure of Celik et al. (2008), the one the ASME Journal
of Fluids Engineering requires of its authors, with three additions that time-averaged
results of an unsteady solver need:

* **Statistical noise.** Each grid's value is a time average with a standard error.
  When the differences between grids are not larger than about twice their combined
  standard errors, the observed order measures the noise, not the discretisation:
  the study is marked ``noise_limited`` and its order is not used as if it were
  known. A Monte Carlo propagation (the values resampled within their standard
  errors, a fixed seed so the numbers repeat) gives the spread of the order and of
  the extrapolated value that the noise alone causes.
* **Honest asymptotic check.** Roache's ratio GCI₃₂ / (r^p GCI₂₁) equals |φ₁/φ₂|
  identically when the order p is computed from the same three grids at a constant
  refinement ratio, so it cannot fail and is reported for completeness only. The
  verdict ``asymptotic`` instead compares the observed order with the formal order
  of the scheme (within :data:`ORDER_TOLERANCE`, the criterion of Oberkampf & Roy
  2010, section 8.7) and, with four or more grids, checks that successive triplets
  give the same order.
* **Combined band.** The numerical band (GCI, about 95 % coverage by Roache's
  calibration) and the statistical band (two standard errors, about 95 %) are
  combined in quadrature for the total uncertainty of the fine-grid value. That
  assumes independent errors with similar coverage factors; both parts are always
  reported separately as well.

Conventions: grid 1 is the finest (smallest representative cell size h), grid 3 the
coarsest; ε₂₁ = φ₂ − φ₁ and ε₃₂ = φ₃ − φ₂; r₂₁ = h₂/h₁ > 1. Uncertainties are
reported both relative to |φ₁| (Celik's form, undefined when φ₁ ≈ 0, as for the lift
of a symmetric body) and absolute (``u_num``, the half-width of the band on φ₁).

Only numpy and the standard library are used (DESIGN.md section 2).

References: P. J. Roache, *Verification and Validation in Computational Science and
Engineering*, Hermosa 1998; I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H.
Coleman, P. E. Raad, "Procedure for estimation and reporting of uncertainty due to
discretization in CFD applications", J. Fluids Eng. 130(7) 078001, 2008; W. L.
Oberkampf & C. J. Roy, *Verification and Validation in Scientific Computing*,
Cambridge 2010; L. Eça & M. Hoekstra, "A procedure for the estimation of the
numerical uncertainty of CFD calculations based on grid refinement studies", J.
Comput. Phys. 262, 104-130, 2014 (the least-squares fit of :func:`fit_order`, without
their weighting and safety-factor rules).
"""
from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Safety factor of a three-grid study with an observed order (Roache 1998; Celik et al. 2008).
FS_THREE_GRID = 1.25
#: Safety factor of a two-grid study, whose order can only be assumed (Roache 1998).
FS_TWO_GRID = 3.0
#: Celik et al. (2008) recommend refinement ratios above 1.3, so the grid differences
#: are larger than other error sources (iteration, round-off, here also averaging).
MIN_REFINEMENT_RATIO = 1.3
#: Observed order within this fraction of the formal order counts as asymptotic
#: (Oberkampf & Roy 2010, section 8.7, which then keeps Fs = 1.25).
ORDER_TOLERANCE = 0.1
#: Grid differences smaller than this many combined standard errors are noise.
NOISE_SIGMAS = 2.0
#: Monte Carlo samples for the noise propagation, and the fixed seed (results repeat).
MC_SAMPLES = 4000
MC_SEED = 20081

#: Default formal order: the interior discretisation of both Navier-Stokes solvers
#: is second order (Taylor-Green verification, docs/theory_ns2d.md section 3). The
#: diffuse immersed boundary converges more slowly near walls, so observed orders
#: between 1 and 2 are expected for forces; see docs/methods.md.
FORMAL_ORDER = 2.0

#: Order searched in [P_MIN, P_MAX] when the fixed-point iteration of Celik's
#: equation does not settle (strongly non-uniform refinement ratios).
P_MIN, P_MAX = 1e-3, 20.0

CONVERGENCE_TYPES = ("monotone", "oscillatory", "divergent", "oscillatory-divergent", "insensitive",
                     "converged-fine")

_NAN = float("nan")


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(x: Any) -> float:
    """``x`` as a float, nan for None / non-numbers."""
    try:
        return float(x) if x is not None else _NAN
    except (TypeError, ValueError):
        return _NAN


# ---------------------------------------------------------------------------
# the order of accuracy
# ---------------------------------------------------------------------------

def convergence_type(phi1: float, phi2: float, phi3: float, *, atol: float = 0.0,
                     h: Sequence[float] | None = None) -> tuple[str, float]:
    """How three values on fine, medium and coarse grids converge, and the ratio R = ε₂₁/ε₃₂.

    With equal refinement ratios: 0 < R < 1 is monotone convergence, -1 < R < 0
    oscillatory convergence, R > 1 monotone divergence and R < -1 oscillatory
    divergence (Celik et al. 2008). With unequal ratios (``h`` given) the threshold 1 is
    replaced by ln r₂₁ / ln r₃₂: for φ = φ₀ + C hᵖ with p > 0, ε₃₂/ε₂₁ =
    r₂₁ᵖ (r₃₂ᵖ − 1)/(r₂₁ᵖ − 1), which exceeds ln r₃₂ / ln r₂₁ (its limit as p → 0) and
    grows with p, so a coarse step much smaller than the fine one is not mistaken for
    divergence. Two special cases: "insensitive" when neither difference exceeds
    ``atol`` (the quantity does not depend on the grid, e.g. a symmetry-zero side
    force), and "converged-fine" when only the fine-pair difference vanishes (the order
    is then unbounded and the fine value is taken as grid-converged).
    """
    e21, e32 = phi2 - phi1, phi3 - phi2
    z21, z32 = abs(e21) <= atol, abs(e32) <= atol
    if z21 and z32:
        return "insensitive", _NAN
    if z21:
        return "converged-fine", 0.0
    if z32:
        return "divergent", math.inf
    R = e21 / e32
    limit = 1.0
    if h is not None:
        h1, h2, h3 = (float(v) for v in h)
        if h1 > 0 and h2 > h1 and h3 > h2:
            limit = math.log(h2 / h1) / math.log(h3 / h2)        # = 1 for a constant ratio
    if 0.0 < R < limit:
        return "monotone", R
    if -limit < R < 0.0:
        return "oscillatory", R
    if R >= limit:
        return "divergent", R
    return "oscillatory-divergent", R


def observed_order(h: Sequence[float], phi: Sequence[float], *, tol: float = 1e-12) -> float:
    """Observed order of accuracy p from three grids (fine first), Celik et al. (2008) eq. 3-5.

    ``p = |ln|ε₃₂/ε₂₁| + q(p)| / ln r₂₁`` with ``q(p) = ln((r₂₁^p − s)/(r₃₂^p − s))`` and
    ``s = sign(ε₃₂/ε₂₁)``. For a constant refinement ratio q = 0 and p is explicit.
    Otherwise the root is bracketed on [P_MIN, P_MAX] and bisected. The root of the
    equation *without* the outer absolute value is preferred: for a power law it is
    the true order (ln|ε₃₂/ε₂₁| + q(p) = p ln r₂₁ > 0), while the absolute value can add
    a spurious large-p root when r₃₂ ≫ r₂₁ (the fixed-point iteration Celik et al.
    suggest finds that one: 20 instead of 1.3 for h = 1, 1.05, 4). Returns nan when a
    difference is zero or there is no root in [P_MIN, P_MAX].
    """
    h1, h2, h3 = (float(v) for v in h)
    f1, f2, f3 = (float(v) for v in phi)
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0.0 or e32 == 0.0 or not all(math.isfinite(v) for v in (h1, h2, h3, f1, f2, f3)):
        return _NAN
    r21, r32 = h2 / h1, h3 / h2
    if not (r21 > 1.0 and r32 > 1.0):
        raise ValueError("grids must be ordered fine to coarse with h1 < h2 < h3")
    s = 1.0 if e32 / e21 > 0 else -1.0
    a = math.log(abs(e32 / e21))
    lr21 = math.log(r21)

    def q(p: float) -> float:
        try:
            num, den = r21 ** p - s, r32 ** p - s
        except OverflowError:
            return _NAN
        if num <= 0 or den <= 0:
            return _NAN
        return math.log(num / den)

    if abs(r21 - r32) <= 1e-12 * r21:
        p = abs(a) / lr21
        return p if p > 0 else _NAN

    def signed(p: float) -> float:
        qp = q(p)
        return p * lr21 - (a + qp) if math.isfinite(qp) else _NAN

    def absolute(p: float) -> float:
        qp = q(p)
        return p * lr21 - abs(a + qp) if math.isfinite(qp) else _NAN

    grid = np.geomspace(P_MIN, P_MAX, 600)
    for g in (signed, absolute):
        vals = np.array([g(float(x)) for x in grid])
        ok = np.isfinite(vals)
        for i in range(len(grid) - 1):
            if not (ok[i] and ok[i + 1] and vals[i] * vals[i + 1] <= 0):
                continue
            lo, hi = float(grid[i]), float(grid[i + 1])
            glo = vals[i]
            while hi - lo > tol * max(1.0, hi):
                mid = 0.5 * (lo + hi)
                gm = g(mid)
                if not math.isfinite(gm):
                    break
                if glo * gm <= 0:
                    hi = mid
                else:
                    lo, glo = mid, gm
            return 0.5 * (lo + hi)
    return _NAN


def richardson(phi_fine: float, phi_coarse: float, r: float, p: float) -> float:
    """Richardson-extrapolated value ``(r^p φ_fine − φ_coarse)/(r^p − 1)`` (nan if undefined)."""
    if not (_finite(p) and p > 0 and r > 1):
        return _NAN
    rp = r ** p
    return (rp * phi_fine - phi_coarse) / (rp - 1.0)


# ---------------------------------------------------------------------------
# a three-grid study
# ---------------------------------------------------------------------------

@dataclass
class GridLevel:
    """One grid of a study: representative cell size ``h`` (any consistent unit, e.g.
    reference lengths per cell) and the value with its standard error."""
    h: float
    value: float
    stderr: float = _NAN
    stderr_reliable: bool = True
    label: str = ""
    sim_id: str = ""
    cells: int = 0


@dataclass
class GCIResult:
    """The outcome of a grid-convergence study of one quantity (see the module docstring).

    Relative quantities (``e_a21``, ``e_ext21``, ``gci_fine``, ``gci_coarse``) are fractions
    of |φ₁| (nan when φ₁ is zero); ``u_num``, ``u_stat`` and ``u_total`` are absolute
    half-widths of about 95 % bands on the fine-grid value φ₁.
    """
    quantity: str
    levels: list[GridLevel]
    r21: float = _NAN
    r32: float = _NAN
    convergence: str = ""
    R: float = _NAN
    p: float = _NAN                 # observed order
    p_used: float = _NAN            # order used for the GCI and the extrapolation
    formal_order: float = FORMAL_ORDER
    fs: float = FS_THREE_GRID
    phi_ext: float = _NAN           # Richardson-extrapolated value
    e_a21: float = _NAN             # approximate relative error |(φ1 − φ2)/φ1|
    e_ext21: float = _NAN           # extrapolated relative error |(φext − φ1)/φext|
    gci_fine: float = _NAN          # GCI of the fine grid, relative
    gci_coarse: float = _NAN        # GCI of the medium grid (pair 3-2), relative
    asymptotic_ratio: float = _NAN  # Roache: GCI32 / (r21^p GCI21); ~|φ1/φ2| by construction
    asymptotic: bool | None = None
    u_num: float = _NAN             # absolute numerical band on φ1 (= GCI × |φ1|)
    u_stat: float = _NAN            # absolute statistical band on φ1 (2 standard errors)
    u_total: float = _NAN           # both combined in quadrature
    noise_limited: bool | None = None
    monte_carlo: dict[str, float] = field(default_factory=dict)
    triplet_orders: list[float] = field(default_factory=list)   # 4+ grids: p of each consecutive triplet
    fit: dict[str, float] = field(default_factory=dict)         # 4+ grids: least-squares (φ0, C, p)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fine(self) -> GridLevel:
        return self.levels[0]

    @property
    def usable(self) -> bool:
        """Whether the GCI is a usable uncertainty estimate (converging, not noise-limited)."""
        return (self.convergence in ("monotone", "converged-fine", "insensitive")
                and not self.noise_limited and _finite(self.u_num))

    def to_dict(self) -> dict:
        """JSON-safe dict (nan / inf become None)."""
        def clean(v: Any) -> Any:
            if isinstance(v, float):
                return v if math.isfinite(v) else None
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v
        return clean(asdict(self))

    def summary(self) -> str:
        """One sentence: the extrapolated value, the order and the uncertainty band."""
        f1 = self.fine.value
        if self.convergence == "insensitive":
            return f"{self.quantity}: no change between the grids ({_g(f1)}); no discretisation error is visible."
        if not _finite(self.u_num):
            return (f"{self.quantity}: {self.convergence} ({_g(f1)} on the finest grid); no grid-convergence "
                    "uncertainty can be given.")
        rel = f" ({100 * self.gci_fine:.2g} %)" if _finite(self.gci_fine) else ""
        p = f"observed order {self.p:.2f}" if _finite(self.p) else "order not determined"
        tail = " Noise-limited: the grid differences are within the statistical error." if self.noise_limited else ""
        return (f"{self.quantity}: {_g(f1)} ± {_g(self.u_num)}{rel} (GCI, fine grid; {p}; extrapolated "
                f"{_g(self.phi_ext)}).{tail}")


def _g(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}g}" if _finite(x) else "—"


def _steady(value: float, stderr: float) -> bool:
    """A standard error that is round-off of a steady value (``analysis.physics.steady``)."""
    from .physics import steady
    return steady(value, stderr)


def _levels(h: Sequence[float], phi: Sequence[float], stderr: Sequence[float] | None,
            reliable: Sequence[bool] | None, labels: Sequence[str] | None,
            sim_ids: Sequence[str] | None, cells: Sequence[int] | None) -> list[GridLevel]:
    n = len(h)
    if len(phi) != n:
        raise ValueError(f"{n} grid sizes but {len(phi)} values")
    se = list(stderr) if stderr is not None else [_NAN] * n
    rel = list(reliable) if reliable is not None else [True] * n
    lab = list(labels) if labels is not None else [""] * n
    ids = list(sim_ids) if sim_ids is not None else [""] * n
    cc = list(cells) if cells is not None else [0] * n
    # a steady level's standard error is round-off (the float64 mean of identical samples, flagged
    # "unreliable" by batch means for lack of independent samples): it has no sampling error to
    # estimate, so it does not make the noise test "optimistic"
    rel = [bool(r) or _steady(_f(phi[i]), _f(se[i])) for i, r in enumerate(rel)]
    out = [GridLevel(float(h[i]), _f(phi[i]), _f(se[i]), bool(rel[i]), str(lab[i]), str(ids[i]), int(cc[i] or 0))
           for i in range(n)]
    out.sort(key=lambda lv: lv.h)
    for a, b in itertools.pairwise(out):
        if not b.h > a.h:
            raise ValueError("grid sizes must be distinct")
    if any(not (lv.h > 0 and math.isfinite(lv.h)) for lv in out):
        raise ValueError("grid sizes must be positive and finite")
    return out


def _core(lv: Sequence[GridLevel], formal: float, fs: float | None, method: str) -> dict[str, Any]:
    """GCI numbers of the finest three levels (no noise analysis): a dict of GCIResult fields."""
    l1, l2, l3 = lv[0], lv[1], lv[2]
    f1, f2, f3 = l1.value, l2.value, l3.value
    r21, r32 = l2.h / l1.h, l3.h / l2.h
    scale = max(abs(f1), abs(f2), abs(f3))
    ctype, R = convergence_type(f1, f2, f3, atol=1e-12 * scale if scale else 0.0, h=(l1.h, l2.h, l3.h))
    out: dict[str, Any] = {"r21": r21, "r32": r32, "convergence": ctype, "R": R}
    if ctype == "insensitive":
        out.update(p=_NAN, p_used=_NAN, phi_ext=f1, e_a21=0.0, e_ext21=0.0, gci_fine=0.0, gci_coarse=0.0,
                   u_num=0.0, fs=fs if fs is not None else FS_THREE_GRID)
        return out
    if ctype == "converged-fine":
        e32 = abs(f3 - f2)
        out.update(p=math.inf, p_used=_NAN, phi_ext=f1, e_a21=0.0, e_ext21=0.0, gci_fine=0.0,
                   gci_coarse=(FS_THREE_GRID * e32 / abs(f2)) if f2 else _NAN, u_num=0.0,
                   fs=fs if fs is not None else FS_THREE_GRID)
        return out
    p = observed_order([l1.h, l2.h, l3.h], [f1, f2, f3])
    out["p"] = p
    if ctype not in ("monotone", "oscillatory") or not _finite(p):
        out.update(p_used=_NAN, phi_ext=_NAN, e_a21=abs((f1 - f2) / f1) if f1 else _NAN, e_ext21=_NAN,
                   gci_fine=_NAN, gci_coarse=_NAN, u_num=_NAN, fs=fs if fs is not None else FS_THREE_GRID)
        return out
    # the order and safety factor of the GCI
    if method == "roy" and _finite(formal) and formal > 0:
        if abs(p - formal) <= ORDER_TOLERANCE * formal:
            p_used, fs_used = p, (fs if fs is not None else FS_THREE_GRID)
        else:
            p_used, fs_used = min(max(0.5, p), formal), (fs if fs is not None else FS_TWO_GRID)
    elif method == "celik":
        p_used, fs_used = p, (fs if fs is not None else FS_THREE_GRID)
    else:
        raise ValueError(f"unknown GCI method {method!r}; use 'celik' or 'roy'")
    rp21, rp32 = r21 ** p_used, r32 ** p_used
    e21, e32 = f2 - f1, f3 - f2
    phi_ext = (rp21 * f1 - f2) / (rp21 - 1.0)
    u_num = fs_used * abs(e21) / (rp21 - 1.0)
    u_32 = fs_used * abs(e32) / (rp32 - 1.0)
    out.update(p_used=p_used, fs=fs_used, phi_ext=phi_ext,
               e_a21=abs(e21 / f1) if f1 else _NAN,
               e_ext21=abs((phi_ext - f1) / phi_ext) if phi_ext else _NAN,
               gci_fine=u_num / abs(f1) if f1 else _NAN,
               gci_coarse=u_32 / abs(f2) if f2 else _NAN,
               u_num=u_num)
    if f1 and f2 and u_num > 0:
        out["asymptotic_ratio"] = (u_32 / abs(f2)) / (rp21 * u_num / abs(f1))
    return out


def gci_study(h: Sequence[float], phi: Sequence[float], stderr: Sequence[float] | None = None, *,
              quantity: str = "", reliable: Sequence[bool] | None = None, labels: Sequence[str] | None = None,
              sim_ids: Sequence[str] | None = None, cells: Sequence[int] | None = None,
              formal_order: float = FORMAL_ORDER, fs: float | None = None, method: str = "celik",
              mc_samples: int = MC_SAMPLES, seed: int = MC_SEED) -> GCIResult:
    """Grid-convergence study of one quantity on three or more grids.

    ``h`` are representative cell sizes (any order; sorted fine to coarse here), ``phi``
    the values and ``stderr`` their statistical standard errors (nan or None for
    deterministic solvers). The GCI uses the finest three grids; with four or more
    the order of every consecutive triplet and a least-squares fit over all grids are
    added and the asymptotic verdict requires the last two triplet orders to agree.
    ``method="celik"`` (default) uses the observed order with Fs = 1.25;
    ``method="roy"`` keeps that only when the observed order is within
    :data:`ORDER_TOLERANCE` of ``formal_order`` and otherwise uses Fs = 3 with the
    order limited to [0.5, formal] (Oberkampf & Roy 2010). Raises ``ValueError`` for
    fewer than three grids or repeated grid sizes.
    """
    lv = _levels(h, phi, stderr, reliable, labels, sim_ids, cells)
    if len(lv) < 3:
        raise ValueError("a grid-convergence study needs at least three grids (gci_two_grid for two)")
    if not all(_finite(x.value) for x in lv):
        raise ValueError(f"{quantity or 'value'}: every grid needs a finite value")
    res = GCIResult(quantity=quantity, levels=lv, formal_order=float(formal_order))
    for k, v in _core(lv, formal_order, fs, method).items():
        setattr(res, k, v)
    if min(res.r21, res.r32) < MIN_REFINEMENT_RATIO:
        res.warnings.append(f"Refinement ratio {min(res.r21, res.r32):.2f} is below {MIN_REFINEMENT_RATIO} "
                            "(Celik et al. 2008): the grid differences may be comparable to other errors.")
    if res.convergence == "oscillatory":
        res.warnings.append("The values oscillate between the grids (oscillatory convergence); the GCI is a "
                            "rough estimate and the extrapolation is not reliable.")
    elif res.convergence in ("divergent", "oscillatory-divergent"):
        res.warnings.append("The values do not converge with the grid (the coarse-grid change is not larger "
                            "than the fine-grid change); refine further before quoting a discretisation "
                            "uncertainty.")
    _noise(res, mc_samples, seed, formal_order, fs, method)
    _asymptotic(res, lv, formal_order)
    s1 = lv[0].stderr
    res.u_stat = NOISE_SIGMAS * s1 if _finite(s1) else _NAN
    if _finite(res.u_num):
        res.u_total = math.hypot(res.u_num, res.u_stat) if _finite(res.u_stat) else res.u_num
    if len(lv) >= 4:
        res.triplet_orders = [observed_order([lv[i].h, lv[i + 1].h, lv[i + 2].h],
                                             [lv[i].value, lv[i + 1].value, lv[i + 2].value])
                              for i in range(len(lv) - 2)]
        res.fit = fit_order([x.h for x in lv], [x.value for x in lv])
    return res


def _asymptotic(res: GCIResult, lv: Sequence[GridLevel], formal: float) -> None:
    """The ``asymptotic`` verdict: observed vs formal order, and triplet agreement with 4+ grids."""
    if res.convergence in ("insensitive", "converged-fine"):
        res.asymptotic = True
        return
    if not (_finite(res.p) and res.convergence == "monotone") or res.noise_limited:
        res.asymptotic = None if res.noise_limited else False
        return
    near_formal = _finite(formal) and formal > 0 and abs(res.p - formal) <= ORDER_TOLERANCE * formal
    if len(lv) >= 4:
        ps = [observed_order([lv[i].h, lv[i + 1].h, lv[i + 2].h], [lv[i].value, lv[i + 1].value, lv[i + 2].value])
              for i in range(2)]
        steady = all(_finite(x) for x in ps) and abs(ps[0] - ps[1]) <= ORDER_TOLERANCE * max(ps)
        res.asymptotic = bool(steady)
        if not steady:
            res.notes.append("The observed order changes between the finest and the next triplet of grids: the "
                             "coarser grids are not yet in the asymptotic range.")
        return
    res.asymptotic = bool(near_formal)
    if not near_formal:
        res.notes.append(f"Observed order {res.p:.2f} differs from the formal order {formal:g} by more than "
                         f"{100 * ORDER_TOLERANCE:.0f} %: the grids may not be in the asymptotic range, or a "
                         "lower-order error (such as a diffuse immersed boundary) dominates.")


def _noise(res: GCIResult, n: int, seed: int, formal: float, fs: float | None, method: str) -> None:
    """Noise test and Monte Carlo spread of p and φ_ext from the standard errors."""
    lv = res.levels[:3]
    se = [x.stderr for x in lv]
    if not all(_finite(s) and s >= 0 for s in se) or not any(s > 0 for s in se):
        res.noise_limited = None if any(_finite(s) and s > 0 for s in se) else False
        if res.noise_limited is None:
            res.warnings.append("Some grids have no standard error: the noise test was not made.")
        return
    if not all(x.stderr_reliable for x in lv):
        res.warnings.append("At least one standard error is a rough estimate (too few independent samples): the "
                            "noise test may be optimistic.")
    v = [x.value for x in lv]
    d21, d32 = abs(v[1] - v[0]), abs(v[2] - v[1])
    s21, s32 = math.hypot(se[0], se[1]), math.hypot(se[1], se[2])
    res.noise_limited = bool(d21 <= NOISE_SIGMAS * s21 or d32 <= NOISE_SIGMAS * s32)
    if res.noise_limited:
        res.warnings.append(f"The grid differences ({_g(d21, 3)} fine-medium, {_g(d32, 3)} medium-coarse) are not "
                            f"larger than {NOISE_SIGMAS:g} combined standard errors ({_g(s21, 3)}, {_g(s32, 3)}): the "
                            "observed order and the GCI reflect averaging noise. Average longer or refine more.")
    if n <= 0:
        return
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((n, 3)) * np.asarray(se) + np.asarray(v)
    ps, ext, mono = [], [], 0
    hs = (lv[0].h, lv[1].h, lv[2].h)
    for f1, f2, f3 in draws:
        ctype, _ = convergence_type(float(f1), float(f2), float(f3), h=hs)
        if ctype != "monotone":
            continue
        mono += 1
        p = observed_order(list(hs), [float(f1), float(f2), float(f3)])
        if not _finite(p):
            continue
        ps.append(p)
        ext.append(richardson(float(f1), float(f2), hs[1] / hs[0], p))
    out: dict[str, float] = {"samples": float(n), "seed": float(seed), "fraction_monotone": mono / n}
    if len(ps) >= 20:
        pa, ea = np.asarray(ps), np.asarray(ext)
        ea = ea[np.isfinite(ea)]
        out.update(p_p16=float(np.percentile(pa, 16)), p_p50=float(np.percentile(pa, 50)),
                   p_p84=float(np.percentile(pa, 84)))
        if ea.size >= 20:
            out.update(phi_ext_p16=float(np.percentile(ea, 16)), phi_ext_p50=float(np.percentile(ea, 50)),
                       phi_ext_p84=float(np.percentile(ea, 84)))
    res.monte_carlo = out


def gci_two_grid(h: Sequence[float], phi: Sequence[float], *, quantity: str = "",
                 formal_order: float = FORMAL_ORDER, fs: float = FS_TWO_GRID,
                 stderr: Sequence[float] | None = None) -> GCIResult:
    """Two-grid estimate with an assumed order (Roache: Fs = 3). Less reliable than three grids:
    the order cannot be checked, so the band is wide on purpose."""
    lv = _levels(h, phi, stderr, None, None, None, None)
    if len(lv) != 2:
        raise ValueError("gci_two_grid takes exactly two grids")
    f1, f2 = lv[0].value, lv[1].value
    r = lv[1].h / lv[0].h
    rp = r ** formal_order
    u = fs * abs(f2 - f1) / (rp - 1.0)
    phi_ext = (rp * f1 - f2) / (rp - 1.0)
    res = GCIResult(quantity=quantity, levels=lv, r21=r, convergence="assumed-order", p=_NAN, p_used=formal_order,
                    formal_order=formal_order, fs=fs, phi_ext=phi_ext, e_a21=abs((f1 - f2) / f1) if f1 else _NAN,
                    e_ext21=abs((phi_ext - f1) / phi_ext) if phi_ext else _NAN,
                    gci_fine=u / abs(f1) if f1 else _NAN, u_num=u, asymptotic=None)
    s1 = lv[0].stderr
    res.u_stat = NOISE_SIGMAS * s1 if _finite(s1) else _NAN
    res.u_total = math.hypot(u, res.u_stat) if _finite(res.u_stat) else u
    res.warnings.append(f"Two grids only: the order is assumed ({formal_order:g}), not observed; Fs = {fs:g}.")
    return res


def fit_order(h: Sequence[float], phi: Sequence[float]) -> dict[str, float]:
    """Least-squares fit ``φ = φ0 + C h^p`` over four or more grids.

    p is found by a bounded scan + golden refinement of the residual (φ0 and C are
    linear for a given p). Returns ``{"phi0", "C", "p", "rms_residual"}`` or {} when
    the fit is undefined. This is the basic fit of Eça & Hoekstra (2014); their
    weighted variants and safety-factor rules are not implemented.
    """
    hh = np.asarray(h, float)
    yy = np.asarray(phi, float)
    if hh.size < 4 or not (np.all(np.isfinite(hh)) and np.all(np.isfinite(yy))):
        return {}

    def resid(p: float) -> tuple[float, float, float]:
        A = np.column_stack([np.ones_like(hh), hh ** p])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        r = yy - A @ coef
        return float(np.sqrt(np.mean(r * r))), float(coef[0]), float(coef[1])

    ps = np.linspace(0.25, 6.0, 116)
    vals = [resid(float(p))[0] for p in ps]
    i = int(np.argmin(vals))
    lo, hi = float(ps[max(i - 1, 0)]), float(ps[min(i + 1, len(ps) - 1)])
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if resid(c)[0] < resid(d)[0]:
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    p = 0.5 * (a + b)
    rms, phi0, C = resid(p)
    return {"phi0": phi0, "C": C, "p": p, "rms_residual": rms}


# ---------------------------------------------------------------------------
# studies from simulation results
# ---------------------------------------------------------------------------

#: Quantities a study evaluates per result type: key -> label. Each is read as a
#: Stat (``mean``/``stderr``) or a plain float from the result.
STUDY_KEYS_2D: dict[str, str] = {"cd": "Cd", "cl": "Cl", "cm": "Cm", "strouhal": "St", "nusselt": "Nu",
                                 "heat_rate": "Heat rate"}
STUDY_KEYS_3D: dict[str, str] = {"cd": "CD", "cs": "CS", "cl": "CL", "cm": "Cm", "strouhal": "St",
                                 "nusselt": "Nu", "heat_rate": "Heat rate"}
#: Simulation settings that a grid study may vary; everything else must match.
RESOLUTION_SETTINGS = ("quality", "cells_per_ref", "duration", "average_from", "device")
#: Relative spread of the domain size (cells / cells per reference) that still counts as fixed;
#: the planner rounds grids to FFT-friendly sizes, which moves the edge by a few cells.
DOMAIN_RTOL = 0.04


def _value(result: Any, key: str) -> tuple[float, float, bool]:
    v = getattr(result, key, None)
    if v is None:
        return _NAN, _NAN, True
    if hasattr(v, "mean"):
        return _f(v.mean), _f(getattr(v, "stderr", _NAN)), bool(getattr(v, "stderr_reliable", True))
    return _f(v), _NAN, True


def _dimension(result: Any) -> int:
    grid = tuple(getattr(result, "grid", ()) or ())
    return 3 if len(grid) == 3 else 2


@dataclass
class GridStudy:
    """A grid-convergence study built from simulation results of the same set-up.

    ``sim_ids`` fine to coarse, ``h`` the cell sizes in reference lengths
    (1 / cells_per_ref), ``quantities`` one :class:`GCIResult` per evaluated key,
    ``domain_fixed`` whether the computational domain was the same size on every grid
    (a quality preset may also change the domain, mixing a blockage effect into the
    grid effect).
    """
    sim_ids: list[str]
    h: list[float]
    cells: list[int]
    dimension: int
    quantities: dict[str, GCIResult]
    domain_fixed: bool | None = None
    domain_sizes: list[tuple[float, ...]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sim_ids": list(self.sim_ids), "h": list(self.h), "cells": list(self.cells),
                "dimension": self.dimension, "domain_fixed": self.domain_fixed,
                "domain_sizes": [list(d) for d in self.domain_sizes],
                "quantities": {k: q.to_dict() for k, q in self.quantities.items()},
                "notes": list(self.notes), "warnings": list(self.warnings)}


def grid_study(results: Sequence[Any] | Mapping[str, Any], keys: Iterable[str] | None = None, *,
               formal_order: float = FORMAL_ORDER, method: str = "celik", labels: Mapping[str, str] | None = None,
               mc_samples: int = MC_SAMPLES, seed: int = MC_SEED) -> GridStudy:
    """A :class:`GridStudy` from three or more CFD results of one set-up at different resolutions.

    ``results`` are :class:`~aerolume.model.CFDResult` / :class:`~aerolume.model.CFD3DResult`
    objects (or ``{sim_id: result}``). The cell size of each is ``1 / cells_per_ref``.
    ``keys`` defaults to every quantity of :data:`STUDY_KEYS_2D` / :data:`STUDY_KEYS_3D`
    that all results have finite. A study whose simulated Reynolds numbers differ is
    refused (``ValueError``): that is a Reynolds-number study, not a grid study.
    """
    items = list(results.items()) if isinstance(results, Mapping) else [(getattr(r, "sim_id", ""), r)
                                                                         for r in results]
    if len(items) < 3:
        raise ValueError("a grid study needs at least three results")
    dims = {_dimension(r) for _, r in items}
    if len(dims) != 1:
        raise ValueError("a grid study cannot mix 2D and 3D results")
    dim = dims.pop()
    cpr = [_f(getattr(r, "cells_per_ref", _NAN)) for _, r in items]
    if not all(_finite(c) and c > 0 for c in cpr):
        raise ValueError("every result of a grid study needs its cells per reference length")
    re_sim = [_f(getattr(r, "re_sim", _NAN)) for _, r in items]
    fin = [x for x in re_sim if _finite(x)]
    if fin and (max(fin) - min(fin)) > 1e-6 * max(abs(x) for x in fin):
        raise ValueError("the results simulate different Reynolds numbers "
                         f"({min(fin):.6g} to {max(fin):.6g}); a grid study needs one flow")
    order = sorted(range(len(items)), key=lambda i: -cpr[i])            # fine first
    items = [items[i] for i in order]
    cpr = [cpr[i] for i in order]
    h = [1.0 / c for c in cpr]
    for a, b in itertools.pairwise(h):
        if not b > a * (1 + 1e-9):
            raise ValueError("two results have the same resolution")
    cells = []
    sizes = []
    for (_, r), c in zip(items, cpr, strict=True):
        grid = tuple(int(g) for g in (getattr(r, "grid", ()) or ()))
        cells.append(int(getattr(r, "cells", 0) or (int(np.prod(grid)) if grid else 0)))
        sizes.append(tuple(g / c for g in grid))
    study = GridStudy(sim_ids=[sid for sid, _ in items], h=h, cells=cells, dimension=dim, quantities={},
                      domain_sizes=sizes)
    if all(len(s) == len(sizes[0]) and s for s in sizes):
        arr = np.asarray(sizes, float)
        spread = (arr.max(axis=0) - arr.min(axis=0)) / np.maximum(arr.max(axis=0), 1e-300)
        study.domain_fixed = bool(np.all(spread <= DOMAIN_RTOL))
        if not study.domain_fixed:
            dims_txt = " × ".join(f"{a:.3g}–{b:.3g}" for a, b in zip(arr.min(axis=0), arr.max(axis=0), strict=True))
            study.warnings.append(f"The domain size changes between the grids ({dims_txt} reference lengths): part "
                                  "of the differences is a domain (blockage) effect, not discretisation. Hold the "
                                  "domain fixed (custom quality with explicit domain sizes) for a clean study.")
    names = STUDY_KEYS_3D if dim == 3 else STUDY_KEYS_2D
    wanted = list(keys) if keys is not None else list(names)
    for key in wanted:
        vals = [_value(r, key) for _, r in items]
        if not all(_finite(v[0]) for v in vals):
            if keys is not None:
                study.warnings.append(f"{names.get(key, key)} is not available on every grid; not evaluated.")
            continue
        label = (labels or {}).get(key) or names.get(key, key)
        study.quantities[key] = gci_study(h, [v[0] for v in vals], [v[1] for v in vals], quantity=label,
                                          reliable=[v[2] for v in vals], sim_ids=study.sim_ids, cells=cells,
                                          formal_order=formal_order, method=method, mc_samples=mc_samples,
                                          seed=seed)
    return study


def _setup_key(sim: Any) -> str | None:
    """Canonical text of a simulation's set-up without its resolution settings (None: not CFD)."""
    if getattr(sim, "analysis", None) != "cfd" or not hasattr(sim, "to_dict"):
        return None
    d = sim.to_dict()
    for k in ("id", "name", "status", "created", "finished", "notes", "error"):
        d.pop(k, None)
    cfd = dict(d.get("cfd") or {})
    for k in RESOLUTION_SETTINGS:
        cfd.pop(k, None)
    d["cfd"] = cfd
    return json.dumps(d, sort_keys=True, default=str)


def find_grid_studies(simulations: Mapping[str, Any], results: Mapping[str, Any], *,
                      min_grids: int = 3) -> list[list[str]]:
    """Groups of simulation ids that form grid studies: same design and set-up apart from the
    resolution settings (:data:`RESOLUTION_SETTINGS`), at ``min_grids`` or more different cells per
    reference length, each with a CFD result. Ids fine to coarse; one run per resolution (the
    latest in project order when several share one)."""
    groups: dict[tuple[str, str], dict[float, str]] = {}
    for sid, sim in simulations.items():
        r = results.get(sid)
        if r is None:
            continue
        key = _setup_key(sim)
        cpr = _f(getattr(r, "cells_per_ref", _NAN))
        if key is None or not (_finite(cpr) and cpr > 0):
            continue
        groups.setdefault((str(getattr(sim, "design_id", "")), key), {})[round(cpr, 6)] = sid
    out = []
    for by_res in groups.values():
        if len(by_res) >= min_grids:
            out.append([by_res[c] for c in sorted(by_res, reverse=True)])
    return out


__all__ = [
    "CONVERGENCE_TYPES", "DOMAIN_RTOL", "FORMAL_ORDER", "FS_THREE_GRID", "FS_TWO_GRID", "MIN_REFINEMENT_RATIO",
    "NOISE_SIGMAS", "ORDER_TOLERANCE", "RESOLUTION_SETTINGS", "STUDY_KEYS_2D", "STUDY_KEYS_3D", "GCIResult",
    "GridLevel", "GridStudy", "convergence_type", "find_grid_studies", "fit_order", "gci_study", "gci_two_grid",
    "grid_study", "observed_order", "richardson",
]
