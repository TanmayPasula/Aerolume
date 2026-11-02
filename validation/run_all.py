"""Every validation case of Aerolume in one command, compared with cited references.

Usage (from the repository root)::

    python validation/run_all.py --quick                 # minutes: verification + coarse cases
    python validation/run_all.py --all                   # every case except the stretch ones
    python validation/run_all.py --list                  # the cases, their references and cost
    python validation/run_all.py --cases cavity2d,cyl40  # some cases (prefix match)
    python validation/run_all.py --area thermal --all    # one area
    python validation/run_all.py --all --write-docs      # also rewrite docs/validation.md section 9
    python validation/run_all.py --from-json a.json b.json --write-docs   # document earlier runs

``aerolume-cli validate --quick | --all`` is meant to call :func:`main` with the same flags; a
CLI that does not yet delegate here runs its own smaller subset.

Each case runs one or more simulations, extracts the quantities its reference
publishes and compares each with a tolerance band. A band is the published value
(or the spread of several published values) widened by a stated tolerance that
says what the method is expected to reach at that resolution, and why
(confinement, a diffuse immersed boundary, a coarse quick grid). Every check
records the reference it comes from (:data:`REFERENCES`), so the generated tables
cite their sources.

Cases are grouped by area: verification (exact answers), 2D and 3D flow, heat
transfer, fans and jets, airfoil polars and finite wings. Most run the solvers
directly through the case modules next to this file (``ns2d_cases.py``,
``ns3d_cases.py``, ``airfoil_validation.py``, ``wing_validation.py``), whose
functions the slow tests also assert on; the heat-transfer, fan and jet cases
come from ``thermal2d_cases.py``, whose docstring names the source every
reference number was read from. A case whose solver feature is missing from this installation is reported as
*unavailable* with the reason, never as passed. Other modules may add cases:
any ``validation/*_cases.py`` that defines ``RUN_ALL_CASES`` (a list of
:class:`Case`) is picked up (see validation/README.md).

Outputs: a JSON record of every check (``--out``, default ``validation/out``),
the Markdown tables printed to the console, and with ``--write-docs`` the
generated part of docs/validation.md (between the ``run_all`` markers) and its
figures in docs/img. The exit code is 0 when every available check passed, 1
otherwise (``--strict`` also fails on unavailable cases), 2 for bad arguments.

GPU timings depend on what else uses the GPU; the JSON records the device and
the wall time of each case so a slow run can be recognised.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOCS = REPO / "docs"
IMG = DOCS / "img"
BEGIN = "<!-- run_all:begin -->"
END = "<!-- run_all:end -->"

AREAS = {"verification": "Verification (exact answers)", "flow2d": "2D flow", "flow3d": "3D flow",
         "thermal": "Heat transfer", "fans": "Fans and jets", "airfoil": "Airfoil polars", "wing": "Finite wings"}

#: Short keys -> references, cited by every check that uses them. Titles and venues as the
#: case modules and the theory documents give them (docs/theory_ns2d.md section 14,
#: docs/theory_ns3d.md section 13, docs/validation_wing.md, docs/validation_airfoil.md and the
#: docstring of thermal2d_cases.py, which names the source each thermal value was read from).
REFERENCES: dict[str, str] = {
    "exact": "Exact solution or exact discrete identity (Taylor–Green vortex, projection, conservation).",
    "dennis-chang": ("Dennis & Chang (1970), Numerical solutions for steady flow past a circular cylinder at "
                     "Reynolds numbers up to 100, J. Fluid Mech. 42."),
    "coutanceau": ("Coutanceau & Bouard (1977), Experimental determination of the main features of the viscous "
                   "flow in the wake of a circular cylinder in uniform translation, J. Fluid Mech. 79."),
    "williamson": ("Williamson (1989), Oblique and parallel modes of vortex shedding in the wake of a circular "
                   "cylinder at low Reynolds numbers, J. Fluid Mech. 206."),
    "liu98": ("Liu, Zheng & Sung (1998), Preconditioned multigrid methods for unsteady incompressible flows, "
              "J. Comput. Phys. 139."),
    "schafer-turek": ("Schäfer & Turek (1996), Benchmark computations of laminar flow around a cylinder, Notes on "
                      "Numerical Fluid Mechanics 52."),
    "sohankar": ("Sohankar, Norberg & Davidson (1998), square cylinder at Re 100, as tabulated in "
                 "validation/ns2d_cases.py (with Darekar & Sherwin 2001 and Sharma & Eswaran 2004)."),
    "clift": "Clift, Grace & Weber (1978), Bubbles, Drops and Particles, Academic Press.",
    "johnson-patel": "Johnson & Patel (1999), Flow past a sphere up to a Reynolds number of 300, J. Fluid Mech. 378.",
    "ladson": ("Ladson (1988), Effects of independent variation of Mach and Reynolds numbers on the low-speed "
               "aerodynamic characteristics of the NACA 0012 airfoil section, NASA TM-4074."),
    "xfoil": "XFOIL 6.99 (M. Drela, MIT) on identical geometry, the reference table of validation/airfoil_validation.py.",
    "llt": ("Prandtl's lifting-line theory, elliptic wing: CLα = 2πAR/(AR + 2), e = 1 (e.g. Anderson, Fundamentals "
            "of Aerodynamics, ch. 5)."),
    "falkner": ("Falkner (1948/1952), Calculated loadings due to incidence of a number of straight and swept-back "
                "wings, ARC R&M 2596."),
    "multhopp": ("Multhopp (1950), Methods for calculating the lift distribution of wings (subsonic lifting-surface "
                 "theory), ARC R&M 2884."),
    "weber": "Weber & Brebner, Low-speed tests on 45° swept-back wings, ARC R&M 2882.",
    "devahl": ("de Vahl Davis (1983), Natural convection of air in a square cavity: a bench mark numerical "
               "solution, Int. J. Numer. Meth. Fluids 3, 249–264, Table V."),
    "cylinder-nu": ("Mean Nusselt number of an isothermal cylinder, Pr 0.71: Biswas & Sarkar (2009), Jafroudi & Yang "
                    "(1986), Apelt & Ledwich (1979), as tabulated by Singhal, Dutta & Kalita (2023), Int. J. Thermal "
                    "Sciences 194, 108588, Table 2."),
    "chandrasekhar": ("Chandrasekhar (1961), Hydrodynamic and Hydromagnetic Stability, Oxford: onset of convection "
                      "between rigid isothermal plates at Ra_c = 1707.76 (k_c = 3.117)."),
    "garrett-cummins": ("Garrett & Cummins (2007), The efficiency of a turbine in a tidal channel, J. Fluid Mech. 588: "
                        "actuator-disc momentum theory in a confined channel (Froude's theory for blockage → 0)."),
    "bickley": "Bickley (1937), the plane laminar jet similarity solution: u_c = 0.4543 (K²/(ν x))^(1/3).",
    "schlichting": ("Schlichting (1933), Laminare Strahlausbreitung, ZAMM 13: the laminar round jet similarity "
                    "solution u = u_c (1 + ξ²/4)⁻², 1/u_c = 8πν x / (3K) (Schlichting & Gersten, Boundary-Layer "
                    "Theory, 9th ed., §7.2.6); Landau–Squire's exact solution differs by 1.1 % at this K."),
    "cavity3d": ("Differentially heated cubic cavity, mean hot-wall Nusselt number: Tric, Labrosse & Betrouni (2000), "
                 "Int. J. Heat Mass Transfer 43; Wakashima & Saitoh (2004), Int. J. Heat Mass Transfer 47; Fusegi, "
                 "Hyun, Kuwahara & Farouk (1991), Int. J. Heat Mass Transfer 34; as tabulated in "
                 "validation/thermal3d_cases.py."),
    "cavity3d-plane": ("Symmetry plane y = ½ of the differentially heated cubic cavity, free-fall units: hot-wall "
                       "Nusselt number of Tric, Labrosse & Betrouni (2000) and Fusegi et al. (1991); centre-line "
                       "velocity maxima of Wakashima & Saitoh (2004) and Fusegi et al. (1991); as tabulated in "
                       "validation/thermal3d_cases.py from Li, Yang & Zhang (arXiv:1511.04633), Al-Rashed et al. "
                       "(2017) and Hajabdollahi & Premnath (2017, arXiv:1710.04751)."),
    "sphere-nu": ("Mean Nusselt number of an isothermal sphere: Ranz & Marshall (1952), Chem. Eng. Prog. 48; Whitaker "
                  "(1972), AIChE J. 18; Clift, Grace & Weber (1978), Bubbles, Drops and Particles."),
    "churchill-sphere": ("Churchill (1983), natural convection from an isothermal sphere in a quiescent fluid: "
                         "Nu = 2 + 0.589 Ra^¼ / [1 + (0.469/Pr)^(9/16)]^(4/9) (laminar, Pr ≥ 0.7), as reproduced in "
                         "M. M. Yovanovich, University of Waterloo ME 353 notes, Natural Convection from Isothermal "
                         "Spheres."),
    "celik": ("Celik, Ghia, Roache, Freitas, Coleman & Raad (2008), Procedure for estimation and reporting of "
              "uncertainty due to discretization in CFD applications, J. Fluids Eng. 130, 078001."),
}


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


@dataclass
class Check:
    """One compared quantity: Aerolume's ``value`` against the band [``lo``, ``hi``].

    ``ref`` is the published value (or None for a range), ``ref_text`` how the
    reference is quoted in the tables, ``tolerance`` the words for the band, and
    ``source`` a key of :data:`REFERENCES`.
    """
    quantity: str
    value: float
    lo: float
    hi: float
    source: str
    ref: float | None = None
    ref_text: str = ""
    tolerance: str = ""
    unit: str = ""
    note: str = ""

    @property
    def passed(self) -> bool:
        # a plain bool: NumPy band edges would make the comparison a numpy.bool_, which JSON refuses
        return bool(_finite(self.value) and float(self.lo) <= float(self.value) <= float(self.hi))

    @property
    def deviation(self) -> float:
        """Relative deviation from the reference value (or from the nearest edge of the
        reference range when there is no single value), nan when undefined."""
        if not _finite(self.value):
            return float("nan")
        if self.ref is not None and self.ref != 0:
            return (float(self.value) - self.ref) / abs(self.ref)
        return float("nan")

    def to_dict(self) -> dict:
        d = _jsonable(asdict(self))
        d["passed"] = self.passed
        d["deviation"] = float(self.deviation) if _finite(self.deviation) else None
        d["value"] = float(self.value) if _finite(self.value) else None
        return d


def band(ref: float, rel: float) -> tuple[float, float]:
    """(lo, hi) of ``ref`` ± ``rel`` relative."""
    a, b = ref * (1 - rel), ref * (1 + rel)
    return (min(a, b), max(a, b))


def widen(lo: float, hi: float, rel_lo: float, rel_hi: float | None = None) -> tuple[float, float]:
    """A published range [lo, hi] widened by ``rel_lo`` below and ``rel_hi`` (default the same) above."""
    rel_hi = rel_lo if rel_hi is None else rel_hi
    return lo - abs(lo) * rel_lo, hi + abs(hi) * rel_hi


@dataclass
class CaseResult:
    """The outcome of one case: status "pass" | "fail" | "unavailable" | "error"."""
    key: str
    title: str
    area: str
    status: str
    checks: list[Check] = field(default_factory=list)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "area": self.area, "status": self.status,
                "seconds": round(self.seconds, 2), "notes": list(self.notes), "error": self.error,
                "checks": [c.to_dict() for c in self.checks], "data": _jsonable(self.data)}


class Unavailable(Exception):
    """A case needs a solver feature this installation does not have (reported, not failed)."""


@dataclass
class Context:
    """What a case function receives: the device, the mode, and the case modules."""
    device: str = "auto"
    quick: bool = True
    log: Callable[[str], None] = print
    _modules: dict[str, Any] = field(default_factory=dict)

    def module(self, name: str) -> Any:
        """A case module next to this file (``ns2d_cases``), imported once (Unavailable when missing).

        The validation folder is put on ``sys.path`` first: some case modules import their
        siblings by plain name (``thermal3d_cases`` builds on ``ns3d_cases``), as they do
        when the tests and the bench scripts run them.
        """
        if name not in self._modules:
            path = HERE / f"{name}.py"
            if not path.is_file():
                raise Unavailable(f"validation/{name}.py is missing")
            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            # a module that exists but fails to import is a broken installation: an error, not
            # an unavailable feature
            spec = importlib.util.spec_from_file_location(f"aerolume_validation_{name}", path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            sys.modules[spec.name] = mod                  # type: ignore[union-attr]
            spec.loader.exec_module(mod)                  # type: ignore[union-attr]
            self._modules[name] = mod
        return self._modules[name]


@dataclass
class Case:
    """A validation case: ``run(ctx)`` returns its checks, or (checks, data) with figure data."""
    key: str
    title: str
    area: str
    run: Callable[[Context], Any]
    sources: tuple[str, ...]
    quick: bool = False                # part of --quick
    est_seconds: float = 30.0          # on the development GPU (quick variant when quick)
    gpu: bool = True                   # benefits from a GPU (runs on the CPU, slower)
    stretch: bool = False              # only with --stretch (long runs)


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        return _jsonable(v.tolist())
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def case_tg2d(ctx: Context) -> list[Check]:
    m = ctx.module("ns2d_cases")
    e16, _ = m.taylor_green_error(16)
    e32, rate = m.taylor_green_error(32)
    order = math.log2(e16 / e32)
    return [Check("Taylor–Green 2D spatial order (16 → 32 cells)", order, 1.8, 2.2, "exact", ref=2.0,
                  ref_text="2 (central scheme)", tolerance="±0.2"),
            Check("Energy decay rate / exact (32 cells)", rate, 0.98, 1.02, "exact", ref=1.0, ref_text="1",
                  tolerance="±2 %")]


def case_tg3d(ctx: Context) -> list[Check]:
    m = ctx.module("ns3d_cases")
    e8, _ = m.taylor_green_error(8)
    e16, rate = m.taylor_green_error(16)
    order = math.log2(e8 / e16)
    return [Check("Taylor–Green 3D spatial order (8 → 16 cells)", order, 1.8, 2.2, "exact", ref=2.0,
                  ref_text="2 (central scheme)", tolerance="±0.2"),
            Check("Energy decay rate / exact (16 cells)", rate, 0.97, 1.03, "exact", ref=1.0, ref_text="1",
                  tolerance="±3 %")]


def case_divergence(ctx: Context) -> list[Check]:
    m = ctx.module("ns2d_cases")
    import torch

    from aerolume.solvers.ns2d import FlowSolver
    cfg = m.cylinder_config(100, 16, upstream=4, downstream=8, half_height=4, device=ctx.device, dtype="float32")
    s = FlowSolver(cfg)
    worst = 0.0
    for _ in range(10):
        s.advance(5)
        worst = max(worst, float(s.field("divergence").abs().max()) * cfg.h)
    dev = s.device.type
    del s
    if dev == "cuda":
        torch.cuda.empty_cache()
    return [Check(f"Projected divergence max |∇·u| h/U, float32 ({dev})", worst, 0.0, 1e-5, "exact", ref=0.0,
                  ref_text="0 (round-off)", tolerance="< 10⁻⁵")]


# ---------------------------------------------------------------------------
# 2D flow
# ---------------------------------------------------------------------------

def case_cyl40(ctx: Context) -> tuple[list[Check], dict]:
    m = ctx.module("ns2d_cases")
    cpd, t_end, t_avg = (16, 40.0, 30.0) if ctx.quick else (32, 60.0, 50.0)
    r = m.run_cylinder(40.0, cpd, t_end, t_avg, device=ctx.device)
    note = ("5 % blockage (slip walls 10 D from the axis) raises Cd about 4 % above the unconfined references "
            "(docs/validation.md section 2)")
    return ([Check(f"Cylinder Re 40, Cd ({cpd} cells/D)", r.values["cd"], 1.55, 1.70, "dennis-chang",
                   ref_text="1.50–1.55 unconfined", tolerance="1.55–1.70 with the domain's blockage", note=note),
             Check(f"Cylinder Re 40, wake length L/D ({cpd} cells/D)", r.values["L/D"], 2.0, 2.5, "coutanceau",
                   ref_text="2.13–2.35", tolerance="2.0–2.5")],
            {"grid": r.grid, "steps": r.steps, "wall_s": r.wall_s})


def case_cyl100(ctx: Context) -> tuple[list[Check], dict]:
    m = ctx.module("ns2d_cases")
    cpd = 24
    r = m.run_cylinder(100.0, cpd, 120.0, 80.0, device=ctx.device)
    v = r.values
    return ([Check(f"Cylinder Re 100, mean Cd ({cpd} cells/D)", v["cd"], 1.33, 1.47, "liu98",
                   ref_text="1.33–1.38 unconfined", tolerance="1.33–1.47 (blockage)"),
             Check(f"Cylinder Re 100, lift amplitude Cl′ ({cpd} cells/D)", v["cl_amp"], 0.30, 0.40, "liu98",
                   ref_text="0.32–0.34", tolerance="0.30–0.40"),
             Check(f"Cylinder Re 100, Strouhal number ({cpd} cells/D)", v["st"], 0.160, 0.175, "williamson",
                   ref=0.164, ref_text="0.164–0.167", tolerance="0.160–0.175")],
            {"grid": r.grid, "steps": r.steps, "wall_s": r.wall_s})


def case_schafer_turek(ctx: Context) -> list[Check]:
    m = ctx.module("ns2d_cases")
    ref = m.SCHAFER_TUREK_REF
    out = []
    r1 = m.run_schafer_turek("2D-1", 40, device=ctx.device)
    for key, label, tol in (("cd", "Cd", 0.01), ("cl", "Cl", 0.05), ("La", "recirculation length La [m]", 0.02),
                            ("dp", "pressure difference Δp [Pa]", 0.02)):
        lo, hi = ref["2D-1"][key]
        a, b = widen(lo, hi, tol)
        out.append(Check(f"Schäfer–Turek 2D-1, {label} (40 cells/D)", r1.values[key], a, b, "schafer-turek",
                         ref=0.5 * (lo + hi), ref_text=f"{lo:g}–{hi:g}", tolerance=f"band ± {100 * tol:g} %"))
    r2 = m.run_schafer_turek("2D-2", 40, device=ctx.device)
    for key, label, tol in (("st", "St", 0.01), ("cd_max", "Cd,max", 0.03), ("cl_max", "Cl,max", 0.03)):
        lo, hi = ref["2D-2"][key]
        a, b = widen(lo, hi, tol)
        out.append(Check(f"Schäfer–Turek 2D-2, {label} (40 cells/D)", r2.values[key], a, b, "schafer-turek",
                         ref=0.5 * (lo + hi), ref_text=f"{lo:g}–{hi:g}", tolerance=f"band ± {100 * tol:g} %"))
    return out


#: Resolution allowance of the square-cylinder drag band: the published range ± 5 %, the widest
#: allowance the other 2D bands already used (Schäfer–Turek 2D-1 lift at 40 cells/D). See
#: :func:`case_square` for why this replaced an earlier, wider band.
SQUARE_CD_ALLOWANCE = 0.05
#: Cells per side of the square runs: the documented Standard-quality grid, then the finer one that
#: should pass the drag band (see :func:`case_square`)
SQUARE_GRIDS = (32, 48)


def case_square(ctx: Context) -> tuple[list[Check], dict]:
    """Square cylinder at Re 100 on 32 and 48 cells per side, against Sohankar et al.'s range.

    Band history, stated because it matters: in early development the drag band was the published range
    +8 % above (1.477 … 1.614). That band was written after the +5.9 % deviation at 32 cells/D
    was already documented (docs/validation.md section 3), so it contained a known result and
    was not a test. Before the first release the drag moved to the rule of the other 2D bands: the published
    range widened by a stated resolution allowance, here ± 5 % (:data:`SQUARE_CD_ALLOWANCE`,
    the widest the other 2D bands use; nothing was tuned to this case). The 32 cells/D check
    is expected to fail (+5.9 % in section 3) and stays in the table as a failure; 48 cells/D is
    the grid that should pass (+3.7 % in section 3). Both are measured on every ``--all`` run.
    The Strouhal number keeps its ± 2 % band at both grids.
    """
    m = ctx.module("ns2d_cases")
    lo, hi = m.SQUARE_RE100_REF["cd"]
    slo, shi = m.SQUARE_RE100_REF["st"]
    a, b = widen(lo, hi, SQUARE_CD_ALLOWANCE)
    checks: list[Check] = []
    data: dict[str, Any] = {}
    for cpd in SQUARE_GRIDS:
        r = m.run_square(100.0, cpd, device=ctx.device)
        note = ("one-cell diffuse interface at sharp corners: drag converges from above (+5.9 % at 32 and +3.7 % "
                "at 48 cells/D, docs/validation.md section 3); this grid is expected to fail the band"
                if cpd == SQUARE_GRIDS[0] else "")
        checks.append(Check(f"Square cylinder Re 100, mean Cd ({cpd} cells/D)", r.values["cd"], a, b, "sohankar",
                            ref=0.5 * (lo + hi), ref_text=f"{lo:g}–{hi:g}",
                            tolerance=f"range ± {100 * SQUARE_CD_ALLOWANCE:g} %", note=note))
        checks.append(Check(f"Square cylinder Re 100, Strouhal number ({cpd} cells/D)", r.values["st"], slo * 0.98,
                            shi * 1.02, "sohankar", ref=0.5 * (slo + shi), ref_text=f"{slo:g}–{shi:g}",
                            tolerance="± 2 %"))
        data[str(cpd)] = {"grid": getattr(r, "grid", None), "wall_s": getattr(r, "wall_s", None),
                          "cd": r.values["cd"], "st": r.values["st"]}
    return checks, data


# ---------------------------------------------------------------------------
# 3D flow
# ---------------------------------------------------------------------------

def case_sphere100(ctx: Context) -> tuple[list[Check], dict]:
    m = ctx.module("ns3d_cases")
    cpd = 16 if ctx.quick else 24
    r = m.run_sphere(100.0, cpd, device=ctx.device)
    ref = m.clift_cd(100.0)
    hi = 1.20 if cpd < 24 else 1.15
    checks = [Check(f"Sphere Re 100, Cd ({cpd} cells/D, 2.85 % blockage)", r.values["cd"], 1.05, hi, "clift",
                    ref=ref, ref_text=f"{ref:.3f} unconfined", tolerance=f"1.05–{hi:g} (confinement +3.4 %"
                    + (", coarse grid)" if cpd < 24 else ")"))]
    lw = r.values.get("L/D")
    if _finite(lw):
        checks.append(Check(f"Sphere Re 100, wake length L/D ({cpd} cells/D)", lw, 0.80, 0.95, "johnson-patel",
                            ref=0.88, ref_text="≈ 0.88", tolerance="0.80–0.95"))
    return checks, {"grid": r.grid, "wall_s": r.wall_s, "steps": r.steps}


def case_sphere_symmetry(ctx: Context) -> list[Check]:
    m = ctx.module("ns3d_cases")
    half = m.run_sphere(100.0, 16, device=ctx.device)
    full = m.run_sphere(100.0, 16, device=ctx.device, half=False)
    diff = abs(half.values["cd"] - full.values["cd"]) / abs(full.values["cd"])
    return [Check("Sphere Re 100: half domain vs full domain, |ΔCd|/Cd", diff, 0.0, 0.01, "exact", ref=0.0,
                  ref_text="0 (same discrete problem)", tolerance="< 1 %")]


def case_sphere300(ctx: Context) -> list[Check]:
    m = ctx.module("ns3d_cases")
    r = m.run_sphere_unsteady(300.0, 24, device=ctx.device)
    v = r.values
    return [Check("Sphere Re 300, mean Cd (24 cells/D)", v["cd"], 0.656 * 0.95, 0.656 * 1.10, "johnson-patel",
                  ref=0.656, ref_text="0.656", tolerance="−5 % / +10 % (confinement)"),
            Check("Sphere Re 300, Strouhal number (24 cells/D)", v.get("st", float("nan")), 0.137 * 0.95,
                  0.137 * 1.05, "johnson-patel", ref=0.137, ref_text="0.137", tolerance="± 5 %")]


# ---------------------------------------------------------------------------
# airfoil and wing
# ---------------------------------------------------------------------------

def case_ladson(ctx: Context) -> list[Check]:
    """Ladson's tripped NACA 0012 at Re 6 × 10⁶, M 0.15, in the attached range (0–8.5°). The
    polar is computed at the measured angles themselves (from 0° for the warm start), so no
    interpolation enters the comparison."""
    af = ctx.module("airfoil_validation")
    refs = [(float(a), float(cl), float(cd)) for a, cl, cd in af.LADSON_0012_RE6E6 if 0.0 <= a <= 8.5]
    c = af.run("NACA 0012 Ladson", "0012", 6e6, [0.0] + [a for a, _, _ in refs], tripped=True, mach=0.15)
    out = []
    for a_ref, cl_ref, cd_ref in refs:
        cl = float(c.at("cl", a_ref))
        out.append(Check(f"NACA 0012 Re 6×10⁶ tripped, cl at {a_ref:g}°", cl, cl_ref - 0.05, cl_ref + 0.05,
                         "ladson", ref=cl_ref, ref_text=f"{cl_ref:.3f}", tolerance="± 0.05"))
        if abs(a_ref - 4.04) < 0.2:
            cd = float(c.at("cd", a_ref))
            out.append(Check(f"NACA 0012 Re 6×10⁶ tripped, cd at {a_ref:g}°", cd, cd_ref * 0.85, cd_ref * 1.15,
                             "ladson", ref=cd_ref, ref_text=f"{cd_ref:.5f}", tolerance="± 15 %"))
    return out


def case_xfoil(ctx: Context) -> list[Check]:
    import numpy as np
    af = ctx.module("airfoil_validation")
    out = []
    for code, re_ in af.XFOIL_CASES:
        c = af.case_xfoil(code, re_)
        rows = af.compare_xfoil(c)
        if not len(rows):
            raise RuntimeError(f"no common angles with XFOIL for NACA {code} Re {re_:g}")
        dcl = float(np.max(np.abs(rows[:, 2] - rows[:, 1])))
        dcd = float(np.max(np.abs(rows[:, 4] / rows[:, 3] - 1.0)))
        dcm = float(np.max(np.abs(rows[:, 6] - rows[:, 5])))
        tag = f"NACA {code} Re {re_ / 1e6:g}×10⁶ vs XFOIL"
        out += [Check(f"{tag}: max |Δcl|", dcl, 0.0, 0.02, "xfoil", ref=0.0, ref_text="0", tolerance="< 0.02"),
                Check(f"{tag}: max |Δcd|/cd", dcd, 0.0, 0.05, "xfoil", ref=0.0, ref_text="0", tolerance="< 5 %"),
                Check(f"{tag}: max |Δcm|", dcm, 0.0, 0.005, "xfoil", ref=0.0, ref_text="0", tolerance="< 0.005")]
    return out


def case_wing_theory(ctx: Context) -> list[Check]:
    w = ctx.module("wing_validation")
    tol = w.TOL
    out = []
    for row in w.case_elliptic():
        f = row["formula"]
        out.append(Check(f"Elliptic wing AR {row['ar']:g}, lifting line CLα [1/rad]", row["llt"],
                         *band(f, tol["elliptic_llt_cla"]), "llt", ref=f, ref_text=f"{f:.4f}",
                         tolerance=f"± {100 * tol['elliptic_llt_cla']:g} %"))
        out.append(Check(f"Elliptic wing AR {row['ar']:g}, span efficiency e (VLM)", row["e_vlm"],
                         tol["elliptic_e"], 1.02, "llt", ref=1.0, ref_text="1", tolerance=f"≥ {tol['elliptic_e']}"))
    ls = w.case_lifting_surface()
    out.append(Check("Circular wing, VLM CLα [1/rad]", ls["circle"], *band(w.MULTHOPP_CIRCLE, tol["lifting_surface"]),
                     "multhopp", ref=w.MULTHOPP_CIRCLE, ref_text=f"{w.MULTHOPP_CIRCLE}",
                     tolerance=f"± {100 * tol['lifting_surface']:g} %"))
    for ar, refs in w.FALKNER_RECT.items():
        t = tol["lifting_surface"] if ar >= 4 else tol["lifting_surface_low_ar"]
        lo, hi = widen(min(refs), max(refs), t)
        out.append(Check(f"Rectangular wing AR {ar}, VLM CLα [1/rad]", ls[f"rect{ar}"], lo, hi, "falkner",
                         ref=float(sum(refs) / len(refs)), ref_text="–".join(f"{x:g}" for x in sorted(set(refs))),
                         tolerance=f"± {100 * t:g} %"))
    for (ar, sw), refs in w.FALKNER_SWEPT.items():
        lo, hi = widen(min(refs), max(refs), tol["swept"])
        out.append(Check(f"Swept wing AR {ar}, {sw}°, VLM CLα [1/rad]", ls[f"swept{ar}_{sw}"], lo, hi, "falkner",
                         ref=float(sum(refs) / len(refs)), ref_text=f"{min(refs):g}–{max(refs):g}",
                         tolerance=f"range ± {100 * tol['swept']:g} %"))
    return out


def case_weber(ctx: Context) -> list[Check]:
    w = ctx.module("wing_validation")
    r = w.case_weber()
    t = w.TOL["weber_cl"]
    return [Check(f"Weber & Brebner 45° wing, CL at {a:g}°", float(cl), *band(float(ref), t), "weber",
                  ref=float(ref), ref_text=f"{ref:.3f}", tolerance=f"± {100 * t:g} %")
            for a, cl, ref in zip(w.WEBER_ALPHA, r["cl"], w.WEBER_CL, strict=True)]


# ---------------------------------------------------------------------------
# heat transfer, fans and jets (validation/thermal2d_cases.py)
# ---------------------------------------------------------------------------
#
# The case functions and their reference data belong to thermal2d_cases.py, whose
# docstring names the source every reference number was read from; the runner only
# sets the resolutions and the tolerance bands. Tolerances are those a second-order
# scheme on these grids is expected to meet, not fitted to the results.

#: (Ra, cells) of the square-cavity runs: quick, all
CAVITY_RUNS = {True: ((1e3, 32), (1e4, 64)), False: ((1e3, 64), (1e4, 64), (1e5, 128), (1e6, 256))}


def _pow10(x: float) -> str:
    """A power of ten as it is printed in the tables ("10⁴"); anything else as "%g"."""
    e = round(math.log10(x)) if x > 0 else 0
    if x > 0 and abs(x - 10.0 ** e) <= 1e-9 * x:
        return "10" + str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{x:g}"


def _cavity_tolerance(ra: float) -> float:
    """Nusselt / velocity band: 2 % to Ra 10⁴, 3 % at 10⁵, 5 % at 10⁶ (thinner boundary layers)."""
    return 0.02 if ra <= 1e4 else 0.03 if ra <= 1e5 else 0.05


def case_cavity2d(ctx: Context) -> tuple[list[Check], dict]:
    m = ctx.module("thermal2d_cases")
    checks: list[Check] = []
    data = []
    for ra, n in CAVITY_RUNS[ctx.quick]:
        r = m.run_cavity(ra, n, device=ctx.device)
        v, ref = r.values, m.DE_VAHL_DAVIS[ra]
        tol = _cavity_tolerance(ra)
        tag = f"Cavity Ra {_pow10(ra)} ({n}²)"
        checks += [
            Check(f"{tag}: mean Nu, hot wall", v["nu_hot"], *band(ref["nu_0"], tol), "devahl", ref=ref["nu_0"],
                  ref_text=f"{ref['nu_0']:g}", tolerance=f"± {100 * tol:g} %"),
            Check(f"{tag}: Nu on the vertical mid-plane", v["nu_half"], *band(ref["nu_half"], tol), "devahl",
                  ref=ref["nu_half"], ref_text=f"{ref['nu_half']:g}", tolerance=f"± {100 * tol:g} %"),
            Check(f"{tag}: u_max on x = ½ [κ/L]", v["u_max"], *band(ref["u_max"], tol + 0.01), "devahl",
                  ref=ref["u_max"], ref_text=f"{ref['u_max']:g}", tolerance=f"± {100 * (tol + 0.01):g} %"),
            Check(f"{tag}: v_max on y = ½ [κ/L]", v["v_max"], *band(ref["v_max"], tol + 0.01), "devahl",
                  ref=ref["v_max"], ref_text=f"{ref['v_max']:g}", tolerance=f"± {100 * (tol + 0.01):g} %"),
            Check(f"{tag}: height of u_max [L]", v["u_max_y"], ref["u_max_y"] - 1.0 / n, ref["u_max_y"] + 1.0 / n,
                  "devahl", ref=ref["u_max_y"], ref_text=f"{ref['u_max_y']:g}", tolerance="± 1 cell"),
            Check(f"{tag}: energy balance (Nu_hot − Nu_cold)/Nu_hot", abs(v["energy_balance"]), 0.0, 0.01, "exact",
                  ref=0.0, ref_text="0 (steady state)", tolerance="< 1 %"),
        ]
        data.append({"ra": ra, "n": n, "nu": v["nu_hot"], "nu_ref": ref["nu_0"], "wall_s": r.wall_s,
                     "steps": r.steps, "t_end": v.get("t_end")})
    return checks, {"cavity": data}


def case_heated_cylinder(ctx: Context) -> tuple[list[Check], dict]:
    m = ctx.module("thermal2d_cases")
    runs = ((20.0, 24),) if ctx.quick else ((20.0, 32), (40.0, 32))
    checks: list[Check] = []
    data = []
    for re, cpd in runs:
        r = m.run_heated_cylinder(re, cpd, device=ctx.device)
        refs = m.CYLINDER_NU_REF[re]
        lo, hi = min(refs.values()), max(refs.values())
        a, b = widen(lo, hi, 0.04)
        checks.append(Check(f"Heated cylinder Re {re:g}, Pr 0.71: mean Nu ({cpd} cells/D)", r.values["nu"], a, b,
                            "cylinder-nu", ref=0.5 * (lo + hi), ref_text=f"{lo:g}–{hi:g}",
                            tolerance="published range ± 4 % (5 % blockage)"))
        data.append({"re": re, "nu": r.values["nu"], "wall_s": r.wall_s})
    return checks, {"cylinder": data}


def case_rayleigh_benard(ctx: Context) -> tuple[list[Check], dict]:
    m = ctx.module("thermal2d_cases")
    out = m.rb_onset(1500.0, 2000.0, 24, device=ctx.device)
    return ([Check("Rayleigh–Bénard onset Ra_c (growth rates at Ra 1500 and 2000)", out["ra_c"],
                   *band(m.RB_RA_C, 0.03), "chandrasekhar", ref=m.RB_RA_C, ref_text=f"{m.RB_RA_C:g}",
                   tolerance="± 3 %")], {"rb": out})


def confined_disc_theory(ct: float, blockage: float) -> dict[str, float]:
    """Quasi-inviscid actuator disc in a channel, for a fan that pushes the flow (``ct`` > 0)
    or one that holds it back (``ct`` < 0, a turbine).

    The streamtube model of Garrett & Cummins (2007) (continuity, Bernoulli in the core and
    the bypass, the channel momentum balance) written with the *signed* pressure jump
    Δp = p(after) − p(before) = C_T / 2 in units of ρU², C_T = T / (½ ρ U² A) along the
    stream; ``thermal2d_cases.channel_disc_theory`` is the turbine branch (Δp < 0) of the
    same equations. Unknowns (U = 1): bypass velocity u_b, wake core velocity u_w, wake core
    area fraction s_w; disc velocity u_t = s_w u_w / B. For B → 0 it gives Froude's
    C_T = 4 a (1 + a) for a propeller (u_t = 1 + a) and 4 a (1 − a) for a turbine
    (u_t = 1 − a). ``blockage`` B = disc area / channel cross-section (2D: width / height).
    """
    from scipy.optimize import brentq
    if not (0.0 < blockage < 1.0) or ct == 0.0:
        raise ValueError("blockage must be in (0, 1) and the thrust coefficient non-zero")
    dpp, B = 0.5 * float(ct), float(blockage)

    def parts(ub: float) -> tuple[float, float]:
        uw = math.sqrt(ub * ub + 2.0 * dpp)
        return uw, (1.0 - ub) / (uw - ub)

    def resid(ub: float) -> float:
        uw, sw = parts(ub)
        return (0.5 * (ub * ub - 1.0) + dpp * B) - (sw * uw * uw + (1.0 - sw) * ub * ub - 1.0)

    # bracket u_b to the physical states 0 < s_w < 1: the wake core is faster than the
    # free stream for a propeller (u_w > 1, so u_b > sqrt(1 - 2 dp)) and slower for a turbine
    # (u_w < 1, so u_b < sqrt(1 + 2 |dp|)); the bypass is the other way round (u_b < 1 or > 1)
    eps = 1e-12
    if dpp > 0:
        lo, hi = math.sqrt(max(1.0 - 2.0 * dpp, 0.0)) + eps, 1.0 - eps
    else:
        lo, hi = max(1.0, math.sqrt(-2.0 * dpp)) + eps, math.sqrt(1.0 - 2.0 * dpp) - eps
    ub = brentq(resid, lo, hi, xtol=1e-15)
    uw, sw = parts(ub)
    ut = sw * uw / B
    return {"u_t": ut, "a": abs(ut - 1.0), "u_w": uw, "u_b": ub, "s_w": sw}


def froude_disc(ct: float) -> dict[str, float]:
    """Unconfined Froude momentum theory: induction a, disc and far-wake velocities (U = 1).
    ``ct`` > 0 propeller (C_T = 4a(1 + a)), < 0 turbine (|C_T| = 4a(1 − a), light-loading root)."""
    if ct > 0:
        a = 0.5 * (math.sqrt(1.0 + ct) - 1.0)
        return {"a": a, "u_disc": 1.0 + a, "u_wake": 1.0 + 2.0 * a}
    a = 0.5 * (1.0 - math.sqrt(1.0 + ct))
    return {"a": a, "u_disc": 1.0 - a, "u_wake": 1.0 - 2.0 * a}


def case_actuator2d(ctx: Context) -> tuple[list[Check], dict]:
    """An actuator strip in a slip-walled channel (10 % blockage), C_T 0.5, both ways round.

    ``propeller=True`` blows with the stream (``FanSpec`` direction +x) and accelerates the
    flow through the strip; ``propeller=False`` blows against it, a momentum sink that slows
    the flow like a turbine. Each is compared with its branch of the confined momentum
    theory (:func:`confined_disc_theory`, signed C_T). History: the case module's default
    changed from the propeller to the turbine, and the first run of this version compared a
    turbine (0.878) with the propeller theory (1.103) and failed; the case now names the
    direction it runs and checks both.
    """
    m = ctx.module("thermal2d_cases")
    ct = 0.5
    cpd = 16 if ctx.quick else 32
    checks: list[Check] = []
    data: dict[str, Any] = {}
    for propeller in (True, False):
        r = m.run_actuator_disc(ct, cpd, propeller=propeller, device=ctx.device)
        v = r.values
        signed = ct if propeller else -ct
        th = confined_disc_theory(signed, v["blockage"])
        fr = froude_disc(signed)
        kind = "propeller (blowing with the stream)" if propeller else "turbine (blowing against it)"
        tag = f"Actuator strip C_T {ct:g}, {kind}"
        checks += [
            Check(f"{tag}, blockage {100 * v['blockage']:.0f} %: disc velocity / U ({cpd} cells/D)", v["u_disc"],
                  *band(th["u_t"], 0.03), "garrett-cummins", ref=th["u_t"],
                  ref_text=f"{th['u_t']:.4f} (confined momentum theory)", tolerance="± 3 %",
                  note=f"unconfined Froude value {fr['u_disc']:.4f}"),
            Check(f"{tag}: delivered thrust / set thrust", v["thrust"] / (0.5 * ct), 0.99, 1.01, "exact", ref=1.0,
                  ref_text="1", tolerance="± 1 %"),
        ]
        data["propeller" if propeller else "turbine"] = {
            "u_disc": v["u_disc"], "u_disc_theory": th["u_t"], "u_disc_froude": fr["u_disc"],
            "blockage": v["blockage"], "grid": list(r.grid), "steps": r.steps, "wall_s": r.wall_s}
    return checks, data


#: streamwise span (in port widths b) of the plane-jet measurements: the centreline fit and
#: the momentum-flux drift. Far enough from the port for the similarity profile, short
#: enough that the closed box's return flow has not yet drained the jet (the case module's
#: default, passed explicitly so the labels below always name the span that was measured).
JET_SPAN = (8.0, 24.0)


#: Band of the plane jet's raw momentum-flux drift: the 5 % written before the case first ran. Not
#: widened after the 6.2 % result (see :func:`case_plane_jet`).
JET_DRIFT_BAND = 0.05


def case_plane_jet(ctx: Context) -> tuple[list[Check], dict]:
    """A jet port (width b, Re_j 25) blowing into a closed box, against Bickley's (1937)
    similarity solution: the centreline decay u_c⁻³ ∝ x, the half-width growth, and the
    conservation of the momentum flux K = ∫u² dy that the similarity law rests on.

    K is integrated over the jet's forward-flow core (between the zero crossings of u):
    the box must return the entrained fluid as a weak counter-flow beside the jet, which
    belongs to the box, not to the jet. History: the first version integrated over the
    whole box height and failed (17.5 % drift over 12 ≤ x/b ≤ 40, docs/validation.md
    section 10 of that run); the case module now measures the core, averages longer and
    fits closer to the port, where the counter-flow has drained little momentum.

    The conservation statement is the core's **momentum budget** (``jet_measures``): the
    change of K over the span splits into the shear of the box's return flow on the core and
    the box's adverse pressure gradient, and what is left over is the scheme's error. That
    residual is held below 1 % of K. The raw drift of K keeps the band it had before its first
    run, < 5 % (:data:`JET_DRIFT_BAND`), and **fails**: 6.2 % measured, of which the budget
    attributes −3.6 % to the return flow's shear and −2.5 % to the box's pressure gradient
    (docs/theory_ns2d.md section 21.5). An earlier development build widened that band to 8 %
    after the 6.2 % was known; like the square cylinder's +8 % band that was a band moved to
    pass, so it was withdrawn before the release. The failure stays in the table until a jet in an open domain (or a taller
    box) is validated: momentum conservation of a free jet is not demonstrated by this case.
    """
    m = ctx.module("thermal2d_cases")
    lo_x, hi_x = JET_SPAN
    r = m.run_plane_jet(device=ctx.device, x_fit=JET_SPAN)
    v = r.values
    span = f"{lo_x:g} ≤ x/b ≤ {hi_x:g}"
    kp, k = v.get("Kp_mean"), v.get("K_mean")
    share = f" ({100 * kp / k:+.1f} % of K)" if _finite(kp) and _finite(k) and k else ""
    split = ""
    if _finite(v.get("K_budget_shear_pct")) and _finite(v.get("K_budget_pressure_pct")):
        split = (f"; of the change of K, {v['K_budget_shear_pct']:+.2f} % is the return flow's shear and "
                 f"{v['K_budget_pressure_pct']:+.2f} % the box's pressure gradient")
    note = ("K = ∫u² dy over the jet's forward-flow core; the closed box returns the entrained fluid as a weak "
            "counter-flow beside the jet, which slowly drains it. The pressure part of the momentum flux, "
            f"∫(p − p_edge) dy over the core, averages {_fmt(kp)}{share} over the same span and is not in K{split}")
    checks = [
        Check(f"Plane jet Re 25: centreline decay slope of u_c⁻³ / Bickley ({span})", v["slope"] / v["slope_bickley"],
              0.9, 1.1, "bickley", ref=1.0, ref_text="1 (similarity law)", tolerance="± 10 %"),
    ]
    if _finite(v.get("K_budget_residual_pct")):
        checks.append(Check(f"Plane jet Re 25: momentum budget of the jet core over {span}, residual / K",
                            abs(v["K_budget_residual_pct"]) / 100.0, 0.0, 0.01, "exact", ref=0.0,
                            ref_text="0 (steady x-momentum balance of the core)", tolerance="< 1 %", note=note))
        note = "the confinement's share of this drift is split in the momentum-budget check above"
    checks.append(Check(f"Plane jet Re 25: drift of the core momentum flux ∫u² dy over {span}",
                        abs(v["K_drift_pct"]) / 100.0, 0.0, JET_DRIFT_BAND, "bickley", ref=0.0,
                        ref_text="0 for a free jet (conserved by the similarity law)",
                        tolerance=f"< {100 * JET_DRIFT_BAND:g} % (set before the first run)",
                        note=note + "; a closed box drains K, so this check fails until a free jet is validated "
                        "(docs/validation.md section 10)"))
    if _finite(v.get("b_half_err_pct_x20")):
        checks.append(Check("Plane jet Re 25: half-width b½ at x/b = 20 / Bickley", 1.0 + v["b_half_err_pct_x20"] / 100.0,
                            0.9, 1.1, "bickley", ref=1.0, ref_text="1 (similarity law, fitted virtual origin)",
                            tolerance="± 10 %"))
    return checks, {"jet": v, "span": list(JET_SPAN), "grid": list(r.grid), "steps": r.steps, "wall_s": r.wall_s}


# ---------------------------------------------------------------------------
# heat transfer, fans and emitters in 3D (validation/thermal3d_cases.py)
# ---------------------------------------------------------------------------

def _published(mid: dict, key: str) -> list[float]:
    """The published values of one symmetry-plane quantity of the cubic cavity, whichever way
    the case module tabulates them: one number (``"nu": 2.302``), one per source
    (``"nu_tric": 2.25, "nu_fusegi": 2.302``) or (value, position) pairs per source
    (``"u_max": {"wakashima": (0.198, 0.825), ...}``)."""
    raw = mid.get(key)
    if isinstance(raw, dict):
        return [float(v[0] if isinstance(v, (tuple, list)) else v) for v in raw.values()]
    if raw is not None:
        return [float(raw[0] if isinstance(raw, (tuple, list)) else raw)]
    return [float(v) for k, v in mid.items() if k.startswith(f"{key}_") and isinstance(v, (int, float))]


def case_cavity3d(ctx: Context) -> tuple[list[Check], dict]:
    """Differentially heated cube, Ra 10⁴. The published hot-wall means span 2.054 (Tric et al.,
    pseudo-spectral) to 2.100 (Fusegi et al., 62³ finite volumes); the band is that range widened
    by 3 %: the 2 % the 2D cavity is held to at 32² and 64², plus 1 % for grids that are coarser
    per direction than Fusegi's (32³ quick, 48³ full). The symmetry-plane quantities are held to
    the range of their published values (Tric et al., Wakashima & Saitoh, Fusegi et al., as the
    case module tabulates them) widened by 3 % (Nusselt number) and 5 % (velocity maxima)."""
    m = ctx.module("thermal3d_cases")
    ra, n = 1e4, (32 if ctx.quick else 48)
    r = m.run_cavity(ra, n, device=ctx.device)
    v = r.values
    refs = m.CAVITY_NU[ra]
    lo, hi = min(refs.values()), max(refs.values())
    mid = m.CAVITY_MIDPLANE[ra]
    tag = f"Cubic cavity Ra {_pow10(ra)} ({n}³)"
    checks = [
        Check(f"{tag}: mean Nu, hot wall", v["nu_hot"], *widen(lo, hi, 0.03), "cavity3d", ref=0.5 * (lo + hi),
              ref_text=f"{lo:g}–{hi:g}", tolerance="published range ± 3 %"),
    ]
    for key, value, label, tol in (("nu", v["nu_sym_plane"], "mean Nu of the hot wall in the plane y = ½", 0.03),
                                   ("u_max", v["u_max"], "u_max on the vertical centre line [free-fall units]", 0.05),
                                   ("w_max", v["w_max"], "w_max on the horizontal centre line [free-fall units]", 0.05)):
        pub = _published(mid, key)
        if not pub:
            continue
        a, b = min(pub), max(pub)
        checks.append(Check(f"{tag}: {label}", value, *widen(a, b, tol), "cavity3d-plane", ref=0.5 * (a + b),
                            ref_text=f"{a:g}" if a == b else f"{a:g}–{b:g}",
                            tolerance=f"{'published value' if a == b else 'published range'} ± {100 * tol:g} %"))
    checks.append(Check(f"{tag}: energy balance |Nu_hot − Nu_cold| / Nu_hot", abs(v["balance"]) / abs(v["nu_hot"]),
                        0.0, 0.01, "exact", ref=0.0, ref_text="0 (steady state)", tolerance="< 1 %"))
    return checks, {"cavity3d": {"ra": ra, "n": n, **v}, "grid": list(r.grid), "steps": r.steps, "wall_s": r.wall_s}


def case_heated_sphere(ctx: Context) -> tuple[list[Check], dict]:
    """Forced convection from an isothermal sphere, Pr 0.71. The three correlations of the case
    module scatter by about 10 %; the band is their range widened by 5 %."""
    m = ctx.module("thermal3d_cases")
    re, cpd = 100.0, (16 if ctx.quick else 24)
    r = m.run_heated_sphere(re, cpd, device=ctx.device)
    v = r.values
    corr = {"Ranz–Marshall": v["nu_ranz"], "Whitaker": v["nu_whitaker"], "Clift": v["nu_clift"]}
    lo, hi = min(corr.values()), max(corr.values())
    checks = [
        Check(f"Heated sphere Re {re:g}, Pr 0.71: mean Nu ({cpd} cells/D)", v["nu"], *widen(lo, hi, 0.05),
              "sphere-nu", ref=v["nu_clift"], ref_text=f"{lo:.3f}–{hi:.3f} (Clift {v['nu_clift']:.3f})",
              tolerance="correlation range ± 5 %", note="2.85 % blockage, half domain"),
        Check(f"Heated sphere Re {re:g}: heat leaving the domain / heat from the sphere", v["balance_ratio"], 0.98,
              1.02, "exact", ref=1.0, ref_text="1 (steady state)", tolerance="± 2 %"),
        Check(f"Heated sphere Re {re:g}: drift of Nu over the last quarter of the run", v["nu_drift"], 0.0, 0.01,
              "exact", ref=0.0, ref_text="0 (steady state)", tolerance="< 1 %"),
    ]
    return checks, {"sphere": v, "grid": list(r.grid), "steps": r.steps, "wall_s": r.wall_s}


#: grids of the natural-convection sphere (cells per diameter): a constant refinement ratio of 1.5,
#: every one a whole number of cells across the box (thermal3d_cases.natural_sphere_config)
NATURAL_SPHERE_GRIDS = (12, 18, 27)


def case_natural_sphere(ctx: Context) -> tuple[list[Check], dict]:
    """Natural convection from an isothermal sphere at Ra 10⁴, Pr 0.71, in a closed box of
    cold walls (6 D across, 9 D high), against Churchill (1983), with a grid study.

    Added before the first release after a review found that a Draft hot sphere at 0.4 cells across its
    thermal boundary layer read Nu 2.4× the correlation. This case measures where the 3D
    solver's Nusselt number converges for a buoyancy-driven body flow. Ra 10⁴ puts the layer
    (about D·Ra^-1/4 = 0.1 D) at 1.2, 1.8 and 2.7 cells on the three grids.

    The band is ± 10 % about Churchill's value: the correlation is an empirical fit to
    experiments in an unbounded fluid and its source states no accuracy, so ± 10 % is this
    suite's choice for a single empirical correlation, stated as such. The box warms the fluid
    beside the sphere (``theta_side``, in the note) and its cold walls are a conduction sink
    nearer than infinity; both move Nu by a few per cent and are reported, not corrected.
    The grid study (Celik et al. 2008, ``aerolume.analysis.gci``) is in the note and the data."""
    from aerolume.analysis.gci import gci_study
    m = ctx.module("thermal3d_cases")
    ra = 1e4
    grids = NATURAL_SPHERE_GRIDS[:2] if ctx.quick else NATURAL_SPHERE_GRIDS
    runs = [m.run_natural_sphere(ra, n, device=ctx.device) for n in grids]
    fine = runs[-1].values
    nus = [r.values["nu"] for r in runs]
    note = f"Nu {' / '.join(f'{x:.4g}' for x in nus)} at {' / '.join(map(str, grids))} cells/D"
    study: dict[str, Any] = {}
    if len(runs) >= 3:
        g = gci_study([1.0 / n for n in grids], nus, quantity="Nu", labels=[f"{n}/D" for n in grids])
        study = {"p": g.p, "phi_ext": g.phi_ext, "u_num": g.u_num, "gci_fine": g.gci_fine,
                 "convergence": g.convergence}
        note += (f"; {g.convergence}, observed order {g.p:.2f}, extrapolated {g.phi_ext:.4g}, "
                 f"GCI {100 * g.gci_fine:.2g} %")
    note += (f"; fluid beside the sphere at θ {fine['theta_side']:.3f} of the sphere's excess "
             f"(Nu on that local ambient {fine['nu_local']:.4g})")
    ref = fine["nu_churchill"]
    checks = [
        Check(f"Natural convection, sphere Ra 10⁴, Pr 0.71: mean Nu ({grids[-1]} cells/D)", fine["nu"],
              *band(ref, 0.10), "churchill-sphere", ref=ref, ref_text=f"{ref:.3f} (Churchill 1983)",
              tolerance="± 10 % (this suite's band for one empirical correlation)", note=note),
        Check("Natural-convection sphere: heat leaving through the walls / heat from the sphere",
              fine["balance_ratio"], 0.98, 1.02, "exact", ref=1.0, ref_text="1 (steady state)", tolerance="± 2 %"),
        Check("Natural-convection sphere: drift of Nu over the last 30 free-fall times", fine["nu_drift"], 0.0,
              0.01, "exact", ref=0.0, ref_text="0 (steady state)", tolerance="< 1 %"),
    ]
    data = {"natural_sphere": {"ra": ra, "grids": list(grids), "nu": nus, "ref": ref, **study,
                               "runs": [{**r.values, "grid": list(r.grid), "steps": r.steps, "wall_s": r.wall_s}
                                        for r in runs]}}
    return checks, data


def case_actuator3d(ctx: Context) -> tuple[list[Check], dict]:
    """An actuator disc of diameter 1 in a square tunnel of side 6 D (blockage B = (π/4)/36 =
    2.2 %), both ways round: pushing the flow downstream (``Fan3D`` direction +x, a propeller,
    C_T 0.5) and holding it back (a turbine, C_T 0.4), each against its branch of the confined
    momentum theory. The turbine runs at C_T 0.4 because a heavily loaded turbine (C_T 0.8)
    reads +3.0 % above 1D momentum theory, about a third of it viscous (docs/theory_ns3d.md
    section 17); at 0.4 the theory is the like-for-like reference."""
    m = ctx.module("thermal3d_cases")
    cpd = 12 if ctx.quick else 16
    blockage = (math.pi / 4.0) / (2.0 * 3.0) ** 2
    checks: list[Check] = []
    data: dict[str, Any] = {}
    for propeller, ct in ((True, 0.5), (False, 0.4)):
        r = m.run_actuator_disc(ct, cpd, propeller=propeller, device=ctx.device)
        v = r.values
        signed = ct if propeller else -ct
        th = confined_disc_theory(signed, blockage)
        fr = froude_disc(signed)
        kind = "propeller" if propeller else "turbine"
        checks += [
            Check(f"Actuator disc C_T {ct:g} ({kind}), blockage 2.2 %: disc velocity / U ({cpd} cells/D)",
                  v["u_disc"], *band(th["u_t"], 0.03), "garrett-cummins", ref=th["u_t"],
                  ref_text=f"{th['u_t']:.4f} (confined momentum theory)", tolerance="± 3 %",
                  note=f"unconfined Froude value {fr['u_disc']:.4f}"),
            Check(f"Actuator disc C_T {ct:g} ({kind}): delivered thrust / set thrust", v["ct"] / ct, 0.99, 1.01,
                  "exact", ref=1.0, ref_text="1", tolerance="± 1 %"),
        ]
        data[kind] = {"ct": ct, "u_disc": v["u_disc"], "u_disc_theory": th["u_t"], "u_disc_froude": fr["u_disc"],
                      "u_axis_2d": v["u_axis_2d"], "u_axis_4d": v["u_axis_4d"], "u_wake_froude": fr["u_wake"],
                      "grid": list(r.grid), "steps": r.steps, "wall_s": r.wall_s}
    return checks, data


def case_jet3d(ctx: Context) -> tuple[list[Check], dict]:
    """A laminar round jet (port diameter 1, Re_j 100, 4 cells per diameter) blowing into a
    closed box, against Schlichting's similarity solution: the centre-line decay slope of
    1/u_c for the injected momentum flux, the profile shape inside 2 r½, and the momentum
    budget of the jet core, which separates the box's return-flow shear and pressure gradient
    from the scheme's error (thermal3d_cases.round_jet_measures). Measured by the case module's
    author: residual −0.12 % of K, slope within 1–2 % of the k0 law. The slow half-width growth
    is not checked here: its 4–11 % deviation was shown by a confinement study to come from the
    box (docs/theory_ns3d.md section 17)."""
    m = ctx.module("thermal3d_cases")
    r = m.run_round_jet(100.0, 4, device=ctx.device)
    v = r.values
    checks = [
        Check("Round jet Re 100: momentum budget of the jet core, residual / K", abs(v["budget_residual_pct"]) / 100.0,
              0.0, 0.005, "exact", ref=0.0, ref_text="0 (steady x-momentum balance of the core)", tolerance="< 0.5 %",
              note=(f"change of K {v['budget_dK_pct']:+.2f} %: return-flow shear {v['budget_shear_pct']:+.2f} %, "
                    f"box pressure gradient {v['budget_pressure_pct']:+.2f} %")),
        Check("Round jet Re 100: slope of 1/u_c / Schlichting (injected K)", 1.0 + v["inv_uc_err_k0"], 0.98, 1.02,
              "schlichting", ref=1.0, ref_text="1 (similarity law)", tolerance="± 2 %"),
        Check("Round jet Re 100: rms of u/u_c − (1 + ξ²/4)⁻² inside 2 r½", v["shape_rms"], 0.0, 0.02, "schlichting",
              ref=0.0, ref_text="0 (similarity profile)", tolerance="< 0.02"),
    ]
    return checks, {"jet3d": {k: v[k] for k in v if isinstance(v[k], (int, float))}, "grid": list(r.grid),
                    "steps": r.steps, "wall_s": r.wall_s}


def case_emitters3d(ctx: Context) -> tuple[list[Check], dict]:
    """A heated sphere with a smoke rake and a pulsed disc emitter across the symmetry plane:
    the half-domain run must reproduce the full one (same discrete problem), float64 on the CPU."""
    m = ctx.module("thermal3d_cases")
    out = m.emitter_symmetry()
    names = {"dye": "smoke channel 0 (rake)", "dye1": "smoke channel 1 (pulsed disc)", "temperature": "temperature",
             "u": "velocity u", "heat_rate": "sphere heat rate"}
    checks = [Check(f"Half vs full domain, max relative difference: {label}", out[key], 0.0, 1e-6, "exact", ref=0.0,
                    ref_text="0 (mirror-symmetric problem)", tolerance="< 10⁻⁶ (float64)")
              for key, label in names.items() if key in out]
    return checks, {"emitters3d": out}


# ---------------------------------------------------------------------------
# discretisation uncertainty (analysis.gci on a benchmark with a known answer)
# ---------------------------------------------------------------------------

#: grids of the cavity grid study: cells per side, fine first. The GCI is computed from the
#: three coarser grids; the finest one tests whether that band covers the actual change.
GCI_GRIDS = (128, 64, 32, 16)


def case_gci_cavity(ctx: Context) -> tuple[list[Check], dict]:
    """The uncertainty machinery itself, checked where the answer can be tested.

    The square cavity at Ra 10³ (a steady, deterministic case) is run on 16², 32², 64² and
    128² cells. ``aerolume.analysis.gci.gci_study`` (Celik et al. 2008) on the three coarser
    grids gives the observed order and the GCI band of the 64² value. Three checks:

    1. the observed order is the formal second order (± 25 %: the coarsest grid has 16
       cells across the cavity);
    2. **coverage**: the 128² value lies inside the 64² GCI band. The GCI bounds the
       discretisation error, so it is tested against a finer grid of the same discrete
       problem, not against a benchmark (whose own error and rounding are of the same size
       as the band here);
    3. the Richardson-extrapolated value of the finest three grids lies within the spread of
       de Vahl Davis's own values for quantities that are equal in the exact steady solution
       (hot-wall Nu₀ 1.117, mid-plane Nu 1.118, mean Nu 1.118), widened by their rounding
       (± 0.0005).

    History: the first version of check 2 asked for de Vahl Davis's Nu₀ inside the 64² band
    and failed (|Nu₆₄ − 1.117| = 1.9 GCI half-widths) because the benchmark's own 0.001
    spread exceeds the 0.0007 band; docs/validation.md section 9 records it.
    """
    from aerolume.analysis.gci import gci_study
    m = ctx.module("thermal2d_cases")
    ra = 1e3
    dvd = m.DE_VAHL_DAVIS[ra]
    runs = [m.run_cavity(ra, n, device=ctx.device) for n in GCI_GRIDS]
    vals = [r.values["nu_hot"] for r in runs]
    h = [1.0 / n for n in GCI_GRIDS]
    g = gci_study(h[1:], vals[1:], quantity="Nu (hot wall)", labels=[f"{n}²" for n in GCI_GRIDS[1:]])
    best = gci_study(h[:3], vals[:3], quantity="Nu (hot wall)", labels=[f"{n}²" for n in GCI_GRIDS[:3]])
    cover = abs(vals[0] - vals[1]) / g.u_num if _finite(g.u_num) and g.u_num > 0 else float("nan")
    spread = [dvd["nu_0"], dvd["nu_half"], dvd["nu_bar"]]
    lo, hi = min(spread) - 0.0005, max(spread) + 0.0005
    checks = [
        Check("Cavity Ra 10³, grids 16² / 32² / 64²: observed order of Nu", g.p, 1.5, 2.5, "celik", ref=2.0,
              ref_text="2 (formal order)", tolerance="± 25 %", note=g.convergence),
        Check("GCI coverage: |Nu₁₂₈ − Nu₆₄| / GCI half-width of Nu₆₄", cover, 0.0, 1.0, "celik", ref=None,
              ref_text="≤ 1 (the band holds the finer grid)", tolerance="≤ 1",
              note=f"Nu₆₄ = {vals[1]:.5f} ± {g.u_num:.2g} (GCI {100 * g.gci_fine:.2g} %), Nu₁₂₈ = {vals[0]:.5f}"),
        Check("Richardson-extrapolated Nu (32² / 64² / 128²) vs de Vahl Davis", best.phi_ext, lo, hi, "devahl",
              ref=dvd["nu_0"], ref_text=f"{min(spread):g}–{max(spread):g}",
              tolerance="the benchmark's own spread ± 0.0005 rounding",
              note=f"observed order {best.p:.3f} on the finest three grids"),
    ]
    data = {"gci": {"ra": ra, "n": list(GCI_GRIDS), "nu": vals, "ref": dvd["nu_0"], "ref_spread": spread,
                    "p": g.p, "phi_ext": g.phi_ext, "u_num": g.u_num, "gci_fine": g.gci_fine,
                    "convergence": g.convergence, "p_best": best.p, "phi_ext_best": best.phi_ext,
                    "u_num_best": best.u_num, "steps": [r.steps for r in runs],
                    "wall_s": [r.wall_s for r in runs]}}
    return checks, data


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

CASES: list[Case] = [
    Case("tg2d", "Taylor–Green vortex, 2D", "verification", case_tg2d, ("exact",), quick=True, est_seconds=6,
         gpu=False),
    Case("tg3d", "Taylor–Green vortex, 3D", "verification", case_tg3d, ("exact",), quick=True, est_seconds=20,
         gpu=False),
    Case("divergence", "Discrete incompressibility", "verification", case_divergence, ("exact",), quick=True,
         est_seconds=5),
    Case("cyl40", "Circular cylinder, Re 40 (steady)", "flow2d", case_cyl40, ("dennis-chang", "coutanceau"),
         quick=True, est_seconds=20),
    Case("cyl100", "Circular cylinder, Re 100 (shedding)", "flow2d", case_cyl100, ("liu98", "williamson"),
         est_seconds=60),
    Case("schafer_turek", "Schäfer–Turek 2D-1 and 2D-2", "flow2d", case_schafer_turek, ("schafer-turek",),
         est_seconds=40),
    Case("square", "Square cylinder, Re 100", "flow2d", case_square, ("sohankar",), est_seconds=450),
    Case("sphere100", "Sphere, Re 100", "flow3d", case_sphere100, ("clift", "johnson-patel"), quick=True,
         est_seconds=40),
    Case("sphere_symmetry", "Sphere: symmetry plane vs full domain", "flow3d", case_sphere_symmetry, ("exact",),
         est_seconds=40),
    Case("sphere300", "Sphere, Re 300 (shedding)", "flow3d", case_sphere300, ("johnson-patel",), est_seconds=900,
         stretch=True),
    Case("cavity2d", "Differentially heated square cavity", "thermal", case_cavity2d, ("devahl",), quick=True,
         est_seconds=15),
    Case("heated_cylinder", "Forced convection from a heated cylinder", "thermal", case_heated_cylinder,
         ("cylinder-nu",), quick=True, est_seconds=30),
    Case("rayleigh_benard", "Rayleigh–Bénard onset", "thermal", case_rayleigh_benard, ("chandrasekhar",),
         est_seconds=20),
    Case("cavity3d", "Differentially heated cubic cavity", "thermal", case_cavity3d, ("cavity3d", "cavity3d-plane"),
         quick=True, est_seconds=15),
    Case("heated_sphere", "Forced convection from a heated sphere", "thermal", case_heated_sphere, ("sphere-nu",),
         est_seconds=15),
    Case("natural_sphere", "Natural convection from a heated sphere", "thermal", case_natural_sphere,
         ("churchill-sphere", "celik"), est_seconds=600),
    Case("gci_cavity", "Grid-convergence uncertainty (GCI), square cavity", "thermal", case_gci_cavity,
         ("celik", "devahl"), quick=True, est_seconds=60),
    Case("actuator2d", "Actuator strip in a 2D channel", "fans", case_actuator2d, ("garrett-cummins",), quick=True,
         est_seconds=10),
    Case("actuator3d", "Actuator disc in a 3D tunnel", "fans", case_actuator3d, ("garrett-cummins",), quick=True,
         est_seconds=20),
    Case("plane_jet", "Plane laminar jet from a port", "fans", case_plane_jet, ("bickley", "exact"), est_seconds=20),
    Case("jet3d", "Round laminar jet from a port", "fans", case_jet3d, ("schlichting", "exact"), est_seconds=60),
    Case("emitters3d", "Smoke emitters and heat: symmetry plane vs full domain", "verification", case_emitters3d,
         ("exact",), quick=True, est_seconds=5, gpu=False),
    Case("ladson", "NACA 0012, Re 6×10⁶, tripped", "airfoil", case_ladson, ("ladson",), quick=True,
         est_seconds=30, gpu=False),
    Case("xfoil", "Cross-check against XFOIL 6.99", "airfoil", case_xfoil, ("xfoil",), est_seconds=180, gpu=False),
    Case("wing_theory", "Lifting-line and lifting-surface theory", "wing", case_wing_theory,
         ("llt", "multhopp", "falkner"), quick=True, est_seconds=10, gpu=False),
    Case("weber", "Weber & Brebner 45° swept wing", "wing", case_weber, ("weber",), est_seconds=5, gpu=False),
]


def discover(cases: list[Case]) -> list[Case]:
    """``cases`` plus every ``RUN_ALL_CASES`` of ``validation/*_cases.py`` (keys must be new)."""
    out = list(cases)
    keys = {c.key for c in out}
    for path in sorted(HERE.glob("*_cases.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "RUN_ALL_CASES" not in text:
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"aerolume_validation_{path.stem}", path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            sys.modules[spec.name] = mod                  # type: ignore[union-attr]
            spec.loader.exec_module(mod)                  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - a broken plug-in module is reported, not fatal
            print(f"warning: {path.name} could not be imported ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        for c in getattr(mod, "RUN_ALL_CASES", []) or []:
            usable = isinstance(c, Case) or all(hasattr(c, a) for a in ("key", "title", "area", "run"))
            if usable and c.key not in keys:
                out.append(c)
                keys.add(c.key)
    return out


def select(cases: Sequence[Case], *, quick: bool, names: Iterable[str] | None = None,
           stretch: bool = False, areas: Iterable[str] | None = None) -> list[Case]:
    """The cases to run: by name (exact key or key prefix), else every case of the mode."""
    names = [n.strip() for n in (names or []) if n.strip()]
    areas = set(areas or [])
    if names:
        chosen = [c for c in cases if any(c.key == n or c.key.startswith(n) for n in names)]
        missing = [n for n in names if not any(c.key == n or c.key.startswith(n) for c in cases)]
        if missing:
            raise KeyError(f"unknown case(s): {', '.join(missing)}")
    else:
        chosen = [c for c in cases if (c.quick or not quick) and (stretch or not c.stretch)]
    if areas:
        chosen = [c for c in chosen if c.area in areas]
    return chosen


def run_case(case: Case, ctx: Context) -> CaseResult:
    """Run one case, catching everything: a failed case never stops the others."""
    t0 = time.perf_counter()
    res = CaseResult(case.key, case.title, case.area, "error")
    try:
        out = case.run(ctx)
        checks, data = (out if isinstance(out, tuple) else (out, {}))
        res.checks = list(checks)
        res.data = dict(data or {})
        if not res.checks:
            res.status, res.error = "error", "the case produced no checks"
        else:
            res.status = "pass" if all(c.passed for c in res.checks) else "fail"
    except Unavailable as exc:
        res.status, res.error = "unavailable", str(exc)
    except Exception as exc:  # noqa: BLE001 - reported per case
        res.status = "error"
        res.error = f"{type(exc).__name__}: {exc}"
        res.notes.append(traceback.format_exc(limit=6))
    finally:
        res.seconds = time.perf_counter() - t0
        with contextlib.suppress(Exception):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# environment and reporting
# ---------------------------------------------------------------------------

#: The code a validation number depends on: the solvers and everything they import, the post-processing
#: the cases use (the GCI module), the data model, the package version, and the case modules with their
#: bands. The GUI, the reports, the exports and the comparison tables are left out on purpose, so editing a
#: dialog or a report table does not mark the validation stale.
SOURCE_SCOPE = ("src/aerolume/__init__.py", "src/aerolume/model.py", "src/aerolume/solvers", "src/aerolume/core",
                "src/aerolume/geometry", "src/aerolume/geometry3d", "src/aerolume/analysis/gci.py", "validation")


def source_digest(repo: Path = REPO, scope: Sequence[str] = SOURCE_SCOPE) -> dict[str, Any]:
    """``{"sha256", "files", "scope"}``: one SHA-256 over every ``.py`` file in ``scope``.

    Each file contributes its repository-relative path and its text with line endings normalised
    to LF, in sorted path order, so the digest is the same on Windows and Linux checkouts and
    changes exactly when the measured code (or a band) changes. The installed version string
    alone cannot tell this: an editable install keeps reporting the version it was installed
    with (an older development version here, while the package said another), which is how section 9 came to be stamped
    with a stale version. ``--check-docs`` compares this digest with the one in the docs.
    """
    files: list[Path] = []
    for item in scope:
        path = repo / item
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(q for q in path.rglob("*.py") if "__pycache__" not in q.parts)
    h = hashlib.sha256()
    rels = sorted({q.relative_to(repo).as_posix(): q for q in files}.items())
    for rel, q in rels:
        text = q.read_bytes().replace(b"\r\n", b"\n")
        h.update(rel.encode("utf-8") + b"\0" + str(len(text)).encode() + b"\0" + text)
    return {"sha256": h.hexdigest(), "files": len(rels), "scope": list(scope)}


def package_version() -> str:
    """The version of the Aerolume package that is actually imported (``aerolume.__version__``)."""
    try:
        import aerolume
        return str(aerolume.__version__)
    except Exception:  # noqa: BLE001 - a broken install still gets a record
        return "—"


def _os_name() -> str:
    """"Windows 11" on Windows 11: ``platform.release()`` reports "10" for every build since 10, and
    Microsoft tells the two apart only by build number (22000 and later is Windows 11)."""
    name, release = platform.system(), platform.release()
    if name == "Windows":
        with contextlib.suppress(ValueError, IndexError):
            if int(platform.version().split(".")[2]) >= 22000:
                release = "11"
    return f"{name} {release}"


def environment(device: str) -> dict[str, Any]:
    """Software and hardware of this run: the imported Aerolume's own version and source digest,
    the other packages' versions from their installed metadata."""
    from importlib import metadata
    env: dict[str, Any] = {"python": platform.python_version(), "os": _os_name(),
                           "machine": platform.machine(), "processor": platform.processor(), "device": device,
                           "aerolume": package_version()}
    with contextlib.suppress(Exception):
        env["source"] = source_digest()
    for dist in ("torch", "numpy", "scipy", "vtk"):
        with contextlib.suppress(Exception):
            env[dist] = metadata.version(dist)
    with contextlib.suppress(Exception):
        import torch
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    return env


def _fmt(x: Any, digits: int = 4) -> str:
    if not _finite(x):
        return "—"
    v = float(x)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        e = math.floor(math.log10(abs(v)))
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f"{v / 10 ** e:.2f} × 10{str(e).translate(sup)}"
    return f"{v:.{digits}g}"


def _digits(c: Check) -> int:
    """Significant digits that resolve the band: 4, more when the band is narrow relative to the
    value (a 0.2 % band printed with 4 digits would hide where in it the value lies)."""
    scale = max(abs(c.lo), abs(c.hi), abs(float(c.value)) if _finite(c.value) else 0.0)
    width = c.hi - c.lo
    if not (scale > 0 and width > 0):
        return 4
    return int(min(7, max(4, math.ceil(-math.log10(width / scale)) + 2)))


def _value_text(c: Check) -> str:
    """The measured value as printed: "NaN" (not a dash) when the case produced no finite number,
    so a solver that returned NaN is not mistaken for a quantity that does not apply."""
    return _fmt(c.value, _digits(c)) if _finite(c.value) else "NaN"


def _band_text(c: Check) -> str:
    d = _digits(c)
    return f"{_fmt(c.lo, d)} … {_fmt(c.hi, d)}"


def _cell(text: Any) -> str:
    """Text for a Markdown table cell: a literal "|" (as in |∇·u|) would end the cell."""
    return str(text).replace("|", r"\|")


#: heading of the generated section of docs/validation.md
SECTION_TITLE = "## 9. Automated validation run"
#: how the generated section names the source digest, and how many hex digits it prints (64 bits:
#: enough to tell code versions apart; the record keeps the full digest)
DIGEST_LABEL = "source digest (SHA-256)"
DIGEST_CHARS = 16
#: the record the generated section was written from, kept in the repository (validation/out/ is not)
RECORD_REL = "validation/records/validation-record.json"


def markdown(results: Sequence[CaseResult], meta: dict[str, Any], *, figures: Sequence[str] = ()) -> str:
    """The generated section of docs/validation.md: run facts, a summary per area, every check."""
    n_checks = sum(len(r.checks) for r in results)
    n_pass = sum(c.passed for r in results for c in r.checks)
    n_fail = n_checks - n_pass
    intro = ("This section is written by `python validation/run_all.py --write-docs`; do not edit it by hand. "
             "Every number below was measured in the run described here; each check compares Aerolume with the "
             "cited reference and the tolerance band stated beside it. A band is the reference's own spread "
             "widened by a stated allowance for the resolution or the confinement, written in `run_all.py` "
             "before the case runs; the one band rule revised after a deviation was known (the square "
             "cylinder's drag) says so in section 3 and in its note. A failing check stays in the table. "
             "*Unavailable* cases need a solver feature this installation does not have; they are listed, never "
             "counted as passed.")
    modes = meta.get("mode", "—")
    run = (f"* Run: {meta.get('date', '—')}, mode **{modes}**, {len(results)} cases, {n_pass} of {n_checks} "
           f"checks passed" + (f" (**{n_fail} failed**)" if n_fail else "")
           + f", {meta.get('wall_minutes', 0):.1f} min of case time in total.")
    machine = (f"* Machine: {meta.get('gpu') or meta.get('processor') or '—'}; {meta.get('os', '—')}; Python "
               f"{meta.get('python', '—')}, PyTorch {meta.get('torch', '—')} (CUDA {meta.get('cuda', '—')}), "
               f"Aerolume {meta.get('aerolume', '—')}.")
    src = meta.get("source") or {}
    code = (f"* Code measured: {DIGEST_LABEL} `{str(src.get('sha256', ''))[:DIGEST_CHARS]}` over {src.get('files', '—')} "
            "files (the solvers, geometry, GCI analysis, data model and validation cases; "
            "`python validation/run_all.py --check-docs` says whether the code still matches)."
            if src.get("sha256") else "* Code measured: no source digest in this record (a run made before records carried one).")
    if meta.get("source_mixed"):
        code += (" **The records documented here were measured on different code** (digests "
                 + ", ".join(f"`{d[:DIGEST_CHARS]}`" for d in meta["source_mixed"]) + "): re-run before quoting them.")
    lines = [SECTION_TITLE, "", intro, "", run, machine, code,
             "* GPU wall times are upper bounds: the development GPU is shared with other programs.",
             ("* Reproduce: `python validation/run_all.py --all` (or `--quick`, `--cases <keys>`). Each run writes "
              "its JSON record to `validation/out/` (not kept in the repository); the record this section was "
              f"written from is kept as `{RECORD_REL}`."), ""]
    lines += ["| area | case | checks passed | status | time |", "|---|---|---:|---|---:|"]
    area_order = {a: i for i, a in enumerate(AREAS)}
    for r in sorted(results, key=lambda r: area_order.get(r.area, len(area_order))):
        ok = sum(c.passed for c in r.checks)
        status = {"pass": "pass", "fail": "**FAIL**", "unavailable": "unavailable", "error": "**ERROR**"}[r.status]
        lines.append(f"| {AREAS.get(r.area, r.area)} | {_cell(r.title)} | {ok} / {len(r.checks)} | {status} | "
                     f"{r.seconds:.0f} s |")
    for fig in figures:
        lines += ["", fig]
    for area, title in AREAS.items():
        rs = [r for r in results if r.area == area]
        if not rs:
            continue
        lines += ["", f"### {title}", "", "| check | Aerolume | reference | band | result | source |",
                  "|---|---:|---|---|---|---|"]
        notes: list[str] = []
        for r in rs:
            if r.status in ("unavailable", "error") and not r.checks:
                lines.append(f"| {_cell(r.title)} | — | — | — | {r.status}: {_cell(r.error.splitlines()[0][:160])} | — |")
                continue
            for c in r.checks:
                tol = f"{_band_text(c)}" + (f" ({c.tolerance})" if c.tolerance else "")
                res_txt = "pass" if c.passed else "**fail**"
                src = c.source
                mark = ""
                if c.note:
                    notes.append(c.note)
                    mark = f" ⁽{len(notes)}⁾".translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
                lines.append(f"| {_cell(c.quantity)}{mark} | {_value_text(c)} | {_cell(c.ref_text or _fmt(c.ref))} | "
                             f"{_cell(tol)} | {res_txt} | [{src}](#ref-{src}) |")
        if notes:
            lines += [""] + [f"{i}. {n}" for i, n in enumerate(notes, 1)]
    used = sorted({c.source for r in results for c in r.checks} | {s for r in results for s in _sources(r)})
    lines += ["", "### References of the automated run", ""]
    for key in used:
        if key in REFERENCES:
            lines.append(f"* <a id=\"ref-{key}\"></a>**{key}**: {REFERENCES[key]}")
    return "\n".join(lines) + "\n"


def _sources(r: CaseResult) -> list[str]:
    case = next((c for c in CASES if c.key == r.key), None)
    return list(case.sources) if case is not None else []


def load_records(paths: Iterable[str | Path]) -> tuple[list[CaseResult], dict[str, Any]]:
    """Case results and run facts from JSON records written by earlier runs (``--from-json``).

    Several records merge in the order given; a case that appears in more than one keeps
    its latest run. So the areas can be run separately (one GPU job at a time on a shared
    machine) and the documentation written from all of them. The facts are those of the
    last record, with the modes and dates of all of them and the summed case time.
    """
    by_key: dict[str, CaseResult] = {}
    metas: list[dict[str, Any]] = []
    for p in paths:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        metas.append(dict(rec.get("meta") or {}))
        for d in rec.get("results") or []:
            checks = [Check(quantity=c["quantity"], value=c["value"] if c["value"] is not None else float("nan"),
                            lo=c["lo"], hi=c["hi"], source=c["source"], ref=c.get("ref"),
                            ref_text=c.get("ref_text", ""), tolerance=c.get("tolerance", ""), unit=c.get("unit", ""),
                            note=c.get("note", "")) for c in d.get("checks") or []]
            by_key.pop(d["key"], None)
            by_key[d["key"]] = CaseResult(d["key"], d["title"], d["area"], d["status"], checks,
                                          float(d.get("seconds") or 0.0), list(d.get("notes") or []),
                                          d.get("error", ""), dict(d.get("data") or {}))
    if not metas:
        raise ValueError("no records given")
    order = {c.key: i for i, c in enumerate(CASES)}
    results = sorted(by_key.values(), key=lambda r: order.get(r.key, len(order)))
    meta = dict(metas[-1])
    meta["mode"] = " + ".join(dict.fromkeys(str(m.get("mode", "—")) for m in metas))
    dates = sorted(str(m.get("date", "")) for m in metas if m.get("date"))
    if dates:
        meta["date"] = dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}"
    meta["wall_minutes"] = sum(r.seconds for r in results) / 60.0
    digests = list(dict.fromkeys(str((m.get("source") or {}).get("sha256") or "") for m in metas))
    if len(digests) > 1:
        # records of different code merged into one table: said in the docs, never hidden
        meta["source_mixed"] = [d or "none" for d in digests]
    return results, meta


def docs_digest(path: Path = DOCS / "validation.md") -> str | None:
    """The source digest printed in the generated section of ``path`` (its first
    :data:`DIGEST_CHARS` hex digits), or None when the section has none."""
    import re
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(re.escape(DIGEST_LABEL) + r" `([0-9a-f]+)`", text)
    return m.group(1) if m else None


def check_docs(path: Path = DOCS / "validation.md", out: Any = None) -> int:
    """0 when the generated section of ``path`` was measured on the code in this checkout, else 1.

    Meant for a release checklist: after any change to the solvers, the geometry, the analysis,
    the model or a validation case, section 9 quotes numbers the shipped code may no longer
    produce, and this says so (``--check-docs``).
    """
    out = out or sys.stdout
    stamped = docs_digest(path)
    now = source_digest()["sha256"][:DIGEST_CHARS]
    if stamped is None:
        print(f"{path}: no source digest in the generated section; re-run with --write-docs", file=out)
        return 1
    if stamped != now:
        print(f"{path}: STALE. Section 9 was measured on {stamped}; this checkout is {now}. Re-run "
              "`python validation/run_all.py --all --write-docs`.", file=out)
        return 1
    print(f"{path}: up to date (source digest {now}).", file=out)
    return 0


#: what a path outside the repository is recorded as (see :func:`public_argv`)
OUTSIDE_REPO = "<path outside the repository>"


def _looks_like_path(text: str) -> bool:
    """True for an argument that names a file or folder rather than a case key, area or device."""
    return bool(text) and (Path(text).is_absolute() or "/" in text or "\\" in text or text.startswith("~"))


def public_path(text: str, repo: Path = REPO, cwd: Path | None = None) -> str:
    """``text`` as the public record may show it: relative to the repository, or :data:`OUTSIDE_REPO`."""
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (cwd or Path.cwd()) / p
    try:
        return p.resolve().relative_to(repo.resolve()).as_posix() or "."
    except (OSError, ValueError):
        return OUTSIDE_REPO


def public_argv(argv: Sequence[str], repo: Path = REPO, cwd: Path | None = None) -> list[str]:
    """The command line as the record keeps it, without the measuring machine's folders.

    Why: the record of a documentation run is committed (:data:`RECORD_REL`) and quoted by the
    website, so it is public. A run started with an absolute ``--out`` in a scratch folder once
    put the developer's user name and folder layout into that public file. Every argument that names a path (``--out``, ``--from-json`` records,
    ``--docs``...) is therefore written relative to the repository when it lies inside it, and as
    :data:`OUTSIDE_REPO` otherwise; flags, case keys, areas and devices are kept as typed, so
    the record still says how the run was made.
    """
    out: list[str] = []
    for arg in argv:
        opt, eq, value = arg.partition("=")
        if arg.startswith("-") and eq:
            out.append(f"{opt}={public_path(value, repo, cwd)}" if _looks_like_path(value) else arg)
        elif not arg.startswith("-") and _looks_like_path(arg):
            out.append(public_path(arg, repo, cwd))
        else:
            out.append(arg)
    return out


def write_record(results: Sequence[CaseResult], meta: dict[str, Any], path: Path = REPO / RECORD_REL) -> Path:
    """Keep the record a documentation run was written from in the repository (``RECORD_REL``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(meta.get("argv"), list):
        # a --from-json run carries the argv of the record it documents, which may predate
        # public_argv; arguments already made public are repository-relative, hence cwd=REPO
        meta = {**meta, "argv": public_argv([str(a) for a in meta["argv"]], REPO, cwd=REPO)}
    record = {"meta": meta, "results": [r.to_dict() for r in results]}
    path.write_text(json.dumps(_jsonable(record), indent=1, default=str, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def write_figures(results: Sequence[CaseResult], img: Path) -> list[str]:
    """Write every figure of :data:`FIGURES` that has data; their Markdown image lines."""
    out = []
    for name, build, alt in FIGURES:
        if build(results, img / name) is not None:
            caption = FIGURE_NOTES.get(name)
            out.append(f"![{alt}](img/{name})" + (f"\n\n*{caption}*" if caption else ""))
    return out


def write_docs(text: str, path: Path = DOCS / "validation.md") -> Path:
    """Replace the generated part of ``path`` (between :data:`BEGIN` and :data:`END`), or append it."""
    old = path.read_text(encoding="utf-8") if path.is_file() else "# Validation report\n"
    block = f"{BEGIN}\n{text}{END}\n"
    if BEGIN in old and END in old:
        head, rest = old.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new = head + block + tail.lstrip("\n")
    else:
        new = old.rstrip("\n") + "\n\n" + block
    path.write_text(new, encoding="utf-8", newline="\n")
    return path


@contextlib.contextmanager
def _figure(size: tuple[float, float], path: Path):
    """``(figure, style)`` in the print style of Aerolume's reports (so the validation figures
    read like the rest of the documentation), saved to ``path`` when the block ends cleanly."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from aerolume.viz import figures as F
    st = F.get_style("print")
    with F.style_context(st):
        fig = Figure(figsize=size, layout="constrained", facecolor=st.bg)
        FigureCanvasAgg(fig)
        yield fig, st
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=st.bg)


def figure_deviations(results: Sequence[CaseResult], path: Path) -> Path | None:
    """Relative deviation of every check that has a single non-zero reference value, drawn on
    its tolerance band (grey). Exact-zero references (round-off checks) are left out: they have
    no relative deviation."""
    rows = [(r, c) for r in results for c in r.checks if c.ref not in (None, 0) and _finite(c.value)]
    if not rows:
        return None
    with _figure((7.0, min(1.0 + 0.2 * len(rows), 16.0)), path) as (fig, st):
        ax = fig.add_subplot()
        for i, (_r, c) in enumerate(rows[::-1]):
            ref = float(c.ref)
            lo, hi = (c.lo - ref) / abs(ref) * 100, (c.hi - ref) / abs(ref) * 100
            ax.plot([lo, hi], [i, i], color=st.grid, lw=5.5, solid_capstyle="butt", zorder=1)
            ax.plot([c.deviation * 100], [i], "o" if c.passed else "X", ms=4.5 if c.passed else 6.0,
                    color=st.series[0] if c.passed else st.bad, zorder=3)
        ax.axvline(0.0, color=st.muted, lw=0.8)
        ax.set_yticks(range(len(rows)), [c.quantity for _, c in rows[::-1]], fontsize=6.0)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_xlabel("Deviation from the reference value [%]")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=st.grid, lw=5.5, label="tolerance band"),
                   Line2D([], [], color=st.series[0], marker="o", ls="none", ms=4.5, label="pass"),
                   Line2D([], [], color=st.bad, marker="X", ls="none", ms=6.0, label="fail")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    return path


def figure_cavity(results: Sequence[CaseResult], path: Path) -> Path | None:
    """Mean Nusselt number of the square cavity against Ra with de Vahl Davis's values."""
    data = next((r.data.get("cavity") for r in results if r.key == "cavity2d" and r.data.get("cavity")), None)
    if not data:
        return None
    with _figure((4.6, 3.0), path) as (fig, st):
        ax = fig.add_subplot()
        ax.plot([d["ra"] for d in data], [d["nu_ref"] for d in data], "s", mfc="none", mec=st.text, ms=7,
                ls="none", label="de Vahl Davis (1983)")
        ax.plot([d["ra"] for d in data], [d["nu"] for d in data], "o", color=st.series[0], ms=4.5, ls="none",
                label="Aerolume")
        for d in data:
            if d.get("n"):
                ax.annotate(f"{d['n']}²", (d["ra"], d["nu"]), textcoords="offset points", xytext=(6, -10),
                            fontsize=6.5, color=st.muted)
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 4, 8])
        ax.set_ylim(0.85, 12.0)
        ax.set_xlim(5e2, 2e6)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Rayleigh number Ra")
        ax.set_ylabel("Mean Nusselt number, hot wall")
        ax.set_title("Square cavity, Pr 0.71")
        ax.legend(loc="upper left")
        ax.grid(True, which="both")
    return path


def figure_gci(results: Sequence[CaseResult], path: Path) -> Path | None:
    """The grid study of ``gci_cavity``: Nu against h, the Richardson curve of the three coarser
    grids, the 64² GCI band, the 128² value it should hold, and de Vahl Davis's spread."""
    data = next((r.data.get("gci") for r in results if r.key == "gci_cavity" and r.data.get("gci")), None)
    if not data or not all(_finite(x) for x in data["nu"]):
        return None
    import numpy as np
    n = np.asarray(data["n"], float)
    nu = np.asarray(data["nu"], float)
    x = n[1] / n                                       # h / h(64²): 0.5, 1, 2, 4
    spread = data.get("ref_spread") or [data["ref"]]
    with _figure((4.8, 3.1), path) as (fig, st):
        ax = fig.add_subplot()
        ax.axhspan(min(spread) - 0.0005, max(spread) + 0.0005, color=st.panel, ec=st.grid, zorder=0,
                   label=f"de Vahl Davis {min(spread):g}–{max(spread):g} (± rounding)")
        if _finite(data.get("phi_ext")) and _finite(data.get("p")):
            xs = np.linspace(0.0, 4.2, 160)
            ax.plot(xs, data["phi_ext"] + (nu[1] - data["phi_ext"]) * xs ** data["p"], color=st.series[1],
                    lw=1.1, label=f"Richardson (16²–64²), p = {data['p']:.2f}")
            ax.plot([0.0], [data["phi_ext"]], "o", mfc=st.bg, mec=st.series[1], mew=1.2, ms=6, ls="none")
        if _finite(data.get("u_num")):
            ax.errorbar([1.0], [nu[1]], yerr=[[data["u_num"]], [data["u_num"]]], fmt="none", ecolor=st.text,
                        capsize=4, elinewidth=1.2, zorder=5, label="GCI band of 64²")
        ax.plot(x[1:], nu[1:], "o", color=st.series[0], ms=4.5, ls="none", zorder=6, label="16², 32², 64²")
        ax.plot(x[:1], nu[:1], "D", color=st.good, ms=4.5, ls="none", zorder=6, label="128² (coverage test)")
        ax.set_xlim(-0.1, 4.3)
        ax.set_xlabel("Cell size h / h(64²)")
        ax.set_ylabel("Mean Nusselt number, hot wall")
        ax.set_title("Cavity Ra 10³: grid convergence and the GCI")
        ax.legend(loc="upper left", fontsize=6.3)
    return path


#: figures of the generated section: file name, builder, alt text
FIGURES: tuple[tuple[str, Callable[[Sequence[CaseResult], Path], Path | None], str], ...] = (
    ("validation_run_all.png", figure_deviations,
     "Deviation of every check with a single reference value from that value, drawn on its tolerance band"),
    ("validation_cavity.png", figure_cavity,
     ("Mean Nusselt number of the differentially heated square cavity against the Rayleigh number, with de Vahl "
      "Davis's benchmark values")),
    ("validation_gci.png", figure_gci,
     ("Grid study of the square cavity at Ra 1000: Nusselt number against cell size, the Richardson curve, the GCI "
      "band of the 64-cell grid, the 128-cell value inside it, and the band of de Vahl Davis's values")),
)


#: a line under a figure where the picture alone could mislead
FIGURE_NOTES = {
    "validation_run_all.png": ("Checks against an exact zero (round-off, symmetry, conservation drift) have no "
                               "relative deviation and are not drawn; they, and any failure among them, are in the "
                               "tables below."),
}


def print_table(results: Sequence[CaseResult], out: Any = None) -> None:
    out = out or sys.stdout
    for r in results:
        head = f"[{r.status.upper():>11}] {r.key:<16} {r.title} ({r.seconds:.0f} s)"
        print(head, file=out)
        if r.error:
            print(f"              {r.error.splitlines()[0][:200]}", file=out)
        for c in r.checks:
            mark = "ok  " if c.passed else "FAIL"
            print(f"      {mark} {c.quantity:<64} {_value_text(c):>12}   band {_band_text(c)}", file=out)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Command-line entry; ``aerolume-cli validate`` is meant to delegate here with the same flags."""
    p = argparse.ArgumentParser(prog="run_all.py", description="Run Aerolume's validation cases.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="the quick subset (default)")
    mode.add_argument("--all", action="store_true", help="every case")
    p.add_argument("--stretch", action="store_true", help="include the long stretch cases (sphere Re 300)")
    p.add_argument("--cases", default="", help="comma-separated case keys (prefix match)")
    p.add_argument("--area", action="append", choices=sorted(AREAS), help="only these areas (repeatable)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--out", default=str(HERE / "out"), help="folder for the JSON record (default validation/out)")
    p.add_argument("--write-docs", action="store_true", help="rewrite docs/validation.md section 9 and its figures")
    p.add_argument("--docs", default=str(DOCS / "validation.md"), help=argparse.SUPPRESS)
    p.add_argument("--img", default=str(IMG), help=argparse.SUPPRESS)
    p.add_argument("--list", action="store_true", help="list the cases and exit")
    p.add_argument("--strict", action="store_true", help="unavailable cases also fail the run")
    p.add_argument("--json", action="store_true", help="print the JSON record instead of the table")
    p.add_argument("--from-json", nargs="+", metavar="RECORD",
                   help="run nothing: report (and with --write-docs document) earlier JSON records, merged in order")
    p.add_argument("--check-docs", action="store_true",
                   help="run nothing: exit 1 when docs/validation.md section 9 was measured on other code")
    p.add_argument("--record", default=str(REPO / RECORD_REL), help=argparse.SUPPRESS)
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code in (0, None) else 2
    if args.check_docs:
        return check_docs(Path(args.docs))
    if args.from_json:
        try:
            results, meta = load_records(args.from_json)
        except (OSError, ValueError, KeyError) as exc:
            print(f"error: cannot read the records ({type(exc).__name__}: {exc})", file=sys.stderr)
            return 2
        print_table(results)
        if args.write_docs:
            write_docs(markdown(results, meta, figures=write_figures(results, Path(args.img))), Path(args.docs))
            print(f"record kept: {write_record(results, meta, Path(args.record))}")
        bad = sum(r.status in ("fail", "error") for r in results)
        return 1 if bad or (args.strict and any(r.status == "unavailable" for r in results)) else 0
    cases = discover(CASES)
    quick = not args.all
    if args.list:
        for c in cases:
            tag = "quick" if c.quick else ("stretch" if c.stretch else "all")
            print(f"{c.key:<18} {tag:<8} ~{c.est_seconds:>5.0f} s  {AREAS.get(c.area, c.area):<28} {c.title}  "
                  f"[{', '.join(c.sources)}]")
        return 0
    try:
        chosen = select(cases, quick=quick, names=args.cases.split(",") if args.cases else None,
                        stretch=args.stretch, areas=args.area)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    ctx = Context(device=args.device, quick=quick)
    source_before = None
    with contextlib.suppress(Exception):
        source_before = source_digest()
    t0 = time.perf_counter()
    results = []
    for c in chosen:
        print(f"running {c.key} ({c.title}) ...", flush=True)
        res = run_case(c, ctx)
        results.append(res)
        buf = io.StringIO()
        print_table([res], buf)
        print(buf.getvalue(), end="", flush=True)
    meta = environment(args.device)
    meta.update(date=time.strftime("%Y-%m-%d %H:%M"), mode="quick" if quick else "all",
                wall_minutes=(time.perf_counter() - t0) / 60.0, argv=public_argv(argv or sys.argv[1:]))
    if source_before is not None:
        after = meta.get("source") or {}
        if after.get("sha256") != source_before["sha256"]:
            # someone edited the measured code while the cases ran: the record says which code
            # the run started on, and the docs refuse to present it as one version
            meta["source"] = source_before
            meta["source_changed_during_run"] = True
            meta["source_mixed"] = [source_before["sha256"], str(after.get("sha256") or "none")]
            print("warning: the measured code changed during the run; the record is marked mixed", file=sys.stderr)
    record = {"meta": meta, "results": [r.to_dict() for r in results]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # the name sorts in time order; two runs in one millisecond still keep both records
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    target = out_dir / f"validation-{meta['mode']}-{stamp}.json"
    k = 2
    while target.exists():
        target = out_dir / f"validation-{meta['mode']}-{stamp}-{k}.json"
        k += 1
    # anything a case put in its data that JSON does not know is written as text: a record is
    # never lost at the end of a long run over one odd value
    target.write_text(json.dumps(_jsonable(record), indent=1, default=str), encoding="utf-8")
    print(f"record: {target}")
    if args.write_docs:
        write_docs(markdown(results, meta, figures=write_figures(results, Path(args.img))), Path(args.docs))
        print(f"record kept: {write_record(results, meta, Path(args.record))}")
    if args.json:
        print(json.dumps(_jsonable(record), indent=1, default=str))
    n_fail = sum(r.status in ("fail", "error") for r in results)
    n_unav = sum(r.status == "unavailable" for r in results)
    print(f"\n{len(results)} cases: {sum(r.status == 'pass' for r in results)} passed, {n_fail} failed or errored, "
          f"{n_unav} unavailable; {meta['wall_minutes']:.1f} min.")
    return 1 if n_fail or (args.strict and n_unav) else 0


if __name__ == "__main__":
    raise SystemExit(main())
