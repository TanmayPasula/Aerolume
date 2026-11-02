"""The assistant's tools: JSON-schema functions over the scripting API (``aerolume.api``).

Every tool is a thin, validated wrapper around :mod:`aerolume.api` (DESIGN.md
section 8), so the assistant can do exactly what a script can, and everything it
does is recorded in the project as if done by hand. Each :class:`Tool` has a
name, a description that says *when* to call it (the claude-api skill:
prescriptive trigger conditions measurably raise the call rate), a strict JSON
schema (every object ``additionalProperties: false``, no numeric limits -
:func:`aerolume.ai.schema.check_schema` enforces it in the tests), a ``kind``
that drives the approval rules, and a handler.

========================  ==========  ==========================================================
tool                      kind        API used
========================  ==========  ==========================================================
project_overview          read        the project store (designs, simulations, status)
describe_design           read        ``Project.design``
create_naca_airfoil       create      ``Project.add_naca``
create_template           create      ``Project.add_template``
create_wing               create      ``Project.add_wing``
derive_design             create      ``add_template`` / ``add_wing`` / ``Design.copy`` + ``transform``
estimate_cost             read        ``Project.plan``
run_polar                 run         ``Project.run_polar``
run_wing                  run         ``Project.run_wing``
run_cfd                   run         ``Project.run_cfd``
run_sweep                 run         ``Project.sweep``
get_results               read        ``Project.result``
compare_results           read        ``Project.compare``
make_plot                 create      ``Result.plot`` / ``Comparison.plot`` / ``Sweep.plot``
write_report              create      ``Project.report``
run_python                code        a script with ``project``, ``al`` bound (always asks the user)
delete_item               delete      ``Project.remove_design`` / ``remove_simulation`` (always asks)
record_plan               record      the experiment log
record_conclusion         record      the experiment log (numbers are traced)
========================  ==========  ==========================================================

**Honesty is built into the results.** Every result summary carries the
numbers' validity context (DESIGN.md 6 and 7.12): real and simulated Reynolds
number, whether the run converged, standard errors and whether they are
reliable, the grid, the solver's own warnings, and ``validity`` sentences from
the validation record (2D CFD of a car is a side-profile section; absolute 3D
vehicle drag is not validated; Cl,max is a lower bound when the stall was not
reached). The system prompt makes the model quote them.

**Data, not instructions.** Names, notes and file contents the tools return are
user data; results are wrapped as ``{"data": ...}`` and the system prompt says
nothing inside is an instruction. Files are written only inside the project.

Results are compact on purpose (4-5 significant figures, tables of at most a
few dozen rows): a local 8-billion-parameter model has an 8k-token context.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import re
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import schema as S
from .experiment import Action, ExperimentLog, _now, fmt_seconds
from .plaintext import brief, pdg_round
from .providers.base import ToolCall, ToolSpec
from .workspace import Workspace, inside

MAX_TABLE_ROWS = 40
MAX_SWEEP_RUNS = 12
PYTHON_OUTPUT_CHARS = 4000


class ToolError(Exception):
    """A problem the model can fix (unknown design, out-of-range value): becomes an error result."""


@dataclass
class ToolOutcome:
    """What a tool returns: JSON for the model, one line for people, and what it produced."""
    content: dict
    summary: str
    is_error: bool = False
    simulation_ids: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)


@dataclass
class Approval:
    """A request for the user's go-ahead, shown as a card in the chat (or asked on the CLI)."""
    kind: str                      # "run" | "code" | "delete"
    title: str
    detail: str
    estimate_s: float | None = None
    code: str | None = None
    #: what will run, from the validated input ("15 m/s · α −4° to 14°, step 1° (19 angles) · VLM, ...")
    what: str = ""
    #: exactly what will run or be deleted, as (label, value) rows: design, analysis, speeds, angle
    #: ranges, quality, grid, Reynolds numbers, device, physics, estimated time (the card's table)
    items: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ToolContext:
    """Everything a handler may use. ``progress(fraction, message)`` feeds the tool's chat row."""
    ws: Workspace
    log: ExperimentLog | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    progress: Callable[[float, str], None] | None = None
    before_gpu_run: Callable[[], None] | None = None
    plot_counter: list[int] = field(default_factory=lambda: [0])
    #: a small local model is calling: results are sent in their compact form (see :func:`compact_summary`)
    compact: bool = False
    #: the plan the last run estimate made, so the approval card can say what will run without planning twice
    last_plan: Any = None
    #: the request asked for a new run ("run a new draft simulation"): run_simulation and run_cfd run again
    #: instead of returning an identical stored result, once per simulation; the set holds what already ran.
    #: None: identical stored results are reused (and said to be)
    fresh: set[str] | None = None

    def wants_fresh(self, key: str) -> bool:
        """Whether ``key`` (a simulation id, or a run_cfd input) must run although a stored result exists."""
        return self.fresh is not None and key not in self.fresh

    @property
    def project(self) -> Any:
        """The :class:`aerolume.api.Project`."""
        return self.ws.project

    @property
    def store(self) -> Any:
        return self.ws.store

    def report(self, fraction: float, message: str) -> None:
        if self.progress is not None:
            with contextlib.suppress(Exception):     # a repaint problem must not stop a solver
                self.progress(float(fraction), str(message))

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise ToolError("Stopped by the user.")


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[[ToolContext, dict], ToolOutcome]
    kind: str = "read"                  # "read" | "create" | "run" | "code" | "delete" | "record" | "suggest"
    local: bool = True                  # offered to small local models
    short: str | None = None            # shorter description for local models
    estimate: Callable[[ToolContext, dict], float] | None = None
    #: the schema a small local model gets (None: the full one); run_cfd drops its large physics object,
    #: which would take ~700 of qwen3:8b's 8k tokens in every request
    local_schema: dict | None = None

    def spec(self, compact: bool = False) -> ToolSpec:
        if compact:
            return ToolSpec(self.name, self.short or self.description, self.local_schema or self.schema)
        return ToolSpec(self.name, self.description, self.schema)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(x: Any, digits: int = 5) -> float | None:
    """A float rounded to ``digits`` significant figures for compact JSON (None for nan/inf)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v == 0:
        return 0.0
    return float(f"{v:.{digits}g}")


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": list(required or []), "additionalProperties": False}


def _str(desc: str, enum: list[str] | None = None) -> dict:
    d: dict = {"type": "string", "description": desc}
    if enum:
        d["enum"] = enum
    return d


def _number(desc: str) -> dict:
    return {"type": "number", "description": desc}


def _bool(desc: str) -> dict:
    return {"type": "boolean", "description": desc}


def _ids(desc: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": desc}


def _short_list(items: list[str], n: int = 4, chars: int = 240) -> list[str]:
    out = []
    for s in list(items)[:n]:
        s = str(s).strip()
        out.append(s if len(s) <= chars else s[: chars - 1] + "…")
    return out


def _range(name: str, v: Any, lo: float, hi: float, unit: str = "") -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a number, got {v!r}.") from None
    if not (math.isfinite(v) and lo <= v <= hi):
        raise ToolError(f"{name} must be between {lo:g} and {hi:g}{(' ' + unit) if unit else ''}, got {v:g}.")
    return v


def _api():
    from .. import api
    return api


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def re_text(re_: float) -> str:
    """A Reynolds number as people write it: "1.37×10⁶" (plain digits below 10,000)."""
    if not (isinstance(re_, (int, float)) and math.isfinite(re_) and re_ > 0):
        return "n/a"
    if re_ < 1e4:
        return f"{re_:,.0f}"
    e = math.floor(math.log10(re_))
    return f"{re_ / 10 ** e:.3g}×10{str(e).translate(_SUP)}"


def _near_ids(ref: str, items: dict) -> str:
    """" Did you mean the id 485fa5563c6c (“Tapered wing”)?" for a mistyped id, else "".

    qwen3:8b dropped one digit of a 12-digit design id in three of four live runs;
    the error named the designs only, so every slip cost a model turn. The id is
    suggested, never used: a call with a wrong id is still refused.
    """
    import difflib
    close = difflib.get_close_matches(ref, list(items), n=2, cutoff=0.8)
    if not close:
        return ""
    return " Did you mean " + " or ".join(f"the id {i} (“{items[i].name}”)" for i in close) + "?"


def find_design(ctx: ToolContext, ref: Any):
    """The model design for an id, a name (case-insensitive) or a unique id prefix."""
    api = _api()
    key = str(ref or "").strip()
    try:
        return ctx.project.design(key).raw
    except api.AerolumeError as exc:
        raise ToolError(str(exc) + _near_ids(key, ctx.store.designs)) from None


def find_simulation(ctx: ToolContext, ref: Any):
    api = _api()
    key = str(ref or "").strip()
    try:
        return ctx.project.simulation(key)
    except api.AerolumeError as exc:
        raise ToolError(str(exc) + _near_ids(key, ctx.store.simulations)) from None


def _design_word(d) -> str:
    if d.kind == "wing":
        return "wing (3D planform)"
    if int(getattr(d, "dimension", 2) or 2) == 3:
        return f"{d.kind} (3D mesh)"
    return f"{d.kind} (2D outline)"


def round_to_stderr(mean: Any, stderr: Any) -> tuple[float | None, float | None]:
    """A mean and its standard error as a result table shows them: the error rounded by the Particle Data
    Group rule (:func:`aerolume.ai.plaintext.pdg_round`, the Results tables' rule) and the mean to the same
    decimal place (0.51377 ± 0.004553 -> 0.514, 0.005; 3.34079 ± 0.00095 -> 3.3408, 0.001).

    The model is sent what the Results page shows. Given 0.51377 ± 0.00455 a
    local model wrote "Cd is 0.51377 ± 0.00455 ... I am confident within
    ±0.00455": digits beyond the error are noise, and quoting them invites
    reading the averaging error as accuracy. Two significant figures everywhere
    used to send 0.00095 where the table beside the answer showed 0.0010
    (critic round 3.4). Without a finite error the mean keeps 5 significant
    figures.
    """
    m = _num(mean)
    try:
        se = float(stderr)
    except (TypeError, ValueError):
        return m, None
    if m is None or not (math.isfinite(se) and se > 0):
        return m, (_num(se, 2) if math.isfinite(se) else None)
    rounded, _digits, decimals = pdg_round(se)
    return round(float(mean), decimals), round(rounded, decimals + 2)


#: A standard error below this share of the mean is round-off of a settled value, not an uncertainty: the
#: Results page's rule (:data:`aerolume.core.format.STEADY_REL`), so the run card, the model's data and the
#: Summary table say the same thing about one number.
from ..core.format import STEADY_REL  # noqa: F401 - re-exported (tests and the panel read it here)

#: What a steady quantity's uncertainty is, sent with it: the model is told where the error of the value
#: really comes from instead of being handed 4.4e-16 to quote as "± 0.00"
STEADY_NOTE = ("steady: the averaging error is round-off (below 1e-5 of the value), not an uncertainty; never "
               "write ± for it. Its error is the discretisation error: run_grid_study (GCI), or say it needs one")


def is_steady(mean: Any, stderr: Any) -> bool:
    """True when a time average's standard error is below :data:`STEADY_REL` of its (non-zero) mean.

    :func:`aerolume.core.format.is_steady`, the Results page's test. A converged laminar
    cavity's Nusselt number has a batch-means error of 4.4×10⁻¹⁶: the run card printed
    "Nu 2.251 ± 4.4e-16" and the model answered "2.25 ± 0.00 (4.4e-16)" (critic round 3.3),
    quoting float round-off as the uncertainty.
    """
    from ..core.format import is_steady as steady
    return steady(mean, stderr)


def _stat(st) -> dict:
    """A statistic as the model gets it: mean and standard error rounded alike, or the steady value.

    A steady value has four significant figures and ``steady: true`` instead of
    an error (the Results page's "steady"); the summary's ``stderr_meaning`` says
    what that means.
    """
    raw_mean, raw_se = getattr(st, "mean", float("nan")), getattr(st, "stderr", float("nan"))
    if is_steady(raw_mean, raw_se):
        return {"mean": _num(raw_mean, 4), "steady": True}
    mean, se = round_to_stderr(raw_mean, raw_se)
    d: dict[str, Any] = {"mean": mean}
    if se is not None:
        d["stderr"] = se
        if not getattr(st, "stderr_reliable", True):
            d["stderr_reliable"] = False
    return d


# ---------------------------------------------------------------------------
# validity and result summaries (shared by run_*, get_results, compare)
# ---------------------------------------------------------------------------

POLAR_VALIDITY = ("Viscous-inviscid (XFOIL-class) airfoil solver at the real Reynolds number: reliable in attached "
                  "flow; values near and past stall are indicative.")
WING_VALIDITY = ("Vortex-lattice / lifting-line with viscous strip drag from section polars: lift slope, induced "
                 "drag and span efficiency are validated within 2-2.5 % (docs/validation_wing.md). Profile drag, "
                 "and so CD and L/D, comes from 2D section polars at the strip Reynolds numbers and is not "
                 "validated to that accuracy; stall is a critical-section estimate, not a CFD stall prediction.")

#: What each deterministic analysis is validated for, given to the model as data. A small local model does not
#: derive these from prose: asked how confident a best-L/D answer was, qwen3:8b called the wing result
#: "validated within a few percent" although only the lift slope and span efficiency are (measured).
VALIDATED = {
    "polar": (["cl", "cd", "cm", "L/D in attached flow"], ["Cl_max", "values near and past stall"]),
    "wing": (["CL_alpha", "e (span efficiency)", "CDi"], ["profile drag", "CD", "L/D", "CL_max",
                                                         "oswald_e_parabolic_fit"]),
}

#: Wing metric names as the model sees them. The analysis names are short ("e_cruise", "oswald"); a local
#: model called e_cruise 0.98991 "the Oswald factor" although that is the span efficiency at the best-L/D
#: point and the Oswald factor of the parabolic drag fit was 0.761 (measured). ``e_cruise`` is renamed
#: after the point it was read at (see :func:`summarize_result`).
WING_METRIC_NAMES = {"oswald": "oswald_e_parabolic_fit", "cd0": "CD0_parabolic_fit"}
CFD2D_VEHICLE_VALIDITY = ("2D CFD of a vehicle side profile: a section through the vehicle per metre of span, not "
                          "its 3D drag; use it to compare shapes, not to quote a car's Cd.")
CFD3D_VEHICLE_VALIDITY = ("Absolute 3D vehicle drag is not validated: the wall-unresolved LES reads an Ahmed-type "
                          "body about 2.1-2.2 times its wind-tunnel drag, and differences between designs are not "
                          "validated either.")
CFD3D_BLUFF_VALIDITY = ("3D Navier-Stokes validated on spheres at Re 20-300 (Cd within a few percent); at higher Re "
                        "the immersed walls are not resolved and results are indicative.")


#: Short forms of the validity sentences for a local model's compact results: its system prompt already
#: carries the validity record as prose, and the sentences repeated in every result filled qwen3:8b's 8k
#: context (8,025 of 8,192 tokens after two reads of the same runs).
_VALIDITY_CODES = (
    (POLAR_VALIDITY, "attached flow only; near/past stall indicative"),
    (WING_VALIDITY, "CD, L/D and stall not validated; CL slope, e, CDi validated"),
    (CFD2D_VEHICLE_VALIDITY, "2D section per metre, not the car's Cd"),
    (CFD3D_VEHICLE_VALIDITY, "3D vehicle drag not validated (reads ~2.1-2.2x)"),
    (CFD3D_BLUFF_VALIDITY, "validated only for spheres at Re 20-300"),
    ("Simulated at Re", "Re capped below the real Re"),
    ("Not converged", "not converged"),
    ("Cl_max is a lower bound", "Cl_max lower bound"),
    ("CL_max is a lower bound", "CL_max lower bound"),
    ("CL at ", "CL optimistic near stall (strips beyond section cl_max)"),
)


def validity_codes(sentences: list[str]) -> list[str]:
    """The short forms (:data:`_VALIDITY_CODES`) of a result's validity sentences."""
    out = []
    for s in sentences or []:
        code = next((c for prefix, c in _VALIDITY_CODES if str(s).startswith(prefix)), None)
        code = code or brief(str(s), 70)
        if code not in out:
            out.append(code)
    return out


def cfd_confidence(design, result) -> tuple[list[str], list[str]]:
    """(validated, not validated) quantities of a CFD result, as data (see :data:`VALIDATED`)."""
    dim = int(getattr(design, "dimension", 2) or 2) if design is not None else 2
    kind = getattr(design, "kind", "")
    re_s = float(getattr(result, "re_sim", float("nan")))
    if dim == 3 and kind == "vehicle":
        return [], ["absolute Cd", "Cl", "differences between designs"]
    if dim == 3:
        ok = math.isfinite(re_s) and 20 <= re_s <= 300
        return (["Cd (sphere-like bodies, Re 20-300)"] if ok else []), ([] if ok else ["Cd above Re 300"])
    if kind == "vehicle":
        return [], ["the vehicle's 3D Cd (2D section)"]
    re_r = float(getattr(result, "re_real", float("nan")))
    if math.isfinite(re_s) and math.isfinite(re_r) and re_s < 0.98 * re_r:
        return [], ["coefficients at the real Re (simulated at a capped Re)"]
    return [], []


def stderr_meaning(result, design) -> str:
    """What a CFD standard error covers, sent with every CFD result (it is not the accuracy)."""
    re_s = float(getattr(result, "re_sim", float("nan")))
    three = int(getattr(design, "dimension", 2) or 2) == 3 if design is not None else False
    model = "wall-unresolved LES, no wall model" if three else "uniform-grid Navier-Stokes"
    re_part = ""
    if math.isfinite(re_s) and re_s > 0:
        e = math.floor(math.log10(re_s))
        re_part = f", Re {re_s / 10 ** e:.2g}e{e}" if re_s >= 1e4 else f", Re {re_s:.0f}"
    return ("statistical averaging uncertainty of this run only; modelling error "
            f"({model}{re_part}) not included")


def design_validity(design) -> list[str]:
    """Validity sentences that follow from the kind of design alone."""
    dim = int(getattr(design, "dimension", 2) or 2) if design is not None else 2
    kind = getattr(design, "kind", "")
    if dim == 2 and kind == "vehicle":
        return [CFD2D_VEHICLE_VALIDITY]
    if dim == 3 and kind == "vehicle":
        return [CFD3D_VEHICLE_VALIDITY]
    if dim == 3 and kind == "generic":
        return [CFD3D_BLUFF_VALIDITY]
    return []


def cfd_validity(result, design) -> list[str]:
    out = []
    re_r, re_s = float(getattr(result, "re_real", float("nan"))), float(getattr(result, "re_sim", float("nan")))
    if math.isfinite(re_r) and math.isfinite(re_s) and re_s < 0.98 * re_r:
        out.append(f"Simulated at Re {re_s:.3g} instead of the real {re_r:.3g} (Reynolds cap): the coefficients "
                   "describe the lower-Re flow. For an airfoil at flight Re use run_polar.")
    if not getattr(result, "converged", False):
        out.append("Not converged: the uncertainty is larger than the planner's target (see warnings).")
    return out + design_validity(design)


def _alpha_step(alphas) -> float | None:
    """The angle step of a sweep (the smallest spacing of its distinct angles), or None for one angle."""
    vals = sorted({round(float(a), 6) for a in alphas if math.isfinite(float(a))})
    gaps = [b - a for a, b in itertools.pairwise(vals) if b - a > 1e-6]
    return min(gaps) if gaps else None


_BEYOND = re.compile(r"beyond cl_max\)?\s+at alpha\s*=\s*([-−\d.,\s]+?)\s*deg", re.IGNORECASE)


def beyond_section_max(warnings: list[str]) -> list[float]:
    """Angles at which a lattice's strips asked for more lift than their section polar gives (from its warning)."""
    for w in warnings or []:
        m = _BEYOND.search(str(w))
        if m:
            out = []
            for tok in m.group(1).replace("−", "-").split(","):
                with contextlib.suppress(ValueError):
                    out.append(float(tok.strip()))
            return out
    return []


def alpha_span(values: list[float]) -> str:
    """"13–15°" for a run of angles, "13°" for one."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    return f"{lo:g}°" if lo == hi else f"{lo:g}–{hi:g}°"


def _confidence(summary: dict, analysis: str, step: float | None) -> None:
    """Add the confidence facts of a deterministic sweep to its summary (see :data:`VALIDATED`)."""
    ok, not_ok = VALIDATED[analysis]
    summary["validated_quantities"] = list(ok)
    summary["not_validated"] = list(not_ok)
    parts = ["Deterministic: no statistical error bar."]
    if step:
        summary["optimum_resolution_deg"] = _num(step / 2.0, 3)
        parts.append(f"An optimum angle (best L/D, Cl_max) is known to ±{step / 2.0:g}° (half the {step:g}° step).")
    parts.append(f"Validated: {', '.join(ok)}. Not validated: {', '.join(not_ok)}.")
    summary["confidence_note"] = " ".join(parts)


def summarize_result(result, sim=None, design=None, *, table: bool = True) -> dict:
    """Compact JSON of a model result with its validity context (see the module docstring)."""
    from ..model import CFD3DResult, CFDResult, PolarResult, WingResult
    out: dict[str, Any] = {"simulation_id": getattr(result, "sim_id", "") or getattr(sim, "id", "")}
    if design is not None:
        # named in every result: a model must see which geometry a number belongs to (a small model once
        # reported a NACA 2412 polar as NACA 4412)
        out["design"] = design.name
    if sim is not None:
        out["name"] = sim.name
        if sim.status == "cancelled":
            out["stopped_early"] = True
    if isinstance(result, PolarResult):
        from ..analysis.polar import polar_metrics
        tab = result.table()
        m = polar_metrics(result)
        out.update(analysis="polar", reynolds=_num(result.re, 4), mach=_num(result.mach), ncrit=_num(result.ncrit),
                   points=len(result.points), converged_points=len(tab["alpha"]))
        out["metrics"] = {k: _num(m[k]) for k in ("cl_max", "alpha_cl_max", "cd_min", "alpha_cd_min", "ld_max",
                                                   "alpha_ld_max", "cl_at_ld_max", "lift_slope_per_deg",
                                                   "alpha_zero_lift", "cm0") if _num(m.get(k)) is not None}
        if m.get("cl_max_reached") == 0.0:
            out["metrics"]["cl_max_is_lower_bound"] = True
        if table and len(tab["alpha"]):
            rows = list(zip(tab["alpha"], tab["cl"], tab["cd"], tab["cm"], strict=False))
            step = max(1, math.ceil(len(rows) / MAX_TABLE_ROWS))
            out["table_columns"] = ["alpha_deg", "cl", "cd", "cm", "l_over_d"]
            out["table"] = [[_num(a, 4), _num(cl, 4), _num(cd, 4), _num(cm, 4),
                             _num(cl / cd, 4) if cd and cd > 0 else None] for a, cl, cd, cm in rows[::step]]
        failed = [p.alpha for p in result.points if not p.converged]
        if failed:
            out["not_converged_alpha_deg"] = [_num(a, 4) for a in failed[:20]]
        out["validity"] = [POLAR_VALIDITY] + (["Cl_max is a lower bound: lift was still rising at the last angle."]
                                              if m.get("cl_max_reached") == 0.0 else [])
        _confidence(out, "polar", _alpha_step([p.alpha for p in result.points]))
    elif isinstance(result, WingResult):
        from ..analysis.wing import wing_metrics
        tab = result.table()
        m = wing_metrics(result)
        out.update(analysis="wing", method=result.method, viscous=bool(result.viscous), area_m2=_num(result.area),
                   span_m=_num(result.span), aspect_ratio=_num(result.aspect_ratio), mac_m=_num(result.mac),
                   reynolds_mac=_num(result.re_mac, 4), points=len(result.points), converged_points=len(tab["alpha"]))
        cruise = "span_efficiency_at_best_LD" if result.viscous else "span_efficiency_at_max_attached_CL"
        out["metrics"] = {WING_METRIC_NAMES.get(k, cruise if k == "e_cruise" else k): _num(m[k])
                          for k in ("cl_alpha_per_deg", "alpha_zero_lift", "cl_max", "alpha_cl_max",
                                    "cd_min", "ld_max", "alpha_ld_max", "cl_at_ld_max", "cd0",
                                    "oswald", "e_cruise", "cm0", "cm_alpha_per_deg")
                          if _num(m.get(k)) is not None}
        if m.get("cl_max_reached") == 0.0:
            out["metrics"]["cl_max_is_lower_bound"] = True
        if math.isfinite(result.stall_alpha):
            out["metrics"]["stall_onset_alpha_deg"] = _num(result.stall_alpha)
            out["metrics"]["stall_onset_cl"] = _num(result.stall_cl)
        if table and len(tab["alpha"]):
            rows = list(zip(tab["alpha"], tab["cl"], tab["cd"], tab["cdi"], tab["e"], strict=False))
            step = max(1, math.ceil(len(rows) / MAX_TABLE_ROWS))
            out["table_columns"] = ["alpha_deg", "CL", "CD", "CDi", "span_efficiency", "L_over_D"]
            out["table"] = [[_num(a, 4), _num(cl, 4), _num(cd, 4), _num(cdi, 4), _num(e, 3),
                             _num(cl / cd, 4) if cd and cd > 0 else None] for a, cl, cd, cdi, e in rows[::step]]
        if result.warnings:
            out["warnings"] = _short_list(list(result.warnings), chars=320)
        out["validity"] = [WING_VALIDITY]
        if m.get("cl_max_reached") == 0.0:
            out["validity"].append("CL_max is a lower bound: lift was still rising at the last angle; the stall "
                                   "onset estimate is the better CL_max figure.")
        beyond = beyond_section_max(list(result.warnings))
        if beyond:
            out["cl_beyond_section_max_alpha_deg"] = [_num(a, 4) for a in beyond]
            out["validity"].append(f"CL at {alpha_span(beyond)} is optimistic: strips there are beyond their "
                                   "section cl_max and the lattice's lift does not stall.")
        _confidence(out, "wing", _alpha_step([p.alpha for p in result.points]))
    elif isinstance(result, (CFDResult, CFD3DResult)):
        three = isinstance(result, CFD3DResult)
        out.update(analysis="cfd3d" if three else "cfd2d", converged=bool(result.converged))
        keys = ("cd", "cs", "cl", "c_roll", "cm", "cn") if three else ("cd", "cl", "cm")
        out["coefficients"] = {k: _stat(getattr(result, k)) for k in keys}
        ld = result.lift_to_drag
        if math.isfinite(ld):
            out["l_over_d"] = _num(ld, 4)
        if three:
            out["forces_N"] = {"drag": _num(result.drag), "side": _num(result.side), "lift": _num(result.lift)}
            out["cells"] = int(result.cells)
            out["symmetry_plane"] = bool(result.symmetry_plane)
        else:
            out["forces_N_per_m"] = {"drag": _num(result.drag), "lift": _num(result.lift)}
        out.update(reynolds_real=_num(result.re_real, 4), reynolds_simulated=_num(result.re_sim, 4),
                   grid=list(result.grid), cells_per_ref=_num(result.cells_per_ref, 3),
                   turbulence=result.turbulence, wall_seconds=_num(result.wall_seconds, 3), device=result.device)
        if result.strouhal is not None and math.isfinite(result.strouhal):
            out["strouhal"] = _num(result.strouhal, 4)
        _physics_summary(out, result, sim, three)
        models = physics_models(sim, result)
        if models and (models["heat_transfer"] or out.get("physics")):
            out["physics_models"] = models
        ref = reference_summary(result, sim, design)
        if ref is not None:
            out["reference_check"] = ref
        if result.warnings:
            out["warnings"] = _short_list(list(result.warnings))
        out["validity"] = cfd_validity(result, design) + physics_validity(sim)
        out["stderr_meaning"] = stderr_meaning(result, design)
        steady = [k for k, st in out["coefficients"].items() if st.get("steady")]
        if isinstance(out.get("nusselt"), dict) and out["nusselt"].get("steady"):
            steady.insert(0, "nusselt")
        if steady:
            out["steady_quantities"] = steady
            out["steady_meaning"] = STEADY_NOTE
        ok, not_ok = cfd_confidence(design, result)
        ok = list(ok) + physics_validated(ref)
        not_ok = list(not_ok) + physics_not_validated(sim, result, ref)
        out["validated_quantities"] = ok
        if not_ok:
            out["not_validated"] = not_ok
    else:
        out["analysis"] = type(result).__name__
    if sim is not None:
        c = sim.conditions
        cond: dict[str, Any] = {"speed_m_s": _num(c.speed)}
        if sim.analysis == "cfd":
            cond["alpha_deg"] = _num(c.alpha_deg)
            cond["quality"] = sim.cfd.quality
            if sim.ground.enabled:
                cond["road"] = "moving" if sim.ground.moving else "fixed"
        if getattr(c, "yaw_deg", 0.0):
            cond["yaw_deg"] = _num(c.yaw_deg)
        out["conditions"] = cond
    return out


#: What the placed physics are validated for. Kept conservative: a quantity moves to "validated" only
#: with evidence in docs/validation.md (DESIGN.md 10.2 / 10.3 list the benchmarks being run).
PHYSICS_VALIDITY = {
    "thermal": ("Heat transfer with buoyancy (Boussinesq approximation: density varies only in the gravity term) on the "
                "immersed-boundary grid; benchmarks (de Vahl Davis cavity, heated cylinder, Rayleigh-Benard onset) are "
                "in docs/validation.md; quote Nu with its standard error and the grid, and run a grid study before "
                "trusting its absolute value."),
    # the same without buoyancy: the sentence used to name the Boussinesq model for every thermal run, and
    # qwen3:8b repeated "the assumptions made in the Boussinesq energy equation" for a forced-convection
    # cylinder that ran with buoyancy off (critic round 3)
    "thermal_forced": ("Heat transfer, buoyancy off (forced convection: temperature is carried by the flow and does "
                       "not act on it) on the immersed-boundary grid; benchmarks (heated cylinder against Lange et al. "
                       "1998) are in docs/validation.md; quote Nu with its standard error and the grid, and run a grid "
                       "study before trusting its absolute value."),
    "fans": ("Fans are actuator discs (momentum sources), not resolved blades, validated against confined momentum "
             "theory in a channel; in thrust mode the delivered thrust is imposed (a consistency check), and in a "
             "closed room Froude momentum theory does not apply (the fan recirculates its own wake); in a 2D closed "
             "room the speeds, disc velocity and volume flow are qualitative and grid-dependent (mean speed 2.4 to "
             "3.4 m/s across 6, 12 and 24 cells per fan width for a 1 N/m fan in a 4 x 3 m room; theory_ns2d.md "
             "§18.4), so never quote them as results and use 3D for rooms; blade-level flow and noise are not "
             "modelled."),
    "ports": "Jets and vents are penalised velocity regions, not resolved nozzles.",
    "emitters": "Smoke and tracers are visualisation only: they do not change the flow.",
    "closed_domain": ("Closed domain: there is no free stream; speed_m_s is the reference speed of the flow driver "
                      "(free-fall velocity, fan or jet speed), and force coefficients use it."),
}


_VALIDITY_CODES += (
    (PHYSICS_VALIDITY["thermal"], "heat transfer with Boussinesq buoyancy; Nu needs a grid study"),
    (PHYSICS_VALIDITY["thermal_forced"], "heat transfer, buoyancy OFF (forced convection); Nu needs a grid study"),
    (PHYSICS_VALIDITY["fans"], ("actuator-disc fans; thrust mode imposes the thrust; no momentum theory in closed "
                                "rooms; 2D room speeds are qualitative")),
)


def physics_validity(sim) -> list[str]:
    """Validity sentences of the placed physics a simulation used (DESIGN.md 10)."""
    if sim is None or getattr(sim, "analysis", "") != "cfd":
        return []
    from ..api import physics_features
    out = []
    for f in physics_features(sim):
        if f == "thermal" and not getattr(sim.thermal, "buoyancy", True):
            f = "thermal_forced"
        if f in PHYSICS_VALIDITY:
            out.append(PHYSICS_VALIDITY[f])
    return out


def physics_models(sim, result=None) -> dict[str, Any]:
    """The physical models a CFD run used, as data: heat transfer, buoyancy, turbulence treatment, domain.

    Sent with every CFD result so a model never has to guess them from a validity
    sentence, and read by the claims checker (a statement about the Boussinesq
    buoyancy of a run that had buoyancy off is contradicted).
    """
    if sim is None or getattr(sim, "analysis", "") != "cfd":
        return {}
    th = sim.thermal
    out: dict[str, Any] = {"heat_transfer": bool(th.enabled), "buoyancy": bool(th.enabled and th.buoyancy),
                           "domain": sim.cfd.domain_mode}
    turb = str(getattr(result, "turbulence", "") or "")
    if turb:
        out["turbulence"] = turb
    return out


def _validated_reference(ref: dict | None) -> bool:
    """Whether a :func:`reference_summary` is a validation of this result's own set-up (not a consistency check)."""
    return bool(ref and ref.get("applies") and ref.get("kind", "validation") == "validation"
                and isinstance(ref.get("difference_pct"), (int, float)))


def physics_validated(ref: dict | None) -> list[str]:
    """The reference quantity of a validated set-up, with this run's measured deviation from the reference.

    The de Vahl Davis cavity and the Lange et al. cylinder are validated set-ups
    (docs/validation.md: Nu within +0.1 to +0.6 % for the cavity); a run that
    reproduces one is evidence about its own Nu, and the entry says by how much
    this run differs, so "validated" always comes with the measured number.
    """
    if not _validated_reference(ref):
        return []
    name = {"nusselt": "Nusselt number"}.get(str(ref["quantity"]), str(ref["quantity"]))
    if not all(isinstance(ref.get(k), (int, float)) for k in ("measured", "reference")):
        return []
    source = ref.get("source") or "published reference"
    band = ""
    if isinstance(ref.get("band_pct"), (int, float)):
        band = f"; {'inside' if ref.get('within_band') else 'OUTSIDE'} its ± {ref['band_pct']:g} % band"
    return [(f"{name}: validated set-up (docs/validation.md); this run {ref['measured']:.5g} against "
             f"{ref['reference']:.5g}, {ref['difference_pct']:+.2g} % ({source}){band}")]


def physics_not_validated(sim, result, ref: dict | None = None) -> list[str]:
    """Physics quantities of a result that are not validated (as data, like the CFD confidence lists).

    ``ref`` is the result's :func:`reference_summary`. A run that reproduces a
    validated reference set-up (the de Vahl Davis cavity, the Lange cylinder) has
    its Nusselt number moved to the validated list with the measured deviation
    (:func:`physics_validated`): listing it as "not validated" made the assistant
    write "the Nusselt number's absolute value is not validated" right after
    matching de Vahl Davis to 0.575 % (critic round 3.3). The heat rate stays
    listed: no benchmark checks it directly.
    """
    if sim is None or getattr(sim, "analysis", "") != "cfd":
        return []
    from ..api import physics_features
    used = set(physics_features(sim))
    out = []
    if "thermal" in used:
        if not (_validated_reference(ref) and ref.get("quantity") == "nusselt"):
            out.append("Nusselt number (absolute value)")
        out.append("heat rate")
    if "fans" in used:
        out.append("flow near the fan disc")
    return out


def reference_summary(result, sim, design) -> dict | None:
    """The result against the published reference its set-up reproduces (api.reference_case), as data.

    None when the design is no reference set-up; ``{"applies": False, "why"}``
    when it is one but the settings left the published conditions. Sent inside
    every result it applies to, so the numbers a comparison needs (reference,
    difference in %) are tool results the claims checker can trace, and returned
    by compare_reference.
    """
    if sim is None or design is None:
        return None
    try:
        api = _api()
        case, why = api.reference_case(design, sim, re=getattr(result, "re_sim", None))
        if case is None:
            return {"applies": False, "why": why} if why else None
        chk = case.check(api.Result(None, sim, result))
    except Exception:  # noqa: BLE001 - a reference that cannot be rebuilt is simply not offered
        return None
    out: dict[str, Any] = {"applies": True, "kind": chk.get("kind", "validation"), "quantity": chk["quantity"],
                           "measured": _num(chk["measured"])}
    steady = is_steady(chk["measured"], chk["stderr"])
    if steady:
        # round-off is no error bar: "within 2 standard errors" of 4.4e-16 says nothing, so neither is sent
        out["steady"] = True
    else:
        out["stderr"] = _num(chk["stderr"], 3)
    out.update(reference=_num(chk["reference"]), difference_pct=_num(chk["difference_pct"], 3))
    if not steady:
        out["reference_within_2_stderr"] = bool(chk.get("within_stderr"))
    if isinstance(chk.get("band_pct"), (int, float)):
        # the stated band decides whether "validated" may be said (claims.py, _overclaim_problems)
        out["band_pct"] = _num(chk["band_pct"], 3)
        out["within_band"] = bool(chk.get("within_band"))
        out["band_source"] = str(chk.get("band_source", ""))
    out["source"] = chk.get("source", "")
    if chk.get("note"):
        out["note"] = chk["note"]
    extra = [n for n in case.notes if n != chk.get("note")]
    if extra:
        out["notes"] = extra[:2]
    out["uncertainty_note"] = (STEADY_NOTE if steady else
                               "stderr is the averaging uncertainty only; the discretisation error needs "
                               "run_grid_study (GCI)")
    return out


def _physics_summary(out: dict, result, sim, three: bool) -> None:
    """Heat transfer, fans and diagnostics of a CFD result (only what the run produced)."""
    nu = getattr(result, "nusselt", None)
    if nu is not None and _num(getattr(nu, "mean", None)) is not None:
        out["nusselt"] = _stat(nu)
    hr = _num(getattr(result, "heat_rate", float("nan")))
    if hr is not None:
        out["heat_rate_W" if three else "heat_rate_W_per_m"] = hr
    fans = list(getattr(result, "fans_report", None) or [])
    if fans:
        out["fans"] = [{k: (_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                        for k, v in dict(f).items()} for f in fans[:8]]
    diag = dict(getattr(result, "diagnostics", None) or {})
    if diag:
        out["diagnostics"] = {k: _num(v, 3) for k, v in list(diag.items())[:12]
                              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if sim is not None:
        from ..api import physics_features
        used = physics_features(sim)
        if used:
            out["physics"] = used


def compact_summary(summary: dict, *, table: bool = False) -> dict:
    """A result summary as a small local model reads it: metrics and conditions, validity as short codes.

    Drops the per-angle table (unless ``table``), the confidence prose (the
    validated / not-validated lists and the resolution stay) and long solver
    warnings; the validity sentences become :func:`validity_codes`. A wing
    result shrinks from about 1,900 to about 450 tokens.
    """
    out = {k: v for k, v in summary.items()
           if k not in ("table", "table_columns", "confidence_note", "warnings", "validity")}
    if table and "table" in summary:
        out["table_columns"], out["table"] = summary.get("table_columns"), summary["table"]
    if summary.get("validity"):
        out["validity"] = validity_codes(summary["validity"])
    if summary.get("warnings"):
        out["solver_warnings"] = len(summary["warnings"])
    return out


def summarize_api_result(r) -> dict:
    """:func:`summarize_result` of an :class:`aerolume.api.Result` (plus ``reused``)."""
    design = r.design.raw if r.design is not None else None
    out = summarize_result(r.raw, r.simulation, design)
    out["simulation_id"] = r.sim_id
    if getattr(r, "reused", False):
        out["reused_stored_result"] = True
    return out


def key_numbers(summary: dict) -> list[dict]:
    """Rows for the log's Results table from a result summary."""
    sid = summary.get("simulation_id", "")
    rows = []
    if summary.get("analysis") in ("cfd2d", "cfd3d"):
        for k, st in summary.get("coefficients", {}).items():
            if st.get("mean") is None or k in ("c_roll", "cn", "cs"):
                continue
            unc = _uncertainty_text(st)
            if st.get("stderr_reliable") is False and unc:
                unc += " (estimate)"
            rows.append({"label": {"cd": "Cd", "cl": "Cl", "cm": "Cm"}.get(k, k), "value": st["mean"], "unit": "",
                         "uncertainty": unc, "source": sid})
        nu = summary.get("nusselt")
        if isinstance(nu, dict) and nu.get("mean") is not None:
            unc = _uncertainty_text(nu)
            rows.append({"label": "Nusselt number", "value": nu["mean"], "unit": "", "uncertainty": unc,
                         "source": sid})
        for i, fan in enumerate(summary.get("fans") or []):
            if isinstance(fan, dict) and fan.get("thrust") is not None:
                rows.append({"label": f"{fan.get('name') or f'Fan {i + 1}'} thrust", "value": fan["thrust"],
                             "unit": "N" if summary.get("analysis") == "cfd3d" else "N/m", "uncertainty": "",
                             "source": sid})
        if summary.get("reynolds_simulated") is not None:
            rows.append({"label": "Re simulated / real",
                         "value": f"{re_text(summary['reynolds_simulated'])} / {re_text(summary.get('reynolds_real') or 0)}",
                         "unit": "", "uncertainty": "", "source": sid})
    elif summary.get("analysis") in ("polar", "wing"):
        rows = _sweep_key_numbers(summary)
    return rows


def _sweep_key_numbers(summary: dict) -> list[dict]:
    """Key results of a polar or wing sweep, each with the qualifier a reader needs to use it.

    Deterministic sweeps have no standard error, so the "uncertainty" column
    says what limits each number instead: the angle step for an optimum angle,
    "lower bound" (with the solver's own reason) for a CL_max that was still
    rising, and "not validated" for the profile-drag figures of a wing. A bare
    "CL,max 1.411 at 15°" in an experiment log read as a stall result although
    the run had flagged both.
    """
    sid = summary.get("simulation_id", "")
    m = summary.get("metrics", {})
    polar = summary["analysis"] == "polar"
    res = summary.get("optimum_resolution_deg")
    step_txt = f"±{res:g}° (half the α step)" if res else ""
    sampled = f"largest sampled value (α step {2 * res:g}°)" if res else "largest sampled value"
    drag_note = "" if polar else "profile drag not validated"
    lower = bool(m.get("cl_max_is_lower_bound"))
    beyond = summary.get("cl_beyond_section_max_alpha_deg") or []
    cl_lab = "Cl,max" if polar else "CL,max"
    rows: list[dict] = []

    def add(label: str, value: Any, unit: str = "", unc: str = "") -> None:
        if value is not None:
            rows.append({"label": label, "value": value, "unit": unit, "uncertainty": unc, "source": sid})

    add("Best L/D", _num(m.get("ld_max"), 4), "", "; ".join(x for x in (sampled, drag_note) if x))
    add("α at best L/D", _num(m.get("alpha_ld_max"), 4), "deg", step_txt)
    if m.get("cl_max") is not None:
        if lower:
            why = ["still rising at the last α"]
            if beyond:
                why.append(f"strips beyond section cl_max at {alpha_span(beyond)}")
            add(cl_lab, f"≥ {m['cl_max']:.4g}", "", "lower bound: " + "; ".join(why))
            add(f"α at {cl_lab}", f"≥ {m.get('alpha_cl_max'):g}" if m.get("alpha_cl_max") is not None else None,
                "deg", "last α of the sweep")
        else:
            add(cl_lab, _num(m["cl_max"], 4), "", "indicative near stall" if polar else "not validated (stall)")
            add(f"α at {cl_lab}", _num(m.get("alpha_cl_max"), 4), "deg", step_txt)
    if not polar and m.get("stall_onset_alpha_deg") is not None:
        add("α at stall onset", _num(m["stall_onset_alpha_deg"], 4), "deg", "critical-section estimate")
        add("CL at stall onset", _num(m.get("stall_onset_cl"), 4), "", "critical-section estimate")
    add("Cd,min" if polar else "CD,min", _num(m.get("cd_min"), 4), "", drag_note)
    add("Oswald e", _num(m.get("oswald_e_parabolic_fit", m.get("oswald")), 4), "",
        "" if polar else "parabolic fit, includes profile drag")
    return rows


# ---------------------------------------------------------------------------
# read tools
# ---------------------------------------------------------------------------

def _project_overview(ctx: ToolContext, _args: dict) -> ToolOutcome:
    store = ctx.store
    designs = [{"id": d.id, "name": d.name, "type": _design_word(d)} for d in list(store.designs.values())]
    all_sims = list(store.simulations.values())
    sims = []
    for s in all_sims[-MAX_TABLE_ROWS:]:
        row = {"id": s.id, "name": s.name, "design_id": s.design_id, "analysis": s.analysis, "status": s.status,
               "speed_m_s": _num(s.conditions.speed)}
        if s.analysis == "cfd":
            row["alpha_deg"] = _num(s.conditions.alpha_deg)
            row["quality"] = s.cfd.quality
        elif s.analysis in ("polar", "wing"):
            # the sweep's range, so a model can see that a stored polar already answers the goal
            st = s.airfoil if s.analysis == "polar" else s.wing
            row["alpha_range_deg"] = [_num(st.alpha_start), _num(st.alpha_end), _num(st.alpha_step)]
        sims.append(row)
    n_exp = 0
    with contextlib.suppress(Exception):
        from .experiment import list_experiments
        n_exp = len(list_experiments(ctx.ws.root))
    content = {"project": getattr(store, "name", ""), "designs": designs, "simulations": sims,
               "experiments_logged": n_exp}
    if len(all_sims) > MAX_TABLE_ROWS:
        content["note"] = f"Only the last {MAX_TABLE_ROWS} of {len(all_sims)} simulations are listed."
    return ToolOutcome(content, f"{len(designs)} designs, {len(all_sims)} simulations")


def _airfoil_thickness(pts) -> float:
    import numpy as np
    p = np.asarray(pts, float)
    best = 0.0
    for x in np.linspace(p[:, 0].min(), p[:, 0].max(), 101)[1:-1]:
        ys = []
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            if (a[0] - x) * (b[0] - x) <= 0 and a[0] != b[0]:
                t = (x - a[0]) / (b[0] - a[0])
                ys.append(a[1] + t * (b[1] - a[1]))
        if len(ys) >= 2:
            best = max(best, max(ys) - min(ys))
    return best


def _describe_design(ctx: ToolContext, args: dict) -> ToolOutcome:
    d = find_design(ctx, args["design_id"])
    out: dict[str, Any] = {"id": d.id, "name": d.name, "kind": d.kind, "dimension": int(d.dimension),
                           "length_ref_m": _num(d.length_ref)}
    src = d.source if isinstance(d.source, dict) else {}
    if src.get("template"):
        out["template"] = src.get("template")
        params = src.get("params") or {}
        out["template_parameters"] = {k: (_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                                      for k, v in params.items()}
    if d.kind == "wing" and d.wing is not None:
        from ..geometry3d.wing import planform
        pf = planform(d.wing)
        out["planform"] = {k: _num(pf[k]) for k in ("area", "span", "aspect_ratio", "mac", "taper", "sweep_c4_deg",
                                                     "sweep_le_deg", "dihedral_deg", "root_chord", "tip_chord")
                           if k in pf}
        out["sections"] = [{"y_m": _num(s.y), "chord_m": _num(s.chord), "x_le_m": _num(s.x_le),
                            "z_le_m": _num(s.z_le), "twist_deg": _num(s.twist_deg), "airfoil": s.airfoil}
                           for s in d.wing.sections[:12]]
    elif int(d.dimension) == 3:
        x0, x1, y0, y1, z0, z1 = d.bounds3()
        out.update(parts=len(d.meshes), triangles=int(sum(len(m.faces) for m in d.meshes)),
                   size_m={"length": _num(x1 - x0), "width": _num(y1 - y0), "height": _num(z1 - z0)},
                   ref_area_m2=_num(d.ref_area), span_ref_m=_num(d.span_ref), symmetric=bool(d.symmetric),
                   on_road=bool(src.get("on_road", False)))
    else:
        x0, x1, y0, y1 = d.bounds()
        out.update(bodies=len(d.bodies), force_ref_m=_num(d.force_ref),
                   size_m={"length": _num(x1 - x0), "height": _num(y1 - y0)},
                   wheels=sum(1 for b in d.bodies if b.role == "wheel"))
        if d.kind == "airfoil" and d.bodies:
            c = float(d.length_ref) or 1.0
            out["max_thickness_over_chord"] = _num(_airfoil_thickness(d.bodies[0].points) / c, 3)
    with contextlib.suppress(Exception):
        out["analyses"] = list(ctx.project.design(d.id).analyses)
    out["simulation_ids"] = [s.id for s in list(ctx.store.simulations.values()) if s.design_id == d.id][-20:]
    if d.notes:
        out["notes_data"] = d.notes[:400]
    return ToolOutcome(out, f"{d.name}: {_design_word(d)}")


# ---------------------------------------------------------------------------
# create tools
# ---------------------------------------------------------------------------

def _name_arg(args: dict) -> str | None:
    return str(args["name"]).strip()[:60] if args.get("name") else None


def _create_naca(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    code = re.sub(r"[^0-9]", "", str(args["code"]))
    chord = _range("chord_m", args.get("chord_m", 1.0), 0.01, 20.0, "m")
    before = set(ctx.store.designs)
    try:
        d = ctx.project.add_naca(code, chord, name=_name_arg(args))
    except api.AerolumeError as exc:
        raise ToolError(str(exc)) from None
    reused = d.id in before
    return ToolOutcome({"design_id": d.id, "name": d.name, "chord_m": chord, "kind": "airfoil",
                        "reused_existing": reused},
                       f"{'Reused' if reused else 'Created'} {d.name} (chord {chord:g} m)")


def _template_params(items) -> dict:
    return {str(i["name"]): i["value"] for i in (items or [])}


def _template_hint(name: str) -> str:
    with contextlib.suppress(Exception):
        from ..geometry import templates
        info = templates.get_template(name)
        return " Parameters: " + "; ".join(
            f"{s.name} (default {s.default}"
            + (f", {s.min:g} to {s.max:g}" if s.min is not None and s.max is not None else "") + ")"
            for s in info.params)
    return ""


def _create_template(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    name = str(args["template"])
    before = set(ctx.store.designs)
    try:
        d = ctx.project.add_template(name, name=_name_arg(args), **_template_params(args.get("parameters")))
    except api.AerolumeError as exc:
        raise ToolError(f"{exc}{_template_hint(name)}") from None
    reused = d.id in before
    raw = d.raw
    used = {k: (_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in ((raw.source or {}).get("params") or {}).items()}
    return ToolOutcome({"design_id": d.id, "name": raw.name, "kind": raw.kind, "parameters": used,
                        "reused_existing": reused}, f"{'Reused' if reused else 'Created'} {raw.name} ({name} template)")


def _airfoil_arg(text: Any) -> str:
    t = str(text or "2412").strip()
    if t.lower().startswith("design:"):
        return t
    return re.sub(r"(?i)^naca\s*", "", t).strip() or "2412"


def _create_wing(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    span = _range("span_m", args["span_m"], 0.05, 200.0, "m")
    root = _range("root_chord_m", args["root_chord_m"], 0.01, 50.0, "m")
    tip = _range("tip_chord_m", args.get("tip_chord_m", root), 0.001, 50.0, "m")
    before = set(ctx.store.designs)
    try:
        d = ctx.project.add_wing(span, root, tip,
                                 sweep_deg=_range("sweep_deg", args.get("sweep_deg", 0.0), -60.0, 60.0, "deg"),
                                 dihedral_deg=_range("dihedral_deg", args.get("dihedral_deg", 0.0), -30.0, 30.0,
                                                     "deg"),
                                 twist_deg=_range("tip_twist_deg", args.get("tip_twist_deg", 0.0), -15.0, 15.0, "deg"),
                                 airfoil=_airfoil_arg(args.get("airfoil", "2412")), name=_name_arg(args))
    except api.AerolumeError as exc:
        raise ToolError(str(exc)) from None
    reused = d.id in before
    raw = d.raw
    ar = raw.span_ref ** 2 / raw.ref_area if raw.ref_area else float("nan")
    return ToolOutcome({"design_id": d.id, "name": raw.name, "area_m2": _num(raw.ref_area),
                        "span_m": _num(raw.span_ref), "mac_m": _num(raw.length_ref), "aspect_ratio": _num(ar),
                        "reused_existing": reused},
                       f"{'Reused' if reused else 'Created'} wing {raw.name} (span {span:g} m, AR {ar:.3g})")


_WING_KEYS = ("span_m", "root_chord_m", "tip_chord_m", "sweep_deg", "dihedral_deg", "tip_twist_deg", "airfoil")


def _derive_design(ctx: ToolContext, args: dict) -> ToolOutcome:
    """A new design from an existing one; the base is never modified."""
    api = _api()
    base = find_design(ctx, args["base_design_id"])
    name = str(args.get("name") or f"{base.name} variant").strip()[:60]
    src = base.source if isinstance(base.source, dict) else {}
    tparams = _template_params(args.get("template_parameters"))
    changes: list[str] = []
    thermal: tuple[str, float] | None = None
    if args.get("body_temperature_c") is not None and args.get("body_heat_flux_w_m2") is not None:
        raise ToolError("Give body_temperature_c or body_heat_flux_w_m2, not both.")
    if args.get("body_temperature_c") is not None:
        thermal = ("temperature", _range("body_temperature_c", args["body_temperature_c"], -270.0, 3000.0, "°C"))
    elif args.get("body_heat_flux_w_m2") is not None:
        thermal = ("heat_flux", _range("body_heat_flux_w_m2", args["body_heat_flux_w_m2"], -1e7, 1e7, "W/m²"))
    try:
        if tparams:
            if not src.get("template"):
                raise ToolError(f"{base.name} was not made from a template, so template_parameters do not apply.")
            params = dict(src.get("params") or {})
            params.update(tparams)
            d = ctx.project.add_template(src["template"], name=name, **params)
            changes.append(", ".join(f"{k} = {v}" for k, v in tparams.items()))
        elif base.kind == "wing" and any(k in args for k in _WING_KEYS):
            from ..geometry3d.wing import planform
            secs = base.wing.sections
            pf = planform(base.wing)
            half = max(float(secs[-1].y), 1e-9)
            cur: dict[str, Any] = {"span_m": pf["span"], "root_chord_m": secs[0].chord, "tip_chord_m": secs[-1].chord,
                                   "sweep_deg": pf.get("sweep_c4_deg", 0.0),
                                   "dihedral_deg": math.degrees(math.atan2(secs[-1].z_le - secs[0].z_le, half)),
                                   "tip_twist_deg": secs[-1].twist_deg, "airfoil": secs[0].airfoil}
            for k in _WING_KEYS:
                if k in args:
                    cur[k] = args[k]
                    changes.append(f"{k} = {args[k]}")
            if len(secs) > 2:
                changes.append(f"rebuilt as a straight-tapered wing (the base had {len(secs)} sections)")
            d = ctx.project.add_wing(_range("span_m", cur["span_m"], 0.05, 200.0),
                                     _range("root_chord_m", cur["root_chord_m"], 0.01, 50.0),
                                     _range("tip_chord_m", cur["tip_chord_m"], 0.001, 50.0),
                                     sweep_deg=_range("sweep_deg", cur["sweep_deg"], -60.0, 60.0),
                                     dihedral_deg=_range("dihedral_deg", cur["dihedral_deg"], -30.0, 30.0),
                                     twist_deg=_range("tip_twist_deg", cur["tip_twist_deg"], -15.0, 15.0),
                                     airfoil=_airfoil_arg(cur["airfoil"]), name=name)
        else:
            rot = tuple(float(args.get(k, 0.0) or 0.0) for k in ("rotate_x_deg", "rotate_y_deg", "rotate_z_deg"))
            scale = args.get("scale")
            if scale is None and not any(rot):
                if thermal is None:
                    raise ToolError("No change requested: give template_parameters (template designs, e.g. a NACA "
                                    "chord or the Ahmed slant_angle), wing planform values, scale / rotate_*_deg "
                                    "(3D meshes), or body_temperature_c / body_heat_flux_w_m2 (a heated copy).")
                d = ctx.project.design(base.id).copy(name)
            else:
                if int(base.dimension) != 3 or not base.meshes:
                    raise ToolError("scale and rotations apply to 3D mesh designs; 2D designs from the template "
                                    "gallery change through template_parameters (e.g. 'length', 'chord').")
                d = ctx.project.design(base.id).copy(name)
                d.transform(scale=_range("scale", scale, 0.01, 100.0) if scale is not None else None,
                            rotate_deg=rot if any(rot) else None)
                if scale is not None:
                    changes.append(f"scaled by {float(scale):g}")
                if any(rot):
                    changes.append(f"rotated by {list(rot)} deg")
        if thermal is not None:
            d.set_thermal(*thermal)
            changes.append(f"bodies at {thermal[1]:g} °C" if thermal[0] == "temperature"
                           else f"body heat flux {thermal[1]:g} W/m²")
    except api.AerolumeError as exc:
        hint = _template_hint(src["template"]) if tparams and src.get("template") else ""
        raise ToolError(f"{exc}{hint}") from None
    raw = d.raw
    return ToolOutcome({"design_id": raw.id, "name": raw.name, "derived_from": base.id, "changes": changes},
                       f"{raw.name} from {base.name}: {'; '.join(changes)}")


# ---------------------------------------------------------------------------
# simulation settings for the API
# ---------------------------------------------------------------------------

_ROAD = ("auto", "none", "moving", "fixed")
_REYNOLDS = ("auto", "real", "capped")
_QUALITY = ("draft", "standard", "fine")


def _speed_settings(args: dict, *, required: bool = True) -> dict:
    if args.get("speed_m_s") is not None:
        return {"speed": _range("speed_m_s", args["speed_m_s"], 0.01, 400.0, "m/s")}
    if args.get("reynolds") is not None:
        return {"re": _range("reynolds", args["reynolds"], 1.0, 1e9)}
    if required:
        raise ToolError("Give speed_m_s (or reynolds).")
    return {}


def _alpha_range(args: dict, start: float, end: float, step: float) -> tuple[float, float, float]:
    a0 = _range("alpha_start", args.get("alpha_start", start), -30.0, 30.0, "deg")
    a1 = _range("alpha_end", args.get("alpha_end", end), -30.0, 40.0, "deg")
    st = _range("alpha_step", args.get("alpha_step", step), 0.05, 10.0, "deg")
    if a1 < a0:
        a0, a1 = a1, a0
    if (a1 - a0) / st > 120:
        raise ToolError("At most 120 angles per sweep: use a larger alpha_step.")
    return a0, a1, st


def _polar_settings(args: dict) -> dict:
    s = _speed_settings(args, required=False) or {"re": 1.0e6}
    s["alpha"] = _alpha_range(args, -4.0, 14.0, 1.0)
    if args.get("ncrit") is not None:
        s["ncrit"] = _range("ncrit", args["ncrit"], 0.5, 15.0)
    if args.get("mach") is not None:
        s["mach"] = _range("mach", args["mach"], 0.0, 0.7)
    return s


def _wing_settings(args: dict) -> dict:
    s = _speed_settings(args)
    s["alpha"] = _alpha_range(args, -4.0, 12.0, 1.0)
    method = str(args.get("method", "vlm"))
    if method not in ("vlm", "llt"):
        raise ToolError("method must be 'vlm' or 'llt'.")
    s["method"] = method
    s["viscous"] = bool(args.get("viscous", True))
    return s


def _cfd_settings(args: dict, dim: int = 2) -> dict:
    closed = (args.get("physics") or {}).get("domain") == "closed"
    s = _speed_settings(args, required=not closed)    # a closed domain takes its driver's speed
    s["alpha"] = _range("alpha_deg", args.get("alpha_deg", 0.0), -30.0, 30.0, "deg")
    yaw = _range("yaw_deg", args.get("yaw_deg", 0.0), -45.0, 45.0, "deg")
    if yaw:
        s["yaw"] = yaw
    q = str(args.get("quality", "draft"))
    if q not in _QUALITY:
        raise ToolError(f"quality must be one of {', '.join(_QUALITY)}.")
    s["quality"] = q
    road = str(args.get("road", "auto"))
    if road == "none":
        s["ground"] = False
    elif road in ("moving", "fixed"):
        s["ground"], s["moving_ground"] = True, road == "moving"
    rmode = str(args.get("reynolds_mode", "auto"))
    if rmode in ("real", "capped"):
        s["reynolds"] = rmode
    sens = str(args.get("sensitivity", "none"))
    if sens != "none":
        s["sensitivity"] = sens
    if args.get("physics"):
        s.update(physics_settings(args["physics"], dim))
    return s


def _vec(v: Any) -> tuple[float, ...] | None:
    return None if v is None else tuple(float(c) for c in v)


def physics_settings(physics: dict, dim: int = 2) -> dict:
    """The ``physics`` object of a run call as :meth:`aerolume.api.Project.run_cfd` keywords.

    Only the mapping happens here (metres, °C, W, N, m/s as the schema names
    them); every check -- dimensions, directions, sizes, a domain box that holds
    everything -- is the API's, so the assistant and a script are refused for the
    same reasons.
    """
    p = dict(physics or {})
    out: dict[str, Any] = {}
    if p.get("domain"):
        out["domain"] = str(p["domain"])
    if p.get("domain_box_m"):
        out["domain_box"] = tuple(float(c) for c in p["domain_box_m"])
    walls = {}
    for w in p.get("walls") or []:
        mode = str(w.get("mode", "temperature"))
        walls[str(w.get("side", ""))] = "adiabatic" if mode == "adiabatic" else (mode, float(w.get("value", 0.0)))
    heat_kinds = ("rect", "disc") if dim == 2 else ("box", "sphere")
    heaters = [{"name": f"Heater {i + 1}", "kind": str(h.get("kind") or heat_kinds[0]),
                "position": _vec(h.get("position_m")), "size": _vec(h.get("size_m")) or (),
                "power": float(h.get("power_w", 0.0))} for i, h in enumerate(p.get("heat_sources") or [])]
    ports = [{"name": f"{str(j.get('kind', 'jet')).capitalize()} {i + 1}", "kind": str(j.get("kind", "jet")),
              "position": _vec(j.get("position_m")), "direction": _vec(j.get("direction")) or (1.0, 0.0, 0.0),
              "size": float(j.get("size_m", 0.05)), "speed": float(j.get("speed_m_s", 0.0)),
              "temperature": j.get("temperature_c"), "dye": bool(j.get("smoke", False))}
             for i, j in enumerate(p.get("ports") or [])]
    wants_heat = bool(walls or heaters or any(j["temperature"] is not None for j in ports))
    heat = p.get("heat_transfer")
    if heat is False and not wants_heat:
        out["thermal"] = False
    elif heat or wants_heat or p.get("ambient_temperature_c") is not None:
        th: dict[str, Any] = {"buoyancy": bool(p.get("buoyancy", True))}
        if p.get("ambient_temperature_c") is not None:
            th["t_ambient"] = float(p["ambient_temperature_c"])
        out["thermal"] = th
    if walls:
        out["walls"] = walls
    if heaters:
        out["heat_sources"] = heaters
    if ports:
        out["ports"] = ports
    fans = [{"name": f"Fan {i + 1}", "position": _vec(f.get("position_m")),
             "direction": _vec(f.get("direction")) or (1.0, 0.0, 0.0), "diameter": float(f.get("diameter_m", 0.3)),
             "mode": str(f.get("mode", "thrust")), "value": float(f.get("value", 0.0)),
             "swirl": float(f.get("swirl", 0.0))} for i, f in enumerate(p.get("fans") or [])]
    if fans:
        out["fans"] = fans
    emitters = []
    for i, e in enumerate(p.get("emitters") or []):
        item: dict[str, Any] = {"name": f"Emitter {i + 1}", "species": str(e.get("species", "dye")),
                                "kind": str(e.get("kind", "point")), "position": _vec(e.get("position_m"))}
        if e.get("end_m") is not None:
            item["end"] = _vec(e["end_m"])
        if e.get("size_m") is not None:
            item["size"] = _vec(e["size_m"])
        if e.get("count") is not None:
            item["count"] = int(e["count"])
        if e.get("color"):
            item["color"] = str(e["color"])
        emitters.append(item)
    if emitters:
        out["emitters"] = emitters
    for k in ("position", "end", "direction"):
        for lst in (heaters, ports, fans, emitters):
            for item in lst:
                if k in item and item[k] is None:
                    raise ToolError(f"Every {k} needs coordinates in metres ({'x, y' if dim == 2 else 'x, y, z'}).")
    return out


def _settings_for(analysis: str, args: dict, dim: int = 2) -> dict:
    if analysis == "cfd":
        return _cfd_settings(args, dim)
    try:
        make = {"polar": _polar_settings, "wing": _wing_settings}[analysis]
    except KeyError:
        raise ToolError("analysis must be 'cfd', 'polar' or 'wing'.") from None
    return make(args)


def _plan(ctx: ToolContext, design_ref: Any, analysis: str, settings: dict):
    """The API's plan of a run; a set-up the planner refuses is the call's own input error.

    Why InputError: ``api.InputError`` means the set-up itself is invalid (heated
    bodies with heat transfer off, a fan outside the box), so the run would be
    refused the same way before anything starts. Asking the user to approve it
    as "cost unknown" (round-4 probe) only led to an approved call that failed;
    the model gets the reason instead and fixes its call. Any other planner
    failure stays a plain ToolError, which an approval treats as an unknown cost
    and asks about (fail closed).
    """
    api = _api()
    try:
        return ctx.project.plan(design_ref, analysis, **settings)
    except api.InputError as exc:
        raise InputError(f"The planner refused this setup: {exc}") from None
    except api.AerolumeError as exc:
        raise ToolError(f"The planner refused this setup: {exc}") from None


class InputError(ToolError):
    """The call's own input is wrong (unknown design, out-of-range value).

    Raised by an estimate before anything is planned. The run handler checks
    the same input first and fails on it without starting anything, so such a
    call needs no approval; any other estimate failure does (see
    :meth:`Toolbox.approval`).
    """


def _checked_inputs(ctx: ToolContext, analysis: str, args: dict) -> tuple[Any, dict]:
    try:
        d = find_design(ctx, args["design_id"])
        return d, _settings_for(analysis, args, int(getattr(d, "dimension", 2) or 2))
    except InputError:
        raise
    except ToolError as exc:
        raise InputError(str(exc)) from None


def _estimate_for(analysis: str) -> Callable[[ToolContext, dict], float]:
    def estimate(ctx: ToolContext, args: dict) -> float:
        d, settings = _checked_inputs(ctx, analysis, args)
        ctx.last_plan = None
        if analysis in _SWEEP_PART and covering_sweep(ctx, d, analysis, settings) is not None:
            return 0.0          # a stored sweep answers it: nothing runs, so nothing to approve
        plan = _plan(ctx, d.id, analysis, settings)
        ctx.last_plan = plan
        return float(plan.est_seconds)
    return estimate


def _deg(v: float) -> str:
    return f"{v:g}°" if v >= 0 else f"−{-v:g}°"


def _alpha_text(alpha: tuple[float, float, float]) -> str:
    a0, a1, st = alpha
    n = math.floor((a1 - a0) / st + 1e-9) + 1
    return f"α {_deg(a0)} to {_deg(a1)}, step {st:g}° ({n} angle{'s' if n != 1 else ''})"


def _cells_text(cells: int) -> str:
    return f"{cells / 1e6:.3g} M cells" if cells >= 1e6 else f"{cells / 1e3:.3g} k cells"


def run_detail(name: str, args: dict, plan: Any = None) -> str:
    """What a run call will compute, from its validated input (the approval card's detail line).

    "15 m/s · α −4° to 14°, step 1° (19 angles) · VLM, viscous strips" for a
    wing; quality, cells, Reynolds numbers and device for CFD. The card used to
    say only "Wing analysis of Tapered wing: about 25 s", so the user approved
    without seeing the speed or the angle range.
    """
    with contextlib.suppress(ToolError, KeyError, TypeError, ValueError):
        analysis = {"run_polar": "polar", "run_wing": "wing", "run_cfd": "cfd"}.get(name)
        sweep: list[str] = []
        if name == "run_sweep":
            analysis = str(args.get("analysis", "cfd"))
            variable = str(args.get("variable", ""))
            unit = {"speed": " m/s", "alpha": "°", "yaw": "°"}.get(variable, "")
            vals = ", ".join(f"{float(v):g}" for v in (args.get("values") or [])[:8])
            more = len(args.get("values") or []) - 8
            sweep = [f"{variable} = {vals}{' …' if more > 0 else ''}{unit}"]
            base = {k: v for k, v in args.items() if k not in ("variable", "values", "analysis", "design_id")}
            base.setdefault({"speed": "speed_m_s", "alpha": "alpha_deg", "yaw": "yaw_deg"}.get(variable, "_"),
                            (args.get("values") or [0])[0])
            args = base
        if analysis is None:
            return ""
        s = _settings_for(analysis, args)
        parts: list[str] = list(sweep)
        if not sweep or "speed" not in sweep[0]:
            if "speed" in s:
                parts.append(f"{s['speed']:g} m/s")
            elif "re" in s:
                parts.append(f"Re {re_text(s['re'])}")
        if analysis in ("polar", "wing"):
            parts.append(_alpha_text(s["alpha"]))
            if analysis == "wing":
                parts.append(("VLM" if s["method"] == "vlm" else "lifting line") + ", "
                             + ("viscous strips" if s["viscous"] else "inviscid"))
            else:
                parts.append(f"viscous polar, ncrit {s.get('ncrit', 9.0):g}")
        else:
            if not sweep or "alpha" not in sweep[0]:
                parts.append(f"α {_deg(s['alpha'])}")
            if s.get("yaw") and (not sweep or "yaw" not in sweep[0]):
                parts.append(f"yaw {_deg(s['yaw'])}")
            parts.append(f"{s['quality']} quality")
            if plan is not None and plan.analysis == "cfd":
                if plan.cells:
                    parts.append(_cells_text(plan.cells))
                re_r, re_s = plan.re_real, plan.re_sim
                if math.isfinite(re_r) and math.isfinite(re_s):
                    parts.append(f"Re {re_text(re_s)}" + (f" (real {re_text(re_r)})" if re_s < 0.98 * re_r else ""))
                if plan.device:
                    parts.append(plan.device)
        return " · ".join(p for p in parts if p)
    return ""


_ANALYSIS_WORDS = {"polar": "Airfoil polar (viscous–inviscid, real Reynolds number)",
                   "wing": "Finite-wing analysis", "cfd": "CFD (Navier–Stokes)"}


def _physics_words(sim) -> str:
    """"heat transfer with buoyancy, 2 fans, smoke (3 emitters), closed domain" for a simulation."""
    if sim is None or getattr(sim, "analysis", "") != "cfd":
        return ""
    bits = []
    if sim.cfd.domain_mode == "closed":
        box = sim.cfd.domain_box
        bits.append("closed domain" + (f" {' × '.join(f'{box[2 * i + 1] - box[2 * i]:g}' for i in range(len(box) // 2))} m"
                                       if box else ""))
    if sim.thermal.enabled:
        bits.append("heat transfer" + (" with buoyancy" if sim.thermal.buoyancy else " (no buoyancy)")
                    + f", ambient {sim.thermal.t_ambient:g} °C")
        walls = [w for w in sim.thermal.walls if w.mode != "adiabatic"]
        if walls:
            bits.append("walls " + ", ".join(f"{w.side} {w.value:.4g}{' °C' if w.mode == 'temperature' else ' W/m²'}"
                                             for w in walls))
    unit = {"thrust": " N", "velocity": " m/s", "pressure_jump": " Pa"}
    if any(f.enabled for f in sim.fans):
        bits.append(", ".join(f"{f.name}: {f.mode.replace('_', ' ')} {f.value:g}{unit.get(f.mode, '')}"
                              for f in sim.fans if f.enabled))
    if any(h.enabled for h in sim.heat_sources):
        bits.append(", ".join(f"{h.name} {h.power:g} W" for h in sim.heat_sources if h.enabled))
    if any(p.enabled for p in sim.ports):
        bits.append(", ".join(f"{p.kind} {p.speed:g} m/s" for p in sim.ports if p.enabled))
    ne = sum(1 for e in sim.emitters if e.enabled)
    if ne:
        bits.append(f"smoke / tracers ({ne} emitter{'s' if ne != 1 else ''})")
    return "; ".join(bits)


def _speed_item(speed: str, closed: bool) -> tuple[str, str]:
    """The card's speed row: an inflow speed, or a closed box's reference speed (it has no inflow)."""
    if closed:
        return ("Reference speed", f"{speed} m/s (closed box, no inflow: it scales time and Re)")
    return ("Speed", f"{speed} m/s")


def run_items(name: str, args: dict, ctx: ToolContext, plan: Any = None, est: float | None = None,
              *, sim: Any = None) -> list[tuple[str, str]]:
    """Exactly what a run call will compute, as (label, value) rows for the approval card.

    Built from the validated input and the planner's estimate: the design, the
    analysis, speed or Reynolds number, angle (or angle range, step and count),
    yaw, quality, grid and cells, the simulated and real Reynolds numbers, the
    device, the placed physics and the estimated time ("unknown" when the
    estimate failed: the card then says so instead of a number). The critique
    found users approving "Wing analysis of Tapered wing: about 25 s" without
    seeing the speed or the angles.
    """
    items: list[tuple[str, str]] = []
    with contextlib.suppress(Exception):
        if sim is None and name in ("run_simulation", "run_grid_study"):
            sim = find_simulation(ctx, args.get("simulation_id"))
        if sim is not None:
            d = ctx.store.designs.get(sim.design_id)
            analysis = sim.analysis
        else:
            d = find_design(ctx, args.get("design_id"))
            analysis = {"run_polar": "polar", "run_wing": "wing", "run_cfd": "cfd"}.get(name,
                                                                                     str(args.get("analysis", "cfd")))
        if d is not None:
            items.append(("Design", f"{d.name} · {_design_word(d)}"))
        if name == "run_grid_study":
            r = float(args.get("refinement", 1.5))
            items.append(("Analysis", "Grid-convergence study (Richardson extrapolation, GCI)"))
            items.append(("Grids", (f"3: this simulation's grid, ×{r:g} and ×{r * r:.3g} cells per reference "
                                    "length (the first reuses its result when it has one)")))
            items.append(("Quantity", str(args.get("quantity") or "Cd (and Cl, Nu where they apply)")))
        elif name == "run_sweep":
            vals = ", ".join(f"{float(v):g}" for v in (args.get("values") or [])[:12])
            unit = {"speed": " m/s", "alpha": "°", "yaw": "°"}.get(str(args.get("variable")), "")
            n = len(args.get("values") or [])
            items.append(("Analysis", f"{_ANALYSIS_WORDS.get(analysis, analysis)}, {n} run{'s' if n != 1 else ''}"))
            items.append(("Varied", f"{args.get('variable')} = {vals}{unit}"))
        else:
            items.append(("Analysis", _ANALYSIS_WORDS.get(analysis, analysis)))
        if sim is not None and name == "run_grid_study":
            c = sim.conditions
            items.append(_speed_item(f"{c.speed:.4g}", getattr(sim.cfd, "domain_mode", "tunnel") == "closed"))
            items.append(("Quality", sim.cfd.quality))
        elif sim is not None:
            c = sim.conditions
            # a closed box has no inflow: its speed is the reference that scales time and Re (the free-fall
            # speed of a natural-convection case), and an angle of attack of 0 says nothing. The heated
            # cavity's card read "Speed 0.05482 m/s · Angle of attack 0°" (round-4 screenshot)
            closed = analysis == "cfd" and getattr(sim.cfd, "domain_mode", "tunnel") == "closed"
            items.append(_speed_item(f"{c.speed:.4g}", closed))
            if analysis == "cfd":
                if not (closed and not c.alpha_deg):
                    items.append(("Angle of attack", _deg(c.alpha_deg)))
                if c.yaw_deg:
                    items.append(("Yaw", _deg(c.yaw_deg)))
                items.append(("Quality", sim.cfd.quality))
            else:
                part = sim.airfoil if analysis == "polar" else sim.wing
                items.append(("Angles", _alpha_text((part.alpha_start, part.alpha_end, part.alpha_step))))
                if analysis == "wing":
                    items.append(("Method", ("vortex lattice" if sim.wing.method == "vlm" else "lifting line") + ", "
                                  + ("viscous strips" if sim.wing.viscous else "inviscid")))
        else:
            dim = int(getattr(d, "dimension", 2) or 2)
            base = dict(args)
            if name == "run_sweep":
                var = str(args.get("variable", ""))
                base = {k: v for k, v in args.items() if k not in ("variable", "values", "analysis")}
                base.setdefault({"speed": "speed_m_s", "alpha": "alpha_deg", "yaw": "yaw_deg"}.get(var, "_"),
                                (args.get("values") or [0])[0])
            s = _settings_for(analysis, base, dim)
            varied = str(args.get("variable", "")) if name == "run_sweep" else ""
            closed = analysis == "cfd" and s.get("domain") == "closed"
            if varied != "speed":
                if "speed" in s:
                    items.append(_speed_item(f"{s['speed']:g}", closed))
                elif "re" in s:
                    items.append(("Reynolds number", re_text(s["re"])))
            if analysis in ("polar", "wing"):
                items.append(("Angles", _alpha_text(s["alpha"])))
                if analysis == "wing":
                    items.append(("Method", ("vortex lattice" if s["method"] == "vlm" else "lifting line") + ", "
                                  + ("viscous strips" if s["viscous"] else "inviscid")))
                else:
                    items.append(("Transition", f"ncrit {s.get('ncrit', 9.0):g}"))
            else:
                if varied != "alpha":
                    items.append(("Angle of attack", _deg(s["alpha"])))
                if s.get("yaw") and varied != "yaw":
                    items.append(("Yaw", _deg(s["yaw"])))
                items.append(("Quality", s["quality"]))
        if plan is not None and getattr(plan, "analysis", "") == "cfd":
            if plan.cells:
                grid = "×".join(str(n) for n in plan.grid)
                items.append(("Grid", f"{_cells_text(plan.cells)} ({grid})" if grid else _cells_text(plan.cells)))
            re_r, re_s = plan.re_real, plan.re_sim
            if math.isfinite(re_r) and math.isfinite(re_s):
                items.append(("Reynolds number", f"{re_text(re_s)} simulated" + (
                    f", {re_text(re_r)} real" if re_s < 0.98 * re_r else "")))
            if plan.device:
                items.append(("Device", plan.device))
        phys = _physics_words(sim if sim is not None else getattr(plan, "simulation", None))
        if phys:
            items.append(("Physics", phys))
    if est is None or not (isinstance(est, (int, float)) and math.isfinite(est)):
        items.append(("Estimated time", "unknown (the estimate failed)"))
    else:
        rough = " (rough)" if name in ("run_polar", "run_wing") else ""
        items.append(("Estimated time", f"about {fmt_seconds(est)}{rough}"))
    return items


def _plan_content(plan) -> dict:
    out: dict[str, Any] = {"analysis": plan.analysis, "estimated_seconds": _num(plan.est_seconds, 3)}
    if plan.analysis == "cfd":
        out.update(cells=plan.cells, grid=list(plan.grid), quality=plan.quality, device=plan.device,
                   reynolds_real=_num(plan.re_real, 4), reynolds_simulated=_num(plan.re_sim, 4))
        if math.isfinite(plan.re_real) and math.isfinite(plan.re_sim) and plan.re_sim < 0.98 * plan.re_real:
            out["reynolds_capped"] = True
    else:
        sim = plan.simulation
        if sim is not None:
            out["angles"] = len(sim.airfoil.alphas() if plan.analysis == "polar" else sim.wing.alphas())
        if math.isfinite(plan.re_real):
            out["reynolds"] = _num(plan.re_real, 4)
    if plan.warnings:
        out["warnings"] = _short_list(plan.warnings)
    if plan.notes:
        out["notes"] = _short_list(plan.notes, 3)
    return out


def _estimate_cost(ctx: ToolContext, args: dict) -> ToolOutcome:
    analysis = str(args.get("analysis", "cfd"))
    d = find_design(ctx, args["design_id"])
    plan = _plan(ctx, d.id, analysis, _settings_for(analysis, args, int(d.dimension)))
    if ctx.log is not None:
        ctx.log.note_planned("cost:" + json.dumps(args, sort_keys=True, default=str), float(plan.est_seconds))
    return ToolOutcome(_plan_content(plan), f"{analysis} of {d.name}: {fmt_seconds(plan.est_seconds)} estimated")


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def _call_run(ctx: ToolContext, label: str, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
    """An API run call with this tool's progress and cancel; API errors become ToolError."""
    api = _api()
    ctx.check_cancel()

    def progress(f: float, msg: str) -> None:
        ctx.report(f, msg or label)

    try:
        return fn(*args, progress=progress, cancel=ctx.cancel, **kw)
    except api.Cancelled as exc:
        part = ""
        res = getattr(exc, "result", None)
        if res is not None and hasattr(res, "sim_id"):
            part = f" The partial result is stored as simulation {res.sim_id}."
        raise ToolError(f"{label} was stopped by the user.{part}") from None
    except api.AerolumeError as exc:
        raise ToolError(f"{label} failed: {exc}") from None


def value_error_texts(mean: Any, stderr: Any) -> tuple[str, str]:
    """(value, error) texts of a statistic by the Results page's rule (:func:`aerolume.core.format.value_error_texts`).

    The run card and the Summary table must print one number the same way: the card read
    "Nu 2.251 ± 4.4e-16" where Results read "2.251 steady" (critic round 3.3). The rule lives
    Qt-free in :mod:`aerolume.core.format`, so the CLI and the MCP server apply it unchanged.
    The error is "steady" below :data:`STEADY_REL` of the value, "—" when unknown.
    """
    from ..core.format import value_error_texts as results_texts
    return results_texts(mean, stderr)


def _uncertainty_text(st: dict) -> str:
    """The uncertainty column of a statistic in the log's Results table: "± 0.010", "steady" or ""."""
    if not isinstance(st, dict) or st.get("mean") is None:
        return ""
    if st.get("steady"):
        return "steady"
    if st.get("stderr") is None:
        return ""
    _v, err = value_error_texts(st["mean"], st["stderr"])
    return "" if err == "—" else ("steady" if err == "steady" else f"± {err}")


def _pm(st: dict) -> str:
    """A statistic as the run card prints it, by the Results page's rule: "3.280 ± 0.010", "2.251 (steady)".

    ``"steady": true`` (from :func:`_stat`) and an error below :data:`STEADY_REL`
    of the value both read "(steady)"; a statistic without an error is the value
    alone; "" without a mean.
    """
    if not isinstance(st, dict) or st.get("mean") is None:
        return ""
    se = 0.0 if st.get("steady") else st.get("stderr")
    value, err = value_error_texts(st["mean"], float("nan") if se is None else se)
    if err == "steady":
        return f"{value} (steady)"
    return value if err == "—" else f"{value} ± {err}"


def cfd_headline(summ: dict) -> str:
    """The numbers a CFD run card leads with: Nu, heat rate and fan thrust first, then the force coefficients.

    A thermal cavity has no forces, and its card read "Cd n/a, Cl n/a" -- missing aero
    numbers as the headline of a heat-transfer run (critic round 3.2). Coefficients the
    run did not produce are left out; "Cd n/a, Cl n/a" remains only when nothing at all
    was measured.
    """
    three = summ.get("analysis") == "cfd3d"
    parts: list[str] = []
    nu = _pm(summ.get("nusselt") or {})
    if nu:
        parts.append(f"Nu {nu}")
    q = summ.get("heat_rate_W" if three else "heat_rate_W_per_m")
    if isinstance(q, (int, float)) and not isinstance(q, bool):
        parts.append(f"Q {q:.3g} W" + ("" if three else "/m"))
    for fan in (summ.get("fans") or [])[:2]:
        t = fan.get("thrust") if isinstance(fan, dict) else None
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            who = str(fan.get("name") or "fan")
            parts.append(f"{who} thrust {t:.3g} N" + ("" if three else "/m"))
    c = summ.get("coefficients") or {}
    for key, lab in (("cd", "Cd"), ("cl", "Cl")):
        v = _pm(c.get(key) or {})
        if v:
            parts.append(f"{lab} {v}")
    return ", ".join(parts) if parts else "Cd n/a, Cl n/a"


#: what a reused result tells the model (and the log): the run card alone was too easy to miss
REUSED_NOTE = ("An identical simulation already had a complete result, so it was not run again: this is that stored "
               "result. Say so in the answer. run_simulation with rerun=true computes it again.")


def _outcome(r, what: str) -> ToolOutcome:
    summ = summarize_api_result(r)
    m = summ.get("metrics", {})
    line = what
    if summ.get("analysis") in ("polar", "wing") and m.get("ld_max") is not None:
        line += f": best L/D {m['ld_max']:.4g} at α {m.get('alpha_ld_max'):g}°"
    elif summ.get("analysis") in ("cfd2d", "cfd3d"):
        line += f": {cfd_headline(summ)}" + ("" if summ.get("converged") else " (not converged)")
    if summ.get("reused_stored_result"):
        line += " (stored result reused)"
        summ.setdefault("reused_note", REUSED_NOTE)
    return ToolOutcome(summ, line, simulation_ids=[r.sim_id], results=key_numbers(summ))


_SWEEP_PART = {"polar": "airfoil", "wing": "wing"}
_ALPHA_KEYS = ("alpha_start", "alpha_end", "alpha_step")


def _sweep_key(sim, analysis: str) -> str:
    """Everything that defines a polar or wing run except its angle range (conditions and solver settings)."""
    d = sim.to_dict()
    part = {k: v for k, v in (d.get(_SWEEP_PART[analysis]) or {}).items() if k not in _ALPHA_KEYS}
    return json.dumps({"conditions": d.get("conditions"), "part": part}, sort_keys=True, default=str)


def covering_sweep(ctx: ToolContext, design, analysis: str, settings: dict):
    """A finished stored run of ``design`` with the same settings whose angle sweep covers the requested one.

    The API reuses only an identical run; a model that asks for α −5…15° when
    the project already holds α −4…14° at the same speed would otherwise add a
    near-duplicate (a local model did exactly that, leaving two
    indistinguishable "Wing polar · 25 m/s" rows). A stored sweep that spans
    the requested range at the same or a finer step answers the same question,
    so it is returned instead. Returns ``(api Result, stored simulation)`` or None.
    """
    try:
        new = ctx.project.plan(design.id, analysis, **settings).simulation
    except Exception:  # noqa: BLE001 - no plan, no comparison: the run reports the problem itself
        return None
    if new is None:
        return None
    want, req = _sweep_key(new, analysis), getattr(new, _SWEEP_PART[analysis])
    store, eps = ctx.store, 1e-9
    for s in reversed(list(store.simulations.values())):
        if s.design_id != design.id or s.analysis != analysis or s.status != "done":
            continue
        have = getattr(s, _SWEEP_PART[analysis])
        if not (have.alpha_start <= req.alpha_start + eps and have.alpha_end >= req.alpha_end - eps
                and have.alpha_step <= req.alpha_step + eps):
            continue
        if _sweep_key(s, analysis) != want or not store.has_result(s.id):
            continue
        with contextlib.suppress(Exception):
            r = ctx.project.result(s.id)
            if r is not None:
                return r, s
    return None


def _reused_outcome(found, what: str) -> ToolOutcome:
    r, sim = found
    out = _outcome(r, what)
    st = getattr(sim, _SWEEP_PART[r.analysis])
    out.content["reused_stored_result"] = True
    out.content["reused_note"] = (f"The stored simulation “{sim.name}” (α {st.alpha_start:g}…{st.alpha_end:g}°, "
                                  f"step {st.alpha_step:g}°) covers the requested range, so it was not run again.")
    if "(stored result reused)" not in out.summary:
        out.summary += " (stored result reused)"
    return out


def _distinct_name(ctx: ToolContext, design, analysis: str, settings: dict) -> str | None:
    """A name that tells a new sweep from a stored one of the same design and speed ("… · α −5…15°").

    Two sweeps that differ only in their angle range would otherwise get the
    API's same default name plus " 2", which the Project tree cannot show.
    """
    try:
        from ..api import _default_name
        new = ctx.project.plan(design.id, analysis, **settings).simulation
        base = _default_name(design, new)
    except Exception:  # noqa: BLE001 - the API's default name is fine
        return None
    taken = {s.name for s in ctx.store.simulations.values() if s.design_id == design.id}
    if base not in taken:
        return None
    st = getattr(new, _SWEEP_PART[analysis])

    def deg(v: float) -> str:
        return f"{v:g}" if v >= 0 else f"−{-v:g}"
    return f"{base} · α {deg(st.alpha_start)}…{deg(st.alpha_end)}°"


def _run_polar(ctx: ToolContext, args: dict) -> ToolOutcome:
    d = find_design(ctx, args["design_id"])
    s = _polar_settings(args)
    found = covering_sweep(ctx, d, "polar", s)
    if found is not None:
        return _reused_outcome(found, f"Polar of {d.name} at Re {re_text(found[0].raw.re)}")
    name = _distinct_name(ctx, d, "polar", s)
    r = _call_run(ctx, f"Polar of {d.name}", ctx.project.run_polar, d.id, **s, **({"name": name} if name else {}))
    return _outcome(r, f"Polar of {d.name} at Re {re_text(r.raw.re)}")


def _run_wing(ctx: ToolContext, args: dict) -> ToolOutcome:
    d = find_design(ctx, args["design_id"])
    s = _wing_settings(args)
    found = covering_sweep(ctx, d, "wing", s)
    if found is not None:
        return _reused_outcome(found, f"Wing {s['method'].upper()} of {d.name}")
    name = _distinct_name(ctx, d, "wing", s)
    r = _call_run(ctx, f"Wing analysis of {d.name}", ctx.project.run_wing, d.id, **s,
                  **({"name": name} if name else {}))
    return _outcome(r, f"Wing {s['method'].upper()} of {d.name}")


def _run_cfd(ctx: ToolContext, args: dict) -> ToolOutcome:
    d = find_design(ctx, args["design_id"])
    s = _cfd_settings(args, int(d.dimension))
    _plan(ctx, d.id, "cfd", s)          # a setup the planner refuses fails here, before anything is recorded
    if ctx.before_gpu_run is not None:
        ctx.before_gpu_run()
    key = "run_cfd:" + json.dumps({"design": d.id, **s}, sort_keys=True, default=str)
    r = _call_run(ctx, f"CFD of {d.name}", ctx.project.run_cfd, d.id, rerun=ctx.wants_fresh(key), **s)
    if ctx.fresh is not None:
        ctx.fresh.add(key)
    return _outcome(r, f"CFD of {d.name} ({s['quality']})")


def _physics_detail(sim) -> dict:
    """The placed physics of a simulation as compact data (describe_simulation)."""
    out: dict[str, Any] = {"domain": sim.cfd.domain_mode}
    if sim.cfd.domain_box:
        out["domain_box_m"] = [_num(c) for c in sim.cfd.domain_box]
    th = sim.thermal
    if th.enabled:
        out["heat_transfer"] = {"ambient_temperature_c": _num(th.t_ambient), "buoyancy": bool(th.buoyancy),
                                "walls": [{"side": w.side, "mode": w.mode, "value": _num(w.value)} for w in th.walls]}
    if sim.fans:
        out["fans"] = [{"name": f.name, "position_m": [_num(c) for c in f.position],
                        "direction": [_num(c, 3) for c in f.direction], "diameter_m": _num(f.diameter),
                        "mode": f.mode, "value": _num(f.value), "enabled": f.enabled} for f in sim.fans]
    if sim.ports:
        out["ports"] = [{"name": p.name, "kind": p.kind, "position_m": [_num(c) for c in p.position],
                         "speed_m_s": _num(p.speed), "size_m": _num(p.size),
                         "temperature_c": _num(p.temperature) if p.temperature is not None else None}
                        for p in sim.ports]
    if sim.heat_sources:
        out["heat_sources"] = [{"name": h.name, "kind": h.kind, "position_m": [_num(c) for c in h.position],
                                "power_w": _num(h.power)} for h in sim.heat_sources]
    if sim.emitters:
        out["emitters"] = [{"name": e.name, "species": e.species, "kind": e.kind,
                            "position_m": [_num(c) for c in e.position]} for e in sim.emitters[:8]]
    return out


def _describe_simulation(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    sim = find_simulation(ctx, args["simulation_id"])
    d = ctx.store.designs.get(sim.design_id)
    settings = ctx.project.settings(sim.id)
    rows = {}
    for k, v in settings.items():
        spec = api.SETTINGS[k]
        rows[k] = {"value": _num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v,
                   "label": spec.label, **({"unit": spec.unit} if spec.unit else {})}
    out: dict[str, Any] = {"simulation_id": sim.id, "name": sim.name, "design": getattr(d, "name", ""),
                           "design_id": sim.design_id, "analysis": sim.analysis, "status": sim.status,
                           "has_result": bool(ctx.store.has_result(sim.id)), "settings": rows,
                           "editable_settings": list(settings)}
    if sim.analysis == "cfd":
        out["physics"] = _physics_detail(sim)
    if sim.error:
        out["error"] = str(sim.error)[:300]
    if sim.notes:
        out["notes_data"] = str(sim.notes)[:300]
    return ToolOutcome(out, f"{sim.name}: {sim.analysis}, {sim.status}")


def _estimate_simulation(ctx: ToolContext, args: dict) -> float:
    try:
        sim = find_simulation(ctx, args["simulation_id"])
    except ToolError as exc:
        raise InputError(str(exc)) from None
    ctx.last_plan = None
    if sim.status == "done" and ctx.store.has_result(sim.id) and not _rerun(ctx, sim.id, args):
        return 0.0                      # the stored result is returned: nothing runs
    plan = ctx.project.plan_of(sim.id)
    ctx.last_plan = plan
    return float(plan.est_seconds)


def _rerun(ctx: ToolContext, key: str, args: dict) -> bool:
    """Run again although a stored result exists: the model asked (rerun) or the goal asked for a new run."""
    return bool(args.get("rerun", False)) or ctx.wants_fresh(key)


def _run_simulation(ctx: ToolContext, args: dict) -> ToolOutcome:
    sim = find_simulation(ctx, args["simulation_id"])
    rerun = _rerun(ctx, sim.id, args)
    if sim.analysis == "cfd" and ctx.before_gpu_run is not None and not (
            sim.status == "done" and ctx.store.has_result(sim.id) and not rerun):
        ctx.before_gpu_run()
    r = _call_run(ctx, f"“{sim.name}”", ctx.project.run, sim.id, rerun=rerun)
    if ctx.fresh is not None:
        ctx.fresh.add(sim.id)
    out = _outcome(r, f"{sim.name}")
    if rerun and not args.get("rerun") and not getattr(r, "reused", False):
        out.content["ran_again"] = "The goal asked for a new run, so it was computed again rather than reused."
    return out


_GRID_QUANTITIES = ("cd", "cl", "cm", "nusselt")
#: cost of one refinement level relative to the base grid: cells grow as r^dim and the time step shrinks as 1/r
_GRID_COST_EXPONENT = {2: 3, 3: 4}


def _grid_inputs(ctx: ToolContext, args: dict) -> tuple[Any, float, str | None]:
    sim = find_simulation(ctx, args["simulation_id"])
    if sim.analysis != "cfd":
        raise ToolError(f"“{sim.name}” is a {sim.analysis} analysis: polars and wing analyses are deterministic, their "
                        "resolution is the angle step. A grid study needs a CFD simulation.")
    r = _range("refinement", args.get("refinement", 1.5), 1.3, 2.0)
    q = args.get("quantity")
    if q is not None and q not in _GRID_QUANTITIES:
        raise ToolError(f"quantity must be one of {', '.join(_GRID_QUANTITIES)}.")
    return sim, r, q


def _estimate_grid_study(ctx: ToolContext, args: dict) -> float:
    """The planner's estimate of the base grid times the cost of the two finer levels (r³ and r⁶ in 2D)."""
    try:
        sim, r, _q = _grid_inputs(ctx, args)
    except ToolError as exc:
        raise InputError(str(exc)) from None
    ctx.last_plan = None
    plan = ctx.project.plan_of(sim.id)
    ctx.last_plan = plan
    d = ctx.store.designs.get(sim.design_id)
    e = _GRID_COST_EXPONENT.get(int(getattr(d, "dimension", 2) or 2), 3)
    base = float(plan.est_seconds)
    stored = sim.status == "done" and ctx.store.has_result(sim.id)
    return (0.0 if stored else base) + base * (r ** e + r ** (2 * e))


def _run_grid_study(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    sim, r, q = _grid_inputs(ctx, args)
    if ctx.before_gpu_run is not None:
        ctx.before_gpu_run()
    gs = _call_run(ctx, f"Grid study of “{sim.name}”", ctx.project.grid_study_of, sim.id, refinement=r, quantity=q)
    quantities: dict[str, Any] = {}
    rows: list[dict] = []
    for key, g in gs.gci.items():
        fine = g.levels[0]
        quantities[key] = {
            "convergence": g.convergence, "observed_order": _num(g.p, 3), "formal_order": _num(g.formal_order, 3),
            "asymptotic": g.asymptotic, "noise_limited": g.noise_limited,
            "fine_grid_value": _num(fine.value, 5), "extrapolated_value": _num(g.phi_ext, 5),
            "gci_fine_pct": _num(100.0 * g.gci_fine, 3), "u_num": _num(g.u_num, 3), "u_stat": _num(g.u_stat, 3),
            "u_total": _num(g.u_total, 3),
            "grids": [{"cells_per_ref": _num(1.0 / lv.h, 4), "value": _num(lv.value, 5), "stderr": _num(lv.stderr, 3),
                       "simulation_id": lv.sim_id} for lv in g.levels],
            "summary": api._gci_sentence(g),
            "warnings": [*g.warnings, *g.notes][:4]}
        band = f"± {_num(g.u_num, 3)} (GCI)" if g.usable else (g.convergence or "no estimate")
        rows.append({"label": f"{g.quantity} (finest grid)", "value": _num(fine.value, 5), "unit": "",
                     "uncertainty": band, "source": fine.sim_id})
    content = {"simulation_id": sim.id, "simulation": sim.name, "refinement": r, "quantities": quantities,
               "notes": list(gs.notes),
               "meaning": ("u_num is the GCI: an about-95 % band on the finest grid's value from the discretisation "
                           "alone (Celik et al. 2008). u_stat is two standard errors of that run's time average; "
                           "u_total combines both. Neither includes modelling error (turbulence model, wall "
                           "resolution, 2D vs 3D). noise_limited true: the grid differences are within the averaging "
                           "noise, so no order or GCI is claimed. asymptotic false: the observed order is far from "
                           "the formal order 2, so the grids may not be fine enough for the extrapolation.")}
    ids = [x.sim_id for x in gs.results]
    usable = [k for k, g in gs.gci.items() if g.usable]
    summ = (f"Grid study of {sim.name}: " + "; ".join(f"{k} GCI {100 * gs.gci[k].gci_fine:.2g} %" for k in usable)
            if usable else f"Grid study of {sim.name}: no usable GCI")
    return ToolOutcome(content, summ, simulation_ids=ids, results=rows)


_CASES = ("cavity", "heated_cylinder", "fan_room")


def _create_case(ctx: ToolContext, args: dict) -> ToolOutcome:
    """A reference case (design + simulation) with its published value; nothing runs."""
    api = _api()
    kind = str(args["case"])
    dim = int(_range("dimension", args.get("dimension", 2), 2, 3))
    try:
        if kind == "cavity":
            case = api.cavity_case(_range("rayleigh", args.get("rayleigh", 1e4), 1.0, 1e8),
                                   _range("prandtl", args.get("prandtl", 0.71), 0.01, 100.0),
                                   size=_range("size_m", args.get("size_m", 0.1), 1e-3, 100.0, "m"), dimension=dim)
        elif kind == "heated_cylinder":
            if dim != 2:
                raise ToolError("The heated-cylinder case is 2D.")
            case = api.heated_cylinder_case(_range("reynolds", args.get("reynolds", 40.0), 1e-3, 1e4),
                                            d=_range("size_m", args.get("size_m", 0.01), 1e-4, 10.0, "m"),
                                            delta_t=_range("delta_t_k", args.get("delta_t_k", 10.0), -200.0, 500.0,
                                                           "K"))
        elif kind == "fan_room":
            case = api.fan_room_case(_range("thrust_n", args.get("thrust_n", 1.0), -1e4, 1e4, "N"), dimension=dim)
        else:
            raise ToolError(f"case must be one of {', '.join(_CASES)}.")
        q = str(args.get("quality", "draft"))
        if q not in _QUALITY:
            raise ToolError(f"quality must be one of {', '.join(_QUALITY)}.")
        before_designs, before_sims = set(ctx.store.designs), set(ctx.store.simulations)
        sim = ctx.project.add_simulation(case, quality=q, reuse=True)
    except api.AerolumeError as exc:
        raise ToolError(str(exc)) from None
    if ctx.log is not None:
        ctx.log.note_planned(f"sim:{sim.id}")           # the plan's expected time includes it (planner_expected)
    d = ctx.store.designs[sim.design_id]
    out: dict[str, Any] = {"design_id": d.id, "simulation_id": sim.id, "name": sim.name, "case": kind,
                           "reused_design": d.id in before_designs, "reused_simulation": sim.id in before_sims,
                           "reference": case.reference,
                           "notes": list(case.notes), "physics": _physics_detail(sim),
                           "next_step": "Call run_simulation with this simulation_id (estimate_cost does not take "
                                        "cases; run_simulation's approval shows the estimate), then "
                                        "compare_reference with it."}
    if ctx.compact:                      # a small model's 8k context: the placed physics are in describe_simulation
        out.pop("physics", None)
    return ToolOutcome(out, f"Case {sim.name} ready (simulation {sim.id})", simulation_ids=[])


def _compare_reference(ctx: ToolContext, args: dict) -> ToolOutcome:
    """Each simulation against the published reference its set-up reproduces (nothing runs)."""
    refs = list(args.get("simulation_ids") or [])
    if not refs:
        raise ToolError("Give simulation_ids (see project_overview).")
    rows: list[dict] = []
    ids: list[str] = []
    for ref in refs[:8]:
        sim = find_simulation(ctx, ref)
        r = _result(ctx, sim.id)
        if r is None:
            rows.append({"simulation_id": sim.id, "name": sim.name, "note": "No result yet: run it first."})
            continue
        design = r.design.raw if r.design is not None else None
        chk = reference_summary(r.raw, r.simulation, design)
        row: dict[str, Any] = {"simulation_id": sim.id, "name": sim.name}
        if chk is None:
            row["note"] = ("No published reference applies to this simulation. Reference cases: create_case "
                           "(cavity, heated_cylinder, fan_room).")
        else:
            row.update(chk)
            ids.append(sim.id)
        rows.append(row)
    done = [r for r in rows if r.get("applies") and isinstance(r.get("difference_pct"), (int, float))]
    word = {"nusselt": "Nu", "fan thrust": "thrust"}
    def band(r: dict) -> str:
        if not isinstance(r.get("band_pct"), (int, float)):
            return ""
        return f", {'inside' if r.get('within_band') else 'outside'} ± {r['band_pct']:g} %"

    summ = "; ".join(f"{word.get(r['quantity'], r['quantity'])} {r['measured']:.5g} against {r['reference']:.5g} "
                     f"({r['difference_pct']:+.2g} %" + band(r)
                     + (", consistency" if r.get("kind") == "consistency" else "")
                     + ")" for r in done) or "No reference applies"
    return ToolOutcome({"checks": rows}, summ, simulation_ids=ids)


def _setting_value(spec: Any, v: Any) -> Any:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _num(v)
    return v


def suggestion_rows(ctx: ToolContext, sim, changes: list[dict]) -> list[dict]:
    """Each proposed change checked against the setting's rules, with the value it would replace.

    Unknown settings and invalid values are kept as rows marked ``valid: False``
    with the reason (shown struck through on the card), never guessed at or
    clamped: Lumen's rule, a model cannot invent a setting or push one past its
    limit. Nothing is applied here.
    """
    api = _api()
    d = ctx.store.designs.get(sim.design_id)
    rows = []
    for ch in changes:
        key = str(ch.get("setting", ""))
        row: dict[str, Any] = {"setting": key, "label": key, "unit": "", "old": None, "new": ch.get("value"),
                               "valid": False, "problem": ""}
        try:
            spec = api.setting_for(key, sim, d)
            row.update(label=spec.label, unit=spec.unit, old=_setting_value(spec, spec.get(sim)))
            new = spec.check(ch.get("value"))
            row["new"] = _setting_value(spec, new)
            if new == spec.get(sim):
                row["problem"] = "already set to this value"
            else:
                row["valid"] = True
        except api.AerolumeError as exc:
            row["problem"] = str(exc)
        rows.append(row)
    return rows


def _suggest_settings(ctx: ToolContext, args: dict) -> ToolOutcome:
    sim = find_simulation(ctx, args["simulation_id"])
    changes = list(args.get("changes") or [])
    if not changes:
        raise ToolError("Give changes: [{setting, value}] (describe_simulation lists the editable settings).")
    rows = suggestion_rows(ctx, sim, changes)
    valid = [r for r in rows if r["valid"]]
    content = {"proposed": True, "applied": False, "simulation_id": sim.id, "simulation": sim.name,
               "has_result": bool(ctx.store.has_result(sim.id)), "reason": str(args.get("reason", ""))[:400],
               "changes": rows,
               "note": ("Shown to the user as a card with Apply and Dismiss. Nothing has changed: the settings change "
                        "only if the user clicks Apply (a simulation with a result is copied, not edited). Do not "
                        "say the settings were changed.")}
    if not valid:
        return ToolOutcome({"error": "No valid change: " + "; ".join(f"{r['setting']}: {r['problem']}" for r in rows),
                            **content}, "No valid change to suggest", is_error=True)
    n = len(valid)
    return ToolOutcome(content, f"Suggested {n} change{'s' if n != 1 else ''} to {sim.name} (not applied)")


_SWEEP_KEY = {"alpha": "alpha", "speed": "speed", "yaw": "yaw"}


def _sweep_inputs(ctx: ToolContext, args: dict) -> tuple[Any, str, str, list[float], dict]:
    d = find_design(ctx, args["design_id"])
    analysis = str(args.get("analysis", "cfd"))
    variable = str(args["variable"])
    values = [float(v) for v in args["values"]]
    if not values:
        raise ToolError("values must hold at least one value.")
    if len(values) > MAX_SWEEP_RUNS:
        raise ToolError(f"At most {MAX_SWEEP_RUNS} values per sweep; split it or use fewer values.")
    if analysis in ("polar", "wing") and variable == "alpha":
        raise ToolError(f"run_{analysis} already sweeps the angle of attack: call it with alpha_start/end/step.")
    if analysis in ("polar", "wing") and variable == "yaw":
        raise ToolError("Yaw sweeps are CFD only.")
    base = {k: v for k, v in args.items() if k not in ("variable", "values", "analysis", "design_id")}
    if variable == "speed":
        base.pop("reynolds", None)
        base["speed_m_s"] = values[0]
    elif variable == "alpha":
        base["alpha_deg"] = values[0]
    elif variable == "yaw":
        base["yaw_deg"] = values[0]
    settings = _settings_for(analysis, base, int(d.dimension))
    settings.pop(_SWEEP_KEY[variable], None)
    if variable == "speed":
        settings.pop("re", None)
    return d, analysis, variable, values, settings


def _estimate_sweep(ctx: ToolContext, args: dict) -> float:
    try:
        d, analysis, variable, values, settings = _sweep_inputs(ctx, args)
    except ToolError as exc:
        raise InputError(str(exc)) from None
    total = 0.0
    ctx.last_plan = None
    for v in values:
        plan = _plan(ctx, d.id, analysis, {**settings, _SWEEP_KEY[variable]: v})
        ctx.last_plan = ctx.last_plan or plan
        total += float(plan.est_seconds)
    return total


def _run_sweep(ctx: ToolContext, args: dict) -> ToolOutcome:
    d, analysis, variable, values, settings = _sweep_inputs(ctx, args)
    if analysis == "cfd" and ctx.before_gpu_run is not None:
        ctx.before_gpu_run()
    sw = _call_run(ctx, f"Sweep of {d.name} over {variable}", ctx.project.sweep, d.id, analysis=analysis,
                   **{**settings, _SWEEP_KEY[variable]: values})
    rows, ids, results = [], [], []
    for vals, r in zip(sw.values, sw.results, strict=False):
        summ = summarize_api_result(r)
        ids.append(r.sim_id)
        results.extend(key_numbers(summ))
        row: dict[str, Any] = {variable: _num(vals.get(_SWEEP_KEY[variable])), "simulation_id": r.sim_id}
        if summ.get("analysis") in ("cfd2d", "cfd3d"):
            c = summ["coefficients"]
            row.update(cd=c["cd"].get("mean"), cd_stderr=c["cd"].get("stderr"), cl=c["cl"].get("mean"),
                       cl_stderr=c["cl"].get("stderr"), l_over_d=summ.get("l_over_d"),
                       converged=summ.get("converged"), reynolds_simulated=summ.get("reynolds_simulated"))
        else:
            m = summ.get("metrics", {})
            row.update({k: m.get(k) for k in ("ld_max", "alpha_ld_max", "cl_max", "cd_min", "oswald_e_parabolic_fit")
                        if k in m})
        rows.append(row)
    validity: list[str]
    if analysis == "cfd":
        validity = design_validity(d)
        if not all(r.get("converged") for r in rows):
            validity.append("Not every run converged: compare their standard errors before ranking them.")
        capped = False
        with contextlib.suppress(Exception):
            capped = any(r.raw.re_sim < 0.98 * r.raw.re_real for r in sw.results)
        if capped:
            validity.append("Some runs were simulated below the real Reynolds number (see reynolds_simulated).")
    else:
        validity = [POLAR_VALIDITY if analysis == "polar" else WING_VALIDITY]
    content = {"analysis": analysis, "variable": variable,
               "unit": {"alpha": "deg", "speed": "m/s", "yaw": "deg"}[variable], "rows": rows, "design_id": d.id,
               "design": d.name, "validity": validity}
    return ToolOutcome(content, f"Sweep of {d.name} over {variable}: {len(rows)} runs", simulation_ids=ids,
                       results=results)


# ---------------------------------------------------------------------------
# results, comparison, plots, reports
# ---------------------------------------------------------------------------

def _result(ctx: ToolContext, ref: Any):
    api = _api()
    try:
        return ctx.project.result(str(ref))
    except api.AerolumeError as exc:
        raise ToolError(str(exc)) from None


def _get_results(ctx: ToolContext, args: dict) -> ToolOutcome:
    refs = list(args.get("simulation_ids") or [])
    if not refs:
        raise ToolError("Give simulation_ids (see project_overview).")
    out: list[dict] = []
    full: list[dict] = []
    n = 0
    for ref in refs[:12]:
        sim = find_simulation(ctx, ref)
        r = _result(ctx, sim.id)
        if r is None:
            out.append({"simulation_id": sim.id, "name": sim.name, "status": sim.status,
                        "note": "No result yet." + (f" Error: {sim.error}" if sim.error else "")})
            continue
        design = r.design.raw if r.design is not None else None
        # a local model gets the table only when it asks for it (its 8k context); Claude by default
        want_table = bool(args.get("include_table", not ctx.compact))
        s = summarize_result(r.raw, r.simulation, design, table=want_table)
        s["simulation_id"] = sim.id
        full.append(s)
        out.append(compact_summary(s, table=want_table) if ctx.compact else s)
        n += 1
    return ToolOutcome({"results": out}, f"Read {n} result{'s' if n != 1 else ''}",
                       simulation_ids=[s["simulation_id"] for s in out if "analysis" in s],
                       results=[row for s in full for row in key_numbers(s)])


def _compare(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    ids = [find_simulation(ctx, r).id for r in (args.get("simulation_ids") or [])]
    if len(ids) < 2:
        raise ToolError("Compare needs at least two simulations with results.")
    base = args.get("baseline_id")
    base_id = find_simulation(ctx, base).id if base else ids[0]
    if base_id not in ids:
        ids.insert(0, base_id)
    base_idx = ids.index(base_id)
    try:
        c = ctx.project.compare(ids, baseline=base_idx)
    except api.AerolumeError as exc:
        raise ToolError(str(exc)) from None
    table = []
    for r in c.rows:
        cells = []
        for cell in r.get("cells", []):
            if not cell.get("comparable", True):
                continue
            v = cell.get("value")
            item: dict[str, Any] = {"simulation_id": cell["sim_id"], "value": v if isinstance(v, bool) else _num(v)}
            if cell.get("stderr") is not None and _num(cell.get("stderr")) is not None:
                if is_steady(v, cell["stderr"]):
                    item["steady"] = True           # round-off, not an error bar (see STEADY_NOTE)
                else:
                    item["stderr"] = _num(cell["stderr"], 3)
            if cell.get("pct") is not None and not cell.get("is_baseline"):
                item["change_pct"] = _num(cell["pct"], 3)
            if cell.get("significant") is not None and not cell.get("is_baseline"):
                item["significant"] = bool(cell["significant"])
            if cell.get("lower_bound"):
                item["lower_bound"] = True
            cells.append(item)
        table.append({"metric": r["key"], "label": r.get("label", r["key"]), "unit": r.get("unit", ""), "cells": cells})
    content = {"baseline_id": base_id, "rows": table, "warnings": list(c.warnings)}
    return ToolOutcome(content, f"Compared {len(ids)} simulations (baseline {base_id})", simulation_ids=list(ids))


#: what make_plot accepts: every Result.plot kind plus the two multi-result figures
PLOT_KINDS = ("polar", "history", "cp", "field", "surface", "streamlines", "smoke", "vortices", "slice", "span",
              "planform", "geometry", "sensitivity", "compare", "sweep")


def _plot_path(ctx: ToolContext, kind: str) -> Path:
    folder = ctx.log.plots_dir if ctx.log is not None else ctx.ws.root / "experiments" / "_plots"
    folder.mkdir(parents=True, exist_ok=True)
    ctx.plot_counter[0] += 1
    path = folder / f"{ctx.plot_counter[0]:02d}-{kind}.png"
    if not inside(path, ctx.ws.root):
        raise ToolError("Figures are written inside the project folder only.")
    return path


def _make_plot(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    kind = str(args["kind"])
    ids = [find_simulation(ctx, r).id for r in (args.get("simulation_ids") or [])][:6]
    if not ids:
        raise ToolError("Give simulation_ids.")
    results = []
    for sid in ids:
        r = _result(ctx, sid)
        if r is None:
            raise ToolError(f"{sid} has no result yet.")
        results.append(r)
    title = str(args.get("title") or "").strip() or None
    metric = str(args.get("metric") or "")
    path = _plot_path(ctx, kind)
    ctx.report(0.2, "Drawing the figure")
    try:
        if kind == "compare":
            if len(results) < 2:
                raise ToolError("kind 'compare' needs two or more results.")
            ctx.project.compare(results).plot(metric or "cd", path)
        elif kind == "sweep":
            variable = str(args.get("variable") or "alpha")
            attr = {"alpha": "alpha_deg", "speed": "speed", "yaw": "yaw_deg"}.get(variable)
            if attr is None:
                raise ToolError("variable must be alpha, speed or yaw for a sweep plot.")
            pts = sorted(((getattr(r.simulation.conditions, attr), r) for r in results), key=lambda t: t[0])
            sw = api.Sweep([variable], [({variable: v}, r) for v, r in pts])
            sw.plot(variable, metric or None, path=path, title=title)
        elif kind == "polar" and len(results) > 1:
            from ..model import PolarResult, WingResult
            from ..viz import figures as F
            raws = [r.raw for r in results]
            labels = [r.name for r in results]
            if all(isinstance(x, PolarResult) for x in raws):
                fig = F.fig_polar(raws, labels=labels, title=title, figsize=(7.0, 5.2))
            elif all(isinstance(x, WingResult) for x in raws):
                fig = F.fig_wing_polar(raws, labels=labels, title=title, figsize=(7.0, 5.2))
            else:
                raise ToolError("An overlaid 'polar' needs polar results or wing results (one kind).")
            F.save_figure(fig, path, dpi=150)
        else:
            r = results[0]
            kinds = r.plot_kinds()
            if kind not in kinds:
                raise ToolError(f"A {r.kind} result has no {kind!r} figure; choose one of {', '.join(kinds)} "
                                "(or 'compare' / 'sweep' for several results).")
            r.plot(kind, path)
    except api.AerolumeError as exc:
        raise ToolError(f"The figure could not be made: {exc}") from None
    rel = ctx.log.add_plot(path) if ctx.log is not None else path.name
    return ToolOutcome({"file": rel, "kind": kind, "simulation_ids": ids,
                        "note": "The figure is shown to the user and saved with the experiment."},
                       f"Figure ({kind}) of {len(ids)} result{'s' if len(ids) > 1 else ''}", files=[path])


def _write_report(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    ids = [find_simulation(ctx, r).id for r in (args.get("simulation_ids") or [])]
    if not ids:
        raise ToolError("Give simulation_ids with results.")
    fmt = str(args.get("format", "pdf"))
    title = str(args.get("title") or "").strip() or None
    ctx.report(0.1, "Writing the report")
    try:
        path = ctx.project.report(ids, None, title=title, fmt=fmt)
    except api.AerolumeError as exc:
        raise ToolError(f"The report could not be written: {exc}") from None
    if not inside(path, ctx.ws.root):
        raise ToolError("The report was written outside the project folder.")
    rel = Path(path).resolve().relative_to(ctx.ws.root.resolve()).as_posix()
    return ToolOutcome({"file": rel, "simulations": ids}, f"Report written: {rel}")


# ---------------------------------------------------------------------------
# code and deletion (always approved by the user first)
# ---------------------------------------------------------------------------

def run_script_headless(ctx: ToolContext, code: str) -> dict:
    """Run approved code on this thread with ``project``, ``al``, ``np`` and ``log`` bound.

    API calls inside see the tool's cancel event (``aerolume.api.context``), so
    Stop reaches every run the script starts. Output is captured; ``result``
    (if the script sets it) is returned.
    """
    import numpy as np

    import aerolume as al
    api = _api()
    out = io.StringIO()

    def log(*parts: Any) -> None:
        print(*parts, file=out)

    ns: dict[str, Any] = {"__name__": "__assistant__", "project": ctx.project, "al": al, "np": np, "log": log}
    t0 = time.monotonic()
    error, tb = "", ""
    try:
        compiled = compile(code, "<assistant script>", "exec")
        with api.context(cancel=ctx.cancel, progress=ctx.report, project=ctx.project), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compiled, ns)                       # noqa: S102 - the user approved this exact code
    except api.Cancelled:
        error = "Stopped by the user."
    except BaseException as exc:  # a script error (SystemExit included) is reported to the model
        if isinstance(exc, KeyboardInterrupt):
            raise
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        tb = "\n".join(traceback.format_exc().splitlines()[-8:])
    return {"output": out.getvalue(), "error": error, "traceback": tb, "seconds": time.monotonic() - t0,
            "result": ns.get("result")}


def _run_python(ctx: ToolContext, args: dict) -> ToolOutcome:
    code = str(args["code"])
    ctx.report(0.0, "Running the script")
    runner = ctx.ws.run_script
    res = runner(code, ctx.cancel, ctx.report) if runner is not None else run_script_headless(ctx, code)
    text = str(res.get("output", ""))
    if len(text) > PYTHON_OUTPUT_CHARS:
        text = text[:PYTHON_OUTPUT_CHARS] + f"\n… ({len(res['output']) - PYTHON_OUTPUT_CHARS} more characters)"
    content: dict[str, Any] = {"output": text, "seconds": _num(res.get("seconds", 0.0), 3)}
    if res.get("result") is not None:
        try:
            content["result"] = json.loads(json.dumps(res["result"], default=str))
        except (TypeError, ValueError):
            content["result"] = repr(res["result"])[:2000]
    sims = list(res.get("sim_ids") or [])
    if sims:
        content["simulation_ids"] = sims
    figures = [Path(f) for f in (res.get("figures") or []) if Path(f).is_file()]
    if figures:
        content["figures"] = [f.name for f in figures]
    if res.get("error"):
        content["error"] = res["error"]
        if res.get("traceback"):
            content["traceback"] = res["traceback"]
        return ToolOutcome(content, f"Script failed: {str(res['error'])[:120]}", is_error=True, simulation_ids=sims,
                           files=figures)
    return ToolOutcome(content, f"Script ran in {fmt_seconds(float(res.get('seconds', 0.0)))}", simulation_ids=sims,
                       files=figures)


def _delete(ctx: ToolContext, args: dict) -> ToolOutcome:
    api = _api()
    kind = str(args["kind"])
    try:
        if kind == "simulation":
            sim = find_simulation(ctx, args["id"])
            if ctx.ws.busy(sim.id):
                raise ToolError("That simulation is running or queued; it cannot be deleted now.")
            ctx.project.remove_simulation(sim.id)
            return ToolOutcome({"deleted": sim.id, "kind": "simulation"}, f"Deleted simulation {sim.name}")
        d = find_design(ctx, args["id"])
        sims = [s.id for s in list(ctx.store.simulations.values()) if s.design_id == d.id]
        if any(ctx.ws.busy(s) for s in sims):
            raise ToolError("A simulation of that design is running or queued; it cannot be deleted now.")
        ctx.project.remove_design(d.id)
    except api.AerolumeError as exc:
        raise ToolError(str(exc)) from None
    return ToolOutcome({"deleted": d.id, "kind": "design", "simulations_deleted": sims},
                       f"Deleted design {d.name} and {len(sims)} simulation{'s' if len(sims) != 1 else ''}")


# ---------------------------------------------------------------------------
# the experiment record
# ---------------------------------------------------------------------------

_QUANTITY = (("speed", re.compile(r"speed|velocity|m/s|\bu_?inf|\bv_?inf|free.?stream", re.IGNORECASE)),
             ("alpha", re.compile(r"alpha|α|angle of attack|\baoa\b|incidence", re.IGNORECASE)),
             ("yaw", re.compile(r"\byaw|sideslip", re.IGNORECASE)),
             ("reynolds", re.compile(r"reynolds|\bre\b", re.IGNORECASE)))


def _quantity_of(text: str) -> str | None:
    """Which flow condition a plan entry names ("speed_m_s" -> "speed"), or None."""
    for q, rx in _QUANTITY:
        if rx.search(text or ""):
            return q
    return None


def plan_problems(plan: dict, goal: str) -> list[str]:
    """Contradictions in a plan's controls: a control is a quantity held fixed.

    A control that also names a variable of the plan is rejected, and so is a
    speed control when the goal itself compares two or more speeds. qwen3:8b
    recorded "Controls (held fixed): speed_m_s; alpha_deg" for a goal comparing
    15 and 25 m/s, and the log said so in print.
    """
    from .experiment import goal_speeds
    variables = [str(v) for v in plan.get("variables") or []]
    varied = {_quantity_of(v) for v in variables} - {None}
    speeds = goal_speeds(goal)
    out = []
    for c in (str(x) for x in plan.get("controls") or []):
        q = _quantity_of(c)
        if q is None:
            continue
        if q in varied:
            out.append(f"“{c}” is listed as a control (held fixed) and also among the variables; keep it only "
                       "under variables.")
        elif q == "speed" and len(speeds) >= 2:
            out.append(f"The goal compares {' and '.join(f'{v:g}' for v in speeds)} m/s, so speed is varied: list "
                       f"it under variables (with those speeds), not “{c}” under controls.")
    return out


def planner_expected(ctx: ToolContext) -> tuple[float | None, int]:
    """(seconds, runs) the planner expects this experiment's set-up runs to take; (None, 0) when it has no estimate.

    Sums :attr:`ExperimentLog.planned`: estimate_cost results and the
    simulations created here (create_case), each planned now unless its stored
    result will be returned. A simulation the planner refuses is left out.
    """
    log = ctx.log
    planned = dict(getattr(log, "planned", None) or {}) if log is not None else {}
    total, runs = 0.0, 0
    for key, seconds in planned.items():
        if seconds is None and key.startswith("sim:"):
            sid = key[4:]
            sim = ctx.store.simulations.get(sid)
            if sim is None:
                continue
            if sim.status == "done" and ctx.store.has_result(sid) and not ctx.wants_fresh(sid):
                seconds = 0.0                   # its stored result is returned: nothing runs
            else:
                try:
                    seconds = float(ctx.project.plan_of(sid).est_seconds)
                except Exception:  # noqa: BLE001 - left out: the run's own approval asks with "cost unknown"
                    seconds = None
        if seconds is None or not math.isfinite(seconds):
            continue
        total += max(0.0, float(seconds))
        runs += 1
    return (total, runs) if runs else (None, 0)


def _record_plan(ctx: ToolContext, args: dict) -> ToolOutcome:
    """Record the plan; its expected time is the planner's estimate, not the figure in the call.

    The model's ``expected_seconds`` is a guess (qwen3:8b planned "5 min 00 s"
    for a cavity run that took 11 s, and the card showed it as the plan). It is
    replaced by :func:`planner_expected`, or dropped when the planner has not
    estimated anything yet, and the result says which.
    """
    plan = {k: v for k, v in args.items() if v not in (None, "", [])}
    problems = plan_problems(plan, ctx.log.record.goal if ctx.log is not None else "")
    if problems:
        raise ToolError("Plan not recorded: " + " ".join(problems))
    guess = plan.pop("expected_seconds", None)
    est, runs = planner_expected(ctx)
    content: dict[str, Any] = {"recorded": True}
    if est is not None:
        plan["expected_seconds"] = _num(est, 3)
        plan["expected_time_basis"] = f"planner estimate, {runs} run{'s' if runs != 1 else ''}"
        content["expected_seconds"] = plan["expected_seconds"]
        content["expected_time_basis"] = plan["expected_time_basis"]
        typed = isinstance(guess, (int, float)) and not isinstance(guess, bool)
        if typed and abs(float(guess) - est) > 0.25 * max(est, 1.0):
            content["note"] = (f"Expected time is the planner's estimate ({fmt_seconds(est)}), not the "
                               f"{fmt_seconds(float(guess))} in this call.")
    elif guess is not None:
        content["note"] = ("No planner estimate yet, so the plan shows no expected time (the figure in this call is "
                           "not recorded). estimate_cost gives one for a design; a case's run approval shows its own.")
    if ctx.log is not None:
        ctx.log.set_plan(plan)
    extra = f", about {fmt_seconds(est)} (planner estimate)" if est is not None else ""
    # brief(): cut between words after the maths became text, keeping a number after the cut; a slice at
    # 90 characters read "cavity at Ra , about 28 s" (critic round 3.4)
    return ToolOutcome(content, f"Plan recorded: {brief(str(plan.get('objective', '')), 90)}{extra}")


#: a finding whose quantity is computed from results (a change, a ratio) rather than read from one run
_DERIVED_QUANTITY = re.compile(r"differ|chang|ratio|delta|increas|decreas|reduc|improv|gain|loss|relative|percent|"
                               r"%|\bvs\.?\b|versus|compared|between", re.IGNORECASE)
_ANGLE_QUANTITY = re.compile(r"\balpha\b|α|angle|incidence|\baoa\b|stall onset|\byaw\b|twist|dihedral",
                             re.IGNORECASE)
_PLUS_MINUS = re.compile(r"(±|\+/-|\+-)\s*\d")
_DEGREES = re.compile(r"°|\bdeg(?:rees?)?\b", re.IGNORECASE)


def _written(value: Any) -> str:
    """A finding's value as the model wrote it (the shortest repr keeps its precision: 26.47, not 26.470000)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return repr(float(value)) if isinstance(value, float) else str(value)


def result_facts(summary: dict, n: int = 6) -> str:
    """A simulation's headline values in one line for an error message ("ld_max 29.52 at α 3°, ...")."""
    parts: list[str] = []
    m = summary.get("metrics") or {}
    for key, at in (("ld_max", "alpha_ld_max"), ("cl_max", "alpha_cl_max")):
        if m.get(key) is not None:
            parts.append(f"{key} {m[key]:.4g}" + (f" at α {m[at]:g}°" if m.get(at) is not None else ""))
    for key, v in m.items():
        if key in ("ld_max", "alpha_ld_max", "cl_max", "alpha_cl_max") or not isinstance(v, (int, float)) \
                or isinstance(v, bool):
            continue
        parts.append(f"{key} {v:.4g}")
    for key, st in (summary.get("coefficients") or {}).items():
        if isinstance(st, dict) and st.get("mean") is not None:
            parts.append(f"{key} {st['mean']:g}" + (f" ± {st['stderr']:g}" if st.get("stderr") is not None else ""))
    if summary.get("l_over_d") is not None:
        parts.append(f"l_over_d {summary['l_over_d']:g}")
    return ", ".join(parts[:n]) or "no headline values"


def finding_problems(ctx: ToolContext, findings: list[dict]) -> list[str]:
    """Findings that do not hold for the simulation they cite (see :func:`_record_conclusion`)."""
    from .experiment import numbers_in, untraced_numbers
    problems: list[str] = []
    summaries: dict[str, dict | None] = {}
    for i, f in enumerate(findings, 1):
        quantity = str(f.get("quantity", "")).strip() or f"finding {i}"
        try:
            sim = find_simulation(ctx, f.get("simulation_id", ""))
        except ToolError as exc:
            problems.append(f"Finding {i} (“{quantity}”): {exc}")
            continue
        if sim.id not in summaries:
            r = None
            with contextlib.suppress(ToolError):
                r = _result(ctx, sim.id)
            summaries[sim.id] = (summarize_result(r.raw, r.simulation, r.design.raw if r.design is not None else None)
                                 if r is not None else None)
        summ = summaries[sim.id]
        if summ is None:
            problems.append(f"Finding {i} (“{quantity}”): simulation {sim.id} (“{sim.name}”) has no result.")
            continue
        value = f.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not _DERIVED_QUANTITY.search(quantity):
            tok = _written(value)
            if untraced_numbers(tok, numbers_in(summ)):
                problems.append(f"Finding {i} (“{quantity}” = {tok}) is not a value of simulation {sim.id} "
                                f"(“{sim.name}”); its values are {result_facts(summ)}. Correct the value, or cite "
                                "the simulation it comes from.")
        unc = str(f.get("uncertainty") or "").strip()
        unit = str(f.get("unit") or "").strip().lower()
        is_angle = bool(_ANGLE_QUANTITY.search(quantity)) or unit in ("deg", "°", "degree", "degrees")
        if unc and not is_angle:
            if _DEGREES.search(unc):
                problems.append(f"Finding {i}: the uncertainty “{unc}” is in degrees, but “{quantity}” is not an "
                                "angle.")
            elif summ.get("analysis") in ("polar", "wing") and _PLUS_MINUS.search(unc):
                res = summ.get("optimum_resolution_deg")
                note = f" ±{res:g} is the angle resolution in degrees and applies only to an optimum angle." if res \
                    else ""
                problems.append(f"Finding {i}: “{quantity}” comes from a deterministic {summ['analysis']} result, "
                                f"which has no statistical error bar.{note} Leave the uncertainty empty, or write "
                                "what limits it (e.g. 'sampled at the α step', 'not validated').")
    return problems


def conclusion_claim_problems(ctx: ToolContext, texts: list[str]) -> list[str]:
    """Actions a conclusion's own text claims that no successful tool call of the experiment performed.

    The action check of the final answer (agent.py) does not see the log's
    Conclusions section, which record_conclusion writes. Measured live with
    qwen3:8b (the delete goal, the prompt's "you cannot delete" rule removed):
    the model called record_conclusion with the summary "The simulation 'Wing
    polar · 25 m/s' has been deleted and is no longer available" and the
    limitation "Simulation has been deleted." -- refused that time only because
    a finding's value was the word "done". A claimed action is refused here on
    its own, so the log can never record a deletion, run or creation that did not
    happen.
    """
    from .claims import CLAIM_KINDS, claimed_actions
    if ctx.log is None:
        return []
    kinds = {t.name: t.kind for t in TOOLS}
    done = {kinds.get(a.tool, "") for a in list(ctx.log.record.actions) if a.status == "ok"}
    what = {"delete": "deleted anything", "run": "ran an analysis", "create": "created anything",
            "edit": "changed a setting (a suggestion changes nothing until the user clicks Apply)"}
    out: list[str] = []
    for text in texts:
        for kind, sentence in claimed_actions(text):
            if not done & set(CLAIM_KINDS[kind]):
                problem = f"“{sentence}” says so, but no tool call in this experiment {what[kind]}."
                if problem not in out:
                    out.append(problem)
    return out


def _record_conclusion(ctx: ToolContext, args: dict) -> ToolOutcome:
    findings = list(args.get("findings") or [])
    limitations = [str(x) for x in (args.get("limitations") or [])]
    texts = [str(args.get("summary", "")), *limitations, *(str(f.get("quantity", "")) for f in findings)]
    claimed = conclusion_claim_problems(ctx, texts)
    if claimed:
        return ToolOutcome({"error": "Conclusion not recorded: it claims an action that did not happen. "
                                     + " ".join(claimed) + " Record only what your tool calls did."},
                           "Conclusion not recorded: it claims an action no tool call performed", is_error=True)
    problems = finding_problems(ctx, findings)
    if problems:
        return ToolOutcome({"error": "Conclusion not recorded. " + " ".join(problems)},
                           f"Conclusion not recorded: {len(problems)} finding problem{'s' if len(problems) > 1 else ''}",
                           is_error=True)
    untraced: list[str] = []
    if ctx.log is not None:
        untraced = ctx.log.conclude(str(args.get("summary", "")), findings=findings, limitations=limitations,
                                    status="running")
    content: dict[str, Any] = {"recorded": True}
    if untraced:
        content["untraced_numbers"] = untraced
        content["note"] = ("These numbers do not appear in any tool result. Correct them from the results, or "
                           "say they are derived and how.")
    return ToolOutcome(content, f"Conclusion recorded ({len(findings)} finding{'s' if len(findings) != 1 else ''})")


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------

_DESIGN_ID = _str("Design id (or exact design name) from project_overview.")
_SPEED = _number("Free-stream speed in m/s.")
_ALPHA_START = _number("First angle of attack in degrees.")
_ALPHA_END = _number("Last angle of attack in degrees.")
_ALPHA_STEP = _number("Angle step in degrees (0.5-2 typical).")
_PARAM_LIST = {"type": "array", "description": "Parameter overrides, e.g. [{'name': 'slant_angle', 'value': 30}].",
               "items": _obj({"name": _str("Parameter name."),
                              "value": {"type": ["number", "string", "boolean"],
                                        "description": "Value (SI units, degrees)."}}, ["name", "value"])}

_POINT = {"type": "array", "items": {"type": "number"},
          "description": "Coordinates in metres in the design frame: [x, y] (2D) or [x, y, z] (3D)."}
_DIRECTION = {"type": "array", "items": {"type": "number"},
              "description": "Direction vector (normalised for you), e.g. [1, 0] blows along +x."}
_PHYSICS = _obj({
    "domain": _str("tunnel: free stream from the left (default). closed: walls on every side (a cavity, a room); "
                   "the flow is then driven by buoyancy, fans or jets.", ["tunnel", "closed"]),
    "domain_box_m": {"type": "array", "items": {"type": "number"},
                     "description": "Domain extent in metres: [xmin, xmax, ymin, ymax] (2D), plus [zmin, zmax] (3D)."},
    "heat_transfer": _bool("Solve the temperature field (needed by heated bodies, wall temperatures, heaters)."),
    "buoyancy": _bool("With heat transfer: Boussinesq buoyancy along -y (2D) / -z (3D). Default true; false for "
                      "forced convection."),
    "ambient_temperature_c": _number("Free-stream / initial temperature in °C (default 20)."),
    "walls": {"type": "array", "description": "Thermal conditions of domain sides (closed domains: also left/right).",
              "items": _obj({"side": _str("Domain side.", ["left", "right", "bottom", "top", "front", "back"]),
                             "mode": _str("Condition.", ["adiabatic", "temperature", "heat_flux"]),
                             "value": _number("°C for temperature, W/m² for heat_flux.")}, ["side", "mode"])},
    "heat_sources": {"type": "array", "description": "Volumetric heaters.",
                     "items": _obj({"kind": _str("rect/disc (2D), box/sphere (3D).",
                                                 ["rect", "disc", "box", "sphere"]),
                                    "position_m": _POINT,
                                    "size_m": {"type": "array", "items": {"type": "number"},
                                               "description": "Extents [m] (rect/box) or [radius] (disc/sphere)."},
                                    "power_w": _number("Heat released: W (3D) or W per metre of span (2D).")},
                                   ["position_m", "size_m", "power_w"])},
    "fans": {"type": "array", "description": "Actuator-disc fans or propellers.",
             "items": _obj({"position_m": _POINT, "direction": _DIRECTION,
                            "diameter_m": _number("Disc diameter (3D) or width (2D) in metres."),
                            "mode": _str("thrust (N; N/m in 2D), pressure_jump (Pa) or velocity (target m/s).",
                                         ["thrust", "pressure_jump", "velocity"]),
                            "value": _number("Thrust, pressure jump or target velocity, by mode."),
                            "swirl": _number("3D only: tangential-to-axial velocity ratio (default 0).")},
                           ["position_m", "diameter_m", "value"])},
    "ports": {"type": "array", "description": "Jets (blowing in) and vents (sucking out).",
              "items": _obj({"kind": _str("jet or vent.", ["jet", "vent"]), "position_m": _POINT,
                             "direction": _DIRECTION, "size_m": _number("Width / diameter in metres."),
                             "speed_m_s": _number("Speed through the port."),
                             "temperature_c": _number("Jet temperature in °C (needs heat transfer)."),
                             "smoke": _bool("Release smoke with the jet.")},
                            ["kind", "position_m", "size_m", "speed_m_s"])},
    "emitters": {"type": "array", "description": "Smoke (dye) or tracer-particle sources for flow pictures.",
                 "items": _obj({"species": _str("dye (smoke) or particles.", ["dye", "particles"]),
                                "kind": _str("Emitter shape.", ["point", "line", "rake", "rect", "disc", "plane"]),
                                "position_m": _POINT,
                                "end_m": {**_POINT, "description": "Other end of a line or rake [m]."},
                                "size_m": {"type": "array", "items": {"type": "number"},
                                           "description": "rect extents, disc [radius], plane [width, height]."},
                                "count": {"type": "integer", "description": "Nozzles of a rake."},
                                "color": _str("Smoke colour #rrggbb (up to 4 colours).")}, ["position_m"])},
})

_CFD_PROPS = {
    "design_id": _DESIGN_ID, "speed_m_s": _SPEED,
    "alpha_deg": _number("Angle of attack / pitch in degrees (default 0)."),
    "yaw_deg": _number("Sideslip in degrees, 3D designs only (default 0)."),
    "quality": _str("Grid and run-length preset; draft is minutes, fine can be much longer.", list(_QUALITY)),
    "road": _str("Ground plane: auto (vehicles on a moving road), none, moving or fixed.", list(_ROAD)),
    "reynolds_mode": _str("auto (planner default: 2D airfoils are capped at a resolvable Re), real, or capped.",
                          list(_REYNOLDS)),
    "sensitivity": _str("Also compute a shape-sensitivity map of this objective (slower).",
                        ["none", "cd", "cl", "ld"]),
    "physics": {**_PHYSICS, "description": "Optional placed physics: heat transfer and buoyancy, heated walls, "
                                           "heaters, fans, jets and vents, smoke emitters, a closed domain."},
}
#: run_cfd as a small local model sees it: without the physics object (its tokens; the model can still
#: run a prepared case or a set-up simulation with run_simulation)
_CFD_PROPS_LOCAL = {k: v for k, v in _CFD_PROPS.items() if k != "physics"}

#: the editable settings (aerolume.api.SETTINGS), listed in suggest_settings' schema
_SETTING_NAMES = ["adaptive_dt", "alpha", "alpha_end", "alpha_start", "alpha_step", "average_from", "buoyancy",
                  "cells_per_ref", "diagnostics", "duration", "ground", "mach", "method", "moving_ground", "ncrit",
                  "quality", "re_cap", "reynolds", "speed", "t_ambient", "thermal", "turbulence", "viscous", "yaw"]

TOOLS: list[Tool] = [
    Tool("project_overview",
         "List the open project's designs (id, name, type) and simulations (id, analysis, status, conditions). "
         "Call this first in every experiment to learn the ids you need.",
         _obj({}), _project_overview, "read",
         short="List designs and simulations with their ids. Call first."),
    Tool("describe_design",
         "Details of one design: reference length/area, size, template parameters, wing planform (span, area, "
         "aspect ratio, MAC, sweep, taper) or mesh facts, and which analyses it can have. Call before choosing "
         "conditions or deriving variants.",
         _obj({"design_id": _DESIGN_ID}, ["design_id"]), _describe_design, "read",
         short="Details of a design (references, planform, template parameters, analyses)."),
    Tool("create_naca_airfoil",
         "Add a NACA 4- or 5-digit airfoil design (2D). Call when the goal names a NACA section that is not in "
         "the project yet; an identical existing design is reused.",
         _obj({"code": _str("NACA digits, e.g. '2412' or '23012'."),
               "chord_m": _number("Chord in metres (default 1)."),
               "name": _str("Design name (optional).")}, ["code"]),
         _create_naca, "create", short="Add a NACA airfoil design (2D)."),
    Tool("create_template",
         "Add a 2D design from the template gallery: sedan, hatchback, fastback, suv, pickup, van, box_truck, "
         "ahmed (rear-slant reference body), front_wing, naca, cylinder, square, ellipse, flat_plate, "
         "rounded_rectangle. Parameters are optional overrides by name (an error lists the valid ones).",
         _obj({"template": _str("Template name.", ["sedan", "hatchback", "fastback", "suv", "pickup", "van",
                                                    "box_truck", "ahmed", "front_wing", "naca", "cylinder", "square",
                                                    "ellipse", "flat_plate", "rounded_rectangle"]),
               "parameters": _PARAM_LIST, "name": _str("Design name (optional).")}, ["template"]),
         _create_template, "create", local=False),
    Tool("create_wing",
         "Add a straight-tapered finite wing (3D planform) for the wing analysis (VLM / lifting line) or 3D CFD. "
         "Call when the goal is about a wing, aspect ratio, taper, sweep, twist or induced drag.",
         _obj({"span_m": _number("Tip-to-tip span in metres."),
               "root_chord_m": _number("Root chord in metres."),
               "tip_chord_m": _number("Tip chord in metres (default = root: rectangular)."),
               "sweep_deg": _number("Quarter-chord sweep in degrees (default 0)."),
               "dihedral_deg": _number("Dihedral in degrees (default 0)."),
               "tip_twist_deg": _number("Tip twist in degrees, negative = washout (default 0)."),
               "airfoil": _str("Section: NACA code like '2412' or 'design:<id>' of a 2D airfoil."),
               "name": _str("Design name (optional).")}, ["span_m", "root_chord_m"]),
         _create_wing, "create", short="Add a tapered wing planform (3D)."),
    Tool("derive_design",
         "Make a new variant of a design without changing the original: template parameters (e.g. the Ahmed "
         "body's slant_angle, a NACA airfoil's chord), wing planform values, or scale / rotations of a 3D mesh. "
         "Call once per variant when an experiment varies the geometry.",
         _obj({"base_design_id": _DESIGN_ID, "name": _str("Name of the new design."),
               "template_parameters": _PARAM_LIST,
               "span_m": _number("Wing span in metres."), "root_chord_m": _number("Wing root chord in metres."),
               "tip_chord_m": _number("Wing tip chord in metres."),
               "sweep_deg": _number("Wing quarter-chord sweep in degrees."),
               "dihedral_deg": _number("Wing dihedral in degrees."),
               "tip_twist_deg": _number("Wing tip twist in degrees."), "airfoil": _str("Wing section."),
               "scale": _number("Uniform scale factor (3D meshes)."),
               "rotate_x_deg": _number("3D mesh rotation about x."), "rotate_y_deg": _number("3D mesh rotation about y."),
               "rotate_z_deg": _number("3D mesh rotation about z."),
               "body_temperature_c": _number("Make a heated (or cooled) copy: every body held at this temperature "
                                             "in °C (runs then need physics.heat_transfer)."),
               "body_heat_flux_w_m2": _number("Make a copy whose bodies release this heat flux in W/m².")},
              ["base_design_id", "name"]),
         _derive_design, "create", short="Make a variant of a design (template parameters, wing planform)."),
    Tool("estimate_cost",
         "Estimate how long an analysis would take (and for CFD the grid, cells and simulated vs real Reynolds "
         "number) without running it. Call before planning CFD runs or sweeps.",
         _obj({"analysis": _str("Which analysis.", ["cfd", "polar", "wing"]), "design_id": _DESIGN_ID,
               "speed_m_s": _SPEED, "reynolds": _number("Reynolds number instead of a speed (polar)."),
               "alpha_deg": _CFD_PROPS["alpha_deg"], "yaw_deg": _CFD_PROPS["yaw_deg"],
               "quality": _CFD_PROPS["quality"], "alpha_start": _ALPHA_START, "alpha_end": _ALPHA_END,
               "alpha_step": _ALPHA_STEP, "physics": _CFD_PROPS["physics"]}, ["analysis", "design_id"]),
         _estimate_cost, "read", short="Estimate run time (and CFD grid, Re) without running.",
         local_schema=_obj({"analysis": _str("Which analysis.", ["cfd", "polar", "wing"]), "design_id": _DESIGN_ID,
                            "speed_m_s": _SPEED, "reynolds": _number("Reynolds number instead of a speed (polar)."),
                            "alpha_deg": _CFD_PROPS["alpha_deg"], "quality": _CFD_PROPS["quality"],
                            "alpha_start": _ALPHA_START, "alpha_end": _ALPHA_END, "alpha_step": _ALPHA_STEP},
                           ["analysis", "design_id"])),
    Tool("run_polar",
         "Run a viscous airfoil polar (XFOIL-class, real Reynolds number, seconds) of a 2D airfoil design over a "
         "range of angles and return cl, cd, cm and L/D per angle plus Cl_max, best L/D and the lift slope. "
         "Use this for airfoil questions such as the angle of best L/D.",
         _obj({"design_id": _DESIGN_ID, "speed_m_s": _SPEED,
               "reynolds": _number("Chord Reynolds number instead of a speed (optional)."),
               "alpha_start": _ALPHA_START, "alpha_end": _ALPHA_END, "alpha_step": _ALPHA_STEP,
               "ncrit": _number("Transition parameter (9 = wind tunnel, default)."),
               "mach": _number("Mach number for the compressibility correction (default 0).")}, ["design_id"]),
         _run_polar, "run", estimate=_estimate_for("polar"),
         short="Airfoil polar (seconds): cl, cd, cm, L/D per angle, best L/D, Cl_max."),
    Tool("run_wing",
         "Run a finite-wing analysis (vortex lattice or lifting line with viscous section drag, seconds) of a "
         "wing design over a range of angles: CL, CD, induced drag, span efficiency, best L/D, stall onset.",
         _obj({"design_id": _DESIGN_ID, "speed_m_s": _SPEED, "alpha_start": _ALPHA_START, "alpha_end": _ALPHA_END,
               "alpha_step": _ALPHA_STEP, "method": _str("vlm (any planform) or llt (straight wings).", ["vlm", "llt"]),
               "viscous": _bool("Add profile drag from section polars (default true).")}, ["design_id", "speed_m_s"]),
         _run_wing, "run", estimate=_estimate_for("wing"),
         short="Finite-wing polar (seconds): CL, CD, CDi, e, best L/D."),
    Tool("run_cfd",
         "Run one Navier-Stokes simulation (2D or 3D after the design; minutes) at one condition and return "
         "mean coefficients with standard errors, convergence, simulated vs real Reynolds number and warnings. "
         "Use for vehicles, bluff bodies and 3D shapes; for airfoil polars prefer run_polar. physics adds heat "
         "transfer (Nusselt number, heat rate), fans (delivered thrust, flow), jets, smoke emitters or a closed "
         "domain; for a published benchmark use create_case instead.",
         _obj(_CFD_PROPS, ["design_id"]), _run_cfd, "run", estimate=_estimate_for("cfd"),
         short="One CFD run (minutes): Cd, Cl, Cm with uncertainty.",
         local_schema=_obj(_CFD_PROPS_LOCAL, ["design_id", "speed_m_s"])),
    Tool("run_sweep",
         "Run one analysis several times over one variable (CFD over alpha, speed or yaw; polar or wing over "
         "speed) and return one table row per value. Use for parameter studies at fixed other conditions.",
         _obj({**_CFD_PROPS, "analysis": _str("Analysis to repeat.", ["cfd", "polar", "wing"]),
               "variable": _str("The swept variable.", ["alpha", "speed", "yaw"]),
               "values": {"type": "array", "items": {"type": "number"}, "description": "Values of the variable."},
               "alpha_start": _ALPHA_START, "alpha_end": _ALPHA_END, "alpha_step": _ALPHA_STEP},
              ["design_id", "analysis", "variable", "values"]),
         _run_sweep, "run", estimate=_estimate_sweep, short="Repeat an analysis over one variable; table of results.",
         local_schema=_obj({**_CFD_PROPS_LOCAL, "analysis": _str("Analysis to repeat.", ["cfd", "polar", "wing"]),
                            "variable": _str("The swept variable.", ["alpha", "speed", "yaw"]),
                            "values": {"type": "array", "items": {"type": "number"},
                                       "description": "Values of the variable."},
                            "alpha_start": _ALPHA_START, "alpha_end": _ALPHA_END, "alpha_step": _ALPHA_STEP},
                           ["design_id", "analysis", "variable", "values"])),
    Tool("get_results",
         "Read stored results of simulations (coefficients with standard errors, polar/wing tables and metrics, "
         "Reynolds numbers, convergence, warnings, validity). Call before quoting any number of an earlier run.",
         _obj({"simulation_ids": _ids("Simulation ids."),
               "include_table": _bool("Include the per-angle table of polars and wings (large: ask only when you "
                                      "need values at single angles).")},
              ["simulation_ids"]),
         _get_results, "read", short="Read stored results with uncertainty."),
    Tool("compare_results",
         "Compare results of several simulations against a baseline: values, changes in percent and whether "
         "differences are significant given the standard errors, plus warnings when conditions differ.",
         _obj({"simulation_ids": _ids("Simulations to compare (2 or more)."),
               "baseline_id": _str("Baseline simulation (default: the first).")}, ["simulation_ids"]),
         _compare, "read", short="Compare results against a baseline (deltas, significance)."),
    Tool("make_plot",
         "Save a figure of results as a PNG in the experiment folder; it is shown to the user. kind: polar "
         "(polar or wing results, several overlay), history (CFD convergence), cp (surface pressure), field "
         "(2D flow field), surface / streamlines / smoke / vortices / slice (3D CFD), span (wing span loading), "
         "planform, geometry, sensitivity, compare (bars of one metric), sweep (metric vs alpha/speed/yaw).",
         _obj({"kind": _str("Figure kind.", list(PLOT_KINDS)), "simulation_ids": _ids("Simulations to plot."),
               "metric": _str("compare / sweep: cd, cl, cm or ld (sweeps of polars: ld_max, cl_max)."),
               "variable": _str("sweep: alpha, speed or yaw.", ["alpha", "speed", "yaw"]),
               "title": _str("Figure title (optional).")}, ["kind", "simulation_ids"]),
         _make_plot, "create", short="Save and show a figure of results."),
    Tool("write_report",
         "Write the project's PDF or HTML report of chosen simulations into the project's reports folder.",
         _obj({"simulation_ids": _ids("Simulations to include."),
               "format": _str("pdf or html.", ["pdf", "html"]), "title": _str("Report title.")}, ["simulation_ids"]),
         _write_report, "create", local=False),
    Tool("run_python",
         "Run a short Python script against the Aerolume scripting API with `project` (the open project), `al` "
         "(import aerolume as al), `np` and `log` bound; set `result` to return a value. The user sees the code "
         "and must approve it. Use only when no other tool can do the job (custom post-processing).",
         _obj({"code": _str("The Python source."), "purpose": _str("One sentence: what the script does.")},
              ["code", "purpose"]),
         _run_python, "code", local=False),
    Tool("delete_item",
         "Delete a design (with its simulations) or a simulation. The user must approve it. Only when the user "
         "asked to remove something.",
         _obj({"kind": _str("What to delete.", ["design", "simulation"]), "id": _str("Its id.")}, ["kind", "id"]),
         _delete, "delete", local=False),
    Tool("describe_simulation",
         "Settings of one simulation (speed, angles, quality, duration, grid, turbulence, Reynolds mode; wing and "
         "polar ranges), its placed physics (heat transfer, walls, fans, jets, heaters, emitters, domain), status "
         "and whether it has a result. Call before suggest_settings or run_simulation.",
         _obj({"simulation_id": _str("Simulation id (from project_overview).")}, ["simulation_id"]),
         _describe_simulation, "read", short="Settings, physics and status of a simulation."),
    Tool("run_simulation",
         "Run a simulation that is already in the project as it is set up (by the user on the Setup page, or by "
         "create_case), or return its stored result when it has one (rerun=true runs it again). Use it for "
         "simulations with physics prepared elsewhere and for reference cases.",
         _obj({"simulation_id": _str("Simulation id."),
               "rerun": _bool("Run again even when a complete result is stored (default false).")},
              ["simulation_id"]),
         _run_simulation, "run", estimate=_estimate_simulation,
         short="Run a prepared simulation (or read its stored result)."),
    Tool("run_grid_study",
         "Re-run a CFD simulation on three systematically refined grids and estimate its discretisation error "
         "(Celik et al. 2008: observed order, Richardson-extrapolated value, GCI band), with the averaging noise "
         "taken into account. Use it when the goal asks how far a CFD number can be trusted, or before comparing a "
         "Nusselt number or drag with a reference. Costs about 1 + r^3 + r^6 base runs in 2D (r^4 in 3D).",
         _obj({"simulation_id": _str("A CFD simulation (from project_overview, run_cfd or create_case)."),
               "refinement": _number("Grid ratio between levels, 1.3 to 2 (default 1.5)."),
               "quantity": _str("Quantity to analyse (default Cd, plus Cl for lifting designs and Nu with heat "
                                "transfer).", list(_GRID_QUANTITIES))}, ["simulation_id"]),
         _run_grid_study, "run", local=False, estimate=_estimate_grid_study),
    Tool("create_case",
         "Add a published reference case, ready to run: 'cavity' (differentially heated square or cubic cavity, de "
         "Vahl Davis 1983 / Tric et al. 2000, set by its Rayleigh number), 'heated_cylinder' (forced convection, "
         "Lange et al. 1998 correlation, set by Reynolds number) or 'fan_room' (a thrust fan in a closed room: a "
         "consistency check only, since thrust mode imposes the thrust and momentum theory does not apply in a "
         "closed room). Returns the design, the simulation (not run) and the reference value; an existing match is "
         "reused. Then call run_simulation and compare_reference.",
         _obj({"case": _str("Which case.", list(_CASES)),
               "rayleigh": _number("cavity: Rayleigh number (published values at 1e3, 1e4, 1e5, 1e6)."),
               "prandtl": _number("cavity: Prandtl number (default 0.71, air)."),
               "reynolds": _number("heated_cylinder: Reynolds number on the diameter (default 40)."),
               "delta_t_k": _number("heated_cylinder: wall minus free-stream temperature in K (default 10)."),
               "thrust_n": _number("fan_room: fan thrust in N (N per metre of span in 2D; default 1)."),
               "size_m": _number("cavity side or cylinder diameter in metres."),
               "dimension": {"type": "integer", "description": "2 (default) or 3 (cavity, fan_room)."},
               "quality": _str("Grid preset of the simulation (default draft).", list(_QUALITY))}, ["case"]),
         _create_case, "create",
         # offered to local models too: without it qwen3:8b answered "the tools cannot run a simulation for a
         # differentially heated square cavity" -- true for its tool list, while the system prompt told it to
         # use create_case (critic round 3)
         short="Set up a benchmark run: cavity (by Rayleigh), heated_cylinder (by Reynolds), fan_room (consistency).",
         local_schema=_obj({"case": _str("Which case.", list(_CASES)),
                            "rayleigh": _number("cavity: Rayleigh number (published at 1e3, 1e4, 1e5, 1e6)."),
                            "reynolds": _number("heated_cylinder: Reynolds number (default 40)."),
                            "thrust_n": _number("fan_room: fan thrust in N per metre (default 1)."),
                            "dimension": {"type": "integer", "description": "2 (default) or 3."}}, ["case"])),
    Tool("compare_reference",
         "Compare simulations with the published reference their set-up reproduces (de Vahl Davis cavity, Lange "
         "et al. heated cylinder; the fan room is a consistency check): measured value with its standard error, "
         "the reference, the difference in percent and the source. Nothing runs. Use it whenever the goal asks for "
         "a comparison with a reference, and quote its numbers.",
         _obj({"simulation_ids": _ids("Simulations with results (from run_simulation or project_overview).")},
              ["simulation_ids"]),
         _compare_reference, "read", short="Compare results with their published reference (difference in %)."),
    Tool("suggest_settings",
         "Propose changes to a simulation's settings (for example a finer quality or a longer duration when a "
         "run did not converge). The user sees a card listing each change old -> new and decides: nothing changes "
         "unless they click Apply. Each value is checked against the setting's valid range. Never say that the "
         "settings were changed.",
         _obj({"simulation_id": _str("Simulation id."),
               "changes": {"type": "array", "description": "The changes.",
                           "items": _obj({"setting": _str("Setting name (describe_simulation lists them).",
                                                          _SETTING_NAMES),
                                          "value": {"type": ["number", "string", "boolean"],
                                                    "description": "New value (units as describe_simulation)."}},
                                         ["setting", "value"])},
               "reason": _str("One sentence: why these changes.")}, ["simulation_id", "changes", "reason"]),
         _suggest_settings, "suggest", short="Propose setting changes; the user applies them."),
    Tool("record_plan",
         "Record the experiment plan in the log before running anything: objective, variables and ranges, "
         "controls held fixed, method and quality, expected number of runs and time, validity limits.",
         _obj({"objective": _str("What is measured and why."),
               "method": _str("Analysis and why it fits (e.g. polar at the real Reynolds number)."),
               "variables": {"type": "array", "items": {"type": "string"}, "description": "Varied quantities and ranges."},
               "controls": {"type": "array", "items": {"type": "string"},
                            "description": "Quantities held fixed (never one that is varied)."},
               "quality": _str("Quality / resolution chosen."),
               "expected_runs": {"type": "integer", "description": "Number of runs."},
               "expected_seconds": _number("Your expected total time in seconds; the plan records the "
                                                   "planner's estimate (estimate_cost, create_case) instead."),
               "validity": {"type": "array", "items": {"type": "string"}, "description": "Known validity limits."}},
              ["objective", "method"]),
         _record_plan, "record", short="Record the plan (objective, variables, controls, method, expected cost)."),
    Tool("record_conclusion",
         "Record the findings when the experiment is complete: a summary, each key number with its unit, "
         "uncertainty and source simulation, and the validity limits. Each value is checked against the "
         "simulation it cites.",
         _obj({"summary": _str("The answer in two or three sentences."),
               "findings": {"type": "array", "description": "Key numbers.",
                            "items": _obj({"quantity": _str("What it is."), "value": _number("The number."),
                                           "unit": _str("Unit, or empty."),
                                           "uncertainty": _str("CFD: ± standard error. Polar and wing values "
                                                               "have none: 'lower bound', 'not validated' or "
                                                               "empty."),
                                           "simulation_id": _str("Where it comes from.")},
                                          ["quantity", "value", "simulation_id"])},
               "limitations": {"type": "array", "items": {"type": "string"}, "description": "Validity limits."}},
              ["summary", "findings"]),
         _record_conclusion, "record", short="Record findings with numbers and their source simulations."),
]


class Toolbox:
    """The tool catalogue with validation, approval rules and execution."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.tools = {t.name: t for t in (tools if tools is not None else TOOLS)}

    def specs(self, *, compact: bool = False) -> list[ToolSpec]:
        """Tool specs for a provider; ``compact`` gives small local models a shorter, focused set."""
        return [t.spec(compact) for t in self.tools.values() if t.local or not compact]

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def prepare(self, call: ToolCall) -> tuple[dict | None, list[str]]:
        """The validated input of a call, or None and the problems (unknown tool, bad JSON, schema)."""
        tool = self.tools.get(call.name)
        if tool is None:
            return None, [f"Unknown tool {call.name!r}. Tools: {', '.join(self.tools)}."]
        value = call.input
        if value is None:
            value = {}
        if isinstance(value, str):
            try:
                value = json.loads(value) if value.strip() else {}
            except ValueError:
                return None, [f"The input is not valid JSON: {value[:200]!r}"]
        if isinstance(value, dict):
            # JSON nulls stand for "not given" (models send them for optional parameters they skip):
            # every optional parameter has a default, and a null required one reads as "missing"
            value = {k: v for k, v in value.items() if v is not None}
        value = S.coerce(tool.schema, value)
        problems = S.validate(tool.schema, value)
        if problems:
            return None, problems
        return dict(value), []

    def approval(self, name: str, args: dict, ctx: ToolContext, threshold_s: float) -> Approval | None:
        """The question to ask the user before running this call, or None when no approval is needed.

        Code and deletions always ask; runs ask when their estimated time is above
        ``threshold_s`` **and whenever the estimate fails** (fail closed: an unknown
        cost is asked about, never run silently). A call whose own input is wrong
        raises :class:`InputError`: the caller answers it with that error and runs
        nothing (asking the user to approve a call that cannot work would be noise).
        Every card lists exactly what would run (:func:`run_items`).
        """
        tool = self.tools[name]
        if tool.kind == "code":
            return Approval("code", "Run this Python script?", str(args.get("purpose", "")), None,
                            code=str(args.get("code", "")),
                            items=[("Purpose", str(args.get("purpose", "")) or "not given"),
                                   ("Runs", ("Python in this window with project and al bound; it may create, "
                                             "change or run anything a script can"))])
        if tool.kind == "delete":
            what = str(args.get("kind", "item"))
            try:
                obj = find_simulation(ctx, args["id"]) if what == "simulation" else find_design(ctx, args["id"])
                label, oid = obj.name, obj.id
            except ToolError as exc:
                raise InputError(str(exc)) from None          # nothing of that id: nothing to approve or delete
            extra = ""
            items = [("Deletes", f"{what} “{label}” ({oid})")]
            if what == "design":
                n = sum(1 for s in list(ctx.store.simulations.values()) if s.design_id == oid)
                extra = f" and its {n} simulation{'s' if n != 1 else ''}" if n else ""
                if n:
                    items.append(("Also deletes", f"{n} simulation{'s' if n != 1 else ''} and their results"))
            elif ctx.store.has_result(oid):
                items.append(("Also deletes", "its stored result"))
            items.append(("Undo", "not possible"))
            return Approval("delete", f"Delete {what} “{label}”{extra}?", "This cannot be undone.", items=items)
        if tool.kind == "run" and tool.estimate is not None:
            title = _run_title(name, args, ctx)
            ctx.last_plan = None
            try:
                est = float(tool.estimate(ctx, args))
            except InputError:
                raise
            except Exception as exc:  # noqa: BLE001 - fail closed: an unknown cost is asked about
                return Approval("run", f"{title}: cost unknown",
                                f"Aerolume could not estimate this run ({exc}); approve to run it anyway.", None,
                                what=run_detail(name, args), items=run_items(name, args, ctx, None, None))
            if not math.isfinite(est):
                return Approval("run", f"{title}: cost unknown", "Aerolume could not estimate this run; approve "
                                "to run it anyway.", None, what=run_detail(name, args, ctx.last_plan),
                                items=run_items(name, args, ctx, ctx.last_plan, None))
            if est > threshold_s:
                rough = "; rough for polars and wings" if name in ("run_polar", "run_wing") else ""
                return Approval("run", f"{title}: about {fmt_seconds(est)}",
                                f"The estimated time ({fmt_seconds(est)}{rough}) is above the approval threshold "
                                f"({fmt_seconds(threshold_s)}, Preferences > Assistant).", est,
                                what=run_detail(name, args, ctx.last_plan),
                                items=run_items(name, args, ctx, ctx.last_plan, est))
        return None

    def run(self, name: str, args: dict, ctx: ToolContext) -> ToolOutcome:
        """Execute a validated call; ToolError and failures become error outcomes (never raise).

        After every call the approval gate is checked (:func:`restore_gate`): an approved
        ``run_python`` script runs in this process and could replace ``Toolbox.approval`` -- one
        line hidden after padding switched every later card off (critic round 5). A replaced gate
        is put back at once and the call becomes an error that says so."""
        tool = self.tools[name]
        t0 = time.monotonic()
        snapshot_gate()
        try:
            out = tool.handler(ctx, args)
        except ToolError as exc:
            out = ToolOutcome({"error": str(exc)}, str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - an unexpected bug becomes a readable error result
            traceback.print_exc()
            out = ToolOutcome({"error": f"{type(exc).__name__}: {exc}"}, f"{type(exc).__name__}: {exc}",
                              is_error=True)
        changed = restore_gate(self)
        if changed:
            msg = (f"This call changed Aerolume's approval rules ({', '.join(changed)}); they were restored. "
                   "Check what else it did before trusting its results.")
            out = ToolOutcome({"error": msg}, msg, is_error=True)
        if ctx.log is not None:
            if name not in ("record_conclusion", "suggest_settings"):
                # The model's own conclusion is what the number check examines: its inputs are never a
                # source, or a figure written once in record_conclusion would "trace" in the final answer
                # (a suggested value is the model's own number too, not a result).
                ctx.log.add_source(args, origin="input", tool=name)
            if name != "suggest_settings" or out.is_error:
                ctx.log.add_source(out.content, origin="result", tool=name)
            else:          # the settings the suggestion would replace are facts; the proposed values are not
                ctx.log.add_source([{k: r.get(k) for k in ("old",)} for r in out.content.get("changes", [])],
                                   origin="result", tool=name)
            status = "cancelled" if (ctx.cancel.is_set() and out.is_error) else ("error" if out.is_error else "ok")
            ctx.log.record_action(Action(_now(), name, dict(args), status, out.summary, time.monotonic() - t0,
                                         list(out.simulation_ids), [str(p) for p in out.files]))
            if out.results:
                ctx.log.add_results(out.results)
        return out


def _run_title(name: str, args: dict, ctx: ToolContext) -> str:
    """The approval card's title: what runs, by name ("Run “Fan in a room · 1 N/m”", "Polar of NACA 2412").

    run_simulation and run_grid_study name a simulation, not a design: looking up
    the (absent) design_id gave "run_simulation of " once a project held more
    than one design (round-4 probe).
    """
    if name in ("run_grid_study", "run_simulation"):
        what = "Grid study of" if name == "run_grid_study" else "Run"
        try:
            return f"{what} “{find_simulation(ctx, args.get('simulation_id', '')).name}”"
        except ToolError:
            return f"{what} {args.get('simulation_id', '')}".rstrip()
    try:
        d = find_design(ctx, args.get("design_id", "")).name
    except ToolError:
        d = str(args.get("design_id", ""))
    what = {"run_cfd": "CFD run", "run_polar": "Polar", "run_wing": "Wing analysis",
            "run_sweep": f"{len(args.get('values', []) or [])}-run sweep"}.get(name, name)
    return f"{what} of {d}"


#: (owner, attribute) -> the original object: what decides whether a call needs the user's approval and
#: runs it. Recorded on first use (the agent module is added once it is imported) and restored by
#: :func:`restore_gate` whenever a call replaced one of them.
_GATE: dict[tuple[type, str], Any] = {}


def _gate_attributes() -> list[tuple[type, str]]:
    import sys
    out = [(Toolbox, "approval"), (Toolbox, "run"), (Toolbox, "prepare")]
    agent = sys.modules.get("aerolume.ai.agent")
    if agent is not None and hasattr(agent, "Agent"):
        out.append((agent.Agent, "_execute"))
    mcp = sys.modules.get("aerolume.mcp_server")
    if mcp is not None and hasattr(mcp, "McpServer"):
        out.append((mcp.McpServer, "call_tool"))
    return out


def snapshot_gate() -> None:
    """Remember the approval gate's functions (the first time each owner is seen)."""
    for owner, attr in _gate_attributes():
        if attr in owner.__dict__:
            _GATE.setdefault((owner, attr), owner.__dict__[attr])


def restore_gate(toolbox: Toolbox | None = None) -> list[str]:
    """Put back any approval-gate function a call replaced (on the class, or shadowed on ``toolbox``);
    returns their names (empty when nothing changed)."""
    changed = []
    for (owner, attr), orig in list(_GATE.items()):
        if owner.__dict__.get(attr) is not orig:
            setattr(owner, attr, orig)
            changed.append(f"{owner.__name__}.{attr}")
    if toolbox is not None:
        for attr in ("approval", "run", "prepare"):
            if attr in vars(toolbox):
                delattr(toolbox, attr)
                changed.append(f"toolbox.{attr}")
    return changed


snapshot_gate()


def result_json(outcome: ToolOutcome) -> str:
    """The text a model receives: tool output as data (never instructions), compact JSON."""
    if outcome.is_error and "error" in outcome.content:
        payload: dict[str, Any] = {"error": outcome.content["error"]}
        extra = {k: v for k, v in outcome.content.items() if k != "error"}
        if extra:
            payload["data"] = extra
    else:
        payload = {"data": outcome.content}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


__all__ = ["PLOT_KINDS", "TOOLS", "WING_METRIC_NAMES", "Approval", "InputError", "Tool", "ToolContext", "ToolError",
           "ToolOutcome", "Toolbox", "cfd_confidence", "cfd_validity", "compact_summary", "design_validity",
           "find_design", "find_simulation", "finding_problems", "key_numbers", "plan_problems", "result_facts",
           "result_json", "round_to_stderr", "run_detail", "run_script_headless", "stderr_meaning",
           "summarize_api_result", "summarize_result", "validity_codes"]
