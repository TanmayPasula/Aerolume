"""Pieces shared by the 2D (:mod:`.content`) and 3D (:mod:`.content3d`) section builders.

:class:`Entry` is one simulation as the report sees it, with the type tests every
section needs: a 2D Navier–Stokes result, an airfoil polar, a 3D Navier–Stokes
result or a finite-wing sweep. The rest are the small formatting rules that must
read the same in every section: a standard error that is only an estimate is
marked "~", a difference within the noise is marked "~" and never coloured, and
so on.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..model import CFD3DResult, CFDResult, PolarResult, WingResult
from . import fmt
from .document import Cell, FigureBlock, Heading


@dataclass
class Entry:
    """One simulation as the report sees it."""
    sim_id: str
    sim: Any
    design: Any | None
    result: Any | None
    error: str | None = None
    #: label used throughout the report; set by ``content.load_entries`` when simulation
    #: names alone would be ambiguous (two designs each with a "CFD · 30 m/s")
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or str(getattr(self.sim, "name", self.sim_id))

    @property
    def is_cfd(self) -> bool:
        """A 2D Navier–Stokes result (per unit span)."""
        r = self.result
        if isinstance(r, (CFD3DResult, WingResult)):
            return False
        return r is not None and hasattr(r, "cd") and hasattr(r.cd, "mean")

    @property
    def is_polar(self) -> bool:
        """A 2D airfoil polar."""
        r = self.result
        if isinstance(r, (WingResult, CFD3DResult)):
            return False
        return r is not None and hasattr(r, "points") and hasattr(r, "table")

    @property
    def is_cfd3d(self) -> bool:
        """A 3D Navier–Stokes result (whole body, newtons)."""
        return isinstance(self.result, CFD3DResult)

    @property
    def is_wing(self) -> bool:
        """A finite-wing sweep (vortex lattice / lifting line)."""
        return isinstance(self.result, WingResult)

    @property
    def dimension(self) -> int:
        """3 for 3D designs and results, else 2 (decides wording such as "per metre of span")."""
        if isinstance(self.result, (CFD3DResult, WingResult)):
            return 3
        if isinstance(self.result, (CFDResult, PolarResult)):
            return 2
        return 3 if int(getattr(self.design, "dimension", 2) or 2) == 3 or self.analysis == "wing" else 2

    @property
    def kind(self) -> str:
        return str(getattr(self.design, "kind", "generic"))

    @property
    def analysis(self) -> str:
        return str(getattr(self.sim, "analysis", "cfd"))

    @property
    def type_label(self) -> str:
        """Short analysis type for overview tables: "CFD", "CFD 3D", "Polar", "Wing"."""
        if self.analysis == "wing":
            return "Wing"
        if self.analysis == "polar":
            return "Polar"
        return "CFD 3D" if self.dimension == 3 else "CFD"

    @property
    def ground(self) -> bool:
        g = getattr(self.sim, "ground", None)
        return bool(getattr(g, "enabled", False)) or self.kind == "vehicle"


def fig(builder: Any, *args: Any, caption: str = "", size: tuple[float, float] | None = None,
        width: float = 1.0, **kw: Any) -> FigureBlock:
    """A lazily built figure (the renderer calls it with ``style`` and ``figsize``)."""
    return FigureBlock(functools.partial(builder, *args, **kw), caption=caption, size=size, width=width)


def sub(blocks: list, entries: Sequence[Entry], e: Entry, sec_no: str, k: int) -> None:
    """Subsection heading per simulation, only when a section covers several."""
    if len(entries) > 1:
        blocks.append(Heading(e.name, level=2, number=f"{sec_no}.{k}"))


def fluid(e: Entry) -> Any:
    return e.sim.conditions.fluid


def dyn_pressure(e: Entry) -> float:
    c = e.sim.conditions
    return 0.5 * c.fluid.density * c.speed ** 2


def rel_err(stat: Any) -> float:
    try:
        return abs(float(stat.stderr) / float(stat.mean)) if stat.mean else float("nan")
    except (TypeError, ValueError, ZeroDivisionError, AttributeError):
        return float("nan")


#: Relative standard error below which a mean is "ranking grade" (green), and above which it is
#: poor (red); the drag's in the force sections and the Nusselt number's for runs without forces.
STDERR_OK = 0.01
STDERR_WARN = 0.03


#: Prefix of a standard error that is only an estimate (``Stat.stderr_reliable`` False:
#: too few independent samples for batch means); explained under the tables that use it.
ROUGH = "~"
ROUGH_NOTE = ("~ marks a standard error estimated from too few effectively independent samples: a rough "
              "figure, not a measured error bar; run longer before relying on differences of that size.")


def rough(stat: Any) -> bool:
    return not bool(getattr(stat, "stderr_reliable", True)) and fmt.finite(getattr(stat, "stderr", float("nan")))


def err_text(stat: Any) -> str:
    """The standard error for a table cell, marked ~ when it is only an estimate."""
    text = fmt.err_only(stat.stderr)
    return ROUGH + text if rough(stat) and text != fmt.DASH else text


def pm_text(stat: Any) -> str:
    """'mean ± stderr' with the same ~ marking."""
    text = fmt.pm(stat.mean, stat.stderr)
    return text.replace("± ", "± " + ROUGH) if rough(stat) else text


def turbulence_text(code: str, fallback: str = "") -> str:
    c = (code or fallback or "").lower()
    if not c or c == "auto":
        return "chosen automatically"
    if c in ("laminar", "none"):
        return "none: laminar, all scales resolved (direct simulation)"
    if c.startswith("les"):
        model = c.split("-", 1)[1] if "-" in c else ""
        names = {"wale": "WALE", "smagorinsky": "Smagorinsky", "vreman": "Vreman"}
        return f"large-eddy simulation, {names.get(model, model or 'sub-grid')} model"
    return c


def sci(x: float) -> str:
    if not fmt.finite(x):
        return fmt.DASH
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    if -3 <= e <= 4:
        return fmt.num(x, 4)
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{x / 10 ** e:.4g} × 10{str(e).translate(sup)}"


def compare_rows(results: Mapping[str, Any]) -> list[dict] | None:
    try:
        from ..analysis.compare import compare_rows as rows
        return rows(results)
    except Exception:  # noqa: BLE001 - optional sibling module
        return None


def comparison_warnings(results: Mapping[str, Any], conditions: Mapping[str, Any]) -> list[str]:
    """Why compared runs differ for reasons other than the design (shared with the Compare page).

    ``conditions`` maps sim id -> the Simulation (preferred: closed rooms are then judged by their
    driving inputs, not by a reference "speed") or its FlowConditions."""
    try:
        from ..analysis.compare import comparison_warnings as warn
        return warn(results, conditions)
    except Exception:  # noqa: BLE001 - optional sibling module
        return []


def delta_cell(cell: dict | None, better: str | None = None) -> Cell:
    """A Δ% table cell from an ``analysis.compare_rows`` cell: green / red only when significant."""
    if cell and cell.get("is_baseline"):
        return Cell("baseline", "neutral")
    if not cell or cell.get("pct") is None:
        return Cell(fmt.DASH, "neutral")
    txt = fmt.pct(cell["pct"]) if abs(cell["pct"]) >= 0.05 else "0.0 %"
    sig = cell.get("significant")
    if sig is False:
        return Cell(txt + " ~", "neutral")
    imp = cell.get("improved")
    tone = "neutral" if imp is None else ("ok" if imp else "bad")
    return Cell(txt + (" ✱" if sig else ""), tone)
