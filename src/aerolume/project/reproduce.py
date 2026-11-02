"""Reproduction scripts: a stored result turned back into the code that recomputes it.

:meth:`Project.export_reproduction_script` writes a plain Python file that uses only the
public scripting API (``aerolume.api``, DESIGN.md section 8) plus the two helpers of
this module, :func:`rebuild` and :func:`check_reproduction`. The file embeds the design
and the simulation settings *as they were run* (from the result's provenance record),
creates its own project next to itself, runs the simulation through the dispatcher the
app, the CLI and the API share, and compares the new numbers with the recorded ones.
So a reviewer can rerun a figure of a paper with ``python reproduce_x.py`` and get a
pass / fail answer, on any machine with Aerolume installed.

What "the same number" means depends on the solver:

* Navier-Stokes means are time averages of a chaotic or periodic signal, and GPU
  reductions are not bitwise reproducible across devices, so a reproduced mean must lie
  within :data:`CFD_SIGMAS` combined standard errors of the recorded one, with a floor of
  :data:`CFD_RTOL_FLOOR` (0.1 %) for steady flows whose standard error is ~0;
* the polar and wing solvers are deterministic: relative :data:`EXACT_RTOL` (10⁻⁵, which
  allows for a different BLAS or CPU) or absolute :data:`EXACT_ATOL`.

The script is regenerated, not edited, when the simulation changes; it says in its
header where its data came from, and whether the design or settings were edited after
the run (then the run's own settings are used, which is the point).
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..model import CFD3DResult, CFDResult, Design, PolarResult, Simulation, WingResult, new_id, now_iso
from . import provenance as prov

#: A reproduced CFD mean agrees when within this many combined standard errors ...
CFD_SIGMAS = 3.0
#: ... or within this relative difference (steady flows have a standard error near zero).
CFD_RTOL_FLOOR = 1e-3
#: Deterministic solvers (polar, wing): relative and absolute tolerance.
EXACT_RTOL = 1e-5
EXACT_ATOL = 1e-7

_CFD_KEYS = ("cd", "cl", "cm")
_CFD3D_KEYS = ("cd", "cs", "cl", "c_roll", "cm", "cn")
_SWEEP_KEYS = ("cl", "cd", "cm")


# ---------------------------------------------------------------------------
# expected values
# ---------------------------------------------------------------------------

def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def expected_values(result: Any) -> dict[str, Any]:
    """The numbers a reproduction is checked against: ``{"kind", "values": {key: [value, stderr]}}``.

    CFD: the mean coefficients with their standard errors (plus the Nusselt number and
    heat rate of a thermal run); polars and wings: cl, cd and cm at every converged
    angle (keys ``"cl@4"``), without errors (the solvers are deterministic).
    """
    vals: dict[str, list[float]] = {}
    if isinstance(result, (CFDResult, CFD3DResult)):
        kind = "cfd" if isinstance(result, CFDResult) else "cfd3d"
        for k in (_CFD_KEYS if kind == "cfd" else _CFD3D_KEYS):
            st = getattr(result, k)
            if math.isfinite(_num(st.mean)):
                vals[k] = [_num(st.mean), _num(st.stderr)]
        nu = getattr(result, "nusselt", None)
        if nu is not None and math.isfinite(_num(nu.mean)):
            vals["nusselt"] = [_num(nu.mean), _num(nu.stderr)]
        if math.isfinite(_num(getattr(result, "heat_rate", float("nan")))):
            vals["heat_rate"] = [_num(result.heat_rate), float("nan")]
    elif isinstance(result, (PolarResult, WingResult)):
        kind = "polar" if isinstance(result, PolarResult) else "wing"
        for p in result.points:
            if not p.converged:
                continue
            for k in _SWEEP_KEYS:
                v = _num(getattr(p, k))
                if math.isfinite(v):
                    vals[f"{k}@{p.alpha:g}"] = [v, float("nan")]
    else:
        raise TypeError(f"not a result: {type(result).__name__}")
    return {"kind": kind, "values": vals}


def check_reproduction(result: Any, expected: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Compare a reproduced ``result`` with :func:`expected_values` of the original.

    Returns ``(rows, ok)``: one row per expected value (``key``, ``recorded``,
    ``recorded_stderr``, ``value``, ``stderr``, ``difference``, ``tolerance``, ``ok`` and a
    printable ``text``) and whether every value agrees. A value the new run does not have
    (an angle that did not converge this time) fails.
    """
    new = expected_values(result)
    kind = str(expected.get("kind", new["kind"]))
    stochastic = kind in ("cfd", "cfd3d")
    rows: list[dict[str, Any]] = []
    for key, (v0, se0) in (expected.get("values") or {}).items():
        v1, se1 = new["values"].get(key, [float("nan"), float("nan")])
        v0, se0, v1, se1 = _num(v0), _num(se0), _num(v1), _num(se1)
        diff = v1 - v0
        if stochastic:
            se = math.hypot(se0 if math.isfinite(se0) else 0.0, se1 if math.isfinite(se1) else 0.0)
            tol = max(CFD_SIGMAS * se, CFD_RTOL_FLOOR * abs(v0), 1e-12)
            how = f"±{tol:.3g} (3 s.e. or 0.1 %)"
        else:
            tol = max(EXACT_RTOL * abs(v0), EXACT_ATOL)
            how = f"±{tol:.3g}"
        ok = bool(math.isfinite(v1) and abs(diff) <= tol)
        text = (f"{key:>12}  recorded {v0:.6g}  now {v1:.6g}  difference {diff:+.3g}  tolerance {how}  "
                f"{'ok' if ok else 'DIFFERENT'}") if math.isfinite(v1) else \
            f"{key:>12}  recorded {v0:.6g}  now: missing  DIFFERENT"
        rows.append({"key": key, "recorded": v0, "recorded_stderr": se0, "value": v1, "stderr": se1,
                     "difference": diff, "tolerance": tol, "ok": ok, "text": text})
    return rows, bool(rows) and all(r["ok"] for r in rows)


# ---------------------------------------------------------------------------
# rebuilding a design and a simulation inside an API project
# ---------------------------------------------------------------------------

_DESIGN_REF = re.compile(r"^design:(.+)$")


def _wing_refs(design_dict: Mapping[str, Any]) -> list[str]:
    wing = design_dict.get("wing") or {}
    out = []
    for s in wing.get("sections") or []:
        m = _DESIGN_REF.match(str(s.get("airfoil", "")))
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def rebuild(project: Any, designs: list[Mapping[str, Any]], simulation: Mapping[str, Any], *,
            mesh_file: str | os.PathLike | None = None) -> Simulation:
    """Add the designs and the simulation of a reproduction script to an API ``project``.

    ``designs`` are ``Design.to_dict()`` documents, the simulated design first and any
    airfoil designs its wing sections name ("design:<id>") after it; their ids are mapped
    to the ids they get in ``project``. ``mesh_file`` holds the 3D mesh arrays of the first
    design (``Design.mesh_arrays()``). A simulation with exactly these settings that is
    already in the project (the script run before) is reused, so running the script
    twice re-runs one simulation instead of piling up copies. Returns the model
    simulation, ready for ``project.run(sim.id, rerun=True)``.
    """
    from .. import model as M
    if not designs:
        raise ValueError("a reproduction needs the design it simulates")
    id_map: dict[str, str] = {}
    built: list[Design] = []
    # airfoils first, so the wing's references can be remapped before it is added
    order = list(range(1, len(designs))) + [0]
    for i in order:
        doc = json.loads(json.dumps(designs[i]))            # a private copy
        old_id = str(doc.get("id") or "")
        if doc.get("wing"):
            for s in doc["wing"].get("sections") or []:
                m = _DESIGN_REF.match(str(s.get("airfoil", "")))
                if m and m.group(1) in id_map:
                    s["airfoil"] = f"design:{id_map[m.group(1)]}"
        d = M.Design.from_dict(doc)
        if i == 0 and mesh_file is not None:
            with np.load(mesh_file, allow_pickle=False) as z:
                d.attach_mesh_arrays({k: z[k] for k in z.files})
        d.id = new_id()
        added = project.add_design(d)                       # reused when name + geometry match
        raw = getattr(added, "raw", added)
        id_map[old_id] = raw.id
        built.append(raw)
    target = built[-1]
    sim = M.Simulation.from_dict(json.loads(json.dumps(simulation)))
    sim.design_id = target.id
    key = prov.settings_digest(sim)
    for s in project.raw.simulations_of(target.id):
        if prov.settings_digest(s) == key:
            return s
    sim.id = new_id()
    sim.status, sim.finished, sim.error = "new", None, None
    sim.created = now_iso()
    project.host.add_simulation(project.raw, sim)
    project.save()
    return project.raw.simulations[sim.id]


# ---------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------

def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower().encode("ascii", "ignore").decode()).strip("_")
    return (s[:n].rstrip("_") or "simulation")


def _pretty(obj: Any, level: int = 0, width: int = 100) -> str:
    """JSON with objects indented one key per line and flat lists (numbers, or rows of
    numbers such as polygon points) packed onto wrapped lines: a 200-point outline takes
    about 40 lines instead of 1,000."""
    pad, inner = " " * level, " " * (level + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{inner}{json.dumps(str(k), ensure_ascii=True)}: {_pretty(v, level + 1, width)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        def scalar(x: Any) -> bool:
            return not isinstance(x, (dict, list))
        flat = all(scalar(x) for x in obj) or all(isinstance(x, list) and all(scalar(y) for y in x) for x in obj)
        if not flat:
            return "[\n" + ",\n".join(inner + _pretty(x, level + 1, width) for x in obj) + "\n" + pad + "]"
        parts = [json.dumps(x, ensure_ascii=True, allow_nan=True, separators=(",", ":")) for x in obj]
        lines: list[str] = []
        cur = ""
        for t in parts:
            if cur and len(inner) + len(cur) + len(t) + 1 > width:
                lines.append(cur + ",")
                cur = t
            else:
                cur = f"{cur},{t}" if cur else t
        lines.append(cur)
        if len(lines) == 1 and len(pad) + len(lines[0]) + 2 <= width:
            return "[" + lines[0] + "]"
        return "[\n" + "\n".join(inner + ln for ln in lines) + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=True, allow_nan=True)


def _literal(obj: Any) -> str:
    """``obj`` as a Python expression that rebuilds it: readable JSON in a raw string when safe."""
    text = _pretty(obj)
    if "'''" in text:
        return f"json.loads({json.dumps(obj, ensure_ascii=True)!r})"
    return f"json.loads(r'''{text}''')"


def _fmt_seconds(s: Any) -> str:
    v = _num(s)
    if not math.isfinite(v):
        return "run time not recorded"
    if v < 10:
        return f"{v:.1f} s"
    return f"{v:.0f} s" if v < 600 else f"{v / 60:.1f} min"


def write_reproduction_script(project: Any, sim_id: str, path: str | os.PathLike | None = None) -> Path:
    """Implementation of :meth:`Project.export_reproduction_script` (see there)."""
    from .fileio import atomic_writer
    if sim_id not in project.simulations:
        raise KeyError(f"unknown simulation {sim_id!r}")
    sim = project.simulations[sim_id]
    rec = project.provenance(sim_id) or {}
    result = project.load_result(sim_id, arrays=False)
    notes: list[str] = []
    cur_design = project.designs.get(sim.design_id)
    sim_doc = rec.get("simulation") if isinstance(rec.get("simulation"), dict) else None
    if sim_doc is None:
        sim_doc = sim.to_dict()
        notes.append("No provenance record was saved with this result (it predates them): the script uses the "
                     "simulation's current settings.")
    elif prov.settings_digest(sim_doc) != prov.settings_digest(sim):
        notes.append("The simulation was edited after this result was computed; the script uses the settings "
                     "as they were run.")
    design_doc = rec.get("design") if isinstance(rec.get("design"), dict) else None
    mesh_design: Design | None = None
    if cur_design is not None and int(getattr(cur_design, "dimension", 2) or 2) == 3:
        # the record keeps a 3D design without its mesh arrays; take them from the project
        mesh_design = cur_design
        if rec.get("design_hash") and prov.design_digest(cur_design) != rec.get("design_hash"):
            notes.append("The 3D design was edited after this result was computed and the record holds no copy of "
                         "the old mesh: the script uses the current geometry, so a difference may be the edit.")
        design_doc = cur_design.to_dict()
    elif design_doc is None:
        if cur_design is None:
            raise KeyError(f"the design of simulation {sim_id!r} is not in the project")
        design_doc = cur_design.to_dict()
    elif cur_design is not None and rec.get("design_hash") and prov.design_digest(cur_design) != rec["design_hash"]:
        notes.append("The design was edited after this result was computed; the script uses the geometry as run.")
    designs = [design_doc]
    for ref in _wing_refs(design_doc):
        d = project.designs.get(ref)
        if d is None:
            notes.append(f"A wing section names airfoil design {ref}, which is no longer in the project: the "
                         "script cannot rebuild it.")
            continue
        designs.append(d.to_dict())
    expected = expected_values(result) if result is not None else {"kind": "", "values": {}}
    if result is None:
        notes.append("The simulation has no stored result: the script runs it but has nothing to compare with.")

    if path is None:
        base = project.scripts_dir / f"reproduce_{_slug(sim.name)}_{sim_id[:6]}.py"
    else:
        base = Path(path)
        if base.suffix.lower() != ".py":
            base = base.with_suffix(".py")
    mesh_name = None
    if mesh_design is not None and mesh_design.meshes:
        mesh_name = f"{base.stem}_mesh.npz"
        with atomic_writer(base.with_name(mesh_name), "wb") as fh:
            np.savez_compressed(fh, **mesh_design.mesh_arrays())
    run = rec.get("run") or {}
    plan = rec.get("plan") or {}
    sw = rec.get("software") or {}
    src = "the result's provenance record" if rec else "the project's current settings"
    hash_line = f"Configuration hash {rec.get('config_hash') or prov.config_hash(cur_design, sim)}"
    if plan.get("solver_hash"):
        hash_line += f",\nsolver-settings hash {plan['solver_hash']}"
    finished = str(run.get("finished") or rec.get("saved") or sim.finished or "—").replace("T", " ")
    deterministic = isinstance(result, (PolarResult, WingResult)) or sim.analysis in ("polar", "wing")
    device = run.get("device") or getattr(result, "device", "") or ("CPU" if deterministic else "—")
    precision = run.get("precision") or getattr(result, "precision", "") or ("float64" if deterministic else "—")
    header_notes = "".join(f"\nNote: {n}" for n in notes)
    proj_dir = f"{base.stem}.aerolume"
    tol_text = ("time-averaged CFD values must agree within three combined standard errors (or 0.1 %)"
                if expected["kind"] in ("cfd", "cfd3d") else
                "the deterministic polar / wing values must agree to 1e-5 (relative)")
    text = f'''"""Reproduce “{sim.name}” of project “{project.name}” (simulation {sim_id}).

Generated by Aerolume {__version__} on {now_iso().replace("T", " ")[:16]} from {src}.
Recorded run: {device}, {precision}, \
{_fmt_seconds(run.get("wall_seconds", getattr(result, "wall_seconds", None)))}, saved {finished}
(Aerolume {sw.get("aerolume", "—")}, Python {sw.get("python", "—")}).
{hash_line}.

Run it with plain Python (Aerolume installed):   python {base.name}
or with the command line:                         aerolume-cli script {base.name}
or open it in the app's Script console and press Run.

It makes its own project next to this file ({proj_dir}), rebuilds the design and the
simulation exactly as they were run, runs the simulation through the dispatcher the app
uses, and compares the new numbers with the recorded ones: {tol_text}.
The exit code is 0 when everything agrees and 1 otherwise.{header_notes}
"""
from __future__ import annotations

import json
from pathlib import Path

import aerolume as al
from aerolume.project.reproduce import check_reproduction, rebuild

#: the simulated design first, then any airfoil designs its wing sections name
DESIGNS = {_literal(designs)}

SIMULATION = {_literal(sim_doc)}

#: recorded values: key -> [value, standard error]
EXPECTED = {_literal(expected)}

MESH_FILE = {mesh_name!r}


def main() -> int:
    here = Path(globals().get("__file__") or ".").resolve().parent
    project = al.new_project(here / {proj_dir!r}, exist_ok=True)
    sim = rebuild(project, DESIGNS, SIMULATION, mesh_file=here / MESH_FILE if MESH_FILE else None)
    result = project.run(sim.id, rerun=True, verbose=True)
    rows, ok = check_reproduction(result.raw, EXPECTED)
    for row in rows:
        print(row["text"])
    if not rows:
        print("Nothing recorded to compare with.")
        return 0
    print("Reproduced: every value agrees." if ok else "NOT reproduced: see the rows marked DIFFERENT.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
    with atomic_writer(base, "w") as fh:
        fh.write(text)
    return base


__all__ = ["CFD_RTOL_FLOOR", "CFD_SIGMAS", "EXACT_ATOL", "EXACT_RTOL", "check_reproduction", "expected_values",
           "rebuild", "write_reproduction_script"]
