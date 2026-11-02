"""Research exports: solver diagnostics, fan reports, provenance records and grid studies.

The files a paper's supplementary material needs next to the result tables of
:mod:`.tables`: the time series that show a run behaved (divergence, change norms, CFL,
time step), the fan table, the record of what produced a result (software, settings,
hashes; ``project/provenance.py``) and the grid-convergence table with observed orders
and GCI bands (``analysis/gci.py``). CSV conventions as in :mod:`.tables` (one header
row, ``repr`` floats, ``nan`` for missing values); every file is written atomically.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..project.fileio import atomic_writer
from .tables import _write_rows

#: History series that are solver diagnostics (besides any ``diag_*`` key): the
#: AeroJAX-style health record of DESIGN.md 10.4.
DIAGNOSTIC_SERIES = ("divergence_rms", "divergence_max", "cfl", "dt", "steps_per_s", "pressure_residual")
_DIAG = re.compile(r"^diag_.+|^change_.+")


def diagnostic_keys(history: Mapping[str, Any]) -> list[str]:
    """The history keys that are solver diagnostics, in stored order."""
    return [k for k in history if k != "t" and (k in DIAGNOSTIC_SERIES or _DIAG.match(k))]


def diagnostics_csv(result: Any, path: str | os.PathLike) -> Path:
    """Solver diagnostics over time, one row per recorded step.

    Taken from the run's diagnostics table (``result.diagnostics_history``, the solver's
    ``DIAG_COLUMNS``: step, t, dt, cfl, divergence, pressure residual, change norms), or
    else from diagnostic series in the force history (:data:`DIAGNOSTIC_SERIES` and any
    ``diag_*`` / ``change_*`` key, written as ``t,<series...>``). A result with only the
    final values (``result.diagnostics``) gets ``quantity,value`` rows instead, so the
    file always says what the run reported. ``ValueError`` when there is none of these.
    """
    table = getattr(result, "diagnostics_history", None) or {}
    if table:
        keys = list(table)
        cols = [np.asarray(table[k], np.float64).ravel() for k in keys]
        if any(c.size != cols[0].size for c in cols):
            raise ValueError("diagnostics columns differ in length")
        return _write_rows(path, keys, zip(*(c.tolist() for c in cols), strict=True))
    hist = getattr(result, "history", None) or {}
    keys = diagnostic_keys(hist)
    if keys and "t" in hist:
        cols = [np.asarray(hist["t"], np.float64).ravel()] + [np.asarray(hist[k], np.float64).ravel() for k in keys]
        n = cols[0].size
        if any(c.size != n for c in cols):
            raise ValueError("diagnostic series differ in length from the time column")
        return _write_rows(path, ["t", *keys], zip(*(c.tolist() for c in cols), strict=True))
    diag = getattr(result, "diagnostics", None) or {}
    if not diag:
        raise ValueError("the result records no solver diagnostics (CFDSettings.record_diagnostics was off, or the "
                         "result predates them)")
    return _write_rows(path, ("quantity", "value"), [(k, v) for k, v in diag.items()])


def fans_csv(result: Any, path: str | os.PathLike) -> Path:
    """One row per fan: ``name,mode,thrust,thrust_stderr,volume_flow,induced_velocity,ideal_power,
    pressure_jump,target`` (SI; per metre of span in 2D). ``ValueError`` without fans."""
    from ..analysis.physics import fan_entries
    fans = fan_entries(result)
    if not fans:
        raise ValueError("the result has no fans")
    keys = ("thrust", "thrust_stderr", "volume_flow", "induced_velocity", "ideal_power", "pressure_jump", "target")
    return _write_rows(path, ("name", "mode", *keys), [(f["name"], f["mode"], *(f[k] for k in keys)) for f in fans])


def provenance_json(record: Mapping[str, Any] | None, path: str | os.PathLike) -> Path:
    """Write a provenance record (``Project.provenance(sim_id)``) as indented JSON.

    ``ValueError`` for a missing record (a result saved before provenance existed),
    rather than an empty file that would look like a record.
    """
    if not record:
        raise ValueError("no provenance record: the result was saved before Aerolume recorded provenance")
    path = Path(path)
    with atomic_writer(path, "w") as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False, allow_nan=True)
        fh.write("\n")
    return path


def grid_study_csv(study: Any, path: str | os.PathLike) -> Path:
    """A grid-convergence study (``analysis.gci.GridStudy``), one row per quantity and grid.

    Columns: ``quantity,grid,sim_id,h,cells,value,stderr`` then, on every row of the
    quantity, its study numbers ``r21,r32,convergence,p,phi_ext,gci_fine,u_num,u_stat,
    u_total,asymptotic,noise_limited`` (``h`` in reference lengths per cell; ``grid`` 1 =
    finest).
    """
    rows: list[tuple] = []
    for q in study.quantities.values():
        tail = (q.r21, q.r32, q.convergence, q.p, q.phi_ext, q.gci_fine, q.u_num, q.u_stat, q.u_total,
                "" if q.asymptotic is None else q.asymptotic, "" if q.noise_limited is None else q.noise_limited)
        for i, lv in enumerate(q.levels, 1):
            rows.append((q.quantity, i, lv.sim_id, lv.h, lv.cells, lv.value, lv.stderr, *tail))
    header = ("quantity", "grid", "sim_id", "h", "cells", "value", "stderr", "r21", "r32", "convergence", "p",
              "phi_ext", "gci_fine", "u_num", "u_stat", "u_total", "asymptotic", "noise_limited")
    return _write_rows(path, header, rows)


__all__ = ["DIAGNOSTIC_SERIES", "diagnostic_keys", "diagnostics_csv", "fans_csv", "grid_study_csv",
           "provenance_json"]
