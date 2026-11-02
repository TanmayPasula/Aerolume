"""Research exports (export/research.py) and the placed-physics rows of result_csv."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aerolume import export
from aerolume.analysis import gci as G
from aerolume.model import CFD3DResult, CFDResult, Stat


def _thermal_result() -> CFDResult:
    t = np.linspace(0, 10, 50)
    return CFDResult(sim_id="h", cd=Stat(1.0, 0.0, 1, 1, 0.01), cl=Stat(0, 0, 0, 0, 0.001),
                     cm=Stat(0, 0, 0, 0, 0.001), nusselt=Stat(3.3, 0.02, 3.2, 3.4, 0.004), heat_rate=4.4,
                     fans_report=[{"name": "F1", "mode": "thrust", "thrust": 2.0, "flow": 1.5, "power": 11.0}],
                     diagnostics={"divergence_rms": 2e-7, "cfl_max": 0.5, "heat_in_bodies": 4.4,
                                  "heat_out_outlet": 4.39},
                     history={"t": t, "cd": np.ones(50), "diag_div_rms": np.full(50, 1e-7), "dt": np.full(50, 0.01),
                              "change_u_rel": np.linspace(1e-2, 1e-6, 50)})


def test_result_csv_has_physics_rows(tmp_path):
    path = export.result_csv(_thermal_result(), tmp_path / "r.csv")
    _, rows = export.read_csv(path)
    by = {r[0]: r for r in rows}
    assert float(by["nusselt"][1]) == 3.3 and float(by["nusselt"][2]) == 0.004
    assert by["heat_rate"][3] == "W/m" and float(by["heat_rate"][1]) == 4.4
    assert float(by["fan:0:thrust"][1]) == 2.0 and by["fan:0:volume_flow"][3] == "m^2/s"
    assert float(by["heat_balance:residual"][1]) == pytest.approx(0.01)
    assert float(by["diag:cfl_max"][1]) == 0.5
    plain = CFDResult(sim_id="p")
    _, rows = export.read_csv(export.result_csv(plain, tmp_path / "p.csv"))
    assert not any(r[0].startswith(("nusselt", "heat_", "fan:", "diag:")) for r in rows)
    r3 = CFD3DResult(sim_id="q", heat_rate=10.0)
    _, rows = export.read_csv(export.result_csv(r3, tmp_path / "q.csv"))
    assert next(r for r in rows if r[0] == "heat_rate")[3] == "W"


def test_diagnostics_csv_series_or_final_values(tmp_path):
    r = _thermal_result()
    header, rows = export.read_csv(export.diagnostics_csv(r, tmp_path / "d.csv"))
    assert header == ["t", "diag_div_rms", "dt", "change_u_rel"] and len(rows) == 50
    r.history = {"t": r.history["t"], "cd": r.history["cd"]}
    header, rows = export.read_csv(export.diagnostics_csv(r, tmp_path / "e.csv"))
    assert header == ["quantity", "value"] and ["cfl_max", "0.5"] in rows
    with pytest.raises(ValueError, match="no solver diagnostics"):
        export.diagnostics_csv(CFDResult(sim_id="x"), tmp_path / "f.csv")


def test_diagnostics_table_is_preferred_and_kept_off_the_history(tmp_path):
    r = _thermal_result()
    r.diagnostics_history = {"step": np.arange(0, 50, 10.0), "t": np.linspace(0, 10, 5),
                             "div_rms": np.full(5, 1e-7), "du_rel": np.logspace(-2, -6, 5)}
    r.history["div_sampled"] = np.zeros(5)                       # another clock: not in the force CSV
    header, rows = export.read_csv(export.diagnostics_csv(r, tmp_path / "d.csv"))
    assert header == ["step", "t", "div_rms", "du_rel"] and len(rows) == 5
    header, rows = export.read_csv(export.history_csv(r, tmp_path / "h.csv"))
    assert "div_sampled" not in header and len(rows) == 50


def test_fans_csv(tmp_path):
    header, rows = export.read_csv(export.fans_csv(_thermal_result(), tmp_path / "f.csv"))
    assert header[:4] == ["name", "mode", "thrust", "thrust_stderr"]
    assert rows[0][0] == "F1" and float(rows[0][2]) == 2.0 and rows[0][3] == "nan"
    with pytest.raises(ValueError):
        export.fans_csv(CFDResult(sim_id="x"), tmp_path / "g.csv")


def test_provenance_json(tmp_path):
    rec = {"format": "aerolume-provenance", "config_hash": "ab", "run": {"wall_seconds": float("nan")}}
    path = export.provenance_json(rec, tmp_path / "p.json")
    assert json.loads(path.read_text(encoding="utf-8"))["config_hash"] == "ab"
    with pytest.raises(ValueError, match="before Aerolume recorded provenance"):
        export.provenance_json(None, tmp_path / "q.json")


def test_grid_study_csv(tmp_path):
    rs = [CFDResult(sim_id=s, cd=Stat(v, 0, v, v, 1e-4), cl=Stat(0.2, 0, 0.2, 0.2, 1e-4), cm=Stat(0, 0, 0, 0, 0),
                    re_sim=100.0, grid=(int(c * 20), int(c * 10)), cells_per_ref=c)
          for s, c, v in (("f", 64.0, 1.50), ("m", 32.0, 1.52), ("c", 16.0, 1.60))]
    st = G.grid_study(rs)
    header, rows = export.read_csv(export.grid_study_csv(st, tmp_path / "g.csv"))
    assert header[:7] == ["quantity", "grid", "sim_id", "h", "cells", "value", "stderr"]
    cd_rows = [r for r in rows if r[0] == "Cd"]
    assert [r[2] for r in cd_rows] == ["f", "m", "c"] and cd_rows[0][1] == "1"
    p = float(cd_rows[0][header.index("p")])
    assert p == pytest.approx(math.log2(4.0))
    assert cd_rows[0][header.index("convergence")] == "monotone"
