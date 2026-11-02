"""Grid studies of an existing CFD simulation: ``Project.grid_study_of`` and the ``run_grid_study`` tool.

Real solver runs on tiny CPU grids (a NACA 0012 at Re 200, 12 / 18 / 27 cells per
chord, two convective times: seconds each). What is checked is the bookkeeping
and the honesty data, not the physics: the coarse grid is the simulation itself
(its stored result reused), the finer ones are copies that differ only in
resolution and are reused on a re-run, the analysis is ``aerolume.analysis.gci``'s,
and the tool's result says what the GCI covers. Such short runs sit inside the
averaging noise, so "noise_limited" is an acceptable outcome; a claimed order is
only checked when the study is usable.
"""
from __future__ import annotations

import math

import pytest
import test_ai_fakes as fakes

from aerolume.ai.claims import check_answer
from aerolume.ai.experiment import ExperimentLog
from aerolume.ai.tools import InputError, Toolbox, ToolContext, result_json
from aerolume.ai.workspace import Workspace

TINY = {"re": 200, "device": "cpu", "cells_per_ref": 12, "duration": 2}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A project with a NACA 0012, one tiny CFD run of it, and a polar (seconds)."""
    from aerolume.simulation import plan as planmod
    saved = {n: getattr(planmod, n) for n in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE")}
    p = fakes.make_project(tmp_path_factory.mktemp("grid"), naca="0012")
    did = next(iter(p.raw.designs))
    base = p.run_cfd(did, alpha=4, **TINY)
    for n, v in saved.items():
        setattr(planmod, n, v)
    return p, did, base


def test_grid_study_of_reuses_the_simulation_and_its_finer_copies(tiny):
    import aerolume as al
    p, did, base = tiny
    n_before = len(p.raw.simulations)
    gs = p.grid_study_of(base.sim_id)
    assert gs.results[0].sim_id == base.sim_id and gs.results[0].reused     # the coarse grid
    assert gs.cells_per_ref == [12.0, 18.0, 27.0]
    names = [r.name for r in gs.results[1:]]
    assert all(n.endswith(f"· {c} cells/ref") for n, c in zip(names, (18, 27), strict=True)), names
    copies = [p.raw.simulations[r.sim_id] for r in gs.results[1:]]
    ref = p.raw.simulations[base.sim_id]
    for c in copies:                         # everything but the resolution is the original's
        assert c.conditions.to_dict() == ref.conditions.to_dict() and c.cfd.quality == ref.cfd.quality
        assert c.cfd.duration == ref.cfd.duration
    assert len(p.raw.simulations) == n_before + 2
    assert set(gs.gci) == {"cd", "cl"}                                                    # a lifting design
    g = gs.gci["cd"]
    assert [round(1 / lv.h) for lv in g.levels] == [27, 18, 12]
    assert g.noise_limited is not None                                                    # stderrs went in
    if g.usable:
        assert math.isfinite(g.p) and math.isfinite(g.u_num)
    again = p.grid_study_of(base.sim_id)                                                  # nothing new runs
    assert all(r.reused for r in again.results) and len(p.raw.simulations) == n_before + 2
    with pytest.raises(al.InputError, match="three different"):
        p.grid_study_of(base.sim_id, cells_per_ref=[12, 12, 18])
    pol = p.run_polar(did, re=2e5, alpha=(0, 2, 2))
    with pytest.raises(al.InputError, match="deterministic"):
        p.grid_study_of(pol.sim_id)


def test_the_tool_asks_with_the_grids_listed_and_reports_what_the_gci_covers(tiny, tmp_path):
    p, _did, base = tiny
    tb = Toolbox()
    log = ExperimentLog(p.path, "How far can the Cd of this run be trusted?", provider="test", model="t")
    ctx = ToolContext(Workspace(p), log)
    args = {"simulation_id": base.sim_id, "refinement": 1.5}
    approval = tb.approval("run_grid_study", args, ctx, 0.0)
    assert approval is not None and approval.title.startswith("Grid study of “")
    rows = dict(approval.items)
    assert rows["Grids"].startswith("3: this simulation's grid, ×1.5 and ×2.25")
    assert rows["Analysis"].startswith("Grid-convergence study") and "Estimated time" in rows
    with pytest.raises(InputError, match="between 1.3 and 2"):
        tb.approval("run_grid_study", {**args, "refinement": 5}, ctx, 0.0)
    out = tb.run("run_grid_study", args, ctx)
    assert not out.is_error, out.content
    q = out.content["quantities"]["cd"]
    assert len(q["grids"]) == 3 and q["grids"][0]["cells_per_ref"] == 27
    assert "Celik et al. 2008" in out.content["meaning"] and "modelling error" in out.content["meaning"]
    assert len(out.simulation_ids) == 3 and out.results and out.results[0]["label"].startswith("Cd")
    assert '"quantities"' in result_json(out)
    # a number of the study is a source the claims check can verify
    fine = q["fine_grid_value"]
    rep = check_answer(f"On the finest grid Cd is {fine}.", sources=log.labelled_sources(), sims=[], actions=[],
                       designs=[])
    assert rep.statements[0].status == "verified", rep.to_dict()
