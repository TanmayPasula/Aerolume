"""Pre-release (0.1.0) integration fixes (the contract requests the round-4 fixers could not apply themselves).

* the Results header says how much of the heat input is still being stored when it is a sizeable
  share, next to a "Converged" verdict that judges only the monitored quantities (the heated
  cylinder example read "Converged" with 9 % of its heat still warming the wake);
* the Setup inspector is never narrower than its form, so no field or button is cut off at
  1280 x 800 beside the Assistant, and the Estimate / Planner strip stacks when it is narrow;
* the placed-item buttons are never narrower than their own text.
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from aerolume.model import CFD3DResult, CFDResult, Simulation, Stat

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    theme.apply(app, "light")
    return app


def pump(app, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def _heated(storage: float, *, converged: bool = True, cls=CFDResult):
    """A tunnel run with 2.699 W/m from its body, ``storage`` W/m of it still warming the domain."""
    diag = {"heat_in_bodies": 2.699, "heat_in_inlet": 0.0, "heat_out_outlet": 2.699 - storage,
            "heat_storage_rate": storage}
    kw = {"grid": (128, 96, 64)} if cls is CFD3DResult else {"grid": (128, 96)}
    return cls(sim_id="s", converged=converged, re_real=40.0, re_sim=40.0, heat_rate=2.699,
               nusselt=Stat(mean=3.3, stderr=0.01), diagnostics=diag, **kw)


@pytest.mark.parametrize("storage, text, unsettled", [
    (0.2482, "9 % of the heat still being stored", False),     # the heated-cylinder example
    (0.5398, "20 % of the heat still being stored", True),     # above the runner's 10 % threshold
    (0.0270, None, False),                                      # 1 %: settled enough to say nothing
])
def test_stored_heat_fact_states_the_share_and_flags_it_above_the_runners_threshold(storage, text, unsettled):
    from aerolume.app.pages import results as R
    fact = R.stored_heat_fact(_heated(storage))
    if text is None:
        assert fact is None
    else:
        assert fact == (text, unsettled)


def test_stored_heat_fact_ignores_results_without_a_heat_balance():
    from aerolume.app.pages import results as R
    assert R.stored_heat_fact(CFDResult(sim_id="s")) is None
    assert R.stored_heat_fact(CFDResult(sim_id="s", diagnostics={"heat_storage_rate": float("nan")})) is None


@pytest.mark.parametrize("cls", [CFDResult, CFD3DResult])
def test_converged_header_names_the_heat_still_stored_and_warns_when_unsettled(cls):
    from aerolume.app.pages import results as R
    sim = Simulation(name="hot", design_id="d")
    status, kind, facts = R.header_facts(_heated(0.2482, cls=cls), sim)
    assert status == "Converged" and kind == "good" and "9 % of the heat still being stored" in facts
    status, kind, facts = R.header_facts(_heated(0.5398, cls=cls), sim)
    assert status == "Converged" and kind == "warn" and "20 % of the heat still being stored" in facts
    status, kind, facts = R.header_facts(_heated(0.0, cls=cls), sim)
    assert kind == "good" and "still being stored" not in facts
    # a run that did not converge keeps its own verdict and still says what is stored
    status, kind, facts = R.header_facts(_heated(0.5398, converged=False, cls=cls), sim)
    assert status == "Not converged" and kind == "warn" and "20 %" in facts


def _block_design():
    import numpy as np

    from aerolume.model import Body, Design
    sq = 0.05 * np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    return Design(name="Block", kind="generic", bodies=[Body("Block", sq)], length_ref=0.1, force_ref=0.1,
                  moment_ref=(0.0, 0.0))


def _setup_page(qapp, folder, size=(900, 700)):
    from aerolume.app.pages import setup as S
    from aerolume.app.state import AppState
    from aerolume.core import fluid as fluids
    from aerolume.model import Emitter, FlowConditions
    from aerolume.project import Project

    state = AppState()
    state.set_project(Project.create(folder / "Width", "Width"))
    d = _block_design()
    state.add_design(d)
    sim = Simulation(name="run", design_id=d.id, conditions=FlowConditions(fluid=fluids.air(20.0), speed=2.0),
                     emitters=[Emitter(name="Smoke", position=(-0.3, 0.0))])
    state.add_simulation(sim, select=True)
    page = S.SetupPage(state, None)
    page.resize(*size)
    page.show()
    pump(qapp, 0.3)
    return page


def test_setup_inspector_is_never_narrower_than_its_form(qapp, tmp_path):
    from aerolume.app.pages import setup as S
    from aerolume.app.widgets.physics_editor import PlacedItemsSection

    page = _setup_page(qapp, tmp_path)
    panel = page.findChild(QtWidgets.QFrame, "setup_inspector")
    need = page._inspector_min.need()
    assert panel.minimumWidth() >= min(max(S.INSPECTOR_MIN_WIDTH, need), panel.maximumWidth())
    assert page.grid.minimumSizeHint().width() <= page.inspector_scroll.viewport().width()
    # the page's minimum size hint includes it, so the dock manager's page guard can protect the form
    assert page.minimumSizeHint().width() >= panel.minimumWidth()
    for sec in page.findChildren(PlacedItemsSection):
        for b in (sec.add_button, sec.dup_button, sec.remove_button):
            assert b.minimumWidth() >= b.sizeHint().width() - 1, b.text()
    page.close()


def test_estimate_and_planner_stack_on_a_narrow_strip_and_return_side_by_side(qapp, tmp_path):
    from aerolume.app.pages import setup as S

    page = _setup_page(qapp, tmp_path, (1600, 1000))
    assert page._info.width() > S.INFO_STACK_WIDTH + 20 and not page.info_stacked()
    # the strip's width is the page's minus the inspector's: about 370 px at 800 px, as in the
    # Analysis workspace at 1280 x 800
    page.resize(800, 700)
    pump(qapp, 0.2)
    assert page._info.width() < S.INFO_STACK_WIDTH
    assert page.info_stacked() and page._info_hdiv.isVisibleTo(page) and not page._info_vdiv.isVisibleTo(page)
    page.resize(1600, 1000)
    pump(qapp, 0.2)
    assert not page.info_stacked() and page._info_vdiv.isVisibleTo(page) and not page._info_hdiv.isVisibleTo(page)
    # inside the 20 px band above the threshold a stacked strip stays stacked (no flicker while dragging)
    page._info_box.setDirection(QtWidgets.QBoxLayout.Direction.TopToBottom)
    page.resize(page.width() - page._info.width() + S.INFO_STACK_WIDTH + 10, 1000)
    pump(qapp, 0.2)
    assert S.INFO_STACK_WIDTH <= page._info.width() < S.INFO_STACK_WIDTH + 20 and page.info_stacked()
    page.close()
