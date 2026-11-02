"""Flow visualisation of 3D runs in Setup: the smoke settings, the emitter layout and its preview.

DESIGN.md 7.13: a 3D CFD run releases smoke (a passive dye) upstream of the body
from a rake, a grid of nozzles or a plane band. The Setup page edits that choice
(Flow visualisation group), shows where the emitters stand in the 3D case
preview, and states the nozzle count in the Estimate. The pure parts - the
automatic layout rule, the planner override, the storage adapter and the
free-stream arrow placement - are tested on their own; the page is tested with
the real planner on the ``examples/geometry3d`` samples.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from aerolume.model import Simulation, WingSection, WingSpec

pytestmark = pytest.mark.gui

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "geometry3d"
CAR_BOX = (0.0, 4.5, -0.91, 0.91, 0.0, 1.41)          # a car's wind-frame box [m]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    from aerolume.app import theme
    theme.apply(app)
    yield app


@pytest.fixture
def state(qapp, tmp_path):
    from aerolume.app.state import AppState
    from aerolume.project import Project
    st = AppState()
    st.set_project(Project.create(tmp_path / "Smoke3D", "Smoke 3D"))
    yield st


@pytest.fixture
def planner():
    from aerolume.app.pages import setup as setup_mod
    loader = setup_mod._PlannerLoader.get()
    if not loader.wait(180):
        pytest.skip(f"planner not importable: {loader.error}")
    return loader


def _vehicle():
    import aerolume.geometry3d as g
    return g.design_from_meshes(g.read_mesh(EXAMPLES / "body_and_wheel.obj"), "Sample car", "vehicle")


def _sphere():
    import aerolume.geometry3d as g
    return g.design_from_meshes(g.read_mesh(EXAMPLES / "sphere_d100mm_binary.stl"), "Sphere", "generic",
                                units="mm")


def _wing():
    import aerolume.geometry3d as g
    return g.design_from_wing(WingSpec(sections=[WingSection(0.0, 0.3, airfoil="2412"),
                                                 WingSection(1.0, 0.18, x_le=0.03, airfoil="2412")]), "Wing")


# ---------------------------------------------------------------------------
# the rake: the Setup settings resolved by the runner's own resolver
# ---------------------------------------------------------------------------

DOMAIN = (-3.4, 11.4, -4.55, 4.55, 0.0, 5.0)            # a full (mirrored) tunnel around CAR_BOX
CELL = 0.057                                             # solver cell size [m]


def _resolve(spec):
    from aerolume.viz3d.flow import resolve_rake
    return resolve_rake(spec, CAR_BOX, DOMAIN, purpose="smoke", h=CELL)


def test_nice_round_is_the_nearest_125_value():
    from aerolume.app.widgets.geometry_canvas import nice_round
    assert nice_round(0.27) == pytest.approx(0.2)
    assert nice_round(0.4) == pytest.approx(0.5)
    assert nice_round(0.75) == pytest.approx(1.0)
    assert nice_round(1.4) == pytest.approx(1.0) and nice_round(1.5) == pytest.approx(2.0)
    assert nice_round(0.0) == 1.0 and nice_round(float("nan")) == 1.0


def test_seed_rake_spec_writes_only_what_the_user_fixed():
    from aerolume.app.widgets.geometry_canvas import seed_rake_spec
    assert seed_rake_spec(None) == {"kind": "rake"}
    assert seed_rake_spec({"kind": "spray"}) == {"kind": "rake"}
    assert seed_rake_spec({"kind": "rake", "z_min": 0.5, "z_max": 0.9}) == {"kind": "rake", "z": 0.5}
    assert seed_rake_spec({"kind": "grid", "z_min": 1.0, "z_max": 0.2, "n": [5, 3], "spacing": 0.3}) == \
        {"kind": "grid", "z": [0.2, 1.0], "n": [5, 3], "spacing": 0.3}
    assert seed_rake_spec({"kind": "vertical", "z_min": 0.1, "z_max": 1.0, "n": 7}) == \
        {"kind": "vertical", "z": [0.1, 1.0], "n": 7}
    assert seed_rake_spec({"kind": "plane", "z_min": 0.1, "z_max": 1.0, "n": 7, "spacing": 0.2}) == \
        {"kind": "plane", "z": [0.1, 1.0]}                   # a band has no nozzles to count
    assert seed_rake_spec({"kind": "grid", "z_min": 0.2}) == {"kind": "grid"}    # half a range is no range


def test_counts_for_spacing():
    from aerolume.app.widgets.geometry_canvas import MAX_NOZZLES_PER_AXIS, counts_for_spacing
    assert counts_for_spacing("rake", (-1.0, 1.0), (0.5, 0.5), 0.25) == 9
    assert counts_for_spacing("vertical", (0.0, 0.0), (0.1, 1.1), 0.1) == 11
    assert counts_for_spacing("grid", (-1.0, 1.0), (0.0, 1.0), 0.5) == [5, 3]
    assert counts_for_spacing("rake", (-1.0, 1.0), (0.5, 0.5), 10.0) == 2          # at least the two ends
    assert counts_for_spacing("grid", (-1.0, 1.0), (0.0, 1.0), 1e-9) == [MAX_NOZZLES_PER_AXIS] * 2


def test_layout_is_the_runners_rake():
    """The preview resolves the same rake dict, body box, domain and cell size as
    ``simulation.live3d.resolve_seed_rake``: the nozzles drawn are the nozzles run."""
    from aerolume.app.widgets.geometry_canvas import smoke_rake_layout
    lay = smoke_rake_layout({"kind": "rake"}, CAR_BOX, extent3=DOMAIN, cell=CELL)
    sr = _resolve({"kind": "rake"})
    assert lay["kind"] == "rake" and np.allclose(lay["points"], sr.points) and lay["count"] == len(sr.points)
    assert lay["x"] == pytest.approx(sr.x) and sr.x < 0.0 and lay["y"] == pytest.approx(sr.y)
    assert np.allclose(lay["points"][:, 2], sr.z[0])                             # one height
    assert lay["spacing"] == pytest.approx((sr.y[1] - sr.y[0]) / (sr.n[0] - 1))
    assert lay["auto"]["z_min"] == lay["auto"]["z_max"] == pytest.approx(sr.z[0])
    assert lay["body_x0"] == 0.0 and lay["spec"] == {"kind": "rake"}
    for kind in ("vertical", "grid"):
        got = smoke_rake_layout({"kind": kind}, CAR_BOX, extent3=DOMAIN, cell=CELL)
        ref = _resolve({"kind": kind})
        assert np.allclose(got["points"], ref.points) and got["n"] == tuple(ref.n)
    vert = smoke_rake_layout({"kind": "vertical"}, CAR_BOX, extent3=DOMAIN, cell=CELL)
    assert vert["spacing"] == pytest.approx(vert["spacing_yz"][1]) and np.ptp(vert["points"][:, 1]) == 0
    plane = smoke_rake_layout({"kind": "plane"}, CAR_BOX, extent3=DOMAIN, cell=CELL)
    ref = _resolve({"kind": "plane"})
    assert plane["count"] == 0 and len(plane["points"]) == 0
    assert plane["y"] == pytest.approx(ref.y) and plane["z"] == pytest.approx(ref.z)


def test_typed_heights_and_spacing_reach_the_rake():
    from aerolume.app.widgets.geometry_canvas import smoke_rake_layout
    lay = smoke_rake_layout({"kind": "rake", "z_min": 0.3, "z_max": 0.3, "spacing": 0.25}, CAR_BOX,
                            extent3=DOMAIN, cell=CELL)
    assert np.allclose(lay["points"][:, 2], 0.3)
    assert lay["spec"]["z"] == 0.3 and lay["spec"]["n"] == round((lay["y"][1] - lay["y"][0]) / 0.25) + 1
    assert lay["spacing"] == pytest.approx(0.25, rel=0.15)        # the real pitch, a whole number of gaps
    grid = smoke_rake_layout({"kind": "grid", "z_min": 0.2, "z_max": 1.0, "spacing": 0.2}, CAR_BOX,
                             extent3=DOMAIN, cell=CELL)
    assert grid["z"] == pytest.approx((0.2, 1.0)) and grid["n"][1] == 5
    assert grid["count"] == grid["n"][0] * grid["n"][1]
    # no plan yet: a box around the body stands in for the domain, on the road for vehicles
    loose = smoke_rake_layout({"kind": "grid"}, CAR_BOX, on_road=True)
    assert loose["z"][0] > 0.0 and loose["x"] < 0.0
    assert smoke_rake_layout({"smoke": False}, CAR_BOX) is None
    assert smoke_rake_layout({"kind": "rake"}, None) is None


def test_an_unresolvable_rake_says_why(monkeypatch):
    from aerolume.app.pages.setup import smoke_fact, smoke_note
    from aerolume.app.widgets import geometry_canvas as gc

    def broken(*_a, **_k):
        raise ValueError("no domain")

    monkeypatch.setattr(gc, "_resolve", broken)
    lay = gc.smoke_rake_layout({"kind": "rake"}, CAR_BOX, extent3=DOMAIN)
    assert lay["error"].endswith("no domain") and lay["count"] == 0
    assert smoke_note(lay, {"smoke": True}).startswith("The smoke rake cannot be placed")
    assert smoke_fact(lay, {"smoke": True}) == "—"


# ---------------------------------------------------------------------------
# where the settings live (the model field is the core's; the page adapts)
# ---------------------------------------------------------------------------

WANT = {"smoke": False, "kind": "grid", "z_min": 0.2, "z_max": 1.2, "spacing": 0.25, "n": [9, 5]}
WANT_RAKE = {"kind": "grid", "z": [0.2, 1.2], "n": [9, 5], "spacing": 0.25}


def test_flow_viz_storage_adapter_shapes():
    from aerolume.app.pages.setup import FLOW_VIZ_DEFAULTS, copy_flow_viz, flow_viz_of, set_flow_viz
    # 1. a container on the simulation (None at first): written as {smoke, rake: viz3d.flow rake dict}
    s1 = SimpleNamespace(id="a", cfd=SimpleNamespace(), flow_viz=None)
    assert flow_viz_of(s1) == FLOW_VIZ_DEFAULTS
    set_flow_viz(s1, WANT)
    assert s1.flow_viz == {"smoke": False, "rake": WANT_RAKE} and flow_viz_of(s1) == WANT

    # 2. a dataclass with a nested rake dataclass (heights as z_min / z_max) and the flag called "dye"
    @dataclasses.dataclass
    class Rake:
        kind: str = "rake"
        z_min: float | None = None
        z_max: float | None = None
        spacing: float | None = None
        n: object = None

    @dataclasses.dataclass
    class FlowViz:
        dye: bool = True
        rake: Rake = dataclasses.field(default_factory=Rake)

    s2 = SimpleNamespace(id="b", cfd=SimpleNamespace(flow_viz=FlowViz()))
    set_flow_viz(s2, WANT)
    assert s2.cfd.flow_viz.dye is False and s2.cfd.flow_viz.rake == Rake("grid", 0.2, 1.2, 0.25, [9, 5])
    assert flow_viz_of(s2) == WANT
    # 3. a bool on CFDSettings with its rake dict beside it
    s3 = SimpleNamespace(id="c", cfd=SimpleNamespace(smoke=True, seed_rake=None))
    set_flow_viz(s3, WANT)
    assert s3.cfd.smoke is False and s3.cfd.seed_rake == WANT_RAKE and flow_viz_of(s3) == WANT
    # 4. an extras dict
    s4 = SimpleNamespace(id="d", cfd=SimpleNamespace(extras={}))
    set_flow_viz(s4, WANT)
    assert s4.cfd.extras["flow_viz"]["rake"] == WANT_RAKE and flow_viz_of(s4) == WANT
    # 5. no place in the model: kept for the session, copied with a duplicate, never set on the object
    a, b = Simulation(name="a", design_id="x"), Simulation(name="b", design_id="x")
    holder_free = not any(hasattr(o, n) for o in (a, a.cfd) for n in ("flow_viz", "smoke", "dye"))
    set_flow_viz(a, WANT)
    assert flow_viz_of(a) == WANT
    copy_flow_viz(a, b)
    assert flow_viz_of(b) == WANT
    if holder_free:
        assert "flow_viz" not in a.__dict__ and "flow_viz" not in a.to_dict()
    # a rake dict as viz3d.flow writes it reads back (a single height is a rake's)
    s5 = SimpleNamespace(id="e", cfd=SimpleNamespace(), flow_viz={"smoke": True, "rake": {"kind": "rake", "z": 0.7}})
    assert flow_viz_of(s5) == {**FLOW_VIZ_DEFAULTS, "z_min": 0.7, "z_max": 0.7}
    # damaged values read as automatic, unknown kinds as the default
    s6 = SimpleNamespace(id="f", cfd=SimpleNamespace(), flow_viz={"kind": "spray", "z": "high", "n": [0, -1],
                                                                   "spacing": float("nan"), "smoke": True})
    assert flow_viz_of(s6) == FLOW_VIZ_DEFAULTS


def test_stored_settings_are_the_hand_over_to_the_run():
    """What a run (Simulate's rake, the job's ViewSettings.seed_rake) takes from Setup:
    ``stored_flow_viz(flow_viz_of(sim))``, whose rake dict the runner's resolver accepts as is."""
    from aerolume.app.pages.setup import flow_viz_of, set_flow_viz, stored_flow_viz
    from aerolume.viz3d.flow import resolve_rake
    s = SimpleNamespace(id="r1", cfd=SimpleNamespace(), flow_viz=None)
    assert stored_flow_viz(flow_viz_of(s)) == {"smoke": True, "rake": {"kind": "rake"}}   # all automatic
    set_flow_viz(s, WANT)
    out = stored_flow_viz(flow_viz_of(s))
    assert out == {"smoke": False, "rake": WANT_RAKE}
    sr = resolve_rake(out["rake"], CAR_BOX, DOMAIN, purpose="smoke", h=CELL)
    assert sr.kind == "grid" and tuple(sr.n) == (9, 5) and sr.z == pytest.approx((0.2, 1.2))
    # the default rake resolves to the preview's nozzles (the runner's own call)
    from aerolume.app.widgets.geometry_canvas import smoke_rake_layout
    lay = smoke_rake_layout({"kind": "rake"}, CAR_BOX, extent3=DOMAIN, cell=CELL)
    auto = resolve_rake({"kind": "rake"}, CAR_BOX, DOMAIN, purpose="smoke", h=CELL)
    np.testing.assert_allclose(lay["points"], auto.points)


def test_smoke_texts():
    from aerolume.app.pages.setup import smoke_fact, smoke_note
    from aerolume.app.widgets.geometry_canvas import smoke_rake_layout
    on = {"smoke": True, "kind": "rake"}
    lay = smoke_rake_layout(on, CAR_BOX, extent3=DOMAIN, cell=CELL)
    assert smoke_fact(lay, on) == f"Rake, {lay['count']} nozzles"
    assert smoke_fact(None, {"smoke": False}) == "Off"
    note = smoke_note(lay, on)
    assert note.startswith(f"{lay['count']} nozzles") and "ahead of the body" in note and "z = " in note
    grid = smoke_rake_layout({"kind": "grid"}, CAR_BOX, extent3=DOMAIN, cell=CELL)
    assert f"{grid['n'][0]} × {grid['n'][1]} = {grid['count']} nozzles" in smoke_note(grid, {"smoke": True})
    vert = smoke_rake_layout({"kind": "vertical"}, CAR_BOX, extent3=DOMAIN, cell=CELL)
    assert smoke_fact(vert, {"smoke": True}).startswith("Vertical rake, ")
    assert "from z = " in smoke_note(vert, {"smoke": True})
    plane = smoke_rake_layout({"kind": "plane"}, CAR_BOX, extent3=DOMAIN, cell=CELL)
    assert smoke_fact(plane, {"smoke": True}) == "Plane band"
    assert smoke_note(plane, {"smoke": True}).startswith("A band from z = ")
    assert smoke_note(None, {"smoke": False}).startswith("Off")


# ---------------------------------------------------------------------------
# the free-stream arrow of the case preview
# ---------------------------------------------------------------------------

def _iso_projection(scale: float = 200.0, centre=(500.0, 300.0)):
    """A fixed oblique projection like the iso camera: x right-down, y left-up, z up (px)."""
    ex = np.array([0.80, 0.45])                  # +x goes right and down
    ey = np.array([-0.80, -0.30])                # +y goes left and up
    ez = np.array([0.0, -1.0])                   # +z goes up

    def project(p):
        p = np.asarray(p, dtype=float).reshape(-1, 3)
        return np.asarray(centre) + scale * (p[:, :1] * ex + p[:, 1:2] * ey + p[:, 2:3] * ez)
    return project


def test_flow_arrow_stays_clear_of_a_wing_the_rake_and_the_view_edge():
    from aerolume.app.widgets.geometry_canvas import (
        _convex_hull,
        _point_polygon_distance,
        flow_arrow_placement,
    )
    proj = _iso_projection()
    wing = (0.0, 0.3, -1.0, 1.0, 0.0, 0.05)
    corners = np.array([[x, y, z] for x in wing[:2] for y in wing[2:4] for z in wing[4:]])
    hull = _convex_hull(proj(corners))
    tail, head = flow_arrow_placement(wing, proj, view=(1000.0, 600.0))
    assert head[0] > tail[0] and head[1] == tail[1] and head[2] == tail[2]        # along +x
    seg = proj(np.linspace(tail, head, 9))
    assert min(_point_polygon_distance(q, hull) for q in seg) >= 14.0
    # a rake across the span in front of the wing: the arrow keeps clear of it too
    rake = np.array([[-0.1, y, 0.025] for y in np.linspace(-1.1, 1.1, 12)])
    tail, head = flow_arrow_placement(wing, proj, avoid=rake, view=(1000.0, 600.0), label=(80.0, 16.0))
    seg = proj(np.linspace(tail, head, 9))
    d = np.hypot(seg[:, None, 0] - proj(rake)[None, :, 0], seg[:, None, 1] - proj(rake)[None, :, 1])
    assert d.min() > 10.0
    # seen from the side nothing is in the way: the arrow keeps its natural place, in line
    # with the centre of the body
    def side(p):
        p = np.asarray(p, dtype=float).reshape(-1, 3)
        return np.column_stack([500.0 + 100.0 * p[:, 0], 300.0 - 100.0 * p[:, 2]])

    tail, head = flow_arrow_placement(CAR_BOX, side, road=True, view=(4000.0, 4000.0))
    assert tail[1] == pytest.approx(0.0) and tail[2] == pytest.approx(0.705) and head[0] < 0.0
    # on a road the arrow never goes below the body
    t2, _h2 = flow_arrow_placement((0, 1, -0.1, 0.1, 0.0, 0.2), proj, road=True, avoid=np.array([[-0.3, 0.0, 0.1]]))
    assert t2[2] >= 0.0
    assert flow_arrow_placement(CAR_BOX, lambda p: np.full((len(np.atleast_2d(p)), 2), np.nan)) is None


def test_speed_label_never_covers_its_arrow():
    """The speed label sits beside the arrow at any slope (a label centred above the middle
    covered the head of the steep arrow the iso view gives a wing)."""
    from aerolume.app.widgets.geometry_canvas import arrow_label_centre
    w, h = 84.0, 16.0
    for ang in np.linspace(0.0, 2 * np.pi, 37):
        a = np.array([400.0, 300.0])
        b = a + 70.0 * np.array([np.cos(ang), np.sin(ang)])
        c = arrow_label_centre(a, b, (w, h))
        seg = a + np.linspace(0.0, 1.0, 41)[:, None] * (b - a)
        inside = (np.abs(seg[:, 0] - c[0]) <= w / 2 + 2) & (np.abs(seg[:, 1] - c[1]) <= h / 2 + 2)
        assert not inside.any(), np.degrees(ang)
        # on the upper side of the arrow (screen y grows downwards); left of a vertical one
        mid = 0.5 * (a + b)
        if abs(np.sin(ang)) < 0.999:
            assert c[1] < mid[1]
        else:
            assert c[0] < mid[0]
        # and close: no further out than its half-diagonal plus the gap
        assert np.hypot(*(c - mid)) <= 0.5 * np.hypot(w, h) + 5.0 + 1e-9
        # the other side mirrors it through the arrow's middle
        assert np.allclose(arrow_label_centre(a, b, (w, h), side=-1), 2 * mid - c)


def test_speed_label_moves_below_when_the_upper_side_is_taken():
    from aerolume.app.widgets.geometry_canvas import arrow_label_centre, flow_arrow_layout

    def side(p):
        p = np.asarray(p, dtype=float).reshape(-1, 3)
        return np.column_stack([500.0 + 100.0 * p[:, 0], 300.0 - 100.0 * p[:, 2]])

    lab = (84.0, 16.0)
    tail, head, s = flow_arrow_layout(CAR_BOX, side, road=True, view=(4000.0, 4000.0), label=lab)
    assert s == 1                                     # nothing in the way: the caption sits above
    pe = side(np.array([tail, head]))
    up = arrow_label_centre(pe[0], pe[1], lab)
    # something (e.g. the smoke label) in the caption's upper half, 19 px off the shaft: the
    # arrow keeps its place, the caption goes below
    t2, h2, s2 = flow_arrow_layout(CAR_BOX, side, road=True, view=(4000.0, 4000.0), label=lab,
                                   avoid_px=np.array([up - (0.0, 6.0)]))
    assert s2 == -1 and np.allclose(t2, tail) and np.allclose(h2, head)


def test_point_polygon_distance_sign():
    from aerolume.app.widgets.geometry_canvas import _convex_hull, _point_polygon_distance
    sq = _convex_hull(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=float))
    assert len(sq) == 4
    assert _point_polygon_distance(np.array([5.0, 5.0]), sq) == pytest.approx(-5.0)
    assert _point_polygon_distance(np.array([13.0, 5.0]), sq) == pytest.approx(3.0)
    assert _point_polygon_distance(np.array([5.0, 5.0]), sq[::-1]) == pytest.approx(-5.0)   # either winding


# ---------------------------------------------------------------------------
# the Setup page
# ---------------------------------------------------------------------------

def _page(state, design):
    from aerolume.app.pages.setup import SetupPage
    state.add_design(design, select=True)
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    page.on_shown()
    return page


def test_setup_flow_visualisation_group(qapp, state, planner):
    from aerolume.app.pages.setup import flow_viz_of
    page = _page(state, _vehicle())
    page.update_summary()
    assert page.plan_status == "ok", page.planner_lines
    assert page.flowviz_group.isVisibleTo(page)
    assert page.smoke.isChecked() and page.smoke_kind.currentData() == "rake"
    assert page.smoke_auto_z.isChecked() and page.smoke_auto_spacing.isChecked()
    sm = page.preview3d.smoke()
    assert sm is not None and sm["kind"] == "rake" and sm["count"] > 0
    assert dict(page.estimate.items())["Smoke"] == f"Rake, {sm['count']} nozzles"
    assert page.smoke_note.text().startswith(f"{sm['count']} nozzles")
    assert "until the app closes" not in page.smoke_note.text()             # nothing edited, nothing to lose
    # the fields show the automatic values while on Auto; the rake has no "Height to"
    assert page.smoke_z_min.si_value() == pytest.approx(sm["z"][0], abs=1e-3)
    assert not page._smoke_zmax_row.label.isVisibleTo(page)
    # the planned (wind-frame) body sets the layout: the rake stands upstream of it, and the
    # nozzles drawn are the ones the runner resolves from the stored rake dict
    from aerolume.app.pages.setup import stored_flow_viz
    from aerolume.simulation.live3d import resolve_seed_rake
    assert sm["x"] < float(page.last_plan.body_bounds[0])

    def run_rake():
        return resolve_seed_rake(page.last_plan, stored_flow_viz(flow_viz_of(state.simulation))["rake"])

    assert np.allclose(run_rake().points, sm["points"])
    # a grid: the range rows appear, the count is the product
    page.smoke_kind.setCurrentIndex(page.smoke_kind.findData("grid"))
    page.update_summary()
    assert flow_viz_of(state.simulation)["kind"] == "grid"
    assert page._smoke_zmax_row.label.isVisibleTo(page) and page._smoke_zmin_row.label.text() == "Height from"
    assert " × " in page.smoke_note.text() and page.preview3d.smoke()["kind"] == "grid"
    from aerolume.app.pages.setup import _flow_viz_holder
    if _flow_viz_holder(state.simulation) is None:          # a model without a field for the rake: said
        assert "until the app closes; Convert to emitter keeps them" in page.smoke_note.text()
    # typing a spacing fixes it (Auto off): the stored rake gets the counts that give it, and
    # the run resolves the very nozzles the preview shows
    page.smoke_spacing.set_si_value(0.1, emit=True)
    page.update_summary()
    fv = flow_viz_of(state.simulation)
    shown = page.preview3d.smoke()
    assert fv["spacing"] == pytest.approx(0.1) and not page.smoke_auto_spacing.isChecked()
    assert fv["n"] == list(shown["n"]) and shown["spacing"] == pytest.approx(0.1, rel=0.2)
    assert np.allclose(run_rake().points, shown["points"])
    page.smoke_auto_spacing.click()
    fv = flow_viz_of(state.simulation)
    assert fv["spacing"] is None and fv["n"] is None
    # leaving Auto through one height fixes the whole range at what the fields show
    shown_hi = page.smoke_z_max.si_value()
    page.smoke_z_min.set_si_value(0.3, emit=True)
    fv = flow_viz_of(state.simulation)
    assert fv["z_min"] == pytest.approx(0.3) and fv["z_max"] == pytest.approx(shown_hi)
    page.update_summary()
    assert page.preview3d.smoke()["z"][0] == pytest.approx(0.3)
    assert np.allclose(run_rake().points, page.preview3d.smoke()["points"])
    # a vertical rake: up the height on the centre plane
    page.smoke_kind.setCurrentIndex(page.smoke_kind.findData("vertical"))
    page.update_summary()
    vert = page.preview3d.smoke()
    assert vert["kind"] == "vertical" and np.ptp(vert["points"][:, 1]) == 0 and vert["count"] > 2
    assert dict(page.estimate.items())["Smoke"] == f"Vertical rake, {vert['count']} nozzles"
    # a new kind: the heights follow the body again
    page.smoke_kind.setCurrentIndex(page.smoke_kind.findData("plane"))
    fv = flow_viz_of(state.simulation)
    assert fv["kind"] == "plane" and fv["z_min"] is None and fv["z_max"] is None
    page.update_summary()
    assert not page._smoke_spacing_row.label.isVisibleTo(page)
    assert dict(page.estimate.items())["Smoke"] == "Plane band"
    # off: the emitter rows are disabled, nothing is drawn, the Estimate says so
    page.smoke.setChecked(False)
    page.update_summary()
    assert flow_viz_of(state.simulation)["smoke"] is False
    assert not page.smoke_kind.isEnabled() and page.preview3d.smoke() is None
    assert dict(page.estimate.items())["Smoke"] == "Off"
    page.smoke.setChecked(True)
    # a duplicate carries the settings
    dup = page.duplicate_simulation()
    assert flow_viz_of(dup)["kind"] == "plane"
    assert not page.grab().isNull()


def test_setup_flow_visualisation_for_cfd_and_the_automatic_rake_for_3d(qapp, state, planner):
    from aerolume.app.dialogs.import_dialogs import naca_design
    from aerolume.app.pages.setup import flow_viz_of, set_flow_viz
    page = _page(state, _wing())
    assert state.simulation.analysis == "wing"
    assert not page.flowviz_group.isVisibleTo(page)            # the wing analysis has no flow field
    page.analysis.buttons["cfd"].click()
    page.update_summary()
    assert page.flowviz_group.isVisibleTo(page)               # the lofted wing in 3D CFD has smoke
    sm = page.preview3d.smoke()
    assert sm is not None and sm["y"][1] > 1.0                # spans the wing
    # 2D CFD: the emitters (DESIGN.md 10.1), but no automatic rake - a 2D run's smoke comes in at the inlet
    state.add_design(naca_design("0012"), select=True)
    page.on_shown()
    assert page.flowviz_group.isVisibleTo(page) and page.emitters_section.empty.isVisibleTo(page)
    assert not page.smoke_row.label.isVisibleTo(page) and "inlet" in page.emitters_section.empty.text()
    # a new 3D simulation of another body of the study takes its smoke kind, not its heights
    import aerolume.geometry3d as g
    box = g.design_from_meshes([g.box(0.3, 0.2, 0.2, name="Box")], "Box", "generic")
    state.add_design(box, select=True)
    page.on_shown()
    first = state.simulation
    set_flow_viz(first, {"smoke": True, "kind": "grid", "z_min": 0.1, "z_max": 0.3, "spacing": 0.05})
    sph = _sphere()
    state.add_design(sph, select=True)
    page.on_shown()
    fv = flow_viz_of(state.simulation)
    assert state.simulation.design_id == sph.id and state.simulation is not first
    assert fv["kind"] == "grid" and fv["z_min"] is None and fv["spacing"] is None
    # a second run of the same body keeps everything
    nxt = page.new_simulation()
    sph_first = flow_viz_of(next(s for s in state.project.simulations.values() if s.design_id == sph.id))
    assert flow_viz_of(nxt) == sph_first


def test_setup_smoke_settings_survive_save_and_reopen_when_the_model_holds_them(qapp, state, planner, tmp_path):
    from aerolume.app.pages.setup import _flow_viz_holder, flow_viz_of, set_flow_viz
    from aerolume.project import Project
    page = _page(state, _vehicle())
    sim = state.simulation
    if _flow_viz_holder(sim) is None:
        pytest.skip("the data model has no field for the smoke settings yet (session-only fallback)")
    want = {"smoke": True, "kind": "grid", "z_min": 0.1, "z_max": 0.4, "spacing": 0.05}
    set_flow_viz(sim, want)
    state.update_simulation(sim)
    page.flush()
    state.project.save()
    again = Project.open(state.project.path)
    assert flow_viz_of(again.simulations[sim.id]) == want


def test_case_preview_draws_each_emitter_kind(qapp, tmp_path):
    from aerolume.app.widgets.geometry_canvas import CasePreview3D, smoke_rake_layout
    car = _vehicle()
    view = CasePreview3D()
    view.resize(900, 520)
    view.show()
    b = car.bounds3()
    ext = (b[0] - 3, b[1] + 8, -3.0, 3.0, 0.0, 3.0)
    for kind in ("rake", "grid", "plane"):
        lay = smoke_rake_layout({"kind": kind}, b, on_road=True, extent3=ext)
        view.set_case(car, meshes=car.meshes, extent3=ext, symmetry=True, ground="moving", speed_text="U = 30 m/s",
                      attitude_text="Pitch 0°", smoke=lay)
        for _ in range(5):
            qapp.processEvents()
        assert view.smoke() is lay
        img = view.grab()
        assert not img.isNull()
        img.save(str(tmp_path / f"preview_{kind}.png"))
        # the framing includes the emitters (they stand upstream of the body)
        fx0 = view.scene.focus_bounds()[0]
        assert fx0 <= lay["x"] + 1e-9
        # the free-stream arrow was drawn, and neither it nor its label sits on the axes triad
        assert view.last_arrow is not None
        a, b2 = view._project(np.asarray(view.last_arrow[:2]))
        triad = view._triad_px()
        seg = a + np.linspace(0.0, 1.0, 21)[:, None] * (b2 - a)
        assert np.hypot(seg[:, None, 0] - triad[None, :, 0], seg[:, None, 1] - triad[None, :, 1]).min() > 4.0
    view.set_case(car, meshes=car.meshes, extent3=ext, ground="moving", smoke=None)
    assert view.smoke() is None
    view.scene.close()


def test_geometry_and_setup_offer_the_back_view(qapp, state):
    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    assert "back" in page.view_buttons
    from aerolume.app.pages.setup import SetupPage
    sp = SetupPage(state, None)
    assert "back" in sp.view_buttons
    page.release_3d()
    sp.release_3d()
