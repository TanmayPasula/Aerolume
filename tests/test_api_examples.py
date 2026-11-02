"""The physics examples (examples/scripts) against the API: what they set up, and how they stop.

The examples are the Script console's snippets (a console test keeps the files
and the snippets equal). These tests run the fast part of each: the reference
case it builds and the run it asks for. When this build's solver does not publish
the physics an example needs, the run must be refused with that reason before
anything is recorded -- a script then ends with one readable error, not with a
run that silently ignored its fan or its heating. The solver runs themselves are
covered by the validation tests of the solver packages.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import aerolume as al
from aerolume import api

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "scripts"
PHYSICS = {"heated_cavity": {"thermal", "closed_domain"}, "heated_cylinder": {"thermal"},
           "fan_room": {"fans", "emitters", "closed_domain"}}


def pretend(monkeypatch, features) -> None:
    """Make this build report ``features`` as supported, under both names a script may use.

    ``import aerolume as al`` resolves ``al.supported_physics`` on first use and keeps
    it, so patching only ``api.supported_physics`` would leave a stale (or patched)
    copy on the package for later tests.
    """
    def fn(dimension=None):
        return frozenset(features)
    monkeypatch.setattr(api, "supported_physics", fn)
    monkeypatch.setattr(al, "supported_physics", fn, raising=False)


@pytest.fixture
def lacking(monkeypatch):
    """A solver that implements none of the placed physics (the state before the solver agents land)."""
    pretend(monkeypatch, ())


@pytest.mark.parametrize("key", sorted(PHYSICS))
def test_a_physics_example_stops_with_the_reason_and_records_no_run(tmp_path, lacking, key, capsys):
    project = al.new_project(tmp_path / f"{key}")
    with pytest.raises(al.InputError, match="does not implement") as err:
        runpy.run_path(str(EXAMPLES / f"{key}.py"), init_globals={"project": project})
    words = {"thermal": "heat transfer", "closed_domain": "closed domains", "fans": "fans",
             "emitters": "smoke and particle emitters"}
    for feature in PHYSICS[key]:
        assert words[feature] in str(err.value), (key, feature)
    assert project.simulations == []                        # nothing half-recorded
    assert project.designs == []                            # not even the case's design (critic round 3)
    out = capsys.readouterr().out
    assert "<Case " in out                                  # the case and its reference were printed first


def test_the_examples_build_the_cases_their_comments_describe():
    cav = al.cavity_case(ra=1e4)
    assert cav.reference["value"] == 2.238 and cav.design.length_ref == pytest.approx(0.1)
    cyl = al.heated_cylinder_case(re=40)
    assert cyl.reference["value"] == pytest.approx(3.28, abs=0.005)          # "Lange et al. 3.28" in the docs
    assert cyl.design.bodies[0].thermal_value - cyl.settings["thermal"].t_ambient == pytest.approx(10.0)
    room = al.fan_room_case(thrust=1.0)
    assert room.settings["domain_box"] == (0.0, 4.0, 0.0, 3.0) and room.settings["fans"][0].diameter == 0.5
    src = (EXAMPLES / "fan_room.py").read_text(encoding="utf-8")
    assert '"position": (0.4, 0.4), "end": (0.4, 2.6)' in src                # the rake lies inside the room


def test_the_3d_smoke_example_places_its_rakes_upstream_of_the_sphere(tmp_path, monkeypatch):
    """Run the 3D example up to its run call: two rakes of two colours 1.5 diameters upstream."""
    seen = {}

    def fake_run_cfd(self, design, **kw):
        seen.update(kw, design=design)
        raise al.Cancelled("stopped by the test before the solver")

    pretend(monkeypatch, {"emitters"})
    monkeypatch.setattr(api.Project, "run_cfd", fake_run_cfd)
    project = al.new_project(tmp_path / "smoke")
    with pytest.raises(al.Cancelled):
        runpy.run_path(str(EXAMPLES / "smoke_3d.py"), init_globals={"project": project})
    rakes = seen["emitters"]
    assert seen["re"] == 300 and seen["quality"] == "draft" and len(rakes) == 2
    assert {r["color"] for r in rakes} == {"#f5f5f5", "#f0a040"}
    assert all(r["position"][0] == pytest.approx(-0.3) and r["end"][0] == pytest.approx(-0.3) for r in rakes)
    # the rakes span the sphere (d 0.2 m, centred at the origin) without touching it: |y|, |z| <= 0.08
    assert all(abs(c) <= 0.08 for r in rakes for c in (*r["position"][1:], *r["end"][1:]))
    sphere = project.design(seen["design"].id if hasattr(seen["design"], "id") else seen["design"])
    assert sphere.dimension == 3
    # the emitters pass the API's own checks for a 3D design
    from aerolume import model as M
    items = api._items("emitters", rakes, M.Emitter)
    api._check_emitters(items, 3)
    assert [e.kind for e in items] == ["rake", "rake"]


def test_without_3d_emitters_the_smoke_example_falls_back_and_says_so(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_cfd(self, design, **kw):
        seen.update(kw)
        raise al.Cancelled("stopped by the test")

    pretend(monkeypatch, ())
    monkeypatch.setattr(api.Project, "run_cfd", fake_run_cfd)
    with pytest.raises(al.Cancelled):
        runpy.run_path(str(EXAMPLES / "smoke_3d.py"), init_globals={"project": al.new_project(tmp_path / "s")})
    assert seen["emitters"] is None
    assert "default smoke rake" in capsys.readouterr().out
