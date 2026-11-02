"""The scripting API's placed physics (DESIGN.md 10): checks, reference cases, grid studies.

Everything here is fast and solver-free: the physics keywords are validated
before a run is recorded, so a wrong one is an :class:`al.InputError` with nothing
left in the project; the reference cases are pure set-ups whose published values
are checked against the papers; the grid-convergence numbers are those of
``aerolume.analysis.gci`` (checked against the worked example of Celik et al.
2008). One end-to-end cavity run is made only when this build's solver publishes
heat transfer (``al.supported_physics``) and is marked slow.
"""
from __future__ import annotations

import csv
import math
import types

import pytest

import aerolume as al
from aerolume import api
from aerolume import model as M


@pytest.fixture
def proj(tmp_path):
    return al.new_project(tmp_path / "Physics API")


def pretend(monkeypatch, features) -> None:
    """Make this build report ``features`` as supported, as ``api.supported_physics`` and ``al.supported_physics``
    (the package keeps the function it resolved first, so patching one name would leak into later tests)."""
    def fn(dimension=None):
        return frozenset(features)
    monkeypatch.setattr(api, "supported_physics", fn)
    monkeypatch.setattr(al, "supported_physics", fn, raising=False)


@pytest.fixture
def all_physics(monkeypatch):
    """Pretend the solver implements every placed-physics feature (to reach the API's own checks)."""
    pretend(monkeypatch, api.PHYSICS_FEATURES)


@pytest.fixture
def no_physics(monkeypatch):
    pretend(monkeypatch, ())


# ---------------------------------------------------------------------------
# reference cases
# ---------------------------------------------------------------------------

def test_cavity_case_sets_the_rayleigh_number_exactly_and_cites_de_vahl_davis():
    case = al.cavity_case(ra=1e4)
    th = case.settings["thermal"]
    walls = {w.side: w for w in th.walls}
    dT = walls["left"].value - walls["right"].value
    nu = case.settings["fluid"].kinematic_viscosity
    L = case.design.length_ref
    ra = th.gravity * th.beta * dT * L ** 3 / (nu * nu / th.prandtl)       # g β ΔT L³ / (ν κ), κ = ν / Pr
    assert ra == pytest.approx(1e4, rel=1e-12)
    assert walls["left"].value + walls["right"].value == pytest.approx(2 * th.t_ambient)
    assert case.settings["domain"] == "closed" and case.settings["domain_box"] == (0.0, L, 0.0, L)
    assert case.reference == {"quantity": "nusselt", "value": 2.238, "kind": "validation",
                              "source": "de Vahl Davis (1983), Nu₀: mean Nu of the hot wall",
                              # the band docs/validation.md checks it with (validation/run_all.py)
                              "band_pct": 2.0, "band_source": "docs/validation.md section 9, ± 2 % on the 64² grid"}
    # every published Rayleigh number in 2D and 3D, and no reference off the benchmark's Prandtl number
    assert [al.cavity_case(ra=r).reference["value"] for r in (1e3, 1e5, 1e6)] == [1.117, 4.509, 8.817]
    assert al.cavity_case(ra=1e5, dimension=3).reference["value"] == 4.3370
    assert al.cavity_case(ra=1e4, pr=7.0).reference is None
    assert al.cavity_case(ra=3e4).reference is None
    with pytest.raises(al.InputError, match="dimension"):
        al.cavity_case(dimension=4)


def test_heated_cylinder_follows_the_lange_correlation_and_turns_buoyancy_off():
    case = al.heated_cylinder_case(re=40)
    # Lange, Durst & Breuer (1998): Nu = 0.082 Re^0.5 + 0.734 Re^(0.05 + 0.226 Re^0.085)
    x = 0.05 + 0.226 * 40 ** 0.085
    assert case.reference["value"] == pytest.approx(0.082 * 40 ** 0.5 + 0.734 * 40 ** x, abs=5e-5)
    assert case.reference["value"] == pytest.approx(3.28, abs=0.01)
    assert case.settings["re"] == 40 and case.settings["thermal"].buoyancy is False
    body = case.design.bodies[0]
    assert body.thermal == "temperature" and body.thermal_value == pytest.approx(25.0)
    assert al.heated_cylinder_case(re=500).reference is None          # outside the fit's 1e-4..200
    with pytest.raises(al.InputError, match="delta_t"):
        al.heated_cylinder_case(delta_t=0)


def test_fan_room_case_puts_one_fan_in_a_closed_room():
    case = al.fan_room_case(thrust=2.0)
    fan = case.settings["fans"][0]
    assert case.settings["domain"] == "closed" and case.settings["domain_box"] == (0.0, 4.0, 0.0, 3.0)
    assert fan.position == pytest.approx((4 / 3, 1.5)) and fan.mode == "thrust" and fan.value == 2.0
    assert case.reference["quantity"] == "fan thrust" and case.reference["value"] == 2.0
    three = al.fan_room_case(dimension=3)
    assert len(three.settings["fans"][0].position) == 3 and three.design.dimension == 3
    with pytest.raises(al.InputError, match="does not fit"):
        al.fan_room_case(diameter=3.5)


def test_the_cavity_reference_is_the_validated_hot_wall_nusselt_number():
    """The API, the validation runner and docs/validation.md quote the same de Vahl Davis value.

    The API once used the paper's cavity-average Nū (2.243 at Ra 10⁴) against a
    measured hot-wall Nu, while the validation table compares with Nu₀ (2.238):
    two references for one run (critic round 3). The runner's table is read from
    its source (importing it would load torch).
    """
    import ast
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "validation" / "thermal2d_cases.py").read_text(encoding="utf-8"))
    table = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "DE_VAHL_DAVIS" for t in n.targets))
    assert api.CAVITY_NU_2D == {ra: row["nu_0"] for ra, row in table.items()}
    doc = (root / "docs" / "validation.md").read_text(encoding="utf-8")
    sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    rows = {float("1e" + m.group(1).translate(sup)): float(m.group(2)) for m in re.finditer(
        r"^\| Cavity Ra 10([⁰¹²³⁴⁵⁶⁷⁸⁹]+) \([^)]*\): mean Nu, hot wall \| [^|]+ \| ([\d.]+) \|", doc, re.MULTILINE)}
    assert rows == api.CAVITY_NU_2D
    for ra, nu in api.CAVITY_NU_2D.items():
        assert al.cavity_case(ra=ra).reference["value"] == nu


def test_the_fan_room_check_is_labelled_a_consistency_check():
    case = al.fan_room_case(thrust=1.0)
    assert case.kind == "consistency" and "consistency" in repr(case)
    res = types.SimpleNamespace(fans=[{"thrust": 1.0, "thrust_stderr": 0.0}])
    out = case.check(res)
    assert out["kind"] == "consistency" and out["difference_pct"] == 0.0
    assert "not a validation" in out["note"] and "momentum theory" in out["note"]
    assert al.cavity_case(ra=1e4).kind == "validation" and al.heated_cylinder_case(re=40).kind == "validation"


def test_case_check_compares_with_the_reference_and_its_standard_error():
    case = al.cavity_case(ra=1e4)
    res = types.SimpleNamespace(nusselt=2.25, nusselt_stat=M.Stat(mean=2.25, stderr=0.01))
    out = case.check(res)
    assert out["reference"] == 2.238 and out["measured"] == 2.25 and out["kind"] == "validation"
    assert out["difference_pct"] == pytest.approx(100 * (2.25 - 2.238) / 2.238)
    assert out["within_stderr"] is True                    # |2.25 - 2.238| = 0.012 <= 2 x 0.01
    far = case.check(types.SimpleNamespace(nusselt=2.4, nusselt_stat=M.Stat(mean=2.4, stderr=0.01)))
    assert far["within_stderr"] is False
    with pytest.raises(al.InputError, match="no published reference"):
        al.cavity_case(ra=2e4).check(res)


def test_add_case_adds_the_design_once(proj):
    a = proj.add_case(al.cavity_case(ra=1e4))
    b = proj.add_case(al.cavity_case(ra=1e4))
    assert a.id == b.id and len(proj.designs) == 1
    assert a.name == "Heated cavity · Ra 1.0×10⁴"


# ---------------------------------------------------------------------------
# the physics keywords are checked before anything is recorded
# ---------------------------------------------------------------------------

BAD = [
    ({"emitters": [{"kind": "rake", "position": (0, 0)}]}, "needs end="),
    ({"emitters": [{"kind": "disc", "position": (0, 0), "size": (0.1,)}]}, "is 3D"),
    ({"emitters": [{"knd": "point", "position": (0, 0)}]}, "Did you mean 'kind'"),
    ({"emitters": [{"kind": "point", "position": (0, 0), "color": c} for c in
                   ("#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff")]}, "At most 4 smoke colours"),
    ({"fans": [{"position": (1, 0, 0.5), "diameter": 0.2, "value": 2}]}, "needs 2 coordinates"),
    ({"fans": [{"position": (1, 0), "diameter": 0.2, "value": 0}]}, "zero thrust does nothing"),
    ({"fans": [{"position": (1, 0), "diameter": 0.2, "value": 2, "swirl": 0.3}]}, "swirl needs a 3D design"),
    ({"heat_sources": [{"kind": "rect", "position": (1, 0), "size": (0.2, 0.2), "power": 50}]},
     "needs heat transfer"),
    ({"thermal": True, "walls": {"left": 30.0}}, "left side is the inlet"),
    ({"thermal": True, "ports": [{"kind": "vent", "position": (1, 0), "size": 0.1, "speed": 1, "temperature": 40}]},
     "only a jet carries a temperature"),
    ({"ports": [{"kind": "jet", "position": (1, 0), "size": 0.1, "speed": 1, "temperature": 40}]},
     "needs heat transfer"),
    ({"domain": "closed", "ground": True}, "drop ground=True"),
    ({"domain": "box"}, "'tunnel'"),
    ({"domain": "closed", "domain_box": (0, 2, 0, 1), "fans": [{"position": (3, 0.5), "diameter": 0.2, "value": 1}]},
     "outside the domain box"),
    ({"adaptive_dt": "magic"}, "'cfl'"),
    ({"thermal": {"t_ambient": -300}}, "below absolute zero"),
    ({"thermal": {"colour": 3}}, "unknown field"),
]


@pytest.mark.parametrize(("kw", "match"), BAD, ids=[m for _k, m in BAD])
def test_invalid_physics_is_refused_with_a_reason_and_records_nothing(proj, all_physics, kw, match):
    d = proj.add_template("ahmed")
    with pytest.raises(al.InputError, match=match):
        proj.run_cfd(d, speed=2, device="cpu", **kw)
    assert proj.simulations == []


def test_physics_the_solver_lacks_is_refused_by_name_with_the_run_named(proj, no_physics):
    d = proj.add_template("ahmed")
    fan = al.Fan(position=(1.0, 0.2), direction=(1, 0), diameter=0.2, value=3.0)
    with pytest.raises(al.InputError) as err:
        proj.run_cfd(d, speed=2, device="cpu", fans=[fan], thermal=True)
    text = str(err.value)
    assert "does not implement heat transfer, fans yet" in text and "nothing was run" in text
    assert "“CFD · 2 m/s · α 0° · heated · 1 fan”" in text          # named as it would have been recorded
    assert proj.simulations == []
    # plain runs never ask the simulation layer about physics at all
    assert api.physics_features(M.Simulation(name="plain", design_id="x", analysis="cfd")) == []


def test_a_refused_case_run_or_plan_leaves_no_design_behind(proj, no_physics):
    """A refused case used to add its design first: an empty "Heated cavity" per attempt (critic round 3)."""
    for attempt in (lambda: proj.run_case(al.cavity_case(ra=1e4)), lambda: proj.plan(al.heated_cylinder_case()),
                    lambda: proj.run_cfd(al.fan_room_case(), quality="draft")):
        with pytest.raises(al.InputError, match="nothing was run"):
            attempt()
    with pytest.raises(al.InputError):                       # a bad keyword is refused before the design too
        proj.run_case(al.cavity_case(ra=1e4), quality="ultra")
    assert proj.designs == [] and proj.simulations == []
    # preparing a case without running it still works (a later build runs it)
    sim = proj.add_simulation(al.cavity_case(ra=1e4))
    assert len(proj.designs) == 1 and sim.status == "new"


def test_re_running_an_add_and_heat_script_reuses_the_unused_design(proj):
    """``d = add(...); d.set_thermal(...)`` twice gave "Hot sphere" and "Hot sphere 2" (critic round 3)."""
    for _ in range(2):
        d = proj.add_template("cylinder", name="Hot cylinder")
        d.set_thermal("temperature", 60.0)
    assert [x.name for x in proj.designs] == ["Hot cylinder"]
    assert proj.designs[0].raw.bodies[0].thermal == "temperature"
    # a design that has simulations is never reset: the unheated request gets its own design
    proj.add_simulation(d, speed=1.0, thermal={"buoyancy": False})
    plain = proj.add_template("cylinder", name="Hot cylinder")
    assert plain.name == "Hot cylinder 2" and plain.raw.bodies[0].thermal == "adiabatic"
    assert proj.design("Hot cylinder").raw.bodies[0].thermal == "temperature"


def _fan_result(sim_id: str) -> M.CFDResult:
    return M.CFDResult(sim_id=sim_id, re_real=3e4, re_sim=3e4, turbulence="laminar", fans_report=[
        {"name": "Fan", "mode": "thrust", "thrust": 1.0, "thrust_stderr": 0.002, "thrust_reliable": True,
         "pressure_jump": 2.0, "volume_flow": 0.0718, "disc_velocity": 0.1436, "induced_velocity": 0.1436,
         "ideal_power": 0.1436, "target": 1.0, "diameter": 0.5}])


def test_the_fan_rows_carry_units_and_the_thrust_stderr_sits_in_its_column(proj):
    case = al.fan_room_case(thrust=1.0)
    sim = proj.add_simulation(case)
    r = api.Result(proj, sim, _fan_result(sim.id))
    rows = {row["key"]: row for row in r.table()}
    assert "fan0.thrust_stderr" not in rows and rows["fan0.thrust"]["stderr"] == 0.002
    units = {k: rows[f"fan0.{k}"]["unit"] for k in ("thrust", "disc_velocity", "induced_velocity", "ideal_power",
                                                     "target", "diameter", "volume_flow", "pressure_jump")}
    assert units == {"thrust": "N/m", "disc_velocity": "m/s", "induced_velocity": "m/s", "ideal_power": "W/m",
                     "target": "N/m", "diameter": "m", "volume_flow": "m²/s", "pressure_jump": "Pa"}
    text = str(r)
    assert "thrust stderr" not in text and "Fan: ideal power" in text
    line = next(ln for ln in text.splitlines() if ln.startswith("Fan: thrust"))
    assert "0.002" in line and line.rstrip().endswith("N/m")
    # the room's check is the consistency of the imposed thrust, and it says so
    chk = r.reference_check()
    assert chk["applies"] and chk["kind"] == "consistency" and chk["stderr"] == 0.002
    assert "not a validation" in chk["note"]


def test_a_stored_cylinder_run_is_checked_against_lange_only_in_forced_convection(proj):
    case = al.heated_cylinder_case(re=40)
    sim = proj.add_simulation(case)
    raw = M.CFDResult(sim_id=sim.id, re_real=40.0, re_sim=40.0, turbulence="laminar",
                      nusselt=M.Stat(3.3402, 0.01, 3.32, 3.36, 0.001))
    chk = api.Result(proj, sim, raw).reference_check()
    lange = round(api.lange_nusselt(40.0), 4)
    assert chk["applies"] and chk["reference"] == lange and chk["kind"] == "validation"
    assert chk["difference_pct"] == pytest.approx(100 * (3.3402 - lange) / lange)
    hot = proj.edit_simulation(sim, buoyancy=True)
    chk2 = api.Result(proj, hot, raw).reference_check()
    assert chk2 == {"applies": False, "why": chk2["why"]} and "forced convection" in chk2["why"]
    # a design that is no reference set-up has no check at all
    plain = proj.add_simulation(proj.add_template("ahmed"), speed=2.0)
    assert api.Result(proj, plain, M.CFDResult(sim_id=plain.id)).reference_check() is None


# ---------------------------------------------------------------------------
# placing physics on a simulation that is already in the project
# ---------------------------------------------------------------------------

def test_add_emitter_fan_heater_and_jet_to_a_set_up_simulation(proj):
    room = proj.add_case(al.fan_room_case(thrust=1.0))
    sim = proj.add_simulation(room, domain="closed", domain_box=(0, 4, 0, 3), speed=1.0)
    s1 = proj.add_emitter(sim, kind="rake", position=(0.4, 0.4), end=(0.4, 2.6), count=8)
    assert s1.id == sim.id                                        # no result yet: edited in place
    assert [e.kind for e in s1.emitters] == ["rake"] and s1.emitters[0].end == (0.4, 2.6)
    assert "smoke" in s1.name                                     # the automatic name follows the physics
    s2 = proj.add_fan(s1, al.Fan(position=(1.3, 1.5), direction=(2, 0), diameter=0.5, value=1.0))
    assert s2.fans[0].direction == (1.0, 0.0)                     # normalised like run_cfd(fans=)
    with pytest.raises(al.InputError, match="thermal=True"):
        proj.add_heat_source(s2, kind="rect", position=(2, 0.3), size=(0.4, 0.2), power=300)
    s3 = proj.edit_simulation(s2, thermal=True)
    s4 = proj.add_heat_source(s3, {"kind": "rect", "position": (2, 0.3), "size": (0.4, 0.2), "power": 300})
    s5 = proj.add_port(s4, kind="jet", position=(3.5, 2.5), direction=(0, -1), size=0.1, speed=2, temperature=40)
    assert len(s5.heat_sources) == 1 and s5.ports[0].temperature == 40.0
    assert len(proj.simulations) == 1                              # every step edited the one simulation


@pytest.mark.parametrize(("call", "kw", "match"), [
    ("add_emitter", {"kind": "rake", "position": (1, 1)}, "needs end="),
    ("add_emitter", {"knd": "point", "position": (1, 1)}, "Did you mean 'kind'"),
    ("add_fan", {"position": (9, 1), "diameter": 0.2, "value": 1}, "outside the domain box"),
    ("add_fan", {"position": (1, 1, 0.5), "diameter": 0.2, "value": 1}, "needs 2 coordinates"),
    ("add_port", {"kind": "vent", "position": (1, 1), "size": 0.1, "speed": 1, "temperature": 30},
     "needs heat transfer"),
    ("add_emitter", {}, "Describe the Emitter"),
])
def test_a_wrong_placement_is_refused_and_leaves_the_simulation_as_it_was(proj, call, kw, match):
    room = proj.add_case(al.fan_room_case(thrust=1.0))
    sim = proj.add_simulation(room, domain="closed", domain_box=(0, 4, 0, 3), speed=1.0)
    before = sim.to_dict()
    with pytest.raises(al.InputError, match=match):
        getattr(proj, call)(sim, **kw)
    assert proj.simulation(sim.id).to_dict() == before and len(proj.simulations) == 1


def test_a_fifth_smoke_colour_counts_the_emitters_already_there(proj):
    d = proj.add_template("ahmed")
    colours = ("#ff0000", "#00ff00", "#0000ff", "#ffff00")
    sim = proj.add_simulation(d, speed=2, emitters=[{"kind": "point", "position": (-1, 0.2 + 0.1 * i), "color": c}
                                                     for i, c in enumerate(colours)])
    with pytest.raises(al.InputError, match="At most 4 smoke colours"):
        proj.add_emitter(sim, kind="point", position=(-1, 0.9), color="#00ffff")
    s = proj.add_emitter(sim, kind="point", position=(-1, 0.9), color="#ff0000")    # an existing channel
    assert len(s.emitters) == 5


def test_placing_physics_on_a_simulation_with_a_result_makes_a_copy(proj, monkeypatch):
    d = proj.add_template("ahmed")
    sim = proj.add_simulation(d, speed=2)
    monkeypatch.setattr(proj.raw, "has_result", lambda sid: sid == sim.id)
    s = proj.add_emitter(sim, kind="point", position=(-1, 0.3))
    assert s.id != sim.id and s.status == "new" and len(s.emitters) == 1
    assert proj.simulation(sim.id).emitters == []                  # the run with the result is untouched
    with pytest.raises(al.InputError, match="has a result"):
        proj.add_emitter(sim, kind="point", position=(-1, 0.3), copy=False)
    pol = proj.add_simulation(proj.add_naca("2412"), "polar")
    with pytest.raises(al.InputError, match="belong to CFD simulations"):
        proj.add_fan(pol, position=(1, 0), diameter=0.2, value=1)


@pytest.mark.parametrize("dim", [2, 3])
def test_a_closed_room_heated_only_by_a_heater_or_a_flux_body_is_driven_by_buoyancy(proj, dim):
    """Critic round 3.2: a room with only a 300 W heater was refused as "nothing drives the flow".

    The heater's temperature scale is its estimated rise q''L/(k Nu) (DESIGN.md 10.2), and the
    run's reference speed is the free-fall velocity the planner states for it.
    """
    from aerolume.simulation.physics import thermal_estimate
    case = al.fan_room_case(dimension=dim)
    size = (0.3, 0.2) if dim == 2 else (0.3, 0.2, 0.3)
    pos = (3.5, 0.3) if dim == 2 else (3.5, 1.5, 0.3)
    plan = proj.plan(case, "cfd", quality="draft", fans=[], thermal={"buoyancy": True, "t_ambient": 15.0},
                     heat_sources=[{"name": "Heater", "kind": "rect" if dim == 2 else "box", "position": pos,
                                    "size": size, "power": 300.0}])
    sim = plan.simulation
    est = thermal_estimate(case.design.raw if hasattr(case.design, "raw") else case.design, sim)
    assert est is not None and est.natural and est.delta_t_basis.startswith("estimated rise at the heater")
    assert est.delta_t > 1.0
    assert sim.conditions.speed == pytest.approx(est.speed)
    assert sim.conditions.speed == pytest.approx(math.sqrt(est.g_beta * est.delta_t * est.length_ra))
    # nothing heated, nothing blowing: still refused, with the heater named among the remedies
    with pytest.raises(al.InputError, match="a heater"):
        proj.plan(case, "cfd", quality="draft", fans=[], thermal={"buoyancy": True})


def test_a_heat_flux_wall_drives_a_closed_room(proj):
    case = al.fan_room_case()
    plan = proj.plan(case, "cfd", quality="draft", fans=[],
                     thermal={"buoyancy": True, "walls": [{"side": "bottom", "mode": "heat_flux", "value": 50.0}]})
    assert plan.simulation.conditions.speed > 0


def test_a_reused_result_says_so_and_tells_the_host(proj, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(api.Host, "result_reused", lambda self, store, sim_id: seen.append(sim_id), raising=False)
    d = proj.add_naca("0012")
    first = proj.run_polar(d, re=5e5, alpha=(0, 2, 1))
    assert not first.reused and seen == [] and "reused" not in str(first)
    again = proj.run_polar(d, re=5e5, alpha=(0, 2, 1))
    assert again.reused and seen == [again.sim_id]
    assert "Stored result of an identical run, reused (not run again; rerun=True runs it again)." in str(again)


def test_supported_physics_reads_what_the_simulation_layer_publishes(monkeypatch):
    from aerolume import simulation as S
    monkeypatch.setattr(S, "SUPPORTED_PHYSICS", {2: {"thermal", "fans"}, 3: {"thermal"}}, raising=False)
    assert al.supported_physics(2) == {"thermal", "fans"} and al.supported_physics(3) == {"thermal"}
    assert al.supported_physics() == {"thermal"}
    monkeypatch.setattr(S, "SUPPORTED_PHYSICS", ["emitters"], raising=False)
    assert al.supported_physics(3) == {"emitters"}


# ---------------------------------------------------------------------------
# grid convergence: the numbers of analysis.gci
# ---------------------------------------------------------------------------

def test_grid_convergence_reproduces_the_celik_worked_example():
    """Celik et al. (2008) Table 1, column 1: N = 18000, 8000, 4500 cells (2D), φ = 6.063, 5.972, 5.863."""
    h = [(1 / n) ** 0.5 for n in (18000, 8000, 4500)]
    g = al.grid_convergence(h, [6.063, 5.972, 5.863], quantity="reattachment length")
    assert g.convergence == "monotone"
    assert g.p == pytest.approx(1.53, abs=0.005)
    assert g.phi_ext == pytest.approx(6.1685, abs=5e-4)
    assert 100 * g.e_a21 == pytest.approx(1.5, abs=0.05)
    assert 100 * g.e_ext21 == pytest.approx(1.7, abs=0.05)
    assert 100 * g.gci_fine == pytest.approx(2.2, abs=0.05)
    assert g.summary().startswith("reattachment length: 6.063 ±")
    from aerolume.analysis import gci as G
    ref = G.gci_study(h, [6.063, 5.972, 5.863])
    assert (g.p, g.phi_ext, g.gci_fine) == (ref.p, ref.phi_ext, ref.gci_fine)       # one implementation


def test_grid_convergence_does_not_claim_an_order_inside_the_averaging_noise():
    g = al.grid_convergence([1.0, 1.5, 2.25], [1.000, 1.004, 1.010], stderr=[0.004, 0.004, 0.004])
    assert g.noise_limited is True and not g.usable
    assert any("standard errors" in w for w in g.warnings)
    with pytest.raises(al.InputError, match="three grids"):
        al.grid_convergence([1, 2], [1.0, 1.1])
    with pytest.raises(al.InputError, match="distinct"):
        al.grid_convergence([1, 1, 2], [1.0, 1.1, 1.2])


class _GridResult:
    """The parts of an API Result a GridStudy reads (a 2D CFD run on one grid)."""

    def __init__(self, sid: str, cpr: int, cd: float, se: float) -> None:
        self.sim_id, self.name, self.dimension = sid, f"CFD · {cpr} cells/ref", 2
        self.raw = types.SimpleNamespace(cells=cpr * cpr * 8, grid=(cpr * 4, cpr * 2), cells_per_ref=cpr)
        self.stats = {"cd": M.Stat(mean=cd, stderr=se)}
        self.nusselt_stat = None

    def summary(self) -> dict:
        return {"cd": self.stats["cd"].mean}


def test_grid_study_uses_analysis_gci_and_exports_its_csv(tmp_path):
    # Cd = 1.0 + 0.4 h^2 on h = 1/16, 1/24, 1/36 (refinement 1.5), standard errors far below the changes
    cprs = [16, 24, 36]
    rs = [_GridResult(f"s{c}", c, 1.0 + 0.4 / c ** 2, 1e-6) for c in cprs]
    gs = al.GridStudy(rs, cprs, ["cd"])
    g = gs.gci["cd"]
    assert g.convergence == "monotone" and g.p == pytest.approx(2.0, abs=1e-6)
    assert g.phi_ext == pytest.approx(1.0, abs=1e-9) and g.asymptotic is True and g.noise_limited is False
    assert [lv.sim_id for lv in g.levels] == ["s36", "s24", "s16"]          # fine first, as the report
    assert gs.study.quantities["cd"] is g and gs.study.sim_ids == ["s36", "s24", "s16"]
    text = str(gs)
    # the fine value to the band's second significant figure (not "1 ± 0.0003858")
    assert "Cd: 1.00031 ± 0.00039 (0.039 %)" in text and "observed order 2.00" in text
    assert "extrapolated 1.00000" in text and gs.summary()["cd"] in text
    assert "cd GCI" in repr(gs)
    path = gs.export(tmp_path / "grid.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["sim_id"] for r in rows] == ["s36", "s24", "s16"] and rows[0]["convergence"] == "monotone"
    assert math.isclose(float(rows[0]["p"]), 2.0, abs_tol=1e-6)
    with pytest.raises(al.InputError, match=".csv"):
        gs.export(tmp_path / "grid.json")
    fig = gs.plot("cd")
    assert fig.axes and fig.axes[0].get_xlabel().startswith("h = 1 / cells")


def test_a_grid_study_with_a_missing_value_says_so():
    rs = [_GridResult("a", 16, 1.0, 0.0), _GridResult("b", 24, float("nan"), 0.0), _GridResult("c", 36, 1.1, 0.0)]
    gs = al.GridStudy(rs, [16, 24, 36], ["cd"])
    assert "cd" not in gs.gci and any("finite value" in n for n in gs.notes)


# ---------------------------------------------------------------------------
# the ambient temperature of a thermal run
# ---------------------------------------------------------------------------

def _fluid_warnings(plan) -> list[str]:
    return [w for w in plan.warnings if "ambient temperature is" in w]


@pytest.mark.parametrize("thermal", [True, {"buoyancy": True}])
def test_a_default_thermal_run_has_the_fluids_temperature_as_ambient(proj, thermal):
    """Critic round 3.3: every API thermal run began with "the fluid is air at 15 °C but the ambient
    temperature is 20 °C". With no ambient stated, it is the temperature the fluid was made at."""
    case = al.fan_room_case()
    plan = proj.plan(case, "cfd", quality="draft", thermal=thermal,
                     heat_sources=[{"name": "Heater", "position": (3.5, 0.3), "size": (0.3, 0.2), "power": 300.0}])
    sim = plan.simulation
    assert sim.thermal.t_ambient == pytest.approx(15.0)       # ISA sea-level air
    assert _fluid_warnings(plan) == []
    d = proj.add_sphere(0.1, name="Hot sphere")
    d.set_thermal("temperature", 60.0)
    plan = proj.plan(d, "cfd", re=100, quality="draft", thermal=True)
    assert plan.simulation.thermal.t_ambient == pytest.approx(15.0) and _fluid_warnings(plan) == []


def test_a_stated_ambient_without_a_fluid_runs_in_air_at_that_temperature(proj):
    from aerolume.core import fluid as F
    d = proj.add_sphere(0.1, name="Hot sphere")
    d.set_thermal("temperature", 60.0)
    plan = proj.plan(d, "cfd", re=100, quality="draft", thermal={"t_ambient": 25.0, "buoyancy": True})
    sim = plan.simulation
    assert sim.conditions.fluid.name == "Air, 25 C" and sim.thermal.t_ambient == 25.0
    assert sim.conditions.fluid.density == pytest.approx(F.air(25.0).density)
    # re= is turned into a speed with the 25 °C air, not the 15 °C default
    assert sim.conditions.reynolds(0.1) == pytest.approx(100.0, rel=1e-9)
    assert _fluid_warnings(plan) == []
    # "ambient" is the same keyword
    alias = proj.plan(d, "cfd", re=100, quality="draft", thermal={"ambient": 25.0})
    assert alias.simulation.conditions.fluid.name == "Air, 25 C"


def test_a_stated_fluid_and_ambient_that_disagree_still_get_the_warning(proj):
    """A mismatch the caller typed is theirs to see: only the default is made consistent."""
    d = proj.add_sphere(0.1, name="Hot sphere")
    d.set_thermal("temperature", 60.0)
    plan = proj.plan(d, "cfd", speed=0.02, quality="draft", fluid="air", thermal={"t_ambient": 30.0})
    assert plan.simulation.conditions.fluid.name.startswith("Air, ISA") and _fluid_warnings(plan)
    water = proj.plan(d, "cfd", speed=0.02, quality="draft", fluid="water", thermal=True)
    assert water.simulation.thermal.t_ambient == pytest.approx(20.0) and _fluid_warnings(water) == []


# ---------------------------------------------------------------------------
# an end-to-end physics run (only when the solver publishes it)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif("thermal" not in al.supported_physics(2) or "closed_domain" not in al.supported_physics(2),
                    reason="this build's 2D solver does not publish heat transfer in closed domains yet")
def test_the_ra_1e3_cavity_runs_and_reports_its_nusselt_number(proj):
    case = al.cavity_case(ra=1e3)
    r = proj.run_case(case, quality="draft")
    assert "thermal" in r.physics and "closed_domain" in r.physics
    check = case.check(r)
    assert math.isfinite(check["measured"]) and abs(check["difference_pct"]) < 10.0
