"""The 3D frame and sign conventions (DESIGN.md 7.2), checked across the modules that apply them.

Frame: x downstream, y to the left of an observer looking downstream -- which is the
*right* of a driver or pilot facing forward (the nose points to -x) -- and z up.
Positive yaw rotates the geometry by -yaw about +z: the nose turns toward +y, the free
stream in the vehicle's own frame is U (cos yaw, sin yaw, 0), and the wind comes from
the vehicle's left (-y). Wheels rolling on a road moving at +U spin at -U/r about +y.

Every module that touches these conventions must agree: the model's docstrings, the
3D planner (geometry into the wind frame, wheel speeds), the 3D solver's attitude
helper and wheel kinematics, and the finite-wing solver's sideslip.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.geometry3d import box, cylinder, design_from_meshes
from aerolume.model import FlowConditions, GroundConfig, WingSection, WingSpec
from aerolume.simulation import defaults as D
from aerolume.simulation.plan3d import rotation3, wind_frame_meshes
from aerolume.solvers.ns3d.geometry import rotation_matrix as ns3d_rotation
from aerolume.solvers.wing import analyze_wing, freestream_direction


def _vehicle():
    parts = [box(4.0, 1.8, 1.2, center=(2.0, 0.0, 0.9), name="body")]
    for x in (0.8, 3.2):
        for y in (-0.8, 0.8):
            parts.append(cylinder(0.6, 0.25, axis="y", n=32, center=(x, y, 0.3), name=f"wheel {x:g} {y:g}"))
    return design_from_meshes(parts, "car", kind="vehicle", units="m")


@pytest.mark.parametrize("yaw", [5.0, 20.0])
def test_positive_yaw_turns_the_nose_to_plus_y_and_the_wind_comes_from_the_left(yaw):
    r = rotation3(0.0, yaw)
    nose = r @ np.array([-1.0, 0.0, 0.0])                    # the nose points to -x before yawing
    assert nose[1] > 0 and nose[0] < 0
    # the free stream (+x in the wind frame) seen from the vehicle: rotate back with r^T
    u_body = r.T @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(u_body, [math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0],
                               atol=1e-15)
    # the vehicle's left, for a driver facing forward (-x) with z up, is -y: the wind enters from there
    forward, up = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)
    np.testing.assert_allclose(left, [0.0, -1.0, 0.0])
    assert float(u_body @ left) < 0                           # flowing from the left side to the right
    # the solver's attitude helper and the wing solver use the same rotation
    np.testing.assert_allclose(ns3d_rotation(0.0, yaw), r, atol=1e-15)
    np.testing.assert_allclose(freestream_direction(0.0, yaw), u_body, atol=1e-15)


def test_the_planner_yaws_the_geometry_the_documented_way():
    d = _vehicle()
    wind = wind_frame_meshes(d.meshes, 0.0, 10.0, d.moment_ref3, ground=True)
    body = wind[0]
    nose = body.vertices[np.argmin(d.meshes[0].vertices[:, 0])]   # the same vertex, after yawing
    tail = body.vertices[np.argmax(d.meshes[0].vertices[:, 0])]
    assert nose[1] > tail[1]                                       # the nose swung toward +y
    assert "-y side" in FlowConditions.__doc__ and "left" in FlowConditions.__doc__


def test_rolling_wheels_spin_minus_u_over_r_about_plus_y():
    """The planner gives road wheels omega = -U/r about +y; the contact point then moves with
    the road (+U in x), the top of the wheel upstream."""
    from aerolume.simulation import plan_simulation

    d = _vehicle()
    sim = D.default_simulation(d, "wheels", FlowConditions(speed=20.0))
    sim.ground = GroundConfig(enabled=True, moving=True, rotate_wheels=True)
    sim.cfd.device = "cpu"
    sim.cfd.quality = "draft"
    plan = plan_simulation(d, sim, device="cpu")
    U = 1.0                                                   # solver units: speed 1, length length_ref
    wheels = [b for b in plan.solver_config.bodies if b.role == "wheel"]
    assert len(wheels) == 4
    for w in wheels:
        c = np.asarray(w.center)
        r = 0.3 / plan.length_ref
        ax = np.asarray(w.axis) / np.linalg.norm(w.axis)
        assert ax[1] == pytest.approx(1.0)
        assert w.omega == pytest.approx(-U / r, rel=0.02)
        contact = c - np.array([0.0, 0.0, r])
        velocity = w.omega * np.cross(ax, contact - c)
        assert velocity[0] == pytest.approx(U, rel=0.02)          # moves with the road
        assert (w.omega * np.cross(ax, np.array([0.0, 0.0, r])))[0] < 0   # the top moves upstream
    assert "-U / r about +y" in " ".join(type(d.meshes[0]).__doc__.split())


def test_wing_sideslip_loads_the_windward_left_wing():
    """Positive sideslip on a wing with dihedral: the wind comes from the left (-y), the left
    half sees a larger angle of attack and carries more lift, so the wing rolls right wing
    down -- a negative rolling moment about +x (x points aft)."""
    w = WingSpec([WingSection(0.0, 1.0, 0.0, 0.0, airfoil="0012"),
                  WingSection(4.0, 1.0, 0.0, 4.0 * math.tan(math.radians(6.0)), airfoil="0012")],
                 n_chord=4, n_span=12)
    p = analyze_wing(w, alpha_deg=4.0, speed=30.0, method="vlm", viscous=False, beta_deg=5.0)
    assert p.converged
    left = p.cl_local[p.y < 0].sum()
    right = p.cl_local[p.y > 0].sum()
    assert left > right
    assert p.c_roll < 0
    p0 = analyze_wing(w, alpha_deg=4.0, speed=30.0, method="vlm", viscous=False, beta_deg=0.0)
    assert abs(p0.c_roll) < 1e-9
