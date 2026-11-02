"""Subnormal flushing (aerolume.solvers.fpmode): on only while CPU solvers step, never left on.

``torch.set_flush_denormal(True)`` sets the calling thread's flush-to-zero bits until
someone clears them, and scipy's ``cKDTree`` then recurses until the native stack
overflows -- a fatal crash of the whole process. The solvers now restore the previous
mode after stepping, and the geometry code's k-d trees run with IEEE subnormals even if
a caller left flushing on.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from aerolume.solvers.fpmode import flush_denormal, flushing_subnormals, ieee_subnormals


@pytest.fixture(autouse=True)
def ieee_after():
    yield
    torch.set_flush_denormal(False)


def test_flush_denormal_restores_the_previous_mode():
    assert not flushing_subnormals()
    with flush_denormal():
        assert flushing_subnormals()
        with flush_denormal():
            assert flushing_subnormals()
        assert flushing_subnormals()                     # the inner block found it on and leaves it on
    assert not flushing_subnormals()
    with flush_denormal(False):
        assert not flushing_subnormals()
    torch.set_flush_denormal(True)
    with flush_denormal():
        pass
    assert flushing_subnormals()                         # it was on before: still on


def test_ieee_subnormals_switches_flushing_off_and_back():
    torch.set_flush_denormal(True)
    with ieee_subnormals():
        assert not flushing_subnormals()
        assert float(np.float64(1e-310) * 1.0) > 0
    assert flushing_subnormals()


def test_cpu_solvers_leave_the_mode_as_they_found_it():
    from aerolume.solvers.ns2d import FlowSolver, SolverConfig
    from aerolume.solvers.ns3d import FlowSolver3D, SolverConfig3D

    s2 = FlowSolver(SolverConfig(nx=32, ny=16, h=1 / 8, x0=-1.0, y0=-1.0, re=50.0, device="cpu", perturb=False))
    s2.advance(3)
    assert not flushing_subnormals()
    s3 = FlowSolver3D(SolverConfig3D(nx=12, ny=8, nz=8, h=1 / 4, x0=-1.5, y0=-1.0, z0=-1.0, re=20.0, bodies=[],
                                     device="cpu", dtype="float64", perturb=False))
    s3.advance(3)
    assert not flushing_subnormals()


def test_kd_tree_code_survives_a_caller_that_left_flushing_on():
    """The original crash, in a child process (a regression would kill the test runner)."""
    code = textwrap.dedent("""
        import torch
        from aerolume.geometry3d import box, design_from_meshes
        torch.set_flush_denormal(True)
        meshes = [box(1, 1, 1, center=(0, 0, 0.5), name="a"), box(0.4, 0.4, 0.4, center=(1.2, 0, 0.3), name="b"),
                  box(0.4, 0.4, 0.4, center=(-1.2, 0, 0.3), name="c")]
        d = design_from_meshes(meshes, "three boxes", kind="vehicle", units="m")
        print("ok", d.symmetric)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0 and "ok True" in r.stdout, (r.returncode, r.stdout[-500:], r.stderr[-2000:])
