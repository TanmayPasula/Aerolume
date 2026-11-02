"""Robust voxelisation and narrow-band signed distance (aerolume.geometry3d.voxel)."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import aerolume.geometry3d as G
from aerolume.geometry3d import voxel as VX
from aerolume.model import Mesh3D

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "geometry3d"


def _sphere_grid(d=1.0, cells_per_d=32, n=64, offset=(0.0, 0.0, 0.0)):
    s = G.sphere(d, n, center=offset)
    h = d / cells_per_d
    g = VX.Grid(44, 44, 44, h, offset[0] - 22 * h + 0.37 * h, offset[1] - 22 * h + 0.11 * h,
                offset[2] - 22 * h - 0.29 * h)          # off-centre on purpose
    return s, g


def _cap_removed(m: Mesh3D, direction, half_angle_deg: float) -> Mesh3D:
    c = m.vertices[m.faces].mean(1)
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    keep = (c / np.linalg.norm(c, axis=1, keepdims=True)) @ d < math.cos(math.radians(half_angle_deg))
    return Mesh3D("leaky", m.vertices, m.faces[keep])


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def test_grid_conventions():
    g = VX.Grid(4, 3, 2, 0.5, 1.0, -1.0, 0.0)
    assert g.shape == (2, 3, 4) and g.cells == 24
    np.testing.assert_allclose(g.centres("x"), [1.25, 1.75, 2.25, 2.75])
    assert g.extent == (1.0, 3.0, -1.0, 0.5, 0.0, 1.0)
    b = G.box(1, 1, 1)
    ga = VX.grid_around(b, 0.1, pad=2)
    assert ga.x0 == pytest.approx(-0.7) and ga.nx == 14
    # cell (k, j, i) is centred at (x0 + (i + .5) h, ...): a box covering exactly one cell
    one = G.box(0.5, 0.5, 0.5, center=(1.75, -0.25, 0.75))
    ins = VX.inside_mask(one, g, device="cpu")
    assert ins.sum() == 1 and bool(ins[1, 1, 1])


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def test_sphere_volume_from_fraction_within_half_percent():
    s, g = _sphere_grid()
    info: dict = {}
    frac = VX.voxelize(s, g, device="cpu", info=info)
    vol = float(frac.sum()) * g.h ** 3
    assert vol == pytest.approx(G.volume(s), rel=0.005)
    assert vol == pytest.approx(math.pi / 6, rel=0.005)
    # the remaining bias is the ramp's curvature term, +1/(2 R^2) with R in cells
    assert vol / G.volume(s) - 1 == pytest.approx(1 / (2 * 16**2), abs=0.0008)
    assert info["layers"] == 1 and info["odd_rays"] == [0, 0, 0] and info["undecided_cells"] == 0
    assert 0.0 <= float(frac.min()) and float(frac.max()) <= 1.0


def test_sdf_band_is_exact_for_a_box():
    b = G.box(1.0, 0.6, 0.4, center=(0.05, 0.0, -0.02))
    g = VX.grid_around(b, 0.05, pad=3)
    band = 0.12
    sd = VX.sdf_band(b, g, band, device="cpu", dtype=torch.float64).numpy()
    zc, yc, xc = np.meshgrid(g.centres("z"), g.centres("y"), g.centres("x"), indexing="ij")
    q = np.abs(np.stack([xc - 0.05, yc, zc + 0.02])) - np.array([0.5, 0.3, 0.2])[:, None, None, None]
    out = np.linalg.norm(np.maximum(q, 0), axis=0)
    exact = out + np.minimum(q.max(0), 0)
    inband = np.abs(exact) < band - 1e-9
    np.testing.assert_allclose(sd[inband], exact[inband], atol=1e-12)
    np.testing.assert_allclose(sd[~inband], np.sign(exact[~inband]) * band, atol=0)


def test_sdf_band_sphere_distance_and_sign():
    s, g = _sphere_grid(n=96)
    sd = VX.sdf_band(s, g, 3 * g.h, device="cpu", dtype=torch.float64).numpy()
    zc, yc, xc = np.meshgrid(g.centres("z"), g.centres("y"), g.centres("x"), indexing="ij")
    exact = np.sqrt(xc**2 + yc**2 + zc**2) - 0.5
    inband = np.abs(sd) < 3 * g.h
    facet = 0.5 * (1 - math.cos(math.pi / 96)) * 2          # inscribed polyhedron: at most ~2 sagittas
    assert np.abs(sd[inband] - exact[inband]).max() < facet + 1e-9
    assert np.all(np.sign(sd[np.abs(exact) > g.h]) == np.sign(exact[np.abs(exact) > g.h]))


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

def test_leaky_sample_file_voxelises_like_the_closed_sphere():
    (leaky,) = G.read_mesh(EXAMPLES / "leaky_sphere.stl")
    assert not G.is_watertight(leaky)
    closed = G.sphere(1.0, 32)
    h = 1 / 32
    g = VX.grid_around(closed, h, pad=3)
    info: dict = {}
    fl = VX.voxelize(leaky, g, device="cpu", info=info)
    fc = VX.voxelize(closed, g, device="cpu")
    assert sum(info["odd_rays"]) > 0                         # rays through the hole abstained
    # the hole (20 degree half-angle) is closed flat across its rim: the volume lost is
    # the spherical cap above the rim, 0.62 % of the sphere -- nothing else changes
    rim = math.radians(20)
    hc = 0.5 * (1 - math.cos(rim))
    cap = math.pi * hc * hc * (3 * 0.5 - hc) / 3
    lost = float(fc.sum() - fl.sum()) * h**3
    assert lost == pytest.approx(cap, rel=0.25)
    ins_l = VX.inside_mask(leaky, g, device="cpu")
    ins_c = VX.inside_mask(closed, g, device="cpu")
    assert int((ins_l != ins_c).sum()) < 0.02 * int(ins_c.sum())


@pytest.mark.parametrize("deg", [5, 12])
def test_small_holes_change_almost_nothing(deg):
    s, g = _sphere_grid()
    ref = VX.voxelize(s, g, device="cpu")
    f = VX.voxelize(_cap_removed(s, (0.3, -0.6, 0.74), deg), g, device="cpu")
    assert float(f.sum()) == pytest.approx(float(ref.sum()), rel=0.001)


def test_flipped_faces_soup_and_duplicates_change_nothing():
    """Same inside cells; fractions equal up to float32 rounding (corner order enters it)."""
    s, g = _sphere_grid(n=48)
    ref = VX.voxelize(s, g, device="cpu")
    ref_in = VX.inside_mask(s, g, device="cpu")
    rng = np.random.default_rng(7)
    f = s.faces.copy()
    flip = rng.random(len(f)) < 0.35
    f[flip] = f[flip][:, [0, 2, 1]]
    t = s.vertices[s.faces].reshape(-1, 3)
    variants = [Mesh3D("flip", s.vertices, f),
                Mesh3D("soup", t, np.arange(len(t)).reshape(-1, 3)),
                Mesh3D("dup", s.vertices, np.vstack([s.faces, s.faces[::5, [1, 0, 2]]]))]
    for m in variants:
        assert torch.equal(VX.inside_mask(m, g, device="cpu"), ref_in), m.name
        assert float((VX.voxelize(m, g, device="cpu") - ref).abs().max()) < 1e-6, m.name
    # flipped AND leaky at once
    fl = _cap_removed(Mesh3D("fl", s.vertices, f), (1, 1, 1), 10)
    assert float(VX.voxelize(fl, g, device="cpu").sum()) == pytest.approx(float(ref.sum()), rel=0.002)


def test_overlapping_shells_in_one_mesh_are_united():
    a = G.sphere(0.6, 48, center=(-0.15, 0, 0))
    b = G.sphere(0.6, 48, center=(0.15, 0, 0))
    merged = G.merge([a, b])
    h = 1 / 40
    g = VX.grid_around(merged, h)
    info: dict = {}
    fm = VX.voxelize(merged, g, device="cpu", info=info)
    fs = VX.voxelize([a, b], g, device="cpu")
    assert info["layers"] == 2 and info["buried_triangles"] > 0
    assert torch.equal(fm, fs)
    # the polyhedral union, counted on a 4x finer grid of the two separate inside tests
    gf = VX.Grid(4 * g.nx, 4 * g.ny, 4 * g.nz, h / 4, g.x0, g.y0, g.z0)
    fine = VX.inside_mask(a, gf, device="cpu") | VX.inside_mask(b, gf, device="cpu")
    assert float(fm.sum()) * h**3 == pytest.approx(float(fine.sum()) * (h / 4) ** 3, rel=0.004)
    r, dd = 0.3, 0.3                                          # union of the true spheres (faceting: -0.6 %)
    lens = math.pi * (4 * r + dd) * (2 * r - dd) ** 2 / 12
    union = 2 * 4 / 3 * math.pi * r**3 - lens
    assert float(fm.sum()) * h**3 == pytest.approx(union, rel=0.012)


def _box_sdf(xc, yc, zc, centre, size):
    q = np.abs(np.stack([xc - centre[0], yc - centre[1], zc - centre[2]])) - 0.5 * np.asarray(size)[:, None, None, None]
    return np.linalg.norm(np.maximum(q, 0), axis=0) + np.minimum(q.max(0), 0)


def test_part_pushed_through_the_body_surface():
    """A small box (4 cells wide) through the big box's floor, in one mesh: the buried
    patch of floor and the buried part of the small box must not leave an internal
    surface in the distance field.

    Outside the union the field equals the exact union distance; inside, the depth
    is within about half a cell of the exact one (only near the crease, where a
    sub-cell sliver of buried surface may survive), so everything more than 1.3
    cells deep is fully solid. Without the buried-face rules the patch of floor
    inside the small box left an internal layer with fraction ~0.6 several cells
    deep.
    """
    big_c, big_s = (0.0, 0.0, 0.0), (1.0, 0.6, 0.4)
    sm_c, sm_s = (0.13, 0.05, -0.2), (0.21, 0.11, 0.3)
    m = G.merge([G.box(*big_s, center=big_c), G.box(*sm_s, center=sm_c)])
    for h in (1 / 40, 1 / 37.3):                               # two alignments of the grid
        g = VX.grid_around(m, h)
        info: dict = {}
        sd = VX.sdf_band(m, g, 3 * h, device="cpu", dtype=torch.float64, info=info).numpy()
        zc, yc, xc = np.meshgrid(g.centres("z"), g.centres("y"), g.centres("x"), indexing="ij")
        exact = np.minimum(_box_sdf(xc, yc, zc, big_c, big_s), _box_sdf(xc, yc, zc, sm_c, sm_s))
        out = exact > 0
        np.testing.assert_allclose(np.minimum(sd[out], 3 * h), np.minimum(exact[out], 3 * h), atol=1e-12)
        near = (exact < 0) & (exact > -3 * h)
        err = (sd[near] - exact[near]) / h
        assert err.min() > -0.65 and err.max() < 0.3, (err.min(), err.max())
        assert np.all(sd[exact < -1.3 * h] <= -h)               # fraction exactly 1
        assert info["layers"] == 2 and info["buried_triangles"] > 0
    # a sphere half sunk into a box: the fraction against the analytic union
    s = G.sphere(0.3, 64, center=(0.0, 0.0, 0.2))
    mm = G.merge([G.box(1.0, 0.6, 0.4), s])
    h = 1 / 50
    g = VX.grid_around(mm, h)
    f = VX.voxelize(mm, g, device="cpu").numpy()
    zc, yc, xc = np.meshgrid(g.centres("z"), g.centres("y"), g.centres("x"), indexing="ij")
    exact = np.minimum(_box_sdf(xc, yc, zc, (0, 0, 0), (1.0, 0.6, 0.4)),
                       np.sqrt(xc**2 + yc**2 + (zc - 0.2) ** 2) - 0.15)
    fe = VX.solid_fraction(torch.tensor(exact), h).numpy()
    assert np.abs(f - fe).max() < 0.15                          # measured 0.106 (faceting included)
    assert f.sum() == pytest.approx(fe.sum(), rel=0.003)


def test_box_with_a_missing_face_and_open_sheet():
    b = G.box(0.8, 0.6, 0.5)
    h = 1 / 32
    g = VX.grid_around(b, h, pad=3)
    keep = np.ones(12, dtype=bool)
    keep[[8, 9]] = False                                       # the whole +x face is missing
    open_ = Mesh3D("open", b.vertices, b.faces[keep])
    fo = VX.voxelize(open_, g, device="cpu")
    assert float(fo.sum()) * h**3 == pytest.approx(0.24, rel=0.02)
    # a single square sheet: thickened to one cell by default, a smear without
    sq = Mesh3D("sheet", [[-0.3, -0.3, 0.013], [0.3, -0.3, 0.013], [0.3, 0.3, 0.013], [-0.3, 0.3, 0.013]],
                [[0, 1, 2], [0, 2, 3]])
    info: dict = {}
    f1 = float(VX.voxelize(sq, g, device="cpu", info=info).sum()) * h**3
    f0 = float(VX.voxelize(sq, g, device="cpu", sheet_thickness=0).sum()) * h**3
    f2 = float(VX.voxelize(sq, g, device="cpu", sheet_thickness=2 * h).sum()) * h**3
    assert info["sheet_faces"] == 2
    assert f1 == pytest.approx(0.36 * h, rel=0.25)
    # unthickened, the ramp of a zero-thickness sheet sums to 0.25-0.5 h per area depending
    # on where the sheet sits between cell centres: a weak, half-strength smear
    assert 0.2 * 0.36 * h < f0 < 0.6 * 0.36 * h
    assert f2 == pytest.approx(2 * f1, rel=0.2)
    # a thin but closed plate is never thickened (its shape is defined)
    plate = G.box(0.6, 0.6, 0.3 * h, center=(0, 0, 0.013))
    info = {}
    VX.voxelize(plate, g, device="cpu", info=info)
    assert info["sheet_faces"] == 0


def test_inputs_chunking_and_edge_cases():
    s, g = _sphere_grid(n=24)
    ref = VX.voxelize(s, g, device="cpu")
    vt = torch.tensor(s.vertices)
    assert torch.equal(VX.voxelize((vt, torch.tensor(s.faces)), g, device="cpu"), ref)
    assert torch.equal(VX.voxelize([(s.vertices, s.faces)], g, device="cpu"), ref)
    assert torch.equal(VX.voxelize(s, g, device="cpu", budget_mb=0.05), ref)     # many tiny chunks
    assert float(VX.voxelize([], g, device="cpu").sum()) == 0
    # a body cut by the domain boundary: what lies inside the grid is still right
    cut = G.sphere(1.0, 48, center=(0.6, 0, 0))
    gi = VX.Grid(32, 48, 48, 1 / 32, -0.5, -0.75, -0.75)
    ins = VX.inside_mask(cut, gi, device="cpu")
    zc, yc, xc = np.meshgrid(gi.centres("z"), gi.centres("y"), gi.centres("x"), indexing="ij")
    exact = (xc - 0.6) ** 2 + yc**2 + zc**2 < 0.25
    assert int((ins.numpy() != exact).sum()) < 0.01 * exact.sum()
    with pytest.raises(ValueError):
        VX.voxelize(s, (0, 4, 4, 0.1, 0, 0, 0), device="cpu")
    with pytest.raises(ValueError):
        VX.voxelize(Mesh3D("bad", s.vertices, s.faces + 10_000), g, device="cpu")


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _tiny():
    s = G.sphere(0.6, 12)
    s.vertices = s.vertices + np.random.default_rng(1).normal(0, 0.01, s.vertices.shape)
    return s


def test_sdf_band_gradient_matches_finite_differences():
    s = _tiny()
    h = 0.05
    g = VX.grid_around(s, h, pad=3)
    inside = VX.inside_mask(s, g, device="cpu")
    F = torch.tensor(s.faces)
    w = torch.rand(g.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    def J(V):
        return (w * VX.sdf_band([(V, F)], g, 2.5 * h, inside=inside, sheet_thickness=0)).sum()

    V = torch.tensor(s.vertices, requires_grad=True)
    J(V).backward()
    rng = np.random.default_rng(2)
    eps = 1e-6
    rel = []
    for _ in range(12):
        i, c = int(rng.integers(len(s.vertices))), int(rng.integers(3))
        Vp, Vm = V.detach().clone(), V.detach().clone()
        Vp[i, c] += eps
        Vm[i, c] -= eps
        fd = (J(Vp).item() - J(Vm).item()) / (2 * eps)
        rel.append(abs(V.grad[i, c].item() - fd) / max(abs(fd), 1e-9))
    assert np.median(rel) < 1e-5 and max(rel) < 1e-3


def test_fraction_volume_derivative_equals_mesh_volume_derivative():
    """d/de of the summed solid fraction under a uniform scaling (1 + e) is about 3 V.

    The exact statement is that autograd equals the central difference of the same
    discrete volume, and that this equals the central difference computed with the exact
    signed distance (generalised winding number, no ray jitter): this mesh puts one cell
    centre 0.0006 cells inside a face, where the jittered parity says "outside" and only
    the through-the-surface rule gives the true side (and derivative). 3 V holds only to
    the discretisation of the ramp (0.5 % here).
    """
    from aerolume.solvers.ns3d.geometry import mesh_signed_distance

    s = _tiny()
    h = 0.05
    g = VX.grid_around(s, h, pad=3)
    inside = VX.inside_mask(s, g, device="cpu")
    F = torch.tensor(s.faces)

    def vol(e):
        V = torch.tensor(s.vertices) * (1 + e)
        sd = VX.sdf_band([(V, F)], g, h, inside=inside, sheet_thickness=0)
        return VX.solid_fraction(sd, h).sum() * h**3, sd

    e = torch.zeros((), dtype=torch.float64, requires_grad=True)
    v0, sd = vol(e)
    v0.backward()
    assert sd.dtype == torch.float64 and sd.requires_grad
    eps = 1e-6
    fd = (vol(eps)[0].item() - vol(-eps)[0].item()) / (2 * eps)
    assert e.grad.item() == pytest.approx(fd, rel=1e-8)
    zc, yc, xc = np.meshgrid(g.centres("z"), g.centres("y"), g.centres("x"), indexing="ij")
    P = torch.tensor(np.stack([xc.ravel(), yc.ravel(), zc.ravel()], 1))

    def vol_exact(e):
        d = mesh_signed_distance(torch.tensor(s.vertices) * (1 + e), F, P).clamp(-h, h)
        return float(VX.solid_fraction(d, h).sum()) * h**3
    assert e.grad.item() == pytest.approx((vol_exact(eps) - vol_exact(-eps)) / (2 * eps), rel=1e-8)
    exact0 = mesh_signed_distance(torch.tensor(s.vertices), F, P).reshape(g.shape).clamp(-h, h)
    torch.testing.assert_close(sd.detach(), exact0, rtol=0, atol=1e-12)    # every sign right, jitter or not
    assert e.grad.item() == pytest.approx(3 * G.volume(s), rel=1e-2)


def _box_on_grid(centre):
    """A 0.8 x 0.6 x 0.4 box and a grid of h = 0.1 whose cell centres fall where ``centre`` puts them."""
    g = VX.Grid(16, 14, 12, 0.1, -0.8, -0.65, -0.55)
    return G.box(0.8, 0.6, 0.4, center=centre), g


@pytest.mark.parametrize("centre, tol", [
    ((0.05, 0.0412, 0.0291), 1e-4),     # x faces exactly on rows of cell centres
    ((0.05, 0.05, 0.05), 1e-4),         # and the y / z faces at equal offsets: medial planes through centres
])
def test_distance_is_differentiable_through_the_surface(centre, tol):
    """Cell centres exactly on a face get the face's signed plane distance (value 0, the face's
    derivative), and cells as far from two faces the mean of their derivatives: autograd then
    matches central differences for face moves and random vertex moves. Before, a centre on a
    face had zero derivative (sqrt at 0) and the jittered parity sign, which flips as the face
    moves through it (the objective jumped), and a tie credited one face only: median error
    25-30 % over these modes."""
    b, g = _box_on_grid(centre)
    V0 = torch.tensor(b.vertices, dtype=torch.float64)
    F = torch.tensor(b.faces)
    w = torch.rand(g.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(3))

    def J(V):
        sd = VX.sdf_band([(V, F)], g, 2.5 * g.h, device="cpu")
        return (w * VX.solid_fraction(sd, g.h)).sum(), sd

    V = V0.clone().requires_grad_(True)
    j0, sd = J(V)
    j0.backward()
    on_face = int((sd.detach().abs() < 1e-12).sum())
    assert on_face >= 2 * 6 * 4                                      # two whole x faces of centres
    modes = []
    for axis in range(3):
        for side in (-1.0, 1.0):
            m = torch.zeros_like(V0)
            m[torch.sign(V0[:, axis] - V0[:, axis].mean()) == side, axis] = 1.0
            modes.append(m)
    rng = np.random.default_rng(5)
    modes += [torch.tensor(rng.normal(size=V0.shape)) for _ in range(4)]
    eps = 1e-6
    for m in modes:
        fd = (J(V0 + eps * m)[0].item() - J(V0 - eps * m)[0].item()) / (2 * eps)
        ad = float((V.grad * m).sum())
        assert abs(ad - fd) <= tol * abs(fd), (ad, fd)


def _open_half_sphere() -> Mesh3D:
    s = G.sphere(1.0, 48)
    return Mesh3D("half", s.vertices, s.faces[(s.vertices[s.faces][:, :, 1] >= -1e-12).all(1)])


CUT_CASES = {
    # a symmetric "car" of two overlapping boxes plus an open roof sheet, half domain (y >= 0) on the road
    "half domain": (lambda: [G.merge([G.box(1.0, 0.5, 0.2, center=(0.0, 0.0, 0.2)),
                                      G.box(0.5, 0.4, 0.2, center=(0.05, 0.0, 0.33))]),
                             Mesh3D("spoiler", [[0.3, -0.3, 0.461], [0.5, -0.3, 0.461], [0.5, 0.3, 0.461],
                                                [0.3, 0.3, 0.461]], [[0, 1, 2], [0, 2, 3]])],
                    VX.Grid(32, 14, 16, 0.05, -0.8, 0.0, 0.0)),
    # the open half of a sphere seen through a window that cuts it across x
    "window": (lambda: [_open_half_sphere()], VX.Grid(12, 32, 32, 1 / 16, -0.25, -1.0, -1.0)),
    # a wing modelled as one open surface, crossing the symmetry plane: stays a thickened sheet
    "sheet across the plane": (lambda: [Mesh3D("sheet", [[-0.2, -0.8, 0.013], [0.2, -0.8, 0.013],
                                                         [0.2, 0.8, 0.013], [-0.2, 0.8, 0.013]],
                                               [[0, 1, 2], [0, 2, 3]])],
                               VX.Grid(16, 20, 12, 0.05, -0.4, 0.0, -0.3)),
}


@pytest.mark.parametrize("case", sorted(CUT_CASES))
def test_a_grid_that_cuts_the_surface_matches_a_covering_grid(case):
    """Symmetry planes, roads and solver windows cut the body: the distance inside the cut grid
    must equal that of a grid holding the whole surface. Before, faces whose probes left the
    grid were taken for open sheets and thickened (0.42 h too small through the window)."""
    make, g = CUT_CASES[case]
    meshes = make()
    pad = 12
    big = VX.Grid(g.nx + 2 * pad, g.ny + 2 * pad, g.nz + 2 * pad, g.h, g.x0 - pad * g.h, g.y0 - pad * g.h,
                  g.z0 - pad * g.h)
    ic: dict = {}
    ib: dict = {}
    cut = VX.sdf_band(meshes, g, 2.32 * g.h, device="cpu", dtype=torch.float64, info=ic)
    cover = VX.sdf_band(meshes, big, 2.32 * g.h, device="cpu", dtype=torch.float64, info=ib)
    cover = cover[pad:pad + g.nz, pad:pad + g.ny, pad:pad + g.nx]
    assert float((cut - cover).abs().max()) < 1e-9 * g.h
    assert ic["sheet_faces"] == ib["sheet_faces"]
    assert any(ic["cut_padding"])                        # the classification field was grown past the cut


def test_staggered_grids_share_the_preparation():
    """MAC face grids: centres at the face centres; sdf_bands == sdf_band on each grid."""
    g = VX.Grid(10, 8, 6, 0.1, 0.0, -0.4, -0.3)
    gu = g.faces("x")
    assert gu.shape == (6, 8, 11)
    np.testing.assert_allclose(gu.centres("x"), np.arange(11) * 0.1, atol=1e-15)   # x0 + i h
    np.testing.assert_allclose(gu.centres("y"), g.centres("y"))
    assert g.faces("z").shape == (7, 8, 10) and g.faces("y").shape == (6, 9, 10)
    s = G.sphere(0.5, 24, center=(0.5, 0.0, 0.0))
    grids = [g, gu, g.faces("y"), g.faces("z")]
    V = torch.tensor(s.vertices, requires_grad=True)
    many = VX.sdf_bands([(V, torch.tensor(s.faces))], grids, 0.25)
    for gr, sd in zip(grids, many):
        assert sd.shape == gr.shape
        torch.testing.assert_close(sd.detach(), VX.sdf_band(s, gr, 0.25, device="cpu", dtype=torch.float64))
    sum(m.sum() for m in many).backward()
    assert V.grad is not None and float(V.grad.abs().sum()) > 0


def test_no_graph_without_requires_grad():
    s = _tiny()
    g = VX.grid_around(s, 0.05)
    sd = VX.sdf_band([(torch.tensor(s.vertices), torch.tensor(s.faces))], g, 0.1)
    assert not sd.requires_grad and sd.dtype == torch.float64


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@needs_cuda
def test_gpu_matches_cpu():
    s, g = _sphere_grid(n=48)
    lk = _cap_removed(s, (0.2, 0.9, -0.3), 15)
    for m in (s, lk):
        fc = VX.voxelize(m, g, device="cpu")
        fg = VX.voxelize(m, g, device="cuda")
        assert fg.device.type == "cuda"
        assert torch.equal(VX.inside_mask(m, g, device="cpu"), VX.inside_mask(m, g, device="cuda").cpu())
        assert float((fg.cpu() - fc).abs().max()) < 1e-5


@pytest.mark.gpu
@needs_cuda
def test_gpu_gradient_matches_cpu():
    s = _tiny()
    g = VX.grid_around(s, 0.05)
    grads = []
    for dev in ("cpu", "cuda"):
        V = torch.tensor(s.vertices, device=dev, requires_grad=True)
        sd = VX.sdf_band([(V, torch.tensor(s.faces, device=dev))], g, 0.1)
        VX.solid_fraction(sd, 0.05).sum().backward()
        grads.append(V.grad.cpu())
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-9, atol=1e-12)


@pytest.mark.gpu
@needs_cuda
def test_gpu_performance_target_256x128x128_200k_triangles():
    """DESIGN 7.4 target: 256 x 128 x 128 cells x 200k triangles in < 2 s on the GPU."""
    s = G.sphere(1.0, 448)
    v = s.vertices * np.array([2.6, 1.0, 0.8])
    v[:, 2] += 0.15 * np.sin(3 * v[:, 0])
    m = Mesh3D("body", v, s.faces)
    assert len(m.faces) >= 199_000
    g = VX.Grid(256, 128, 128, 4.0 / 256, -2.0, -1.0, -1.0)
    VX.voxelize(G.sphere(0.5, 16), g, device="cuda")                 # warm-up (kernel loading)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    frac = VX.voxelize(m, g, device="cuda")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() - base
    assert dt < 2.0, f"{dt:.2f} s"
    assert peak < 1.5 * 2**30, f"{peak / 2**20:.0f} MB"
    assert float(frac.sum()) * g.h**3 == pytest.approx(G.volume(m), rel=0.005)
    del frac
    torch.cuda.empty_cache()
