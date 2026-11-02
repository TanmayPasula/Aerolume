"""3D shape-sensitivity maps: dJ/dn on surface meshes by autograd through the 3D solver.

This is the 3D counterpart of :mod:`aerolume.optimize.sensitivity` (DESIGN.md
7.8): the AirShaper-style "adjoint map" of any imported shape, computed by one
reverse sweep through a short window of the differentiable 3D Navier-Stokes
solver (:mod:`aerolume.solvers.ns3d`). It answers, for every vertex of the
user's mesh at once, "if the surface moved outward here, how much would the
drag (lift, L/D, pitching moment) change?". Conventions, sign and honesty notes
are those of the 2D maps (``docs/theory_sensitivity.md``, whose section 9 is
the 3D part: method, verification, measured cost and magnitude).

Pipeline
--------
1. **Developed flow.** The caller's :class:`~aerolume.solvers.ns3d.FlowSolver3D`
   (the runner passes its finished solver), or the flow is developed here up to
   the plan's ``t_avg_start``. Its state ``(u, v, w)`` is held fixed.
2. **Parametrisation.** Every body mesh is refined by conforming edge bisection
   until no edge is longer than half a cell (inserted vertices lie on the
   original edges and faces, so the shape -- and every mask -- is unchanged; the
   user's vertices keep their indices). Each refined vertex ``i`` gets a
   displacement ``d_i`` along its outward normal (area-weighted, with every face
   oriented towards the fluid by probing the solid fraction, so meshes of any
   winding work). ``d -> vertices -> sdf_band (shared voxeliser) -> C1 masks ->
   implicit penalisation -> projection -> force impulse`` is differentiable.
3. **Objective.** From the fixed state the solver runs ``lead + average``
   convective times at a fixed time step; J is CD, CL, L/D or Cm of the
   coefficients averaged over the last ``average`` (full-body values: a
   half-domain run reports the mirrored body, as the solver does).
4. **Reverse sweep** with O(sqrt(N)) memory: the forward pass keeps the velocity
   every ~sqrt(N) steps (on the host when the GPU is short), the reverse pass
   recomputes one segment at a time and pulls the adjoint back step by step
   with a vector-Jacobian product of one solver step, captured -- forward and
   backward -- as a CUDA graph (``torch.cuda.make_graphed_callables``): 8 ms per
   VJP at 194 k cells instead of 120 ms eager (RTX 4060 Ti, float32, measured).
   This is checkpointing done by hand rather than with
   :func:`~aerolume.solvers.ns3d.rollout3d`, whose ``torch.utils.checkpoint``
   chunks re-record the activations of a whole chunk of steps at once and run
   eagerly; the per-step VJP holds one step's activations at a time and replays
   as a graph. Gradients reach the mask tensors only; one final backward through
   the geometry build turns them into dJ/dd_i.
5. **Map.** dJ/dd_i scales with the vertex's area; dividing by the vertex dual
   area gives the density g with ``dJ = integral g dn dA``. Per-vertex values
   alias with the grid (each vertex feels only the few faces of the one-cell
   mask band around it), so both the derivatives and the areas are smoothed by
   the same few dozen steps (44-99 measured) of a surface-Laplacian random walk on the refined
   mesh (a kernel of about 1.5 cells, the 2D smoothing) and their ratio is
   taken. With a symmetry plane the solver sees only y >= 0: the half-domain
   derivative is that of a mirror-symmetric pair of bumps (twice the one-sided
   value), so it is halved, and vertices on the missing half get the value of
   their mirror image.

Units and sign
--------------
Returned values are ``L_ref^2 * g`` in **1/m**: the change of J per metre of
outward normal displacement applied over a patch of surface of area L_ref^2.
For a small displacement field dn [m] at vertices with areas dA_i [m^2]::

    dJ ~= (1 / L_ref^2) * sum_i sens_i * dn_i * dA_i

(the 2D maps use L_ref * g; both are the non-dimensional density divided by
L_ref). Positive = pushing the surface outward (adding material) there
increases J. The pattern is reliable, the magnitude indicative: every map
carries :data:`MAGNITUDE_WARNING` (the 2D measurements of the window and
immersed-boundary biases, docs/theory_sensitivity.md section 5, and the 3D ones
of section 9.5).

Memory budget
-------------
A reverse sweep holds one differentiated step (its CUDA-graph pool), the
sqrt(N) checkpoints, the recomputed segment and the adjoint: measured in float32
about 130 bytes per cell plus 12 bytes per cell for every velocity state held
(1.28 GB above the flow solver at 1.56 M cells and 1026 steps).
:func:`compute_sensitivity3d` estimates that before doing anything costly and raises
:class:`SensitivityBudgetError` (a ``ValueError`` with a readable message, which
the runner turns into a warning) when the grid exceeds ``max_cells`` (default
:data:`DEFAULT_MAX_CELLS`: the ~1.5 M-cell sensitivity budget of DESIGN 7.1 with
a margin) or the estimate exceeds the free device memory.

How the 3D runner calls it
--------------------------
::

    from aerolume.optimize import Cancelled, attach_sensitivity3d, compute_sensitivity3d
    try:
        maps = compute_sensitivity3d(plan.design, plan.sim, plan=plan, solver=solver,
                                     objective=plan.sim.cfd.sensitivity_objective,
                                     progress=sub_progress, cancel=cancel)
        attach_sensitivity3d(result, maps)   # fills SurfaceData3D.sensitivity, adds notes / warnings
    except Cancelled:
        result.warnings.append("Sensitivity map skipped: stopped by the user.")
    except Exception as exc:                # SensitivityBudgetError included: the forces stay valid
        result.warnings.append(f"Sensitivity map skipped: {exc}")
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import torch

from ..model import CFD3DResult, Design, Simulation
from ..solvers.fpmode import ieee_subnormals
from ..solvers.ns3d import (
    FlowSolver3D,
    ImmersedMesh,
    SolverConfig3D,
    State3D,
    build_geometry3d,
    make_operators3d,
)
from ..solvers.ns3d.core import Ops3D, cfl_dt3d, momentum_rhs3d, step3d_full
from ..solvers.ns3d.fields import trilinear
from ..solvers.ns3d.geometry import Geometry3D, body_window3d
from ..solvers.ns3d.poisson import ieee_matmul
from .common import Progress, ProgressFn, check_cancel
from .sensitivity import (
    CHECK_EVERY,
    DT_SAFETY,
    MAX_STEPS,
    OFFLOAD_FRACTION,
    WindowValue,
    choose_window,
    shedding_period,
)

__all__ = [
    "DEFAULT_MAX_CELLS",
    "MAGNITUDE_WARNING",
    "OBJECTIVES3D",
    "SensitivityBudgetError",
    "SensitivityMaps3D",
    "SurfaceParam",
    "WindowedObjective3D",
    "attach_sensitivity3d",
    "compute_sensitivity3d",
    "develop_flow3d",
    "estimate_memory_bytes",
    "estimate_seconds",
    "finite_difference_check3d",
    "normal_modes3d",
    "refine_mesh",
    "surface_smoother",
    "vertex_frame",
    "visible_dual_area",
]

#: Objectives and display names (3D coefficient symbols, DESIGN 7.2). J is formed from the
#: window-mean full-body coefficients [CD, CS, CL, C_roll, Cm, Cn] summed over bodies;
#: "ld" is mean CL / mean CD.
OBJECTIVES3D = {"cd": "CD", "cl": "CL", "ld": "L/D", "cm": "Cm"}
#: Refined-mesh edge length [cells]: below the grid scale, as in 2D, so the map can show
#: everything the masks can feel. Coarser when the refined mesh would exceed MAX_REFINED_FACES.
REFINE_CELLS = 0.5
MAX_REFINED_FACES = 600_000
#: Map smoothing [cells]: the per-direction standard deviation of the surface random walk
#: that smooths derivatives and areas (the 2D value; per-vertex derivatives alias with the
#: grid, see the module docstring).
SMOOTH_CELLS = 1.5
#: Laziness of the random walk (probability of moving per step) and an iteration cap.
SMOOTH_LAMBDA = 0.5
SMOOTH_MAX_ITER = 400
#: Extra cells around the bodies in the fixed mask window (finite-difference steps are far
#: below a cell; 2D keeps 4 for its optimiser, which 3D does not have).
WINDOW_MARGIN_CELLS = 2.0
#: Probe distance [cells] of the face-orientation test (the solid fraction is compared a
#: cell outside and a cell inside along each face normal).
ORIENT_PROBE_CELLS = 1.0
#: Cell limits per device class above which the map is refused (DESIGN 7.1: sensitivity
#: runs <= ~1.5 M cells; every Draft grid passes). Time is the binding limit on a CPU.
DEFAULT_MAX_CELLS = {"gpu": 2_100_000, "cpu": 400_000}
#: Device bytes per cell of one differentiated step (the graph pool of a CUDA-graphed
#: forward + backward replay and its static buffers), float32; float64 doubles it. Measured
#: on the development GPU (docs/theory_sensitivity.md section 9): ~130 B/cell including the
#: constant states, fitted over 0.19 / 1.56 M cells; 200 keeps a margin.
STEP_BYTES_PER_CELL = 200
#: Device bytes per mask-window cell (masks, solid velocities, force weights and their
#: gradient accumulators: ~32 floats) and a fixed allowance (Poisson matrices, CUDA
#: context growth, allocator slack).
WINDOW_BYTES_PER_CELL = 32 * 4
FIXED_BYTES = 256 * 2**20
#: Share of the free device memory a sweep may plan to use.
MEMORY_FRACTION = 0.85
#: Cost of one step of the sweep (forward pass + recomputation + VJP) in nanoseconds per
#: cell, float32: measured 54 on the development GPU alone and 115 while another job used it
#: (CUDA graphs, 1.56 M cells), 5200 on its CPU (4 threads, eager, 58 k cells).
SWEEP_NS_PER_CELL = {"gpu": 80.0, "cpu": 5000.0}

#: The warning every 3D map carries (docs/theory_sensitivity.md 9.5 has the measurements).
MAGNITUDE_WARNING = (
    "Read the 3D sensitivity map's pattern rather than its exact size: it is the derivative of a short time "
    "average through the immersed boundary, which reacts more strongly to sub-cell shape changes than a real shape "
    "change does. On a sphere it was within 6 % of the physical slope at Re 30 and 28 % above it at Re 100 (16 cells "
    "per diameter), and the same method can be up to about twice too large in 2D; signs and the places that "
    "matter most are reliable.")

_GEOM3 = ("chi_u", "chi_v", "chi_w", "a_u", "a_v", "a_w", "k_u", "k_v", "k_w")
_COEF = {"cd": 0, "cs": 1, "cl": 2, "c_roll": 3, "cm": 4, "cn": 5}


class SensitivityBudgetError(ValueError):
    """The sensitivity sweep would not fit the cell or memory budget (message says why).

    Raised before any expensive work; the 3D runner turns it into a warning on
    the result (the forces are unaffected).
    """


# ---------------------------------------------------------------------------
# mesh helpers (numpy)
# ---------------------------------------------------------------------------

def _split_faces(V: np.ndarray, F: np.ndarray, M: np.ndarray) -> np.ndarray:
    """New faces for faces F whose edges (a,b), (b,c), (c,a) carry midpoints M (-1 = unsplit).

    Every child keeps its parent's winding. Two split edges: the shorter of the two
    possible diagonals is used, which keeps the children well shaped.
    """
    cnt = (M >= 0).sum(1)
    out = [F[cnt == 0]]
    s3 = cnt == 3
    if s3.any():
        a, b, c = F[s3].T
        m0, m1, m2 = M[s3].T
        out += [np.stack(t, 1) for t in ((a, m0, m2), (m0, b, m1), (m2, m1, c), (m0, m1, m2))]

    def rotated(sel: np.ndarray, first: np.ndarray):
        idx = (first[:, None] + np.arange(3)[None, :]) % 3
        return np.take_along_axis(F[sel], idx, 1), np.take_along_axis(M[sel], idx, 1)

    s1 = np.flatnonzero(cnt == 1)
    if s1.size:
        fr, mr = rotated(s1, np.argmax(M[s1] >= 0, axis=1))       # split edge -> (a, b)
        a, b, c = fr.T
        m0 = mr[:, 0]
        out += [np.stack((a, m0, c), 1), np.stack((m0, b, c), 1)]
    s2 = np.flatnonzero(cnt == 2)
    if s2.size:
        fr, mr = rotated(s2, (np.argmax(M[s2] < 0, axis=1) + 1) % 3)   # unsplit edge -> (c, a)
        a, b, c = fr.T
        m0, m1 = mr[:, 0], mr[:, 1]
        short = np.linalg.norm(V[a] - V[m1], axis=1) <= np.linalg.norm(V[m0] - V[c], axis=1)
        out.append(np.stack((m0, b, m1), 1))
        out.append(np.where(short[:, None], np.stack((a, m0, m1), 1), np.stack((m0, m1, c), 1)))
        out.append(np.where(short[:, None], np.stack((a, m1, c), 1), np.stack((a, m0, c), 1)))
    return np.concatenate(out).astype(np.int64)


def refine_mesh(vertices, faces, max_edge: float, *, max_faces: int | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Conforming edge bisection until no edge is longer than ``max_edge``.

    Every edge longer than the limit gets its midpoint (shared by the faces on
    both sides, so a welded mesh stays welded and closed components stay
    closed), and each face is split into 2, 3 or 4 children by the usual
    patterns; repeated until all edges are short. The original vertices keep
    their indices 0..N-1 (new ones are appended) and every new vertex lies on an
    original edge or face, so the shape is exactly unchanged. Winding is kept.
    Raises ``ValueError`` when the refined mesh would exceed ``max_faces``.
    """
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not (max_edge > 0 and math.isfinite(max_edge)) or len(F) == 0:
        return V.copy(), F.copy()
    for _ in range(64):
        e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)              # (M, 3, 2)
        length = np.linalg.norm(V[e[..., 1]] - V[e[..., 0]], axis=-1)
        split = length > max_edge * (1.0 + 1e-9)
        if not split.any():
            return V, F
        pairs = np.sort(e[split], axis=1)
        key = pairs[:, 0] * len(V) + pairs[:, 1]
        uk, inv = np.unique(key, return_inverse=True)
        lo, hi = uk // len(V), uk % len(V)
        mid = len(V) + np.arange(len(uk))
        V = np.vstack([V, 0.5 * (V[lo] + V[hi])])
        M = np.full(F.shape, -1, dtype=np.int64)
        M[split] = mid[inv.reshape(-1)]
        F = _split_faces(V, F, M)
        if max_faces is not None and len(F) > max_faces:
            raise ValueError(f"refining the mesh to edges of {max_edge:.3g} would need more than "
                             f"{max_faces:,} faces")
    return V, F


def vertex_frame(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(unit vertex normals (N, 3), vertex dual areas (N,)) of a triangle mesh.

    Normals are the area-weighted mean of the incident face normals (``face_normals``
    (M, 3) unit vectors when given -- e.g. oriented towards the fluid -- else the
    winding's own); the dual area is a third of the incident face areas
    (barycentric dual), so the dual areas sum to the surface area.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    if face_normals is None:
        fn = cr
    else:
        fn = np.asarray(face_normals, dtype=np.float64) * (2.0 * area)[:, None]
    n = np.zeros_like(V)
    dual = np.zeros(len(V))
    for k in range(3):
        np.add.at(n, F[:, k], fn)
        np.add.at(dual, F[:, k], area / 3.0)
    norm = np.linalg.norm(n, axis=1)
    return n / np.where(norm > 0, norm, 1.0)[:, None], dual


def surface_smoother(vertices: np.ndarray, faces: np.ndarray, sigma: float, *, lam: float = SMOOTH_LAMBDA,
                     max_iter: int = SMOOTH_MAX_ITER) -> Callable[[np.ndarray], np.ndarray]:
    """A function applying ``k`` steps of a lazy random walk on the mesh graph (surface Laplacian).

    One step is ``x <- (1 - lam) x + lam * (mean of the neighbours)``, i.e.
    ``x <- x + lam D^-1 L x`` with the graph Laplacian L. The neighbours are the
    vertices sharing an edge, plus vertices at the same position (meshes that
    repeat vertices along hard edges, or triangle soups, stay connected). After
    k steps the walk has spread by ``sigma^2 = k lam <l^2> / 2`` per surface
    direction (``<l^2>``: mean squared edge length), so k is chosen from
    ``sigma`` (at most ``max_iter``). The operator is row-stochastic: the ratio
    of two smoothed extensive quantities (derivative / area) is a kernel-weighted
    density, the 3D form of the 2D ``density_on_outline``.
    """
    from scipy import sparse

    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    l2 = float(np.mean(np.sum((V[e[:, 1]] - V[e[:, 0]]) ** 2, axis=1))) if len(e) else 0.0
    # vertices at the same position (to ~1e-9 of the mesh size): chain each group
    scale = max(float(np.ptp(V, axis=0).max()) if n else 1.0, 1e-300)
    q = np.round(V / (1e-9 * scale)).astype(np.int64)
    _, grp = np.unique(q, axis=0, return_inverse=True)
    grp = grp.reshape(-1)
    order = np.argsort(grp, kind="stable")
    same = grp[order[1:]] == grp[order[:-1]]
    dup = np.stack([order[:-1][same], order[1:][same]], 1)
    pairs = np.concatenate([e, dup]) if len(dup) else e
    if len(pairs) == 0 or l2 <= 0 or not sigma > 0:
        return lambda x: np.asarray(x, dtype=np.float64).copy()
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    W = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    W.sum_duplicates()
    W.data[:] = 1.0
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0)
    walk = sparse.diags(inv) @ W
    keep = np.where(deg > 0, 1.0 - lam, 1.0)
    k = int(min(max_iter, max(1, math.ceil(2.0 * sigma * sigma / (lam * l2)))))

    def apply(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        for _ in range(k):
            y = keep * y + lam * (walk @ y)
        return y

    apply.iterations = k            # type: ignore[attr-defined]
    return apply


# ---------------------------------------------------------------------------
# refined surface of one body
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SurfaceParam:
    """The displacement parametrisation of one body (non-dimensional solver units).

    ``vertices`` / ``faces``: the refined mesh (the body's own vertices first,
    ``n_original`` of them); ``normals``: unit outward vertex normals;
    ``area``: vertex dual areas; ``visible_area``: the part of it the solver
    sees (faces whose centroid lies in the domain -- with a symmetry plane, the
    y >= y0 half); ``mirror``: for vertices on the missing half of a
    symmetry-plane run, the index of the visible vertex nearest to the mirror
    image, else -1; ``partner``: for every vertex of a symmetry-plane run, the
    vertex at its mirror image (within a quarter cell; itself on the plane),
    else -1.
    """
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    area: np.ndarray
    visible_area: np.ndarray
    mirror: np.ndarray
    partner: np.ndarray
    n_original: int
    flipped_faces: int = 0


def _window_sampler(geom: Geometry3D, ops: Ops3D):
    """Trilinear sampler of the window's cell-centre solid fraction at points (P, 3) (numpy);
    points on the missing half of a symmetry-plane run are mirrored into the domain."""
    k0, _, j0, _, i0, _ = geom.window
    win = SimpleNamespace(x0=ops.x0 + i0 * ops.h, y0=ops.y0 + j0 * ops.h, z0=ops.z0 + k0 * ops.h, h=ops.h)
    chi = geom.chi_c.detach()

    def sample(p: np.ndarray) -> np.ndarray:
        q = np.array(p, dtype=np.float64, copy=True)
        if ops.symmetry:
            low = q[:, 1] < ops.y0
            q[low, 1] = 2.0 * ops.y0 - q[low, 1]
        t = torch.as_tensor(q, device=chi.device, dtype=chi.dtype)
        with torch.no_grad():
            return trilinear(chi, t, win).double().cpu().numpy()
    return sample


def _inside_domain(p: np.ndarray, ops: Ops3D) -> np.ndarray:
    """True for points inside the solver domain (periodic axes never clip)."""
    ok = np.ones(len(p), dtype=bool)
    lo = (ops.x0, ops.y0, ops.z0)
    n = (ops.nx, ops.ny, ops.nz)
    for a in range(3):
        if ops.periodic_axes[a]:
            continue
        ok &= (p[:, a] >= lo[a] - 1e-12) & (p[:, a] <= lo[a] + n[a] * ops.h + 1e-12)
    return ok


def _domain_planes(ops: Ops3D) -> list[tuple[int, float, float]]:
    """The clipping half-spaces of the domain as (axis, bound, side): inside where
    ``side * (x[axis] - bound) >= 0``. Periodic axes do not clip."""
    planes = []
    for a, (o, n) in enumerate(((ops.x0, ops.nx), (ops.y0, ops.ny), (ops.z0, ops.nz))):
        if not ops.periodic_axes[a]:
            planes += [(a, o, 1.0), (a, o + n * ops.h, -1.0)]
    return planes


def _clipped_hat_weights(tri: np.ndarray, planes) -> np.ndarray:
    """``(integral of each corner's hat function over the part of triangle ``tri`` (3, 3)
    inside all ``planes``) / area(tri)``, shape (3,): exact.

    The triangle is clipped in barycentric space (Sutherland-Hodgman against each
    half-space, which is linear in the barycentric coordinates), the remaining
    convex polygon is fanned into triangles, and a hat function -- a barycentric
    coordinate -- integrates over each to its area times its mean at the corners.
    """
    poly = [np.eye(3)[k] for k in range(3)]
    for a, b, s in planes:
        f = [s * (float(lam @ tri[:, a]) - b) for lam in poly]
        if min(f) >= 0.0:
            continue
        out = []
        for k in range(len(poly)):
            p, q, fp, fq = poly[k], poly[(k + 1) % len(poly)], f[k], f[(k + 1) % len(poly)]
            if fp >= 0.0:
                out.append(p)
            if (fp >= 0.0) != (fq >= 0.0):
                out.append(p + (q - p) * (fp / (fp - fq)))
        poly = out
        if len(poly) < 3:
            return np.zeros(3)
    w = np.zeros(3)
    for k in range(1, len(poly) - 1):
        p0, p1, p2 = poly[0], poly[k], poly[k + 1]
        twice_area = abs((p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1]))
        w += twice_area * (p0 + p1 + p2) / 3.0                     # reference triangle area is 1/2
    return w


def visible_dual_area(V: np.ndarray, F: np.ndarray, ops: Ops3D) -> np.ndarray:
    """``A_i = integral over the visible surface of the hat function of vertex i`` (N,).

    "Visible" is the part of the surface inside the solver domain (with a
    symmetry plane: y >= y0). A face entirely inside gives each corner a third
    of its area, one entirely outside nothing, and a face that crosses the
    domain boundary is clipped exactly. This is the weight the derivative
    actually carries: the solver's cells feel only the visible part of a
    crossing face, but its gradient reaches all three corners through their hat
    functions, so the derivative / area ratio stays consistent up to the plane.
    """
    a = V[F]                                                       # (M, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1)
    inside = _inside_domain(a.reshape(-1, 3), ops).reshape(-1, 3)
    full = inside.all(1)
    w = np.where(full[:, None], 1.0 / 3.0, 0.0) * np.ones((1, 3))
    planes = _domain_planes(ops)
    gone = np.zeros(len(F), dtype=bool)                            # all corners beyond one plane
    for ax, b, sd in planes:
        gone |= np.all(sd * (a[:, :, ax] - b) < 0.0, axis=1)
    for m in np.flatnonzero(~full & ~gone):                        # faces crossing the boundary
        w[m] = _clipped_hat_weights(a[m], planes)
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], area * w[:, k])
    return out


def build_surface_param(body: ImmersedMesh, ops: Ops3D, *, refine_cells: float = REFINE_CELLS,
                        max_faces: int = MAX_REFINED_FACES) -> SurfaceParam:
    """Refined mesh, dual areas and visibility of one body (normals still unoriented)."""
    V0 = body.vertices_numpy()
    F0 = np.asarray(body.faces, dtype=np.int64)
    spacing = refine_cells * ops.h
    for _ in range(12):
        try:
            V, F = refine_mesh(V0, F0, spacing, max_faces=max_faces)
            break
        except ValueError:
            spacing *= 1.4
    else:                                                      # pragma: no cover - absurd meshes only
        V, F = V0.copy(), F0.copy()
    nrm, dual = vertex_frame(V, F)
    vis = visible_dual_area(V, F, ops)
    mirror = np.full(len(V), -1, dtype=np.int64)
    partner = np.full(len(V), -1, dtype=np.int64)
    if ops.symmetry:
        from scipy.spatial import cKDTree
        q = V.copy()
        q[:, 1] = 2.0 * ops.y0 - q[:, 1]
        with ieee_subnormals():             # runs inside the CPU sweep's flush-to-zero block
            dist, idx = cKDTree(V).query(q)
            ok = dist < 0.25 * ops.h
            partner[ok] = idx[ok]
            low = np.flatnonzero(V[:, 1] < ops.y0 - 1e-9 * ops.h)
            high = np.flatnonzero(V[:, 1] >= ops.y0 - 1e-9 * ops.h)
            if low.size and high.size:
                _, idx = cKDTree(V[high]).query(q[low])
                mirror[low] = high[idx]
    return SurfaceParam(body.name, V, F, nrm, dual, vis, mirror, partner, len(V0))


def orient_surface(param: SurfaceParam, sample, ops: Ops3D) -> SurfaceParam:
    """Point every face normal towards the fluid, then recompute the vertex normals.

    The cell-centre solid fraction is compared one cell outside and one cell
    inside along each face normal; a face is flipped when the "outside" probe is
    more solid. Faces where the test is undecided (open sheets with fluid on
    both sides) or meaningless (outside the domain, e.g. a tyre patch below the
    road; on the missing half of a symmetry-plane run the mirror image is
    probed) keep the mesh's winding. Per-face rather than per-vertex because an
    inconsistently wound mesh averages opposite face normals into meaningless
    vertex normals.
    """
    V, F = param.vertices, param.faces
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn = cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-300)[:, None]
    cen = V[F].mean(axis=1)
    d = ORIENT_PROBE_CELLS * ops.h
    s_out, s_in = sample(cen + d * fn), sample(cen - d * fn)
    seen = cen.copy()
    if ops.symmetry:
        seen[:, 1] = np.where(seen[:, 1] < ops.y0, 2.0 * ops.y0 - seen[:, 1], seen[:, 1])
    flip = (s_out > s_in + 1e-3) & _inside_domain(seen, ops)
    fn = np.where(flip[:, None], -fn, fn)
    param.normals, _ = vertex_frame(V, F, fn)
    param.flipped_faces = int(flip.sum())
    return param


# ---------------------------------------------------------------------------
# one differentiable 3D step, optionally CUDA-graphed
# ---------------------------------------------------------------------------

class _Stepper3D:
    """One 3D solver step as a pure function of (u, v, w, mask tensors), with its VJP.

    As in 2D, ``chi_c`` (cell-centre solid fraction: only switches the LES eddy
    viscosity off inside solids) is passed without gradient; everything that
    places the wall (face masks, solid velocities, force / moment weights) is
    differentiated. The reference functional core (``step3d_full``) is used: the
    fused CUDA kernels of the production solver have no autograd. Captures and
    replays run under IEEE float32 matmuls (the projection's transforms must
    never run in TF32, forward or backward).
    """

    def __init__(self, ops: Ops3D, window, nbody: int, dt: float, *, use_graphs: bool | None = None):
        self.ops, self.window, self.nbody = ops, tuple(window), int(nbody)
        self.dt = torch.tensor(float(dt), device=ops.device, dtype=ops.dtype)
        self._zero = torch.zeros((), device=ops.device, dtype=ops.dtype)
        self.fn: Callable = self._step
        self.graphed = False
        self.note: str | None = None
        self._want_graphs = (ops.device.type == "cuda") if use_graphs is None else bool(use_graphs)
        self._tried = False

    def _step(self, u, v, w, chi_u, chi_v, chi_w, a_u, a_v, a_w, k_u, k_v, k_w, chi_c, dt):
        g = Geometry3D(self.window, chi_u, chi_v, chi_w, chi_c, a_u, a_v, a_w, k_u, k_v, k_w, self.nbody)
        z = self._zero
        new, c = step3d_full(State3D(u, v, w, z, z, z), g, self.ops, dt)
        return new.u, new.v, new.w, c

    def prepare(self, u, v, w, geom_t: Sequence[torch.Tensor], chi_c: torch.Tensor) -> None:
        """Capture the CUDA graph (once) from sample tensors shaped like these."""
        if self._tried or not self._want_graphs:
            return
        self._tried = True
        try:
            with ieee_matmul(), torch.enable_grad():
                sample = ([x.detach().clone().requires_grad_(True) for x in (u, v, w)]
                          + [t.detach().clone().requires_grad_(True) for t in geom_t]
                          + [chi_c.detach().clone(), self.dt.clone()])
                self.fn = torch.cuda.make_graphed_callables(self._step, tuple(sample), num_warmup_iters=3,
                                                            allow_unused_input=True)
            self.graphed = True
        except Exception as exc:  # noqa: BLE001 - any capture failure: the eager path computes the same thing
            self.fn = self._step
            self.graphed = False
            self.note = (f"CUDA graphs were not available for the reverse sweep ({type(exc).__name__}); "
                         "it ran in eager mode, which is slower but gives the same result.")
            if self.ops.device.type == "cuda":
                torch.cuda.synchronize(self.ops.device)

    def forward(self, u, v, w, geom_t, chi_c):
        """(u, v, w, per-body coefficients) after one step, without autograd. With CUDA
        graphs the outputs are static buffers overwritten by the next call: clone to keep."""
        with torch.no_grad(), ieee_matmul():
            return self.fn(u, v, w, *geom_t, chi_c, self.dt)

    def vjp(self, u, v, w, geom_t, chi_c, lam_u, lam_v, lam_w, wc):
        """Pull (lam_u, lam_v, lam_w, wc) on the step outputs back to (u, v, w, mask tensors).

        Returns gradients (None where an input does not influence the outputs; with
        a symmetry plane the side-force weights ``k_v`` never do). With CUDA graphs
        these alias static buffers: consume them before the next call.
        """
        u, v, w = (x.detach().requires_grad_(True) for x in (u, v, w))
        with torch.enable_grad(), ieee_matmul():
            un, vn, wn, c = self.fn(u, v, w, *geom_t, chi_c, self.dt)
            return torch.autograd.grad((un, vn, wn, c), (u, v, w, *geom_t), (lam_u, lam_v, lam_w, wc),
                                       allow_unused=True)


# ---------------------------------------------------------------------------
# the windowed objective
# ---------------------------------------------------------------------------

def _objective_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "ld":
        return lambda m: m[_COEF["cl"]] / m[_COEF["cd"]]
    if name in OBJECTIVES3D:
        i = _COEF[name]
        return lambda m: m[i]
    raise ValueError(f"unknown objective {name!r}; use one of {', '.join(OBJECTIVES3D)}")


def _velocity(state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, FlowSolver3D | None]:
    """(u, v, w, solver or None) from a FlowSolver3D, a State3D, a (u, v, w) tuple or a state_dict."""
    if isinstance(state, FlowSolver3D):
        s = state.state
        return s.u, s.v, s.w, state
    if isinstance(state, State3D):
        return state.u, state.v, state.w, None
    if isinstance(state, dict) and all(k in state for k in "uvw"):
        return torch.as_tensor(state["u"]), torch.as_tensor(state["v"]), torch.as_tensor(state["w"]), None
    if isinstance(state, (tuple, list)) and len(state) == 3:
        return torch.as_tensor(state[0]), torch.as_tensor(state[1]), torch.as_tensor(state[2]), None
    raise TypeError("the starting state must be a FlowSolver3D, a State3D, a state_dict or (u, v, w)")


class WindowedObjective3D:
    """J(d): the time-averaged force objective after a shape change, as a function of
    per-vertex normal displacements ``d`` (non-dimensional, one array per body over
    its refined vertices).

    Built from a solver config (non-dimensional, wind frame) and a developed
    starting state, which is held fixed. :meth:`evaluate` runs the window
    forward; :meth:`gradient` returns J and dJ/dd for every refined vertex by one
    reverse sweep. Both evaluate the same discrete function (same fixed mask
    window, time step and step count), so central differences of
    :meth:`evaluate` check :meth:`gradient` exactly
    (:func:`finite_difference_check3d`).
    """

    def __init__(self, cfg: SolverConfig3D, state0, *, objective: str = "cd", lead: float = 1.0,
                 average: float = 1.0, dt: float | None = None, refine_cells: float = REFINE_CELLS,
                 margin_cells: float = WINDOW_MARGIN_CELLS, use_graphs: bool | None = None,
                 max_steps: int = MAX_STEPS):
        if not cfg.bodies:
            raise ValueError("there is no body to compute a sensitivity for")
        if not (lead >= 0 and average > 0):
            raise ValueError("the window needs lead >= 0 and average > 0 convective times")
        self.objective = objective
        self._fobj = _objective_fn(objective)
        self.cfg = dataclasses.replace(cfg, bodies=list(cfg.bodies), use_cuda_graphs=False, fused_kernels=False)
        self.ops = make_operators3d(self.cfg)
        ops = self.ops
        dev, dtype = ops.device, ops.dtype
        u0, v0, w0, _ = _velocity(state0)
        self.u0, self.v0, self.w0 = (x.detach().to(device=dev, dtype=dtype).clone() for x in (u0, v0, w0))
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        if (tuple(self.u0.shape) != (nz, ny, nx + 1) or tuple(self.v0.shape) != (nz, ny + 1, nx)
                or tuple(self.w0.shape) != (nz + 1, ny, nx)):
            raise ValueError("the starting state does not match the solver grid")
        # -- refined surfaces, fixed window, oriented normals
        self.surfaces = [build_surface_param(b, ops, refine_cells=refine_cells) for b in self.cfg.bodies]
        margin = ops.mask_width + 3.0 + abs(ops.mask_shift) + margin_cells
        self.window = body_window3d([s.vertices for s in self.surfaces], ops, margin_cells=margin)
        if self.window[1] <= self.window[0]:
            raise ValueError("the bodies lie outside the solver domain")
        self._v0 = [torch.as_tensor(s.vertices, device=dev, dtype=dtype) for s in self.surfaces]
        self._faces = [s.faces for s in self.surfaces]
        with torch.no_grad():
            base = self.geometry()
        sample = _window_sampler(base, ops)
        for s in self.surfaces:
            orient_surface(s, sample, ops)
        self._n0 = [torch.as_tensor(s.normals, device=dev, dtype=dtype) for s in self.surfaces]
        # -- time step and window
        if dt is None:
            if cfg.dt is not None:
                dt = float(cfg.dt)
            else:
                nut = None
                if ops.turbulence != "none":
                    with torch.no_grad():
                        nut = momentum_rhs3d(self.u0, self.v0, self.w0, base, ops)[3]
                dt = DT_SAFETY * float(cfl_dt3d(self.u0, self.v0, self.w0, ops, nut))
        self.dt = float(dt)
        if not (self.dt > 0 and math.isfinite(self.dt)):
            raise ValueError("could not determine a time step from the starting state")
        self.n_lead = math.ceil(lead / self.dt - 1e-9)
        self.n_avg = max(1, math.ceil(average / self.dt - 1e-9))
        if self.n_lead + self.n_avg > max_steps:
            raise ValueError(f"the window needs {self.n_lead + self.n_avg:,} steps (limit {max_steps:,}); "
                             "shorten it")
        self.lead, self.average = self.n_lead * self.dt, self.n_avg * self.dt
        self._stepper = _Stepper3D(ops, self.window, len(self.surfaces), self.dt, use_graphs=use_graphs)
        self.offload: bool | None = None          # set by gradient(): checkpoints on the host

    # -------------------------------------------------------------- geometry
    @property
    def steps(self) -> int:
        return self.n_lead + self.n_avg

    @property
    def cuda_graphs(self) -> bool:
        return self._stepper.graphed

    @property
    def notes(self) -> list[str]:
        return [self._stepper.note] if self._stepper.note else []

    @property
    def window_cells(self) -> int:
        k0, k1, j0, j1, i0, i1 = self.window
        return (k1 - k0) * (j1 - j0) * (i1 - i0)

    def _as_disp(self, disp) -> list[torch.Tensor]:
        dev, dtype = self.ops.device, self.ops.dtype
        if disp is None:
            return [torch.zeros(len(s.vertices), device=dev, dtype=dtype) for s in self.surfaces]
        if len(disp) != len(self.surfaces):
            raise ValueError("one displacement array per body is needed")
        out = []
        for d, s in zip(disp, self.surfaces):
            t = d if isinstance(d, torch.Tensor) else torch.as_tensor(np.asarray(d, dtype=np.float64))
            t = t.to(device=dev, dtype=dtype)
            if t.shape != (len(s.vertices),):
                raise ValueError(f"displacement for a {len(s.vertices)}-vertex mesh has shape {tuple(t.shape)}")
            out.append(t)
        return out

    def vertices(self, disp=None) -> list[torch.Tensor]:
        """Displaced refined vertices ``x_i + d_i n_i`` (non-dimensional tensors; keeps autograd).
        Before the normals are oriented (during construction) the displacement must be zero."""
        if disp is None or not hasattr(self, "_n0"):
            return list(self._v0)
        return [p + d[:, None] * n for p, d, n in zip(self._v0, self._as_disp(disp), self._n0)]

    def geometry(self, disp=None) -> Geometry3D:
        """Masks of the displaced bodies on the fixed window (differentiable in ``disp``)."""
        bodies = []
        for b, v, f in zip(self.cfg.bodies, self.vertices(disp), self._faces):
            bodies.append(ImmersedMesh(v, f, name=b.name, role=b.role, omega=b.omega, center=b.center,
                                       axis=b.axis))
        return build_geometry3d(bodies, self.ops, window=self.window)

    # ------------------------------------------------------------- forward
    def _forward(self, geom_t, chi_c, *, keep_every: int = 0, keep_on=None, progress=None, cancel=None,
                 work=(0.0, 1.0)):
        """Run the window; returns (sum of per-body coefficients over the averaging
        steps, stored (u, v, w) states every ``keep_every`` steps on ``keep_on``)."""
        st = self._stepper
        st.prepare(self.u0, self.v0, self.w0, geom_t, chi_c)
        u, v, w = self.u0, self.v0, self.w0
        acc = torch.zeros(len(self.surfaces), 6, device=self.ops.device, dtype=self.ops.dtype)
        kept = []
        n = self.steps
        for i in range(n):
            if keep_every and i % keep_every == 0:
                if keep_on is None:
                    kept.append((u.clone(), v.clone(), w.clone()))
                else:
                    kept.append(tuple(x.to(keep_on, copy=True) for x in (u, v, w)))
            u, v, w, c = st.forward(u, v, w, geom_t, chi_c)
            if i >= self.n_lead:
                acc += c
            if i % CHECK_EVERY == 0:
                check_cancel(cancel)
                if progress is not None:
                    progress(work[0] + (work[1] - work[0]) * i / n, "Running the sensitivity window")
        return acc, kept

    def _window_value(self, acc: torch.Tensor) -> tuple[torch.Tensor, WindowValue]:
        per_body = acc / self.n_avg
        mean = per_body.sum(0).detach().requires_grad_(True)
        with torch.enable_grad():
            j = self._fobj(mean)
            (dj,) = torch.autograd.grad(j, mean)
        val = WindowValue(float(j.detach()), mean.detach().cpu().double().numpy(),
                          per_body.detach().cpu().double().numpy())
        return dj, val

    def evaluate(self, disp=None, *, cancel: threading.Event | None = None) -> WindowValue:
        """J for displaced vertices (forward only). ``coefficients`` holds the six window means."""
        with torch.no_grad():
            g = self.geometry(disp)
            geom_t = [getattr(g, k) for k in _GEOM3]
            acc, _ = self._forward(geom_t, g.chi_c, cancel=cancel)
        return self._window_value(acc)[1]

    def advance(self, duration: float, disp=None, *, cancel: threading.Event | None = None,
                progress: ProgressFn | None = None) -> WindowValue | None:
        """Move the starting state forward by ``duration`` (L/U) with the (displaced) shape,
        at the window's fixed step (used to start ensemble windows at other phases of a
        shedding cycle). Returns the mean coefficients over the span (None if empty)."""
        n = round(duration / self.dt)
        if n <= 0:
            return None
        with torch.no_grad():
            g = self.geometry(disp)
            geom_t = [getattr(g, k) for k in _GEOM3]
            st = self._stepper
            st.prepare(self.u0, self.v0, self.w0, geom_t, g.chi_c)
            u, v, w = self.u0, self.v0, self.w0
            acc = torch.zeros(len(self.surfaces), 6, device=self.ops.device, dtype=self.ops.dtype)
            for i in range(n):
                u, v, w, c = st.forward(u, v, w, geom_t, g.chi_c)
                acc += c
                if i % CHECK_EVERY == 0:
                    check_cancel(cancel)
                    if progress is not None:
                        progress(i / n, "Letting the flow settle")
            self.u0, self.v0, self.w0 = u.clone(), v.clone(), w.clone()
            if not all(bool(torch.isfinite(x).all()) for x in (self.u0, self.v0, self.w0)):
                raise RuntimeError("The flow became unstable while advancing the sensitivity start state.")
        per_body = (acc / n).cpu().double().numpy()
        return WindowValue(float("nan"), per_body.sum(0), per_body)

    # ------------------------------------------------------------- reverse
    def memory_estimate(self, *, offload: bool = False) -> int:
        """Device bytes this window's reverse sweep needs (see :func:`estimate_memory_bytes`)."""
        return estimate_memory_bytes(self.cfg.cells, steps=self.steps, window_cells=self.window_cells,
                                     dtype=self.cfg.dtype, offload=offload)

    def gradient(self, disp=None, *, progress: ProgressFn | None = None,
                 cancel: threading.Event | None = None) -> tuple[WindowValue, list[np.ndarray]]:
        """J and dJ/dd_i for every refined vertex of every body (non-dimensional).

        Memory: one differentiated step, about 2 sqrt(N) velocity states (the
        checkpoints move to host memory when the device is short) and the
        adjoint. Raises :class:`~aerolume.optimize.Cancelled` when ``cancel`` is set.
        """
        rep = Progress(progress)
        dvec = [d.detach().clone().requires_grad_(True) for d in self._as_disp(disp)]
        with torch.enable_grad():
            g_full = self.geometry(dvec)
        full = [getattr(g_full, k) for k in _GEOM3]
        geom_t = [t.detach().clone().requires_grad_(True) for t in full]
        chi_c = g_full.chi_c.detach()
        n = self.steps
        seg = max(1, math.ceil(math.sqrt(n)))
        f_fwd = 1.0 / 6.0
        keep_on = self._offload_target(n, seg)
        self.offload = keep_on is not None
        acc, kept = self._forward(geom_t, chi_c, keep_every=seg, keep_on=keep_on, progress=rep, cancel=cancel,
                                  work=(0.0, f_fwd))
        dj, val = self._window_value(acc)
        nb = len(self.surfaces)
        w_avg = (dj / self.n_avg).expand(nb, 6).contiguous()
        w_zero = torch.zeros_like(w_avg)
        lam = [torch.zeros_like(x) for x in (self.u0, self.v0, self.w0)]
        grads = [torch.zeros_like(t) for t in geom_t]
        st = self._stepper
        done = 0
        for b in reversed(range(len(kept))):
            start, stop = b * seg, min(n, (b + 1) * seg)
            u, v, w = kept[b]
            if keep_on is not None:
                u, v, w = (x.to(self.ops.device) for x in (u, v, w))
            states = [(u, v, w)]
            for _ in range(start, stop - 1):
                u, v, w, _c = st.forward(u, v, w, geom_t, chi_c)
                states.append((u.clone(), v.clone(), w.clone()))
            kept[b] = None
            for i in reversed(range(start, stop)):
                ui, vi, wi = states[i - start]
                out = st.vjp(ui, vi, wi, geom_t, chi_c, *lam, w_avg if i >= self.n_lead else w_zero)
                for a, gk in zip(lam, out[:3]):
                    a.copy_(gk)
                for a, gk in zip(grads, out[3:]):
                    if gk is not None:
                        a.add_(gk)
                states[i - start] = None
                done += 1
                if done % CHECK_EVERY == 0:
                    check_cancel(cancel)
                    rep(f_fwd + (1.0 - f_fwd) * done / n, "Differentiating the flow (reverse sweep)")
            del states
        pairs = [(t, gk) for t, gk in zip(full, grads) if t.requires_grad]
        if not all(bool(torch.isfinite(gk).all()) for _, gk in pairs):
            raise RuntimeError("The sensitivity is not finite: the differentiated flow model has a non-smooth "
                               "point here (sub-grid models where the strain vanishes). Use laminar or WALE.")
        if pairs:
            torch.autograd.backward([t for t, _ in pairs], [gk for _, gk in pairs])
        out = []
        for d in dvec:
            gd = d.grad if d.grad is not None else torch.zeros_like(d)
            out.append(gd.detach().cpu().double().numpy())
        if not all(np.all(np.isfinite(o)) for o in out):
            raise RuntimeError("The sensitivity is not finite (the geometry derivative overflowed).")
        rep(1.0, "Sensitivity computed")
        return val, out

    def _offload_target(self, n: int, seg: int):
        """'cpu' when the checkpoints would take more than OFFLOAD_FRACTION of the free GPU
        memory, else None (keep them on the device). Only the ~n/seg checkpoints can wait
        on the host; the recomputed segment and the adjoint stay on the device."""
        if self.ops.device.type != "cuda":
            return None
        per_state = sum(x.numel() for x in (self.u0, self.v0, self.w0)) * self.u0.element_size()
        need = per_state * math.ceil(n / seg)
        try:
            free, _ = torch.cuda.mem_get_info(self.ops.device)
            free += torch.cuda.memory_reserved(self.ops.device) - torch.cuda.memory_allocated(self.ops.device)
        except Exception:  # noqa: BLE001 - no reliable reading: keep everything on the device
            return None
        return "cpu" if need > OFFLOAD_FRACTION * free else None

    # ------------------------------------------------------------- the map
    def density(self, vertex_grad: Sequence[np.ndarray], *, smooth_cells: float = SMOOTH_CELLS
                ) -> list[np.ndarray]:
        """Smoothed density g (non-dimensional, ``dJ = integral g dn dA``) at every refined
        vertex of every body, from per-vertex derivatives.

        ``g = S^k G / S^k A_visible`` with the surface random walk S of
        :func:`surface_smoother` (``smooth_cells`` cells; 0 = no smoothing). Vertices the
        solver cannot see get 0, except on the missing half of a symmetry-plane run, where
        they take their mirror image's value. With a symmetry plane the half-domain
        derivative belongs to a mirror-symmetric pair of bumps and is halved, and each
        vertex first receives its mirror partner's derivative and visible area (the
        solver sees one of the two): the smoothing kernel then straddles the plane exactly
        as in a full-domain run, instead of being cut off by it.
        """
        out = []
        half = 0.5 if self.ops.symmetry else 1.0
        for s, g0 in zip(self.surfaces, vertex_grad):
            sm = surface_smoother(s.vertices, s.faces, smooth_cells * self.ops.h)
            gv = np.array(g0, dtype=np.float64)
            area = s.visible_area.copy()
            if self.ops.symmetry:
                has = s.partner >= 0
                gv[has] += np.asarray(g0, dtype=np.float64)[s.partner[has]]
                area[has] += s.visible_area[s.partner[has]]
            num, den = sm(gv), sm(area)
            tiny = 1e-9 * max(float(np.max(s.visible_area, initial=0.0)), 1e-300)
            g = np.where(den > tiny, num / np.maximum(den, tiny), 0.0) * half
            m = s.mirror >= 0
            g[m] = g[s.mirror[m]]
            out.append(g)
        return out


# ---------------------------------------------------------------------------
# finite-difference verification
# ---------------------------------------------------------------------------

def normal_modes3d(problem: WindowedObjective3D, centres: Sequence[Sequence[float]], width: float, *,
                   body: int = 0, mirror_y: float | None = None) -> np.ndarray:
    """Smooth displacement modes over one body's refined vertices: (n_modes, n_vertices).

    Mode k is a Gaussian bump ``exp(-|x - c_k|^2 / (2 width^2))`` (peak 1,
    non-dimensional) centred at ``centres[k]`` (solver units). ``mirror_y``
    adds the mirror bump about the plane y = mirror_y (a mirror-symmetric mode,
    what a half-domain run with its symmetry plane there actually perturbs).
    """
    x = problem.surfaces[body].vertices
    out = []
    for c in centres:
        c = np.asarray(c, dtype=np.float64)
        phi = np.exp(-np.sum((x - c) ** 2, axis=1) / (2.0 * width * width))
        if mirror_y is not None:
            cm = c.copy()
            cm[1] = 2.0 * float(mirror_y) - cm[1]
            if np.linalg.norm(cm - c) > 1e-12:
                phi = phi + np.exp(-np.sum((x - cm) ** 2, axis=1) / (2.0 * width * width))
        out.append(phi)
    return np.asarray(out)


def finite_difference_check3d(problem: WindowedObjective3D, modes: np.ndarray, *, eps: float = 1e-4,
                              body: int = 0, vertex_grad: Sequence[np.ndarray] | None = None,
                              progress: ProgressFn | None = None,
                              cancel: threading.Event | None = None) -> list[dict]:
    """Compare autograd with central differences of the same J along displacement modes.

    For each mode phi (over the refined vertices of ``body``) returns ``{"mode",
    "ad", "fd", "rel_err", "j_plus", "j_minus"}`` with ``ad = sum_i dJ/dd_i
    phi_i`` and ``fd = (J(+eps phi) - J(-eps phi)) / (2 eps)``; ``eps`` is a
    non-dimensional amplitude (1e-4 L is ~1/1000 of a cell at 10 cells per L).
    """
    rep = Progress(progress)
    if vertex_grad is None:
        _, vertex_grad = problem.gradient(progress=rep.sub(0.0, 0.3), cancel=cancel)
    out = []
    modes = np.atleast_2d(np.asarray(modes, dtype=np.float64))
    for k, phi in enumerate(modes):
        disp = [np.zeros(len(s.vertices)) for s in problem.surfaces]
        disp[body] = eps * phi
        jp = problem.evaluate(disp, cancel=cancel).value
        disp[body] = -eps * phi
        jm = problem.evaluate(disp, cancel=cancel).value
        fd = (jp - jm) / (2.0 * eps)
        ad = float(np.dot(vertex_grad[body], phi))
        out.append({"mode": k, "ad": ad, "fd": fd, "rel_err": abs(ad - fd) / max(abs(fd), 1e-300),
                    "j_plus": jp, "j_minus": jm})
        rep(0.3 + 0.7 * (k + 1) / len(modes), f"Finite-difference check, mode {k + 1} of {len(modes)}")
    return out


# ---------------------------------------------------------------------------
# budget and cost
# ---------------------------------------------------------------------------

def estimate_memory_bytes(cells: int, *, steps: int, window_cells: int = 0, dtype: str = "float32",
                          offload: bool = False) -> int:
    """Device memory a 3D reverse sweep needs beyond what the flow solver already holds.

    One differentiated step (:data:`STEP_BYTES_PER_CELL`, measured with CUDA
    graphs), the velocity states that stay on the device -- the sqrt(N)-step
    recomputed segment, the starting state, the adjoint and the step's in/out
    buffers -- plus the sqrt(N) checkpoints unless they are offloaded to the
    host, the mask window and a fixed allowance. float64 doubles everything.
    """
    scale = 2 if dtype == "float64" else 1
    state = 3 * 4 * scale                                       # bytes per cell of one (u, v, w) state
    seg = max(1, math.ceil(math.sqrt(max(int(steps), 1))))
    held = seg + 6 + (0 if offload else math.ceil(max(int(steps), 1) / seg))
    return int(cells * (STEP_BYTES_PER_CELL * scale + held * state) + window_cells * WINDOW_BYTES_PER_CELL * scale
               + FIXED_BYTES)


def _available_bytes(device: torch.device) -> int | None:
    """Free device memory including the caching allocator's unused reserve (CUDA), else None."""
    if device.type != "cuda":
        return None
    try:
        free, _ = torch.cuda.mem_get_info(device)
        return int(free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    except Exception:  # noqa: BLE001 - no reading: the caller skips the memory test
        return None


def _check_memory(need: int, device: torch.device, cells: int, steps: int | None = None) -> None:
    """Raise :class:`SensitivityBudgetError` when ``need`` bytes exceed MEMORY_FRACTION of the
    free device memory (CUDA only; no reading, no refusal)."""
    avail = _available_bytes(device)
    if avail is None or need <= MEMORY_FRACTION * avail:
        return
    what = f"{cells / 1e6:.2g} M cells" + (f" and {steps:,} steps" if steps else "")
    raise SensitivityBudgetError(
        f"the 3D sensitivity map needs about {need / 2**30:.1f} GB of GPU memory for {what}, but only "
        f"{avail / 2**30:.1f} GB is free; it was skipped (the forces are unaffected). Run the map at Draft "
        "quality, or with fewer other GPU programs open.")


def check_budget(cells: int, dclass: str, max_cells: int | None) -> None:
    """Raise :class:`SensitivityBudgetError` when the grid exceeds the cell budget."""
    limit = DEFAULT_MAX_CELLS[dclass] if max_cells is None else int(max_cells)
    if cells > limit:
        where = "on the GPU" if dclass == "gpu" else "on a CPU"
        raise SensitivityBudgetError(
            f"the 3D sensitivity map is limited to {limit / 1e6:.2g} M cells {where} and this grid has "
            f"{cells / 1e6:.2g} M; it was skipped (the forces are unaffected). Run the map at Draft quality, "
            "where it costs about as much as the flow solve.")


def estimate_seconds(plan) -> float:
    """Rough wall time [s] of :func:`compute_sensitivity3d` reusing the run's solver.

    A steady window of 6 + 2 body time units on a GPU (3 + 1 on a CPU) at 0.9 of
    the plan's time step, times the measured cost of one sweep step (forward,
    recomputation and VJP: :data:`SWEEP_NS_PER_CELL`), plus graph capture and the
    geometry derivative. Shedding flows cost about 2-4x (whole periods, three
    phases). For the Setup page's run-time estimate.
    """
    from .sensitivity import STEADY_WINDOW
    cfg = plan.solver_config
    gpu = not str(getattr(plan, "device", "cuda")).startswith("cpu")
    L = float(plan.length_ref)
    unit = max(float(getattr(plan, "strouhal_length", L) or L) / L, 1e-6)
    lead, avg = STEADY_WINDOW["gpu" if gpu else "cpu"]
    steps = (lead + avg) * unit / max(DT_SAFETY * float(plan.dt), 1e-12)
    rate = SWEEP_NS_PER_CELL["gpu" if gpu else "cpu"] * (2.0 if cfg.dtype == "float64" else 1.0)
    return float(steps * cfg.cells * rate * 1e-9 + (10.0 if gpu else 5.0))


# ---------------------------------------------------------------------------
# public driver
# ---------------------------------------------------------------------------

class SensitivityMaps3D(list):
    """``list[np.ndarray]``: one 3D sensitivity map per body, dJ/dn [1/m] (``L_ref^2 g``, see
    the module docstring) at that body's own mesh vertices, in their order. Metadata:

    ``objective``, ``value`` (J over the window), ``coefficients`` (window means
    {"cd","cs","cl","c_roll","cm","cn"}), ``window`` ((lead, average) in L/U),
    ``dt``, ``steps``, ``wall_seconds``, ``peak_memory_bytes`` /
    ``extra_memory_bytes`` (CUDA: process peak during the sweep and its rise
    above the memory in use before it; None on CPU), ``estimated_memory_bytes``,
    ``offloaded`` (checkpoints held on the host), ``cells``, ``device``,
    ``cuda_graphs``, ``bodies`` (names), ``points`` / ``normals`` (wind frame,
    metres / unit outward) of the map's vertices, ``refined_vertices`` (count per
    body), ``vertex_gradient`` (dJ/dd per refined vertex, non-dimensional),
    ``symmetry`` and ``symmetry_y`` (plane position, metres), ``period``,
    ``ensemble``, ``smooth_cells``, ``length_ref``, ``notes`` and ``warnings``.
    """

    _FIELDS: ClassVar[dict] = {
        "objective": "cd", "value": float("nan"), "coefficients": {}, "window": (float("nan"), float("nan")),
        "dt": float("nan"), "steps": 0, "wall_seconds": float("nan"), "peak_memory_bytes": None,
        "extra_memory_bytes": None, "estimated_memory_bytes": None, "offloaded": False, "cells": 0,
        "device": "", "cuda_graphs": False, "bodies": [], "points": [], "normals": [], "refined_vertices": [],
        "vertex_gradient": [], "symmetry": False, "symmetry_y": 0.0, "period": None, "ensemble": 1,
        "smooth_cells": SMOOTH_CELLS, "length_ref": 1.0, "notes": [], "warnings": [],
    }

    def __init__(self, arrays: Sequence[np.ndarray] = (), **meta):
        super().__init__(np.asarray(a, dtype=np.float64) for a in arrays)
        unknown = set(meta) - set(self._FIELDS)
        if unknown:
            raise TypeError(f"unknown SensitivityMaps3D fields: {', '.join(sorted(unknown))}")
        for k, default in self._FIELDS.items():
            v = meta.get(k, default)
            setattr(self, k, list(v) if isinstance(default, list) else (dict(v) if isinstance(default, dict) else v))

    def summary(self) -> str:
        """One sentence for result notes and logs."""
        sym = OBJECTIVES3D.get(self.objective, self.objective)
        mem = (f", peak GPU memory {self.peak_memory_bytes / 2**20:.0f} MB" if self.peak_memory_bytes else "")
        return (f"3D shape sensitivity of {sym} by automatic differentiation through the solver: {sym} averaged "
                f"over t = {self.window[0]:.3g} ... {self.window[0] + self.window[1]:.3g} L/U after the shape "
                f"change ({self.steps:,} steps of {self.dt:.3g} L/U on {self.cells / 1e6:.2g} M cells), computed "
                f"in {self.wall_seconds:.0f} s{mem}.")


def develop_flow3d(cfg: SolverConfig3D, t_end: float, rep: Progress, cancel) -> FlowSolver3D:
    """Develop the flow from the impulsive start to ``t_end`` (fast path, CUDA graphs)."""
    solver = FlowSolver3D(cfg)
    rate = 50 if solver.device.type == "cpu" else 500
    while solver.t < t_end - 1e-12:
        check_cancel(cancel)
        n = max(1, min(rate, math.ceil((t_end - solver.t) / max(solver.dt, 1e-12))))
        if solver.device.type == "cuda" and n >= 20:
            n -= n % 10
        solver.advance(n)
        if not np.all(np.isfinite(solver.coefficients())):
            raise RuntimeError(f"The flow became unstable at t = {solver.t:.3g} while developing it.")
        rep(min(1.0, solver.t / t_end), f"Developing the flow for the sensitivity (t = {solver.t:.1f} of {t_end:g})")
    return solver


def _plan3d(design: Design, sim: Simulation, device: str | None):
    """The 3D plan (``simulation.plan.plan_simulation`` dispatches on the dimension)."""
    from ..simulation.plan import plan_simulation
    plan = plan_simulation(design, sim, device=device)
    if not isinstance(getattr(plan, "solver_config", None), SolverConfig3D):
        try:
            from ..simulation import plan3d  # older dispatch: call the 3D planner directly
        except ImportError as exc:
            raise ValueError("the 3D sensitivity map needs a 3D plan (simulation.plan3d is missing)") from exc
        fn = getattr(plan3d, "plan_simulation3d", None) or plan3d.plan_simulation
        plan = fn(design, sim, device=device)
    return plan


def _wind_points(plan, cfg: SolverConfig3D, L: float) -> list[np.ndarray]:
    """Map points in the wind frame (metres): the plan's own wind-frame meshes when they
    match the solver bodies vertex for vertex, else the solver vertices times L_ref."""
    wind = list(getattr(plan, "bodies_wind3", None) or [])
    out = []
    for i, b in enumerate(cfg.bodies):
        v = b.vertices_numpy()
        if i < len(wind) and len(getattr(wind[i], "vertices", ())) == len(v):
            out.append(np.asarray(wind[i].vertices, dtype=np.float64).copy())
        else:
            out.append(v * L)
    return out


def compute_sensitivity3d(design: Design, sim: Simulation, *, objective: str | None = None, plan=None,
                          solver=None, window: float | tuple[float, float] | None = None,
                          device: str | None = None, progress: ProgressFn | None = None,
                          cancel: threading.Event | None = None, max_cells: int | None = None,
                          ensemble: int | None = None, smooth_cells: float = SMOOTH_CELLS) -> SensitivityMaps3D:
    """Per-vertex surface sensitivity of ``objective`` for every mesh of a 3D CFD set-up, in 1/m.

    The values are the derivative of a windowed time average from a fixed
    developed flow, taken through the immersed boundary's masks: the pattern and
    signs are reliable, the size indicative (biased high, see
    :data:`MAGNITUDE_WARNING`). One array per solver body (= per design mesh, or
    the lofted wing), at that body's vertices in their order; with a symmetry
    plane the vertices of the missing half carry their mirror image's value.

    ``objective``: "cd", "cl", "ld" or "cm" (default ``sim.cfd.sensitivity_objective``).
    ``plan``: the 3D plan (planned here when None). ``solver``: the developed
    :class:`FlowSolver3D` of the run (only read), a ``State3D`` or a state_dict;
    None develops the flow to ``plan.t_avg_start`` first. ``window``: averaging
    time or ``(lead, average)`` in L/U; default as in 2D (steady: 6 + 2 body time
    units on a GPU, 3 + 1 on a CPU; shedding: whole periods from ``ensemble``
    phases). ``max_cells``: refuse larger grids (default
    :data:`DEFAULT_MAX_CELLS` per device class). ``smooth_cells``: kernel width
    of the map smoothing in cells. ``progress(fraction, message)`` and ``cancel``
    as usual.

    Raises :class:`SensitivityBudgetError` (before any costly work) when the grid
    or the estimated device memory exceeds the budget, and
    :class:`~aerolume.optimize.Cancelled` when cancelled.
    """
    from ..core.device import resolve
    from ..simulation.live import cpu_threads
    from ..solvers.fpmode import flush_denormal

    wall0 = time.perf_counter()
    obj = (objective or getattr(getattr(sim, "cfd", None), "sensitivity_objective", None) or "cd").lower()
    _objective_fn(obj)                                     # validates
    rep = Progress(progress)
    if plan is None:
        plan = _plan3d(design, sim, device)
    cfg = plan.solver_config
    if not isinstance(cfg, SolverConfig3D):
        raise TypeError("compute_sensitivity3d needs a 3D plan; use compute_sensitivity for 2D designs")
    if device not in (None, "", "auto"):
        cfg = dataclasses.replace(cfg, device=resolve(device), bodies=list(cfg.bodies))
    dev_name = str(cfg.device if cfg.device not in ("auto", None, "") else resolve("auto"))
    dclass = "cpu" if dev_name.startswith("cpu") else "gpu"
    check_budget(cfg.cells, dclass, max_cells)
    L = float(plan.length_ref)
    unit = max(float(getattr(plan, "strouhal_length", L) or L) / L, 1e-6)
    plan_dt = float(getattr(plan, "dt", 0.0) or 0.0)
    if dclass == "gpu" and plan_dt > 0:
        # rough pre-check before anything costly (the exact one follows once the window is known):
        # the steady window at the plan's step, with the checkpoints on the host
        lead0, avg0, _ = choose_window(unit, dclass, None, window, ensemble)
        _check_memory(estimate_memory_bytes(cfg.cells, steps=math.ceil((lead0 + avg0) / (DT_SAFETY * plan_dt)),
                                            dtype=cfg.dtype, offload=True),
                      torch.device(dev_name), cfg.cells)
    warn: list[str] = [MAGNITUDE_WARNING]
    notes: list[str] = []

    with cpu_threads(dev_name), flush_denormal(dclass == "cpu"):
        if solver is None:
            t_dev = max(float(getattr(plan, "t_avg_start", 0.0) or 0.0), 4.0 * unit)
            flow = develop_flow3d(cfg, t_dev, rep.sub(0.0, 0.3), cancel)
            state, live = flow.state, flow
            rest = rep.sub(0.3, 1.0)
        else:
            state, live = solver, (solver if isinstance(solver, FlowSolver3D) else None)
            rest = rep
        t_state = float(live.t) if live is not None else float("nan")
        period = shedding_period(live, 0.5 * t_state if math.isfinite(t_state) else 0.0)
        lead, avg, members = choose_window(unit, dclass, period, window, ensemble)
        if period is not None:
            warn.append(f"The flow is unsteady (the lift oscillates with a period of {period:.3g} L/U). The map "
                        f"is the sensitivity of a time average over {avg / period:.0f} period(s), averaged over "
                        f"{members} start phase(s); it also reflects how the shape shifts the shedding, so read "
                        "it as a guide to where the shape matters rather than as exact numbers.")
        problem = WindowedObjective3D(cfg, state, objective=obj, lead=lead, average=avg)
        if live is not None and solver is None:
            del flow, state, live                          # free the developing solver before the sweep
            if problem.ops.device.type == "cuda":
                torch.cuda.empty_cache()
        dev = problem.ops.device
        est = problem.memory_estimate()
        avail = _available_bytes(dev)
        if avail is not None and est > MEMORY_FRACTION * avail:
            est = problem.memory_estimate(offload=True)
            _check_memory(est, dev, cfg.cells, problem.steps)
        is_cuda = dev.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(dev)
            before = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        vals, grads = [], []
        shift = (period or 0.0) / members
        for m in range(members):
            if m:
                problem.advance(shift, cancel=cancel)
            sub = rest.sub(0.97 * m / members, 0.97 * (m + 1) / members)
            val_m, g_m = problem.gradient(progress=sub, cancel=cancel)
            vals.append(val_m)
            grads.append(g_m)
        peak = extra = None
        if is_cuda:
            torch.cuda.synchronize(dev)
            peak = int(torch.cuda.max_memory_allocated(dev))
            extra = max(0, peak - int(before))
        notes.extend(problem.notes)
        vgrad = [np.mean([g[k] for g in grads], axis=0) for k in range(len(grads[0]))]
        coef = np.mean([v.coefficients for v in vals], axis=0)
        value = float(np.mean([v.value for v in vals]))
        dens = problem.density(vgrad, smooth_cells=smooth_cells)
        arrays = [g[:s.n_original] / L for g, s in zip(dens, problem.surfaces)]
        normals = [s.normals[:s.n_original].copy() for s in problem.surfaces]
        flipped = sum(s.flipped_faces for s in problem.surfaces)
        if flipped:
            notes.append(f"{flipped:,} refined faces were wound inward and were turned towards the flow for the "
                         "map's outward normals.")

    coeffs = {k: float(coef[i]) for k, i in _COEF.items()}
    maps = SensitivityMaps3D(
        arrays, objective=obj, value=value, coefficients=coeffs, window=(problem.lead, problem.average),
        dt=problem.dt, steps=problem.steps * members, wall_seconds=time.perf_counter() - wall0,
        peak_memory_bytes=peak, extra_memory_bytes=extra, estimated_memory_bytes=est,
        offloaded=bool(problem.offload), cells=cfg.cells, device=str(dev), cuda_graphs=problem.cuda_graphs,
        bodies=[b.name for b in cfg.bodies], points=_wind_points(plan, cfg, L), normals=normals,
        refined_vertices=[len(s.vertices) for s in problem.surfaces], vertex_gradient=vgrad,
        symmetry=bool(problem.ops.symmetry), symmetry_y=float(problem.ops.y0) * L, period=period,
        ensemble=members, smooth_cells=float(smooth_cells), length_ref=L, notes=notes, warnings=warn)
    maps.notes.insert(0, maps.summary())
    rep(1.0, "Sensitivity map ready")
    return maps


def attach_sensitivity3d(result: CFD3DResult, arrays: Sequence[np.ndarray]) -> CFD3DResult:
    """Fill ``SurfaceData3D.sensitivity`` of ``result`` from :func:`compute_sensitivity3d` output.

    Bodies are matched by name when ``arrays`` is a :class:`SensitivityMaps3D`
    (else, and for surfaces whose name no map carries, by order, never taking a
    map that another surface matched by name). An array with one value per surface vertex is used as it
    is; otherwise (the result's surface was mirrored to the full body, or
    re-meshed) every surface vertex takes the value of the nearest map vertex
    (``SensitivityMaps3D.points``, wind frame metres -- the frame of
    ``SurfaceData3D.vertices``), with vertices on the missing half of a
    symmetry-plane run looked up at their mirror image. The map's summary and
    warnings are merged into the result's notes and warnings. Returns ``result``.
    """
    meta = arrays if isinstance(arrays, SensitivityMaps3D) else None
    names = list(meta.bodies) if meta is not None else []
    seq = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    taken = {sd.body for sd in result.surface} & set(names)
    for i, sd in enumerate(result.surface):
        if sd.body in names:
            j = names.index(sd.body)
        elif i < len(names) and names[i] in taken:
            continue                        # that map belongs to another surface of this result
        else:
            j = i
        if j >= len(seq):
            continue
        a = seq[j]
        verts = np.asarray(sd.vertices, dtype=np.float64).reshape(-1, 3)
        if a.size == len(verts):
            sd.sensitivity = a.copy()
            continue
        if meta is None or j >= len(meta.points) or len(meta.points[j]) != a.size or a.size == 0:
            continue
        from scipy.spatial import cKDTree
        src = np.asarray(meta.points[j], dtype=np.float64)
        q = verts.copy()
        if meta.symmetry:
            y0 = float(meta.symmetry_y)
            vis = src[:, 1] >= y0 - 1e-9 * max(float(np.ptp(src)), 1e-300)
            if vis.any():
                src, a = src[vis], a[vis]
            low = q[:, 1] < y0
            q[low, 1] = 2.0 * y0 - q[low, 1]
        with ieee_subnormals():
            _, idx = cKDTree(src).query(q)
        sd.sensitivity = a[idx]
    if meta is not None:
        for line in meta.notes:
            if line not in result.notes:
                result.notes.append(line)
        for line in meta.warnings:
            if line not in result.warnings:
                result.warnings.append(line)
    return result
