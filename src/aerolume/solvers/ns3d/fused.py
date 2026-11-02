"""Fused CUDA kernels for the 3D fast path (PyTorch jiterator, no autograd).

Why: a 3D step written as plain torch operations is bound by memory traffic.
Each upwind3 flux is ~9 elementwise passes over a field and a step evaluates 27
of them (9 per stage), so the reference core (:mod:`.core`) runs 256 x 128 x 128
cells at ~12 steps/s on the development GPU against the 40 steps/s target. The
jiterator (``torch.cuda.jiterator``) compiles small elementwise CUDA kernels at
run time with NVRTC (shipped with PyTorch: no compiler needed) and takes
*strided views* as inputs, so a whole stencil - five shifted views of the padded
velocity plus the advecting velocities - is read once and written once.
Neighbouring reads hit the L2 cache: a 7-input stencil costs about one
contiguous read-and-write pass.

Kernels (same arithmetic as the reference core, different evaluation order):

* ``crossd0`` / ``crossd``: advection along a *cross* direction of a momentum
  component (fluxes on the two edges, with the precomputed edge velocity sums)
  plus the diffusion along that direction; ``crossd`` adds to an accumulator;
* ``updself``: the advection and diffusion along the component's *own*
  direction, fused with the Runge-Kutta stage update
  ``lerp(u_prev, u0, a) + b dt L`` - the stage value never needs a separate pass;
* ``div`` / ``gsub``: the projection's divergence and gradient subtraction;
* ``sum2``: the edge velocity sums.

jiterator takes at most 8 tensor inputs, which is what shapes this split. The
viscosity enters as a launch argument (``ops.nu_value``), so a captured CUDA
graph bakes it in and :meth:`FlowSolver3D.set_inflow` re-captures (the reference
core reads the ``nu`` tensor instead). Boundary faces are fixed after the fused
update (outlet: convective tendency; walls: zero; inlet: by the inlet/outlet
correction). Everything runs under ``torch.no_grad``; gradients go through
:mod:`.core`. Agreement with the reference core: 1e-15 in float64 after five
steps for every boundary-condition combination (tests/test_ns3d_solver.py).
"""
from __future__ import annotations

import torch

from ..ns2d.core import FORCE_WEIGHTS, SSP_RK3
from .core import (
    Ops3D,
    State3D,
    StepOut3D,
    advection_scale,
    cfl_dt3d,
    face_flux,
    force_coefficients,
    pad_velocity,
)
from .geometry import Geometry3D

_FLUX = {
    # 24/7 x the upwind3 flux (see core.face_flux): central4 part + |V| x 4th difference
    "upwind3": "#define FLX(v, a, b, c, d) ((v) * (((b) + (c)) - ((a) + (d)) / T(7)) "
               "+ T(3) / T(7) * abs(v) * (((b) - (c)) - ((a) - (d)) / T(3)))",
    # 4 x the central flux
    "central": "#define FLX(v, a, b, c, d) ((v) * ((b) + (c)))",
}

_CROSSD0 = """template <typename T> T ns3d_crossd0_{s}(T a, T b, T c, T d, T e, T vlo, T vhi, T kadv, T nuh2) {{
{flux}
  T r = kadv * (FLX(vlo, a, b, c, d) - FLX(vhi, b, c, d, e)) + nuh2 * ((b + d) - T(2) * c);
#undef FLX
  return r;
}}"""

_CROSSD = """template <typename T> T ns3d_crossd_{s}(T a, T b, T c, T d, T e, T vlo, T vhi, T acc, T kadv, T nuh2) {{
{flux}
  T r = acc + kadv * (FLX(vlo, a, b, c, d) - FLX(vhi, b, c, d, e)) + nuh2 * ((b + d) - T(2) * c);
#undef FLX
  return r;
}}"""

_UPDSELF = """template <typename T> T ns3d_updself_{s}(T a, T b, T c, T d, T e, T u0, T lc, T bdt, T al, T kadv, T nuh2) {{
{flux}
  T l = lc + kadv * (FLX(b + c, a, b, c, d) - FLX(c + d, b, c, d, e)) + nuh2 * ((b + d) - T(2) * c);
#undef FLX
  return c + al * (u0 - c) + bdt * l;
}}"""

_DIV = """template <typename T> T ns3d_div(T u1, T u0, T v1, T v0, T w1, T w0, T inv_h) {
  return ((u1 - u0) + (v1 - v0) + (w1 - w0)) * inv_h;
}"""

_GSUB = """template <typename T> T ns3d_gsub(T u, T p1, T p0, T inv_h) {
  return u - (p1 - p0) * inv_h;
}"""

_SUM2 = """template <typename T> T ns3d_sum2(T a, T b) {
  return a + b;
}"""

# --- subgrid model (core.sgs_divergence3d, same arithmetic) -------------------------------
_D2 = """template <typename T> T ns3d_d2(T a, T b, T s) {
  return (a - b) * s;
}"""

_D4 = """template <typename T> T ns3d_d4(T a, T b, T c, T d, T s) {
  return ((a - b) + (c - d)) * s;
}"""

# eddy viscosity from the traceless gradient (g33 = -(g11 + g22)): exactly 8 inputs
_NUT_HEAD = """template <typename T> T ns3d_nut_{m}(T g11, T g22, T g12, T g13, T g21, T g23, T g31, T g32, T c) {{
  T g33 = -(g11 + g22);
"""
_NUT_BODY = {
    "smagorinsky": """  T s12 = T(0.5) * (g12 + g21), s13 = T(0.5) * (g13 + g31), s23 = T(0.5) * (g23 + g32);
  T ss = g11 * g11 + g22 * g22 + g33 * g33 + T(2) * (s12 * s12 + s13 * s13 + s23 * s23);
  return c * sqrt(T(2) * ss + T(1e-30));
}""",
    "wale": """  T s12 = T(0.5) * (g12 + g21), s13 = T(0.5) * (g13 + g31), s23 = T(0.5) * (g23 + g32);
  T ss = g11 * g11 + g22 * g22 + g33 * g33 + T(2) * (s12 * s12 + s13 * s13 + s23 * s23);
  T a11 = g11 * g11 + g12 * g21 + g13 * g31, a12 = g11 * g12 + g12 * g22 + g13 * g32;
  T a13 = g11 * g13 + g12 * g23 + g13 * g33, a21 = g21 * g11 + g22 * g21 + g23 * g31;
  T a22 = g21 * g12 + g22 * g22 + g23 * g32, a23 = g21 * g13 + g22 * g23 + g23 * g33;
  T a31 = g31 * g11 + g32 * g21 + g33 * g31, a32 = g31 * g12 + g32 * g22 + g33 * g32;
  T a33 = g31 * g13 + g32 * g23 + g33 * g33;
  T tr3 = (a11 + a22 + a33) / T(3);
  T d11 = a11 - tr3, d22 = a22 - tr3, d33 = a33 - tr3;
  T d12 = T(0.5) * (a12 + a21), d13 = T(0.5) * (a13 + a31), d23 = T(0.5) * (a23 + a32);
  T sd = d11 * d11 + d22 * d22 + d33 * d33 + T(2) * (d12 * d12 + d13 * d13 + d23 * d23);
  sd = sd > T(0) ? sd : T(0);
  ss = ss > T(0) ? ss : T(0);
  return c * pow(sd, T(1.5)) / (pow(ss, T(2.5)) + pow(sd, T(1.25)) + T(1e-30));
}""",
    "vreman": """  T b11 = g11 * g11 + g12 * g12 + g13 * g13, b22 = g21 * g21 + g22 * g22 + g23 * g23;
  T b33 = g31 * g31 + g32 * g32 + g33 * g33, b12 = g11 * g21 + g12 * g22 + g13 * g23;
  T b13 = g11 * g31 + g12 * g32 + g13 * g33, b23 = g21 * g31 + g22 * g32 + g23 * g33;
  T bb = b11 * b22 - b12 * b12 + b11 * b33 - b13 * b13 + b22 * b33 - b23 * b23;
  bb = bb > T(0) ? bb : T(0);
  return c * sqrt(bb / ((b11 + b22 + b33) + T(1e-30)) + T(1e-30));
}""",
}

_TAU_EDGE = """template <typename T> T ns3d_tau_edge(T n1, T n2, T n3, T n4, T ua, T ub, T va, T vb, T s) {
  return T(0.25) * ((n1 + n2) + (n3 + n4)) * ((ua - ub) + (va - vb)) * s;
}"""

_TAU_DIAG = """template <typename T> T ns3d_tau_diag(T n, T a, T b, T s) {
  return T(2) * n * ((a - b) * s);
}"""

_SGS_ACC = """template <typename T> T ns3d_sgs_acc(T acc, T a1, T a0, T b1, T b0, T c1, T c0, T s) {
  return acc + ((a1 - a0) + (b1 - b0) + (c1 - c0)) * s;
}"""

# --- passive smoke (dye.face_flux / dye.advect_dye3d, same arithmetic) --------------------
# MUSCL face value with the van Leer slope (harmonic mean of the one-sided differences,
# zero at extrema), taken from the upwind side, times the face velocity
_DYE_FLUX = """template <typename T> T ns3d_dye_flux(T a, T b, T c, T d, T vel) {
  T p, q, pq, s, r;
  if (vel > T(0)) {
    p = b - a; q = c - b; pq = p * q;
    s = pq > T(0) ? T(2) * pq / (p + q) : T(0);
    r = b + T(0.5) * s;
  } else {
    p = c - b; q = d - c; pq = p * q;
    s = pq > T(0) ? T(2) * pq / (p + q) : T(0);
    r = c - T(0.5) * s;
  }
  return vel * r;
}"""

# forward Euler update, clipped to 1 and floored at DYE_FLOOR (dt is a device scalar input)
_DYE_UPD = """template <typename T> T ns3d_dye_upd(T c, T fx1, T fx0, T fy1, T fy0, T fz1, T fz0, T dt, T inv_h, T floor) {
  T r = c - (dt * inv_h) * (((fx1 - fx0) + (fy1 - fy0)) + (fz1 - fz0));
  r = r > T(1) ? T(1) : r;
  return r < floor ? T(0) : r;
}"""


# --- heat equation (physics.ThermalOps.fluxes / stage, same arithmetic) -------------------
# the smoke's MUSCL face value times the face velocity, minus the diffusive flux kf (c - b)
# (kf = kappa / h: a launch argument when laminar, a face tensor with the LES diffusivity)
_TH_MUSCL = """  T p, q, pq, s, r;
  if (vel > T(0)) {
    p = b - a; q = c - b; pq = p * q;
    s = pq > T(0) ? T(2) * pq / (p + q) : T(0);
    r = b + T(0.5) * s;
  } else {
    p = c - b; q = d - c; pq = p * q;
    s = pq > T(0) ? T(2) * pq / (p + q) : T(0);
    r = c - T(0.5) * s;
  }
  return vel * r - kf * (c - b);
}"""
_TH_FLUX_S = "template <typename T> T ns3d_th_flux_s(T a, T b, T c, T d, T vel, T kf) {\n" + _TH_MUSCL
_TH_FLUX_T = "template <typename T> T ns3d_th_flux_t(T a, T b, T c, T d, T vel, T kf) {\n" + _TH_MUSCL

# Runge-Kutta stage of theta: lerp(c, c0, al) - b dt div(F) (bdt is a device scalar input)
_TH_UPD = """template <typename T> T ns3d_th_upd(T c, T c0, T dv, T bdt, T al) {
  return c + al * (c0 - c) - bdt * dv;
}"""


class FusedKernels:
    """The kernels for one advection scheme (jiterator compiles them per dtype on first call)."""

    def __init__(self, scheme: str):
        from torch.cuda.jiterator import _create_jit_fn
        if scheme not in _FLUX:
            raise ValueError(scheme)
        fl = _FLUX[scheme]
        self.scheme = scheme
        self.crossd0 = _create_jit_fn(_CROSSD0.format(s=scheme, flux=fl), kadv=1.0, nuh2=0.0)
        self.crossd = _create_jit_fn(_CROSSD.format(s=scheme, flux=fl), kadv=1.0, nuh2=0.0)
        self.updself = _create_jit_fn(_UPDSELF.format(s=scheme, flux=fl), al=0.0, kadv=1.0, nuh2=0.0)
        self.div = _create_jit_fn(_DIV, inv_h=1.0)
        self.gsub = _create_jit_fn(_GSUB, inv_h=1.0)
        self.sum2 = _create_jit_fn(_SUM2)
        self.d2 = _create_jit_fn(_D2, s=1.0)
        self.d4 = _create_jit_fn(_D4, s=1.0)
        self.tau_edge = _create_jit_fn(_TAU_EDGE, s=1.0)
        self.tau_diag = _create_jit_fn(_TAU_DIAG, s=1.0)
        self.sgs_acc = _create_jit_fn(_SGS_ACC, s=1.0)
        self.nut = {m: _create_jit_fn(_NUT_HEAD.format(m=m) + body, c=1.0) for m, body in _NUT_BODY.items()}
        self.dye_flux = _create_jit_fn(_DYE_FLUX)
        self.dye_update = _create_jit_fn(_DYE_UPD, inv_h=1.0, floor=0.0)
        self.th_flux_s = _create_jit_fn(_TH_FLUX_S, kf=0.0)
        self.th_flux_t = _create_jit_fn(_TH_FLUX_T)
        self.th_upd = _create_jit_fn(_TH_UPD, al=0.0)


_CACHE: dict[str, FusedKernels] = {}


def load_kernels(scheme: str, device: torch.device, dtype: torch.dtype) -> FusedKernels:
    """Kernels for ``scheme``, compiled and smoke-tested on ``device`` / ``dtype``.

    Raises RuntimeError when the fused path is unavailable (no CUDA, no
    jiterator, an NVRTC failure or a wrong result), so the caller can fall back
    to the reference core.
    """
    if device.type != "cuda":
        raise RuntimeError("the fused kernels need a CUDA device")
    try:
        k = _CACHE.get(scheme) or FusedKernels(scheme)
        x = torch.linspace(-1.0, 1.0, 9 * 16, device=device, dtype=dtype).reshape(9, 16)
        a, b, c, d, e, u0, lc = x[:7]
        bdt = torch.tensor(0.3, device=device, dtype=dtype)
        r = k.updself(a, b, c, d, e, u0, lc, bdt, al=0.25, kadv=0.7, nuh2=0.5)
        up = scheme == "upwind3"
        lref = lc + 0.7 * (face_flux(None, a, b, c, d, up) - face_flux(None, b, c, d, e, up)) + 0.5 * (b + d - 2 * c)
        ref = c + 0.25 * (u0 - c) + 0.3 * lref
        if not torch.allclose(r, ref, rtol=1e-5, atol=1e-5):
            raise RuntimeError("fused kernel smoke test gave a wrong result")
        _CACHE[scheme] = k
        return k
    except RuntimeError:
        raise
    except Exception as exc:              # any jiterator / NVRTC failure means "not available"
        raise RuntimeError(f"fused kernels unavailable ({type(exc).__name__}: {exc})") from exc


# ---------------------------------------------------------------------------
# the fused step (no autograd)
# ---------------------------------------------------------------------------

def _cross_terms(U, V, W, geom: Geometry3D, ops: Ops3D, kern: FusedKernels, out: dict | None = None):
    """Cross-direction advection + diffusion of each component, plus LES and forcing
    (``out`` receives the eddy viscosity field ``"nut"`` with LES)."""
    nz, ny, nx = ops.nz, ops.ny, ops.nx
    ka = advection_scale(ops)
    nh = ops.nu_value / (ops.h * ops.h)
    m = slice(2, -2)
    # u: cross y (edges carry v summed over x), cross z (w summed over x)
    s = kern.sum2(V[m, 2:ny + 3, 1:nx + 2], V[m, 2:ny + 3, 2:nx + 3])
    lu = kern.crossd0(U[m, 0:ny, m], U[m, 1:ny + 1, m], U[m, 2:ny + 2, m], U[m, 3:ny + 3, m], U[m, 4:ny + 4, m],
                      s[:, :ny], s[:, 1:], kadv=ka, nuh2=nh)
    s = kern.sum2(W[2:nz + 3, m, 1:nx + 2], W[2:nz + 3, m, 2:nx + 3])
    lu = kern.crossd(U[0:nz, m, m], U[1:nz + 1, m, m], U[2:nz + 2, m, m], U[3:nz + 3, m, m], U[4:nz + 4, m, m],
                     s[:nz], s[1:], lu, kadv=ka, nuh2=nh)
    # v: cross x (u summed over y), cross z (w summed over y)
    s = kern.sum2(U[m, 1:ny + 2, m], U[m, 2:ny + 3, m])
    lv = kern.crossd0(V[m, m, 0:nx], V[m, m, 1:nx + 1], V[m, m, 2:nx + 2], V[m, m, 3:nx + 3], V[m, m, 4:nx + 4],
                      s[..., :nx], s[..., 1:], kadv=ka, nuh2=nh)
    s = kern.sum2(W[2:nz + 3, 1:ny + 2, m], W[2:nz + 3, 2:ny + 3, m])
    lv = kern.crossd(V[0:nz, m, m], V[1:nz + 1, m, m], V[2:nz + 2, m, m], V[3:nz + 3, m, m], V[4:nz + 4, m, m],
                     s[:nz], s[1:], lv, kadv=ka, nuh2=nh)
    # w: cross x (u summed over z), cross y (v summed over z)
    s = kern.sum2(U[1:nz + 2, m, m], U[2:nz + 3, m, m])
    lw = kern.crossd0(W[m, m, 0:nx], W[m, m, 1:nx + 1], W[m, m, 2:nx + 2], W[m, m, 3:nx + 3], W[m, m, 4:nx + 4],
                      s[..., :nx], s[..., 1:], kadv=ka, nuh2=nh)
    s = kern.sum2(V[1:nz + 2, m, m], V[2:nz + 3, m, m])
    lw = kern.crossd(W[m, 0:ny, m], W[m, 1:ny + 1, m], W[m, 2:ny + 2, m], W[m, 3:ny + 3, m], W[m, 4:ny + 4, m],
                     s[:, :ny], s[:, 1:], lw, kadv=ka, nuh2=nh)
    nut_max = None
    if ops.turbulence != "none":
        lu, lv, lw, nut_max = _fused_sgs(U, V, W, geom, ops, kern, lu, lv, lw, out=out)
    if ops.forcing is not None:
        lu.add_(ops.forcing[0])
        lv.add_(ops.forcing[1])
        lw.add_(ops.forcing[2])
    return lu, lv, lw, nut_max


def nut_constant(ops: Ops3D) -> float:
    """The model constant times its filter-width factor, as the nut kernels take it."""
    d2 = ops.h * ops.h
    return (ops.les_c * d2) if ops.turbulence == "vreman" else (ops.les_c * ops.les_c * d2)


def _fused_sgs(U, V, W, geom: Geometry3D, ops: Ops3D, kern: FusedKernels, lu, lv, lw, out: dict | None = None):
    """Fused equivalent of :func:`core.sgs_divergence3d`, added to (lu, lv, lw).

    Gradient components at the cell centres -1..n (the off-diagonals are the
    averages of the four surrounding edge differences, i.e. central differences
    averaged over the two neighbouring faces), the eddy viscosity in one kernel,
    the edge and diagonal stresses, then their divergence accumulated into the
    tendencies: 18 kernels instead of ~250 elementwise passes.
    """
    nz, ny, nx = ops.nz, ops.ny, ops.nx
    ih = 1.0 / ops.h
    q = 0.25 * ih
    Z, Y, X = slice(1, nz + 3), slice(1, ny + 3), slice(1, nx + 3)        # cells -1..n
    g11 = kern.d2(U[Z, Y, 2:nx + 4], U[Z, Y, X], s=ih)
    g22 = kern.d2(V[Z, 2:ny + 4, X], V[Z, Y, X], s=ih)
    g12 = kern.d4(U[Z, 2:ny + 4, X], U[Z, 0:ny + 2, X], U[Z, 2:ny + 4, 2:nx + 4], U[Z, 0:ny + 2, 2:nx + 4], s=q)
    g13 = kern.d4(U[2:nz + 4, Y, X], U[0:nz + 2, Y, X], U[2:nz + 4, Y, 2:nx + 4], U[0:nz + 2, Y, 2:nx + 4], s=q)
    g21 = kern.d4(V[Z, Y, 2:nx + 4], V[Z, Y, 0:nx + 2], V[Z, 2:ny + 4, 2:nx + 4], V[Z, 2:ny + 4, 0:nx + 2], s=q)
    g23 = kern.d4(V[2:nz + 4, Y, X], V[0:nz + 2, Y, X], V[2:nz + 4, 2:ny + 4, X], V[0:nz + 2, 2:ny + 4, X], s=q)
    g31 = kern.d4(W[Z, Y, 2:nx + 4], W[Z, Y, 0:nx + 2], W[2:nz + 4, Y, 2:nx + 4], W[2:nz + 4, Y, 0:nx + 2], s=q)
    g32 = kern.d4(W[Z, 2:ny + 4, X], W[Z, 0:ny + 2, X], W[2:nz + 4, 2:ny + 4, X], W[2:nz + 4, 0:ny + 2, X], s=q)
    nut = kern.nut[ops.turbulence](g11, g22, g12, g13, g21, g23, g31, g32, c=nut_constant(ops))
    del g11, g22, g12, g13, g21, g23, g31, g32
    if not geom.empty:
        k0, k1, j0, j1, i0, i1 = geom.window
        nut[k0 + 1:k1 + 1, j0 + 1:j1 + 1, i0 + 1:i1 + 1].mul_(1.0 - geom.chi_c)
    nut_max = nut.amax()
    if out is not None:
        out["nut"] = nut
    t12 = kern.tau_edge(nut[1:nz + 1, 0:ny + 1, 0:nx + 1], nut[1:nz + 1, 1:ny + 2, 0:nx + 1],
                        nut[1:nz + 1, 0:ny + 1, 1:nx + 2], nut[1:nz + 1, 1:ny + 2, 1:nx + 2],
                        U[2:nz + 2, 2:ny + 3, 2:nx + 3], U[2:nz + 2, 1:ny + 2, 2:nx + 3],
                        V[2:nz + 2, 2:ny + 3, 2:nx + 3], V[2:nz + 2, 2:ny + 3, 1:nx + 2], s=ih)
    t13 = kern.tau_edge(nut[0:nz + 1, 1:ny + 1, 0:nx + 1], nut[1:nz + 2, 1:ny + 1, 0:nx + 1],
                        nut[0:nz + 1, 1:ny + 1, 1:nx + 2], nut[1:nz + 2, 1:ny + 1, 1:nx + 2],
                        U[2:nz + 3, 2:ny + 2, 2:nx + 3], U[1:nz + 2, 2:ny + 2, 2:nx + 3],
                        W[2:nz + 3, 2:ny + 2, 2:nx + 3], W[2:nz + 3, 2:ny + 2, 1:nx + 2], s=ih)
    t23 = kern.tau_edge(nut[0:nz + 1, 0:ny + 1, 1:nx + 1], nut[1:nz + 2, 0:ny + 1, 1:nx + 1],
                        nut[0:nz + 1, 1:ny + 2, 1:nx + 1], nut[1:nz + 2, 1:ny + 2, 1:nx + 1],
                        V[2:nz + 3, 2:ny + 3, 2:nx + 2], V[1:nz + 2, 2:ny + 3, 2:nx + 2],
                        W[2:nz + 3, 2:ny + 3, 2:nx + 2], W[2:nz + 3, 1:ny + 2, 2:nx + 2], s=ih)
    t11 = kern.tau_diag(nut[1:nz + 1, 1:ny + 1, :], U[2:nz + 2, 2:ny + 2, 2:nx + 4], U[2:nz + 2, 2:ny + 2, 1:nx + 3],
                        s=ih)
    t22 = kern.tau_diag(nut[1:nz + 1, :, 1:nx + 1], V[2:nz + 2, 2:ny + 4, 2:nx + 2], V[2:nz + 2, 1:ny + 3, 2:nx + 2],
                        s=ih)
    t33 = kern.tau_diag(nut[:, 1:ny + 1, 1:nx + 1], W[2:nz + 4, 2:ny + 2, 2:nx + 2], W[1:nz + 3, 2:ny + 2, 2:nx + 2],
                        s=ih)
    lu = kern.sgs_acc(lu, t11[..., 1:], t11[..., :-1], t12[:, 1:], t12[:, :-1], t13[1:], t13[:-1], s=ih)
    lv = kern.sgs_acc(lv, t12[..., 1:], t12[..., :-1], t22[:, 1:], t22[:, :-1], t23[1:], t23[:-1], s=ih)
    lw = kern.sgs_acc(lw, t13[..., 1:], t13[..., :-1], t23[:, 1:], t23[:, :-1], t33[1:], t33[:-1], s=ih)
    return lu, lv, lw, nut_max


def _pad_psi(psi: torch.Tensor, ops: Ops3D) -> torch.Tensor:
    """psi with one ghost layer on the six faces: replicated (Neumann: zero gradient on
    boundary faces) or wrapped (periodic, including the duplicate face)."""
    nz, ny, nx = psi.shape
    P = psi.new_empty(nz + 2, ny + 2, nx + 2)
    P[1:-1, 1:-1, 1:-1] = psi
    px, py, pz = ops.periodic_axes
    P[1:-1, 1:-1, 0] = psi[..., -1] if px else psi[..., 0]
    P[1:-1, 1:-1, -1] = psi[..., 0] if px else psi[..., -1]
    P[1:-1, 0, 1:-1] = psi[:, -1] if py else psi[:, 0]
    P[1:-1, -1, 1:-1] = psi[:, 0] if py else psi[:, -1]
    P[0, 1:-1, 1:-1] = psi[-1] if pz else psi[0]
    P[-1, 1:-1, 1:-1] = psi[0] if pz else psi[-1]
    return P


def fast_step(state: State3D, geom: Geometry3D, ops: Ops3D, kern: FusedKernels, dt=None
              ) -> tuple[State3D, torch.Tensor]:
    """Fused equivalent of :func:`core.step3d_full` (call under ``torch.no_grad``)."""
    out = fast_step_phys(state, geom, ops, kern, dt)
    return out.state, out.coeffs


def fast_step_phys(state: State3D, geom: Geometry3D, ops: Ops3D, kern: FusedKernels, dt=None, *, phys=None,
                   theta=None, diag: bool = False, theta_kernels: bool = True) -> StepOut3D:
    """Fused equivalent of :func:`core.step3d_phys` (call under ``torch.no_grad``).

    With ``phys=None`` it executes exactly the v0.2 fused step. ``theta_kernels``
    selects the fused heat-equation kernels (False: the reference torch
    arithmetic for theta, used when their self-test failed).
    """
    u0, v0, w0 = state.u, state.v, state.w
    up, vp, wp = u0, v0, w0
    acc = None
    psi_w = None
    nz, ny, nx = ops.nz, ops.ny, ops.nx
    h, inv_h = ops.h, 1.0 / ops.h
    ka = advection_scale(ops)
    nh = ops.nu_value / (h * h)
    px, py, pz = ops.periodic_axes
    n_in = 1.0 / (nz * ny)
    m = slice(2, -2)
    thermal = phys is not None and phys.thermal is not None and theta is not None
    t0 = tp = theta
    want_nut = thermal and ops.turbulence != "none"
    source = phys.mass_flux if phys is not None else None
    tkern = kern if theta_kernels else None
    for s, ((a, b), wgt) in enumerate(zip(SSP_RK3, FORCE_WEIGHTS)):
        U, V, W = pad_velocity(up, vp, wp, ops)
        extra = {} if want_nut else None
        lu, lv, lw, nut_max = _cross_terms(U, V, W, geom, ops, kern, out=extra)
        if s == 0:
            if dt is None:
                if ops.dt_fixed is not None:
                    dt = ops.extras["dt_fixed"]
                elif phys is not None:
                    dt = phys.dt_rule(u0, v0, w0, ops, nut_max, theta=t0)
                else:
                    dt = cfl_dt3d(u0, v0, w0, ops, nut_max)
            if not torch.is_tensor(dt):
                dt = torch.tensor(float(dt), device=ops.device, dtype=ops.dtype)
            if phys is not None:
                phys.begin_step(u0, v0, w0, dt)
        if phys is not None:
            lu, lv, lw = phys.momentum_sources(lu, lv, lw, up, vp, wp, tp, geom)
        bdt = dt * b
        uh = kern.updself(U[m, m, 0:nx + 1], U[m, m, 1:nx + 2], U[m, m, 2:nx + 3], U[m, m, 3:nx + 4],
                          U[m, m, 4:nx + 5], u0, lu, bdt, al=a, kadv=ka, nuh2=nh)
        vh = kern.updself(V[m, 0:ny + 1, m], V[m, 1:ny + 2, m], V[m, 2:ny + 3, m], V[m, 3:ny + 4, m],
                          V[m, 4:ny + 5, m], v0, lv, bdt, al=a, kadv=ka, nuh2=nh)
        wh = kern.updself(W[0:nz + 1, m, m], W[1:nz + 2, m, m], W[2:nz + 3, m, m], W[3:nz + 4, m, m],
                          W[4:nz + 5, m, m], w0, lw, bdt, al=a, kadv=ka, nuh2=nh)
        if not px:
            if ops.closed:
                uh[..., 0] = 0.0
                uh[..., -1] = 0.0
            else:
                # convective outlet u_t + u_x = 0 on the last face, then inlet value and mass balance
                uo, ui = up[..., -1], up[..., -2]
                uh[..., -1] = uo + a * (u0[..., -1] - uo) + bdt * ((ui - uo) * inv_h)
                corr = (ops.u_inlet.sum() - uh[..., -1].sum()) * n_in
                if source is not None:
                    corr = corr + source * n_in
                uh[..., -1].add_(corr)
                uh[..., 0] = ops.u_inlet[..., 0]
        if not py:
            vh[:, 0] = 0.0
            vh[:, -1] = 0.0
        if not pz:
            wh[0] = 0.0
            wh[-1] = 0.0
        if not geom.empty:
            k0, k1, j0, j1, i0, i1 = geom.window
            k = bdt / ops.eta
            incs = []
            for arr, sl, chi, tgt in ((uh, (slice(k0, k1), slice(j0, j1), slice(i0, i1 + 1)), geom.chi_u, geom.a_u),
                                      (vh, (slice(k0, k1), slice(j0, j1 + 1), slice(i0, i1)), geom.chi_v, geom.a_v),
                                      (wh, (slice(k0, k1 + 1), slice(j0, j1), slice(i0, i1)), geom.chi_w, geom.a_w)):
                win = arr[sl]
                new = (win + k * tgt) / (1.0 + k * chi)
                incs.append((new - win).mul_(wgt))
                win.copy_(new)
            if acc is None:
                acc = incs
            else:
                for total, inc in zip(acc, incs):
                    total.add_(inc)
        if phys is not None and phys.paint is not None:
            phys.penalise(uh, vh, wh, bdt, ops)
        if px:
            uh[..., -1] = uh[..., 0]
        if py:
            vh[:, -1] = vh[:, 0]
        if pz:
            wh[-1] = wh[0]
        if thermal and phys.buoyant:
            # after the penalisation, everywhere (physics.ThermalOps.add_buoyancy)
            wh, share = phys.buoyancy(wh, tp, bdt, wgt, geom, ops, inplace=True)
            if share is not None:
                acc[2].add_(share)
        if thermal:
            tp_new = phys.theta_stage(tp, t0, up, vp, wp, a, bdt, wgt, ops,
                                      nut=extra.get("nut") if extra else None, kern=tkern)
        div = kern.div(uh[..., 1:], uh[..., :-1], vh[:, 1:], vh[:, :-1], wh[1:], wh[:-1], inv_h=inv_h)
        if phys is not None:
            div = phys.continuity(div)
            if diag and s == 2:
                phys.record_divergence(div)
        psi = ops.poisson.solve(div)
        P = _pad_psi(psi, ops)
        up = kern.gsub(uh, P[1:-1, 1:-1, 1:nx + 2], P[1:-1, 1:-1, 0:nx + 1], inv_h=inv_h)
        vp = kern.gsub(vh, P[1:-1, 1:ny + 2, 1:-1], P[1:-1, 0:ny + 1, 1:-1], inv_h=inv_h)
        wp = kern.gsub(wh, P[1:nz + 2, 1:-1, 1:-1], P[0:nz + 1, 1:-1, 1:-1], inv_h=inv_h)
        if thermal:
            tp = tp_new
        if psi_w is None:
            psi_w = psi.mul_(wgt)
        else:
            psi_w.add_(psi, alpha=wgt)
    p = psi_w.div_(dt)
    if geom.empty:
        coeffs = torch.zeros(geom.nbody, 6, device=ops.device, dtype=ops.dtype)
    else:
        coeffs = force_coefficients(tuple(acc), geom, ops, dt)
    if phys is not None:
        phys.end_step(dt, ops)
    return StepOut3D(State3D(up, vp, wp, p, state.t + dt, dt * torch.ones_like(state.t)), coeffs,
                     tp if thermal else None)


def check_theta_kernels(kern: FusedKernels, device: torch.device, dtype: torch.dtype) -> None:
    """Compile the heat-equation kernels and compare them with the reference arithmetic on a
    small random field (RuntimeError on a compile failure or a wrong result: the solver then
    advances theta with the plain-torch reference)."""
    from .dye import face_flux
    try:
        g = torch.Generator(device=device).manual_seed(11)
        a, b, c, d, vel, kf, c0, dv = (torch.randn(5, 6, 7, generator=g, device=device, dtype=dtype) for _ in range(8))
        bdt = torch.tensor(0.07, device=device, dtype=dtype)
        r1 = kern.th_flux_s(a, b, c, d, vel, kf=0.3)
        r2 = kern.th_flux_t(a, b, c, d, vel, kf.abs())
        r3 = kern.th_upd(c, c0, dv, bdt, al=0.25)
    except Exception as exc:
        raise RuntimeError(f"fused heat-equation kernels unavailable ({type(exc).__name__}: {exc})") from exc
    ref1 = face_flux(a, b, c, d, vel) - 0.3 * (c - b)
    ref2 = face_flux(a, b, c, d, vel) - kf.abs() * (c - b)
    ref3 = c + 0.25 * (c0 - c) - bdt * dv
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    for r, ref in ((r1, ref1), (r2, ref2), (r3, ref3)):
        if not torch.allclose(r, ref, rtol=tol, atol=tol):
            raise RuntimeError("fused heat-equation kernel self-test gave a wrong result")


def check_dye_kernels(kern: FusedKernels, device: torch.device, dtype: torch.dtype) -> None:
    """Compile the smoke kernels and compare one step with the reference scheme on a
    small random field (raises RuntimeError on a compile failure or a wrong result;
    the solver then advects the smoke with the plain-torch reference)."""
    from types import SimpleNamespace

    from .dye import advect_dye3d
    try:
        g = torch.Generator(device=device).manual_seed(7)
        shape = (6, 5, 7)
        c = torch.rand(shape, generator=g, device=device, dtype=dtype)
        u = torch.randn((6, 5, 8), generator=g, device=device, dtype=dtype) * 0.1
        v = torch.randn((6, 6, 7), generator=g, device=device, dtype=dtype) * 0.1
        w = torch.randn((7, 5, 7), generator=g, device=device, dtype=dtype) * 0.1
        dt = torch.tensor(0.05, device=device, dtype=dtype)
        ops = SimpleNamespace(periodic_axes=(False, False, False), h=0.1)
        a = fast_dye(c, u, v, w, dt, ops, kern)                 # type: ignore[arg-type]
        b = advect_dye3d(c, u, v, w, dt, ops)
    except Exception as exc:
        raise RuntimeError(f"fused smoke kernels unavailable ({type(exc).__name__}: {exc})") from exc
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    if not torch.allclose(a, b, rtol=tol, atol=tol):
        raise RuntimeError("fused smoke kernel smoke test gave a wrong result")


def fast_dye(c: torch.Tensor, u, v, w, dt, ops: Ops3D, kern: FusedKernels) -> torch.Tensor:
    """Fused equivalent of :func:`.dye.advect_dye3d`: three face-flux kernels (one per
    direction, each reading the four-cell stencil as strided views of the padded field)
    and one update kernel. About a dozen field passes against ~400 for a flow step."""
    from .dye import DYE_FLOOR, pad_dye
    nz, ny, nx = c.shape
    P = pad_dye(c, ops)
    m = slice(2, -2)
    fx = kern.dye_flux(P[m, m, 0:nx + 1], P[m, m, 1:nx + 2], P[m, m, 2:nx + 3], P[m, m, 3:nx + 4], u)
    fy = kern.dye_flux(P[m, 0:ny + 1, m], P[m, 1:ny + 2, m], P[m, 2:ny + 3, m], P[m, 3:ny + 4, m], v)
    fz = kern.dye_flux(P[0:nz + 1, m, m], P[1:nz + 2, m, m], P[2:nz + 3, m, m], P[3:nz + 4, m, m], w)
    return kern.dye_update(c, fx[..., 1:], fx[..., :-1], fy[:, 1:], fy[:, :-1], fz[1:], fz[:-1], dt,
                           inv_h=1.0 / ops.h, floor=DYE_FLOOR)


__all__ = ["FusedKernels", "check_dye_kernels", "check_theta_kernels", "fast_dye", "fast_step", "fast_step_phys",
           "load_kernels"]
