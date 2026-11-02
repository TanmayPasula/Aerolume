"""Exact pressure-Poisson solver on the 3D cell-centred grid.

The projection needs ``L psi = f`` with ``L`` the standard 7-point Laplacian on
cell centres. Along every axis the boundary condition is either homogeneous
Neumann (every boundary-normal velocity is prescribed: inlet, outlet after the
mass-flux correction, walls, the symmetry plane) or periodic. ``L`` is then a
Kronecker sum ``L = Lx (+) Ly (+) Lz`` of 1D operators, and the eigenvectors of
each 1D operator are known in closed form:

* Neumann, cell-centred: the DCT-II basis
  ``q_k(j) = s_k cos(pi k (j + 1/2) / n)``, eigenvalue ``(2 cos(pi k / n) - 2) / h^2``;
* periodic: the discrete Hartley basis ``q_k(j) = cas(2 pi k j / n) / sqrt(n)``
  (``cas = cos + sin``), eigenvalue ``(2 cos(2 pi k / n) - 2) / h^2``. The
  Hartley vectors are real, symmetric and orthonormal, and since the periodic
  Laplacian is a symmetric circulant matrix (``lambda_k = lambda_(n-k)``) each of
  them is an eigenvector.

With the orthonormal matrices ``Q`` (rows = eigenvectors) a solve is three
forward transforms, a division by ``lambda_x + lambda_y + lambda_z`` and three
inverse transforms (``Q^T``): exact to round-off, no iterations, the same cost
on every call - what CUDA-graph capture and autograd want. The constant mode is
the null space of the all-Neumann (or periodic) operator; it is set to zero, so
``psi`` has zero mean (the projection makes ``f`` sum to zero by construction).

Why dense matrix products instead of FFT-based DCTs (2D uses Makhoul's packed
DCT): on the grids this solver runs (n <= ~600 per axis) one transform along an
axis is a single cuBLAS GEMM that reads and writes the field once, while a
DCT via FFT needs a gather, a complex FFT, a twiddle multiply and a repack. On
the development GPU a forward + inverse 3D transform of 256 x 128 x 128 takes
0.9 ms as GEMMs against ~2.5 ms for packed DCTs (``docs/theory_ns3d.md`` 4).
The cost per element grows like n, so very long axes would favour FFTs again.

TF32 must never be used here: its 10-bit mantissa would leave a divergence of
~1e-3 after the projection. :func:`ieee_matmul` forces IEEE float32 GEMMs for
the duration of a solve whatever the process-wide setting is.
"""
from __future__ import annotations

import contextlib
import math

import torch


@contextlib.contextmanager
def ieee_matmul():
    """Force IEEE (non-TF32) float32 cuBLAS matmuls inside the block.

    Uses the per-backend ``fp32_precision`` switch of recent PyTorch (reading
    the legacy ``allow_tf32`` flag raises once the new API has been used
    anywhere in the process). On a PyTorch without it, the legacy flag is used.
    """
    backends = [torch.backends.cuda.matmul]
    mkl = getattr(getattr(torch.backends, "mkldnn", None), "matmul", None)   # CPU: bf16/tf32 via oneDNN
    if mkl is not None:
        backends.append(mkl)
    changed = []
    try:
        for m in backends:
            if hasattr(m, "fp32_precision"):
                old = m.fp32_precision
                if old != "ieee":
                    m.fp32_precision = "ieee"
                    changed.append((m, "fp32_precision", old))
            elif m is torch.backends.cuda.matmul and m.allow_tf32:   # pragma: no cover - older PyTorch
                m.allow_tf32 = False
                changed.append((m, "allow_tf32", True))
        yield
    finally:
        for m, name, old in reversed(changed):
            setattr(m, name, old)


def neumann_basis(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal DCT-II matrix (rows = eigenvectors) and 1D eigenvalues for h = 1, float64."""
    k = torch.arange(n, dtype=torch.float64)[:, None]
    j = torch.arange(n, dtype=torch.float64)[None, :]
    q = torch.cos(math.pi * k * (j + 0.5) / n) * math.sqrt(2.0 / n)
    q[0] *= math.sqrt(0.5)
    lam = 2.0 * torch.cos(math.pi * torch.arange(n, dtype=torch.float64) / n) - 2.0
    return q, lam


def periodic_basis(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal discrete Hartley matrix (symmetric) and periodic 1D eigenvalues for h = 1, float64."""
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k[:, None] * k[None, :] / n
    q = (torch.cos(ang) + torch.sin(ang)) / math.sqrt(n)
    lam = 2.0 * torch.cos(2.0 * math.pi * k / n) - 2.0
    return q, lam


class PoissonSolver3D:
    """Direct solver for ``L psi = f`` on an ``(nz, ny, nx)`` cell-centred grid.

    ``periodic = (px, py, pz)`` selects periodic (True) or homogeneous Neumann
    (False) conditions per axis. The solution has zero mean and the zero mode
    of ``f`` is ignored. Works on CPU and CUDA, in float32 and float64, and is
    differentiable (matrix products and one elementwise product).
    """

    def __init__(self, nx: int, ny: int, nz: int, h: float, *, periodic: tuple[bool, bool, bool],
                 device: torch.device, dtype: torch.dtype):
        self.nx, self.ny, self.nz, self.h = int(nx), int(ny), int(nz), float(h)
        self.periodic = tuple(bool(p) for p in periodic)
        mats, lams = [], []
        for n, per in zip((self.nx, self.ny, self.nz), self.periodic):
            q, lam = periodic_basis(n) if per else neumann_basis(n)
            mats.append(q)
            lams.append(lam / (self.h * self.h))
        qx, qy, qz = mats
        self.qx = qx.to(device=device, dtype=dtype)
        self.qy = qy.to(device=device, dtype=dtype)
        self.qz = qz.to(device=device, dtype=dtype)
        # contiguous transposes: cuBLAS handles transposed operands, but explicit
        # copies keep the inverse transforms on the same (fast) code path
        self.qxt = self.qx.t().contiguous()
        self.qyt = self.qy.t().contiguous()
        self.qzt = self.qz.t().contiguous()
        lx, ly, lz = lams
        lam = lz[:, None, None] + ly[None, :, None] + lx[None, None, :]
        inv = torch.where(lam != 0, 1.0 / torch.where(lam != 0, lam, torch.ones_like(lam)), torch.zeros_like(lam))
        inv[0, 0, 0] = 0.0                      # remove the constant (null-space) mode
        self.inv_lam = inv.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ transforms
    def _along_x(self, a: torch.Tensor, mt: torch.Tensor) -> torch.Tensor:
        """Apply a matrix ``m`` along the last axis, given its transpose ``mt`` as a contiguous
        tensor: ``out[..., k] = sum_j m[k, j] a[..., j] = (a @ mt)[..., k]``. Passing the
        transpose pre-made keeps cuBLAS on its "nn" kernels, which are ~4x faster here
        than the transposed-operand ones (measured, 256 x 128 x 128 float32)."""
        nz, ny, nx = a.shape
        return (a.reshape(-1, nx) @ mt).reshape(nz, ny, nx)

    def _along_y(self, a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.matmul(m, a)                       # nz batched (ny x ny) @ (ny x nx)

    def _along_z(self, a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        nz, ny, nx = a.shape
        return (m @ a.reshape(nz, -1)).reshape(nz, ny, nx)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Coefficients of ``f`` in the eigenbasis (orthonormal transform)."""
        with ieee_matmul():
            return self._along_z(self._along_y(self._along_x(f, self.qxt), self.qy), self.qz)

    def inverse(self, c: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`forward`."""
        with ieee_matmul():
            return self._along_x(self._along_y(self._along_z(c, self.qzt), self.qyt), self.qx)

    def solve(self, f: torch.Tensor) -> torch.Tensor:
        """Solution ``psi`` (same shape as ``f``) of ``L psi = f`` with zero mean."""
        with ieee_matmul():
            c = self._along_z(self._along_y(self._along_x(f, self.qxt), self.qy), self.qz)
            c = c * self.inv_lam
            return self._along_x(self._along_y(self._along_z(c, self.qzt), self.qyt), self.qx)

    # ------------------------------------------------------------------ diagnostics
    def laplacian(self, psi: torch.Tensor) -> torch.Tensor:
        """The discrete operator ``L`` itself (tests and diagnostics)."""
        pe = psi
        for dim, per in zip((2, 1, 0), self.periodic):
            n = pe.shape[dim]
            if per:
                lo, hi = pe.narrow(dim, n - 1, 1), pe.narrow(dim, 0, 1)
            else:
                lo, hi = pe.narrow(dim, 0, 1), pe.narrow(dim, n - 1, 1)
            pe = torch.cat([lo, pe, hi], dim=dim)
        c = pe[1:-1, 1:-1, 1:-1]
        ring = (pe[1:-1, 1:-1, 2:] + pe[1:-1, 1:-1, :-2] + pe[1:-1, 2:, 1:-1] + pe[1:-1, :-2, 1:-1]
                + pe[2:, 1:-1, 1:-1] + pe[:-2, 1:-1, 1:-1])
        return (ring - 6.0 * c) / (self.h * self.h)

    @property
    def nbytes(self) -> int:
        """Device memory held by the solver (matrices and inverse eigenvalues)."""
        return sum(t.numel() * t.element_size() for t in (self.qx, self.qy, self.qz, self.qxt, self.qyt,
                                                          self.qzt, self.inv_lam))
