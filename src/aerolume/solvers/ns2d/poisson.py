"""Exact pressure-Poisson solvers on the cell-centred grid.

The projection step needs ``L psi = f`` where ``L`` is the standard 5-point
Laplacian on cell centres. With homogeneous Neumann conditions on all four
sides (the normal velocity on every domain boundary is prescribed, so the
pressure correction has zero normal gradient there) the eigenvectors of ``L``
are exactly the DCT-II basis functions. A forward 2D DCT, a division by the
eigenvalues and an inverse DCT therefore solve the discrete problem to
round-off: no iterations, no tolerance, and the same cost on every call, which
is what CUDA-graph capture and autograd want. In periodic mode the eigenvectors
are Fourier modes and plain real FFTs do the same job.

DCT via FFT
-----------
A length-N DCT-II is computed with Makhoul's reordering (evens ascending, odds
descending) followed by one *real* FFT of length N and a quarter-wave twiddle.
For real input the half spectrum ``Z_k = exp(-i pi k / 2N) V_k`` (k = 0..N/2)
already holds every DCT coefficient::

    X_k     =  2 Re Z_k        k = 0 .. N/2
    X_(N-k) = -2 Im Z_k        k = 1 .. (N-1)/2

so we never form the full complex spectrum. We go one step further and never
put the coefficients back in natural order either: the solver keeps them in
the *packed* layout ``[Re Z_0..Re Z_(N/2), Im Z_1..Im Z_m]`` (factors 2 and -2
dropped because they cancel in the inverse) and permutes the eigenvalues once
at construction instead. The inverse rebuilds ``Z`` from the packed array,
untwiddles, runs ``irfft`` and undoes the reordering with one gather.

Everything is ordinary differentiable torch (``index_select``, ``fft.rfft``,
``fft.irfft``, elementwise ops), so gradients flow through the solve, and it
runs unchanged on CUDA, CPU and MPS (where FFTs are supported).
"""
from __future__ import annotations

import math

import torch


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


class _DCTAxis:
    """Packed DCT-II / inverse along one axis of length ``n``.

    Holds the reorder permutations, the twiddles and the packed eigen-index
    for one dimension so that repeated solves allocate nothing new except
    their outputs.
    """

    def __init__(self, n: int, dim: int, device: torch.device, dtype: torch.dtype):
        self.n, self.dim = n, dim
        self.half = n // 2 + 1                    # rfft length
        self.m = (n - 1) // 2                     # number of packed imaginary parts
        perm = list(range(0, n, 2)) + list(range(1, n, 2))[::-1]
        inv = [0] * n
        for pos, src in enumerate(perm):
            inv[src] = pos
        self.perm = torch.tensor(perm, device=device, dtype=torch.long)
        self.iperm = torch.tensor(inv, device=device, dtype=torch.long)
        k = torch.arange(self.half, device=device, dtype=torch.float64)
        ang = -math.pi * k / (2 * n)
        shape = [1, 1]
        shape[dim] = self.half
        cd = _complex_dtype(dtype)
        self.tw = torch.polar(torch.ones_like(ang), ang).to(cd).reshape(shape)
        self.itw = self.tw.conj().resolve_conj()
        # natural wavenumber of each packed slot: 0..N/2, then N-1 .. N-m
        self.k_packed = list(range(self.half)) + [n - k for k in range(1, self.m + 1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.index_select(self.dim, self.perm)
        z = torch.fft.rfft(v, dim=self.dim) * self.tw
        return torch.cat([z.real, z.imag.narrow(self.dim, 1, self.m)], dim=self.dim)

    def inverse(self, p: torch.Tensor) -> torch.Tensor:
        re = p.narrow(self.dim, 0, self.half)
        parts = [torch.zeros_like(re.narrow(self.dim, 0, 1)), p.narrow(self.dim, self.half, self.m)]
        if self.n % 2 == 0:
            # Nyquist: Z_(N/2) = X_(N/2) (1 - i) / 2, so its imaginary part is -Re
            parts.append(-re.narrow(self.dim, self.half - 1, 1))
        im = torch.cat(parts, dim=self.dim)
        z = torch.complex(re, im) * self.itw
        v = torch.fft.irfft(z, n=self.n, dim=self.dim)
        return v.index_select(self.dim, self.iperm)


def laplacian_eigenvalues_neumann(n: int, h: float) -> list[float]:
    """Eigenvalues (2 cos(pi k / n) - 2) / h^2 of the 1D Neumann cell-centred Laplacian."""
    return [(2.0 * math.cos(math.pi * k / n) - 2.0) / h**2 for k in range(n)]


class PoissonSolver:
    """Direct solver for ``L psi = f`` on an ``(ny, nx)`` cell-centred grid.

    ``periodic=False``: homogeneous Neumann on all sides (DCT-II). The
    operator is singular (constants are in its null space); the zero mode of
    the solution is set to zero, so ``psi`` has zero mean, and the zero mode of
    ``f`` is ignored. The projection makes ``f`` sum to zero by construction
    (global mass-flux correction at the outlet), so nothing is lost.

    ``periodic=True``: periodic in both directions (real 2D FFT).
    """

    def __init__(self, nx: int, ny: int, h: float, *, periodic: bool,
                 device: torch.device, dtype: torch.dtype):
        self.nx, self.ny, self.h, self.periodic = nx, ny, h, periodic
        if periodic:
            kx = torch.arange(nx // 2 + 1, device=device, dtype=torch.float64)
            ky = torch.arange(ny, device=device, dtype=torch.float64)
            lx = (2 * torch.cos(2 * math.pi * kx / nx) - 2) / h**2
            ly = (2 * torch.cos(2 * math.pi * ky / ny) - 2) / h**2
            lam = ly[:, None] + lx[None, :]
        else:
            self.ax = _DCTAxis(nx, 1, device, dtype)
            self.ay = _DCTAxis(ny, 0, device, dtype)
            ex = torch.tensor(laplacian_eigenvalues_neumann(nx, h), dtype=torch.float64)
            ey = torch.tensor(laplacian_eigenvalues_neumann(ny, h), dtype=torch.float64)
            lx = ex[self.ax.k_packed]
            ly = ey[self.ay.k_packed]
            lam = (ly[:, None] + lx[None, :]).to(device)
        inv = torch.where(lam != 0, 1.0 / torch.where(lam != 0, lam, 1.0), 0.0)
        inv[0, 0] = 0.0                         # remove the constant (null-space) mode
        self.inv_lam = inv.to(dtype)

    def solve(self, f: torch.Tensor) -> torch.Tensor:
        """Solution ``psi`` (same shape as ``f``) of ``L psi = f`` with zero mean."""
        if self.periodic:
            fh = torch.fft.rfft2(f) * self.inv_lam
            return torch.fft.irfft2(fh, s=(self.ny, self.nx))
        fh = self.ay.forward(self.ax.forward(f)) * self.inv_lam
        return self.ax.inverse(self.ay.inverse(fh))

    def laplacian(self, psi: torch.Tensor) -> torch.Tensor:
        """The discrete operator ``L`` itself (used by tests and diagnostics)."""
        if self.periodic:
            pe = torch.cat([psi[:, -1:], psi, psi[:, :1]], dim=1)
            pe = torch.cat([pe[-1:], pe, pe[:1]], dim=0)
        else:
            pe = torch.cat([psi[:, :1], psi, psi[:, -1:]], dim=1)
            pe = torch.cat([pe[:1], pe, pe[-1:]], dim=0)
        return (pe[1:-1, 2:] + pe[1:-1, :-2] + pe[2:, 1:-1] + pe[:-2, 1:-1] - 4 * pe[1:-1, 1:-1]) / self.h**2
