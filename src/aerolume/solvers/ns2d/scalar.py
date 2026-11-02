"""Cell-centred scalar transport: bounded advection and flux-form diffusion.

Temperature (theta) and the smoke channels live at cell centres of the MAC grid,
so the face velocities the momentum solver already has are exactly the
advecting velocities of a finite-volume flux. Two building blocks:

* :func:`tvd_fluxes` - the advective flux ``u c`` at every face, with the face
  value reconstructed by MUSCL and limited by van Leer's harmonic slope. The
  forward-Euler step of this flux is total-variation diminishing for face
  Courant numbers up to 1 / (1 + psi_max / 2) = 1/2, which is the solver's
  default CFL rule; SSP-RK3 is a convex combination of such steps, so the
  temperature integrated with the momentum stages stays bounded too (no new
  extrema; theory doc 16.1). It is the existing smoke scheme
  (:func:`~aerolume.solvers.ns2d.core.advect_dye`) with boundary-aware ghosts and
  any number of leading channel dimensions.
* :func:`diffusive_fluxes` - ``F = -kappa grad c`` at every face with the
  domain-wall conditions applied on the boundary faces themselves (Dirichlet
  over half a cell, prescribed flux, adiabatic), so the boundary heat rates the
  solver reports are the fluxes the scheme actually used: in a steady state the
  heat through a wall equals the heat through any line across the domain to
  round-off (discrete conservation), which is what makes the cavity Nusselt
  numbers second-order accurate (theory doc 16.4).

Everything is pure torch with static control flow (CUDA-graph capturable,
differentiable).
"""
from __future__ import annotations

import torch

#: smoke concentrations below this are invisible and set to exactly zero (subnormal guard)
DYE_FLOOR = 1e-6


def van_leer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Van Leer limited slope: harmonic mean of one-sided differences, 0 at extrema."""
    ab = a * b
    s = a + b
    return torch.where(ab > 0, 2.0 * ab / torch.where(ab > 0, s, torch.ones_like(s)), torch.zeros_like(ab))


def _ghost_x(c: torch.Tensor, ops, inflow) -> torch.Tensor:
    """c with two ghost columns per side: wrap (periodic), inflow value at the
    inlet / replicate at the outlet (tunnel), replicate at walls (closed)."""
    if ops.periodic:
        return torch.cat([c[..., -2:], c, c[..., :2]], dim=-1)
    right = c[..., -1:].expand(*c.shape[:-1], 2)
    if getattr(ops, "closed", False) or inflow is None:
        left = c[..., :1].expand(*c.shape[:-1], 2)
    else:
        left = inflow.expand(*c.shape[:-1], 2) if torch.is_tensor(inflow) else torch.full_like(right, float(inflow))
    return torch.cat([left, c, right], dim=-1)


def _ghost_y(c: torch.Tensor, ops) -> torch.Tensor:
    if ops.periodic:
        return torch.cat([c[..., -2:, :], c, c[..., :2, :]], dim=-2)
    shape = list(c.shape)
    shape[-2] = 2
    return torch.cat([c[..., :1, :].expand(shape), c, c[..., -1:, :].expand(shape)], dim=-2)


def tvd_fluxes(c: torch.Tensor, u: torch.Tensor, v: torch.Tensor, ops, inflow=None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Advective fluxes (u c at x-faces, v c at y-faces) of a cell-centred scalar.

    ``c`` has shape (..., ny, nx); ``u`` (ny, nx+1) and ``v`` (ny+1, nx) are the
    face velocities. ``inflow`` is the inlet value in tunnel mode (a tensor
    broadcastable to (..., ny, 1), or a number; None = zero gradient). Walls
    have zero normal velocity, so their flux is zero whatever the ghosts hold;
    the ghosts only keep the limiter's slopes symmetric there.
    """
    cx = _ghost_x(c, ops, inflow)
    s = van_leer(cx[..., 1:-1] - cx[..., :-2], cx[..., 2:] - cx[..., 1:-1])        # cells -1..nx
    left = cx[..., 1:-2] + 0.5 * s[..., :-1]
    right = cx[..., 2:-1] - 0.5 * s[..., 1:]
    fx = u * torch.where(u > 0, left, right)
    cy = _ghost_y(c, ops)
    s = van_leer(cy[..., 1:-1, :] - cy[..., :-2, :], cy[..., 2:, :] - cy[..., 1:-1, :])
    low = cy[..., 1:-2, :] + 0.5 * s[..., :-1, :]
    high = cy[..., 2:-1, :] - 0.5 * s[..., 1:, :]
    fy = v * torch.where(v > 0, low, high)
    return fx, fy


def flux_divergence(fx: torch.Tensor, fy: torch.Tensor, h: float) -> torch.Tensor:
    """-(div F) at cell centres for face fluxes F: the tendency of a conserved scalar."""
    return (fx[..., :-1] - fx[..., 1:] + fy[..., :-1, :] - fy[..., 1:, :]) * (1.0 / h)


# ---------------------------------------------------------------------------
# diffusion with domain-wall conditions
# ---------------------------------------------------------------------------

#: boundary modes understood by :func:`diffusive_fluxes`
BOUNDARY_MODES = ("adiabatic", "temperature", "heat_flux", "inlet", "outlet", "periodic")


def diffusive_fluxes(c: torch.Tensor, kx, ky, ops, modes: tuple[str, str, str, str], values: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffusive fluxes ``F = -kappa dc/dn`` (positive along +x / +y) at every face.

    ``kx`` / ``ky`` are the diffusivities at the x-faces (ny, nx+1) and y-faces
    (ny+1, nx), or 0-d tensors for a uniform value. ``modes`` gives the
    (left, right, bottom, top) boundary type and ``values`` (4,) the matching
    wall temperature ("temperature", "inlet") or dimensionless gradient G into
    the fluid ("heat_flux", flux kappa G). "temperature" / "inlet" take the
    one-sided difference over the half cell between the wall and the first
    centre; "adiabatic" and "outlet" have zero diffusive flux; "periodic" wraps.
    """
    h = ops.h
    inv_h, inv_half = 1.0 / h, 2.0 / h
    kx_t, ky_t = torch.is_tensor(kx) and kx.dim() > 0, torch.is_tensor(ky) and ky.dim() > 0
    kxi = kx[..., 1:-1] if kx_t else kx
    kyi = ky[..., 1:-1, :] if ky_t else ky
    fxi = -(kxi * (c[..., 1:] - c[..., :-1])) * inv_h
    fyi = -(kyi * (c[..., 1:, :] - c[..., :-1, :])) * inv_h
    kl = kx[..., :1] if kx_t else kx
    kr = kx[..., -1:] if kx_t else kx
    kb = ky[..., :1, :] if ky_t else ky
    kt = ky[..., -1:, :] if ky_t else ky
    first_x, last_x = c[..., :1], c[..., -1:]
    first_y, last_y = c[..., :1, :], c[..., -1:, :]

    def side(mode, k, inner, value, sign, wrap):
        # flux on a boundary face, positive along +axis; sign = +1 on the low side, -1 on the high side
        if mode in ("temperature", "inlet"):
            return (-sign * inv_half) * k * (inner - value)
        if mode == "heat_flux":
            return (sign * k) * value * torch.ones_like(inner)
        if mode == "periodic":
            return wrap
        return torch.zeros_like(inner)

    wrap_x = -(kl * (first_x - last_x)) * inv_h if modes[0] == "periodic" else None
    wrap_y = -(kb * (first_y - last_y)) * inv_h if modes[2] == "periodic" else None
    fl = side(modes[0], kl, first_x, values[0], +1.0, wrap_x)
    fr = side(modes[1], kr, last_x, values[1], -1.0, wrap_x)
    fb = side(modes[2], kb, first_y, values[2], +1.0, wrap_y)
    ft = side(modes[3], kt, last_y, values[3], -1.0, wrap_y)
    return torch.cat([fl, fxi, fr], dim=-1), torch.cat([fb, fyi, ft], dim=-2)


def boundary_heat_rates(fx: torch.Tensor, fy: torch.Tensor, h: float) -> torch.Tensor:
    """Heat rate into the fluid through (left, right, bottom, top), per unit span.

    From the diffusive face fluxes of :func:`diffusive_fluxes` (the advective
    flux through a wall is zero). Positive = heat entering the fluid.
    """
    return torch.stack([fx[..., :, 0].sum(-1), -fx[..., :, -1].sum(-1),
                        fy[..., 0, :].sum(-1), -fy[..., -1, :].sum(-1)], dim=-1) * h


def face_average_x(q: torch.Tensor, ops) -> torch.Tensor:
    """A cell-centred field averaged to the x-faces (ny, nx+1); boundary faces copy the
    adjacent cell (wrap when periodic)."""
    mid = 0.5 * (q[..., 1:] + q[..., :-1])
    if ops.periodic:
        w = 0.5 * (q[..., :1] + q[..., -1:])
        return torch.cat([w, mid, w], dim=-1)
    return torch.cat([q[..., :1], mid, q[..., -1:]], dim=-1)


def face_average_y(q: torch.Tensor, ops) -> torch.Tensor:
    """A cell-centred field averaged to the y-faces (ny+1, nx)."""
    mid = 0.5 * (q[..., 1:, :] + q[..., :-1, :])
    if ops.periodic:
        w = 0.5 * (q[..., :1, :] + q[..., -1:, :])
        return torch.cat([w, mid, w], dim=-2)
    return torch.cat([q[..., :1, :], mid, q[..., -1:, :]], dim=-2)


__all__ = [
    "BOUNDARY_MODES",
    "DYE_FLOOR",
    "boundary_heat_rates",
    "diffusive_fluxes",
    "face_average_x",
    "face_average_y",
    "flux_divergence",
    "tvd_fluxes",
    "van_leer",
]
