"""Shape parametrisations: CST airfoils and smooth normal-displacement modes.

Two complementary ways to describe *changes* of a shape with a few numbers:

* **CST** (Kulfan's Class-Shape Transformation, AIAA 2008-7085) for airfoils:
  ``y(psi) = C(psi) S(psi) + psi * dz_te`` with the class function
  ``C = psi^0.5 (1 - psi)`` (round nose, sharp tail) and a Bernstein-polynomial
  shape function ``S``. Order 8 reproduces almost any practical section to ~1e-4
  chord, and every coefficient set gives a smooth, valid airfoil -- ideal for
  optimisation and for "fit, then tweak" in the GUI.
* **Normal modes** for any closed outline (cars included): smooth periodic
  Gaussian bumps along the arc length; a displacement ``d = sum a_k phi_k`` along
  the outward normal deforms the shape locally without creating kinks. The
  optimiser pairs these with the solver's surface sensitivity dJ/dn.

The torch parts are imported lazily, inside the functions that need them, so this
module (and the GUI that lists templates) stays importable without paying for
torch start-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .airfoils import cosine_spacing
from .design import _te_thickness, normalize_airfoil
from .polygon import arc_length, as_polygon, perimeter

__all__ = [
    "CSTParams",
    "apply_normal_displacement",
    "cst_eval",
    "cst_fit",
    "cst_surface",
    "normal_basis",
    "vertex_normals",
]


# ---------------------------------------------------------------------------
# CST
# ---------------------------------------------------------------------------

@dataclass
class CSTParams:
    """CST coefficients of an airfoil (chord-normalised).

    ``upper`` / ``lower``: Bernstein coefficients (length ``order + 1``) of the upper
    and lower surface; the lower ones are normally negative. ``te_upper`` /
    ``te_lower``: trailing-edge y of each surface (blunt TE: te_upper - te_lower > 0).
    ``n1``, ``n2``: class-function exponents (0.5, 1.0 = round nose, sharp tail).
    ``residual``: RMS fit error in y/c when produced by :func:`cst_fit`.
    """
    upper: np.ndarray
    lower: np.ndarray
    te_upper: float = 0.0
    te_lower: float = 0.0
    n1: float = 0.5
    n2: float = 1.0
    chord: float = 1.0
    residual: float = float("nan")
    max_error: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def order(self) -> int:
        return len(self.upper) - 1

    @property
    def te_thickness(self) -> float:
        """Trailing-edge thickness in chords."""
        return float(self.te_upper - self.te_lower)

    def to_dict(self) -> dict:
        return {"upper": list(map(float, self.upper)), "lower": list(map(float, self.lower)),
                "te_upper": self.te_upper, "te_lower": self.te_lower, "n1": self.n1, "n2": self.n2,
                "chord": self.chord, "residual": self.residual, "max_error": self.max_error}

    @classmethod
    def from_dict(cls, d: dict) -> CSTParams:
        return cls(upper=np.asarray(d["upper"], float), lower=np.asarray(d["lower"], float),
                   te_upper=float(d.get("te_upper", 0.0)), te_lower=float(d.get("te_lower", 0.0)),
                   n1=float(d.get("n1", 0.5)), n2=float(d.get("n2", 1.0)), chord=float(d.get("chord", 1.0)),
                   residual=float(d.get("residual", float("nan"))),
                   max_error=float(d.get("max_error", float("nan"))))


def _bernstein(psi: np.ndarray, order: int) -> np.ndarray:
    """(len(psi), order + 1) Bernstein basis K_i psi^i (1 - psi)^(n - i)."""
    i = np.arange(order + 1)
    k = np.array([math.comb(order, j) for j in i], dtype=float)
    return k * psi[:, None] ** i * (1.0 - psi[:, None]) ** (order - i)


def _class(psi: np.ndarray, n1: float, n2: float) -> np.ndarray:
    return np.clip(psi, 0, None) ** n1 * np.clip(1.0 - psi, 0, None) ** n2


def cst_surface(coeffs: np.ndarray, psi: np.ndarray, te: float = 0.0, n1: float = 0.5,
                n2: float = 1.0) -> np.ndarray:
    """y/c of one CST surface at stations ``psi`` (x/c)."""
    psi = np.asarray(psi, dtype=float)
    a = np.asarray(coeffs, dtype=float)
    return _class(psi, n1, n2) * (_bernstein(psi, len(a) - 1) @ a) + psi * te


def _split_surfaces(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper (LE -> TE) and lower (LE -> TE) of a normalised Selig-ordered airfoil."""
    ile = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))       # LE is at the origin after normalising
    upper = q[:ile + 1][::-1]
    # sharp TE: point 0 is shared, so the lower surface ends there too
    lower = q[ile:] if _te_thickness(q) > 0 else np.vstack([q[ile:], q[:1]])
    return upper, lower


def cst_fit(coords: np.ndarray, order: int = 8, n1: float = 0.5, n2: float = 1.0) -> CSTParams:
    """Least-squares CST fit of an airfoil (any position / scale / point order).

    The airfoil is normalised to unit chord first (:func:`normalize_airfoil`); the
    trailing-edge y of each surface is taken from the data and the Bernstein
    coefficients of each surface are fitted by linear least squares in y at the
    given x stations. ``residual`` / ``max_error`` report the RMS / max error in y/c
    over the input points. Points slightly ahead of the chord-line leading edge
    (cambered sections have some) are evaluated at psi = 0.
    """
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    q = normalize_airfoil(coords, 1.0)
    upper, lower = _split_surfaces(q)
    res = []
    out = {}
    for name, srf in (("upper", upper), ("lower", lower)):
        psi = np.clip(srf[:, 0], 0.0, 1.0)
        te = float(srf[-1, 1]) if srf[-1, 0] > 0.999 else 0.0
        A = _class(psi, n1, n2)[:, None] * _bernstein(psi, order)
        rhs = srf[:, 1] - psi * te
        # the LE point (psi = 0) carries no information about S; drop all-zero rows
        rows = np.abs(A).sum(1) > 0
        coef, *_ = np.linalg.lstsq(A[rows], rhs[rows], rcond=None)
        out[name] = (coef, te)
        res.append(A @ coef + psi * te - srf[:, 1])
    err = np.concatenate(res)
    return CSTParams(upper=out["upper"][0], lower=out["lower"][0], te_upper=out["upper"][1],
                     te_lower=out["lower"][1], n1=n1, n2=n2, chord=1.0,
                     residual=float(np.sqrt(np.mean(err**2))), max_error=float(np.abs(err).max()))


def cst_eval(params: CSTParams, n: int = 160) -> np.ndarray:
    """Airfoil coordinates from CST parameters: Selig order, CCW, implicitly closed.

    ``n`` points total with cosine spacing (``2 * (n // 2)`` for a sharp TE, one more
    for a blunt one, matching :func:`aerolume.geometry.naca4`). Scaled by
    ``params.chord``.
    """
    psi = cosine_spacing(max(int(n) // 2, 4))
    yu = cst_surface(params.upper, psi, params.te_upper, params.n1, params.n2)
    yl = cst_surface(params.lower, psi, params.te_lower, params.n1, params.n2)
    upper = np.column_stack([psi[::-1], yu[::-1]])
    lower = np.column_stack([psi[1:], yl[1:]])
    if abs(params.te_upper - params.te_lower) < 1e-12:
        lower = lower[:-1]
    return np.vstack([upper, lower]) * params.chord


# ---------------------------------------------------------------------------
# normal-displacement modes (torch)
# ---------------------------------------------------------------------------

def normal_basis(poly, n_modes: int, width: float | None = None):
    """Smooth periodic bump modes along the outline: tensor ``(n_modes, N)``.

    Mode ``k`` is a Gaussian in arc length centred at ``s_k = k P / n_modes`` (P =
    perimeter, measured from vertex 0), wrapped periodically, peak value 1.
    ``width`` is the Gaussian standard deviation **as a fraction of the perimeter**
    (default ``1 / n_modes``: neighbouring bumps overlap so their sum is smooth and
    nearly uniform). Arc length is computed on the input outline, so the modes are
    attached to the material points and survive vertex clustering. Returned in the
    dtype / device of ``poly`` when it is a tensor (float64 CPU otherwise), without
    gradient: the basis is a fixed design parametrisation.
    """
    import torch

    if isinstance(poly, torch.Tensor):
        dev, dt = poly.device, poly.dtype
        p = poly.detach().cpu().double().numpy()
    else:
        dev, dt = torch.device("cpu"), torch.float64
        p = np.asarray(poly, dtype=float)
    p = as_polygon(p, drop_closing=False)
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    per = perimeter(p)
    s = arc_length(p) / per                                   # 0..1
    sig = width if width is not None else 1.0 / n_modes
    if not sig > 0:
        raise ValueError("width must be positive")
    centres = np.arange(n_modes) / n_modes
    d = np.abs(s[None, :] - centres[:, None])
    d = np.minimum(d, 1.0 - d)                                # periodic distance
    phi = np.exp(-0.5 * (d / sig) ** 2)
    return torch.as_tensor(phi, dtype=dt, device=dev)


def vertex_normals(poly_t):
    """Outward unit normals at the vertices of a CCW polygon tensor ``(N, 2)``.

    Central-difference tangent ``p[i+1] - p[i-1]`` rotated clockwise (outward for a
    CCW outline). Differentiable.
    """
    import torch

    t = torch.roll(poly_t, -1, 0) - torch.roll(poly_t, 1, 0)
    n = torch.stack([t[:, 1], -t[:, 0]], dim=1)
    return n / torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(torch.finfo(poly_t.dtype).tiny)


def apply_normal_displacement(poly_t, d):
    """Move every vertex of a CCW polygon tensor by ``d[i]`` along its outward normal.

    ``d`` is an ``(N,)`` tensor (typically ``coeffs @ normal_basis(...)``); positive
    values push the surface outward (adding material). Differentiable with respect
    to both the polygon and ``d``, so a design gradient dJ/d(coeffs) follows from
    the surface sensitivity by the chain rule.
    """
    import torch

    poly_t = torch.as_tensor(poly_t)
    d = torch.as_tensor(d, dtype=poly_t.dtype, device=poly_t.device)
    if d.shape != poly_t.shape[:1]:
        raise ValueError(f"displacement must have shape ({poly_t.shape[0]},), got {tuple(d.shape)}")
    return poly_t + d[:, None] * vertex_normals(poly_t)
