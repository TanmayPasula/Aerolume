# mfoil (vendored)

| | |
|---|---|
| Upstream | mfoil, "class and methods for subsonic airfoil analysis", Python version |
| Author | Krzysztof J. Fidkowski, University of Michigan |
| Version | v 2023-06-28 (the date in the file header) |
| Home | <https://websites.umich.edu/~kfid/codes.html> (Matlab and Python versions) |
| Licence | MIT, see [`LICENSE`](LICENSE) (also reproduced verbatim at the top of the vendored copy) |
| Reference | K. J. Fidkowski, "A coupled inviscid–viscous airfoil analysis solver, revisited", *AIAA Journal*, 2022 (preprint and code documentation: <https://websites.umich.edu/~kfid/MYPUBS/Fidkowski_2023_mfoil.pdf>) |

## What lives where

* `third_party/mfoil/mfoil.py` — the **pristine** upstream file, byte for byte. Never edit it; it is
  the baseline every local change is reviewed against.
* `src/aerolume/solvers/airfoil/_mfoil.py` — the copy Aerolume imports. It keeps the full
  copyright and licence header and adds an "AEROLUME MODIFICATIONS" block directly below it that
  lists every patch (P1–P12) plus a `# ruff: noqa` line that exempts the vendored file from the
  project's lint rules. Each changed line carries an `AEROLUME P<n>` comment, so

  ```
  diff third_party/mfoil/mfoil.py src/aerolume/solvers/airfoil/_mfoil.py
  ```

  reads as that list.
* `src/aerolume/solvers/airfoil/_fast.py` — Aerolume's own (Apache-2.0) vectorised re-implementation
  of mfoil's hot loops (panel influence matrices, wake, ue–source sensitivities, residual/Jacobian
  assembly, Newton solve, transition march). It does not modify mfoil; it calls the same
  closures and is tested against the vendored originals to round-off in
  `tests/test_airfoil_fast.py`.

## Local patches (summary)

The authoritative descriptions are in the header of `_mfoil.py`.

| # | Kind | What |
|---|------|------|
| P1 | integration | matplotlib imported lazily inside the plotting functions, so importing the solver never selects a pyplot backend (the Qt app must not get one). |
| P2 | integration | quiet, headless defaults: `Param.verb = 0`, `Param.doplot = False`. |
| P3 | bug fix | `set_coords()`: `X.shape(1)` → `X.shape[1]` (the original raised `TypeError`, so `mfoil(coords=...)` never worked); the input is copied so callers' arrays are not modified. |
| P4 | compatibility | `build_glob_sys()`: compared a sparse matrix with a tuple (a `ValueError` in SciPy ≥ 1.15); compares `.shape`. |
| P5 | bug fix | `calc_force()`: chord normalisation of cl/cm/cdpi was inside the panel loop (only harmless for chord = 1). |
| P6 | latent bugs | 5-digit NACA call syntax, sonic-Cp plot line, `np.zeros(Nsys,1)`, `stgt == None`, bare `except:` — all on paths Aerolume does not use. |
| P7 | bug fix | `get_cteq()`: typo `HK_U` → `Hk_U` when Hk is clipped (linearisation now matches the residual). |
| P8 | feature | forced transition (XFOIL's XTR trip), `Oper.xtr = [lower, upper]` x/c, ≥ 1 = free. |
| P9 | bug fix | exactly sharp trailing edges (zero TE gap): divisions by zero / wrong wake direction / failed assertion. |
| P10 | bug fix | sharp-TE consistency: the gamma-extrapolation row of the influence system kept a non-zero right-hand side (skewed the TE vorticity with alpha). |
| P11 | bug fix | `init_boundary_layer()`: when the march failed at the transition station, its fallback copied the laminar amplification n (~9) into the first turbulent node's sqrt(ctau) slot, giving residuals ~3e3 that Newton never recovered from (cold starts at Re ≥ 6e6). |
| P12 | robustness | `residual_transition()`: the step-limited Newton search for the transition location could cycle without converging; a bracketed bisection now finishes it and the linearisation is evaluated at the final location. Converged cases are unchanged. |

Patches P1–P4 and P6–P7 do not change any result on the paths Aerolume uses; P5 only matters for
chord ≠ 1 (Aerolume always passes unit chord). P8–P12 add or repair behaviour; with free
transition, a blunt trailing edge and a boundary-layer march that does not hit the P11/P12
fallbacks the solution is the upstream one.

Review notes on the upstream code (not changed, worth knowing): the laminar/turbulent closures,
transition and Newton driver are a faithful port of the published method; the code is written for
readability (per-station Python loops, `lil_matrix` assembly), hence `_fast.py`; there is no
global mutable state other than the module-level constants, so separate `mfoil` objects are safe
on separate threads; ruff's one "undefined name" (`Hktgt` in `init_boundary_layer`) is a false
positive (it is always assigned before inverse mode is entered).

## Updating

To take a newer upstream release: replace `mfoil.py` here, re-apply the patches listed in the
`_mfoil.py` header (each is a few lines), re-run `pytest tests/test_airfoil*.py` (the `_fast`
equivalence tests will flag any change in the discretisation the vectorised kernels mirror), and
re-run `validation/airfoil_validation.py`.
