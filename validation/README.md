# Validation

This folder holds the evidence that Aerolume's numbers are right: every verification and
validation case, the scripts that run them, and the runner that compares them with their
published references and writes the tables of [docs/validation.md](../docs/validation.md).
The methods behind the solvers are in [docs/methods.md](../docs/methods.md).

## Run it

From the repository root, with Aerolume installed (`pip install -e .`):

```
python validation/run_all.py --quick                  # every area, coarse grids: a few minutes on a GPU
python validation/run_all.py --all                    # every case at its documented resolution
python validation/run_all.py --all --stretch          # also the long cases (sphere Re 300)
python validation/run_all.py --list                   # the cases, their references and their cost
python validation/run_all.py --cases cavity2d,cyl40   # some cases (key or key prefix)
python validation/run_all.py --all --area thermal     # one area (repeatable)
python validation/run_all.py --device cpu --quick     # force the CPU (slower, same checks)
```

Add `--write-docs` to regenerate section 9 of `docs/validation.md` and its figures in
`docs/img/`. Everything else in that document is hand-written and is left alone: the
generated part sits between the `<!-- run_all:begin -->` and `<!-- run_all:end -->` markers.

Before a release, check that section 9 still describes the code being shipped:

```
python validation/run_all.py --check-docs     # exit 0: measured on this code; 1: stale, re-run
```

Every run records a SHA-256 digest of the code its numbers depend on (the solvers, `core`,
`geometry`, `geometry3d`, the GCI module, `model.py`, the package version and the
`validation/*.py` case modules with their bands; line endings normalised). Section 9 prints
its first 16 hex digits, and `--check-docs` recomputes it. The GUI, the reports and the
comparison tables are outside the digest, so editing them does not mark the validation stale.
The installed package metadata is not used for the version: an editable install keeps
reporting the version it was installed with, which is how section 9 once carried the wrong
version during development.

On a machine shared with other GPU work, run the areas one after the other and document
them together afterwards; each run writes a JSON record, and `--from-json` reports any set of
records without running anything (a case in several records keeps its latest run):

```
python validation/run_all.py --all --area thermal --out validation/out
python validation/run_all.py --all --area fans --out validation/out
python validation/run_all.py --from-json validation/out/*.json --write-docs
```

`aerolume-cli validate` runs its own smaller built-in subset (Taylor–Green, the Re 40
cylinder, NACA 0012 against Ladson; `--full` adds the Re 100 shedding cylinder and the heated
cavity through the planner) and works from an installed program without this folder. Use this
script for the complete set.

## What a run reports

Every case returns one or more **checks**: a measured quantity, the band it must fall in, the
reference value or range and the source it comes from. A case's status is

| status | meaning | exit code |
|---|---|---|
| pass | every check is inside its band | 0 |
| fail | at least one check is outside its band | 1 |
| error | the case crashed (the traceback is in the JSON record) | 1 |
| unavailable | the installation lacks a solver feature the case needs | 0 (1 with `--strict`) |

An unavailable case is listed, never counted as passed. Bad arguments exit with 2.

The JSON record (`--out`, default `validation/out/`, which is git-ignored) holds every check,
the case data the figures are drawn from, the wall time of every case and the machine: GPU,
OS, Python, PyTorch, CUDA, the Aerolume version and the source digest. With `--write-docs`
the record the documentation was written from (merged, for `--from-json`) is also kept in
the repository as `validation/records/validation-record.json`, so every number in section 9
can be traced to its run. Records measured on different code are marked as mixed in the
documentation, and a run during which the measured code changed is marked the same way.

## How the bands are set

A band is the published value, or the spread of several published values, widened by a stated
tolerance that says what the method should reach at that resolution and why: confinement of
the computational domain, the one-cell diffuse immersed boundary, a coarse quick grid, the
scatter of empirical correlations. Each band is written next to its check in `run_all.py`, with
the reason in the case's docstring, **before** the case is run, and is not moved afterwards to
make a result pass. If a check turns out to test the wrong thing, the case is changed for a
stated physical reason and the failed version is recorded in the case's docstring and in
`docs/validation.md`. Examples: the grid-convergence case (`case_gci_cavity`) first asked for
the benchmark value inside the GCI band, which the benchmark's own rounding cannot meet; the
actuator strip (`case_actuator2d`) once compared a fan blowing against the stream with the
theory of one blowing with it, and now runs and checks both directions; the plane jet
(`case_plane_jet`) first integrated the momentum flux over the whole closed box, including the
return flow the box forces beside the jet, and then over the jet's forward-flow core, where it
still drifted 6.2 % against a 5 % band and failed. The momentum budget of the core
(`thermal2d_cases.jet_measures`) then split that drift into the shear of the box's return flow
(−3.6 %) and the box's adverse pressure gradient (−2.5 %), leaving −0.27 % of K to the scheme.
That budget's residual is now a check of its own (< 1 % of K). The raw drift keeps the 5 % band
it had before its first run and fails: a development build before 0.1.0 widened it to an 8 % "sanity bound"
after the 6.2 % was known, which is a band moved to pass, so it was withdrawn. The round jet
(`jet3d`) was designed with the budget from the start.

The other band set the wrong way round has been replaced too. In early development the square cylinder's
drag band (`case_square`) was the published range +8 % above, written after the +5.9 %
deviation at 32 cells per side was already documented; it contained a known result and tested
nothing. Before the first release it moved to the rule of the other 2D bands: the published range ± 5 %, the
widest resolution allowance those bands already used (Schäfer–Turek 2D-1 lift). The case now
runs 32 and 48 cells per side; the 32-cell check fails and stays in the table as a failure, and
48 cells is the grid that passes. Nothing about the rule was tuned to this case, but it was
chosen knowing the 48-cell value, and docs/validation.md section 3 says so.

Exact checks (verification) compare with an exact answer: a solution of the equations
(Taylor–Green), an identity of the discretisation (a divergence-free projection, a symmetry
plane equal to the full domain) or a conservation law (energy balance of a steady cavity).

## Cases

`python validation/run_all.py --list` prints the current registry with costs. The areas:

| area | cases | references |
|---|---|---|
| verification | Taylor–Green 2D and 3D, discrete incompressibility, smoke and heat across a symmetry plane | exact |
| 2D flow | cylinder Re 40 and 100, Schäfer–Turek 2D-1 and 2D-2, square cylinder Re 100 | Dennis & Chang, Coutanceau & Bouard, Williamson, Liu et al., Schäfer & Turek, Sohankar et al. |
| 3D flow | sphere Re 100, half vs full domain, sphere Re 300 (stretch) | Clift et al., Johnson & Patel |
| heat transfer | square cavity Ra 10³–10⁶, heated cylinder Re 20 and 40, Rayleigh–Bénard onset, cubic cavity Ra 10⁴, heated sphere Re 100, grid-convergence uncertainty | de Vahl Davis, Biswas & Sarkar et al., Chandrasekhar, Tric et al. / Fusegi et al., Ranz–Marshall / Whitaker / Clift, Celik et al. |
| fans and jets | actuator strip (2D) and disc (3D) in a tunnel, each as a propeller and as a turbine; plane jet (2D) and round jet (3D) from a port into a closed box, with the momentum budget of the jet core | momentum theory (Froude; Garrett & Cummins for confinement), Bickley, Schlichting |
| airfoil polars | NACA 0012 at Re 6 × 10⁶, cross-check against XFOIL 6.99 | Ladson, XFOIL |
| finite wings | lifting-line and lifting-surface theory, Weber & Brebner swept wing | Prandtl, Multhopp, Falkner, Weber & Brebner |

## Files

| file | what it is |
|---|---|
| `run_all.py` | the runner: case registry, checks, bands, references, JSON records, the generated documentation and figures |
| `ns2d_cases.py`, `ns3d_cases.py` | 2D and 3D flow cases and their reference data (also used by the slow tests) |
| `thermal2d_cases.py`, `thermal3d_cases.py` | heat-transfer, fan, jet and emitter cases; their docstrings name the source of every reference number |
| `airfoil_validation.py`, `wing_validation.py` | polar and finite-wing validation with their reference tables |
| `bench_ns2d.py`, `bench_ns3d.py` | the detailed studies behind the theory documents (grid studies, performance; `bench_ns2d.py physics-perf` and `bench_ns3d.py physics` measure what each placed-physics feature costs, `bench_ns3d.py thermal --parts jet` runs the round-jet study with its grid and confinement variants) |
| `make_validation_figures.py` | the hand-picked figures of docs/validation.md sections 1–8 |

## Adding a case

1. Write the case function in `run_all.py` (or in a case module): it receives a
   `Context` (`ctx.device`, `ctx.quick`, `ctx.module("name")` to import a case module next to
   this file) and returns a list of `Check`, or `(checks, data)` where `data` is JSON-able
   and kept in the record for figures.
2. Give each `Check` its band (`band(ref, rel)`, `widen(lo, hi, rel)` or explicit numbers), the
   reference as published (`ref`, `ref_text`), the tolerance in words and a `source` key.
3. Add the source to `REFERENCES` (authors, year, title, venue) and the case to `CASES` with
   its area, its sources, whether it belongs to `--quick` and its measured cost.
4. Raise `Unavailable("why")` when the installation lacks a feature, rather than failing.
5. `tests/test_validation_runner.py` checks that every case cites a known reference and that
   `--quick` touches every area; run it.

A module of another area can contribute cases without editing `run_all.py`: any
`validation/*_cases.py` that defines `RUN_ALL_CASES = [Case(...), ...]` is picked up (keys must
be new).
