# Contributing to Aerolume

Thanks for helping. Aerolume aims to be the open-source aerodynamics tool that an
engineer trusts and a newcomer can use on day one, so contributions are judged on
both: is it correct, and is it easy?

## Getting set up

```bash
git clone <your fork>
cd Aerolume
python -m venv .venv            # keep it outside cloud-synced folders if you can
# GPU (NVIDIA): pick the CUDA build matching your driver from https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu130
# or CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q                       # fast suite, CPU
pytest -q -m slow               # validation-grade runs (minutes, GPU recommended)
aerolume                        # launch the desktop app
```

## Ground rules

* **Read `DESIGN.md` first.** It is the contract between modules (data model, units,
  sign conventions, public signatures). A change to a public signature updates
  `DESIGN.md` in the same pull request.
* **Numbers need evidence.** A change to the solver, the airfoil analysis or any
  post-processing that moves a coefficient must say by how much, on which validation
  case, and why the new value is closer to the reference. `validation/` holds the
  benchmark scripts; `docs/validation*.md` holds the results.
* **SI everywhere inside.** Unit conversion happens at the edges (`aerolume.core.units`).
* **The UI never blocks.** Anything slower than ~50 ms runs on a worker thread.
* **Import boundaries.** `model`, `core` and `analysis` depend only on numpy;
  torch stays in the solver/geometry-SDF/optimise/viz-render layers; Qt stays under
  `app/`; matplotlib is used through `matplotlib.figure.Figure`, never `pyplot`.
* Tests: `@pytest.mark.slow` for anything over ~20 s, `@pytest.mark.gpu` for CUDA-only
  tests, `@pytest.mark.gui` for Qt tests (they run offscreen).
* Style: `ruff check src tests`. Docstrings explain *why*, not just *what*.

## Generated material

Several files in the repository are produced by scripts; regenerate them, do not edit them,
when the code they depend on changes (each needs a CUDA GPU to finish in minutes):

| what | script |
|---|---|
| example projects (`src/aerolume/examples/projects/`) | `python examples/make_example_projects.py` |
| CFD validation numbers (docs/theory_ns2d.md, docs/validation.md) | `python validation/bench_ns2d.py all` (or one case) |
| validation figures (`docs/img/validation_*.png`) | `python validation/make_validation_figures.py` |
| airfoil validation tables and figures | `python validation/airfoil_validation.py` |
| application screenshots (website and user guide) | `python website/tools/make_screens.py --work <scratch folder>` |
| the website | `python website/build.py` |

Write scratch output (temporary projects, logs, screenshots) outside the repository.

## Release checklist

In this order, once nobody is editing the tree (each later step depends on the earlier ones):

1. `pytest -q` and `pytest -q -m slow` green; `ruff check src tests validation examples website packaging`.
2. `python validation/run_all.py --check-docs`: exit 1 means docs/validation.md section 9 was
   measured on other solver, geometry, model or validation code. Then run
   `python validation/run_all.py --all --write-docs` (about half an hour on a GPU) and carry any
   changed number into README.md and CHANGELOG.md.
3. `python examples/make_example_projects.py` when the solver, the planner or the result format
   changed; then `python website/tools/make_screens.py --work <scratch folder>` and
   `python website/build.py --strict` (tests/test_report_website_release.py fails until both ran).
4. `python packaging/build.py --variant cuda` (full build, smoke test, installer), then
   `python packaging/build.py --write-measurements` (fills docs/install.md and the CHANGELOG's
   build summary from the measured run) and `python packaging/build.py --refresh-docs`.
   `python packaging/build.py --status` must say CURRENT before the installer is published,
   together with the `-release-notes.md` file beside it (it carries the installer's SHA-256).

## Licensing

Aerolume is Apache-2.0. By contributing you agree your contribution is licensed under
it. Third-party code must be compatible (MIT/BSD/Apache-2.0) and listed in
`THIRD_PARTY_NOTICES.md`. Qt is used through PySide6 (LGPL) — please do not add PyQt.
