"""Figures for docs/validation.md, drawn from the documentation's own tables.

Usage (from the repository root)::

    python validation/make_validation_figures.py            # all figures -> docs/img/validation_*.png
    python validation/make_validation_figures.py --no-tg    # skip the Taylor-Green runs (~20 s CPU)

The measured numbers live in the Markdown tables that ``bench_ns2d.py`` and the
other validation scripts print (docs/theory_ns2d.md sections 10-11,
docs/theory_sensitivity.md section 4, docs/validation.md). The figures are drawn
from those tables, so a figure can never show a different number from the text
next to it. Only the Taylor-Green convergence plot computes something: the error
at four resolutions (float64, CPU, a few seconds per grid), because the theory
document only lists the resulting orders.

Matplotlib is used through ``matplotlib.figure.Figure`` with the Agg canvas, never
pyplot (DESIGN.md section 2).
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOCS = REPO / "docs"
IMG = DOCS / "img"

#: docs/ui_guidelines.md section 2 plot series: blue, orange, green, red, purple, grey.
SERIES = ["#2463c4", "#d9730d", "#2f855a", "#b8322b", "#6b46c1", "#5f6b7a"]
INK, INK2, GRID, BAND = "#1d2129", "#454c59", "#d3d6dc", "#e6e9ee"


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

def _cells(line: str) -> list[str]:
    s = line.strip().removeprefix("|").removesuffix("|")
    return [c.strip() for c in s.split("|")]


def tables(md: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """(heading, header, rows) of every pipe table in a Markdown text, outside code fences."""
    out, heading, fence = [], "", False
    lines = md.splitlines()
    k = 0
    while k < len(lines):
        ln = lines[k]
        if ln.lstrip().startswith("```"):
            fence = not fence
        elif not fence and ln.startswith("#"):
            heading = ln.lstrip("#").strip()
        elif (not fence and "|" in ln and k + 1 < len(lines)
              and re.match(r"^\s*\|?\s*:?-{3,}", lines[k + 1])):
            header, rows = _cells(ln), []
            k += 2
            while k < len(lines) and lines[k].strip().startswith("|"):
                rows.append(_cells(lines[k]))
                k += 1
            out.append((heading, header, rows))
            continue
        k += 1
    return out


def find_table(md: str, first_header: str, heading: str = ""):
    for h, header, rows in tables(md):
        if header and header[0].lower().startswith(first_header.lower()) and heading.lower() in h.lower():
            return header, rows
    raise KeyError(f"table starting with {first_header!r} under {heading!r}")


def num(cell: str) -> float:
    """First number in a table cell ('**493**' -> 493, '1.3533 (-8.5 %)' -> 1.3533, '—' -> nan)."""
    m = re.search(r"[-−]?\d+(?:\.\d+)?(?:e[-−]?\d+)?", cell.replace("−", "-").replace(",", ""))
    return float(m.group(0).replace("−", "-")) if m else float("nan")


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------

def figure(ncols: int, width: float, height: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, ncols))
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
    return fig, axes


def save(fig, name: str) -> Path:
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / name
    fig.savefig(path, facecolor="white")
    print(f"wrote {path.relative_to(REPO)}")
    return path


def band(ax, lo: float, hi: float, label: str) -> None:
    ax.axhspan(lo, hi, color=BAND, zorder=0, label=label)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_cylinder() -> Path:
    """Grid study of the circular cylinder (docs/theory_ns2d.md section 11)."""
    md = (DOCS / "theory_ns2d.md").read_text(encoding="utf-8")
    header, rows = find_table(md, "case", "circular cylinder")
    c = {h: i for i, h in enumerate(header)}
    data = {}
    for r in rows:
        data.setdefault(r[0], []).append((num(r[c["cells/D"]]), num(r[c["Cd"]]), num(r[c["Cl′"]]),
                                          num(r[c["St"]])))
    re40 = np.array(data["Re40"])
    re100 = np.array(data["Re100"])
    fig, (a1, a2, a3) = figure(3, 11.0, 3.4)
    band(a1, 1.50, 1.55, "unconfined references")
    a1.plot(re40[:, 0], re40[:, 1], "o-", color=SERIES[0], label="Re 40")
    a1.axhspan(1.33, 1.38, color=BAND, zorder=0)
    a1.plot(re100[:, 0], re100[:, 1], "s-", color=SERIES[1], label="Re 100")
    a1.set_title("Mean drag", color=INK, fontsize=10)
    a1.set_ylabel("$C_d$", color=INK2)
    band(a2, 0.32, 0.34, "references")
    a2.plot(re100[:, 0], re100[:, 2], "s-", color=SERIES[1], label="Re 100")
    a2.set_title("Lift amplitude, Re 100", color=INK, fontsize=10)
    a2.set_ylabel("$C_l'$", color=INK2)
    band(a3, 0.164, 0.167, "references")
    a3.plot(re100[:, 0], re100[:, 3], "s-", color=SERIES[1], label="Re 100")
    a3.set_title("Strouhal number, Re 100", color=INK, fontsize=10)
    a3.set_ylabel("St", color=INK2)
    for ax in (a1, a2, a3):
        ax.set_xlabel("cells per diameter", color=INK2)
        ax.set_xticks([16, 24, 32, 48, 64])
        ax.legend(fontsize=8, frameon=True, edgecolor=GRID)
    return save(fig, "validation_cylinder.png")


def _load_cases():
    spec = importlib.util.spec_from_file_location("ns2d_cases", HERE / "ns2d_cases.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ns2d_cases"] = mod
    spec.loader.exec_module(mod)
    return mod


def fig_taylor_green() -> Path:
    """Error of the decaying Taylor-Green vortex against resolution, both advection schemes."""
    cases = _load_cases()
    ns = np.array([16, 32, 64, 128])
    from matplotlib.ticker import NullFormatter, NullLocator
    fig, (ax,) = figure(1, 5.2, 3.6)
    first = None
    for k, adv in enumerate(("central", "upwind3")):
        errs = np.array([cases.taylor_green_error(int(n), advection=adv)[0] for n in ns])
        orders = np.log2(errs[:-1] / errs[1:])
        print(f"Taylor-Green {adv}: errors {errs}, orders {np.round(orders, 3)}")
        ax.loglog(ns, errs, "o-", color=SERIES[k], label=f"{adv} (order {orders[-1]:.2f} at N = 64→128)")
        first = errs[0] if first is None else first
    ref = 0.2 * first * (ns / ns[0]) ** -2.0    # below both curves (the central one lies on slope 2)
    ax.loglog(ns, ref, "--", color=SERIES[5], linewidth=1.0, label="slope 2")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("cells per 2π", color=INK2)
    ax.set_ylabel("max |u − u_exact| at t = 1", color=INK2)
    ax.set_title("Taylor–Green vortex, Re 10, float64", color=INK, fontsize=10)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend(fontsize=8, frameon=True, edgecolor=GRID)
    return save(fig, "validation_taylor_green.png")


def fig_performance() -> Path:
    """Solver throughput (docs/theory_ns2d.md section 10)."""
    md = (DOCS / "theory_ns2d.md").read_text(encoding="utf-8")
    _, rows = find_table(md, "device", "Performance")
    labels, values, colours = [], [], []
    device = ""
    for r in rows:
        device = r[0] if r[0] else device
        gpu = "CUDA" in device or "CUDA" in r[0]
        labels.append(f"{'GPU' if gpu else 'CPU'} {r[1]}, {r[2]}")
        values.append(num(r[3]))
        colours.append(SERIES[0] if gpu else SERIES[1])
    fig, (ax,) = figure(1, 8.0, 3.8)
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, color=colours, height=0.6)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 8, yi, f"{v:.0f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("time steps per second (float32)", color=INK2)
    ax.set_title("Navier–Stokes solver throughput, RTX 4060 Ti / i7-13700F", color=INK, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.12)
    return save(fig, "validation_performance.png")


def fig_sensitivity() -> Path:
    """Autograd against central finite differences, cylinder modes (docs/theory_sensitivity.md section 4)."""
    md = (DOCS / "theory_sensitivity.md").read_text(encoding="utf-8")
    header, rows = find_table(md, "mode", "Verification against finite differences")
    by = {r[0].lower(): r[1:] for r in rows}
    ad = np.array([num(v) for v in by["autograd"]])
    fd = np.array([num(v) for v in by["central fd"]])
    labels = [h.replace(" (", "\n(") for h in header[1:]]
    fig, (ax,) = figure(1, 7.0, 3.4)
    x = np.arange(len(ad))
    ax.bar(x - 0.18, ad, 0.36, color=SERIES[0], label="autograd (reverse sweep)")
    ax.bar(x + 0.18, fd, 0.36, color=SERIES[1], label="central finite difference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("bump mode on the outline", color=INK2)
    ax.set_ylabel("dJ / d(mode amplitude)", color=INK2)
    worst = float(np.max(np.abs(ad / fd - 1)))
    ax.set_title(f"Cd sensitivity, cylinder Re 30 (largest difference {worst:.1e})", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=True, edgecolor=GRID)
    return save(fig, "validation_sensitivity.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-tg", action="store_true", help="skip the Taylor-Green figure (it runs the solver)")
    a = ap.parse_args(argv)
    fig_cylinder()
    fig_performance()
    fig_sensitivity()
    if not a.no_tg:
        fig_taylor_green()
    return 0


if __name__ == "__main__":
    sys.exit(main())
