"""Render web versions of the airfoil validation figures for the documentation pages.

``validation/airfoil_validation.py`` draws ``docs/img/airfoil_*.png`` for the repository:
one wide multi-panel PNG per figure on a white ground. On the site those PNGs are fitted
into a 46 rem column, which shrinks their legends to 6-7 px, and they glare on the dark
theme. This script recomputes the same cases with the same functions (so the curves are
the validated ones, not a copy) and draws every panel as its own small figure:

* one panel per file, designed at 360 x 300 CSS px with 13 px text, so two panels sit side
  by side in the docs column at their natural size and a phone shows one panel per row at
  about the same size -- the text stays readable in both;
* a light and a dark variant per panel, drawn on the site's --sheet colour with the
  site's ink and rule colours, and the application's plot series colours
  (docs/ui_guidelines.md section 2: blue #2463c4, orange #d9730d, ...);
* Reynolds numbers as ``Re = 6×10⁶``, the notation the text uses;
* saved at 2.5x for high-density screens.

The output goes to ``website/assets/docfig/`` with ``docfig.json``, which build.py reads:
a lone ``![...](img/airfoil_0012.png)`` in the Markdown becomes the panel grid, and each
panel links to the full-resolution repository PNG. Files missing from the JSON keep the
plain image, so the docs never depend on this script having run.

The polars take about three minutes on one CPU process (measured: 170 s); the computed curves are cached in
``--cache`` so ``--render-only`` redraws in seconds.

Usage::

    python website/tools/make_doc_figures.py [--cache DIR] [--render-only]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEBSITE = HERE.parent
REPO = WEBSITE.parent
OUT = WEBSITE / "assets" / "docfig"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "aerolume-website" / "docfig"

#: Design size of one panel in CSS pixels, and the pixel density it is saved at.
PANEL_W, PANEL_H, DENSITY = 360, 300, 2.5
FONT_PT = 9.5                     # at 100 px per inch: 13.2 CSS px

#: Theme colours, copied from website/assets/site.css (--sheet, --ink, --ink-2, --ink-3,
#: --rule) and the application's plot series (docs/ui_guidelines.md section 2). The dark
#: blue is the app's dark-theme accent: #2463c4 is too dim on a graphite ground.
THEMES = {
    "light": {"bg": "#fbfcfd", "ink": "#151a22", "ink2": "#414957", "ink3": "#5d6571", "rule": "#dde1e7",
              "blue": "#2463c4", "orange": "#d9730d", "green": "#2f855a", "exp": "#151a22"},
    "dark": {"bg": "#1a1d21", "ink": "#e4e6ea", "ink2": "#b6bbc4", "ink3": "#8b919b", "rule": "#2d3137",
             "blue": "#5b93ee", "orange": "#e98a2e", "green": "#57b37c", "exp": "#e4e6ea"},
}
FONTS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def _validation_module():
    """Import validation/airfoil_validation.py (a script folder, not a package)."""
    path = REPO / "validation" / "airfoil_validation.py"
    spec = importlib.util.spec_from_file_location("airfoil_validation", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["airfoil_validation"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- data
def compute(cache: Path) -> Path:
    """Run every case of the airfoil validation and store the curves the panels draw."""
    av = _validation_module()
    from aerolume.geometry.airfoils import naca4
    from aerolume.solvers.airfoil import analyze

    arrays: dict[str, np.ndarray] = {}

    def keep(prefix: str, case) -> None:
        for key in ("alpha", "cl", "cd", "cm"):
            arrays[f"{prefix}_{key}"] = np.asarray(case.table[key], float)
        print(f"  {case.name}: {case.seconds:.1f} s, {case.n_failed()} unconverged")

    keep("c6t", av.case_0012_re6e6_tripped())
    keep("c3t", av.case_0012_re3e6_tripped())
    keep("c6f", av.case_0012_re6e6_free())
    for code in ("2412", "4412"):
        keep(f"n{code}", av.case_cambered(code))
    p = analyze(naca4("2412", 200), 1e6, 4.0)
    for key in ("x", "cp", "cp_inviscid", "cf"):
        arrays[f"cp_{key}"] = np.asarray(getattr(p, key), float)
    arrays["cp_scalars"] = np.array([p.cl, p.cd, p.xtr_upper, p.xtr_lower], float)
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / "airfoil_curves.npz"
    np.savez_compressed(out, **arrays)
    return out


# --------------------------------------------------------------------------- drawing
def _figure(theme: dict):
    """A 360 x 300 px panel in the theme's colours (Figure + Agg canvas; never pyplot)."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONTS, "font.size": FONT_PT,
        "mathtext.fontset": "custom", "mathtext.rm": FONTS[0], "mathtext.it": f"{FONTS[0]}:italic",
        "mathtext.bf": f"{FONTS[0]}:bold", "axes.unicode_minus": True,
        "text.color": theme["ink"], "axes.titlecolor": theme["ink"], "axes.labelcolor": theme["ink2"],
    })
    fig = Figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100, layout="constrained", facecolor=theme["bg"])
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_facecolor(theme["bg"])
    ax.grid(True, color=theme["rule"], linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["ink3"])
    ax.tick_params(colors=theme["ink2"], labelsize=FONT_PT - 0.5, length=3)
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])
    ax.title.set_color(theme["ink"])
    return fig, ax


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=FONT_PT + 0.5, fontweight="semibold", pad=8)


def _legend(fig, ax, theme: dict, ncol: int = 1) -> None:
    """Legend under the axes, where it never covers a curve."""
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=ncol, frameon=False,
                    fontsize=FONT_PT - 0.5, handlelength=1.8, borderaxespad=0.0, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(theme["ink2"])


def _sorted(d: dict, prefix: str) -> dict[str, np.ndarray]:
    i = np.argsort(d[f"{prefix}_alpha"])
    return {k: d[f"{prefix}_{k}"][i] for k in ("alpha", "cl", "cd", "cm")}


def _prestall(t: dict[str, np.ndarray]) -> slice:
    return slice(0, int(np.argmax(t["cl"])) + 1)


def panels(d: dict, av) -> dict[str, list[tuple[str, str, int, callable]]]:
    """Panel builders: {repository PNG: [(panel name, alt text, height, draw(fig, ax, theme))]}."""
    c6t, c3t, c6f = _sorted(d, "c6t"), _sorted(d, "c3t"), _sorted(d, "c6f")
    cam = {code: _sorted(d, f"n{code}") for code in ("2412", "4412")}
    lad, gre = av.LADSON_0012_RE6E6, av.GREGORY_0012_RE3E6
    abd = av.ABBOTT_0012_RE6E6_CD
    ref = av.CAMBERED_REF

    def lift_0012(fig, ax, th):
        _title(ax, "NACA 0012 lift, tripped")
        ax.plot(c6t["alpha"], c6t["cl"], "-", color=th["blue"], lw=2, label="Aerolume, Re = 6×10⁶")
        ax.plot(c3t["alpha"], c3t["cl"], "--", color=th["blue"], lw=1.4, label="Aerolume, Re = 3×10⁶")
        ax.plot(lad[:, 0], lad[:, 1], "o", color=th["exp"], ms=4.5, label="Ladson TM-4074, Re = 6×10⁶")
        ax.plot(gre[:, 0], gre[:, 1], "s", mfc="none", color=th["orange"], ms=5, mew=1.3,
                label="Gregory & O'Reilly, Re = 3×10⁶")
        ax.set_xlabel("α [deg]")
        ax.set_ylabel("$c_l$")
        _legend(fig, ax, th)

    def polar_0012(fig, ax, th):
        _title(ax, "NACA 0012 drag polar, Re = 6×10⁶")
        ax.plot(c6t["cd"] * 1e4, c6t["cl"], "-", color=th["blue"], lw=2, label="Aerolume, tripped at x/c = 0.05")
        ax.plot(c6f["cd"] * 1e4, c6f["cl"], ":", color=th["blue"], lw=2,
                label="Aerolume, free transition, $N_{crit}$ = 9")
        pre = lad[:-1]
        ax.plot(pre[:, 2] * 1e4, pre[:, 1], "o", color=th["exp"], ms=4.5, label="Ladson, tripped")
        ax.plot(abd[:, 1] * 1e4, abd[:, 0], "^", mfc="none", color=th["orange"], ms=5.5, mew=1.3,
                label="Abbott & von Doenhoff, smooth")
        ax.set_xlabel("$c_d$ [counts, 10⁻⁴]")
        ax.set_ylabel("$c_l$")
        ax.set_xlim(0, 300)
        _legend(fig, ax, th)

    colors = {"2412": "blue", "4412": "orange"}

    def lift_cam(fig, ax, th):
        _title(ax, "Lift, Re = 6×10⁶, free transition")
        for code, t in cam.items():
            col = th[colors[code]]
            ax.plot(t["alpha"], t["cl"], "-", color=col, lw=2, label=f"Aerolume, NACA {code}")
            ax.plot([ref[code]["alpha_l0"], 0.0], [0.0, ref[code]["cl0"]], "o", color=col, mfc="none", ms=6,
                    mew=1.3, label=f"A&vD {code}: $α_{{L0}}$ and $c_l$(0)")
            ax.axhline(ref[code]["clmax"], color=col, lw=0.9, ls=":")
        ax.set_xlabel("α [deg]")
        ax.set_ylabel("$c_l$")
        ax.text(0.99, 0.02, "dotted: A&vD $c_{l,max}$", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FONT_PT - 1.5, color=th["ink3"])
        _legend(fig, ax, th)

    def polar_cam(fig, ax, th):
        _title(ax, "Drag polar (predicted), Re = 6×10⁶")
        for code, t in cam.items():
            s = _prestall(t)
            ax.plot(t["cd"][s] * 1e4, t["cl"][s], "-", color=th[colors[code]], lw=2, label=f"NACA {code}")
        ax.set_xlabel("$c_d$ [counts, 10⁻⁴]")
        ax.set_ylabel("$c_l$")
        ax.set_xlim(0, 300)
        _legend(fig, ax, th, ncol=2)

    def moment_cam(fig, ax, th):
        _title(ax, "Pitching moment about c/4")
        for code, t in cam.items():
            col = th[colors[code]]
            ax.plot(t["alpha"], t["cm"], "-", color=col, lw=2, label=f"Aerolume, NACA {code}")
            ax.axhline(ref[code]["cm"], color=col, lw=0.9, ls=":", label=f"A&vD {code} (plot reading)")
        ax.set_xlabel("α [deg]")
        ax.set_ylabel("$c_{m,c/4}$")
        ax.set_ylim(-0.16, 0.02)
        _legend(fig, ax, th, ncol=2)

    x, cp, cpi, cf = d["cp_x"], d["cp_cp"], d["cp_cp_inviscid"], d["cp_cf"]
    cl, cd, xtu, xtl = (float(v) for v in d["cp_scalars"])
    ile = int(np.argmin(x))

    def cp_2412(fig, ax, th):
        _title(ax, "NACA 2412, α = 4°, Re = 10⁶")
        ax.plot(x, cpi, "--", color=th["ink2"], lw=1.2, label="inviscid (panel method)")
        ax.plot(x, cp, "-", color=th["blue"], lw=2, label=f"viscous (coupled): $c_l$ = {cl:.3f}, $c_d$ = {cd:.5f}")
        ax.invert_yaxis()
        ax.set_xlabel("x/c")
        ax.set_ylabel("$C_p$")
        _legend(fig, ax, th)

    def cf_2412(fig, ax, th):
        _title(ax, "Skin friction, NACA 2412, α = 4°")
        ax.plot(x[:ile + 1], cf[:ile + 1], "-", color=th["blue"], lw=2, label="upper surface")
        ax.plot(x[ile:], cf[ile:], "-", color=th["orange"], lw=1.6, label="lower surface")
        for xt, key in ((xtu, "blue"), (xtl, "orange")):
            if xt < 1:
                ax.axvline(xt, color=th[key], ls=":", lw=1)
        ax.set_xlabel("x/c")
        ax.set_ylabel("$c_f$")
        ax.text(0.99, 0.97, "dotted: transition", transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_PT - 1.5, color=th["ink3"])
        _legend(fig, ax, th, ncol=2)

    return {
        "airfoil_0012.png": [
            ("lift", ("NACA 0012 lift coefficient against angle of attack: Aerolume at Re 6 and 3 million, "
                      "tripped, with Ladson's and Gregory and O'Reilly's measurements."), 390, lift_0012),
            ("polar", ("NACA 0012 drag polar at Re 6 million: Aerolume tripped and free-transition curves with "
                       "Ladson's tripped data and Abbott and von Doenhoff's smooth-model data."), 390, polar_0012),
        ],
        "airfoil_cambered.png": [
            ("lift", ("Lift of NACA 2412 and 4412 against angle of attack, with the zero-lift angles, lift at "
                      "zero angle and maximum lift read from Abbott and von Doenhoff."), 390, lift_cam),
            ("polar", "Predicted drag polars of NACA 2412 and 4412 at Re 6 million.", 330, polar_cam),
            ("moment", ("Quarter-chord pitching moment of NACA 2412 and 4412 against angle of attack, with the "
                        "book values as dotted lines."), 350, moment_cam),
        ],
        "airfoil_cp_2412.png": [
            ("cp", ("Pressure coefficient along NACA 2412 at 4 degrees and Re 1 million: viscous solution "
                    "against the inviscid panel solution."), 350, cp_2412),
            ("cf", ("Skin friction on the upper and lower surface of the same case, with the transition "
                    "points dotted."), 350, cf_2412),
        ],
    }


def _save(fig, path: Path, bg: str) -> None:
    """PNG at DENSITY x, reduced to a 96-colour palette.

    A line plot has a few flat colours plus their anti-aliasing ramps, which 96 colours
    hold without visible banding; the files shrink to about a third (1 MB -> 0.35 MB for
    all fourteen panels).
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100 * DENSITY, facecolor=bg)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


def render(cache: Path) -> dict:
    """Draw every panel in both themes into website/assets/docfig/ and return the index."""
    av = _validation_module()
    d = dict(np.load(cache / "airfoil_curves.npz"))
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for source, items in panels(d, av).items():
        stem = Path(source).stem
        entry = []
        for name, alt, height, draw in items:
            files = {}
            for theme_name, theme in THEMES.items():
                fig, ax = _figure(theme)
                fig.set_size_inches(PANEL_W / 100, height / 100)
                draw(fig, ax, theme)
                fname = f"{stem}-{name}-{theme_name}.png"
                _save(fig, OUT / fname, theme["bg"])
                files[theme_name] = fname
            entry.append({"name": name, "alt": alt, "width": PANEL_W, "height": height, **files})
            print(f"  {stem}-{name}: {PANEL_W}x{height} CSS px")
        index[source] = {"panels": entry}
    (OUT / "docfig.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # the cache holds solver fields, never committed or published: by default a folder under the
    # system temp directory, so no checkout depends on (or writes into) one person's folder layout
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help=f"folder for the computed data (default {DEFAULT_CACHE})")
    ap.add_argument("--render-only", action="store_true", help="redraw from the cached curves")
    args = ap.parse_args()
    if not args.render_only:
        compute(args.cache)
    render(args.cache)


if __name__ == "__main__":
    main()
