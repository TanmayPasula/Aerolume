"""Generate the sample geometry files in this folder.

Every file is produced programmatically, so the samples are reproducible and
carry no third-party copyright:

* ``naca2412_selig.dat``      NACA 2412, Selig format (upper TE -> LE -> lower TE)
* ``naca23012_lednicer.dat``  NACA 23012, Lednicer format (point counts, two surfaces)
* ``car_profile.svg``         a car side view drawn with absolute + relative path commands,
                              cubic / smooth / quadratic curves, arcs and a group transform
* ``car_profile.dxf``         a hatchback outline split into LWPOLYLINE, SPLINE, LINE chains
                              (shuffled, some reversed) and POLYLINE pieces, plus wheels as
                              CIRCLEs and a bulge-arc LWPOLYLINE on a separate layer
* ``car_silhouette.png``      an anti-aliased black silhouette (SUV) on white
* ``two_element_wing.csv``    main element + flap as x,y blocks separated by a blank line

Run: ``python examples/geometry/make_examples.py``
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def make_dat() -> None:
    from aerolume.geometry import format_airfoil_dat, naca
    (HERE / "naca2412_selig.dat").write_text(format_airfoil_dat("NACA 2412", naca("2412", 120), "selig"))
    (HERE / "naca23012_lednicer.dat").write_text(
        format_airfoil_dat("NACA 23012", naca("23012", 120), "lednicer"))


def make_svg() -> None:
    # Drawn in a 1000 x 400 px canvas (y down), car facing left. The body path mixes
    # absolute and relative commands; the group transform scales and shifts it.
    body = (
        "M 60,300 "                                  # underbody start, behind the nose
        "L 760,300 "                                 # flat floor
        "l 140,-30 "                                 # diffuser (relative)
        "C 930,262 940,230 935,200 "                 # rear bumper
        "S 925,160 905,150 "                         # smooth continuation to the boot lid edge
        "L 800,145 "                                 # boot lid
        "Q 700,70 610,62 "                           # rear window (quadratic)
        "T 470,58 "                                  # roof (smooth quadratic)
        "c -60,4 -110,40 -150,82 "                   # windscreen (relative cubic)
        "L 120,160 "                                 # bonnet
        "a 70,70 0 0 0 -70,70 "                      # nose radius (relative arc)
        "A 80,70 0 0 0 60,300 "                      # lower bumper (absolute arc)
        "Z")
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="250mm" height="100mm" viewBox="0 0 1000 400">
  <title>Aerolume sample: car side profile</title>
  <defs><linearGradient id="g"><stop offset="0" stop-color="#888"/></linearGradient></defs>
  <g transform="translate(20 10) scale(0.96)">
    <g transform="rotate(0 500 200)">
      <path d="{body}" fill="#2a2a2a" stroke="none"/>
    </g>
    <circle cx="210" cy="300" r="62" fill="#111"/>
    <circle cx="790" cy="300" r="62" fill="#111"/>
  </g>
  <text x="10" y="390" font-size="12">not geometry: ignored</text>
</svg>
"""
    (HERE / "car_profile.svg").write_text(svg, encoding="utf-8")


def make_dxf() -> None:
    import ezdxf

    from aerolume.geometry import templates
    d = templates.make("hatchback")
    p = d.bodies[0].points * 1000.0               # millimetres, like most CAD
    n = len(p)
    loop = np.vstack([p, p[:1]])
    cuts = [0, n // 4, n // 2, 3 * n // 4, n]
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4                    # millimetres
    msp = doc.modelspace()
    pieces = []
    a = loop[cuts[0]:cuts[1] + 1]
    pieces.append(lambda: msp.add_lwpolyline([tuple(q) for q in a]))
    b = loop[cuts[1]:cuts[2] + 1]
    fit = b[::3] if (len(b) - 1) % 3 == 0 else np.vstack([b[::3], b[-1:]])   # end point exactly once
    pieces.append(lambda: msp.add_spline(fit_points=[(x, y, 0.0) for x, y in fit]))
    c = loop[cuts[2]:cuts[3] + 1][::-1]            # reversed direction on purpose
    def lines():
        for q0, q1 in itertools.pairwise(c):
            msp.add_line(tuple(q0), tuple(q1))
    pieces.append(lines)
    e = loop[cuts[3]:cuts[4] + 1]
    pieces.append(lambda: msp.add_polyline2d([tuple(q) for q in e]))
    rng = np.random.default_rng(3)
    for k in rng.permutation(len(pieces)):          # entity order scrambled
        pieces[k]()
    for xc in (0.17 * 4100, 0.80 * 4100):
        msp.add_circle((xc, 310.0), 310.0, dxfattribs={"layer": "WHEELS"})
    # a stadium (two semicircular bulges) on its own layer: a separate small body
    msp.add_lwpolyline([(1500, 1800, 0, 0, 1.0), (1700, 1800, 0, 0, 1.0)], format="xyseb", close=True,
                       dxfattribs={"layer": "EXTRAS"})
    doc.saveas(HERE / "car_profile.dxf")


def make_png() -> None:
    from PIL import Image, ImageDraw

    from aerolume.geometry import templates
    d = templates.make("suv")
    p = d.bodies[0].points
    W, H, ss = 1200, 500, 4                         # supersample for anti-aliasing
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    s = (W * 0.85) / (x1 - x0)
    px = (p[:, 0] - x0) * s + W * 0.075
    py = H * 0.85 - p[:, 1] * s                     # road at 85 % of the height
    img = Image.new("L", (W * ss, H * ss), 255)
    ImageDraw.Draw(img).polygon([(float(a * ss), float(b * ss)) for a, b in zip(px, py)], fill=0)
    img.resize((W, H), Image.LANCZOS).save(HERE / "car_silhouette.png")


def make_csv() -> None:
    from aerolume.geometry import naca, transform
    main = naca("4412", 80, 0.3)
    flap = transform(naca("4412", 60, 0.12), rotate_deg=-20.0, translate=(0.29, -0.02))
    lines = ["x,y"]
    lines += [f"{x:.6f},{y:.6f}" for x, y in main]
    lines += [""]
    lines += [f"{x:.6f},{y:.6f}" for x, y in flap]
    (HERE / "two_element_wing.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    make_dat()
    make_svg()
    make_dxf()
    make_png()
    make_csv()
    print("wrote samples to", HERE)


if __name__ == "__main__":
    main()
