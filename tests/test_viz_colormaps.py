"""Colour maps and per-field metadata (aerolume.viz.colormaps)."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from aerolume.viz import colormaps as C


def test_every_map_is_a_readonly_256x3_uint8_lut():
    assert set(C.CMAP_NAMES) == set(C.COLORMAPS)
    for name in ("viridis", "magma", "inferno", "plasma", "cividis", "turbo",
                 "coolwarm", "RdBu_r", "icefire", "grayscale"):
        lut = C.COLORMAPS[name]
        assert lut.shape == (256, 3) and lut.dtype == np.uint8, name
        assert not lut.flags.writeable, name


def test_matplotlib_maps_are_sampled_in_the_right_direction():
    # Published end points of viridis: dark purple -> yellow.
    v = C.COLORMAPS["viridis"].astype(int)
    assert np.abs(v[0] - (68, 1, 84)).max() <= 1
    assert np.abs(v[-1] - (253, 231, 37)).max() <= 1
    g = C.COLORMAPS["grayscale"]
    assert np.array_equal(g[:, 0], np.arange(256)) and np.array_equal(g[:, 0], g[:, 2])


@pytest.mark.parametrize("name", ["viridis", "magma", "inferno", "plasma", "cividis",
                                  "coolwarm", "RdBu_r", "turbo"])
def test_fallback_anchor_luts_match_matplotlib(name):
    """The no-matplotlib fallback must look like the real map (screen-scale accuracy)."""
    fallback = C.lut_from_anchors(C._ANCHORS[name]).astype(float)
    err = np.abs(fallback - C.COLORMAPS[name].astype(float)).mean()
    assert err < 3.0, err


def test_icefire_has_symmetric_v_shaped_lightness():
    L = C.srgb_to_lab(C.COLORMAPS["icefire"] / 255.0)[:, 0]
    assert 120 <= int(np.argmin(L)) <= 135                 # darkest at the centre (zero)
    assert np.all(np.diff(L[:120]) < 0.5)                  # monotone down ...
    assert np.all(np.diff(L[136:]) > -0.5)                 # ... and monotone up
    assert np.abs(L - L[::-1]).max() < 8.0                 # +w and -w equally strong
    assert L[0] > 80 and L[-1] > 80 and L.min() < 15


def test_lab_round_trip():
    rgb = np.random.default_rng(0).random((100, 3))
    assert np.allclose(C.lab_to_srgb(C.srgb_to_lab(rgb)), rgb, atol=1e-6)


def test_get_lut_aliases_and_errors():
    assert C.get_lut("rdbu_r") is C.COLORMAPS["RdBu_r"]
    assert np.array_equal(C.get_lut("viridis_r"), C.COLORMAPS["viridis"][::-1])
    assert C.get_lut(None) is C.COLORMAPS["viridis"]
    with pytest.raises(KeyError, match="available"):
        C.get_lut("jet-ish")


def test_lut_rgba_and_colorbar_image():
    rgba = C.lut_rgba("magma", alpha=200)
    assert rgba.shape == (256, 4) and (rgba[:, 3] == 200).all()
    bar = C.colorbar_image("viridis", length=100, thickness=10)
    assert bar.shape == (100, 10, 4)
    assert np.array_equal(bar[0, 0, :3], C.COLORMAPS["viridis"][-1])     # high value on top
    assert np.array_equal(bar[-1, 0, :3], C.COLORMAPS["viridis"][0])
    hbar = C.colorbar_image("viridis", length=100, thickness=10, vertical=False)
    assert hbar.shape == (10, 100, 4)
    assert np.array_equal(hbar[0, -1, :3], C.COLORMAPS["viridis"][-1])   # high value on the right


def test_field_info_contract():
    for name, info in C.FIELD_INFO.items():
        label, kind, diverging = info                      # DESIGN.md: a 3-tuple
        assert isinstance(label, str) and kind in C.UNIT_SYMBOL and isinstance(diverging, bool)
        assert name in C.DEFAULT_CMAP and C.DEFAULT_CMAP[name] in C.COLORMAPS
        if diverging and name != "v":
            assert C.DEFAULT_CMAP[name] in C.DIVERGING_CMAPS, name
    for name in ("speed", "u", "v", "p", "cp", "vorticity", "dye", "mask", "nut", "divergence"):
        assert name in C.FIELD_INFO
    assert C.field_info("unknown").units_kind == "dimensionless"
    assert C.default_cmap("unknown") == "viridis"


def test_nondimensional_to_si_conversion():
    U, rho, L = 30.0, 1.2, 2.0
    assert C.to_si("speed", 1.5, speed=U, density=rho, length=L) == pytest.approx(45.0)
    assert C.to_si("p", 0.5, speed=U, density=rho, length=L) == pytest.approx(0.5 * rho * U**2)
    assert C.to_si("cp", -1.2, speed=U, density=rho, length=L) == pytest.approx(-1.2)
    assert C.to_si("vorticity", 2.0, speed=U, density=rho, length=L) == pytest.approx(2.0 * U / L)
    assert C.to_si("nut", 1e-3, speed=U, density=rho, length=L) == pytest.approx(1e-3 * U * L)
    arr = np.ones(4)
    assert np.allclose(C.to_si("u", arr, speed=U, density=rho, length=L), U)
    assert C.FIELD_INFO["p"].unit == "Pa" and C.FIELD_INFO["cp"].unit == ""


def test_colormaps_import_needs_neither_torch_nor_pyplot():
    code = ("import sys, aerolume.viz; from aerolume.viz import COLORMAPS, FIELD_INFO; "
            "print('torch' in sys.modules, 'matplotlib.pyplot' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]
