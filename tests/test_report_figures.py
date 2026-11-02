"""Tests for aerolume.viz.figures: every builder, both styles, headless, no pyplot."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import report_fixtures as rf
from matplotlib.figure import Figure

from aerolume.model import Body, FieldData, PolarResult
from aerolume.viz import figures as F


@pytest.fixture(scope="module")
def project():
    return rf.make_project(field_res=(120, 60))


@pytest.fixture(scope="module")
def cfd(project):
    return project.load_result("s-2412-cfd")


@pytest.fixture(scope="module")
def car(project):
    return project.load_result("s-saloon")


def _png_ok(fig: Figure) -> None:
    data = F.figure_to_png(fig, dpi=50)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_no_pyplot_qt_or_torch_imported():
    code = ("import sys; sys.path.insert(0, 'tests'); import report_fixtures as rf; "
            "from aerolume.viz import figures as F; import aerolume.report as R; "
            "pr = rf.make_project(field_res=(40, 20)); r = pr.load_result('s-2412-cfd'); "
            "F.figure_to_png(F.fig_field(r.mean_fields, 'speed', r.bodies_wind, streamlines=True)); "
            "R.build_document(pr, list(pr.simulations)); "
            "bad = [m for m in ('matplotlib.pyplot', 'PySide6', 'torch') if m in sys.modules]; "
            "print('BAD', bad); sys.exit(1 if bad else 0)")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize("style", ["print", "dark"])
def test_every_builder_renders(project, cfd, car, style):
    polars = [project.load_result("s-2412-polar"), project.load_result("s-4412-polar")]
    figs = [
        F.fig_geometry(project.designs["d-naca2412"], alpha_deg=6, style=style),
        F.fig_geometry(project.designs["d-saloon"], style=style),
        F.fig_geometry([b.points for b in car.bodies_wind], style=style),
        F.fig_cp(cfd.surface, show_cf=True, style=style),
        F.fig_cp([polars[0].points[4], polars[0].points[10]], style=style),
        F.fig_cp(car.surface, kind="vehicle", body_names=["body"], style=style),
        F.fig_history(cfd.history, cfd.t_avg_start, cfd, style=style),
        F.fig_polar(polars, labels=["a", "b"], style=style),
        F.fig_field(cfd.mean_fields, "speed", cfd.bodies_wind, streamlines=True, style=style),
        F.fig_field(car.mean_fields, "vorticity", car.bodies_wind, ground=True, style=style),
        F.fig_compare_bars([{"label": "A", "cd": cfd.cd}, {"label": "B", "cd": car.cd}], ["cd"], style=style),
        F.fig_sensitivity(cfd.surface, style=style),
    ]
    for fig in figs:
        assert isinstance(fig, Figure)
        _png_ok(fig)
    assert figs[0].get_facecolor()[:3] == pytest.approx(
        (1.0, 1.0, 1.0) if style == "print" else tuple(int(F.STYLES["dark"].bg[i:i + 2], 16) / 255 for i in (1, 3, 5)))


def test_cp_axis_inverted_and_branches(cfd):
    fig = F.fig_cp(cfd.surface)
    ax = fig.axes[0]
    assert ax.yaxis_inverted()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["upper surface", "lower surface"]
    # upper branch carries the suction peak of a lifting airfoil
    up, lo = ax.lines[0].get_ydata(), ax.lines[1].get_ydata()
    assert np.min(up) < np.min(lo)


def test_split_loop_on_rotated_airfoil():
    pts = rf.rotate(rf.naca4("2412", 61), 10.0, (0.25, 0.0))
    up, lo, t, nrm = F._split_loop(pts[:, 0], pts[:, 1])
    assert t.min() == pytest.approx(0.0, abs=1e-12) and t.max() == pytest.approx(1.0, abs=0.02)
    assert nrm[up].mean() > nrm[lo].mean()
    assert len(up) + len(lo) == len(pts) + 2      # the LE and TE belong to both branches


def test_overlay_cp_legend_is_compact(project):
    p = project.load_result("s-2412-polar")
    fig = F.fig_cp([p.points[2], p.points[6], p.points[10]], labels=["a", "b", "c"])
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[:3] == ["a", "b", "c"]
    assert "upper surface" in labels and "lower surface" in labels
    assert len(labels) <= 6


def test_history_shows_window_and_mean(cfd):
    fig = F.fig_history(cfd.history, cfd.t_avg_start, cfd)
    assert len(fig.axes) == 3
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any(t.startswith("mean 0.025") and "±" in t for t in texts)
    # the y-range frames the averaging window, not the impulsive-start spike
    _, hi = fig.axes[0].get_ylim()
    assert hi < 0.1


def test_history_empty():
    fig = F.fig_history({"t": np.zeros(0)})
    _png_ok(fig)


def test_polar_handles_no_converged_points():
    p = PolarResult(sim_id="x", re=1e6)
    fig = F.fig_polar(p)
    assert len(fig.axes) == 4
    _png_ok(fig)


def test_field_units_and_derived_quantities(cfd):
    fd = cfd.mean_fields
    fig = F.fig_field(fd, "speed", cfd.bodies_wind)
    cbar_label = fig.axes[-1].get_ylabel()
    assert "[m/s]" in cbar_label
    only_uv = FieldData(fd.extent, {"u": fd.fields["u"], "v": fd.fields["v"], "p": fd.fields["p"]}, fd.mask)
    fig = F.fig_field(only_uv, "vorticity", cfd.bodies_wind)
    assert "[1/s]" in fig.axes[-1].get_ylabel()
    fig = F.fig_field(only_uv, "cp", cfd.bodies_wind, q_inf=980.0)
    assert "Cp" in fig.axes[-1].get_ylabel()
    fig = F.fig_field(only_uv, "nut")            # missing and not derivable: a note, not an exception
    assert any("not available" in t.get_text() for t in fig.axes[0].texts)


def test_field_view_frames_body(cfd):
    fig = F.fig_field(cfd.mean_fields, "speed", cfd.bodies_wind)
    x0, x1 = fig.axes[0].get_xlim()
    ext = cfd.mean_fields.extent
    assert x0 >= ext[0] - 1e-9 and x1 <= ext[1] + 1e-9
    assert (x1 - x0) < (ext[1] - ext[0])        # auto view zooms on the body and near wake
    w, h = fig.get_size_inches()
    assert 2.0 <= h <= 6.0 and w == pytest.approx(6.7)


def test_compare_bars_delta_labels(cfd, car):
    rows = [{"label": "base", "cd": 0.5, "cd_err": 0.01}, {"label": "new", "cd": 0.45, "cd_err": 0.01}]
    fig = F.fig_compare_bars(rows, "cd")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("(-10.0 %)" in t for t in texts)
    fig = F.fig_compare_bars([{"name": "a", "cl": cfd.cl}, {"name": "b", "cl": float("nan")}], ["cl"])
    assert any("n/a" in t.get_text() for t in fig.axes[0].texts)


def test_sensitivity_with_separate_arrays():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    circle = rf.circle(0.5, 0.5, 0.5, 64)
    sens = np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False))
    fig = F.fig_sensitivity([Body("c", circle), Body("sq", sq + 2)], [sens, None], objective="cl")
    _png_ok(fig)
    assert "Cl" in fig.axes[0].get_title(loc="left")
    assert any("increase" in fig.texts[i].get_text() for i in range(len(fig.texts)))


def test_save_figure(tmp_path, cfd):
    out = F.save_figure(F.fig_history(cfd.history, cfd.t_avg_start, cfd), tmp_path / "h.png", dpi=60)
    assert out.read_bytes()[:4] == b"\x89PNG"
    out = F.save_figure(F.fig_polar(rf.FakeProject("x").results.get("none") or PolarResult("x", 1e6)),
                        tmp_path / "p.svg")
    assert out.stat().st_size > 1000


def test_unknown_style():
    with pytest.raises(ValueError):
        F.get_style("neon")
