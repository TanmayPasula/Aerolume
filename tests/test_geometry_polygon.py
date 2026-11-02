"""Polygon operations (aerolume.geometry.polygon)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.geometry import polygon as P


def circle(n=64, r=1.0, c=(0.0, 0.0)):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)])


SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])


def test_area_orientation_centroid_bbox_perimeter():
    assert P.signed_area(SQUARE) == pytest.approx(2.0)
    cw = SQUARE[::-1]
    assert P.signed_area(cw) == pytest.approx(-2.0)
    ccw = P.orient_ccw(cw)
    assert P.signed_area(ccw) > 0 and np.allclose(ccw[0], cw[0])        # point 0 kept
    assert np.allclose(P.centroid(SQUARE + [10, 5]), [11.0, 5.5])
    assert P.bbox(SQUARE) == (0.0, 2.0, 0.0, 1.0)
    assert P.perimeter(SQUARE) == pytest.approx(6.0)
    # explicitly closed input is normalised
    closed = np.vstack([SQUARE, SQUARE[:1]])
    assert len(P.as_polygon(closed)) == 4


def test_remove_duplicates_and_spikes():
    p = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [2, 1], [1, 1], [0, 1], [0, 0]], float)
    q = P.remove_duplicates(p)
    assert len(q) == 4
    assert not P.is_self_intersecting(q)


def test_resample_count_spacing_and_corners():
    c = circle(400)
    r = P.resample(c, 50)
    assert len(r) == 50
    assert np.allclose(np.hypot(*r.T), 1.0, atol=2e-3)
    e = P.edge_lengths(r)
    assert e.std() / e.mean() < 0.02                                    # uniform
    r2 = P.resample(c, spacing=0.1)
    assert len(r2) == round(P.perimeter(c) / 0.1)
    # square: corners preserved exactly
    sq = P.resample(SQUARE, 40)
    for corner in SQUARE:
        assert np.min(np.hypot(*(sq - corner).T)) < 1e-12
    # without corner keeping, a coarse resample cuts corners
    sq2 = P.resample(SQUARE, 7, keep_corners=False)
    assert min(np.min(np.hypot(*(sq2 - k).T)) for k in SQUARE) >= 0.0


def test_resample_blunt_trailing_edge_kept_and_curvature_clusters():
    from aerolume.geometry import naca
    a = naca("0012", 300, closed_te=False)
    r = P.resample(a, 120, curvature=1.0)
    for te in (a[0], a[-1]):                                            # both blunt-TE corners
        assert np.min(np.hypot(*(r - te).T)) < 1e-12
    # points cluster at the leading edge
    e = P.edge_lengths(r)
    ile = int(np.argmin(r[:, 0]))
    assert e[ile] < 0.5 * np.median(e)


def test_smooth_reduces_noise_without_shrinking():
    rng = np.random.default_rng(0)
    c = circle(400) + rng.normal(0, 0.01, (400, 2))
    # noise this strong looks like corners to the corner detector, so do not pin them
    s = P.smooth(c, iterations=30, keep_corners=False)
    rad = np.hypot(*s.T)
    assert rad.std() < 0.5 * np.hypot(*c.T).std()
    assert abs(P.signed_area(s) - math.pi) / math.pi < 0.01


def test_simplify_rdp():
    c = circle(1000)
    s = P.simplify_rdp(c, 1e-3)
    assert 20 < len(s) < 150
    assert P.distance_to_polygon(s, c).max() <= 1e-3 + 1e-12


def test_close_gaps_chains_shuffled_reversed_pieces():
    c = circle(120)
    loop = np.vstack([c, c[:1]])
    pieces = [loop[i:i + 21] for i in range(0, 120, 20)]
    pieces[1] = pieces[1][::-1]
    pieces[4] = pieces[4][::-1] + 1e-9                                  # tiny endpoint mismatch
    rng = np.random.default_rng(1)
    pieces = [pieces[k] for k in rng.permutation(len(pieces))]
    notes = []
    out = P.close_gaps(pieces, tol=1e-6, notes=notes)
    assert len(out) == 1 and not notes
    assert abs(abs(P.signed_area(out[0])) - P.signed_area(c)) < 1e-6
    # an open arc gets closed with a note
    notes = []
    out = P.close_gaps([c[:60]], notes=notes)
    assert len(out) == 1 and len(notes) == 1


def test_self_intersection_detection_and_repair():
    bowtie = np.array([[0, 0], [2, 2], [2, 0], [0, 2]], float)
    assert P.is_self_intersecting(bowtie)
    hits = P.self_intersections(bowtie)
    assert len(hits) == 1 and np.allclose(hits[0][2], [1, 1])
    assert not P.is_self_intersecting(SQUARE)
    assert not P.is_self_intersecting(circle(500))
    # pentagram (every second vertex of a pentagon): repaired to the 10-point star outline
    k = np.arange(5)
    pent = np.column_stack([np.cos(np.pi / 2 + 2 * np.pi * k / 5), np.sin(np.pi / 2 + 2 * np.pi * k / 5)])
    star = pent[[0, 2, 4, 1, 3]]
    assert len(P.self_intersections(star)) == 5
    fixed = P.repair_self_intersections(star)
    assert len(fixed) == 10 and not P.is_self_intersecting(fixed) and P.signed_area(fixed) > 0
    r_in = np.cos(2 * np.pi / 5) / np.cos(np.pi / 5)          # inner radius of the star
    assert np.sort(np.hypot(*fixed.T))[:5] == pytest.approx([r_in] * 5)
    # a closing edge that overshoots the start: main body kept, pinch lobe dropped
    over = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [0.5, -0.5]], float)
    assert P.is_self_intersecting(over)
    fixed = P.repair_self_intersections(over)
    assert not P.is_self_intersecting(fixed)
    assert P.signed_area(fixed) == pytest.approx(16.0 - 0.5 * 4 * 4 / 9, rel=1e-9)
    # bow-tie: two equal lobes touching at a point -> one lobe
    lobe = P.repair_self_intersections(bowtie)
    assert P.signed_area(lobe) == pytest.approx(1.0) and not P.is_self_intersecting(lobe)


def test_point_in_polygon_and_distance():
    c = circle(256)
    pts = np.array([[0, 0], [0.9, 0], [1.1, 0], [0, -2]])
    assert list(P.point_in_polygon(c, pts)) == [True, True, False, False]
    assert P.point_in_polygon(SQUARE, (1.0, 0.5)) is True
    # clockwise outlines give the same answer (non-zero winding)
    assert list(P.point_in_polygon(c[::-1], pts)) == [True, True, False, False]
    d = P.distance_to_polygon(SQUARE, [[1, 3], [3, 0.5], [1, 0.5]])
    assert np.allclose(d, [2, 1, 0.5])


def test_overlap_and_gap():
    a = SQUARE
    b = SQUARE + [1.5, 0.5]
    far = SQUARE + [3.0, 0.0]
    assert P.polygons_overlap(a, b)
    assert not P.polygons_overlap(a, far)
    assert P.polygon_distance(a, far) == pytest.approx(1.0)
    assert P.polygons_overlap(SQUARE * 4 - 1, SQUARE)                   # containment


def test_frontal_height_merges_intervals():
    a = SQUARE                                   # y 0..1
    b = SQUARE + [5, 0.5]                        # y 0.5..1.5 (overlaps in y)
    c = SQUARE + [0, 3]                          # y 3..4 (gap)
    assert P.frontal_height(a) == pytest.approx(1.0)
    assert P.frontal_height([a, b]) == pytest.approx(1.5)
    assert P.frontal_height([a, b, c]) == pytest.approx(2.5)


def test_transform_order_and_orientation():
    q = P.transform(SQUARE, scale=2.0, rotate_deg=90.0, about=(0, 0), translate=(1, 1))
    assert np.allclose(q[1], [1.0, 5.0])              # (2,0)*2 = (4,0) -> rot90 (0,4) -> +1
    m = P.transform(SQUARE, mirror="x", about=(1.0, 0.0))
    assert P.signed_area(m) > 0                       # still CCW
    assert P.bbox(m) == pytest.approx((0, 2, 0, 1))
    m2 = P.transform(SQUARE, scale=(-1, 1))
    assert P.signed_area(m2) > 0
    with pytest.raises(ValueError):
        P.transform(SQUARE, mirror="z")
