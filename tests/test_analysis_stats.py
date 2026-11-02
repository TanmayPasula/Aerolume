"""Statistics on synthetic signals with known answers: Strouhal, stderr calibration, MSER."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.analysis import (
    batch_means,
    batch_means_stderr,
    coefficient_stats,
    convergence_state,
    detect_steady,
    dominant_frequency,
    integrated_autocorr_time,
    stats,
    strouhal,
    summarize_history,
    transient_end,
)


def ar1(n: int, phi: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """``reps`` independent AR(1) series of length n (unit innovations), stationary start."""
    x = np.empty((reps, n))
    x[:, 0] = rng.standard_normal(reps) / math.sqrt(1 - phi * phi)
    e = rng.standard_normal((reps, n))
    for i in range(1, n):
        x[:, i] = phi * x[:, i - 1] + e[:, i]
    return x


def shedding(t: np.ndarray, f: float, rng: np.random.Generator, *, noise=0.05, tau=8.0) -> np.ndarray:
    """Cl-like signal: decaying offset + oscillation that grows in, + white noise."""
    return (0.8 * np.exp(-t / tau) + 0.4 * np.sin(2 * np.pi * f * t) * (1 - np.exp(-t / 10.0))
            + noise * rng.standard_normal(t.size))


# --- Strouhal ---------------------------------------------------------------

@pytest.mark.parametrize("f0", [0.137, 0.165, 0.21, 0.3, 0.45])
def test_strouhal_within_one_percent(f0):
    rng = np.random.default_rng(int(f0 * 1000))
    t = np.linspace(0, 200, 20001)
    st = strouhal(t, shedding(t, f0, rng))
    assert st is not None
    assert abs(st - f0) / f0 < 0.01


def test_strouhal_scales_with_length_and_speed():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 20, 8001)          # seconds
    f = 2.5                               # Hz
    cl = np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(t.size)
    st = strouhal(t, cl, length=0.2, speed=3.0)
    assert st == pytest.approx(f * 0.2 / 3.0, rel=0.005)


def test_strouhal_nonuniform_time_step():
    rng = np.random.default_rng(4)
    dt = 0.01 * (1 + 0.5 * np.sin(np.arange(30000) / 300.0) ** 2)     # CFL-like varying step
    t = np.cumsum(dt)
    st = strouhal(t, np.sin(2 * np.pi * 0.2 * t) + 0.02 * rng.standard_normal(t.size))
    assert st == pytest.approx(0.2, rel=0.005)


def test_strouhal_none_for_non_periodic():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 200, 20001)
    assert strouhal(t, rng.standard_normal(t.size)) is None                    # broadband noise
    assert strouhal(t, 1.0 + np.exp(-t / 5.0)) is None                         # steady flow
    assert strouhal(t, np.exp(-t / 20.0) * np.sin(t)) is None                  # decaying perturbation
    assert strouhal(t, np.full(t.size, 0.3)) is None                           # constant
    assert strouhal(t[:5], np.sin(t[:5])) is None                              # too short


def test_zero_crossing_fallback_few_cycles():
    rng = np.random.default_rng(6)
    t = np.linspace(0, 17.5, 3501)        # 3.5 cycles at f = 0.2
    f = dominant_frequency(t, np.sin(2 * np.pi * 0.2 * t + 0.3) + 0.03 * rng.standard_normal(t.size),
                           t_start=0.0)
    assert f == pytest.approx(0.2, rel=0.01)


# --- standard error calibration (Monte Carlo) ---------------------------------

@pytest.mark.parametrize("phi", [0.0, 0.9, 0.97])
def test_stderr_calibrated_against_monte_carlo_ar1(phi):
    rng = np.random.default_rng(10 + int(100 * phi))
    x = ar1(8000, phi, 300, rng)
    est = np.array([batch_means_stderr(row) for row in x])
    true = x.mean(axis=1).std(ddof=1)
    assert np.mean(est) / true == pytest.approx(1.0, abs=0.3)


def test_stderr_calibrated_for_periodic_plus_noise():
    rng = np.random.default_rng(11)
    n, reps = 8000, 300
    i = np.arange(n)
    phase = rng.uniform(0, 2 * np.pi, (reps, 1))
    x = np.sin(2 * np.pi * i / 50.7 + phase) + 0.2 * ar1(n, 0.8, reps, rng)
    est = np.array([batch_means_stderr(row) for row in x])
    true = x.mean(axis=1).std(ddof=1)
    assert np.mean(est) / true == pytest.approx(1.0, abs=0.3)


def test_autocorrelation_time_of_ar1():
    rng = np.random.default_rng(12)
    phi = 0.9
    x = ar1(100_000, phi, 1, rng)[0]
    assert integrated_autocorr_time(x) == pytest.approx((1 + phi) / (1 - phi), rel=0.15)
    assert integrated_autocorr_time(rng.standard_normal(50_000)) == pytest.approx(1.0, abs=0.1)


def test_batch_means_short_series():
    assert math.isnan(batch_means_stderr(np.arange(5.0)))
    bm = batch_means(np.random.default_rng(0).standard_normal(200))
    assert bm["n_batches"] >= 10 and math.isfinite(bm["stderr"])


# --- transient detection (MSER-5) ---------------------------------------------

@pytest.mark.parametrize("tau", [5.0, 10.0, 20.0])
def test_mser_detects_mean_transient(tau):
    rng = np.random.default_rng(int(tau))
    t = np.linspace(0, 300, 15001)
    x = 1.0 + np.exp(-t / tau) + 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.1 * rng.standard_normal(t.size)
    te = transient_end(t, x, floor=0.0)
    # the offset must have decayed below the noise but not much more data thrown away
    assert 1.5 * tau <= te <= 6 * tau


def test_mser_detects_amplitude_growth():
    rng = np.random.default_rng(21)
    t = np.linspace(0, 300, 15001)
    x = 0.6 * np.sin(2 * np.pi * 0.2 * t) / (1 + np.exp(-(t - 60.0) / 5.0)) + 0.02 * rng.standard_normal(t.size)
    te = transient_end(t, x, floor=0.0)
    assert 60.0 <= te <= 110.0


def test_mser_stationary_signal_uses_floor():
    rng = np.random.default_rng(22)
    t = np.linspace(0, 300, 15001)
    for _ in range(5):
        x = 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.1 * rng.standard_normal(t.size)
        assert transient_end(t, x, floor=0.0) == pytest.approx(0.0, abs=1.0)
        assert transient_end(t, x) == pytest.approx(30.0, abs=0.1)            # default 10 % floor
    assert transient_end(t, x, t_min=50.0) >= 50.0


# --- coefficient statistics ---------------------------------------------------

def test_coefficient_stats_values():
    t = np.linspace(0, 100, 10001)
    x = 1.5 + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    s = coefficient_stats(t, x, t_start=20.0)
    assert s.mean == pytest.approx(1.5, abs=2e-4)
    assert s.rms == pytest.approx(0.2 / math.sqrt(2), rel=1e-3)
    assert s.min == pytest.approx(1.3, abs=1e-6) and s.max == pytest.approx(1.7, abs=1e-6)
    assert 0 <= s.stderr < 1e-3
    assert stats(x, t, 20.0) == s


def test_coefficient_stats_is_time_weighted_and_robust():
    # Variable step: the first half is sampled 10x more densely at value 0, the second
    # half sparsely at value 1. A sample mean would say ~0.09; the time mean is 0.5.
    t = np.concatenate([np.linspace(0, 50, 5000, endpoint=False), np.linspace(50, 100, 500)])
    x = (t >= 50).astype(float)
    assert coefficient_stats(t, x, t_start=0.0).mean == pytest.approx(0.5, abs=0.02)
    s = coefficient_stats([0, 1, 2], [1.0, np.nan, 3.0], t_start=0.0)
    assert s.mean == pytest.approx(2.0) and math.isnan(s.stderr)
    assert math.isnan(coefficient_stats(t, x, t_start=1e9).mean)


# --- convergence ------------------------------------------------------------

def test_convergence_states():
    rng = np.random.default_rng(31)
    t = np.linspace(0, 200, 20001)
    cd = 1.3 + 0.02 * np.sin(2 * np.pi * 0.2 * t) + 0.01 * rng.standard_normal(t.size)
    c = convergence_state(t, cd)
    assert c["state"] == "converged" and c["converged"]
    assert c["periodic"] and c["frequency"] == pytest.approx(0.2, rel=0.01)
    assert c["rel_stderr"] < 0.01

    drift = 1.0 + 0.002 * t + 0.01 * rng.standard_normal(t.size)
    c = convergence_state(t, drift, t_start=50.0)
    assert c["state"] == "transient" and not c["converged"]
    assert c["drift"] == pytest.approx(0.002 * 75, rel=0.05)

    growth = np.sin(2 * np.pi * 0.2 * t) * (1 + t / 50.0)
    assert convergence_state(t, growth, t_start=50.0, atol=0.5)["state"] == "transient"

    noisy = 0.05 + ar1(20001, 0.99, 1, rng)[0]           # mean ~0 known only to ~0.2
    c = convergence_state(t, noisy, t_start=0.0, atol=1e-3)
    assert c["state"] == "converging"

    assert convergence_state(t[:10], cd[:10], t_start=0.0)["state"] == "insufficient"
    steady = 1.0 + 1e-9 * np.exp(-t)
    c = convergence_state(t, steady)
    assert c["steady"] and c["converged"]


def test_detect_steady_and_summary():
    rng = np.random.default_rng(32)
    t = np.linspace(0, 150, 15001)
    cd = 1.2 + 0.5 * np.exp(-t / 5) + 0.01 * np.sin(2 * np.pi * 0.2 * t) + 0.003 * rng.standard_normal(t.size)
    cl = shedding(t, 0.2, rng)
    assert detect_steady(t, cd)
    assert detect_steady(t, cd, window=50.0)
    assert not detect_steady(t[:3000], (1 + t / 10)[:3000])
    s = summarize_history({"t": t, "cd": cd, "cl": cl, "cm": -0.1 + 0 * t})
    assert s["t_start"] >= 15.0
    assert s["stats"]["cd"].mean == pytest.approx(1.2, abs=2e-3)
    assert s["strouhal"] == pytest.approx(0.2, rel=0.01)
    assert set(s["convergence"]) == {"cd", "cl", "cm"}


# --- short windows: the error bar is flagged and made conservative (review round) -------

def test_short_red_noise_window_is_flagged_and_its_error_bar_covers_the_truth():
    """AR(1) with tau = 320 samples in a 2,240-sample window (about 7 effective samples):
    the shrunk batches gave |z| > 2 in 26 % of windows; the flagged bound must not."""
    rng = np.random.default_rng(320)
    phi = 319.0 / 321.0                                   # tau = (1 + phi) / (1 - phi) = 320
    x = ar1(2240, phi, 200, rng)
    bms = [batch_means(row) for row in x]
    assert sum(not b["reliable"] for b in bms) >= 0.95 * len(bms)
    se = np.array([b["stderr"] for b in bms])
    shrunk = np.array([b["stderr_batches"] for b in bms])
    assert np.all(se >= shrunk)
    true = x.mean(axis=1).std(ddof=1)                     # ~3.3 for unit innovations
    assert np.median(se) >= 0.9 * true
    z = np.abs(x.mean(axis=1)) / se
    assert np.mean(z > 2) <= 0.10
    assert np.mean(np.abs(x.mean(axis=1)) / shrunk > 2) > 0.15     # what the old value did


def test_enough_data_keeps_the_plain_batch_means_value():
    rng = np.random.default_rng(21)
    x = ar1(20_000, 0.9, 1, rng)[0]
    b = batch_means(x)
    assert b["reliable"] and b["stderr"] == b["stderr_batches"]


def test_a_short_periodic_window_stays_reliable():
    """Six cycles of a shedding-like oscillation: batches shorter than a period over-state
    the error, so the value is safe to compare with (laminar shedding runs rely on it)."""
    rng = np.random.default_rng(22)
    i = np.arange(6 * 200 + 37)
    x = 0.3 * np.sin(2 * np.pi * i / 200.0 + 0.4) + 0.01 * rng.standard_normal(i.size)
    b = batch_means(x)
    assert b["reliable"]
    assert b["stderr"] >= abs(x.mean())                   # the mean of a sine is ~0


def test_coefficient_stats_carries_the_reliability_flag():
    rng = np.random.default_rng(23)
    t = np.arange(2240) * 0.01
    st = coefficient_stats(t, ar1(2240, 319.0 / 321.0, 1, rng)[0], t_start=0.0)
    assert st.stderr_reliable is False and math.isfinite(st.stderr)
    st = coefficient_stats(np.arange(20_000) * 0.01, ar1(20_000, 0.5, 1, rng)[0], t_start=0.0)
    assert st.stderr_reliable is True
