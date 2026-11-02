"""Statistics of solver time series: means with honest error bars, transients, Strouhal.

A CFD force history is an autocorrelated, often periodic signal that starts with a
start-up transient. Three things decide whether a number shown to the user is
trustworthy, and each has its own tool here:

* **Where the transient ends** — :func:`transient_end` uses MSER-5 (White 1997), the
  standard truncation rule for simulation output: it discards the prefix that
  minimises the standard error of the remaining mean. It is applied to the signal
  *and* to its squared deviation, because vortex shedding grows in amplitude while
  its mean barely moves, and a mean-only rule would happily average the growth phase.
* **How uncertain the mean is** — :func:`batch_means_stderr` uses non-overlapping
  batch means. The batch size comes from the integrated autocorrelation time
  (Sokal's automatic window) and from the first zero of the autocorrelation, so a
  batch spans whole shedding periods; batches of half a period would alternate in
  sign and overstate the error by orders of magnitude.
* **Whether it has settled** — :func:`convergence_state` combines the relative
  standard error, the drift between the two halves of the averaging window and the
  amplitude trend into one verdict the runner uses for early stopping and the UI
  shows as a badge.

Time series may come with a variable time step (CFL-limited solvers). Everything
here first resamples onto a uniform grid with the same number of samples, so means
are time-weighted and spectra are valid. Non-finite samples are dropped.

Only numpy and the standard library are used (DESIGN.md section 2).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..model import Stat

# Below this many samples in the averaging window nothing statistical is attempted.
MIN_SAMPLES = 20
# Fewer batches than this make the batch-means variance itself useless (dof < 9).
MIN_BATCHES = 10
#: A series counts as periodic for the short-window rule of batch_means when its
#: autocorrelation one period after the first zero crossing (lag 4 x first zero) is
#: above this: a sine keeps 1 - P/n there (0.75 at four cycles), while red noise of the
#: same window length is near zero or negative (AR(1), tau 320, n 2,240: 1 % of windows).
PERIODIC_RHO = 0.3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean(t, x) -> tuple[np.ndarray, np.ndarray]:
    """Float64 copies with non-finite samples removed and time strictly increasing."""
    t = np.asarray(t, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if t.shape != x.shape:
        raise ValueError(f"t and x differ in length ({t.size} vs {x.size})")
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size > 1 and np.any(np.diff(t) <= 0):
        order = np.argsort(t, kind="stable")
        t, x = t[order], x[order]
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, x = t[keep], x[keep]
    return t, x


def _uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample onto a uniform grid when the step varies by more than 5 %.

    Same sample count, so the statistics see the same amount of information; linear
    interpolation is exact for the time-weighted (trapezoidal) mean.
    """
    if t.size < 3:
        return t, x
    dt = np.diff(t)
    if dt.max() <= 1.05 * dt.min():
        return t, x
    tu = np.linspace(t[0], t[-1], t.size)
    return tu, np.interp(tu, t, x)


def _window(t, x, t_start: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Clean, cut at ``t_start`` (None = automatic transient end) and make uniform."""
    t, x = _clean(t, x)
    if t.size == 0:
        return t, x
    if t_start is None:
        t_start = transient_end(t, x)
    sel = t >= t_start
    return _uniform(t[sel], x[sel])


def autocorrelation(x) -> np.ndarray:
    """Normalised autocorrelation rho[k], k = 0..n-1 (biased estimator, via FFT).

    The biased (divide-by-n) estimator is the one Sokal's windowing is designed for:
    its tail is damped, so the running sum of rho stays bounded.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.zeros(0)
    y = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n]
    if acf[0] <= 0.0:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    return acf / acf[0]


def integrated_autocorr_time(x, c: float = 5.0) -> float:
    """Integrated autocorrelation time tau (in samples), ``Var(mean) = sigma^2 tau / n``.

    Sokal's automatic window: sum rho up to the smallest lag M with M >= c * tau(M).
    White noise gives 1. The result is clamped to [1, n]: anticorrelated signals would
    give tau < 1, and reporting less uncertainty than independent samples is never
    worth the risk.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 1.0
    rho = autocorrelation(x)
    taus = 2.0 * np.cumsum(rho) - 1.0
    ok = np.arange(n) >= c * taus
    m = int(np.argmax(ok)) if ok.any() else n - 1
    return float(min(max(taus[m], 1.0), n))


def effective_sample_size(x) -> float:
    """Number of effectively independent samples, n / tau."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return x.size / integrated_autocorr_time(x) if x.size else 0.0


def auto_batch_size(x) -> int:
    """Batch length for batch means: long enough that batch means are uncorrelated.

    max(5 tau, 4 * first zero of rho). For a periodic signal the first zero of rho is
    a quarter period, so the batch spans one full period and the periodic part
    averages out inside each batch. For red noise 5 tau keeps the batch-to-batch
    correlation below ~10 %.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 1
    rho = autocorrelation(x)
    tau = integrated_autocorr_time(x)
    nonpos = np.nonzero(rho <= 0.0)[0]
    first_zero = int(nonpos[0]) if nonpos.size else n
    return max(1, math.ceil(5.0 * tau), 4 * first_zero)


def batch_means(x, batch_size: int | None = None) -> dict:
    """Batch-means standard error of the mean, with the numbers behind it.

    Returns ``{"stderr", "stderr_batches", "batch_size", "n_batches", "tau", "ess",
    "reliable"}``. Leading samples that do not fill a batch are the ones dropped,
    because they are the closest to the transient.

    **Short series.** When the series is too short for :data:`MIN_BATCHES` batches of
    the ideal size (:func:`auto_batch_size`), the batches are shrunk to
    ``n // MIN_BATCHES`` samples. For a periodic signal (its autocorrelation one
    period after the first zero crossing above :data:`PERIODIC_RHO`) that is
    harmless: batches shorter than a period scatter with the phase and over-state
    the error. For noise-like signals with fewer than ``MIN_BATCHES * 5 * tau``
    samples (fewer than ~50 effective samples) they *under*-state it. Measured:
    AR(1) noise with tau = 320 samples in a 2,240-sample window gave a true
    standard error 1.6x the shrunk-batch value and |z| > 2 in 26 % of windows
    instead of 5 %; a Draft NACA 4412 run averaged over 20 L/U missed its
    converged mean by 4.2 of its stated errors. Such a value is never returned as
    if it were reliable: ``reliable`` is False and ``stderr`` is the conservative

        max(shrunk-batch value, rms * sqrt(tau_eff / n), |drift| / 2)

    with ``drift`` the difference between the means of the two halves (it
    catches start-up transient left in the window, which in the NACA 4412 case
    was most of the error) and ``tau_eff = max(tau, ideal / 5, 2 * first zero of
    the autocorrelation)``, which does not trust the short-series tau (Sokal's
    estimate came out 116 for a true 320). Measured on AR(1) noise, tau = 320, 400
    windows each: |z| > 2 in 4 % of 2,240-sample windows and 6 % of 6,400-sample
    ones (nominal 5 %; the shrunk batches gave 26 % and 12 %). ``stderr_batches``
    keeps the plain batch-means value; with enough data ``stderr`` is that value.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    nan = float("nan")
    if n < MIN_SAMPLES:
        return {"stderr": nan, "stderr_batches": nan, "batch_size": 0, "n_batches": 0, "tau": nan,
                "ess": float(n), "reliable": False}
    tau = integrated_autocorr_time(x)
    ideal = int(batch_size) if batch_size else auto_batch_size(x)
    cap = n // MIN_BATCHES
    b = max(1, min(ideal, cap))
    nb = n // b
    means = x[n - nb * b:].reshape(nb, b).mean(axis=1)
    se = float(means.std(ddof=1) / math.sqrt(nb))
    reliable, out = True, se
    if ideal > cap:
        rho = autocorrelation(x)
        nonpos = np.nonzero(rho <= 0.0)[0]
        first_zero = int(nonpos[0]) if nonpos.size else n
        periodic = 4 * first_zero < n and rho[4 * first_zero] > PERIODIC_RHO
        h = n // 2
        drift = abs(float(x[n - h:].mean() - x[:h].mean())) if h else 0.0
        if periodic:
            # batches shorter than a period scatter with the phase: they over-state the
            # error of a periodic signal, never under-state it
            out = max(se, 0.5 * drift)
        elif n < MIN_BATCHES * 5.0 * tau:
            reliable = False
            tau_eff = max(tau, ideal / 5.0, 2.0 * first_zero)
            out = max(se, float(x.std()) * math.sqrt(min(tau_eff, n) / n), 0.5 * drift)
    return {"stderr": out, "stderr_batches": se, "batch_size": b, "n_batches": nb, "tau": tau,
            "ess": n / tau, "reliable": bool(reliable)}


def batch_means_stderr(x, batch_size: int | None = None) -> float:
    """Standard error of the mean of an autocorrelated series (``nan`` if too short).

    For a series too short to estimate it reliably this is the conservative
    bound of :func:`batch_means` (check its ``reliable`` flag when it matters).
    """
    return batch_means(x, batch_size)["stderr"]


# ---------------------------------------------------------------------------
# transient detection
# ---------------------------------------------------------------------------

def _mser_index(y: np.ndarray, batch: int = 5, limit: int | None = None) -> int:
    """MSER-k truncation point (sample index) of ``y``; searched over the first half.

    MSER(d) = sum_{i>=d} (Y_i - mean_d)^2 / (m - d)^2 on batch means Y of ``batch``
    samples. Minima in the second half are not trusted (the statistic becomes
    unstable as the remaining sample shrinks), the usual MSER practice. ``limit``
    (sample index, exclusive) narrows the search further.
    """
    nb = y.size // batch
    if nb < 4:
        return 0
    yb = y[: nb * batch].reshape(nb, batch).mean(axis=1)
    s1 = np.cumsum(yb[::-1])[::-1]
    s2 = np.cumsum((yb * yb)[::-1])[::-1]
    m = np.arange(nb, 0, -1, dtype=np.float64)
    sse = np.maximum(s2 - s1 * s1 / m, 0.0)
    mser = sse / (m * m)
    dmax = nb // 2
    if limit is not None:
        dmax = min(dmax, max(0, (limit - 1) // batch))
    return int(np.argmin(mser[: dmax + 1])) * batch


def transient_end(t, x, *, floor: float = 0.1, t_min: float | None = None, batch: int = 5) -> float:
    """Time at which the start-up transient of ``x`` has ended (MSER-5 with a floor).

    MSER-5 is run on the signal (catches a drifting mean, e.g. Cd relaxing after the
    impulsive start) and MSER in period-long batches on its squared deviation from the
    late mean (catches a growing or decaying oscillation, e.g. Cl while shedding
    develops); the later of the two wins. The result is never earlier than ``floor`` of the record (the
    impulsive start always pollutes the first part, whatever the statistic says) nor
    earlier than ``t_min`` when given.
    """
    t, x = _clean(t, x)
    if t.size == 0:
        return float("nan")
    t0, t1 = float(t[0]), float(t[-1])
    lower = t0 + floor * (t1 - t0)
    if t_min is not None:
        lower = max(lower, float(t_min))
    if t.size < 4 * batch:
        return min(lower, t1)
    tu, xu = _uniform(t, x)
    i_mean = _mser_index(xu, batch)
    # Envelope test. The squared deviation of a sine is sin^2, whose own fluctuation
    # swamps the bias of a growth phase unless each batch spans a whole period, so the
    # batch length is the period estimated from the late (settled) half: four times the
    # first zero of its autocorrelation.
    late = xu[xu.size // 2:]
    rho = autocorrelation(late)
    nonpos = np.nonzero(rho <= 0.0)[0]
    period = 4 * int(nonpos[0]) if nonpos.size else batch
    env_batch = int(min(max(batch, period), max(batch, xu.size // 20)))
    env = (xu - late.mean()) ** 2
    # With period-long batches there are few of them, and MSER then sometimes finds a
    # spurious minimum well after the real transient. A cut is accepted only if the
    # stretch just before it (the later half of the discarded prefix, at least one
    # batch) really had a different amplitude, by more than 15 % in mean square;
    # otherwise the search is repeated on the part before the rejected cut.
    i_amp, limit = 0, None
    for _ in range(6):
        i_amp = _mser_index(env, env_batch, limit)
        if i_amp == 0:
            break
        lo = max(0, i_amp - max(env_batch, i_amp // 2))
        pre, post = env[lo:i_amp].mean(), env[i_amp:].mean()
        if abs(pre - post) > 0.15 * post:
            break
        limit, i_amp = i_amp, 0
    i = max(i_mean, i_amp)
    return min(max(float(tu[i]), lower), t1)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def coefficient_stats(t, x, t_start: float | None = None) -> Stat:
    """Mean, fluctuation rms, min, max and standard error of ``x`` for ``t >= t_start``.

    ``t_start=None`` picks the averaging start with :func:`transient_end`. An empty
    window gives an all-``nan`` :class:`~aerolume.model.Stat`; a window too short for
    batch means keeps the mean but leaves ``stderr`` as ``nan``. When the window
    holds too few correlation lengths for a reliable estimate, ``stderr`` is the
    conservative bound of :func:`batch_means` and ``stderr_reliable`` is False, so
    the Compare page and the report never treat it as a measured error bar.
    """
    _, xw = _window(t, x, t_start)
    if xw.size == 0:
        return Stat(stderr_reliable=False)
    bm = batch_means(xw)
    return Stat(mean=float(xw.mean()), rms=float(xw.std()), min=float(xw.min()),
                max=float(xw.max()), stderr=bm["stderr"], stderr_reliable=bool(bm["reliable"]))


def stats(series, t, t_start: float | None = None) -> Stat:
    """DESIGN.md 5.5 spelling of :func:`coefficient_stats` (series first)."""
    return coefficient_stats(t, series, t_start)


# ---------------------------------------------------------------------------
# frequency / Strouhal
# ---------------------------------------------------------------------------

def _zero_crossing_frequency(t: np.ndarray, x: np.ndarray) -> float | None:
    """Mean frequency from upward mean crossings with hysteresis, or None if irregular.

    A crossing counts only after the signal has gone below -h and then above +h
    (h = a quarter of the rms), so noise riding on the signal cannot add crossings.
    The crossing instant itself is the last zero crossing before the +h event,
    linearly interpolated.
    """
    y = x - x.mean()
    h = 0.25 * y.std()
    if h <= 0:
        return None
    s = np.where(y > h, 1, np.where(y < -h, -1, 0))
    idx = np.where(s != 0, np.arange(s.size), 0)
    np.maximum.accumulate(idx, out=idx)
    ff = s[idx]
    rises = np.nonzero((ff[1:] == 1) & (ff[:-1] == -1))[0] + 1
    tc = []
    for i in rises:
        j = int(i)
        while j > 0 and y[j - 1] >= 0.0:
            j -= 1
        if j == 0:
            continue
        frac = -y[j - 1] / (y[j] - y[j - 1])
        tc.append(t[j - 1] + frac * (t[j] - t[j - 1]))
    if len(tc) < 3:
        return None
    periods = np.diff(tc)
    if periods.std() > 0.25 * periods.mean():
        return None
    return float((len(tc) - 1) / (tc[-1] - tc[0]))


def dominant_frequency(t, x, *, t_start: float | None = None, min_fraction: float = 0.2) -> float | None:
    """Frequency [1/time unit] of the dominant oscillation, or None if not periodic.

    Windowed FFT (Hann, 4x zero padding) peak refined by a parabola through the log
    power of the three top bins (exact for a Gaussian-like main lobe, bias well under
    0.1 % of a bin). The signal counts as periodic when the peak's main lobe holds at
    least ``min_fraction`` of the fluctuation energy and the oscillation is not dying
    out (a decaying start-up perturbation of a steady flow is not shedding). With
    fewer than four cycles in the record the FFT resolution is poor, so the
    zero-crossing estimate is preferred when it is regular.
    """
    tw, xw = _window(t, x, t_start)
    n = xw.size
    if n < 16:
        return None
    y = xw - np.polyval(np.polyfit(tw - tw[0], xw, 1), tw - tw[0])
    scale = max(1.0, float(np.abs(xw).max()))
    if y.std() <= 1e-9 * scale:
        return None
    third = n // 3
    r_first, r_last = y[:third].std(), y[-third:].std()
    if r_last < 0.5 * r_first:
        return None
    dt = (tw[-1] - tw[0]) / (n - 1)
    span = dt * n
    nfft = 4 * (1 << (n - 1).bit_length())
    p = np.abs(np.fft.rfft(y * np.hanning(n), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, dt)
    valid = freqs >= 1.5 / span
    if not valid.any():
        return None
    k0 = int(np.nonzero(valid)[0][0])
    k = k0 + int(np.argmax(p[valid]))
    band = np.abs(freqs - freqs[k]) <= 2.5 / span
    if p[band].sum() < min_fraction * p[valid].sum():
        return _zero_crossing_frequency(tw, xw) if span * freqs[k] < 4 else None
    f = freqs[k]
    if 0 < k < p.size - 1 and min(p[k - 1], p[k], p[k + 1]) > 0:
        a, b, c = np.log(p[k - 1]), np.log(p[k]), np.log(p[k + 1])
        den = a - 2.0 * b + c
        if den < 0:
            f = (k + 0.5 * (a - c) / den) * (freqs[1] - freqs[0])
    if f * span < 4:
        fz = _zero_crossing_frequency(tw, xw)
        if fz is not None:
            return fz
        if f * span < 2:
            return None
    return float(f)


def strouhal(t, cl, length: float = 1.0, speed: float = 1.0, *, t_start: float | None = None) -> float | None:
    """Strouhal number St = f L / U of the lift oscillation, or None if not periodic.

    With the solver's convective time (``t`` in L/U) the defaults give St = f directly.
    """
    f = dominant_frequency(t, cl, t_start=t_start)
    return None if f is None else f * length / speed


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

def convergence_state(t, x, t_start: float | None = None, *, rtol: float = 0.01,
                      atol: float = 1e-3) -> dict:
    """How settled the averaging window of ``x`` is, as numbers plus a verdict.

    The mean must be known to ``tol = max(rtol * |mean|, atol)`` (``atol`` matters
    for quantities that average to ~0, like Cl of a symmetric body). Keys:

    ``t_start, window, n, mean, stderr, rel_stderr, rms, tau`` (time units),
    ``ess, drift`` (mean of 2nd half minus 1st half), ``drift_sigma`` (drift over its
    standard error), ``amplitude_ratio`` (fluctuation rms, 2nd half / 1st half),
    ``steady`` (fluctuations negligible), ``periodic, frequency, cycles``,
    ``converged`` (bool), ``state`` ("insufficient" | "transient" | "converging" |
    "converged") and ``reason`` (one human-readable sentence for the UI).

    Rules, in order: fewer than 50 samples -> insufficient; a drift of the
    half-window means larger than ``tol`` and than the spread inside either half ->
    transient; fewer than 20 effective samples -> insufficient; a drift larger than
    both 3 standard errors and ``tol``, or a fluctuation amplitude changing by more
    than 25 % between the halves -> transient; a standard error above ``tol`` ->
    converging (just needs more averaging); otherwise converged.
    """
    tc, xc = _clean(t, x)
    if t_start is None and tc.size:
        t_start = transient_end(tc, xc)
    tw, xw = _window(tc, xc, t_start if t_start is not None else -math.inf)
    n = int(xw.size)
    nan = float("nan")
    out: dict = {"t_start": float(t_start) if t_start is not None else nan,
                 "window": float(tw[-1] - tw[0]) if n > 1 else 0.0, "n": n,
                 "mean": nan, "stderr": nan, "rel_stderr": nan, "rms": nan, "tau": nan,
                 "ess": 0.0, "drift": nan, "drift_sigma": nan, "amplitude_ratio": nan,
                 "steady": False, "periodic": False, "frequency": None, "cycles": None,
                 "converged": False, "state": "insufficient", "reason": ""}
    if n < 2:
        out["reason"] = "No samples in the averaging window yet."
        return out
    mean, rms = float(xw.mean()), float(xw.std())
    bm = batch_means(xw)
    dt = out["window"] / (n - 1)
    tol = max(rtol * abs(mean), atol)
    out.update(mean=mean, rms=rms, stderr=bm["stderr"], ess=float(bm["ess"]),
               tau=bm["tau"] * dt if math.isfinite(bm["tau"]) else nan)
    out["rel_stderr"] = bm["stderr"] / abs(mean) if mean != 0 else math.inf
    out["steady"] = steady = rms <= 0.1 * tol
    if n < 50:
        out["reason"] = "Too few samples in the averaging window; keep running."
        return out

    h = n // 2
    a, b = xw[:h], xw[n - h:]
    se_a, se_b = batch_means_stderr(a), batch_means_stderr(b)
    drift = float(b.mean() - a.mean())
    se_d = math.sqrt(se_a ** 2 + se_b ** 2) if math.isfinite(se_a) and math.isfinite(se_b) else nan
    out["drift"] = drift
    out["drift_sigma"] = abs(drift) / se_d if se_d and math.isfinite(se_d) else (
        0.0 if drift == 0 else math.inf)
    # A drift larger than the spread inside either half is a trend whatever the
    # sample size says (a ramp has a huge autocorrelation time, so its ESS is tiny,
    # but calling it "insufficient data" instead of "still drifting" would mislead).
    gross_drift = abs(drift) > tol and abs(drift) > max(a.std(), b.std())
    if not steady and bm["ess"] < 20 and not gross_drift:
        out["reason"] = "Too few independent samples to judge convergence; keep running."
        return out
    if not steady:
        ra, rb = a.std(), b.std()
        out["amplitude_ratio"] = float(rb / ra) if ra > 0 else math.inf
        f = dominant_frequency(tw, xw, t_start=tw[0])
        if f is not None:
            out.update(periodic=True, frequency=f, cycles=f * out["window"])

    drifting = gross_drift or (abs(drift) > tol and not (math.isfinite(se_d) and abs(drift) <= 3.0 * se_d))
    amp_changing = (not steady) and not (0.8 <= out["amplitude_ratio"] <= 1.25)
    if drifting:
        out.update(state="transient", reason=f"The running mean is still drifting "
                   f"({drift:+.3g} between the two halves of the window).")
    elif amp_changing:
        out.update(state="transient", reason=f"The oscillation amplitude is still changing "
                   f"(x{out['amplitude_ratio']:.2f} between the two halves of the window).")
    elif not (bm["stderr"] <= tol or (steady and not math.isfinite(bm["stderr"]))):
        out.update(state="converging", reason=f"Mean known to ±{bm['stderr']:.2g}; "
                   f"target ±{tol:.2g} needs a longer average.")
    else:
        out.update(state="converged", converged=True,
                   reason="Steady: no fluctuation, mean settled." if steady else
                   f"Statistically converged: mean {mean:.4g} ± {bm['stderr']:.2g}.")
    return out


def detect_steady(t, x, window: float | None = None, *, rtol: float = 0.01, atol: float = 1e-3) -> bool:
    """True when ``x`` is statistically settled and its mean known to tolerance.

    ``window`` (time units) looks only at the trailing part of the record; None uses
    the automatic transient end. This is the early-stop test of the runner.
    """
    tc, xc = _clean(t, x)
    if tc.size == 0:
        return False
    t_start = None if window is None else float(tc[-1]) - float(window)
    return bool(convergence_state(tc, xc, t_start, rtol=rtol, atol=atol)["converged"])


def summarize_history(history: Mapping[str, np.ndarray], *, length: float = 1.0, speed: float = 1.0,
                      t_start: float | None = None, keys: tuple[str, ...] = ("cd", "cl", "cm"),
                      rtol: float = 0.01, atol: float = 1e-3) -> dict:
    """Stats, Strouhal and convergence for a solver history ``{"t", "cd", "cl", "cm"}``.

    One common averaging start is used for every coefficient (the latest of their
    transient ends), so Cd and Cl are averaged over the same physical interval.
    Returns ``{"t_start", "stats": {key: Stat}, "convergence": {key: dict},
    "strouhal", "converged"}``.
    """
    t = np.asarray(history["t"], dtype=np.float64)
    present = [k for k in keys if k in history and len(history[k]) == t.size]
    if t_start is None:
        ends = [transient_end(t, history[k]) for k in present]
        ends = [e for e in ends if math.isfinite(e)]
        t_start = max(ends) if ends else (float(t[0]) if t.size else float("nan"))
    st = {k: coefficient_stats(t, history[k], t_start) for k in present}
    conv = {k: convergence_state(t, history[k], t_start, rtol=rtol, atol=atol) for k in present}
    sh = strouhal(t, history["cl"], length, speed, t_start=t_start) if "cl" in present else None
    return {"t_start": t_start, "stats": st, "convergence": conv, "strouhal": sh,
            "converged": bool(conv) and all(c["converged"] for c in conv.values())}


#: The six coefficient histories of a 3D run (DESIGN.md 7.2, ``FlowSolver3D.history()``).
HISTORY_KEYS_3D = ("cd", "cs", "cl", "c_roll", "cm", "cn")


def summarize_history3d(history: Mapping[str, np.ndarray], *, length: float = 1.0, speed: float = 1.0,
                        t_start: float | None = None, rtol: float = 0.01, atol: float = 1e-3) -> dict:
    """:func:`summarize_history` for a 3D run's ``{"t", "cd", "cs", "cl", "c_roll", "cm", "cn"}``.

    The same statistics as in 2D (one common averaging start, MSER transient end,
    batch-means error bars) for all six coefficients. The Strouhal number is
    taken from whichever of lift and side force fluctuates more in the averaging
    window: a 3D wake sheds in a plane of its own choosing (a sphere at Re 300
    sheds sideways as readily as vertically), so lift alone can miss it.
    ``converged`` requires every coefficient present to have settled; coefficients
    that are identically zero (Cs, C_roll, Cn of a symmetry-plane run) count as
    settled via ``atol``.
    """
    out = summarize_history(history, length=length, speed=speed, t_start=t_start,
                            keys=HISTORY_KEYS_3D, rtol=rtol, atol=atol)
    st = out["stats"]
    cands = [k for k in ("cl", "cs") if k in st and math.isfinite(st[k].rms)]
    if cands:
        k = max(cands, key=lambda c: st[c].rms)
        out["strouhal"] = strouhal(history["t"], history[k], length, speed, t_start=out["t_start"])
        out["strouhal_from"] = k
    else:
        out["strouhal_from"] = None
    return out
