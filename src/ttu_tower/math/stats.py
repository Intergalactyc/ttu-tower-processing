"""Detrending and distribution-shape statistics. Pure numpy."""
import numpy as np

from ttu_tower.math import fits


def linear_detrend(x: np.ndarray) -> np.ndarray:
    """x minus its least-squares linear fit against sample index. NaN samples
    don't affect the fit and stay NaN in the output.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(x.size, dtype=np.float64)
    a, b = fits.ls_linear_fit(t, x)
    return x - (a + b * t)


def moments(x: np.ndarray) -> tuple[float, float]:
    """Population skewness (m3/m2^1.5) and kurtosis (m4/m2^2; Gaussian = 3)
    of the finite values of x.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.nan, np.nan
    d = finite - finite.mean()
    m2 = np.mean(d**2)
    m3 = np.mean(d**3)
    m4 = np.mean(d**4)
    return float(m3 / m2**1.5), float(m4 / m2**2)


def autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """S(l) = sum_i x_i * x_{i+l} for l = 0..max_lag (an unnormalized sum, not
    a mean), by FFT with zero-padding to the next power of two >= 2*len(x) so
    the result is the exact linear (non-circular) autocovariance.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    size = 1
    while size < 2 * n:
        size *= 2
    spectrum = np.fft.rfft(x, size)
    full = np.fft.irfft(spectrum * np.conj(spectrum), size)
    return full[: max_lag + 1]


def efolding_integral(rho: np.ndarray) -> float:
    """Trapezoid integral of rho (values at lags 0, 1, 2, ...) from lag 0 to
    the first crossing of 1/e, linearly interpolating the crossing point
    between the two bracketing lags. NaN if rho never reaches 1/e.
    """
    rho = np.asarray(rho, dtype=np.float64)
    threshold = 1.0 / np.e
    below = np.flatnonzero(rho <= threshold)
    if below.size == 0:
        return np.nan
    first = int(below[0])
    if first == 0:
        return 0.0
    r0, r1 = rho[first - 1], rho[first]
    frac = (r0 - threshold) / (r0 - r1) if r0 != r1 else 0.0
    lags = np.arange(first + 1, dtype=np.float64)
    lags[-1] = (first - 1) + frac
    values = np.concatenate([rho[:first], [threshold]])
    return float(np.trapezoid(values, lags))
