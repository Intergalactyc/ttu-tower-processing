"""Slow-sensor (t, rh, p) smoothing and the reference-pressure vpts."""
import numpy as np

from ttu_tower.constants import P_REF, SAMPLE_HZ
from ttu_tower.physics import thermo


def hann_kernel(width_s: float) -> np.ndarray:
    """A Hann window of full width `width_s`, evaluated at every sample; zero
    at +-width_s/2 (one sample beyond its 2H-1 taps, H = width_s*25).
    """
    half = round(width_s * SAMPLE_HZ / 2)
    j = np.arange(-(half - 1), half, dtype=np.float64)
    return np.cos(np.pi * j / (2 * half)) ** 2


def smooth(x: np.ndarray, ok: np.ndarray, k: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """A NaN-aware normalized convolution of x by kernel k, evaluated at
    every sample: usable where the kernel's weighted coverage of `ok` is >=
    c * sum(k), NaN elsewhere. Direct convolution (not FFT), so each output is
    a fixed-order sum over its own window - identical whatever the span.
    """
    x = np.asarray(x, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool)
    x0 = np.where(ok, x, 0.0)
    s = np.convolve(x0, k, "same")
    w = np.convolve(ok.astype(np.float64), k, "same")
    usable = w >= c * k.sum()
    xs = np.full(x.size, np.nan)
    xs[usable] = s[usable] / w[usable]
    return xs, usable


def vpts(ts: np.ndarray, boom: int) -> np.ndarray:
    """Sonic virtual potential temperature at the boom's fixed reference
    pressure P_REF (the measured-pressure correction is applied in secondary).
    """
    return thermo.potential_temperature(ts, P_REF[boom])
