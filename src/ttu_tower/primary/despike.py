"""Despiking: an interpolated median/MAD reference (VM97), outlier runs split
at median crossings, short runs removed as spikes, longer ones kept and
recorded as excursions.
"""
from dataclasses import dataclass

import numpy as np

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.flags import mask_to_intervals

_MAD_TO_SIGMA = 0.6745


@dataclass
class Despiked:
    x: np.ndarray
    spike: np.ndarray
    excursion: np.ndarray
    unchecked: np.ndarray


def _reference_centers(x: np.ndarray, g0: int, stride: int, half_window: int, min_mad: float, c: float):
    """Median/MAD at every global multiple of `stride` whose full
    [center - half_window, center + half_window) window lies in x's span.
    """
    n = x.size
    if n < 2 * half_window:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f, np.empty(0, dtype=bool)

    first_center = -(-(g0 + half_window) // stride) * stride  # smallest multiple of stride with center - H >= g0
    last_center = ((g0 + n - half_window) // stride) * stride  # largest multiple of stride with center + H <= g0+n
    if first_center > last_center:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f, np.empty(0, dtype=bool)

    centers = np.arange(first_center, last_center + 1, stride, dtype=np.int64)
    starts_local = centers - half_window - g0

    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * half_window)
    selected = windows[starts_local]  # (n_centers, 2*half_window)

    finite_count = np.sum(np.isfinite(selected), axis=1)
    valid = finite_count >= c * 2 * half_window

    m = np.nanmedian(selected, axis=1)
    mad = np.maximum(np.nanmedian(np.abs(selected - m[:, None]), axis=1), min_mad)

    return centers, m, mad, valid


def _interpolated_reference(g: np.ndarray, centers: np.ndarray, m_c: np.ndarray, mad_c: np.ndarray, valid_c: np.ndarray, half_window: int):
    """Per-sample median/MAD, interpolated between the bracketing reference
    centers (falling back to the single valid one, or the nearest valid
    center within half_window, or unchecked - see despike()'s docstring).
    """
    n = g.size
    m = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    unchecked = np.ones(n, dtype=bool)
    if centers.size == 0:
        return m, mad, unchecked

    idx_left = np.searchsorted(centers, g, side="right") - 1
    idx_right = idx_left + 1
    left_exists = idx_left >= 0
    right_exists = idx_right < centers.size
    idx_left_c = np.clip(idx_left, 0, centers.size - 1)
    idx_right_c = np.clip(idx_right, 0, centers.size - 1)
    left_valid = left_exists & valid_c[idx_left_c]
    right_valid = right_exists & valid_c[idx_right_c]

    both = left_valid & right_valid
    c_left, c_right = centers[idx_left_c[both]], centers[idx_right_c[both]]
    t = (g[both] - c_left) / (c_right - c_left)
    m[both] = m_c[idx_left_c[both]] + t * (m_c[idx_right_c[both]] - m_c[idx_left_c[both]])
    mad[both] = mad_c[idx_left_c[both]] + t * (mad_c[idx_right_c[both]] - mad_c[idx_left_c[both]])
    unchecked[both] = False

    only_left = left_valid & ~right_valid
    m[only_left] = m_c[idx_left_c[only_left]]
    mad[only_left] = mad_c[idx_left_c[only_left]]
    unchecked[only_left] = False

    only_right = right_valid & ~left_valid
    m[only_right] = m_c[idx_right_c[only_right]]
    mad[only_right] = mad_c[idx_right_c[only_right]]
    unchecked[only_right] = False

    neither = ~left_valid & ~right_valid
    valid_centers, valid_m, valid_mad = centers[valid_c], m_c[valid_c], mad_c[valid_c]
    if np.any(neither) and valid_centers.size > 0:
        idx_neither = np.flatnonzero(neither)
        g_n = g[idx_neither]
        iv = np.searchsorted(valid_centers, g_n)
        left_i = np.clip(iv - 1, 0, valid_centers.size - 1)
        right_i = np.clip(iv, 0, valid_centers.size - 1)
        left_d = np.abs(g_n - valid_centers[left_i])
        right_d = np.abs(g_n - valid_centers[right_i])
        use_left = left_d <= right_d
        nearest_d = np.where(use_left, left_d, right_d)
        nearest_i = np.where(use_left, left_i, right_i)
        within = nearest_d <= half_window

        found = idx_neither[within]
        m[found] = valid_m[nearest_i[within]]
        mad[found] = valid_mad[nearest_i[within]]
        unchecked[found] = False

    return m, mad, unchecked


def _split_and_classify(x: np.ndarray, m: np.ndarray, outlier: np.ndarray, max_spike_samples: int):
    spike = np.zeros(x.size, dtype=bool)
    excursion = np.zeros(x.size, dtype=bool)
    sign = np.sign(x - m)
    starts, ends = mask_to_intervals(outlier, g0=0)
    for s, e in zip(starts, ends):
        seg_sign = sign[s:e]
        splits = np.flatnonzero(np.diff(seg_sign) != 0) + 1
        piece_starts = np.concatenate(([0], splits)) + s
        piece_ends = np.concatenate((splits, [e - s])) + s
        for ps, pe in zip(piece_starts, piece_ends):
            if pe - ps <= max_spike_samples:
                spike[ps:pe] = True
            else:
                excursion[ps:pe] = True
    return spike, excursion


def despike(x: np.ndarray, g0: int, cfg_despike, min_mad: float, c: float) -> Despiked:
    """A 5-min median/MAD reference (VM97), evaluated every `stride_s` and
    linearly interpolated between reference centers. Per sample: if both
    bracketing centers are valid (finite coverage >= c), interpolate; if
    exactly one is valid, use it; if neither is, use the nearest valid center
    within half a window; otherwise the sample is `unchecked`.

    An outlier (z = 0.6745*|x-m|/MAD > z_threshold) run is split wherever
    x - m changes sign, since a real signal can't jump across the reference
    without passing through it - a run that does is a noise burst however
    long. Each piece of at most `max_spike_samples` is a spike (removed);
    a longer one is an excursion (kept).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    stride = round(cfg_despike.stride_s * SAMPLE_HZ)
    half_window = round(cfg_despike.window_s * SAMPLE_HZ / 2)

    centers, m_c, mad_c, valid_c = _reference_centers(x, g0, stride, half_window, min_mad, c)
    g = g0 + np.arange(n, dtype=np.int64)
    m, mad, unchecked = _interpolated_reference(g, centers, m_c, mad_c, valid_c, half_window)

    z = _MAD_TO_SIGMA * np.abs(x - m) / mad  # NaN (unchecked, or x is NaN) is never > threshold
    outlier = z > cfg_despike.z_threshold

    spike, excursion = _split_and_classify(x, m, outlier, cfg_despike.max_spike_samples)

    x_out = x.copy()
    x_out[spike] = np.nan
    return Despiked(x=x_out, spike=spike, excursion=excursion, unchecked=unchecked)
