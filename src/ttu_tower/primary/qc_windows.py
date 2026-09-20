"""Resolution, dropouts (100-s windows) and higher moments (per slot)."""
import numpy as np

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.flags import mask_to_intervals
from ttu_tower.math import stats
from ttu_tower.timegrid import SAMPLES_PER_SLOT


def _bin_range(sub: np.ndarray) -> tuple[float, float]:
    """[mean-3.5sigma, mean+3.5sigma] if narrower than [min, max], else [min, max]."""
    mean, std = np.nanmean(sub), np.nanstd(sub)
    data_min, data_max = np.nanmin(sub), np.nanmax(sub)
    sigma_lo, sigma_hi = mean - 3.5 * std, mean + 3.5 * std
    if (sigma_hi - sigma_lo) <= (data_max - data_min):
        return sigma_lo, sigma_hi
    return data_min, data_max


def _run_filter(marked: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only runs of >= min_run consecutive True values."""
    if min_run <= 1 or not marked.any():
        return marked
    starts, ends = mask_to_intervals(marked, g0=0)
    keep = (ends - starts) >= min_run
    out = np.zeros_like(marked)
    for s, e in zip(starts[keep], ends[keep]):
        out[s:e] = True
    return out


def resolution_dropouts(x: np.ndarray, g0: int, cfg_windows, c: float, is_ts: bool = False):
    """Per-window resolution and dropout flags, on half-overlapping windows
    anchored to the global grid (VM97's binning test plus a minimum run
    length, a multi-window confirmation rule, and a per-window dropout-pair
    fraction). Returns ((resolution_starts, resolution_ends), (dropout_starts,
    dropout_ends)), global sample intervals.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    window = round(cfg_windows.resolution_window_s * SAMPLE_HZ)
    stride = round(cfg_windows.resolution_stride_s * SAMPLE_HZ)
    bins = cfg_windows.ts_resolution_bins if is_ts else cfg_windows.resolution_bins
    min_run = cfg_windows.ts_dropout_min_run if is_ts else cfg_windows.dropout_min_run

    first = -(-g0 // stride) * stride
    last = ((g0 + n - window) // stride) * stride if n >= window else first - stride

    res_starts, res_ends = [], []
    n_pairs_total = max(n - 1, 0)
    mark_count = np.zeros(n_pairs_total, dtype=np.int32)
    contain_count = np.zeros(n_pairs_total, dtype=np.int32)
    dropout_windows: list[tuple[int, int]] = []  # (local pair start, local pair end) per evaluated window

    for start in range(first, last + 1, stride) if last >= first else ():
        lo = start - g0
        sub = x[lo : lo + window]
        finite = np.isfinite(sub)
        if finite.sum() < c * window:
            continue

        low, high = _bin_range(sub[finite])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        width = (high - low) / bins

        in_range = finite & (sub >= low) & (sub <= high)
        idx = np.clip(np.floor((sub - low) / width), 0, bins - 1)
        counts = np.bincount(idx[in_range].astype(np.int64), minlength=bins)
        empty_bins = bins - np.count_nonzero(counts)

        if empty_bins > int(bins * cfg_windows.max_empty_bin_fraction):
            res_starts.append(start)
            res_ends.append(start + window)
            continue  # dropouts are skipped in resolution-flagged windows

        both_finite = in_range[:-1] & in_range[1:]
        if is_ts:
            same = np.abs(sub[1:] - sub[:-1]) <= cfg_windows.ts_repeat_tolerance
        else:
            same = idx[1:] == idx[:-1]
        candidate = both_finite & same
        candidate = _run_filter(candidate, min_run)

        pair_lo = lo  # pair i (local) covers samples [lo+i, lo+i+2)
        pair_hi = lo + window - 1
        mark_count[pair_lo:pair_hi] += candidate
        contain_count[pair_lo:pair_hi] += 1
        dropout_windows.append((start, pair_lo, pair_hi))

    drop_starts, drop_ends = [], []
    if dropout_windows:
        min_windows = cfg_windows.dropout_min_windows
        required = np.minimum(min_windows, contain_count)
        is_dropout_pair = (contain_count > 0) & (mark_count >= required)
        for start, pair_lo, pair_hi in dropout_windows:
            n_window_pairs = pair_hi - pair_lo
            fraction = is_dropout_pair[pair_lo:pair_hi].sum() / n_window_pairs
            if fraction > cfg_windows.max_dropout_fraction:
                drop_starts.append(start)
                drop_ends.append(start + window)

    return (
        (np.array(res_starts, dtype=np.int64), np.array(res_ends, dtype=np.int64)),
        (np.array(drop_starts, dtype=np.int64), np.array(drop_ends, dtype=np.int64)),
    )


def higher_moments(x: np.ndarray, g0: int, slots, cfg_windows, c: float):
    """Per-slot skewness and kurtosis of the linearly detrended slot, and
    whether each falls outside skew_range/kurt_range. NaN (no flag) if
    coverage < c.
    """
    x = np.asarray(x, dtype=np.float64)
    slots = np.asarray(slots)
    skew = np.full(slots.size, np.nan)
    kurt = np.full(slots.size, np.nan)
    skew_flag = np.zeros(slots.size, dtype=bool)
    kurt_flag = np.zeros(slots.size, dtype=bool)

    skew_lo, skew_hi = cfg_windows.skew_range
    kurt_lo, kurt_hi = cfg_windows.kurt_range

    for i, k in enumerate(slots):
        start = SAMPLES_PER_SLOT * int(k) - g0
        seg = x[start : start + SAMPLES_PER_SLOT]
        if np.isfinite(seg).sum() < c * SAMPLES_PER_SLOT:
            continue
        detrended = stats.linear_detrend(seg)
        s, ku = stats.moments(detrended)
        skew[i], kurt[i] = s, ku
        skew_flag[i] = not (skew_lo <= s <= skew_hi)
        kurt_flag[i] = not (kurt_lo <= ku <= kurt_hi)

    return skew, kurt, skew_flag, kurt_flag
