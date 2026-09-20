"""Interior data-gap detection and short-gap linear filling."""
import numpy as np

from ttu_tower.flags import mask_to_intervals


def nan_runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Runs of NaN in x (interior and edge), as local [start, end) indices."""
    return mask_to_intervals(np.isnan(np.asarray(x)), g0=0)


def _interpolate_runs(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.array(x, dtype=np.float64, copy=True)
    filled = np.zeros(x.size, dtype=bool)
    for s, e in zip(starts, ends):
        left, right = x[s - 1], x[e]
        length = e - s
        t = np.arange(1, length + 1, dtype=np.float64) / (length + 1)
        out[s:e] = left + t * (right - left)
        filled[s:e] = True
    return out, filled


def fill_short(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill interior NaN runs of at most `max_gap` samples (with
    finite neighbours on both sides). Returns (x_filled, filled_mask).
    """
    x = np.asarray(x, dtype=np.float64)
    starts, ends = nan_runs(x)
    interior = (starts > 0) & (ends < x.size) & (ends - starts <= max_gap)
    return _interpolate_runs(x, starts[interior], ends[interior])


def fill_within_runs(x: np.ndarray, file_run_id: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """As fill_short, but only gaps whose two neighbours share a
    `file_run_id` >= 0 (the same file run) are filled.
    """
    x = np.asarray(x, dtype=np.float64)
    file_run_id = np.asarray(file_run_id)
    starts, ends = nan_runs(x)
    interior = (starts > 0) & (ends < x.size) & (ends - starts <= max_gap)
    starts, ends = starts[interior], ends[interior]
    same_run = (file_run_id[starts - 1] >= 0) & (file_run_id[starts - 1] == file_run_id[ends])
    return _interpolate_runs(x, starts[same_run], ends[same_run])
