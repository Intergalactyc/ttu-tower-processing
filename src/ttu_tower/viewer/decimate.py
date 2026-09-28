"""NaN-aware min/max decimation for drawing long, gappy series: each bucket
of samples becomes its max and min, so spikes survive; a bucket with no
finite value becomes a gap. (pyqtgraph's own "peak" downsampling uses
max/min, which turns any bucket containing a NaN into NaN.)
"""
import warnings

import numpy as np


def visible_slice(x: np.ndarray, x0: float, x1: float, margin: int = 1) -> slice:
    """Indices of sorted `x` inside [x0, x1], widened by `margin` on each side
    so a line leaving the view still reaches the edge.
    """
    lo = max(int(np.searchsorted(x, x0, side="left")) - margin, 0)
    hi = min(int(np.searchsorted(x, x1, side="right")) + margin, x.size)
    return slice(lo, hi)


def minmax_decimate(x: np.ndarray, y: np.ndarray, x0: float, x1: float, buckets: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) to draw for the visible range [x0, x1] with about `buckets`
    horizontal buckets. Returns the raw points when they already fit.
    """
    sl = visible_slice(x, x0, x1)
    xv, yv = x[sl], y[sl]
    buckets = max(int(buckets), 1)
    if xv.size <= 2 * buckets:
        return xv, yv

    per = int(np.ceil(xv.size / buckets))
    n = int(np.ceil(xv.size / per))
    pad = n * per - xv.size
    ypad = np.concatenate([yv, np.full(pad, np.nan)]).reshape(n, per)
    xpad = np.concatenate([xv, np.full(pad, xv[-1])]).reshape(n, per)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN buckets are gaps, on purpose
        ymax = np.nanmax(ypad, axis=1)
        ymin = np.nanmin(ypad, axis=1)
    xc = xpad[:, 0] + (xpad[:, -1] - xpad[:, 0]) / 2
    out_x = np.repeat(xc, 2)
    out_y = np.empty(2 * n)
    out_y[0::2] = ymax
    out_y[1::2] = ymin
    return out_x, out_y


def break_wraps(x: np.ndarray, y: np.ndarray, period: float = 360.0) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN wherever consecutive values of a circular quantity jump by
    more than half a period, so a line doesn't cross the plot at 0/360.
    """
    jumps = np.flatnonzero(np.abs(np.diff(y)) > period / 2) + 1
    if jumps.size == 0:
        return x, y
    x_mid = (x[jumps - 1] + x[jumps]) / 2
    return np.insert(x, jumps, x_mid), np.insert(y.astype(np.float64), jumps, np.nan)
