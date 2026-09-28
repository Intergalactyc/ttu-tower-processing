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


def nearest_on_curve(x: np.ndarray, y: np.ndarray, vx: float, vy: float, px: float, py: float, radius_px: float,
                     lines: bool = True, wrap: float | None = None) -> tuple[float, int] | None:
    """(pixel distance, sample index) of the curve sample nearest a clicked view
    position, or None beyond `radius_px`. Every sample within the radius counts,
    however many share a pixel column; with `lines`, a click on the segment
    joining two consecutive finite samples also counts and picks the nearer end
    (a segment spanning more than half of `wrap`, e.g. 360 for directions, is
    not drawn, so it doesn't count).
    """
    lo = max(int(np.searchsorted(x, vx - radius_px * px, side="left")) - 1, 0)
    hi = min(int(np.searchsorted(x, vx + radius_px * px, side="right")) + 1, x.size)
    if hi <= lo:
        return None
    sx, sy = (x[lo:hi] - vx) / px, (y[lo:hi] - vy) / py  # pixels from the click
    d = np.hypot(sx, sy)
    best_d, best_i = np.inf, -1
    if np.isfinite(d).any():
        j = int(np.nanargmin(d))
        best_d, best_i = float(d[j]), lo + j
    if lines and hi - lo >= 2:
        ax, ay, bx, by = sx[:-1], sy[:-1], sx[1:], sy[1:]
        ok = np.isfinite(ay) & np.isfinite(by)
        if wrap is not None:
            ok &= np.abs(y[lo + 1:hi] - y[lo:hi - 1]) <= wrap / 2
        seg_x, seg_y = bx - ax, by - ay
        length2 = seg_x**2 + seg_y**2
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip(-(ax * seg_x + ay * seg_y) / length2, 0.0, 1.0)
        t = np.where(length2 > 0, t, 0.0)
        ds = np.hypot(ax + t * seg_x, ay + t * seg_y)
        ds = np.where(ok, ds, np.inf)
        k = int(np.argmin(ds))
        if ds[k] < best_d:
            best_d, best_i = float(ds[k]), lo + k + (1 if t[k] >= 0.5 else 0)
    return (best_d, best_i) if best_d <= radius_px else None
