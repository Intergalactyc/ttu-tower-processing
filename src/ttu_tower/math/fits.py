"""Least-squares fits: linear, weighted, power-law and log. Pure numpy.
NaN pairs/triples are dropped before fitting; non-finite inputs elsewhere are
the caller's responsibility.
"""
import numpy as np


def _drop_nan_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def _drop_nan_triples(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(w))
    return x[mask], y[mask], w[mask]


def ls_linear_fit(xvals, yvals) -> tuple[float, float]:
    """Least-squares fit to y = a + b*x. Returns (a, b), or (0.0, 0.0) if
    fewer than two (x, y) pairs are both finite - too few points to fit a
    line, and `det` would be zero anyway.
    """
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0, 0.0
    x, y = _drop_nan_pairs(x, y)
    if x.size < 2:
        return 0.0, 0.0
    n = x.size
    sum_x = x.sum()
    sum_x2 = np.sum(x * x)
    sum_xy = np.sum(x * y)
    sum_y = y.sum()
    det = n * sum_x2 - sum_x * sum_x
    a = (sum_y * sum_x2 - sum_x * sum_xy) / det
    b = (n * sum_xy - sum_x * sum_y) / det
    return float(a), float(b)


def ls_weighted_linear_fit(xvals, yvals, weights) -> tuple[float, float]:
    """Weighted least-squares fit to y = a + b*x. Returns (a, b), or
    (0.0, 0.0) if fewer than two (x, y, w) triples are all finite.
    """
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0, 0.0
    x, y, w = _drop_nan_triples(x, y, w)
    if x.size < 2:
        return 0.0, 0.0
    sum_w = w.sum()
    sum_wx = np.sum(w * x)
    sum_wx2 = np.sum(w * x * x)
    sum_wy = np.sum(w * y)
    sum_wxy = np.sum(w * x * y)
    det = sum_w * sum_wx2 - sum_wx * sum_wx
    a = (sum_wy * sum_wx2 - sum_wx * sum_wxy) / det
    b = (sum_w * sum_wxy - sum_wx * sum_wy) / det
    return float(a), float(b)


def constrained_linear_fit(xvals, yvals, a: float | None = None, b: float | None = None) -> tuple[float, float]:
    """ls_linear_fit with either the intercept a or the slope b fixed."""
    if a is None and b is None:
        raise ValueError("Either a or b must be specified (for unconstrained, use ls_linear_fit)")
    if a is not None and b is not None:
        raise ValueError("Only one of a or b may be specified")
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    x, y = _drop_nan_pairs(x, y)
    n = x.size
    if a is not None:
        if n == 0:
            return a, 0.0
        sum_x2 = np.sum(x * x)
        sum_xdy = np.sum(x * (y - a))
        return a, float(sum_xdy / sum_x2)
    if n == 0:
        return 0.0, b
    a_fit = (y.sum() - b * x.sum()) / n
    return float(a_fit), b


def power_fit(xvals, yvals, require: int = 2) -> tuple[float, float]:
    """Least-squares fit to y = a*x^b, weighted by y^2 in log space so squared
    residuals are approximately minimized in real y-space (an unweighted
    log-space fit underweights large-y points). b is the wind-shear exponent
    for a wind-speed power-law profile fit.
    """
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    x, y = _drop_nan_pairs(x, y)
    if np.any(y == 0):
        return 0.0, np.nan
    if x.size < require:
        return np.nan, np.nan
    weights = y * y
    ln_a, b = ls_weighted_linear_fit(np.log(x), np.log(y), weights)
    return float(np.exp(ln_a)), b


def log_fit(xvals, yvals) -> tuple[float, float]:
    """Least-squares fit to y = a + b*log(x)."""
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    x, y = _drop_nan_pairs(x, y)
    if np.any(x <= 0):
        raise ValueError("Cannot do log fit with nonpositive x values")
    return ls_linear_fit(np.log(x), y)


def constrained_log_fit(xvals, yvals, a: float | None = None, b: float | None = None) -> tuple[float, float]:
    """log_fit with either a or b fixed."""
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    x, y = _drop_nan_pairs(x, y)
    if np.any(x <= 0):
        raise ValueError("Cannot do log fit with nonpositive x values")
    return constrained_linear_fit(np.log(x), y, a=a, b=b)
