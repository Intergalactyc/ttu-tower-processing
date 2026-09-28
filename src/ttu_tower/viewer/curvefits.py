"""Fitting y against x for the scatter view: the curve fits of
ttu-windprofiles' curve_fits (limits, percentile limits and binning before
fitting), correlation statistics, and a binned central line.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

from ttu_tower.math.fits import constrained_linear_fit, log_fit, ls_linear_fit, power_fit
from ttu_tower.physics.constants import KAPPA
from ttu_tower.physics.most import neutral_loglaw_fit


@dataclass
class FitResult:
    func: Callable[[np.ndarray], np.ndarray]
    params: dict
    label: str
    n: int = 0  # points (or bins) fitted
    rmse: float = np.nan
    notes: list = field(default_factory=list)


@dataclass(frozen=True)
class FitOptions:
    lim_x: tuple[float, float] | None = None  # data limits (before percentiles)
    lim_y: tuple[float, float] | None = None
    pct_x: tuple[float, float] | None = None  # percentile limits
    pct_y: tuple[float, float] | None = None
    bin_count: int | None = None  # fit to binned centres instead of the raw points
    bin_method: str = "mean"
    bin_min_count: int = 1


def bin_average(x, y, bin_count: int, method: str = "mean", bin_range=None, min_count: int = 1):
    """Centres and mean/median of y in `bin_count` equal x bins (sparse bins dropped)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size == 0:
        return np.empty(0), np.empty(0)
    lo, hi = bin_range if bin_range is not None else (np.nanmin(x), np.nanmax(x))
    edges = np.linspace(lo, hi, bin_count + 1)
    stat, edges, _ = sp_stats.binned_statistic(x, y, statistic=method, bins=edges)
    count, _, _ = sp_stats.binned_statistic(x, y, statistic="count", bins=edges)
    centres = edges[:-1] + np.diff(edges) / 2
    keep = np.isfinite(stat) & (count >= min_count)
    return centres[keep], stat[keep]


def prepare(x, y, opts: FitOptions) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    for lim, pct, axis in ((opts.lim_x, opts.pct_x, 0), (opts.lim_y, opts.pct_y, 1)):
        v = x if axis == 0 else y
        if pct is not None and v.size:
            lim = tuple(np.percentile(v, pct))
        if lim is not None:
            keep = (v >= lim[0]) & (v <= lim[1])
            x, y = x[keep], y[keep]
    if opts.bin_count:
        x, y = bin_average(x, y, opts.bin_count, opts.bin_method, min_count=opts.bin_min_count)
    return x, y


# --- the fits (each core takes the prepared x, y) ------------------------------------------

def _linear(x, y):
    a, b = ls_linear_fit(x, y)
    return lambda v: a + b * v, {"a": a, "b": b}, f"y = {a:.4g} + {b:.4g}·x"


def _origin(x, y):
    _, b = constrained_linear_fit(x, y, a=0.0)
    return lambda v: b * v, {"b": b}, f"y = {b:.4g}·x"


def _power(x, y):
    keep = (x > 0) & (y > 0)
    a, b = power_fit(x[keep], y[keep])
    return lambda v: a * np.asarray(v) ** b, {"a": a, "b": b}, f"y = {a:.4g}·x^{b:.4g}"


def _log(x, y):
    keep = x > 0
    a, b = log_fit(x[keep], y[keep])
    return lambda v: a + b * np.log(v), {"a": a, "b": b}, f"y = {a:.4g} + {b:.4g}·ln x"


def _exponential(x, y):
    keep = y > 0
    ln_a, ln_b = ls_linear_fit(x[keep], np.log(y[keep]))
    A, B = np.exp(ln_a), np.exp(ln_b)
    return lambda v: A * B ** np.asarray(v), {"A": A, "B": B}, f"y = {A:.4g}·{B:.4g}^x"


def _loglaw(x, y):
    keep = x > 0
    ustar, z0 = neutral_loglaw_fit(x[keep], y[keep])
    return (lambda v: ustar / KAPPA * np.log(np.asarray(v) / z0), {"ustar": ustar, "z0": z0},
            f"y = u*/κ·ln(x/z0): u* = {ustar:.4g}, z0 = {z0:.4g}")


def _double_linear_model(x, A, B, C, D):
    return (B - A) * C * (np.exp(-x / C) - 1) + B * x + D


def _double_linear(x, y, intercept=None):
    a0, b0 = ls_linear_fit(x, y)
    c0 = max((x.max() - x.min()) / 4, 1e-6)
    if intercept is None:
        (A, B, C, D), *_ = curve_fit(_double_linear_model, x, y, p0=[b0, b0, c0, a0], maxfev=5000,
                                     bounds=([-np.inf, -np.inf, 1e-9, -np.inf], [np.inf] * 4))
    else:
        D = intercept
        (A, B, C), *_ = curve_fit(lambda v, A, B, C: _double_linear_model(v, A, B, C, D), x, y, p0=[b0, b0, c0],
                                  maxfev=5000, bounds=([-np.inf, -np.inf, 1e-9], [np.inf] * 3))
    return (lambda v: _double_linear_model(np.asarray(v, dtype=float), A, B, C, D),
            {"A": A, "B": B, "C": C, "D": D}, f"double linear: slopes {A:.4g} → {B:.4g}, scale {C:.4g}, y(0) {D:.4g}")


def _sparam_ri(x, y):
    """ζ against Ri: x<0: y = K·x; x≥0: y = K·x/(1 + B·x)."""
    _, k0 = constrained_linear_fit(x[x < 0], y[x < 0], a=0.0) if (x < 0).any() else (0.0, 1.0)

    def model(v, K, B):
        return np.where(v < 0, K * v, K * v / (1 + B * v))

    (K, B), *_ = curve_fit(model, x, y, p0=[k0, 1.0], maxfev=2000)
    return lambda v: model(np.asarray(v, dtype=float), K, B), {"K": K, "B": B}, f"x<0: y = {K:.4g}x; x≥0: y = {K:.4g}x/(1+{B:.4g}x)"


def _alpha_ri(x, y, cutoffs=(-0.05, 0.05), critical=0.25):
    """α against Ri: α0(1 + a·Ri)^b either side of neutral, constant above Ri_c."""
    alpha0 = float(y[(x > cutoffs[0]) & (x < cutoffs[1])].mean())

    def model(ri, a, b):
        return alpha0 * (1 + a * ri) ** b

    def fit(mask, guess):
        params, *_ = curve_fit(model, x[mask], y[mask], p0=guess, maxfev=2000, bounds=([-50.0, -1.0], [50.0, 1.0]))
        return params

    a_s, b_s = fit((x > 0) & (x < critical), [1.0, 1.0])
    a_u, b_u = fit(x < 0, [-0.5, -0.5])
    crit = alpha0 * (1 + a_s * critical) ** b_s

    def func(ri):
        ri = np.asarray(ri, dtype=float)
        return np.where(ri >= critical, crit, np.where(ri > 0, model(ri, a_s, b_s), model(ri, a_u, b_u)))

    return (func, {"alpha0": alpha0, "a_stable": a_s, "b_stable": b_s, "a_unstable": a_u, "b_unstable": b_u,
                   "alpha_critical": crit},
            f"α0 {alpha0:.3g}; stable ({a_s:.3g}, {b_s:.3g}); unstable ({a_u:.3g}, {b_u:.3g}); Ri ≥ {critical:g}: {crit:.3g}")


FITS: dict[str, Callable] = {
    "linear": _linear,
    "linear through origin": _origin,
    "power law": _power,
    "logarithmic": _log,
    "exponential": _exponential,
    "neutral log law (x = height)": _loglaw,
    "double linear": _double_linear,
    "double linear through origin": lambda x, y: _double_linear(x, y, intercept=0.0),
    "ζ(Ri) (stability parameter)": _sparam_ri,
    "α(Ri) (shear exponent)": _alpha_ri,
}


def fit(name: str, x, y, opts: FitOptions = FitOptions()) -> FitResult:
    xf, yf = prepare(x, y, opts)
    if xf.size < 2:
        raise ValueError("fewer than two points to fit")
    func, params, label = FITS[name](xf, yf)
    with np.errstate(all="ignore"):
        resid = yf - func(xf)
    rmse = float(np.sqrt(np.nanmean(resid ** 2))) if resid.size else np.nan
    return FitResult(func=func, params={k: float(v) for k, v in params.items()}, label=label, n=int(xf.size), rmse=rmse)


# --- statistics --------------------------------------------------------------------------------

def cohens_kappa(x, y) -> float:
    """Agreement of the signs of x and y beyond chance."""
    a = np.sum((x > 0) & (y > 0))
    b = np.sum((x > 0) & (y < 0))
    c = np.sum((x < 0) & (y > 0))
    d = np.sum((x < 0) & (y < 0))
    n = a + b + c + d
    if n == 0:
        return np.nan
    po = (a + d) / n
    pe = (a + b) / n * (a + c) / n + (c + d) / n * (b + d) / n
    return float((po - pe) / (1 - pe)) if not np.isclose(pe, 1.0) else np.nan


def correlation(x, y) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"N": int(x.size), "r": np.nan, "ρ": np.nan, "κ": cohens_kappa(x, y)}
    return {"N": int(x.size), "r": float(sp_stats.pearsonr(x, y).statistic),
            "ρ": float(sp_stats.spearmanr(x, y).statistic), "κ": cohens_kappa(x, y)}
