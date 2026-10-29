"""Fitting y against x for the scatter view: the curve fits of
ttu-windprofiles' curve_fits (limits, percentile limits and binning before
fitting), correlation statistics, and a binned central line.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit, least_squares

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
    bin_log: bool = False  # bins equal in log x (only positive x is binned)
    bin_range: tuple[float, float] | None = None  # the bins' span (else the points' x span)
    errors: str = "y"  # least squares in "y", or in "ln y" (relative errors; positive y only)


def bin_average(x, y, bin_count: int, method: str = "mean", bin_range=None, min_count: int = 1, log: bool = False):
    """Centres and mean/median of y in `bin_count` equal x bins (sparse bins
    dropped); with `log`, bins equal in log x with geometric centres, over the
    positive x only.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if log:
        keep = x > 0
        x, y = x[keep], y[keep]
    if x.size == 0:
        return np.empty(0), np.empty(0)
    lo, hi = bin_range if bin_range is not None else (np.nanmin(x), np.nanmax(x))
    if log:
        lo = max(lo, np.nanmin(x))
        edges = np.geomspace(lo, hi, bin_count + 1) if hi > lo else np.array([lo, lo * 1.0001])
    else:
        edges = np.linspace(lo, hi, bin_count + 1)
    stat, edges, _ = sp_stats.binned_statistic(x, y, statistic=method, bins=edges)
    count, _, _ = sp_stats.binned_statistic(x, y, statistic="count", bins=edges)
    centres = np.sqrt(edges[:-1] * edges[1:]) if log else edges[:-1] + np.diff(edges) / 2
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
        x, y = bin_average(x, y, opts.bin_count, opts.bin_method, bin_range=opts.bin_range,
                           min_count=opts.bin_min_count, log=opts.bin_log)
    return x, y


def rmse(f, y) -> tuple[float, int]:
    """(RMSE of y about the curve's values f, N) where both are finite."""
    f, y = np.asarray(f, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(f) & np.isfinite(y)
    return (float(np.sqrt(np.mean((y[ok] - f[ok]) ** 2))) if ok.any() else np.nan), int(ok.sum())


# --- the fits (each core takes the prepared x, y; `relative`: least squares in ln y) ---------

def _in_ln_y(model, x, y, p0, lower=None, upper=None) -> tuple:
    """The parameters of `model(x, *p)` minimising Σ(ln y − ln f)² over the positive y, from p0
    (its fit in y): relative errors, so every decade of y weighs alike.
    """
    keep = y > 0
    x, ly = x[keep], np.log(y[keep])
    if ly.size < len(p0):
        raise ValueError(f"only {ly.size} points (or bins) with y > 0, which errors in ln y need")
    if not np.all(np.isfinite(p0)):
        raise ValueError("the fit in y it starts from failed")
    eps = 1e-6 * float(np.exp(np.median(ly)))

    def resid(p):
        with np.errstate(all="ignore"):
            f = np.asarray(model(x, *p), dtype=float)
        # below eps, ln f goes on as its tangent: a curve dipping to 0 or below is pushed back up
        return np.where(f > eps, np.log(np.maximum(f, eps)), np.log(eps) + (f - eps) / eps) - ly

    n = len(p0)
    res = least_squares(resid, np.asarray(p0, dtype=float), x_scale="jac",
                        bounds=(lower if lower is not None else [-np.inf] * n, upper if upper is not None else [np.inf] * n))
    with np.errstate(all="ignore"):
        if not (np.asarray(model(x, *res.x), dtype=float) > 0).all():
            raise ValueError("no curve of this form stays above 0 at every point with y > 0, as errors in ln y need")
    return tuple(float(v) for v in res.x)


def _linear(x, y, relative=False):
    a, b = ls_linear_fit(x, y)
    if relative:
        a, b = _in_ln_y(lambda v, a, b: a + b * v, x, y, [a, b])
    return lambda v: a + b * v, {"a": a, "b": b}, f"y = {a:.4g} + {b:.4g}·x"


def _constrained(x, y, a=0.0, relative=False):
    """y = a + b·x with the intercept a given."""
    _, b = constrained_linear_fit(x, y, a=a)
    if relative:
        (b,) = _in_ln_y(lambda v, b: a + b * v, x, y, [b])
    label = f"y = {b:.4g}·x" if a == 0 else f"y = {a:g} + {b:.4g}·x (intercept fixed)"
    return lambda v: a + b * np.asarray(v, dtype=float), {"a": a, "b": b}, label


def _power(x, y, relative=False):
    """In ln y, weighted by y² (close to errors in y), or unweighted when `relative`."""
    keep = (x > 0) & (y > 0)
    if relative:
        ln_a, b = ls_linear_fit(np.log(x[keep]), np.log(y[keep]))
        a = float(np.exp(ln_a))
    else:
        a, b = power_fit(x[keep], y[keep])
    return lambda v: a * np.asarray(v) ** b, {"a": a, "b": b}, f"y = {a:.4g}·x^{b:.4g}"


def _power_fixed(x, y, b=0.5, relative=False):
    """y = a·x^b with b given: least squares in y (as the constrained linear fit),
    a = Σ y·x^b / Σ x^2b, over the positive x; when `relative`, in ln y,
    ln a = mean(ln y − b·ln x), over the positive x and y.
    """
    keep = x > 0
    if relative:
        keep &= y > 0
        if not keep.any():
            raise ValueError("no point with positive x and y to fit")
        a = float(np.exp(np.mean(np.log(y[keep]) - b * np.log(x[keep]))))
    else:
        xb = x[keep] ** b
        denom = np.sum(xb * xb)
        if not np.isfinite(denom) or denom == 0:
            raise ValueError("no positive x to fit")
        a = float(np.sum(y[keep] * xb) / denom)
    return lambda v: a * np.asarray(v, dtype=float) ** b, {"a": a, "b": b}, f"y = {a:.4g}·x^{b:g} (exponent fixed)"


def _log(x, y, relative=False):
    keep = x > 0
    a, b = log_fit(x[keep], y[keep])
    if relative:
        a, b = _in_ln_y(lambda v, a, b: a + b * np.log(v), x[keep], y[keep], [a, b])
    return lambda v: a + b * np.log(v), {"a": a, "b": b}, f"y = {a:.4g} + {b:.4g}·ln x"


def _exponential(x, y):
    keep = y > 0
    ln_a, ln_b = ls_linear_fit(x[keep], np.log(y[keep]))
    A, B = np.exp(ln_a), np.exp(ln_b)
    return lambda v: A * B ** np.asarray(v), {"A": A, "B": B}, f"y = {A:.4g}·{B:.4g}^x"


def _loglaw(x, y, relative=False):
    keep = x > 0
    ustar, z0 = neutral_loglaw_fit(x[keep], y[keep])
    if relative:
        ustar, z0 = _in_ln_y(lambda v, u, z: u / KAPPA * np.log(v / z), x[keep], y[keep], [ustar, z0],
                             lower=[-np.inf, 1e-12])
    return (lambda v: ustar / KAPPA * np.log(np.asarray(v) / z0), {"ustar": ustar, "z0": z0},
            f"y = u*/κ·ln(x/z0): u* = {ustar:.4g}, z0 = {z0:.4g}")


def _double_linear_model(x, A, B, C, D):
    return (B - A) * C * (np.exp(-x / C) - 1) + B * x + D


def _double_linear(x, y, intercept=None, relative=False):
    a0, b0 = ls_linear_fit(x, y)
    c0 = max((x.max() - x.min()) / 4, 1e-6)
    if intercept is None:
        (A, B, C, D), *_ = curve_fit(_double_linear_model, x, y, p0=[b0, b0, c0, a0], maxfev=5000,
                                     bounds=([-np.inf, -np.inf, 1e-9, -np.inf], [np.inf] * 4))
    else:
        D = intercept
        (A, B, C), *_ = curve_fit(lambda v, A, B, C: _double_linear_model(v, A, B, C, D), x, y, p0=[b0, b0, c0],
                                  maxfev=5000, bounds=([-np.inf, -np.inf, 1e-9], [np.inf] * 3))
    if relative and intercept is None:
        A, B, C, D = _in_ln_y(_double_linear_model, x, y, [A, B, C, D], lower=[-np.inf, -np.inf, 1e-9, -np.inf])
    elif relative:
        A, B, C = _in_ln_y(lambda v, A, B, C: _double_linear_model(v, A, B, C, D), x, y, [A, B, C],
                           lower=[-np.inf, -np.inf, 1e-9])
    return (lambda v: _double_linear_model(np.asarray(v, dtype=float), A, B, C, D),
            {"A": A, "B": B, "C": C, "D": D}, f"double linear: slopes {A:.4g} → {B:.4g}, scale {C:.4g}, y(0) {D:.4g}")


def _double_power_model(lx, la, b1, b2, lxb, width):
    """ln y of a smoothly broken power law, from ln x: a·(x/x_b)^b1·(1 + (x/x_b)^(1/Δ))^((b2-b1)·Δ),
    slope b1 well below the break x_b and b2 well above it, over a width Δ in ln x.
    """
    d = lx - lxb
    return la + b1 * d + (b2 - b1) * width * np.logaddexp(0.0, d / width)


def _double_power(x, y):
    """Fitted in log space (positive x and y), so each decade of x weighs alike."""
    keep = (x > 0) & (y > 0)
    if keep.sum() < 5:
        raise ValueError("fewer than five points with positive x and y")
    lx, ly = np.log(x[keep]), np.log(y[keep])
    a0, b0 = ls_linear_fit(lx, ly)
    lo, hi = float(lx.min()), float(lx.max())
    mid = float(np.median(lx))
    span = max(hi - lo, 1e-6)
    (la, b1, b2, lxb, width), *_ = curve_fit(
        _double_power_model, lx, ly, p0=[a0 + b0 * mid, b0, b0, mid, span / 8], maxfev=10_000,
        bounds=([-np.inf, -np.inf, -np.inf, lo, span / 200], [np.inf, np.inf, np.inf, hi, span]))
    a, xb = float(np.exp(la)), float(np.exp(lxb))

    def func(v):
        v = np.asarray(v, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, np.exp(_double_power_model(np.log(v), la, b1, b2, lxb, width)), np.nan)

    return (func, {"a": a, "b1": b1, "b2": b2, "x_b": xb, "width": width},
            f"double power law: x^{b1:.3g} → x^{b2:.3g} about x_b {xb:.4g} (a {a:.4g}, width {width:.3g} in ln x)")


def _sparam_ri(x, y):
    """ζ against Ri: x<0: y = K·x; x≥0: y = K·x/(1 + B·x)."""
    _, k0 = constrained_linear_fit(x[x < 0], y[x < 0], a=0.0) if (x < 0).any() else (0.0, 1.0)

    def model(v, K, B):
        return np.where(v < 0, K * v, K * v / (1 + B * v))

    (K, B), *_ = curve_fit(model, x, y, p0=[k0, 1.0], maxfev=2000)
    return lambda v: model(np.asarray(v, dtype=float), K, B), {"K": K, "B": B}, f"x<0: y = {K:.4g}x; x≥0: y = {K:.4g}x/(1+{B:.4g}x)"


ALPHA_RI = "alpha-Ri: α(Ri), shear exponent vs Ri"


def _alpha_ri(x, y, ri_neutral_lo=-0.05, ri_neutral_hi=0.05, ri_critical=0.25):
    """ttu-windprofiles' fit_alpha_ri: α0 (the mean α over the neutral band)
    times (1 + a·Ri)^b, fitted separately for 0 < Ri < Ri_c and Ri < 0, and
    held at its Ri_c value above Ri_c.
    """
    neutral = (x > ri_neutral_lo) & (x < ri_neutral_hi)
    stable, unstable = (x > 0) & (x < ri_critical), x < 0
    for name, mask in (("neutral", neutral), ("stable (0 < Ri < Ri_c)", stable), ("unstable (Ri < 0)", unstable)):
        if mask.sum() < 3:
            raise ValueError(f"only {int(mask.sum())} points (or bins) in the {name} range; widen it, or bin less finely")
    alpha0 = float(y[neutral].mean())

    def model(ri, a, b):
        return alpha0 * (1 + a * ri) ** b

    def fit(mask, guess):
        params, *_ = curve_fit(model, x[mask], y[mask], p0=guess, maxfev=2000, bounds=([-50.0, -1.0], [50.0, 1.0]))
        return params

    a_s, b_s = fit(stable, [1.0, 1.0])
    a_u, b_u = fit(unstable, [-0.5, -0.5])
    crit = alpha0 * (1 + a_s * ri_critical) ** b_s
    high = x >= ri_critical
    mean_high = float(y[high].mean()) if high.any() else np.nan

    def func(ri):
        ri = np.asarray(ri, dtype=float)
        return np.where(ri >= ri_critical, crit, np.where(ri > 0, model(ri, a_s, b_s), model(ri, a_u, b_u)))

    return (func, {"alpha0": alpha0, "a_stable": a_s, "b_stable": b_s, "a_unstable": a_u, "b_unstable": b_u,
                   "alpha_critical": crit, "mean_high_ri": mean_high},
            f"α0 {alpha0:.3g}; 0<Ri<{ri_critical:g}: α0(1+{a_s:.3g}Ri)^{b_s:.3g}; Ri≥{ri_critical:g}: {crit:.3g}; "
            f"Ri≤0: α0(1{a_u:+.3g}Ri)^{b_u:.3g}")


FITS: dict[str, Callable] = {
    "linear": _linear,
    "constrained linear": _constrained,
    "power law": _power,
    "power law, fixed exponent": _power_fixed,
    "double power law": _double_power,
    "logarithmic": _log,
    "exponential": _exponential,
    "neutral log law (x = height)": _loglaw,
    "double linear": _double_linear,
    "constrained double linear": lambda x, y, D=0.0, relative=False: _double_linear(x, y, intercept=D,
                                                                                    relative=relative),
    "ζ(Ri) (stability parameter)": _sparam_ri,
    ALPHA_RI: _alpha_ri,
}

# fits with settings of their own: name -> {argument: (label, default)}
FIT_SETTINGS: dict[str, dict[str, tuple[str, float]]] = {
    "constrained linear": {"a": ("intercept", 0.0)},
    "constrained double linear": {"D": ("intercept", 0.0)},
    "power law, fixed exponent": {"b": ("exponent", 0.5)},
    ALPHA_RI: {"ri_neutral_lo": ("neutral band from Ri", -0.05), "ri_neutral_hi": ("to Ri", 0.05),
               "ri_critical": ("Ri_c (α held above)", 0.25)},
}

IN_LN_Y = {"exponential", "double power law"}  # always fitted in ln y
IN_Y = {"ζ(Ri) (stability parameter)", ALPHA_RI}  # always in y: ζ changes sign; α-Ri is fitted in pieces


def errors_in(name: str | None, errors: str = "y") -> str:
    """What a fit's least squares are in: "y" or "ln y" as chosen, unless the curve has its own."""
    return "ln y" if name in IN_LN_Y else "y" if name in IN_Y else errors


RENAMED = {"linear through origin": ("constrained linear", {"a": "0"}),  # old name -> (name, its settings)
           "double linear through origin": ("constrained double linear", {"D": "0"})}


def _n(v: float) -> str:
    """A coefficient for a formula: 6 significant figures, bracketed if negative."""
    return f"{v:.6g}" if v >= 0 else f"({v:.6g})"


def fit_formula(name: str, params: dict, settings: dict | None = None) -> list[tuple[str, str, str]]:
    """The fitted curve as formulas for `expression`: [(y in x, from x, to x)], "" for no bound."""
    p, s = {k: _n(v) for k, v in params.items()}, settings or {}
    simple = {
        "linear": lambda: f"{p['a']} + {p['b']}*x",
        "constrained linear": lambda: f"{p['a']} + {p['b']}*x",
        "power law": lambda: f"{p['a']}*x^{p['b']}",
        "power law, fixed exponent": lambda: f"{p['a']}*x^{p['b']}",
        "double power law": lambda: (f"{p['a']}*(x/{p['x_b']})^{p['b1']}*(1 + (x/{p['x_b']})^(1/{p['width']}))"
                                     f"^(({p['b2']} - {p['b1']})*{p['width']})"),
        "logarithmic": lambda: f"{p['a']} + {p['b']}*ln(x)",
        "exponential": lambda: f"{p['A']}*{p['B']}^x",
        "neutral log law (x = height)": lambda: f"{p['ustar']}/{_n(KAPPA)}*ln(x/{p['z0']})",
        "double linear": lambda: f"({p['B']} - {p['A']})*{p['C']}*(exp(-x/{p['C']}) - 1) + {p['B']}*x + {p['D']}",
        "constrained double linear": lambda: (f"({p['B']} - {p['A']})*{p['C']}*(exp(-x/{p['C']}) - 1) "
                                              f"+ {p['B']}*x + {p['D']}"),
    }
    if name in simple:
        return [(simple[name](), "", "")]
    if name == "ζ(Ri) (stability parameter)":
        return [(f"{p['K']}*x", "", "0"), (f"{p['K']}*x/(1 + {p['B']}*x)", "0", "")]
    if name == ALPHA_RI:
        ri_c = _n(s.get("ri_critical", 0.25))
        return [(f"{p['alpha0']}*(1 + {p['a_unstable']}*x)^{p['b_unstable']}", "", "0"),
                (f"{p['alpha0']}*(1 + {p['a_stable']}*x)^{p['b_stable']}", "0", ri_c),
                (p["alpha_critical"], ri_c, "")]
    raise ValueError(f"no formula for {name}")


def fit(name: str, x, y, opts: FitOptions = FitOptions(), **settings) -> FitResult:
    xf, yf = prepare(x, y, opts)
    if xf.size < 2:
        raise ValueError("fewer than two points to fit")
    in_ln_y = errors_in(name, opts.errors) == "ln y"
    relative = {"relative": True} if in_ln_y and name not in IN_LN_Y else {}
    func, params, label = FITS[name](xf, yf, **settings, **relative)
    with np.errstate(all="ignore"):
        resid = yf - func(xf)
    rmse = float(np.sqrt(np.nanmean(resid ** 2))) if resid.size else np.nan
    notes = []
    if in_ln_y and (yf <= 0).any():
        notes.append(f"{int((yf <= 0).sum())} {'bins' if opts.bin_count else 'points'} with y ≤ 0 left out (ln y)")
    return FitResult(func=func, params={k: float(v) for k, v in params.items()}, label=label, n=int(xf.size), rmse=rmse,
                     notes=notes)


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
