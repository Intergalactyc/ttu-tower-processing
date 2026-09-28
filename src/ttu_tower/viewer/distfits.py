"""Fitting distributions to what a histogram shows: maximum-likelihood fits
per boom, their PDFs to overlay, and how well each fits (KS statistic, AIC).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats as sp_stats

# name -> (scipy distribution, fit keyword arguments, needs positive values)
_DISTRIBUTIONS = {
    "normal": (sp_stats.norm, {}, False),
    "log-normal": (sp_stats.lognorm, {"floc": 0.0}, True),
    "Weibull": (sp_stats.weibull_min, {"floc": 0.0}, True),
    "gamma": (sp_stats.gamma, {"floc": 0.0}, True),
}
CIRCULAR = "von Mises"
_PARAM_NAMES = {"normal": ("μ", "σ"), "log-normal": ("s", "loc", "scale"), "Weibull": ("k", "loc", "λ"),
                "gamma": ("a", "loc", "scale")}


@dataclass
class DistFit:
    name: str
    params: dict
    pdf: Callable[[np.ndarray], np.ndarray]
    n: int
    ks: float
    aic: float


def available(values: np.ndarray, circular: bool) -> list[str]:
    if circular:
        return [CIRCULAR]
    finite = values[np.isfinite(values)]
    positive = finite.size > 0 and (finite > 0).all()
    return [name for name, (_, _, pos) in _DISTRIBUTIONS.items() if positive or not pos]


def fit(name: str, values: np.ndarray, max_points: int = 50_000) -> DistFit | None:
    """None when there's too little data. Directions are in degrees; the von
    Mises PDF comes back per degree.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return None
    if v.size > max_points:  # a year of 10 booms fits in a second this way; the estimates barely move
        v = np.random.default_rng(0).choice(v, max_points, replace=False)
    if name == CIRCULAR:
        rad = np.radians(v)
        kappa, loc, _ = sp_stats.vonmises.fit(rad, fscale=1.0)
        frozen = sp_stats.vonmises(kappa, loc=loc)
        loglik = float(np.sum(frozen.logpdf(rad)))
        ks = float(sp_stats.kstest(np.mod(rad - loc + np.pi, 2 * np.pi) - np.pi,
                                   sp_stats.vonmises(kappa).cdf).statistic)
        return DistFit(name, {"κ": float(kappa), "mean [deg]": float(np.degrees(loc) % 360)},
                       lambda x: frozen.pdf(np.radians(np.asarray(x, dtype=float))) * np.pi / 180.0,
                       int(v.size), ks, 2 * 2 - 2 * loglik)
    dist, kwargs, positive = _DISTRIBUTIONS[name]
    if positive and (v <= 0).any():
        return None
    params = dist.fit(v, **kwargs)
    frozen = dist(*params)
    loglik = float(np.sum(frozen.logpdf(v)))
    k = len(params) - len(kwargs)
    ks = float(sp_stats.kstest(v, frozen.cdf).statistic)
    names = _PARAM_NAMES[name]
    shown = {n: float(p) for n, p in zip(names, params) if not (n == "loc" and kwargs.get("floc") is not None)}
    return DistFit(name, shown, frozen.pdf, int(v.size), ks, 2 * k - 2 * loglik)
