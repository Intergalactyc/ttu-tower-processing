"""Fitting distributions to what a histogram shows: maximum-likelihood fits
per boom, their PDFs and CDFs to overlay, how well each fits (KS statistic,
AIC), and their formulas.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import special
from scipy import stats as sp_stats


@dataclass(frozen=True)
class _Spec:
    dist: object  # a scipy distribution
    params: tuple  # its parameters in scipy's order, as the table names them
    about: dict  # table name -> a few words on it
    formula: str  # the PDF in the table's symbols, as rich text
    filled: Callable  # the table's parameters -> the PDF with their values in, as rich text
    fixed: dict = field(default_factory=dict)  # scipy fit keywords holding parameters fixed
    domain: str = "any"  # "any", "positive", or "unit" (strictly between 0 and 1)
    fit_points: int | None = None  # estimate from at most this many values; KS and AIC still use them all
    shown: Callable | None = None  # scipy's parameters -> the table's, when they differ
    start: Callable | None = None  # values -> starting shape parameters and scale, or None


def _ncx2_start(v: np.ndarray):
    """Shapes and scale matching the first three moments: about twice as fast as scipy's own start."""
    mu, var = v.mean(), v.var()
    m3 = np.mean((v - mu) ** 3)
    disc = var ** 2 - mu * m3 / 2
    if disc <= 0:
        return None
    for s in ((var + np.sqrt(disc)) / (2 * mu), (var - np.sqrt(disc)) / (2 * mu)):
        if s > 0:
            nc = var / (2 * s * s) - mu / s
            df = mu / s - nc
            if df > 0 and nc > 0:
                return (df, nc), s
    return None


def _n(v: float) -> str:
    return f"{v:.4g}".replace("-", "−")


def _less(v: float) -> str:
    """"− v", written "+ |v|" when v is negative."""
    return f"− {_n(v)}" if v >= 0 else f"+ {_n(-v)}"


def _sup(v: float) -> str:
    return f"<sup>{_n(v)}</sup>"


def _normal(p):
    return f"f(x) = {_n(1 / (p['σ'] * _SQRT2PI))} exp[−(x {_less(p['μ'])})²/{_n(2 * p['σ'] ** 2)}]"


def _lognormal(p):
    c = 1 / (p["σ"] * _SQRT2PI)
    return f"f(x) = ({_n(c)}/x) exp[−(ln x {_less(p['μ'])})²/{_n(2 * p['σ'] ** 2)}]"


def _weibull(p):
    k, lam = p["k"], p["λ"]
    return f"f(x) = {_n(k / lam)} (x/{_n(lam)}){_sup(k - 1)} exp[−(x/{_n(lam)}){_sup(k)}]"


def _gamma(p):
    k, theta = p["k"], p["θ"]
    c = np.exp(-special.gammaln(k) - k * np.log(theta))
    return f"f(x) = {_n(c)} x{_sup(k - 1)} e<sup>−x/{_n(theta)}</sup>"


def _exponential(p):
    return f"f(x) = {_n(1 / p['β'])} e<sup>−x/{_n(p['β'])}</sup>"


def _rayleigh(p):
    return f"f(x) = {_n(1 / p['σ'] ** 2)} x exp[−x²/{_n(2 * p['σ'] ** 2)}]"


def _chi(p):
    k, s = p["k"], p["σ"]
    c = np.exp(-(k / 2 - 1) * np.log(2) - special.gammaln(k / 2) - k * np.log(s))
    return f"f(x) = {_n(c)} x{_sup(k - 1)} exp[−x²/{_n(2 * s ** 2)}]"


def _chi2(p):
    k, s2 = p["k"], p["σ²"]
    c = np.exp(-k / 2 * np.log(2 * s2) - special.gammaln(k / 2))
    return f"f(x) = {_n(c)} x{_sup(k / 2 - 1)} e<sup>−x/{_n(2 * s2)}</sup>"


def _ncx2(p):
    k, lam, s2 = p["k"], p["λ"], p["σ²"]
    return (f"f(x) = {_n(1 / (2 * s2))} exp[−(x/{_n(s2)} + {_n(lam)})/2] (x/{_n(lam * s2)}){_sup(k / 4 - 0.5)} "
            f"I<sub>{_n(k / 2 - 1)}</sub>({_n(np.sqrt(lam / s2))}√x)")


def _beta(p):
    a, b = p["α"], p["β"]
    return f"f(x) = {_n(np.exp(-special.betaln(a, b)))} x{_sup(a - 1)} (1 − x){_sup(b - 1)}"


def _uniform(p):
    return f"f(x) = {_n(1 / (p['b'] - p['a']))}, {_n(p['a'])} ≤ x ≤ {_n(p['b'])}"


def _von_mises(p):
    """Per degree, as the histogram's drawn; I₀ scaled, as it overflows for a large κ."""
    c = np.exp(-p["κ"] - np.log(360 * special.i0e(p["κ"])))
    return f"f(θ) = {_n(c)} exp[{_n(p['κ'])} cos(θ {_less(p['μ'])}°)], per degree"


_SQRT2PI = np.sqrt(2 * np.pi)
_LOC0 = {"floc": 0.0}
_VARIANCE = "each component's variance (scale)"
_DISTRIBUTIONS = {
    "normal": _Spec(sp_stats.norm, ("μ", "σ"), {"μ": "mean", "σ": "standard deviation"},
                    "f(x) = exp[−(x − μ)²/(2σ²)] / (σ√(2π))", _normal),
    "log-normal": _Spec(sp_stats.lognorm, ("σ", "loc", "scale"),
                        {"μ": "mean of ln x", "σ": "standard deviation of ln x"},
                        "f(x) = exp[−(ln x − μ)²/(2σ²)] / (xσ√(2π)), x &gt; 0", _lognormal,
                        _LOC0, "positive", shown=lambda p: {"μ": np.log(p[2]), "σ": p[0]}),
    "Weibull": _Spec(sp_stats.weibull_min, ("k", "loc", "λ"), {"k": "shape", "λ": "scale"},
                     "f(x) = (k/λ) (x/λ)<sup>k−1</sup> exp[−(x/λ)<sup>k</sup>], x ≥ 0", _weibull,
                     _LOC0, "positive"),
    "gamma": _Spec(sp_stats.gamma, ("k", "loc", "θ"), {"k": "shape", "θ": "scale"},
                   "f(x) = x<sup>k−1</sup> e<sup>−x/θ</sup> / (Γ(k) θ<sup>k</sup>), x &gt; 0", _gamma,
                   _LOC0, "positive"),
    "exponential": _Spec(sp_stats.expon, ("loc", "β"), {"β": "scale, the mean"},
                         "f(x) = e<sup>−x/β</sup> / β, x ≥ 0", _exponential, _LOC0, "positive"),
    "Rayleigh": _Spec(sp_stats.rayleigh, ("loc", "σ"), {"σ": "scale, the mode"},
                      "f(x) = (x/σ²) exp[−x²/(2σ²)], x ≥ 0", _rayleigh, _LOC0, "positive"),
    "chi": _Spec(
        sp_stats.chi, ("k", "loc", "σ"), {"k": "degrees of freedom", "σ": "each component's σ (scale)"},
        "f(x) = x<sup>k−1</sup> exp[−x²/(2σ²)] / (2<sup>k/2−1</sup> Γ(k/2) σ<sup>k</sup>), x ≥ 0", _chi,
        _LOC0, "positive"),
    "chi-squared": _Spec(
        sp_stats.chi2, ("k", "loc", "σ²"), {"k": "degrees of freedom", "σ²": _VARIANCE},
        "f(x) = x<sup>k/2−1</sup> e<sup>−x/(2σ²)</sup> / ((2σ²)<sup>k/2</sup> Γ(k/2)), x &gt; 0", _chi2,
        _LOC0, "positive"),
    "noncentral chi-squared": _Spec(
        sp_stats.ncx2, ("k", "λ", "loc", "σ²"),
        {"k": "degrees of freedom", "λ": "noncentrality", "σ²": _VARIANCE},
        "f(x) = exp[−(x/σ² + λ)/2] (x/(λσ²))<sup>k/4−1/2</sup> I<sub>k/2−1</sub>(√(λx)/σ) / (2σ²), "
        "x &gt; 0",
        _ncx2, _LOC0, "positive", fit_points=10_000, start=_ncx2_start),  # slow: ~1 s per 10k
    "beta": _Spec(sp_stats.beta, ("α", "β", "loc", "scale"), {"α": "shape", "β": "shape"},
                  "f(x) = x<sup>α−1</sup> (1 − x)<sup>β−1</sup> / B(α, β), 0 &lt; x &lt; 1", _beta,
                  {"floc": 0.0, "fscale": 1.0}, "unit"),
    "uniform": _Spec(sp_stats.uniform, ("loc", "scale"), {"a": "lower end", "b": "upper end"},
                     "f(x) = 1/(b − a), a ≤ x ≤ b", _uniform, shown=lambda p: {"a": p[0], "b": p[0] + p[1]}),
}
CIRCULAR = "von Mises"
_CIRCULAR_ABOUT = {"κ": "concentration", "μ": "mean direction, degrees"}
_CIRCULAR_FORMULA = "f(θ) = exp[κ cos(θ − μ)] / (2π I<sub>0</sub>(κ)), per radian"
_NEEDS = {"any": "", "positive": "every value positive", "unit": "every value between 0 and 1"}


@dataclass
class DistFit:
    name: str
    params: dict
    pdf: Callable[[np.ndarray], np.ndarray]
    cdf: Callable[[np.ndarray], np.ndarray]
    n: int
    ks: float
    aic: float


def names(circular: bool) -> list[str]:
    """Every fit on offer, whether or not the values suit it."""
    return [CIRCULAR] if circular else list(_DISTRIBUTIONS)


def needs(name: str) -> str:
    """What the values must be for `name` to fit ("" for anything)."""
    return _NEEDS[_DISTRIBUTIONS[name].domain] if name in _DISTRIBUTIONS else ""


def about(name: str, param: str) -> str:
    """A few words on one of `name`'s parameters, as the table names it."""
    words = _CIRCULAR_ABOUT if name == CIRCULAR else _DISTRIBUTIONS[name].about
    return words.get(param, "")


def formula(name: str) -> str:
    """`name`'s PDF, and what its parameters are, as rich text."""
    words = _CIRCULAR_ABOUT if name == CIRCULAR else _DISTRIBUTIONS[name].about
    pdf = _CIRCULAR_FORMULA if name == CIRCULAR else _DISTRIBUTIONS[name].formula
    return f"{pdf}<br>" + ", ".join(f"{p}: {w}" for p, w in words.items())


def filled(f: "DistFit") -> str:
    """The fit's PDF with its parameters' values in, as rich text."""
    return _von_mises(f.params) if f.name == CIRCULAR else _DISTRIBUTIONS[f.name].filled(f.params)


def _suits(v: np.ndarray, domain: str) -> bool:
    if domain == "any":
        return True
    if v.size == 0:
        return False
    return bool((v > 0).all() and (domain == "positive" or (v < 1).all()))


def available(values: np.ndarray, circular: bool) -> list[str]:
    if circular:
        return [CIRCULAR]
    finite = values[np.isfinite(values)]
    return [name for name, spec in _DISTRIBUTIONS.items() if _suits(finite, spec.domain)]


def fit(name: str, values: np.ndarray, max_points: int = 50_000) -> DistFit | None:
    """None when there's too little data, or values the distribution can't
    take. Directions are in degrees: the von Mises PDF comes back per degree,
    and its CDF runs from 0° to 360°.
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
        at_zero = float(frozen.cdf(0.0))  # scipy's CDF climbs by 1 every turn, so this starts it at 0°
        return DistFit(name, {"κ": float(kappa), "μ": float(np.degrees(loc) % 360)},
                       lambda x: frozen.pdf(np.radians(np.asarray(x, dtype=float))) * np.pi / 180.0,
                       lambda x: frozen.cdf(np.radians(np.asarray(x, dtype=float))) - at_zero,
                       int(v.size), ks, 2 * 2 - 2 * loglik)
    spec = _DISTRIBUTIONS[name]
    if not _suits(v, spec.domain):
        return None
    sample = v
    if spec.fit_points is not None and v.size > spec.fit_points:
        sample = np.random.default_rng(1).choice(v, spec.fit_points, replace=False)
    guess = spec.start(sample) if spec.start is not None else None
    if guess is not None:
        params = spec.dist.fit(sample, *guess[0], scale=guess[1], **spec.fixed)
    else:
        params = spec.dist.fit(sample, **spec.fixed)
    frozen = spec.dist(*params)
    loglik = float(np.sum(frozen.logpdf(v)))
    k = len(params) - len(spec.fixed)
    ks = float(sp_stats.kstest(v, frozen.cdf).statistic)
    if spec.shown is not None:
        shown = {n: float(p) for n, p in spec.shown(params).items()}
    else:
        shown = {n: float(p) for n, p in zip(spec.params, params) if f"f{n}" not in spec.fixed}  # fixed ones left out
    return DistFit(name, shown, frozen.pdf, frozen.cdf, int(v.size), ks, 2 * k - 2 * loglik)
