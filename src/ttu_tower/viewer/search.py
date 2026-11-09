"""Quantity search: every word of a query must appear in what a quantity is
called, with Greek letters spelled out (θ ≡ theta, so "theta_v" finds θv),
separators and primes ignored ("w theta v" finds w'θv'), and long names
standing in for symbols ("buoyancy flux" finds w'θv' and w'b').
"""
import re
import unicodedata

from ttu_tower.tertiary import gradients

_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "ϑ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "µ": "mu",
    "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "*": "star", "²": "2", "³": "3",
}

# long names of a variable, or of a piece of one ("sigma_w" -> sigma + w) when
# the whole variable has none
_NAMES = {
    "ue": "east wind component eastward", "vn": "north wind component northward",
    "u": "streamwise velocity", "v": "cross-stream velocity", "w": "vertical velocity",
    "ws": "horizontal wind speed", "veer": "wind veer direction change turning",
    "uv": "horizontal momentum flux reynolds stress", "uw": "vertical momentum flux reynolds stress",
    "vw": "vertical momentum flux reynolds stress",
    "wvpts": "buoyancy flux virtual heat flux sonic virtual potential temperature flux",
    "buoyancy_flux": "buoyancy flux", "wpt": "kinematic sensible heat flux potential temperature flux",
    "ts": "sonic temperature", "vpts": "sonic virtual potential temperature",
    "t": "air temperature", "pt": "potential temperature", "vt": "virtual temperature",
    "vpt": "virtual potential temperature", "td": "dewpoint dew point temperature",
    "rh": "relative humidity", "p": "pressure", "es": "saturation vapor vapour pressure",
    "e": "vapor vapour pressure", "r": "water vapor mixing ratio", "q": "water vapor specific humidity",
    "sigma": "standard deviation", "ti": "turbulence intensity", "tke": "turbulent kinetic energy",
    "ctke": "covariance turbulent kinetic energy", "ustar": "friction velocity",
    "obukhov_length": "monin-obukhov length", "zeta": "monin-obukhov stability parameter z/L",
    "theta_star": "temperature scale",
    "phi_m_bd": "dimensionless wind shear similarity stability function monin-obukhov",
    "phi_h_bd": "dimensionless temperature gradient similarity stability function monin-obukhov",
    "psi_m_bd": "integrated stability correction similarity function monin-obukhov",
    "psi_h_bd": "integrated stability correction similarity function monin-obukhov",
    "phi_m": "dimensionless wind shear similarity function",
    "phi_h": "dimensionless potential temperature gradient similarity function",
    "phi_h_v": "dimensionless virtual potential temperature vpt gradient similarity function",
    "dpt_dz": "potential temperature gradient lapse rate",
    "dvpt_dz": "virtual potential temperature gradient lapse rate",
    "dws_dz": "wind speed gradient wind shear", "shear": "wind shear vector",
    "n2": "brunt-väisälä buoyancy frequency squared", "rig": "gradient richardson number",
    "rif": "flux richardson number", "rib": "bulk richardson number",
    "km": "eddy viscosity momentum diffusivity", "kh": "eddy diffusivity heat diffusivity",
    "prandtl": "turbulent prandtl number","its": "integral time scale", "ils": "integral length scale",
    "alpha": "wind shear exponent power law", "gamma": "turbulence intensity exponent power law",
    "loglaw_ustar": "friction velocity", "loglaw_z0": "roughness length",
    "loglaw_z0_constrained": "roughness length", "h_bl_momt": "boundary layer height",
    "sun_elevation": "solar elevation angle", "tau": "averaging timescale time scale",
}
_STAT_NAMES = {"std": "standard deviation", "var": "variance", "cov": "covariance", "its": "integral time scale",
               "te": "transport efficiency"}


def fold(text: str) -> str:
    """Lower case, Greek letters spelled out, accents dropped."""
    text = "".join(_GREEK.get(c, c) for c in text.lower())
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", fold(text))


def terms(*texts: str) -> str:
    """The text a query is matched against, from what something is called."""
    return "|".join(_compact(t) for t in texts if t)


def long_names(variable: str, stat: str | None = None) -> str:
    """A variable's (and stat's) long names, for a quantity's search terms."""
    out = []
    for v in dict.fromkeys(fold(v).replace(" ", "_") for v in (variable, gradients.split(variable)[0])):
        out.append(_NAMES.get(v) or " ".join(_NAMES.get(p, "") for p in v.split("_")))
    if stat:
        out.append(_STAT_NAMES.get(stat, ""))
    return " ".join(o for o in out if o.strip())


def quantity_terms(q) -> str:
    """A catalog quantity's search terms: its title, names, group and long names."""
    return terms(q.title, q.label, q.variable, q.key, q.table, q.group, q.stat or "", long_names(q.variable, q.stat))


class Query:
    """A typed search: true where every word matches (empty: everything)."""

    def __init__(self, text: str):
        self.words = [w for w in (_compact(w) for w in fold(text).split()) if w]

    def __bool__(self) -> bool:
        return bool(self.words)

    def matches(self, haystack: str) -> bool:
        return all(w in haystack for w in self.words)
