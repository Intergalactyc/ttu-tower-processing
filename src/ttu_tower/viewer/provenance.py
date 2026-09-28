"""What a clicked quantity was computed from: which high-frequency view (and
which series in it) explains a plotted value.
"""
from dataclasses import dataclass

from ttu_tower.viewer.catalog import Quantity

_SONIC = ("ue", "vn", "w")
_WIND = _SONIC


@dataclass(frozen=True)
class DrillTarget:
    kind: str  # "series" | "fluctuation" | "flux" | "acf" | "spectra" | "profile" | "qc" | "none"
    variables: tuple[str, ...]  # the 50-Hz series to show, emphasised first
    spectrum: str | None = None  # MRD (co)spectrum, for flux/spectra targets


_FLUXES = {"uw": ("w", "ue", "vn"), "vw": ("w", "ue", "vn"), "uv": ("ue", "vn"), "wvpts": ("w", "vpts"),
           "ustar": ("w", "ue", "vn"), "obukhov_length": ("w", "vpts", "ue", "vn"), "zeta": ("w", "vpts", "ue", "vn")}
_FLUX_SPECTRUM = {"uw": "uw", "vw": "vw", "uv": "uv", "wvpts": "wvpts", "ustar": "uw", "obukhov_length": "wvpts",
                  "zeta": "wvpts"}
_SCALES = {"u": _WIND, "v": _WIND, "w": ("w",), "vpts": ("vpts",)}


def drill_target(q: Quantity) -> DrillTarget:
    v, stat = q.variable, q.stat
    if q.kind != "boom" or q.table in ("slot_final",):
        return DrillTarget("profile" if q.table in ("pairs", "profile") else "none", ())
    if q.table in ("tau_final", "tau"):
        return DrillTarget("spectra", ("w", "vpts") if "heat" in v else ("w", "ue", "vn"),
                           spectrum="uw" if "momentum" in v else "wvpts")
    if q.table in ("coverage", "slot_qc", "flags", "slot_boom"):
        variables = _WIND if v in ("momentum", "ws", "wd") else (("w", "vpts") if v == "heat" else (v,))
        return DrillTarget("qc", variables)
    if q.table == "mrd_frame":
        return DrillTarget("series", _WIND)

    if stat == "its" or v.startswith(("ils_", "its_ratio_", "its_short")):
        base = v.split("_")[-1] if v.startswith(("ils_", "its_ratio_")) else v
        if v.startswith("its_short"):
            base = "vpts" if v.endswith("vpts") else "u"
        return DrillTarget("acf", _SCALES.get(base, _WIND))
    if stat == "te" or v in _FLUXES or stat == "cov":
        return DrillTarget("flux", _FLUXES.get(v, _WIND), spectrum=_FLUX_SPECTRUM.get(v))
    if stat in ("var", "std") or v.startswith(("sigma_", "ti", "tke", "ctke", "aniso")):
        if v in ("ts", "vpts"):
            return DrillTarget("fluctuation", (v,))
        return DrillTarget("fluctuation", _WIND)
    if v in ("ws", "wd"):
        return DrillTarget("series", _WIND)
    if v in ("ue", "vn", "w", "ts", "vpts", "t", "rh", "p"):
        return DrillTarget("series", tuple(dict.fromkeys((v,) + (_WIND if v in _SONIC else ()))))
    if v in ("es", "e", "r", "q", "td", "pt", "vt", "vpt", "p_factor", "p_measured"):
        return DrillTarget("series", ("t", "rh", "p"))
    if v == "w_ratio":
        return DrillTarget("series", ("w", "ue", "vn"))
    return DrillTarget("series", _WIND)


def ordered_variables(target: DrillTarget, available) -> list[str]:
    """The window's variables with the target's first and the rest after, so a
    series view always has everything but leads with what explains the value.
    """
    first = [v for v in target.variables if v in available]
    return first + [v for v in available if v not in first]
