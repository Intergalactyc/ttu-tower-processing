"""Slow-sensor thermodynamics (t, rh, p) and each slot's pressure factor,
which secondary/derived.py applies to every heat quantity, at every tau and
variant.
"""
import numpy as np
import pandas as pd

from ttu_tower.constants import P_REF
from ttu_tower.physics import thermo
from ttu_tower.physics.constants import R_CP

_SLOW_VARS = ("t", "rh", "p", "vpts")


def slow_table(means: pd.DataFrame, coverage: pd.DataFrame, cfg) -> pd.DataFrame:
    """The `slow` table, for every (slot, boom) with `means` rows: humidity
    and potential-temperature quantities from the smoothed t/rh/p means, plus
    the slot's pressure factor and pressure-corrected vpts.
    """
    mask = means["variable"].isin(_SLOW_VARS) & (means["stat"] == "mean")
    wide = means[mask].pivot_table(index=["slot", "boom"], columns="variable", values="value", aggfunc="first")
    wide = wide.reindex(columns=list(_SLOW_VARS))
    if wide.empty:
        return pd.DataFrame(columns=["slot", "boom", "variable", "value"])

    t, rh, p, vpts_ref = wide["t"], wide["rh"], wide["p"], wide["vpts"]
    es = thermo.saturation_vapor_pressure(t)
    e = thermo.water_partial_pressure(rh, es)
    r = thermo.water_air_mixing_ratio(e, p)
    q = thermo.specific_humidity(r)
    td = thermo.dewpoint_temperature(t, rh)
    pt = thermo.potential_temperature(t, p)
    vt = thermo.virtual_temperature(t, r)
    vpt = thermo.virtual_potential_temperature(pt, r)

    p_cov = coverage[(coverage["variable"] == "p") & (coverage["layer"] == "usable")]
    p_cov = p_cov.set_index(["slot", "boom"])["fraction"].reindex(wide.index).fillna(0.0)
    coverage_ok = p_cov >= cfg.qc.min_coverage

    p_ref = wide.index.get_level_values("boom").map(P_REF).to_numpy(dtype=float)
    p_factor = np.ones(len(p))
    p_ok = p.to_numpy(dtype=float)[coverage_ok.to_numpy()]
    p_factor[coverage_ok.to_numpy()] = (p_ref[coverage_ok.to_numpy()] / p_ok) ** R_CP
    p_measured = np.where(coverage_ok, 1.0, 0.0)
    vpts = vpts_ref * p_factor

    result = pd.DataFrame({
        "t": t, "rh": rh, "p": p, "es": es, "e": e, "r": r, "q": q, "td": td,
        "pt": pt, "vt": vt, "vpt": vpt, "p_factor": p_factor, "p_measured": p_measured, "vpts": vpts,
    }, index=wide.index)
    return result.reset_index().melt(id_vars=["slot", "boom"], var_name="variable", value_name="value")
