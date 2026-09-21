"""Profile fits per slot (`profile`): power laws of wind speed, TI and wd
standard deviation vs height, and the neutral log law - each fit's five
rules (booms considered, filtering-first, the tau policy, the `min_booms`
floor, and the fit method) live in one place here, driven by
`[tertiary.fits]`.
"""
import numpy as np
import pandas as pd

from ttu_tower.constants import HEIGHTS
from ttu_tower.math.fits import power_fit
from ttu_tower.physics.most import constrained_neutral_loglaw_fit, neutral_loglaw_fit

_PROFILE_COLUMNS = ["slot", "variant", "variable", "value"]
_SECOND_MOMENT_VARIANTS = ("mrd", "naive", "mrd_unexcised")


def _wide(boom_final: pd.DataFrame, variant: str, variable: str, stat: str | None = None) -> pd.DataFrame:
    """slot x boom, for filtered values already in `boom_final`."""
    mask = (boom_final["variant"] == variant) & (boom_final["variable"] == variable)
    mask &= boom_final["stat"].isna() if stat is None else boom_final["stat"] == stat
    series = boom_final[mask].set_index(["slot", "boom"])["value"]
    return series.unstack("boom")


def _excluded_booms(tau_selected: pd.DataFrame, variant: str, exclude_tau) -> pd.Series:
    """slot -> set of booms excluded by the tau policy. Always empty for
    `naive`, which has no tau exclusions.
    """
    if variant == "naive":
        return pd.Series(dtype=object)
    sub = tau_selected[tau_selected["variant"] == variant]
    bad = sub[sub["source_status"].isin(exclude_tau) | sub["source"].isin(exclude_tau)]
    return bad.groupby("slot")["boom"].apply(set)


def _capped_booms(tau_selected: pd.DataFrame, variant: str) -> pd.Series:
    sub = tau_selected[(tau_selected["variant"] == variant) & (tau_selected["source_status"] == "capped")]
    return sub.groupby("slot")["boom"].apply(set)


def _mask_excluded(values: pd.DataFrame, booms: list[int], excluded_by_slot: pd.Series) -> pd.DataFrame:
    values = values.reindex(columns=booms).copy()
    for slot in values.index:
        excluded = excluded_by_slot.get(slot, set())
        if excluded:
            values.loc[slot, [b for b in booms if b in excluded]] = np.nan
    return values


def _power_law_rows(values: pd.DataFrame, booms: list[int], variant: str, name: str, min_booms: int,
                     capped_by_slot: pd.Series | None = None) -> list[tuple]:
    heights = np.array([HEIGHTS[b] for b in booms], dtype=float)
    rows = []
    for slot, row in values.iterrows():
        y = row.to_numpy(dtype=float)
        used = ~np.isnan(y)
        n = int(used.sum())
        _, exponent = power_fit(heights, y, require=min_booms)
        rows.append((slot, variant, name, float(exponent)))
        rows.append((slot, variant, f"{name}_n", float(n)))
        if capped_by_slot is not None:
            capped = capped_by_slot.get(slot, set())
            n_capped = sum(1 for used_i, b in zip(used, booms) if used_i and b in capped)
            rows.append((slot, variant, f"{name}_n_capped", float(n_capped)))
    return rows


def _alpha_rows(boom_final: pd.DataFrame, cfg_fits) -> list[tuple]:
    ws = _wide(boom_final, "none", "ws", "mean").reindex(columns=cfg_fits.alpha_booms)
    return _power_law_rows(ws, cfg_fits.alpha_booms, "none", "alpha", cfg_fits.min_booms)


def _second_moment_power_rows(boom_final: pd.DataFrame, tau_selected: pd.DataFrame, variant: str, name: str,
                               variable: str, stat: str | None, booms: list[int], cfg_fits) -> list[tuple]:
    values = _wide(boom_final, variant, variable, stat)
    excluded_by_slot = _excluded_booms(tau_selected, variant, cfg_fits.exclude_tau)
    capped_by_slot = _capped_booms(tau_selected, variant)
    values = _mask_excluded(values, booms, excluded_by_slot)
    return _power_law_rows(values, booms, variant, name, cfg_fits.min_booms, capped_by_slot)


def _loglaw_unconstrained_rows(boom_final: pd.DataFrame, cfg_fits) -> list[tuple]:
    booms = cfg_fits.loglaw_booms
    ws = _wide(boom_final, "none", "ws", "mean").reindex(columns=booms)
    heights = [HEIGHTS[b] for b in booms]
    rows = []
    for slot, row in ws.iterrows():
        y = row.to_numpy(dtype=float)
        n = int(np.sum(~np.isnan(y)))
        if n < cfg_fits.min_booms:
            ustar, z0 = np.nan, np.nan
        else:
            ustar, z0 = neutral_loglaw_fit(heights, y)
        rows.append((slot, "none", "loglaw_ustar", float(ustar)))
        rows.append((slot, "none", "loglaw_z0", float(z0)))
        rows.append((slot, "none", "loglaw_n", float(n)))
    return rows


def _loglaw_constrained_rows(boom_final: pd.DataFrame, tau_selected: pd.DataFrame, variant: str, cfg_fits) -> list[tuple]:
    booms = cfg_fits.loglaw_booms
    heights = [HEIGHTS[b] for b in booms]
    ws = _wide(boom_final, "none", "ws", "mean").reindex(columns=booms)
    ustar = _wide(boom_final, variant, "ustar").reindex(columns=booms)
    excluded_by_slot = _excluded_booms(tau_selected, variant, cfg_fits.exclude_tau)
    capped_by_slot = _capped_booms(tau_selected, variant)

    slots = ws.index.union(ustar.index)
    ws = _mask_excluded(ws.reindex(slots), booms, excluded_by_slot)
    ustar = _mask_excluded(ustar.reindex(slots), booms, excluded_by_slot)

    rows = []
    for slot in slots:
        y = ws.loc[slot].to_numpy(dtype=float)
        u = ustar.loc[slot].to_numpy(dtype=float)
        used = ~np.isnan(y) & ~np.isnan(u)
        n = int(used.sum())
        if n < cfg_fits.min_booms:
            z0 = np.nan
        else:
            z0 = constrained_neutral_loglaw_fit(heights, y, ustar=float(np.nanmedian(u)))
        capped = capped_by_slot.get(slot, set())
        n_capped = sum(1 for used_i, b in zip(used, booms) if used_i and b in capped)
        rows.append((slot, variant, "loglaw_z0_constrained", float(z0)))
        rows.append((slot, variant, "loglaw_z0_constrained_n", float(n)))
        rows.append((slot, variant, "loglaw_z0_constrained_n_capped", float(n_capped)))
    return rows


def profiles(boom_final: pd.DataFrame, tau_selected: pd.DataFrame, cfg_fits) -> pd.DataFrame:
    rows = []
    rows += _alpha_rows(boom_final, cfg_fits)
    for variant in _SECOND_MOMENT_VARIANTS:
        rows += _second_moment_power_rows(boom_final, tau_selected, variant, "gamma", "ti", None,
                                           cfg_fits.gamma_booms, cfg_fits)
        rows += _second_moment_power_rows(boom_final, tau_selected, variant, "wdgamma", "wd", "std",
                                           cfg_fits.wdgamma_booms, cfg_fits)
    rows += _loglaw_unconstrained_rows(boom_final, cfg_fits)
    for variant in _SECOND_MOMENT_VARIANTS:
        rows += _loglaw_constrained_rows(boom_final, tau_selected, variant, cfg_fits)
    return pd.DataFrame(rows, columns=_PROFILE_COLUMNS)
