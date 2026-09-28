"""One slot's vertical profile: each boom's value of a quantity (and, where
tertiary filtered it, the value it had before), the profile fits tertiary
stored for the slot redrawn as curves, and the boom-pair quantities.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttu_tower.constants import HEIGHTS
from ttu_tower.math.fits import power_fit
from ttu_tower.physics.constants import KAPPA
from ttu_tower.tertiary.filtering import QUANTITY_GROUPS, build_candidate


@dataclass(frozen=True)
class ProfileQuantity:
    key: str
    label: str
    unit: str
    variable: str
    stat: str | None
    per_variant: bool  # second-moment (tau-dependent) values follow the inspector's variant


PROFILE_QUANTITIES = (
    ProfileQuantity("ws", "wind speed", "m/s", "ws", "mean", False),
    ProfileQuantity("wd", "wind direction", "deg", "wd", "mean", False),
    ProfileQuantity("veer", "veer", "deg", "veer", None, False),
    ProfileQuantity("vpt", "θv (slow sensors)", "K", "vpt", None, False),
    ProfileQuantity("tau", "selected τ", "s", "tau", None, True),
    ProfileQuantity("ti", "TI", "", "ti", None, True),
    ProfileQuantity("sigma_w", "σw", "m/s", "sigma_w", None, True),
    ProfileQuantity("ustar", "u*", "m/s", "ustar", None, True),
    ProfileQuantity("wvpts", "w'θv'", "K m/s", "wvpts", "cov", True),
    ProfileQuantity("wd_std", "wd std", "deg", "wd", "std", True),
)


def _stat_mask(df: pd.DataFrame, stat: str | None) -> pd.Series:
    return df["stat"].isna() if stat is None else df["stat"] == stat


def profile_points(across, pq: ProfileQuantity, variant: str) -> pd.DataFrame:
    """Per boom: height, the final value, the value before filtering, whether
    tertiary filtered it, and filter_log's reasons.
    """
    booms = list(across.booms)
    out = pd.DataFrame({"boom": booms, "height": [HEIGHTS[b] for b in booms]})
    out["value"] = np.nan
    out["prefilter"] = np.nan
    out["reasons"] = ""
    v = variant if pq.per_variant else "none"
    if pq.key == "tau":
        sel = across.table("tau_selected")
        sel = sel[sel["variant"] == variant].set_index("boom")
        out["value"] = out["boom"].map(sel["tau_s"]).astype(float)
        out["prefilter"] = out["value"]
        out["reasons"] = out["boom"].map(sel["source"].astype(str) + " (" + sel["source_status"].astype(str) + ")").fillna("")
    elif pq.key == "veer":
        pairs = across.table("pairs")
        rows = pairs[pairs["variable"] == "veer"].set_index("boom")
        out["value"] = out["boom"].map(rows["value"]).astype(float)
        out["prefilter"] = out["value"]
    else:
        final = across.table("boom_final")
        rows = final[(final["variant"] == v) & (final["variable"] == pq.variable) & _stat_mask(final, pq.stat)]
        out["value"] = out["boom"].map(rows.set_index("boom")["value"]).astype(float)
        cand = build_candidate(across.table("means"), across.table("slow"), across.table("boom_stats"))
        rows = cand[(cand["variant"] == v) & (cand["variable"] == pq.variable) & _stat_mask(cand, pq.stat)]
        out["prefilter"] = out["boom"].map(rows.set_index("boom")["value"]).astype(float)
        log = across.table("filter_log")
        log = log[(log["variant"] == v) & (log["group"] == QUANTITY_GROUPS.get(pq.variable))]
        text = pd.Series([f"{c} {x}" if isinstance(x, str) else str(c) for c, x in zip(log["criterion"], log["variable"])],
                         index=log.index, dtype=object)
        reasons = text.groupby(log["boom"]).agg(lambda parts: ", ".join(sorted(parts)))
        out["reasons"] = out["boom"].map(reasons).fillna("")
    out["filtered"] = out["value"].isna() & out["prefilter"].notna()
    return out


def _excluded(across, variant: str, exclude_tau) -> set[int]:
    if variant == "naive":
        return set()
    sel = across.table("tau_selected")
    sel = sel[sel["variant"] == variant]
    bad = sel[sel["source_status"].isin(exclude_tau) | sel["source"].isin(exclude_tau)]
    return set(bad["boom"].astype(int))


def _stored(across, variant: str, name: str) -> float:
    prof = across.table("profile")
    row = prof[(prof["variant"] == variant) & (prof["variable"] == name)]
    return float(row["value"].iloc[0]) if not row.empty else np.nan


@dataclass
class Fit:
    name: str
    label: str
    z: np.ndarray
    y: np.ndarray
    text: str


def _z_grid(heights) -> np.ndarray:
    """Heights to draw a fit over: the span of the booms it was fitted to, a little beyond."""
    heights = [h for h in heights if np.isfinite(h)]
    return np.geomspace(min(heights) * 0.9, max(heights) * 1.1, 60) if heights else np.empty(0)


def _power_curve(across, points: pd.DataFrame, booms, excluded: set, min_booms: int, stored: float, name: str,
                 label: str) -> Fit | None:
    """The power law through the booms tertiary fitted: the exponent is the
    stored one; the coefficient comes from refitting the same values.
    """
    sub = points[points["boom"].isin(booms) & ~points["boom"].isin(excluded)]
    a, b = power_fit(sub["height"].to_numpy(float), sub["value"].to_numpy(float), require=min_booms)
    if not np.isfinite(stored) or not np.isfinite(a):
        return None
    z = _z_grid(sub.loc[sub["value"].notna(), "height"])
    note = "" if np.isclose(b, stored, rtol=1e-9, atol=1e-12) else f" (refit gives {b:.4g})"
    return Fit(name, label, z, a * z ** stored, f"{label} = {stored:.3g}{note}")


def profile_fits(across, key: str, points: pd.DataFrame, variant: str, cfg_fits) -> list[Fit]:
    """The stored fits of one profile plot as curves."""
    fits = []
    if key == "ws":
        f = _power_curve(across, points, cfg_fits.alpha_booms, set(), cfg_fits.min_booms,
                         _stored(across, "none", "alpha"), "alpha", "α")
        if f is not None:
            fits.append(f)
        ustar, z0 = _stored(across, "none", "loglaw_ustar"), _stored(across, "none", "loglaw_z0")
        used = points[points["boom"].isin(cfg_fits.loglaw_booms) & points["value"].notna()]
        z = _z_grid(used["height"])
        if np.isfinite(ustar) and np.isfinite(z0) and z0 > 0:
            fits.append(Fit("loglaw", "log law", z, ustar / KAPPA * np.log(z / z0),
                            f"log law u* {ustar:.3g} m/s, z0 {z0:.3g} m"))
        z0c = _stored(across, variant, "loglaw_z0_constrained")
        if np.isfinite(z0c) and z0c > 0:
            ustar_c = _constrained_ustar(across, variant, cfg_fits)
            if np.isfinite(ustar_c):
                fits.append(Fit("loglaw_c", "log law (u* fixed)", z, ustar_c / KAPPA * np.log(z / z0c),
                                f"log law with u* = median {ustar_c:.3g} m/s: z0 {z0c:.3g} m"))
    elif key == "ti":
        f = _power_curve(across, points, cfg_fits.gamma_booms, _excluded(across, variant, cfg_fits.exclude_tau),
                         cfg_fits.min_booms, _stored(across, variant, "gamma"), "gamma", "γ")
        if f is not None:
            fits.append(f)
    elif key == "wd_std":
        f = _power_curve(across, points, cfg_fits.wdgamma_booms, _excluded(across, variant, cfg_fits.exclude_tau),
                         cfg_fits.min_booms, _stored(across, variant, "wdgamma"), "wdgamma", "γ_wd")
        if f is not None:
            fits.append(f)
    return fits


def _constrained_ustar(across, variant: str, cfg_fits) -> float:
    """The median u* the constrained log law was fitted with: the log-law
    booms' u* after the tau policy.
    """
    booms = [b for b in cfg_fits.loglaw_booms if b not in _excluded(across, variant, cfg_fits.exclude_tau)]
    final = across.table("boom_final")
    us = final[(final["variant"] == variant) & (final["variable"] == "ustar") & final["stat"].isna()].set_index("boom")["value"]
    u = np.array([us.get(b, np.nan) for b in booms], dtype=float)
    return float(np.nanmedian(u)) if np.isfinite(u).any() else np.nan


def fit_summary(across, variant: str) -> list[str]:
    """Every stored profile row of the slot, as text."""
    prof = across.table("profile")
    lines = []
    for v in ("none", variant):
        rows = prof[prof["variant"] == v]
        for r in rows.itertuples(index=False):
            lines.append(f"{r.variable} ({v}) = {r.value:.4g}" if np.isfinite(r.value) else f"{r.variable} ({v}) = —")
    return lines


def pair_table(across, stability_pair) -> pd.DataFrame:
    """Ri_b and dθv/dz for every boom pair, the stability pair first."""
    pairs = across.table("pairs")
    rows = pairs[pairs["variable"].isin(["rib", "lapse_vpt"])]
    if rows.empty:
        return pd.DataFrame(columns=["boom", "boom2", "rib", "lapse_vpt"])
    wide = rows.pivot_table(index=["boom", "boom2"], columns="variable", values="value", aggfunc="first",
                            dropna=False).reset_index()
    wide.columns.name = None
    for col in ("rib", "lapse_vpt"):
        if col not in wide.columns:
            wide[col] = np.nan
    first = (wide["boom"] == stability_pair[0]) & (wide["boom2"] == stability_pair[1])
    return pd.concat([wide[first], wide[~first]], ignore_index=True)[["boom", "boom2", "rib", "lapse_vpt"]]
