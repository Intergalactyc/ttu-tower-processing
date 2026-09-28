"""tau what-if: secondary's own detection and selection rerun on one slot's
stored spectra (every boom) with edited parameters, compared with what the
run stored, plus u* and σw read off the ladder at each tau.
"""
import dataclasses

import numpy as np
import pandas as pd

from ttu_tower.physics.most import friction_velocity
from ttu_tower.secondary.derived import materialize_missing_unexcised, tau_tables

LADDER_RUNGS = (9.375, 18.75, 37.5, 75.0, 150.0, 300.0, 600.0, 1200.0)


def edited(cfg_secondary, detection: dict | None = None, selection: dict | None = None):
    """A copy of `cfg_secondary` with some detection/selection fields replaced."""
    det = dataclasses.replace(cfg_secondary.detection, **(detection or {}))
    sel = dataclasses.replace(cfg_secondary.selection, **(selection or {}))
    return dataclasses.replace(cfg_secondary, detection=det, selection=sel)


def problems(cfg_secondary) -> list[str]:
    """The config loader's cross-key rules for these fields, as messages."""
    det, sel = cfg_secondary.detection, cfg_secondary.selection
    out = []
    if not det.min_tau_s <= sel.fallback_tau_s <= det.max_tau_s <= 1200:
        out.append("needs min τ ≤ fallback τ ≤ max τ ≤ 1200 s")
    if not det.min_scale_s < det.min_tau_s:
        out.append("needs min scale < min τ")
    if det.peak_significance_se < 0:
        out.append("needs a significance ≥ 0")
    if sorted(sel.priority) != ["heat", "momentum"]:
        out.append("priority must name heat and momentum once each")
    return out


def rerun(across, cfg_secondary) -> tuple[pd.DataFrame, pd.DataFrame]:
    """secondary's (tau, tau_selected) for the slot, every boom, as it would
    compute them with `cfg_secondary`.
    """
    mrd = across.mrd
    if mrd is None or mrd.empty:
        return pd.DataFrame(), pd.DataFrame()
    tau, sel = tau_tables(mrd, across.table("coverage"), cfg_secondary)
    tables = materialize_missing_unexcised({"tau": tau, "tau_selected": sel}, across.table("slot_boom"))
    return tables["tau"], tables["tau_selected"]


def _describe(status, tau_s, tau_lb_s) -> str:
    if pd.isna(status):
        return "—"
    if status == "unresolved":
        return f"unresolved ≥ {tau_lb_s:g} s" if np.isfinite(tau_lb_s) else "unresolved"
    return f"{status} {tau_s:g} s" if np.isfinite(tau_s) else str(status)


def _ladder_values(ladder: pd.DataFrame, boom: int, variant: str, rung: float) -> tuple[float, float]:
    """(u*, σw) from the ladder at one rung (a variant without its own ladder
    rows was never excised, so it reads mrd's).
    """
    if ladder.empty or not np.isfinite(rung):
        return np.nan, np.nan
    lad = ladder[(ladder["boom"] == boom) & (ladder["rung_s"] == rung)]
    own = lad[lad["variant"] == variant]
    lad = own if not own.empty else lad[lad["variant"] == "mrd"]

    def get(variable, stat):
        row = lad[(lad["variable"] == variable) & (lad["stat"] == stat)]
        return float(row["value"].iloc[0]) if not row.empty else np.nan

    with np.errstate(invalid="ignore"):
        return float(friction_velocity(get("uw", "cov"), get("vw", "cov"))), float(np.sqrt(get("w", "var")))


def compare(across, cfg_secondary) -> pd.DataFrame:
    """One row per (boom, variant): stored vs what-if heat, momentum and
    selected tau, whether anything changed, and u*/σw at either selected tau.
    """
    new_tau, new_sel = rerun(across, cfg_secondary)
    old_tau, old_sel = across.table("tau"), across.table("tau_selected")
    ladder = across.table("ladder")
    rows = []
    if new_sel.empty:
        return pd.DataFrame()
    for r in new_sel.sort_values(["boom", "variant"]).itertuples(index=False):
        boom, variant = int(r.boom), r.variant
        row = {"boom": boom, "variant": variant}
        changed = False
        for fam in ("heat", "momentum"):
            new = new_tau[(new_tau["boom"] == boom) & (new_tau["variant"] == variant) & (new_tau["cospectrum"] == fam)]
            old = old_tau[(old_tau["boom"] == boom) & (old_tau["variant"] == variant) & (old_tau["cospectrum"] == fam)]
            n = new.iloc[0] if not new.empty else None
            o = old.iloc[0] if not old.empty else None
            row[f"{fam}_stored"] = _describe(o["status"], o["tau_s"], o["tau_lb_s"]) if o is not None else "—"
            row[f"{fam}_new"] = _describe(n["status"], n["tau_s"], n["tau_lb_s"]) if n is not None else "—"
            changed |= row[f"{fam}_stored"] != row[f"{fam}_new"]
        old = old_sel[(old_sel["boom"] == boom) & (old_sel["variant"] == variant)]
        o = old.iloc[0] if not old.empty else None
        row["selected_tau_stored"] = float(o["tau_s"]) if o is not None else np.nan
        row["selected_tau_new"] = float(r.tau_s)
        row["source_stored"] = f"{o['source']} ({o['source_status']})" if o is not None else "—"
        row["source_new"] = f"{r.source} ({r.source_status})"
        same_tau = (np.isnan(row["selected_tau_stored"]) and np.isnan(row["selected_tau_new"])) or \
            row["selected_tau_stored"] == row["selected_tau_new"]
        changed |= not same_tau or row["source_stored"] != row["source_new"]
        row["ustar_stored"], row["sigma_w_stored"] = _ladder_values(ladder, boom, variant, row["selected_tau_stored"])
        row["ustar_new"], row["sigma_w_new"] = _ladder_values(ladder, boom, variant, row["selected_tau_new"])
        row["changed"] = bool(changed)
        rows.append(row)
    return pd.DataFrame(rows)
