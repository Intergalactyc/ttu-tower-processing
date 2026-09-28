"""The QC picture of one slot of one boom, from stored rows only: its
coverage ladder, how much of each series every test flagged, its slot-level
statistics against their limits, and tertiary's filtering gates recomputed
with their values (which the stored filter_log only names).
"""
import numpy as np
import pandas as pd

from ttu_tower.flags import BOOM_LEVEL_VARIABLES, TESTS, FlagStore
from ttu_tower.primary.products import detection_window, ladder_window
from ttu_tower.tertiary.filtering import (
    _GROUP_INPUT_VARIABLES, _GROUP_REQUIRED_FAMILIES, _WINDOW_MARGIN_SAMPLES, QUANTITY_GROUPS, build_candidate,
)
from ttu_tower.timegrid import SAMPLES_PER_SLOT

LAYERS = ("present", "filled", "usable_l1", "usable", "unexcised")
COVERAGE_VARIABLES = ("ue", "vn", "w", "ts", "vpts", "t", "rh", "p", "momentum", "heat")
FLAG_VARIABLES = ("ue", "vn", "w", "ts", "t", "rh", "p")


def coverage_ladder(coverage: pd.DataFrame) -> pd.DataFrame:
    """variable x layer fractions (NaN where a layer doesn't exist for a variable)."""
    if coverage.empty:
        return pd.DataFrame(index=list(COVERAGE_VARIABLES), columns=list(LAYERS), dtype=float)
    table = coverage.pivot_table(index="variable", columns="layer", values="fraction", aggfunc="first")
    return table.reindex(index=list(COVERAGE_VARIABLES), columns=list(LAYERS))


def windows(k: int) -> dict[str, tuple[int, int]]:
    """[start, end) sample ranges the QC tab reports flag fractions over."""
    g0, n = detection_window(k)
    l0, ln = ladder_window(k)
    s0 = SAMPLES_PER_SLOT * k
    return {"slot": (s0, s0 + SAMPLES_PER_SLOT), "ladder window (20 min)": (l0, l0 + ln),
            "detection window (80 min)": (g0, g0 + n)}


def flag_fractions(flags: pd.DataFrame, boom: int, start: int, end: int) -> pd.DataFrame:
    """test x variable: the fraction of [start, end) each test flagged (boom-level
    tests count toward ue/vn/w/ts; NaN where a test never applies to a variable).
    """
    store = FlagStore(flags)
    out = pd.DataFrame(index=list(TESTS), columns=list(FLAG_VARIABLES), dtype=float)
    for test, spec in TESTS.items():
        applies = spec.variables if spec.variables is not None else BOOM_LEVEL_VARIABLES
        for var in FLAG_VARIABLES:
            if var in applies:
                out.loc[test, var] = float(store.fraction(test, boom, var, [start], [end])[0])
    return out


def slot_qc_checks(slot_qc: pd.DataFrame, cfg_qc) -> pd.DataFrame:
    """Each slot-level statistic with the limit its test applies and whether it
    tripped; `removes` says whether that test makes samples unusable.
    """
    rows = []
    value = {(r.variable, r.stat): r.value for r in slot_qc.itertuples(index=False)}
    w, sl = cfg_qc.windows, cfg_qc.second_layer
    for var in BOOM_LEVEL_VARIABLES:
        for stat, (lo, hi) in (("skew", w.skew_range), ("kurt", w.kurt_range)):
            v = value.get((var, stat), np.nan)
            rows.append((stat, var, stat, v, f"[{lo:g}, {hi:g}]", bool(np.isfinite(v) and not lo <= v <= hi)))
    lo, hi = sl.shadow_sector
    wd = value.get(("wd", "mean_l1"), np.nan)
    rows.append(("direction", "wd", "mean_l1", wd, f"outside ({lo:g}, {hi:g})", bool(np.isfinite(wd) and lo < wd < hi)))
    for stat, limit in (("mean_l1", sl.bounce_max_ws_mean), ("std_l1", sl.bounce_max_ws_std)):
        v = value.get(("ws", stat), np.nan)
        rows.append(("bounce", "ws", stat, v, f"≤ {limit:g}", bool(np.isfinite(v) and v > limit)))
    out = pd.DataFrame(rows, columns=["test", "variable", "stat", "value", "allowed", "tripped"])
    out["removes"] = out["test"].isin(cfg_qc.unusable_tests)
    return out


def filter_gates(k: int, boom: int, means: pd.DataFrame, slow: pd.DataFrame, boom_stats: pd.DataFrame,
                 coverage: pd.DataFrame, tau_selected: pd.DataFrame, flags: pd.DataFrame,
                 cfg_tertiary) -> pd.DataFrame:
    """Every criterion tertiary tests for this slot and boom, per (variant,
    group) present, with the value it tested, its limit and whether it failed.
    `flags` must reach 5 min past the slot on both sides.
    """
    candidate = build_candidate(means, slow, boom_stats)
    if candidate.empty:
        return pd.DataFrame(columns=["variant", "group", "criterion", "variable", "value", "limit", "failed"])
    candidate = candidate.assign(group=candidate["variable"].astype(object).map(QUANTITY_GROUPS))
    keys = candidate[["variant", "group"]].dropna().drop_duplicates()
    tau = {r.variant: r.tau_s for r in tau_selected.itertuples(index=False)}
    usable = {r.variable: r.fraction for r in coverage[coverage["layer"] == "usable"].itertuples(index=False)}
    variant_cov = {(r.variant, r.variable): r.value
                   for r in candidate[candidate["stat"] == "coverage"].itertuples(index=False)}
    store = FlagStore(flags)
    s0 = SAMPLES_PER_SLOT * k

    rows = []
    for variant, group in sorted(keys.itertuples(index=False, name=None), key=lambda r: (str(r[0]), str(r[1]))):
        for fam in _GROUP_REQUIRED_FAMILIES[group]:
            v = usable.get(fam) if variant == "none" else variant_cov.get((variant, fam))
            v = 0.0 if v is None or pd.isna(v) else float(v)
            rows.append((variant, group, "coverage", fam, v, cfg_tertiary.min_coverage, v < cfg_tertiary.min_coverage))
        wide = variant != "none" and tau.get(variant) == 1200.0
        start, end = (s0 - _WINDOW_MARGIN_SAMPLES, s0 + SAMPLES_PER_SLOT + _WINDOW_MARGIN_SAMPLES) if wide else (
            s0, s0 + SAMPLES_PER_SLOT)
        for var in _GROUP_INPUT_VARIABLES[group]:
            for test, limit in (("bounds", cfg_tertiary.max_bounds_fraction), ("spike", cfg_tertiary.max_spike_fraction)):
                v = float(store.fraction(test, boom, var, [start], [end])[0])
                rows.append((variant, group, test, var, v, limit, v > limit))
    return pd.DataFrame(rows, columns=["variant", "group", "criterion", "variable", "value", "limit", "failed"])


def logged_failures(gates: pd.DataFrame) -> set[tuple]:
    """The gates' failures in filter_log's own terms: (variant, group, criterion,
    variable), with one variable-less row per failing coverage key.
    """
    failed = gates[gates["failed"]]
    return {(r.variant, r.group, r.criterion, None if r.criterion == "coverage" else r.variable)
            for r in failed.itertuples(index=False)}


def stored_failures(filter_log: pd.DataFrame, boom: int) -> set[tuple]:
    rows = filter_log[filter_log["boom"] == boom] if "boom" in filter_log.columns else filter_log
    return {(r.variant, r.group, r.criterion, None if pd.isna(r.variable) else r.variable)
            for r in rows.itertuples(index=False)}
