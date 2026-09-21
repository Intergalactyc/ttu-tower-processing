"""Boom-level filtering: a value fails when a required family's coverage is
below `min_coverage`, or when too much of an input variable's support was
flagged `bounds`/`spike`. Filtering runs before every multi-boom quantity and
every profile fit, so their NaN propagation from a failing boom is automatic
- a whole record is never dropped.
"""
import numpy as np
import pandas as pd

from ttu_tower.timegrid import SAMPLES_PER_SLOT

_WINDOW_MARGIN_SAMPLES = 15_000
_CANDIDATE_COLUMNS = ["slot", "boom", "variant", "variable", "stat", "value"]
_FILTER_LOG_COLUMNS = ["slot", "boom", "variant", "group", "criterion", "variable"]

# Every `variable` string that can appear in `means`, `slow` or `boom_stats`,
# mapped to the group it belongs to. The group is a pure function of the
# variable name; `stat` never disambiguates it.
QUANTITY_GROUPS: dict[str, str] = {
    # momentum family: raw wind statistics, gusts, and every derived quantity
    # that only needs ue/vn/w.
    "ue": "momentum", "vn": "momentum", "w": "momentum", "u": "momentum", "v": "momentum",
    "ws": "momentum", "wd": "momentum", "uv": "momentum", "uw": "momentum", "vw": "momentum",
    "sigma_u": "momentum", "sigma_v": "momentum", "sigma_w": "momentum",
    "ti": "momentum", "ti_u": "momentum", "ti_v": "momentum", "ti_w": "momentum",
    "ti_ratio_vu": "momentum", "ti_ratio_wu": "momentum",
    "tke": "momentum", "ctke": "momentum", "ustar": "momentum",
    "ils_u": "momentum", "ils_v": "momentum", "ils_w": "momentum",
    "ils_ratio_vu": "momentum", "ils_ratio_wu": "momentum",
    "its_ratio_u": "momentum", "its_ratio_v": "momentum", "its_ratio_w": "momentum",
    "its_short": "momentum",
    "aniso_l1": "momentum", "aniso_l2": "momentum", "aniso_l3": "momentum",
    "aniso_beta": "momentum", "aniso_phi": "momentum", "w_ratio": "momentum",
    "momentum": "momentum", "its": "momentum",
    # heat family: needs both heat and momentum (obukhov_length needs ustar).
    "wvpts": "heat", "obukhov_length": "heat", "zeta": "heat", "ils_vpts": "heat", "heat": "heat",
    # ts family: ts/vpts statistics, their ITS/ratio, and the pressure factor
    # that travels with the vpts values it converted.
    "ts": "ts", "vpts": "ts", "its_ratio_vpts": "ts", "its_short_vpts": "ts",
    "p_factor": "ts", "p_measured": "ts", "its_vpts": "ts",
    # slow sensors, each its own group.
    "t": "t", "rh": "rh", "p": "p",
    # vpt group: thermodynamic quantities needing all three slow sensors.
    "es": "vpt", "e": "vpt", "r": "vpt", "q": "vpt", "td": "vpt", "pt": "vpt", "vt": "vpt", "vpt": "vpt",
}

# The families to threshold against `min_coverage` for each group. `its` and
# `its_vpts` are never listed here: an ITS computed from few blocks is a
# diagnostic (reported via the `its`/`its_vpts` coverage rows themselves,
# same as any other momentum/ts-group value), not a separate gate.
_GROUP_REQUIRED_FAMILIES: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum",),
    "heat": ("heat", "momentum"),
    "ts": ("ts",),
    "t": ("t",),
    "rh": ("rh",),
    "p": ("p",),
    "vpt": ("t", "rh", "p"),
}

# The input variables whose `bounds`/`spike` flag fractions gate each group.
_GROUP_INPUT_VARIABLES: dict[str, tuple[str, ...]] = {
    "momentum": ("ue", "vn", "w"),
    "heat": ("ue", "vn", "w", "ts"),
    "ts": ("ts",),
    "t": ("t",),
    "rh": ("rh",),
    "p": ("p",),
    "vpt": ("t", "rh", "p"),
}


def build_candidate(means: pd.DataFrame, slow: pd.DataFrame, boom_stats: pd.DataFrame) -> pd.DataFrame:
    """The unfiltered `boom_final`-shaped frame: `means` and `slow` rows
    tagged `variant="none"`, plus every `boom_stats` row. `slow` has no
    `stat` column, so it's set to null before the concat.
    """
    means_rows = means[["slot", "boom", "variable", "stat", "value"]].copy()
    means_rows["variant"] = "none"

    slow_rows = slow[["slot", "boom", "variable", "value"]].copy()
    slow_rows["stat"] = None
    slow_rows["variant"] = "none"

    return pd.concat(
        [means_rows[_CANDIDATE_COLUMNS], slow_rows[_CANDIDATE_COLUMNS], boom_stats[_CANDIDATE_COLUMNS]],
        ignore_index=True,
    )


def _support_bounds(keys: pd.DataFrame) -> pd.DataFrame:
    """Per key, the [start, end) sample support: the slot, or the 20-min
    window when a variant row's selected tau is 1200s. A `variant` row's own
    `ladder_coverage` at that rung is already computed over the right
    support; only the `bounds`/`spike` FlagStore reach needs the wider
    window explicitly.
    """
    keys = keys.copy()
    g0 = SAMPLES_PER_SLOT * keys["slot"]
    use_window = (keys["variant"] != "none") & (keys["tau_s"] == 1200.0)
    keys["start"] = np.where(use_window, g0 - _WINDOW_MARGIN_SAMPLES, g0).astype(np.int64)
    keys["end"] = np.where(use_window, g0 + SAMPLES_PER_SLOT + _WINDOW_MARGIN_SAMPLES, g0 + SAMPLES_PER_SLOT).astype(np.int64)
    return keys


def _coverage_fail_keys(keys: pd.DataFrame, coverage: pd.DataFrame, ladder_coverage: pd.DataFrame, min_coverage: float) -> set:
    fam_map = pd.DataFrame(
        [(g, f) for g, fams in _GROUP_REQUIRED_FAMILIES.items() for f in fams], columns=["group", "family"],
    )
    req = keys.merge(fam_map, on="group", how="inner")

    none_cov = coverage[coverage["layer"] == "usable"].rename(columns={"variable": "family", "fraction": "cov_none"})
    req = req.merge(none_cov[["slot", "boom", "family", "cov_none"]], on=["slot", "boom", "family"], how="left")

    variant_cov = ladder_coverage.rename(columns={"rung_s": "tau_s", "coverage": "cov_variant"})
    req = req.merge(
        variant_cov[["slot", "boom", "variant", "tau_s", "family", "cov_variant"]],
        on=["slot", "boom", "variant", "tau_s", "family"], how="left",
    )

    is_none = req["variant"] == "none"
    req["coverage_value"] = np.where(is_none, req["cov_none"], req["cov_variant"])
    req["coverage_value"] = req["coverage_value"].fillna(0.0)
    failing = req[req["coverage_value"] < min_coverage]
    return set(zip(failing["slot"], failing["boom"], failing["variant"], failing["group"]))


def _flag_fraction_checks(keys: pd.DataFrame, flag_stores: dict) -> pd.DataFrame:
    var_map = pd.DataFrame(
        [(g, v) for g, vs in _GROUP_INPUT_VARIABLES.items() for v in vs], columns=["group", "input_variable"],
    )
    checks = keys.merge(var_map, on="group", how="inner")
    checks["bounds_fraction"] = 0.0
    checks["spike_fraction"] = 0.0

    for (boom, variable), idx in checks.groupby(["boom", "input_variable"], sort=False).groups.items():
        store = flag_stores.get(boom)
        if store is None:
            continue
        sub = checks.loc[idx]
        starts, ends = sub["start"].to_numpy(), sub["end"].to_numpy()
        checks.loc[idx, "bounds_fraction"] = store.fraction("bounds", boom, variable, starts, ends)
        checks.loc[idx, "spike_fraction"] = store.fraction("spike", boom, variable, starts, ends)
    return checks


def evaluate(candidate: pd.DataFrame, coverage: pd.DataFrame, ladder_coverage: pd.DataFrame,
             tau_selected: pd.DataFrame, flag_stores: dict, cfg_tertiary) -> tuple[set, pd.DataFrame]:
    """Per (slot, boom, variant, group) present in `candidate`: whether it
    fails, and the `filter_log` rows explaining why. `flag_stores` maps
    boom -> a `FlagStore` built from that boom's flag fragments reaching
    +-5 min past the batch (so a 20-min-window query can see the halves of
    the neighbouring slots).
    """
    group = candidate["variable"].map(QUANTITY_GROUPS)
    keys = candidate.assign(group=group)[["slot", "boom", "variant", "group"]].drop_duplicates()
    keys = keys.merge(tau_selected[["slot", "boom", "variant", "tau_s"]], on=["slot", "boom", "variant"], how="left")
    keys = _support_bounds(keys)

    coverage_fail = _coverage_fail_keys(keys, coverage, ladder_coverage, cfg_tertiary.min_coverage)
    checks = _flag_fraction_checks(keys, flag_stores)
    bounds_fail = checks[checks["bounds_fraction"] > cfg_tertiary.max_bounds_fraction]
    spike_fail = checks[checks["spike_fraction"] > cfg_tertiary.max_spike_fraction]

    log_rows = [(slot, boom, variant, grp, "coverage", None) for slot, boom, variant, grp in coverage_fail]
    log_rows += list(zip(bounds_fail["slot"], bounds_fail["boom"], bounds_fail["variant"], bounds_fail["group"],
                          ["bounds"] * len(bounds_fail), bounds_fail["input_variable"]))
    log_rows += list(zip(spike_fail["slot"], spike_fail["boom"], spike_fail["variant"], spike_fail["group"],
                          ["spike"] * len(spike_fail), spike_fail["input_variable"]))
    filter_log = pd.DataFrame(log_rows, columns=_FILTER_LOG_COLUMNS)

    fail_keys = coverage_fail.copy()
    fail_keys |= set(zip(bounds_fail["slot"], bounds_fail["boom"], bounds_fail["variant"], bounds_fail["group"]))
    fail_keys |= set(zip(spike_fail["slot"], spike_fail["boom"], spike_fail["variant"], spike_fail["group"]))
    return fail_keys, filter_log


def apply(candidate: pd.DataFrame, fail_keys: set) -> pd.DataFrame:
    """NaN every value whose (slot, boom, variant, group) failed a criterion."""
    out = candidate.copy()
    out["group"] = out["variable"].map(QUANTITY_GROUPS)
    fail_mask = pd.MultiIndex.from_frame(out[["slot", "boom", "variant", "group"]]).isin(fail_keys)
    out.loc[fail_mask, "value"] = np.nan
    return out.drop(columns=["group"])
