import numpy as np
import pandas as pd
import pytest

from ttu_tower.flags import FlagStore
from ttu_tower.tertiary import filtering
from ttu_tower.timegrid import SAMPLES_PER_SLOT

_SLOT = 1000
_BOOM = 1
_G0 = SAMPLES_PER_SLOT * _SLOT


class _Cfg:
    min_coverage = 0.75
    max_bounds_fraction = 0.01
    max_spike_fraction = 0.01


def _means_row(slot, boom, variable, stat, value):
    return {"slot": slot, "boom": boom, "variant": "none", "variable": variable, "stat": stat, "value": value}


def _boom_stats_row(slot, boom, variant, variable, value):
    return {"slot": slot, "boom": boom, "variant": variant, "variable": variable, "stat": None, "value": value}


def _coverage_row(slot, boom, variant, family, value):
    """A `boom_stats` coverage row, exactly as `secondary.derived.selected_stats`
    produces it - the real source of variant coverage in `candidate` now that
    tertiary no longer reads primary's `ladder_coverage` directly (that table
    never has a `naive` row, and copied `mrd_unexcised` rows come from
    `boom_stats`, not from primary).
    """
    return {"slot": slot, "boom": boom, "variant": variant, "variable": family, "stat": "coverage", "value": value}


def _flag_store(boom, test, variable, starts, ends) -> FlagStore:
    rows = [{"start": s, "end": e, "test": test, "kind": "quality", "boom": boom, "variable": variable}
            for s, e in zip(starts, ends)]
    return FlagStore(pd.DataFrame(rows, columns=["start", "end", "test", "kind", "boom", "variable"]))


def _empty_tau_selected():
    return pd.DataFrame(columns=["slot", "boom", "variant", "tau_s", "source", "source_status"])


def test_build_candidate_tags_variant_none_and_nulls_slow_stat():
    means = pd.DataFrame([_means_row(_SLOT, _BOOM, "ue", "mean", 1.0)])
    slow = pd.DataFrame([{"slot": _SLOT, "boom": _BOOM, "variable": "t", "value": 290.0}])
    boom_stats = pd.DataFrame([_boom_stats_row(_SLOT, _BOOM, "mrd", "ustar", 0.3)])

    candidate = filtering.build_candidate(means, slow, boom_stats)

    assert set(candidate["variant"]) == {"none", "mrd"}
    slow_row = candidate[(candidate["variable"] == "t")].iloc[0]
    assert slow_row["stat"] is None
    assert len(candidate) == 3


def test_quantity_groups_completeness():
    produced_variables = {
        # means
        "ue", "vn", "w", "ws", "wd", "ts", "vpts", "t", "rh", "p",
        # slow
        "es", "e", "r", "q", "td", "pt", "vt", "vpt", "p_factor", "p_measured",
        # boom_stats ladder passthrough
        "u", "v", "uv", "uw", "vw", "momentum", "heat", "its", "its_vpts",
        # boom_stats derived
        "sigma_u", "sigma_v", "sigma_w", "ti", "ti_u", "ti_v", "ti_w",
        "ti_ratio_vu", "ti_ratio_wu", "tke", "ctke", "ustar", "obukhov_length",
        "zeta", "ils_u", "ils_v", "ils_w", "ils_vpts", "ils_ratio_vu", "ils_ratio_wu",
        "its_ratio_u", "its_ratio_v", "its_ratio_w", "its_ratio_vpts",
        "its_short", "its_short_vpts", "wvpts", "aniso_l1", "aniso_l2", "aniso_l3",
        "aniso_beta", "aniso_phi", "w_ratio",
    }
    missing = produced_variables - set(filtering.QUANTITY_GROUPS)
    assert not missing, f"no group for: {missing}"
    assert set(filtering.QUANTITY_GROUPS.values()) <= {"momentum", "heat", "ts", "t", "rh", "p", "vpt"}


def test_low_coverage_fails_only_that_group():
    candidate = pd.DataFrame([
        _means_row(_SLOT, _BOOM, "ue", "mean", 1.0),
        _means_row(_SLOT, _BOOM, "ts", "mean", 290.0),
    ])
    coverage = pd.DataFrame([
        {"slot": _SLOT, "boom": _BOOM, "variable": "momentum", "layer": "usable", "fraction": 0.5},
        {"slot": _SLOT, "boom": _BOOM, "variable": "ts", "layer": "usable", "fraction": 1.0},
    ])
    fail_keys, log = filtering.evaluate(candidate, coverage, _empty_tau_selected(), {}, _Cfg())
    filtered = filtering.apply(candidate, fail_keys)

    assert filtered.loc[filtered["variable"] == "ue", "value"].isna().all()
    assert not filtered.loc[filtered["variable"] == "ts", "value"].isna().any()
    assert (log["criterion"] == "coverage").any()
    assert (log["group"] == "momentum").any()
    assert not (log["group"] == "ts").any()


def test_bad_ts_bounds_fails_heat_and_ts_not_momentum():
    tau_selected = pd.DataFrame([{"slot": _SLOT, "boom": _BOOM, "variant": "mrd", "tau_s": 37.5,
                                   "source": "momentum", "source_status": "found"}])
    candidate = pd.DataFrame([
        _means_row(_SLOT, _BOOM, "ue", "mean", 1.0),
        _means_row(_SLOT, _BOOM, "vn", "mean", 0.0),
        _means_row(_SLOT, _BOOM, "w", "mean", 0.0),
        _means_row(_SLOT, _BOOM, "ts", "mean", 290.0),
        _boom_stats_row(_SLOT, _BOOM, "mrd", "wvpts", 0.01),
        _coverage_row(_SLOT, _BOOM, "mrd", "momentum", 1.0),
        _coverage_row(_SLOT, _BOOM, "mrd", "heat", 1.0),
    ])
    coverage = pd.DataFrame([
        {"slot": _SLOT, "boom": _BOOM, "variable": "momentum", "layer": "usable", "fraction": 1.0},
        {"slot": _SLOT, "boom": _BOOM, "variable": "ts", "layer": "usable", "fraction": 1.0},
    ])
    # 5% of the slot flagged bounds on ts - well above max_bounds_fraction.
    flagged = int(0.05 * SAMPLES_PER_SLOT)
    flag_stores = {_BOOM: _flag_store(_BOOM, "bounds", "ts", [_G0], [_G0 + flagged])}

    fail_keys, log = filtering.evaluate(candidate, coverage, tau_selected, flag_stores, _Cfg())
    filtered = filtering.apply(candidate, fail_keys)

    assert filtered.loc[filtered["variable"] == "ts", "value"].isna().all()
    assert filtered.loc[filtered["variable"] == "wvpts", "value"].isna().all()
    assert not filtered.loc[filtered["variable"] == "ue", "value"].isna().any()
    assert not filtered.loc[filtered["variable"] == "vn", "value"].isna().any()
    assert not filtered.loc[filtered["variable"] == "w", "value"].isna().any()
    bounds_rows = log[log["criterion"] == "bounds"]
    assert set(bounds_rows["group"]) == {"heat", "ts"}


@pytest.mark.parametrize("fraction,expect_fail", [(0.009, False), (0.011, True)])
def test_spike_fraction_threshold_flips_decision(fraction, expect_fail):
    candidate = pd.DataFrame([_means_row(_SLOT, _BOOM, "t", "mean", 290.0)])
    coverage = pd.DataFrame([{"slot": _SLOT, "boom": _BOOM, "variable": "t", "layer": "usable", "fraction": 1.0}])
    flagged = int(fraction * SAMPLES_PER_SLOT)
    flag_stores = {_BOOM: _flag_store(_BOOM, "spike", "t", [_G0], [_G0 + flagged])}

    fail_keys, _ = filtering.evaluate(candidate, coverage, _empty_tau_selected(), flag_stores, _Cfg())
    filtered = filtering.apply(candidate, fail_keys)
    assert filtered["value"].isna().iloc[0] == expect_fail


def test_1200s_tau_uses_20min_window_support():
    tau_selected = pd.DataFrame([{"slot": _SLOT, "boom": _BOOM, "variant": "mrd", "tau_s": 1200.0,
                                   "source": "momentum", "source_status": "found"}])
    candidate = pd.DataFrame([
        _boom_stats_row(_SLOT, _BOOM, "mrd", "ustar", 0.3),
        _coverage_row(_SLOT, _BOOM, "mrd", "momentum", 1.0),
    ])
    coverage = pd.DataFrame(columns=["slot", "boom", "variable", "layer", "fraction"])
    # A bounds flag entirely in the neighbouring slot's half (before slot
    # start, within the 20-min window's +-5 min reach but outside the slot).
    flag_stores = {_BOOM: _flag_store(_BOOM, "bounds", "ue", [_G0 - 10_000], [_G0 - 1_000])}

    fail_keys, log = filtering.evaluate(candidate, coverage, tau_selected, flag_stores, _Cfg())
    assert (_SLOT, _BOOM, "mrd", "momentum") in fail_keys
    assert (log["criterion"] == "bounds").any()


def test_slot_only_support_ignores_flags_outside_the_slot():
    tau_selected = pd.DataFrame([{"slot": _SLOT, "boom": _BOOM, "variant": "mrd", "tau_s": 37.5,
                                   "source": "momentum", "source_status": "found"}])
    candidate = pd.DataFrame([
        _boom_stats_row(_SLOT, _BOOM, "mrd", "ustar", 0.3),
        _coverage_row(_SLOT, _BOOM, "mrd", "momentum", 1.0),
    ])
    coverage = pd.DataFrame(columns=["slot", "boom", "variable", "layer", "fraction"])
    flag_stores = {_BOOM: _flag_store(_BOOM, "bounds", "ue", [_G0 - 10_000], [_G0 - 1_000])}

    fail_keys, _ = filtering.evaluate(candidate, coverage, tau_selected, flag_stores, _Cfg())
    assert (_SLOT, _BOOM, "mrd", "momentum") not in fail_keys


def test_low_variant_coverage_nans_only_that_variant():
    """Regression test: `naive` and `mrd_unexcised`-copied rows never have a
    primary `ladder_coverage` row of their own (naive borrows mrd's 600-s
    rung; a copied mrd_unexcised row comes from `boom_stats`, not primary).
    Coverage must come from `boom_stats`'s own per-variant rows in
    `candidate`, or every `naive`/copied-`mrd_unexcised` value is wrongly
    NaN'd regardless of its actual coverage.
    """
    tau_selected = pd.DataFrame([
        {"slot": _SLOT, "boom": _BOOM, "variant": "mrd", "tau_s": 37.5, "source": "momentum", "source_status": "found"},
        {"slot": _SLOT, "boom": _BOOM, "variant": "naive", "tau_s": 600.0, "source": "fixed", "source_status": "fixed"},
    ])
    candidate = pd.DataFrame([
        _boom_stats_row(_SLOT, _BOOM, "mrd", "ustar", 0.3),
        _boom_stats_row(_SLOT, _BOOM, "naive", "ustar", 0.3),
        _coverage_row(_SLOT, _BOOM, "mrd", "momentum", 0.5),
        _coverage_row(_SLOT, _BOOM, "naive", "momentum", 1.0),
    ])
    coverage = pd.DataFrame(columns=["slot", "boom", "variable", "layer", "fraction"])

    fail_keys, _ = filtering.evaluate(candidate, coverage, tau_selected, {}, _Cfg())
    filtered = filtering.apply(candidate, fail_keys)

    assert filtered.loc[filtered["variant"] == "mrd", "value"].isna().all()
    assert not filtered.loc[filtered["variant"] == "naive", "value"].isna().any()


def test_naive_variant_has_no_ladder_coverage_row_and_is_not_penalized_for_it():
    """The exact shape of the bug: `candidate` (built from real `boom_stats`)
    never has a primary-style `ladder_coverage` row for `naive` at all - only
    its own `boom_stats` coverage row. A `naive` value must survive on that
    alone, not be treated as zero coverage because no separate table has it.
    """
    tau_selected = pd.DataFrame([
        {"slot": _SLOT, "boom": _BOOM, "variant": "naive", "tau_s": 600.0, "source": "fixed", "source_status": "fixed"},
    ])
    candidate = pd.DataFrame([
        _boom_stats_row(_SLOT, _BOOM, "naive", "ustar", 0.3),
        _coverage_row(_SLOT, _BOOM, "naive", "momentum", 0.9),
    ])
    coverage = pd.DataFrame(columns=["slot", "boom", "variable", "layer", "fraction"])

    fail_keys, _ = filtering.evaluate(candidate, coverage, tau_selected, {}, _Cfg())
    filtered = filtering.apply(candidate, fail_keys)

    assert not filtered.loc[filtered["variable"] == "ustar", "value"].isna().any()


def test_apply_to_labels_nulls_aniso_class_when_momentum_group_fails():
    boom_labels = pd.DataFrame([
        {"slot": _SLOT, "boom": _BOOM, "variant": "mrd", "label": "aniso_class", "value": "2c"},
        {"slot": _SLOT, "boom": _BOOM, "variant": "naive", "label": "aniso_class", "value": "3c"},
    ])
    fail_keys = {(_SLOT, _BOOM, "mrd", "momentum")}

    result = filtering.apply_to_labels(boom_labels, fail_keys)

    mrd_row = result[result["variant"] == "mrd"]
    naive_row = result[result["variant"] == "naive"]
    assert pd.isna(mrd_row["value"].iloc[0])
    assert naive_row["value"].iloc[0] == "3c"


def test_apply_to_labels_leaves_an_already_null_class_alone():
    boom_labels = pd.DataFrame([
        {"slot": _SLOT, "boom": _BOOM, "variant": "mrd", "label": "aniso_class", "value": None},
    ])

    result = filtering.apply_to_labels(boom_labels, fail_keys=set())

    assert pd.isna(result["value"].iloc[0])
