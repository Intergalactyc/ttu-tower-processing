import math

import numpy as np
import pandas as pd
from pytest import approx

from ttu_tower.config.model import DetectionConfig, SecondaryConfig, SelectionConfig
from ttu_tower.constants import HEIGHTS
from ttu_tower.physics import most
from ttu_tower.physics.earth import local_gravity
from ttu_tower.secondary.derived import (
    derived, materialize_missing_unexcised, naive_tau_selected, selected_stats, tau_tables,
)

_SIGN_TYPE = [0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.5, 1.0, 0.4, -0.3, -0.6, -0.4, -0.2, 0.3, 0.5]
_SCALES = [0.29296875 * 2.0**k for k in range(15)]

_SEC_CFG = SecondaryConfig(
    min_tau_its_ratio=10.0,
    detection=DetectionConfig(peak_significance_se=2.0, min_scale_s=0.5, min_tau_s=9.375, max_tau_s=1200.0),
    selection=SelectionConfig(rule="priority", priority=("heat", "momentum"), fallback_tau_s=600.0),
)


def _mrd_rows(slot, boom, variant, spectrum, values):
    return [
        {"slot": slot, "boom": boom, "variant": variant, "spectrum": spectrum, "scale_s": scale,
         "value": value, "se": 0.05, "n_pairs": 2 ** (15 - i)}
        for i, (scale, value) in enumerate(zip(_SCALES, values), start=1)
    ]


def test_tau_tables_wires_detect_and_select():
    mrd = pd.DataFrame(
        _mrd_rows(100, 1, "mrd", "wvpts", _SIGN_TYPE) + _mrd_rows(100, 1, "mrd", "uw", _SIGN_TYPE)
    )
    coverage = pd.DataFrame([
        {"slot": 100, "boom": 1, "variable": "heat", "layer": "usable", "fraction": 0.9},
        {"slot": 100, "boom": 1, "variable": "momentum", "layer": "usable", "fraction": 0.9},
    ])

    tau_df, tau_selected_df = tau_tables(mrd, coverage, _SEC_CFG)

    assert len(tau_df) == 2  # one row per cospectrum
    heat_row = tau_df[tau_df["cospectrum"] == "heat"].iloc[0]
    momentum_row = tau_df[tau_df["cospectrum"] == "momentum"].iloc[0]
    assert heat_row["status"] == "found"
    assert momentum_row["status"] == "found"
    assert heat_row["tau_s"] == approx(75.0)

    assert len(tau_selected_df) == 1
    sel = tau_selected_df.iloc[0]
    assert sel["source"] == "heat"  # heat has priority and is "found"
    assert sel["source_status"] == "found"
    assert sel["tau_s"] == approx(75.0)


def test_tau_tables_no_data_family():
    mrd = pd.DataFrame(
        _mrd_rows(100, 1, "mrd", "wvpts", [np.nan] * 15) + _mrd_rows(100, 1, "mrd", "uw", _SIGN_TYPE)
    )
    coverage = pd.DataFrame([
        {"slot": 100, "boom": 1, "variable": "heat", "layer": "usable", "fraction": 0.0},
        {"slot": 100, "boom": 1, "variable": "momentum", "layer": "usable", "fraction": 0.9},
    ])
    tau_df, tau_selected_df = tau_tables(mrd, coverage, _SEC_CFG)
    heat_row = tau_df[tau_df["cospectrum"] == "heat"].iloc[0]
    assert heat_row["status"] == "no_data"
    sel = tau_selected_df.iloc[0]
    assert sel["source"] == "momentum"


def test_naive_tau_selected_is_fixed_600s():
    mrd_selected = pd.DataFrame([
        {"slot": 100, "boom": 1, "variant": "mrd", "tau_s": 75.0, "source": "heat", "source_status": "found"},
        {"slot": 101, "boom": 2, "variant": "mrd", "tau_s": np.nan, "source": "none", "source_status": "none"},
    ])
    naive = naive_tau_selected(mrd_selected)
    assert (naive["variant"] == "naive").all()
    assert (naive["tau_s"] == 600.0).all()
    assert (naive["source"] == "fixed").all()
    assert (naive["source_status"] == "fixed").all()
    assert len(naive) == 2


def _ladder_row(slot, boom, variant, rung_s, variable, stat, value):
    return {"slot": slot, "boom": boom, "variant": variant, "rung_s": rung_s, "variable": variable, "stat": stat, "value": value}


def _coverage_row(slot, boom, variant, rung_s, family, used, total):
    return {"slot": slot, "boom": boom, "variant": variant, "rung_s": rung_s, "family": family,
            "blocks_used": used, "blocks_total": total, "coverage": used / total}


def test_selected_stats_applies_pressure_factor_and_naive_borrows_mrd():
    ladder = pd.DataFrame([
        _ladder_row(100, 1, "mrd", 75.0, "u", "var", 1.0),
        _ladder_row(100, 1, "mrd", 75.0, "wvpts", "cov", 0.02),
        _ladder_row(100, 1, "mrd", 75.0, "vpts", "mean", 300.0),
        _ladder_row(100, 1, "mrd", 75.0, "vpts", "var", 4.0),
        _ladder_row(100, 1, "mrd", 600.0, "u", "var", 2.0),
    ])
    ladder_coverage = pd.DataFrame([
        _coverage_row(100, 1, "mrd", 75.0, "momentum", 8, 8),
        _coverage_row(100, 1, "mrd", 600.0, "momentum", 1, 1),
    ])
    tau_selected = pd.concat([
        pd.DataFrame([{"slot": 100, "boom": 1, "variant": "mrd", "tau_s": 75.0, "source": "heat", "source_status": "found"}]),
        naive_tau_selected(pd.DataFrame([{"slot": 100, "boom": 1, "variant": "mrd", "tau_s": 75.0}])),
    ], ignore_index=True)
    p_factor = pd.Series([2.0], index=pd.MultiIndex.from_tuples([(100, 1)], names=["slot", "boom"]))

    stats = selected_stats(ladder, ladder_coverage, tau_selected, p_factor)

    def value(variant, variable, stat):
        m = stats[(stats["variant"] == variant) & (stats["variable"] == variable) & (stats["stat"] == stat)]
        assert len(m) == 1
        return m["value"].iloc[0]

    assert value("mrd", "u", "var") == approx(1.0)  # untouched by p_factor
    assert value("mrd", "wvpts", "cov") == approx(0.04)  # x p_factor
    assert value("mrd", "vpts", "mean") == approx(600.0)  # x p_factor
    assert value("mrd", "vpts", "var") == approx(16.0)  # x p_factor^2
    assert value("mrd", "momentum", "coverage") == approx(1.0)

    # naive borrows the mrd variant's 600-s rung, not its own 75-s selected tau
    assert value("naive", "u", "var") == approx(2.0)


def test_selected_stats_empty_gives_empty_frame():
    empty = pd.DataFrame(columns=["slot", "boom", "variant", "tau_s", "source", "source_status"])
    out = selected_stats(pd.DataFrame(), pd.DataFrame(), empty, pd.Series(dtype=float))
    assert out.empty


def test_derived_quantities_hand_calculation():
    var_u, var_v, var_w = 1.0, 0.5, 0.25
    cov_uw, cov_vw, cov_uv = -0.1, 0.02, 0.05
    u_mean, w_mean, ws_mean, ws_std = 5.0, 0.1, 5.2, 0.8
    wvpts_cov = 0.05
    its_u, its_v, its_w, its_vpts = 5.0, 6.0, 7.0, 6.5
    tau_s = 75.0
    boom = 4
    vpt_slow = 300.0

    stats_rows = [
        (100, boom, "mrd", "u", "var", var_u), (100, boom, "mrd", "v", "var", var_v), (100, boom, "mrd", "w", "var", var_w),
        (100, boom, "mrd", "uw", "cov", cov_uw), (100, boom, "mrd", "vw", "cov", cov_vw), (100, boom, "mrd", "uv", "cov", cov_uv),
        (100, boom, "mrd", "u", "mean", u_mean), (100, boom, "mrd", "w", "mean", w_mean),
        (100, boom, "mrd", "ws", "mean", ws_mean), (100, boom, "mrd", "ws", "std", ws_std),
        (100, boom, "mrd", "wvpts", "cov", wvpts_cov),
        (100, boom, "mrd", "u", "its", its_u), (100, boom, "mrd", "v", "its", its_v),
        (100, boom, "mrd", "w", "its", its_w), (100, boom, "mrd", "vpts", "its", its_vpts),
        (100, boom, "mrd", "its", "blocks_used", 5.0), (100, boom, "mrd", "its_vpts", "blocks_used", 5.0),
    ]
    stats = pd.DataFrame(stats_rows, columns=["slot", "boom", "variant", "variable", "stat", "value"])
    tau_selected = pd.DataFrame([{"slot": 100, "boom": boom, "variant": "mrd", "tau_s": tau_s}])
    slow = pd.DataFrame([{"slot": 100, "boom": boom, "variable": "vpt", "value": vpt_slow}])

    from types import SimpleNamespace
    cfg = SimpleNamespace(secondary=_SEC_CFG)

    stat_df, label_df = derived(stats, tau_selected, slow, cfg)

    def value(variable):
        m = stat_df[stat_df["variable"] == variable]
        assert len(m) == 1
        return m["value"].iloc[0]

    assert value("sigma_u") == approx(math.sqrt(var_u))
    assert value("tke") == approx(0.5 * (var_u + var_v + var_w))
    assert value("ti") == approx(ws_std / ws_mean)
    assert value("ti_u") == approx(math.sqrt(var_u) / ws_mean)
    assert value("ctke") == approx(0.5 * math.sqrt(cov_uw**2 + cov_vw**2 + cov_uv**2))

    expected_ustar = most.friction_velocity(cov_uw, cov_vw)
    assert value("ustar") == approx(expected_ustar)

    g = local_gravity(33.61055, 1014.0 + HEIGHTS[boom])
    expected_l = most.obukhov_length(expected_ustar, vpt_slow, wvpts_cov, gravity=g)
    assert value("obukhov_length") == approx(expected_l)
    assert value("zeta") == approx(HEIGHTS[boom] / expected_l)

    assert value("ils_u") == approx(its_u * u_mean)
    assert value("ils_v") == approx(its_v * abs(u_mean))
    assert value("ils_vpts") == approx(its_vpts * abs(u_mean))
    assert value("its_ratio_u") == approx(tau_s / its_u)
    assert value("its_short") == approx(0.0)  # tau_s/its_x >= 10 for u, v, w -> not short
    assert value("w_ratio") == approx(w_mean / ws_mean)

    a1, a2, a3 = value("aniso_l1"), value("aniso_l2"), value("aniso_l3")
    assert a1 >= a2 >= a3
    assert math.isclose(a1 + a2 + a3, 0.0, abs_tol=1e-9)

    label_row = label_df[(label_df["slot"] == 100) & (label_df["boom"] == boom)].iloc[0]
    assert label_row["label"] == "aniso_class"
    assert label_row["value"] is not None


def test_its_short_flagged_when_ratio_below_threshold():
    stats_rows = [
        (100, 1, "mrd", "u", "var", 1.0), (100, 1, "mrd", "v", "var", 1.0), (100, 1, "mrd", "w", "var", 1.0),
        (100, 1, "mrd", "uw", "cov", 0.0), (100, 1, "mrd", "vw", "cov", 0.0), (100, 1, "mrd", "uv", "cov", 0.0),
        (100, 1, "mrd", "u", "mean", 5.0), (100, 1, "mrd", "w", "mean", 0.0),
        (100, 1, "mrd", "ws", "mean", 5.0), (100, 1, "mrd", "ws", "std", 0.5),
        (100, 1, "mrd", "wvpts", "cov", 0.01),
        # u, v: tau_s/its = 75/7 ~= 10.7 >= 10 -> fine. w: 75/20 = 3.75 < 10 -> short.
        (100, 1, "mrd", "u", "its", 7.0), (100, 1, "mrd", "v", "its", 7.0), (100, 1, "mrd", "w", "its", 20.0),
        (100, 1, "mrd", "vpts", "its", 7.0),
        (100, 1, "mrd", "its", "blocks_used", 5.0), (100, 1, "mrd", "its_vpts", "blocks_used", 5.0),
    ]
    stats = pd.DataFrame(stats_rows, columns=["slot", "boom", "variant", "variable", "stat", "value"])
    tau_selected = pd.DataFrame([{"slot": 100, "boom": 1, "variant": "mrd", "tau_s": 75.0}])
    slow = pd.DataFrame([{"slot": 100, "boom": 1, "variable": "vpt", "value": 300.0}])

    from types import SimpleNamespace
    cfg = SimpleNamespace(secondary=_SEC_CFG)
    stat_df, _ = derived(stats, tau_selected, slow, cfg)
    its_short = stat_df[stat_df["variable"] == "its_short"]["value"].iloc[0]
    assert its_short == 1.0


def test_its_short_nan_when_no_usable_block():
    stats_rows = [
        (100, 1, "mrd", "u", "var", 1.0), (100, 1, "mrd", "v", "var", 1.0), (100, 1, "mrd", "w", "var", 1.0),
        (100, 1, "mrd", "uw", "cov", 0.0), (100, 1, "mrd", "vw", "cov", 0.0), (100, 1, "mrd", "uv", "cov", 0.0),
        (100, 1, "mrd", "u", "mean", 5.0), (100, 1, "mrd", "w", "mean", 0.0),
        (100, 1, "mrd", "ws", "mean", 5.0), (100, 1, "mrd", "ws", "std", 0.5),
        (100, 1, "mrd", "wvpts", "cov", 0.01),
        (100, 1, "mrd", "u", "its", np.nan), (100, 1, "mrd", "v", "its", np.nan), (100, 1, "mrd", "w", "its", np.nan),
        (100, 1, "mrd", "vpts", "its", np.nan),
        (100, 1, "mrd", "its", "blocks_used", 0.0), (100, 1, "mrd", "its_vpts", "blocks_used", 0.0),
    ]
    stats = pd.DataFrame(stats_rows, columns=["slot", "boom", "variant", "variable", "stat", "value"])
    tau_selected = pd.DataFrame([{"slot": 100, "boom": 1, "variant": "mrd", "tau_s": 75.0}])
    slow = pd.DataFrame([{"slot": 100, "boom": 1, "variable": "vpt", "value": 300.0}])

    from types import SimpleNamespace
    cfg = SimpleNamespace(secondary=_SEC_CFG)
    stat_df, _ = derived(stats, tau_selected, slow, cfg)
    its_short = stat_df[stat_df["variable"] == "its_short"]["value"].iloc[0]
    assert math.isnan(its_short)


def test_materialize_missing_unexcised_copies_mrd_rows():
    slot_boom = pd.DataFrame([
        {"slot": 100, "boom": 1, "status": "computed", "unexcised_computed": False},
        {"slot": 101, "boom": 1, "status": "computed", "unexcised_computed": True},
        {"slot": 102, "boom": 1, "status": "no_data", "unexcised_computed": False},
    ])
    tau_selected = pd.DataFrame([
        {"slot": 100, "boom": 1, "variant": "mrd", "tau_s": 75.0, "source": "heat", "source_status": "found"},
        {"slot": 101, "boom": 1, "variant": "mrd", "tau_s": 150.0, "source": "heat", "source_status": "found"},
        {"slot": 101, "boom": 1, "variant": "mrd_unexcised", "tau_s": 300.0, "source": "heat", "source_status": "found"},
    ])
    out = materialize_missing_unexcised({"tau_selected": tau_selected}, slot_boom)["tau_selected"]

    copied = out[(out["slot"] == 100) & (out["variant"] == "mrd_unexcised")]
    assert len(copied) == 1
    assert copied["tau_s"].iloc[0] == approx(75.0)

    # slot 101 already had a real mrd_unexcised row - not touched/duplicated.
    real = out[(out["slot"] == 101) & (out["variant"] == "mrd_unexcised")]
    assert len(real) == 1
    assert real["tau_s"].iloc[0] == approx(300.0)

    # slot 102 (no_data) gets no mrd_unexcised row at all.
    assert out[(out["slot"] == 102) & (out["variant"] == "mrd_unexcised")].empty
