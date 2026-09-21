from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import HEIGHTS
from ttu_tower.math.fits import power_fit
from ttu_tower.physics.most import constrained_neutral_loglaw_fit, neutral_loglaw_fit
from ttu_tower.tertiary.profiles import profiles

_SLOT = 7
_BOOMS = [1, 2, 3, 4]


@dataclass(frozen=True)
class _FitsCfg:
    exclude_tau: tuple = ("unresolved", "fallback")
    min_booms: int = 2
    alpha_booms: tuple = tuple(_BOOMS)
    gamma_booms: tuple = tuple(_BOOMS)
    wdgamma_booms: tuple = tuple(_BOOMS)
    loglaw_booms: tuple = tuple(_BOOMS)


def _none_row(slot, boom, variable, value, stat="mean"):
    return {"slot": slot, "boom": boom, "variant": "none", "variable": variable, "stat": stat, "value": value}


def _stat_row(slot, boom, variant, variable, value):
    return {"slot": slot, "boom": boom, "variant": variant, "variable": variable, "stat": None, "value": value}


def _tau_row(slot, boom, variant, status):
    return {"slot": slot, "boom": boom, "variant": variant, "tau_s": 37.5, "source": status, "source_status": status}


def _get(df, slot, variant, variable):
    row = df[(df["slot"] == slot) & (df["variant"] == variant) & (df["variable"] == variable)]
    assert len(row) == 1, f"expected exactly one ({slot}, {variant}, {variable}) row, got {len(row)}"
    return row["value"].iloc[0]


def test_alpha_matches_direct_power_fit_and_counts_booms():
    ws = {1: 4.0, 2: 5.0, 3: 6.0, 4: np.nan}
    boom_final = pd.DataFrame([_none_row(_SLOT, b, "ws", v) for b, v in ws.items()])
    tau_selected = pd.DataFrame(columns=["slot", "boom", "variant", "tau_s", "source", "source_status"])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    heights = np.array([HEIGHTS[b] for b in _BOOMS])
    y = np.array([ws[b] for b in _BOOMS])
    _, expected_exponent = power_fit(heights, y, require=2)
    assert _get(result, _SLOT, "none", "alpha") == pytest.approx(expected_exponent)
    assert _get(result, _SLOT, "none", "alpha_n") == 3.0


def test_gamma_excludes_unresolved_boom_for_mrd_but_not_naive():
    ti = {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25}
    boom_final = pd.DataFrame(
        [_stat_row(_SLOT, b, "mrd", "ti", v) for b, v in ti.items()]
        + [_stat_row(_SLOT, b, "naive", "ti", v) for b, v in ti.items()],
    )
    tau_selected = pd.DataFrame([
        _tau_row(_SLOT, 1, "mrd", "found"), _tau_row(_SLOT, 2, "mrd", "found"),
        _tau_row(_SLOT, 3, "mrd", "found"), _tau_row(_SLOT, 4, "mrd", "unresolved"),
        _tau_row(_SLOT, 1, "naive", "fixed"), _tau_row(_SLOT, 2, "naive", "fixed"),
        _tau_row(_SLOT, 3, "naive", "fixed"), _tau_row(_SLOT, 4, "naive", "fixed"),
    ])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    assert _get(result, _SLOT, "mrd", "gamma_n") == 3.0
    assert _get(result, _SLOT, "naive", "gamma_n") == 4.0

    heights_excl = np.array([HEIGHTS[b] for b in (1, 2, 3)])
    y_excl = np.array([ti[b] for b in (1, 2, 3)])
    _, expected_mrd = power_fit(heights_excl, y_excl, require=2)
    assert _get(result, _SLOT, "mrd", "gamma") == pytest.approx(expected_mrd)


def test_gamma_counts_capped_boom_as_included():
    ti = {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25}
    boom_final = pd.DataFrame([_stat_row(_SLOT, b, "mrd", "ti", v) for b, v in ti.items()])
    tau_selected = pd.DataFrame([
        _tau_row(_SLOT, 1, "mrd", "found"), _tau_row(_SLOT, 2, "mrd", "found"),
        _tau_row(_SLOT, 3, "mrd", "capped"), _tau_row(_SLOT, 4, "mrd", "found"),
    ])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    assert _get(result, _SLOT, "mrd", "gamma_n") == 4.0
    assert _get(result, _SLOT, "mrd", "gamma_n_capped") == 1.0


def test_min_booms_floor_gives_nan_fit():
    ti = {1: 0.1, 2: np.nan, 3: np.nan, 4: np.nan}
    boom_final = pd.DataFrame([_stat_row(_SLOT, b, "mrd", "ti", v) for b, v in ti.items()])
    tau_selected = pd.DataFrame([_tau_row(_SLOT, b, "mrd", "found") for b in _BOOMS])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    assert _get(result, _SLOT, "mrd", "gamma_n") == 1.0
    assert np.isnan(_get(result, _SLOT, "mrd", "gamma"))


def test_loglaw_unconstrained_matches_direct_fit():
    ws = {1: 4.0, 2: 5.0, 3: 6.0, 4: 7.0}
    boom_final = pd.DataFrame([_none_row(_SLOT, b, "ws", v) for b, v in ws.items()])
    tau_selected = pd.DataFrame(columns=["slot", "boom", "variant", "tau_s", "source", "source_status"])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    heights = [HEIGHTS[b] for b in _BOOMS]
    y = [ws[b] for b in _BOOMS]
    expected_ustar, expected_z0 = neutral_loglaw_fit(heights, y)
    assert _get(result, _SLOT, "none", "loglaw_ustar") == pytest.approx(expected_ustar)
    assert _get(result, _SLOT, "none", "loglaw_z0") == pytest.approx(expected_z0)


def test_loglaw_constrained_uses_median_ustar_over_included_booms():
    ws = {1: 4.0, 2: 5.0, 3: 6.0, 4: 7.0}
    ustar = {1: 0.2, 2: 0.3, 3: 0.4, 4: 10.0}  # boom 4 excluded (unresolved)
    boom_final = pd.DataFrame(
        [_none_row(_SLOT, b, "ws", v) for b, v in ws.items()]
        + [_stat_row(_SLOT, b, "mrd", "ustar", v) for b, v in ustar.items()],
    )
    tau_selected = pd.DataFrame([
        _tau_row(_SLOT, 1, "mrd", "found"), _tau_row(_SLOT, 2, "mrd", "found"),
        _tau_row(_SLOT, 3, "mrd", "found"), _tau_row(_SLOT, 4, "mrd", "unresolved"),
    ])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    heights_excl = [HEIGHTS[b] for b in (1, 2, 3)]
    y_excl = [ws[b] for b in (1, 2, 3)]
    expected_z0 = constrained_neutral_loglaw_fit(heights_excl, y_excl, ustar=float(np.median([0.2, 0.3, 0.4])))
    assert _get(result, _SLOT, "mrd", "loglaw_z0_constrained") == pytest.approx(expected_z0)
    assert _get(result, _SLOT, "mrd", "loglaw_z0_constrained_n") == 3.0


def test_loglaw_constrained_counts_capped_boom_as_included():
    ws = {1: 4.0, 2: 5.0, 3: 6.0, 4: 7.0}
    ustar = {1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}
    boom_final = pd.DataFrame(
        [_none_row(_SLOT, b, "ws", v) for b, v in ws.items()]
        + [_stat_row(_SLOT, b, "mrd", "ustar", v) for b, v in ustar.items()],
    )
    tau_selected = pd.DataFrame([
        _tau_row(_SLOT, 1, "mrd", "found"), _tau_row(_SLOT, 2, "mrd", "found"),
        _tau_row(_SLOT, 3, "mrd", "capped"), _tau_row(_SLOT, 4, "mrd", "found"),
    ])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    assert _get(result, _SLOT, "mrd", "loglaw_z0_constrained_n") == 4.0
    assert _get(result, _SLOT, "mrd", "loglaw_z0_constrained_n_capped") == 1.0


def test_loglaw_unconstrained_reports_a_single_shared_n():
    # loglaw_ustar/loglaw_z0 come from one shared fit and one boom count, so
    # they report a single loglaw_n rather than a duplicate per variable.
    ws = {1: 4.0, 2: 5.0, 3: 6.0, 4: np.nan}
    boom_final = pd.DataFrame([_none_row(_SLOT, b, "ws", v) for b, v in ws.items()])
    tau_selected = pd.DataFrame(columns=["slot", "boom", "variant", "tau_s", "source", "source_status"])

    result = profiles(boom_final, tau_selected, _FitsCfg())

    assert _get(result, _SLOT, "none", "loglaw_n") == 3.0
