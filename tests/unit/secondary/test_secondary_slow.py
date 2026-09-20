from types import SimpleNamespace

import numpy as np
import pandas as pd
from pytest import approx

from ttu_tower.constants import P_REF
from ttu_tower.physics import thermo
from ttu_tower.physics.constants import R_CP
from ttu_tower.secondary.slow import slow_table

_CFG = SimpleNamespace(qc=SimpleNamespace(min_coverage=0.75))


def _means_row(slot, boom, variable, value):
    return {"slot": slot, "boom": boom, "variable": variable, "stat": "mean", "value": value}


def _coverage_row(slot, boom, variable, layer, fraction):
    return {"slot": slot, "boom": boom, "variable": variable, "layer": layer, "fraction": fraction}


def _row(df, slot, boom, variable):
    match = df[(df["slot"] == slot) & (df["boom"] == boom) & (df["variable"] == variable)]
    assert len(match) == 1, f"expected exactly one ({slot},{boom},{variable}) row, got {len(match)}"
    return match["value"].iloc[0]


def test_slow_thermodynamics_and_pressure_factor_hand_rows():
    t1, rh1, p1, vpts1 = 290.0, 0.5, 90.0, 291.0
    t2, rh2, p2, vpts2 = 275.0, 0.8, 85.0, 274.0

    means = pd.DataFrame([
        *[_means_row(100, 1, v, val) for v, val in zip(("t", "rh", "p", "vpts"), (t1, rh1, p1, vpts1))],
        *[_means_row(100, 2, v, val) for v, val in zip(("t", "rh", "p", "vpts"), (t2, rh2, p2, vpts2))],
    ])
    coverage = pd.DataFrame([
        _coverage_row(100, 1, "p", "usable", 0.99),  # >= c: pressure is measured
        _coverage_row(100, 2, "p", "usable", 0.10),  # < c: falls back to reference
    ])

    out = slow_table(means, coverage, _CFG)

    es1 = thermo.saturation_vapor_pressure(t1)
    e1 = rh1 * es1
    r1 = thermo.water_air_mixing_ratio(e1, p1)
    pt1 = thermo.potential_temperature(t1, p1)
    vpt1 = thermo.virtual_potential_temperature(pt1, r1)

    assert _row(out, 100, 1, "es") == approx(es1, rel=1e-12)
    assert _row(out, 100, 1, "e") == approx(e1, rel=1e-12)
    assert _row(out, 100, 1, "r") == approx(r1, rel=1e-12)
    assert _row(out, 100, 1, "q") == approx(r1 / (1 + r1), rel=1e-12)
    assert _row(out, 100, 1, "pt") == approx(pt1, rel=1e-12)
    assert _row(out, 100, 1, "vt") == approx(t1 * (1 + r1 / 0.622) / (1 + r1), rel=1e-12)
    assert _row(out, 100, 1, "vpt") == approx(vpt1, rel=1e-12)
    assert _row(out, 100, 1, "td") == approx(thermo.dewpoint_temperature(t1, rh1), rel=1e-12)

    expected_factor_1 = (P_REF[1] / p1) ** R_CP
    assert _row(out, 100, 1, "p_factor") == approx(expected_factor_1, rel=1e-12)
    assert _row(out, 100, 1, "p_measured") == 1.0
    assert _row(out, 100, 1, "vpts") == approx(vpts1 * expected_factor_1, rel=1e-12)

    # boom 2: pressure coverage below c -> reference pressure kept
    assert _row(out, 100, 2, "p_factor") == approx(1.0, abs=1e-12)
    assert _row(out, 100, 2, "p_measured") == 0.0
    assert _row(out, 100, 2, "vpts") == approx(vpts2, rel=1e-12)


def test_slow_table_empty_input_gives_empty_output():
    means = pd.DataFrame(columns=["slot", "boom", "variable", "stat", "value"])
    coverage = pd.DataFrame(columns=["slot", "boom", "variable", "layer", "fraction"])
    out = slow_table(means, coverage, _CFG)
    assert out.empty


def test_slow_table_missing_pressure_coverage_row_treated_as_zero():
    means = pd.DataFrame([
        _means_row(200, 3, "t", 280.0), _means_row(200, 3, "rh", 0.4),
        _means_row(200, 3, "p", 80.0), _means_row(200, 3, "vpts", 281.0),
    ])
    coverage = pd.DataFrame(columns=["slot", "boom", "variable", "layer", "fraction"])
    out = slow_table(means, coverage, _CFG)
    assert _row(out, 200, 3, "p_measured") == 0.0
    assert _row(out, 200, 3, "p_factor") == approx(1.0)
    assert not np.isnan(_row(out, 200, 3, "vpts"))
