import sys

import numpy as np
import pandas as pd
import pytest

from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.tertiary import mesonet


def _synthetic_mesonet_df(seed: int = 0, n_rows: int = 60) -> pd.DataFrame:
    """A DataFrame matching mesonet._COL_MAP's raw column names, with a
    5-min-cadence DatetimeIndex, engineered so scalar-mean and vector-mean
    wind speed provably differ: constant instantaneous speed but rotating
    direction across the averaging window.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2013-11-01 00:00", periods=n_rows, freq="5min", tz="UTC")
    speed = np.full(n_rows, 5.0)
    direction = np.linspace(0, 360, n_rows, endpoint=False)
    return pd.DataFrame(
        {
            "10m WS scalar": speed,
            "10m WS vector": speed,
            "10m WD": direction,
            "WD Standard Deviation": rng.uniform(1, 10, n_rows),
            "WS Standard Deviation": rng.uniform(0.1, 1, n_rows),
            "9m Temp": rng.uniform(10, 25, n_rows),
            "1.5m RH": rng.uniform(20, 80, n_rows),
            "barometric pressure": rng.uniform(950, 1000, n_rows),
            "2m Solar Radiation": rng.uniform(0, 800, n_rows),
            "Precip": np.zeros(n_rows),
        },
        index=index,
    )


def test_ws_mean_is_scalar_not_vector(monkeypatch):
    synthetic = _synthetic_mesonet_df(n_rows=60)
    monkeypatch.setattr(mesonet, "get_df", lambda data_path: synthetic)

    result = mesonet.mesonet_data("unused-path", "UTC")

    assert "ws_meso_mean" in result.columns
    assert "ws_meso_vector_mean" in result.columns
    # ws is already m/s (no conversion); scalar mean survives resampling
    # unchanged, since instantaneous speed is constant in the fixture.
    assert result["ws_meso_mean"].dropna().iloc[0] == pytest.approx(5.0)
    # Vector-mean speed must be strictly smaller (direction rotates within
    # each averaging window, so vector-averaging cancels some of it out).
    assert (result["ws_meso_vector_mean"].dropna() < result["ws_meso_mean"].dropna()).all()


def test_column_renaming_and_unit_conversion(monkeypatch):
    synthetic = _synthetic_mesonet_df(n_rows=30)
    monkeypatch.setattr(mesonet, "get_df", lambda data_path: synthetic)

    result = mesonet.mesonet_data("unused-path", "UTC")

    for col in ["ws_meso_mean", "wd_meso_std", "ws_meso_std", "t_meso_mean", "rh_meso_mean", "p_meso_mean", "ti_meso"]:
        assert col in result.columns
    assert "_WSVEC" not in result.columns
    assert "_WD" not in result.columns

    # C -> K, % -> fraction, mb -> kPa.
    assert (result["t_meso_mean"] > 273.0).all()
    assert (result["rh_meso_mean"] <= 1.0).all() and (result["rh_meso_mean"] >= 0.0).all()
    assert result["p_meso_mean"].mean() == pytest.approx(97.5, abs=1.0)


def test_ti_meso_zero_speed_guard(monkeypatch):
    synthetic = _synthetic_mesonet_df(n_rows=10)
    synthetic["10m WS scalar"] = 0.0
    monkeypatch.setattr(mesonet, "get_df", lambda data_path: synthetic)

    result = mesonet.mesonet_data("unused-path", "UTC")

    assert result["ti_meso"].isna().all()


def test_fills_gap_spanning_a_full_output_bin(monkeypatch):
    # Native cadence is 5min; the fixed 10-min output cadence means a real
    # multi-tick mesonet outage (not just "finer output grid than native")
    # is what actually produces a hole to interpolate across.
    synthetic = _synthetic_mesonet_df(n_rows=12)
    synthetic.loc[synthetic.index[4:6], "9m Temp"] = np.nan
    monkeypatch.setattr(mesonet, "get_df", lambda data_path: synthetic)

    result = mesonet.mesonet_data("unused-path", "UTC")

    assert not result["t_meso_mean"].isna().any()
    v0, v1, v2 = result["t_meso_mean"].iloc[1], result["t_meso_mean"].iloc[2], result["t_meso_mean"].iloc[3]
    assert v1 == pytest.approx((v0 + v2) / 2)


def test_extrapolates_leading_trailing_gaps(monkeypatch):
    synthetic = _synthetic_mesonet_df(n_rows=8)
    synthetic.loc[synthetic.index[0:2], "9m Temp"] = np.nan
    synthetic.loc[synthetic.index[-2:], "9m Temp"] = np.nan
    monkeypatch.setattr(mesonet, "get_df", lambda data_path: synthetic)

    result = mesonet.mesonet_data("unused-path", "UTC")

    assert not result["t_meso_mean"].isna().any()
    v1, v2 = result["t_meso_mean"].iloc[1], result["t_meso_mean"].iloc[2]
    assert result["t_meso_mean"].iloc[0] == pytest.approx(v1 - (v2 - v1))


def test_direction_consistent_with_gap_filled_vector_components(monkeypatch):
    synthetic = _synthetic_mesonet_df(n_rows=12)
    monkeypatch.setattr(mesonet, "get_df", lambda data_path: synthetic)

    result = mesonet.mesonet_data("unused-path", "UTC")

    expected_speed, expected_dir = vector_to_bearing(result["u_meso_mean"], result["v_meso_mean"])
    np.testing.assert_allclose(result["ws_meso_vector_mean"].to_numpy(), expected_speed.to_numpy())
    np.testing.assert_allclose(result["wd_meso_mean"].to_numpy(), expected_dir.to_numpy())


def test_index_localized_to_requested_timezone(monkeypatch):
    synthetic = _synthetic_mesonet_df(n_rows=6)
    monkeypatch.setattr(mesonet, "get_df", lambda data_path: synthetic)

    result = mesonet.mesonet_data("unused-path", "Etc/GMT+6")

    assert str(result.index.tz) == "Etc/GMT+6"
    assert result.index.name == "slot_start"


def test_combine_non_overlapping_meso_rows_nan_not_dropped():
    slots = pd.DataFrame({
        "slot": [1, 2, 3],
        "slot_start": pd.date_range("2013-11-01", periods=3, freq="30min", tz="UTC"),
    })
    meso = pd.DataFrame(
        {"meso_val": [10.0]},
        index=pd.DatetimeIndex(["2013-11-01 00:30"], tz="UTC", name="slot_start"),
    )

    result = mesonet.combine(slots, meso)

    assert len(result) == 3
    assert result["meso_val"].isna().sum() == 2
    assert result.loc[result["slot"] == 2, "meso_val"].iloc[0] == 10.0


def test_get_df_raises_clear_error_without_wtxmeso(monkeypatch):
    monkeypatch.setitem(sys.modules, "wtxmeso", None)
    with pytest.raises(ImportError, match="mesonet"):
        mesonet.get_df("unused-path")
