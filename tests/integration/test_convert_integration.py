"""Integration tests against real production data and the old repo's raw-file
regression fixture (both read-only).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import ROWS_PER_FILE
from ttu_tower.io.convert import convert_raw_file
from ttu_tower.io.load import load_boom

_FIXTURE_ZIP = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "FT2_E07_C03_R01457_D20131101_T0000.zip"
_REAL_PARQUET_NAME = "FT2_E07_C03_R01457_D20131101_T0000.parquet"

_VARS = ("u", "v", "w", "ts", "t", "rh", "p")


@pytest.mark.integration
def test_converter_reproduces_real_parquet(tmp_path, raw_dir):
    df = convert_raw_file(str(_FIXTURE_ZIP), str(tmp_path))
    real = pd.read_parquet(raw_dir / _REAL_PARQUET_NAME).iloc[: len(df)].reset_index(drop=True)

    assert list(df.columns) == list(real.columns)
    assert set(df.dtypes.unique()) == {np.dtype("float32")}
    for col in df.columns:
        np.testing.assert_array_equal(df[col].to_numpy(), real[col].to_numpy())


@pytest.mark.integration
def test_load_boom_returns_90000_samples_for_a_real_file(raw_dir):
    result = load_boom(raw_dir / _REAL_PARQUET_NAME, boom=5, temp_dir=str(raw_dir))
    assert set(result) == set(_VARS)
    for v in _VARS:
        assert result[v].shape == (ROWS_PER_FILE,)
        assert result[v].dtype == np.float64


@pytest.mark.integration
def test_load_boom_raw_route_matches_parquet_route(tmp_path, raw_dir, monkeypatch):
    # The fixture zip is truncated to 3000 rows, not a full 90,000-row
    # half-hour file - compare against a matching truncated parquet so both
    # routes see the same length.
    monkeypatch.setattr("ttu_tower.io.load.ROWS_PER_FILE", 3000)
    boom = 5

    truncated_parquet = tmp_path / "truncated.parquet"
    pd.read_parquet(raw_dir / _REAL_PARQUET_NAME).iloc[:3000].reset_index(drop=True).to_parquet(truncated_parquet)

    raw_result = load_boom(str(_FIXTURE_ZIP), boom=boom, temp_dir=str(tmp_path))
    parquet_result = load_boom(str(truncated_parquet), boom=boom, temp_dir=str(tmp_path))

    for v in _VARS:
        np.testing.assert_array_equal(raw_result[v], parquet_result[v])
