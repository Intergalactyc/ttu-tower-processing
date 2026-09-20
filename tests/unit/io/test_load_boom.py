import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import SOURCE_HEADERS_NEW
from ttu_tower.io.load import BadFileError, load_boom

_VARS = ("u", "v", "w", "ts", "t", "rh", "p")


def _write_parquet(path, boom, n_rows):
    rng = np.random.default_rng(1)
    cols = {f"{v}_{boom}": rng.normal(size=n_rows).astype(np.float32) for v in _VARS}
    cols[f"u_{boom + 1}"] = rng.normal(size=n_rows).astype(np.float32)  # a decoy column
    pd.DataFrame(cols).to_parquet(path)
    return cols


def _write_raw_csv(path, n_rows):
    rng = np.random.default_rng(2)
    data = rng.normal(size=(n_rows, len(SOURCE_HEADERS_NEW))).astype(np.float32)
    lines = [",".join(f"{v:.6f}" for v in row) for row in data]
    path.write_text("\n".join(lines) + "\n")


def test_load_boom_parquet_returns_float64_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr("ttu_tower.io.load.ROWS_PER_FILE", 10)
    path = tmp_path / "f.parquet"
    cols = _write_parquet(path, boom=3, n_rows=10)

    result = load_boom(str(path), boom=3, temp_dir=str(tmp_path))

    assert set(result) == set(_VARS)
    for v in _VARS:
        assert result[v].dtype == np.float64
        np.testing.assert_allclose(result[v], cols[f"{v}_3"])


def test_load_boom_parquet_wrong_length_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ttu_tower.io.load.ROWS_PER_FILE", 10)
    path = tmp_path / "f.parquet"
    _write_parquet(path, boom=1, n_rows=5)
    with pytest.raises(BadFileError):
        load_boom(str(path), boom=1, temp_dir=str(tmp_path))


def test_load_boom_parquet_missing_boom_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ttu_tower.io.load.ROWS_PER_FILE", 10)
    path = tmp_path / "f.parquet"
    _write_parquet(path, boom=1, n_rows=10)
    with pytest.raises(Exception):
        load_boom(str(path), boom=5, temp_dir=str(tmp_path))


def test_load_boom_raw_route_uses_convert_raw_file(tmp_path, monkeypatch):
    monkeypatch.setattr("ttu_tower.io.load.ROWS_PER_FILE", 4)
    _write_raw_csv(tmp_path / "raw.csv", n_rows=4)

    result = load_boom(str(tmp_path / "raw.csv"), boom=3, temp_dir=str(tmp_path))

    assert set(result) == set(_VARS)
    assert result["u"].shape == (4,)
    assert result["u"].dtype == np.float64


def test_load_boom_raw_route_wrong_length_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ttu_tower.io.load.ROWS_PER_FILE", 999)
    _write_raw_csv(tmp_path / "raw.csv", n_rows=4)
    with pytest.raises(BadFileError):
        load_boom(str(tmp_path / "raw.csv"), boom=1, temp_dir=str(tmp_path))
