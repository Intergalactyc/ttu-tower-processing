import pandas as pd
import pytest

from ttu_tower import schema


def test_every_table_has_a_spec():
    expected = {
        "files", "slots", "slot_boom", "coverage", "means", "slot_qc", "flags",
        "ladder", "ladder_coverage", "mrd", "mrd_frame",
        "tau", "tau_selected", "boom_stats", "boom_labels", "slow", "slot_stats",
        "boom_final", "tau_final", "pairs", "profile", "slot_final", "filter_log",
        "boom_labels_final", "wide",
    }
    assert set(schema.TABLES) == expected


def test_cast_orders_columns_and_casts_dtype():
    df = pd.DataFrame({
        "boom": [1, 2], "value": [1, 2], "variable": ["ue", "vn"], "slot": [0, 1], "stat": ["mean", "mean"],
    })
    out = schema.cast("means", df)
    assert list(out.columns) == ["slot", "boom", "variable", "stat", "value"]
    assert out["value"].dtype == "float64"
    assert str(out["variable"].dtype) == "category"
    assert out["boom"].dtype == "int8"


def test_cast_rejects_missing_column():
    df = pd.DataFrame({"slot": [0], "boom": [1], "variable": ["ue"], "value": [1.0]})
    with pytest.raises(schema.SchemaError, match="missing"):
        schema.cast("means", df)


def test_cast_rejects_extra_column():
    df = pd.DataFrame({
        "slot": [0], "boom": [1], "variable": ["ue"], "stat": ["mean"], "value": [1.0], "bogus": [1],
    })
    with pytest.raises(schema.SchemaError, match="extra"):
        schema.cast("means", df)


def test_cast_nullable_int_columns():
    df = pd.DataFrame({
        "path": ["a"], "name": ["a"], "record": [None], "name_time": [pd.Timestamp.now(tz="UTC")],
        "offset_min": [0], "file_start": [pd.Timestamp.now(tz="UTC")], "half_hour": [0],
        "n_rows": [90000], "status": ["accepted"],
    })
    out = schema.cast("files", df)
    assert out["record"].isna().all()
    assert out["n_rows"].iloc[0] == 90000


def test_cast_timestamp_column_left_as_datetime():
    df = pd.DataFrame({
        "slot": [0], "slot_start": [pd.Timestamp("2014-01-01", tz="Etc/GMT+6")], "half_hour": [0],
        "file_status": ["file"], "record": [1],
    })
    out = schema.cast("slots", df)
    assert pd.api.types.is_datetime64_any_dtype(out["slot_start"])
    assert out["slot_start"].dt.tz is not None
