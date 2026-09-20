import json

import pandas as pd

from ttu_tower import schema
from ttu_tower.io import store


def test_write_fragment_and_read_table_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": pd.Categorical(["x", "y"])})
    store.write_fragment(tmp_path, "frag1", df)
    out = store.read_table(tmp_path)
    assert out["a"].tolist() == [1, 2]
    assert out["b"].tolist() == ["x", "y"]


def test_write_fragment_is_atomic_no_tmp_left(tmp_path):
    df = pd.DataFrame({"a": [1]})
    path = store.write_fragment(tmp_path, "frag", df)
    assert path.exists()
    assert not (tmp_path / "frag.parquet.tmp").exists()


def test_read_table_empty_directory(tmp_path):
    assert store.read_table(tmp_path).empty


def test_read_table_missing_directory(tmp_path):
    assert store.read_table(tmp_path / "missing").empty


def test_read_table_null_only_categorical_column_not_dropped(tmp_path):
    # One fragment's "variable" column is entirely null (boom-level flag rows);
    # another's is populated. Naive multi-file schema inference must not drop
    # the populated fragment's real values.
    boom_level = schema.cast("flags", pd.DataFrame({
        "start": [0], "end": [10], "test": ["direction"], "kind": ["quality"],
        "boom": [1], "variable": [None],
    }))
    per_variable = schema.cast("flags", pd.DataFrame({
        "start": [20], "end": [30], "test": ["bounds"], "kind": ["quality"],
        "boom": [1], "variable": ["ue"],
    }))
    store.write_fragment(tmp_path, "a", boom_level)
    store.write_fragment(tmp_path, "b", per_variable)

    out = store.read_table(tmp_path)
    assert len(out) == 2
    assert out.loc[out["test"] == "bounds", "variable"].iloc[0] == "ue"
    assert pd.isna(out.loc[out["test"] == "direction", "variable"].iloc[0])


def test_write_json_atomic(tmp_path):
    store.write_json_atomic(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert not (tmp_path / "x.json.tmp").exists()


def test_manifest_write_and_read(tmp_path):
    store.write_manifest_entry(tmp_path / "manifest", "unit1", {"status": "success"})
    store.write_manifest_entry(tmp_path / "manifest", "unit2", {"status": "failed"})
    manifest = store.read_manifest(tmp_path / "manifest")
    assert manifest == {"unit1": {"status": "success"}, "unit2": {"status": "failed"}}


def test_read_manifest_missing_directory(tmp_path):
    assert store.read_manifest(tmp_path / "missing") == {}
