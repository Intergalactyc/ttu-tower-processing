import shutil

import pandas as pd
import pytest

from ttu_tower import results, schema
from ttu_tower.config import config_from_dict
from ttu_tower.io import runs, store


def _cfg(tag: str, results_dir: str = ""):
    raw = {
        "tag": tag,
        "paths": {"raw_dirs": ["/data"], "results_dir": results_dir},
        "files": {"bad_records": [[1, 2]]},
    }
    return config_from_dict(raw)


def _write_run(tmp_path, tag: str, results_dir: str = ""):
    cfg = _cfg(tag, results_dir=results_dir)
    return runs.register_run(cfg, tmp_path / f"{tag}.toml", package_version="2.0.0")


def test_load_results_fragment_table(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    run_dir = _write_run(tmp_path, "runA", results_dir=str(tmp_path / "external"))

    df = schema.cast("means", pd.DataFrame({
        "slot": [0, 1], "boom": [1, 1], "variable": ["ue", "ue"], "stat": ["mean", "mean"], "value": [1.0, 2.0],
    }))
    store.write_fragment(run_dir / "primary" / "data" / "means", "unit1", df)
    store.write_json_atomic(run_dir / "primary" / "run_summary.json", {"stage": "primary"})

    out = results.load_results("runA", "means")
    assert len(out) == 2


def test_load_results_single_file_table(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    run_dir = _write_run(tmp_path, "runB")  # default results_dir

    df = schema.cast("files", pd.DataFrame({
        "path": ["a"], "name": ["a"], "record": [1], "name_time": [pd.Timestamp.now(tz="UTC")],
        "offset_min": [0], "file_start": [pd.Timestamp.now(tz="UTC")], "half_hour": [0],
        "n_rows": [90000], "status": ["accepted"],
    }))
    (run_dir / "primary").mkdir(parents=True)
    df.to_parquet(run_dir / "primary" / "files.parquet")
    store.write_json_atomic(run_dir / "primary" / "run_summary.json", {"stage": "primary"})

    out = results.load_results("runB", "files")
    assert len(out) == 1


def test_load_results_wide_concatenates_months_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    run_dir = _write_run(tmp_path, "runC")

    wide_dir = run_dir / "tertiary" / "wide"
    wide_dir.mkdir(parents=True)
    pd.DataFrame({"slot_start": pd.to_datetime(["2014-01-01"]), "x": [1]}).to_parquet(wide_dir / "2014-01.parquet")
    pd.DataFrame({"slot_start": pd.to_datetime(["2014-02-01"]), "x": [2]}).to_parquet(wide_dir / "2014-02.parquet")
    store.write_json_atomic(run_dir / "tertiary" / "run_summary.json", {"stage": "tertiary"})

    out = results.load_results("runC", "wide")
    assert out["x"].tolist() == [1, 2]


def test_load_results_unknown_tag_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    with pytest.raises(RuntimeError, match="unknown tag"):
        results.load_results("nope", "means")


def test_load_results_missing_run_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    run_dir = _write_run(tmp_path, "runD")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    with pytest.raises(RuntimeError, match="missing"):
        results.load_results("runD", "means")


def test_load_results_warns_when_run_unfinished(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    run_dir = _write_run(tmp_path, "runE")
    (run_dir / "primary" / "data" / "means").mkdir(parents=True)
    with pytest.warns(UserWarning):
        results.load_results("runE", "means")


def test_load_results_unknown_table_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    _write_run(tmp_path, "runF")
    with pytest.raises(ValueError, match="unknown table"):
        results.load_results("runF", "bogus")


def test_list_runs_reports_both_external_and_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    run_dir1 = _write_run(tmp_path, "run1", results_dir=str(tmp_path / "ext"))
    run_dir2 = _write_run(tmp_path, "run2")
    run_dir1.mkdir(parents=True)  # registration alone doesn't create the run directory
    run_dir2.mkdir(parents=True)
    df = results.list_runs()
    assert set(df["tag"]) == {"run1", "run2"}
    assert df["exists"].all()
