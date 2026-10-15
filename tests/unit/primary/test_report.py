from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io.runs import find_home
from ttu_tower.primary.report import write_primary_reports
from ttu_tower.primary.runner import run_primary
from ttu_tower.timegrid import half_hour_to_time
from ttu_tower.validation.synthetic import write_raw_dataset


def _args(config, **overrides):
    base = dict(config=config, nproc=1, test=False, redo_failures=False, force=False, allow_non_parquet=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _cfg(raw_dir, tag, start, end, **sections):
    raw = {
        "tag": tag, "paths": {"raw_dirs": [str(raw_dir)]},
        "period": {"start": start, "end": end},
        "files": {"bad_records": [], "booms": [1]},
        "tertiary": {"veer_reference_boom": 1, "fits": {k: [1] for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    }
    raw.update(sections)
    return config_from_dict(raw)


def _out_of_bounds_burst(h, boom, df):
    # every file gets a flag to report, whatever the despike threshold
    df = df.copy()
    df.loc[df.index[1000:1003], f"u_{boom}"] = 500.0
    return df


def test_write_primary_reports_produces_all_csvs(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    raw_dir = tmp_path / "raw"
    half_hours = list(range(7000, 7008))
    rng = np.random.default_rng(3)
    write_raw_dataset(raw_dir, half_hours, [1], rng, mutate=_out_of_bounds_burst)

    start = half_hour_to_time(7001).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(7007).strftime("%Y-%m-%d %H:%M")
    cfg = _cfg(raw_dir, "reporttest", start, end)
    run_primary(cfg, _args(str(tmp_path / "reporttest.toml")))

    run_dir = find_home() / "results" / "reporttest"
    reports_dir = run_dir / "reports"
    write_primary_reports(run_dir / "primary", reports_dir, cfg)

    for name in ("primary_files", "primary_slots", "primary_flags", "primary_coverage", "primary_second_layer"):
        path = reports_dir / f"{name}.csv"
        assert path.is_file(), name
        df = pd.read_csv(path)
        assert not df.empty, name

    files_df = pd.read_csv(reports_dir / "primary_files.csv")
    assert set(files_df["month"]) >= {"overall"}
    assert files_df[files_df["month"] == "overall"]["count"].sum() == 8  # all 8 half-hours found

    coverage_df = pd.read_csv(reports_dir / "primary_coverage.csv")
    assert {"mean", "q10", "q50", "q90", "frac_ge_c"} <= set(coverage_df.columns)
    assert (coverage_df["mean"].between(0, 1)).all()


def _slots(n):
    return pd.DataFrame({"slot": np.arange(n, dtype=np.int64),
                         "slot_start": pd.date_range("2014-04-01", periods=n, freq="10min", tz="Etc/GMT+6")})


def test_flags_report_counts_only_present_samples():
    from ttu_tower.primary.report import flags_report

    coverage = pd.DataFrame([(0, 1, "ts", "present", 0.5), (1, 1, "ts", "present", 1.0)],
                            columns=["slot", "boom", "variable", "layer", "fraction"])
    slot_qc = pd.DataFrame([(0, 1, "ts", "flagged_unchecked", 0.1), (1, 1, "ts", "flagged_unchecked", 0.05),
                            (0, 1, "ts", "skew", 0.3)],
                           columns=["slot", "boom", "variable", "stat", "value"])
    report = flags_report(slot_qc, coverage, _slots(2))
    overall = report[report["month"] == "overall"]
    assert overall["test"].tolist() == ["unchecked"]
    assert overall["fraction"].iloc[0] == pytest.approx(0.15 / 1.5)


def test_second_layer_report_counts_merged_slots():
    from ttu_tower.primary.report import second_layer_report

    flags = pd.DataFrame({"start": [0], "end": [60_000], "test": ["direction"], "kind": ["quality"],
                          "boom": [1], "variable": [None]})
    report = second_layer_report(flags, _slots(4), [1])
    direction = report[report["test"] == "direction"]["fraction"].iloc[0]
    assert direction == pytest.approx(2 / 4)  # one merged row, two slots


def test_predates_flagged_counts():
    from ttu_tower.primary.report import predates_flagged_counts

    old_qc = pd.DataFrame([(0, 1, "ts", "skew", 0.3)], columns=["slot", "boom", "variable", "stat", "value"])
    new_qc = pd.concat([old_qc, pd.DataFrame([(0, 1, "ts", "flagged_spike", 0.01)], columns=old_qc.columns)])
    flags = pd.DataFrame({"start": [0], "end": [10], "test": ["spike"], "kind": ["quality"], "boom": [1],
                          "variable": ["ts"]})
    assert predates_flagged_counts(old_qc, flags)
    assert not predates_flagged_counts(new_qc, flags)
    assert not predates_flagged_counts(old_qc, flags.iloc[0:0])
