from types import SimpleNamespace

import numpy as np
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io.runs import find_home
from ttu_tower.io.store import read_table
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.tertiary import filtering
from ttu_tower.tertiary.runner import run_tertiary
from ttu_tower.timegrid import half_hour_to_time
from ttu_tower.validation.synthetic import write_raw_dataset

_HALF_HOURS = list(range(6100, 6108))  # 8 half-hours
_BOOMS = [1, 2]
_MISSING = [6103]  # one outage, to exercise the mrd_unexcised copy path
_TABLES = ("boom_final", "tau_final", "pairs", "profile", "slot_final", "filter_log", "boom_labels_final")


def _args(config, **overrides):
    base = dict(config=config, test=False, force=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _cfg(raw_dir, tag):
    start = half_hour_to_time(_HALF_HOURS[0] + 1).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(_HALF_HOURS[-1]).strftime("%Y-%m-%d %H:%M")
    raw = {
        "tag": tag, "paths": {"raw_dirs": [str(raw_dir)]},
        "period": {"start": start, "end": end},
        "files": {"bad_records": [], "booms": _BOOMS},
        "primary": {"batch_max_files": 3},
        "tertiary": {"veer_reference_boom": 1, "fits": {k: _BOOMS for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    }
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tertiary_acceptance") / "raw"
    rng = np.random.default_rng(11)
    write_raw_dataset(d, _HALF_HOURS, _BOOMS, rng, missing=_MISSING)
    return d


def _run_upstream(cfg, config_path):
    run_primary(cfg, _args(str(config_path), nproc=1, redo_failures=False, allow_non_parquet=False))
    run_secondary(cfg, _args(str(config_path)))


def _read_all(run_dir):
    return {t: read_table(run_dir / "tertiary" / "data" / t) for t in _TABLES}


def test_run_tertiary_end_to_end(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "ter_e2e")
    config_path = tmp_path / "ter_e2e.toml"
    config_path.write_text("tag = \"ter_e2e\"\n")

    _run_upstream(cfg, config_path)
    summary = run_tertiary(cfg, _args(str(config_path)))

    assert summary["status_counts"]["processed"] > 0
    run_dir = find_home() / "results" / "ter_e2e"
    tables = _read_all(run_dir)

    boom_final = tables["boom_final"]
    assert not boom_final.empty
    assert set(boom_final["variant"].unique()) == {"none", "mrd", "mrd_unexcised", "naive"}
    assert boom_final["slot_start"].notna().all()
    # boom_final is exactly means+slow+boom_stats concatenated (filtering
    # only NaNs values, never drops rows or renames variables), so every
    # variable those three real tables actually produced must have a group -
    # a stronger, non-hand-maintained version of test_filtering's
    # completeness check.
    unmapped = set(boom_final["variable"].unique()) - set(filtering.QUANTITY_GROUPS)
    assert not unmapped, f"no QUANTITY_GROUPS entry for: {unmapped}"

    tau_final = tables["tau_final"]
    assert set(tau_final["variant"].unique()) == {"mrd", "mrd_unexcised", "naive"}

    pairs = tables["pairs"]
    assert set(pairs["variable"].unique()) == {"rib", "lapse_vpt", "veer"}
    assert {(1, 2), (1, 1)} <= set(zip(pairs["boom"], pairs["boom2"]))

    profile = tables["profile"]
    expected_vars = {"alpha", "alpha_n", "gamma", "gamma_n", "gamma_n_capped",
                      "wdgamma", "wdgamma_n", "wdgamma_n_capped", "loglaw_ustar", "loglaw_z0", "loglaw_n",
                      "loglaw_z0_constrained_n", "loglaw_z0_constrained_n_capped"}
    assert expected_vars <= set(profile["variable"].unique())
    assert set(profile[profile["variable"] == "gamma"]["variant"].unique()) == {"mrd", "mrd_unexcised", "naive"}

    slot_final = tables["slot_final"]
    assert set(slot_final["slot"]) == set(boom_final["slot"].unique()) | set(slot_final["slot"])
    sun_rows = slot_final[slot_final["variable"] == "sun_elevation"]
    assert not sun_rows.empty
    assert sun_rows["value"].notna().all()
    night_rows = slot_final[slot_final["variable"] == "night"]
    assert set(night_rows["value"].unique()) <= {0.0, 1.0}

    filter_log = tables["filter_log"]
    assert list(filter_log.columns) == ["slot", "boom", "variant", "group", "criterion", "variable"]

    boom_labels_final = tables["boom_labels_final"]
    assert not boom_labels_final.empty
    assert set(boom_labels_final["label"].unique()) == {"aniso_class"}
    assert boom_labels_final["slot_start"].notna().all()
    # aniso_class is null everywhere aniso_l1 (same momentum group) is NaN in
    # boom_final, and only there - the two tables can't disagree.
    aniso_l1 = boom_final[boom_final["variable"] == "aniso_l1"].set_index(["slot", "boom", "variant"])["value"]
    joined = boom_labels_final.set_index(["slot", "boom", "variant"])
    joined = joined.join(aniso_l1.rename("aniso_l1"), how="inner")
    assert (joined["value"].isna() == joined["aniso_l1"].isna()).all()

    wide_files = sorted((run_dir / "tertiary" / "wide").glob("*.parquet"))
    assert wide_files
    import pandas as pd
    wide_df = pd.concat((pd.read_parquet(p) for p in wide_files), ignore_index=True)
    assert "ws_mean_b1" in wide_df.columns
    assert "alpha" in wide_df.columns
    assert "rib_b1-b2" in wide_df.columns


def test_run_tertiary_resumes_by_skipping_existing_batches(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "ter_resume")
    config_path = tmp_path / "ter_resume.toml"
    config_path.write_text("tag = \"ter_resume\"\n")

    _run_upstream(cfg, config_path)
    first = run_tertiary(cfg, _args(str(config_path)))
    second = run_tertiary(cfg, _args(str(config_path)))

    assert first["status_counts"]["processed"] > 0
    assert second["status_counts"]["processed"] == 0
    assert second["status_counts"]["skipped"] == first["status_counts"]["processed"]
