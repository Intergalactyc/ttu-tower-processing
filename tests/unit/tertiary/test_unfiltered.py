"""load_results(..., filtered=False): tertiary's tables recomputed without its filtering, on a
synthetic run whose boom 1 fails filtering (an out-of-bounds burst every file, no bounds flags
allowed)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

from ttu_tower import load_results
from ttu_tower.config import config_from_dict
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.tertiary import unfiltered
from ttu_tower.tertiary.runner import run_tertiary
from ttu_tower.timegrid import half_hour_to_time
from ttu_tower.validation.synthetic import write_raw_dataset

_HALF_HOURS = list(range(6200, 6206))
_BOOMS = [1, 2, 3]
_TAG = "ter_unfiltered"
_KEYS = {"boom_final": ["slot", "boom", "variant", "variable", "stat"],
         "pairs": ["slot", "boom", "boom2", "variant", "variable"],
         "profile": ["slot", "variant", "variable"]}


def _burst_on_boom_1(h, boom, df):
    if boom != 1:
        return df
    df = df.copy()
    df.loc[df.index[1000:1003], f"u_{boom}"] = 500.0
    return df


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The run's tag, after primary, secondary and tertiary; TTU_TOWER_HOME is set for the module."""
    root = tmp_path_factory.mktemp("tertiary_unfiltered")
    write_raw_dataset(root / "raw", _HALF_HOURS, _BOOMS, np.random.default_rng(17), mutate=_burst_on_boom_1)
    cfg = config_from_dict({
        "tag": _TAG, "paths": {"raw_dirs": [str(root / "raw")]},
        "period": {"start": half_hour_to_time(_HALF_HOURS[0] + 1).strftime("%Y-%m-%d %H:%M"),
                   "end": half_hour_to_time(_HALF_HOURS[-1]).strftime("%Y-%m-%d %H:%M")},
        "files": {"bad_records": [], "booms": _BOOMS},
        "primary": {"batch_max_files": 3},
        "tertiary": {"veer_reference_boom": 2, "max_bounds_fraction": 0.0,
                     "fits": {k: _BOOMS for k in ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    })
    config_path = root / f"{_TAG}.toml"
    config_path.write_text(f"tag = \"{_TAG}\"\n")
    mp = pytest.MonkeyPatch()
    mp.setenv("TTU_TOWER_HOME", str(root / "home"))
    args = SimpleNamespace(config=str(config_path), test=False, force=False)
    run_primary(cfg, SimpleNamespace(**vars(args), nproc=1, redo_failures=False, allow_non_parquet=False))
    run_secondary(cfg, args)
    run_tertiary(cfg, args)
    yield _TAG
    mp.undo()


def _indexed(df, keys):
    return df.astype({k: str for k in keys if k not in ("slot", "boom", "boom2")}).set_index(keys)["value"].sort_index()


def test_filtering_failed_boom_1(run):
    filter_log = load_results(run, "filter_log")
    failed = filter_log[filter_log["criterion"] == "bounds"]
    assert not failed.empty
    assert set(failed["boom"]) == {1}


@pytest.mark.parametrize("table", list(_KEYS))
def test_unfiltered_keeps_rows_and_stored_values(run, table):
    stored = _indexed(load_results(run, table), _KEYS[table])
    unfilt = _indexed(load_results(run, table, filtered=False), _KEYS[table])
    assert unfilt.index.equals(stored.index)
    has = stored.notna()
    if table == "profile":  # a fit over more booms is a different fit
        n = unfilt.index.get_level_values("variable").str.endswith("_n")
        assert (unfilt[n].fillna(0) >= stored[n].fillna(0)).all() and (unfilt[n].fillna(0) > stored[n].fillna(0)).any()
    else:
        np.testing.assert_allclose(unfilt[has], stored[has], rtol=1e-12)
        assert (unfilt.notna() & ~has).any()


def test_unfiltered_boom_final_restores_failed_values(run):
    filter_log = load_results(run, "filter_log")
    failed = filter_log[(filter_log["variant"] == "none") & (filter_log["group"] == "momentum")]
    keys = ["slot", "boom", "variant", "variable", "stat"]
    stored = _indexed(load_results(run, "boom_final"), keys)
    unfilt = _indexed(load_results(run, "boom_final", filtered=False), keys)
    for slot, boom in zip(failed["slot"], failed["boom"]):
        key = (slot, boom, "none", "ue", "mean")
        assert np.isnan(stored[key]) and np.isfinite(unfilt[key])
    # the gradient products are recomputed from the restored values
    level = unfilt.index.get_level_values
    rig = (level("boom") == 1) & (level("variant") == "none") & (level("variable") == "rig")
    assert unfilt[rig].notna().sum() > stored[rig].notna().sum()


def test_unfiltered_wide_matches_export_layout(run):
    stored = load_results(run, "wide").set_index("slot_start")
    unfilt = load_results(run, "wide", filtered=False).set_index("slot_start")
    assert unfilt.index.equals(stored.index)
    assert set(unfilt.columns) == set(stored.columns)
    unfilt = unfilt[stored.columns]
    numeric = stored.select_dtypes("number").columns.drop(
        [c for c in stored.columns if c.startswith(("alpha", "gamma", "wdgamma", "loglaw", "h_"))])
    has = stored[numeric].notna().to_numpy()
    np.testing.assert_allclose(unfilt[numeric].to_numpy(float)[has], stored[numeric].to_numpy(float)[has], rtol=1e-12)
    assert unfilt["ue_mean_b1"].notna().sum() > stored["ue_mean_b1"].notna().sum()

    some = load_results(run, "wide", filtered=False, columns=["slot_start", "ue_mean_b1", "rig_b2"])
    assert list(some.columns) == ["slot_start", "ue_mean_b1", "rig_b2"]
    pd.testing.assert_series_equal(some["ue_mean_b1"], unfilt["ue_mean_b1"].reset_index(drop=True))


def test_unfiltered_columns_and_filters(run):
    rig = load_results(run, "boom_final", filtered=False, columns=["slot", "boom", "value"],
                       filters=(ds.field("variable") == "rig") & (ds.field("boom") == 1))
    assert list(rig.columns) == ["slot", "boom", "value"]
    assert len(rig) > 0 and set(rig["boom"]) == {1}


def test_tables_filtering_leaves_alone_are_read_as_stored(run):
    pd.testing.assert_frame_equal(load_results(run, "tau_final", filtered=False), load_results(run, "tau_final"))
    assert "filter_log" not in unfiltered.TABLES
    with pytest.raises(ValueError, match="isn't changed by tertiary filtering"):
        unfiltered.load(None, "tau_final")
