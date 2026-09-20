from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io.store import read_manifest, read_table
from ttu_tower.primary.runner import run_primary
from ttu_tower.validation.synthetic import write_raw_dataset


def _args(config, **overrides):
    base = dict(config=config, nproc=1, test=False, redo_failures=False, force=False, allow_non_parquet=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _cfg(raw_dir, tag, start, end, **sections):
    raw = {
        "tag": tag, "paths": {"raw_dirs": [str(raw_dir)]},
        "period": {"start": start, "end": end},
        "files": {"bad_records": []},
        "tertiary": {
            "veer_reference_boom": 1,
            "fits": {"alpha_booms": [1], "gamma_booms": [1], "wdgamma_booms": [1], "loglaw_booms": [1]},
        },
    }
    raw.update(sections)
    return config_from_dict(raw)


def _write_dataset(tmp_path, half_hours, booms, seed=0, missing=()):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(seed)
    write_raw_dataset(raw_dir, half_hours, booms, rng, missing=missing)
    return raw_dir


def test_run_primary_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    half_hours = list(range(3000, 3008))
    raw_dir = _write_dataset(tmp_path, half_hours, [1, 2])

    from ttu_tower.timegrid import half_hour_to_time
    start = half_hour_to_time(3001).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(3007).strftime("%Y-%m-%d %H:%M")
    cfg = _cfg(raw_dir, "runA", start, end, files={"bad_records": [], "booms": [1, 2]})

    summary = run_primary(cfg, _args(str(tmp_path / "runA.toml")))
    assert summary["status_counts"]["success"] > 0
    assert summary["status_counts"]["failed"] == 0

    from ttu_tower.io.runs import find_home
    run_dir = find_home() / "results" / "runA"
    files_df = pd.read_parquet(run_dir / "primary" / "files.parquet")
    assert (files_df["status"] == "accepted").all()

    slot_boom = read_table(run_dir / "primary" / "data" / "slot_boom")
    n_half_hours = 3007 - 3001  # [3001, 3007) exclusive of the period end half-hour
    assert len(slot_boom) == n_half_hours * 3 * 2  # slots x booms

    manifest = read_manifest(run_dir / "primary" / "manifest")
    assert len(manifest) == len(set(slot_boom_unit for slot_boom_unit in manifest))
    assert all(e["status"] == "success" for e in manifest.values())


def test_run_primary_hash_guard_blocks_config_change(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    half_hours = list(range(3100, 3106))
    raw_dir = _write_dataset(tmp_path, half_hours, [1])

    from ttu_tower.timegrid import half_hour_to_time
    start = half_hour_to_time(3101).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(3105).strftime("%Y-%m-%d %H:%M")
    cfg1 = _cfg(raw_dir, "runB", start, end, files={"bad_records": [], "booms": [1]})
    run_primary(cfg1, _args(str(tmp_path / "runB.toml")))

    cfg2 = _cfg(raw_dir, "runB", start, end, files={"bad_records": [[1, 1]], "booms": [1]})
    with pytest.raises(Exception):
        run_primary(cfg2, _args(str(tmp_path / "runB.toml")))

    # --force clears and reruns successfully
    summary = run_primary(cfg2, _args(str(tmp_path / "runB.toml"), force=True))
    assert summary["status_counts"]["failed"] == 0


def test_run_primary_test_flag_only_first_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    half_hours = list(range(3200, 3210))
    raw_dir = _write_dataset(tmp_path, half_hours, [1])

    from ttu_tower.timegrid import half_hour_to_time
    start = half_hour_to_time(3201).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(3209).strftime("%Y-%m-%d %H:%M")
    cfg = _cfg(raw_dir, "runC", start, end, files={"bad_records": [], "booms": [1]}, primary={"batch_max_files": 2})

    run_primary(cfg, _args(str(tmp_path / "runC.toml"), test=True))

    from ttu_tower.io.runs import find_home
    run_dir = find_home() / "results" / "testing" / "runC"
    slot_boom = read_table(run_dir / "primary" / "data" / "slot_boom")
    assert slot_boom["slot"].nunique() <= 2 * 3  # only the first (2-file) batch
