from types import SimpleNamespace

import numpy as np
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io.runs import find_home
from ttu_tower.io.store import read_table
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.timegrid import half_hour_to_time
from ttu_tower.validation.synthetic import write_raw_dataset

_HALF_HOURS = list(range(6000, 6008))  # 8 half-hours
_BOOMS = [1, 2]
_MISSING = [6003]  # one outage, to exercise the mrd_unexcised copy path
_TABLES = ("tau", "tau_selected", "boom_stats", "boom_labels", "slow", "slot_stats")


def _args(config, **overrides):
    base = dict(config=config, test=False, force=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _cfg(raw_dir, tag, **sections):
    start = half_hour_to_time(_HALF_HOURS[0] + 1).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(_HALF_HOURS[-1]).strftime("%Y-%m-%d %H:%M")
    raw = {
        "tag": tag, "paths": {"raw_dirs": [str(raw_dir)]},
        "period": {"start": start, "end": end},
        "files": {"bad_records": [], "booms": _BOOMS},
        "primary": {"batch_max_files": 3},
        "tertiary": {"veer_reference_boom": 1, "fits": {k: [1] for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    }
    raw.update(sections)
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("secondary_acceptance") / "raw"
    rng = np.random.default_rng(7)
    write_raw_dataset(d, _HALF_HOURS, _BOOMS, rng, missing=_MISSING)
    return d


def _read_all(run_dir):
    return {t: read_table(run_dir / "secondary" / "data" / t) for t in _TABLES}


def test_run_secondary_end_to_end(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "sec_e2e")
    config_path = tmp_path / "sec_e2e.toml"
    config_path.write_text("tag = \"sec_e2e\"\n")  # content unused; run_secondary reads cfg directly

    run_primary(cfg, _args(str(config_path), nproc=1, redo_failures=False, allow_non_parquet=False))
    summary = run_secondary(cfg, _args(str(config_path)))

    assert summary["status_counts"]["processed"] > 0
    run_dir = find_home() / "results" / "sec_e2e"
    tables = _read_all(run_dir)

    for name, df in tables.items():
        assert not df.empty, f"{name} is empty"

    tau_selected = tables["tau_selected"]
    assert set(tau_selected["variant"].unique()) == {"mrd", "mrd_unexcised", "naive"}
    naive = tau_selected[tau_selected["variant"] == "naive"]
    assert (naive["tau_s"] == 600.0).all()
    assert (naive["source"] == "fixed").all()

    # every (slot, boom, variant) with a tau_selected row should also have
    # boom_stats rows (at least the coverage rows, always present).
    boom_stats = tables["boom_stats"]
    sel_keys = set(zip(tau_selected["slot"], tau_selected["boom"], tau_selected["variant"]))
    stats_keys = set(zip(boom_stats["slot"], boom_stats["boom"], boom_stats["variant"]))
    assert sel_keys <= stats_keys

    # boom_labels carries the anisotropy class for every boom_stats (slot,boom,variant).
    boom_labels = tables["boom_labels"]
    label_keys = set(zip(boom_labels["slot"], boom_labels["boom"], boom_labels["variant"]))
    assert sel_keys <= label_keys

    slow = tables["slow"]
    assert set(slow["variable"].unique()) >= {"t", "rh", "p", "p_factor", "p_measured", "vpts"}

    slot_stats = tables["slot_stats"]
    assert slot_stats["sun_elevation"].notna().all()


def test_run_secondary_resumes_by_skipping_existing_batches(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "sec_resume")
    config_path = tmp_path / "sec_resume.toml"
    config_path.write_text("tag = \"sec_resume\"\n")

    run_primary(cfg, _args(str(config_path), nproc=1, redo_failures=False, allow_non_parquet=False))
    first = run_secondary(cfg, _args(str(config_path)))
    second = run_secondary(cfg, _args(str(config_path)))

    assert first["status_counts"]["processed"] > 0
    assert second["status_counts"]["processed"] == 0
    assert second["status_counts"]["skipped"] == first["status_counts"]["processed"]
