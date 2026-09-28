import json
import shutil

import pytest

from ttu_tower.viewer.run import RunHandle, RunNotFound, list_registered


def test_open_by_tag_and_by_directory_agree(primary_run):
    by_tag = RunHandle.open(primary_run.tag)
    by_dir = RunHandle.from_dir(primary_run.run_dir)
    assert by_tag.run_dir == by_dir.run_dir == primary_run.run_dir
    assert by_tag.stages == by_dir.stages == ["primary"]
    assert by_tag.period == by_dir.period
    assert any(r["tag"] == primary_run.tag for r in list_registered())


def test_unknown_tag_raises(primary_run):
    with pytest.raises(RunNotFound):
        RunHandle.open("no_such_tag_anywhere")


def test_opening_a_run_never_creates_a_home(primary_run, tmp_path, monkeypatch):
    nowhere = tmp_path / "no_home_here"
    monkeypatch.setenv("TTU_TOWER_HOME", str(nowhere))
    assert list_registered() == []
    with pytest.raises(RunNotFound):
        RunHandle.open(primary_run.tag)
    run = RunHandle.from_dir(primary_run.run_dir)
    assert run.cfg == primary_run.cfg
    assert not nowhere.exists()


def test_config_comes_from_the_resolved_copy(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    assert run.config_source == "resolved"
    assert run.cfg == primary_run.cfg
    assert run.can_reprocess


def _copy_without_resolved_config(primary_run, tmp_path):
    run_dir = tmp_path / "copy"
    shutil.copytree(primary_run.run_dir, run_dir)
    (run_dir / "primary" / "config.resolved.json").unlink()
    return run_dir


def test_registered_config_fallback_checks_the_hash(primary_run, tmp_path):
    run_dir = _copy_without_resolved_config(primary_run, tmp_path)
    run = RunHandle.from_dir(run_dir, config_path=str(primary_run.config_path))
    assert run.config_source == "registered"
    assert run.config_problems == []
    assert run.can_reprocess


def test_registered_config_that_no_longer_matches_disables_reprocessing(primary_run, tmp_path):
    run_dir = _copy_without_resolved_config(primary_run, tmp_path)
    edited = tmp_path / "edited.toml"
    edited.write_text(primary_run.config_path.read_text() + "[qc]\nmin_coverage = 0.5\n")
    run = RunHandle.from_dir(run_dir, config_path=str(edited))
    assert run.config_problems and "primary" in run.config_problems[0]
    assert not run.can_reprocess


def test_unreadable_registered_config_is_reported_not_raised(primary_run, tmp_path):
    run_dir = _copy_without_resolved_config(primary_run, tmp_path)
    dummy = tmp_path / "dummy.toml"
    dummy.write_text('tag = "x"\n')  # missing required keys
    run = RunHandle.from_dir(run_dir, config_path=str(dummy))
    assert run.cfg is None
    assert run.config_problems
    assert not run.can_reprocess


def test_meta_exposes_timezone_and_unusable_tests(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    with open(primary_run.run_dir / "primary" / "run_meta.json") as f:
        meta = json.load(f)
    assert run.timezone == meta["timezone"]
    assert run.unusable_tests == tuple(meta["unusable_tests"])
