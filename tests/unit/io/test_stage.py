import pytest

from ttu_tower.io.stage import (
    StageHashMismatch, check_hash_guard, read_run_meta, write_run_meta, write_run_summary,
)


def test_write_and_read_run_meta(tmp_path):
    write_run_meta(tmp_path, stage="primary", tag="t", config_hash="abc123", upstream_hash=None,
                    package_version="2.0.0", timezone="Etc/GMT+6", unusable_tests=["skew", "kurt"])
    meta = read_run_meta(tmp_path)
    assert meta["stage"] == "primary"
    assert meta["config_hash"] == "abc123"
    assert meta["unusable_tests"] == ["skew", "kurt"]
    assert meta["timezone"] == "Etc/GMT+6"


def test_read_run_meta_missing_returns_none(tmp_path):
    assert read_run_meta(tmp_path / "nope") is None


def test_check_hash_guard_passes_with_no_existing_meta(tmp_path):
    check_hash_guard(tmp_path, "abc", force=False)  # no raise


def test_check_hash_guard_passes_with_matching_hash(tmp_path):
    write_run_meta(tmp_path, stage="primary", tag="t", config_hash="abc", upstream_hash=None,
                    package_version="2.0.0", timezone="Etc/GMT+6")
    check_hash_guard(tmp_path, "abc", force=False)  # no raise


def test_check_hash_guard_raises_on_mismatch_without_force(tmp_path):
    write_run_meta(tmp_path, stage="primary", tag="t", config_hash="abc", upstream_hash=None,
                    package_version="2.0.0", timezone="Etc/GMT+6")
    with pytest.raises(StageHashMismatch):
        check_hash_guard(tmp_path, "xyz", force=False)


def test_check_hash_guard_clears_stage_dir_with_force(tmp_path):
    write_run_meta(tmp_path, stage="primary", tag="t", config_hash="abc", upstream_hash=None,
                    package_version="2.0.0", timezone="Etc/GMT+6")
    (tmp_path / "leftover.txt").write_text("x")
    check_hash_guard(tmp_path, "xyz", force=True)
    assert not (tmp_path / "leftover.txt").exists()
    assert not (tmp_path / "run_meta.json").exists()


def test_check_hash_guard_fresh_clears_stage_dir_even_with_matching_hash(tmp_path):
    """`fresh` is for a code-only fix: the config hash never changes, so the
    ordinary guard (even with `force`) would see a match and skip every
    batch by marker instead of actually rerunning them.
    """
    write_run_meta(tmp_path, stage="primary", tag="t", config_hash="abc", upstream_hash=None,
                    package_version="2.0.0", timezone="Etc/GMT+6")
    (tmp_path / "leftover.txt").write_text("x")
    check_hash_guard(tmp_path, "abc", force=False, fresh=True)
    assert not (tmp_path / "leftover.txt").exists()
    assert not (tmp_path / "run_meta.json").exists()


def test_check_hash_guard_fresh_is_a_noop_on_an_empty_stage_dir(tmp_path):
    check_hash_guard(tmp_path, "abc", force=False, fresh=True)  # no raise, nothing to clear


def test_write_run_summary(tmp_path):
    write_run_summary(tmp_path, stage="primary", config_hash="abc", package_version="2.0.0",
                       started="t0", finished="t1", status_counts={"success": 3}, totals={"files_loaded": 10},
                       timings={"stage_a": 1.5})
    import json
    data = json.loads((tmp_path / "run_summary.json").read_text())
    assert data["status_counts"] == {"success": 3}
    assert data["totals"]["files_loaded"] == 10


def test_write_run_meta_records_git_dirty(tmp_path):
    write_run_meta(tmp_path, stage="primary", tag="t", config_hash="abc", upstream_hash=None,
                    package_version="2.0.0", timezone="Etc/GMT+6")
    meta = read_run_meta(tmp_path)
    assert "git_dirty" in meta
    assert meta["git_dirty"] in (True, False, None)


def test_git_helpers_outside_a_checkout_return_none(tmp_path):
    from ttu_tower.io.stage import git_commit, git_dirty

    assert git_commit(cwd=tmp_path) is None
    assert git_dirty(cwd=tmp_path) is None


def test_write_config_copies_copies_toml_and_resolves(tmp_path):
    import json

    from ttu_tower.config import config_from_dict
    from ttu_tower.io.stage import write_config_copies

    cfg = config_from_dict({"tag": "t", "paths": {"raw_dirs": ["/data"]}, "files": {"bad_records": []}})
    toml_path = tmp_path / "t.toml"
    toml_path.write_text("tag = \"t\"\n# a comment kept verbatim\n")
    stage_dir = tmp_path / "run" / "primary"

    write_config_copies(stage_dir, cfg, toml_path)

    assert (stage_dir / "config.toml").read_text() == toml_path.read_text()
    resolved = json.loads((stage_dir / "config.resolved.json").read_text())
    assert config_from_dict(resolved) == cfg


def test_write_config_copies_without_a_config_file_writes_only_resolved(tmp_path):
    from ttu_tower.config import config_from_dict
    from ttu_tower.io.stage import write_config_copies

    cfg = config_from_dict({"tag": "t", "paths": {"raw_dirs": ["/data"]}, "files": {"bad_records": []}})
    write_config_copies(tmp_path, cfg, tmp_path / "missing.toml")
    assert not (tmp_path / "config.toml").exists()
    assert (tmp_path / "config.resolved.json").is_file()
