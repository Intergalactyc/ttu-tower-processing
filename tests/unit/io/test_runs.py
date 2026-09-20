import json
from pathlib import Path

import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io import runs


def test_autouse_fixture_isolates_real_user_home():
    # tests/conftest.py's autouse fixture sets TTU_TOWER_HOME for the whole
    # session, so find_home() here must never resolve under the real
    # ~/.ttu-tower.
    home = runs.find_home()
    assert home.parent != Path.home()


def _cfg(tag: str, results_dir: str = ""):
    raw = {
        "tag": tag,
        "paths": {"raw_dirs": ["/data"], "results_dir": results_dir},
        "files": {"bad_records": [[1, 2]]},
    }
    return config_from_dict(raw)


# --- home discovery ----------------------------------------------------------

def test_find_home_creates_ttu_tower_when_nothing_there(tmp_path, monkeypatch):
    monkeypatch.delenv("TTU_TOWER_HOME", raising=False)
    home = runs.find_home(user_home=tmp_path)
    assert home == tmp_path / ".ttu-tower"
    data = json.loads((home / "ttu-tower-home.json").read_text())
    assert data["signature"] == "ttu-tower-processing home"


def test_find_home_creates_ttu_tower0_when_ttu_tower_is_foreign_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TTU_TOWER_HOME", raising=False)
    (tmp_path / ".ttu-tower").mkdir()
    (tmp_path / ".ttu-tower" / "somefile.txt").write_text("not a home")
    assert runs.find_home(user_home=tmp_path) == tmp_path / ".ttu-tower0"


def test_find_home_considers_every_candidate_not_just_the_first_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TTU_TOWER_HOME", raising=False)
    (tmp_path / ".ttu-tower").mkdir()  # foreign
    real = tmp_path / ".ttu-tower1"  # .ttu-tower0 absent, .ttu-tower1 ours
    real.mkdir()
    (real / "ttu-tower-home.json").write_text(json.dumps({"signature": "ttu-tower-processing home"}))

    home = runs.find_home(user_home=tmp_path)
    assert home == real
    assert not (tmp_path / ".ttu-tower0").exists()


def test_find_home_treats_file_named_ttu_tower_as_foreign(tmp_path, monkeypatch):
    monkeypatch.delenv("TTU_TOWER_HOME", raising=False)
    (tmp_path / ".ttu-tower").write_text("i am a file")
    assert runs.find_home(user_home=tmp_path) == tmp_path / ".ttu-tower0"


def test_find_home_treats_wrong_signature_content_as_foreign(tmp_path, monkeypatch):
    monkeypatch.delenv("TTU_TOWER_HOME", raising=False)
    d = tmp_path / ".ttu-tower"
    d.mkdir()
    (d / "ttu-tower-home.json").write_text(json.dumps({"signature": "something else"}))
    assert runs.find_home(user_home=tmp_path) == tmp_path / ".ttu-tower0"


def test_find_home_second_call_finds_existing(tmp_path, monkeypatch):
    monkeypatch.delenv("TTU_TOWER_HOME", raising=False)
    home1 = runs.find_home(user_home=tmp_path)
    home2 = runs.find_home(user_home=tmp_path)
    assert home1 == home2
    candidates = [p for p in tmp_path.iterdir() if p.name.startswith(".ttu-tower")]
    assert len(candidates) == 1


# --- TTU_TOWER_HOME ------------------------------------------------------------

def test_ttu_tower_home_env_adopts_existing_signature(tmp_path, monkeypatch):
    d = tmp_path / "myhome"
    d.mkdir()
    (d / "ttu-tower-home.json").write_text(json.dumps({"signature": "ttu-tower-processing home"}))
    monkeypatch.setenv("TTU_TOWER_HOME", str(d))
    assert runs.find_home() == d


def test_ttu_tower_home_env_initializes_missing(tmp_path, monkeypatch):
    d = tmp_path / "newhome"
    monkeypatch.setenv("TTU_TOWER_HOME", str(d))
    assert runs.find_home() == d
    assert (d / "ttu-tower-home.json").is_file()


def test_ttu_tower_home_env_initializes_empty_dir(tmp_path, monkeypatch):
    d = tmp_path / "emptyhome"
    d.mkdir()
    monkeypatch.setenv("TTU_TOWER_HOME", str(d))
    assert runs.find_home() == d
    assert (d / "ttu-tower-home.json").is_file()


def test_ttu_tower_home_env_errors_on_nonempty_foreign_dir(tmp_path, monkeypatch):
    d = tmp_path / "foreign"
    d.mkdir()
    (d / "other.txt").write_text("x")
    monkeypatch.setenv("TTU_TOWER_HOME", str(d))
    with pytest.raises(RuntimeError):
        runs.find_home()


# --- run_dir_for --------------------------------------------------------------

def test_run_dir_for_default_uses_home_results(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg = _cfg("myrun")
    assert runs.run_dir_for(cfg, test=False) == home / "results" / "myrun"
    assert runs.run_dir_for(cfg, test=True) == home / "results" / "testing" / "myrun"


def test_run_dir_for_external_results_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg = _cfg("myrun", results_dir=str(tmp_path / "external"))
    assert runs.run_dir_for(cfg, test=False) == tmp_path / "external" / "myrun"


# --- registration --------------------------------------------------------------

def test_register_run_writes_new_link(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg = _cfg("myrun", results_dir=str(tmp_path / "results"))
    run_dir = runs.register_run(cfg, tmp_path / "myrun.toml", package_version="2.0.0")
    assert run_dir == tmp_path / "results" / "myrun"
    data = json.loads((home / "runs" / "myrun.json").read_text())
    assert data["run_dir"] == str(run_dir)
    assert data["tag"] == "myrun"
    assert data["package_version"] == "2.0.0"


def test_register_run_same_dir_is_noop(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg = _cfg("myrun", results_dir=str(tmp_path / "results"))
    run_dir1 = runs.register_run(cfg, tmp_path / "myrun.toml", package_version="2.0.0")
    run_dir2 = runs.register_run(cfg, tmp_path / "myrun.toml", package_version="2.0.0")
    assert run_dir1 == run_dir2


def test_register_run_repoints_when_old_dir_gone(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg1 = _cfg("myrun", results_dir=str(tmp_path / "results1"))
    runs.register_run(cfg1, tmp_path / "myrun.toml", package_version="2.0.0")

    cfg2 = _cfg("myrun", results_dir=str(tmp_path / "results2"))
    run_dir2 = runs.register_run(cfg2, tmp_path / "myrun.toml", package_version="2.0.0")
    assert run_dir2 == tmp_path / "results2" / "myrun"


def test_register_run_collision_raises_without_force(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg1 = _cfg("myrun", results_dir=str(tmp_path / "results1"))
    run_dir1 = runs.register_run(cfg1, tmp_path / "myrun.toml", package_version="2.0.0")
    run_dir1.mkdir(parents=True)

    cfg2 = _cfg("myrun", results_dir=str(tmp_path / "results2"))
    with pytest.raises(RuntimeError, match="myrun"):
        runs.register_run(cfg2, tmp_path / "myrun.toml", package_version="2.0.0")


def test_register_run_force_reregisters_without_deleting_old(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg1 = _cfg("myrun", results_dir=str(tmp_path / "results1"))
    run_dir1 = runs.register_run(cfg1, tmp_path / "myrun.toml", package_version="2.0.0")
    run_dir1.mkdir(parents=True)

    cfg2 = _cfg("myrun", results_dir=str(tmp_path / "results2"))
    run_dir2 = runs.register_run(cfg2, tmp_path / "myrun.toml", package_version="2.0.0", force=True)
    assert run_dir2 == tmp_path / "results2" / "myrun"
    assert run_dir1.exists()


def test_register_run_test_flag_uses_separate_namespace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTU_TOWER_HOME", str(home))
    cfg = _cfg("myrun", results_dir=str(tmp_path / "results"))
    normal_dir = runs.register_run(cfg, tmp_path / "myrun.toml", package_version="2.0.0")
    test_dir = runs.register_run(cfg, tmp_path / "myrun.toml", test=True, package_version="2.0.0")
    assert normal_dir != test_dir
    assert (home / "runs" / "myrun.json").is_file()
    assert (home / "runs" / "testing" / "myrun.json").is_file()
