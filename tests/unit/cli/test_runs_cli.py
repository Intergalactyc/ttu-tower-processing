import json

from ttu_tower.cli.runs import main
from ttu_tower.io.runs import find_home


def test_lists_registered_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    home = find_home()
    run_dir = tmp_path / "results" / "alpha"
    (run_dir / "primary").mkdir(parents=True)
    (run_dir / "primary" / "run_meta.json").write_text("{}")
    (home / "runs").mkdir()
    (home / "runs" / "alpha.json").write_text(json.dumps({
        "tag": "alpha", "run_dir": str(run_dir), "config": "", "registered": "2026-01-01T00:00:00-06:00",
        "package_version": "2.0.0",
    }))

    main([])
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "primary" in out


def test_no_registered_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    main(["--test"])
    assert "no registered runs" in capsys.readouterr().out
