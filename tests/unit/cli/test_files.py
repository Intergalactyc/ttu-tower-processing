import numpy as np

from ttu_tower.cli.files import main
from ttu_tower.validation.synthetic import write_raw_dataset


def _write_config(path, raw_dir, start="", end=""):
    raw_dir_toml = str(raw_dir).replace("\\", "/")
    path.write_text(
        'tag = "clifilestest"\n'
        f'[paths]\nraw_dirs = ["{raw_dir_toml}"]\n'
        "[files]\nbad_records = []\n"
        f'[period]\nstart = "{start}"\nend = "{end}"\n'
    )


def test_empty_period_defaults_from_accepted_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(1)
    write_raw_dataset(raw_dir, range(4000, 4004), [1], rng)

    config_path = tmp_path / "config.toml"
    _write_config(config_path, raw_dir)  # no [period] at all -> both default to ""

    main([str(config_path)])
    out = capsys.readouterr().out
    assert "period coverage" in out
    assert "4/4" in out  # all 4 half-hours accepted, defaulting to their own range


def test_empty_period_with_no_accepted_files_skips_coverage(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    config_path = tmp_path / "config.toml"
    _write_config(config_path, raw_dir)

    main([str(config_path)])  # must not raise
    out = capsys.readouterr().out
    assert "period coverage" not in out


def test_explicit_period_still_works(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(2)
    write_raw_dataset(raw_dir, range(4100, 4106), [1], rng)

    from ttu_tower.timegrid import half_hour_to_time
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path, raw_dir,
        start=half_hour_to_time(4101).strftime("%Y-%m-%d %H:%M"),
        end=half_hour_to_time(4105).strftime("%Y-%m-%d %H:%M"),
    )

    main([str(config_path)])
    out = capsys.readouterr().out
    assert "period coverage: 4/4" in out
