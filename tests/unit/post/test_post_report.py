from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io.runs import find_home
from ttu_tower.post.report import write_stability_reports
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.tertiary.runner import run_tertiary
from ttu_tower.timegrid import half_hour_to_time
from ttu_tower.validation.synthetic import write_raw_dataset

_HALF_HOURS = list(range(6200, 6208))  # 8 half-hours
_BOOMS = [2, 4]  # matches the default post.stability.pair


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
        "tertiary": {"veer_reference_boom": 2, "fits": {k: _BOOMS for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    }
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("post_report") / "raw"
    rng = np.random.default_rng(13)
    write_raw_dataset(d, _HALF_HOURS, _BOOMS, rng)
    return d


def test_write_stability_reports_produces_all_csvs(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "post_e2e")
    config_path = tmp_path / "post_e2e.toml"
    config_path.write_text("tag = \"post_e2e\"\n")

    run_primary(cfg, _args(str(config_path), nproc=1, redo_failures=False, allow_non_parquet=False))
    run_secondary(cfg, _args(str(config_path)))
    run_tertiary(cfg, _args(str(config_path)))

    run_dir = find_home() / "results" / "post_e2e"
    reports_dir = run_dir / "reports"
    write_stability_reports(run_dir, reports_dir, cfg)

    for name in ("stability_yield", "stability_flags", "stability_tau"):
        path = reports_dir / f"{name}.csv"
        assert path.is_file(), name
        df = pd.read_csv(path)
        assert not df.empty, name

    yield_df = pd.read_csv(reports_dir / "stability_yield.csv")
    assert set(yield_df["quantity"].unique()) == {"ustar", "wvpts_cov", "ti"}
    assert set(yield_df["variant"].unique()) == {"mrd", "naive", "mrd_unexcised"}
    assert (yield_df["fraction"].between(0, 1)).all()
    assert (yield_df["n_survived"] <= yield_df["n_computed"]).all()

    flags_df = pd.read_csv(reports_dir / "stability_flags.csv")
    assert set(flags_df["boom"].unique()) <= set(_BOOMS)
    assert (flags_df["fraction"].dropna().between(0, 1)).all()

    tau_df = pd.read_csv(reports_dir / "stability_tau.csv")
    assert set(tau_df["variant"].unique()) == {"mrd", "naive", "mrd_unexcised"}
    fractions_by_group = tau_df.groupby(["stability_class", "boom", "variant"])["fraction"].sum()
    assert (fractions_by_group.round(6) == 1.0).all()
