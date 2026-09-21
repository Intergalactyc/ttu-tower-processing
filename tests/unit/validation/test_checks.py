import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io.runs import find_home
from ttu_tower.post.classify import stability_classes
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.tertiary.runner import run_tertiary
from ttu_tower.timegrid import half_hour_to_time
from ttu_tower.validation import checks
from ttu_tower.validation.synthetic import correlated_ou, inject_spikes, ou, write_raw_dataset

_HALF_HOURS = list(range(6300, 6312))  # 12 half-hours, enough for a few stability classes
_BOOMS = [2, 4]


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
        "primary": {"batch_max_files": 4},
        "tertiary": {"veer_reference_boom": 2, "fits": {k: _BOOMS for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    }
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("validation_checks") / "raw"
    rng = np.random.default_rng(21)
    write_raw_dataset(d, _HALF_HOURS, _BOOMS, rng)
    return d


@pytest.fixture(scope="module")
def run(tmp_path_factory, raw_dir):
    """A full primary -> secondary -> tertiary run, shared read-only across
    every check test in this module.
    """
    previous_home = os.environ.get("TTU_TOWER_HOME")
    os.environ["TTU_TOWER_HOME"] = str(tmp_path_factory.mktemp("validation_checks_home"))
    try:
        cfg = _cfg(raw_dir, "validation_checks")
        config_path = tmp_path_factory.mktemp("validation_checks_cfg") / "cfg.toml"
        config_path.write_text('tag = "validation_checks"\n')

        run_primary(cfg, _args(str(config_path), nproc=1, redo_failures=False, allow_non_parquet=False))
        run_secondary(cfg, _args(str(config_path)))
        run_tertiary(cfg, _args(str(config_path)))

        run_dir = find_home() / "results" / "validation_checks"
        yield run_dir, cfg
    finally:
        if previous_home is None:
            os.environ.pop("TTU_TOWER_HOME", None)
        else:
            os.environ["TTU_TOWER_HOME"] = previous_home


@pytest.fixture(scope="module")
def classes(run):
    from ttu_tower.io.store import read_table
    run_dir, cfg = run
    return stability_classes(read_table(run_dir / "tertiary" / "data" / "pairs"), cfg.post)


def _read(run_dir, stage, table):
    from ttu_tower.io.store import read_table
    return read_table(run_dir / stage / "data" / table)


def test_tau_agreement_structure(run, classes):
    run_dir, cfg = run
    result = checks.tau_agreement(_read(run_dir, "secondary", "tau"), _read(run_dir, "secondary", "tau_selected"), classes)
    assert set(result) == {"agreement", "selected_source", "status_distribution"}
    for df in result.values():
        assert not df.empty
    assert (result["selected_source"].groupby(["stability_class", "boom"])["fraction"].sum().round(6) == 1.0).all()
    agreement = result["agreement"]
    assert (agreement["frac_within_1_rung"] >= agreement["frac_exact"]).all()


def test_file_stability_class_handles_an_all_missing_series_regardless_of_dtype():
    # A `stability_classes()` Series built from a mix of strings and `None`
    # can get inferred as pandas' "str" dtype rather than plain "object" -
    # which silently turns the unmatched `None`s into float NaN. A filter
    # that only checks `is not None` misses that and leaves NaN in the list,
    # which crashes `pd.Series(...).mode().iloc[0]` (mode() drops NaN by
    # default, so an all-NaN group's mode is empty). Regression for exactly
    # that: found on the real April 2014 pilot run, where most slots have no
    # stability class at all.
    classes_object_dtype = pd.Series([None, "neutral", None], index=[9, 10, 11], dtype=object)
    classes_str_dtype = pd.Series([None, "neutral", None], index=[9, 10, 11]).astype("str")
    assert classes_str_dtype.iloc[0] is not None  # confirms the dtype quirk this test guards against
    assert pd.isna(classes_str_dtype.iloc[0])

    for classes in (classes_object_dtype, classes_str_dtype):
        assert checks._file_stability_class(3, classes) == "neutral"  # half_hour 3 -> slots 9, 10, 11

    all_missing = pd.Series([None, None, None], index=[18, 19, 20]).astype("str")
    assert checks._file_stability_class(6, all_missing) is None  # half_hour 6 -> slots 18, 19, 20


def test_clean_injection_positions_leave_room_for_the_longest_injected_run():
    # despike_calibration injects 1-3 sample runs at chosen "clean"
    # positions; a position within 2 samples of the array's end overflows
    # when a 3-sample run is drawn for it (found on the real pilot run, only
    # surfaced with the full ~90000-sample files and enough injections to
    # hit the last few samples by chance). The fix restricts candidate
    # positions to `< x.size - 3`; confirm the longest run then always fits.
    x = np.arange(10, dtype=float)
    clean = np.flatnonzero(np.ones(10, dtype=bool))
    clean = clean[clean < x.size - 3]
    assert clean.max() == 6
    injected = inject_spikes(x, [clean.max()], [3], [5.0])
    assert injected.size == x.size


def test_despike_calibration_structure(run):
    run_dir, cfg = run
    rng = np.random.default_rng(1)
    result = checks.despike_calibration(cfg, run_dir, rng, n_per_class=3, thresholds=(4.0, 6.0), inject_sigmas=(6.0, 12.0))
    assert set(result) >= {"removal_rates", "old_method", "excursion_lengths", "injection", "numpy_warnings"}
    assert list(result["numpy_warnings"].columns) == ["message", "count"]
    rates = result["removal_rates"]
    assert not rates.empty
    assert (rates["spike_fraction"].between(0, 1)).all()
    # a higher z_threshold should never flag more than a lower one, on average.
    by_var = rates.groupby(["variable", "z_threshold"])["spike_fraction"].mean().unstack("z_threshold")
    assert (by_var[6.0] <= by_var[4.0] + 1e-9).all()
    injection = result["injection"]
    if not injection.empty:
        assert (injection["detection_rate"].between(0, 1)).all()
        assert (injection["false_removal_rate"].between(0, 1)).all()


def test_its_bias_synthetic_ratio_approaches_one_for_long_tau():
    cfg = SimpleNamespace(ladder=SimpleNamespace(its_max_lag_fraction=0.3))
    rng = np.random.default_rng(2)
    df = checks._its_bias_synthetic(cfg, rng, T_values=(1.0,), n_realizations=8, n_top_blocks=10)
    long_tau = df[df["tau_over_T"] >= 100]
    assert abs(long_tau["its_ratio"].mean() - 1.0) < 0.25


def test_its_bias_real_structure(run, classes):
    run_dir, cfg = run
    result = checks.its_bias(cfg, _read(run_dir, "secondary", "boom_stats"), _read(run_dir, "secondary", "tau_selected"),
                              classes, np.random.default_rng(3), T_values=(1.0,), n_realizations=2, n_top_blocks=5)
    assert set(result) == {"synthetic", "real_ratio", "real_short_rates"}
    assert set(result["real_ratio"]["variable"].unique()) <= {"u", "v", "w", "vpts"}


def test_interp_bias_synthetic_bias_small_for_tiny_gap_fraction():
    cfg = SimpleNamespace(qc=SimpleNamespace(max_fill_gap_samples=50), ladder=SimpleNamespace(its_max_lag_fraction=0.3))
    rng = np.random.default_rng(4)
    x, y = correlated_ou(10 * 60_000, 1.0, [1.0, 1.0], [[1.0, -0.3], [-0.3, 1.0]], rng)
    df = checks._interp_bias_from_series(x, y, cfg, rng, fractions=(0.001,), n_realizations=5)
    finest = df[df["rung_s"] == df["rung_s"].min()]
    assert abs(finest["var_x_bias"].mean()) < 0.05
    assert abs(finest["cov_bias"].mean()) < 0.05


def test_floor_peak_structure(run, classes):
    run_dir, cfg = run
    result = checks.floor_peak(_read(run_dir, "primary", "mrd"), _read(run_dir, "secondary", "tau"), classes, cfg)
    assert set(result) == {"peak_scale_distribution", "floor_rate", "clip_rate"}
    assert (result["floor_rate"]["floor_fraction"].between(0, 1)).all()


def test_clip_rate_catches_a_reversal_one_rung_above_min_tau_s():
    # tau_s = max(scale(r-1), min_tau_s) (secondary/detect.py), so a reversal
    # one rung above min_tau_s (18.75 s) still floors tau_s to min_tau_s
    # (9.375 s) - reversal_scale_s <= min_tau_s alone would miss this row
    # entirely and undercount the true floor-hit rate.
    tau = pd.DataFrame([
        {"slot": 1, "boom": 1, "variant": "mrd", "cospectrum": "momentum", "status": "found",
         "tau_s": 9.375, "reversal_scale_s": 18.75, "peak_scale_s": 2.34},
        {"slot": 2, "boom": 1, "variant": "mrd", "cospectrum": "momentum", "status": "found",
         "tau_s": 37.5, "reversal_scale_s": 75.0, "peak_scale_s": 4.69},
    ])
    mrd = pd.DataFrame(columns=["slot", "boom", "variant", "spectrum", "scale_s"])
    classes = pd.Series(["neutral", "neutral"], index=[1, 2])
    cfg = SimpleNamespace(secondary=SimpleNamespace(detection=SimpleNamespace(min_tau_s=9.375)))

    result = checks.floor_peak(mrd, tau, classes, cfg, booms=(1,))
    clip_rate = result["clip_rate"]
    row = clip_rate[(clip_rate["boom"] == 1) & (clip_rate["cospectrum"] == "momentum")].iloc[0]
    assert row["clip_fraction"] == 0.5


def test_yield_report_structure(run, classes):
    run_dir, cfg = run
    result = checks.yield_report(run_dir / "primary", run_dir / "tertiary", classes)
    assert set(result) == {"flag_rates", "survival_by_variant"}
    survival = result["survival_by_variant"]
    assert (survival["fraction"].between(0, 1)).all()
    assert set(survival["variant"].unique()) == {"none", "mrd", "mrd_unexcised", "naive"}


def test_tau_profile_structure(run, classes):
    run_dir, cfg = run
    result = checks.tau_profile(_read(run_dir, "secondary", "tau_selected"), classes)
    assert set(result) == {"tau_profile", "source_mix"}
    profile = result["tau_profile"]
    assert (profile["q75"] >= profile["median"]).all()
    assert (profile["median"] >= profile["q25"]).all()


def test_mrd_vs_naive_structure(run, classes):
    run_dir, cfg = run
    result = checks.mrd_vs_naive(_read(run_dir, "tertiary", "boom_final"), _read(run_dir, "primary", "slot_boom"), classes)
    assert set(result) == {"mrd_vs_naive", "mrd_vs_unexcised"}
    assert set(result["mrd_vs_naive"]["quantity"].unique()) <= {"ustar", "wvpts_cov", "ti", "sigma_u", "ils_u", "zeta"}


def test_sanity_structure(run, classes):
    run_dir, cfg = run
    result = checks.sanity(
        _read(run_dir, "secondary", "slow"), _read(run_dir, "secondary", "tau_selected"),
        _read(run_dir, "primary", "slot_boom"), _read(run_dir, "primary", "mrd"),
        _read(run_dir, "primary", "mrd_frame"), _read(run_dir, "primary", "means"), classes,
    )
    assert set(result) == {
        "p_measured_rate", "p_factor_distribution", "tau_status_rates",
        "unexcised_computed_rate", "n_pairs_at_20min", "wind_direction_diff",
    }
    assert (result["unexcised_computed_rate"]["unexcised_computed_fraction"].between(0, 1)).all()


def test_write_validation_reports_writes_every_check(run):
    run_dir, cfg = run
    checks.write_validation_reports(run_dir, cfg, checks=["tau_profile", "sanity"], seed=5)
    base = run_dir / "reports" / "validation"
    assert (base / "tau_profile" / "tau_profile.csv").is_file()
    assert (base / "sanity" / "p_factor_distribution.csv").is_file()


def test_run_check_rejects_unknown_name(run):
    run_dir, cfg = run
    with pytest.raises(ValueError, match="unknown check"):
        checks.run_check("not_a_real_check", run_dir, cfg, np.random.default_rng(0))
