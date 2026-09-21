import json
from pathlib import Path

import pytest

from ttu_tower.config import ConfigError, config_from_dict, load_config, to_resolved_dict

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "configs" / "templates" / "oneyear.toml"


def _minimal_raw() -> dict:
    return {
        "paths": {"raw_dirs": ["/data/raw"]},
        "files": {"bad_records": [[1, 2]]},
    }


def test_minimal_config_matches_documented_defaults():
    cfg = config_from_dict(_minimal_raw(), default_tag="mytag")

    assert cfg.tag == "mytag"
    assert cfg.paths.raw_dirs == ("/data/raw",)
    assert cfg.paths.results_dir == ""
    assert cfg.paths.mesonet_dir == ""
    assert cfg.output.timezone == "Etc/GMT+6"
    assert cfg.period.start == ""
    assert cfg.period.end == ""
    assert cfg.files.bad_records == ((1, 2),)
    assert cfg.files.max_name_offset_min == 2
    assert cfg.files.booms == tuple(range(1, 11))
    assert cfg.qc.min_coverage == 0.75
    assert cfg.qc.max_fill_gap_samples == 50
    assert cfg.qc.unusable_tests == ("unchecked", "resolution", "dropouts", "skew", "kurt", "direction", "bounce")
    assert cfg.qc.bounds.sonic == (-45.0, 45.0)
    assert cfg.qc.bounds.ts == (233.15, 333.15)
    assert cfg.qc.bounds.t == (223.15, 323.15)
    assert cfg.qc.bounds.rh == (0.0, 1.0)
    assert cfg.qc.bounds.p == (50.0, 110.0)
    assert cfg.qc.despike.window_s == 300.0
    assert cfg.qc.despike.stride_s == 30.0
    assert cfg.qc.despike.z_threshold == {
        "ue": 3.5, "vn": 3.5, "w": 3.5, "ts": 3.5, "t": 3.5, "rh": 3.5, "p": 3.5,
    }
    assert cfg.qc.despike.max_spike_samples == 3
    assert cfg.qc.despike.min_mad == {
        "ue": 0.001, "vn": 0.001, "w": 0.001, "ts": 0.01, "t": 0.002, "rh": 0.00002, "p": 0.0004,
    }
    assert cfg.qc.windows.resolution_window_s == 100.0
    assert cfg.qc.windows.resolution_stride_s == 50.0
    assert cfg.qc.windows.resolution_bins == 100
    assert cfg.qc.windows.ts_resolution_bins == 30
    assert cfg.qc.windows.max_empty_bin_fraction == 0.8
    assert cfg.qc.windows.dropout_min_run == 10
    assert cfg.qc.windows.ts_dropout_min_run == 50
    assert cfg.qc.windows.ts_repeat_tolerance == 0.05
    assert cfg.qc.windows.max_dropout_fraction == 0.05
    assert cfg.qc.windows.dropout_min_windows == 2
    assert cfg.qc.windows.skew_range == (-2.0, 2.0)
    assert cfg.qc.windows.kurt_range == (1.0, 8.0)
    assert cfg.qc.slow.smoothing_width_s == {"t": 20.0, "rh": 20.0, "p": 2.0}
    assert cfg.qc.second_layer.shadow_sector == (105.0, 170.0)
    assert cfg.qc.second_layer.bounce_max_ws_mean == 30.0
    assert cfg.qc.second_layer.bounce_max_ws_std == 6.75
    assert cfg.mrd.floor_level == 15
    assert cfg.ladder.its_max_lag_fraction == 0.25
    assert cfg.ladder.gust_periods_s == (1, 2, 3, 5, 10, 30, 60)
    assert cfg.primary.unexcised is True
    assert cfg.primary.unexcised_max_fill_gap_s == 600
    assert cfg.primary.batch_max_files == 96
    assert cfg.primary.nproc == 0
    assert cfg.secondary.min_tau_its_ratio == 10.0
    assert cfg.secondary.detection.peak_significance_se == 2.0
    assert cfg.secondary.detection.min_scale_s == 0.5
    assert cfg.secondary.detection.min_tau_s == 9.375
    assert cfg.secondary.detection.max_tau_s == 1200.0
    assert cfg.secondary.selection.rule == "priority"
    assert cfg.secondary.selection.priority == ("heat", "momentum")
    assert cfg.secondary.selection.fallback_tau_s == 600.0
    assert cfg.tertiary.min_coverage == 0.75
    assert cfg.tertiary.max_bounds_fraction == 0.01
    assert cfg.tertiary.max_spike_fraction == 0.01
    assert cfg.tertiary.veer_reference_boom == 4
    assert cfg.tertiary.fits.exclude_tau == ("unresolved", "fallback")
    assert cfg.tertiary.fits.min_booms == 2
    assert cfg.tertiary.fits.alpha_booms == tuple(range(1, 11))
    assert cfg.tertiary.fits.gamma_booms == tuple(range(1, 11))
    assert cfg.tertiary.fits.wdgamma_booms == tuple(range(1, 11))
    assert cfg.tertiary.fits.loglaw_booms == (1, 2, 3, 4, 5, 6, 7)
    assert cfg.post.stability.pair == (2, 4)
    assert cfg.post.stability.classes == (
        ("strongly unstable", "(-inf,-0.05)"),
        ("unstable", "[-0.05,-0.01)"),
        ("neutral", "[-0.01,0.01)"),
        ("stable", "[0.01,0.1)"),
        ("strongly stable", "[0.1,inf)"),
    )


def test_load_config_tag_defaults_to_file_stem(tmp_path):
    path = tmp_path / "myrun.toml"
    path.write_text('[paths]\nraw_dirs = ["/data"]\n\n[files]\nbad_records = [[1, 2]]\n')
    cfg = load_config(path)
    assert cfg.tag == "myrun"


def test_explicit_tag_overrides_file_stem(tmp_path):
    path = tmp_path / "myrun.toml"
    path.write_text('tag = "custom"\n[paths]\nraw_dirs = ["/data"]\n\n[files]\nbad_records = [[1, 2]]\n')
    cfg = load_config(path)
    assert cfg.tag == "custom"


def test_template_loads_and_matches_documented_defaults():
    cfg = load_config(TEMPLATE_PATH)
    minimal = config_from_dict(_minimal_raw(), default_tag=cfg.tag)
    assert cfg.qc == minimal.qc
    assert cfg.mrd == minimal.mrd
    assert cfg.ladder == minimal.ladder
    assert cfg.primary == minimal.primary
    assert cfg.secondary == minimal.secondary
    assert cfg.tertiary == minimal.tertiary
    assert cfg.post == minimal.post


def test_round_trip_through_resolved_json():
    cfg = load_config(TEMPLATE_PATH)
    reloaded_raw = json.loads(json.dumps(to_resolved_dict(cfg) | {"tag": cfg.tag}))
    cfg2 = config_from_dict(reloaded_raw)
    assert cfg2 == cfg


# --- unknown keys -----------------------------------------------------------

def test_unknown_top_level_key_raises():
    raw = _minimal_raw()
    raw["bogus"] = 1
    with pytest.raises(ConfigError, match="bogus"):
        config_from_dict(raw, default_tag="t")


def test_unknown_nested_key_raises():
    raw = _minimal_raw()
    raw["qc"] = {"bogus": 1}
    with pytest.raises(ConfigError, match=r"qc\.bogus"):
        config_from_dict(raw, default_tag="t")


def test_unknown_key_in_min_mad_table_raises():
    raw = _minimal_raw()
    raw["qc"] = {"despike": {"min_mad": {"bogus": 1.0}}}
    with pytest.raises(ConfigError, match=r"min_mad\.bogus"):
        config_from_dict(raw, default_tag="t")


def test_z_threshold_overrides_one_variable_and_keeps_the_rest_at_default():
    raw = _minimal_raw()
    raw["qc"] = {"despike": {"z_threshold": {"p": 6.0}}}
    cfg = config_from_dict(raw, default_tag="t")

    assert cfg.qc.despike.z_threshold["p"] == 6.0
    assert cfg.qc.despike.z_threshold["ue"] == 3.5
    assert cfg.qc.despike.z_threshold["t"] == 3.5


def test_unknown_key_in_z_threshold_table_raises():
    raw = _minimal_raw()
    raw["qc"] = {"despike": {"z_threshold": {"bogus": 5.0}}}
    with pytest.raises(ConfigError, match=r"z_threshold\.bogus"):
        config_from_dict(raw, default_tag="t")


# --- required keys and the tag -----------------------------------------------

def test_missing_raw_dirs_raises():
    with pytest.raises(ConfigError, match="raw_dirs"):
        config_from_dict({"files": {"bad_records": [[1, 2]]}}, default_tag="t")


def test_missing_bad_records_raises():
    with pytest.raises(ConfigError, match="bad_records"):
        config_from_dict({"paths": {"raw_dirs": ["/x"]}}, default_tag="t")


def test_invalid_tag_raises():
    raw = _minimal_raw()
    raw["tag"] = "bad tag!"
    with pytest.raises(ConfigError):
        config_from_dict(raw)


# --- cross-key rules ---------------------------------------------------------

def test_cross_key_floor_level_too_low():
    raw = _minimal_raw()
    raw["mrd"] = {"floor_level": 8}
    with pytest.raises(ConfigError, match="floor_level"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_despike_ratio_not_even_integer():
    raw = _minimal_raw()
    raw["qc"] = {"despike": {"window_s": 90.0, "stride_s": 30.0}}  # ratio 3, odd
    with pytest.raises(ConfigError, match="window_s"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_stride_must_divide_1800():
    raw = _minimal_raw()
    raw["qc"] = {"despike": {"window_s": 14.0, "stride_s": 7.0}}  # ratio 2 (even), but 1800/7 not integer
    with pytest.raises(ConfigError, match="stride_s"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_resolution_stride_must_divide_1800():
    raw = _minimal_raw()
    raw["qc"] = {"windows": {"resolution_stride_s": 7.0}}
    with pytest.raises(ConfigError, match="resolution_stride_s"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_stage_b_reach_exceeds_margin():
    raw = _minimal_raw()
    raw["qc"] = {"despike": {"window_s": 1200.0, "stride_s": 100.0}}  # reach 1300s > 600s
    with pytest.raises(ConfigError, match="margin"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_tau_ordering():
    raw = _minimal_raw()
    raw["secondary"] = {"selection": {"fallback_tau_s": 5.0}}  # below min_tau_s
    with pytest.raises(ConfigError, match="min_tau_s"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_tau_must_be_ladder_rung():
    raw = _minimal_raw()
    raw["secondary"] = {"selection": {"fallback_tau_s": 20.0}}  # in range, not a rung
    with pytest.raises(ConfigError, match="ladder rung"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_min_scale_must_be_below_min_tau():
    raw = _minimal_raw()
    raw["secondary"] = {"detection": {"min_scale_s": 9.375, "min_tau_s": 9.375}}
    with pytest.raises(ConfigError, match="min_scale_s"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_unusable_tests_must_be_known():
    raw = _minimal_raw()
    raw["qc"] = {"unusable_tests": ["bogus_test"]}
    with pytest.raises(ConfigError, match="unusable_tests"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_veer_reference_boom_must_be_in_booms():
    raw = _minimal_raw()
    raw["files"] = {"bad_records": [[1, 2]], "booms": [1, 2, 3]}
    raw["tertiary"] = {"veer_reference_boom": 5}
    with pytest.raises(ConfigError, match="veer_reference_boom"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_fits_booms_must_be_subset_of_booms():
    raw = _minimal_raw()
    raw["files"] = {"bad_records": [[1, 2]], "booms": [1, 2, 3, 4]}  # keep default veer_reference_boom (4) valid
    raw["tertiary"] = {"fits": {"alpha_booms": [1, 5]}}
    with pytest.raises(ConfigError, match="alpha_booms"):
        config_from_dict(raw, default_tag="t")


def test_cross_key_timezone_must_be_allowed():
    raw = _minimal_raw()
    raw["output"] = {"timezone": "America/Chicago"}
    with pytest.raises(ConfigError):
        config_from_dict(raw, default_tag="t")


def test_cross_key_results_dir_must_be_absolute():
    raw = _minimal_raw()
    raw["paths"]["results_dir"] = "relative/path"
    with pytest.raises(ConfigError, match="results_dir"):
        config_from_dict(raw, default_tag="t")


# --- per-key notes ------------------------------------------------------------

def test_gust_periods_must_divide_600():
    raw = _minimal_raw()
    raw["ladder"] = {"gust_periods_s": [7]}
    with pytest.raises(ConfigError, match="gust_periods_s"):
        config_from_dict(raw, default_tag="t")


def test_smoothing_width_must_be_positive_multiple_of_0_04():
    raw = _minimal_raw()
    raw["qc"] = {"slow": {"smoothing_width_s": {"p": 2.03}}}
    with pytest.raises(ConfigError, match="smoothing_width_s"):
        config_from_dict(raw, default_tag="t")


def test_period_must_be_on_10_minute_boundary():
    raw = _minimal_raw()
    raw["period"] = {"start": "2013-11-01 00:05"}
    with pytest.raises(ConfigError, match="10-minute"):
        config_from_dict(raw, default_tag="t")
