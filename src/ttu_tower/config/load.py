"""TOML -> Config: fills defaults, validates types/ranges/cross-key rules, and
rejects unknown keys.
"""
import os
import re
import tomllib
from datetime import datetime
from pathlib import Path

from ttu_tower.constants import BOOMS as ALL_BOOMS
from ttu_tower.flags import TESTS as _TEST_REGISTRY
from ttu_tower.config.model import (
    BoundsConfig,
    Config,
    DespikeConfig,
    DetectionConfig,
    FilesConfig,
    FitsConfig,
    LadderConfig,
    MRDConfig,
    OutputConfig,
    PathsConfig,
    PeriodConfig,
    PostConfig,
    PrimaryConfig,
    QCConfig,
    SecondaryConfig,
    SecondLayerConfig,
    SelectionConfig,
    SlowConfig,
    StabilityConfig,
    TertiaryConfig,
    WindowsConfig,
    to_jsonable,
)


class ConfigError(Exception):
    """An invalid, missing or unknown config key."""


_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_QUALITY_TESTS = frozenset(name for name, spec in _TEST_REGISTRY.items() if spec.kind == "quality")
_LADDER_RUNGS = (9.375, 18.75, 37.5, 75.0, 150.0, 300.0, 600.0, 1200.0)

_MIN_MAD_DEFAULT = {"ue": 0.001, "vn": 0.001, "w": 0.001, "ts": 0.01, "t": 0.002, "rh": 0.00002, "p": 0.0004}
_Z_THRESHOLD_DEFAULT = {"ue": 3.5, "vn": 3.5, "w": 3.5, "ts": 3.5, "t": 3.5, "rh": 3.5, "p": 6.0}
_SMOOTHING_WIDTH_DEFAULT = {"t": 20.0, "rh": 20.0, "p": 2.0}
_UNUSABLE_TESTS_DEFAULT = ["unchecked", "resolution", "dropouts", "skew", "kurt", "direction", "bounce"]
_GUST_PERIODS_DEFAULT = [1, 2, 3, 5, 10, 30, 60]
_ALL_BOOMS_DEFAULT = list(range(1, 11))
_LOGLAW_BOOMS_DEFAULT = [1, 2, 3, 4, 5, 6, 7]
_CLASSES_DEFAULT = [
    ["strongly unstable", "(-inf,-0.05)"],
    ["unstable", "[-0.05,-0.01)"],
    ["neutral", "[-0.01,0.01)"],
    ["stable", "[0.01,0.1)"],
    ["strongly stable", "[0.1,inf)"],
]

_MISSING = object()


def _err(path: str, msg: str):
    raise ConfigError(f"'{path}': {msg}")


def _section(raw: dict, key: str, path: str) -> dict:
    value = raw.pop(key, {})
    if not isinstance(value, dict):
        _err(f"{path}.{key}" if path else key, "must be a table")
    return dict(value)


def _check_unknown(d: dict, path: str):
    if d:
        key = next(iter(d))
        _err(f"{path}.{key}" if path else key, "unknown key")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pop_bool(d: dict, key: str, default: bool, path: str) -> bool:
    if key not in d:
        return default
    value = d.pop(key)
    if not isinstance(value, bool):
        _err(f"{path}.{key}", "must be a boolean")
    return value


def _pop_str(d: dict, key: str, default, path: str, *, choices: set[str] | None = None) -> str:
    if key not in d:
        if default is _MISSING:
            _err(f"{path}.{key}", "is required")
        return default
    value = d.pop(key)
    if not isinstance(value, str):
        _err(f"{path}.{key}", "must be a string")
    if choices is not None and value not in choices:
        _err(f"{path}.{key}", f"must be one of {sorted(choices)}")
    return value


def _pop_number(d: dict, key: str, default, path: str, *, kind=float, min_=None, max_=None, min_incl: bool = True, max_incl: bool = True):
    if key not in d:
        if default is _MISSING:
            _err(f"{path}.{key}", "is required")
        return default
    value = d.pop(key)
    if not _is_number(value):
        _err(f"{path}.{key}", "must be a number")
    if kind is int and not isinstance(value, int):
        _err(f"{path}.{key}", "must be an integer")
    value = kind(value)
    if min_ is not None and (value < min_ if min_incl else value <= min_):
        _err(f"{path}.{key}", f"must be {'>=' if min_incl else '>'} {min_}")
    if max_ is not None and (value > max_ if max_incl else value >= max_):
        _err(f"{path}.{key}", f"must be {'<=' if max_incl else '<'} {max_}")
    return value


def _pop_int_list(d: dict, key: str, default, path: str) -> tuple[int, ...]:
    if key not in d:
        return tuple(default)
    value = d.pop(key)
    if not isinstance(value, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        _err(f"{path}.{key}", "must be a list of integers")
    return tuple(value)


def _pop_str_list(d: dict, key: str, default, path: str) -> tuple[str, ...]:
    if key not in d:
        return tuple(default)
    value = d.pop(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _err(f"{path}.{key}", "must be a list of strings")
    return tuple(value)


def _pop_pair(d: dict, key: str, default, path: str, *, kind=float):
    if key not in d:
        return tuple(default)
    value = d.pop(key)
    if not isinstance(value, list) or len(value) != 2 or not all(_is_number(v) for v in value):
        _err(f"{path}.{key}", "must be a list of two numbers")
    return (kind(value[0]), kind(value[1]))


def _pop_float_dict(d: dict, key: str, defaults: dict[str, float], path: str) -> dict[str, float]:
    value = d.pop(key, {})
    if not isinstance(value, dict):
        _err(f"{path}.{key}", "must be a table")
    result = dict(defaults)
    for k, v in value.items():
        if k not in defaults:
            _err(f"{path}.{key}.{k}", "unknown key")
        if not _is_number(v):
            _err(f"{path}.{key}.{k}", "must be a number")
        result[k] = float(v)
    return result


def _pop_bad_records(d: dict, key: str, path: str) -> tuple[tuple[int, int], ...]:
    if key not in d:
        _err(f"{path}.{key}", "is required")
    value = d.pop(key)
    if not isinstance(value, list):
        _err(f"{path}.{key}", "must be a list of [start, end] pairs")
    records = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in item):
            _err(f"{path}.{key}", "each entry must be a [start, end] integer pair")
        start, end = item
        if start > end:
            _err(f"{path}.{key}", f"range [{start}, {end}] has start > end")
        records.append((start, end))
    return tuple(records)


def _pop_classes(d: dict, key: str, default, path: str) -> tuple[tuple[str, str], ...]:
    value = d.pop(key, None)
    if value is None:
        value = default
    if not isinstance(value, list):
        _err(f"{path}.{key}", "must be a list of [name, interval] pairs")
    result = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2 or not all(isinstance(v, str) for v in item):
            _err(f"{path}.{key}", "each entry must be a [name, interval] string pair")
        result.append((item[0], item[1]))
    return tuple(result)


def _validate_period_timestamp(value: str, key: str):
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        _err(key, "must be 'YYYY-MM-DD HH:MM'")
    if dt.minute % 10 != 0:
        _err(key, "must be on a 10-minute boundary")


def _is_multiple(value: float, step: float, tol: float = 1e-9) -> bool:
    if step == 0:
        return False
    ratio = value / step
    return abs(ratio - round(ratio)) < tol


def _parse_paths(d: dict, path: str) -> PathsConfig:
    raw_dirs = d.pop("raw_dirs", _MISSING)
    if raw_dirs is _MISSING:
        _err(f"{path}.raw_dirs", "is required")
    if not isinstance(raw_dirs, list) or not raw_dirs or not all(isinstance(v, str) for v in raw_dirs):
        _err(f"{path}.raw_dirs", "must be a non-empty list of strings")
    results_dir = _pop_str(d, "results_dir", "", path)
    if results_dir:
        expanded = os.path.expanduser(results_dir)
        if not Path(expanded).is_absolute():
            _err(f"{path}.results_dir", "must be absolute (after ~ expansion), or empty")
        results_dir = expanded
    mesonet_dir = _pop_str(d, "mesonet_dir", "", path)
    _check_unknown(d, path)
    return PathsConfig(raw_dirs=tuple(raw_dirs), results_dir=results_dir, mesonet_dir=mesonet_dir)


def _parse_output(d: dict, path: str) -> OutputConfig:
    timezone = _pop_str(d, "timezone", "Etc/GMT+6", path, choices={"Etc/GMT+6", "UTC"})
    _check_unknown(d, path)
    return OutputConfig(timezone=timezone)


def _parse_period(d: dict, path: str) -> PeriodConfig:
    start = _pop_str(d, "start", "", path)
    end = _pop_str(d, "end", "", path)
    if start:
        _validate_period_timestamp(start, f"{path}.start")
    if end:
        _validate_period_timestamp(end, f"{path}.end")
    _check_unknown(d, path)
    return PeriodConfig(start=start, end=end)


def _parse_files(d: dict, path: str) -> FilesConfig:
    bad_records = _pop_bad_records(d, "bad_records", path)
    max_name_offset_min = _pop_number(d, "max_name_offset_min", 2, path, kind=int, min_=0, max_=29)
    booms = _pop_int_list(d, "booms", _ALL_BOOMS_DEFAULT, path)
    for b in booms:
        if b not in ALL_BOOMS:
            _err(f"{path}.booms", f"boom {b} is not one of {ALL_BOOMS}")
    _check_unknown(d, path)
    return FilesConfig(bad_records=bad_records, max_name_offset_min=max_name_offset_min, booms=booms)


def _parse_bounds(d: dict, path: str) -> BoundsConfig:
    sonic = _pop_pair(d, "sonic", (-45.0, 45.0), path)
    ts = _pop_pair(d, "ts", (233.15, 333.15), path)
    t = _pop_pair(d, "t", (223.15, 323.15), path)
    rh = _pop_pair(d, "rh", (0.0, 1.0), path)
    p = _pop_pair(d, "p", (50.0, 110.0), path)
    _check_unknown(d, path)
    return BoundsConfig(sonic=sonic, ts=ts, t=t, rh=rh, p=p)


def _parse_despike(d: dict, path: str) -> DespikeConfig:
    window_s = _pop_number(d, "window_s", 300.0, path)
    stride_s = _pop_number(d, "stride_s", 30.0, path)
    z_threshold = _pop_float_dict(d, "z_threshold", _Z_THRESHOLD_DEFAULT, path)
    max_spike_samples = _pop_number(d, "max_spike_samples", 3, path, kind=int)
    min_mad = _pop_float_dict(d, "min_mad", _MIN_MAD_DEFAULT, path)
    _check_unknown(d, path)
    return DespikeConfig(window_s=window_s, stride_s=stride_s, z_threshold=z_threshold,
                          max_spike_samples=max_spike_samples, min_mad=min_mad)


def _parse_windows(d: dict, path: str) -> WindowsConfig:
    resolution_window_s = _pop_number(d, "resolution_window_s", 100.0, path)
    resolution_stride_s = _pop_number(d, "resolution_stride_s", 50.0, path)
    resolution_bins = _pop_number(d, "resolution_bins", 100, path, kind=int)
    ts_resolution_bins = _pop_number(d, "ts_resolution_bins", 30, path, kind=int)
    max_empty_bin_fraction = _pop_number(d, "max_empty_bin_fraction", 0.8, path, min_=0, max_=1)
    dropout_min_run = _pop_number(d, "dropout_min_run", 10, path, kind=int)
    ts_dropout_min_run = _pop_number(d, "ts_dropout_min_run", 50, path, kind=int)
    ts_repeat_tolerance = _pop_number(d, "ts_repeat_tolerance", 0.05, path)
    max_dropout_fraction = _pop_number(d, "max_dropout_fraction", 0.05, path, min_=0, max_=1)
    dropout_min_windows = _pop_number(d, "dropout_min_windows", 2, path, kind=int)
    skew_range = _pop_pair(d, "skew_range", (-2.0, 2.0), path)
    kurt_range = _pop_pair(d, "kurt_range", (1.0, 8.0), path)
    _check_unknown(d, path)
    return WindowsConfig(
        resolution_window_s=resolution_window_s, resolution_stride_s=resolution_stride_s,
        resolution_bins=resolution_bins, ts_resolution_bins=ts_resolution_bins,
        max_empty_bin_fraction=max_empty_bin_fraction, dropout_min_run=dropout_min_run,
        ts_dropout_min_run=ts_dropout_min_run, ts_repeat_tolerance=ts_repeat_tolerance,
        max_dropout_fraction=max_dropout_fraction, dropout_min_windows=dropout_min_windows,
        skew_range=skew_range, kurt_range=kurt_range,
    )


def _parse_slow(d: dict, path: str) -> SlowConfig:
    smoothing_width_s = _pop_float_dict(d, "smoothing_width_s", _SMOOTHING_WIDTH_DEFAULT, path)
    for k, v in smoothing_width_s.items():
        if v <= 0 or not _is_multiple(v, 0.04):
            _err(f"{path}.smoothing_width_s.{k}", "must be a positive multiple of 0.04 s")
    _check_unknown(d, path)
    return SlowConfig(smoothing_width_s=smoothing_width_s)


def _parse_second_layer(d: dict, path: str) -> SecondLayerConfig:
    shadow_sector = _pop_pair(d, "shadow_sector", (105.0, 170.0), path)
    bounce_max_ws_mean = _pop_number(d, "bounce_max_ws_mean", 30.0, path)
    bounce_max_ws_std = _pop_number(d, "bounce_max_ws_std", 6.75, path)
    _check_unknown(d, path)
    return SecondLayerConfig(shadow_sector=shadow_sector, bounce_max_ws_mean=bounce_max_ws_mean,
                              bounce_max_ws_std=bounce_max_ws_std)


def _parse_qc(d: dict, path: str) -> QCConfig:
    min_coverage = _pop_number(d, "min_coverage", 0.75, path, min_=0, min_incl=False, max_=1)
    max_fill_gap_samples = _pop_number(d, "max_fill_gap_samples", 50, path, kind=int)
    unusable_tests = _pop_str_list(d, "unusable_tests", _UNUSABLE_TESTS_DEFAULT, path)
    bounds = _parse_bounds(_section(d, "bounds", path), f"{path}.bounds")
    despike = _parse_despike(_section(d, "despike", path), f"{path}.despike")
    windows = _parse_windows(_section(d, "windows", path), f"{path}.windows")
    slow = _parse_slow(_section(d, "slow", path), f"{path}.slow")
    second_layer = _parse_second_layer(_section(d, "second_layer", path), f"{path}.second_layer")
    _check_unknown(d, path)
    return QCConfig(min_coverage=min_coverage, max_fill_gap_samples=max_fill_gap_samples,
                     unusable_tests=unusable_tests, bounds=bounds, despike=despike, windows=windows,
                     slow=slow, second_layer=second_layer)


def _parse_mrd(d: dict, path: str) -> MRDConfig:
    floor_level = _pop_number(d, "floor_level", 15, path, kind=int)
    _check_unknown(d, path)
    return MRDConfig(floor_level=floor_level)


def _parse_ladder(d: dict, path: str) -> LadderConfig:
    its_max_lag_fraction = _pop_number(d, "its_max_lag_fraction", 0.25, path, min_=0, min_incl=False, max_=1)
    gust_periods_s = _pop_int_list(d, "gust_periods_s", _GUST_PERIODS_DEFAULT, path)
    for p in gust_periods_s:
        if p <= 0 or 600 % p != 0:
            _err(f"{path}.gust_periods_s", f"{p} must divide 600")
    _check_unknown(d, path)
    return LadderConfig(its_max_lag_fraction=its_max_lag_fraction, gust_periods_s=gust_periods_s)


def _parse_primary(d: dict, path: str) -> PrimaryConfig:
    unexcised = _pop_bool(d, "unexcised", True, path)
    unexcised_max_fill_gap_s = _pop_number(d, "unexcised_max_fill_gap_s", 600, path, kind=int)
    batch_max_files = _pop_number(d, "batch_max_files", 96, path, kind=int)
    nproc = _pop_number(d, "nproc", 0, path, kind=int, min_=0)
    _check_unknown(d, path)
    return PrimaryConfig(unexcised=unexcised, unexcised_max_fill_gap_s=unexcised_max_fill_gap_s,
                          batch_max_files=batch_max_files, nproc=nproc)


def _parse_detection(d: dict, path: str) -> DetectionConfig:
    peak_significance_se = _pop_number(d, "peak_significance_se", 2.0, path)
    min_scale_s = _pop_number(d, "min_scale_s", 0.5, path)
    min_tau_s = _pop_number(d, "min_tau_s", 9.375, path)
    max_tau_s = _pop_number(d, "max_tau_s", 1200.0, path)
    _check_unknown(d, path)
    return DetectionConfig(peak_significance_se=peak_significance_se, min_scale_s=min_scale_s,
                            min_tau_s=min_tau_s, max_tau_s=max_tau_s)


def _parse_selection(d: dict, path: str) -> SelectionConfig:
    rule = _pop_str(d, "rule", "priority", path, choices={"priority", "longest_significant"})
    priority = _pop_str_list(d, "priority", ["heat", "momentum"], path)
    for p in priority:
        if p not in ("heat", "momentum"):
            _err(f"{path}.priority", f"'{p}' must be 'heat' or 'momentum'")
    fallback_tau_s = _pop_number(d, "fallback_tau_s", 600.0, path)
    _check_unknown(d, path)
    return SelectionConfig(rule=rule, priority=priority, fallback_tau_s=fallback_tau_s)


def _parse_secondary(d: dict, path: str) -> SecondaryConfig:
    min_tau_its_ratio = _pop_number(d, "min_tau_its_ratio", 10.0, path)
    detection = _parse_detection(_section(d, "detection", path), f"{path}.detection")
    selection = _parse_selection(_section(d, "selection", path), f"{path}.selection")
    _check_unknown(d, path)
    return SecondaryConfig(min_tau_its_ratio=min_tau_its_ratio, detection=detection, selection=selection)


def _parse_fits(d: dict, path: str) -> FitsConfig:
    exclude_tau = _pop_str_list(d, "exclude_tau", ["unresolved", "fallback"], path)
    for s in exclude_tau:
        if s not in ("found", "capped", "unresolved", "fallback"):
            _err(f"{path}.exclude_tau", f"'{s}' is not a valid source/source_status")
    min_booms = _pop_number(d, "min_booms", 2, path, kind=int)
    alpha_booms = _pop_int_list(d, "alpha_booms", _ALL_BOOMS_DEFAULT, path)
    gamma_booms = _pop_int_list(d, "gamma_booms", _ALL_BOOMS_DEFAULT, path)
    wdgamma_booms = _pop_int_list(d, "wdgamma_booms", _ALL_BOOMS_DEFAULT, path)
    loglaw_booms = _pop_int_list(d, "loglaw_booms", _LOGLAW_BOOMS_DEFAULT, path)
    _check_unknown(d, path)
    return FitsConfig(exclude_tau=exclude_tau, min_booms=min_booms, alpha_booms=alpha_booms,
                       gamma_booms=gamma_booms, wdgamma_booms=wdgamma_booms, loglaw_booms=loglaw_booms)


def _parse_tertiary(d: dict, path: str) -> TertiaryConfig:
    min_coverage = _pop_number(d, "min_coverage", 0.75, path, min_=0, min_incl=False, max_=1)
    max_bounds_fraction = _pop_number(d, "max_bounds_fraction", 0.01, path, min_=0, max_=1)
    max_spike_fraction = _pop_number(d, "max_spike_fraction", 0.01, path, min_=0, max_=1)
    veer_reference_boom = _pop_number(d, "veer_reference_boom", 4, path, kind=int)
    fits = _parse_fits(_section(d, "fits", path), f"{path}.fits")
    _check_unknown(d, path)
    return TertiaryConfig(min_coverage=min_coverage, max_bounds_fraction=max_bounds_fraction,
                           max_spike_fraction=max_spike_fraction, veer_reference_boom=veer_reference_boom,
                           fits=fits)


def _parse_stability(d: dict, path: str) -> StabilityConfig:
    pair = _pop_pair(d, "pair", (2, 4), path, kind=int)
    classes = _pop_classes(d, "classes", _CLASSES_DEFAULT, path)
    _check_unknown(d, path)
    return StabilityConfig(pair=pair, classes=classes)


def _parse_post(d: dict, path: str) -> PostConfig:
    stability = _parse_stability(_section(d, "stability", path), f"{path}.stability")
    _check_unknown(d, path)
    return PostConfig(stability=stability)


def _validate_cross_key(cfg: Config):
    if cfg.mrd.floor_level < 9:
        _err("mrd.floor_level", "must be >= 9, so floor blocks divide the finest (9.375 s) block")

    despike = cfg.qc.despike
    ratio = despike.window_s / despike.stride_s
    if not _is_multiple(ratio, 1.0) or round(ratio) % 2 != 0:
        _err("qc.despike.window_s", "window_s / stride_s must be an even integer")
    if not _is_multiple(1800, despike.stride_s):
        _err("qc.despike.stride_s", "1800 must be a multiple of stride_s")
    if not _is_multiple(1800, cfg.qc.windows.resolution_stride_s):
        _err("qc.windows.resolution_stride_s", "must divide 1800")

    reach = {
        "qc.despike.{window_s,stride_s}": despike.window_s + despike.stride_s,
        "qc.windows.{resolution_window_s,resolution_stride_s}": cfg.qc.windows.resolution_window_s + cfg.qc.windows.resolution_stride_s,
        "qc.slow.smoothing_width_s": max(cfg.qc.slow.smoothing_width_s.values()) / 2,
        "primary.unexcised_max_fill_gap_s": float(cfg.primary.unexcised_max_fill_gap_s),
    }
    for key, value in reach.items():
        if value > 600:
            _err(key, f"reach {value}s exceeds the 600 s Stage B margin")

    det, sel = cfg.secondary.detection, cfg.secondary.selection
    if not (det.min_tau_s <= sel.fallback_tau_s <= det.max_tau_s <= 1200):
        _err("secondary.detection/selection", "must have min_tau_s <= fallback_tau_s <= max_tau_s <= 1200")
    for key, value in (
        ("secondary.detection.min_tau_s", det.min_tau_s),
        ("secondary.detection.max_tau_s", det.max_tau_s),
        ("secondary.selection.fallback_tau_s", sel.fallback_tau_s),
    ):
        if not any(abs(value - rung) < 1e-9 for rung in _LADDER_RUNGS):
            _err(key, f"must be a ladder rung {_LADDER_RUNGS}")
    if not det.min_scale_s < det.min_tau_s:
        _err("secondary.detection.min_scale_s", "must be < min_tau_s")

    for test in cfg.qc.unusable_tests:
        if test not in _QUALITY_TESTS:
            _err("qc.unusable_tests", f"'{test}' is not a known quality test")

    if cfg.tertiary.veer_reference_boom not in cfg.files.booms:
        _err("tertiary.veer_reference_boom", "must be in files.booms")
    for key in ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms"):
        booms = getattr(cfg.tertiary.fits, key)
        if not set(booms) <= set(cfg.files.booms):
            _err(f"tertiary.fits.{key}", "must be a subset of files.booms")


def config_from_dict(raw: dict, *, default_tag: str = "") -> Config:
    """Build a Config from a plain nested dict (a parsed TOML file, or a
    previously resolved config.resolved.json read back with json.load).
    """
    raw = dict(raw)
    tag = raw.pop("tag", None) or default_tag
    if not tag:
        _err("tag", "is required (or inferred from the config file name)")
    if not _TAG_RE.match(tag):
        _err("tag", f"'{tag}' must match {_TAG_RE.pattern}")

    paths = _parse_paths(_section(raw, "paths", ""), "paths")
    output = _parse_output(_section(raw, "output", ""), "output")
    period = _parse_period(_section(raw, "period", ""), "period")
    files = _parse_files(_section(raw, "files", ""), "files")
    qc = _parse_qc(_section(raw, "qc", ""), "qc")
    mrd = _parse_mrd(_section(raw, "mrd", ""), "mrd")
    ladder = _parse_ladder(_section(raw, "ladder", ""), "ladder")
    primary = _parse_primary(_section(raw, "primary", ""), "primary")
    secondary = _parse_secondary(_section(raw, "secondary", ""), "secondary")
    tertiary = _parse_tertiary(_section(raw, "tertiary", ""), "tertiary")
    post = _parse_post(_section(raw, "post", ""), "post")
    _check_unknown(raw, "")

    cfg = Config(tag=tag, paths=paths, output=output, period=period, files=files, qc=qc, mrd=mrd,
                 ladder=ladder, primary=primary, secondary=secondary, tertiary=tertiary, post=post)
    _validate_cross_key(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    """Read and validate a TOML config file."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return config_from_dict(raw, default_tag=path.stem)


def to_resolved_dict(cfg: Config) -> dict:
    """Every key of `cfg`, defaults filled in, as plain JSON-serializable types."""
    return to_jsonable(cfg)
