"""Frozen dataclasses for the run configuration, one nested dataclass per
TOML section.
"""
from dataclasses import dataclass, fields, is_dataclass


@dataclass(frozen=True)
class PathsConfig:
    raw_dirs: tuple[str, ...]
    results_dir: str
    mesonet_dir: str


@dataclass(frozen=True)
class OutputConfig:
    timezone: str


@dataclass(frozen=True)
class PeriodConfig:
    start: str
    end: str


@dataclass(frozen=True)
class FilesConfig:
    bad_records: tuple[tuple[int, int], ...]
    max_name_offset_min: int
    booms: tuple[int, ...]


@dataclass(frozen=True)
class BoundsConfig:
    sonic: tuple[float, float]
    ts: tuple[float, float]
    t: tuple[float, float]
    rh: tuple[float, float]
    p: tuple[float, float]


@dataclass(frozen=True)
class DespikeConfig:
    window_s: float
    stride_s: float
    z_threshold: float
    max_spike_samples: int
    min_mad: dict[str, float]


@dataclass(frozen=True)
class WindowsConfig:
    resolution_window_s: float
    resolution_stride_s: float
    resolution_bins: int
    ts_resolution_bins: int
    max_empty_bin_fraction: float
    dropout_min_run: int
    ts_dropout_min_run: int
    ts_repeat_tolerance: float
    max_dropout_fraction: float
    dropout_min_windows: int
    skew_range: tuple[float, float]
    kurt_range: tuple[float, float]


@dataclass(frozen=True)
class SlowConfig:
    smoothing_width_s: dict[str, float]


@dataclass(frozen=True)
class SecondLayerConfig:
    shadow_sector: tuple[float, float]
    bounce_max_ws_mean: float
    bounce_max_ws_std: float


@dataclass(frozen=True)
class QCConfig:
    min_coverage: float
    max_fill_gap_samples: int
    unusable_tests: tuple[str, ...]
    bounds: BoundsConfig
    despike: DespikeConfig
    windows: WindowsConfig
    slow: SlowConfig
    second_layer: SecondLayerConfig


@dataclass(frozen=True)
class MRDConfig:
    floor_level: int


@dataclass(frozen=True)
class LadderConfig:
    its_max_lag_fraction: float
    gust_periods_s: tuple[int, ...]


@dataclass(frozen=True)
class PrimaryConfig:
    unexcised: bool
    unexcised_max_fill_gap_s: int
    batch_max_files: int
    nproc: int


@dataclass(frozen=True)
class DetectionConfig:
    peak_significance_se: float
    min_scale_s: float
    min_tau_s: float
    max_tau_s: float


@dataclass(frozen=True)
class SelectionConfig:
    rule: str
    priority: tuple[str, ...]
    fallback_tau_s: float


@dataclass(frozen=True)
class SecondaryConfig:
    min_tau_its_ratio: float
    detection: DetectionConfig
    selection: SelectionConfig


@dataclass(frozen=True)
class FitsConfig:
    exclude_tau: tuple[str, ...]
    min_booms: int
    alpha_booms: tuple[int, ...]
    gamma_booms: tuple[int, ...]
    wdgamma_booms: tuple[int, ...]
    loglaw_booms: tuple[int, ...]


@dataclass(frozen=True)
class TertiaryConfig:
    min_coverage: float
    max_bounds_fraction: float
    max_spike_fraction: float
    veer_reference_boom: int
    fits: FitsConfig


@dataclass(frozen=True)
class StabilityConfig:
    pair: tuple[int, int]
    classes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PostConfig:
    stability: StabilityConfig


@dataclass(frozen=True)
class Config:
    tag: str
    paths: PathsConfig
    output: OutputConfig
    period: PeriodConfig
    files: FilesConfig
    qc: QCConfig
    mrd: MRDConfig
    ladder: LadderConfig
    primary: PrimaryConfig
    secondary: SecondaryConfig
    tertiary: TertiaryConfig
    post: PostConfig


def to_jsonable(obj):
    """Recursively convert a (possibly nested) dataclass/tuple/dict/scalar
    into plain JSON-serializable Python objects (dataclass -> dict, tuple ->
    list). Used for config.resolved.json and the stage config hashes.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj
