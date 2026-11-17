"""`era5.toml` -> frozen dataclasses. Separate from the tower run config."""
import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ERA5ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Stencil:
    south: float
    north: float
    west: float
    east: float


@dataclass(frozen=True)
class DownloadConfig:
    area: tuple[float, float, float, float]  # N, W, S, E
    pressure_levels: tuple[int, ...]
    model_level_top: int
    chunk_days: dict[str, int]
    parallel: int


@dataclass(frozen=True)
class ProcessConfig:
    source: str
    stencil: Stencil
    heights_agl: tuple[float, ...]


@dataclass(frozen=True)
class ERA5Config:
    data_dir: Path
    start: dt.date
    end: dt.date
    download: DownloadConfig
    process: ProcessConfig


_KEYS = {
    "paths": {"data_dir"},
    "period": {"start", "end"},
    "download": {"area", "pressure_levels", "model_level_top", "chunk_days", "parallel"},
    "process": {"source", "stencil", "heights_agl"},
}


def _check_keys(d: dict, allowed: set, path: str):
    for key in d:
        if key not in allowed:
            raise ERA5ConfigError(f"{path}{key}: unknown key")
    for key in allowed:
        if key not in d:
            raise ERA5ConfigError(f"{path}{key}: missing")


def load(path) -> ERA5Config:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    _check_keys(d, set(_KEYS), "")
    for section, keys in _KEYS.items():
        _check_keys(d[section], keys, f"{section}.")

    period = d["period"]
    start, end = dt.date.fromisoformat(period["start"]), dt.date.fromisoformat(period["end"])
    if end < start:
        raise ERA5ConfigError("period.end: before period.start")

    dl = d["download"]
    if len(dl["area"]) != 4:
        raise ERA5ConfigError("download.area: expected [N, W, S, E]")
    if not 1 <= dl["model_level_top"] <= 137:
        raise ERA5ConfigError("download.model_level_top: 1-137")
    _check_keys(dl["chunk_days"], {"surface", "pressure", "model"}, "download.chunk_days.")

    pr = d["process"]
    if pr["source"] not in ("model", "pressure"):
        raise ERA5ConfigError("process.source: 'model' or 'pressure'")
    _check_keys(pr["stencil"], {"south", "north", "west", "east"}, "process.stencil.")

    return ERA5Config(
        data_dir=Path(d["paths"]["data_dir"]),
        start=start,
        end=end,
        download=DownloadConfig(
            area=tuple(float(x) for x in dl["area"]),
            pressure_levels=tuple(int(x) for x in dl["pressure_levels"]),
            model_level_top=int(dl["model_level_top"]),
            chunk_days={k: int(v) for k, v in dl["chunk_days"].items()},
            parallel=int(dl["parallel"]),
        ),
        process=ProcessConfig(
            source=pr["source"],
            stencil=Stencil(**{k: float(v) for k, v in pr["stencil"].items()}),
            heights_agl=tuple(float(h) for h in pr["heights_agl"]),
        ),
    )
