"""CDS request builders for the three ERA5 sources, and a resumable runner.

One file per request; a file that exists is skipped, so an interrupted run
resumes where it stopped. File names carry the days they cover, so a partial
test range never shadows a full chunk.
"""
import datetime as dt
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ttu_tower.era5.config import ERA5Config

SURFACE_DATASET = "reanalysis-era5-single-levels"
PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
MODEL_DATASET = "reanalysis-era5-complete"

SURFACE_VARIABLES = [
    "forecast_surface_roughness",
    "boundary_layer_height",
    "instantaneous_surface_sensible_heat_flux",
    "instantaneous_moisture_flux",
    "friction_velocity",
    "instantaneous_eastward_turbulent_surface_stress",
    "instantaneous_northward_turbulent_surface_stress",
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "skin_temperature",
]
INVARIANT_VARIABLES = ["geopotential"]
PRESSURE_VARIABLES = [
    "geopotential", "temperature", "specific_humidity",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
]
# z, t, u, v, q, w (omega), lnsp; z and lnsp exist only at level 1.
MODEL_PARAMS = "129/130/131/132/133/135/152"

HOURS = [f"{h:02d}:00" for h in range(24)]


@dataclass(frozen=True)
class Request:
    dataset: str
    request: dict
    target: Path


def days(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def chunks(ds: list[dt.date], max_days: int, within_month: bool = True) -> list[list[dt.date]]:
    """Sorted days split into runs of consecutive days of at most `max_days`,
    optionally never crossing a month.
    """
    out: list[list[dt.date]] = []
    for d in ds:
        if (out and len(out[-1]) < max_days and d - out[-1][-1] == dt.timedelta(days=1)
                and (not within_month or out[-1][0].month == d.month)):
            out[-1].append(d)
        else:
            out.append([d])
    return out


def _covered_days(out_dir: Path, prefix: str) -> set[dt.date]:
    """Days covered by existing `<prefix>_YYYYMMDD[-YYYYMMDD].*` files."""
    covered = set()
    for f in out_dir.glob(f"{prefix}_*") if out_dir.is_dir() else []:
        if f.name.endswith(".part"):
            continue
        span = f.name[len(prefix) + 1:].split(".")[0].split("-")
        try:
            first = dt.datetime.strptime(span[0], "%Y%m%d").date()
            last = dt.datetime.strptime(span[-1], "%Y%m%d").date()
        except ValueError:
            continue
        covered.update(days(first, last))
    return covered


def _span(ds: list[dt.date]) -> str:
    return f"{ds[0]:%Y%m%d}-{ds[-1]:%Y%m%d}"


def _cds_time(ds: list[dt.date]) -> dict:
    return {
        "year": [f"{ds[0].year}"],
        "month": [f"{ds[0].month:02d}"],
        "day": [f"{d.day:02d}" for d in ds],
        "time": HOURS,
    }


def surface_requests(cfg: ERA5Config, out_dir: Path, start: dt.date, end: dt.date) -> list[Request]:
    area = list(cfg.download.area)
    reqs = [Request(
        SURFACE_DATASET,
        {"product_type": ["reanalysis"], "variable": INVARIANT_VARIABLES,
         "year": [f"{start.year}"], "month": [f"{start.month:02d}"], "day": [f"{start.day:02d}"],
         "time": ["00:00"], "area": area, "data_format": "netcdf", "download_format": "unarchived"},
        out_dir / "sfc_invariant.nc",
    )]
    for ds in chunks(days(start, end), cfg.download.chunk_days["surface"]):
        reqs.append(Request(
            SURFACE_DATASET,
            {"product_type": ["reanalysis"], "variable": SURFACE_VARIABLES, **_cds_time(ds),
             "area": area, "data_format": "netcdf", "download_format": "unarchived"},
            out_dir / f"sfc_{_span(ds)}.nc",
        ))
    return reqs


def pressure_requests(cfg: ERA5Config, out_dir: Path, start: dt.date, end: dt.date) -> list[Request]:
    levels = [str(p) for p in cfg.download.pressure_levels]
    return [
        Request(
            PRESSURE_DATASET,
            {"product_type": ["reanalysis"], "variable": PRESSURE_VARIABLES, "pressure_level": levels,
             **_cds_time(ds), "area": list(cfg.download.area), "data_format": "netcdf",
             "download_format": "unarchived"},
            out_dir / f"pl_{_span(ds)}.nc",
        )
        for ds in chunks(days(start, end), cfg.download.chunk_days["pressure"])
    ]


def model_levelist(top: int) -> str:
    """Level 1 (for z and lnsp) plus `top`..137."""
    return "/".join(str(k) for k in [1, *range(max(top, 2), 138)])


def model_requests(cfg: ERA5Config, out_dir: Path, start: dt.date, end: dt.date) -> list[Request]:
    """GRIB files of up to `chunk_days.model` days (crossing months), over the
    days not already covered by an existing file. The CDS queue wait is per
    request and dwarfs the run time (E1: 18.5 h queued, 3 min run), so
    requests are as large as the cost limit allows. NetCDF conversion costs 6x
    in that limit; GRIB is also what carries the L137 a/b table.
    """
    n, w, s, e = cfg.download.area
    covered = _covered_days(out_dir, "ml")
    todo = [d for d in days(start, end) if d not in covered]
    return [
        Request(
            MODEL_DATASET,
            {"class": "ea", "expver": "1", "stream": "oper", "type": "an", "levtype": "ml",
             "levelist": model_levelist(cfg.download.model_level_top), "param": MODEL_PARAMS,
             "date": f"{ds[0]:%Y-%m-%d}/to/{ds[-1]:%Y-%m-%d}", "time": "00/to/23/by/1",
             "grid": "0.25/0.25", "area": f"{n}/{w}/{s}/{e}", "data_format": "grib"},
            out_dir / f"ml_{_span(ds)}.grib",
        )
        for ds in chunks(todo, cfg.download.chunk_days["model"], within_month=False)
    ]


BUILDERS = {"surface": surface_requests, "pressure": pressure_requests, "model": model_requests}

_print_lock = threading.Lock()


def _log(msg: str):
    with _print_lock:
        print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _fetch(req: Request) -> Path:
    import cdsapi  # lazy: only the download path needs the 'era5' extra

    tmp = req.target.with_name(req.target.name + ".part")
    t0 = time.monotonic()
    _log(f"submit {req.target.name}")
    cdsapi.Client(quiet=True, progress=False).retrieve(req.dataset, req.request, str(tmp))
    os.replace(tmp, req.target)
    _log(f"done   {req.target.name} ({req.target.stat().st_size / 1e6:.1f} MB, "
         f"{(time.monotonic() - t0) / 60:.1f} min)")
    return req.target


def run(reqs: list[Request], parallel: int) -> list[Request]:
    """Fetch every request whose target doesn't exist yet; return the ones that failed."""
    todo = [r for r in reqs if not r.target.exists()]
    _log(f"{len(reqs) - len(todo)} of {len(reqs)} files present; fetching {len(todo)}")
    if not todo:
        return []
    todo[0].target.parent.mkdir(parents=True, exist_ok=True)
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {pool.submit(_fetch, r): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                fut.result()
            except Exception:
                failed.append(r)
                _log(f"FAILED {r.target.name}\n{traceback.format_exc(limit=3)}")
    _log(f"finished: {len(todo) - len(failed)} fetched, {len(failed)} failed"
         + (" (rerun to retry)" if failed else ""))
    return failed
