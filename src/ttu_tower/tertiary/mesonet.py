"""Mesonet merge: the WTX mesonet station, resampled to the tower's slot
cadence and joined onto it by `slot_start`.
"""
import os

import numpy as np
import pandas as pd

from ttu_tower.math.polar import bearing_to_vector, vector_to_bearing

_COL_MAP = {
    "10m WS scalar": "ws_meso_mean",
    "10m WS vector": "_WSVEC",
    "10m WD": "_WD",
    "WD Standard Deviation": "wd_meso_std",
    "WS Standard Deviation": "ws_meso_std",
    "9m Temp": "t_meso_mean",
    "1.5m RH": "rh_meso_mean",
    "barometric pressure": "p_meso_mean",
    "2m Solar Radiation": "solar",
    "Precip": "precip",
}
_MB_TO_KPA = 0.1
_AVG_PERIOD = "10min"


def get_df(data_path: str) -> pd.DataFrame:
    """The mesonet station's raw DataFrame (`stationinfo.xls`/`data/` under
    `data_path`). Imports `wtxmeso` lazily, only when a merge is requested,
    with a clear error if the `mesonet` extra isn't installed.
    """
    try:
        import wtxmeso as wtx
    except ImportError as e:
        raise ImportError(
            "reading mesonet data needs the 'mesonet' extra: pip install 'ttu-tower-processing[mesonet]'"
        ) from e
    reader = wtx.Reader(os.path.join(data_path, "stationinfo.xls"), "atmospheric")
    reader.read_directory(os.path.join(data_path, "data"))
    return reader.stations[0].df


def _extrapolate_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly extrapolate leading/trailing NaN runs in each column from
    that column's two nearest valid samples. Assumes a uniformly-spaced
    index (so integer position is proportional to time) and that interior
    gaps are already filled, e.g. by `interpolate(limit_area="inside")`.
    """
    df = df.copy()
    positions = np.arange(len(df))
    for col in df.columns:
        values = df[col].to_numpy(dtype=float, copy=True)
        valid = np.flatnonzero(~np.isnan(values))
        if valid.size < 2:
            continue
        first, second = valid[0], valid[1]
        if first > 0:
            slope = (values[second] - values[first]) / (second - first)
            lead = positions[:first]
            values[:first] = values[first] + slope * (lead - first)
        last, second_last = valid[-1], valid[-2]
        if last < len(values) - 1:
            slope = (values[last] - values[second_last]) / (last - second_last)
            trail = positions[last + 1:]
            values[last + 1:] = values[last] + slope * (trail - last)
        df[col] = values
    return df


def _fill_resampling_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the NaN bins left by resampling to a finer grid than the native
    mesonet cadence: interior gaps by linear interpolation, leading/trailing
    gaps by linear extrapolation. Must run before any column derived from
    these (e.g. a vector mean/bearing from u/v components) is computed, so
    the derived column stays consistent with its gap-filled inputs.
    """
    df = df.interpolate(method="linear", limit_area="inside")
    df = _extrapolate_edges(df)
    return df


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Mesonet source units (C, %, mb) -> the pipeline's internal SI (K,
    fraction, kPa); ws (m/s) and wd (degrees) already match.
    """
    df = df.copy()
    df["t_meso_mean"] = df["t_meso_mean"] + 273.15
    df["rh_meso_mean"] = df["rh_meso_mean"] / 100.0
    df["p_meso_mean"] = df["p_meso_mean"] * _MB_TO_KPA
    return df


def mesonet_data(data_path: str, timezone: str) -> pd.DataFrame:
    """The mesonet station's columns, resampled to the fixed 10-min slot
    cadence and indexed by `slot_start` (tz-aware, in `timezone`).
    """
    df = get_df(data_path)
    df = df.drop(columns=[c for c in df.columns if c not in _COL_MAP])
    df = df.rename(mapper=_COL_MAP, axis="columns")
    df["u_meso_mean"], df["v_meso_mean"] = bearing_to_vector(df["_WSVEC"], df["_WD"])
    df = df.drop(columns=["_WSVEC", "_WD"])
    df = _convert_units(df)
    df = df.resample(_AVG_PERIOD).mean()
    df = _fill_resampling_gaps(df)
    # wd_meso_mean genuinely needs a vector average (direction can't be
    # scalar-averaged); computed AFTER gap-filling so it's always consistent
    # with u_meso_mean/v_meso_mean, and so a wrapping bearing (e.g. 350deg ->
    # 10deg) is never interpolated directly through 180deg.
    df["ws_meso_vector_mean"], df["wd_meso_mean"] = vector_to_bearing(df["u_meso_mean"], df["v_meso_mean"])
    df = df.asfreq(_AVG_PERIOD)
    df["ti_meso"] = df["ws_meso_std"] / df["ws_meso_mean"].replace(0, np.nan)
    df.index = df.index.tz_convert(timezone)
    df.index.name = "slot_start"
    return df


def combine(slots: pd.DataFrame, meso: pd.DataFrame) -> pd.DataFrame:
    """Left-join `meso` (indexed by `slot_start`) onto `slots` (with a
    `slot_start` column): non-overlapping mesonet rows leave NaN, not a
    dropped tower slot.
    """
    return slots.join(meso, on="slot_start")
