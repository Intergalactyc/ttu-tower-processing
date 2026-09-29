"""A quantity by hour of day and month (the diurnal and annual cycles at once):
a statistic per (hour, month) cell in the run's time zone, pooling years.
"""
import warnings

import numpy as np
import pandas as pd

from ttu_tower.math.polar import yamartino_std
from ttu_tower.viewer.decimate import visible_slice

STATS = ("median", "mean", "std", "count")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _local(curve, tz: str) -> tuple[np.ndarray, np.ndarray]:
    t = pd.to_datetime(curve.x, unit="s", utc=True).tz_convert(tz)
    return np.asarray(t.hour), np.asarray(t.month)


def _in_range(curve, x_range):
    return slice(None) if x_range is None else visible_slice(curve.x, x_range[0], x_range[1], margin=0)


def grid(curve, tz: str, x_range=None, stat: str = "median", circular: bool = False,
         category: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(values[hour, month-1], counts[hour, month-1]). A direction's median and
    mean are its vector mean and its std the Yamartino σ; with `category`, the
    value is the fraction of the cell's slots in that category.
    """
    sl = _in_range(curve, x_range)
    y = curve.y[sl]
    hour, month = _local(type(curve)(slots=curve.slots[sl], y=y), tz)
    ok = np.isfinite(y)
    values = np.full((24, 12), np.nan)
    counts = np.zeros((24, 12), dtype=np.int64)
    if not ok.any():
        return values, counts
    frame = pd.DataFrame({"h": hour[ok], "m": month[ok] - 1, "y": y[ok]})
    if category is not None:
        frame["y"] = (frame["y"].round() == category).astype(float)
    groups = frame.groupby(["h", "m"])["y"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if stat == "count":
            reduced = groups.size().astype(float)
        elif circular and category is None:
            reduced = groups.apply(lambda v: _circular(v.to_numpy(), stat))
        else:
            reduced = getattr(groups, stat)()
    sizes = groups.size()
    h, m = reduced.index.get_level_values(0), reduced.index.get_level_values(1)
    values[h, m] = reduced.to_numpy()
    counts[sizes.index.get_level_values(0), sizes.index.get_level_values(1)] = sizes.to_numpy()
    return values, counts


def _circular(v: np.ndarray, stat: str) -> float:
    s, c = np.sin(np.radians(v)).mean(), np.cos(np.radians(v)).mean()
    if stat == "std":
        return float(yamartino_std(s, c))
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def cell_slots(curve, tz: str, x_range, hour: int, month: int) -> np.ndarray:
    """The slots (with a value) of one cell; `month` is 1-12."""
    sl = _in_range(curve, x_range)
    sub = type(curve)(slots=curve.slots[sl], y=curve.y[sl])
    h, m = _local(sub, tz)
    return sub.slots[(h == hour) & (m == month) & np.isfinite(sub.y)]
