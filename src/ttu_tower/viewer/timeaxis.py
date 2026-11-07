"""Slot and sample indices <-> unix seconds (the viewer's plotting x axis)
and display-timezone offsets, on the viewer's current cadence: the open
run's (`set_cadence`, called when a run is opened; the default 10-min
cadence until then). The viewer shows one run's slot grid at a time, so
this is the one place that knows its slot length.
"""
import numpy as np
import pandas as pd

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.timegrid import DEFAULT_CADENCE, EPOCH, Cadence

EPOCH_UNIX = EPOCH.timestamp()
_cadence: Cadence = DEFAULT_CADENCE


def set_cadence(cadence: Cadence) -> None:
    """Make `cadence` (the open run's) the viewer's slot grid."""
    global _cadence
    _cadence = cadence


def cadence() -> Cadence:
    """The viewer's current cadence."""
    return _cadence


def slot_seconds() -> float:
    """The current slot length in seconds (600 at the default cadence)."""
    return _cadence.slot_s


def __getattr__(name):
    if name == "SLOT_SECONDS":  # the old constant, now the current cadence's
        return _cadence.slot_s
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def slot_to_unix(k):
    """Slot index (scalar or array) -> unix seconds of its start."""
    return EPOCH_UNIX + _cadence.slot_s * np.asarray(k, dtype=np.float64)


def unix_to_slot(x):
    """Unix seconds -> the slot containing them (scalar or int64 array)."""
    k = np.floor((np.asarray(x, dtype=np.float64) - EPOCH_UNIX) / _cadence.slot_s).astype(np.int64)
    return int(k) if k.ndim == 0 else k


def sample_to_unix(g):
    return EPOCH_UNIX + np.asarray(g, dtype=np.float64) / SAMPLE_HZ


def display_offset_s(tz: str) -> int:
    """pyqtgraph DateAxisItem's `utcOffset` for a fixed-offset zone: seconds
    *behind* UTC, i.e. positive west (the sign convention of `time.timezone`).
    """
    offset = pd.Timestamp("2000-01-01", tz=tz).utcoffset()
    return -int(offset.total_seconds())


def parse_time(text: str, tz: str) -> pd.Timestamp:
    """A user-typed time, read in `tz`, floored to its slot."""
    t = pd.Timestamp(text)
    t = t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)
    return t.floor(f"{_cadence.slot_min}min")


def format_slot(k: int, tz: str) -> str:
    start = pd.Timestamp(slot_to_unix(k), unit="s", tz="UTC").tz_convert(tz)
    return f"{start:%Y-%m-%d %H:%M}–{start + pd.Timedelta(minutes=_cadence.slot_min):%H:%M} ({start:%z})"


def slot_to_time(k):
    """Slot index (or indices) -> start time(s), tz-aware, on the current cadence."""
    return _cadence.slot_to_time(k)


def time_to_slot(t):
    """A slot-aligned time -> its slot index, on the current cadence."""
    return _cadence.time_to_slot(t)
