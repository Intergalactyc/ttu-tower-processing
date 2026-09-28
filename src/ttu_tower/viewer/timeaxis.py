"""Slot and sample indices <-> unix seconds (the viewer's plotting x axis)
and display-timezone offsets.
"""
import numpy as np
import pandas as pd

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.timegrid import EPOCH, SAMPLES_PER_SLOT

EPOCH_UNIX = EPOCH.timestamp()
SLOT_SECONDS = SAMPLES_PER_SLOT / SAMPLE_HZ  # 600


def slot_to_unix(k):
    """Slot index (scalar or array) -> unix seconds of its start."""
    return EPOCH_UNIX + SLOT_SECONDS * np.asarray(k, dtype=np.float64)


def unix_to_slot(x):
    """Unix seconds -> the slot containing them (scalar or int64 array)."""
    k = np.floor((np.asarray(x, dtype=np.float64) - EPOCH_UNIX) / SLOT_SECONDS).astype(np.int64)
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
    return t.floor("10min")


def format_slot(k: int, tz: str) -> str:
    start = pd.Timestamp(slot_to_unix(k), unit="s", tz="UTC").tz_convert(tz)
    return f"{start:%Y-%m-%d %H:%M}–{start + pd.Timedelta(minutes=10):%H:%M} ({start:%z})"
