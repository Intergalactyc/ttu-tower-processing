"""From jet slots to jet events, the same for every detection algorithm: short
runs of rejected slots (and, optionally, slots missing data) between jet slots
are bridged, runs shorter than the minimum are dropped, and what's left is
numbered in time order with its statistics. Only `is_jet`, `status` and the
nose columns of a detection are read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttu_tower.constants import SOURCE_TIMEZONE
from ttu_tower.jets.identify import NO_DATA, Params, Profiles, param
from ttu_tower.timegrid import slot_to_time

SLOT_MINUTES = 10


@dataclass(frozen=True)
class EventParams(Params):
    max_gap_slots: int = param(
        1, label="bridge gaps up to", unit="slots", kind="int", lo=0, hi=36,
        help="Rejected slots (with data) between two jet slots are counted as jet if there are at most this many; "
             "missing data isn't bridged unless asked.")
    bridge_missing: bool = param(
        False, label="also bridge missing data", kind="bool",
        help="Count slots without the data the algorithm needs (an outage, or a required boom missing or "
             "flagged) as bridgeable gaps too.")
    min_event_slots: int = param(
        6, label="min event length", unit="slots", kind="int", lo=1, hi=144,
        help="Shorter runs of jet slots (counting bridged ones) are dropped.")


SOURCES = ("auto", "edited", "manual")  # as detected; boundaries edited by hand; made by hand


@dataclass(frozen=True)
class Events:
    table: pd.DataFrame  # one row per event, indexed by event_id
    slots: pd.DataFrame  # per slot: event_id (Int32, NA outside events), filled (bridged), manual (added by hand)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of each run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def fill_gaps(is_jet: np.ndarray, bridgeable: np.ndarray, max_gap: int) -> np.ndarray:
    """Which slots get bridged: interior runs of non-jet slots no longer than
    `max_gap`, every one of them `bridgeable`.
    """
    filled = np.zeros(len(is_jet), bool)
    if max_gap <= 0 or not is_jet.any():
        return filled
    starts, ends = _runs(~is_jet)
    blocked = np.concatenate([[0], np.cumsum(~bridgeable)])
    for a, b in zip(starts, ends):
        if a > 0 and b < len(is_jet) and b - a <= max_gap and blocked[b] == blocked[a]:
            filled[a:b] = True
    return filled


def find_events(detection: pd.DataFrame, params: EventParams | None = None, profiles: Profiles | None = None,
                tz: str = SOURCE_TIMEZONE) -> Events:
    """Events of a detection (one row per slot, on a consecutive slot grid).
    `profiles` adds the wind direction at the nose when it carries "wd".
    """
    params = EventParams() if params is None else params
    errors = params.validate()
    if errors:
        raise ValueError("; ".join(errors))
    slots = detection.index.to_numpy(np.int64)
    is_jet = detection["is_jet"].to_numpy(bool)
    if params.bridge_missing:
        bridgeable = np.ones(len(slots), bool)
    else:
        bridgeable = (detection["status"] != NO_DATA).to_numpy(bool)
    filled = fill_gaps(is_jet, bridgeable, params.max_gap_slots)
    starts, ends = _runs(is_jet | filled)
    keep = ends - starts >= params.min_event_slots
    starts, ends = starts[keep], ends[keep]

    return assemble(detection, starts, ends, filled, np.zeros(len(slots), bool), profiles, tz)


def assemble(detection: pd.DataFrame, starts, ends, filled: np.ndarray, manual: np.ndarray,
             profiles: Profiles | None = None, tz: str = SOURCE_TIMEZONE, source=None, rejected=None,
             edits=None) -> Events:
    """Events from spans of the detection's slot grid ([start, end) positions,
    in time order, not overlapping): `filled` marks slots bridged by the rule,
    `manual` slots added by hand; `source` ("auto", "edited", "manual"),
    `rejected` and `edits` (ids) describe each span.
    """
    slots = detection.index.to_numpy(np.int64)
    starts, ends = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    n = len(starts)
    event_id = np.full(len(slots), -1)
    for i, (a, b) in enumerate(zip(starts, ends)):
        event_id[a:b] = i
    inside = event_id >= 0
    per_slot = pd.DataFrame({"event_id": pd.Series(event_id, index=detection.index, dtype="Int32").mask(~inside),
                             "filled": filled & inside, "manual": manual & inside})
    start_slot, end_slot = slots[starts], slots[ends - 1]
    table = pd.DataFrame({
        "start_slot": start_slot,
        "end_slot": end_slot,
        "n_slots": ends - starts,
        "n_filled": np.array([filled[a:b].sum() for a, b in zip(starts, ends)], dtype=np.int64),
        "n_manual": np.array([manual[a:b].sum() for a, b in zip(starts, ends)], dtype=np.int64),
        "duration_min": SLOT_MINUTES * (ends - starts),
        "start_time": slot_to_time(start_slot).tz_convert(tz),
        "end_time": slot_to_time(end_slot + 1).tz_convert(tz),
        "truncated": (starts == 0) | (ends == len(slots)),
        "source": pd.Categorical(["auto"] * n if source is None else list(source), categories=SOURCES),
        "rejected": np.zeros(n, bool) if rejected is None else np.asarray(rejected, bool),
    }, index=pd.RangeIndex(n, name="event_id"))
    table["edits"] = pd.Series([()] * n if edits is None else [tuple(e) for e in edits], index=table.index,
                               dtype=object)
    return Events(table.join(nose_stats(detection, per_slot["event_id"], profiles)), per_slot)


def _circular_mean_deg(deg: pd.Series) -> float:
    rad = np.deg2rad(deg.dropna().to_numpy())
    if rad.size == 0:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


NOSE_FROM = ("jets", "candidates")  # which slots' noses describe an event


def nose_rows(detection: pd.DataFrame, event_of: pd.Series, nose_from: str = "jets") -> pd.DataFrame:
    """The detection rows whose nose describes their event, with its id: the
    jet slots ("jets"), or every slot of the event with a candidate ("candidates").
    """
    keep = event_of.notna() & (detection["is_jet"] if nose_from == "jets" else detection["u_nose"].notna())
    return detection[keep].assign(event_id=event_of[keep].astype(int))


def nose_stats(detection: pd.DataFrame, event_of: pd.Series, profiles: Profiles | None,
               nose_from: str = "jets") -> pd.DataFrame:
    """Each event's nose statistics, from the slots `nose_from` names: the
    observed peak (`z_nose`/`u_nose`, the boom the detection found) and, where the
    detection carries them, the estimated nose (`z_nose_est`/`u_nose_est`). An
    event with no such slot has NaN statistics and no peak boom; its peak slot is
    its first slot.
    """
    rows = nose_rows(detection, event_of, nose_from)
    for col in ("z_nose_est", "u_nose_est"):
        if col not in rows:
            rows[col] = np.nan
    if profiles is not None and "wd" in profiles.extra and len(rows):
        at = np.searchsorted(profiles.slots, rows.index.to_numpy())
        cols = [profiles.column(int(b)) for b in rows["nose_boom"]]
        rows["wd_nose"] = profiles.extra["wd"][at, cols]
    else:
        rows["wd_nose"] = np.nan
    ids = pd.Index(sorted(event_of.dropna().astype(int).unique()), name="event_id")
    g = rows.groupby("event_id")
    peak = g["u_nose"].idxmax()  # the fastest nose
    first = event_of.dropna().astype(int).reset_index().groupby("event_id")["slot"].min()
    out = pd.DataFrame({
        "n_detected": g.size(),
        "peak_slot": peak,
        "peak_boom": pd.Series(rows.loc[peak.to_numpy(), "nose_boom"].to_numpy(), index=peak.index),
        "z_nose_median": g["z_nose"].median(), "z_nose_min": g["z_nose"].min(), "z_nose_max": g["z_nose"].max(),
        "u_nose_max": g["u_nose"].max(), "u_nose_mean": g["u_nose"].mean(),
        "z_est_median": g["z_nose_est"].median(), "z_est_min": g["z_nose_est"].min(),
        "z_est_max": g["z_nose_est"].max(), "u_est_max": g["u_nose_est"].max(), "u_est_mean": g["u_nose_est"].mean(),
        "falloff_max": g["falloff_abs"].max(), "falloff_mean": g["falloff_abs"].mean(),
        "wd_nose": g["wd_nose"].apply(_circular_mean_deg),
    }).reindex(ids)
    out["n_detected"] = out["n_detected"].fillna(0).astype(np.int64)
    out["peak_slot"] = out["peak_slot"].fillna(first).astype(np.int64)
    out["peak_boom"] = pd.to_numeric(out["peak_boom"], errors="coerce").astype("Int8")
    return out


def detected_slots(events: Events, detection: pd.DataFrame, event_ids=None) -> pd.Series:
    """The event id of every detected (not bridged) slot inside an event, optionally only these events."""
    ids = events.slots["event_id"]
    mask = detection["is_jet"] & ids.notna()
    if event_ids is not None:
        mask &= ids.isin(list(event_ids))
    return ids[mask].astype(int)
