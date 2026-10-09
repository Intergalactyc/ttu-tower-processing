import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import BOOMS
from ttu_tower.jets.events import EventParams, detected_slots, fill_gaps, find_events
from ttu_tower.jets.identify import NOSE_STATUSES, Profiles
from ttu_tower.timegrid import slot_to_time

SLOT0 = 728_000
_CODES = {"J": "jet", "r": "weak_falloff", "n": "no_data"}


def _detection(pattern: str, z=116.5, u=14.0):
    """One slot per character: J jet, r rejected with data, n no data."""
    status = [_CODES[c] for c in pattern]
    jet = np.array([s == "jet" for s in status])
    has = np.array([s != "no_data" for s in status])
    n = len(pattern)
    return pd.DataFrame({
        "is_jet": jet,
        "status": pd.Categorical(status, categories=NOSE_STATUSES),
        "nose_boom": pd.Series(8, index=range(n), dtype="Int8").where(has).array,
        "z_nose": np.where(has, z, np.nan), "u_nose": np.where(has, u, np.nan),
        "z_min_above": np.where(has, 200.0, np.nan), "u_min_above": np.where(has, u - 3, np.nan),
        "falloff_abs": np.where(has, 3.0, np.nan), "falloff_rel": np.where(has, 3.0 / u, np.nan),
    }, index=pd.Index(np.arange(SLOT0, SLOT0 + n), name="slot"))


def _events(pattern, **params):
    return find_events(_detection(pattern), EventParams(**params))


def test_a_rejected_slot_is_bridged_and_counts_toward_the_length():
    ev = _events("JJJrJJ")
    assert len(ev.table) == 1
    row = ev.table.iloc[0]
    assert (row.n_slots, row.n_filled, row.n_detected, row.duration_min) == (6, 1, 5, 60)
    assert ev.slots["filled"].tolist() == [False, False, False, True, False, False]


def test_missing_data_is_never_bridged():
    assert len(_events("JJJnJJJ").table) == 0
    ev = _events("JJJnJJJ", min_event_slots=3)
    assert len(ev.table) == 2 and not ev.slots["filled"].any()


def test_only_gaps_up_to_the_limit_are_bridged():
    assert len(_events("JJJrrJJJ").table) == 0
    assert _events("JJJrrJJJ", max_gap_slots=2).table.iloc[0].n_filled == 2
    assert len(_events("JJJrJJ", max_gap_slots=0).table) == 0
    assert not fill_gaps(np.array([False, True, False]), np.ones(3, bool), 5).any()


def test_short_runs_are_dropped_and_events_touching_the_ends_are_truncated():
    ev = _events("JJJJJrrrJJJJJJ" + "r" + "JJJJJJ", max_gap_slots=1)
    assert len(ev.table) == 1  # the first run (5) is too short; the last two are bridged into one
    assert ev.table.iloc[0].n_slots == 13 and ev.table.iloc[0].truncated
    assert ev.table.iloc[0].start_slot == SLOT0 + 8 and ev.table.iloc[0].end_slot == SLOT0 + 20
    middle = _events("rJJJJJJr")
    assert not middle.table.iloc[0].truncated


def test_times_and_ids_follow_the_slots():
    ev = find_events(_detection("rJJJJJJrrrJJJJJJ"), tz="UTC")
    assert list(ev.table.index) == [0, 1]
    first = ev.table.iloc[0]
    assert first.start_time == slot_to_time(SLOT0 + 1).tz_convert("UTC")
    assert first.end_time == slot_to_time(SLOT0 + 7).tz_convert("UTC")
    assert str(first.start_time.tz) == "UTC"
    ids = ev.slots["event_id"]
    assert ids.iloc[0] is pd.NA and ids.iloc[1] == 0 and ids.iloc[-1] == 1


def test_statistics_come_from_detected_slots_only():
    det = _detection("JJJrJJ")
    det.loc[SLOT0 + 3, ["z_nose", "u_nose", "falloff_abs"]] = [158.2, 30.0, 0.5]  # the bridged slot's own candidate
    det.loc[SLOT0, "u_nose"] = 16.0
    row = find_events(det).table.iloc[0]
    assert row.z_nose_max == 116.5 and row.u_nose_max == 16.0 and row.falloff_mean == 3.0
    assert row.u_nose_mean == pytest.approx((16.0 + 4 * 14.0) / 5)
    assert row.peak_slot == SLOT0 and row.peak_boom == 8  # the fastest detected slot, not the bridged one
    assert list(detected_slots(find_events(det), det).index) == [SLOT0 + i for i in (0, 1, 2, 4, 5)]


def test_the_nose_direction_is_a_circular_mean():
    det = _detection("JJJJJJ")
    wd = np.full((6, 10), np.nan)
    wd[:, BOOMS.index(8)] = [350.0, 10.0, 350.0, 10.0, 355.0, 5.0]
    profiles = Profiles(det.index.to_numpy(), tuple(BOOMS), np.zeros((6, 10)), extra={"wd": wd})
    row = find_events(det, profiles=profiles).table.iloc[0]
    assert min(row.wd_nose, 360.0 - row.wd_nose) == pytest.approx(0.0, abs=1e-9)


def test_no_jets_gives_an_empty_table_with_its_columns():
    ev = _events("rrrnnn")
    assert ev.table.empty and {"start_time", "z_nose_median", "wd_nose"} <= set(ev.table.columns)
    assert ev.slots["event_id"].isna().all()


def test_invalid_event_parameters_are_refused():
    with pytest.raises(ValueError):
        _events("JJJJJJ", min_event_slots=0)



def test_missing_data_is_bridged_only_when_asked():
    assert len(_events("JJJnJJJ").table) == 0
    ev = _events("JJJnJJJ", bridge_missing=True)
    assert len(ev.table) == 1 and ev.table.iloc[0].n_filled == 1 and ev.table.iloc[0].n_slots == 7
    assert ev.slots.loc[SLOT0 + 3, "filled"]
    assert len(_events("JJJnrJJ", max_gap_slots=2, bridge_missing=True).table) == 1
    assert len(_events("JJJnnJJJ", bridge_missing=True).table) == 0  # still no longer than the limit
    assert EventParams(bridge_missing="yes").validate()
