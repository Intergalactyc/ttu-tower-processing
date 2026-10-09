import numpy as np
import pandas as pd
import pytest

from ttu_tower.jets.events import find_events
from ttu_tower.jets.filters import (
    JetFilters, apply_filters, context_needs, event_quantities, rib_column, vpt_column,
)
from ttu_tower.jets.identify import NOSE_STATUSES
from ttu_tower.timegrid import time_to_slot

LOCAL = "Etc/GMT+6"


def _slot(text):
    return int(time_to_slot(pd.Timestamp(text, tz=LOCAL)))


def _detection(start_slot, pattern):
    """One slot per character from start_slot: J jet, r rejected with data."""
    status = ["jet" if c == "J" else "weak_falloff" for c in pattern]
    n = len(pattern)
    return pd.DataFrame({
        "is_jet": np.array([c == "J" for c in pattern]),
        "status": pd.Categorical(status, categories=NOSE_STATUSES),
        "nose_boom": pd.array([8] * n, dtype="Int8"),
        **{c: np.full(n, 1.0) for c in ("z_nose", "u_nose", "z_min_above", "u_min_above", "falloff_abs",
                                        "falloff_rel")},
    }, index=pd.Index(np.arange(start_slot, start_slot + n), name="slot"))


def _three_events():
    """Events starting 23:00, 03:00 and 12:00 local, one rejected slot inside the first."""
    first = _slot("2014-01-10 23:00")
    det = _detection(first, "JJJrJJ" + "r" * 18 + "JJJJJJ" + "r" * 48 + "JJJJJJ")
    events = find_events(det)
    assert len(events.table) == 3
    return det, events


def _context(det, dvpt=None, rib=None, night=None):
    ctx = pd.DataFrame(index=det.index)
    ctx[vpt_column(2)] = 290.0
    ctx[vpt_column(10)] = 290.0 + (np.zeros(len(det)) if dvpt is None else dvpt)
    ctx[rib_column(2, 4)] = 0.0 if rib is None else rib
    ctx["night"] = 1.0 if night is None else night
    return ctx


def test_a_time_window_may_wrap_midnight():
    det, events = _three_events()
    q = event_quantities(events, det, None, JetFilters())
    assert list(q["start_minute"]) == [23 * 60, 3 * 60, 12 * 60]
    night_window = apply_filters(q, JetFilters(time_window=("18:00", "09:00")))
    assert list(night_window["passed"]) == [True, True, False]
    day = apply_filters(q, JetFilters(time_window=("09:00", "17:00")))
    assert list(day["time_of_day"]) == [False, False, True]


def test_the_time_window_is_read_in_the_events_zone():
    det, _ = _three_events()
    q = event_quantities(find_events(det, tz="UTC"), det, None, JetFilters())
    assert q["start_minute"].iloc[0] == 5 * 60  # 23:00 at UTC-6


def test_night_uses_the_start_slot():
    det, events = _three_events()
    night = pd.Series(0.0, index=det.index)
    night.loc[events.table["start_slot"].iloc[1]] = 1.0
    q = event_quantities(events, det, _context(det, night=night), JetFilters())
    assert list(apply_filters(q, JetFilters(night_only=True))["night"]) == [False, True, False]


def test_inversion_is_the_mean_over_every_slot_of_the_event():
    det, events = _three_events()
    dvpt = pd.Series(2.0, index=det.index)
    dvpt.iloc[3] = 8.0  # the bridged slot of the first event counts too
    dvpt.iloc[24:30] = np.nan  # the second event has no value
    dvpt.iloc[78:81] = 0.5
    ctx = _context(det, dvpt=dvpt.to_numpy())
    q = event_quantities(events, det, ctx, JetFilters())
    assert q["dvpt_mean"].iloc[0] == pytest.approx(3.0) and np.isnan(q["dvpt_mean"].iloc[1])
    assert q["dvpt_mean"].iloc[2] == pytest.approx(1.25)
    passed = apply_filters(q, JetFilters(min_inversion=1.5))
    assert list(passed["inversion"]) == [True, False, False]


def test_inversion_booms_are_ordered_by_height():
    det, events = _three_events()
    ctx = _context(det, dvpt=np.full(len(det), 3.0))
    q = event_quantities(events, det, ctx, JetFilters(inversion_booms=(10, 2)))
    assert (q["dvpt_mean"] == 3.0).all()
    assert context_needs(JetFilters(inversion_booms=(10, 2), rib_booms=(4, 2))) == ((2, 10), ((2, 4),))


def test_richardson_ranges_may_be_open():
    det, events = _three_events()
    rib = np.concatenate([np.full(24, 0.1), np.full(54, 0.5), np.full(len(det) - 78, -0.2)])
    q = event_quantities(events, det, _context(det, rib=rib), JetFilters())
    assert list(q["rib_median"]) == pytest.approx([0.1, 0.5, -0.2])
    assert list(apply_filters(q, JetFilters(rib_max=0.25))["richardson"]) == [True, False, True]
    assert list(apply_filters(q, JetFilters(rib_min=0.0, rib_max=0.25))["richardson"]) == [True, False, False]
    assert list(apply_filters(q, JetFilters(rib_min=0.3))["richardson"]) == [False, True, False]


def test_without_context_an_enabled_filter_fails_and_a_disabled_one_passes():
    det, events = _three_events()
    q = event_quantities(events, det, None, JetFilters())
    assert apply_filters(q, JetFilters())["passed"].all()
    assert not apply_filters(q, JetFilters(min_inversion=0.0))["passed"].any()


def test_filter_settings_validate_and_round_trip():
    assert JetFilters(time_window=("25:00", "06:00")).validate()
    assert JetFilters(rib_booms=(3, 3)).validate()
    assert JetFilters(rib_min=1.0, rib_max=0.0).validate()
    with pytest.raises(ValueError):
        apply_filters(pd.DataFrame(columns=["start_minute"]), JetFilters(inversion_booms=(2, 12)))
    f = JetFilters(time_window=("18:00", "09:00"), night_only=True, min_inversion=1.0, rib_max=0.25)
    assert JetFilters.from_dict(f.to_dict() | {"stale": 1}) == f
    assert f.enabled() == {"time_of_day": True, "night": True, "inversion": True, "richardson": True}
