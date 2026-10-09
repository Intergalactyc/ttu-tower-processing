import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import BOOMS
from ttu_tower.jets.analysis import JetSettings, analyse, sweep, sweep_values
from ttu_tower.jets.events import EventParams
from ttu_tower.jets.filters import JetFilters, vpt_column
from ttu_tower.jets.identify import NoseParams, Profiles

JET = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 15.0, 12.0, 12.5]  # nose at boom 8, falloff 3 m/s
RISING = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]


def _profiles(pattern: str, slot0=1000):
    """J jet, r rejected, n no data (boom 10 missing)."""
    rows = []
    for c in pattern:
        row = list(JET if c == "J" else RISING)
        if c == "n":
            row[9] = np.nan
        rows.append(row)
    return Profiles(np.arange(slot0, slot0 + len(pattern)), tuple(BOOMS), np.array(rows))


def test_counts_follow_each_rule_and_each_filter():
    profiles = _profiles("JJJrJJJ" + "rrr" + "JJJJJJ" + "n" + "JJJ")
    context = pd.DataFrame({vpt_column(2): 290.0, vpt_column(10): 291.0, "night": 1.0}, index=profiles.slots)
    context.loc[profiles.slots[10]:, vpt_column(10)] = np.nan  # the second event has no ΔVPT
    settings = JetSettings(filters=JetFilters(min_inversion=0.5))
    result = analyse(profiles, settings, context)
    counts = result.counts()
    assert counts["status"] == {"no_data": 1, "no_peak": 4, "too_slow": 0, "weak_falloff": 0, "jet": 15}
    assert (counts["jet_slots"], counts["events"], counts["passed"]) == (15, 2, 1)
    assert counts["failed"] == {"inversion": 1} and counts["no_value"] == {"inversion": 1}
    assert list(result.event_slots()) == list(profiles.slots[:7])
    assert len(result.event_slots(passed_only=False)) == 13
    assert list(result.event_slots(event_ids=[1], passed_only=False)) == list(profiles.slots[10:16])


def test_invalid_settings_are_refused():
    with pytest.raises(ValueError, match="falloff"):
        analyse(_profiles("JJJJJJ"), JetSettings(params=NoseParams(min_falloff_abs=None)))


def test_settings_round_trip_and_keep_other_algorithms_parameters():
    s = JetSettings(params=NoseParams(min_nose_speed=12.0, required_booms=(8, 9, 10)),
                    events=EventParams(max_gap_slots=2), filters=JetFilters(night_only=True, rib_max=0.25))
    saved = s.to_dict({"params": {"other": {"x": 1}}})
    assert saved["params"]["other"] == {"x": 1}
    assert JetSettings.from_dict(saved) == s


def test_stale_or_broken_saves_fall_back_to_defaults():
    empty = JetSettings.from_dict({})
    assert empty.algorithm_params() == NoseParams() and empty.events == EventParams()
    assert empty.filters == JetFilters()
    assert JetSettings.from_dict({"algorithm": "gone"}).algorithm == "nose"
    assert JetSettings.from_dict({"params": {"nose": {"min_nose_speed": 9.0, "stale": 1}}}).params == \
        NoseParams(min_nose_speed=9.0)
    assert JetSettings.from_dict({"filters": {"rib_booms": "nonsense"}}).filters.validate()


def test_counts_events_and_slots_within_a_range():
    profiles = _profiles("JJJJJJ" + "rrrr" + "JJJJJJJ")
    result = analyse(profiles, JetSettings())
    slots = profiles.slots
    assert result.counts()["events"] == 2
    first = (int(slots[0]), int(slots[8]))  # the second event starts after it
    c = result.counts(first)
    assert (c["events"], c["jet_slots"]) == (1, 6)
    assert sum(c["status"].values()) == 8
    assert list(result.events_in(first).index) == [0]
    detected = result.detected_in([1])
    assert list(detected.index) == list(slots[10:17]) and (detected["event_id"] == 1).all()


def test_a_sweep_counts_jets_as_one_setting_varies():
    profiles = _profiles("JJJJJJ" + "rrrr" + "JJJJJJJ")
    settings = JetSettings()
    speeds = sweep(profiles, settings, "params", "min_nose_speed", [10.0, 15.0, 16.0])
    assert list(speeds["jet_slots"]) == [13, 13, 0] and list(speeds["events"]) == [2, 2, 0]
    lengths = sweep(profiles, settings, "events", "min_event_slots", [6, 7, 8])
    assert list(lengths["events"]) == [2, 1, 0] and list(lengths["jet_slots"]) == [13, 13, 13]
    switched_on = sweep(profiles, settings, "params", "min_falloff_rel", [0.1, 0.3])
    assert list(switched_on["jet_slots"]) == [13, 0]  # falloff 3 m/s of 15 m/s: 20 %


def test_sweep_values_span_the_current_value_within_the_limits():
    speed = sweep_values({"kind": "float", "lo": 0.0, "hi": 50.0}, 10.0)
    assert speed[0] == 0.0 and speed[-1] == 25.0 and len(speed) == 26
    rel = sweep_values({"kind": "optional_float", "lo": 0.0, "hi": 1.0, "suggested": 0.25}, None)
    assert rel[-1] == pytest.approx(0.625)
    gaps = sweep_values({"kind": "int", "lo": 0, "hi": 36}, 1)
    assert list(gaps) == list(range(0, 7))
