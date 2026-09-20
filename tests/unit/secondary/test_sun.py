import pandas as pd

from ttu_tower.secondary.sun import slot_stats

_TZ = "Etc/GMT+6"


def _slots_df(timestamps):
    return pd.DataFrame({"slot": range(len(timestamps)), "slot_start": pd.DatetimeIndex(timestamps, tz=_TZ)})


def test_midday_is_not_night_and_has_positive_elevation():
    slots = _slots_df(["2024-06-21 12:00:00"])
    out = slot_stats(slots, _TZ)
    assert out["night"].iloc[0] == False  # noqa: E712
    assert out["sun_elevation"].iloc[0] > 30.0


def test_midnight_is_night_and_has_negative_elevation():
    slots = _slots_df(["2024-06-21 00:00:00"])
    out = slot_stats(slots, _TZ)
    assert out["night"].iloc[0] == True  # noqa: E712
    assert out["sun_elevation"].iloc[0] < 0.0


def test_summer_midday_elevation_higher_than_winter_midday():
    slots = _slots_df(["2024-06-21 12:00:00", "2024-12-21 12:00:00"])
    out = slot_stats(slots, _TZ)
    summer, winter = out["sun_elevation"].iloc[0], out["sun_elevation"].iloc[1]
    assert summer > winter


def test_sunrise_sunset_cached_per_utc_date_gives_consistent_night_flags():
    # Two slots on the same UTC date, one clearly day and one clearly night.
    slots = _slots_df(["2024-03-15 13:00:00", "2024-03-15 02:00:00"])
    out = slot_stats(slots, _TZ)
    assert out["night"].iloc[0] == False  # noqa: E712
    assert out["night"].iloc[1] == True  # noqa: E712


def test_output_has_one_row_per_slot_in_order():
    slots = _slots_df(["2024-01-01 00:00:00", "2024-01-01 00:10:00", "2024-01-01 00:20:00"])
    out = slot_stats(slots, _TZ)
    assert list(out["slot"]) == [0, 1, 2]
    assert len(out) == 3
