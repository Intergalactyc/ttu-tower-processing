import numpy as np
import pytest

from ttu_tower.viewer import hodograph
from ttu_tower.viewer.hodograph import MESONET, UV


def test_a_slots_wind_becomes_blows_toward_components():
    w = hodograph._uv(np.array([1, 2, 3]), np.array([5.0, 2.0, np.nan]), np.array([270.0, 0.0, 90.0]))
    assert list(w.slots) == [1, 2]  # no speed: left out
    assert w.u == pytest.approx([5.0, 0.0], abs=1e-12) and w.v == pytest.approx([0.0, -2.0], abs=1e-12)


def test_the_summary_averages_vectors_bottom_to_top_with_the_mesonet_last():
    uvs = {10: UV(np.array([1, 2]), np.array([4.0, 4.0]), np.array([0.0, 2.0])),
           MESONET: UV(np.array([2]), np.array([1.0]), np.array([1.0])),
           1: UV(np.array([1, 2, 3]), np.array([1.0, -1.0, 0.0]), np.array([0.0, 0.0, 3.0]))}
    rows = hodograph.summary(uvs)
    assert list(rows["member"]) == [1, 10, MESONET] and list(rows["n"]) == [3, 2, 1]
    low = rows.iloc[0]
    assert (low["u"], low["v"]) == pytest.approx((0.0, 1.0)) and low["from"] == pytest.approx(180.0)
    assert low["steadiness"] == pytest.approx(1.0 / (5.0 / 3.0))  # |mean| 1 over a mean speed of 5/3
    assert list(hodograph.common_slots(uvs, [1, 10])) == [1, 2]
    assert hodograph.height(MESONET) == 10.0


def test_jet_profiles_keep_the_booms_mostly_present_and_the_slots_all_of_them_have():
    ws = np.array([[5.0, 6.0, np.nan], [5.0, np.nan, np.nan], [5.0, 6.0, 7.0], [5.0, 6.0, np.nan]])
    wd = np.full(ws.shape, 270.0)  # from the west: toward east
    uvs = hodograph.from_profiles(np.array([10, 11, 12, 13]), (1, 2, 3), ws, wd)
    assert list(uvs) == [1, 2]  # boom 3 has wind in only 1 of 4 slots
    assert list(uvs[1].slots) == [10, 12, 13] and uvs[2].u == pytest.approx([6.0, 6.0, 6.0])
    assert hodograph.from_profiles(np.array([], dtype=np.int64), (1,), np.empty((0, 1)), np.empty((0, 1))) == {}
