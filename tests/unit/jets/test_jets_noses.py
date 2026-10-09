import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.jets import identify, noses
from ttu_tower.jets.analysis import JetSettings, analyse
from ttu_tower.jets.identify import NoseParams, Profiles

Z = np.array([HEIGHTS[b] for b in BOOMS])


def _parabola(z_top: float, u_top: float, width: float = 3000.0) -> np.ndarray:
    return u_top - (Z - z_top) ** 2 / width


def test_the_vertex_of_three_points_is_exact():
    z = np.array([[74.7, 116.5, 158.2], [10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    u = -(z - np.array([[100.0], [17.0], [0.0]])) ** 2 + np.array([[15.0], [8.0], [0.0]])
    u[2] = [1.0, 2.0, 3.0]  # a straight line has no maximum
    zv, uv = noses.vertex(z, u)
    assert zv[:2] == pytest.approx([100.0, 17.0]) and uv[:2] == pytest.approx([15.0, 8.0])
    assert np.isnan(zv[2]) and np.isnan(uv[2])


def _detection(rows, params=NoseParams()):
    profiles = Profiles(np.arange(100, 100 + len(rows)), tuple(BOOMS), np.array(rows))
    return identify.detect("nose", profiles, params), profiles


def test_each_peak_gets_the_nose_between_its_neighbours():
    rows = [_parabola(100.0, 15.0), _parabola(130.0, 14.0)]
    det, profiles = _detection(rows)
    assert list(det["nose_boom"]) == [8, 8]
    est = noses.estimate(det, profiles)
    assert est["z_nose_est"].to_numpy() == pytest.approx([100.0, 130.0])
    assert est["u_nose_est"].to_numpy() == pytest.approx([15.0, 14.0])
    assert (est["z_nose_est"] > HEIGHTS[7]).all() and (est["z_nose_est"] < HEIGHTS[9]).all()


def test_the_neighbours_are_those_of_the_detections_profile():
    row = _parabola(100.0, 15.0)
    row[8] = np.nan  # boom 9: the full profile can't confirm the peak, but a profile skipping it can
    params = NoseParams(profile_booms=(5, 6, 7, 8, 10), required_booms=(7, 8, 10))
    det, profiles = _detection([row], params)
    assert det["nose_boom"].iloc[0] == 8
    est = noses.estimate(det, profiles, params.profile_booms)
    z, u = HEIGHTS[7], HEIGHTS[10]
    zv, uv = noses.vertex(np.array([[z, HEIGHTS[8], u]]), np.array([[row[6], row[7], row[9]]]))
    assert est["z_nose_est"].iloc[0] == pytest.approx(zv[0]) and est["u_nose_est"].iloc[0] == pytest.approx(uv[0])


def test_slots_without_a_peak_or_a_neighbour_have_no_estimate():
    rising = np.linspace(4.0, 13.0, 10)
    det, profiles = _detection([rising, _parabola(100.0, 15.0)])
    est = noses.estimate(det, profiles)
    assert est["z_nose_est"].isna().iloc[0] and est["z_nose_est"].notna().iloc[1]
    gone = Profiles(profiles.slots, profiles.booms, profiles.ws.copy())
    gone.ws[1, BOOMS.index(9)] = np.nan
    assert noses.estimate(det, gone).isna().all().all()


def test_analysis_describes_events_by_peak_and_estimated_nose():
    rows = [np.linspace(4.0, 13.0, 10)] * 2 + [_parabola(100.0, 15.0)] * 6 + [np.linspace(4.0, 13.0, 10)] * 2
    result = analyse(Profiles(np.arange(500, 510), tuple(BOOMS), np.array(rows)), JetSettings())
    assert {"z_nose_est", "u_nose_est"} <= set(result.detection.columns)
    e = result.events.table.iloc[0]
    assert e.z_nose_median == HEIGHTS[8] and e.z_est_median == pytest.approx(100.0)
    assert e.u_est_max == pytest.approx(15.0) and e.u_nose_max < e.u_est_max
    assert not pd.isna(result.table("candidates").iloc[0].z_est_min)
