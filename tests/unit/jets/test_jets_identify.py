import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.jets import identify
from ttu_tower.jets.identify import NoseParams, Profiles

from jets_reference import reference_nose

NAN = np.nan
BASE = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]  # rising to the top: no jet


def _profiles(*rows, slot0=1000):
    return Profiles(np.arange(slot0, slot0 + len(rows)), tuple(BOOMS), np.array(rows, dtype=float))


def _with(**speeds):
    """BASE with some booms changed, e.g. _with(b8=15.0, b9=12.0)."""
    row = list(BASE)
    for name, v in speeds.items():
        row[int(name[1:]) - 1] = v
    return row


def _nose(*rows, **params):
    return identify.detect("nose", _profiles(*rows), NoseParams(**params))


def test_a_jet_is_a_local_maximum_with_falloff_above_it():
    out = _nose(_with(b8=15.0, b9=12.5, b10=12.0))
    r = out.iloc[0]
    assert r.is_jet and r.status == "jet" and r.nose_boom == 8
    assert r.z_nose == HEIGHTS[8] and r.u_nose == 15.0
    assert r.u_min_above == 12.0 and r.z_min_above == HEIGHTS[10]
    assert r.falloff_abs == pytest.approx(3.0) and r.falloff_rel == pytest.approx(0.2)


def test_a_peak_at_the_top_boom_is_never_a_nose():
    out = _nose(_with(b10=20.0))
    assert out.iloc[0].status == "no_peak" and pd.isna(out.iloc[0].nose_boom)


def test_a_dip_below_a_faster_top_is_not_a_jet():
    # the old method flagged this: speed falls between 74.7 and 116.5 m, but the top is fastest
    out = _nose(_with(b7=12.0, b8=10.5, b9=13.0, b10=14.0))
    r = out.iloc[0]
    assert not r.is_jet and r.status == "weak_falloff" and r.nose_boom == 7
    assert r.falloff_abs == pytest.approx(1.5)


def test_a_missing_required_boom_means_no_data():
    out = _nose(_with(b8=15.0, b9=12.0, b10=NAN))
    assert out.iloc[0].status == "no_data" and not out.iloc[0].is_jet


def test_a_missing_neighbour_drops_the_candidate():
    out = _nose(_with(b6=NAN, b7=15.0, b8=12.0, b9=12.5, b10=12.8))
    assert out.iloc[0].status == "no_peak"
    out = _nose(_with(b6=9.0, b7=15.0, b8=12.0, b9=12.5, b10=12.8))
    assert out.iloc[0].status == "jet" and out.iloc[0].nose_boom == 7


def test_speed_and_falloff_thresholds_are_inclusive():
    exact = _with(b7=9.0, b8=10.0, b9=8.0, b10=8.5)
    assert _nose(exact).iloc[0].status == "jet"
    assert _nose(_with(b7=9.0, b8=9.99, b9=7.5, b10=8.0)).iloc[0].status == "too_slow"
    assert _nose(_with(b8=12.0, b9=10.01, b10=10.5)).iloc[0].status == "weak_falloff"


def test_relative_and_combined_falloff():
    row = _with(b8=16.0, b9=13.0, b10=13.5)  # falloff 3 m/s = 18.75 %
    assert _nose(row, min_falloff_abs=None, min_falloff_rel=0.15).iloc[0].is_jet
    assert not _nose(row, min_falloff_abs=None, min_falloff_rel=0.25).iloc[0].is_jet
    assert not _nose(row, min_falloff_abs=2.0, min_falloff_rel=0.25).iloc[0].is_jet
    assert not _nose(row, min_falloff_abs=4.0, min_falloff_rel=0.15).iloc[0].is_jet
    assert _nose(row, min_falloff_abs=2.0, min_falloff_rel=0.15).iloc[0].is_jet


def test_the_fastest_qualifying_candidate_is_the_nose_and_ties_go_low():
    two = [4.0, 5.0, 6.0, 7.0, 8.0, 14.0, 11.0, 16.0, 13.0, 13.5]
    assert _nose(two).iloc[0].nose_boom == 8
    tied = [4.0, 5.0, 6.0, 7.0, 8.0, 15.0, 12.0, 15.0, 12.0, 12.5]
    assert _nose(tied).iloc[0].nose_boom == 6


def test_rejected_slots_report_their_best_candidate():
    slow = _nose([4.0, 5.0, 9.0, 7.0, 8.0, 9.5, 8.0, 9.0, 8.5, 8.9])  # local maxima at 3, 6, 8; all < 10
    assert slow.iloc[0].status == "too_slow" and slow.iloc[0].nose_boom == 6
    weak = _nose([4.0, 5.0, 6.0, 7.0, 8.0, 14.0, 13.0, 12.5, 12.5, 13.0])  # 6: 14 -> min above 12.5
    assert weak.iloc[0].status == "weak_falloff" and weak.iloc[0].nose_boom == 6
    assert weak.iloc[0].falloff_abs == pytest.approx(1.5) and not weak.iloc[0].is_jet
    weak2 = _nose([4.0, 5.0, 6.0, 7.0, 8.0, 14.0, 13.0, 14.5, 13.0, 13.5])  # two fast; falloffs 1.0 and 1.5
    assert weak2.iloc[0].status == "weak_falloff" and weak2.iloc[0].nose_boom == 8


def test_the_profile_can_be_narrowed():
    row = _with(b5=16.0, b8=15.0, b9=12.0, b10=12.5)  # boom 5 is faster but outside the profile
    out = _nose(row, profile_booms=(6, 7, 8, 9, 10))
    assert out.iloc[0].nose_boom == 8


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError, match="falloff"):
        identify.detect("nose", _profiles(BASE), NoseParams(min_falloff_abs=None, min_falloff_rel=None))
    with pytest.raises(ValueError, match="required booms"):
        identify.detect("nose", _profiles(BASE), NoseParams(profile_booms=(1, 2, 3, 4), required_booms=(9,)))
    assert NoseParams(min_nose_speed=-1.0).validate()
    assert NoseParams(profile_booms=(1, 11, 3)).validate()
    assert not NoseParams().validate()


def test_the_registry_describes_the_parameters_for_a_form():
    algorithm = identify.ALGORITHMS["nose"]
    assert algorithm.params is NoseParams and algorithm.description
    fields = {f["name"]: f for f in identify.describe(NoseParams)}
    assert fields["min_falloff_rel"]["kind"] == "optional_float" and fields["min_falloff_rel"]["suggested"] == 0.25
    assert fields["required_booms"]["default"] == (7, 8, 9, 10)
    params = NoseParams(required_booms=(8, 9, 10), min_falloff_rel=0.3)
    saved = identify.to_dict(params)
    assert saved["required_booms"] == [8, 9, 10]
    assert identify.from_dict(NoseParams, saved | {"stale_key": 1}) == params
    assert identify.from_dict(NoseParams, {}) == NoseParams()


def test_profiles_need_a_consecutive_grid_and_frames_are_filled_out():
    with pytest.raises(ValueError, match="consecutive"):
        Profiles(np.array([1, 2, 4]), tuple(BOOMS), np.zeros((3, 10)))
    frame = pd.DataFrame([BASE, BASE], index=[10, 13], columns=list(BOOMS))
    p = Profiles.from_frame(frame, {"wd": frame * 10})
    assert list(p.slots) == [10, 11, 12, 13] and np.isnan(p.ws[1:3]).all() and p.extra["wd"].shape == (4, 10)


# --- against a literal reading of the rules ----------------------------------------------------

@pytest.mark.parametrize("params", [
    NoseParams(),
    NoseParams(min_nose_speed=7.0, min_falloff_abs=1.0, min_falloff_rel=0.1),
    NoseParams(min_falloff_abs=None, min_falloff_rel=0.15, required_booms=(8, 10)),
    NoseParams(profile_booms=(3, 5, 6, 7, 8, 9, 10), required_booms=(5, 9)),
])
def test_matches_a_literal_reading_of_the_rules_on_random_profiles(params):
    rng = np.random.default_rng(3)
    n = 3000
    ws = np.round(rng.uniform(3.0, 18.0, size=(n, 10)) * 2) / 2  # half-m/s steps, so ties happen
    ws[rng.random((n, 10)) < 0.08] = np.nan
    out = identify.detect("nose", _profiles(*ws), params)
    for i in range(n):
        status, boom = reference_nose(ws[i], params)
        row = out.iloc[i]
        assert row.status == status, (i, ws[i])
        assert (pd.isna(row.nose_boom) and boom is None) or row.nose_boom == boom, (i, ws[i])
