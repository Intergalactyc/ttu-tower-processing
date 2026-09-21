import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import HEIGHTS, SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.physics.earth import local_gravity
from ttu_tower.physics.richardson import bulk_richardson_number
from ttu_tower.tertiary.multiboom import multiboom

_SLOT = 5


def _none_row(slot, boom, variable, value, stat="mean"):
    return {"slot": slot, "boom": boom, "variant": "none", "variable": variable, "stat": stat, "value": value}


def test_rib_and_lapse_vpt_match_direct_computation():
    boom_final = pd.DataFrame([
        _none_row(_SLOT, 1, "ue", 5.0), _none_row(_SLOT, 1, "vn", 0.0), _none_row(_SLOT, 1, "vpt", 290.0, stat=None),
        _none_row(_SLOT, 4, "ue", 6.0), _none_row(_SLOT, 4, "vn", 1.0), _none_row(_SLOT, 4, "vpt", 292.0, stat=None),
        _none_row(_SLOT, 1, "wd", 270.0), _none_row(_SLOT, 4, "wd", 260.0),
    ])
    result = multiboom(boom_final, [1, 4], veer_reference_boom=4)

    rib_row = result[(result["boom"] == 1) & (result["boom2"] == 4) & (result["variable"] == "rib")]
    expected_g = local_gravity(SITE_LATITUDE, SITE_ELEVATION + (HEIGHTS[1] + HEIGHTS[4]) / 2)
    expected_rib = bulk_richardson_number(
        290.0, 292.0, HEIGHTS[1], HEIGHTS[4], 5.0, 6.0, 0.0, 1.0, components=True, gravity=expected_g,
    )
    assert rib_row["value"].iloc[0] == pytest.approx(expected_rib)

    lapse_row = result[(result["boom"] == 1) & (result["boom2"] == 4) & (result["variable"] == "lapse_vpt")]
    expected_lapse = (292.0 - 290.0) / (HEIGHTS[4] - HEIGHTS[1])
    assert lapse_row["value"].iloc[0] == pytest.approx(expected_lapse)


def test_veer_matches_signed_angular_distance_to_reference():
    boom_final = pd.DataFrame([
        _none_row(_SLOT, 1, "wd", 100.0), _none_row(_SLOT, 4, "wd", 90.0),
        _none_row(_SLOT, 1, "ue", 1.0), _none_row(_SLOT, 1, "vn", 1.0), _none_row(_SLOT, 1, "vpt", 290.0, stat=None),
        _none_row(_SLOT, 4, "ue", 1.0), _none_row(_SLOT, 4, "vn", 1.0), _none_row(_SLOT, 4, "vpt", 290.0, stat=None),
    ])
    result = multiboom(boom_final, [1, 4], veer_reference_boom=4)

    veer_1 = result[(result["boom"] == 1) & (result["boom2"] == 4) & (result["variable"] == "veer")]
    assert veer_1["value"].iloc[0] == pytest.approx(10.0)

    # The reference boom's own row: boom == boom2, and veer is 0 (not NaN),
    # since it goes through the same call comparing wd to itself.
    veer_ref = result[(result["boom"] == 4) & (result["boom2"] == 4) & (result["variable"] == "veer")]
    assert len(veer_ref) == 1
    assert veer_ref["value"].iloc[0] == pytest.approx(0.0)


def test_filtered_reference_boom_gives_nan_veer_everywhere_including_its_own_row():
    boom_final = pd.DataFrame([
        _none_row(_SLOT, 1, "wd", 100.0), _none_row(_SLOT, 4, "wd", np.nan),
        _none_row(_SLOT, 1, "ue", 1.0), _none_row(_SLOT, 1, "vn", 1.0), _none_row(_SLOT, 1, "vpt", 290.0, stat=None),
        _none_row(_SLOT, 4, "ue", np.nan), _none_row(_SLOT, 4, "vn", np.nan), _none_row(_SLOT, 4, "vpt", np.nan, stat=None),
    ])
    result = multiboom(boom_final, [1, 4], veer_reference_boom=4)

    veer = result[result["variable"] == "veer"]
    assert veer["value"].isna().all()


def test_filtered_boom_gives_nan_rib_for_every_pair_containing_it():
    boom_final = pd.DataFrame([
        _none_row(_SLOT, 1, "ue", np.nan), _none_row(_SLOT, 1, "vn", np.nan), _none_row(_SLOT, 1, "vpt", np.nan, stat=None),
        _none_row(_SLOT, 2, "ue", 5.0), _none_row(_SLOT, 2, "vn", 0.0), _none_row(_SLOT, 2, "vpt", 290.0, stat=None),
        _none_row(_SLOT, 3, "ue", 5.5), _none_row(_SLOT, 3, "vn", 0.0), _none_row(_SLOT, 3, "vpt", 291.0, stat=None),
    ])
    result = multiboom(boom_final, [1, 2, 3], veer_reference_boom=2)

    involving_1 = result[((result["boom"] == 1) | (result["boom2"] == 1)) & result["variable"].isin(["rib", "lapse_vpt"])]
    assert involving_1["value"].isna().all()
    not_involving_1 = result[(result["boom"] == 2) & (result["boom2"] == 3) & (result["variable"] == "rib")]
    assert not not_involving_1["value"].isna().any()
