import numpy as np
from pytest import approx

from ttu_tower.physics.richardson import bulk_richardson_number


def test_bulk_richardson_number_matches_manual_formula():
    vpt_lower, vpt_upper = 295.0, 297.5
    height_lower, height_upper = 10.0, 47.0
    ws_lower, ws_upper = 4.2, 6.8
    wd_lower, wd_upper = 137.0, 152.5

    ri = bulk_richardson_number(
        vpt_lower, vpt_upper, height_lower, height_upper,
        ws_lower, ws_upper, wd_lower, wd_upper,
    )

    # Independent reference: standard Ri_bulk formula from scratch, using the
    # textbook FROM-bearing -> velocity relationship
    # (u = -speed*sin(from_rad), v = -speed*cos(from_rad)).
    def uv(speed, from_bearing_deg):
        rad = np.deg2rad(from_bearing_deg)
        return -speed * np.sin(rad), -speed * np.cos(rad)

    u_l, v_l = uv(ws_lower, wd_lower)
    u_u, v_u = uv(ws_upper, wd_upper)
    shear_sq = (u_u - u_l) ** 2 + (v_u - v_l) ** 2
    g = 9.80665
    expected = g * (vpt_upper - vpt_lower) * (height_upper - height_lower) / (((vpt_upper + vpt_lower) / 2) * shear_sq)

    assert ri == approx(expected, abs=1e-9)


def test_bulk_richardson_number_scalar_returns_plain_float():
    ri = bulk_richardson_number(295.0, 297.5, 10.0, 47.0, 4.2, 6.8, 137.0, 152.5)
    assert isinstance(ri, float)
    assert not isinstance(ri, np.ndarray)


def test_bulk_richardson_number_zero_shear_is_nan():
    ri = bulk_richardson_number(295.0, 297.5, 10.0, 47.0, 5.0, 5.0, 90.0, 90.0)
    assert isinstance(ri, float)
    assert np.isnan(ri)


def test_bulk_richardson_number_array_inputs_return_array():
    vpt_lower = np.array([295.0, 296.0])
    vpt_upper = np.array([297.5, 298.0])
    ws_lower = np.array([4.2, 5.0])
    ws_upper = np.array([6.8, 5.0])
    wd_lower = np.array([137.0, 90.0])
    wd_upper = np.array([152.5, 90.0])
    ri = bulk_richardson_number(vpt_lower, vpt_upper, 10.0, 47.0, ws_lower, ws_upper, wd_lower, wd_upper)
    assert isinstance(ri, np.ndarray)
    assert np.isnan(ri[1])  # zero shear
