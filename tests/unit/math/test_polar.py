import numpy as np
from pytest import approx

from ttu_tower.math.polar import (
    bearing_to_vector,
    polar_wind,
    rotate_covariance_streamwise,
    series_signed_angular_distance,
    signed_angular_distance,
    streamwise_angle,
    vector_to_bearing,
    wind_components,
    yamartino_std,
)


def test_polar_wind_toward_bearing_axis_aligned():
    assert polar_wind(0.0, 1.0)[1] == approx(0.0)  # points N
    assert polar_wind(1.0, 0.0)[1] == approx(90.0)  # points E
    assert polar_wind(0.0, -1.0)[1] == approx(180.0)  # points S
    assert polar_wind(-1.0, 0.0)[1] == approx(270.0)  # points W


def test_wind_components_polar_wind_round_trip():
    rng = np.random.default_rng(0)
    speed = rng.uniform(0.1, 20, 25)
    toward = rng.uniform(0, 360, 25)
    u, v = wind_components(speed, toward)
    speed_out, toward_out = polar_wind(u, v)
    assert np.allclose(speed_out, speed, atol=1e-9)
    assert np.allclose((toward_out - toward + 180) % 360 - 180, 0, atol=1e-7)


def test_wind_components_zero_speed_ignores_direction():
    u, v = wind_components(0.0, np.nan)
    assert (u, v) == (0.0, 0.0)


def test_bearing_to_vector_from_bearing_axis_aligned():
    # A "south wind" (FROM 180) points due north; a "west wind" (FROM 270)
    # points due east - the two axis-aligned cases that pin down the sign.
    u, v = bearing_to_vector(1.0, 180.0)
    assert (u, v) == approx((0.0, 1.0), abs=1e-9)
    u, v = bearing_to_vector(1.0, 270.0)
    assert (u, v) == approx((1.0, 0.0), abs=1e-9)


def test_vector_to_bearing_matches_ground_truth():
    assert vector_to_bearing(0.0, 1.0)[1] == approx(180.0)  # points N -> FROM S
    assert vector_to_bearing(1.0, 0.0)[1] == approx(270.0)  # points E -> FROM W
    assert vector_to_bearing(0.0, -1.0)[1] == approx(0.0)  # points S -> FROM N
    assert vector_to_bearing(-1.0, 0.0)[1] == approx(90.0)  # points W -> FROM E


def test_bearing_to_vector_vector_to_bearing_round_trip():
    rng = np.random.default_rng(1)
    speed = rng.uniform(0.1, 20, 25)
    from_bearing = rng.uniform(0, 360, 25)
    u, v = bearing_to_vector(speed, from_bearing)
    speed_out, from_out = vector_to_bearing(u, v)
    assert np.allclose(speed_out, speed, atol=1e-9)
    assert np.allclose((from_out - from_bearing + 180) % 360 - 180, 0, atol=1e-7)


def test_signed_angular_distance_matches_series_version():
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, 360, 100)
    phi = rng.uniform(0, 360, 100)
    series = series_signed_angular_distance(theta, phi)
    scalar = np.array([signed_angular_distance(t, p) for t, p in zip(theta, phi)])
    assert np.allclose(series, scalar, atol=1e-9)


def test_signed_angular_distance_sign_convention():
    # theta clockwise of phi -> positive
    assert signed_angular_distance(10.0, 350.0) == approx(20.0)
    assert signed_angular_distance(350.0, 10.0) == approx(-20.0)


def test_yamartino_std_matches_direct_formula():
    s_bar, c_bar = 0.3, 0.7
    epsilon = np.sqrt(max(0.0, 1 - s_bar**2 - c_bar**2))
    expected_rad = np.arcsin(epsilon) * (1 + (2 / np.sqrt(3) - 1) * epsilon**3)
    assert yamartino_std(s_bar, c_bar, degrees=False) == approx(expected_rad)
    assert yamartino_std(s_bar, c_bar, degrees=True) == approx(np.degrees(expected_rad))


def test_yamartino_std_symmetric_in_arguments():
    assert yamartino_std(0.3, 0.7) == approx(yamartino_std(0.7, 0.3))


def test_yamartino_std_zero_for_no_spread():
    assert yamartino_std(0.0, 1.0) == approx(0.0, abs=1e-9)


def test_yamartino_std_approaches_circular_std_for_small_spreads():
    rng = np.random.default_rng(3)
    true_std_rad = 0.02
    angles = rng.normal(0, true_std_rad, 100_000)
    s_bar, c_bar = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    result_rad = yamartino_std(s_bar, c_bar, degrees=False)
    assert result_rad == approx(true_std_rad, rel=0.05)


def test_rotate_covariance_streamwise_aligned_flow_is_a_no_op():
    var_ue, var_vn, cov_ue_vn = 2.0, 0.5, 0.1
    cov_ue_w, cov_vn_w = 0.3, -0.2
    u_var, v_var, uv_cov, uw_cov, vw_cov = rotate_covariance_streamwise(
        var_ue, var_vn, cov_ue_vn, cov_ue_w, cov_vn_w, phi=0.0
    )
    assert (u_var, v_var, uv_cov, uw_cov, vw_cov) == approx(
        (var_ue, var_vn, cov_ue_vn, cov_ue_w, cov_vn_w)
    )


def test_rotate_covariance_streamwise_preserves_trace():
    rng = np.random.default_rng(4)
    var_ue, var_vn = rng.uniform(0.5, 3, 5), rng.uniform(0.5, 3, 5)
    cov_ue_vn = rng.uniform(-1, 1, 5)
    cov_ue_w, cov_vn_w = rng.uniform(-1, 1, 5), rng.uniform(-1, 1, 5)
    phi = rng.uniform(-np.pi, np.pi, 5)
    u_var, v_var, _, _, _ = rotate_covariance_streamwise(var_ue, var_vn, cov_ue_vn, cov_ue_w, cov_vn_w, phi)
    assert np.allclose(u_var + v_var, var_ue + var_vn, rtol=1e-12)


def test_rotate_covariance_streamwise_matches_matrix_rotation():
    var_ue, var_vn, cov_ue_vn = 1.5, 2.5, 0.4
    cov_ue_w, cov_vn_w = 0.2, -0.6
    phi = 0.7
    result = rotate_covariance_streamwise(var_ue, var_vn, cov_ue_vn, cov_ue_w, cov_vn_w, phi)

    q = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    c_full = np.array([
        [var_ue, cov_ue_vn, cov_ue_w],
        [cov_ue_vn, var_vn, cov_vn_w],
        [cov_ue_w, cov_vn_w, 0.0],  # w_var irrelevant to the entries checked below
    ])
    rotated = q @ c_full @ q.T
    assert result[0] == approx(rotated[0, 0])
    assert result[1] == approx(rotated[1, 1])
    assert result[2] == approx(rotated[0, 1])
    assert result[3] == approx(rotated[0, 2])
    assert result[4] == approx(rotated[1, 2])


def test_streamwise_angle_matches_atan2():
    assert streamwise_angle(1.0, 1.0) == approx(np.pi / 4)
