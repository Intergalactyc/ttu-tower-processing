import numpy as np
from pytest import approx

from ttu_tower.math.polar import (
    bearing_to_vector,
    polar_wind,
    series_signed_angular_distance,
    signed_angular_distance,
    vector_to_bearing,
    wind_components,
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
