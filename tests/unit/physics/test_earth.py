from pytest import approx

from ttu_tower.physics.earth import coriolis, local_gravity


def test_local_gravity():
    assert local_gravity(45, 5000) == approx(9.79077, abs=5e-6)


def test_coriolis_zero_at_equator():
    assert coriolis(0.0) == approx(0.0, abs=1e-12)


def test_coriolis_matches_formula():
    from ttu_tower.physics.constants import EARTH_ROTATION_RATE
    import numpy as np

    lat = 33.61055
    expected = 2 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(lat))
    assert coriolis(lat) == approx(expected, rel=1e-12)
