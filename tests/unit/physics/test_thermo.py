import numpy as np
from pytest import approx

from ttu_tower.physics import thermo
from ttu_tower.physics.constants import R_CP

# Standard published saturation-vapor-pressure-over-water table (kPa), e.g. WMO/CIMO.
_PUBLISHED_ES_KPA = {
    0.0: 0.6112,
    10.0: 1.2280,
    20.0: 2.3370,
    30.0: 4.2430,
}


def test_saturation_vapor_pressure_matches_published_table():
    for t_c, expected in _PUBLISHED_ES_KPA.items():
        es = thermo.saturation_vapor_pressure(t_c + 273.15)
        assert es == approx(expected, rel=0.01)


def test_saturation_vapor_pressure_exact_at_zero_celsius():
    # e_s(0C) = 0.61094 * exp(0) = 0.61094 exactly (the AERK coefficient itself).
    assert thermo.saturation_vapor_pressure(273.15) == approx(0.61094, abs=1e-12)


def test_dewpoint_is_exact_inverse_of_saturation_vapor_pressure():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = rng.uniform(233.15, 313.15)
        rh = rng.uniform(0.05, 1.0)
        td = thermo.dewpoint_temperature(t, rh)
        e = rh * thermo.saturation_vapor_pressure(t)
        assert thermo.saturation_vapor_pressure(td) == approx(e, rel=1e-12)


def test_isa_pressure_at_tower_boom_heights():
    # site elevation 1014 m + boom height: ~89.7 kPa at boom 1 (0.9 m), ~87.6 kPa at boom 10 (200 m).
    assert thermo.isa_pressure(1014.9) == approx(89.7, abs=0.05)
    assert thermo.isa_pressure(1214.0) == approx(87.6, abs=0.05)


def test_isa_pressure_sea_level():
    assert thermo.isa_pressure(0.0) == approx(101.325, abs=1e-9)


def test_potential_temperature_identity_at_reference_pressure():
    assert thermo.potential_temperature(300.0, 100.0) == approx(300.0, abs=1e-12)


def test_potential_temperature_matches_formula():
    t, p = 295.0, 87.0
    expected = t * (100.0 / p) ** R_CP
    assert thermo.potential_temperature(t, p) == approx(expected, abs=1e-9)


def test_water_air_mixing_ratio_and_specific_humidity():
    e, p = 1.5, 90.0
    r = thermo.water_air_mixing_ratio(e, p)
    assert r == approx(0.622 * 1.5 / (90.0 - 1.5), rel=1e-12)
    q = thermo.specific_humidity(r)
    assert q == approx(r / (1 + r), rel=1e-12)


def test_vpt_from_3_matches_manual_pipeline():
    rh, p, t = 0.6, 88.0, 293.0
    vpt = thermo.vpt_from_3(rh, p, t)

    svp = thermo.saturation_vapor_pressure(t)
    avp = rh * svp
    w = thermo.water_air_mixing_ratio(avp, p)
    pt = thermo.potential_temperature(t, p)
    expected = pt * (1 + w / 0.622) / (1 + w)

    assert vpt == approx(expected, rel=1e-12)
