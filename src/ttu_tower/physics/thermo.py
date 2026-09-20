"""Thermodynamics: pressure, humidity, potential temperature. Saturation
vapor pressure and dewpoint use the AERK Magnus form (Alduchov & Eskridge
1996) rather than the Tetens approximation.
"""
import numpy as np

from ttu_tower.physics.constants import CP, R_CP, REFERENCE_PRESSURE, STANDARD_GRAVITY, WATER_AIR_MWR

# ISA (International Standard Atmosphere)
_ISA_SEA_LEVEL_PRESSURE = 101.325  # kPa
_ISA_SEA_LEVEL_TEMPERATURE = 288.15  # K
_ISA_LAPSE_COEFFICIENT = 2.25577e-5  # 1/m
_ISA_LAPSE_EXPONENT = 5.25588

# AERK Magnus coefficients (Alduchov & Eskridge 1996), for saturation vapor pressure over water
_MAGNUS_A = 0.61094  # kPa
_MAGNUS_B = 17.625
_MAGNUS_C = 243.04  # degrees C


def isa_pressure(z: float | np.ndarray) -> float | np.ndarray:
    """ISA barometric pressure (kPa) at height z (m above sea level)."""
    return _ISA_SEA_LEVEL_PRESSURE * (1 - _ISA_LAPSE_COEFFICIENT * z) ** _ISA_LAPSE_EXPONENT


def pressure_to_slp(value, meters_asl, gravity: float = STANDARD_GRAVITY):
    """Sea-level-equivalent pressure from a value measured at meters_asl."""
    exponent = -1 / R_CP
    coefficient = 1 - (gravity * meters_asl) / (CP * _ISA_SEA_LEVEL_TEMPERATURE)
    return value * coefficient**exponent


def pressure_above_msl(value, meters_asl, gravity: float = STANDARD_GRAVITY):
    """Local barometric pressure at meters_asl from a sea-level pressure."""
    exponent = 1 / R_CP
    coefficient = 1 - (gravity * meters_asl) / (CP * _ISA_SEA_LEVEL_TEMPERATURE)
    return value * coefficient**exponent


def saturation_vapor_pressure(temperature):
    """Saturation vapor pressure (kPa) over water, AERK Magnus form. temperature in K."""
    t_c = temperature - 273.15
    return _MAGNUS_A * np.exp(_MAGNUS_B * t_c / (t_c + _MAGNUS_C))


def water_partial_pressure(relative_humidity, sat_vapor_pressure):
    """Partial pressure of water (e = RH * e_s)."""
    return relative_humidity * sat_vapor_pressure


actual_vapor_pressure = water_partial_pressure  # alias


def water_air_mixing_ratio(actual_vapor_pressure, barometric_air_pressure):
    """Dimensionless water:air mixing ratio, given two pressures of the same units."""
    return WATER_AIR_MWR * actual_vapor_pressure / (barometric_air_pressure - actual_vapor_pressure)


def specific_humidity(mixing_ratio):
    """Specific humidity (mass of vapor / total air mass) from mixing ratio."""
    return mixing_ratio / (1 + mixing_ratio)


def virtual_temperature(temperature, mixing_ratio):
    """Virtual temperature (K) from temperature (K) and mixing ratio."""
    return temperature * (1 + mixing_ratio / WATER_AIR_MWR) / (1 + mixing_ratio)


def potential_temperature(temperature, barometric_air_pressure):
    """Potential temperature (K) from temperature (K) and pressure (kPa)."""
    return temperature * (REFERENCE_PRESSURE / barometric_air_pressure) ** R_CP


def virtual_potential_temperature(potential_temperature, mixing_ratio, approximate: bool = False):
    """Virtual potential temperature (K) from potential temperature (K) and mixing ratio."""
    if approximate:
        return potential_temperature * (1 + 0.607 * mixing_ratio)
    return potential_temperature * (1 + mixing_ratio / WATER_AIR_MWR) / (1 + mixing_ratio)


def dewpoint_temperature(temperature, relative_humidity):
    """Dewpoint (K) from temperature (K) and relative humidity (fraction), AERK
    Magnus inverse: exact inverse of saturation_vapor_pressure, so
    saturation_vapor_pressure(dewpoint_temperature(T, rh)) == rh * saturation_vapor_pressure(T).
    """
    e = relative_humidity * saturation_vapor_pressure(temperature)
    gamma = np.log(e / _MAGNUS_A)
    t_c = _MAGNUS_C * gamma / (_MAGNUS_B - gamma)
    return t_c + 273.15


def vpt_from_3(relative_humidity, barometric_air_pressure, temperature):
    """Full pipeline: virtual potential temperature (K) from rh (fraction), pressure
    (kPa) and temperature (K)."""
    svp = saturation_vapor_pressure(temperature)
    avp = relative_humidity * svp
    w = water_air_mixing_ratio(actual_vapor_pressure=avp, barometric_air_pressure=barometric_air_pressure)
    pt = potential_temperature(temperature=temperature, barometric_air_pressure=barometric_air_pressure)
    return virtual_potential_temperature(potential_temperature=pt, mixing_ratio=w, approximate=False)
