"""Local gravity and the Coriolis parameter."""
import numpy as np

from ttu_tower.physics.constants import EARTH_ROTATION_RATE

# International Gravity Formula 1980 coefficients
_IGF_G0 = 9.780327
_IGF_B1 = 0.0053024
_IGF_B2 = 0.0000058
_FREE_AIR_GRADIENT = -3.086e-6  # per meter


def local_gravity(latitude, elevation, *, latitude_degrees: bool = True):
    """Theoretical local gravity (m/s^2) from latitude (deg) and elevation (m)."""
    rad = np.deg2rad(latitude) if latitude_degrees else latitude
    igf = _IGF_G0 * (1 + _IGF_B1 * np.sin(rad) ** 2 - _IGF_B2 * np.sin(2 * rad) ** 2)
    free_air = _FREE_AIR_GRADIENT * elevation
    return igf + free_air


def coriolis(latitude, *, latitude_degrees: bool = True, omega: float = EARTH_ROTATION_RATE):
    """Coriolis parameter (rad/s) from latitude (deg)."""
    rad = np.deg2rad(latitude) if latitude_degrees else latitude
    return 2 * omega * np.sin(rad)
