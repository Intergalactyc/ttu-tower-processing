"""Universal physical constants shared across physics/. A formula's own
coefficients (Magnus, ISA, Businger-Dyer, ...) live next to that formula
instead.
"""

R = 8.314462618 / 0.02896968  # gas constant of dry air, J/(kg K)
CP = 1004.68506  # specific heat capacity of air at constant pressure, J/(kg K)
R_CP = R / CP  # ~0.286

REFERENCE_PRESSURE = 100.0  # kPa, potential-temperature reference pressure
WATER_AIR_MWR = 0.622  # water:air molecular weight ratio

STANDARD_GRAVITY = 9.80665  # m/s^2
KAPPA = 0.41  # von Karman constant
EARTH_ROTATION_RATE = 7.2921e-5  # rad/s
