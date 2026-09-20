"""Vector <-> compass-bearing conversions, pure numpy.

`wind_components`/`polar_wind` are a plain, convention-free conversion
between a vector (u, v) = (east, north) and the compass bearing (degrees
clockwise of north) that the vector itself points toward: `direction` in
`wind_components`, and out of `polar_wind`, always describes (u, v)'s own
heading, whatever that heading means physically - feed in a "wind blows
FROM here" vector and you get a FROM-bearing back, feed in a "wind blows
TOWARD here" vector and you get a TOWARD-bearing back.

`bearing_to_vector`/`vector_to_bearing` are the meteorological pair: they
convert between a reported wind direction (a FROM-bearing - the direction
the wind is blowing from, the standard convention for a measured or
reported wind direction) and the "blows-toward" velocity vector that
represents the actual air motion, 180 degrees from the FROM-bearing. Use
these two, not the generic pair above, whenever a real wind direction is
involved.
"""
import numpy as np

FloatOrArray = float | np.ndarray


def wind_components(speed: FloatOrArray, direction: FloatOrArray, degrees: bool = True):
    """(speed, direction) -> (u, v) = (east, north): the vector of this speed
    that points toward compass bearing `direction`. Exact inverse of
    polar_wind. Zero speed gives (0, 0) regardless of direction, so a NaN
    direction at zero speed doesn't poison the result.
    """
    direction_rad = np.deg2rad(direction) if degrees else direction
    u = speed * np.sin(direction_rad)
    v = speed * np.cos(direction_rad)
    zero = np.asarray(speed) == 0
    u = np.where(zero, 0.0, u)
    v = np.where(zero, 0.0, v)
    if np.ndim(speed) == 0 and np.ndim(direction) == 0:
        return float(u), float(v)
    return u, v


def polar_wind(u: FloatOrArray, v: FloatOrArray, degrees: bool = True):
    """(u, v) = (east, north) -> (speed, direction): the compass bearing that
    (u, v) itself points toward. Exact inverse of wind_components.
    """
    speed = np.hypot(u, v)
    direction = np.arctan2(u, v)
    if degrees:
        direction = np.rad2deg(direction) % 360
    else:
        direction = direction % (2 * np.pi)
    if np.ndim(u) == 0 and np.ndim(v) == 0:
        return float(speed), float(direction)
    return speed, direction


def bearing_to_vector(speed: FloatOrArray, from_bearing: FloatOrArray, degrees: bool = True):
    """(speed, FROM-bearing) -> (u, v) = (east, north): the "blows-toward"
    velocity vector for a wind reported as coming from `from_bearing`.
    Exact inverse of vector_to_bearing.
    """
    mod = 360 if degrees else 2 * np.pi
    toward_bearing = (from_bearing + mod / 2) % mod
    return wind_components(speed, toward_bearing, degrees=degrees)


def vector_to_bearing(u: FloatOrArray, v: FloatOrArray, degrees: bool = True):
    """(u, v) = (east, north) "blows-toward" velocity components -> (speed,
    FROM-bearing): the standard reported wind direction. Exact inverse of
    bearing_to_vector.
    """
    mod = 360 if degrees else 2 * np.pi
    speed, toward_bearing = polar_wind(u, v, degrees=degrees)
    from_bearing = (toward_bearing + mod / 2) % mod
    return speed, from_bearing


def signed_angular_distance(theta: float, phi: float, degrees: bool = True, reverse: bool = False) -> float:
    """Signed minimal angle from phi to theta (positive = clockwise), scalar inputs.

    Convention-agnostic: depends only on theta - phi.
    """
    flip_sign = -1 if reverse else 1
    mod = 360 if degrees else 2 * np.pi
    d0 = (theta - phi) % mod
    d1 = mod - d0
    if d0 > d1:
        return -flip_sign * d1
    return flip_sign * d0


def series_signed_angular_distance(theta: np.ndarray, phi: np.ndarray, degrees: bool = True, reverse: bool = False) -> np.ndarray:
    """Vectorized signed_angular_distance, for arrays."""
    flip_sign = -1 if reverse else 1
    mod = 360 if degrees else 2 * np.pi
    d0 = (theta - phi) % mod
    d1 = mod - d0
    return flip_sign * np.where(d1 < d0, -d1, d0)
