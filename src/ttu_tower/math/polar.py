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


def streamwise_angle(mean_ue: FloatOrArray, mean_vn: FloatOrArray) -> FloatOrArray:
    """phi = atan2(mean_vn, mean_ue) (radians): the angle of a mean wind
    vector, for rotate_streamwise.
    """
    return np.arctan2(mean_vn, mean_ue)


def rotate_streamwise(ue: FloatOrArray, vn: FloatOrArray, phi: FloatOrArray):
    """Rotate (ue, vn) = (east, north) into the streamwise frame at angle
    phi: u along the mean wind, v 90 degrees to its left.
    """
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    u = ue * cos_phi + vn * sin_phi
    v = -ue * sin_phi + vn * cos_phi
    return u, v


def rotate_covariance_streamwise(var_ue, var_vn, var_ue_vn, cov_ue_w, cov_vn_w, phi):
    """Rotate an Earth-frame (ue, vn, w) covariance matrix into the streamwise
    frame at angle phi (C' = Q C Q^T; w is unaffected). Returns (u_var, v_var,
    uv_cov, uw_cov, vw_cov); w_var is unchanged and not returned.
    """
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    u_var = cos_phi**2 * var_ue + 2 * cos_phi * sin_phi * var_ue_vn + sin_phi**2 * var_vn
    v_var = sin_phi**2 * var_ue - 2 * cos_phi * sin_phi * var_ue_vn + cos_phi**2 * var_vn
    uv_cov = (cos_phi**2 - sin_phi**2) * var_ue_vn + cos_phi * sin_phi * (var_vn - var_ue)
    uw_cov = cos_phi * cov_ue_w + sin_phi * cov_vn_w
    vw_cov = -sin_phi * cov_ue_w + cos_phi * cov_vn_w
    return u_var, v_var, uv_cov, uw_cov, vw_cov


def yamartino_std(s_bar: FloatOrArray, c_bar: FloatOrArray, degrees: bool = True) -> FloatOrArray:
    """Yamartino's single-pass wind-direction standard deviation, from the
    mean sin/cos of the sample directions (s_bar, c_bar; symmetric in the
    two - only s_bar**2 + c_bar**2 ever matters).
    """
    epsilon = np.sqrt(np.maximum(0.0, 1.0 - s_bar**2 - c_bar**2))
    sigma = np.arcsin(epsilon) * (1.0 + (2.0 / np.sqrt(3.0) - 1.0) * epsilon**3)
    return np.degrees(sigma) if degrees else sigma
