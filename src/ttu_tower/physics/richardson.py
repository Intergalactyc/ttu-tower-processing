"""Bulk Richardson number."""
import numpy as np

from ttu_tower.math import polar
from ttu_tower.physics.constants import STANDARD_GRAVITY


def bulk_richardson_number(
    vpt_lower,
    vpt_upper,
    height_lower: float,
    height_upper: float,
    ws_lower,
    ws_upper,
    wd_lower,
    wd_upper,
    *,
    components: bool = False,
    gravity: float = STANDARD_GRAVITY,
):
    """Bulk Richardson number between two heights.

    wd_lower/wd_upper (when components=False) are FROM-bearings, converted via
    bearing_to_vector. Ri_b depends only on the squared magnitude of the
    (u_upper-u_lower, v_upper-v_lower) difference vector, invariant under any
    consistent rotation/reflection applied identically at both heights.
    Returns a plain float for scalar inputs, an ndarray otherwise.
    """
    # Convert to numpy dtype up front: plain-float scalar inputs would otherwise
    # do plain Python division below, which raises ZeroDivisionError on zero
    # shear instead of producing NaN.
    vpt_lower = np.asarray(vpt_lower, dtype=float)
    vpt_upper = np.asarray(vpt_upper, dtype=float)
    ws_lower = np.asarray(ws_lower, dtype=float)
    ws_upper = np.asarray(ws_upper, dtype=float)
    wd_lower = np.asarray(wd_lower, dtype=float)
    wd_upper = np.asarray(wd_upper, dtype=float)

    delta_vpt = vpt_upper - vpt_lower
    delta_z = height_upper - height_lower

    if components:
        u_lower, u_upper = ws_lower, ws_upper
        v_lower, v_upper = wd_lower, wd_upper
    else:
        u_lower, v_lower = polar.bearing_to_vector(ws_lower, wd_lower)
        u_upper, v_upper = polar.bearing_to_vector(ws_upper, wd_upper)

    delta_u = u_upper - u_lower
    delta_v = v_upper - v_lower
    shear_sq = delta_u**2 + delta_v**2
    vpt_avg = (vpt_upper + vpt_lower) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        ri = np.where(shear_sq == 0, np.nan, (gravity * delta_vpt * delta_z) / (vpt_avg * shear_sq))

    if np.ndim(ri) == 0:
        return float(ri)
    return ri
