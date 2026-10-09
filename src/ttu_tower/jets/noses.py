"""Where the nose lies between the booms: the quadratic through a slot's
observed peak and the booms directly below and above it, at its maximum.

For analysing identified events only; detection works on the observed
profile. The peak is a strict local maximum, so the quadratic opens downward
and its maximum lies between the two neighbouring booms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ttu_tower.jets.identify import Profiles

COLUMNS = ("z_nose_est", "u_nose_est")


def vertex(z: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The maximum (height, speed) of the quadratic through three points per
    row (`z`, `u`: n x 3); NaN where the points are missing or the quadratic
    doesn't open downward.
    """
    z0, z1, z2 = z[:, 0], z[:, 1], z[:, 2]
    u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        d0, d1, d2 = (z0 - z1) * (z0 - z2), (z1 - z0) * (z1 - z2), (z2 - z0) * (z2 - z1)
        a = u0 / d0 + u1 / d1 + u2 / d2
        b = -(u0 * (z1 + z2) / d0 + u1 * (z0 + z2) / d1 + u2 * (z0 + z1) / d2)
        c = u0 * z1 * z2 / d0 + u1 * z0 * z2 / d1 + u2 * z0 * z1 / d2
        zv = -b / (2 * a)
        uv = c - b * b / (4 * a)
    ok = np.isfinite(zv) & np.isfinite(uv) & (a < 0)
    return np.where(ok, zv, np.nan), np.where(ok, uv, np.nan)


def estimate(detection: pd.DataFrame, profiles: Profiles, booms=None) -> pd.DataFrame:
    """Per slot of `detection` with a peak (`nose_boom`): the estimated nose
    height and speed, from the peak and its neighbours among `booms` (the
    profile the detection used; default all of `profiles`).
    """
    out = pd.DataFrame(np.nan, index=detection.index, columns=list(COLUMNS))
    p = profiles.subset(booms) if booms is not None else profiles
    if len(p.booms) < 3 or detection.empty:
        return out
    peak = detection["nose_boom"]
    has = peak.notna().to_numpy()
    if not has.any():
        return out
    column = {b: i for i, b in enumerate(p.booms)}
    cols = np.array([column.get(int(b), -1) for b in peak[has]])
    rows = np.searchsorted(p.slots, detection.index.to_numpy()[has])
    inside = (cols >= 1) & (cols <= len(p.booms) - 2) & (rows < p.slots.size)
    take = np.flatnonzero(has)[inside]
    cols, rows = cols[inside], rows[inside]
    trio = np.stack([cols - 1, cols, cols + 1], axis=1)
    zv, uv = vertex(p.heights[trio], p.ws[rows[:, None], trio])
    out.iloc[take, 0] = zv
    out.iloc[take, 1] = uv
    return out
