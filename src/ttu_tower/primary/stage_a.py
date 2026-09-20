"""Stage A: per-sample transforms on one boom's raw arrays, in order -
units, bounds, triplet coupling, tilt, rotation to Earth axes.
"""
from dataclasses import dataclass

import numpy as np

from ttu_tower.constants import TILT_ANGLES, TILT_AXES

_MPH_TO_MS = 1 / 2.23694
_INHG_TO_KPA = 3.38639


def to_si(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Raw source units (mph, F, %, inHg) -> SI (m/s, K, fraction, kPa)."""
    return {
        "u": raw["u"] * _MPH_TO_MS,
        "v": raw["v"] * _MPH_TO_MS,
        "w": raw["w"] * _MPH_TO_MS,
        "ts": (raw["ts"] - 32.0) * (5.0 / 9.0) + 273.15,
        "t": (raw["t"] - 32.0) * (5.0 / 9.0) + 273.15,
        "rh": raw["rh"] / 100.0,
        "p": raw["p"] * _INHG_TO_KPA,
    }


def apply_bounds(si: dict[str, np.ndarray], cfg_bounds) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Set out-of-[qc.bounds] samples to NaN. A sonic-triplet violation in any
    of u, v, w removes all three. Returns (si, removed) with a boolean removal
    mask per variable.
    """
    out = dict(si)
    removed = {}

    lo, hi = cfg_bounds.sonic
    sonic_bad = np.zeros_like(si["u"], dtype=bool)
    for var in ("u", "v", "w"):
        x = si[var]
        sonic_bad |= (x < lo) | (x > hi)
    for var in ("u", "v", "w"):
        x = si[var].copy()
        x[sonic_bad] = np.nan
        out[var] = x
        removed[var] = sonic_bad.copy()

    for var in ("ts", "t", "rh", "p"):
        lo, hi = getattr(cfg_bounds, var)
        x = si[var].copy()
        bad = (x < lo) | (x > hi)
        x[bad] = np.nan
        out[var] = x
        removed[var] = bad

    return out, removed


def couple_triplet(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A sample NaN in any of u, v, w becomes NaN in all three."""
    bad = np.isnan(u) | np.isnan(v) | np.isnan(w)
    return np.where(bad, np.nan, u), np.where(bad, np.nan, v), np.where(bad, np.nan, w)


def tilt_matrix(boom: int) -> np.ndarray:
    """The Kelly & Ennis residual-tilt correction matrix for `boom`, in the
    datalogger's raw (north, west, up) frame: R = Rnz(theta) @ Ry(gamma) @ Rz(theta),
    a rotation to the tilt axis, the tilt correction, and back.
    """
    gamma = np.deg2rad(TILT_ANGLES[boom])
    theta = np.deg2rad(TILT_AXES[boom])
    cg, sg = np.cos(gamma), np.sin(gamma)
    ct, st = np.cos(theta), np.sin(theta)
    rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cg, 0.0, sg], [0.0, 1.0, 0.0], [-sg, 0.0, cg]])
    r_nz = np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])
    return r_nz @ ry @ rz


def tilt_correct(u: np.ndarray, v: np.ndarray, w: np.ndarray, boom: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply `boom`'s tilt_matrix to the raw (north, west, up) triplet."""
    r = tilt_matrix(boom)
    u_new = r[0, 0] * u + r[0, 1] * v + r[0, 2] * w
    v_new = r[1, 0] * u + r[1, 1] * v + r[1, 2] * w
    w_new = r[2, 0] * u + r[2, 1] * v + r[2, 2] * w
    return u_new, v_new, w_new


def rotate_to_earth(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw (north, west) sonic components -> true (east, north) blows-toward components."""
    return v, -u


@dataclass
class StageA:
    ue: np.ndarray
    vn: np.ndarray
    w: np.ndarray
    ts: np.ndarray
    t: np.ndarray
    rh: np.ndarray
    p: np.ndarray
    bounds_removed: dict[str, np.ndarray]


def stage_a(raw: dict[str, np.ndarray], boom: int, cfg) -> StageA:
    """The full Stage A chain for one boom's raw arrays. `cfg` is `[qc]`."""
    si = to_si(raw)
    si, removed = apply_bounds(si, cfg.bounds)
    u, v, w = couple_triplet(si["u"], si["v"], si["w"])
    u, v, w = tilt_correct(u, v, w, boom)
    ue, vn = rotate_to_earth(u, v)
    return StageA(ue=ue, vn=vn, w=w, ts=si["ts"], t=si["t"], rh=si["rh"], p=si["p"], bounds_removed=removed)
