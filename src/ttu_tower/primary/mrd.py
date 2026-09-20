"""The multiresolution decomposition (HM97): dyadic mode products from floor
sums, and the per-slot detection outputs (cospectra and variance spectra)
built from them.
"""
import numpy as np

from ttu_tower.math.polar import rotate_streamwise, streamwise_angle, vector_to_bearing
from ttu_tower.timegrid import BlockGrid

SPECTRA = ("uw", "vw", "uv", "uu", "vv", "ww", "wvpts", "tsts")


def mode_scales_s(floor_level: int, sample_hz: float) -> np.ndarray:
    """P_i (seconds) for mode i = 1..K: the width of the blocks mode i is
    computed in, in units of 2^i floor blocks.
    """
    floor_block_s = BlockGrid.floor(floor_level).length_samples / sample_hz
    return floor_block_s * 2.0 ** np.arange(1, floor_level + 1)


def mrd_modes(sx: np.ndarray, sy: np.ndarray, n: np.ndarray, block_len: float, c: float):
    """The K = log2(len(sx)) dyadic modes of the window's floor sums for one
    (x, y) pair. Mode i (1-indexed) aggregates floor blocks into halves of
    2^(i-1) blocks; a block (a pair of halves) is valid if both halves have
    coverage >= c; its value is Delta_x * Delta_y / 4 with Delta the
    difference of the two half means. Returns (value, se, n_pairs), each of
    length K: value is the mean over valid blocks (NaN if none); se is
    std(ddof=1)/sqrt(n_valid) (NaN if n_valid < 2); n_pairs is n_valid.
    """
    n_floor = sx.size
    k = int(round(np.log2(n_floor)))
    value = np.full(k, np.nan)
    se = np.full(k, np.nan)
    n_pairs = np.zeros(k, dtype=np.int32)

    for idx in range(k):
        h = 2 ** idx  # 2^(i-1) for mode i = idx + 1
        sx_h = sx.reshape(-1, h).sum(axis=1)
        sy_h = sy.reshape(-1, h).sum(axis=1)
        n_h = n.reshape(-1, h).sum(axis=1)

        coverage = n_h / (h * block_len)
        valid_half = coverage >= c
        mean_x = sx_h / n_h
        mean_y = sy_h / n_h

        valid_block = valid_half[0::2] & valid_half[1::2]
        dx = mean_x[1::2] - mean_x[0::2]
        dy = mean_y[1::2] - mean_y[0::2]
        p = dx * dy / 4.0

        p_valid = p[valid_block]
        n_valid = p_valid.size
        n_pairs[idx] = n_valid
        if n_valid > 0:
            value[idx] = p_valid.mean()
        if n_valid >= 2:
            se[idx] = p_valid.std(ddof=1) / np.sqrt(n_valid)

    return value, se, n_pairs


def _nan_spectrum(k: int):
    return np.full(k, np.nan), np.full(k, np.nan), np.zeros(k, dtype=np.int32)


def detection_outputs(floor: dict, c: float):
    """Detection outputs for one slot's window: the 8 spectra (value, se,
    n_pairs each), the momentum-family wind direction, and the momentum/heat
    coverage of the whole window. Momentum (or heat/ts) spectra and wd_deg
    are NaN wherever that family has no usable weight at all in the window.
    """
    momentum, heat, ts = floor["momentum"], floor["heat"], floor["ts"]
    nb = momentum.n.size
    k = int(round(np.log2(nb)))
    block_len = BlockGrid.floor(k).length_samples

    window_nominal = nb * block_len
    coverage_momentum = momentum.n.sum() / window_nominal
    coverage_heat = heat.n.sum() / window_nominal

    spectra = {}
    wd_deg = np.nan

    if momentum.n.sum() > 0:
        ue_sum = momentum.sx[:, momentum.columns.index("ue")]
        vn_sum = momentum.sx[:, momentum.columns.index("vn")]
        w_sum = momentum.sx[:, momentum.columns.index("w")]

        _, wd_deg = vector_to_bearing(ue_sum.sum(), vn_sum.sum())
        phi = streamwise_angle(ue_sum.sum(), vn_sum.sum())
        sx_u, sx_v = rotate_streamwise(ue_sum, vn_sum, phi)

        spectra["uu"] = mrd_modes(sx_u, sx_u, momentum.n, block_len, c)
        spectra["vv"] = mrd_modes(sx_v, sx_v, momentum.n, block_len, c)
        spectra["ww"] = mrd_modes(w_sum, w_sum, momentum.n, block_len, c)
        spectra["uw"] = mrd_modes(sx_u, w_sum, momentum.n, block_len, c)
        spectra["vw"] = mrd_modes(sx_v, w_sum, momentum.n, block_len, c)
        spectra["uv"] = mrd_modes(sx_u, sx_v, momentum.n, block_len, c)
    else:
        for key in ("uu", "vv", "ww", "uw", "vw", "uv"):
            spectra[key] = _nan_spectrum(k)

    if heat.n.sum() > 0:
        w_sum = heat.sx[:, heat.columns.index("w")]
        theta_sum = heat.sx[:, heat.columns.index("theta")]
        spectra["wvpts"] = mrd_modes(w_sum, theta_sum, heat.n, block_len, c)
    else:
        spectra["wvpts"] = _nan_spectrum(k)

    if ts.n.sum() > 0:
        tau_sum = ts.sx[:, ts.columns.index("tau_s")]
        spectra["tsts"] = mrd_modes(tau_sum, tau_sum, ts.n, block_len, c)
    else:
        spectra["tsts"] = _nan_spectrum(k)

    return spectra, wd_deg, coverage_momentum, coverage_heat
