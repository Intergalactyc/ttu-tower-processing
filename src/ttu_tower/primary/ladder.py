"""The statistics ladder: second-moment statistics at every rung, aggregated
exactly from finest-block sums.
"""
import numpy as np

from ttu_tower.math.polar import rotate_covariance_streamwise, streamwise_angle, yamartino_std
from ttu_tower.primary.blocks import FamilySums
from ttu_tower.timegrid import BlockGrid

RUNG_SECONDS = tuple(9.375 * 2.0**j for j in range(8))
_SLOT_SUPPORT_SAMPLES = 30_000
_WINDOW_SUPPORT_SAMPLES = 60_000


def _grouped(fam: FamilySums, lo: int, hi: int, group_size: int) -> FamilySums:
    m = len(fam.columns)
    n = fam.n[lo:hi].reshape(-1, group_size).sum(axis=1)
    sx = fam.sx[lo:hi].reshape(-1, group_size, m).sum(axis=1)
    return FamilySums(n=n, sx=sx, columns=fam.columns)


def _col(fam: FamilySums, name: str) -> np.ndarray:
    return fam.sx[:, fam.columns.index(name)]


def _rung_mean(per_block: np.ndarray, valid: np.ndarray) -> float:
    return float(per_block[valid].mean()) if valid.any() else np.nan


def _momentum_rows(momentum: FamilySums, direction: FamilySums, block_len: float, c: float):
    n = momentum.n
    valid = (n / block_len) >= c

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ue, mean_vn, mean_w = _col(momentum, "ue") / n, _col(momentum, "vn") / n, _col(momentum, "w") / n
        mean_ws, mean_ws2 = _col(momentum, "ws") / n, _col(momentum, "ws2") / n
        var_ue = np.maximum(_col(momentum, "ue2") / n - mean_ue**2, 0.0)
        var_vn = np.maximum(_col(momentum, "vn2") / n - mean_vn**2, 0.0)
        var_w = np.maximum(_col(momentum, "w2") / n - mean_w**2, 0.0)
        cov_ue_vn = _col(momentum, "ue_vn") / n - mean_ue * mean_vn
        cov_ue_w = _col(momentum, "ue_w") / n - mean_ue * mean_w
        cov_vn_w = _col(momentum, "vn_w") / n - mean_vn * mean_w
        ws_var = np.maximum(mean_ws2 - mean_ws**2, 0.0)

        n_dir = direction.n
        s_bar = _col(direction, "vn_ws") / n_dir
        c_bar = _col(direction, "ue_ws") / n_dir

    phi = streamwise_angle(mean_ue, mean_vn)
    u_var, v_var, uv_cov, uw_cov, vw_cov = rotate_covariance_streamwise(var_ue, var_vn, cov_ue_vn, cov_ue_w, cov_vn_w, phi)
    u_mean = np.hypot(mean_ue, mean_vn)
    wd_std_block = yamartino_std(s_bar, c_bar, degrees=True)

    rows = [
        ("u", "var", _rung_mean(u_var, valid)),
        ("v", "var", _rung_mean(v_var, valid)),
        ("w", "var", _rung_mean(var_w, valid)),
        ("uv", "cov", _rung_mean(uv_cov, valid)),
        ("uw", "cov", _rung_mean(uw_cov, valid)),
        ("vw", "cov", _rung_mean(vw_cov, valid)),
        ("u", "mean", _rung_mean(u_mean, valid)),
        ("w", "mean", _rung_mean(mean_w, valid)),
        ("ws", "mean", _rung_mean(mean_ws, valid)),
        ("ws", "std", np.sqrt(_rung_mean(ws_var, valid))),
        ("wd", "std", np.sqrt(_rung_mean(wd_std_block**2, valid))),
    ]
    return rows, int(valid.sum())


def _heat_rows(heat: FamilySums, block_len: float, c: float):
    n = heat.n
    valid = (n / block_len) >= c
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_w = _col(heat, "w") / n
        mean_theta = _col(heat, "theta") / n
        mean_theta2 = _col(heat, "theta2") / n
        mean_w_theta = _col(heat, "w_theta") / n

    wvpts_cov = mean_w_theta - mean_w * mean_theta
    vpts_var = np.maximum(mean_theta2 - mean_theta**2, 0.0)
    vpts_mean = mean_theta + 273.15

    rows = [
        ("wvpts", "cov", _rung_mean(wvpts_cov, valid)),
        ("vpts", "var", _rung_mean(vpts_var, valid)),
        ("vpts", "mean", _rung_mean(vpts_mean, valid)),
    ]
    return rows, int(valid.sum())


def _ts_rows(ts: FamilySums, block_len: float, c: float):
    n = ts.n
    valid = (n / block_len) >= c
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_tau = _col(ts, "tau_s") / n
        mean_tau2 = _col(ts, "tau_s2") / n

    ts_var = np.maximum(mean_tau2 - mean_tau**2, 0.0)
    ts_mean = mean_tau + 273.15

    rows = [
        ("ts", "var", _rung_mean(ts_var, valid)),
        ("ts", "mean", _rung_mean(ts_mean, valid)),
    ]
    return rows, int(valid.sum())


def rung_statistics(finest: dict[str, FamilySums], c: float):
    """Second-moment statistics at every rung (9.375 s .. 20 min), from finest
    sums over the 128 finest blocks [64k-32, 64k+96) centered on a slot.
    Returns (values, coverage): values rows (rung_s, variable, stat, value);
    coverage rows (rung_s, family, blocks_used, blocks_total, coverage).
    """
    finest_len = BlockGrid.FINEST.length_samples
    values, coverage = [], []

    for j in range(8):
        rung_s = RUNG_SECONDS[j]
        if j < 7:
            lo, hi, group = 32, 96, 2**j
            support = _SLOT_SUPPORT_SAMPLES
        else:
            lo, hi, group = 0, 128, 128
            support = _WINDOW_SUPPORT_SAMPLES
        n_total_blocks = (hi - lo) // group
        block_len = finest_len * group

        momentum = _grouped(finest["momentum"], lo, hi, group)
        heat = _grouped(finest["heat"], lo, hi, group)
        ts = _grouped(finest["ts"], lo, hi, group)
        direction = _grouped(finest["direction"], lo, hi, group)

        mom_rows, mom_used = _momentum_rows(momentum, direction, block_len, c)
        values += [(rung_s, var, stat, val) for var, stat, val in mom_rows]
        coverage.append((rung_s, "momentum", mom_used, n_total_blocks, momentum.n.sum() / support))

        heat_rows, heat_used = _heat_rows(heat, block_len, c)
        values += [(rung_s, var, stat, val) for var, stat, val in heat_rows]
        coverage.append((rung_s, "heat", heat_used, n_total_blocks, heat.n.sum() / support))

        ts_rows, ts_used = _ts_rows(ts, block_len, c)
        values += [(rung_s, var, stat, val) for var, stat, val in ts_rows]
        coverage.append((rung_s, "ts", ts_used, n_total_blocks, ts.n.sum() / support))

    return values, coverage
