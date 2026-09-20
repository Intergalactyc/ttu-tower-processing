"""Integral timescales and transport efficiencies at every rung, computed
directly from raw samples (they don't aggregate from block sums the way the
ladder's second moments do).
"""
import numpy as np

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.math.polar import rotate_streamwise, streamwise_angle
from ttu_tower.math.stats import autocovariance, efolding_integral
from ttu_tower.physics.turbulence import transport_efficiency
from ttu_tower.timegrid import BlockGrid, majority_block

RUNG_SECONDS = tuple(9.375 * 2.0**j for j in range(8))
_SLOT_SAMPLES = 30_000
_WINDOW_SAMPLES = 60_000
_WINDOW_MARGIN_SAMPLES = 15_000


def _tau_block_index(g: np.ndarray, group: int, slot_first_finest: int) -> np.ndarray:
    finest_idx = majority_block(g, BlockGrid.FINEST) - slot_first_finest
    return finest_idx // group


def _fully_usable_blocks(block_idx: np.ndarray, mask: np.ndarray, n_blocks: int) -> np.ndarray:
    usable = np.ones(n_blocks, dtype=bool)
    np.logical_and.at(usable, block_idx, mask)
    return usable


def _pooled_its(fluctuations: list[np.ndarray], max_lag: int) -> float:
    if not fluctuations:
        return np.nan
    pooled = np.zeros(max_lag + 1)
    for x in fluctuations:
        pooled += autocovariance(x, max_lag)
    if pooled[0] == 0:
        return np.nan
    rho = pooled / pooled[0]
    return (1.0 / SAMPLE_HZ) * efolding_integral(rho)


def _its_momentum(ue, vn, w, mask, block_idx, n_blocks, max_lag):
    usable = _fully_usable_blocks(block_idx, mask, n_blocks)
    u_fluct, v_fluct, w_fluct = [], [], []
    for b in np.flatnonzero(usable):
        sel = block_idx == b
        mean_ue, mean_vn, mean_w = ue[sel].mean(), vn[sel].mean(), w[sel].mean()
        phi = streamwise_angle(mean_ue, mean_vn)
        u_p, v_p = rotate_streamwise(ue[sel] - mean_ue, vn[sel] - mean_vn, phi)
        u_fluct.append(u_p)
        v_fluct.append(v_p)
        w_fluct.append(w[sel] - mean_w)
    its_u = _pooled_its(u_fluct, max_lag)
    its_v = _pooled_its(v_fluct, max_lag)
    its_w = _pooled_its(w_fluct, max_lag)
    return its_u, its_v, its_w, int(usable.sum())


def _its_vpts(vpts, mask, block_idx, n_blocks, max_lag):
    usable = _fully_usable_blocks(block_idx, mask, n_blocks)
    fluct = [vpts[block_idx == b] - vpts[block_idx == b].mean() for b in np.flatnonzero(usable)]
    return _pooled_its(fluct, max_lag), int(usable.sum())


def _te_momentum_heat(ue, vn, w, theta, momentum_mask, heat_mask, block_idx, n_blocks, nominal_len, c):
    s_all_uw = s_pos_uw = s_neg_uw = 0.0
    s_all_wt = s_pos_wt = s_neg_wt = 0.0
    for b in range(n_blocks):
        sel = block_idx == b

        mom_ok = momentum_mask[sel]
        if mom_ok.sum() / nominal_len >= c:
            ue_b, vn_b, w_b = ue[sel][mom_ok], vn[sel][mom_ok], w[sel][mom_ok]
            phi = streamwise_angle(ue_b.mean(), vn_b.mean())
            u_p, _ = rotate_streamwise(ue_b - ue_b.mean(), vn_b - vn_b.mean(), phi)
            uw_p = u_p * (w_b - w_b.mean())
            s_all_uw += uw_p.sum()
            s_pos_uw += uw_p[uw_p > 0].sum()
            s_neg_uw += uw_p[uw_p < 0].sum()

        heat_ok = heat_mask[sel]
        if heat_ok.sum() / nominal_len >= c:
            w_b2, theta_b = w[sel][heat_ok], theta[sel][heat_ok]
            wt_p = (w_b2 - w_b2.mean()) * (theta_b - theta_b.mean())
            s_all_wt += wt_p.sum()
            s_pos_wt += wt_p[wt_p > 0].sum()
            s_neg_wt += wt_p[wt_p < 0].sum()

    te_uw = transport_efficiency(s_all_uw, s_pos_uw, s_neg_uw)
    te_wt = transport_efficiency(s_all_wt, s_pos_wt, s_neg_wt)
    return te_uw, te_wt


def rung_its_te(series: dict[str, np.ndarray], masks: dict[str, np.ndarray], g0: int, k: int, cfg_ladder, c: float):
    """ITS (u, v, w, vpts) and TE (uw, wvpts) at every rung, for slot k.
    `series`/`masks` must cover samples [30000k-15000, 30000k+45000). Returns
    (values, coverage): values rows (rung_s, variable, stat, value); coverage
    rows (rung_s, family, blocks_used, blocks_total, coverage=blocks_used/blocks_total).
    """
    slot_start_g = _SLOT_SAMPLES * k
    slot_first_finest = 64 * k
    values, coverage = [], []

    ue, vn, w, vpts = series["ue"], series["vn"], series["w"], series["vpts"]
    momentum_mask, heat_mask, ts_mask = masks["momentum"], masks["heat"], masks["ts"]
    theta = vpts - 273.15

    for j in range(8):
        rung_s = RUNG_SECONDS[j]
        group = 2**j
        nominal_len = round(rung_s * SAMPLE_HZ)
        max_lag = int(np.floor(cfg_ladder.its_max_lag_fraction * nominal_len))

        if j < 7:
            # 64 // group, not _SLOT_SAMPLES // nominal_len: nominal_len is
            # already rounded (469, not 468.75), which would silently lose
            # a block (63 instead of 64) at the finest rungs.
            span_g0, span_n, n_blocks = slot_start_g, _SLOT_SAMPLES, 64 // group
        else:
            span_g0, span_n, n_blocks = slot_start_g - _WINDOW_MARGIN_SAMPLES, _WINDOW_SAMPLES, 1

        local = slice(span_g0 - g0, span_g0 - g0 + span_n)
        ue_l, vn_l, w_l, vpts_l, theta_l = ue[local], vn[local], w[local], vpts[local], theta[local]
        mom_l, heat_l, ts_l = momentum_mask[local], heat_mask[local], ts_mask[local]

        if j < 7:
            g_arr = span_g0 + np.arange(span_n)
            block_idx = _tau_block_index(g_arr, group, slot_first_finest)
        else:
            block_idx = np.zeros(span_n, dtype=np.int64)

        its_u, its_v, its_w, blocks_its = _its_momentum(ue_l, vn_l, w_l, mom_l, block_idx, n_blocks, max_lag)
        its_vpts, blocks_its_vpts = _its_vpts(vpts_l, ts_l, block_idx, n_blocks, max_lag)

        values.append((rung_s, "u", "its", its_u))
        values.append((rung_s, "v", "its", its_v))
        values.append((rung_s, "w", "its", its_w))
        values.append((rung_s, "vpts", "its", its_vpts))
        coverage.append((rung_s, "its", blocks_its, n_blocks, blocks_its / n_blocks))
        coverage.append((rung_s, "its_vpts", blocks_its_vpts, n_blocks, blocks_its_vpts / n_blocks))

        te_uw, te_wt = _te_momentum_heat(ue_l, vn_l, w_l, theta_l, mom_l, heat_l, block_idx, n_blocks, nominal_len, c)
        values.append((rung_s, "uw", "te", te_uw))
        values.append((rung_s, "wvpts", "te", te_wt))

    return values, coverage
