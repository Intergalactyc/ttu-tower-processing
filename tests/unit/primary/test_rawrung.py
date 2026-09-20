from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import lfilter

from ttu_tower.primary.rawrung import RUNG_SECONDS, _its_vpts, _te_momentum_heat, rung_its_te


def _ou_process(rng, n, tau, dt=0.02):
    """A discretized Ornstein-Uhlenbeck process with exact ACF exp(-lag*dt/tau)."""
    alpha = np.exp(-dt / tau)
    sigma = np.sqrt(1 - alpha**2)
    noise = rng.normal(0, 1, n)
    return lfilter([sigma], [1.0, -alpha], noise)


def _cfg_ladder(its_max_lag_fraction=0.25):
    return SimpleNamespace(its_max_lag_fraction=its_max_lag_fraction)


def _window_series(rng, n=60000):
    return {
        "ue": rng.normal(3, 1, n),
        "vn": rng.normal(1, 1, n),
        "w": rng.normal(0, 0.5, n),
        "ts": rng.normal(290, 1, n),
        "vpts": rng.normal(295, 1, n),
    }


# --- ITS: vpts uses the ts family's blocks, and its value is scale-invariant --

def test_its_vpts_equals_its_of_ts_for_affine_related_data():
    rng = np.random.default_rng(10)
    n = 60000
    ts_fluct = _ou_process(rng, n, tau=5.0)
    ts = 290.0 + ts_fluct
    vpts = 2.5 * ts + 10.0  # an affine transform: ITS must be identical (scale/shift invariant)
    mask = np.ones(n, dtype=bool)
    block_idx = np.zeros(n, dtype=np.int64)

    its_from_ts, used_ts = _its_vpts(ts, mask, block_idx, 1, max_lag=3000)
    its_from_vpts, used_vpts = _its_vpts(vpts, mask, block_idx, 1, max_lag=3000)

    assert used_ts == used_vpts
    assert its_from_vpts == pytest.approx(its_from_ts, rel=1e-12)


def test_its_vpts_uses_ts_mask_not_momentum_mask():
    rng = np.random.default_rng(11)
    g0 = -15000
    n = 60000
    series = _window_series(rng, n)
    series["vpts"] = series["ts"].copy()

    mask_full = np.ones(n, dtype=bool)
    masks_baseline = {"momentum": mask_full.copy(), "heat": mask_full.copy(), "ts": mask_full.copy()}
    masks_knocked = {"momentum": mask_full.copy(), "heat": mask_full.copy(), "ts": mask_full.copy()}
    masks_knocked["momentum"][15100] = False  # a sample inside the slot, momentum-only

    cfg_ladder = _cfg_ladder()
    values_a, coverage_a = rung_its_te(series, masks_baseline, g0=g0, k=0, cfg_ladder=cfg_ladder, c=0.75)
    values_b, coverage_b = rung_its_te(series, masks_knocked, g0=g0, k=0, cfg_ladder=cfg_ladder, c=0.75)

    cov_a = {(r, f): used for r, f, used, total, cov in coverage_a}
    cov_b = {(r, f): used for r, f, used, total, cov in coverage_b}

    rung0 = RUNG_SECONDS[0]
    assert cov_b[(rung0, "its")] < cov_a[(rung0, "its")]  # momentum ITS blocks_used decreased
    assert cov_b[(rung0, "its_vpts")] == cov_a[(rung0, "its_vpts")]  # vpts ITS blocks_used unaffected


# --- ITS on a known-timescale process -----------------------------------------

def test_its_ou_process_20min_rung_close_to_analytic():
    tau_true = 5.0
    expected = (1 - 1 / np.e) * tau_true  # the e-folding integral truncates at the crossing
    results = []
    for seed in range(20):
        rng = np.random.default_rng(3000 + seed)
        series = _window_series(rng)
        series["vpts"] = 295.0 + _ou_process(rng, 60000, tau_true)
        mask = np.ones(60000, dtype=bool)
        masks = {"momentum": mask, "heat": mask, "ts": mask}
        values, _ = rung_its_te(series, masks, g0=-15000, k=0, cfg_ladder=_cfg_ladder(), c=0.75)
        by_key = {(r, v, s): val for r, v, s, val in values}
        results.append(by_key[(RUNG_SECONDS[7], "vpts", "its")])

    assert np.mean(results) == pytest.approx(expected, rel=0.10)


def test_its_ou_process_9_375s_rung_biased_or_nan():
    # max_lag at this rung is floor(0.25 * 469) = 117 samples = 2.34 s, well
    # under tau=5s, so the pooled ACF may never cross 1/e within the window;
    # where it does, noise-driven early crossings bias the estimate low.
    tau_true = 5.0
    expected = (1 - 1 / np.e) * tau_true
    results = []
    for seed in range(20):
        rng = np.random.default_rng(4000 + seed)
        series = _window_series(rng)
        series["vpts"] = 295.0 + _ou_process(rng, 60000, tau_true)
        mask = np.ones(60000, dtype=bool)
        masks = {"momentum": mask, "heat": mask, "ts": mask}
        values, _ = rung_its_te(series, masks, g0=-15000, k=0, cfg_ladder=_cfg_ladder(), c=0.75)
        by_key = {(r, v, s): val for r, v, s, val in values}
        results.append(by_key[(RUNG_SECONDS[0], "vpts", "its")])

    assert all(np.isnan(r) or r < expected for r in results)


# --- transport efficiency ------------------------------------------------------

def test_te_all_same_sign_gives_one():
    n = 30000
    rng = np.random.default_rng(20)
    mean_ue, mean_vn = 4.0, 3.0
    ue_fluct = rng.normal(0, 1, n)
    ue = mean_ue + ue_fluct
    vn = np.full(n, mean_vn)  # constant -> v' = 0 always, so phi doesn't matter for u'
    phi = np.arctan2(mean_vn, mean_ue)
    w = np.cos(phi) * ue_fluct  # w' = cos(phi)*ue_fluct = u' exactly -> u'w' = u'^2 >= 0 always
    theta = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    heat_mask = np.zeros(n, dtype=bool)
    block_idx = np.zeros(n, dtype=np.int64)

    te_uw, te_wt = _te_momentum_heat(ue, vn, w, theta, mask, heat_mask, block_idx, n_blocks=1, nominal_len=n, c=0.5)
    assert te_uw == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(te_wt)  # heat never valid here


def test_te_pools_across_blocks_matches_analytic_3_to_1():
    n_per_block = 100
    pattern = np.tile([1.0, -1.0], n_per_block // 2)
    ue = np.concatenate([5.0 + pattern, 5.0 + pattern])
    vn = np.zeros(2 * n_per_block)
    # block 0: w' = 3*pattern (same sign as u') -> u'w' = +3 every sample
    # block 1: w' = -1*pattern (opposite sign) -> u'w' = -1 every sample
    w = np.concatenate([10.0 + 3 * pattern, 20.0 - 1 * pattern])
    theta = np.zeros(2 * n_per_block)
    mask = np.ones(2 * n_per_block, dtype=bool)
    heat_mask = np.zeros(2 * n_per_block, dtype=bool)
    block_idx = np.concatenate([np.zeros(n_per_block, dtype=np.int64), np.ones(n_per_block, dtype=np.int64)])

    te_uw, _ = _te_momentum_heat(ue, vn, w, theta, mask, heat_mask, block_idx, n_blocks=2, nominal_len=n_per_block, c=0.5)
    # pooled: S_all = 300 - 100 = 200, S_pos = 300 (S_all > 0) -> TE = 200/300
    assert te_uw == pytest.approx(2.0 / 3.0, abs=1e-9)
