import numpy as np
import pytest

from ttu_tower.math.polar import rotate_covariance_streamwise, streamwise_angle, yamartino_std
from ttu_tower.primary.blocks import finest_sums
from ttu_tower.primary.ladder import RUNG_SECONDS, rung_statistics
from ttu_tower.timegrid import BlockGrid, block_sums


def _random_window(rng):
    n = 60000  # the 128-finest-block window: [64k-32, 64k+96) = 60000 samples
    g0 = -15000  # slot k=0's window starts at 30000*0 - 15000
    series = {
        "ue": rng.normal(3, 1, n),
        "vn": rng.normal(1, 1, n),
        "w": rng.normal(0, 0.5, n),
        "ts": rng.normal(290, 1, n),
        "vpts": rng.normal(295, 1, n),
    }
    masks = {
        "momentum": rng.random(n) > 0.1,
        "heat": rng.random(n) > 0.1,
        "ts": rng.random(n) > 0.1,
    }
    return g0, series, masks


def _momentum_block_sums(series, mask, g0, grid, b0, nb):
    ue, vn, w = series["ue"], series["vn"], series["w"]
    cols = np.stack([ue, vn, w, ue**2, vn**2, w**2, ue * vn, ue * w, vn * w], axis=1)
    sx, n = block_sums(cols, mask.astype(float), g0, grid, b0, nb)
    ws = np.hypot(ue, vn)
    sx_ws, _ = block_sums(np.stack([ws, ws**2], axis=1), mask.astype(float), g0, grid, b0, nb)
    return sx, sx_ws, n


def _momentum_stats(sx, sx_ws, n, block_len, c):
    valid = (n / block_len) >= c
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ue, mean_vn, mean_w = sx[:, 0] / n, sx[:, 1] / n, sx[:, 2] / n
        var_ue = np.maximum(sx[:, 3] / n - mean_ue**2, 0.0)
        var_vn = np.maximum(sx[:, 4] / n - mean_vn**2, 0.0)
        var_w = np.maximum(sx[:, 5] / n - mean_w**2, 0.0)
        cov_ue_vn = sx[:, 6] / n - mean_ue * mean_vn
        cov_ue_w = sx[:, 7] / n - mean_ue * mean_w
        cov_vn_w = sx[:, 8] / n - mean_vn * mean_w
        mean_ws, mean_ws2 = sx_ws[:, 0] / n, sx_ws[:, 1] / n
        ws_var = np.maximum(mean_ws2 - mean_ws**2, 0.0)

    phi = streamwise_angle(mean_ue, mean_vn)
    u_var, v_var, uv_cov, uw_cov, vw_cov = rotate_covariance_streamwise(var_ue, var_vn, cov_ue_vn, cov_ue_w, cov_vn_w, phi)
    u_mean = np.hypot(mean_ue, mean_vn)

    def rmean(x):
        return float(x[valid].mean()) if valid.any() else np.nan

    return {
        ("u", "var"): rmean(u_var), ("v", "var"): rmean(v_var), ("w", "var"): rmean(var_w),
        ("uv", "cov"): rmean(uv_cov), ("uw", "cov"): rmean(uw_cov), ("vw", "cov"): rmean(vw_cov),
        ("u", "mean"): rmean(u_mean), ("w", "mean"): rmean(mean_w),
        ("ws", "mean"): rmean(mean_ws), ("ws", "std"): float(np.sqrt(rmean(ws_var))),
    }


def _heat_ts_block_sums(series, heat_mask, ts_mask, g0, grid, b0, nb):
    w, vpts, ts = series["w"], series["vpts"], series["ts"]
    theta, tau_s = vpts - 273.15, ts - 273.15
    sx_h, n_h = block_sums(np.stack([w, theta, theta**2, w * theta], axis=1), heat_mask.astype(float), g0, grid, b0, nb)
    sx_t, n_t = block_sums(np.stack([tau_s, tau_s**2], axis=1), ts_mask.astype(float), g0, grid, b0, nb)
    return sx_h, n_h, sx_t, n_t


def _heat_ts_stats(sx_h, n_h, sx_t, n_t, block_len, c):
    valid_h = (n_h / block_len) >= c
    valid_t = (n_t / block_len) >= c
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_w, mean_theta = sx_h[:, 0] / n_h, sx_h[:, 1] / n_h
        mean_theta2, mean_w_theta = sx_h[:, 2] / n_h, sx_h[:, 3] / n_h
        mean_tau, mean_tau2 = sx_t[:, 0] / n_t, sx_t[:, 1] / n_t

    wvpts_cov = mean_w_theta - mean_w * mean_theta
    vpts_var = np.maximum(mean_theta2 - mean_theta**2, 0.0)
    vpts_mean = mean_theta + 273.15
    ts_var = np.maximum(mean_tau2 - mean_tau**2, 0.0)
    ts_mean = mean_tau + 273.15

    def rmean(x, valid):
        return float(x[valid].mean()) if valid.any() else np.nan

    return {
        ("wvpts", "cov"): rmean(wvpts_cov, valid_h),
        ("vpts", "var"): rmean(vpts_var, valid_h),
        ("vpts", "mean"): rmean(vpts_mean, valid_h),
        ("ts", "var"): rmean(ts_var, valid_t),
        ("ts", "mean"): rmean(ts_mean, valid_t),
    }


def test_aggregation_matches_direct_computation_every_rung():
    c = 0.75
    for seed in range(20):
        rng = np.random.default_rng(seed)
        g0, series, masks = _random_window(rng)
        finest = finest_sums(series, masks, g0=g0, b0=-32, nb=128)

        values, _ = rung_statistics(finest, c)
        by_key = {(rung_s, var, stat): val for rung_s, var, stat, val in values}

        for j in range(8):
            rung_s = RUNG_SECONDS[j]
            if j < 7:
                # slot 0's own finest blocks start at global finest-block 0,
                # so a grid scaled up by the group size aligns block 0 with it.
                group = 2**j
                nb = 64 // group
                grid = BlockGrid(1875 * group, 4)
                mom_sx, mom_sx_ws, mom_n = _momentum_block_sums(series, masks["momentum"], g0, grid, 0, nb)
                ht_sx, ht_n, ts_sx, ts_n = _heat_ts_block_sums(series, masks["heat"], masks["ts"], g0, grid, 0, nb)
                block_len = grid.length_samples
            else:
                # the 20-min window is centered on the slot (finest blocks
                # [-32, 96)), not aligned with any block-0 of a scaled grid,
                # so sum 128 individual finest blocks together by hand instead.
                grid = BlockGrid.FINEST
                mom_sx, mom_sx_ws, mom_n = _momentum_block_sums(series, masks["momentum"], g0, grid, -32, 128)
                ht_sx, ht_n, ts_sx, ts_n = _heat_ts_block_sums(series, masks["heat"], masks["ts"], g0, grid, -32, 128)
                mom_sx, mom_sx_ws, mom_n = mom_sx.sum(0, keepdims=True), mom_sx_ws.sum(0, keepdims=True), mom_n.sum(keepdims=True)
                ht_sx, ht_n = ht_sx.sum(0, keepdims=True), ht_n.sum(keepdims=True)
                ts_sx, ts_n = ts_sx.sum(0, keepdims=True), ts_n.sum(keepdims=True)
                block_len = grid.length_samples * 128

            direct_mom = _momentum_stats(mom_sx, mom_sx_ws, mom_n, block_len, c)
            direct_ht = _heat_ts_stats(ht_sx, ht_n, ts_sx, ts_n, block_len, c)

            for key, expected in {**direct_mom, **direct_ht}.items():
                actual = by_key[(rung_s, *key)]
                if np.isnan(expected):
                    assert np.isnan(actual), (seed, j, key)
                else:
                    assert actual == pytest.approx(expected, rel=1e-10), (seed, j, key)


def test_rotation_aligned_flow_gives_zero_cross_covariance():
    rng = np.random.default_rng(100)
    n = 60000
    mean_speed, injected_v_var = 5.0, 0.7
    # flow aligned with east: fluctuations only in ue (streamwise) and an
    # independent v-like component injected orthogonally via vn... to keep
    # the mean aligned with east exactly, hold vn's mean at 0.
    ue = mean_speed + rng.normal(0, 1.0, n)
    vn = rng.normal(0, np.sqrt(injected_v_var), n)
    w = rng.normal(0, 0.3, n)
    series = {"ue": ue, "vn": vn, "w": w, "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0)}
    masks = {"momentum": np.ones(n, dtype=bool), "heat": np.ones(n, dtype=bool), "ts": np.ones(n, dtype=bool)}
    finest = finest_sums(series, masks, g0=-15000, b0=-32, nb=128)
    values, _ = rung_statistics(finest, c=0.75)
    by_key = {(rung_s, var, stat): val for rung_s, var, stat, val in values}

    v_var = by_key[(RUNG_SECONDS[7], "v", "var")]  # 20-min rung: most averaging, closest to injected value
    uv_cov = by_key[(RUNG_SECONDS[7], "uv", "cov")]
    assert v_var == pytest.approx(injected_v_var, rel=0.1)
    assert abs(uv_cov) < 0.05


def test_rotation_invariance_under_global_rotation():
    rng = np.random.default_rng(101)
    n = 60000
    ue = 3.0 + rng.normal(0, 1, n)
    vn = 1.0 + rng.normal(0, 1, n)
    w = 0.2 * (ue - ue.mean()) + rng.normal(0, 0.3, n)
    ts, vpts = rng.normal(290, 1, n), rng.normal(295, 1, n)
    masks = {"momentum": np.ones(n, dtype=bool), "heat": np.ones(n, dtype=bool), "ts": np.ones(n, dtype=bool)}

    def run(ue_in, vn_in):
        series = {"ue": ue_in, "vn": vn_in, "w": w, "ts": ts, "vpts": vpts}
        finest = finest_sums(series, masks, g0=-15000, b0=-32, nb=128)
        values, _ = rung_statistics(finest, c=0.75)
        return {(rung_s, var, stat): val for rung_s, var, stat, val in values}

    base = run(ue, vn)
    alpha = np.deg2rad(53.0)
    ue_r = ue * np.cos(alpha) - vn * np.sin(alpha)
    vn_r = ue * np.sin(alpha) + vn * np.cos(alpha)
    rotated = run(ue_r, vn_r)

    for key in base:
        if key[1] in ("u", "v", "w", "uv", "uw", "vw", "ws", "wd"):
            assert base[key] == pytest.approx(rotated[key], rel=1e-9, abs=1e-12, nan_ok=True) if not np.isnan(base[key]) else np.isnan(rotated[key])


def test_yamartino_matches_single_pass_formula_on_samples():
    rng = np.random.default_rng(102)
    n = 60000
    bearing_deg = rng.normal(200.0, 5.0, n)  # tight spread around 200 degrees
    speed = rng.uniform(1.0, 10.0, n)
    bearing_rad = np.deg2rad(bearing_deg)
    ue = speed * np.sin(bearing_rad + np.pi)  # blows-toward = from_bearing + 180
    vn = speed * np.cos(bearing_rad + np.pi)
    w = rng.normal(0, 0.3, n)
    series = {"ue": ue, "vn": vn, "w": w, "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0)}
    masks = {"momentum": np.ones(n, dtype=bool), "heat": np.ones(n, dtype=bool), "ts": np.ones(n, dtype=bool)}
    finest = finest_sums(series, masks, g0=-15000, b0=-32, nb=128)
    values, _ = rung_statistics(finest, c=0.75)
    by_key = {(rung_s, var, stat): val for rung_s, var, stat, val in values}
    wd_std = by_key[(RUNG_SECONDS[7], "wd", "std")]

    ws_sample = np.hypot(ue, vn)
    s_bar = np.mean(vn / ws_sample)
    c_bar = np.mean(ue / ws_sample)
    expected = yamartino_std(s_bar, c_bar)
    assert wd_std == pytest.approx(expected, rel=1e-9)


def test_coverage_blanking_one_block_drops_it_and_averages_the_rest():
    rng = np.random.default_rng(103)
    n = 60000
    series = {
        "ue": rng.normal(3, 0.2, n), "vn": rng.normal(1, 0.2, n), "w": rng.normal(0, 0.1, n),
        "ts": rng.normal(290, 0.1, n), "vpts": rng.normal(295, 0.1, n),
    }
    mask = np.ones(n, dtype=bool)
    masks = {"momentum": mask.copy(), "heat": mask.copy(), "ts": mask.copy()}
    g0, b0 = -15000, -32
    finest_a = finest_sums(series, masks, g0=g0, b0=b0, nb=128)
    values_a, coverage_a = rung_statistics(finest_a, c=0.75)

    # blank one whole finest block (global index 40, well inside the slot: the
    # slot's own finest blocks are global indices 0..63 here since k=0)
    masks_b = {k: v.copy() for k, v in masks.items()}
    finest_block_start = round(40 * 1875 / 4)
    finest_block_end = round(41 * 1875 / 4)
    local_lo, local_hi = finest_block_start - g0, finest_block_end - g0
    masks_b["momentum"][local_lo:local_hi] = False

    finest_b = finest_sums(series, masks_b, g0=g0, b0=b0, nb=128)
    values_b, coverage_b = rung_statistics(finest_b, c=0.75)

    cov_by_key_a = {(r, f): (used, total, cov) for r, f, used, total, cov in coverage_a}
    cov_by_key_b = {(r, f): (used, total, cov) for r, f, used, total, cov in coverage_b}

    # at the finest rung (j=0, group=1), exactly one block should have flipped from valid to invalid
    used_a, total_a, _ = cov_by_key_a[(RUNG_SECONDS[0], "momentum")]
    used_b, total_b, _ = cov_by_key_b[(RUNG_SECONDS[0], "momentum")]
    assert total_a == total_b
    assert used_b == used_a - 1
