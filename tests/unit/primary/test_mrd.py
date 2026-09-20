import numpy as np
import pytest

from ttu_tower.primary.blocks import FamilySums
from ttu_tower.primary.mrd import detection_outputs, mrd_modes
from ttu_tower.timegrid import BlockGrid, block_sums


def test_identity_rung_sums_equal_direct_covariance():
    k = 15
    nb = 2**k
    block_len = BlockGrid.floor(k).length_samples
    c = 0.5  # window is fully valid regardless of c

    for seed in range(20):
        rng = np.random.default_rng(seed)
        mean_x = rng.normal(0, 1, nb)
        mean_y = rng.normal(0, 1, nb)
        n = np.full(nb, block_len)
        sx, sy = mean_x * n, mean_y * n

        value, se, n_pairs = mrd_modes(sx, sy, n, block_len, c)

        for j in range(8):  # rung tau_j = P_(j+6): the 8 ladder rungs
            i_max = j + 6
            mrd_sum = value[:i_max].sum()

            group_size = 2**i_max
            gx = mean_x.reshape(-1, group_size)
            gy = mean_y.reshape(-1, group_size)
            direct = np.mean((gx - gx.mean(axis=1, keepdims=True)) * (gy - gy.mean(axis=1, keepdims=True)), axis=1).mean()

            assert mrd_sum == pytest.approx(direct, rel=1e-10), (seed, j)

        full_sum = value.sum()
        window_cov = np.mean((mean_x - mean_x.mean()) * (mean_y - mean_y.mean()))
        assert full_sum == pytest.approx(window_cov, rel=1e-10), seed


def test_floor_invariance_raw_samples_vs_preaveraged_blocks():
    rng = np.random.default_rng(100)
    n_samples = 2**18
    x = rng.normal(0, 1, n_samples)
    y = rng.normal(0, 1, n_samples)
    weights = np.ones(n_samples)

    grid_raw = BlockGrid.integer(1)
    sx_raw, n_raw = block_sums(x, weights, 0, grid_raw, 0, n_samples)
    sy_raw, _ = block_sums(y, weights, 0, grid_raw, 0, n_samples)
    value_raw, se_raw, np_raw = mrd_modes(sx_raw, sy_raw, n_raw, grid_raw.length_samples, c=0.8)

    grid_8 = BlockGrid.integer(8)
    nb_8 = n_samples // 8
    sx_8, n_8 = block_sums(x, weights, 0, grid_8, 0, nb_8)
    sy_8, _ = block_sums(y, weights, 0, grid_8, 0, nb_8)
    value_8, se_8, np_8 = mrd_modes(sx_8, sy_8, n_8, grid_8.length_samples, c=0.8)

    for i8 in range(1, len(value_8) + 1):
        i_raw = i8 + 3  # scale = 2^i8 * 8 samples = 2^(i8+3) * 1 sample
        assert np_8[i8 - 1] == np_raw[i_raw - 1]
        assert value_8[i8 - 1] == pytest.approx(value_raw[i_raw - 1], rel=1e-9)
        if np_8[i8 - 1] >= 2:
            assert se_8[i8 - 1] == pytest.approx(se_raw[i_raw - 1], rel=1e-9)


def test_valid_pair_rule_matches_table():
    rng = np.random.default_rng(2)
    k = 15
    nb = 2**k
    block_len = BlockGrid.floor(k).length_samples
    c = 0.8  # so no coverage lands exactly on the threshold

    blank_len_blocks = round(75 * 50 / block_len)
    assert blank_len_blocks == 512

    n = np.full(nb, block_len)
    n[:blank_len_blocks] = 0.0
    x, y = rng.normal(0, 1, nb), rng.normal(0, 1, nb)
    sx, sy = x * n, y * n  # zero contribution where n=0

    _, _, n_pairs = mrd_modes(sx, sy, n, block_len, c)

    expected = {i: 2 ** (15 - i) - 256 // 2 ** (i - 1) for i in range(1, 10)}
    expected |= {10: 31, 11: 15, 12: 7, 13: 4, 14: 2, 15: 1}

    for i in range(1, 16):
        assert n_pairs[i - 1] == expected[i], i


def test_white_noise_mode_expectation():
    k = 15
    nb = 2**k
    block_len = BlockGrid.floor(k).length_samples
    c = 0.5
    n_windows = 200

    values = np.empty((n_windows, k))
    for w_idx in range(n_windows):
        rng = np.random.default_rng(1000 + w_idx)
        x = rng.normal(0, 1, nb)
        n = np.full(nb, block_len)
        value, _, _ = mrd_modes(x * n, x * n, n, block_len, c)
        values[w_idx] = value

    mean_value = values.mean(axis=0)
    se_value = values.std(axis=0, ddof=1) / np.sqrt(n_windows)
    expected = 1.0 / (2 * 2.0 ** np.arange(k))

    assert np.all(np.abs(mean_value - expected) <= 3 * se_value)


def test_se_matches_brute_force_recomputation():
    rng = np.random.default_rng(3)
    k = 15
    nb = 2**k
    block_len = BlockGrid.floor(k).length_samples
    c = 0.7

    x, y = rng.normal(0, 1, nb), rng.normal(0, 1, nb)
    n = np.full(nb, block_len)
    blank = rng.random(nb) < 0.1
    n = np.where(blank, 0.0, n)
    sx, sy = np.where(blank, 0.0, x * n), np.where(blank, 0.0, y * n)

    value, se, n_pairs = mrd_modes(sx, sy, n, block_len, c)

    i = 5  # spot-check one interior mode
    h = 2 ** (i - 1)
    n_halves = nb // h
    half_x = sx.reshape(n_halves, h).sum(axis=1)
    half_y = sy.reshape(n_halves, h).sum(axis=1)
    half_n = n.reshape(n_halves, h).sum(axis=1)
    half_valid = (half_n / (h * block_len)) >= c
    with np.errstate(invalid="ignore"):
        mean_x, mean_y = half_x / half_n, half_y / half_n

    p_list = [
        (mean_x[2 * bj + 1] - mean_x[2 * bj]) * (mean_y[2 * bj + 1] - mean_y[2 * bj]) / 4
        for bj in range(n_halves // 2)
        if half_valid[2 * bj] and half_valid[2 * bj + 1]
    ]
    p_arr = np.array(p_list)

    assert n_pairs[i - 1] == p_arr.size
    assert value[i - 1] == pytest.approx(p_arr.mean(), rel=1e-10)
    assert se[i - 1] == pytest.approx(p_arr.std(ddof=1) / np.sqrt(p_arr.size), rel=1e-10)


# --- detection_outputs / rotation --------------------------------------------

def _empty_family(nb, columns):
    return FamilySums(n=np.zeros(nb), sx=np.zeros((nb, len(columns))), columns=columns)


def test_westerly_wind_gives_wd_270():
    nb = 2**15
    block_len = BlockGrid.floor(15).length_samples
    n = np.full(nb, block_len)
    momentum = FamilySums(
        n=n, sx=np.stack([np.full(nb, block_len * 5.0), np.zeros(nb), np.zeros(nb)], axis=1),
        columns=("ue", "vn", "w"),
    )
    floor = {"momentum": momentum, "heat": _empty_family(nb, ("w", "theta")), "ts": _empty_family(nb, ("tau_s",))}
    _, wd_deg, _, _ = detection_outputs(floor, c=0.5)
    assert wd_deg == pytest.approx(270.0, abs=1e-9)


def test_no_momentum_weight_gives_nan_spectra_and_wd():
    nb = 2**15
    floor = {
        "momentum": _empty_family(nb, ("ue", "vn", "w")),
        "heat": _empty_family(nb, ("w", "theta")),
        "ts": _empty_family(nb, ("tau_s",)),
    }
    spectra, wd_deg, cov_m, cov_h = detection_outputs(floor, c=0.5)
    assert np.isnan(wd_deg)
    for key in ("uu", "vv", "ww", "uw", "vw", "uv"):
        value, se, n_pairs = spectra[key]
        assert np.isnan(value).all()
        assert (n_pairs == 0).all()
    assert cov_m == 0.0


def _make_floor(ue, vn, w, ts, vpts, n):
    momentum = FamilySums(n=n, sx=np.stack([ue * n, vn * n, w * n], axis=1), columns=("ue", "vn", "w"))
    theta = vpts - 273.15
    heat = FamilySums(n=n, sx=np.stack([w * n, theta * n], axis=1), columns=("w", "theta"))
    tau_s = ts - 273.15
    ts_fam = FamilySums(n=n, sx=(tau_s * n)[:, None], columns=("tau_s",))
    return {"momentum": momentum, "heat": heat, "ts": ts_fam}


def test_frame_invariance_under_global_rotation():
    rng = np.random.default_rng(4)
    nb = 2**15
    block_len = BlockGrid.floor(15).length_samples
    n = np.full(nb, block_len)

    ue = 3.0 + rng.normal(0, 1, nb)
    vn = 1.5 + rng.normal(0, 1, nb)
    w = 0.3 * (ue - ue.mean()) + 0.5 * rng.normal(0, 1, nb)
    ts = rng.normal(290, 1, nb)
    vpts = rng.normal(295, 1, nb)

    floor_a = _make_floor(ue, vn, w, ts, vpts, n)
    spectra_a, wd_a, cm_a, ch_a = detection_outputs(floor_a, c=0.5)

    alpha = np.deg2rad(37.0)
    ue_r = ue * np.cos(alpha) - vn * np.sin(alpha)
    vn_r = ue * np.sin(alpha) + vn * np.cos(alpha)
    floor_b = _make_floor(ue_r, vn_r, w, ts, vpts, n)
    spectra_b, wd_b, cm_b, ch_b = detection_outputs(floor_b, c=0.5)

    for key in ("uu", "vv", "ww", "uw", "vw", "uv", "wvpts", "tsts"):
        va, sea, na = spectra_a[key]
        vb, seb, nb_ = spectra_b[key]
        assert np.allclose(va, vb, rtol=1e-9, equal_nan=True), key
        assert np.array_equal(na, nb_), key
    assert cm_a == cm_b and ch_a == ch_b
    # a counterclockwise (standard math convention) rotation of (ue, vn) by
    # alpha is a clockwise-bearing decrease of alpha.
    assert wd_b == pytest.approx((wd_a - 37.0) % 360, abs=1e-6)


def test_rotation_applied_as_vector_to_uw_vw():
    rng = np.random.default_rng(5)
    nb = 2**15
    block_len = BlockGrid.floor(15).length_samples
    n = np.full(nb, block_len)

    ue = 4.0 + rng.normal(0, 1, nb)
    vn = 3.0 + rng.normal(0, 1, nb)
    w = 0.3 * (ue - ue.mean()) - 0.2 * (vn - vn.mean()) + 0.5 * rng.normal(0, 1, nb)
    ts = np.full(nb, 290.0)
    vpts = np.full(nb, 295.0)

    floor = _make_floor(ue, vn, w, ts, vpts, n)
    spectra, wd_deg, _, _ = detection_outputs(floor, c=0.5)
    uw_actual, _, _ = spectra["uw"]
    vw_actual, _, _ = spectra["vw"]

    total_ue, total_vn = (ue * n).sum(), (vn * n).sum()
    phi = np.arctan2(total_vn, total_ue)

    raw_uw, _, _ = mrd_modes(ue * n, w * n, n, block_len, 0.5)
    raw_vw, _, _ = mrd_modes(vn * n, w * n, n, block_len, 0.5)
    expected_uw = np.cos(phi) * raw_uw + np.sin(phi) * raw_vw
    expected_vw = -np.sin(phi) * raw_uw + np.cos(phi) * raw_vw

    assert np.allclose(uw_actual, expected_uw, rtol=1e-9, equal_nan=True)
    assert np.allclose(vw_actual, expected_vw, rtol=1e-9, equal_nan=True)
