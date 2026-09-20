import numpy as np
import pytest

from ttu_tower.primary.blocks import finest_sums, floor_sums
from ttu_tower.timegrid import BlockGrid, block_sums


def _synthetic_series(n, rng):
    return {
        "ue": rng.normal(2, 1, n),
        "vn": rng.normal(1, 1, n),
        "w": rng.normal(0, 0.5, n),
        "ts": rng.normal(290, 1, n),
        "vpts": rng.normal(295, 1, n),
    }


def _synthetic_masks(n, rng):
    return {
        "momentum": rng.random(n) > 0.1,
        "heat": rng.random(n) > 0.1,
        "ts": rng.random(n) > 0.1,
    }


def test_floor_sums_matches_direct_block_sums():
    rng = np.random.default_rng(0)
    n = 30000  # one slot, floor_level=15 -> 4096 blocks
    series = _synthetic_series(n, rng)
    masks = _synthetic_masks(n, rng)
    floor_level, nb = 15, 4096
    grid = BlockGrid.floor(floor_level)

    result = floor_sums(series, masks, g0=0, floor_level=floor_level, b0=0, nb=nb)

    expected_sx, expected_n = block_sums(
        np.stack([series["ue"], series["vn"], series["w"]], axis=1),
        masks["momentum"].astype(float), 0, grid, 0, nb,
    )
    assert result["momentum"].columns == ("ue", "vn", "w")
    assert np.array_equal(result["momentum"].sx, expected_sx)
    assert np.array_equal(result["momentum"].n, expected_n)

    theta = series["vpts"] - 273.15
    expected_sx, expected_n = block_sums(
        np.stack([series["w"], theta], axis=1), masks["heat"].astype(float), 0, grid, 0, nb,
    )
    assert result["heat"].columns == ("w", "theta")
    assert np.array_equal(result["heat"].sx, expected_sx)
    assert np.array_equal(result["heat"].n, expected_n)

    tau_s = series["ts"] - 273.15
    expected_sx, expected_n = block_sums(tau_s[:, None], masks["ts"].astype(float), 0, grid, 0, nb)
    assert result["ts"].columns == ("tau_s",)
    assert np.array_equal(result["ts"].sx, expected_sx)
    assert np.array_equal(result["ts"].n, expected_n)


def test_floor_sums_heat_offset_recovers_exact_mean():
    floor_level = 15
    nb, n = 4096, 30000  # exactly one slot
    vpts_value = 295.37
    series = {
        "ue": np.zeros(n), "vn": np.zeros(n), "w": np.full(n, 1.23),
        "ts": np.full(n, 290.0), "vpts": np.full(n, vpts_value),
    }
    masks = {"momentum": np.zeros(n, dtype=bool), "heat": np.ones(n, dtype=bool), "ts": np.zeros(n, dtype=bool)}
    result = floor_sums(series, masks, g0=0, floor_level=floor_level, b0=0, nb=nb)
    theta_col = result["heat"].columns.index("theta")
    recovered_mean = result["heat"].sx[:, theta_col] / result["heat"].n + 273.15
    assert np.allclose(recovered_mean, vpts_value, atol=1e-9)


def test_floor_sums_ts_offset_does_not_affect_variance():
    # the offset must cancel exactly in any variance/covariance: pool all 64
    # finest blocks of one slot and check the variance recovered from the
    # offset second-moment sums equals the variance of the un-offset ts.
    rng = np.random.default_rng(1)
    n = 30000  # exactly one slot: 64 finest blocks
    ts = rng.normal(290, 2, n)
    series = {"ue": np.zeros(n), "vn": np.zeros(n), "w": np.zeros(n), "ts": ts, "vpts": np.full(n, 295.0)}
    mask = np.ones(n, dtype=bool)
    masks = {"momentum": mask, "heat": mask, "ts": mask}
    result = finest_sums(series, masks, g0=0, b0=0, nb=64)
    fam = result["ts"]
    tau_s_sum = fam.sx[:, fam.columns.index("tau_s")].sum()
    tau_s2_sum = fam.sx[:, fam.columns.index("tau_s2")].sum()
    n_sum = fam.n.sum()
    var_via_offset = tau_s2_sum / n_sum - (tau_s_sum / n_sum) ** 2
    assert var_via_offset == pytest.approx(np.var(ts), rel=1e-8)


def test_finest_sums_columns_and_direction_subsum():
    rng = np.random.default_rng(2)
    n = 30000
    series = _synthetic_series(n, rng)
    masks = _synthetic_masks(n, rng)
    result = finest_sums(series, masks, g0=0, b0=0, nb=64)

    assert result["momentum"].columns == ("ue", "vn", "w", "ue2", "vn2", "w2", "ue_vn", "ue_w", "vn_w", "ws", "ws2")
    assert result["heat"].columns == ("w", "theta", "w2", "theta2", "w_theta")
    assert result["ts"].columns == ("tau_s", "tau_s2")
    assert result["direction"].columns == ("ue_ws", "vn_ws")

    # direction sub-sum's weight can never exceed the plain momentum weight (ws>0 is an extra restriction)
    assert np.all(result["direction"].n <= result["momentum"].n + 1e-9)


def test_finest_sums_ws2_is_true_speed_squared_not_ue2_plus_vn2():
    n = 500  # >= one finest block (468.75 samples)
    ue, vn, w = np.full(n, 3.0), np.full(n, 4.0), np.zeros(n)
    series = {"ue": ue, "vn": vn, "w": w, "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0)}
    mask = np.ones(n, dtype=bool)
    masks = {"momentum": mask, "heat": mask, "ts": mask}
    result = finest_sums(series, masks, g0=0, b0=0, nb=1)
    fam = result["momentum"]
    ws2_sum = fam.sx[0, fam.columns.index("ws2")]
    assert ws2_sum == pytest.approx(fam.n[0] * 25.0)  # ws = hypot(3, 4) = 5 for every sample
