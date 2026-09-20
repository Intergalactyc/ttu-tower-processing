import numpy as np
import pandas as pd
import pytest

from ttu_tower import timegrid as tg


def test_sample_time_round_trip_random_instants():
    rng = np.random.default_rng(0)
    g_values = rng.integers(0, 2_000_000_000, size=200, dtype=np.int64)
    for g in g_values:
        t = tg.sample_to_time(int(g))
        assert tg.time_to_sample(t) == int(g)

    t_arr = tg.sample_to_time(g_values)
    g_back = tg.time_to_sample(t_arr)
    assert np.array_equal(g_back, g_values)


def test_sample_time_round_trip_specific_dates():
    for date_str in ["2012-02-29 12:34:56", "2014-11-01 00:00:00", "2014-11-01 01:30:00", "2014-11-02 02:00:00"]:
        t = pd.Timestamp(date_str, tz="Etc/GMT+6")
        g = tg.time_to_sample(t)
        assert tg.sample_to_time(g) == t


def test_time_to_sample_requires_tz_aware():
    with pytest.raises(ValueError):
        tg.time_to_sample(pd.Timestamp("2014-01-01 00:00:00"))


def test_time_to_sample_requires_tick_alignment():
    t = tg.sample_to_time(0) + pd.Timedelta(microseconds=1)
    with pytest.raises(ValueError):
        tg.time_to_sample(t)


def test_slot_and_half_hour_round_trip():
    for k in [0, 1, 123456]:
        assert tg.time_to_slot(tg.slot_to_time(k)) == k
    for h in [0, 1, 41152]:
        assert tg.time_to_half_hour(tg.half_hour_to_time(h)) == h

    ks = np.array([0, 5, 999])
    assert np.array_equal(tg.time_to_slot(tg.slot_to_time(ks)), ks)


def test_time_to_slot_requires_alignment():
    t = tg.slot_to_time(5) + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        tg.time_to_slot(t)


def test_time_to_half_hour_requires_alignment():
    t = tg.half_hour_to_time(3) + pd.Timedelta(minutes=10)
    with pytest.raises(ValueError):
        tg.time_to_half_hour(t)


def test_half_hour_equals_three_slots():
    h = 7
    assert tg.half_hour_to_time(h) == tg.slot_to_time(3 * h)


# --- block_sums --------------------------------------------------------------

def test_block_sums_integer_grid_equals_reshape_sum():
    rng = np.random.default_rng(1)
    length, nb = 50, 12
    n = length * nb
    values = rng.normal(size=(n, 3))
    weights = np.ones(n)
    grid = tg.BlockGrid.integer(length)
    sx, sw = tg.block_sums(values, weights, g0=0, grid=grid, b0=0, nb=nb)
    expected = values.reshape(nb, length, 3).sum(axis=1)
    assert np.allclose(sx, expected, rtol=1e-12)
    assert np.allclose(sw, length, rtol=1e-12)


def test_block_sums_floor_grid_weight_equals_num_over_den():
    grid = tg.BlockGrid.floor(15)
    nb = 4096  # exactly one slot
    n = 30000
    sx, sw = tg.block_sums(np.zeros(n), np.ones(n), g0=0, grid=grid, b0=0, nb=nb)
    assert np.allclose(sw, grid.num / grid.den, atol=1e-12)


def test_block_sums_finest_grid_weight_equals_num_over_den():
    grid = tg.BlockGrid.FINEST
    nb = 64  # exactly one slot
    n = 30000
    sx, sw = tg.block_sums(np.zeros(n), np.ones(n), g0=0, grid=grid, b0=0, nb=nb)
    assert np.allclose(sw, grid.num / grid.den, atol=1e-12)


def test_block_sums_total_matches_sample_total():
    rng = np.random.default_rng(2)
    grid = tg.BlockGrid.floor(15)
    n = 30000  # exactly 4096 floor blocks, no leftover
    values = rng.normal(size=n)
    weights = np.ones(n)
    sx, sw = tg.block_sums(values, weights, g0=0, grid=grid, b0=0, nb=4096)
    assert sx.sum() == pytest.approx(values.sum(), rel=1e-10)
    assert sw.sum() == pytest.approx(n, rel=1e-12)


def test_block_sums_translation_invariance():
    rng = np.random.default_rng(3)
    grid = tg.BlockGrid.floor(15)
    n_full = 30000
    values_full = rng.normal(size=n_full)
    weights_full = np.ones(n_full)
    sx_full, sw_full = tg.block_sums(values_full, weights_full, g0=0, grid=grid, b0=0, nb=4096)

    b0_sub, nb_sub = 1000, 2000
    g_start = int(np.floor(b0_sub * grid.num / grid.den))
    g_end = int(np.ceil((b0_sub + nb_sub) * grid.num / grid.den))
    sub_values = values_full[g_start:g_end]
    sub_weights = weights_full[g_start:g_end]
    sx_sub, sw_sub = tg.block_sums(sub_values, sub_weights, g0=g_start, grid=grid, b0=b0_sub, nb=nb_sub)

    assert np.array_equal(sx_sub, sx_full[b0_sub : b0_sub + nb_sub])
    assert np.array_equal(sw_sub, sw_full[b0_sub : b0_sub + nb_sub])


def test_block_sums_nan_with_zero_weight_does_not_propagate():
    grid = tg.BlockGrid.integer(10)
    n = 30
    values = np.zeros(n)
    values[5] = np.nan
    weights = np.ones(n)
    weights[5] = 0.0
    sx, sw = tg.block_sums(values, weights, g0=0, grid=grid, b0=0, nb=3)
    assert not np.any(np.isnan(sx))


def test_block_sums_asserts_full_coverage():
    grid = tg.BlockGrid.integer(10)
    with pytest.raises(AssertionError):
        tg.block_sums(np.zeros(15), np.ones(15), g0=0, grid=grid, b0=0, nb=3)  # only covers 1.5 blocks


def test_majority_block_matches_formula():
    grid = tg.BlockGrid.FINEST
    g = np.array([0, 100, 468, 469, 937])
    expected = ((2 * g + 1) * grid.den) // (2 * grid.num)
    assert np.array_equal(tg.majority_block(g, grid), expected)
