import numpy as np

from ttu_tower.primary.gaps import fill_short, fill_within_runs, nan_runs


def test_nan_runs_interior_and_edge():
    x = np.array([np.nan, np.nan, 1.0, 2.0, np.nan, 3.0, np.nan])
    starts, ends = nan_runs(x)
    assert list(zip(starts.tolist(), ends.tolist())) == [(0, 2), (4, 5), (6, 7)]


def test_fill_short_50_vs_51():
    x = np.concatenate([[1.0], np.full(50, np.nan), [2.0], np.full(51, np.nan), [3.0]])
    filled, mask = fill_short(x, max_gap=50)
    assert not np.any(np.isnan(filled[1:51]))  # the 50-gap is filled
    assert np.all(mask[1:51])
    assert np.all(np.isnan(filled[52:103]))  # the 51-gap is left
    assert not np.any(mask[52:103])
    # linear interpolation check
    assert filled[1] > filled[2] > filled[3] or filled[1] < filled[2] < filled[3]


def test_fill_short_linear_values():
    x = np.array([0.0, np.nan, np.nan, np.nan, 4.0])
    filled, mask = fill_short(x, max_gap=3)
    assert np.allclose(filled, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(mask, [False, True, True, True, False])


def test_fill_short_edge_runs_not_filled():
    x = np.array([np.nan, np.nan, 1.0, 2.0, np.nan])
    filled, mask = fill_short(x, max_gap=5)
    assert np.isnan(filled[0]) and np.isnan(filled[1])
    assert np.isnan(filled[4])
    assert not mask.any()


def test_fill_within_runs_blocks_across_file_run_boundary():
    x = np.array([1.0, np.nan, np.nan, 2.0])
    file_run_id = np.array([5, 5, 6, 6])  # gap straddles a file-run change
    filled, mask = fill_within_runs(x, file_run_id, max_gap=5)
    assert np.isnan(filled[1]) and np.isnan(filled[2])
    assert not mask.any()


def test_fill_within_runs_fills_within_same_run():
    x = np.array([1.0, np.nan, np.nan, 2.0])
    file_run_id = np.array([5, 5, 5, 5])
    filled, mask = fill_within_runs(x, file_run_id, max_gap=5)
    assert np.allclose(filled, [1.0, 4 / 3, 5 / 3, 2.0])
    assert np.array_equal(mask, [False, True, True, False])


def test_fill_within_runs_longer_than_cap_not_filled():
    x = np.array([1.0, np.nan, np.nan, np.nan, 2.0])
    file_run_id = np.array([5, 5, 5, 5, 5])
    filled, mask = fill_within_runs(x, file_run_id, max_gap=2)
    assert np.isnan(filled[1:4]).all()
    assert not mask.any()


def test_fill_within_runs_placeholder_run_id_never_filled():
    x = np.array([1.0, np.nan, np.nan, 2.0])
    file_run_id = np.array([-1, -1, -1, -1])
    filled, mask = fill_within_runs(x, file_run_id, max_gap=5)
    assert np.isnan(filled[1:3]).all()
    assert not mask.any()
