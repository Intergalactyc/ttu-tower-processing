import numpy as np
import pytest

from ttu_tower.flags import BOOM_LEVEL_VARIABLES, FlagRows, FlagStore, mask_to_intervals


def test_mask_to_intervals_runs():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    starts, ends = mask_to_intervals(mask, g0=100)
    assert list(zip(starts.tolist(), ends.tolist())) == [(101, 103), (105, 106), (107, 110)]


def test_mask_to_intervals_edges():
    mask = np.array([1, 1, 0, 0, 1], dtype=bool)
    starts, ends = mask_to_intervals(mask, g0=0)
    assert list(zip(starts.tolist(), ends.tolist())) == [(0, 2), (4, 5)]


def test_mask_to_intervals_empty():
    starts, ends = mask_to_intervals(np.array([], dtype=bool), g0=0)
    assert len(starts) == 0 and len(ends) == 0


def test_flagrows_merges_touching_and_overlapping():
    rows = FlagRows()
    rows.add("bounds", 1, "ue", np.array([0, 10, 25]), np.array([10, 20, 30]))
    frame = rows.frame()
    sub = frame[(frame["test"] == "bounds") & (frame["boom"] == 1) & (frame["variable"] == "ue")]
    intervals = sorted(zip(sub["start"].tolist(), sub["end"].tolist()))
    assert intervals == [(0, 20), (25, 30)]


def test_flagrows_frame_kind_and_dtypes():
    rows = FlagRows()
    rows.add("excursion", 5, "w", np.array([0]), np.array([5]))
    frame = rows.frame()
    assert frame["kind"].iloc[0] == "record"
    assert str(frame["test"].dtype) == "category"


def test_flagrows_frame_empty():
    frame = FlagRows().frame()
    assert list(frame.columns) == ["start", "end", "test", "kind", "boom", "variable"]
    assert len(frame) == 0


def _build_store(rows) -> FlagStore:
    accumulator = FlagRows()
    for test, boom, variable, starts, ends in rows:
        accumulator.add(test, boom, variable, np.array(starts), np.array(ends))
    return FlagStore(accumulator.frame())


def test_flagstore_count_matches_brute_force_random():
    rng = np.random.default_rng(0)
    n = 2000
    mask = rng.random(n) < 0.1
    starts, ends = mask_to_intervals(mask, g0=0)
    store = _build_store([("resolution", 1, "ue", starts, ends)])

    qs = rng.integers(0, n - 100, size=30)
    qe = qs + rng.integers(1, 100, size=30)
    counts = store.count("resolution", 1, "ue", qs, qe)

    brute = np.array([mask[a:b].sum() for a, b in zip(qs, qe)])
    assert np.array_equal(counts, brute)


def test_flagstore_boom_level_included_for_sonic_variables_only():
    store = _build_store([("direction", 1, None, [10], [20])])
    for var in BOOM_LEVEL_VARIABLES:
        assert store.count("direction", 1, var, [0], [30])[0] == 10
    assert store.count("direction", 1, "t", [0], [30])[0] == 0


def test_flagstore_intervals_split_across_fragments_match_merged():
    split = _build_store([("bounds", 2, "ue", [100, 150], [150, 200])])
    whole = _build_store([("bounds", 2, "ue", [100], [200])])
    qs, qe = [0], [1000]
    assert split.count("bounds", 2, "ue", qs, qe)[0] == whole.count("bounds", 2, "ue", qs, qe)[0] == 100


def test_flagstore_fraction():
    store = _build_store([("skew", 3, "w", [50], [60])])
    frac = store.fraction("skew", 3, "w", [0], [100])
    assert frac[0] == pytest.approx(0.1)


def test_flagstore_mask():
    store = _build_store([("resolution", 4, "ts", [5, 15], [10, 18])])
    m = store.mask(["resolution"], 4, "ts", g0=0, n=20)
    expected = np.zeros(20, dtype=bool)
    expected[5:10] = True
    expected[15:18] = True
    assert np.array_equal(m, expected)


def test_flagstore_mask_unions_multiple_tests():
    store = _build_store([
        ("skew", 1, "ue", [0], [5]),
        ("kurt", 1, "ue", [10], [15]),
    ])
    m = store.mask(["skew", "kurt"], 1, "ue", g0=0, n=20)
    assert m[:5].all() and m[10:15].all() and not m[5:10].any() and not m[15:].any()


def test_flagstore_empty_frame():
    store = FlagStore(FlagRows().frame())
    assert store.count("bounds", 1, "ue", [0], [10])[0] == 0
    assert not store.mask(["bounds"], 1, "ue", g0=0, n=10).any()
