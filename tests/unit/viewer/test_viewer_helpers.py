import threading
import time

import numpy as np
import pandas as pd
import pytest

from ttu_tower.timegrid import slot_to_time, time_to_slot
from ttu_tower.viewer.cache import LRUCache
from ttu_tower.viewer.catalog import Quantity
from ttu_tower.viewer.provenance import drill_target, ordered_variables
from ttu_tower.viewer.timeaxis import display_offset_s, parse_time, slot_to_unix, unix_to_slot


def test_slot_unix_round_trip():
    k = np.array([0, 728857, 780191])
    x = slot_to_unix(k)
    assert (unix_to_slot(x) == k).all()
    assert (unix_to_slot(x + 599.9) == k).all()
    assert pd.Timestamp(x[1], unit="s", tz="UTC") == slot_to_time(728857)


def test_display_offset_sign_is_positive_west():
    assert display_offset_s("Etc/GMT+6") == 6 * 3600
    assert display_offset_s("UTC") == 0


def test_parse_time_floors_to_the_slot_in_the_given_zone():
    t = parse_time("2013-11-09 12:14", "Etc/GMT+6")
    assert time_to_slot(t) == time_to_slot(pd.Timestamp("2013-11-09 12:10", tz="Etc/GMT+6"))


def test_cache_evicts_least_recently_used():
    cache = LRUCache(2)
    cache.get("a", lambda: 1)
    cache.get("b", lambda: 2)
    cache.get("a", lambda: -1)  # touch a
    cache.get("c", lambda: 3)
    assert "a" in cache and "c" in cache and "b" not in cache


def test_cache_shares_one_computation_between_threads():
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.05)
        return 42

    cache = LRUCache(4)
    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get("k", slow))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [42] * 5 and calls == [1]


def test_cache_failure_is_not_cached():
    cache = LRUCache(2)
    with pytest.raises(ValueError):
        cache.get("k", lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert cache.get("k", lambda: 7) == 7


def _q(table, variable, stat=None, kind="boom"):
    return Quantity(key="x", table=table, variable=variable, stat=stat, kind=kind, variants=("mrd",), label="",
                    unit="", group="")


@pytest.mark.parametrize("table,variable,stat,kind,expected", [
    ("boom_final", "ws", "mean", "series", ("ue", "vn", "w")),
    ("boom_final", "ts", "mean", "series", ("ts",)),
    ("boom_final", "t", "mean", "series", ("t",)),
    ("boom_final", "u", "its", "acf", ("ue", "vn", "w")),
    ("boom_final", "ils_vpts", None, "acf", ("vpts",)),
    ("boom_final", "uw", "cov", "flux", ("w", "ue", "vn")),
    ("boom_final", "wvpts", "cov", "flux", ("w", "vpts")),
    ("boom_final", "sigma_w", None, "fluctuation", ("ue", "vn", "w")),
    ("tau_final", "tau", None, "spectra", ("w", "ue", "vn")),
    ("coverage", "heat", None, "qc", ("w", "vpts")),
])
def test_drill_targets(table, variable, stat, kind, expected):
    target = drill_target(_q(table, variable, stat))
    assert target.kind == kind
    assert target.variables == expected


def test_pair_and_profile_quantities_point_at_the_profile():
    assert drill_target(_q("pairs", "rib", kind="pair")).kind == "profile"
    assert drill_target(_q("slot_final", "night", kind="slot")).kind == "none"


def test_ordered_variables_leads_with_the_target():
    target = drill_target(_q("boom_final", "ts", "mean"))
    assert ordered_variables(target, ["ue", "vn", "w", "ts", "t"]) == ["ts", "ue", "vn", "w", "t"]


def test_minmax_decimate_keeps_spikes_and_gaps():
    from ttu_tower.viewer.decimate import minmax_decimate

    x = np.arange(100_000, dtype=float)
    y = np.zeros_like(x)
    y[1::2] = np.nan  # NaN-dense, as a filtered timeline is
    y[50_000] = 9.0
    y[20_000:30_000] = np.nan  # a real gap
    dx, dy = minmax_decimate(x, y, 0, x[-1], buckets=500)
    assert dx.size <= 1002
    assert np.nanmax(dy) == 9.0  # the spike survives
    assert np.isfinite(dy).mean() > 0.85  # NaN-dense buckets still draw
    gap = (dx > 20_500) & (dx < 29_500)
    assert np.isnan(dy[gap]).all()


def test_minmax_decimate_returns_raw_points_when_they_fit():
    from ttu_tower.viewer.decimate import minmax_decimate

    x = np.arange(100, dtype=float)
    y = np.sin(x)
    dx, dy = minmax_decimate(x, y, 10, 20, buckets=500)
    assert dx[0] == 9 and dx[-1] == 21  # one sample of margin on each side
    np.testing.assert_array_equal(dy, y[9:22])


def test_break_wraps_splits_lines_at_the_0_360_seam():
    from ttu_tower.viewer.decimate import break_wraps

    x = np.arange(5, dtype=float)
    y = np.array([350.0, 355.0, 3.0, 8.0, 12.0])
    bx, by = break_wraps(x, y)
    assert bx.size == 6 and np.isnan(by[2]) and bx[2] == 1.5
    assert np.array_equal(by[[0, 1, 3, 4, 5]], y)


def test_circular_summary_uses_the_vector_mean():
    pytest.importorskip("PySide6")
    from ttu_tower.viewer.ui.distribution import summary

    n, nan_pct, mean, median, sigma, p5, p95 = summary(np.array([350.0, 10.0, np.nan]), circular=True)
    assert n == 2 and nan_pct == pytest.approx(100 / 3)
    assert min(mean, 360 - mean) < 1e-9  # 0°, not the linear 180°
    assert 0 < sigma < 15 and np.isnan(median)
