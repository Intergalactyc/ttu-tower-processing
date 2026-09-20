from types import SimpleNamespace

import numpy as np
import pytest

from ttu_tower.math.polar import bearing_to_vector, vector_to_bearing
from ttu_tower.primary.slotstats import slot_means


def _slow_dict(n, mask=None):
    mask = np.ones(n, dtype=bool) if mask is None else mask
    return {
        "t": (np.full(n, 290.0), mask),
        "rh": (np.full(n, 0.5), mask),
        "p": (np.full(n, 90.0), mask),
    }


def test_wind_direction_unit_mean_equals_mean_at_constant_speed():
    n = 30000
    bearing, speed = 137.0, 5.0
    ue0, vn0 = bearing_to_vector(speed, bearing)
    series = {
        "ue": np.full(n, ue0), "vn": np.full(n, vn0), "w": np.zeros(n),
        "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0),
    }
    mask = np.ones(n, dtype=bool)
    masks = {"momentum": mask, "ts": mask}
    cfg = SimpleNamespace(gust_periods_s=[])
    rows = slot_means(series, masks, _slow_dict(n), g0=0, k=0, cfg=cfg, c=0.75)
    by_key = {(v, s): val for v, s, val in rows}

    assert by_key[("wd", "mean")] == pytest.approx(by_key[("wd", "unit_mean")], abs=1e-9)
    assert by_key[("wd", "mean")] == pytest.approx(bearing, abs=1e-6)


def test_wind_direction_mean_follows_fast_unit_mean_near_middle():
    n = 30000
    half = n // 2
    slow_bearing, fast_bearing = 0.0, 90.0
    ue_slow, vn_slow = bearing_to_vector(0.1, slow_bearing)
    ue_fast, vn_fast = bearing_to_vector(10.0, fast_bearing)
    ue = np.concatenate([np.full(half, ue_slow), np.full(n - half, ue_fast)])
    vn = np.concatenate([np.full(half, vn_slow), np.full(n - half, vn_fast)])
    series = {"ue": ue, "vn": vn, "w": np.zeros(n), "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0)}
    mask = np.ones(n, dtype=bool)
    masks = {"momentum": mask, "ts": mask}
    cfg = SimpleNamespace(gust_periods_s=[])
    rows = slot_means(series, masks, _slow_dict(n), g0=0, k=0, cfg=cfg, c=0.75)
    by_key = {(v, s): val for v, s, val in rows}

    _, expected_mean = vector_to_bearing(ue.mean(), vn.mean())
    ws_full = np.hypot(ue, vn)
    _, expected_unit_mean = vector_to_bearing((ue / ws_full).mean(), (vn / ws_full).mean())
    assert by_key[("wd", "mean")] == pytest.approx(expected_mean, abs=1e-9)
    assert by_key[("wd", "unit_mean")] == pytest.approx(expected_unit_mean, abs=1e-9)

    def circ_dist(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    assert circ_dist(by_key[("wd", "mean")], fast_bearing) < circ_dist(by_key[("wd", "unit_mean")], fast_bearing)
    assert circ_dist(by_key[("wd", "unit_mean")], 45.0) < 5.0


def test_gusts_step_signal_known_maxima_and_invalid_block_skipped():
    n = 30000
    p = 30
    block_samples = 50 * p
    n_blocks = n // block_samples

    ue, vn, w = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n_blocks):
        u, v = bearing_to_vector(float(i + 1), 0.0)
        sl = slice(i * block_samples, (i + 1) * block_samples)
        ue[sl], vn[sl], w[sl] = u, v, float(i)

    mask = np.ones(n, dtype=bool)
    mask[(n_blocks - 1) * block_samples :] = False  # invalidate the block that would have the highest speed

    series = {"ue": ue, "vn": vn, "w": w, "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0)}
    masks = {"momentum": mask, "ts": np.ones(n, dtype=bool)}
    cfg = SimpleNamespace(gust_periods_s=[p])
    rows = slot_means(series, masks, _slow_dict(n), g0=0, k=0, cfg=cfg, c=0.75)
    by_key = {(v, s): val for v, s, val in rows}

    assert by_key[("ws", f"max_{p}s")] == pytest.approx(float(n_blocks - 1), abs=1e-6)
    assert by_key[("w", f"max_{p}s")] == pytest.approx(float(n_blocks - 2), abs=1e-6)


def test_gusts_std_nan_with_fewer_than_two_valid_blocks():
    n = 30000
    p = 60  # 3000 samples/block -> 10 blocks/slot
    block_samples = 50 * p
    n_blocks = n // block_samples
    ue, vn, w = np.full(n, 1.0), np.zeros(n), np.zeros(n)
    mask = np.zeros(n, dtype=bool)
    mask[:block_samples] = True  # only one valid block

    series = {"ue": ue, "vn": vn, "w": w, "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0)}
    masks = {"momentum": mask, "ts": np.ones(n, dtype=bool)}
    cfg = SimpleNamespace(gust_periods_s=[p])
    rows = slot_means(series, masks, _slow_dict(n), g0=0, k=0, cfg=cfg, c=0.75)
    by_key = {(v, s): val for v, s, val in rows}

    assert np.isnan(by_key[("ws", f"std_{p}s")])
    assert not np.isnan(by_key[("ws", f"max_{p}s")])


def test_means_use_their_own_masks():
    n = 30000
    ue, vn, w = np.full(n, 2.0), np.full(n, 1.0), np.full(n, 0.1)
    ts, vpts = np.full(n, 291.0), np.full(n, 296.0)
    mom_mask = np.ones(n, dtype=bool)
    ts_mask = np.zeros(n, dtype=bool)
    ts_mask[:15000] = True  # only half usable

    series = {"ue": ue, "vn": vn, "w": w, "ts": ts, "vpts": vpts}
    masks = {"momentum": mom_mask, "ts": ts_mask}
    t_mask = np.ones(n, dtype=bool)
    t_mask[:100] = False
    slow = {"t": (np.full(n, 280.0), t_mask), "rh": (np.full(n, 0.4), np.ones(n, dtype=bool)), "p": (np.full(n, 88.0), np.ones(n, dtype=bool))}
    cfg = SimpleNamespace(gust_periods_s=[])
    rows = slot_means(series, masks, slow, g0=0, k=0, cfg=cfg, c=0.75)
    by_key = {(v, s): val for v, s, val in rows}

    assert by_key[("ts", "mean")] == pytest.approx(291.0)
    assert by_key[("vpts", "mean")] == pytest.approx(296.0)
    assert by_key[("t", "mean")] == pytest.approx(280.0)
    assert by_key[("rh", "mean")] == pytest.approx(0.4)
    assert by_key[("p", "mean")] == pytest.approx(88.0)


def test_no_momentum_data_gives_nan_rows():
    n = 30000
    series = {"ue": np.zeros(n), "vn": np.zeros(n), "w": np.zeros(n), "ts": np.full(n, 290.0), "vpts": np.full(n, 295.0)}
    masks = {"momentum": np.zeros(n, dtype=bool), "ts": np.ones(n, dtype=bool)}
    cfg = SimpleNamespace(gust_periods_s=[10])
    rows = slot_means(series, masks, _slow_dict(n), g0=0, k=0, cfg=cfg, c=0.75)
    by_key = {(v, s): val for v, s, val in rows}
    for key in [("ue", "mean"), ("wd", "mean"), ("ws", "max"), ("ws", "max_10s")]:
        assert np.isnan(by_key[key])
    assert by_key[("ts", "mean")] == pytest.approx(290.0)
