import numpy as np
import pytest

from ttu_tower.config.model import SecondLayerConfig
from ttu_tower.math.polar import bearing_to_vector
from ttu_tower.primary.masks import family_masks, first_layer, second_layer
from ttu_tower.timegrid import SAMPLES_PER_SLOT


def test_first_layer_combines_finite_and_intervals():
    n = 100
    finite = {"ue": np.ones(n, dtype=bool), "t": np.ones(n, dtype=bool)}
    finite["ue"][50] = False  # not finite there
    intervals = {"ue": {"resolution": (np.array([10]), np.array([20]))}}
    out = first_layer(finite, intervals, unusable_tests=["resolution", "kurt"], g0=0, n=n)
    assert not out["ue"][10:20].any()
    assert out["ue"][20:50].all()
    assert not out["ue"][50]
    assert out["t"].all()  # no intervals recorded for t


def test_first_layer_excludes_direction_and_bounce():
    n = 50
    finite = {"ue": np.ones(n, dtype=bool)}
    intervals = {"ue": {"direction": (np.array([0]), np.array([n]))}}
    out = first_layer(finite, intervals, unusable_tests=["direction", "bounce"], g0=0, n=n)
    assert out["ue"].all()  # direction/bounce never applied at first layer


def test_first_layer_clips_intervals_to_span():
    n = 20
    finite = {"ue": np.ones(n, dtype=bool)}
    intervals = {"ue": {"unchecked": (np.array([90]), np.array([110]))}}  # global, span is g0=100..120
    out = first_layer(finite, intervals, unusable_tests=["unchecked"], g0=100, n=n)
    assert not out["ue"][:10].any()  # global [100,110) -> local [0,10)
    assert out["ue"][10:].all()


def test_family_masks():
    masks = {
        "ue": np.array([True, True, False]),
        "vn": np.array([True, True, True]),
        "w": np.array([True, False, True]),
        "vpts": np.array([True, True, True]),
        "ts": np.array([True, False, False]),
    }
    fam = family_masks(masks)
    assert fam["momentum"].tolist() == [True, False, False]
    assert fam["heat"].tolist() == [True, False, True]
    assert fam["ts"].tolist() == [True, False, False]


def _second_layer_cfg():
    return SecondLayerConfig(shadow_sector=(105.0, 170.0), bounce_max_ws_mean=30.0, bounce_max_ws_std=6.75)


def _constant_slot(speed, bearing, n=SAMPLES_PER_SLOT):
    ue, vn = bearing_to_vector(speed, bearing)
    return np.full(n, ue), np.full(n, vn)


def test_second_layer_direction_boundary_cases():
    cfg = _second_layer_cfg()
    for bearing, expect_direction in [(104.9, False), (105.1, True), (170.0, False)]:
        ue, vn = _constant_slot(10.0, bearing)
        mask = np.ones(SAMPLES_PER_SLOT, dtype=bool)
        wd, ws_mean, ws_std, direction_flag, bounce_flag = second_layer(
            ue, vn, mask, g0=0, slots=[0], cfg_second=cfg, c=0.75
        )
        assert direction_flag[0] == expect_direction, bearing
        assert not bounce_flag[0]
        assert wd[0] == pytest.approx(bearing, abs=1e-6)


def test_second_layer_bounce_on_mean():
    cfg = _second_layer_cfg()
    ue, vn = _constant_slot(30.1, 0.0)  # wd=0, outside shadow sector
    mask = np.ones(SAMPLES_PER_SLOT, dtype=bool)
    wd, ws_mean, ws_std, direction_flag, bounce_flag = second_layer(ue, vn, mask, g0=0, slots=[0], cfg_second=cfg, c=0.75)
    assert ws_mean[0] > 30.0
    assert bounce_flag[0]
    assert not direction_flag[0]


def test_second_layer_bounce_on_std():
    cfg = _second_layer_cfg()
    n = SAMPLES_PER_SLOT
    half = n // 2
    speeds = np.concatenate([np.full(half, 13.2), np.full(n - half, 26.8)])  # mean 20, pop std 6.8
    ue, vn = bearing_to_vector(speeds, np.zeros(n))
    mask = np.ones(n, dtype=bool)
    wd, ws_mean, ws_std, direction_flag, bounce_flag = second_layer(ue, vn, mask, g0=0, slots=[0], cfg_second=cfg, c=0.75)
    assert ws_std[0] > 6.75
    assert ws_mean[0] < 30.0
    assert bounce_flag[0]
    assert not direction_flag[0]


def test_second_layer_below_coverage_gives_nan_no_flags():
    cfg = _second_layer_cfg()
    n = SAMPLES_PER_SLOT
    ue, vn = _constant_slot(10.0, 150.0)  # would otherwise flag direction
    mask = np.zeros(n, dtype=bool)
    mask[: int(0.5 * n)] = True  # below c=0.75
    wd, ws_mean, ws_std, direction_flag, bounce_flag = second_layer(ue, vn, mask, g0=0, slots=[0], cfg_second=cfg, c=0.75)
    assert np.isnan(wd[0])
    assert not direction_flag[0]
    assert not bounce_flag[0]
