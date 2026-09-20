"""The two-layer usable mask and the combined per-family masks."""
import numpy as np

from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.timegrid import SAMPLES_PER_SLOT

_SECOND_LAYER_TESTS = ("direction", "bounce")


def first_layer(finite: dict[str, np.ndarray], intervals: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
                 unusable_tests: list[str], g0: int, n: int) -> dict[str, np.ndarray]:
    """Per variable: finite (after short gap filling) and outside every
    interval of a window test in `unusable_tests`, excluding `direction` and
    `bounce` (computed from first-layer means, so they can't feed themselves).

    `intervals[variable][test]` holds that (test, variable)'s global
    [start, end) sample intervals.
    """
    tests = [t for t in unusable_tests if t not in _SECOND_LAYER_TESTS]
    result = {}
    for var, finite_mask in finite.items():
        bad = np.zeros(n, dtype=bool)
        var_intervals = intervals.get(var, {})
        for test in tests:
            starts, ends = var_intervals.get(test, ((), ()))
            for s, e in zip(starts, ends):
                a, b = max(s, g0) - g0, min(e, g0 + n) - g0
                if a < b:
                    bad[a:b] = True
        result[var] = finite_mask & ~bad
    return result


def family_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """momentum = ue & vn & w; heat = w & vpts; ts = ts."""
    return {
        "momentum": masks["ue"] & masks["vn"] & masks["w"],
        "heat": masks["w"] & masks["vpts"],
        "ts": masks["ts"],
    }


def second_layer(ue: np.ndarray, vn: np.ndarray, momentum_mask_l1: np.ndarray, g0: int, slots, cfg_second, c: float):
    """Per slot, only where first-layer momentum coverage >= c: wd of the
    vector mean, and the mean/population-std of per-sample horizontal speed.
    `direction` if wd falls in the (open) shadow sector; `bounce` if the mean
    or std exceeds its limit.
    """
    slots = list(slots)
    n_slots = len(slots)
    wd = np.full(n_slots, np.nan)
    ws_mean = np.full(n_slots, np.nan)
    ws_std = np.full(n_slots, np.nan)
    direction_flag = np.zeros(n_slots, dtype=bool)
    bounce_flag = np.zeros(n_slots, dtype=bool)

    low, high = cfg_second.shadow_sector

    for i, k in enumerate(slots):
        start = SAMPLES_PER_SLOT * int(k) - g0
        sl = slice(start, start + SAMPLES_PER_SLOT)
        mask = momentum_mask_l1[sl]
        if mask.sum() < c * SAMPLES_PER_SLOT:
            continue
        ue_s, vn_s = ue[sl][mask], vn[sl][mask]
        ws_sample = np.hypot(ue_s, vn_s)
        _, wd_i = vector_to_bearing(ue_s.mean(), vn_s.mean())
        wd[i] = wd_i
        ws_mean[i] = ws_sample.mean()
        ws_std[i] = ws_sample.std()
        direction_flag[i] = low < wd_i < high
        bounce_flag[i] = (ws_mean[i] > cfg_second.bounce_max_ws_mean) or (ws_std[i] > cfg_second.bounce_max_ws_std)

    return wd, ws_mean, ws_std, direction_flag, bounce_flag
