import numpy as np

from ttu_tower.config.model import DespikeConfig
from ttu_tower.primary.despike import despike


def _cfg(window_s=300.0, stride_s=30.0, max_spike_samples=3):
    return DespikeConfig(window_s=window_s, stride_s=stride_s, z_threshold={},
                          max_spike_samples=max_spike_samples, min_mad={})


def test_spikes_classified_by_run_length():
    rng = np.random.default_rng(0)
    n = 30000
    x = rng.normal(0, 1, n)
    cfg = _cfg()

    burst_lengths = [1, 2, 3, 4, 5, 6]
    burst_starts = {}
    pos = 5000
    for length in burst_lengths:
        x[pos : pos + length] = 8.0
        burst_starts[length] = pos
        pos += 500

    result = despike(x, g0=0, cfg_despike=cfg, min_mad=1e-6, z_threshold=3.5, c=0.75)

    for length in burst_lengths:
        s = burst_starts[length]
        if length <= cfg.max_spike_samples:
            assert result.spike[s : s + length].all(), length
            assert np.isnan(result.x[s : s + length]).all(), length
        else:
            assert result.excursion[s : s + length].all(), length
            assert not np.isnan(result.x[s : s + length]).any(), length


def test_alternating_burst_split_vs_same_sign_burst_kept_whole():
    n = 30000
    x = np.zeros(n)
    cfg = _cfg()

    pos_a = 10000
    x[pos_a : pos_a + 6] = [-10.0, 10.0, 8.0, -7.0, 15.0, -12.0]

    pos_b = 20000
    x[pos_b : pos_b + 6] = [10.0, 14.0, 7.0, 12.0, 9.0, 11.0]

    result = despike(x, g0=0, cfg_despike=cfg, min_mad=1.0, z_threshold=3.5, c=0.75)

    # split at every sign change -> five pieces, all length <= 3 -> all spike
    assert result.spike[pos_a : pos_a + 6].all()
    assert not result.excursion[pos_a : pos_a + 6].any()

    # one same-sign run of length 6, not split despite large internal jumps -> excursion
    assert result.excursion[pos_b : pos_b + 6].all()
    assert not result.spike[pos_b : pos_b + 6].any()
    assert not np.isnan(result.x[pos_b : pos_b + 6]).any()


def test_unchecked_deep_in_a_long_gap_but_not_at_its_edges():
    rng = np.random.default_rng(2)
    n = 80000
    x = rng.normal(0, 1, n)
    cfg = _cfg()  # window_s=300 -> half_window = 7500 samples

    gap_start, gap_len = 20000, 20000  # much longer than 2*half_window
    x[gap_start : gap_start + gap_len] = np.nan

    result = despike(x, g0=0, cfg_despike=cfg, min_mad=1e-6, z_threshold=3.5, c=0.75)

    middle = gap_start + gap_len // 2
    assert result.unchecked[middle]
    assert not result.unchecked[gap_start - 100]
    assert not result.unchecked[gap_start + gap_len + 100]

    gap_unchecked = result.unchecked[gap_start : gap_start + gap_len]
    assert 0 < gap_unchecked.sum() < gap_len  # edges of the gap are still reachable


def test_mad_floor_prevents_mass_flagging_of_near_constant_quantized_data():
    rng = np.random.default_rng(3)
    n = 30000
    x = np.full(n, 280.0)
    jitter_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    x[jitter_idx] += 0.01  # one quantization step
    cfg = _cfg()

    result = despike(x, g0=0, cfg_despike=cfg, min_mad=0.01, z_threshold=3.5, c=0.75)
    flagged_fraction = (result.spike | result.excursion).mean()
    assert flagged_fraction < 0.01


def test_translation_invariance_margin_vs_wider_margin():
    rng = np.random.default_rng(4)
    full = rng.normal(0, 1, 300000)
    full[100000:100002] = 8.0
    full[150000] = -9.0
    cfg = _cfg()

    core_start, core_len = 120000, 30000
    margin = 30000  # exceeds despike's reach (window_s/2 + stride_s = 180 s = 9000 samples)

    def core_result(margin_size):
        g0 = core_start - margin_size
        span = full[g0 : core_start + core_len + margin_size]
        result = despike(span, g0=g0, cfg_despike=cfg, min_mad=1e-6, z_threshold=3.5, c=0.75)
        offset = core_start - g0
        return result, offset

    small, off_small = core_result(margin)
    large, off_large = core_result(3 * margin)

    sl = slice(off_small, off_small + core_len)
    ll = slice(off_large, off_large + core_len)
    assert np.array_equal(small.x[sl], large.x[ll], equal_nan=True)
    assert np.array_equal(small.spike[sl], large.spike[ll])
    assert np.array_equal(small.excursion[sl], large.excursion[ll])
    assert np.array_equal(small.unchecked[sl], large.unchecked[ll])
