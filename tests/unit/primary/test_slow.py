import numpy as np

from ttu_tower.constants import P_REF, SAMPLE_HZ
from ttu_tower.physics.constants import R_CP
from ttu_tower.primary.slow import hann_kernel, smooth, vpts


def test_hann_kernel_shape_symmetric_positive_and_weight_count():
    k20 = hann_kernel(20.0)
    assert k20.size == 999
    assert np.all(k20 > 0)
    assert np.allclose(k20, k20[::-1])

    k2 = hann_kernel(2.0)
    assert k2.size == 99
    assert np.all(k2 > 0)
    assert np.allclose(k2, k2[::-1])


def test_hann_kernel_minus_3db_point():
    for width_s in (20.0, 2.0):
        k = hann_kernel(width_s)
        k_norm = k / k.sum()
        n_fft = 1 << 20
        response = np.abs(np.fft.rfft(k_norm, n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_HZ)
        target = response[0] / np.sqrt(2)
        idx = np.argmax(response <= target)
        f_3db = freqs[idx]
        expected = 0.72 / width_s
        assert abs(f_3db - expected) / expected < 0.02


def test_smoothing_constant_and_ramp_unchanged_away_from_gaps():
    n = 5000
    k = hann_kernel(20.0)  # reach 499
    ok = np.ones(n, dtype=bool)

    const = np.full(n, 5.0)
    xs, usable = smooth(const, ok, k, c=0.75)
    assert np.allclose(xs[600:-600], 5.0, atol=1e-9)
    assert usable[600:-600].all()

    ramp = np.arange(n, dtype=np.float64) * 0.1
    xs2, _ = smooth(ramp, ok, k, c=0.75)
    assert np.allclose(xs2[600:-600], ramp[600:-600], atol=1e-6)


def test_smoothing_nan_gap_renormalizes_correctly():
    n = 3000
    k = hann_kernel(2.0)  # reach 99; a 10-sample gap centered on the peak weight keeps coverage ~0.80
    ok = np.ones(n, dtype=bool)
    ramp = np.arange(n, dtype=np.float64) * 0.01
    ok[1500:1510] = False
    x = ramp.copy()
    x[1500:1510] = np.nan

    xs, usable = smooth(x, ok, k, c=0.75)
    assert usable[1505]
    assert abs(xs[1505] - ramp[1505]) < 0.05


def test_smoothing_weighted_coverage_below_c_is_unusable():
    n = 3000
    k = hann_kernel(2.0)  # reach 99
    ok = np.ones(n, dtype=bool)
    x = np.full(n, 5.0)
    ok[1400:1600] = False  # gap wider than the kernel's reach
    x[1400:1600] = np.nan

    _, usable = smooth(x, ok, k, c=0.75)
    assert not usable[1500]


def test_smoothing_has_no_steps():
    rng = np.random.default_rng(5)
    n = 10000
    walk = np.cumsum(rng.normal(0, 0.001, n))
    walk_quantized = np.round(walk / 0.002) * 0.002
    ok = np.ones(n, dtype=bool)
    k = hann_kernel(20.0)

    xs, usable = smooth(walk_quantized, ok, k, c=0.75)
    diffs = np.abs(np.diff(xs[usable]))
    assert diffs.max() < 0.15 * 0.002


def test_smoothing_translation_invariance():
    rng = np.random.default_rng(6)
    full = rng.normal(280.0, 1.0, 300000)
    ok_full = np.ones(300000, dtype=bool)
    k = hann_kernel(20.0)  # reach 499
    core_start, core_len = 120000, 30000
    margin = 30000

    def core_result(margin_size):
        g0 = core_start - margin_size
        span = full[g0 : core_start + core_len + margin_size]
        ok_span = ok_full[g0 : core_start + core_len + margin_size]
        xs, usable = smooth(span, ok_span, k, c=0.75)
        offset = core_start - g0
        return xs[offset : offset + core_len], usable[offset : offset + core_len]

    xs_small, usable_small = core_result(margin)
    xs_large, usable_large = core_result(3 * margin)
    assert np.array_equal(xs_small, xs_large, equal_nan=True)
    assert np.array_equal(usable_small, usable_large)


def test_vpts_formula_and_p_ref_matches_isa():
    from ttu_tower.constants import HEIGHTS, SITE_ELEVATION
    from ttu_tower.physics import thermo

    n = 100
    ts = np.full(n, 300.0)
    boom = 3
    result = vpts(ts, boom)
    expected = ts * (100.0 / P_REF[boom]) ** R_CP
    assert np.allclose(result, expected)
    assert P_REF[boom] == thermo.isa_pressure(SITE_ELEVATION + HEIGHTS[boom])
