import numpy as np

from ttu_tower.config.model import WindowsConfig
from ttu_tower.primary.qc_windows import higher_moments, resolution_dropouts
from ttu_tower.timegrid import SAMPLES_PER_SLOT


def _cfg(**overrides):
    defaults = dict(
        resolution_window_s=100.0, resolution_stride_s=50.0, resolution_bins=100,
        ts_resolution_bins=30, max_empty_bin_fraction=0.8, dropout_min_run=10,
        ts_dropout_min_run=50, ts_repeat_tolerance=0.05, max_dropout_fraction=0.05,
        dropout_min_windows=2, skew_range=(-2.0, 2.0), kurt_range=(1.0, 8.0),
    )
    defaults.update(overrides)
    return WindowsConfig(**defaults)


def test_resolution_flags_coarse_quantization_not_continuous_gaussian():
    rng = np.random.default_rng(0)
    n = 20000
    cfg = _cfg()

    x_coarse = np.round(rng.normal(0, 1, n))  # only ~7 distinct values
    (res_s, res_e), _ = resolution_dropouts(x_coarse, g0=0, cfg_windows=cfg, c=0.75)
    assert len(res_s) > 0

    x_gauss = rng.normal(0, 1, n)
    (res_s2, res_e2), _ = resolution_dropouts(x_gauss, g0=0, cfg_windows=cfg, c=0.75)
    assert len(res_s2) == 0


def test_dropouts_flag_above_threshold_not_below():
    rng = np.random.default_rng(1)
    n = 30000
    cfg = _cfg()

    def make_series(seg_count, seg_len=11, cluster_start=10000, spacing=60):
        x = rng.normal(0, 1, n)
        pos = cluster_start
        for _ in range(seg_count):
            x[pos : pos + seg_len] = 0.0
            pos += spacing
        return x

    # 40 segments of 11 identical samples (10 marked pairs each) clustered
    # tightly enough that a single window sees all of them: ~400/4999 = 8%.
    x_high = make_series(seg_count=40, spacing=60)
    (_, _), (drop_s, _) = resolution_dropouts(x_high, g0=0, cfg_windows=cfg, c=0.75)
    assert len(drop_s) > 0

    # 3 well-separated segments: well under 5% in any one window.
    x_low = make_series(seg_count=3, spacing=2000)
    (_, _), (drop_s2, _) = resolution_dropouts(x_low, g0=0, cfg_windows=cfg, c=0.75)
    assert len(drop_s2) == 0


def test_ts_dropouts_flag_on_exact_repeats_within_tolerance():
    rng = np.random.default_rng(2)
    n = 30000
    cfg = _cfg()  # ts_dropout_min_run = 50 -> each stuck run needs 51 samples
    x = 280.0 + rng.normal(0, 0.5, n)

    pos = 10000
    for _ in range(8):
        x[pos : pos + 51] = 280.0  # |delta| = 0 <= ts_repeat_tolerance, 50 marked pairs each
        pos += 100

    (_, _), (drop_s, _) = resolution_dropouts(x, g0=0, cfg_windows=cfg, c=0.75, is_ts=True)
    assert len(drop_s) > 0


def test_dropouts_skipped_in_resolution_flagged_windows():
    # A window that's resolution-flagged (coarse quantization) should not
    # also emit a dropout interval for that same window, even if it also has
    # a stuck run - dropouts are skipped there by design.
    rng = np.random.default_rng(3)
    n = 20000
    cfg = _cfg()
    x = np.round(rng.normal(0, 1, n))  # coarse -> resolution-flagged everywhere
    x[10000:10011] = 0.0  # a stuck run too

    (res_s, res_e), (drop_s, drop_e) = resolution_dropouts(x, g0=0, cfg_windows=cfg, c=0.75)
    assert len(res_s) > 0
    for ds, de in zip(drop_s, drop_e):
        assert not np.any((res_s <= ds) & (de <= res_e))


def test_higher_moments_flags_skew_and_kurt_independently():
    rng = np.random.default_rng(4)
    n = SAMPLES_PER_SLOT
    cfg = _cfg()

    x_lognormal = rng.lognormal(mean=0.0, sigma=0.7, size=n)
    skew, kurt, skew_flag, kurt_flag = higher_moments(x_lognormal, g0=0, slots=[0], cfg_windows=cfg, c=0.75)
    assert skew[0] > 2.0
    assert skew_flag[0]

    is_heavy = rng.random(n) < 0.05
    x_mixture = np.where(is_heavy, rng.normal(0, 5, n), rng.normal(0, 1, n))
    skew2, kurt2, skew_flag2, kurt_flag2 = higher_moments(x_mixture, g0=0, slots=[0], cfg_windows=cfg, c=0.75)
    assert kurt2[0] > 8.0
    assert kurt_flag2[0]
    assert not skew_flag2[0]

    x_gauss = rng.normal(0, 1, n)
    skew3, kurt3, skew_flag3, kurt_flag3 = higher_moments(x_gauss, g0=0, slots=[0], cfg_windows=cfg, c=0.75)
    assert not skew_flag3[0]
    assert not kurt_flag3[0]


def test_higher_moments_nan_below_coverage():
    n = SAMPLES_PER_SLOT
    cfg = _cfg()
    x = np.full(n, np.nan)
    x[:1000] = 1.0  # far below c*n
    skew, kurt, skew_flag, kurt_flag = higher_moments(x, g0=0, slots=[0], cfg_windows=cfg, c=0.75)
    assert np.isnan(skew[0]) and np.isnan(kurt[0])
    assert not skew_flag[0] and not kurt_flag[0]
