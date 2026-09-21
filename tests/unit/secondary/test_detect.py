import numpy as np
import pytest

from ttu_tower.config.model import DetectionConfig
from ttu_tower.secondary.detect import detect

# P_i for i = 1..15 (0.29296875 * 2**(i-1)), 0-indexed here (P[0] = P_1)
P = [0.29296875 * 2.0**k for k in range(15)]

_CFG = DetectionConfig(peak_significance_se=2.0, min_scale_s=0.5, min_tau_s=9.375, max_tau_s=1200.0)

_SIGN_TYPE = [0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.5, 1.0, 0.4, -0.3, -0.6, -0.4, -0.2, 0.3, 0.5]
_RISING = [0.1 + i * (2.9 / 14) for i in range(15)]


def _n_default():
    return np.array([2 ** (15 - i) for i in range(1, 16)])


def _run(d, se=0.05, n=None, family_has_data=True, cfg=_CFG):
    d = np.asarray(d, dtype=np.float64)
    se_arr = np.full(d.size, se, dtype=np.float64) if np.isscalar(se) else np.asarray(se, dtype=np.float64)
    n_arr = _n_default() if n is None else np.asarray(n)
    return detect(d, se_arr, n_arr, family_has_data, cfg)


def test_sign_type_gap():
    r = _run(_SIGN_TYPE)
    assert r.status == "found"
    assert r.peak_scale_s == pytest.approx(P[5])
    assert r.reversal_scale_s == pytest.approx(P[9])
    assert r.reversal_type == "sign"
    assert r.tau_s == pytest.approx(P[8])
    assert r.sign == 1


def test_increase_type_gap():
    d = [0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.4, 0.9, 0.5, 0.3, 0.4, 0.8, 1.4, 2.0, 2.5]
    r = _run(d)
    assert r.status == "found"
    assert r.peak_scale_s == pytest.approx(P[5])
    assert r.reversal_scale_s == pytest.approx(P[10])
    assert r.reversal_type == "increase"
    assert r.tau_s == pytest.approx(P[9])


def test_wobble_before_the_peak():
    d = [0.1, 0.5, 1.0, 0.9, 1.6, 1.8, 1.5, 1.0, 0.4, -0.2, -0.4, -0.2, 0.1, 0.2, 0.1]
    r = _run(d, se=0.1)
    assert r.status == "found"
    assert r.peak_scale_s == pytest.approx(P[5])
    assert r.reversal_type == "sign"
    assert r.tau_s == pytest.approx(P[8])


def test_insignificant_bump_first():
    d = [0.0, 0.02, 0.06, 0.02, 0.0, 0.3, 0.8, 1.5, 2.0, 1.5, 0.6, -0.2, -0.3, -0.1, 0.1]
    r = _run(d)
    assert r.status == "found"
    assert r.peak_scale_s == pytest.approx(P[8])
    assert r.reversal_type == "sign"
    assert r.tau_s == pytest.approx(P[10])


def test_mode_1_ignored():
    d = list(_SIGN_TYPE)
    d[0] = -5.0
    r = _run(d)
    baseline = _run(_SIGN_TYPE)
    for field in ("status", "tau_s", "tau_lb_s", "sign", "peak_scale_s", "reversal_scale_s", "reversal_type"):
        a, b = getattr(r, field), getattr(baseline, field)
        if isinstance(a, float) and np.isnan(a) and np.isnan(b):
            continue
        assert a == b


def test_weak():
    d = [0.05, -0.04, 0.08, 0.02, 0.09, -0.05, 0.03, 0, 0.07, -0.02, 0.01, 0.05, -0.03, 0.02, 0.01]
    r = _run(d)
    assert r.status == "weak"


def test_rising_through_40_min():
    r = _run(_RISING)
    assert r.status == "capped"
    assert r.tau_s == pytest.approx(1200.0)
    assert r.reversal_type is None


def test_rising_80_min_sign_change():
    d = _RISING[:14] + [-1.0]
    r = _run(d)
    assert r.status == "capped"
    assert r.reversal_scale_s == pytest.approx(P[14])
    assert r.reversal_type == "sign"


def test_peak_slow_decline_80_min_rise():
    d = [0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 1.5]
    r = _run(d)
    assert r.status == "capped"
    assert r.reversal_scale_s == pytest.approx(P[14])
    assert r.reversal_type == "increase"


def test_unusable_mode():
    n = _n_default()
    n[11] = 1  # N_12 = 1 (1-indexed mode 12 -> array index 11)
    r = _run(_RISING, n=n)
    assert r.status == "unresolved"
    assert r.tau_lb_s == pytest.approx(P[10])


def test_gap_below_an_unusable_mode():
    n = _n_default()
    n[11] = 1
    r = _run(_SIGN_TYPE, n=n)
    assert r.status == "found"
    assert r.tau_s == pytest.approx(P[8])


def test_negative_flux():
    baseline = _run(_SIGN_TYPE)
    r = _run([-x for x in _SIGN_TYPE])
    assert r.status == "found"
    assert r.tau_s == pytest.approx(baseline.tau_s)
    assert r.sign == -1


def test_se_zero_matches_se_default():
    baseline = _run(_SIGN_TYPE)
    r = _run(_SIGN_TYPE, se=0.0)
    assert r.status == baseline.status
    assert r.tau_s == pytest.approx(baseline.tau_s)


def test_all_zero_se_zero_is_weak():
    r = _run([0.0] * 15, se=0.0)
    assert r.status == "weak"


def test_no_usable_mode():
    n = _n_default()
    n[1] = 1  # N_2 = 1 -> even i_lo is unusable
    r = _run(_SIGN_TYPE, n=n)
    assert r.status == "unresolved"
    assert r.tau_lb_s == pytest.approx(P[0])


def test_no_data():
    r = _run(_SIGN_TYPE, family_has_data=False)
    assert r.status == "no_data"
    assert np.isnan(r.tau_s)
    assert np.isnan(r.tau_lb_s)
    assert r.sign == 0


def test_insignificant_reversal_at_the_floor_is_skipped_for_a_real_one_further_out():
    # A small, noise-level dip right after the peak (mode 5, |d|=0.02 against
    # se=0.05 - well under the 2-SE bar) would flip sign and floor tau under
    # a bare reversal test, at exactly the kind of small scale where SE can
    # be spuriously tiny in real (especially strongly unstable) data. The
    # real, clearly significant reversal is much further out at mode 8.
    d = [0.0, 1.0, 0.5, 0.05, -0.02, -0.03, -0.05, -0.08, -0.1, -0.5, -1.5, -0.2, -0.1, -0.05, -0.02]
    r = _run(d)
    assert r.status == "found"
    assert r.peak_scale_s == pytest.approx(P[1])
    assert r.reversal_scale_s == pytest.approx(P[7])
    assert r.reversal_type == "sign"
    assert r.tau_s == pytest.approx(P[6])
    assert r.tau_s > _CFG.min_tau_s  # not floored


def test_significant_reversal_at_the_floor_still_floors():
    # A reversal right after the peak that IS clearly significant (0.1 vs.
    # se=0.05, a 3.3-SE swing after smoothing) must still floor tau exactly
    # as before - the added gate only ever screens out insignificant
    # candidates, never a real one just because it's close to the floor.
    d = [0.0, 0.3, 1.0, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = _run(d)
    assert r.status == "found"
    assert r.peak_scale_s == pytest.approx(P[1])
    assert r.reversal_scale_s == pytest.approx(P[3])
    assert r.reversal_type == "sign"
    assert r.tau_s == pytest.approx(_CFG.min_tau_s)
