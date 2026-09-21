import math
import warnings

import numpy as np
from pytest import approx
from scipy.optimize import curve_fit

from ttu_tower.math.fits import (
    constrained_linear_fit,
    ls_linear_fit,
    ls_weighted_linear_fit,
    power_fit,
)


def test_ls_linear_fit_basic():
    xvals = [1.0, 2.0, 3.0, 4.0]
    yvals = [10.25, 10.5, 10.75, 11.0]  # y = 10 + 0.25*x, exactly
    a, b = ls_linear_fit(xvals, yvals)
    assert a == approx(10.0, abs=1e-9)
    assert b == approx(0.25, abs=1e-9)


def test_ls_linear_fit_duplicate_values_with_nan():
    xvals = [1.0, 2.0, 2.0, float("nan"), 3.0, 2.0]
    yvals = [10.0, 20.0, 21.0, 99.0, 30.0, float("nan")]

    a, b = ls_linear_fit(xvals, yvals)

    pairs = [(x, y) for x, y in zip(xvals, yvals) if not (math.isnan(x) or math.isnan(y))]
    assert pairs == [(1.0, 10.0), (2.0, 20.0), (2.0, 21.0), (3.0, 30.0)]
    n = len(pairs)
    sum_x = sum(p[0] for p in pairs)
    sum_y = sum(p[1] for p in pairs)
    sum_x2 = sum(p[0] * p[0] for p in pairs)
    sum_xy = sum(p[0] * p[1] for p in pairs)
    det = n * sum_x2 - sum_x * sum_x
    expected_a = (sum_y * sum_x2 - sum_x * sum_xy) / det
    expected_b = (n * sum_xy - sum_x * sum_y) / det

    assert a == approx(expected_a, abs=1e-9)
    assert b == approx(expected_b, abs=1e-9)


def test_ls_linear_fit_all_nan_input_gives_zero_with_no_warning():
    """A caller (e.g. linear_detrend on a fully-missing slot/file) can pass an
    array that's entirely NaN; the pre-drop `x.size == 0` guard doesn't catch
    this since x.size is nonzero before dropping - only after.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a, b = ls_linear_fit([1.0, 2.0, 3.0], [float("nan")] * 3)
    assert (a, b) == (0.0, 0.0)


def test_ls_linear_fit_single_surviving_point_gives_zero_with_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a, b = ls_linear_fit([1.0, float("nan")], [5.0, float("nan")])
    assert (a, b) == (0.0, 0.0)


def test_ls_weighted_linear_fit_all_nan_input_gives_zero_with_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a, b = ls_weighted_linear_fit([1.0, 2.0], [float("nan")] * 2, [1.0, 1.0])
    assert (a, b) == (0.0, 0.0)


def test_constrained_linear_fit_fixed_intercept():
    xvals = [1.0, 2.0, 3.0, 4.0]
    yvals = [10.0, 12.0, 14.0, 16.0]  # y = 8 + 2x
    a, b = constrained_linear_fit(xvals, yvals, a=8.0)
    assert a == 8.0
    assert b == approx(2.0, abs=1e-9)


def test_ls_weighted_linear_fit_equal_weights_matches_unweighted():
    xvals = [1.0, 2.0, 3.0, 4.0, 5.5]
    yvals = [10.25, 10.5, 10.75, 11.0, 11.375]
    a_u, b_u = ls_linear_fit(xvals, yvals)
    a_w, b_w = ls_weighted_linear_fit(xvals, yvals, weights=[1.0] * len(xvals))
    assert a_w == approx(a_u, abs=1e-9)
    assert b_w == approx(b_u, abs=1e-9)


def test_ls_weighted_linear_fit_basic():
    xvals = [1.0, 2.0, 3.0, 4.0]
    yvals = [10.0, 12.0, 14.0, 16.0]  # y = 8 + 2*x, exactly
    weights = [1.0, 4.0, 9.0, 16.0]
    a, b = ls_weighted_linear_fit(xvals, yvals, weights)
    assert a == approx(8.0, abs=1e-9)
    assert b == approx(2.0, abs=1e-9)


def test_power_fit_recovers_exact_powerlaw():
    xvals = [1.0, 2.0, 4.0, 8.0, 16.0]
    a_true, b_true = 3.0, 0.5
    yvals = [a_true * x**b_true for x in xvals]
    a, b = power_fit(xvals, yvals)
    assert a == approx(a_true, abs=1e-9)
    assert b == approx(b_true, abs=1e-9)


def test_power_fit_weighting_reduces_bias_vs_nls():
    # y = 3*x^0.5 plus roughly-constant-magnitude noise: a weighted log-log fit
    # should track true nonlinear least squares more closely than an
    # unweighted log-log fit, which underweights large-y points.
    xvals = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    a_true, b_true = 3.0, 0.5
    noise = np.array([-1.0, 0.8, -0.6, 0.5, 1.0, -0.9])
    yvals = a_true * xvals**b_true + noise

    _, b_unweighted = ls_linear_fit(np.log(xvals), np.log(yvals))
    _, b_weighted = power_fit(xvals, yvals)

    (_, b_nls), _ = curve_fit(lambda x, a, b: a * x**b, xvals, yvals, p0=[a_true, b_true])

    assert abs(b_weighted - b_nls) < abs(b_unweighted - b_nls)
    assert b_weighted == approx(0.4625034431995891, abs=1e-6)
