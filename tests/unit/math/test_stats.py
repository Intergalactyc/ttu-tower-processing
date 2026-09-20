import numpy as np
import pytest

from ttu_tower.math.stats import autocovariance, efolding_integral, linear_detrend, moments


def test_autocovariance_matches_direct_sum():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    max_lag = 20
    result = autocovariance(x, max_lag)
    for lag in range(max_lag + 1):
        expected = np.sum(x[: len(x) - lag] * x[lag:])
        assert result[lag] == pytest.approx(expected, rel=1e-9)


def test_autocovariance_lag_zero_is_sum_of_squares():
    x = np.array([1.0, 2.0, -1.0, 3.0])
    result = autocovariance(x, max_lag=2)
    assert result[0] == pytest.approx(np.sum(x**2))


def test_efolding_integral_exponential_acf():
    # rho(l) = exp(-l/T): the integral is truncated at the 1/e crossing (lag
    # T exactly), so it's the partial integral (1 - 1/e)*T, not the full T.
    T = 5.0
    lags = np.arange(0, 200)
    rho = np.exp(-lags / T)
    result = efolding_integral(rho)
    # the trapezoid rule on integer lag steps slightly overestimates a convex
    # curve like this one, so allow a bit more than pure floating-point slop
    assert result == pytest.approx((1 - 1 / np.e) * T, rel=5e-3)


def test_efolding_integral_never_crosses_gives_nan():
    rho = np.full(50, 0.9)  # stays well above 1/e throughout
    assert np.isnan(efolding_integral(rho))


def test_efolding_integral_crosses_immediately():
    rho = np.array([1.0 / np.e - 0.01, 0.1, 0.05])
    assert efolding_integral(rho) == 0.0


def test_efolding_integral_interpolates_crossing():
    # rho drops linearly from 1 to 0 over lags 0..10: crosses 1/e at lag 10*(1-1/e)
    lags = np.arange(11)
    rho = 1.0 - lags / 10.0
    result = efolding_integral(rho)
    crossing = 10 * (1 - 1 / np.e)
    # integral of a straight line from 0 to crossing = 0.5*(1+1/e)*crossing
    expected = 0.5 * (1.0 + 1.0 / np.e) * crossing
    assert result == pytest.approx(expected, rel=1e-9)
