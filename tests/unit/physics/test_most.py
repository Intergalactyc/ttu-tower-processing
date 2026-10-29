import numpy as np
from pytest import approx

from ttu_tower.physics import most
from ttu_tower.physics.constants import KAPPA


def test_friction_velocity():
    assert most.friction_velocity(3.0, 4.0) == approx((9.0 + 16.0) ** 0.25, abs=1e-12)


def test_obukhov_length_matches_formula():
    u_star, vpt, flux = 0.3, 295.0, 0.05
    expected = -(u_star**3) * vpt / (KAPPA * 9.80665 * flux)
    assert most.obukhov_length(u_star, vpt, flux) == approx(expected, abs=1e-9)


def test_temperature_scale():
    assert most.temperature_scale(0.3, 0.06) == approx(-0.2, abs=1e-12)


def test_businger_dyer_functions_at_neutral():
    for f in (most.businger_dyer_phi_m, most.businger_dyer_phi_h):
        assert f(0.0) == approx(1.0, abs=1e-12)
    for f in (most.businger_dyer_psi_m, most.businger_dyer_psi_h):
        assert f(0.0) == approx(0.0, abs=1e-12)


def test_businger_dyer_phi_stable_unstable_branches():
    assert most.businger_dyer_phi_m(0.5) == approx(1 + 5.0 * 0.5, abs=1e-12)
    assert most.businger_dyer_phi_h(0.5) == approx(1 + 5.0 * 0.5, abs=1e-12)
    assert most.businger_dyer_phi_m(-0.5) == approx((1 - 16.0 * (-0.5)) ** (-0.25), abs=1e-12)
    assert most.businger_dyer_phi_h(-0.5) == approx((1 - 16.0 * (-0.5)) ** (-0.5), abs=1e-12)


def test_businger_dyer_no_warning_on_strongly_stable(recwarn):
    zeta = np.array([-1.0, 0.0, 0.5, 2.0, np.nan])
    for f in (most.businger_dyer_phi_m, most.businger_dyer_phi_h, most.businger_dyer_psi_m, most.businger_dyer_psi_h):
        out = f(zeta)
        assert np.all(np.isfinite(out[:4])) and np.isnan(out[4])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_businger_dyer_psi_integrates_phi():
    # psi(zeta) = integral_0^zeta (1 - phi(x))/x dx
    for zeta in (-2.0, -0.3, 0.4):
        x = np.linspace(zeta * 1e-9, zeta, 200_001)
        for phi, psi in ((most.businger_dyer_phi_m, most.businger_dyer_psi_m),
                         (most.businger_dyer_phi_h, most.businger_dyer_psi_h)):
            assert psi(zeta) == approx(np.trapezoid((1 - phi(x)) / x, x), abs=1e-6)


def test_neutral_loglaw_fit_recovers_known_profile():
    ustar_true, z0_true = 0.35, 0.1
    z = np.array([2.0, 5.0, 10.0, 20.0, 50.0])
    u = (ustar_true / KAPPA) * np.log(z / z0_true)
    ustar, z0 = most.neutral_loglaw_fit(z, u)
    assert ustar == approx(ustar_true, abs=1e-9)
    assert z0 == approx(z0_true, abs=1e-9)


def test_constrained_neutral_loglaw_fit_matches_unconstrained():
    ustar_true, z0_true = 0.35, 0.1
    z = np.array([2.0, 5.0, 10.0, 20.0, 50.0])
    u = (ustar_true / KAPPA) * np.log(z / z0_true)
    z0 = most.constrained_neutral_loglaw_fit(z, u, ustar=ustar_true)
    assert z0 == approx(z0_true, abs=1e-9)
