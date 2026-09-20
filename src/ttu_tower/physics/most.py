"""Monin-Obukhov similarity theory: friction velocity, Obukhov length, the
neutral log-law fits.
"""
import numpy as np

from ttu_tower.math import fits
from ttu_tower.physics.constants import KAPPA, STANDARD_GRAVITY

# Businger-Dyer dimensionless wind-shear coefficients
_ALPHA = 4.7
_BETA = 15.0

# Constraints on fit domain for log law wind profile fits, and zero-division guard bound
_Z0_MAX = 10.0
_USTAR_MAX = 10.0
_B_NEAR_ZERO_ATOL = 1e-6


def friction_velocity(wu_covariance, wv_covariance):
    """u* (m/s) from the two momentum-flux covariances."""
    return (wu_covariance**2 + wv_covariance**2) ** 0.25


def obukhov_length(u_star, vpt, vpt_flux, gravity: float = STANDARD_GRAVITY):
    """Obukhov length L (m)."""
    return -(u_star**3) * vpt / (KAPPA * gravity * vpt_flux)


def businger_dyer_phi(z_over_l):
    """Dimensionless wind-shear function phi(z/L)."""
    phi_stable = 1 + _ALPHA * z_over_l
    phi_unstable = (1 - _BETA * z_over_l) ** (-0.25)
    return np.where(z_over_l >= 0, phi_stable, phi_unstable)


def most_wind_gradient(u_star, l, z, method: str = "businger dyer"):
    """du/dz (1/s) from MOST + Businger-Dyer, assuming u aligned with the mean wind."""
    m = method.lower().replace("-", " ").replace("_", " ")
    if m != "businger dyer":
        raise ValueError(f"Wind shear function method '{method}' unrecognized")
    return u_star / (KAPPA * z) * businger_dyer_phi(z / l)


def neutral_loglaw_fit(zvals, uvals, displacement: float = 0.0) -> tuple[float, float]:
    """Least-squares fit to u = (u*/kappa) * log((z-d)/z0). Returns (ustar, z0)."""
    z = np.asarray(zvals, dtype=float)
    u = np.asarray(uvals, dtype=float)
    mask = z > displacement
    a, b = fits.log_fit(z[mask] - displacement, u[mask])
    ustar = b * KAPPA
    z0 = np.exp(-a / b) if not np.isclose(b, 0.0, atol=_B_NEAR_ZERO_ATOL) else 0.0
    if z0 > _Z0_MAX or abs(ustar) > _USTAR_MAX:
        return np.nan, np.nan
    return ustar, z0


def constrained_neutral_loglaw_fit(zvals, uvals, ustar: float, displacement: float = 0.0) -> float:
    """neutral_loglaw_fit with u* fixed. Returns z0."""
    z = np.asarray(zvals, dtype=float)
    u = np.asarray(uvals, dtype=float)
    mask = z > displacement
    a, b = fits.constrained_log_fit(z[mask] - displacement, u[mask], b=ustar / KAPPA)
    z0 = np.exp(-a / b) if not np.isclose(b, 0.0, atol=_B_NEAR_ZERO_ATOL) else 0.0
    return np.nan if z0 > _Z0_MAX else z0
