import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import HEIGHTS, SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.math.stats import log_gradient_weights
from ttu_tower.physics.constants import KAPPA
from ttu_tower.physics.earth import local_gravity
from ttu_tower.tertiary import gradients
from ttu_tower.tertiary.gradients import gradients as compute

_SLOT = 7
_BOOMS = [3, 4, 5, 6]
_STENCILS = ("log_linear", "log")


def _row(boom, variable, value, stat=None, variant="none"):
    return {"slot": _SLOT, "boom": boom, "variant": variant, "variable": variable, "stat": stat, "value": value}


def _boom_final(speed=lambda z: 2.0 * np.log(z) + 0.05 * z, pt=lambda z: 290 + 0.5 * np.log(z) + 0.01 * z,
                vpt=lambda z: 291 + 0.8 * np.log(z) + 0.02 * z, angle=lambda z: 0.3, ustar=0.4, theta_star=0.05,
                uw=-0.16, vw=0.0, wvpts=-0.02, sigma_w=0.3, drop=()):
    """ue/vn from a speed and an angle (radians from east) per height; the flux rows per variant."""
    rows = []
    for boom in _BOOMS:
        z = HEIGHTS[boom]
        gone = boom in drop
        rows += [_row(boom, "ue", np.nan if gone else speed(z) * np.cos(angle(z)), "mean"),
                 _row(boom, "vn", speed(z) * np.sin(angle(z)), "mean"),
                 _row(boom, "pt", pt(z)), _row(boom, "vpt", np.nan if gone else vpt(z))]
        for variant in ("mrd", "naive"):
            rows += [_row(boom, "ustar", ustar, variant=variant), _row(boom, "theta_star", theta_star, variant=variant),
                     _row(boom, "sigma_w", sigma_w, variant=variant), _row(boom, "uw", uw, "cov", variant),
                     _row(boom, "vw", vw, "cov", variant), _row(boom, "wvpts", wvpts, "cov", variant)]
    return pd.DataFrame(rows)


def _value(out, boom, variable, variant="none"):
    m = out[(out["boom"] == boom) & (out["variable"] == variable) & (out["variant"] == variant)]
    assert len(m) == 1, (boom, variable, variant)
    return m["value"].iloc[0]


def test_weights_are_exact_for_their_profiles():
    z1, z, z2 = 4.0, 10.1, 16.8
    two = log_gradient_weights([z1, z2], z)
    assert two @ (3.0 * np.log([z1, z2]) + 1.0) == pytest.approx(3.0 / z)
    three = log_gradient_weights([z1, z, z2], z)
    f = lambda h: 1.0 + 3.0 * np.log(h) + 0.2 * h  # noqa: E731
    assert three @ f(np.array([z1, z, z2])) == pytest.approx(3.0 / z + 0.2)
    assert two.sum() == pytest.approx(0.0) and three.sum() == pytest.approx(0.0, abs=1e-12)


def test_stencils_use_the_neighbours_and_two_points_at_the_ends():
    log, loglin = gradients.stencil(_BOOMS, "log"), gradients.stencil(_BOOMS, "log_linear")
    assert log[4][0] == [3, 5] and loglin[4][0] == [3, 4, 5]
    for form in (log, loglin):
        assert form[3][0] == [3, 4] and form[6][0] == [5, 6]
    assert gradients.stencil([2, 5, 9], "log")[5][0] == [2, 9]  # neighbours among the run's booms
    assert gradients.stencil([4], "log") == {}


def test_log_linear_recovers_a_log_linear_profile_and_log_a_log_profile():
    out = compute(_boom_final(), _BOOMS, _STENCILS)
    for boom in (4, 5):
        z = HEIGHTS[boom]
        assert _value(out, boom, "dws_dz") == pytest.approx(2.0 / z + 0.05)
        assert _value(out, boom, "dpt_dz") == pytest.approx(0.5 / z + 0.01)
        assert _value(out, boom, "dvpt_dz") == pytest.approx(0.8 / z + 0.02)
    pure = compute(_boom_final(speed=lambda z: 2.0 * np.log(z)), _BOOMS, _STENCILS)
    for boom in _BOOMS:  # ends included: two points are exact for a log profile
        assert _value(pure, boom, "dws_dz_log") == pytest.approx(2.0 / HEIGHTS[boom])
        assert _value(pure, boom, "dws_dz") == pytest.approx(2.0 / HEIGHTS[boom])


def test_profile_and_flux_products_follow_their_definitions():
    out = compute(_boom_final(), _BOOMS, _STENCILS)
    boom, z = 4, HEIGHTS[4]
    dws, dpt, dvpt = 2.0 / z + 0.05, 0.5 / z + 0.01, 0.8 / z + 0.02
    theta_v = 291 + 0.8 * np.log(z) + 0.02 * z  # the boom's own
    g = local_gravity(SITE_LATITUDE, SITE_ELEVATION + z)
    n2 = g / theta_v * dvpt
    assert _value(out, boom, "shear") == pytest.approx(dws)  # no veer: the vector shear is the speed's
    assert _value(out, boom, "n2") == pytest.approx(n2)
    assert _value(out, boom, "rig") == pytest.approx(n2 / dws**2)

    phi_m = _value(out, boom, "phi_m", "mrd")
    assert phi_m == pytest.approx(KAPPA * z * dws / 0.4)
    assert _value(out, boom, "phi_h", "naive") == pytest.approx(KAPPA * z * dpt / 0.05)
    assert _value(out, boom, "phi_h_v", "mrd") == pytest.approx(KAPPA * z * dvpt / 0.05)
    rif = _value(out, boom, "rif", "mrd")
    assert rif == pytest.approx(g / theta_v * -0.02 / (-0.16 * dws))
    # stress along the wind (uw = -u*^2, vw = 0): Ri_f = zeta/phi_m with zeta = -kappa z g w'θv'/(u*^3 θv)
    assert rif == pytest.approx(-KAPPA * z * g * -0.02 / (0.4**3 * theta_v) / phi_m)
    assert _value(out, boom, "km", "mrd") == pytest.approx(0.16 / dws)
    assert _value(out, boom, "kh", "mrd") == pytest.approx(0.02 / dvpt)
    assert _value(out, boom, "prandtl", "mrd") == pytest.approx(_value(out, boom, "rig") / rif)
    assert _value(out, boom, "l_buoyant", "mrd") == pytest.approx(0.3 / np.sqrt(n2))
    assert out["stat"].isna().all() and set(out["variant"]) == {"none", "mrd", "naive"}


def test_the_second_stencil_is_suffixed_and_differs_off_log():
    out = compute(_boom_final(), _BOOMS, _STENCILS)
    names = set(out["variable"])
    assert {"rig", "rig_log", "prandtl", "prandtl_log"} <= names and "rig_log_linear" not in names
    assert _value(out, 5, "dws_dz_log") != pytest.approx(_value(out, 5, "dws_dz"))
    only_log = compute(_boom_final(), _BOOMS, ("log",))
    assert _value(only_log, 5, "dws_dz") == pytest.approx(_value(out, 5, "dws_dz_log"))


def test_shear_rif_and_km_dont_depend_on_the_frame():
    veering = dict(angle=lambda z: 0.2 + 0.15 * np.log(z), uw=-0.12, vw=0.05)
    a = compute(_boom_final(**veering), _BOOMS, _STENCILS)
    turned = dict(veering, angle=lambda z: 1.4 + 0.15 * np.log(z))  # every boom's wind turned by the same angle
    b = compute(_boom_final(**turned), _BOOMS, _STENCILS)
    for variable, variant in (("shear", "none"), ("dws_dz", "none"), ("rig", "none"), ("rif", "mrd"),
                              ("km", "mrd"), ("rif_log", "mrd")):
        for boom in _BOOMS:
            assert _value(a, boom, variable, variant) == pytest.approx(_value(b, boom, variable, variant))
    # veer: the vector shear exceeds the speed gradient, and the cross-wind stress counts in Ri_f
    assert _value(a, 4, "shear") > abs(_value(a, 4, "dws_dz"))
    aligned = compute(_boom_final(**dict(veering, vw=0.0)), _BOOMS, _STENCILS)
    assert _value(a, 4, "rif", "mrd") != pytest.approx(_value(aligned, 4, "rif", "mrd"))


def test_filtered_points_give_nan_only_where_a_stencil_uses_them():
    out = compute(_boom_final(drop=(4,)), _BOOMS, _STENCILS)
    assert np.isnan(_value(out, 4, "dws_dz")) and np.isnan(_value(out, 3, "dws_dz"))  # log_linear uses 4
    assert np.isfinite(_value(out, 4, "dws_dz_log"))  # log at 4 uses 3 and 5 only
    assert np.isnan(_value(out, 4, "n2_log"))  # but N² takes the boom's own θv
    assert np.isnan(_value(out, 5, "dvpt_dz_log")) and np.isfinite(_value(out, 6, "dvpt_dz"))


def test_degenerate_inputs_give_nan_not_inf():
    out = compute(_boom_final(theta_star=0.0, vpt=lambda z: 291 - 0.8 * np.log(z)), _BOOMS, _STENCILS)
    assert np.isnan(_value(out, 4, "phi_h_v", "mrd"))
    assert _value(out, 4, "n2") < 0 and np.isnan(_value(out, 4, "l_buoyant", "mrd"))
    calm = compute(_boom_final(speed=lambda z: 0 * z), _BOOMS, _STENCILS)
    assert np.isnan(_value(calm, 4, "rig")) and np.isnan(_value(calm, 4, "km", "mrd"))


def test_names_split_back_to_their_product():
    assert gradients.split("rig_log") == ("rig", "log")
    assert gradients.split("phi_h_v_log_linear") == ("phi_h_v", "log_linear")
    assert gradients.split("phi_h_v") == ("phi_h_v", None) and gradients.split("ws") == ("ws", None)
    assert gradients.name("rig", "log", _STENCILS) == "rig_log" and gradients.name("rig", "log_linear", _STENCILS) == "rig"
