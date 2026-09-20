import math

from ttu_tower.physics.turbulence import anisotropy, transport_efficiency


def test_transport_efficiency_all_same_sign_gives_one():
    assert transport_efficiency(s_all=10.0, s_pos=10.0, s_neg=0.0) == 1.0
    assert transport_efficiency(s_all=-10.0, s_pos=0.0, s_neg=-10.0) == 1.0


def test_transport_efficiency_three_to_one_split():
    # 3 units positive, 1 unit negative -> s_all = 2, s_pos = 3, s_neg = -1
    te = transport_efficiency(s_all=2.0, s_pos=3.0, s_neg=-1.0)
    assert te == 2.0 / 3.0


def test_transport_efficiency_negative_dominant():
    te = transport_efficiency(s_all=-2.0, s_pos=1.0, s_neg=-3.0)
    assert te == -2.0 / -3.0


def test_transport_efficiency_nan_when_denominator_zero():
    assert math.isnan(transport_efficiency(s_all=5.0, s_pos=0.0, s_neg=-2.0))


def test_anisotropy_isotropic_turbulence_is_3c():
    a = anisotropy(var_u=1.0, var_v=1.0, var_w=1.0, cov_uv=0.0, cov_uw=0.0, cov_vw=0.0, tke=1.5)
    assert math.isclose(a.l1, 0.0, abs_tol=1e-12)
    assert math.isclose(a.l2, 0.0, abs_tol=1e-12)
    assert math.isclose(a.l3, 0.0, abs_tol=1e-12)
    assert a.aniso_class == "3c"


def test_anisotropy_one_component_turbulence_is_1c():
    a = anisotropy(var_u=2.0, var_v=0.0, var_w=0.0, cov_uv=0.0, cov_uw=0.0, cov_vw=0.0, tke=1.0)
    assert math.isclose(a.l1, 2.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(a.l2, -1.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(a.l3, -1.0 / 3.0, rel_tol=1e-12)
    assert a.aniso_class == "1c"


def test_anisotropy_two_component_turbulence_is_2c():
    a = anisotropy(var_u=1.0, var_v=1.0, var_w=0.0, cov_uv=0.0, cov_uw=0.0, cov_vw=0.0, tke=1.0)
    assert math.isclose(a.l1, 1.0 / 6.0, rel_tol=1e-12)
    assert math.isclose(a.l2, 1.0 / 6.0, rel_tol=1e-12)
    assert math.isclose(a.l3, -1.0 / 3.0, rel_tol=1e-12)
    assert a.aniso_class == "2c"


def test_anisotropy_diagonal_tensor_angles_are_zero_when_streamwise_aligned():
    a = anisotropy(var_u=1.2667, var_v=0.4667, var_w=0.2667, cov_uv=0.0, cov_uw=0.0, cov_vw=0.0, tke=1.0)
    assert math.isclose(a.l1, 0.3, abs_tol=1e-4)
    assert math.isclose(a.l2, -0.1, abs_tol=1e-4)
    assert math.isclose(a.l3, -0.2, abs_tol=1e-4)
    assert math.isclose(a.beta, 0.0, abs_tol=1e-6)
    assert math.isclose(a.phi, 0.0, abs_tol=1e-6)


def test_anisotropy_zero_tke_is_nan():
    a = anisotropy(var_u=1.0, var_v=1.0, var_w=1.0, cov_uv=0.0, cov_uw=0.0, cov_vw=0.0, tke=0.0)
    assert math.isnan(a.l1) and math.isnan(a.beta) and math.isnan(a.phi)
    assert a.aniso_class is None


def test_anisotropy_nan_input_is_nan():
    a = anisotropy(var_u=math.nan, var_v=1.0, var_w=1.0, cov_uv=0.0, cov_uw=0.0, cov_vw=0.0, tke=1.5)
    assert math.isnan(a.l1)
    assert a.aniso_class is None
