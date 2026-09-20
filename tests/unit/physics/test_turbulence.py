import math

from ttu_tower.physics.turbulence import transport_efficiency


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
