import math

import pytest

from ttu_tower.config.model import SelectionConfig
from ttu_tower.secondary.detect import Detection
from ttu_tower.secondary.select import select

_PRIORITY_CFG = SelectionConfig(rule="priority", priority=("heat", "momentum"), fallback_tau_s=600.0)
_PRIORITY_CFG_REV = SelectionConfig(rule="priority", priority=("momentum", "heat"), fallback_tau_s=600.0)
_LONGEST_CFG = SelectionConfig(rule="longest_significant", priority=("heat", "momentum"), fallback_tau_s=600.0)


def _det(status, tau_s=math.nan, tau_lb_s=math.nan):
    return Detection(status, tau_s, tau_lb_s, 1, math.nan, math.nan, None)


@pytest.mark.parametrize("heat_status,momentum_status", [
    (a, b) for a in ("found", "capped", "unresolved", "weak", "no_data")
    for b in ("found", "capped", "unresolved", "weak", "no_data")
])
def test_priority_heat_first(heat_status, momentum_status):
    heat = _det(heat_status, tau_s=75.0, tau_lb_s=150.0)
    momentum = _det(momentum_status, tau_s=300.0, tau_lb_s=600.0)
    tau_s, source, source_status = select(heat, momentum, _PRIORITY_CFG)

    if heat_status in ("found", "capped"):
        assert (tau_s, source, source_status) == (75.0, "heat", heat_status)
    elif momentum_status in ("found", "capped"):
        assert (tau_s, source, source_status) == (300.0, "momentum", momentum_status)
    elif heat_status == "unresolved":
        assert (tau_s, source, source_status) == (max(150.0, 600.0), "heat", "unresolved")
    elif momentum_status == "unresolved":
        assert (tau_s, source, source_status) == (max(600.0, 600.0), "momentum", "unresolved")
    elif heat_status != "no_data" or momentum_status != "no_data":
        assert (tau_s, source, source_status) == (600.0, "fallback", "fallback")
    else:
        assert source == "none" and source_status == "none" and math.isnan(tau_s)


@pytest.mark.parametrize("heat_status,momentum_status", [
    (a, b) for a in ("found", "capped", "unresolved", "weak", "no_data")
    for b in ("found", "capped", "unresolved", "weak", "no_data")
])
def test_priority_momentum_first(heat_status, momentum_status):
    heat = _det(heat_status, tau_s=75.0, tau_lb_s=150.0)
    momentum = _det(momentum_status, tau_s=300.0, tau_lb_s=600.0)
    tau_s, source, source_status = select(heat, momentum, _PRIORITY_CFG_REV)

    if momentum_status in ("found", "capped"):
        assert (tau_s, source, source_status) == (300.0, "momentum", momentum_status)
    elif heat_status in ("found", "capped"):
        assert (tau_s, source, source_status) == (75.0, "heat", heat_status)
    elif momentum_status == "unresolved":
        assert (tau_s, source, source_status) == (max(600.0, 600.0), "momentum", "unresolved")
    elif heat_status == "unresolved":
        assert (tau_s, source, source_status) == (max(150.0, 600.0), "heat", "unresolved")
    elif heat_status != "no_data" or momentum_status != "no_data":
        assert (tau_s, source, source_status) == (600.0, "fallback", "fallback")
    else:
        assert source == "none" and source_status == "none" and math.isnan(tau_s)


def test_longest_significant_picks_larger_tau():
    heat = _det("found", tau_s=75.0)
    momentum = _det("capped", tau_s=1200.0)
    tau_s, source, source_status = select(heat, momentum, _LONGEST_CFG)
    assert (tau_s, source, source_status) == (1200.0, "momentum", "capped")


def test_longest_significant_ties_break_by_priority():
    heat = _det("found", tau_s=300.0)
    momentum = _det("capped", tau_s=300.0)
    tau_s, source, source_status = select(heat, momentum, _LONGEST_CFG)
    assert (tau_s, source, source_status) == (300.0, "heat", "found")


def test_longest_significant_falls_back_to_unresolved_when_none_active():
    heat = _det("unresolved", tau_lb_s=150.0)
    momentum = _det("weak")
    tau_s, source, source_status = select(heat, momentum, _LONGEST_CFG)
    assert (tau_s, source, source_status) == (600.0, "heat", "unresolved")


def test_unresolved_lower_bound_floored_at_fallback():
    heat = _det("unresolved", tau_lb_s=100.0)
    momentum = _det("no_data")
    tau_s, source, source_status = select(heat, momentum, _PRIORITY_CFG)
    assert (tau_s, source, source_status) == (600.0, "heat", "unresolved")


def test_both_no_data_gives_none():
    heat = _det("no_data")
    momentum = _det("no_data")
    tau_s, source, source_status = select(heat, momentum, _PRIORITY_CFG)
    assert source == "none" and source_status == "none" and math.isnan(tau_s)
