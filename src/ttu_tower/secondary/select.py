"""Selecting one operative tau per (slot, boom, variant) from the heat and
momentum cospectra's detections.
"""
from ttu_tower.secondary.detect import Detection

_ACTIVE = ("found", "capped")


def select(heat: Detection, momentum: Detection, cfg_sel) -> tuple[float, str, str]:
    """(tau_s, source, source_status). `source` is "heat", "momentum",
    "fallback" or "none"; `source_status` is the chosen source's own status,
    or "fallback"/"none" to match.
    """
    by_name = {"heat": heat, "momentum": momentum}
    active = [name for name in cfg_sel.priority if by_name[name].status in _ACTIVE]

    if active:
        # max() keeps the first tie, so ties already break by priority order.
        name = max(active, key=lambda n: by_name[n].tau_s) if cfg_sel.rule == "longest_significant" else active[0]
        det = by_name[name]
        return det.tau_s, name, det.status

    for name in cfg_sel.priority:
        det = by_name[name]
        if det.status == "unresolved":
            return max(det.tau_lb_s, cfg_sel.fallback_tau_s), name, det.status

    if heat.status != "no_data" or momentum.status != "no_data":
        return cfg_sel.fallback_tau_s, "fallback", "fallback"

    return float("nan"), "none", "none"
