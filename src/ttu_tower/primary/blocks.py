"""Per-sample sums onto the floor and finest block grids, by family. These
compress a span's samples into per-block sums once, so every coarser
aggregation (MRD modes, ladder rungs) only ever reshapes and sums blocks,
never raw samples again. Temperatures are summed as offsets from 273.15 K
for numerical conditioning purposes: the offset cancels exactly in any variance
or covariance, and keeps the values small for the second-moment sums and
differences taken from them.
"""
from dataclasses import dataclass

import numpy as np

from ttu_tower.timegrid import BlockGrid, block_sums


@dataclass
class FamilySums:
    n: np.ndarray  # (nb,) usable weight sum per block
    sx: np.ndarray  # (nb, m) weighted sum per block, per column
    columns: tuple[str, ...]


def _family_sums(columns: dict[str, np.ndarray], weight: np.ndarray, g0: int, grid: BlockGrid, b0: int, nb: int) -> FamilySums:
    names = tuple(columns)
    values = np.stack([columns[name] for name in names], axis=1)
    sx, n = block_sums(values, weight.astype(np.float64), g0, grid, b0, nb)
    return FamilySums(n=n, sx=sx, columns=names)


def floor_sums(series: dict[str, np.ndarray], masks: dict[str, np.ndarray], g0: int, floor_level: int, b0: int, nb: int) -> dict[str, FamilySums]:
    """First-order block sums on the floor grid, per family, for MRD."""
    grid = BlockGrid.floor(floor_level)
    theta = series["vpts"] - 273.15
    tau_s = series["ts"] - 273.15
    return {
        "momentum": _family_sums(
            {"ue": series["ue"], "vn": series["vn"], "w": series["w"]}, masks["momentum"], g0, grid, b0, nb
        ),
        "heat": _family_sums({"w": series["w"], "theta": theta}, masks["heat"], g0, grid, b0, nb),
        "ts": _family_sums({"tau_s": tau_s}, masks["ts"], g0, grid, b0, nb),
    }


def finest_sums(series: dict[str, np.ndarray], masks: dict[str, np.ndarray], g0: int, b0: int, nb: int) -> dict[str, FamilySums]:
    """First- and second-order block sums on the finest (9.375 s) grid, per
    family, for the statistics ladder. The "direction" entry is a sub-sum of
    ue/ws, vn/ws over momentum-and-moving samples, for a later Yamartino
    wind-direction standard deviation.
    """
    grid = BlockGrid.FINEST
    ue, vn, w = series["ue"], series["vn"], series["w"]
    ws = np.hypot(ue, vn)
    theta = series["vpts"] - 273.15
    tau_s = series["ts"] - 273.15

    momentum = _family_sums(
        {
            "ue": ue, "vn": vn, "w": w,
            "ue2": ue**2, "vn2": vn**2, "w2": w**2,
            "ue_vn": ue * vn, "ue_w": ue * w, "vn_w": vn * w,
            "ws": ws, "ws2": ws**2,
        },
        masks["momentum"], g0, grid, b0, nb,
    )

    dir_mask = masks["momentum"] & (ws > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ue_ws, vn_ws = ue / ws, vn / ws
    direction = _family_sums({"ue_ws": ue_ws, "vn_ws": vn_ws}, dir_mask, g0, grid, b0, nb)

    heat = _family_sums(
        {"w": w, "theta": theta, "w2": w**2, "theta2": theta**2, "w_theta": w * theta},
        masks["heat"], g0, grid, b0, nb,
    )

    ts = _family_sums({"tau_s": tau_s, "tau_s2": tau_s**2}, masks["ts"], g0, grid, b0, nb)

    return {"momentum": momentum, "heat": heat, "ts": ts, "direction": direction}
