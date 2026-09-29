"""Linked brushing: which points a lasso encloses, which slots a dragged time
span covers, and how a new selection combines with the old one.
"""
import numpy as np

from ttu_tower.viewer.timeaxis import SLOT_SECONDS, unix_to_slot

EMPTY = np.empty(0, dtype=np.int64)


def in_polygon(x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Whether each point (x, y) lies inside the polygon (px, py) (even-odd rule;
    the polygon closes itself).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    n = len(px)
    if n < 3:
        return inside
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = px[i], py[i], px[j], py[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < at)
        j = i
    return inside


def slots_in_span(x0: float, x1: float) -> np.ndarray:
    """The slots a dragged span [x0, x1] (unix seconds, either order) touches."""
    lo, hi = sorted((x0, x1))
    return np.arange(int(unix_to_slot(lo)), int(unix_to_slot(hi)) + 1, dtype=np.int64)


def combine(old: np.ndarray, new: np.ndarray, mode: str) -> np.ndarray:
    """`mode`: "replace", "add" or "remove"; sorted unique slots."""
    new = np.unique(np.asarray(new, dtype=np.int64))
    if mode == "add":
        return np.union1d(old, new)
    if mode == "remove":
        return np.setdiff1d(old, new)
    return new


def runs(slots: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive stretches of sorted slots as (first, last + 1)."""
    if slots.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(slots) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [slots.size - 1]))
    return [(int(slots[a]), int(slots[b]) + 1) for a, b in zip(starts, ends)]


def describe(slots: np.ndarray) -> str:
    if slots.size == 0:
        return "no selection"
    stretches = len(runs(slots))
    hours = slots.size * SLOT_SECONDS / 3600
    return (f"{slots.size} slot{'s' if slots.size != 1 else ''} ({hours:.3g} h) in {stretches} "
            f"stretch{'es' if stretches != 1 else ''}")
