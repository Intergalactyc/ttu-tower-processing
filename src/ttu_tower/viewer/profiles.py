"""Vertical profiles of a per-boom quantity over a range of slots: each boom's
mean (± σ) or median (± MAD), over every slot or per stability class, and a
curve fitted up the tower.
"""
import warnings

import numpy as np
import pandas as pd

from ttu_tower.constants import HEIGHTS
from ttu_tower.math.polar import yamartino_std
from ttu_tower.viewer import curvefits
from ttu_tower.viewer.decimate import visible_slice
from ttu_tower.viewer.timeline import in_class

ALL = "all"
SELECTED, REST = "selected", "the rest"
FITS = {"power law": "power law", "log law": "neutral log law (x = height)", "linear": "linear",
        "logarithmic": "logarithmic"}
_SHOWN_PARAMS = {"power law": {"b": "α", "a": "a"}, "log law": {"ustar": "u*", "z0": "z0"},
                 "linear": {"b": "slope", "a": "intercept"}, "logarithmic": {"b": "b", "a": "a"}}
_POWER_LAW_VARIABLES = ("ws", "ti", "ti_u", "ti_v", "ti_w")


def relevant(q) -> bool:
    """A profile needs one value per boom (not pairs, slot-level values or categories)."""
    return q is not None and q.kind == "boom" and not q.categorical


def default_fit(q) -> str | None:
    return "power law" if q is not None and q.variable in _POWER_LAW_VARIABLES and q.table == "boom_final" else None


def _centre(values: np.ndarray, method: str, circular: bool) -> tuple[float, float]:
    """(centre, spread): mean ± σ or median ± MAD; a direction's vector mean ± Yamartino σ."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan, np.nan
    if circular:
        s, c = np.sin(np.radians(v)).mean(), np.cos(np.radians(v)).mean()
        return float(np.degrees(np.arctan2(s, c)) % 360.0), float(yamartino_std(s, c))
    if method == "median":
        m = float(np.median(v))
        return m, float(np.median(np.abs(v - m)))
    return float(v.mean()), float(v.std())


def profile_stats(data, members, x_range, stability, method: str = "mean", by_stability: bool = False,
                  selection: np.ndarray | None = None) -> pd.DataFrame:
    """One row per (group, boom): height, centre, spread and N, where group is
    "all", each stability class, or (given brushed `selection` slots) the
    selected slots and the rest.
    """
    q = data.quantity
    if selection is not None:
        groups = [SELECTED, REST]
    else:
        groups = [ALL] if not by_stability or stability is None else list(stability[1])
    rows = []
    for m in members:
        curve = data.curves.get(m)
        if curve is None or not isinstance(m, (int, np.integer)):
            continue
        sl = slice(None) if x_range is None else visible_slice(curve.x, x_range[0], x_range[1], margin=0)
        y, slots = curve.y[sl], curve.slots[sl]
        chosen = np.isin(slots, selection) if selection is not None else None
        for g in groups:
            if g == ALL:
                v = y
            elif g in (SELECTED, REST):
                v = y[chosen if g == SELECTED else ~chosen]
            else:
                v = y[in_class(stability, slots, g)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centre, spread = _centre(v, method, q.circular)
            rows.append({"group": g, "boom": int(m), "height": HEIGHTS[int(m)], "centre": centre, "spread": spread,
                         "n": int(np.isfinite(v).sum())})
    frame = pd.DataFrame(rows, columns=["group", "boom", "height", "centre", "spread", "n"])
    order = {g: i for i, g in enumerate(groups)}  # classes from unstable to stable, as configured
    return frame.sort_values(["group", "height"], key=lambda c: c.map(order) if c.name == "group" else c,
                             kind="stable").reset_index(drop=True)


def fit_profile(name: str, heights: np.ndarray, values: np.ndarray):
    """A curve of value against height (None where there are too few booms or
    the fit fails); returns (FitResult, shown parameters).
    """
    ok = np.isfinite(heights) & np.isfinite(values)
    if ok.sum() < 3:
        return None, {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = curvefits.fit(FITS[name], heights[ok], values[ok])
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None, {}
    shown = {label: result.params[k] for k, label in _SHOWN_PARAMS[name].items() if k in result.params}
    return result, shown
