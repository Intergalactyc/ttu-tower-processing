"""Two runs side by side: the other run's version of a quantity, and how the
two agree where both have a value (bias, RMSE, correlation), per boom.
"""
import numpy as np
import pandas as pd

from ttu_tower.viewer.decimate import visible_slice
from ttu_tower.viewer.timeline import joined


def counterpart(catalog, q, variant: str):
    """(the other run's quantity, variant) for `q`, or (None, reason)."""
    if catalog is None:
        return None, "the other run's quantities are still loading"
    if q.key not in catalog.quantities:
        return None, f"{q.title} isn't in the other run"
    other = catalog[q.key]
    if variant not in other.variants:
        return None, f"the other run has no {variant} variant of {q.title}"
    return other, variant


def _range(curve, x_range):
    if x_range is None:
        return curve
    sl = visible_slice(curve.x, x_range[0], x_range[1], margin=0)
    return type(curve)(slots=curve.slots[sl], y=curve.y[sl])


def paired(a, b, x_range=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(slots, a values, b values) where both runs have a finite value, in range."""
    slots, ya, yb = joined(_range(a, x_range), _range(b, x_range))
    ok = np.isfinite(ya) & np.isfinite(yb)
    return slots[ok], ya[ok], yb[ok]


def differences(ya: np.ndarray, yb: np.ndarray, circular: bool) -> np.ndarray:
    """b - a (wrapped to ±180° for directions)."""
    d = yb - ya
    return (d + 180.0) % 360.0 - 180.0 if circular else d


def member_stats(a, b, x_range=None, circular: bool = False) -> dict:
    ra, rb = _range(a, x_range), _range(b, x_range)
    _, ya, yb = paired(a, b, x_range)
    d = differences(ya, yb, circular)
    both = d.size
    stats = {"N (this run)": int(np.isfinite(ra.y).sum()), "N (other)": int(np.isfinite(rb.y).sum()), "N both": both}
    if both:
        stats.update({"mean (this)": float(np.mean(ya)) if not circular else np.nan,
                      "mean (other)": float(np.mean(yb)) if not circular else np.nan,
                      "bias (other − this)": float(np.mean(d)), "RMSE": float(np.sqrt(np.mean(d ** 2))),
                      "identical %": 100.0 * float(np.mean(d == 0))})
        varied = both > 2 and not circular and np.std(ya) > 0 and np.std(yb) > 0
        stats["r"] = float(np.corrcoef(ya, yb)[0, 1]) if varied else np.nan
    return stats


def table(data_a, data_b, members, x_range=None, label=str) -> pd.DataFrame:
    circular = data_a.quantity.circular
    rows = {}
    for m in members:
        a, b = data_a.curves.get(m), data_b.curves.get(m)
        if a is None or b is None:
            continue
        rows[label(m)] = member_stats(a, b, x_range, circular)
    return pd.DataFrame.from_dict(rows, orient="index").dropna(axis=1, how="all")
