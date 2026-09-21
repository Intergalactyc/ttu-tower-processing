"""Named interval classification: an ordered list of (name, interval) bins,
applied to a whole array at once. Ports the interval-string syntax of old
windprofiles' `SingleClassifier` (open/closed bounds, `inf`/`-inf`), not its
class hierarchy.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


class ClassifyError(Exception):
    """A class interval string is malformed."""


@dataclass(frozen=True)
class _Interval:
    low: float
    low_inclusive: bool
    high: float
    high_inclusive: bool

    def contains(self, values: np.ndarray) -> np.ndarray:
        lo_ok = values >= self.low if self.low_inclusive else values > self.low
        hi_ok = values <= self.high if self.high_inclusive else values < self.high
        return lo_ok & hi_ok


def _parse_interval(interval: str) -> _Interval:
    """`(a,b)`, `[a,b)`, `(a,b]` or `[a,b]`; `a`/`b` are floats or `inf`/
    `-inf` (case-insensitive, `float()`'s own parsing).
    """
    s = interval.replace(" ", "")
    if len(s) < 3 or s[0] not in "[(" or s[-1] not in "])":
        raise ClassifyError(f"malformed interval '{interval}': must look like '(a,b)' or '[a,b)'")
    low_inclusive = s[0] == "["
    high_inclusive = s[-1] == "]"
    parts = s[1:-1].split(",")
    if len(parts) != 2:
        raise ClassifyError(f"malformed interval '{interval}': must contain exactly one comma")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise ClassifyError(f"malformed interval '{interval}': bounds must be numeric, 'inf' or '-inf'") from e
    return _Interval(low, low_inclusive, high, high_inclusive)


class IntervalClassifier:
    """`classes`: (name, interval-string) pairs, tried in order - the first
    matching interval wins. `.classify(values)` gives None for NaN or when no
    interval matches (both fall out of plain float comparison: a NaN bound
    check is always False, so it can never match any interval).
    """

    def __init__(self, classes: list[tuple[str, str]]):
        self._names = [name for name, _ in classes]
        self._intervals = [_parse_interval(interval) for _, interval in classes]

    def classify(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        result = np.full(values.shape, None, dtype=object)
        assigned = np.zeros(values.shape, dtype=bool)
        for name, interval in zip(self._names, self._intervals):
            hit = interval.contains(values) & ~assigned
            result[hit] = name
            assigned |= hit
        return result


def stability_classes(pairs: pd.DataFrame, cfg_post) -> pd.Series:
    """slot -> stability class name (or None), from tertiary's `pairs` table:
    the `rib` value of the configured `post.stability.pair`.
    """
    boom, boom2 = cfg_post.stability.pair
    rib = pairs[(pairs["variable"] == "rib") & (pairs["boom"] == boom) & (pairs["boom2"] == boom2)]
    classifier = IntervalClassifier(list(cfg_post.stability.classes))
    classes = classifier.classify(rib["value"].to_numpy())
    return pd.Series(classes, index=rib["slot"].to_numpy(), name="stability_class")
