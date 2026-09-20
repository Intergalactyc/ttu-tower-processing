"""The QC test registry and the interval-keyed flag store. Flags are
intervals [start, end) of global sample indices, produced once at each
test's own grain; any later window asks what fraction of it a test flagged.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttu_tower import schema

BOOM_LEVEL_VARIABLES = ("ue", "vn", "w", "ts")

_SEVEN_SERIES = ("ue", "vn", "w", "ts", "t", "rh", "p")
_SONIC_TS = ("ue", "vn", "w", "ts")


@dataclass(frozen=True)
class TestSpec:
    name: str
    kind: str  # "quality" | "record"
    grain: str  # "sample" | "window" | "slot"
    variables: tuple[str, ...] | None  # None = boom-level (applies to BOOM_LEVEL_VARIABLES)


TESTS: dict[str, TestSpec] = {
    spec.name: spec
    for spec in (
        TestSpec("bounds", "quality", "sample", _SEVEN_SERIES),
        TestSpec("spike", "quality", "sample", _SEVEN_SERIES),
        TestSpec("excursion", "record", "sample", _SEVEN_SERIES),
        TestSpec("unchecked", "quality", "sample", _SEVEN_SERIES),
        TestSpec("resolution", "quality", "window", _SONIC_TS),
        TestSpec("dropouts", "quality", "window", _SONIC_TS),
        TestSpec("skew", "quality", "slot", _SONIC_TS),
        TestSpec("kurt", "quality", "slot", _SONIC_TS),
        TestSpec("direction", "quality", "slot", None),
        TestSpec("bounce", "quality", "slot", None),
    )
}


def mask_to_intervals(mask: np.ndarray, g0: int) -> tuple[np.ndarray, np.ndarray]:
    """Runs of True in `mask` as [start, end) global sample indices."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    diff = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(diff == 1) + 1
    ends = np.flatnonzero(diff == -1) + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [mask.size]))
    return (starts + g0).astype(np.int64), (ends + g0).astype(np.int64)


def _merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge touching or overlapping [start, end) intervals."""
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    merged_s, merged_e = [], []
    cur_s, cur_e = s[0], e[0]
    for i in range(1, len(s)):
        if s[i] <= cur_e:
            cur_e = max(cur_e, e[i])
        else:
            merged_s.append(cur_s)
            merged_e.append(cur_e)
            cur_s, cur_e = s[i], e[i]
    merged_s.append(cur_s)
    merged_e.append(cur_e)
    return np.array(merged_s, dtype=np.int64), np.array(merged_e, dtype=np.int64)


class FlagRows:
    """Accumulates flag intervals as they're found, then merges and casts them
    to the `flags` table schema.
    """

    def __init__(self):
        self._rows: list[tuple[str, int, str | None, int, int]] = []

    def add(self, test: str, boom: int, variable: str | None, starts: np.ndarray, ends: np.ndarray) -> None:
        for s, e in zip(np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)):
            self._rows.append((test, boom, variable, int(s), int(e)))

    def frame(self) -> pd.DataFrame:
        columns = ["start", "end", "test", "kind", "boom", "variable"]
        if not self._rows:
            return schema.cast("flags", pd.DataFrame(columns=columns))
        df = pd.DataFrame(self._rows, columns=["test", "boom", "variable", "start", "end"])
        out_rows = []
        for (test, boom, variable), g in df.groupby(["test", "boom", "variable"], dropna=False):
            s, e = _merge_intervals(g["start"].to_numpy(), g["end"].to_numpy())
            kind = TESTS[test].kind
            for a, b in zip(s, e):
                out_rows.append({"start": a, "end": b, "test": test, "kind": kind, "boom": boom, "variable": variable})
        return schema.cast("flags", pd.DataFrame(out_rows, columns=columns))


class FlagStore:
    """Per-(test, boom, variable) sorted, merged interval arrays with prefix
    sums, for O(log n) flagged-sample queries over arbitrary supports.
    """

    def __init__(self, frame: pd.DataFrame):
        self._keys: dict[tuple[str, int, str | None], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if frame.empty:
            return
        for (test, boom, variable), g in frame.groupby(["test", "boom", "variable"], dropna=False, observed=True):
            starts = g["start"].to_numpy(dtype=np.int64)
            order = np.argsort(starts)
            starts = starts[order]
            ends = g["end"].to_numpy(dtype=np.int64)[order]
            prefix = np.concatenate(([0], np.cumsum(ends - starts)))
            var = None if pd.isna(variable) else str(variable)
            self._keys[(str(test), int(boom), var)] = (starts, ends, prefix)

    def _count_key(self, key: tuple[str, int, str | None], qs: np.ndarray, qe: np.ndarray) -> np.ndarray:
        entry = self._keys.get(key)
        if entry is None:
            return np.zeros(len(qs), dtype=np.int64)
        starts, ends, prefix = entry
        lo = np.searchsorted(ends, qs, side="right")
        hi = np.searchsorted(starts, qe, side="left")
        total = np.zeros(len(qs), dtype=np.int64)
        valid = lo < hi
        lo_v, hi_v = lo[valid], hi[valid]
        left_over = np.maximum(0, qs[valid] - starts[lo_v])
        right_over = np.maximum(0, ends[hi_v - 1] - qe[valid])
        total[valid] = prefix[hi_v] - prefix[lo_v] - left_over - right_over
        return total

    def count(self, test: str, boom: int, variable: str, starts, ends) -> np.ndarray:
        """Flagged samples of `variable` within each [start, end) query, by `test`."""
        qs = np.asarray(starts, dtype=np.int64)
        qe = np.asarray(ends, dtype=np.int64)
        total = self._count_key((test, boom, variable), qs, qe)
        if variable in BOOM_LEVEL_VARIABLES:
            total = total + self._count_key((test, boom, None), qs, qe)
        return total

    def fraction(self, test: str, boom: int, variable: str, starts, ends) -> np.ndarray:
        qs = np.asarray(starts, dtype=np.int64)
        qe = np.asarray(ends, dtype=np.int64)
        return self.count(test, boom, variable, qs, qe) / (qe - qs)

    def mask(self, tests: list[str], boom: int, variable: str, g0: int, n: int) -> np.ndarray:
        """A boolean mask of length n, True wherever any of `tests` flagged `variable`."""
        out = np.zeros(n, dtype=bool)
        for test in tests:
            keys = [(test, boom, variable)]
            if variable in BOOM_LEVEL_VARIABLES:
                keys.append((test, boom, None))
            for key in keys:
                entry = self._keys.get(key)
                if entry is None:
                    continue
                starts, ends, _ = entry
                lo = np.searchsorted(ends, g0, side="right")
                hi = np.searchsorted(starts, g0 + n, side="left")
                for s, e in zip(starts[lo:hi], ends[lo:hi]):
                    out[max(s, g0) - g0 : min(e, g0 + n) - g0] = True
        return out
