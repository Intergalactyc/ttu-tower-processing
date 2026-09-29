"""Flagged samples per slot, straight from the stored flag intervals: one
weighted bincount per fragment rather than a FlagStore (which sorts every
interval of a boom first).
"""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ttu_tower.timegrid import SAMPLES_PER_SLOT

_RAW_BOUNDS_NAMES = {"u": "ue", "v": "vn"}  # older runs stored sonic bounds flags under the raw names


def boom_counts(run, index, boom: int, tests=None) -> dict:
    """{(test, variable or None): flagged samples in each period slot} for one
    boom (only `tests`, if given), counted per slot straight from the stored
    intervals.
    """
    slot_a, slot_b = run.period
    n_slots = slot_b - slot_a
    keys: dict[tuple, int] = {}
    totals: list[np.ndarray] = []
    for path in index.table("flags").paths(booms=[boom]):
        table = pq.read_table(path, columns=["start", "end", "test", "variable"],
                              filters=[("test", "in", list(tests))] if tests is not None else None)
        if table.num_rows == 0:
            continue
        test_codes, tests = _codes(table["test"])
        var_codes, variables = _codes(table["variable"])
        combo = test_codes * (len(variables) + 1) + (var_codes + 1)
        lookup = np.full(len(tests) * (len(variables) + 1), -1, dtype=np.int64)  # (test, variable) code -> key
        for c in np.flatnonzero(np.bincount(combo, minlength=lookup.size)):
            t, v = divmod(int(c), len(variables) + 1)
            test, var = tests[t], (variables[v - 1] if v > 0 else None)
            if test == "bounds":
                var = _RAW_BOUNDS_NAMES.get(var, var)
            if (test, var) not in keys:
                keys[(test, var)] = len(keys)
                totals.append(np.zeros(n_slots, dtype=np.int64))
            lookup[c] = keys[(test, var)]
        rows = lookup[combo]
        starts = table["start"].to_numpy().astype(np.int64)
        ends = table["end"].to_numpy().astype(np.int64)
        # only the slots this file covers
        lo = max(int(starts.min()) // SAMPLES_PER_SLOT, slot_a)
        hi = min((int(ends.max()) - 1) // SAMPLES_PER_SLOT + 1, slot_b)
        if hi <= lo:
            continue
        counts = per_slot_samples(starts, ends, rows, len(keys), lo, hi - lo)
        for i in np.unique(rows):
            totals[i][lo - slot_a:hi - slot_a] += counts[i]
    return {key: totals[i].astype(np.int32) for key, i in keys.items()}


def _codes(column) -> tuple[np.ndarray, list]:
    arr = column.combine_chunks()
    if not hasattr(arr, "indices"):
        arr = pc.dictionary_encode(arr)
    return arr.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64), arr.dictionary.to_pylist()


def per_slot_samples(starts, ends, rows, n_rows: int, slot_a: int, n_slots: int) -> np.ndarray:
    """(n_rows, n_slots): samples of [start, end) intervals in each slot, summed
    per row (the intervals of one row don't overlap).
    """
    n = SAMPLES_PER_SLOT
    starts = np.maximum(starts, slot_a * n)
    ends = np.minimum(ends, (slot_a + n_slots) * n)
    keep = ends > starts
    starts, ends, rows = starts[keep], ends[keep], rows[keep]
    first, last = starts // n - slot_a, (ends - 1) // n - slot_a
    width = n_slots + 1
    size = n_rows * width
    one = first == last
    out = np.bincount(rows[one] * width + first[one], weights=(ends - starts)[one], minlength=size).astype(np.float64)
    s, e, r, f, la = starts[~one], ends[~one], rows[~one], first[~one], last[~one]
    if s.size:
        out += np.bincount(r * width + f, weights=(f + slot_a + 1) * n - s, minlength=size)  # the first slot's part
        out += np.bincount(r * width + la, weights=e - (la + slot_a) * n, minlength=size)  # the last slot's
        # whole slots strictly between: +n from first+1 through last-1, as a difference array
        diff = np.bincount(r * width + f + 1, minlength=size).astype(np.float64)
        diff -= np.bincount(r * width + la, minlength=size)
        out += n * np.cumsum(diff.reshape(n_rows, width), axis=1).ravel()
    return np.rint(out.reshape(n_rows, width)[:, :n_slots]).astype(np.int64)
