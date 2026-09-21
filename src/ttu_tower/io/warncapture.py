"""Numpy/runtime-warning capture: turns raw warning spam into a
{message: count} tally recorded in a stage's summary instead of letting it
print, without changing anything the warning was already telling numpy to
do (division/exp results are unaffected - only the printed notice is
suppressed and counted). Originally `primary/stream.py`'s own pattern,
factored out so every stage counts warnings the same way.
"""
import warnings
from contextlib import contextmanager


@contextmanager
def capture_warnings(counts: dict[str, int]):
    """Every warning raised inside the block is tallied into `counts` (by
    its message string) instead of being printed.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        key = str(w.message)
        counts[key] = counts.get(key, 0) + 1
