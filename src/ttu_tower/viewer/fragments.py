"""Targeted reads of a run's tables: which fragment files hold a given boom
and half-hour, so a slot or boom lookup opens a handful of files instead of
scanning a whole table directory.
"""
import bisect
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower import schema
from ttu_tower.io.store import read_paths
from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR

_NAME_RE = re.compile(r"^bat(\d{7})-(\d{7})(?:_b(\d{2}))?$")
_EARTH_NAMES = {"u": "ue", "v": "vn"}


@dataclass(frozen=True)
class Fragment:
    path: Path
    h_a: int
    h_b: int
    boom: int | None


class TableFragments:
    """The fragment files of one table directory, indexed by batch and boom."""

    def __init__(self, table_dir: Path):
        self.table_dir = Path(table_dir)
        self.batched: list[Fragment] = []
        self.other: list[Path] = []  # e.g. slot_boom/outages.parquet
        for path in sorted(self.table_dir.glob("*.parquet")) if self.table_dir.is_dir() else []:
            m = _NAME_RE.match(path.stem)
            if m is None:
                self.other.append(path)
                continue
            boom = int(m.group(3)) if m.group(3) else None
            self.batched.append(Fragment(path, int(m.group(1)), int(m.group(2)), boom))
        self.batched.sort(key=lambda f: (f.h_a, f.boom or 0))
        self._batch_starts = sorted({f.h_a for f in self.batched})
        self._batch_ends = {f.h_a: f.h_b for f in self.batched}

    def batch_for_half_hour(self, h: int) -> tuple[int, int] | None:
        """The (h_a, h_b) of the batch whose core holds half-hour h; None for an
        outage half-hour, which belongs to no batch.
        """
        i = bisect.bisect_right(self._batch_starts, h) - 1
        if i < 0:
            return None
        h_a = self._batch_starts[i]
        h_b = self._batch_ends[h_a]
        return (h_a, h_b) if h <= h_b else None

    def paths(self, booms=None, half_hours=None, include_other: bool = False) -> list[Path]:
        booms = None if booms is None else set(booms)
        batches = None
        if half_hours is not None:
            batches = {b for b in (self.batch_for_half_hour(int(h)) for h in half_hours) if b is not None}
        out = []
        for f in self.batched:
            if booms is not None and f.boom is not None and f.boom not in booms:
                continue
            if batches is not None and (f.h_a, f.h_b) not in batches:
                continue
            out.append(f.path)
        if include_other:
            out.extend(self.other)
        return out


class FragmentIndex:
    """Lazily built `TableFragments` for every table of a run."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._tables: dict[str, TableFragments] = {}

    def table(self, name: str) -> TableFragments:
        if name not in self._tables:
            spec = schema.TABLES[name]
            self._tables[name] = TableFragments(self.run_dir / spec.stage / "data" / name)
        return self._tables[name]

    def read(self, name: str, *, booms=None, half_hours=None, columns=None, filter=None,
             include_other: bool = False) -> pd.DataFrame:
        """Rows of table `name` from only the fragments that can hold them.
        `booms`/`half_hours` select files; `filter` (a pyarrow expression)
        selects rows within them, so pass a boom filter too for per-batch tables.
        """
        paths = self.table(name).paths(booms=booms, half_hours=half_hours, include_other=include_other)
        return read_paths(paths, columns=columns, filter=filter)

    def flags_window(self, boom: int, g_lo: int, g_hi: int) -> pd.DataFrame:
        """Flag intervals of `boom` overlapping samples [g_lo, g_hi). Flags are
        clipped to each work unit's core, so every batch the window touches is
        read. Sonic bounds flags written under the raw names u/v (older runs)
        come back as ue/vn.
        """
        half_hours = range(g_lo // SAMPLES_PER_HALF_HOUR, (g_hi - 1) // SAMPLES_PER_HALF_HOUR + 1)
        f = pa_dataset.field
        flt = (f("boom") == boom) & (f("end") > g_lo) & (f("start") < g_hi)
        flags = self.read("flags", booms=[boom], half_hours=half_hours, filter=flt)
        return alias_bounds_variables(flags)


def alias_bounds_variables(flags: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw-frame u/v of `bounds` rows to ue/vn."""
    if flags.empty or "variable" not in flags.columns:
        return flags
    variable = flags["variable"].astype(object)
    is_raw = (flags["test"].astype(object) == "bounds") & variable.isin(list(_EARTH_NAMES))
    if not is_raw.any():
        return flags
    flags = flags.copy()
    flags["variable"] = np.where(is_raw, variable.map(_EARTH_NAMES), variable)
    flags["variable"] = flags["variable"].astype("category")
    return flags
