"""Reading jet-detection inputs from a run's tertiary output: the slot-mean
wind-speed profiles (and direction), and the per-slot quantities the event
filters use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower import schema
from ttu_tower.constants import BOOMS
from ttu_tower.io.store import read_table
from ttu_tower.jets.filters import rib_column, vpt_column
from ttu_tower.jets.identify import Profiles


def _read(run_dir, table: str, columns: list[str], filter) -> pd.DataFrame:
    spec = schema.TABLES[table]
    return read_table(Path(run_dir) / spec.stage / "data" / table, columns=columns, filter=filter)


def _per_boom(frame: pd.DataFrame, booms) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(np.nan, index=pd.Index([], name="slot"), columns=list(booms))
    wide = frame.pivot_table(index="slot", columns="boom", values="value", aggfunc="first", observed=True,
                             dropna=False)
    wide.columns = [int(b) for b in wide.columns]
    return wide.reindex(columns=list(booms))


def load_profiles(run_dir, booms=BOOMS, with_direction: bool = True) -> Profiles:
    """The `none`-variant slot means of ws (and wd) for `booms`, on the run's
    complete slot grid from its first to its last tertiary slot.
    """
    f = pa_dataset.field
    variables = ["ws", "wd"] if with_direction else ["ws"]
    rows = _read(run_dir, "boom_final", ["slot", "boom", "variable", "value"],
                 (f("variant") == "none") & (f("stat") == "mean") & f("variable").isin(variables)
                 & f("boom").isin(list(booms)))
    rows["variable"] = rows["variable"].astype(str)
    if rows.empty:
        return Profiles(np.array([], np.int64), tuple(booms), np.empty((0, len(booms))))
    ws = _per_boom(rows[rows["variable"] == "ws"], booms)
    grid = pd.RangeIndex(int(rows["slot"].min()), int(rows["slot"].max()) + 1)
    ws = ws.reindex(grid)
    extra = {"wd": _per_boom(rows[rows["variable"] == "wd"], booms)} if with_direction else {}
    return Profiles.from_frame(ws, extra)


def load_context(run_dir, vpt_booms=(), rib_pairs=(), night: bool = True) -> pd.DataFrame:
    """Per slot: `vpt_b{b}` (slow-sensor virtual potential temperature, K),
    `rib_b{a}-b{b}` and `night` (1/0), for the filters.
    """
    f = pa_dataset.field
    parts = []
    if vpt_booms:
        rows = _read(run_dir, "boom_final", ["slot", "boom", "value"],
                     (f("variant") == "none") & (f("variable") == "vpt") & f("boom").isin(list(vpt_booms)))
        wide = _per_boom(rows, vpt_booms)
        wide.columns = [vpt_column(b) for b in wide.columns]
        parts.append(wide)
    for a, b in {tuple(sorted(p)) for p in rib_pairs}:
        rows = _read(run_dir, "pairs", ["slot", "value"],
                     (f("variable") == "rib") & (f("boom") == a) & (f("boom2") == b))
        parts.append(rows.groupby("slot")["value"].first().rename(rib_column(a, b)).to_frame())
    if night:
        rows = _read(run_dir, "slot_final", ["slot", "value"], f("variable") == "night")
        parts.append(rows.groupby("slot")["value"].first().rename("night").to_frame())
    if not parts:
        return pd.DataFrame(index=pd.Index([], name="slot"))
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "slot"
    return out


def load_boom_values(run_dir, variable: str, stat: str | None, variant: str, booms=BOOMS) -> pd.DataFrame:
    """Per slot (rows) and boom (columns): one quantity of tertiary `boom_final`,
    e.g. ("tke", None, "mrd") or ("wd", "std", "mrd").
    """
    f = pa_dataset.field
    flt = (f("variant") == variant) & (f("variable") == variable) & f("boom").isin(list(booms))
    flt &= f("stat").is_null() if stat is None else f("stat") == stat
    return _per_boom(_read(run_dir, "boom_final", ["slot", "boom", "value"], flt), booms)
