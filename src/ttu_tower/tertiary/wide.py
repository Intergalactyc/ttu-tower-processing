"""The monthly wide export (`wide/<YYYY-MM>.parquet`): a `slot_start`-indexed
pivot of every tertiary table, one calendar month (in `[output].timezone`) at
a time. A whole year pivoted at once would need about 1 GB of memory.
"""
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower.io import store

_SOURCE_TABLES = ("boom_final", "tau_final", "pairs", "profile", "slot_final")


def _variant_suffix(variant: str) -> str:
    return "" if variant == "none" else f"_{variant}"


def _pivot_boom_final(boom_final: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    df = boom_final.copy()
    var_part = df["variable"].astype(str).where(df["stat"].isna(), df["variable"].astype(str) + "_" + df["stat"].astype(str))
    df["column"] = var_part + "_b" + df["boom"].astype(str) + df["variant"].astype(str).map(_variant_suffix)
    wide = df.pivot_table(index="slot_start", columns="column", values="value", aggfunc="first")
    return wide.reindex(index)


def _pivot_tau_final(tau_final: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    df = tau_final.copy()
    suffix = "_b" + df["boom"].astype(str) + "_" + df["variant"].astype(str)
    tau_wide = df.assign(column="tau" + suffix).pivot_table(
        index="slot_start", columns="column", values="tau_s", aggfunc="first",
    )
    source_wide = df.assign(column="tau_source" + suffix).pivot(
        index="slot_start", columns="column", values="source",
    )
    return pd.concat([tau_wide.reindex(index), source_wide.reindex(index)], axis=1)


def _pivot_pairs(pairs: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pairs.copy()
    df["column"] = df["variable"].astype(str) + "_b" + df["boom"].astype(str) + "-b" + df["boom2"].astype(str)
    wide = df.pivot_table(index="slot_start", columns="column", values="value", aggfunc="first")
    return wide.reindex(index)


def _pivot_profile(profile: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    df = profile.copy()
    df["column"] = df["variable"].astype(str) + df["variant"].astype(str).map(_variant_suffix)
    wide = df.pivot_table(index="slot_start", columns="column", values="value", aggfunc="first")
    return wide.reindex(index)


def _pivot_slot_final(slot_final: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    wide = slot_final.pivot_table(index="slot_start", columns="variable", values="value", aggfunc="first")
    return wide.reindex(index)


def pivot_tables(tables: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    """One month's (or any slot_start subset's) wide frame, from the 5
    slot_start-keyed tertiary tables (`filter_log` has no place here).
    """
    parts = [
        _pivot_boom_final(tables["boom_final"], index),
        _pivot_tau_final(tables["tau_final"], index),
        _pivot_pairs(tables["pairs"], index),
        _pivot_profile(tables["profile"], index),
        _pivot_slot_final(tables["slot_final"], index),
    ]
    wide = pd.concat(parts, axis=1)
    wide.index.name = "slot_start"
    return wide.reset_index()


def write_wide_export(stage_dir: Path, timezone: str) -> None:
    """One `wide/<YYYY-MM>.parquet` per calendar month, each built from only
    that month's rows of the 5 source tables - read fresh per month via a
    `slot_start` range filter pushed down to the parquet scan, never the
    full year at once (see the module docstring).
    """
    table_dirs = {t: Path(stage_dir) / "data" / t for t in _SOURCE_TABLES}
    all_starts = pd.concat(
        [store.read_table(table_dirs[t], columns=["slot_start"])["slot_start"] for t in _SOURCE_TABLES],
        ignore_index=True,
    ).drop_duplicates()
    if all_starts.empty:
        return
    # A source table with zero rows contributes an object-dtype empty
    # series; concatenating it with the others' real tz-aware timestamps
    # otherwise degrades the whole result to object dtype.
    all_starts = pd.to_datetime(all_starts)

    local = all_starts.dt.tz_convert(timezone)
    month_keys = local.dt.strftime("%Y-%m")
    wide_dir = Path(stage_dir) / "wide"
    wide_dir.mkdir(parents=True, exist_ok=True)

    for month_key in sorted(month_keys.unique()):
        index = pd.DatetimeIndex(sorted(all_starts[month_keys == month_key].unique()))
        lo, hi = index.min(), index.max()
        month_filter = (pa_dataset.field("slot_start") >= lo) & (pa_dataset.field("slot_start") <= hi)
        month_tables = {t: store.read_table(table_dirs[t], filter=month_filter) for t in _SOURCE_TABLES}
        month_df = pivot_tables(month_tables, index)
        store.write_fragment(wide_dir, month_key, month_df)
