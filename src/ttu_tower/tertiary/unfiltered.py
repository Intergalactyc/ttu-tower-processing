"""Tertiary's tables without its filtering, recomputed on demand from a finished run.

Filtering (filtering.py) NaNs a value whose group failed a criterion, and the gradient, multi-boom
and profile products are then computed from what is left. Only the filtered tables are stored,
so the unfiltered ones are rebuilt here, batch by batch, by the same code tertiary runs
(runner.filtered_tables) with no failures: from primary `means` and secondary `slow`,
`boom_stats`, `boom_labels` and `tau_selected`, under the run's own tertiary config. The flags
and coverage that filtering reads are never loaded. The code is the installed package's, so a
product added since the run was made appears here but not in its stored tables.

Unfiltered means only that tertiary's boom-level filtering is skipped. Samples primary marked
unusable (`[qc].unusable_tests`) are still left out of every statistic, and statistics primary
or secondary couldn't compute (no usable data) are still NaN.

The tables filtering doesn't touch (`tau_final`, `slot_final`) are read as stored; `wide` is
pivoted from the recomputed tables and those, with the stored export's column names.
"""
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa

from ttu_tower import schema
from ttu_tower.config import config_from_dict
from ttu_tower.io import store
from ttu_tower.tertiary import runner, wide
from ttu_tower.timegrid import Cadence

TABLES = (*runner.FILTERED_TABLES, "wide")  # the tables load() recomputes


def run_config(run_dir: Path):
    """The config tertiary ran with (its config.resolved.json)."""
    path = Path(run_dir) / "tertiary" / "config.resolved.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the run has no tertiary stage to recompute")
    with open(path) as f:
        return config_from_dict(json.load(f))


def batches(run_dir: Path) -> Iterator[tuple[pd.DataFrame, dict[str, pd.DataFrame]]]:
    """Per batch tertiary processed: its rows of the run's slot table, and runner.FILTERED_TABLES without
    filtering, with slot_start. Batches tertiary skipped (no secondary output, or no usable primary boom) are
    skipped here too."""
    run_dir = Path(run_dir)
    cfg = run_config(run_dir)
    primary_dir, secondary_dir = run_dir / "primary", run_dir / "secondary"
    batch_list, slots_all = runner.plan_run_batches(primary_dir, cfg)
    cadence = Cadence.of(cfg)
    for batch in batch_list:
        secondary = runner._read_secondary_batch(secondary_dir, batch)
        if secondary is None:
            continue
        primary, usable_booms = runner._read_primary_batch(primary_dir, batch, cfg.files.booms, ("means",))
        if not usable_booms:
            continue
        batch_slots = runner.batch_slot_table(batch, slots_all, cadence)
        tables = runner.filtered_tables(primary, secondary, set(), cfg)
        yield batch_slots, runner.with_slot_start(tables, batch_slots)


def _wide_batch(batch_slots: pd.DataFrame, tables: dict[str, pd.DataFrame],
                stored: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One batch's wide rows: its recomputed tables and the stored rows of its slots, indexed as the
    stored export is (every slot_start any of them has)."""
    slots = batch_slots["slot"]
    tables = tables | {name: df[df["slot"].isin(slots)] for name, df in stored.items()}
    starts = pd.concat([tables[t]["slot_start"] for t in wide.SOURCE_TABLES], ignore_index=True).dropna()
    return wide.pivot_tables(tables, pd.DatetimeIndex(sorted(pd.to_datetime(starts).unique())))


def load(run_dir: Path, table: str, columns=None, filters=None) -> pd.DataFrame:
    """`table` (one of TABLES) of the run, without tertiary filtering. `columns` and `filters` (a pyarrow
    expression, fragment tables only, as in load_results) are applied batch by batch."""
    if table not in TABLES:
        raise ValueError(f"'{table}' isn't changed by tertiary filtering; recomputable tables: {list(TABLES)}")
    stored = ({t: store.read_table(Path(run_dir) / "tertiary" / "data" / t) for t in ("tau_final", "slot_final")}
              if table == "wide" else {})
    parts = []
    for batch_slots, tables in batches(run_dir):
        if table == "wide":
            df = _wide_batch(batch_slots, tables, stored)
        else:
            df = schema.cast(table, tables[table])
            if filters is not None:
                df = pa.Table.from_pandas(df, preserve_index=False).filter(filters).to_pandas()
        if columns is not None:
            df = df[[c for c in columns if c in df.columns]]
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True)
    return out if columns is None else out.reindex(columns=columns)
