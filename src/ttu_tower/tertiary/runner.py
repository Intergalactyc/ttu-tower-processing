"""Tertiary orchestration: batches in sequence, in one process, mirroring
secondary - each batch reads that batch's primary and secondary fragments,
computes every tertiary table, and writes one fragment per table. After
every batch, the wide export is written month by month.
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ttu_tower import __version__, schema
from ttu_tower.config.hashing import secondary_hash, tertiary_hash
from ttu_tower.flags import FlagStore
from ttu_tower.io import store
from ttu_tower.io.rawfiles import period_slots_from_table
from ttu_tower.io.runs import register_run
from ttu_tower.io.stage import check_hash_guard, write_run_meta, write_run_summary
from ttu_tower.primary.partition import Batch, plan_batches
from ttu_tower.tertiary import filtering, mesonet, multiboom, profiles, wide

_PRIMARY_TABLES = ("coverage", "means", "ladder_coverage")
_PRIMARY_MARKER_TABLE = "means"  # any primary unit's fragment existing implies the unit succeeded
_SECONDARY_TABLES = ("tau_selected", "boom_stats", "slow", "slot_stats", "boom_labels")
_SECONDARY_MARKER_TABLE = "boom_stats"
_TERTIARY_TABLES = ("boom_final", "tau_final", "pairs", "profile", "slot_final", "filter_log", "boom_labels_final")
_TERTIARY_MARKER_TABLE = "boom_final"  # a batch is "done" once this table's fragment exists
_SLOT_START_TABLES = ("boom_final", "tau_final", "pairs", "profile", "slot_final", "boom_labels_final")

_logger = logging.getLogger(__name__)


def _read_boom_fragment(primary_dir: Path, table: str, batch_id: str, boom: int) -> pd.DataFrame | None:
    path = primary_dir / "data" / table / f"{batch_id}_b{boom:02d}.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _read_primary_batch(primary_dir: Path, batch: Batch, booms) -> tuple[dict[str, pd.DataFrame], list[int]]:
    """A failed primary unit leaves no fragment, and the run is allowed to
    continue past it, so a missing (batch, boom) is skipped here rather than
    raised (same tolerance as secondary/runner.py).
    """
    usable = [b for b in booms if (primary_dir / "data" / _PRIMARY_MARKER_TABLE / f"{batch.id}_b{b:02d}.parquet").is_file()]
    for b in sorted(set(booms) - set(usable)):
        _logger.warning(f"primary unit {batch.id}_b{b:02d} has no output; skipping this boom for the batch")

    tables = {}
    for table in _PRIMARY_TABLES:
        frames = [f for b in usable if (f := _read_boom_fragment(primary_dir, table, batch.id, b)) is not None]
        tables[table] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=schema.TABLES[table].columns)
    return tables, usable


def _flag_store_for_boom(primary_dir: Path, batches: list[Batch], i: int, boom: int) -> FlagStore:
    """A `FlagStore` from this batch's own flags fragment plus its immediate
    neighbours' - the +-5 min reach a 20-min-window bounds/spike query can
    need at a batch edge. The 1200s rung's own `ladder_coverage` needs no
    such reach: primary already computed it over the full 20-min window.
    """
    neighbours = ([batches[i - 1]] if i > 0 else []) + [batches[i]] + ([batches[i + 1]] if i + 1 < len(batches) else [])
    frames = [f for b in neighbours if (f := _read_boom_fragment(primary_dir, "flags", b.id, boom)) is not None]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=schema.TABLES["flags"].columns)
    return FlagStore(combined)


def _read_secondary_batch(secondary_dir: Path, batch: Batch) -> dict[str, pd.DataFrame] | None:
    marker = secondary_dir / "data" / _SECONDARY_MARKER_TABLE / f"{batch.id}.parquet"
    if not marker.is_file():
        _logger.warning(f"secondary batch {batch.id} has no output; skipping it for tertiary")
        return None
    return {table: pd.read_parquet(secondary_dir / "data" / table / f"{batch.id}.parquet") for table in _SECONDARY_TABLES}


def _slot_final_rows(batch_slots: pd.DataFrame, slot_stats: pd.DataFrame, meso_df: pd.DataFrame | None) -> pd.DataFrame:
    """`sun_elevation`/`night` for every slot, regardless of whether a
    mesonet merge is configured; `night` is a 0/1 float, not a bool.
    """
    sun = slot_stats[["slot", "sun_elevation", "night"]].copy()
    sun["night"] = sun["night"].astype(float)
    rows = sun.melt(id_vars=["slot"], var_name="variable", value_name="value")

    if meso_df is not None:
        merged = mesonet.combine(batch_slots[["slot", "slot_start"]], meso_df)
        meso_rows = merged.melt(
            id_vars=["slot", "slot_start"], value_vars=list(meso_df.columns), var_name="variable", value_name="value",
        ).drop(columns=["slot_start"])
        rows = pd.concat([rows, meso_rows], ignore_index=True)
    return rows


def _process_batch(i: int, batches: list[Batch], primary_dir: Path, secondary_dir: Path,
                    slots_all: pd.DataFrame, meso_df: pd.DataFrame | None, cfg) -> dict[str, pd.DataFrame] | None:
    batch = batches[i]
    secondary = _read_secondary_batch(secondary_dir, batch)
    if secondary is None:
        return None
    primary, usable_booms = _read_primary_batch(primary_dir, batch, cfg.files.booms)
    if not usable_booms:
        _logger.warning(f"batch {batch.id} has no usable primary booms; skipping")
        return None

    flag_stores = {b: _flag_store_for_boom(primary_dir, batches, i, b) for b in usable_booms}

    candidate = filtering.build_candidate(primary["means"], secondary["slow"], secondary["boom_stats"])
    fail_keys, filter_log = filtering.evaluate(
        candidate, primary["coverage"], primary["ladder_coverage"], secondary["tau_selected"], flag_stores, cfg.tertiary,
    )
    boom_final = filtering.apply(candidate, fail_keys)
    boom_labels_final = filtering.apply_to_labels(secondary["boom_labels"], fail_keys)

    pairs = multiboom.multiboom(boom_final, cfg.files.booms, cfg.tertiary.veer_reference_boom)
    profile = profiles.profiles(boom_final, secondary["tau_selected"], cfg.tertiary.fits)
    tau_final = secondary["tau_selected"].copy()

    slot_a, slot_b = 3 * batch.h_a, 3 * (batch.h_b + 1)
    batch_slots = slots_all[(slots_all["slot"] >= slot_a) & (slots_all["slot"] < slot_b)]
    slot_final = _slot_final_rows(batch_slots, secondary["slot_stats"], meso_df)

    tables = {
        "boom_final": boom_final, "tau_final": tau_final, "pairs": pairs,
        "profile": profile, "slot_final": slot_final, "filter_log": filter_log,
        "boom_labels_final": boom_labels_final,
    }
    slot_start_by_slot = batch_slots.set_index("slot")["slot_start"]
    for name in _SLOT_START_TABLES:
        tables[name] = tables[name].assign(slot_start=tables[name]["slot"].map(slot_start_by_slot))
    return tables


def run_tertiary(cfg, args) -> dict:
    """Registers the run, guards the config hash against secondary's, and
    processes every batch of the file table primary already wrote, in order.
    Returns the run summary dict.
    """
    run_dir = register_run(cfg, args.config, test=args.test, force=args.force, package_version=__version__)
    primary_dir = run_dir / "primary"
    secondary_dir = run_dir / "secondary"
    stage_dir = run_dir / "tertiary"
    config_hash = tertiary_hash(cfg, __version__)
    check_hash_guard(stage_dir, config_hash, force=args.force)
    stage_dir.mkdir(parents=True, exist_ok=True)

    write_run_meta(stage_dir, stage="tertiary", tag=cfg.tag, config_hash=config_hash,
                    upstream_hash=secondary_hash(cfg, __version__), package_version=__version__,
                    timezone=cfg.output.timezone, unusable_tests=cfg.qc.unusable_tests)

    file_table = pd.read_parquet(primary_dir / "files.parquet")
    slots_all = pd.read_parquet(primary_dir / "slots.parquet")
    period_slots = period_slots_from_table(slots_all)
    batches, _ = plan_batches(file_table, period_slots, cfg.primary.batch_max_files)
    if args.test:
        batches = batches[:1]

    meso_df = mesonet.mesonet_data(cfg.paths.mesonet_dir, cfg.output.timezone) if cfg.paths.mesonet_dir else None

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    status_counts = {"processed": 0, "skipped": 0}

    for i, batch in enumerate(batches):
        marker = stage_dir / "data" / _TERTIARY_MARKER_TABLE / f"{batch.id}.parquet"
        if marker.is_file():
            status_counts["skipped"] += 1
            continue

        tables = _process_batch(i, batches, primary_dir, secondary_dir, slots_all, meso_df, cfg)
        if tables is None:
            status_counts["skipped"] += 1
            continue
        for table in _TERTIARY_TABLES:
            store.write_fragment(stage_dir / "data" / table, batch.id, schema.cast(table, tables[table]))
        status_counts["processed"] += 1

    wide_source_tables = ("boom_final", "tau_final", "pairs", "profile", "slot_final")
    full_tables = {t: store.read_table(stage_dir / "data" / t) for t in wide_source_tables}
    wide.write_wide_export(stage_dir, full_tables, cfg.output.timezone)

    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    write_run_summary(stage_dir, stage="tertiary", config_hash=config_hash, package_version=__version__,
                       started=started, finished=finished, status_counts=status_counts, totals={}, timings={})

    print(f"tertiary: {status_counts['processed']} batches processed, {status_counts['skipped']} skipped")

    return {"status_counts": status_counts}
