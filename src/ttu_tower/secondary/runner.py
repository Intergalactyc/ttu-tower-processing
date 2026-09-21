"""Secondary orchestration: batches in sequence, in one process, each one
reading that batch's primary fragments (every boom), computing every
secondary table, and writing one fragment per table.
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ttu_tower import __version__, schema
from ttu_tower.config.hashing import primary_hash, secondary_hash
from ttu_tower.io import store
from ttu_tower.io.rawfiles import period_slots_from_table
from ttu_tower.io.runs import register_run
from ttu_tower.io.stage import check_hash_guard, write_run_meta, write_run_summary
from ttu_tower.primary.partition import Batch, plan_batches
from ttu_tower.secondary import sun
from ttu_tower.secondary.derived import (
    derived, materialize_missing_unexcised, naive_tau_selected, selected_stats, tau_tables,
)
from ttu_tower.secondary.slow import slow_table

_PRIMARY_TABLES = ("ladder", "ladder_coverage", "mrd", "coverage", "means", "slot_boom")
_SECONDARY_TABLES = ("tau", "tau_selected", "boom_stats", "boom_labels", "slow", "slot_stats")
_MARKER_TABLE = "boom_stats"  # a batch is "done" once this table's fragment exists
_PRIMARY_MARKER_TABLE = "means"  # any primary unit's fragment existing implies the unit succeeded

_logger = logging.getLogger(__name__)


def _read_primary_batch(primary_dir: Path, batch: Batch, booms) -> dict[str, pd.DataFrame]:
    """A failed primary unit leaves no fragment, and the run is allowed to
    continue past it, so a missing (batch, boom) is skipped here rather than
    raised.
    """
    usable_booms = []
    for boom in booms:
        if (primary_dir / "data" / _PRIMARY_MARKER_TABLE / f"{batch.id}_b{boom:02d}.parquet").is_file():
            usable_booms.append(boom)
        else:
            _logger.warning(f"primary unit {batch.id}_b{boom:02d} has no output; skipping this boom for the batch")

    tables = {}
    for table in _PRIMARY_TABLES:
        frames = [pd.read_parquet(primary_dir / "data" / table / f"{batch.id}_b{boom:02d}.parquet") for boom in usable_booms]
        tables[table] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return tables


def _process_batch(batch: Batch, primary_dir: Path, booms, slots_all: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    primary = _read_primary_batch(primary_dir, batch, booms)

    tau_df, tau_selected_real = tau_tables(primary["mrd"], primary["coverage"], cfg.secondary)
    slow_df = slow_table(primary["means"], primary["coverage"], cfg)
    p_factor = slow_df[slow_df["variable"] == "p_factor"].set_index(["slot", "boom"])["value"]

    naive_df = naive_tau_selected(tau_selected_real[tau_selected_real["variant"] == "mrd"])
    tau_selected_all = pd.concat([tau_selected_real, naive_df], ignore_index=True)

    stats_ladder = selected_stats(primary["ladder"], primary["ladder_coverage"], tau_selected_all, p_factor)
    stats_extra, labels_df = derived(stats_ladder, tau_selected_all, slow_df, cfg)
    boom_stats_df = pd.concat([stats_ladder, stats_extra], ignore_index=True)

    slot_a, slot_b = 3 * batch.h_a, 3 * (batch.h_b + 1)
    batch_slots = slots_all[(slots_all["slot"] >= slot_a) & (slots_all["slot"] < slot_b)]
    slot_stats_df = sun.slot_stats(batch_slots, cfg.output.timezone)

    tables = {
        "tau": tau_df, "tau_selected": tau_selected_all, "boom_stats": boom_stats_df,
        "boom_labels": labels_df, "slow": slow_df, "slot_stats": slot_stats_df,
    }
    return materialize_missing_unexcised(tables, primary["slot_boom"])


def _accumulate(totals: dict, tau_selected: pd.DataFrame) -> None:
    mrd_only = tau_selected[tau_selected["variant"] == "mrd"]
    source_status_counts = totals.setdefault("source_status_counts", {})
    for status, n in mrd_only["source_status"].value_counts().items():
        source_status_counts[str(status)] = source_status_counts.get(str(status), 0) + int(n)


def run_secondary(cfg, args) -> dict:
    """Registers the run, guards the config hash against primary's, and
    processes every batch of the file table primary already wrote, in order.
    Returns the run summary dict.
    """
    run_dir = register_run(cfg, args.config, test=args.test, force=args.force, package_version=__version__)
    primary_dir = run_dir / "primary"
    stage_dir = run_dir / "secondary"
    config_hash = secondary_hash(cfg, __version__)
    check_hash_guard(stage_dir, config_hash, force=args.force)
    stage_dir.mkdir(parents=True, exist_ok=True)

    write_run_meta(stage_dir, stage="secondary", tag=cfg.tag, config_hash=config_hash,
                    upstream_hash=primary_hash(cfg, __version__), package_version=__version__,
                    timezone=cfg.output.timezone, unusable_tests=cfg.qc.unusable_tests)

    file_table = pd.read_parquet(primary_dir / "files.parquet")
    slots_all = pd.read_parquet(primary_dir / "slots.parquet")
    period_slots = period_slots_from_table(slots_all)
    batches, _ = plan_batches(file_table, period_slots, cfg.primary.batch_max_files)
    if args.test:
        batches = batches[:1]

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    status_counts = {"processed": 0, "skipped": 0}
    totals: dict = {}

    for batch in batches:
        marker = stage_dir / "data" / _MARKER_TABLE / f"{batch.id}.parquet"
        if marker.is_file():
            status_counts["skipped"] += 1
            continue

        tables = _process_batch(batch, primary_dir, cfg.files.booms, slots_all, cfg)
        for table in _SECONDARY_TABLES:
            store.write_fragment(stage_dir / "data" / table, batch.id, schema.cast(table, tables[table]))
        status_counts["processed"] += 1
        _accumulate(totals, tables["tau_selected"])

    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    write_run_summary(stage_dir, stage="secondary", config_hash=config_hash, package_version=__version__,
                       started=started, finished=finished, status_counts=status_counts, totals=totals, timings={})

    print(f"secondary: {status_counts['processed']} batches processed, {status_counts['skipped']} skipped")
    print(f"tau source status (mrd): {totals.get('source_status_counts', {})}")

    return {"status_counts": status_counts, "totals": totals}
