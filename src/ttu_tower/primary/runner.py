"""Primary orchestration: the file table, batches, the config-hash guard,
dispatch to a process pool with the logging queue, and the run summary.
"""
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from ttu_tower import __version__, schema
from ttu_tower.config.hashing import primary_hash
from ttu_tower.io import store
from ttu_tower.io.rawfiles import build_file_table, check_raw_allowed, resolve_period_slots
from ttu_tower.io.runs import register_run
from ttu_tower.io.stage import check_hash_guard, write_run_meta, write_run_summary
from ttu_tower.logs import configure_worker, log_listener
from ttu_tower.primary.partition import Batch, plan_batches
from ttu_tower.primary.stream import run_unit
from ttu_tower.timegrid import slot_to_time

_TABLES = ("coverage", "means", "slot_qc", "flags", "mrd", "mrd_frame", "ladder", "ladder_coverage", "slot_boom")


def _worker(batch: Batch, boom: int, file_table: pd.DataFrame, period_slots: tuple[int, int], cfg, out_dir: str):
    """Runs in a worker process. A raising unit is caught here (not left to
    the pool), logged, and marked failed in the manifest; the pool continues.
    Success writes its own manifest entry inside run_unit.
    """
    unit_id = f"{batch.id}_b{boom:02d}"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        summary = run_unit(batch, boom, file_table, period_slots, cfg, out_dir)
        return unit_id, "success", None, asdict(summary)
    except Exception as e:
        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        store.write_manifest_entry(Path(out_dir) / "manifest", unit_id, {
            "unit": unit_id, "status": "failed", "error": error,
            "started": started, "finished": finished, "summary": {},
        })
        logging.getLogger("MAIN").error(f"unit {unit_id} failed: {e}", exc_info=True)
        return unit_id, "failed", error, {}


def _write_slots_and_outages(stage_dir: Path, file_table: pd.DataFrame, period_slots: tuple[int, int],
                              outage_half_hours: list[int], booms, timezone: str) -> None:
    slot_a, slot_b = period_slots
    accepted = file_table[file_table["status"] == "accepted"].set_index("half_hour")

    slots = list(range(slot_a, slot_b))
    half_hours = [k // 3 for k in slots]
    is_file = [h in accepted.index for h in half_hours]
    records = [int(accepted.loc[h, "record"]) if is_file[i] else None for i, h in enumerate(half_hours)]
    slot_start = slot_to_time(slots).tz_convert(timezone) if slots else []
    slots_df = pd.DataFrame({
        "slot": slots, "slot_start": slot_start,
        "half_hour": half_hours, "file_status": ["file" if f else "no_file" for f in is_file], "record": records,
    })
    store.write_fragment(stage_dir, "slots", schema.cast("slots", slots_df))

    outage_rows = []
    for h in outage_half_hours:
        for k in (3 * h, 3 * h + 1, 3 * h + 2):
            if slot_a <= k < slot_b:
                for boom in booms:
                    outage_rows.append((k, boom, "no_file", False))
    outages_df = pd.DataFrame(outage_rows, columns=["slot", "boom", "status", "unexcised_computed"])
    store.write_fragment(stage_dir / "data" / "slot_boom", "outages", schema.cast("slot_boom", outages_df))


def run_primary(cfg, args) -> dict:
    """Registers the run, guards the config hash, builds the file table and
    batches, dispatches every (batch, boom) work unit to a process pool, and
    writes the run summary. Returns the run summary dict.
    """
    run_dir = register_run(cfg, args.config, test=args.test, force=args.force, package_version=__version__)
    stage_dir = run_dir / "primary"
    config_hash = primary_hash(cfg, __version__)
    check_hash_guard(stage_dir, config_hash, force=args.force)
    stage_dir.mkdir(parents=True, exist_ok=True)

    write_run_meta(stage_dir, stage="primary", tag=cfg.tag, config_hash=config_hash, upstream_hash=None,
                    package_version=__version__, timezone=cfg.output.timezone, unusable_tests=cfg.qc.unusable_tests)

    file_table = build_file_table(cfg.paths.raw_dirs, cfg.files)
    check_raw_allowed(file_table, args.allow_non_parquet)
    store.write_fragment(stage_dir, "files", file_table)

    period_slots = resolve_period_slots(cfg, file_table)
    batches, outage_half_hours = plan_batches(file_table, period_slots, cfg.primary.batch_max_files)
    if args.test:
        batches = batches[:1]

    _write_slots_and_outages(stage_dir, file_table, period_slots, outage_half_hours, cfg.files.booms, cfg.output.timezone)

    manifest = store.read_manifest(stage_dir / "manifest")
    units = []
    for batch in batches:
        for boom in cfg.files.booms:
            unit_id = f"{batch.id}_b{boom:02d}"
            entry = manifest.get(unit_id)
            if entry is not None and entry.get("status") == "success":
                continue
            if entry is not None and entry.get("status") == "failed" and not args.redo_failures:
                continue
            units.append((batch, boom))

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    status_counts = {"success": 0, "failed": 0}
    totals: dict = {}
    timings: dict = {}

    nproc = args.nproc or cfg.primary.nproc or 1
    ctx = get_context("spawn")
    logs_dir = stage_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    queue = ctx.Queue()
    listener = ctx.Process(target=log_listener, args=(queue, str(logs_dir / "primary.log")))
    listener.start()
    try:
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx, initializer=configure_worker, initargs=(queue,)) as executor:
            futures = [executor.submit(_worker, batch, boom, file_table, period_slots, cfg, str(stage_dir)) for batch, boom in units]
            for future in as_completed(futures):
                unit_id, status, error, summary_dict = future.result()
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "success":
                    _accumulate(totals, timings, summary_dict)
    finally:
        queue.put(None)
        listener.join(timeout=10)

    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    write_run_summary(stage_dir, stage="primary", config_hash=config_hash, package_version=__version__,
                       started=started, finished=finished, status_counts=status_counts, totals=totals, timings=timings)

    print(f"primary: {status_counts.get('success', 0)} succeeded, {status_counts.get('failed', 0)} failed")
    print(f"slots: {totals.get('slot_counts', {})}")

    return {"status_counts": status_counts, "totals": totals, "timings": timings}


def _accumulate(totals: dict, timings: dict, summary: dict) -> None:
    totals["files_loaded"] = totals.get("files_loaded", 0) + summary.get("files_loaded", 0)
    totals["files_missing"] = totals.get("files_missing", 0) + summary.get("files_missing", 0)
    totals["unexcised_computed"] = totals.get("unexcised_computed", 0) + summary.get("unexcised_computed", 0)

    slot_counts = totals.setdefault("slot_counts", {})
    for status, n in summary.get("slot_counts", {}).items():
        slot_counts[status] = slot_counts.get(status, 0) + n

    numpy_warnings = totals.setdefault("numpy_warnings", {})
    for msg, n in summary.get("numpy_warnings", {}).items():
        numpy_warnings[msg] = numpy_warnings.get(msg, 0) + n

    for step, seconds in summary.get("timings", {}).items():
        timings[step] = timings.get(step, 0.0) + seconds
