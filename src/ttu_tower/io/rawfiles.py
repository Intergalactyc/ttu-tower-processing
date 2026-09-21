"""The raw file table: name parsing and the file-timing rule."""
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ttu_tower import schema
from ttu_tower.constants import ROWS_PER_FILE, SOURCE_TIMEZONE
from ttu_tower.timegrid import time_to_half_hour, time_to_slot

_NAME_RE = re.compile(r"^FT2_E07_C03_R(\d+)_D(\d{8})_T(\d{4})\.(parquet|csv|csv\.gz|zip)$")
_RAW_EXTENSIONS = (".parquet", ".csv", ".csv.gz", ".zip")

_logger = logging.getLogger(__name__)


class RawFilesNotAllowed(Exception):
    """Raised when accepted files need conversion and --allow-non-parquet wasn't passed."""


def parse_name(name: str) -> tuple[int, pd.Timestamp, str] | None:
    """Parse a raw file name into (record, name_time, kind); None if it doesn't
    match the naming convention or encodes an invalid date/time. `name_time` is
    tz-aware in the source's zone. `kind` is "parquet" or "raw".
    """
    m = _NAME_RE.match(name)
    if m is None:
        return None
    record_str, date_str, time_str, ext = m.groups()
    try:
        name_time = pd.Timestamp(
            year=int(date_str[0:4]), month=int(date_str[4:6]), day=int(date_str[6:8]),
            hour=int(time_str[0:2]), minute=int(time_str[2:4]), tz=SOURCE_TIMEZONE,
        )
    except ValueError:
        return None
    kind = "parquet" if ext == "parquet" else "raw"
    return int(record_str), name_time, kind


def _is_bad_record(record: int, bad_records) -> bool:
    return any(lo <= record <= hi for lo, hi in bad_records)


def build_file_table(raw_dirs, cfg_files) -> pd.DataFrame:
    """The file table: one row per raw file found in `raw_dirs` (non-recursive),
    with the file-timing rule's status, checked in order: unparseable,
    bad_record, offset, collision, bad_length, accepted.
    """
    paths = sorted({p for raw_dir in raw_dirs for ext in _RAW_EXTENSIONS for p in Path(raw_dir).glob(f"*{ext}")})

    rows = []
    for path in paths:
        name = path.name
        parsed = parse_name(name)
        if parsed is None:
            rows.append({
                "path": str(path), "name": name, "record": pd.NA, "name_time": pd.NaT,
                "offset_min": pd.NA, "file_start": pd.NaT, "half_hour": pd.NA,
                "n_rows": pd.NA, "status": "unparseable",
            })
            continue

        record, name_time, kind = parsed
        file_start = name_time.floor("30min")
        offset_min = round((name_time - file_start).total_seconds() / 60)
        half_hour = int(time_to_half_hour(file_start))
        n_rows = int(pq.ParquetFile(path).metadata.num_rows) if kind == "parquet" else pd.NA

        if _is_bad_record(record, cfg_files.bad_records):
            status = "bad_record"
        elif offset_min > cfg_files.max_name_offset_min:
            status = "offset"
        else:
            status = "candidate"

        rows.append({
            "path": str(path), "name": name, "record": record, "name_time": name_time,
            "offset_min": offset_min, "file_start": file_start, "half_hour": half_hour,
            "n_rows": n_rows, "status": status,
        })

    table = pd.DataFrame(rows, columns=[
        "path", "name", "record", "name_time", "offset_min", "file_start", "half_hour", "n_rows", "status",
    ])
    if table.empty:
        return schema.cast("files", table)

    table["half_hour"] = table["half_hour"].astype("Int64")
    table["n_rows"] = table["n_rows"].astype("Int64")

    candidate_half_hours = table.loc[table["status"] == "candidate", "half_hour"]
    dup = candidate_half_hours.duplicated(keep=False)
    if dup.any():
        _logger.warning(f"{candidate_half_hours[dup].nunique()} half-hour(s) with colliding file names, all rejected")
    table.loc[candidate_half_hours.index[dup], "status"] = "collision"

    candidate = table["status"] == "candidate"
    bad_length = candidate & table["n_rows"].notna() & (table["n_rows"] != ROWS_PER_FILE)
    table.loc[bad_length, "status"] = "bad_length"
    table.loc[table["status"] == "candidate", "status"] = "accepted"

    return schema.cast("files", table)


def accepted_half_hours(table: pd.DataFrame) -> dict[int, Path]:
    """Map half-hour index -> path, for every accepted file."""
    accepted = table[table["status"] == "accepted"]
    return {int(h): Path(p) for h, p in zip(accepted["half_hour"], accepted["path"])}


def resolve_period_slots(cfg, file_table: pd.DataFrame) -> tuple[int, int]:
    """The processing period as `(slot_a, slot_b)`. An empty `[period].start`
    or `end` resolves to the first or last accepted file's own bounds; a set
    value is read in `[output].timezone`. `ttu-files` and `ttu-primary` both
    call this, so they agree on what an empty period means.
    """
    if cfg.period.start:
        slot_a = int(time_to_slot(pd.Timestamp(cfg.period.start, tz=cfg.output.timezone)))
    else:
        accepted_hh = file_table.loc[file_table["status"] == "accepted", "half_hour"]
        if accepted_hh.empty:
            raise ValueError("period.start is empty and there are no accepted files to default from")
        slot_a = 3 * int(accepted_hh.min())

    if cfg.period.end:
        slot_b = int(time_to_slot(pd.Timestamp(cfg.period.end, tz=cfg.output.timezone)))
    else:
        accepted_hh = file_table.loc[file_table["status"] == "accepted", "half_hour"]
        if accepted_hh.empty:
            raise ValueError("period.end is empty and there are no accepted files to default from")
        slot_b = 3 * (int(accepted_hh.max()) + 1)

    return slot_a, slot_b


def period_slots_from_table(slots: pd.DataFrame) -> tuple[int, int]:
    """The `(slot_a, slot_b)` primary actually wrote to `slots.parquet`, as
    the single source of truth for secondary/tertiary: re-resolving
    `[period]` independently in every stage risks the stages silently
    disagreeing about which slots exist if the config is edited between runs.
    """
    return int(slots["slot"].min()), int(slots["slot"].max()) + 1


def check_raw_allowed(table: pd.DataFrame, allow_non_parquet: bool) -> None:
    """Stop (unless `allow_non_parquet`) if any accepted file still needs conversion."""
    raw = table[(table["status"] == "accepted") & ~table["name"].str.endswith(".parquet")]
    if raw.empty:
        return
    dirs = ", ".join(sorted({str(Path(p).parent) for p in raw["path"]}))
    message = (
        f"{len(raw)} unconverted raw files in {dirs}; run `ttu-convert-parquet` first, "
        "or pass --allow-non-parquet (slow; for small runs)"
    )
    if not allow_non_parquet:
        raise RawFilesNotAllowed(message)
    _logger.warning(message)
