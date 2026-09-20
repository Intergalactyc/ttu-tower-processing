"""The rolling-buffer work unit: per (batch, boom), streams Stage A -> Stage B
-> products with bounded context, then writes fragments and its manifest entry.
"""
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ttu_tower import flags, schema
from ttu_tower.constants import SAMPLE_HZ, STAGE_B_MARGIN_S
from ttu_tower.io.load import BadFileError, load_boom
from ttu_tower.io.rawfiles import accepted_half_hours
from ttu_tower.io.store import write_fragment, write_manifest_entry
from ttu_tower.primary.partition import Batch
from ttu_tower.primary.products import file_products
from ttu_tower.primary.stage_a import StageA, stage_a
from ttu_tower.primary.stage_b import StageBOut, build_span, stage_b
from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR

_MARGIN = STAGE_B_MARGIN_S * SAMPLE_HZ
_TABLES = ("coverage", "means", "slot_qc", "flags", "mrd", "mrd_frame", "ladder", "ladder_coverage", "slot_boom")


@dataclass
class UnitSummary:
    files_loaded: int = 0
    files_missing: int = 0
    slot_counts: dict[str, int] = field(default_factory=dict)
    flagged_fractions: dict[str, float] = field(default_factory=dict)
    unexcised_computed: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    numpy_warnings: dict[str, int] = field(default_factory=dict)


def _empty_table(table: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(schema.TABLES[table].columns))


def run_unit(batch: Batch, boom: int, file_table: pd.DataFrame, period_slots: tuple[int, int], cfg, out_dir) -> UnitSummary:
    """Streams half-hours `[batch.h_a - 3, batch.h_b + 3]` through Stage A,
    Stage B (with a rolling buffer bounded to the margin each needs) and
    products, then writes every collected table as one fragment per table and
    this unit's manifest entry. Every slot-keyed table is filtered to
    `period_slots` (`[slot_a, slot_b)`) except `flags`, which stays clipped to
    the core and otherwise unfiltered - reaches beyond the period (a
    neighbour's detection window, tertiary's `FlagStore`) still need it.
    """
    slot_a, slot_b = period_slots
    unit_id = f"{batch.id}_b{boom:02d}"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    accepted = accepted_half_hours(file_table)
    h_a, h_b = batch.h_a, batch.h_b

    A: dict[int, StageA | None] = {}
    B: dict[int, StageBOut] = {}
    collected: dict[str, list[pd.DataFrame]] = {t: [] for t in _TABLES}
    summary = UnitSummary()

    with tempfile.TemporaryDirectory(prefix="ttu_unit_") as temp_dir, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        for h in range(h_a - 3, h_b + 4):
            path = accepted.get(h)
            if path is None:
                A[h] = None
                summary.files_missing += 1
            else:
                t0 = time.perf_counter()
                try:
                    A[h] = stage_a(load_boom(path, boom, temp_dir), boom, cfg.qc)
                    summary.files_loaded += 1
                except BadFileError:
                    A[h] = None
                    summary.files_missing += 1
                summary.timings["stage_a"] = summary.timings.get("stage_a", 0.0) + (time.perf_counter() - t0)

            hb = h - 1
            if h_a - 2 <= hb <= h_b + 2:
                span = build_span(A, hb, _MARGIN)
                t0 = time.perf_counter()
                B[hb] = stage_b(span, hb, boom, cfg)
                summary.timings["stage_b"] = summary.timings.get("stage_b", 0.0) + (time.perf_counter() - t0)
                A.pop(hb - 1, None)

            hp = hb - 2
            if h_a <= hp <= h_b:
                parts = file_products(hp, B, boom, cfg, timings=summary.timings)
                for table, df in parts.items():
                    if table != "flags" and "slot" in df.columns:
                        df = df[(df["slot"] >= slot_a) & (df["slot"] < slot_b)]
                    collected[table].append(df)
                B.pop(hp - 2, None)

        t0 = time.perf_counter()
        for table in _TABLES:
            df = pd.concat(collected[table], ignore_index=True) if collected[table] else _empty_table(table)
            write_fragment(Path(out_dir) / "data" / table, unit_id, schema.cast(table, df))
        summary.timings["writing"] = time.perf_counter() - t0

        warning_counts: dict[str, int] = {}
        for w in caught:
            key = str(w.message)
            warning_counts[key] = warning_counts.get(key, 0) + 1
        summary.numpy_warnings = warning_counts

    slot_boom_all = pd.concat(collected["slot_boom"], ignore_index=True) if collected["slot_boom"] else _empty_table("slot_boom")
    summary.slot_counts = {str(k): int(v) for k, v in slot_boom_all["status"].value_counts().items()}
    summary.unexcised_computed = int(slot_boom_all["unexcised_computed"].sum())

    flags_all = pd.concat(collected["flags"], ignore_index=True) if collected["flags"] else _empty_table("flags")
    n_core_files = h_b - h_a + 1
    for test, group in flags_all.groupby("test", observed=True):
        n_vars = len(flags.TESTS[test].variables) if flags.TESTS[test].variables else 1
        possible = n_core_files * SAMPLES_PER_HALF_HOUR * n_vars
        flagged = int((group["end"] - group["start"]).sum())
        summary.flagged_fractions[str(test)] = flagged / possible if possible else float("nan")

    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    write_manifest_entry(Path(out_dir) / "manifest", unit_id, {
        "unit": unit_id, "status": "success", "error": None,
        "started": started, "finished": finished, "summary": asdict(summary),
    })

    return summary
