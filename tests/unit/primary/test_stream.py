import numpy as np
import pandas as pd

from ttu_tower.config import config_from_dict
from ttu_tower.io.rawfiles import build_file_table
from ttu_tower.io.store import read_table
from ttu_tower.primary.partition import Batch
from ttu_tower.primary.stream import _TABLES, run_unit
from ttu_tower.validation.synthetic import MESOSCALE_TYPICAL, write_raw_dataset


def _cfg(raw_dir, **sections):
    raw = {"tag": "t", "paths": {"raw_dirs": [str(raw_dir)]}, "files": {"bad_records": []}}
    raw.update(sections)
    return config_from_dict(raw)


def test_run_unit_writes_all_fragments_and_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(42)
    half_hours = list(range(1000, 1008))  # 8 half-hours
    write_raw_dataset(raw_dir, half_hours, [1], rng, waves=MESOSCALE_TYPICAL)

    cfg = _cfg(raw_dir)
    table = build_file_table([str(raw_dir)], cfg.files)
    assert (table["status"] == "accepted").all()

    batch = Batch(h_a=1002, h_b=1005)
    out_dir = tmp_path / "primary"
    period_slots = (3 * batch.h_a, 3 * (batch.h_b + 1))
    summary = run_unit(batch, boom=1, file_table=table, period_slots=period_slots, cfg=cfg, out_dir=out_dir)

    assert summary.files_loaded > 0
    assert summary.slot_counts.get("computed", 0) + summary.slot_counts.get("no_data", 0) == 12  # 4 files x 3 slots
    assert "stage_a" in summary.timings and summary.timings["stage_a"] > 0
    assert "stage_b" in summary.timings and summary.timings["stage_b"] > 0
    assert "writing" in summary.timings

    unit_id = f"{batch.id}_b01"
    for t in _TABLES:
        frag = out_dir / "data" / t / f"{unit_id}.parquet"
        assert frag.is_file(), t

    manifest = out_dir / "manifest" / f"{unit_id}.json"
    assert manifest.is_file()
    import json
    entry = json.loads(manifest.read_text())
    assert entry["status"] == "success"
    assert entry["unit"] == unit_id

    slot_boom = read_table(out_dir / "data" / "slot_boom")
    assert len(slot_boom) == 12
    assert set(slot_boom["slot"]) == {3 * h + i for h in range(1002, 1006) for i in range(3)}


def test_run_unit_handles_missing_file_in_batch(tmp_path):
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(7)
    half_hours = list(range(2000, 2008))
    write_raw_dataset(raw_dir, half_hours, [1], rng, missing=[2004])

    cfg = _cfg(raw_dir)
    table = build_file_table([str(raw_dir)], cfg.files)

    batch = Batch(h_a=2003, h_b=2005)
    out_dir = tmp_path / "primary"
    period_slots = (3 * batch.h_a, 3 * (batch.h_b + 1))
    summary = run_unit(batch, boom=1, file_table=table, period_slots=period_slots, cfg=cfg, out_dir=out_dir)

    slot_boom = read_table(out_dir / "data" / "slot_boom")
    no_file_slots = slot_boom[slot_boom["status"] == "no_file"]
    assert len(no_file_slots) == 3
    assert set(no_file_slots["slot"]) == {3 * 2004, 3 * 2004 + 1, 3 * 2004 + 2}
