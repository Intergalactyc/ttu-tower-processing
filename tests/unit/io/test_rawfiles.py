import numpy as np
import pandas as pd
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.io.rawfiles import (
    RawFilesNotAllowed, accepted_half_hours, build_file_table, check_raw_allowed, parse_name,
    resolve_period_slots,
)


def _files_cfg(bad_records=(), max_name_offset_min=2):
    cfg = config_from_dict({
        "tag": "t", "paths": {"raw_dirs": ["/data"]},
        "files": {"bad_records": list(bad_records) or [[999_999, 999_999]], "max_name_offset_min": max_name_offset_min},
    })
    return cfg.files


def _write_parquet(path, n_rows=90_000):
    pd.DataFrame({"u_1": np.zeros(n_rows, dtype=np.float32)}).to_parquet(path)


# --- parse_name ----------------------------------------------------------------

def test_parse_name_valid_parquet():
    result = parse_name("FT2_E07_C03_R01457_D20131101_T0000.parquet")
    assert result is not None
    record, name_time, kind = result
    assert record == 1457
    assert kind == "parquet"
    assert (name_time.year, name_time.month, name_time.day) == (2013, 11, 1)
    assert (name_time.hour, name_time.minute) == (0, 0)
    assert str(name_time.tz) == "Etc/GMT+6"


@pytest.mark.parametrize("ext", ["csv", "csv.gz", "zip"])
def test_parse_name_raw_kinds(ext):
    record, name_time, kind = parse_name(f"FT2_E07_C03_R00001_D20131101_T0000.{ext}")
    assert kind == "raw"


def test_parse_name_crossing_midnight():
    record, name_time, kind = parse_name("FT2_E07_C03_R00001_D20131101_T2359.parquet")
    assert (name_time.hour, name_time.minute) == (23, 59)


@pytest.mark.parametrize("name", [
    "FT2_E07_C03_R01457_D20131101_T0000.txt",  # wrong extension
    "FT2_E07_C03_R01457_D20131101_T0000",  # no extension
    "FT2_E07_C03_D20131101_T0000.parquet",  # missing record
    "FT2_E07_C03_Rabc_D20131101_T0000.parquet",  # non-numeric record
    "FT2_E07_C99_R01457_D20131101_T0000.parquet",  # wrong site code
    "FT2_E07_C03_R01457_D2013110_T0000.parquet",  # short date
    "FT2_E07_C03_R01457_D20131101_T000.parquet",  # short time
])
def test_parse_name_malformed(name):
    assert parse_name(name) is None


def test_parse_name_invalid_embedded_date():
    assert parse_name("FT2_E07_C03_R00001_D20131301_T0000.parquet") is None  # month 13
    assert parse_name("FT2_E07_C03_R00001_D20131101_T2500.parquet") is None  # hour 25


# --- build_file_table: rule order ----------------------------------------------

def test_bad_record_dropped_before_offset_or_collision(tmp_path):
    _write_parquet(tmp_path / "FT2_E07_C03_R00050_D20131101_T0000.parquet")
    cfg = _files_cfg(bad_records=[[50, 50]])
    table = build_file_table([str(tmp_path)], cfg)
    row = table.iloc[0]
    assert row["status"] == "bad_record"


def test_offset_within_tolerance_accepted_beyond_rejected(tmp_path):
    _write_parquet(tmp_path / "FT2_E07_C03_R00001_D20131101_T0002.parquet")
    _write_parquet(tmp_path / "FT2_E07_C03_R00002_D20131101_T0033.parquet")
    cfg = _files_cfg(max_name_offset_min=2)
    table = build_file_table([str(tmp_path)], cfg).set_index("record")
    assert table.loc[1, "status"] == "accepted"
    assert table.loc[1, "offset_min"] == 2
    assert table.loc[2, "status"] == "offset"
    assert table.loc[2, "offset_min"] == 3


def test_two_files_rounding_to_same_half_hour_both_rejected(tmp_path):
    _write_parquet(tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.parquet")
    _write_parquet(tmp_path / "FT2_E07_C03_R00002_D20131101_T0001.parquet")
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    assert set(table["status"]) == {"collision"}


def test_collision_only_among_survivors_not_with_bad_record(tmp_path):
    # record 50 is bad_record and shares a half-hour with record 1, which
    # should still be accepted - collision only applies among survivors.
    _write_parquet(tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.parquet")
    _write_parquet(tmp_path / "FT2_E07_C03_R00050_D20131101_T0000.parquet")
    cfg = _files_cfg(bad_records=[[50, 50]])
    table = build_file_table([str(tmp_path)], cfg).set_index("record")
    assert table.loc[1, "status"] == "accepted"
    assert table.loc[50, "status"] == "bad_record"


def test_bad_length_parquet(tmp_path):
    _write_parquet(tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.parquet", n_rows=100)
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    assert table.iloc[0]["status"] == "bad_length"


def test_bad_length_not_checked_for_raw_files(tmp_path):
    (tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.zip").touch()
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    assert table.iloc[0]["status"] == "accepted"
    assert pd.isna(table.iloc[0]["n_rows"])


def test_unparseable_name_with_recognized_extension(tmp_path):
    (tmp_path / "not_a_tower_file.parquet").touch()
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    row = table.iloc[0]
    assert row["status"] == "unparseable"
    assert pd.isna(row["record"])
    assert pd.isna(row["half_hour"])


def test_unrecognized_extensions_not_scanned(tmp_path):
    (tmp_path / "readme.txt").touch()
    (tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.parquet.bak").touch()
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    assert len(table) == 0


def test_accepted_half_hours(tmp_path):
    _write_parquet(tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.parquet")
    _write_parquet(tmp_path / "FT2_E07_C03_R00002_D20131101_T0030.parquet", n_rows=1)  # bad_length
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    result = accepted_half_hours(table)
    assert len(result) == 1
    (h, path), = result.items()
    assert path.name == "FT2_E07_C03_R00001_D20131101_T0000.parquet"


def test_empty_raw_dirs_returns_empty_table(tmp_path):
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    assert len(table) == 0
    assert list(table.columns) == ["path", "name", "record", "name_time", "offset_min", "file_start", "half_hour", "n_rows", "status"]


# --- check_raw_allowed ----------------------------------------------------------

def test_check_raw_allowed_blocks_accepted_raw_files(tmp_path):
    (tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.zip").touch()
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    with pytest.raises(RawFilesNotAllowed, match=r"1 unconverted raw files"):
        check_raw_allowed(table, allow_non_parquet=False)


def test_check_raw_allowed_passes_with_flag(tmp_path):
    (tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.zip").touch()
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    check_raw_allowed(table, allow_non_parquet=True)  # no raise


def test_check_raw_allowed_ignores_non_accepted_raw_files(tmp_path):
    (tmp_path / "FT2_E07_C03_R00050_D20131101_T0000.zip").touch()
    cfg = _files_cfg(bad_records=[[50, 50]])
    table = build_file_table([str(tmp_path)], cfg)
    check_raw_allowed(table, allow_non_parquet=False)  # bad_record, not accepted - no raise


def test_check_raw_allowed_ignores_accepted_parquet_files(tmp_path):
    _write_parquet(tmp_path / "FT2_E07_C03_R00001_D20131101_T0000.parquet")
    cfg = _files_cfg()
    table = build_file_table([str(tmp_path)], cfg)
    check_raw_allowed(table, allow_non_parquet=False)  # no raise


# --- resolve_period_slots --------------------------------------------------------

def _full_cfg(start="", end="", raw_dirs=("/data",)):
    return config_from_dict({
        "tag": "t", "paths": {"raw_dirs": list(raw_dirs)}, "files": {"bad_records": []},
        "period": {"start": start, "end": end},
    })


def _table_with_accepted(half_hours):
    return pd.DataFrame({
        "half_hour": list(half_hours),
        "status": ["accepted"] * len(half_hours),
    })


def test_resolve_period_slots_explicit_bounds():
    cfg = _full_cfg(start="2013-11-01 00:00", end="2013-11-01 01:00")
    slot_a, slot_b = resolve_period_slots(cfg, _table_with_accepted([]))
    from ttu_tower.timegrid import time_to_slot
    expected_a = int(time_to_slot(pd.Timestamp("2013-11-01 00:00", tz="Etc/GMT+6")))
    assert (slot_a, slot_b) == (expected_a, expected_a + 6)  # 1 hour = 6 slots


def test_resolve_period_slots_defaults_from_accepted_files():
    cfg = _full_cfg(start="", end="")
    table = _table_with_accepted([100, 101, 105])
    slot_a, slot_b = resolve_period_slots(cfg, table)
    assert slot_a == 3 * 100
    assert slot_b == 3 * (105 + 1)


def test_resolve_period_slots_mixes_explicit_start_with_default_end():
    cfg = _full_cfg(start="2013-11-01 00:00", end="")
    table = _table_with_accepted([100, 101, 105])
    slot_a, slot_b = resolve_period_slots(cfg, table)
    from ttu_tower.timegrid import time_to_slot
    assert slot_a == int(time_to_slot(pd.Timestamp("2013-11-01 00:00", tz="Etc/GMT+6")))
    assert slot_b == 3 * 106


def test_resolve_period_slots_empty_default_raises_with_no_accepted_files():
    cfg = _full_cfg(start="", end="")
    with pytest.raises(ValueError):
        resolve_period_slots(cfg, _table_with_accepted([]))
