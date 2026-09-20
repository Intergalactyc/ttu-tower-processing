import pandas as pd

from ttu_tower.primary.partition import Batch, plan_batches


def _table(accepted_half_hours):
    return pd.DataFrame({
        "half_hour": list(accepted_half_hours),
        "status": ["accepted"] * len(accepted_half_hours),
    })


def test_batch_id_format():
    assert Batch(h_a=5, h_b=12).id == "bat0000005-0000012"


def test_no_outages_single_batch():
    table = _table(range(100, 110))
    batches, outages = plan_batches(table, (300, 330), batch_max_files=96)
    assert outages == []
    assert batches == [Batch(100, 109)]


def test_isolated_missing_file_stays_in_one_batch():
    table = _table([100, 101, 103, 104, 105])  # 102 missing, isolated
    batches, outages = plan_batches(table, (300, 318), batch_max_files=96)
    assert outages == []
    assert batches == [Batch(100, 105)]


def test_two_consecutive_missing_is_an_outage():
    table = _table([100, 101, 104, 105])  # 102, 103 missing
    batches, outages = plan_batches(table, (300, 318), batch_max_files=96)
    assert outages == [102, 103]
    assert batches == [Batch(100, 101), Batch(104, 105)]


def test_outage_at_start_and_end():
    table = _table([102, 103, 104])  # 100,101 missing at start; 105,106 missing at end
    batches, outages = plan_batches(table, (300, 321), batch_max_files=96)
    assert outages == [100, 101, 105, 106]
    assert batches == [Batch(102, 104)]


def test_all_missing_gives_one_outage_no_batches():
    table = _table([])
    batches, outages = plan_batches(table, (300, 309), batch_max_files=96)
    assert batches == []
    assert outages == [100, 101, 102]


def test_batch_max_files_splits_a_stretch():
    table = _table(range(100, 110))
    batches, outages = plan_batches(table, (300, 330), batch_max_files=4)
    assert outages == []
    assert batches == [Batch(100, 103), Batch(104, 107), Batch(108, 109)]


def test_period_slot_bounds_map_to_enclosing_half_hours():
    # period covers slots 301..305 (slot 300 = h100's first slot; 301,302 also
    # h100; 303,304,305 = h101) - slot_b=306 excludes slot 306 (h102).
    table = _table([100, 101])
    batches, outages = plan_batches(table, (301, 306), batch_max_files=96)
    assert batches == [Batch(100, 101)]


def test_batch_max_files_one_gives_one_batch_per_half_hour():
    table = _table(range(100, 103))
    batches, outages = plan_batches(table, (300, 309), batch_max_files=1)
    assert batches == [Batch(100, 100), Batch(101, 101), Batch(102, 102)]
