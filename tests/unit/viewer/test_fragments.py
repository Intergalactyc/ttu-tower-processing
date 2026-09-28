import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower.io.store import read_table
from ttu_tower.viewer.fragments import FragmentIndex, alias_bounds_variables

from viewer_fixtures import PRIMARY_MISSING


def test_every_core_half_hour_maps_to_its_batch(primary_run):
    table = FragmentIndex(primary_run.run_dir).table("mrd")
    batches = sorted({(f.h_a, f.h_b) for f in table.batched})
    assert len(batches) == 2
    for h_a, h_b in batches:
        for h in range(h_a, h_b + 1):
            assert table.batch_for_half_hour(h) == (h_a, h_b)
    assert table.batch_for_half_hour(batches[0][0] - 1) is None
    assert table.batch_for_half_hour(batches[-1][1] + 1) is None


def test_targeted_read_equals_a_filtered_whole_table_read(primary_run):
    index = FragmentIndex(primary_run.run_dir)
    h = PRIMARY_MISSING[0] + 1
    k = 3 * h
    flt = pa_dataset.field("slot") == k
    targeted = index.read("means", booms=[1], half_hours=[h], filter=flt)
    whole = read_table(primary_run.run_dir / "primary" / "data" / "means", filter=flt)
    key = ["slot", "boom", "variable", "stat"]
    pd.testing.assert_frame_equal(
        targeted.sort_values(key).reset_index(drop=True), whole.sort_values(key).reset_index(drop=True),
        check_categorical=False,
    )


def test_flags_window_spans_a_batch_edge(primary_run):
    index = FragmentIndex(primary_run.run_dir)
    table = index.table("flags")
    h_edge = min(f.h_b for f in table.batched)  # last half-hour of the first batch
    g_lo, g_hi = h_edge * 90_000 + 45_000, (h_edge + 1) * 90_000 + 45_000
    window = index.flags_window(1, g_lo, g_hi)
    whole = alias_bounds_variables(read_table(primary_run.run_dir / "primary" / "data" / "flags"))
    whole = whole[(whole["boom"] == 1) & (whole["end"] > g_lo) & (whole["start"] < g_hi)]
    assert len(window) == len(whole)


def test_bounds_flags_under_raw_names_are_aliased():
    flags = pd.DataFrame({
        "start": [0, 0, 0, 0], "end": [5, 5, 5, 5], "test": ["bounds", "bounds", "bounds", "spike"],
        "kind": "quality", "boom": 1, "variable": ["u", "v", "w", "ue"],
    })
    out = alias_bounds_variables(flags)
    assert out["variable"].astype(str).tolist() == ["ue", "vn", "w", "ue"]


def test_outage_fragment_is_indexed_separately(primary_run):
    table = FragmentIndex(primary_run.run_dir).table("slot_boom")
    assert [p.stem for p in table.other] == ["outages"]
    assert table.paths(include_other=True)[-1].stem == "outages"
