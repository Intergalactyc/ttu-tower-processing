import json

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.viewer import brush, find, timeline
from ttu_tower.viewer.bookmarks import DEFAULT_LIST, Bookmark, BookmarkStore
from ttu_tower.viewer.catalog import build_catalog
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.run import RunHandle
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, slot_to_unix

from viewer_fixtures import FULL_BOOMS, SPIKY_BOOM, SPIKY_SLOT


def test_lasso_encloses_points_by_the_even_odd_rule():
    # a U shape: its notch is outside
    px = np.array([0, 3, 3, 2, 2, 1, 1, 0], dtype=float)
    py = np.array([0, 0, 3, 3, 1, 1, 3, 3], dtype=float)
    x = np.array([0.5, 1.5, 2.5, 1.5, 4.0])
    y = np.array([2.0, 2.0, 2.0, 0.5, 1.0])
    assert list(brush.in_polygon(x, y, px, py)) == [True, False, True, True, False]
    assert not brush.in_polygon(x, y, px[:2], py[:2]).any()  # too few vertices to enclose anything


def test_brushed_spans_and_combining():
    k = 728857
    span = brush.slots_in_span(slot_to_unix(k + 3) + 10, slot_to_unix(k) + 1)
    assert list(span) == [k, k + 1, k + 2, k + 3]
    old = np.array([1, 2, 3])
    assert list(brush.combine(old, [3, 4], "add")) == [1, 2, 3, 4]
    assert list(brush.combine(old, [2], "remove")) == [1, 3]
    assert list(brush.combine(old, [9, 8, 9], "replace")) == [8, 9]
    assert brush.runs(np.array([1, 2, 3, 7, 9, 10])) == [(1, 4), (7, 8), (9, 11)]
    assert brush.describe(np.array([1, 2, 3, 7])) == f"4 slots ({4 * SLOT_SECONDS / 3600:.3g} h) in 2 stretches"


def test_bookmarks_persist_edit_and_export(tmp_path):
    path = tmp_path / "viewer" / "bookmarks.json"
    store = BookmarkStore(path)
    assert not path.exists()  # nothing is written until the first save
    store.add(Bookmark("oneyear", 728857, 8, note="p spikes τ"), Bookmark("oneyear", 728858, 9, list="figures"),
              Bookmark("other", 5, 1))
    again = BookmarkStore(path)
    assert [(b.slot, b.boom, b.note, b.list) for b in again.items] == [
        (728857, 8, "p spikes τ", DEFAULT_LIST), (728858, 9, "", "figures"), (5, 1, "", DEFAULT_LIST)]
    assert again.lists("oneyear") == [DEFAULT_LIST, "figures"] and len(again.for_run("oneyear")) == 2
    first = again.items[0].id
    again.update(first, note="checked")
    again.remove([again.items[2].id])
    final = BookmarkStore(path)
    assert [b.note for b in final.items] == ["checked", ""]
    final.export(tmp_path / "out.csv", final.items, describe=lambda k: f"t{k}")
    csv = pd.read_csv(tmp_path / "out.csv")
    assert list(csv["time"]) == ["t728857", "t728858"] and list(csv["boom"]) == [8, 9]
    final.export(tmp_path / "out.json", final.items)
    assert len(json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["bookmarks"]) == 2


def test_query_names_and_their_checks():
    known = {n: find.Name(n, "") for n in ("ws_mean", "boom", "stability")}
    expr = "ws_mean > 3 and abs(ws_mean) < 9 and stability == 'stable' and ws_mean.notna() and boom in [1, 2]"
    assert find.referenced(expr, known) == ["ws_mean", "stability", "boom"]
    assert find.unknown(expr, known) == []
    assert find.unknown("ws_meen > 3 and 'not_a_name' == stability", known) == ["ws_meen"]


def test_a_query_that_pins_booms_reads_only_theirs():
    booms = list(range(1, 11))
    assert find.pinned_booms("ws_mean > 3 and boom == 9", booms) == [9]
    assert find.pinned_booms("boom in [2, 4] and ws_mean > 3 and boom != 4", booms) == [2, 4]
    assert find.pinned_booms("boom == 9 or ws_mean > 3", booms) == booms  # an `or` can't narrow them
    assert find.pinned_booms("(boom == 9) and (ws_mean > 3)", booms) == [9]
    assert find.pinned_booms("ws_mean > 3", booms) == booms


@pytest.mark.slow
def test_find_matches_the_stored_values(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    catalog = build_catalog(run)
    known = find.names(catalog)
    assert {"ws_mean", "ustar", "tau_status", "heat_tau_s", "flag_spike_w", "coverage_momentum_usable"} <= set(known)
    assert "rib_1_2" in known and known["rib_1_2"].pair == (1, 2)

    result, wanted = find.find(run, index, catalog, "ws_mean > 1 and boom == 2", "mrd", run.booms, None, None, "UTC")
    assert wanted == ["ws_mean", "boom"]
    ws = timeline.load(run, index, catalog["boom_final|ws|mean"], "none", (2,)).curves[2]
    expected = ws.slots[np.isfinite(ws.y) & (ws.y > 1)]
    assert list(result["slot"]) == list(expected) and (result["boom"] == 2).all()
    assert np.allclose(result["ws_mean"], ws.y[np.isin(ws.slots, expected)])

    sample, _ = find.find(run, index, catalog, "ws_mean > 0", "mrd", run.booms, None, None, "UTC", sample=3, seed=1)
    assert len(sample) == 3 and list(sample["slot"]) == sorted(sample["slot"])

    with pytest.raises(find.FindError, match="did you mean ws_mean"):
        find.find(run, index, catalog, "ws_meen > 1", "mrd", run.booms, None, None, "UTC")
    with pytest.raises(find.FindError, match="must be a condition"):
        find.find(run, index, catalog, "ws_mean + 1", "mrd", run.booms, None, None, "UTC")


@pytest.mark.slow
def test_find_builtins_time_and_filtering(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    catalog = build_catalog(run)
    frame = find.build_frame(run, index, catalog, ["time", "hour", "height", "filtered", "filtered_momentum"], "mrd",
                             FULL_BOOMS, None, None, "Etc/GMT+6")
    assert len(frame) == (run.period[1] - run.period[0]) * len(FULL_BOOMS)
    first = frame.iloc[0]
    local = pd.Timestamp(slot_to_unix(first["slot"]), unit="s", tz="UTC").tz_convert("Etc/GMT+6")
    assert first["time"] == local.tz_localize(None) and first["hour"] == local.hour
    log = index.read("filter_log", filter=pa_dataset.field("variant").isin(["mrd", "none"]))
    logged = set(zip(log["slot"], log["boom"]))
    assert set(zip(frame.loc[frame["filtered"], "slot"], frame.loc[frame["filtered"], "boom"])) == logged
    assert (SPIKY_SLOT, SPIKY_BOOM) in logged
    momentum = log[log["group"] == "momentum"]
    assert frame["filtered_momentum"].sum() == len(set(zip(momentum["slot"], momentum["boom"])))
