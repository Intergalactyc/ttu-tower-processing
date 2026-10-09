import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import BOOMS
from ttu_tower.jets.analysis import JetSettings, analyse
from ttu_tower.jets.edits import EditStore, new_edit
from ttu_tower.jets.filters import JetFilters, vpt_column
from ttu_tower.jets.identify import NoseParams, Profiles

JET = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 15.0, 12.0, 12.5]  # nose at boom 8, falloff 3 m/s
WEAK = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 15.0, 14.0, 14.5]  # a fast nose, falloff only 1 m/s
RISING = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
SLOT0 = 5000


def _profiles(pattern: str):
    """J jet, w a weak candidate, r rising, n no data."""
    rows = []
    for c in pattern:
        row = list({"J": JET, "w": WEAK, "r": RISING, "n": RISING}[c])
        if c == "n":
            row[9] = np.nan
        rows.append(row)
    return Profiles(np.arange(SLOT0, SLOT0 + len(pattern)), tuple(BOOMS), np.array(rows))


PATTERN = "rr" + "JJJJJJ" + "ww" + "rr" + "JJJJJJ" + "rrrr"  # events at 2-7 and 12-17


def _analyse(edits, pattern=PATTERN, **kwargs):
    return analyse(_profiles(pattern), JetSettings(**kwargs), edits=edits)


def test_without_edits_the_events_are_as_detected():
    r = _analyse([])
    t = r.events.table
    assert list(t["start_slot"] - SLOT0) == [2, 12] and list(t["source"]) == ["auto", "auto"]
    assert not t["rejected"].any() and r.edit_report.empty


def test_new_boundaries_extend_an_event_and_mark_the_added_slots():
    peak = SLOT0 + 2
    r = _analyse([new_edit("span", SLOT0, SLOT0 + 9, anchor=peak)])
    first = r.events.table.iloc[0]
    assert (first.start_slot, first.end_slot, first.source) == (SLOT0, SLOT0 + 9, "edited")
    assert first.n_manual == 4 and first.n_detected == 6 and first.duration_min == 100
    assert r.events.slots.loc[SLOT0, "manual"] and not r.events.slots.loc[SLOT0 + 3, "manual"]
    assert list(r.edit_report["status"]) == ["applied"] and r.edit_report["event_id"].iloc[0] == 0
    shrunk = _analyse([new_edit("span", SLOT0 + 3, SLOT0 + 5, anchor=peak)]).events.table.iloc[0]
    assert (shrunk.start_slot, shrunk.n_slots, shrunk.n_detected) == (SLOT0 + 3, 3, 3)


def test_a_span_reaching_another_event_absorbs_it():
    r = _analyse([new_edit("span", SLOT0 + 2, SLOT0 + 13, anchor=SLOT0 + 4)])
    t = r.events.table
    assert len(t) == 1 and t.iloc[0].end_slot == SLOT0 + 13 and t.iloc[0].n_detected == 8


def test_an_edit_whose_event_is_gone_stays_as_a_manual_event():
    edits = [new_edit("span", SLOT0 + 1, SLOT0 + 8, anchor=SLOT0 + 3)]
    r = _analyse(edits, params=NoseParams(min_nose_speed=16.0))  # now nothing is a jet
    t = r.events.table
    assert len(t) == 1 and t.iloc[0].source == "manual" and t.iloc[0].n_detected == 0
    assert "manual event" in r.edit_report["status"].iloc[0]
    assert pd.isna(t.iloc[0].u_nose_max)
    noses = r.table("candidates").iloc[0]
    assert noses.u_nose_max == 15.0 and noses.n_detected == 7 and noses.peak_boom == 8  # 6 jets, 1 weak


def test_an_event_can_be_added_and_overlapping_hand_spans_join():
    r = _analyse([new_edit("add", SLOT0 + 20, SLOT0 + 22)], pattern=PATTERN + "rrr")
    t = r.events.table
    assert list(t["source"]) == ["auto", "auto", "manual"] and t.iloc[2].n_manual == 3
    joined = _analyse([new_edit("add", SLOT0 + 20, SLOT0 + 22), new_edit("add", SLOT0 + 22, SLOT0 + 24)],
                      pattern=PATTERN + "rrr")
    assert joined.events.table.iloc[2].end_slot == SLOT0 + 24 and len(joined.events.table.iloc[2].edits) == 2
    over = _analyse([new_edit("add", SLOT0 + 6, SLOT0 + 13)])
    assert len(over.events.table) == 1 and over.events.table.iloc[0].source == "manual"


def test_a_rejected_event_is_listed_but_not_kept():
    r = _analyse([new_edit("reject", SLOT0 + 12, SLOT0 + 17, anchor=SLOT0 + 14)])
    assert list(r.events.table["rejected"]) == [False, True]
    assert list(r.passed) == [True, False] and r.counts()["rejected"] == 1
    assert len(r.events_in()) == 1
    nothing = _analyse([new_edit("reject", SLOT0, SLOT0, anchor=SLOT0)])
    assert "inactive" in nothing.edit_report["status"].iloc[0] and r.counts()["passed"] == 1


def test_edits_outside_the_data_do_nothing():
    r = _analyse([new_edit("add", SLOT0 + 500, SLOT0 + 510)])
    assert len(r.events.table) == 2 and r.edit_report["status"].iloc[0] == "outside the data"


def test_filters_see_every_slot_of_an_edited_event():
    profiles = _profiles(PATTERN)
    context = np.full(len(PATTERN), 1.0)
    context[0:2] = 11.0
    ctx = pd.DataFrame({vpt_column(2): 290.0, vpt_column(10): 290.0 + context, "night": 1.0}, index=profiles.slots)
    settings = JetSettings(filters=JetFilters(min_inversion=2.0))
    plain = analyse(profiles, settings, ctx)
    assert list(plain.passed) == [False, False]
    edited = analyse(profiles, settings, ctx, edits=[new_edit("span", SLOT0, SLOT0 + 7, anchor=SLOT0 + 3)])
    assert edited.quantities["dvpt_mean"].iloc[0] == pytest.approx((2 * 11 + 6 * 1) / 8)
    assert list(edited.passed) == [True, False]


def test_the_store_keeps_a_runs_edits(tmp_path):
    path = tmp_path / "jet_edits" / "run.json"
    store = EditStore("run", path)
    a, b = new_edit("add", 10, 20, note="by eye"), new_edit("reject", 30, 40, anchor=35)
    store.change(add=[a, b])
    again = EditStore("run", path)
    assert again.edits == [a, b]
    again.change(remove=[a.id])
    assert EditStore("run", path).edits == [b]
    again.clear()
    assert EditStore("run", path).edits == []
    with pytest.raises(ValueError):
        new_edit("span", 10, 5)
