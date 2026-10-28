import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QLabel, QMessageBox  # noqa: E402

from ttu_tower.constants import BOOMS, HEIGHTS  # noqa: E402
from ttu_tower.jets.analysis import JetSettings  # noqa: E402
from ttu_tower.jets.edits import EditStore  # noqa: E402
from ttu_tower.jets.identify import ALGORITHMS, NoseParams, Profiles, describe  # noqa: E402
from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.timeaxis import slot_to_unix  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.inspector.inspector import TABS  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402

from viewer_fixtures import FULL_HALF_HOURS  # noqa: E402

pytestmark = pytest.mark.slow

FIRST = 3 * FULL_HALF_HOURS[0]
N = 3 * len(FULL_HALF_HOURS)
JET = [8.0, 14.0, 9.0, 8.5]  # booms 1-4: a nose at boom 2, falloff 5.5 m/s
RISING = [8.0, 9.0, 10.0, 11.0]
EVENT_A = [FIRST + i for i in range(1, 7)]  # FIRST+4 is a rejected slot, bridged
EVENT_B = [FIRST + i for i in range(12, 20)]
SHORT = [FIRST + 22, FIRST + 23]  # too short to be an event
SETTINGS = JetSettings(params=NoseParams(profile_booms=(1, 2, 3, 4), required_booms=(1, 2, 3)))


def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


def _profiles():
    """Jets the synthetic run doesn't have, on its slots, nosing at a boom it does have."""
    slots = np.arange(FIRST, FIRST + N)
    ws = np.full((N, len(BOOMS)), np.nan)
    for i, k in enumerate(slots):
        ws[i, :4] = JET if (k in EVENT_A or k in EVENT_B or k in SHORT) and k != FIRST + 4 else RISING
    return Profiles(slots, tuple(BOOMS), ws)


@pytest.fixture
def window(qtbot, full_run):
    configure_pyqtgraph()
    w = MainWindow(RunHandle.from_dir(full_run.run_dir))
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    qtbot.waitUntil(lambda: _idle(w) and w.jets.profiles is not None, timeout=60_000)
    w.jets.panel.set_settings(SETTINGS)
    w.jets.profiles = _profiles()
    w.jets.store.clear()  # edits are kept per run, and every test here shares the run
    w.jets.recompute()
    yield w
    for inspector in list(w.inspectors):
        inspector.close()
    w.runner.wait(30_000)


def test_the_form_is_built_from_the_algorithm_and_shows_what_each_rule_removed(window):
    panel = window.jets.panel
    assert [panel.algorithm.itemData(i) for i in range(panel.algorithm.count())] == list(ALGORITHMS)
    assert set(panel.form.widgets) == {f["name"] for f in describe(NoseParams)}
    counts = window.jets.result.counts()
    assert counts["events"] == 2 and counts["jet_slots"] == len(EVENT_A) - 1 + len(EVENT_B) + len(SHORT)
    assert counts["status"]["no_data"] == 0 and counts["status"]["no_peak"] == N - counts["jet_slots"]
    assert "→ <b>2</b> events" in panel.result.text()
    assert window.jets.view.events.table.rowCount() == 2


def test_a_setting_change_reruns_and_bad_settings_are_explained(window, qtbot):
    panel = window.jets.panel
    falloff_box, falloff = panel.form.widgets["min_falloff_abs"]
    falloff.setValue(6.0)
    qtbot.waitUntil(lambda: window.jets.result.counts()["events"] == 0, timeout=5000)
    falloff.setValue(2.0)
    qtbot.waitUntil(lambda: window.jets.result.counts()["events"] == 2, timeout=5000)
    falloff_box.setChecked(False)  # no falloff threshold at all
    assert not panel.errors.isHidden() and "falloff" in panel.errors.text()
    assert window.jets.result.counts()["events"] == 2  # the last good result stays
    falloff_box.setChecked(True)
    assert panel.errors.isHidden()


def test_an_event_picked_in_the_table_is_shown_on_the_timelines_and_the_band(window, qtbot):
    table = window.jets.view.events
    table.select_event(1)
    x0, x1 = window.x_range()
    assert x0 < slot_to_unix(EVENT_B[0]) and x1 > slot_to_unix(EVENT_B[-1] + 1)
    band = window.jets.band
    assert band.highlight.isVisible()
    assert band.highlight.getRegion() == pytest.approx((slot_to_unix(EVENT_B[0]), slot_to_unix(EVENT_B[-1] + 1)))
    middle_slot = (EVENT_B[0] + EVENT_B[-1]) // 2
    pos = band.vb.mapViewToScene(QPointF(slot_to_unix(middle_slot) + 300, 0.5))
    qtbot.wait(50)
    assert band._event_at(pos) == 1
    band.eventClicked.emit(0)
    assert table.selected_event() == 0


def test_select_brushes_every_slot_of_the_kept_events(window):
    window.jets.panel.select_button.click()
    assert sorted(window.selection.tolist()) == EVENT_A + EVENT_B


def test_step_through_opens_the_profile_tab_on_each_events_fastest_slot(window, qtbot):
    window.jets.step_through(1)
    qtbot.waitUntil(lambda: bool(window.inspectors) and window.inspectors[-1].bundle is not None and _idle(window),
                    timeout=60_000)
    inspector = window.inspectors[-1]
    assert (inspector.slot, inspector.boom) == (EVENT_B[0], 2)
    assert inspector.tabs.currentIndex() == TABS.index("profile")
    assert inspector.sequence == [(EVENT_A[0], 2), (EVENT_B[0], 2)]


def test_the_settings_are_remembered_and_follow_the_time_zone(window, qtbot, full_run):
    speed = window.jets.panel.form.widgets["min_nose_speed"]
    speed.setValue(12.5)
    qtbot.waitUntil(lambda: window.jets.result.settings.algorithm_params().min_nose_speed == 12.5, timeout=5000)
    utc = window.tz_box.findData("UTC")
    window.tz_box.setCurrentIndex(utc)
    qtbot.waitUntil(lambda: str(window.jets.result.events.table["start_time"].dt.tz) == "UTC", timeout=5000)
    again = MainWindow(RunHandle.from_dir(full_run.run_dir))
    qtbot.addWidget(again)
    try:
        assert again.jets.panel.settings().algorithm_params().min_nose_speed == 12.5
        assert again.jets.panel.settings().algorithm_params().profile_booms == (1, 2, 3, 4)
    finally:
        qtbot.waitUntil(lambda: _idle(again), timeout=60_000)
        again.close()


class _DoubleClick:
    def __init__(self, scene_pos):
        self._pos = scene_pos

    def double(self):
        return True

    def scenePos(self):
        return self._pos


def test_the_event_tab_maps_the_event_and_steps_between_events(window, qtbot):
    view = window.jets.view
    view.events.select_event(0)
    view.tabs.setCurrentWidget(view.single)
    single = view.single
    assert single.event_id == 0
    assert single.image.image.shape == (len(BOOMS), N)  # the 3 h either side reach both ends of the run
    assert len(single.noses.data) == len(EVENT_A) - 1 and len(single.bridged.data) == 1
    assert single.table.rowCount() == len(EVENT_A) and "bridged" in single.table.item(3, 1).text()
    single.next_button.click()
    assert view.event_id == 1 and view.events.selected_event() == 1
    assert not single.next_button.isEnabled() and single.prev_button.isEnabled()
    assert single.table.rowCount() == len(EVENT_B)


def test_a_double_click_on_the_map_opens_that_slot_and_boom(window, qtbot):
    view = window.jets.view
    view.events.select_event(0)
    view.tabs.setCurrentWidget(view.single)
    vb = view.single.map.getPlotItem().getViewBox()
    slot = EVENT_A[2]
    view.single._on_map_click(_DoubleClick(vb.mapViewToScene(QPointF(slot_to_unix(slot) + 300, 1.5))))
    qtbot.waitUntil(lambda: bool(window.inspectors) and window.inspectors[-1].bundle is not None and _idle(window),
                    timeout=60_000)
    inspector = window.inspectors[-1]
    assert (inspector.slot, inspector.boom) == (slot, 2)
    assert inspector.tabs.currentIndex() == TABS.index("profile")


def test_the_two_profile_panels_load_what_they_are_set_to(window, qtbot):
    view = window.jets.view
    view.events.select_event(1)
    view.tabs.setCurrentWidget(view.single)
    single = view.single
    assert [c.currentData() for c in single.choices] == ["ws", "vpt"]
    qtbot.waitUntil(lambda: "vpt" in single.values, timeout=60_000)
    assert single.values["vpt"][[1, 2]].notna().any().all()
    assert single.panels[1].getPlotItem().listDataItems()
    single.choices[1].setCurrentIndex(single.choices[1].findData("tke"))
    qtbot.waitUntil(lambda: "tke" in single.values, timeout=60_000)
    assert single.panels[1].getPlotItem().listDataItems()
    assert single.panels[1].getAxis("bottom").labelText == "TKE"


def test_the_aggregate_follows_the_interval(window, qtbot):
    window.right_tabs.setCurrentWidget(window.jets.view)
    window.full_period.setChecked(True)
    view = window.jets.view
    view.tabs.setCurrentWidget(view.aggregate)
    agg = view.aggregate
    assert agg.summary.text().startswith(
        f"2 events ({len(EVENT_A) + len(EVENT_B)} slots, {len(EVENT_A) - 1 + len(EVENT_B)} jet slots)")
    bars = next(i for i in agg.height_plot.getPlotItem().items if isinstance(i, pg.BarGraphItem))
    assert bars.opts["height"][BOOMS.index(2)] == 1.0  # both events, as a fraction
    assert agg.composite_plot.getPlotItem().listDataItems()
    window.full_period.setChecked(False)
    window.set_x_range(slot_to_unix(FIRST), slot_to_unix(FIRST + 10))
    qtbot.waitUntil(lambda: agg.summary.text().startswith("1 event "), timeout=5000)



def test_the_aggregate_histograms_take_a_bin_count(window, qtbot):
    window.right_tabs.setCurrentWidget(window.jets.view)
    window.full_period.setChecked(True)
    view = window.jets.view
    view.tabs.setCurrentWidget(view.aggregate)
    agg = view.aggregate

    def edges():
        item = next(i for i in agg.speed_plot.getPlotItem().items if isinstance(i, pg.PlotDataItem))
        return item.xData

    assert agg.bins.text() == "auto" and np.allclose(np.diff(edges()), 1.0)  # 1 m/s bins
    agg.bins.setValue(4)
    x = edges()
    assert x.size == 5 and np.allclose(np.diff(x), np.diff(x)[0])


def test_no_dock_can_widen_the_window_past_the_screen(window, qtbot):
    window.right_tabs.setCurrentWidget(window.jets.view)
    view = window.jets.view
    view.tabs.setCurrentWidget(view.aggregate)
    before = [d.minimumSizeHint().width() for d in (window.controls_dock, window.summary_dock, window)]
    wide = QLabel("x" * 400)  # a control row far wider than any screen
    view.aggregate.layout().addWidget(wide)
    assert wide.minimumSizeHint().width() > 2000
    assert [d.minimumSizeHint().width() for d in (window.controls_dock, window.summary_dock, window)] == before

def test_diagnostics_count_every_status_and_sweep_a_setting(window, qtbot):
    window.right_tabs.setCurrentWidget(window.jets.view)
    window.full_period.setChecked(True)
    view = window.jets.view
    view.tabs.setCurrentWidget(view.diagnostics)
    diag = view.diagnostics
    assert diag.status.rowCount() == 5

    def swept():
        return diag.sweep_note.text() and "sweeping" not in diag.sweep_note.text()
    qtbot.waitUntil(swept, timeout=30_000)
    x, y = diag.slots_plot.getPlotItem().listDataItems()[0].getData()
    assert y[np.argmin(np.abs(x - 10.0))] == window.jets.result.counts()["jet_slots"]
    diag.sweep_choice.setCurrentIndex(diag.sweep_choice.findData("events:min_event_slots"))
    qtbot.waitUntil(lambda: swept() and diag.events_plot.getPlotItem().getAxis("bottom").labelText.startswith(
        "min event length"), timeout=30_000)
    x, y = diag.events_plot.getPlotItem().listDataItems()[0].getData()
    assert y[list(x).index(6)] == 2 and y[list(x).index(7)] == 1


def test_bridging_missing_data_is_a_check_box_on_the_events_form(window, qtbot):
    box = window.jets.panel.event_form.widgets["bridge_missing"]
    assert isinstance(box, QCheckBox) and not box.isChecked()
    box.setChecked(True)
    qtbot.waitUntil(lambda: window.jets.result.settings.events.bridge_missing, timeout=5000)


def _edit_boundaries(window, event_id, first, last):
    view = window.jets.view
    view.events.select_event(event_id)
    view.tabs.setCurrentWidget(view.single)
    single = view.single
    single.edit_button.setChecked(True)
    single.span.setRegion((slot_to_unix(first) + 40, slot_to_unix(last + 1) - 30))  # snapped to the slot edges
    single._snap()
    single.save_button.click()


def test_boundaries_dragged_on_the_map_are_saved_as_a_manual_edit(window, full_run):
    _edit_boundaries(window, 0, FIRST, FIRST + 8)
    table = window.jets.result.events.table
    first = table.iloc[0]
    assert (first.start_slot, first.end_slot, first.source) == (FIRST, FIRST + 8, "edited")
    assert first.n_manual == 3 and window.jets.view.event_id == 0
    assert window.jets.view.events.table.item(0, 1).text().startswith("edited")
    stored = EditStore(full_run.tag).edits
    assert len(stored) == 1 and (stored[0].kind, stored[0].start, stored[0].end) == ("span", FIRST, FIRST + 8)
    band = window.jets.band
    assert window.jets.result.events.slots.loc[FIRST, "manual"] and 0 in set(band.event_of)
    _edit_boundaries(window, 0, FIRST + 1, FIRST + 10)  # a second edit replaces the first
    assert len(window.jets.store.edits) == 1 and window.jets.result.events.table.iloc[0].end_slot == FIRST + 10
    window.jets.view.single.revert_button.click()
    assert not window.jets.store.edits and window.jets.result.events.table.iloc[0].source == "auto"


def test_a_rejected_event_drops_out_until_the_rejection_is_undone(window):
    view = window.jets.view
    view.events.select_event(1)
    view.tabs.setCurrentWidget(view.single)
    view.single.reject_button.click()
    result = window.jets.result
    assert result.events.table.loc[1, "rejected"] and result.counts()["passed"] == 1
    assert window.jets.view.single.reject_button.text().startswith("Rejected")
    view.events.kept_only.setChecked(False)
    assert "rejected by hand" in view.events.table.item(1, view.events.COLUMNS.index("kept")).text()
    view.single.reject_button.click()
    assert window.jets.result.counts()["passed"] == 2 and not window.jets.store.edits


def test_an_event_can_be_made_from_the_selection_and_edits_listed_and_removed(window, qtbot, monkeypatch):
    window.set_selection(np.arange(FIRST + 20, FIRST + 22), "replace")  # rising profiles, no jet
    window.jets.view.events.add_button.click()
    table = window.jets.result.events.table
    assert len(table) == 3 and table.iloc[2].source == "manual" and table.iloc[2].n_slots == 2
    assert np.isnan(table.iloc[2].u_nose_max) and table.iloc[2].n_detected == 0
    window.jets.view.every_slot.setChecked(True)  # the hand-made slots' candidates count as noses
    assert window.jets.view.events.nose_from == "candidates"
    window.jets.view.every_slot.setChecked(False)
    window.jets.view.events.select_event(1)
    window.jets.view.tabs.setCurrentWidget(window.jets.view.single)
    window.jets.view.single.reject_button.click()
    window.jets.view.events.edits_button.click()
    dialog = window.jets.dialog
    assert dialog.isVisible() and dialog.table.rowCount() == 2
    dialog.table.selectRow(0)
    dialog.remove_button.click()
    assert len(window.jets.store.edits) == 1 and dialog.table.rowCount() == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.clear_button.click()
    assert not window.jets.store.edits and dialog.table.rowCount() == 0
    assert len(window.jets.result.events.table) == 2
    dialog.close()


def _drawn_x(panel):
    xs = [item.getData()[0] for item in panel.getPlotItem().listDataItems()]
    xs = np.concatenate([x for x in xs if x is not None and len(x)])
    return np.nanmin(xs), np.nanmax(xs)


def test_profile_panels_summarize_the_event_and_refit_to_each_event(window, qtbot):
    view = window.jets.view
    view.events.select_event(0)
    view.tabs.setCurrentWidget(view.single)
    single = view.single
    single.choices[1].setCurrentIndex(single.choices[1].findData("ti"))
    qtbot.waitUntil(lambda: "ti" in single.values, timeout=60_000)
    single.modes[0].setCurrentIndex(single.modes[0].findData("median"))
    labels = [getattr(i, "export_label", "") for i in single.panels[0].getPlotItem().listDataItems()]
    assert "median" in labels and not any(":" in label for label in labels)  # no per-slot curves
    single.modes[0].setCurrentIndex(single.modes[0].findData("mean"))
    assert "mean" in [getattr(i, "export_label", "") for i in single.panels[0].getPlotItem().listDataItems()]
    for event_id in (1, 0):
        view.events.select_event(event_id)
        for panel in single.panels:
            lo, hi = _drawn_x(panel)
            x0, x1 = panel.getPlotItem().getViewBox().viewRange()[0]
            assert x0 <= lo and hi <= x1


def test_the_aggregate_can_show_one_event_slot_by_slot(window, qtbot):
    window.right_tabs.setCurrentWidget(window.jets.view)
    window.full_period.setChecked(True)
    view = window.jets.view
    view.events.select_event(1)
    view.tabs.setCurrentWidget(view.aggregate)
    agg = view.aggregate
    agg.scope.setCurrentIndex(agg.scope.findData("event"))
    assert agg.summary.text().startswith(f"jet 1, ") and f"{len(EVENT_B)} slots" in agg.summary.text()
    assert not agg.which.isEnabled()
    bars = next(i for i in agg.height_plot.getPlotItem().items if isinstance(i, pg.BarGraphItem))
    assert bars.opts["height"][BOOMS.index(2)] == 1.0  # every slot's nose at boom 2
    speed = agg.speed_plot.getPlotItem().listDataItems()[0]
    x, density = speed.getData()
    assert (density * np.diff(x)).sum() == pytest.approx(1.0)
    view.events.select_event(0)
    view.tabs.setCurrentWidget(view.aggregate)
    assert agg.summary.text().startswith("jet 0, ")


def test_one_events_slots_can_be_brushed_from_the_list_or_the_event_tab(window):
    view = window.jets.view
    assert not view.events.select_button.isEnabled()
    view.events.select_event(0)
    view.events.select_button.click()
    assert sorted(window.selection.tolist()) == EVENT_A  # its bridged slot too
    view.events.select_event(1)
    view.tabs.setCurrentWidget(view.single)
    view.single.select_button.click()
    assert sorted(window.selection.tolist()) == EVENT_B


def test_events_are_described_by_observed_peak_and_estimated_nose(window):
    view = window.jets.view
    columns = view.events.COLUMNS
    assert {"obs. peak (m)", "est. nose (m)", "obs. peak speed (m/s)", "est. nose speed (m/s)"} <= set(columns)
    e = window.jets.result.events.table.loc[0]
    assert HEIGHTS[1] < e.z_est_median < HEIGHTS[3] and e.u_est_max > e.u_nose_max  # between the peak's neighbours
    view.events.select_event(0)
    view.tabs.setCurrentWidget(view.single)
    assert "est. nose" in view.single.label.text()
    headers = [view.single.table.horizontalHeaderItem(j).text() for j in range(view.single.table.columnCount())]
    assert {"obs. peak (m)", "est. nose (m)"} <= set(headers)
    view.tabs.setCurrentWidget(view.aggregate)
    agg = view.aggregate
    assert agg.height_plot.getPlotItem().titleLabel.text.startswith("observed peak height")
    agg.kind.setCurrentIndex(agg.kind.findData("estimated"))
    assert agg.height_plot.getPlotItem().titleLabel.text.startswith("estimated nose height")
    items = agg.height_plot.getPlotItem().listDataItems()
    assert items and not any(isinstance(i, pg.BarGraphItem) for i in agg.height_plot.getPlotItem().items)
    x, density = items[0].getData()
    assert x[0] <= e.z_est_median <= x[-1]


def test_zeta_is_a_profile_panel_option(window, qtbot):
    view = window.jets.view
    view.events.select_event(0)
    view.tabs.setCurrentWidget(view.single)
    single = view.single
    single.choices[1].setCurrentIndex(single.choices[1].findData("zeta"))
    qtbot.waitUntil(lambda: "zeta" in single.values, timeout=60_000)
    assert single.panels[1].getAxis("bottom").labelText == "ζ = z/L"


def test_the_aggregate_draws_the_jet_slots_hodograph(window, qtbot):
    p = window.jets.profiles
    wd = np.where(np.isfinite(p.ws), 180.0, np.nan)  # from the south: v toward north is the speed
    window.jets.profiles = Profiles(p.slots, p.booms, p.ws, extra={"wd": wd})
    window.jets.recompute()
    qtbot.waitUntil(lambda: _idle(window) and window.jets.result is not None, timeout=60_000)
    view = window.jets.view
    window.right_tabs.setCurrentWidget(view)
    window.full_period.setChecked(True)
    view.tabs.setCurrentWidget(view.aggregate)
    agg = view.aggregate
    agg.which.setCurrentIndex(agg.which.findData(False))
    title = agg.hodo_plot.getPlotItem().titleLabel.text
    detected = window.jets.result.detected_in(window.jets.result.events.table.index)
    assert f"mean of {len(detected)} jet slots" in title and "each event" in title
    curves = [c for c in agg.hodo_plot.getPlotItem().listDataItems() if isinstance(c, pg.PlotDataItem)]
    mean = curves[-1]  # drawn last, over each event's
    u, v = mean.getData()
    assert len(u) == 4 and u == pytest.approx(0.0, abs=1e-9) and v == pytest.approx(JET)  # booms 1-4, the jet nose
    assert len(curves) == 1 + window.jets.result.events.table.shape[0]
    agg.scope.setCurrentIndex(agg.scope.findData("event"))
    agg.event_id = int(window.jets.result.events.table.index[0])
    agg.redraw()
    assert "each slot" in agg.hodo_plot.getPlotItem().titleLabel.text
