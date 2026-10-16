import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from ttu_tower.timegrid import slot_to_time  # noqa: E402
from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.timeaxis import display_offset_s, slot_to_unix  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.inspector.numbers_tab import ALL  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402
from ttu_tower.viewer.ui.series_view import MODES as SERIES_MODES, SeriesPanel, rug_differences, rug_rows  # noqa: E402

from viewer_fixtures import FAULT_HALF_HOUR  # noqa: E402


def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


@pytest.fixture
def window(qtbot, primary_run):
    configure_pyqtgraph()
    w = MainWindow(RunHandle.from_dir(primary_run.run_dir))
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    qtbot.waitUntil(lambda: _idle(w) and w.datas[0] is not None, timeout=60_000)
    yield w
    w.runner.wait(30_000)


class _FakeClick:
    def __init__(self, scene_pos, button=Qt.MouseButton.LeftButton, double=False):
        self._pos, self._button, self._double = scene_pos, button, double

    def button(self):
        return self._button

    def double(self):
        return self._double

    def scenePos(self):
        return self._pos

    def screenPos(self):
        return QPointF(10, 10)


def _a_point(plot):
    member = plot.visible_members()[0]
    curve = plot.curves[member]
    i = int(np.flatnonzero(np.isfinite(curve.y))[len(curve.y) // 2])
    return member, int(curve.slots[i]), float(curve.x[i]), float(curve.y[i])


def _zoom_to(window, x):
    window.set_x_range(x - 3 * 3600, x + 3 * 3600)
    window.plots[0].decimator.redraw()


def test_window_loads_a_curve_per_selected_boom(window):
    plot, spec = window.plots[0], window.specs[0]
    assert spec.quantity is not None
    assert set(plot.items) == set(spec.members) == {1}
    assert window.catalog is not None and len(window.catalog) > 50


def test_pick_finds_the_clicked_point(window):
    plot = window.plots[0]
    member, slot, x, y = _a_point(plot)
    _zoom_to(window, x)
    hits = plot.pick(x, y)
    assert hits and hits[0][1] == slot and hits[0][2] == member


def test_a_left_click_on_a_point_emits_it_at_once(window, qtbot):
    plot = window.plots[0]
    member, slot, x, y = _a_point(plot)
    _zoom_to(window, x)
    scene = plot.vb.mapViewToScene(QPointF(x, y))
    emitted = []
    plot.pointClicked.connect(lambda s, m: emitted.append((s, m)))
    assert plot._on_click(_FakeClick(scene))
    assert emitted == [(slot, member)]
    assert not plot._on_click(_FakeClick(scene, button=Qt.MouseButton.RightButton))  # the plot's own menu
    assert not plot._on_click(_FakeClick(scene, double=True))
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)


def test_a_mouse_click_right_on_a_drawn_marker_reaches_the_picker(window, qtbot):
    plot = window.plots[0]
    assert "points" in window.specs[0].style
    member, slot, x, y = _a_point(plot)
    _zoom_to(window, x)
    emitted = []
    plot.pointClicked.disconnect()
    plot.pointClicked.connect(lambda s, m: emitted.append((s, m)))
    pos = plot.widget.mapFromScene(plot.vb.mapViewToScene(QPointF(x, y)))
    qtbot.mouseClick(plot.widget.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    qtbot.waitUntil(lambda: bool(emitted), timeout=5_000)
    assert emitted == [(slot, member)]


def test_log_toggle_and_boom_changes_keep_the_viewed_interval(window, qtbot):
    _, _, x, _ = _a_point(window.plots[0])
    window.set_x_range(x - 7200, x + 7200)
    before = window.x_range()
    window.panels[0].log_y.setChecked(True)
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)
    qtbot.wait(200)
    window.panels[0].style.setCurrentIndex(2)
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)
    qtbot.wait(200)
    assert window.x_range() == pytest.approx(before)


def test_a_click_off_the_data_falls_through(window):
    plot = window.plots[0]
    _, _, x, y = _a_point(plot)
    _zoom_to(window, x)
    (_, _), (y0, y1) = plot.vb.viewRange()
    scene = plot.vb.mapViewToScene(QPointF(x, y1 + 10 * (y1 - y0)))
    assert not plot._on_click(_FakeClick(scene))


def _open_series(window, qtbot, slot):
    key = "flags|spike|ts"  # a flag fraction opens the series with its flag rug
    if window.specs[0].quantity is None or window.specs[0].quantity.key != key:
        window.panels[0].select(key, "none", (1,))
        qtbot.waitUntil(lambda: window.datas[0] is not None and window.datas[0].quantity.key == key, timeout=60_000)
    window.plots[0].pointClicked.emit(slot, 1)
    qtbot.waitUntil(lambda: bool(window.inspectors) and window.inspectors[-1].series.result is not None
                    and window.inspectors[-1].series.slot == slot, timeout=60_000)
    return window.inspectors[-1]


def test_a_click_opens_the_inspector_on_the_unexcised_series(window, qtbot):
    inspector = _open_series(window, qtbot, 3 * FAULT_HALF_HOUR + 1)
    series: SeriesPanel = inspector.series
    assert inspector.tabs.currentWidget() is series
    assert series.current_mode() == "unexcised" and series.result["mode"] == "unexcised"
    assert series.plots, "no series were drawn"
    series.mode.setCurrentIndex([k for _, k in SERIES_MODES].index("qc"))
    qtbot.waitUntil(lambda: series.result["mode"] == "qc", timeout=60_000)
    assert "spike" in series.result["removed"]
    inspector.close()


def test_stepping_and_loading_neighbours_alongside(window, qtbot):
    k = 3 * FAULT_HALF_HOUR + 1
    inspector = _open_series(window, qtbot, k)
    series = inspector.series
    inspector.nav_buttons["next"].click()
    qtbot.waitUntil(lambda: series.result["am"].g0 == (k + 1) * 30_000 - 30_000, timeout=60_000)
    assert (series.lo, series.slot, series.hi) == (k + 1, k + 1, k + 1)
    series.nav_buttons["earlier"].click()
    series.nav_buttons["later"].click()
    qtbot.waitUntil(lambda: series.result["am"].n == series.window_samples()[1], timeout=60_000)
    assert (series.lo, series.slot, series.hi) == (k, k + 1, k + 2)
    assert series.result["am"].n == 3 * 30_000 + 2 * 30_000
    inspector.nav_buttons["prev"].click()  # the neighbours move with the focus
    qtbot.waitUntil(lambda: series.slot == k and series.result["am"].g0 == (k - 1) * 30_000 - 30_000,
                    timeout=60_000)
    assert (series.lo, series.hi) == (k - 1, k + 1)
    series.nav_buttons["single"].click()
    qtbot.waitUntil(lambda: series.result["am"].n == 30_000 + 2 * 30_000, timeout=60_000)
    inspector.close()


def test_series_toggles_streamwise_frame_and_tau_blocks(window, qtbot):
    inspector = _open_series(window, qtbot, 3 * FAULT_HALF_HOUR + 1)
    series = inspector.series
    shown = set(series.plots)
    series.series_boxes["ts"].setChecked(False)
    assert "ts" not in series.plots and set(series.plots) == shown - {"ts"}
    series.frame.setCurrentIndex(1)
    qtbot.waitUntil(lambda: series.result["frame"] == "streamwise", timeout=60_000)
    assert {"u", "v"} <= set(series.plots) and "ue" not in series.plots
    assert series.result["bearings"]
    series.blocks.setCurrentIndex(series.blocks.findData(75.0))
    series.fluct.setChecked(True)
    assert series.plots["w"].getAxis("left").labelText.startswith("w'")
    inspector.close()


def test_numbers_tab_lists_stored_rows_and_verify_is_exact(window, qtbot):
    inspector = _open_series(window, qtbot, 3 * FAULT_HALF_HOUR + 1)
    qtbot.waitUntil(lambda: inspector.bundle is not None, timeout=60_000)
    numbers = inspector.numbers
    numbers.table_box.setCurrentIndex(numbers.table_box.findData("means"))
    assert numbers.shown_rows() > 10
    numbers.search.setText("ts")
    assert 0 < numbers.shown_rows() < len(numbers.current_frame())
    numbers.verify()
    qtbot.waitUntil(lambda: getattr(numbers, "last_report", None) is not None, timeout=120_000)
    report = numbers.last_report
    assert set(report["table"]) >= {"means", "coverage", "mrd", "ladder"}
    assert (report["max abs diff"] == 0).all()
    assert (report["only recomputed"] == 0).all() and (report["only stored"] == 0).all()
    inspector.close()


def _shown(numbers, column):
    return [numbers.proxy.index(r, column).data() for r in range(numbers.shown_rows())]


def _double_click(view, row):
    index = view.model().index(row, 0)
    view.scrollTo(index)
    pos = view.visualRect(index).center()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)  # a real double click starts with a click
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)


def _selected(view, columns):
    rows = view.selectionModel().selectedRows()
    assert len(rows) == 1
    return [view.model().index(rows[0].row(), c).data() for c in columns]


def test_numbers_all_tables_finds_a_value_and_jumps_to_its_row(window, qtbot):
    inspector = _open_series(window, qtbot, 3 * FAULT_HALF_HOUR + 1)
    qtbot.waitUntil(lambda: inspector.bundle is not None, timeout=60_000)
    numbers = inspector.numbers
    inspector.tabs.setCurrentWidget(numbers)
    numbers.table_box.setCurrentIndex(numbers.table_box.findData(ALL))
    assert numbers.shown_rows() == len(numbers.current_frame()) > sum(
        len(df) for df in numbers._tables().values())
    assert set(_shown(numbers, 0)) >= {"means", "coverage", "slot_qc", "mrd", "ladder"}

    numbers.search.setText("stat=skew")  # only the all-tables key has this text
    assert set(_shown(numbers, 0)) == {"slot_qc"}
    _double_click(numbers.view, _shown(numbers, 1).index("variable=w, stat=skew"))
    assert numbers.table_box.currentData() == "slot_qc"
    assert numbers.search.text() == ""
    assert _selected(numbers.view, (2, 3)) == ["w", "skew"]

    numbers.table_box.setCurrentIndex(numbers.table_box.findData(ALL))
    numbers.search.setText("kurt")
    _double_click(numbers.view, _shown(numbers, 0).index("slot_qc"))
    assert numbers.table_box.currentData() == "slot_qc"
    assert numbers.search.text() == "kurt"  # the row still shows, so the filter stays
    assert _selected(numbers.view, (3,)) == ["kurt"]
    inspector.close()


def test_pinned_inspector_is_kept(window, qtbot):
    slot = 3 * FAULT_HALF_HOUR
    _open_series(window, qtbot, slot).pin.setChecked(True)
    window.plots[0].pointClicked.emit(slot + 1, 1)
    qtbot.waitUntil(lambda: len(window.inspectors) == 2 and window.inspectors[1].series.result is not None,
                    timeout=60_000)
    assert window.inspectors[0].slot == slot and window.inspectors[1].slot == slot + 1


def test_distribution_follows_the_visible_range(window):
    dist = window.dists[0]
    whole = sum(v.size for v in dist.values_by_member().values())
    _, _, x, _ = _a_point(window.plots[0])
    window.set_x_range(x - 3600, x + 3600)
    window._refresh_distributions()
    zoomed = sum(v.size for v in dist.values_by_member().values())
    assert 0 < zoomed < whole


def test_time_axis_labels_are_in_the_run_timezone(window):
    axis = window.plots[0].axis
    assert axis.utcOffset == display_offset_s("Etc/GMT+6")
    k = 3 * FAULT_HALF_HOUR + 1
    x = slot_to_unix(k)
    ticks = axis.tickValues(x - 3600, x + 3600, 800)  # also sets the zoom level tickStrings formats by
    label = axis.tickStrings([x], 1.0, ticks[-1][0])[0]
    assert label == f"{slot_to_time(k):%H:%M}"


def test_inspect_from_a_typed_time(window, qtbot):
    k = 3 * FAULT_HALF_HOUR + 1
    window.goto.setText(f"{slot_to_time(k).tz_convert(window.run.timezone):%Y-%m-%d %H:%M}")
    window.goto_boom.setCurrentIndex(window.goto_boom.findData(1))
    window._on_inspect_goto()
    qtbot.waitUntil(lambda: bool(window.inspectors) and window.inspectors[-1].bundle is not None, timeout=60_000)
    assert (window.inspectors[-1].slot, window.inspectors[-1].boom) == (k, 1)
    window.inspectors[-1].close()


def test_long_category_labels_fit_and_axes_stay_aligned(window, qtbot):
    window.panels[1].select("slot_boom|status", "none", (1,))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[1] is not None
                    and window.datas[1].quantity.key == "slot_boom|status", timeout=60_000)
    labels = window.datas[1].categories
    axis = window.plots[1].plot_item.getAxis("left")
    from PySide6.QtGui import QFontMetrics
    widest = max(QFontMetrics(axis.font()).horizontalAdvance(c) for c in labels)
    assert axis.maximumWidth() >= widest + 20  # the width asked for; the geometry follows at the next layout pass
    qtbot.waitUntil(lambda: axis.width() >= widest + 20, timeout=5_000)
    axes = [window.plots[0].plot_item.getAxis("left"), axis, window.stability_band.plot.getPlotItem().getAxis("left"),
            window.overview.getPlotItem().getAxis("left")]
    qtbot.waitUntil(lambda: len({a.width() for a in axes}) == 1, timeout=5_000)


def test_qc_whatif_panel_reprocesses_and_marks_what_changed(window, qtbot):
    from viewer_fixtures import TS_SPIKE_INDEX
    from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR

    inspector = _open_series(window, qtbot, 3 * FAULT_HALF_HOUR + 2)
    series = inspector.series
    series.mode.setCurrentIndex([k for _, k in SERIES_MODES].index("qc"))
    qtbot.waitUntil(lambda: series.result["mode"] == "qc", timeout=60_000)
    panel = series.whatif
    assert not series.rug.isChecked() and "rug" not in " ".join(series.plots)  # the rug starts hidden
    series.whatif_button.setChecked(True)
    assert panel.isVisible()
    panel.apply()  # nothing edited: nothing to apply
    assert series.whatif_rp is None
    panel.editors[("despike", "z_threshold", "ts")].setText("1000000")
    assert "background" in panel.editors[("despike", "z_threshold", "ts")].styleSheet()
    panel.apply()
    qtbot.waitUntil(lambda: series.result.get("whatif") is not None and panel.comparison.rowCount() > 0
                    and _idle(window), timeout=120_000)
    whatif = series.result["whatif"]
    assert series.rug.isChecked()  # turned on to show the what-if's differences
    g = FAULT_HALF_HOUR * SAMPLES_PER_HALF_HOUR + TS_SPIKE_INDEX - whatif.g0
    assert series.result["qc"].removed["spike"]["ts"][g]
    assert not whatif.removed.get("spike", {}).get("ts", np.zeros(whatif.n, bool))[g]
    diffs = rug_differences(rug_rows(series.result["removed"], series.result["qc"].filled),
                            rug_rows(whatif.removed, whatif.filled), whatif.n)
    assert diffs["spike · ts"][1][g]  # flagged only with the run's settings
    assert "z_threshold.ts" in panel.state.text()
    panel.tauRequested.emit()
    qtbot.waitUntil(lambda: _idle(window), timeout=120_000)
    panel.editors[("despike", "window_s")].setText("5000")
    panel.apply()
    assert "window_s" in panel.state.text() and series.whatif_rp is not None  # rejected; the last what-if stays
    panel.reset()
    qtbot.waitUntil(lambda: series.whatif_rp is None and series.result.get("whatif") is None and _idle(window),
                    timeout=60_000)
    inspector.close()


def test_series_start_with_the_wind_and_sonic_theta_and_show_what_was_clicked(window, qtbot):
    k = 3 * FAULT_HALF_HOUR + 1
    inspector = window.inspect(1, k)
    series = inspector.series
    qtbot.waitUntil(lambda: series.result is not None and series.slot == k and _idle(window), timeout=60_000)
    assert set(series.plots) == {"ue", "vn", "w", "vpts"}
    assert not series.series_boxes["t"].isChecked() and series.series_boxes["vpts"].isChecked()
    window.inspect(1, k, window.catalog["flags|spike|t"], "none")  # a value computed from T
    qtbot.waitUntil(lambda: "t" in series.plots and _idle(window), timeout=60_000)
    assert series.series_boxes["t"].isChecked()
    inspector.close()


def test_toolbar_buttons_bring_back_a_closed_panel_even_detached(window, qtbot):
    for title, dock in (("Controls", window.controls_dock), ("Summary", window.summary_dock)):
        action = window.dock_actions[title]
        assert action in window.toolbar.actions() and action.isChecked() and dock.isVisible()
        dock.setFloating(True)
        dock.close()
        qtbot.waitUntil(lambda: not dock.isVisible(), timeout=2000)
        assert not action.isChecked()
        action.trigger()
        assert dock.isVisible() and dock.isFloating() and action.isChecked()
        action.trigger()  # pressed while open: it closes
        assert not dock.isVisible()
        action.trigger()
        dock.setFloating(False)



def test_panel_tabs_wrap_onto_more_rows_when_narrow(window, qtbot):
    tabs = window.right_tabs
    assert not tabs.tabBar().isVisible() and [b.text() for b in tabs.buttons()][-1] == "Jets"

    def rows():
        return len({b.geometry().y() for b in tabs.buttons()})

    window.resize(3600, 900)  # room for one row even in the wide test font
    window.resizeDocks([window.summary_dock], [2400], Qt.Orientation.Horizontal)
    qtbot.waitUntil(lambda: rows() == 1, timeout=2000)
    window.resizeDocks([window.summary_dock], [260], Qt.Orientation.Horizontal)
    qtbot.waitUntil(lambda: rows() > 1, timeout=2000)
    assert all(tabs.bar.rect().contains(b.geometry()) for b in tabs.buttons())  # none cut off
    assert tabs.bar.width() <= window.summary_dock.width()
    tabs.buttons()[1].click()
    assert tabs.currentWidget() is window.scatter
    tabs.setCurrentWidget(window.profiles)
    assert [b.text() for b in tabs.buttons() if b.isChecked()] == ["Profiles"]


def test_every_rug_row_keeps_its_label(qtbot):
    import pyqtgraph as pg
    from PySide6.QtGui import QImage, QPainter

    configure_pyqtgraph()
    graphics = pg.GraphicsLayoutWidget()
    qtbot.addWidget(graphics)
    series, rug = graphics.addPlot(row=0, col=0), graphics.addPlot(row=1, col=0)
    names = ["dropouts · ts", "dropouts · sonic", "excursion · sonic", "coupled · sonic"]
    for plot in (series, rug):
        plot.getAxis("left").setWidth(72)
    rug.getAxis("left").setTicks([list(enumerate(names))])
    rug.setYRange(-0.7, len(names) - 0.3, padding=0)
    graphics.resize(600, 300)
    graphics.show()
    qtbot.waitExposed(graphics)
    SeriesPanel._fit_left_axes([series, rug], rug)
    qtbot.wait(50)
    assert series.getAxis("left").fixedWidth == rug.getAxis("left").fixedWidth > 72  # the time axes stay aligned
    image = QImage(400, 400, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    drawn = [text for _, _, text in rug.getAxis("left").generateDrawSpecs(painter)[2]]
    painter.end()
    assert drawn == names


def test_the_availability_strip_zooms_on_its_own(window, qtbot):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    strip = window.overview
    vb = strip.getViewBox()
    whole, rows = vb.viewRange()[0], vb.viewRange()[1]
    timeline = window.plots[0].plot_item.getViewBox().viewRange()[0]
    viewport = strip.viewport()
    at = QPointF(viewport.width() / 2, viewport.height() / 2)
    for _ in range(3):  # wheel towards the user: zoom in
        QApplication.sendEvent(viewport, QWheelEvent(at, QPointF(viewport.mapToGlobal(at.toPoint())), QPoint(0, 0),
                                                     QPoint(0, 120), Qt.MouseButton.NoButton,
                                                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                                                     False))
    qtbot.wait(50)
    zoomed = vb.viewRange()
    assert zoomed[0][1] - zoomed[0][0] < 0.9 * (whole[1] - whole[0])
    assert whole[0] <= zoomed[0][0] and zoomed[0][1] <= whole[1]  # never past the period
    assert zoomed[1] == rows  # the boom rows stay put
    assert window.plots[0].plot_item.getViewBox().viewRange()[0] == timeline  # the timelines don't follow
    qtbot.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=at.toPoint())
    assert vb.viewRange()[0] == whole


def test_boom_presets_pick_even_and_odd_booms(window):
    from PySide6.QtWidgets import QToolButton
    panel = window.panels[0]
    presets = {b.text(): b for b in panel.findChildren(QToolButton) if b.text() in ("odd", "even", "all")}
    presets["even"].click()
    assert [b for b, box in panel.boom_boxes.items() if box.isChecked()] == [b for b in panel.booms if b % 2 == 0]
    presets["odd"].click()
    assert [b for b, box in panel.boom_boxes.items() if box.isChecked()] == [b for b in panel.booms if b % 2 == 1]
    presets["all"].click()


def test_only_the_shown_tab_sets_a_panels_minimum_width(window):
    tabs = window.left_tabs
    find = next(i for i in range(tabs.count()) if tabs.tabText(i) == "Find")
    tabs.setCurrentIndex(find)
    wide = tabs.minimumSizeHint().width()
    tabs.setCurrentIndex(0)  # Panel A
    assert tabs.minimumSizeHint().width() < wide
    assert tabs.widget(find).sizePolicy().horizontalPolicy() != tabs.widget(0).sizePolicy().horizontalPolicy()
    tabs.setCurrentIndex(find)
    assert tabs.minimumSizeHint().width() == wide

def test_the_step_arrows_move_by_a_whole_or_half_interval(window):
    _, _, x, _ = _a_point(window.plots[0])
    window.set_x_range(x - 3600, x + 3600)
    window.step(1.0)
    assert window.x_range() == pytest.approx((x + 3600, x + 3 * 3600))
    window.step(-0.5)
    assert window.x_range() == pytest.approx((x, x + 2 * 3600))


def test_opening_an_inspector_shows_a_busy_cursor_until_its_slot_is_read(window, qtbot):
    from PySide6.QtWidgets import QApplication
    inspector = window.inspect(1, 3 * FAULT_HALF_HOUR + 1)
    assert QApplication.overrideCursor() is not None
    assert "opening the Slot Inspector" in window.statusBar().currentMessage()
    qtbot.waitUntil(lambda: inspector.bundle is not None, timeout=60_000)
    assert QApplication.overrideCursor() is None and window.statusBar().currentMessage() == ""
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)
    inspector.close()


def test_timeline_csv_has_labelled_columns_and_local_times(window, tmp_path):
    from ttu_tower.viewer.ui.plot_export import plot_frame
    plot = window.plots[0]
    member, _, x, _ = _a_point(plot)
    _zoom_to(window, x)
    frame = plot_frame(plot.plot_item)
    assert frame.index.name == "time (Etc/GMT+6)"
    label = plot.items[member].export_label
    assert label in frame.columns
    curve = plot.curves[member]
    (x0, x1), _ = plot.vb.viewRange()
    keep = (curve.x >= x0) & (curve.x <= x1)
    assert np.allclose(frame[label].to_numpy(dtype=float), curve.y[keep], equal_nan=True)  # every sample, not decimated
    first = pd.Timestamp(curve.x[keep][0], unit="s", tz="UTC").tz_convert("Etc/GMT+6")
    assert frame.index[0] == first.strftime("%Y-%m-%d %H:%M:%S")


def test_the_export_dialog_applies_a_title_and_labels_only_while_exporting(window, qtbot, tmp_path, monkeypatch):
    from pyqtgraph.exporters import ImageExporter
    from PySide6.QtWidgets import QMessageBox
    from ttu_tower.viewer.ui.plot_export import LabelledCSVExporter, ViewerExportDialog
    plot = window.plots[0]
    dialog = ViewerExportDialog(plot.plot_item.scene())
    qtbot.addWidget(dialog)
    dialog.show(plot.plot_item)
    assert dialog.ui.paramTree.minimumHeight() >= 150 and dialog.ui.gridLayout.rowStretch(5) > 0
    names = [dialog.ui.formatList.item(i).text() for i in range(dialog.ui.formatList.count())]
    assert "CSV of plot data (labelled columns)" in names and "CSV of original plot data" not in names
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: told.append(text))
    seen = {}
    original = ImageExporter.export

    def spy(self, fileName=None, toBytes=False, copy=False):
        seen["title"] = self.item.titleLabel.text
        seen["x"] = self.item.getAxis("bottom").labelText
        return original(self, fileName=fileName, toBytes=toBytes, copy=copy)

    monkeypatch.setattr(ImageExporter, "export", spy)
    for i, name in enumerate(names):
        if name.startswith("Image"):
            dialog.ui.formatList.setCurrentRow(i)
    dialog.text["title"].setText("Wind at 16.8 m")
    dialog.text["bottom"].setText("local time")
    path = tmp_path / "timeline.png"
    dialog.currentExporter.fileSaveDialog = lambda **_: dialog.currentExporter.export(fileName=str(path))
    dialog.exportClicked()
    assert path.exists() and told == [f"Saved to {path}"]
    assert seen == {"title": "Wind at 16.8 m", "x": "local time"}
    assert plot.plot_item.getAxis("bottom").labelText != "local time"  # the plot itself as it was
    for i, name in enumerate(names):
        if name.startswith("CSV"):
            dialog.ui.formatList.setCurrentRow(i)
    assert isinstance(dialog.currentExporter, LabelledCSVExporter)
    csv = tmp_path / "timeline.csv"
    dialog.currentExporter.fileSaveDialog = lambda **_: dialog.currentExporter.export(fileName=str(csv))
    dialog.exportClicked()
    assert pd.read_csv(csv).columns[0].startswith("time (") and told[-1] == f"Saved to {csv}"
