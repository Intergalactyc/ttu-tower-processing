import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402

from ttu_tower.timegrid import slot_to_time  # noqa: E402
from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.timeaxis import display_offset_s, slot_to_unix  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402
from ttu_tower.viewer.ui.series_view import MODES as SERIES_MODES, SeriesPanel  # noqa: E402

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
    assert axis.width() >= widest + 20
    widths = {window.plots[0].plot_item.getAxis("left").width(), axis.width(),
              window.stability_band.plot.getPlotItem().getAxis("left").width(),
              window.overview.getPlotItem().getAxis("left").width()}
    assert len(widths) == 1
