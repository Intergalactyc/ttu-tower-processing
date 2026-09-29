import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.timeaxis import slot_to_unix  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402

from viewer_fixtures import FULL_BOOMS, SPIKY_BOOM, SPIKY_SLOT  # noqa: E402

pytestmark = pytest.mark.slow


def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


@pytest.fixture
def window(qtbot, full_run):
    configure_pyqtgraph()
    w = MainWindow(RunHandle.from_dir(full_run.run_dir))
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    qtbot.waitUntil(lambda: _idle(w) and w.datas[0] is not None, timeout=60_000)
    yield w
    for inspector in list(w.inspectors):
        inspector.close()
    w.runner.wait(30_000)


def _inspect(window, qtbot, tab):
    inspector = window.inspect(SPIKY_BOOM, SPIKY_SLOT, window.catalog["boom_final|sigma_w|"], "mrd", tab=tab)
    qtbot.waitUntil(lambda: inspector.bundle is not None and _idle(window), timeout=60_000)
    return inspector


def test_qc_tab_shows_the_failed_gates_as_the_run_logged_them(window, qtbot):
    inspector = _inspect(window, qtbot, "qc")
    qc = inspector.qc
    assert inspector.tabs.currentWidget() is qc
    assert qc.coverage.rowCount() == 10 and qc.flag_table.rowCount() > 5
    results = [qc.gates.item(i, qc.gates.columnCount() - 1).text() for i in range(qc.gates.rowCount())]
    assert "FAIL" in results
    assert "differs" not in qc.gate_note.text()
    qc.window_box.setCurrentIndex(0)
    assert qc.flag_table.rowCount() > 5


def test_profile_tab_ghosts_the_filtered_boom_and_switches_boom_on_click(window, qtbot):
    inspector = _inspect(window, qtbot, "profile")
    profile = inspector.profile
    qtbot.waitUntil(lambda: bool(profile.plots), timeout=60_000)
    assert len(profile.plots) == 10
    points = profile.points["sigma_w"].set_index("boom")
    assert points.loc[SPIKY_BOOM, "filtered"]
    other = next(b for b in FULL_BOOMS if b != SPIKY_BOOM)
    profile.boomClicked.emit(other)
    qtbot.waitUntil(lambda: inspector.bundle is not None and inspector.bundle.boom == other and _idle(window),
                    timeout=60_000)
    assert inspector.boom == other and profile.boom == other


def test_whatif_tab_reproduces_the_run_then_follows_an_edit(window, qtbot):
    inspector = _inspect(window, qtbot, "whatif")
    whatif = inspector.whatif
    qtbot.waitUntil(lambda: whatif.result is not None, timeout=60_000)
    assert not whatif.result["changed"].any() and whatif.is_default()
    assert set(whatif.plots) == {"heat", "momentum"}
    whatif.fields["peak_significance_se"].setValue(50.0)
    assert not whatif.is_default() and whatif.result["changed"].any()
    whatif.fields["min_tau_s"].setCurrentIndex(whatif.fields["min_tau_s"].findData(1200.0))
    assert whatif.problem.text()
    whatif.reset()
    assert whatif.is_default() and not whatif.result["changed"].any() and not whatif.problem.text()


def test_timeline_ghosts_filtered_values_and_a_click_on_one_opens_the_qc_tab(window, qtbot):
    window.panels[0].select("boom_final|sigma_w|", "mrd", tuple(FULL_BOOMS))
    window.panels[0].tau_box.setChecked(True)
    plot = window.plots[0]
    qtbot.waitUntil(lambda: _idle(window) and SPIKY_BOOM in plot.filtered_items, timeout=60_000)
    assert "filtered" in plot.overlay_key.text() and SPIKY_BOOM in plot.filtered_crosses
    plot.legend.buttons[SPIKY_BOOM].setChecked(False)
    assert not plot.filtered_crosses[SPIKY_BOOM].isVisible()
    plot.legend.buttons[SPIKY_BOOM].setChecked(True)
    assert window.night_box.isChecked()
    pre = plot.filtered.curves[SPIKY_BOOM]
    i = int(np.flatnonzero(pre.slots == SPIKY_SLOT)[0])
    x, y = float(pre.x[i]), float(pre.y[i])
    window.set_x_range(x - 3 * 3600, x + 3 * 3600)
    hits = plot.pick(x, y)
    assert hits and hits[0][1:] == (SPIKY_SLOT, SPIKY_BOOM, "filtered")
    assert "spike w" in plot._overlay_note(SPIKY_SLOT, SPIKY_BOOM)
    plot.filteredClicked.emit(SPIKY_SLOT, SPIKY_BOOM)
    qtbot.waitUntil(lambda: bool(window.inspectors) and window.inspectors[-1].bundle is not None, timeout=60_000)
    assert window.inspectors[-1].tabs.currentWidget() is window.inspectors[-1].qc


def test_anisotropy_tab_maps_every_boom_and_follows_clicks(window, qtbot):
    inspector = _inspect(window, qtbot, "anisotropy")
    tab = inspector.anisotropy
    qtbot.waitUntil(lambda: tab.states is not None and tab.track is not None and _idle(window), timeout=60_000)
    assert list(tab.states["boom"]) == FULL_BOOMS and tab.table.rowCount() == len(FULL_BOOMS)
    assert tab.states["state"].notna().any()
    assert len(tab.track) > 1
    tab.whole.setChecked(True)
    qtbot.waitUntil(lambda: tab.density is not None, timeout=60_000)
    assert tab.density[0].size > 10
    tab.slotClicked.emit(SPIKY_SLOT - 2)
    qtbot.waitUntil(lambda: inspector.bundle.slot == SPIKY_SLOT - 2 and tab.across is not None
                    and tab.across.slot == SPIKY_SLOT - 2 and _idle(window), timeout=60_000)


def test_anisotropy_band_rows_follow_the_boom_toggles(window, qtbot):
    band = window.aniso_band
    qtbot.waitUntil(lambda: band.codes is not None, timeout=60_000)
    assert band.rows == FULL_BOOMS and (band.codes >= 0).any()
    band.boom_buttons[FULL_BOOMS[0]].setChecked(False)
    assert band.rows == FULL_BOOMS[1:]
    band.cellClicked.emit(SPIKY_SLOT, SPIKY_BOOM)
    qtbot.waitUntil(lambda: bool(window.inspectors) and window.inspectors[-1].bundle is not None, timeout=60_000)
    assert window.inspectors[-1].tabs.currentWidget() is window.inspectors[-1].anisotropy


def test_scatter_follows_the_interval_and_fits_only_when_asked(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    window.panels[1].select("boom_final|ustar|", "mrd", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[1] is not None
                    and window.datas[1].quantity.key == "boom_final|ustar|", timeout=60_000)
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    window.full_period.setChecked(True)
    whole = scatter.x.size
    assert whole > 10 and scatter.x.size == scatter.y.size
    assert scatter.member[0].currentData() == scatter.member[1].currentData()
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear"))
    assert scatter.fit_result is None  # nothing is fitted until asked
    scatter.run_fit()
    qtbot.waitUntil(lambda: scatter.fit_result is not None, timeout=60_000)
    assert scatter.fit_result.n == whole and "earlier" not in scatter.fit_status.text()
    fitted = scatter.fit_result

    window.full_period.setChecked(False)
    slots = scatter.slots
    window.set_x_range(slot_to_unix(slots[0]), slot_to_unix(slots[len(slots) // 2]))
    window._refresh_distributions()
    assert 0 < scatter.x.size < whole
    assert scatter.fit_result is fitted and "earlier" in scatter.fit_status.text()  # kept, marked stale
    scatter.fit_bins.setValue(5)
    scatter.run_fit()
    qtbot.waitUntil(lambda: scatter.fit_result is not fitted, timeout=60_000)
    assert scatter.fit_result.n <= 5 and "earlier" not in scatter.fit_status.text()
    for i in range(scatter.display.count()):
        scatter.display.setCurrentIndex(i)
    scatter.color_by.setCurrentIndex(scatter.color_by.findData("stability"))
    scatter._swap()
    assert scatter.source[0].currentData() == 1


def test_wind_rose_tab_follows_boom_filtering_and_interval(window, qtbot):
    window.right_tabs.setCurrentWidget(window.windrose)
    rose = window.windrose
    window.full_period.setChecked(True)
    qtbot.waitUntil(lambda: rose.tower is not None and _idle(window), timeout=60_000)
    assert rose.tower_plot.table is not None and rose.tower_plot.table.sum() == pytest.approx(100.0)
    assert rose.mesonet is None and "not merged" in rose.meso_plot.getPlotItem().titleLabel.text
    rose.boom.setCurrentIndex(rose.boom.findData(SPIKY_BOOM))
    qtbot.waitUntil(lambda: _idle(window) and rose.tower is not None, timeout=60_000)
    filtered = rose.tower.slots.size
    rose.filtering.setCurrentIndex(1)
    qtbot.waitUntil(lambda: _idle(window) and rose.tower.slots.size != filtered, timeout=60_000)
    assert rose.tower.slots.size > filtered and SPIKY_SLOT in rose.tower.slots
    shadow = window.run.cfg.qc.second_layer.shadow_sector
    assert rose.bearings.text() == ", ".join(f"{b:g}" for b in shadow)  # the shadow sector's edges by default
    window.full_period.setChecked(False)
    x = slot_to_unix(SPIKY_SLOT)
    window.set_x_range(x - 3600, x + 3600)
    window._refresh_distributions()
    t = slot_to_unix(rose.tower.slots)
    n = int(((t >= x - 3600) & (t < x + 3600)).sum())
    assert 0 < n < rose.tower.slots.size and rose.tower_plot.table.sum() == pytest.approx(100.0)
    assert f"{n} slots" in rose.tower_plot.getPlotItem().titleLabel.text


def test_histograms_take_a_distribution_fit(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.full_period.setChecked(True)
    dist = window.dists[0]
    dist.fit_box.setCurrentIndex(dist.fit_box.findData("Weibull"))
    qtbot.waitUntil(lambda: bool(dist.fits) and _idle(window), timeout=60_000)
    assert all(f is not None and f.params["k"] > 0 for f in dist.fits.values())
    headers = [dist.table.horizontalHeaderItem(j).text() for j in range(dist.table.columnCount())]
    assert dist.table.rowCount() == len(FULL_BOOMS) and {"N", "k", "KS", "AIC"} <= set(headers)


def test_histograms_take_one_stability_class(window, qtbot):
    from ttu_tower.viewer.timeline import Curve
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    data = window.datas[0]
    slots = np.unique(np.concatenate([c.slots for c in data.curves.values()]))
    names = ["unstable", "neutral", "stable"]
    stability = (Curve(slots=slots, y=(slots % 3).astype(float)), names)  # the fixture's booms make no Ri_b pair
    dist = window.dists[0]
    dist.set_data(data, list(data.curves), None, stability)
    everything = sum(np.isfinite(v).sum() for v in dist.values_by_member().values())
    parts = []
    for name in names:
        dist.stability_box.setCurrentIndex(dist.stability_box.findData(name))
        parts.append(sum(np.isfinite(v).sum() for v in dist.values_by_member().values()))
        assert name in dist.title.text()
    assert dist.stability_box.isEnabled() and dist.stability_box.count() == len(names) + 1
    assert all(n > 0 for n in parts) and sum(parts) == everything

def test_scatter_alpha_ri_fit_takes_its_own_settings(window, qtbot):
    from ttu_tower.viewer import curvefits
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData(curvefits.ALPHA_RI))
    assert set(scatter.fit_settings) == {"ri_neutral_lo", "ri_neutral_hi", "ri_critical"}
    scatter.fit_settings["ri_critical"].setText("0.3")
    assert dict(scatter._settings())["ri_critical"] == 0.3
    scatter.fit_settings["ri_critical"].setText("x")
    scatter.run_fit()
    assert "isn't a number" in scatter.fit_status.text()
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear"))
    assert scatter.fit_settings == {}


class _Click:
    def __init__(self, scene_pos):
        self._pos = scene_pos

    def button(self):
        from PySide6.QtCore import Qt
        return Qt.MouseButton.LeftButton

    def double(self):
        return False

    def scenePos(self):
        return self._pos

    def screenPos(self):
        from PySide6.QtCore import QPointF
        return QPointF(10, 10)


def test_a_click_on_any_quantity_opens_its_slot(window, qtbot):
    from PySide6.QtCore import QPointF
    failures = []
    for q in list(window.catalog):
        members = q.pairs[:1] if q.kind == "pair" else ((None,) if q.kind == "slot" else tuple(FULL_BOOMS))
        window.panels[0].select(q.key, q.variants[0], members)
        qtbot.waitUntil(lambda: _idle(window) and window.datas[0] is not None
                        and window.datas[0].quantity.key == q.key, timeout=60_000)
        plot = window.plots[0]
        target = None
        for member in plot.visible_members():
            c = plot.curves[member]
            ok = np.flatnonzero(np.isfinite(c.y) & ((c.y > 0) if plot.log_y else True))
            if ok.size:
                i = ok[ok.size // 2]
                target = (int(c.slots[i]), float(c.x[i]), float(c.y[i]))
                break
        if target is None:
            continue  # nothing plotted for this quantity in the fixture
        slot, x, y = target
        window.set_x_range(x - 3 * 3600, x + 3 * 3600)
        plot.decimator.redraw()
        for w in list(window.inspectors):
            w.close()
        qtbot.waitUntil(lambda: not window.inspectors, timeout=10_000)
        hits = plot.pick(x, y)
        clicked = plot._on_click(_Click(plot.vb.mapViewToScene(QPointF(x, y))))
        qtbot.wait(20)
        if not clicked or not window.inspectors:
            failures.append(f"{q.key}: clicked={clicked}, hits={hits[:3]}")
            continue
        inspector = window.inspectors[-1]
        qtbot.waitUntil(lambda: inspector.bundle is not None and _idle(window), timeout=60_000)
        if inspector.slot != slot:
            failures.append(f"{q.key}: slot {inspector.slot} != {slot}")
    assert not failures, failures


def test_qc_summary_tab_fills_a_row_per_boom(window, qtbot):
    window.right_tabs.setCurrentWidget(window.qc_summary)
    qc = window.qc_summary
    qtbot.waitUntil(lambda: qc.flags is not None and _idle(window), timeout=60_000)
    for table in qc._tables():
        assert table.rowCount() == len(FULL_BOOMS), table.toolTip()
    headers = [qc.flag_table.horizontalHeaderItem(j).text() for j in range(qc.flag_table.columnCount())]
    assert "spike" in headers
    spike = headers.index("spike")
    values = {qc.flag_table.verticalHeaderItem(i).text(): qc.flag_table.item(i, spike).text() for i in range(2)}
    assert float(values[f"b{SPIKY_BOOM} (2.4 m)"]) > 0  # the spiky boom's w spikes


def test_profiles_tab_draws_each_panel_with_its_default_fit(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.profiles)
    view = window.profiles
    plot_a = view.plots[0]
    assert plot_a.fit_box.currentData() == "power law"
    assert list(plot_a.stats["boom"]) == FULL_BOOMS and plot_a.stats["centre"].notna().all()
    assert plot_a.table.rowCount() == 1  # two booms are too few to fit: the row says so
    view.method.setCurrentIndex(1)
    assert view.error_bars.text() == "± MAD"


def test_controls_that_cant_apply_are_greyed(window, qtbot):
    key = next(q.key for q in window.catalog if q.categorical and q.kind == "boom")
    window.panels[0].select(key, None, tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == key, timeout=60_000)
    dist = window.dists[0]
    assert not dist.log_bins.isEnabled() and not dist.fit_box.isEnabled()
    assert not window.panels[0].style.isEnabled() and not window.panels[0].log_y.isEnabled()


def test_distribution_tables_scroll_and_leave_the_histogram_most_of_the_height(window, qtbot):
    from ttu_tower.viewer.ui.distribution import TABLE_SHARE
    dist = window.dists[0]
    qtbot.waitUntil(lambda: dist.table.rowCount() > 0, timeout=60_000)
    qtbot.wait(50)
    plot_height, table_height = dist.split.sizes()
    assert table_height <= dist.table.maximumHeight()  # never blank rows below the last boom
    assert table_height <= max(TABLE_SHARE * (plot_height + table_height) + 1, dist.table.minimumHeight())
    dist.split.moveSplitter(plot_height // 2, 1)  # dragged: the user's height sticks through refreshes
    dist.split.splitterMoved.emit(plot_height // 2, 1)
    dragged = dist.split.sizes()
    dist.refresh()
    assert dist.split.sizes() == dragged


def test_profile_classes_sit_a_few_pixels_apart_with_capped_bars(window, qtbot):
    from ttu_tower.viewer.timeline import Curve
    from ttu_tower.viewer.ui.profiles_view import OFFSET_PX
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.right_tabs.setCurrentWidget(window.profiles)
    view = window.profiles
    data = window.datas[0]
    slots = np.unique(np.concatenate([c.slots for c in data.curves.values()]))
    stability = (Curve(slots=slots, y=(slots % 2).astype(float)), ["unstable", "stable"])
    view.set_sources([(data, list(data.curves)), None], None, stability)
    view.grouping.setCurrentIndex(view.grouping.findData("stability"))
    view.error_bars.setChecked(True)
    plot = view.plots[0]
    (first, bars, _, z, _, _, _), (second, _, _, _, _, _, _) = plot._marks
    py = plot.plot.getViewBox().viewPixelSize()[1]
    gap = second.getData()[1] - first.getData()[1]
    assert np.allclose(gap, OFFSET_PX * py) and np.allclose(first.getData()[1], z - OFFSET_PX * py / 2)
    assert bars is not None and bars.opts["beam"] > 0


def _drag(qtbot, widget, points, modifiers):
    """A left-button drag through `points` (widget coordinates), as real mouse events."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    viewport = widget.viewport()
    for i, p in enumerate(points):
        kind = (QEvent.Type.MouseButtonPress if i == 0 else QEvent.Type.MouseButtonRelease
                if i == len(points) - 1 else QEvent.Type.MouseMove)
        held = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton
        pos = QPointF(p)
        QApplication.sendEvent(viewport, QMouseEvent(kind, pos, QPointF(viewport.mapToGlobal(pos.toPoint())),
                                                     Qt.MouseButton.LeftButton, held, modifiers))
        qtbot.wait(40)


def test_shift_drag_on_a_timeline_brushes_its_span_everywhere(window, qtbot):
    from PySide6.QtCore import QPointF, Qt
    from ttu_tower.viewer import brush
    plot = window.plots[0]
    member = plot.visible_members()[0]
    k = int(plot.curves[member].slots[len(plot.curves[member].slots) // 2])
    x = slot_to_unix(k)
    window.set_x_range(x - 4 * 3600, x + 4 * 3600)
    qtbot.wait(50)
    y = float(np.mean(plot.vb.viewRange()[1]))
    span = (x - 1800 + 300, x + 1800 + 300)  # mid-slot, clear of pixel rounding at slot edges
    ends = [plot.widget.mapFromScene(plot.vb.mapViewToScene(QPointF(v, y))) for v in (span[0], x, span[1])]
    _drag(qtbot, plot.widget, ends, Qt.KeyboardModifier.ShiftModifier)
    assert list(window.selection) == list(brush.slots_in_span(*span))
    assert window.x_range() == pytest.approx((x - 4 * 3600, x + 4 * 3600))  # brushing doesn't pan
    assert set(plot.selected_items) == set(plot.visible_members())
    assert window.overview.selection_image.image is not None and "slots" in window.selection_label.text()
    assert window.scatter.selection.size == window.selection.size
    window.set_selection(np.array([k + 30]), "add")
    assert window.selection.size == brush.slots_in_span(*span).size + 1
    from PySide6.QtTest import QTest
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.selection.size == 0 and not plot.selected_items and not window.selection_clear.isVisible()


def test_a_lasso_on_the_scatter_selects_the_points_inside(window, qtbot):
    from PySide6.QtCore import QPointF, Qt
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    qtbot.waitUntil(lambda: scatter.x.size > 10, timeout=60_000)
    slots, x, y = scatter._shown()
    vx, vy = scatter._to_view(x, y)
    lo_x, hi_x = np.percentile(vx, [20, 80])
    lo_y, hi_y = np.percentile(vy, [20, 80])
    corners = [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y), (lo_x, lo_y)]
    points = [scatter.plot.mapFromScene(scatter.vb.mapViewToScene(QPointF(a, b))) for a, b in corners]
    _drag(qtbot, scatter.plot, points, Qt.KeyboardModifier.ShiftModifier)
    inside = (vx > lo_x) & (vx < hi_x) & (vy > lo_y) & (vy < hi_y)
    assert 0 < window.selection.size and set(window.selection) <= set(slots)
    # the lasso is drawn in screen pixels, so allow the points right on its edge
    assert abs(window.selection.size - inside.sum()) <= max(2, 0.05 * inside.sum())


def test_find_steps_the_inspector_through_its_matches(window, qtbot):
    panel = window.find
    panel.query.setText(f"ws_mean > 0 and boom == {SPIKY_BOOM}")
    panel.run_query()
    qtbot.waitUntil(lambda: panel.result is not None and _idle(window), timeout=60_000)
    matches = list(zip(panel.result["slot"], panel.result["boom"]))
    assert len(matches) > 3 and panel.table.rowCount() == len(matches)
    panel.table.setCurrentCell(1, 0)
    panel.step_button.click()
    inspector = window.inspectors[-1]
    qtbot.waitUntil(lambda: inspector.bundle is not None and _idle(window), timeout=60_000)
    assert (inspector.slot, inspector.boom) == matches[1] and inspector.seq_info.text().startswith("2 /")
    inspector.step_sequence(1)
    qtbot.waitUntil(lambda: inspector.bundle.slot == matches[2][0] and _idle(window), timeout=60_000)
    assert inspector.seq_info.text().startswith("3 /")
    panel.select_button.click()
    assert list(window.selection) == sorted({int(k) for k, _ in matches})
    window.inspect(SPIKY_BOOM, matches[0][0])  # an ordinary click drops the sequence
    assert not inspector.seq_next.isVisible()
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)


def test_bookmarks_are_saved_listed_and_opened(window, qtbot, tmp_path):
    from ttu_tower.viewer.bookmarks import Bookmark, BookmarkStore
    store = BookmarkStore(tmp_path / "bookmarks.json")
    window.bookmarks = window.bookmark_panel.store = store
    tag = window.run.tag
    window._store_bookmarks([Bookmark(tag, SPIKY_SLOT, SPIKY_BOOM, "mrd", "w spikes", "checks"),
                             Bookmark(tag, SPIKY_SLOT + 1, 1, "naive", "", "checks"),
                             Bookmark("another run", 5, 1)])
    panel = window.bookmark_panel
    assert (tmp_path / "bookmarks.json").exists() and panel.table.rowCount() == 2
    panel.all_runs.setChecked(True)
    assert panel.table.rowCount() == 3
    panel.all_runs.setChecked(False)
    panel.list_filter.setCurrentIndex(panel.list_filter.findText("checks"))
    note = panel.table.item(0, 3)
    note.setText("w spikes, fine")
    assert BookmarkStore(tmp_path / "bookmarks.json").items[0].note == "w spikes, fine"
    panel.table.setCurrentCell(1, 0)
    panel.open_button.click()
    inspector = window.inspectors[-1]
    qtbot.waitUntil(lambda: inspector.bundle is not None and _idle(window), timeout=60_000)
    assert (inspector.slot, inspector.boom, inspector.variant()) == (SPIKY_SLOT + 1, 1, "naive")
    assert inspector.seq_info.text().startswith("2 / 2")
