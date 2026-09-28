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
    assert dist.fit_table.rowCount() == len(FULL_BOOMS)
