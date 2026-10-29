import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from ttu_tower.constants import HEIGHTS  # noqa: E402
from ttu_tower.viewer import figures  # noqa: E402
from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.timeaxis import slot_to_unix  # noqa: E402
from ttu_tower.viewer.ui import view_state  # noqa: E402
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
    assert len(profile.plots) == 18 and {"tke", "zeta", "uw", "ils_u", "theta_star", "n2", "rig", "rif"} <= set(profile.plots)
    assert all(profile.points[k]["value"].notna().any() for k in ("uw", "ils_u"))
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
    whatif.smoothing.setCurrentIndex(whatif.smoothing.findText("none"))
    assert not whatif.is_default() and "unsmoothed" in whatif.key.text()
    whatif.smoothing.setEditText("1-2")
    assert whatif.problem.text().startswith("smoothing")
    whatif.smoothing.setEditText("1-3-1")
    assert not whatif.problem.text() and "1-3-1 smoothed" in whatif.key.text()
    whatif.normalize["momentum"].setChecked(True)
    assert "normalized" in whatif.plots["momentum"].titleLabel.text
    assert "normalized" not in whatif.plots["heat"].titleLabel.text
    whatif.reset()
    assert whatif.is_default() and not whatif.result["changed"].any()


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


def test_scatter_can_show_only_the_selection_and_drop_its_rings(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    window.panels[1].select("boom_final|ustar|", "mrd", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[1] is not None
                    and window.datas[1].quantity.key == "boom_final|ustar|", timeout=60_000)
    window.right_tabs.setCurrentWidget(window.scatter)
    window.full_period.setChecked(True)
    scatter = window.scatter
    assert not scatter.points_shown.isEnabled() and scatter.ring.isChecked()  # nothing brushed yet

    def rings():
        return [i for i in scatter.plot.getPlotItem().items if (getattr(i, "name", lambda: "")() or "").startswith(
            "selected")]
    chosen = scatter.slots[::3]
    window.set_selection(chosen, "replace")
    assert scatter.points_shown.isEnabled() and scatter._shown()[0].size == scatter.slots.size and rings()
    scatter.points_shown.setCurrentIndex(scatter.points_shown.findData("selected"))
    shown = scatter._shown()[0]
    assert sorted(shown) == sorted(chosen) and scatter.stats.text().startswith(f"N {chosen.size}")
    scatter.ring.setChecked(False)
    assert not rings()
    assert scatter.figure_spec()[1]["selected"] is None
    window.set_selection(np.empty(0, dtype=np.int64), "replace")
    assert scatter._shown()[0].size == scatter.slots.size  # nothing brushed: every slot again


def test_distributions_show_the_selection_the_rest_or_every_slot(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.right_tabs.setCurrentWidget(window.dist_tab)
    window.full_period.setChecked(True)
    dist = window.dists[0]
    assert dist.showing() == "all" and not dist.show_box.isEnabled()
    slots = window.datas[0].curves[FULL_BOOMS[0]].slots
    window.set_selection(slots[:6], "replace")
    assert dist.showing() == "both" and set(dist._stats) == {"rest", "selected"} and not dist.sel_table.isHidden()

    def labels():
        return [getattr(i, "export_label", "") for i in dist.plot.getPlotItem().listDataItems()]
    assert any(text.endswith("brushed selection") for text in labels())
    dist.show_box.setCurrentIndex(dist.show_box.findData("selected"))
    assert set(dist._stats) == {"selected"} and dist.sel_table.isHidden()
    assert dist._stats["selected"]["N"].max() <= 6 and all(text.endswith("brushed selection") for text in labels())
    dist.show_box.setCurrentIndex(dist.show_box.findData("all"))
    assert set(dist._stats) == {"all"} and dist._stats["all"]["N"].max() > 6
    assert not any("selection" in text or "rest" in text for text in labels())


def test_distribution_bins_can_be_counted_and_ranged(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.right_tabs.setCurrentWidget(window.dist_tab)
    window.full_period.setChecked(True)
    dist = window.dists[0]
    assert dist._edges.size == 61 and not dist.bin_lo.isEnabled()
    auto = (dist._edges[0], dist._edges[-1])
    assert (dist.bin_lo.value(), dist.bin_hi.value()) == pytest.approx(auto, abs=1e-3)
    dist.bin_count.setValue(20)
    assert dist._edges.size == 21
    dist.bin_auto.setChecked(False)
    assert dist.bin_lo.isEnabled() and dist._edges[0] == pytest.approx(auto[0], abs=1e-3)  # starts from auto's
    dist.bin_lo.setValue(0.0)
    dist.bin_hi.setValue(40.0)
    assert dist._edges[0] == 0.0 and dist._edges[-1] == 40.0 and dist._edges.size == 21
    x, _ = dist.plot.getPlotItem().listDataItems()[0].getData()
    assert x[0] == 0.0 and x[-1] == 40.0
    dist.bin_auto.setChecked(True)
    assert dist._edges[0] == pytest.approx(auto[0])


def test_stability_can_be_classified_by_another_pair_or_by_zeta(window, qtbot, full_run):
    from ttu_tower.viewer import stability as criteria
    panel = window.stability_panel
    assert window.stability_spec is None and "the run's own" in panel.in_use.text()
    own = window.stability

    def classified(spec):
        return lambda: _idle(window) and window.stability_spec == spec and window.stability is not own

    rib = criteria.StabilitySpec("rib", (1, 2), "mrd", (-0.5, -0.1, 0.1, 0.5))
    panel.set_spec(rib)
    panel._emit()  # as an edit would
    qtbot.waitUntil(classified(rib), timeout=60_000)
    curve, names = window.stability
    assert names == list(criteria.NAMES) and np.isfinite(curve.y).any()
    assert "Ri_b(b1, b2)" in panel.in_use.text() and "Ri_b(b1, b2)" in window.stability_band.legend.text()
    assert panel.counts.rowCount() == 5
    values = criteria.values(window.index, rib)
    expected = criteria.classify(values, rib.classes()).dropna()
    got = pd.Series(np.asarray(names, dtype=object)[curve.y[np.isfinite(curve.y)].astype(int)],
                    index=curve.slots[np.isfinite(curve.y)])
    assert got.equals(expected.loc[got.index]) and len(got) == len(expected)

    panel.variable.setCurrentIndex(panel.variable.findData("zeta"))
    panel.boom_a.setCurrentIndex(panel.boom_a.findData(2))
    zeta = panel.spec()
    assert zeta.variable == "zeta" and zeta.edges == criteria.ZETA_EDGES and zeta.booms[0] == 2
    qtbot.waitUntil(classified(zeta), timeout=60_000)
    assert "ζ at b2" in window.stability_band.legend.text()

    panel.edges[1].setValue(panel.edges[2].value() + 1)  # boundaries out of order: refused
    assert not panel.errors.isHidden() and window.stability_spec == zeta

    again = MainWindow(RunHandle.from_dir(full_run.run_dir))  # the choice is remembered for the run
    qtbot.addWidget(again)
    try:
        qtbot.waitUntil(lambda: _idle(again) and again.stability_spec == zeta, timeout=60_000)
    finally:
        again.close()
    panel.reset.click()
    qtbot.waitUntil(lambda: _idle(window) and window.stability_spec is None, timeout=60_000)
    assert "the run's own" in panel.in_use.text()


def test_wind_rose_tab_follows_boom_filtering_and_interval(window, qtbot):
    window.right_tabs.setCurrentWidget(window.windrose)
    rose = window.windrose
    window.full_period.setChecked(True)
    qtbot.waitUntil(lambda: rose.tower is not None and _idle(window), timeout=60_000)
    assert rose.tower_plot.table is not None and rose.tower_plot.table.sum() == pytest.approx(100.0)
    assert rose.mesonet is None and "not merged" in rose.meso_plot.getPlotItem().titleLabel.text
    assert rose.filtering.currentData() is False and rose.same_slots.isChecked()  # unfiltered, same slots
    rose.boom.setCurrentIndex(rose.boom.findData(SPIKY_BOOM))
    rose.filtering.setCurrentIndex(0)
    qtbot.waitUntil(lambda: _idle(window) and rose.loaded_key == (SPIKY_BOOM, True), timeout=60_000)
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
    offered = [dist.fit_box.itemData(i) for i in range(dist.fit_box.count())]
    assert {"Rayleigh", "exponential", "beta"} <= set(offered)
    dist.fit_box.setCurrentIndex(dist.fit_box.findData("Rayleigh"))
    qtbot.waitUntil(lambda: bool(dist.fits) and _idle(window) and "σ (fit)" in [
        dist.table.horizontalHeaderItem(j).text() for j in range(dist.table.columnCount())], timeout=60_000)
    assert all(f is not None and f.name == "Rayleigh" for f in dist.fits.values())
    dist.fit_box.setCurrentIndex(dist.fit_box.findData("beta"))  # wind speeds aren't between 0 and 1
    assert not dist.fits and dist.note.text() == "beta needs every value between 0 and 1"
    assert not dist.note.isHidden()


def test_summary_table_labels_explain_themselves_and_cumulative_reuses_the_fit(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.full_period.setChecked(True)
    dist = window.dists[0]
    dist.fit_box.setCurrentIndex(dist.fit_box.findData("gamma"))
    qtbot.waitUntil(lambda: bool(dist.fits) and _idle(window), timeout=60_000)
    table = dist.table
    tips = {table.horizontalHeaderItem(j).text(): table.horizontalHeaderItem(j).toolTip()
            for j in range(table.columnCount())}
    assert {"skew", "kurt"} <= set(tips) and tips["kurt"] == "kurtosis (3 for a normal)"
    assert list(tips)[6:13] == ["min", "p5", "p25", "median", "p75", "p95", "max"]
    assert tips["p25"] == "25th percentile (lower quartile)" and tips["max"] == "largest value"
    assert tips["AIC"].startswith("Akaike information criterion") and tips["k"] == "gamma fit: shape"
    assert tips["θ"] == "gamma fit: scale"
    assert not table.toolTip() and not table.item(0, 0).toolTip()
    row = table.verticalHeaderItem(0).toolTip()
    assert "gamma fit" in row and "f(x) = " in row and "Γ" not in row  # the fitted numbers, not the symbols
    assert "Γ(k)" in dist.fit_box.toolTip()
    assert "Γ(k)" in dist.fit_box.itemData(dist.fit_box.findData("gamma"), Qt.ItemDataRole.ToolTipRole)
    fits = dist.fits
    dist.cumulative.setChecked(True)
    assert dist.fits is fits and not dist.bin_count.isEnabled()  # the same fit, drawn as CDFs
    items = [i for i in dist.plot.getPlotItem().listDataItems() if hasattr(i, "export_label")]
    cdfs = [i for i in items if i.export_label.endswith("gamma CDF")]
    assert len(cdfs) == len(FULL_BOOMS)
    empirical = [i for i in items if i not in cdfs]
    y = empirical[0].getData()[1]
    assert np.all(np.diff(y) >= 0) and 0 <= y[0] < y[-1] <= 1
    spec = dist.figure_spec()
    assert spec["ylabel"] == "cumulative fraction" and len(spec["cdfs"]) == len(FULL_BOOMS)
    assert all("cdf" in f for f in spec["fits"])
    dist.cumulative.setChecked(False)
    assert dist.fits is fits and dist.figure_spec()["hists"]


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


def test_scatter_fixes_a_power_laws_exponent(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    qtbot.waitUntil(lambda: scatter.x.size > 10, timeout=60_000)
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("power law, fixed exponent"))
    assert set(scatter.fit_settings) == {"b"} and scatter.fit_settings["b"].text() == "0.5"
    scatter.fit_settings["b"].setText("0.33")
    scatter.run_fit()
    qtbot.waitUntil(lambda: scatter.fit_result is not None and not scatter._fitting, timeout=60_000)
    assert scatter.fit_result.params["b"] == 0.33 and "x^0.33" in scatter.fit_result.label


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


def test_variant_starts_at_mrd_and_keeps_the_last_choice(window):
    panel = window.panels[0]
    assert panel.current_quantity().key == "boom_final|ws|mean"
    assert panel.variant.currentData() == "mrd"
    panel.select("boom_final|t|mean")
    assert panel.variant.currentData() == "none" and not panel.variant.isEnabled()
    panel.select("boom_final|ws|mean")
    assert panel.variant.currentData() == "mrd"
    panel.variant.setCurrentIndex(panel.variant.findData("naive"))
    panel.select("boom_final|t|mean")
    panel.select("boom_final|ustar|")
    assert panel.variant.currentData() == "naive"


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
    qtbot.wait(300)  # the plot laid out at its size, as it is under a real drag
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
    assert inspector.bookmark_button.text().startswith("★") and "'checks'" in inspector.bookmark_button.toolTip()
    inspector.go_to(SPIKY_SLOT + 3)  # no bookmark there
    assert inspector.bookmark_button.text().startswith("☆")
    inspector.go_to(SPIKY_SLOT + 1)
    assert inspector.bookmark_button.text().startswith("★")
    store.remove(b.id for b in store.items)
    window._bookmarks_changed()
    assert inspector.bookmark_button.text().startswith("☆")
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)


def test_composite_spectra_draw_a_median_per_boom(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.composite)
    view = window.composite
    qtbot.waitUntil(lambda: bool(view.summaries) and _idle(window), timeout=60_000)
    assert set(view.summaries) == set(FULL_BOOMS)
    everything = view.summaries[FULL_BOOMS[0]]["n"].max()
    window.set_selection(np.unique(window.datas[0].curves[FULL_BOOMS[0]].slots)[:5], "replace")
    view.group.setCurrentIndex(view.group.findData("selected"))
    assert 0 < view.summaries[FULL_BOOMS[0]]["n"].max() <= 5 < everything
    view.spectrum.setCurrentIndex(view.spectrum.findData("ww"))
    qtbot.waitUntil(lambda: view._loaded_key == view._key() and bool(view.summaries), timeout=60_000)
    assert view.log_y.isEnabled()
    view.spectrum.setCurrentIndex(view.spectrum.findData("wvpts_te"))
    qtbot.waitUntil(lambda: view._loaded_key == view._key() and bool(view.summaries), timeout=60_000)
    assert not view.normalize.isEnabled() and not view.log_y.isEnabled()
    s = view.summaries[FULL_BOOMS[0]]
    assert len(s) == 8 and ((s["mid"] > 0) & (s["mid"] <= 1)).all()


def test_a_diurnal_cell_click_selects_its_slots(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.diurnal)
    view = window.diurnal
    qtbot.waitUntil(lambda: view.values is not None, timeout=60_000)
    h, m = map(int, np.argwhere(view.counts > 0)[0])
    assert view._clicked(m + 0.5, h + 0.5)
    assert window.selection.size == view.counts[h, m]


def test_the_anisotropy_map_lassoes_slots(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.aniso_map)
    view = window.aniso_map
    qtbot.waitUntil(lambda: view.states is not None and _idle(window), timeout=60_000)
    slots, _, _ = view.shown()
    assert len(view.panels) == 1 and view.display.currentData() == "heatmap"
    view._on_lasso(np.array([-1.0, 2.0, 2.0, -1.0]), np.array([-1.0, -1.0, 2.0, 2.0]), "replace")
    assert list(window.selection) == sorted(set(slots)) and view.table.rowCount() >= 1
    assert view.split.currentData() == "selection" and [p[0] for p in view.panels] == ["selected", "the rest"]
    window.set_selection(slots[:3], "replace")
    rest_panel = view.panels[1]
    view._on_lasso(np.array([-1.0, 2.0, 2.0, -1.0]), np.array([-1.0, -1.0, 2.0, 2.0]), "replace", rest_panel[2])
    assert list(window.selection) == sorted(set(slots) - set(slots[:3]))  # a lasso on one panel takes its slots only


def test_comparing_a_run_with_itself_agrees_everywhere(window, qtbot):
    window._set_other_run(RunHandle.from_dir(window.run.run_dir))
    qtbot.waitUntil(lambda: all(o is not None and not isinstance(o, str) for o in window.others)
                    and _idle(window), timeout=60_000)
    assert set(window.plots[0].compare_items) == set(window.plots[0].items)
    window.right_tabs.setCurrentWidget(window.compare)
    frame = window.compare.frame
    assert frame is not None and len(frame) == len(window.datas[0].curves)
    assert (frame["N both"] == frame["N (this run)"]).all() and (frame["RMSE"] == 0).all()
    window._set_other_run(None)
    assert not window.plots[0].compare_items


def test_figures_export_from_every_tab(window, qtbot, tmp_path):
    from ttu_tower.viewer import figures
    window.full_period.setChecked(True)
    spec = window.timelines_spec()
    figures.save(figures.render(*spec, (6, 4)), tmp_path / "timelines.pdf")
    exported = []
    for i in range(window.right_tabs.count()):
        window.right_tabs.setCurrentIndex(i)
        qtbot.waitUntil(lambda: _idle(window), timeout=60_000)
        qtbot.wait(50)
        spec = window.summary_spec()
        if spec is not None:
            figures.save(figures.render(*spec, (6, 4)), tmp_path / f"{spec[0]}.png", dpi=60)
            exported.append(spec[0])
    assert {"distributions", "scatter", "profiles", "composite", "diurnal", "anisotropy", "windrose"} <= set(exported)
    assert (tmp_path / "timelines.pdf").stat().st_size > 1000


def test_a_selection_splits_each_histogram_into_its_own_density(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.full_period.setChecked(True)
    dist = window.dists[0]
    slots = np.unique(window.datas[0].curves[FULL_BOOMS[0]].slots)
    window.set_selection(slots[:6], "replace")
    assert "brushed selection" in dist.title.text()
    hists = dist.figure_spec()["hists"]
    selected = [h for h in hists if h.get("filled")]
    rest = [h for h in hists if not h.get("filled")]
    assert len(selected) == len(rest) == len(FULL_BOOMS)
    for h in hists:  # each part is a density of its own slots, not a share of the whole
        widths = np.diff(h["edges"])
        assert (h["density"] * widths).sum() == pytest.approx(1.0, abs=0.05)
    window.set_selection(np.empty(0, dtype=np.int64), "replace")
    assert not any(h.get("filled") for h in dist.figure_spec()["hists"])


def test_a_selection_gets_its_own_statistics_table(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|ws|mean", timeout=60_000)
    window.full_period.setChecked(True)
    dist = window.dists[0]
    dist.fit_box.setCurrentIndex(dist.fit_box.findData("normal"))
    qtbot.waitUntil(lambda: bool(dist.fits) and _idle(window), timeout=60_000)
    assert dist.sel_table.isHidden() and dist.table.rowCount() == len(FULL_BOOMS)
    slots = np.unique(window.datas[0].curves[FULL_BOOMS[0]].slots)
    window.set_selection(slots[:6], "replace")
    qtbot.waitUntil(lambda: {p for p, _ in dist.fits} == {"rest", "selected"} and _idle(window), timeout=60_000)
    assert not dist.sel_table.isHidden() and dist.captions[1].text().startswith("brushed selection")

    def column(table, name):
        j = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())].index(name)
        return [float(table.item(r, j).text()) for r in range(table.rowCount())]

    selected_n, rest_n = column(dist.sel_table, "N"), column(dist.table, "N")
    whole = [np.isfinite(window.datas[0].curves[b].y).sum() for b in FULL_BOOMS]
    assert all(0 < n <= 6 for n in selected_n) and [a + b for a, b in zip(selected_n, rest_n)] == whole
    assert "μ" in [dist.sel_table.horizontalHeaderItem(c).text() for c in range(dist.sel_table.columnCount())]
    window.set_selection(np.empty(0, dtype=np.int64), "replace")
    assert dist.sel_table.isHidden() and not dist.captions[0].isVisible()


def test_scatter_log_axes_show_the_values_themselves(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    qtbot.waitUntil(lambda: scatter.x.size > 10, timeout=60_000)
    axis = scatter.axes["bottom"]
    assert not axis.log10
    scatter.log_x.setChecked(True)
    assert axis.log10 and "(log scale)" in scatter.plot.getPlotItem().getAxis("bottom").labelText
    assert axis.tickStrings([0.0, 1.0, 2.0], 1.0, 1.0) == ["1", "10", "100"]  # data is log10(x); ticks read x
    scatter.log_x.setChecked(False)
    assert not axis.log10 and "(log scale)" not in scatter.plot.getPlotItem().getAxis("bottom").labelText
    scatter.fit_bins.setValue(10)
    scatter.fit_bin_log.setChecked(True)
    assert scatter._options().bin_log == scatter.fit_bin_log.isEnabled()


def test_distributions_show_one_side_of_zero_or_absolute_values(window, qtbot):
    window.full_period.setChecked(True)
    dist = window.dists[0]
    for key, variant in (("boom_final|wvpts|cov", "mrd"), ("boom_final|vw|cov", "mrd"), ("boom_final|w|mean", "none"),
                         ("boom_final|uv|cov", "mrd")):  # one whose values cross 0
        if key not in {q.key for q in window.catalog}:
            continue
        window.panels[0].select(key, variant, tuple(FULL_BOOMS))
        qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == key, timeout=60_000)
        raw = np.concatenate(list(dist.values_by_member().values()))
        raw = raw[np.isfinite(raw)]
        if (raw < 0).any() and (raw > 0).any():
            break
    assert (raw < 0).any() and (raw > 0).any() and not dist.log_bins.isEnabled()  # both signs: no log bins
    dist.side.setCurrentIndex(dist.side.findData("negative"))
    neg = np.concatenate(list(dist.values_by_member().values()))
    assert neg.size == (raw < 0).sum() and (neg > 0).all()  # the negative side, as −x
    assert dist.log_bins.isEnabled() and "values &lt; 0 only" in dist.title.text()  # escaped: the title is HTML
    dist.log_bins.setChecked(True)
    axis = dist.plot.getPlotItem().getAxis("bottom")
    assert axis.log10 and "−x" in axis.labelText and "(log scale)" in axis.labelText
    dist.side.setCurrentIndex(dist.side.findData("abs"))
    absolute = np.concatenate(list(dist.values_by_member().values()))
    assert np.isfinite(absolute).sum() == raw.size and (absolute[np.isfinite(absolute)] >= 0).all()
    dist.side.setCurrentIndex(dist.side.findData("all"))
    assert not dist.log_bins.isEnabled() and not axis.log10


def test_scatter_log_axes_show_either_side_of_zero(window, qtbot):
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    scatter.slots = np.arange(5)
    scatter.x = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    scatter.y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scatter.z = np.full(5, np.nan)
    assert scatter.side("x") == 0 and scatter._shown()[0].size == 5  # every value by default
    scatter.log_x.setChecked(True)
    assert scatter.sides["x"].currentData() == 1  # a log axis shows one side
    slots, x, _ = scatter._shown()
    assert list(slots) == [3, 4] and list(x) == [1.0, 3.0]
    scatter.sides["x"].setCurrentIndex(scatter.sides["x"].findData(-1))
    slots, x, _ = scatter._shown()
    assert list(slots) == [0, 1] and list(x) == [2.0, 1.0]  # the negative side, as −x
    assert scatter.sign("x") == -1 and scatter.sign("y") == 1


def test_scatter_shows_one_side_of_zero_on_linear_axes_too(window, qtbot):
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    scatter.slots = np.arange(5)
    scatter.x = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    scatter.y = np.array([-1.0, 2.0, 3.0, 4.0, -5.0])
    scatter.z = np.full(5, np.nan)
    scatter.sides["x"].setCurrentIndex(scatter.sides["x"].findData(-1))
    slots, x, _ = scatter._shown()
    assert list(slots) == [0, 1] and list(x) == [-2.0, -1.0] and scatter.sign("x") == 1  # as they are
    assert "(< 0 only)" in scatter.axis_title(0)
    assert "&lt; 0 only" in scatter.plot.getPlotItem().getAxis("bottom").labelText  # escaped: labels are HTML
    scatter.sides["x"].setCurrentIndex(scatter.sides["x"].findData(1))
    scatter.sides["y"].setCurrentIndex(scatter.sides["y"].findData(1))
    assert list(scatter._shown()[0]) == [3]  # x > 0 and y > 0
    assert scatter._inputs()[-4:-2] == (1, 1)  # a fit made before is stale once a side changes
    scatter.sides["y"].setCurrentIndex(scatter.sides["y"].findData(0))
    assert list(scatter._shown()[0]) == [3, 4]
    assert not scatter.log_x.isChecked() and scatter.sides["x"].model().item(0).isEnabled()


def _wind_and_ustar(window, qtbot):
    window.panels[0].select("boom_final|ws|mean", "none", tuple(FULL_BOOMS))
    window.panels[1].select("boom_final|ustar|", "mrd", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[1] is not None
                    and window.datas[1].quantity.key == "boom_final|ustar|", timeout=60_000)
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.scatter)
    return window.scatter


def _choose_z(window, qtbot, key):
    scatter = window.scatter
    scatter.z_quantity.setCurrentIndex(scatter.z_quantity.findData(key))
    qtbot.waitUntil(lambda: scatter.z_data is not None and _idle(window), timeout=60_000)


class _Point:
    def __init__(self, i):
        self.i = i

    def data(self):
        return self.i


def test_scatter_colors_points_by_a_third_quantity_and_keeps_it_in_a_view(window, qtbot):
    scatter = _wind_and_ustar(window, qtbot)
    assert not scatter.has_z() and not scatter.color_by.model().item(2).isEnabled()  # no z chosen yet
    _choose_z(window, qtbot, "boom_final|sigma_w|")
    assert scatter.member[2].currentData() == scatter.member[0].currentData()  # x's boom by default
    assert scatter.has_z() and scatter.z.size == scatter.x.size
    scatter.color_by.setCurrentIndex(scatter.color_by.findData("z"))
    lo, hi, name = scatter.drawn_z
    assert name == "viridis" and lo < hi and scatter.zbar.isVisible()
    assert scatter.zbar.getPlotItem().getAxis("right").labelText == scatter.axis_title(2)
    spec = scatter.figure_spec()[1]
    assert spec["z"]["cmap"] == "viridis" and any("c" in g for g in spec["groups"])
    figures.render("scatter", spec, (6, 4))  # with a color bar
    scatter.log_z.setChecked(True)
    assert scatter.zbar_axis.log10 and np.nanmin(scatter.shown_z()) > 0
    scatter.z_map.setCurrentIndex(scatter.z_map.findData("diverging"))
    assert scatter.z_colormap() == "viridis"  # diverging only about the 0 of a linear z
    scatter.log_z.setChecked(False)
    assert scatter.z_colormap() == "RdBu_r" and scatter.drawn_z[0] == -scatter.drawn_z[1]

    state = view_state.save(scatter, scatter.STATE_FIELDS)
    member = scatter.member[2].currentData()
    scatter.z_quantity.setCurrentIndex(0)
    assert scatter.z_data is None and not scatter.has_z()
    assert view_state.restore(scatter, state, scatter.STATE_FIELDS) == []  # z's boom waits for its data
    qtbot.waitUntil(lambda: scatter.z_data is not None and _idle(window), timeout=60_000)
    assert scatter.z_quantity.currentData() == "boom_final|sigma_w|" and scatter.member[2].currentData() == member
    assert scatter.drawn_z is not None


def test_scatter_z_can_be_the_boom_height(window, qtbot):
    scatter = _wind_and_ustar(window, qtbot)
    single = scatter.x.size
    scatter.source[2].setCurrentIndex(scatter.source[2].findData(3))
    booms = set(window.datas[0].curves) & set(window.datas[1].curves)
    stacked = set(int(b) for b in np.unique(scatter.point_member))
    assert stacked <= booms and len(stacked) > 1 and scatter.x.size > single
    assert np.array_equal(scatter.z, [HEIGHTS[int(b)] for b in scatter.point_member])
    assert not scatter.member[0].isEnabled() and "each boom" in scatter.axis_title(0)
    assert scatter.axis_title(2) == "height [m]"
    caught = []
    scatter.pointClicked.connect(lambda slot, member: caught.append((slot, member)))
    scatter._on_click(None, [_Point(7)], None)
    assert caught == [(int(scatter._drawn_slots[7]), int(scatter._drawn_members[7]))]  # the point's own boom
    scatter.source[2].setCurrentIndex(0)
    assert scatter.point_member is None and scatter.member[0].isEnabled() and scatter.x.size == single


def test_scatter_marks_the_fits_x_percentiles(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    qtbot.waitUntil(lambda: scatter.x.size > 10, timeout=60_000)

    def marks():
        return sorted(i.value() for i in scatter.plot.getPlotItem().items if isinstance(i, pg.InfiniteLine))
    assert marks() == []
    scatter.pct[0].setValue(10.0)
    scatter.pct[1].setValue(90.0)
    scatter.pct_show.setChecked(True)
    assert marks() == pytest.approx(sorted(np.percentile(scatter._shown()[1], [10, 90])))
    assert [v["label"] for v in scatter.figure_spec()[1]["vlines"]] == ["10 %", "90 %"]
    if scatter.log_x.isEnabled():
        scatter.log_x.setChecked(True)
        assert marks() == pytest.approx(sorted(np.log10(np.percentile(scatter._shown()[1], [10, 90]))))
    scatter.pct_show.setChecked(False)
    assert marks() == []


def test_the_3d_view_follows_scatter(window, qtbot):
    pytest.importorskip("OpenGL")
    from ttu_tower.viewer.ui.scatter3d import ticks
    assert ticks(-0.25, 1.1, False)[1] == ["-0.2", "0", "0.2", "0.4", "0.6", "0.8", "1"]
    assert ticks(-0.3, 2.2, True)[1] == ["1", "10", "100"]
    scatter = _wind_and_ustar(window, qtbot)
    _choose_z(window, qtbot, "boom_final|sigma_w|")
    scatter.open_3d()
    view = scatter.window3d
    assert view.isVisible() and scatter.shows_3d()
    have = int(np.isfinite(scatter.shown_z()).sum())
    assert view.points.pos.shape == (have, 3)
    inside = (view.points.pos >= 0) & (view.points.pos <= 1)
    assert inside.mean() > 0.9  # the box spans the 1st-99th percentiles
    texts = [label.text for label in view.labels]
    assert any(scatter.axis_title(2) in t for t in texts)
    scatter.log_x.setChecked(True)
    assert any("(log scale)" in label.text for label in view.labels)
    window.set_selection(scatter.slots[:20], "replace")
    assert view.chosen.visible() and view.chosen.pos.shape[0] > 0
    scatter.source[2].setCurrentIndex(scatter.source[2].findData(3))
    assert any(label.text.startswith("height [m]") for label in view.labels)
    view.color_by.setCurrentIndex(view.color_by.findData("stability"))
    view.look("x–y")
    view.close()
    assert not scatter.shows_3d()


def test_the_3d_wind_rose_stacks_every_booms_rose(window, qtbot):
    pytest.importorskip("OpenGL")
    from ttu_tower.viewer.ui import windrose3d
    assert list(windrose3d.layer_heights([1.0, 10.0, 100.0], "log height")) == pytest.approx([0.0, 0.45, 0.9])
    assert list(windrose3d.layer_heights([1.0, 10.0, 100.0], "even")) == pytest.approx([0.0, 0.45, 0.9])
    assert windrose3d.layer_heights([0.0, 50.0, 200.0], "height")[1] == pytest.approx(0.9 * 50 / 200)
    table = np.zeros((4, 2))
    table[0] = [10.0, 5.0]
    verts, faces, colors = windrose3d.wedges(table, 20.0, 0.3)
    assert len(faces) == len(colors) == 2 * 2 * 7 and np.allclose(verts[:, 2], 0.3)
    assert np.hypot(verts[:, 0], verts[:, 1]).max() == pytest.approx(0.75)  # 15 % of a 20 % rim

    window.right_tabs.setCurrentWidget(window.windrose)
    rose = window.windrose
    window.full_period.setChecked(True)
    qtbot.waitUntil(lambda: rose.tower is not None and _idle(window), timeout=60_000)
    rose.open_3d()
    view = rose.window3d
    qtbot.waitUntil(lambda: view.winds is not None and _idle(window), timeout=60_000)
    layers = view.layers()
    assert [label.split()[0] for label, _, _ in layers] == [f"b{b}" for b in sorted(FULL_BOOMS)]  # no mesonet here
    assert not view.with_mesonet.isEnabled() and view.mesh.visible()
    texts = [label.text for label in view.labels]
    assert any(t.startswith(f"b{FULL_BOOMS[0]} (") for t in texts) and "N" in texts
    view.common.setChecked(True)
    sizes = {w.ws.size for _, _, w in view.layers()}
    assert len(sizes) == 1  # the same slots in every layer
    rose.filtering.setCurrentIndex(0)  # the tab's filtering reloads every boom
    qtbot.waitUntil(lambda: _idle(window) and view.loaded_key[1] is True, timeout=60_000)
    view.spacing.setCurrentIndex(view.spacing.findData("height"))
    view.look("from above")
    view.close()
    assert not rose.shows_3d()


def test_the_hodograph_tab_follows_the_group_selection_and_interval(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.hodograph)
    hodo = window.hodograph
    qtbot.waitUntil(lambda: hodo.uvs is not None and _idle(window), timeout=60_000)
    assert set(hodo.uvs) == set(FULL_BOOMS)  # no mesonet in the fixture
    label, _, rows = hodo.drawn[0]
    assert label == "all slots" and list(rows["member"]) == sorted(FULL_BOOMS, key=lambda b: HEIGHTS[b])
    assert rows["n"].nunique() == 1 and hodo.table.rowCount() == len(FULL_BOOMS)  # the same slots at every boom
    whole = int(rows["n"].iloc[0])
    hodo.common.setChecked(False)
    assert hodo.drawn[0][2]["n"].max() >= whole
    hodo.common.setChecked(True)

    chosen = hodo.uvs[FULL_BOOMS[0]].slots[:5]
    window.set_selection(chosen, "replace")
    hodo.group.setCurrentIndex(hodo.group.findData("selected"))
    assert hodo.drawn[0][2]["n"].max() <= 5 and hodo.each_slot.isEnabled()
    hodo.each_slot.setChecked(True)
    if hodo.stability is not None:
        hodo.group.setCurrentIndex(hodo.group.findData("classes"))
        assert len(hodo.drawn) == len(hodo.stability[1]) and not hodo.each_slot.isEnabled()
    figures.render(*hodo.figure_spec(), (5, 5))
    hodo.group.setCurrentIndex(0)
    window.full_period.setChecked(False)
    k = int(chosen[0])
    window.set_x_range(slot_to_unix(k), slot_to_unix(k + 2))
    window._refresh_distributions()
    assert hodo.drawn[0][2]["n"].max() <= 2


def test_the_inspector_hodograph_shows_the_slot_and_the_hour_around(window, qtbot):
    inspector = _inspect(window, qtbot, "hodograph")
    tab = inspector.hodograph
    qtbot.waitUntil(lambda: tab.uvs is not None and _idle(window), timeout=60_000)
    have = [b for b in FULL_BOOMS if b in tab.uvs and SPIKY_SLOT in tab.uvs[b].slots]
    assert tab._loaded_slot == SPIKY_SLOT and have and tab.table.rowCount() == len(have)
    missing = sorted(set(FULL_BOOMS) - set(have))  # e.g. a boom whose wind tertiary filtered
    assert ("no wind at" in tab.info.text()) == bool(missing) and "slots before" in tab.info.text()
    curves = len(tab.plot.getPlotItem().listDataItems())
    tab.around.setChecked(False)
    assert len(tab.plot.getPlotItem().listDataItems()) < curves
    inspector.step(1)
    qtbot.waitUntil(lambda: tab._loaded_slot == SPIKY_SLOT + 1 and _idle(window), timeout=60_000)


def test_scatter_overlays_a_typed_piecewise_function_and_takes_the_fit(window, qtbot):
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    qtbot.waitUntil(lambda: scatter.x.size > 10, timeout=60_000)
    overlay = scatter.overlay

    def drawn():
        return next((line for line in scatter.drawn_lines if line["label"] == "overlay"), None)
    assert not scatter.overlay_on.isChecked() and drawn() is None
    scatter.overlay_on.setChecked(True)
    overlay.pieces[0].formula.setText("2*x")
    overlay.pieces[0].formula.editingFinished.emit()
    line = drawn()
    assert line["y"][np.isfinite(line["y"])] == pytest.approx(2 * line["x"][np.isfinite(line["y"])])
    assert overlay.status.text().startswith("RMSE") and f"{scatter.x.size} points" in overlay.status.text()
    mid = float(np.median(scatter.x))
    overlay.pieces[0].hi.setText(f"{mid:.6g}")
    overlay.add_piece("1", f"{mid:.6g}", "")
    overlay.changed.emit()
    line = drawn()
    assert np.isnan(line["y"]).sum() == 1  # one break, between the two pieces
    assert (line["y"][line["x"] > mid * 1.01] == 1.0).all()
    overlay.pieces[1].formula.setText("3x")
    overlay.changed.emit()
    assert "piece 2" in overlay.status.text() and overlay.pieces[1].formula.styleSheet()

    state = view_state.save(scatter, scatter.STATE_FIELDS)
    assert state["overlay"][0][0] == "2*x" and state["overlay_on"] is True
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear"))
    assert not overlay.from_fit.isEnabled()
    scatter.run_fit()
    qtbot.waitUntil(lambda: scatter.fit_result is not None and not scatter._fitting, timeout=60_000)
    overlay.from_fit.click()
    assert len(overlay.pieces) == 1 and "*x" in overlay.pieces[0].formula.text()
    line, fit = drawn(), scatter.fit_result
    finite = np.isfinite(line["y"])
    assert line["y"][finite] == pytest.approx(fit.func(line["x"][finite]), rel=1e-4)
    view_state.restore(scatter, state, scatter.STATE_FIELDS)
    assert [p.formula.text() for p in overlay.pieces] == ["2*x", "3x"]


def test_scatter_can_show_and_fit_one_stability_class(window, qtbot):
    from ttu_tower.viewer.timeline import Curve, in_class
    scatter = _wind_and_ustar(window, qtbot)
    names = ["unstable", "neutral", "stable"]  # the fixture's booms make no Ri_b pair
    stability = (Curve(slots=scatter.slots, y=(scatter.slots % 3).astype(float)), names)
    scatter.set_sources(window.datas, None, stability)
    assert [scatter.only_class.itemData(i) for i in range(scatter.only_class.count())] == [None, *names]
    counts = {n: int(in_class(scatter.stability, scatter.slots, n).sum()) for n in names}
    name = max(counts, key=counts.get)
    assert 2 < counts[name] < scatter.slots.size
    scatter.only_class.setCurrentIndex(scatter.only_class.findData(name))
    shown = scatter._shown()[0]
    assert sorted(shown) == sorted(scatter.slots[in_class(scatter.stability, scatter.slots, name)])
    assert scatter.stats.text().startswith(f"{name} only · N {counts[name]}")
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear"))
    scatter.run_fit()
    qtbot.waitUntil(lambda: scatter.fit_result is not None and not scatter._fitting, timeout=60_000)
    assert scatter.fit_result.n == counts[name]  # only that class's points
    state = view_state.save(scatter, scatter.STATE_FIELDS)
    scatter.only_class.setCurrentIndex(0)
    assert scatter._shown()[0].size == scatter.slots.size and "earlier" in scatter.fit_status.text()
    view_state.restore(scatter, state, scatter.STATE_FIELDS)
    assert scatter.class_shown() == name


def test_the_overlay_can_use_the_booms_heights(window, qtbot):
    scatter = _wind_and_ustar(window, qtbot)
    overlay = scatter.overlay
    scatter.overlay_on.setChecked(True)
    b = scatter.member[0].currentData()
    overlay.set_pieces([("x*zA/zB", "", "")])
    assert scatter.overlay_heights() == {"zA": HEIGHTS[b], "zB": HEIGHTS[scatter.member[1].currentData()]}
    line = next(line for line in scatter.drawn_lines if line["label"] == "overlay")
    finite = np.isfinite(line["y"])
    ratio = HEIGHTS[b] / HEIGHTS[scatter.member[1].currentData()]
    assert line["y"][finite] == pytest.approx(line["x"][finite] * ratio) and overlay.status.text().startswith("RMSE")

    scatter.source[2].setCurrentIndex(scatter.source[2].findData(3))  # z: each boom at its height
    lines = [line for line in scatter.drawn_lines if line["label"] in ("overlay, each boom", None)]
    booms = sorted(set(int(m) for m in scatter.point_member), key=HEIGHTS.get)
    assert len(lines) == len(booms) and [line["label"] for line in lines][0] == "overlay, each boom"
    first = lines[0]
    ok = np.isfinite(first["y"])
    assert first["y"][ok] == pytest.approx(first["x"][ok])  # zA = zB: each boom's own height
    assert overlay.status.text().startswith("RMSE")
    scatter.source[2].setCurrentIndex(scatter.source[2].findData(2))

    scatter.source[0].setCurrentIndex(scatter.source[0].findData(1))  # x and y both from panel B: no zA
    overlay.set_pieces([("zA*x", "", "")])
    assert "zA: Scatter shows no single boom from panel A" in overlay.status.text()
    assert not any(line["label"] == "overlay" for line in scatter.drawn_lines)
    overlay.set_pieces([("zB*x", "", ""), ("3x", "", "")])  # a bad piece: the good one still drawn
    assert "piece 2" in overlay.status.text() and any(line["label"] == "overlay" for line in scatter.drawn_lines)


def test_an_automatic_fit_follows_the_quantities_and_booms(window, qtbot):
    scatter = _wind_and_ustar(window, qtbot)
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear"))
    qtbot.wait(500)
    assert scatter.fit_result is None  # nothing until asked, or "auto"
    scatter.fit_auto.setChecked(True)
    qtbot.waitUntil(lambda: scatter.fit_result is not None and not scatter._fitting, timeout=60_000)
    first = scatter.fit_result
    assert scatter.fit_inputs == scatter._inputs()
    other = next(b for b in FULL_BOOMS if b != scatter.member[0].currentData())
    scatter.member[0].setCurrentIndex(scatter.member[0].findData(other))  # another boom on x
    qtbot.waitUntil(lambda: scatter.fit_result is not first and not scatter._fitting, timeout=60_000)
    assert scatter.fit_inputs == scatter._inputs() and "earlier" not in scatter.fit_status.text()
    second = scatter.fit_result
    window.panels[1].select("boom_final|sigma_w|", "mrd", tuple(FULL_BOOMS))  # another quantity on y
    qtbot.waitUntil(lambda: _idle(window) and scatter.fit_result is not second and not scatter._fitting,
                    timeout=60_000)
    assert scatter.fit_inputs == scatter._inputs()
    scatter.fit_auto.setChecked(False)
    third = scatter.fit_result
    scatter.member[0].setCurrentIndex(0 if scatter.member[0].currentIndex() else 1)
    qtbot.wait(600)
    assert scatter.fit_result is third and "press Fit" in scatter.fit_status.text()  # off: kept, marked earlier


def test_an_automatic_fit_that_fails_is_not_retried_until_something_changes(window, qtbot):
    scatter = _wind_and_ustar(window, qtbot)
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("power law, fixed exponent"))
    scatter.fit_settings["b"].setText("0.5")
    scatter.sides["x"].setCurrentIndex(scatter.sides["x"].findData(-1))  # no x < 0: nothing to fit
    calls = []
    real = scatter.run_fit
    scatter.run_fit = lambda **kw: (calls.append(1), real(**kw))
    scatter.fit_auto.setChecked(True)
    qtbot.wait(1500)
    assert len(calls) <= 1 and scatter.fit_result is None


def test_a_fit_can_take_only_the_selected_points_or_all_but_them(window, qtbot):
    scatter = _wind_and_ustar(window, qtbot)
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear"))
    scatter.fit_subset.setCurrentIndex(scatter.fit_subset.findData("selected"))
    scatter.run_fit()
    qtbot.waitUntil(lambda: not scatter._fitting, timeout=60_000)
    assert scatter.fit_result is None and "nothing is selected" in scatter.fit_status.text()
    outliers = scatter.slots[np.argsort(scatter.y)[-5:]]  # as if lassoed
    window.set_selection(outliers, "replace")
    scatter.run_fit()
    qtbot.waitUntil(lambda: scatter.fit_result is not None and not scatter._fitting, timeout=60_000)
    assert scatter.fit_result.n == 5
    scatter.fit_subset.setCurrentIndex(scatter.fit_subset.findData("unselected"))
    scatter.run_fit()
    qtbot.waitUntil(lambda: scatter.fit_result.n != 5 and not scatter._fitting, timeout=60_000)
    assert scatter.fit_result.n == scatter._shown()[0].size - 5
    window.set_selection(outliers[:2], "replace")  # another selection: the fit is earlier
    assert "earlier" in scatter.fit_status.text()


def test_several_fits_are_drawn_together_and_kept_in_a_view(window, qtbot):
    scatter = _wind_and_ustar(window, qtbot)
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear"))
    scatter.add_fit()  # a copy, then changed
    assert len(scatter.fits) == 2 and scatter.current == 1 and scatter.fits[1].color != scatter.fits[0].color
    scatter.fit_name.setCurrentIndex(scatter.fit_name.findData("linear through origin"))
    scatter.fit_bins.setValue(8)
    assert scatter.fits[0].spec.name == "linear" and scatter.fits[1].spec.bins == 8
    scatter.run_fit()  # every fit
    qtbot.waitUntil(lambda: all(e.result is not None for e in scatter.fits) and not scatter._fitting,
                    timeout=60_000)
    labels = [line["label"] for line in scatter.drawn_lines]
    assert "1. linear" in labels and "2. linear through origin, 8 mean bins" in labels
    assert scatter.fit_list.count() == 2 and "RMSE" in scatter.fit_list.item(1).text()
    assert scatter.fits[1].result.n <= 8
    scatter.fit_list.setCurrentRow(0)  # edit the first
    assert scatter.current == 0 and scatter.fit_name.currentData() == "linear" and scatter.fit_bins.value() == 0
    scatter.fit_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert not scatter.fits[1].shown and not any(line["label"].startswith("2.") for line in scatter.drawn_lines)

    state = view_state.save(scatter, scatter.STATE_FIELDS)
    scatter.remove_current_fit()
    assert len(scatter.fits) == 1 and scatter.fits[0].spec.name == "linear through origin"
    view_state.restore(scatter, state, scatter.STATE_FIELDS)
    assert [e.spec.name for e in scatter.fits] == ["linear", "linear through origin"]
    assert [e.shown for e in scatter.fits] == [True, False] and scatter.fits[1].spec.bins == 8

    scatter.fit_auto.setChecked(True)  # every fit refits by itself
    qtbot.waitUntil(lambda: all(e.result is not None for e in scatter.fits) and not scatter._fitting,
                    timeout=60_000)
    before = [e.result for e in scatter.fits]
    other = next(b for b in FULL_BOOMS if b != scatter.member[0].currentData())
    scatter.member[0].setCurrentIndex(scatter.member[0].findData(other))
    qtbot.waitUntil(lambda: all(e.result is not r for e, r in zip(scatter.fits, before)) and not scatter._fitting,
                    timeout=60_000)


def test_stability_schemes_are_saved_and_loaded_by_name(window, qtbot, tmp_path):
    from ttu_tower.viewer import stability as criteria
    panel = window.stability_panel
    panel.schemes = criteria.SchemeStore.open(tmp_path / "schemes.json")
    panel._fill_schemes()
    assert panel.scheme.itemText(0) == "(no saved schemes)" and not panel.delete_scheme_button.isEnabled()

    rib = criteria.StabilitySpec("rib", (1, 2), "mrd", (-0.5, -0.1, 0.1, 0.5))
    panel.set_spec(rib)
    panel.save_scheme("near surface")
    zeta = criteria.StabilitySpec("zeta", (2, 2), "mrd", (-1.0, -0.1, 0.1, 1.0))
    panel.set_spec(zeta)
    panel.save_scheme("zeta wide")
    assert panel.scheme.currentData() == "zeta wide" and panel.delete_scheme_button.isEnabled()
    panel.edges[3].setValue(2.0)  # an edit: no saved scheme is what's set now
    assert panel.scheme.currentData() is None and panel.scheme.currentText() == "(unsaved changes)"

    panel.load_scheme("near surface")  # every view follows
    qtbot.waitUntil(lambda: _idle(window) and window.stability_spec == rib, timeout=60_000)
    assert panel.scheme.currentData() == "near surface" and panel.spec() == rib
    again = criteria.SchemeStore.open(tmp_path / "schemes.json")  # kept, for every run
    assert again.items == {"near surface": rib, "zeta wide": zeta}
    with pytest.raises(ValueError, match="name"):
        panel.save_scheme("  ")
    panel.delete_scheme()
    assert set(criteria.SchemeStore.open(tmp_path / "schemes.json").items) == {"zeta wide"}
    assert panel.scheme.currentData() is None
    panel.reset.click()
    qtbot.waitUntil(lambda: _idle(window) and window.stability_spec is None, timeout=60_000)


def _hover_at(plot_widget, x, y):
    """Move the cursor to view point (x, y) of a PlotWidget, as the mouse would."""
    from PySide6.QtCore import QPointF
    vb = plot_widget.getPlotItem().getViewBox()
    plot_widget.scene().sigMouseMoved.emit(vb.mapViewToScene(QPointF(x, y)))


def test_hovering_a_summary_plot_reads_the_cursor_in_the_bottom_bar(window, qtbot):
    from ttu_tower.viewer.ui.hover_readout import axis_value
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(window.dist_tab)
    dist = window.dists[0]
    qtbot.wait(300)
    (x0, x1), (y0, y1) = dist.plot.getPlotItem().getViewBox().viewRange()
    _hover_at(dist.plot, (x0 + x1) / 2, (y0 + y1) / 2)
    text = window.hover.text()
    assert text.startswith("Distributions:") and f"= {(x0 + x1) / 2:.4g}" in text and "density" in text

    window.right_tabs.setCurrentWidget(window.scatter)
    scatter = window.scatter
    qtbot.waitUntil(lambda: scatter.x.size > 10, timeout=60_000)
    qtbot.wait(300)
    (x0, x1), (y0, y1) = scatter.vb.viewRange()
    _hover_at(scatter.plot, (x0 + x1) / 2, (y0 + y1) / 2)
    text = window.hover.text()
    assert text.startswith("Scatter:") and f"= {(x0 + x1) / 2:.4g}" in text and f"= {(y0 + y1) / 2:.4g}" in text
    assert scatter.axis_title(0) in text
    if scatter.log_x.isEnabled():
        scatter.log_x.setChecked(True)
        qtbot.wait(200)
        (x0, x1), (y0, y1) = scatter.vb.viewRange()
        _hover_at(scatter.plot, (x0 + x1) / 2, (y0 + y1) / 2)
        assert f"= {10 ** ((x0 + x1) / 2):.4g}" in window.hover.text()  # the value, not its log
        assert "(log scale)" not in window.hover.text()

    axis = pg.AxisItem("bottom")
    axis.setTicks([[(0.0, "Jan"), (1.0, "Feb"), (2.0, "Mar")]])
    assert axis_value(axis, 1.2) == "Feb" and axis_value(axis, 7.0) == "7"  # far from any label: the number
    sparse = pg.AxisItem("left")
    sparse.setTicks([[(h + 0.5, f"{h:02d}") for h in range(0, 24, 3)]])
    assert axis_value(sparse, 3.3) == "03" and axis_value(sparse, 4.6) == "4.6"  # between labels: the hour itself


def test_stability_boundaries_take_five_decimals(window, tmp_path):
    from ttu_tower.viewer import stability as criteria
    panel = window.stability_panel
    panel.schemes = criteria.SchemeStore.open(tmp_path / "schemes.json")
    fine = criteria.StabilitySpec("rib", (1, 2), "mrd", (-0.01234, -0.00125, 0.00125, 0.04321))
    panel.set_spec(fine)
    assert panel.spec() == fine and panel.edges[1].text().endswith("0.00125")
    panel.edges[2].lineEdit().selectAll()
    from PySide6.QtTest import QTest
    QTest.keyClicks(panel.edges[2].lineEdit(), "0.00375")
    panel.edges[2].interpretText()
    assert panel.spec().edges[2] == 0.00375
    panel.save_scheme("fine")
    assert criteria.SchemeStore.open(tmp_path / "schemes.json").items["fine"].edges[2] == 0.00375


def test_scatter_offers_all_unstable_and_all_stable(window, qtbot):
    from ttu_tower.viewer.timeline import Curve, in_class
    scatter = _wind_and_ustar(window, qtbot)
    names = ["strongly unstable", "unstable", "neutral", "stable", "strongly stable"]
    stability = (Curve(slots=scatter.slots, y=(scatter.slots % 5).astype(float)), names)
    scatter.set_sources(window.datas, None, stability)
    items = [scatter.only_class.itemData(i) for i in range(scatter.only_class.count())]
    assert items == [None, *names, "all unstable", "all stable"]
    scatter.only_class.setCurrentIndex(scatter.only_class.findData("all stable"))
    both = in_class(stability, scatter.slots, "stable") | in_class(stability, scatter.slots, "strongly stable")
    assert sorted(scatter._shown()[0]) == sorted(scatter.slots[both])
    assert scatter.stats.text().startswith("all stable only")
