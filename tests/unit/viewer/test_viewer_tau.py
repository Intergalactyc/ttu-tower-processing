import dataclasses

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from ttu_tower.io.store import read_table  # noqa: E402
from ttu_tower.viewer import taustats  # noqa: E402
from ttu_tower.viewer.fragments import FragmentIndex  # noqa: E402
from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402

from viewer_fixtures import FULL_BOOMS  # noqa: E402

pytestmark = pytest.mark.slow


def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


@pytest.fixture
def data(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    return run, taustats.load(run, FragmentIndex(run.run_dir))


def test_tau_rows_match_the_stored_tables(data):
    run, d = data
    sel = read_table(run.run_dir / "secondary" / "data" / "tau_selected")
    sel = sel[sel["variant"].astype(str).isin(taustats.VARIANTS)]
    assert d.slot.size == len(sel) > 0
    cats = taustats.categories(d, "selected")
    assert taustats.shares(cats, "selected").sum() == pytest.approx(100.0)
    rung = cats < len(taustats.RUNGS)
    assert np.array_equal(taustats.tau_values(cats)[rung], d.sel_tau[rung])


def test_reselecting_with_the_runs_rules_reproduces_every_stored_selection(data):
    run, d = data
    tau, source, status = taustats.reselect(d, run.cfg.secondary.selection)
    assert np.array_equal(tau, d.sel_tau, equal_nan=True)
    assert (source == d.sel_source).all() and (status == d.sel_status).all()
    other = dataclasses.replace(run.cfg.secondary.selection, priority=("momentum", "heat"))
    _, source2, _ = taustats.reselect(d, other)
    momentum_first = (d.status["momentum"] == taustats.STATUSES.index("found"))
    assert (source2[momentum_first] == taustats.SOURCES.index("momentum")).all()


def test_the_joint_distribution_and_agreement_add_up(data):
    _, d = data
    heat, mom = taustats.categories(d, "heat"), taustats.categories(d, "momentum")
    counts = taustats.joint(heat, mom)
    assert counts.sum() == d.slot.size
    rungs = len(taustats.RUNGS)
    a = taustats.agreement(heat, mom)
    assert a["N both"] == counts[:rungs, :rungs].sum()
    assert a["% equal"] + a["% heat longer"] + a["% momentum longer"] == pytest.approx(100.0)
    assert a["% equal"] == pytest.approx(100.0 * np.trace(counts[:rungs, :rungs]) / a["N both"])


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
    w.runner.wait(30_000)


def test_tau_tab_draws_distributions_heat_vs_momentum_and_a_whatif(window, qtbot):
    view = window.tau_view
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(view)
    qtbot.waitUntil(lambda: view.data is not None and _idle(window), timeout=60_000)
    assert len(view.dist_plot.getPlotItem().listDataItems()) == len(FULL_BOOMS)
    assert view.dist_table.rowCount() == len(FULL_BOOMS)
    view.which.setCurrentIndex(view.which.findData("heat"))
    assert "% found" in [view.dist_table.horizontalHeaderItem(j).text() for j in range(view.dist_table.columnCount())]

    view.tabs.setCurrentIndex(1)
    n = len(taustats.labels("heat"))
    assert view.joint_table.rowCount() == view.joint_table.columnCount() == n
    keep, heat, mom = view._joint_rows
    i, j = np.unravel_index(np.argmax(taustats.joint(heat[keep], mom[keep])), (n, n))
    view.joint_table.cellClicked.emit(int(i), int(j))
    assert window.selection.size > 0
    assert set(window.selection) <= set(view.data.slot[keep & (heat == i) & (mom == j)])

    window.set_selection(np.empty(0, dtype=np.int64), "replace")
    view.tabs.setCurrentIndex(2)
    assert "run's own settings" in view.whatif_note.text() and not view.changed().any()
    view.priority.setCurrentIndex(view.priority.findData("momentum,heat"))
    assert view.changed().any() and "run's own" not in view.whatif_note.text()
    assert view.whatif_table.rowCount() == len(FULL_BOOMS)
    view.brush_changed.click()
    assert window.selection.size > 0
    view.reset_whatif()
    assert not view.changed().any()


def test_tau_whatif_recomputes_detection_on_every_slot(window, qtbot, data):
    run, d = data
    view = window.tau_view
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(view)
    qtbot.waitUntil(lambda: view.data is not None and _idle(window), timeout=60_000)
    view.tabs.setCurrentIndex(2)
    assert not view.recompute.isEnabled() and "the run's (stored)" in view.det_note.text()
    view.fields["peak_significance_se"].setValue(50.0)
    assert view.recompute.isEnabled() and "press Recompute" in view.det_note.text()
    assert not view.changed("heat").any()  # nothing applied yet
    view.recompute.click()
    qtbot.waitUntil(lambda: view.applied is not None and _idle(window), timeout=120_000)
    assert view.changed("heat").any() and "recomputed" in view.det_note.text()
    assert not view.is_default()
    view.fields["peak_significance_se"].setValue(run.cfg.secondary.detection.peak_significance_se)
    view.smoothing.setEditText("1-2-1")  # the run's weights, typed
    view.recompute.click()  # back to the run's settings: its stored detections again
    assert view.applied is None and not view.changed("heat").any()
    view.reset_whatif()
    assert view.is_default() and not view.changed().any()


def test_redetecting_with_the_runs_settings_reproduces_the_stored_detections(data, full_run):
    run, d = data
    index = FragmentIndex(run.run_dir)
    boom = FULL_BOOMS[0]
    out = taustats.redetect(index, d, boom, run.cfg.secondary.detection)
    rows = taustats.rows_of(d, boom)
    for family in taustats.SPECTRA:
        status, tau, lb = out[family]
        assert np.array_equal(status, d.status[family][rows])
        assert np.array_equal(tau, d.tau[family][rows], equal_nan=True)
        assert np.array_equal(lb, d.tau_lb[family][rows], equal_nan=True)
    plain = taustats.redetect(index, d, boom, run.cfg.secondary.detection, kernel=(1.0,))
    assert any((plain[f][0] != d.status[f][rows]).any() or not np.array_equal(plain[f][1], d.tau[f][rows],
                                                                              equal_nan=True)
               for f in taustats.SPECTRA)
