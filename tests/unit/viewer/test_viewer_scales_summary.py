import numpy as np
import pyarrow.dataset as pa_dataset
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from ttu_tower.viewer import composite, figures  # noqa: E402
from ttu_tower.viewer.fragments import FragmentIndex  # noqa: E402
from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402

from viewer_fixtures import FULL_BOOMS  # noqa: E402

pytestmark = pytest.mark.slow


def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


def test_scale_matrices_are_the_ladders_its_and_its_times_the_mean_speed(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    boom = FULL_BOOMS[0]
    slots, rungs, m = composite.scales_matrices(run, index, boom, "mrd")
    assert slots.size > 0 and list(rungs) == [9.375 * 2 ** j for j in range(8)]
    f = pa_dataset.field
    lad = index.read("ladder", booms=[boom], filter=(f("variant") == "mrd") & (f("boom") == boom))
    k, r = int(slots[len(slots) // 2]), 75.0
    at = lad[(lad["slot"] == k) & (lad["rung_s"] == r)]
    its = float(at[(at["variable"].astype(str) == "w") & (at["stat"].astype(str) == "its")]["value"].iloc[0])
    u = float(at[(at["variable"].astype(str) == "u") & (at["stat"].astype(str) == "mean")]["value"].iloc[0])
    i, j = np.searchsorted(slots, k), np.searchsorted(rungs, r)
    assert m["its_w"][i, j] == its and m["ils_w"][i, j] == pytest.approx(its * abs(u))


def test_summaries_give_a_median_or_a_mean_with_its_spread():
    slots, rungs = np.arange(4), np.array([1.0, 2.0])
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [10.0, np.nan]])
    med = composite.summarize(slots, rungs, values, np.ones(4, dtype=bool))
    mean = composite.summarize(slots, rungs, values, np.ones(4, dtype=bool), center="mean")
    assert med["mid"].tolist() == [2.5, 4.0] and mean["mid"].tolist() == [4.0, 4.0]
    assert mean["hi"][0] - mean["mid"][0] == pytest.approx(np.std([1.0, 2.0, 3.0, 10.0]))


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


def test_scales_tab_draws_a_line_per_boom_and_exports(window, qtbot, tmp_path):
    view = window.scales_view
    window.full_period.setChecked(True)
    window.right_tabs.setCurrentWidget(view)
    qtbot.waitUntil(lambda: bool(view.summaries) and _idle(window), timeout=60_000)
    assert set(view.summaries) == set(FULL_BOOMS) and view.table.rowCount() == len(FULL_BOOMS)
    assert "ITS = rung/" in view.info.text()  # the τ/ITS guide on ITS
    median = view.summaries[FULL_BOOMS[0]]["mid"].to_numpy()
    view.center.setCurrentIndex(view.center.findData("mean"))
    assert not np.allclose(view.summaries[FULL_BOOMS[0]]["mid"].to_numpy(), median)
    view.quantity.setCurrentIndex(view.quantity.findData("ils_u"))
    assert "ITS = rung/" not in view.info.text() and "ILS u" in view.plot.getPlotItem().getAxis("left").labelText
    plain = view.summaries[FULL_BOOMS[0]]["mid"].to_numpy()
    view.corrected.setChecked(True)
    assert view.summaries[FULL_BOOMS[0]]["mid"].to_numpy() == pytest.approx(plain * np.e / (np.e - 1), nan_ok=True)
    assert "ILS u corrected" in view.plot.getPlotItem().getAxis("left").labelText
    spec = window.summary_spec()
    assert spec is not None and spec[0] == "composite" and spec[1]["xlabel"] == "ladder rung (s)"
    figures.save(figures.render(*spec, (6, 4)), tmp_path / "scales.png", dpi=60)
