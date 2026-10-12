import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

import pyqtgraph as pg  # noqa: E402

from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.axis_units import power_of_ten  # noqa: E402


def test_the_power_of_ten_is_factored_out_only_outside_the_readable_range():
    assert power_of_ten(-0.9, 0.9) == 0 and power_of_ten(0.0, 5000.0) == 0 and power_of_ten(0.0, 0.0) == 0
    assert power_of_ten(-3.2e-4, 1.1e-4) == -4 and power_of_ten(0.0, 2.5e5) == 5


@pytest.mark.parametrize("lo, hi, label", [
    (0.0, 0.9, "wd std (deg)"),  # never millidegrees
    (0.0, 3.0e-4, "u'w' (×10⁻⁴ m²/s²)"),  # never mm²/s²
    (0.0, 2.0e5, "count (×10⁵)"),
])
def test_an_axis_keeps_its_units_with_a_power_of_ten(qtbot, lo, hi, label):
    configure_pyqtgraph()
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    unit = {"wd std": "deg", "u'w'": "m²/s²"}.get(label.split(" (")[0], "")
    w.setLabel("left", label.split(" (")[0], units=unit or None)
    w.plot([0, 1], [lo, hi])
    w.setYRange(lo, hi, padding=0)
    axis = w.getAxis("left")
    assert label in axis.labelString()
    values = axis.tickValues(lo, hi, 300)[0][1]
    strings = axis.tickStrings(values, axis.autoSIPrefixScale * axis.scale, values[1] - values[0])
    assert max(abs(float(s.replace("−", "-"))) for s in strings) < 10_000


def test_fixed_ticks_and_unlabelled_axes_are_never_scaled(qtbot):
    configure_pyqtgraph()
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    w.setLabel("bottom", "hour")
    w.plot([0.0, 1e-5], [0, 1])
    w.setXRange(0.0, 1e-5, padding=0)
    assert "×10" in w.getAxis("bottom").labelString()
    w.getAxis("bottom").setTicks([[(0.0, "a"), (1e-5, "b")]])
    assert "×10" not in w.getAxis("bottom").labelString() and w.getAxis("bottom").autoSIPrefixScale == 1.0
    w.setYRange(0.0, 1e-6, padding=0)  # the left axis has no label to carry a factor
    assert w.getAxis("left").autoSIPrefixScale == 1.0
