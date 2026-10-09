import numpy as np
import pytest

pytest.importorskip("PySide6")

from ttu_tower.viewer.ui.jets_event import summarize  # noqa: E402

ROWS = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, np.nan], [6.0, 14.0]])  # slots x booms


def test_mean_and_sd_per_boom():
    centre, low, high = summarize(ROWS, "mean")
    assert centre == pytest.approx([3.0, 12.0])
    assert high - centre == pytest.approx([np.std(ROWS[:, 0], ddof=1), 2.0])
    assert low == pytest.approx(2 * centre - high)


def test_median_and_quartiles_per_boom():
    centre, low, high = summarize(ROWS, "median")
    assert centre == pytest.approx([2.5, 12.0]) and low == pytest.approx([1.75, 11.0])
    assert high == pytest.approx([3.75, 13.0])


def test_directions_are_summarized_across_north_and_stay_continuous():
    rows = np.array([[350.0, 355.0, 5.0], [10.0, 359.0, 15.0], [0.0, 2.0, 10.0], [5.0, 358.0, 12.0]])
    for how in ("mean", "median"):
        centre, low, high = summarize(rows, how, circular=True)
        assert np.all(np.abs(np.diff(centre)) < 20)  # no jump of ~360 between booms
        assert np.all(low <= centre) and np.all(centre <= high) and np.all(high - low < 30)
    assert summarize(rows, "mean", circular=True)[0][0] % 360 == pytest.approx(1.25, abs=0.01)


def test_a_boom_without_values_stays_blank():
    centre, low, high = summarize(np.array([[np.nan, 1.0], [np.nan, 3.0]]), "mean")
    assert np.isnan(centre[0]) and np.isnan(low[0]) and centre[1] == 2.0
