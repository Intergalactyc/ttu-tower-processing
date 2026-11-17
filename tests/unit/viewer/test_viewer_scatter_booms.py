"""Scatter by boom height: one axis's boom fixed, and fits per boom; built from small in-memory data."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from PySide6.QtCore import QItemSelectionModel  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ttu_tower.constants import HEIGHTS  # noqa: E402
from ttu_tower.viewer.timeline import Curve  # noqa: E402
from ttu_tower.viewer.ui import style, view_state  # noqa: E402
from ttu_tower.viewer.ui.scatter import HEIGHT, PerBoomFit, ScatterView  # noqa: E402

SLOTS = np.arange(6, dtype=np.int64)


def _data(key, booms, scale, kind="boom", skip=()):
    curves = {}
    for b in booms:
        slots = np.array([s for s in SLOTS if (b, s) not in skip], dtype=np.int64)
        curves[b] = Curve(slots, scale * (b or 0) + slots.astype(float))
    quantity = SimpleNamespace(key=key, title=key, kind=kind, categorical=False, extra=None)
    return SimpleNamespace(quantity=quantity, variant="none", curves=curves)


@pytest.fixture
def scatter(qtbot):
    view = ScatterView()
    qtbot.addWidget(view)
    # x on booms 1-3 (boom 2 lacks slot 4), y on booms 1-5
    view.set_sources([_data("a", (1, 2, 3), 10.0, skip={(2, 4)}), _data("b", (1, 2, 3, 4, 5), 100.0)], None)
    view.source[2].setCurrentIndex(view.source[2].findData(HEIGHT))
    return view


def _fix(view, axis, boom):
    view.member[axis].setCurrentIndex(view.member[axis].findData(boom))
    view.boom_mode[axis].setCurrentIndex(view.boom_mode[axis].findData("fixed"))


def test_by_default_both_booms_vary_over_the_booms_both_have(scatter):
    assert scatter.boom_mode[0].currentData() == scatter.boom_mode[1].currentData() == "each"
    assert set(scatter.point_member.tolist()) == {1, 2, 3}
    assert np.array_equal(scatter.x % 10, scatter.y % 100)  # same slot
    assert np.array_equal(scatter.x // 10, scatter.point_member)  # same boom
    assert not scatter.member[0].isEnabled() and not scatter.member[1].isEnabled()
    assert scatter.axis_title(2) == "height [m]"


def test_a_fixed_x_pairs_its_boom_with_every_boom_of_y_by_slot(scatter):
    _fix(scatter, 0, 2)
    assert set(scatter.point_member.tolist()) == {1, 2, 3, 4, 5}  # all of y's booms, not only x's
    assert np.all(scatter.x // 10 == 2)  # x always at boom 2
    assert np.array_equal(scatter.y // 100, scatter.point_member)  # y at the point's boom
    assert np.array_equal(scatter.x % 10, scatter.y % 100)  # paired by slot
    assert 4 not in scatter.slots  # x's boom 2 has no slot 4: no pair there
    assert scatter.x.size == 5 * (SLOTS.size - 1)
    assert np.array_equal(scatter.z, [HEIGHTS[int(b)] for b in scatter.point_member])  # z follows y's boom
    assert scatter.member[0].isEnabled() and not scatter.member[1].isEnabled()
    assert "b2" in scatter.axis_title(0) and "each boom" in scatter.axis_title(1)
    assert scatter.axis_title(2) == "height of y's boom [m]"


def test_a_fixed_y_varies_x_instead(scatter):
    _fix(scatter, 1, 5)
    assert set(scatter.point_member.tolist()) == {1, 2, 3}
    assert np.all(scatter.y // 100 == 5) and np.array_equal(scatter.x // 10, scatter.point_member)
    assert scatter.axis_title(2) == "height of x's boom [m]"
    heights = np.array([HEIGHTS[int(b)] for b in scatter.point_member])
    values = scatter._height_values(heights)  # panel A varies with the points; panel B fixed at boom 5
    assert np.array_equal(values["zA"], heights) and values["zB"] == HEIGHTS[5]


def test_both_fixed_shows_nothing_and_says_why(scatter):
    _fix(scatter, 0, 2)
    _fix(scatter, 1, 3)
    assert scatter.x.size == 0 and "both are fixed" in scatter.empty_reason


def test_the_varying_axis_must_be_per_boom(qtbot):
    view = ScatterView()
    qtbot.addWidget(view)
    view.set_sources([_data("a", (1, 2, 3), 10.0), _data("b", (None,), 0.0, kind="slot")], None)
    view.source[2].setCurrentIndex(view.source[2].findData(HEIGHT))
    assert set(view.point_member.tolist()) == {1, 2, 3}  # a slot quantity at every boom, as before
    _fix(view, 0, 2)
    assert view.x.size == 0 and "needs a quantity per boom" in view.empty_reason


def test_the_choice_shows_only_by_boom_height_and_is_saved_and_swapped(scatter):
    _fix(scatter, 0, 2)
    state = view_state.save(scatter, scatter.STATE_FIELDS)
    scatter._swap()
    assert scatter.boom_mode[1].currentData() == "fixed" and scatter.boom_mode[0].currentData() == "each"
    assert scatter.member[1].currentData() == 2 and np.all(scatter.y // 10 == 2)
    assert not {"boom_mode.0", "boom_mode.1"} & set(view_state.restore(scatter, state, scatter.STATE_FIELDS))
    assert scatter.boom_fixed(0) and not scatter.boom_fixed(1) and np.all(scatter.x // 10 == 2)
    scatter.source[2].setCurrentIndex(0)
    assert scatter.boom_mode[0].isHidden() and not scatter.boom_fixed(0)
    assert scatter.point_member is None and scatter.member[0].isEnabled()


# --- fits per boom ---------------------------------------------------------------------------------

EXPONENTS = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.1}


def _power_law_data(lonely=None):
    """x = 1..40 at every boom, y = (b + 1) x^k_b; `lonely`: a boom with a single point."""
    x = np.arange(1.0, 41.0)
    slots = np.arange(x.size, dtype=np.int64)
    xs, ys = {}, {}
    for b, k in EXPONENTS.items():
        xs[b], ys[b] = Curve(slots, x), Curve(slots, (b + 1) * x ** k)
    if lonely is not None:
        xs[lonely], ys[lonely] = Curve(slots[:1], x[:1]), Curve(slots[:1], x[:1])

    def data(key, curves):
        quantity = SimpleNamespace(key=key, title=key, kind="boom", categorical=False, extra=None)
        return SimpleNamespace(quantity=quantity, variant="none", curves=curves)
    return [data("L", xs), data("ILS", ys)]


@pytest.fixture
def per_boom(qtbot):
    view = ScatterView()
    qtbot.addWidget(view)
    view.set_sources(_power_law_data(lonely=5), None)
    view.source[2].setCurrentIndex(view.source[2].findData(HEIGHT))
    view.fit_name.setCurrentIndex(view.fit_name.findData("power law"))
    view.fit_per_boom.setChecked(True)
    view.run_fit()
    return view


def _curves(view):
    return [line for line in view.drawn_lines if line["color"] in {style.boom_color(b).name() for b in range(1, 11)}]


def test_a_per_boom_fit_finds_each_booms_own_exponent(per_boom):
    result = per_boom.fit_result
    assert isinstance(result, PerBoomFit) and set(result.fits) == {1, 2, 3, 4, 5}
    for b, k in EXPONENTS.items():
        assert result.fits[b].params["b"] == pytest.approx(k, rel=1e-6)
        assert result.fits[b].params["a"] == pytest.approx(b + 1, rel=1e-6)
    assert isinstance(result.fits[5], str)  # one point: fails there only
    assert "fitted at 4 of 5 booms" in per_boom.fit_status.text() and "failed at b5" in per_boom.fit_status.text()
    assert len(_curves(per_boom)) == 4 and sum(line["label"] is not None for line in _curves(per_boom)) == 1
    assert not per_boom.overlay.from_fit.isEnabled()  # no single formula to overlay


def test_the_table_lists_every_boom_and_copies(per_boom):
    table = per_boom.boom_table
    assert not table.isHidden() and table.rowCount() == 5
    frame = per_boom.boom_frame(per_boom.current_entry)
    assert list(frame["boom"]) == [style.member_label(b) for b in (1, 2, 3, 4, 5)]
    assert list(frame.columns[:4]) == ["boom", "height [m]", "a", "b"]
    assert frame["b"].iloc[:4].to_numpy() == pytest.approx(list(EXPONENTS.values()), rel=1e-6)
    assert frame["RMSE"].iloc[:4].to_numpy() == pytest.approx(0.0, abs=1e-6)
    assert frame["NRMSE_IQR"].iloc[:4].to_numpy() == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(frame["NRMSE_IQR"].iloc[4])  # its fit failed
    x = np.arange(1.0, 41.0)  # r and ρ: x vs y over each boom's points
    assert frame["r"].iloc[:4].to_numpy() == pytest.approx(
        [np.corrcoef(x, (b + 1) * x ** k)[0, 1] for b, k in EXPONENTS.items()])
    assert frame["ρ"].iloc[:4].to_numpy() == pytest.approx(1.0) and np.isnan(frame["r"].iloc[4])
    assert list(frame.columns[-7:]) == ["N fitted", "RMSE", "NRMSE_IQR", "r", "ρ", "N shown", "note"]
    assert frame["N shown"].iloc[:4].tolist() == [40] * 4 and "fewer than two" in frame["note"].iloc[4]
    per_boom.copy_boom_table()
    copied = QApplication.clipboard().text().splitlines()
    assert copied[0].split("\t")[:4] == ["boom", "height [m]", "a", "b"] and len(copied) == 6


def test_without_every_curve_only_the_picked_booms_are_drawn(per_boom):
    per_boom.fit_boom_curves.setChecked(False)
    assert _curves(per_boom) == []
    model, flags = per_boom.boom_table.selectionModel(), QItemSelectionModel.SelectionFlag
    for row in (0, 2):
        model.select(per_boom.boom_table.model().index(row, 0), flags.Select | flags.Rows)
    assert per_boom.boom_pick == {1, 3}
    assert {line["color"] for line in _curves(per_boom)} == {style.boom_color(b).name() for b in (1, 3)}
    per_boom.redraw()  # the table is refilled: the picks stay
    assert per_boom.boom_pick == {1, 3} and len(per_boom.boom_table.selectionModel().selectedRows()) == 2
    per_boom.fit_boom_curves.setChecked(True)
    assert len(_curves(per_boom)) == 4


def test_a_per_boom_fit_is_saved_and_needs_boom_height(per_boom):
    saved = per_boom.fits_state.get_state()
    assert saved["fits"][0]["per_boom"] is True and saved["fits"][0]["boom_curves"] is True
    per_boom.fits_state.put_state(saved)
    assert per_boom.fit_per_boom.isChecked() and per_boom.fit_result is None  # not fitted yet
    per_boom.source[2].setCurrentIndex(0)
    assert per_boom.fit_per_boom.isEnabled() and per_boom.fit_per_boom.isChecked()  # untickable while ticked
    per_boom.run_fit()
    assert per_boom.fit_result is None and "needs z = boom height" in per_boom.fit_status.text()
    assert per_boom.boom_table.isHidden()
    per_boom.fit_per_boom.setChecked(False)
    assert not per_boom.fit_per_boom.isEnabled()
    per_boom.run_fit()
    assert per_boom.fit_result is not None and not isinstance(per_boom.fit_result, PerBoomFit)


def test_scores_carry_nrmse_iqr_alongside_the_rmse_in_y_and_in_ln_y():
    y = np.arange(1.0, 11.0)
    at = y * 1.1
    bins = (np.arange(5.0), np.full(5, 3.0))  # constant y: IQR 0
    scores = ScatterView._scores(at, y, bins, bins[1] + 0.5)
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    assert scores["nrmse points"] == pytest.approx(scores["points"][0] / iqr)
    ln_iqr = np.subtract(*np.percentile(np.log(y), [75, 25]))  # ln y: the ln RMSE over the IQR of ln y
    assert scores["nrmse ln points"] == pytest.approx(scores["ln points"][0] / ln_iqr)
    assert np.isnan(scores["nrmse bins"]) and np.isnan(scores["nrmse ln bins"])
    text = ScatterView._rmse_text(scores)
    assert f"NRMSE_IQR {scores['nrmse points']:.3g}, —" in text
    assert f"NRMSE_IQR {scores['nrmse ln points']:.3g}, —" in text
    assert "NRMSE_IQR" in ScatterView._rmse_text(scores, short=True)
    assert np.isnan(ScatterView._scores(at[:3], y[:3], None, None)["nrmse points"])  # too few points


def test_switching_per_boom_marks_a_fit_stale(per_boom):
    fitted = per_boom.current_entry.inputs
    per_boom.fit_per_boom.setChecked(False)
    assert per_boom._inputs() != fitted and "press Fit to refit" in per_boom.fit_status.text()
