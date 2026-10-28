import json

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.timeaxis import slot_to_unix  # noqa: E402
from ttu_tower.viewer.ui import view_state  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402
from ttu_tower.viewer.views import SavedView, ViewStore  # noqa: E402

from viewer_fixtures import FULL_BOOMS, SPIKY_SLOT  # noqa: E402

pytestmark = pytest.mark.slow


def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


def _open(qtbot, full_run, **kwargs):
    configure_pyqtgraph()
    w = MainWindow(RunHandle.from_dir(full_run.run_dir), **kwargs)
    w.views = ViewStore(path=None, load=False)
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    qtbot.waitUntil(lambda: _idle(w) and w.datas[0] is not None and w._stability_known, timeout=60_000)
    return w


def test_views_are_kept_by_name_and_run(tmp_path):
    store = ViewStore(tmp_path / "views.json")
    store.add(SavedView("winter", "a", {"x": 1}))
    store.add(SavedView("winter", "b", {"x": 2}))
    store.add(SavedView("winter", "a", {"x": 3}))  # the same name and run: replaced
    again = ViewStore(tmp_path / "views.json")
    assert [(v.run, v.state["x"]) for v in again.ordered("b")] == [("b", 2), ("a", 3)]
    again.remove(again.ordered("a")[0].id)
    assert [v.run for v in ViewStore(tmp_path / "views.json").items] == ["b"]


def test_selected_slots_pack_into_runs():
    slots = [5, 1, 2, 3, 9, 10]
    assert view_state.pack_slots(slots) == [[1, 3], [5, 5], [9, 10]]
    assert view_state.unpack_slots(view_state.pack_slots(slots)) == sorted(slots)


def test_a_saved_view_reopens_the_same_state(qtbot, full_run):
    first = _open(qtbot, full_run)
    first.panels[0].select("boom_final|sigma_w|", "mrd", tuple(FULL_BOOMS[:2]))
    first.panels[0].log_y.setChecked(True)
    qtbot.waitUntil(lambda: _idle(first) and first.datas[0].quantity.key == "boom_final|sigma_w|", timeout=60_000)
    first.plots[0].legend.solo(FULL_BOOMS[1])
    x = slot_to_unix(SPIKY_SLOT)
    first.set_x_range(x - 6 * 3600, x + 6 * 3600)
    first.night_box.setChecked(False)
    first.set_selection(np.arange(SPIKY_SLOT, SPIKY_SLOT + 4), "replace")
    first.right_tabs.setCurrentWidget(first.diurnal)  # its lists fill once it's shown
    qtbot.waitUntil(lambda: _idle(first), timeout=60_000)
    first.diurnal.member.setCurrentIndex(first.diurnal.member.count() - 1)
    first.right_tabs.setCurrentWidget(first.scatter)
    first.scatter.log_x.setChecked(True)
    first.scatter.fit_bins.setValue(12)
    first.dists[0].side.setCurrentIndex(first.dists[0].side.findData("abs"))
    first.composite.spectrum.setCurrentIndex(first.composite.spectrum.findData("ww"))
    first.windrose.sectors.setCurrentIndex(0)
    qtbot.waitUntil(lambda: _idle(first), timeout=60_000)
    state = first.view_state()

    second = _open(qtbot, full_run)
    notes = second.apply_view_state(state)
    qtbot.waitUntil(lambda: _idle(second) and second.datas[0].quantity.key == "boom_final|sigma_w|"
                    and not second._pending_legend, timeout=60_000)
    assert notes == []
    assert second.panels[0].state() == state["panels"][0] and second.panels[1].state() == state["panels"][1]
    assert second.plots[0].visible_members() == [FULL_BOOMS[1]]
    assert np.allclose(second.x_range(), state["x_range"]) and not second.night_box.isChecked()
    assert list(second.selection) == list(range(SPIKY_SLOT, SPIKY_SLOT + 4))
    assert second.right_tabs.currentWidget() is second.scatter
    again = second.view_state()
    for tab in ("Scatter", "Distributions A", "Spectra", "Wind rose"):
        assert again["summary"][tab] == state["summary"][tab], tab
    assert "Diurnal" in second._summary_retry  # its boom list fills when it's shown: restored then
    second.right_tabs.setCurrentWidget(second.diurnal)
    qtbot.waitUntil(lambda: _idle(second) and "Diurnal" not in second._summary_retry, timeout=60_000)
    assert second.view_state()["summary"]["Diurnal"] == state["summary"]["Diurnal"]


def test_a_view_from_another_run_skips_what_this_run_lacks(qtbot, full_run):
    window = _open(qtbot, full_run)
    state = window.view_state()
    state["panels"][0] = dict(state["panels"][0], quantity="boom_final|not_here|")
    state["summary"]["No such tab"] = {}
    notes = window.apply_view_state(state)
    assert any("not_here" in n and "isn't in this run" in n for n in notes)
    assert any("No such tab" in n for n in notes)


@pytest.mark.parametrize("answer", ["yes", "no", "yes, after reading it"])
def test_reopening_a_run_offers_to_resume_where_it_was_left(qtbot, full_run, monkeypatch, answer):
    asked = []
    reply = QMessageBox.StandardButton.No if answer == "no" else QMessageBox.StandardButton.Yes

    def question(*a, **k):
        asked.append(a)
        if answer.endswith("reading it"):  # a real dialog runs the event loop while it waits for the click
            qtbot.wait(3000)
        return reply
    monkeypatch.setattr(MainWindow, "_ask_resume", lambda self, tag: question() == QMessageBox.StandardButton.Yes)
    first = _open(qtbot, full_run, resume="ask")
    key = f"session/{first.run.tag}"
    first.settings.remove(key)
    first.panels[0].select("boom_final|tke|", "mrd", tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(first) and first.datas[0].quantity.key == "boom_final|tke|", timeout=60_000)
    first.close()  # where it was left goes to the settings
    asked.clear()
    second = _open(qtbot, full_run, resume="ask")
    assert len(asked) == 1
    expected = "boom_final|ws|mean" if answer == "no" else "boom_final|tke|"
    assert second.datas[0].quantity.key == expected
    second.resume = "never"  # nothing left for later tests
    second.settings.remove(key)


def test_resuming_after_the_quantities_came_applies_the_view_at_once(qtbot, full_run, monkeypatch):
    monkeypatch.setattr(MainWindow, "_ask_resume", lambda self, tag: True)
    window = _open(qtbot, full_run)
    state = window.view_state()
    state["panels"][0] = dict(state["panels"][0], quantity="boom_final|tke|", variant="mrd")
    key = f"session/{window.run.tag}"
    window.settings.setValue(key, json.dumps(state))
    window.resume = "ask"
    try:
        window._offer_resume(window.run)  # as if the catalog had arrived while the dialog was open
        qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "boom_final|tke|", timeout=60_000)
        assert window._pending_view is None
    finally:
        window.resume = "never"
        window.settings.remove(key)


def test_a_late_combo_takes_its_saved_choice_once_its_items_arrive(qtbot):
    box = view_state.LateCombo()
    qtbot.addWidget(box)
    assert view_state.put(box, {"data": 5, "text": "b5"}) and box.pending is not None
    box.addItem("b2", 2)
    box.addItem("b5", 5)
    box.take_pending()
    assert box.currentData() == 5 and box.pending is None


def test_the_resume_question_is_a_window_of_its_own_until_the_viewer_shows(qtbot, monkeypatch):
    seen = []

    def answer(box):
        seen.append({"parent": box.parent(), "on_top": bool(box.windowFlags() & Qt.WindowType.WindowStaysOnTopHint),
                     "default_yes": box.defaultButton() is box.button(QMessageBox.StandardButton.Yes),
                     "title": box.windowTitle()})
        box.button(QMessageBox.StandardButton.Yes).click()
        return 0
    monkeypatch.setattr(QMessageBox, "exec", answer)
    configure_pyqtgraph()
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._ask_resume("someday") is True
    assert seen[0] == {"parent": None, "on_top": True, "default_yes": True, "title": "ttu-view — someday"}
    window.show()
    qtbot.waitExposed(window)
    window._ask_resume("someday")
    assert seen[1]["parent"] is window and not seen[1]["on_top"]  # on the viewer once it's shown
