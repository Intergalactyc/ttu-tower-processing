import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.right_button import CLICK_S, RIGHT_MOVE_PX, is_click  # noqa: E402


@pytest.fixture
def plot(qtbot, monkeypatch):
    configure_pyqtgraph()
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    w.resize(500, 400)
    w.plot([0, 1, 2], [0, 1, 4])
    w.show()
    qtbot.waitExposed(w)
    w.menus = []
    monkeypatch.setattr(pg.ViewBox, "getMenu", lambda self, ev: w.menus.append(ev))  # counted, never shown
    return w


def _right(qtbot, widget, points, hold_s: float = 0.0):
    """A right press at points[0], moves through the rest, a release at the last."""
    viewport = widget.viewport()
    right, none = Qt.MouseButton.RightButton, Qt.MouseButton.NoButton
    for i, p in enumerate(points):
        kind = (QEvent.Type.MouseButtonPress if i == 0 else QEvent.Type.MouseButtonRelease
                if i == len(points) - 1 else QEvent.Type.MouseMove)
        if kind == QEvent.Type.MouseButtonRelease and hold_s:
            qtbot.wait(int(hold_s * 1000))
        pos = QPointF(*p)
        QApplication.sendEvent(viewport, QMouseEvent(kind, pos, QPointF(viewport.mapToGlobal(pos.toPoint())), right,
                                                     none if kind == QEvent.Type.MouseButtonRelease else right,
                                                     Qt.KeyboardModifier.NoModifier))
        qtbot.wait(10)


def test_a_quick_right_click_opens_the_menu(plot, qtbot):
    _right(qtbot, plot, [(250, 200), (250, 200)])
    assert len(plot.menus) == 1


def test_a_right_hold_without_a_drag_opens_nothing(plot, qtbot):
    _right(qtbot, plot, [(250, 200), (251, 200), (251, 200)], hold_s=CLICK_S + 0.15)
    assert not plot.menus


def test_a_small_quick_right_drag_rescales_without_the_menu(plot, qtbot):
    before = plot.getViewBox().viewRange()
    _right(qtbot, plot, [(250, 200), (252, 200), (254, 200), (254, 200)])
    assert not plot.menus and plot.getViewBox().viewRange() != before


def test_a_drag_pyqtgraph_never_saw_opens_no_menu(plot, qtbot, monkeypatch):
    scene = plot.scene()
    monkeypatch.setattr(scene, "_moveEventIsAllowed", lambda: False)  # every move throttled, as in a fast flick
    _right(qtbot, plot, [(250, 200), (270, 200), (270, 200)])
    assert not plot.menus


def test_no_menu_while_the_right_button_is_down(plot, qtbot, monkeypatch):
    from PySide6.QtGui import QGuiApplication
    monkeypatch.setattr(QGuiApplication, "mouseButtons", staticmethod(lambda: Qt.MouseButton.RightButton))
    _right(qtbot, plot, [(250, 200), (250, 200)])
    assert not plot.menus


def test_a_click_is_short_and_still():
    assert is_click(0.0, 0.1) and is_click(RIGHT_MOVE_PX - 0.5, CLICK_S)
    assert not is_click(RIGHT_MOVE_PX, 0.1) and not is_click(0.0, CLICK_S + 0.01)
