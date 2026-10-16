"""Right button on every plot: a drag rescales, a click opens the plot's menu.

pyqtgraph decides between the two from its own drag bookkeeping, which only
looks at the mouse moves it doesn't throttle, and counts any release before
5 px as a click however long the hold. So a quick flick, jitter at the end of
a drag or a hesitant hold could open the menu. Here a right release is a click
only if the pointer never travelled RIGHT_MOVE_PX from the press (over every
move, throttled or not) and the button came up within CLICK_S; and the menu
never opens while the right button is down.
"""
from time import perf_counter

import pyqtgraph as pg
from pyqtgraph.GraphicsScene import GraphicsScene
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

RIGHT_MOVE_PX = 3
CLICK_S = 0.35
_RIGHT = Qt.MouseButton.RightButton
_installed = False


def is_click(travel_px: float, held_s: float) -> bool:
    return travel_px < RIGHT_MOVE_PX and held_s <= CLICK_S


def _distance(ev, start) -> float:
    p = ev.screenPos()
    return ((p.x() - start[0]) ** 2 + (p.y() - start[1]) ** 2) ** 0.5


def install() -> None:
    """Patch every plot's scene and view box, once."""
    global _installed
    if _installed:
        return
    _installed = True
    press, move, release = GraphicsScene.mousePressEvent, GraphicsScene.mouseMoveEvent, GraphicsScene.mouseReleaseEvent
    raise_menu = pg.ViewBox.raiseContextMenu

    def on_press(self, ev):
        if not hasattr(self, "_default_move_px"):
            self._default_move_px = self._moveDistance
        right = ev.button() == _RIGHT
        self._moveDistance = RIGHT_MOVE_PX if right else self._default_move_px
        if right:
            p = ev.screenPos()
            self._right_press = (p.x(), p.y(), perf_counter())
            self._right_travel = 0.0
        press(self, ev)

    def on_move(self, ev):
        start = getattr(self, "_right_press", None)
        if start is not None and ev.buttons() & _RIGHT:
            self._right_travel = max(self._right_travel, _distance(ev, start))
        move(self, ev)

    def on_release(self, ev):
        start = getattr(self, "_right_press", None)
        if ev.button() == _RIGHT and start is not None:
            travel = max(self._right_travel, _distance(ev, start))
            # a drag pyqtgraph saw ends as a drag anyway; one it missed, or a hold, mustn't become a click
            if _RIGHT not in self.dragButtons and not is_click(travel, perf_counter() - start[2]):
                self.clickEvents = [e for e in self.clickEvents if e.button() != _RIGHT]
            self._right_press = None
        release(self, ev)

    def on_raise_menu(self, ev):
        if QGuiApplication.mouseButtons() & _RIGHT:  # still held: the start of a drag, not a click
            return
        raise_menu(self, ev)

    GraphicsScene.mousePressEvent = on_press
    GraphicsScene.mouseMoveEvent = on_move
    GraphicsScene.mouseReleaseEvent = on_release
    pg.ViewBox.raiseContextMenu = on_raise_menu
