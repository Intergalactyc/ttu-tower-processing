"""Curves that redraw from a NaN-aware min/max decimation of the visible x
range whenever their plot is panned or zoomed.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Signal

from ttu_tower.viewer.decimate import minmax_decimate


def _ignore(ev) -> None:
    ev.ignore()


def pass_clicks(item) -> None:
    """Let clicks on an item's markers through to its ViewBox (a
    ScatterPlotItem takes a click that lands on a marker for itself).
    """
    scatter = item.scatter if isinstance(item, pg.PlotDataItem) else item
    scatter.mouseClickEvent = _ignore


class FixedXViewBox(pg.ViewBox):
    """A ViewBox whose x range only ever changes when asked. Continuous x
    autorange would feed back through decimation (the drawn points reach one
    sample past the view, so each redraw would widen it), so any request to
    enable it - the auto-range button, a log-mode switch - becomes a one-off
    `fitXRequested` instead. Y autorange works as usual.
    """

    fitXRequested = Signal()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.suppress_fit = False  # set while code (not the user) changes plot settings, e.g. log mode
        super().enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

    def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
        if x is None and y is None:
            if axis in (None, pg.ViewBox.XYAxes):
                x = y = enable
            elif axis in (pg.ViewBox.XAxis, 0, "x"):
                x = enable
            else:
                y = enable
        if y is not None:
            super().enableAutoRange(axis=pg.ViewBox.YAxis, enable=y)
        if x and not self.suppress_fit:
            self.fitXRequested.emit()


class DecimatedCurve:
    def __init__(self, item: pg.PlotDataItem, x: np.ndarray, y: np.ndarray):
        self.item = item
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

    def redraw(self, x0: float, x1: float, buckets: int, log_y: bool = False) -> None:
        y = self.y
        if log_y:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.where(y > 0, y, np.nan)
        dx, dy = minmax_decimate(self.x, y, x0, x1, buckets)
        if not np.isfinite(dy).any():  # nothing to draw here (and pyqtgraph warns on all-NaN markers)
            dx = dy = np.empty(0)
        self.item.setData(dx, dy, connect="finite")


class PlotDecimator:
    """Keeps every registered curve of one PlotItem decimated to its view."""

    def __init__(self, plot_item: pg.PlotItem, delay_ms: int = 40):
        self.plot_item = plot_item
        self.curves: list[DecimatedCurve] = []
        self.log_y = False
        self._timer = QTimer(plot_item.getViewBox())  # dies with its plot, so it never fires on a deleted one
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self.redraw)
        plot_item.getViewBox().sigXRangeChanged.connect(lambda *_: self._timer.start())
        plot_item.getViewBox().sigResized.connect(lambda *_: self._timer.start())

    def add(self, item: pg.PlotDataItem, x, y) -> DecimatedCurve:
        curve = DecimatedCurve(item, x, y)
        self.curves.append(curve)
        return curve

    def clear(self) -> None:
        self.curves.clear()

    def redraw(self) -> None:
        vb = self.plot_item.getViewBox()
        (x0, x1), _ = vb.viewRange()
        buckets = max(int(vb.width()), 200)
        for curve in self.curves:
            if curve.item.isVisible():
                curve.redraw(x0, x1, buckets, self.log_y)


def data_bounds(curves) -> tuple[float, float] | None:
    """The x extent of all curves with any finite value."""
    lo, hi = np.inf, -np.inf
    for c in curves:
        finite = np.isfinite(c.y)
        if finite.any():
            lo, hi = min(lo, c.x[finite][0]), max(hi, c.x[finite][-1])
    return (lo, hi) if np.isfinite(lo) else None
