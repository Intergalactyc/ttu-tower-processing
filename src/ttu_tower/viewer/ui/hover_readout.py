"""The cursor's position over any plot under a widget (the Summary panel), as
"<axis label> = <value>" for each axis shown: real values on log axes, the
nearest label on an axis of labelled ticks (hours, months, booms).
"""
import html
import re
from datetime import datetime, timezone

import numpy as np
import pyqtgraph as pg

from ttu_tower.viewer.ui.axes import Log10Axis

_TAG = re.compile(r"<[^>]+>")


def axis_name(axis: pg.AxisItem, fallback: str) -> str:
    text = html.unescape(_TAG.sub("", axis.labelText or "")).replace(" (log scale)", "").strip()
    units = (axis.labelUnits or "").strip()
    return (f"{text} [{units}]" if units and units not in text else text) or fallback


def axis_value(axis: pg.AxisItem, v: float) -> str:
    """The value at view coordinate `v` as the axis reads it."""
    ticks = getattr(axis, "_tickLevels", None)
    if ticks:  # labelled ticks: the one whose cell (half a unit or spacing either side) holds the cursor
        at = [(float(p), str(label)) for level in ticks for p, label in level if str(label)]
        if at:
            spacing = np.median(np.diff(sorted(p for p, _ in at))) if len(at) > 1 else 1.0
            p, label = min(at, key=lambda t: abs(t[0] - v))
            if abs(p - v) <= min(spacing / 2, 0.5):
                return label
    if isinstance(axis, Log10Axis) and getattr(axis, "log10", True) or axis.logMode:
        v = 10.0 ** v
    if isinstance(axis, pg.DateAxisItem):
        return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"{v:.4g}"


def readout(plot_item: pg.PlotItem, scene_pos) -> str | None:
    vb = plot_item.getViewBox()
    if not vb.sceneBoundingRect().contains(scene_pos):
        return None
    if isinstance(plot_item, pg.ColorBarItem):  # its view runs 0-256 along the bar
        p = vb.mapSceneToView(scene_pos)
        lo, hi = plot_item.values
        along = p.x() if plot_item.horizontal else p.y()
        label = axis_name(plot_item.getAxis("bottom" if plot_item.horizontal else "left"), "color scale")
        return f"{label} = {lo + along / 256.0 * (hi - lo):.4g}"
    p = vb.mapSceneToView(scene_pos)
    parts = []
    for sides, coordinate, fallback in ((("bottom", "top"), p.x(), "x"), (("left", "right"), p.y(), "y")):
        axis = next((plot_item.getAxis(s) for s in sides if plot_item.getAxis(s).isVisible()), None)
        if axis is not None:
            name, value = axis_name(axis, fallback), axis_value(axis, coordinate)
            labelled = name == fallback and getattr(axis, "_tickLevels", None)  # untitled, e.g. months: the label says it
            parts.append(value if labelled else f"{name} = {value}")
    return " · ".join(parts) or None


class HoverReadout:
    """Reports the cursor's position over every plot under the widgets it watches."""

    def __init__(self, report):
        self.report = report  # (text) -> None
        self._watched: set[int] = set()

    def watch(self, root) -> None:
        """Every plot under `root`, including ones made since the last call."""
        for view in root.findChildren(pg.GraphicsView):
            if id(view) not in self._watched and view.scene() is not None:
                self._watched.add(id(view))
                view.scene().sigMouseMoved.connect(lambda pos, v=view: self._moved(v, pos))

    def _moved(self, view, pos) -> None:
        for item in view.scene().items(pos):
            while item is not None and not isinstance(item, pg.PlotItem):
                item = item.parentItem()
            if item is not None:
                text = readout(item, pos)
                if text:
                    self.report(text)
                return
