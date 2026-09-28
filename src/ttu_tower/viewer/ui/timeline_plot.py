"""One timeline panel: a quantity over the whole period, one curve per boom
(or boom pair), clickable through to the data behind a point.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QToolButton, QVBoxLayout, QWidget

from ttu_tower.viewer.decimate import break_wraps, nearest_on_curve
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, display_offset_s, unix_to_slot
from ttu_tower.viewer.timeline import Curve, TimelineData
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.curves import FixedXViewBox, PlotDecimator

PICK_RADIUS_PX = 8.0
MIN_AXIS_WIDTH = 60
MARKER_SIZE = 5
AMBIGUOUS_PX = 2.0


class PickViewBox(FixedXViewBox):
    """Clicks near a data point go to `picker(ev)`; anything else (the
    right-click menu with export, panning) behaves as usual.
    """

    def __init__(self, picker=None, **kwargs):
        super().__init__(**kwargs)
        self.picker = picker

    def mouseClickEvent(self, ev):
        if self.picker is not None and self.picker(ev):
            ev.accept()
            return
        super().mouseClickEvent(ev)


class LegendBar(QWidget):
    """One toggle per curve: click shows/hides it, right-click shows only it."""

    toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(4, 0, 4, 0)
        self.layout_.setSpacing(4)
        self.buttons: dict = {}

    def set_members(self, members: list, labels: list[str], colors: list) -> None:
        for btn in self.buttons.values():
            btn.deleteLater()
        self.buttons = {}
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for member, label, color in zip(members, labels, colors):
            btn = QToolButton()
            btn.setText(label or "value")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setStyleSheet(f"QToolButton {{ color: {color.name()}; font-weight: bold; }}"
                              "QToolButton:!checked { color: #bbbbbb; font-weight: normal; }")
            btn.toggled.connect(lambda *_: self.toggled.emit())
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda _pos, m=member: self.solo(m))
            self.buttons[member] = btn
            self.layout_.addWidget(btn)
        self.layout_.addStretch(1)

    def solo(self, member) -> None:
        for m, btn in self.buttons.items():
            btn.blockSignals(True)
            btn.setChecked(m == member)
            btn.blockSignals(False)
        self.toggled.emit()

    def show_all(self) -> None:
        for btn in self.buttons.values():
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
        self.toggled.emit()

    def visible(self) -> set:
        return {m for m, btn in self.buttons.items() if btn.isChecked()}


class TimelinePlot(QWidget):
    pointClicked = Signal(int, object)  # slot, member
    hovered = Signal(int, str)  # slot, values of the visible curves there

    def __init__(self, tz: str = "Etc/GMT+6", parent=None):
        super().__init__(parent)
        self.title = QLabel()
        self.legend = LegendBar()
        self.legend.toggled.connect(self._apply_visibility)
        self.vb = PickViewBox(picker=self._on_click)
        self.axis = pg.DateAxisItem(orientation="bottom", utcOffset=display_offset_s(tz))
        self.widget = pg.PlotWidget(viewBox=self.vb, axisItems={"bottom": self.axis})
        self.plot_item = self.widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.15)
        self.plot_item.getAxis("left").enableAutoSIPrefix(False)
        self.plot_item.getAxis("left").setWidth(MIN_AXIS_WIDTH)
        self.vb.setAutoVisible(y=True)
        self.decimator = PlotDecimator(self.plot_item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addWidget(self.legend, stretch=1)
        layout.addLayout(header)
        layout.addWidget(self.widget, stretch=1)

        self.data: TimelineData | None = None
        self.items: dict = {}  # member -> PlotDataItem
        self.curves: dict = {}  # member -> Curve
        self.log_y = False
        self.lines_drawn = True
        self._night_items: list = []
        self.widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # --- data ----------------------------------------------------------------------------------

    def set_timezone(self, tz: str) -> None:
        self.axis.utcOffset = display_offset_s(tz)
        self.axis.picture = None
        self.axis.update()

    def clear(self, message: str = "") -> None:
        for item in self.items.values():
            self.plot_item.removeItem(item)
        self.items, self.curves, self.data = {}, {}, None
        self.decimator.clear()
        self.legend.set_members([], [], [])
        self.title.setText(message)

    def set_data(self, data: TimelineData, spec) -> None:
        self.clear()
        self.data = data
        q = data.quantity
        if spec.log_y != self.log_y:
            self.vb.setYRange(0, 1, padding=0)  # a linear range read as log (or back) would overflow
        self.log_y = spec.log_y
        self.decimator.log_y = spec.log_y
        self.vb.suppress_fit = True
        self.plot_item.setLogMode(y=spec.log_y)
        self.vb.suppress_fit = False
        variant = "" if data.variant == "none" else f" · {data.variant}"
        self.title.setText(f"<b>{q.title}</b>{variant}")
        self.plot_item.setLabel("left", q.unit if not q.categorical else "")

        symbol = None if spec.style == "lines" else "o"
        pen_on = spec.style != "points" and not q.categorical
        self.lines_drawn = pen_on
        members = [m for m in spec.members if m in data.curves]
        colors = [style.member_color(m, i) for i, m in enumerate(members)]
        for member, color in zip(members, colors):
            curve = data.curves[member]
            item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.2) if pen_on else None,
                                   symbol=symbol or ("o" if q.categorical else None), symbolSize=MARKER_SIZE,
                                   symbolBrush=color, symbolPen=None, connect="finite")
            self.plot_item.addItem(item)
            self.items[member] = item
            self.curves[member] = curve
            x, y = break_wraps(curve.x, curve.y) if q.circular else (curve.x, curve.y)
            self.decimator.add(item, x, y)
        self.legend.set_members(members, [style.member_label(m) for m in members], colors)

        left = self.plot_item.getAxis("left")
        if q.categorical and data.categories:
            left.setTicks([[(i, c) for i, c in enumerate(data.categories)]])
        else:
            left.setTicks(None)
        self.decimator.redraw()
        self.vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def axis_width_needed(self) -> int:
        """Pixels the left axis needs for its widest category label (pyqtgraph
        silently drops tick labels that don't fit the axis width).
        """
        if self.data is None or not self.data.quantity.categorical or not self.data.categories:
            return MIN_AXIS_WIDTH
        axis = self.plot_item.getAxis("left")
        metrics = QFontMetrics(axis.style.get("tickFont") or axis.font())
        widest = max(metrics.horizontalAdvance(c) for c in self.data.categories)
        return max(MIN_AXIS_WIDTH, widest + 30)  # room for the tick marks and the axis label

    def _apply_visibility(self) -> None:
        visible = self.legend.visible()
        for member, item in self.items.items():
            item.setVisible(member in visible)
        self.decimator.redraw()

    def visible_members(self) -> list:
        visible = self.legend.visible()
        return [m for m in self.items if m in visible]

    def set_night(self, night: Curve | None, on: bool) -> None:
        for item in self._night_items:
            self.plot_item.removeItem(item)
        self._night_items = []
        if night is None or not on:
            return
        is_night = np.nan_to_num(night.y) > 0.5
        edges = np.flatnonzero(np.diff(np.concatenate(([0], is_night.astype(np.int8), [0]))))
        x = night.x
        for start, end in zip(edges[0::2], edges[1::2]):
            region = pg.LinearRegionItem((x[start], x[end - 1] + SLOT_SECONDS), movable=False,
                                         brush=pg.mkBrush(style.NIGHT_BRUSH), pen=pg.mkPen(None))
            region.setZValue(-100)
            region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.plot_item.addItem(region, ignoreBounds=True)
            self._night_items.append(region)

    # --- picking -------------------------------------------------------------------------------

    def pick(self, view_x: float, view_y: float) -> list[tuple[float, int, object]]:
        """(pixel distance, slot, member) of each visible curve's nearest point
        (or drawn line) within the pick radius of a view position, nearest first.
        """
        px, py = self.vb.viewPixelSize()
        if not px or not py or self.data is None:
            return []
        circular = self.data.quantity.circular
        hits = []
        for member in self.visible_members():
            curve = self.curves[member]
            if curve.x.size == 0:
                continue
            ys = curve.y
            if self.log_y:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ys = np.where(ys > 0, np.log10(ys), np.nan)
            found = nearest_on_curve(curve.x, ys, view_x, view_y, px, py, PICK_RADIUS_PX, lines=self.lines_drawn,
                                     wrap=360.0 if circular else None)
            if found is not None:
                hits.append((found[0], int(curve.slots[found[1]]), member))
        return sorted(hits, key=lambda h: h[0])

    def _on_click(self, ev) -> bool:
        if ev.button() != Qt.MouseButton.LeftButton or ev.double():
            return False
        pos = self.vb.mapSceneToView(ev.scenePos())
        hits = self.pick(pos.x(), pos.y())
        if not hits:
            return False
        self._resolve(hits, ev.screenPos())
        return True

    def _resolve(self, hits, screen_pos) -> None:
        near = [h for h in hits if h[0] - hits[0][0] <= AMBIGUOUS_PX]
        if len(near) == 1:
            self.pointClicked.emit(near[0][1], near[0][2])
            return
        menu = QMenu(self)
        for _, slot, member in near:
            act = menu.addAction(style.member_label(member) or "value")
            act.triggered.connect(lambda _=False, s=slot, m=member: self.pointClicked.emit(s, m))
        menu.popup(screen_pos.toPoint())

    def _on_mouse_moved(self, scene_pos) -> None:
        if not self.vb.sceneBoundingRect().contains(scene_pos) or not self.curves:
            return
        pos = self.vb.mapSceneToView(scene_pos)
        slot = unix_to_slot(pos.x())
        parts = []
        cats = self.data.categories if self.data is not None else None
        for member in self.visible_members():
            curve = self.curves[member]
            i = int(np.searchsorted(curve.slots, slot))
            if i < curve.slots.size and curve.slots[i] == slot and np.isfinite(curve.y[i]):
                v = curve.y[i]
                text = cats[int(v)] if cats else f"{v:.4g}"
                parts.append(f"{style.member_label(member) or 'value'}: {text}")
        self.hovered.emit(slot, "  ".join(parts))
