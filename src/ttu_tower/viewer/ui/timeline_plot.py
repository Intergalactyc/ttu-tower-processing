"""One timeline panel: a quantity over the whole period, one curve per boom
(or boom pair), clickable through to the data behind a point.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFontMetrics
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QToolButton, QToolTip, QVBoxLayout, QWidget

from ttu_tower.viewer.decimate import break_wraps, nearest_on_curve
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, display_offset_s, unix_to_slot
from ttu_tower.viewer.timeline import Curve, FilteredOverlay, TimelineData
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.curves import FixedXViewBox, PlotDecimator

PICK_RADIUS_PX = 8.0
MIN_AXIS_WIDTH = 60
MARKER_SIZE = 5
AMBIGUOUS_PX = 2.0
FILTERED_COLOR = "#7a7a7a"
TAU_MARKS = {"capped": ("t1", "#ff7f0e"), "unresolved": ("d", "#9467bd"), "fallback": ("s", "#555555"),
             "none": ("x", "#000000")}
_TAU_GLYPHS = {"capped": "▲", "unresolved": "◆", "fallback": "■", "none": "✕"}


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

    def set_members(self, members: list, labels: list[str], colors: list, tips: list[str] | None = None) -> None:
        for btn in self.buttons.values():
            btn.deleteLater()
        self.buttons = {}
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for i, (member, label, color) in enumerate(zip(members, labels, colors)):
            btn = QToolButton()
            btn.setText(label or "value")
            btn.setToolTip((tips[i] + " · " if tips and tips[i] else "") + "click: show/hide · right-click: only this")
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
    filteredClicked = Signal(int, object)  # slot, member of a value tertiary filtered
    hovered = Signal(int, str)  # slot, values of the visible curves there

    def __init__(self, tz: str = "Etc/GMT+6", parent=None):
        super().__init__(parent)
        self.title = QLabel()
        self.title.setWordWrap(True)  # wraps rather than widening the window
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
        self.overlay_key = QLabel()
        self.overlay_key.setTextFormat(Qt.TextFormat.RichText)
        self.overlay_key.setWordWrap(True)
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addWidget(self.legend, stretch=1)
        header.addWidget(self.overlay_key)
        layout.addLayout(header)
        layout.addWidget(self.widget, stretch=1)

        self.data: TimelineData | None = None
        self.items: dict = {}  # member -> PlotDataItem
        self.curves: dict = {}  # member -> Curve
        self.log_y = False
        self.lines_drawn = True
        self._night_items: list = []
        self.filtered: FilteredOverlay | None = None
        self.filtered_items: dict = {}  # member -> PlotDataItem (the ring)
        self.filtered_crosses: dict = {}  # member -> PlotDataItem (the x inside it)
        self.tau_marks: dict = {}  # member -> (slots, statuses, x, y)
        self.tau_items: dict = {}  # member -> ScatterPlotItem
        self._tip_shown = False
        self.widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # --- data ----------------------------------------------------------------------------------

    def set_timezone(self, tz: str) -> None:
        self.axis.utcOffset = display_offset_s(tz)
        self.axis.picture = None
        self.axis.update()

    def clear(self, message: str = "") -> None:
        self.clear_overlays()
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
        self.legend.set_members(members, [style.member_short(m) for m in members], colors,
                                [style.member_label(m) for m in members])

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
        for items in (self.items, self.filtered_items, self.filtered_crosses, self.tau_items):
            for member, item in items.items():
                item.setVisible(member in visible)
        self.decimator.redraw()

    # --- overlays --------------------------------------------------------------------------------

    def _remove(self, *groups) -> None:
        for items in groups:
            for item in items.values():
                self.plot_item.removeItem(item)

    def clear_overlays(self) -> None:
        self._remove(self.filtered_items, self.filtered_crosses, self.tau_items)
        self.filtered, self.filtered_items, self.filtered_crosses, self.tau_marks, self.tau_items = None, {}, {}, {}, {}
        self.overlay_key.setText("")

    def _positive(self, y: np.ndarray) -> np.ndarray:
        """Values a log axis can show (others NaN)."""
        if not self.log_y:
            return y
        with np.errstate(invalid="ignore"):
            return np.where(y > 0, y, np.nan)

    def set_filtered(self, overlay: FilteredOverlay) -> None:
        """The values tertiary filtered, as they were before: a grey ring (filtered)
        around an x in the boom's color (whose value it was).
        """
        self._remove(self.filtered_items, self.filtered_crosses)
        self.filtered, self.filtered_items, self.filtered_crosses = overlay, {}, {}
        visible = self.legend.visible()
        for i, member in enumerate(self.items):
            curve = overlay.curves.get(member)
            if curve is None or curve.slots.size == 0:
                continue
            y = self._positive(curve.y)
            ok = np.isfinite(y)
            if not ok.any():
                continue
            ring = pg.PlotDataItem(curve.x[ok], y[ok], pen=None, symbol="o", symbolSize=MARKER_SIZE + 6,
                                   symbolBrush=None, symbolPen=pg.mkPen(FILTERED_COLOR, width=1.2))
            cross = pg.PlotDataItem(curve.x[ok], y[ok], pen=None, symbol="x", symbolSize=MARKER_SIZE + 1,
                                    symbolPen=None, symbolBrush=style.member_color(member, i))
            for z, it in ((-5, ring), (-4, cross)):
                it.setZValue(z)
                it.setVisible(member in visible)
                self.plot_item.addItem(it)
            self.filtered_items[member] = ring
            self.filtered_crosses[member] = cross
        self._update_key()

    def set_tau_status(self, marks: dict) -> None:
        """A marker on each value whose selected tau wasn't simply found."""
        for item in self.tau_items.values():
            self.plot_item.removeItem(item)
        self.tau_marks, self.tau_items = {}, {}
        visible = self.legend.visible()
        for member, (slots, statuses) in marks.items():
            curve = self.curves.get(member)
            if curve is None or slots.size == 0:
                continue
            y = _values_at(curve, slots)
            if self.filtered is not None and member in self.filtered.curves:
                y = np.where(np.isnan(y), _values_at(self.filtered.curves[member], slots), y)
            y = self._positive(y)
            ok = np.isfinite(y)
            slots, statuses, y = slots[ok], statuses[ok], y[ok]
            x = Curve(slots=slots, y=y).x
            spots = [{"pos": (xi, yi), "symbol": TAU_MARKS[st][0], "brush": pg.mkBrush(TAU_MARKS[st][1]),
                      "pen": pg.mkPen(TAU_MARKS[st][1]), "size": MARKER_SIZE + 4}
                     for xi, yi, st in zip(x, y, statuses)]
            item = pg.ScatterPlotItem(spots=spots)
            item.setZValue(5)
            item.setVisible(member in visible)
            self.plot_item.addItem(item)
            self.tau_items[member] = item
            self.tau_marks[member] = (slots, statuses, x, y)
        self._update_key()

    def _update_key(self) -> None:
        parts = []
        if self.filtered_items:
            parts.append(f"<span style='color:{FILTERED_COLOR}'>⊗ filtered (value before; x in the boom's color)</span>")
        if self.tau_items:
            parts.append("τ " + " ".join(f"<span style='color:{TAU_MARKS[s][1]}'>{g} {s}</span>"
                                         for s, g in _TAU_GLYPHS.items()))
        self.overlay_key.setText("  ".join(parts))

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

    def _view_y(self, y: np.ndarray) -> np.ndarray:
        if not self.log_y:
            return y
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y > 0, np.log10(y), np.nan)

    def pick(self, view_x: float, view_y: float) -> list[tuple[float, int, object, str]]:
        """(pixel distance, slot, member, kind) of each visible curve's nearest
        point (or drawn line) within the pick radius of a view position, nearest
        first; kind is "value", or "filtered" for a ghosted pre-filter value.
        """
        px, py = self.vb.viewPixelSize()
        if not px or not py or self.data is None:
            return []
        circular = self.data.quantity.circular
        hits = []
        for member in self.visible_members():
            layers = [(self.curves[member], "value", self.lines_drawn)]
            if member in self.filtered_items:
                layers.append((self.filtered.curves[member], "filtered", False))
            for curve, kind, lines in layers:
                if curve.x.size == 0:
                    continue
                found = nearest_on_curve(curve.x, self._view_y(curve.y), view_x, view_y, px, py, PICK_RADIUS_PX,
                                         lines=lines, wrap=360.0 if circular else None)
                if found is not None:
                    hits.append((found[0], int(curve.slots[found[1]]), member, kind))
        return sorted(hits, key=lambda h: (h[0], h[3] != "value"))

    def _on_click(self, ev) -> bool:
        if ev.button() != Qt.MouseButton.LeftButton or ev.double():
            return False
        pos = self.vb.mapSceneToView(ev.scenePos())
        hits = self.pick(pos.x(), pos.y())
        if not hits:
            return False
        self._resolve(hits, ev.screenPos())
        return True

    def _emit(self, slot: int, member, kind: str) -> None:
        (self.filteredClicked if kind == "filtered" else self.pointClicked).emit(slot, member)

    def _resolve(self, hits, screen_pos) -> None:
        near = [h for h in hits if h[0] - hits[0][0] <= AMBIGUOUS_PX]
        if len(near) == 1:
            self._emit(*near[0][1:])
            return
        menu = QMenu(self)
        for _, slot, member, kind in near:
            act = menu.addAction((style.member_label(member) or "value") + (" (filtered)" if kind == "filtered" else ""))
            act.triggered.connect(lambda _=False, s=slot, m=member, k=kind: self._emit(s, m, k))
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
            note = self._overlay_note(slot, member)
            if note:
                parts.append(f"{style.member_label(member)}: {note}")
        self.hovered.emit(slot, "  ".join(parts))
        self._overlay_tip(pos)

    def _overlay_note(self, slot: int, member) -> str:
        notes = []
        if member in self.filtered_items:
            pre = self.filtered.curves[member]
            i = int(np.searchsorted(pre.slots, slot))
            if i < pre.slots.size and pre.slots[i] == slot:
                reasons = self.filtered.reasons.get((slot, member), "")
                notes.append(f"filtered {pre.y[i]:.4g}" + (f" ({reasons})" if reasons else ""))
        if member in self.tau_marks:
            slots, statuses, _, _ = self.tau_marks[member]
            i = int(np.searchsorted(slots, slot))
            if i < slots.size and slots[i] == slot:
                notes.append(f"τ {statuses[i]}")
        return ", ".join(notes)

    def _overlay_tip(self, pos) -> None:
        """A tooltip naming why a ghosted value was filtered, or its tau status."""
        best = None
        if self.filtered_items or self.tau_items:
            px, py = self.vb.viewPixelSize()
            for member in self.visible_members():
                layers = []
                if member in self.filtered_items:
                    c = self.filtered.curves[member]
                    layers.append((c.x, c.y, c.slots))
                if member in self.tau_marks:
                    slots, _, x, y = self.tau_marks[member]
                    layers.append((x, y, slots))
                for x, y, slots in layers:
                    if not x.size:
                        continue
                    found = nearest_on_curve(x, self._view_y(y), pos.x(), pos.y(), px, py, PICK_RADIUS_PX, lines=False)
                    if found is not None and (best is None or found[0] < best[0]):
                        best = (found[0], int(slots[found[1]]), member)
        if best is not None:
            QToolTip.showText(QCursor.pos(), f"{style.member_label(best[2])}: {self._overlay_note(best[1], best[2])}",
                              self.widget)
            self._tip_shown = True
        elif self._tip_shown:
            QToolTip.hideText()
            self._tip_shown = False


def _values_at(curve: Curve, slots: np.ndarray) -> np.ndarray:
    """The curve's value at each slot (NaN where it has none)."""
    y = np.full(slots.size, np.nan)
    if curve.slots.size:
        i = np.clip(np.searchsorted(curve.slots, slots), 0, curve.slots.size - 1)
        hit = curve.slots[i] == slots
        y[hit] = curve.y[i[hit]]
    return y
