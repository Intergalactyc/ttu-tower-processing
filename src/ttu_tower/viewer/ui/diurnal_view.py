"""A heatmap of one panel's quantity by hour of day and month, for one boom
(or pair), over the viewed interval or the whole period. Clicking a cell
selects its slots for linked brushing.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ttu_tower.viewer import diurnal
from ttu_tower.viewer.ui import style

_SEQUENTIAL = pg.colormap.get("viridis")
_CYCLIC = pg.colormap.get("CET-C6")  # directions wrap around


class _CellViewBox(pg.ViewBox):
    def __init__(self, on_click, **kwargs):
        super().__init__(**kwargs)
        self.on_click = on_click

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            p = self.mapToView(ev.pos())
            if self.on_click(p.x(), p.y()):
                ev.accept()
                return
        super().mouseClickEvent(ev)


class DiurnalView(QWidget):
    cellSelected = Signal(object)  # the clicked cell's slots

    def __init__(self, parent=None):
        super().__init__(parent)
        self.datas = [None, None]
        self.x_range = None
        self.tz = "UTC"
        self.values = self.counts = None
        self.source = QComboBox()
        self.source.addItem("panel A", 0)
        self.source.addItem("panel B", 1)
        self.source.currentIndexChanged.connect(lambda *_: self._sync_members())
        self.member = QComboBox()
        self.stat = QComboBox()
        for s in diurnal.STATS:
            self.stat.addItem(s, s)
        self.category = QComboBox()
        self.category.setToolTip("categories: the share of each cell's slots in this one")
        for w in (self.member, self.stat, self.category):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.vb = _CellViewBox(self._clicked)
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        item = self.plot.getPlotItem()
        item.getAxis("bottom").setTicks([[(m + 0.5, name) for m, name in enumerate(diurnal.MONTHS)]])
        item.getAxis("left").setTicks([[(h + 0.5, f"{h:02d}") for h in range(0, 24, 3)]])
        item.setLabel("left", "hour of day")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.colorbar = pg.ColorBarItem(colorMap=_SEQUENTIAL, interactive=False, width=12)
        self.colorbar.setImageItem(self.image, insert_in=item)
        self.plot.setXRange(0, 12, padding=0)
        self.plot.setYRange(0, 24, padding=0)
        self.plot.scene().sigMouseMoved.connect(self._hover)
        self.note = QLabel("click a cell to select its slots")
        self.note.setStyleSheet("color:#666;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        for w in (self.source, self.member, self.stat, self.category):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.title)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.note)

    def sizeHint(self) -> QSize:
        return QSize(600, 600)

    def figure_spec(self) -> tuple[str, dict] | None:
        data = self._data()
        if self.values is None or data is None:
            return None
        q = data.quantity
        cyclic = q.circular and self.stat.currentData() in ("median", "mean") and not q.categorical
        title = self.title.text().replace("<b>", "").replace("</b>", "")
        return "diurnal", {"values": self.values, "title": title, "cyclic": cyclic, "months": diurnal.MONTHS,
                           "clabel": q.unit if not q.categorical else "share", "ylabel": f"hour of day ({self.tz})"}

    def set_sources(self, datas, x_range, tz: str) -> None:
        changed = [d is not o for d, o in zip(datas, self.datas)]
        self.datas, self.x_range, self.tz = list(datas), x_range, tz
        if changed[self.source.currentData()]:
            self._sync_members(redraw=False)
        self.redraw()

    def _data(self):
        return self.datas[self.source.currentData()]

    def _sync_members(self, redraw: bool = True) -> None:
        data = self._data()
        current = self.member.currentData()
        self.member.blockSignals(True)
        self.member.clear()
        if data is not None:
            for m in data.curves:
                self.member.addItem(style.member_label(m) or "value", m)
            self.member.setCurrentIndex(max(self.member.findData(current), 0))
        self.member.blockSignals(False)
        self.category.blockSignals(True)
        self.category.clear()
        if data is not None and data.quantity.categorical:
            for i, c in enumerate(data.categories or []):
                self.category.addItem(c, i)
        self.category.blockSignals(False)
        if redraw:
            self.redraw()

    def _curve(self):
        data = self._data()
        if data is None or self.member.count() == 0:
            return None
        return data.curves.get(self.member.currentData())

    def redraw(self) -> None:
        data, curve = self._data(), self._curve()
        self.values = self.counts = None
        if data is None or curve is None:
            self.image.clear()
            self.title.setText("choose a quantity in the panel")
            return
        q = data.quantity
        categorical = q.categorical
        self.category.setVisible(categorical)
        self.stat.setEnabled(not categorical)
        self.stat.setToolTip("categories show the share in the chosen one" if categorical else
                             "directions: vector mean (median and mean) and Yamartino σ")
        category = self.category.currentData() if categorical else None
        stat = "mean" if categorical else self.stat.currentData()
        self.values, self.counts = diurnal.grid(curve, self.tz, self.x_range, stat, q.circular, category)
        cyclic = q.circular and stat in ("median", "mean") and not categorical
        self.colorbar.setColorMap(_CYCLIC if cyclic else _SEQUENTIAL)
        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            self.image.clear()
        else:
            lo, hi = (0.0, 360.0) if cyclic else (float(finite.min()), float(finite.max()))
            self.image.setImage(self.values, levels=(lo, hi if hi > lo else lo + 1))
            self.image.setRect(0, 0, 12, 24)
            self.colorbar.setLevels((lo, hi if hi > lo else lo + 1))
        what = (f"share in {self.category.currentText()}" if categorical else stat)
        scope = "whole period" if self.x_range is None else "visible range"
        self.title.setText(f"<b>{q.title}</b> · {self.member.currentText()} · {what} by hour ({self.tz}) and month"
                           f" · {scope}")

    def _cell(self, x: float, y: float):
        m, h = int(np.floor(x)), int(np.floor(y))
        return (h, m) if 0 <= m < 12 and 0 <= h < 24 else None

    def _hover(self, pos) -> None:
        if self.values is None or not self.vb.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        cell = self._cell(p.x(), p.y())
        if cell is None:
            return
        h, m = cell
        v = self.values[h, m]
        self.plot.setToolTip(f"{diurnal.MONTHS[m]}, {h:02d}:00–{h:02d}:59: "
                             + (f"{v:.4g}" if np.isfinite(v) else "—") + f" (N {self.counts[h, m]})")

    def _clicked(self, x: float, y: float) -> bool:
        curve, cell = self._curve(), self._cell(x, y)
        if curve is None or cell is None or self.counts is None or self.counts[cell] == 0:
            return False
        h, m = cell
        self.cellSelected.emit(diurnal.cell_slots(curve, self.tz, self.x_range, h, m + 1))
        return True
