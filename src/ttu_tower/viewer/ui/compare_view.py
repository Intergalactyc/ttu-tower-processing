"""This run against another: for a panel's quantity, per boom, how many values
each run has and how they agree where both do (bias, RMSE, share identical,
r), and a scatter of one against the other for a chosen boom.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ttu_tower.viewer import compare
from ttu_tower.viewer.ui import style, tables


class CompareView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.datas = [None, None]
        self.others = [None, None]  # the other run's TimelineData per panel (or a reason it has none)
        self.tag = None
        self.x_range = None
        self.frame = None
        self.source = QComboBox()
        self.source.addItem("panel A", 0)
        self.source.addItem("panel B", 1)
        self.member = QComboBox()
        for w in (self.source, self.member):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.table = tables.new_table()
        self.table.setToolTip("N: finite values in range; the rest over the slots both runs have a value for")
        self.plot = pg.PlotWidget()
        self.plot.getPlotItem().export_layout = "per curve"  # CSV: x and y columns per curve
        self.plot.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        for side in ("left", "bottom"):
            self.plot.getPlotItem().getAxis(side).enableAutoSIPrefix(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        for w in (self.source, self.member):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.title)
        layout.addWidget(self.table)
        layout.addWidget(self.plot, stretch=1)
        tables.fit_rows(self.table)

    def sizeHint(self) -> QSize:
        return QSize(600, 600)

    def figure_spec(self) -> tuple[str, dict] | None:
        i = self.source.currentData()
        data, other = self.datas[i], self.others[i]
        if self.frame is None or data is None or other is None or isinstance(other, str):
            return None
        m = self.member.currentData()
        if m not in data.curves or m not in other.curves:
            return None
        _, ya, yb = compare.paired(data.curves[m], other.curves[m], self.x_range)
        return "compare", {"x": ya, "y": yb, "xlabel": f"this run: {data.quantity.title}",
                           "ylabel": f"{self.tag}", "title": f"{self.member.currentText()}"}

    def set_sources(self, datas, others, tag, x_range) -> None:
        self.datas, self.others, self.tag, self.x_range = list(datas), list(others), tag, x_range
        self.redraw()

    def redraw(self) -> None:
        self.plot.clear()
        self.frame = None
        i = self.source.currentData()
        data, other = self.datas[i], self.others[i]
        if self.tag is None:
            self.title.setText("choose a run to compare with in the toolbar")
            self._empty()
            return
        if data is None:
            self.title.setText("choose a quantity in the panel")
            self._empty()
            return
        if isinstance(other, str) or other is None:
            self.title.setText(f"<b>{data.quantity.title}</b>: {other or 'reading the other run…'}")
            self._empty()
            return
        members = list(data.curves)
        self._sync_members(members)
        scope = "whole period" if self.x_range is None else "visible range"
        self.title.setText(f"<b>{data.quantity.title}</b>" + (f" · {data.variant}" if data.variant != "none" else "")
                           + f" — this run against <b>{self.tag}</b>, {scope}")
        self.frame = compare.table(data, other, members, self.x_range, label=lambda m: style.member_label(m) or "value")
        tables.fill(self.table, self.frame, index=True,
                    fmt=lambda col, v: f"{v:.0f}" if col.startswith("N") and np.isfinite(v) else None)
        tables.fit_rows(self.table)
        self._scatter(data, other)

    def _empty(self) -> None:
        tables.fill(self.table, pd.DataFrame())
        tables.fit_rows(self.table)

    def _sync_members(self, members) -> None:
        if [self.member.itemData(i) for i in range(self.member.count())] == members:
            return
        current = self.member.currentData()
        self.member.blockSignals(True)
        self.member.clear()
        for m in members:
            self.member.addItem(style.member_label(m) or "value", m)
        self.member.setCurrentIndex(max(self.member.findData(current), 0))
        self.member.blockSignals(False)

    def _scatter(self, data, other) -> None:
        m = self.member.currentData()
        a, b = data.curves.get(m), other.curves.get(m)
        if a is None or b is None:
            return
        _, ya, yb = compare.paired(a, b, self.x_range)
        if ya.size == 0:
            return
        c = pg.mkColor("#1f4e9c")
        c.setAlpha(140)
        self.plot.addItem(pg.ScatterPlotItem(ya, yb, size=4, pen=None, brush=pg.mkBrush(c)))
        lo, hi = float(min(ya.min(), yb.min())), float(max(ya.max(), yb.max()))
        self.plot.plot([lo, hi], [lo, hi], pen=pg.mkPen("#555555", width=1, style=Qt.PenStyle.DashLine))
        q = data.quantity
        self.plot.setLabel("bottom", f"this run · {q.title}")
        self.plot.setLabel("left", f"{self.tag}")
