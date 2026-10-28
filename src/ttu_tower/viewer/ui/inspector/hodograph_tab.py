"""The slot's hodograph: each boom's vector-mean wind (u east, v north),
joined bottom to top, the mesonet's 10 m wind as a star apart, and, faintly,
the hodographs of the slots around it.
"""
import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ttu_tower.viewer import hodograph
from ttu_tower.viewer.hodograph import MESONET
from ttu_tower.viewer.ui import tables
from ttu_tower.viewer.ui.hodograph_view import draw_hodograph, draw_origin, prepare_plot

AROUND = 3  # slots either side: the hour around the slot


class HodographTab(QWidget):
    def __init__(self, index, runner, booms, parent=None):
        super().__init__(parent)
        self.index, self.runner, self.booms = index, runner, list(booms)
        self.slot = None
        self.uvs = None  # {member: UV} for the slots around
        self._loaded_slot = None
        self.around = QCheckBox("the hour around (faint)")
        self.around.setChecked(True)
        self.around.setToolTip(f"the hodographs of the {AROUND} slots either side, fainter the farther away")
        self.labels = QCheckBox("heights")
        self.labels.setChecked(True)
        for w in (self.around, self.labels):
            w.toggled.connect(lambda *_: self.redraw())
        self.plot = pg.PlotWidget()
        prepare_plot(self.plot)
        self.table = tables.new_table()
        self.info = QLabel()
        self.info.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.around)
        row.addWidget(self.labels)
        row.addStretch(1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.plot)
        split.addWidget(self.table)
        split.setSizes([600, 200])
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(split, stretch=1)
        layout.addWidget(self.info)

    def set_slot(self, slot: int) -> None:
        self.slot = slot
        if self.isVisible():
            self._load()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._load()

    def _load(self) -> None:
        if self.slot is None or self._loaded_slot == self.slot:
            return
        k = self.slot
        self.info.setText("reading the wind…")
        self.runner.submit(hodograph.slot_window, self.index, k, self.booms, AROUND, key=f"hodograph-{id(self)}",
                           label="reading the slot's wind", on_done=lambda uvs: self._loaded(k, uvs),
                           on_error=lambda exc, tb: self.info.setText(f"couldn't read the wind: {exc}"), priority=10)

    def _loaded(self, k, uvs) -> None:
        if not shiboken6.isValid(self) or k != self.slot:
            return
        self.uvs, self._loaded_slot = uvs, k
        self.redraw()

    def _at(self, k: int) -> dict:
        return {m: w.only(w.slots == k) for m, w in self.uvs.items()}

    def redraw(self) -> None:
        self.plot.clear()
        self.plot.getPlotItem().legend.clear()
        if self.uvs is None or self.slot is None:
            return
        draw_origin(self.plot)
        if self.around.isChecked():
            for d in range(-AROUND, AROUND + 1):
                if d == 0:
                    continue
                rows = hodograph.summary(self._at(self.slot + d))
                tower = rows[(rows["member"] != MESONET) & (rows["n"] > 0)]
                color = pg.mkColor("#1f77b4" if d < 0 else "#d62728")  # before blue, after red
                color.setAlpha(int(150 - 30 * (abs(d) - 1)))
                self.plot.plot(tower["u"].to_numpy(float), tower["v"].to_numpy(float), pen=pg.mkPen(color, width=1),
                               symbol="o", symbolSize=4, symbolBrush=color, symbolPen=None)
        rows = hodograph.summary(self._at(self.slot))
        draw_hodograph(self.plot, rows, labels="all" if self.labels.isChecked() else None, width=2.4)
        shown = rows[rows["n"] > 0].copy()
        shown.insert(0, "where", [MESONET if m == MESONET else f"b{m}" for m in shown["member"]])
        tables.fill(self.table, shown.drop(columns=["member", "n", "mean_speed", "steadiness"]).rename(columns={
            "height": "height [m]", "u": "u [m/s]", "v": "v [m/s]", "speed": "speed [m/s]", "from": "from [°]"}))
        missing = [f"b{b}" for b in self.booms if b not in set(shown["member"])]
        note = f" · no wind at {', '.join(missing)}" if missing else ""
        around = (f" · faint: the {AROUND} slots before (blue) and after (red)" if self.around.isChecked() else "")
        self.info.setText("the slot's vector-mean wind at each boom (final values); ★ the mesonet's 10 m wind"
                          + around + note)
