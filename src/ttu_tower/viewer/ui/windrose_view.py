"""Wind roses over the viewed interval (or the whole period): one boom's -
final, or before the shadow-sector removal and any filtering - and the mesonet
station's, on one radial scale, with lines of constant bearing (by default the
shadow sector's edges).
"""
import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGraphicsPolygonItem, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import windrose

_VIRIDIS = pg.colormap.get("viridis")
_COMPASS = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
_BEARING_LINE = "#d62728"


def _speed_labels(edges) -> list[str]:
    return [f"{a:g}–{b:g}" if np.isfinite(b) else f"≥ {a:g}" for a, b in zip(edges[:-1], edges[1:])]


class RosePlot(pg.PlotWidget):
    """One rose: stacked wedges per direction sector, north up, clockwise."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAspectLocked(True)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        self.title = ""
        self.table = None

    def _xy(self, r: float, bearing: float) -> QPointF:
        t = np.radians(bearing)
        return QPointF(r * np.sin(t), r * np.cos(t))

    @staticmethod
    def outer(table: np.ndarray | None) -> float:
        """The longest sector's total (%)."""
        return float(table.sum(axis=1).max()) if table is not None and table.sum() else 0.0

    def draw(self, table: np.ndarray | None, title: str, outer: float | None = None, bearings=()) -> None:
        """`table`: sectors x speed-bin percentages (None: nothing to show);
        `outer`: the radius to scale to (to share one scale between roses);
        `bearings`: where to draw dashed lines of constant direction.
        """
        self.clear()
        self.table = table
        self.setTitle(title, size="9pt")
        if table is None or not table.sum():
            text = pg.TextItem("no wind data in range", color="#666", anchor=(0.5, 0.5))
            self.addItem(text)
            self.setRange(xRange=(-1, 1), yRange=(-1, 1))
            return
        sectors, bins = table.shape
        width = 360.0 / sectors
        outer = max(outer or 0.0, self.outer(table))
        step = next(s for s in (1, 2, 2.5, 5, 10, 20, 25, 50, 100) if outer / s <= 5)
        for ring in np.arange(step, outer + step, step):
            angles = np.linspace(0, 360, 121)
            self.plot([ring * np.sin(np.radians(a)) for a in angles], [ring * np.cos(np.radians(a)) for a in angles],
                      pen=pg.mkPen("#cccccc", width=1))
            label = pg.TextItem(f"{ring:g} %", color="#888", anchor=(0, 1))
            p = self._xy(ring, 22.5)
            label.setPos(p.x(), p.y())
            self.addItem(label)
        rim = np.arange(step, outer + step, step)[-1]
        for name, bearing in _COMPASS.items():
            self.plot([0, rim * np.sin(np.radians(bearing))], [0, rim * np.cos(np.radians(bearing))],
                      pen=pg.mkPen("#dddddd", width=1))
            label = pg.TextItem(name, color="#000", anchor=(0.5, 0.5))
            p = self._xy(rim * 1.1, bearing)
            label.setPos(p.x(), p.y())
            self.addItem(label)
        for bearing in bearings:
            end = self._xy(rim * 1.02, bearing)
            line = self.plot([0, end.x()], [0, end.y()], pen=pg.mkPen(_BEARING_LINE, width=1.5, style=Qt.PenStyle.DashLine))
            line.setZValue(10)
            label = pg.TextItem(f"{bearing:g}°", color=_BEARING_LINE, anchor=(0.5, 0.5))
            p = self._xy(rim * 1.1, bearing)
            label.setPos(p.x(), p.y())
            self.addItem(label)
        for i in range(sectors):
            centre = i * width
            a0, a1 = centre - width * 0.45, centre + width * 0.45
            arc = np.linspace(a0, a1, 8)
            r0 = 0.0
            for j in range(bins):
                r1 = r0 + table[i, j]
                if r1 > r0:
                    points = [self._xy(r1, a) for a in arc] + [self._xy(r0, a) for a in arc[::-1]]
                    item = QGraphicsPolygonItem(QPolygonF(points))
                    item.setBrush(QBrush(_VIRIDIS.map(j / max(bins - 1, 1), mode="qcolor")))
                    item.setPen(QPen(Qt.PenStyle.NoPen))
                    item.setToolTip(f"from {centre:g}° ± {width / 2:g}°, {self._speed_label(j)} m/s: {table[i, j]:.2f} %")
                    self.addItem(item)
                r0 = r1
        self.setRange(xRange=(-rim * 1.2, rim * 1.2), yRange=(-rim * 1.2, rim * 1.2), padding=0)

    def _speed_label(self, j: int) -> str:
        return _speed_labels(windrose.SPEED_EDGES)[j]


def _load(run, index, boom: int, filtered: bool):
    """Runs on a worker thread."""
    return windrose.tower_wind(run, index, boom, filtered), windrose.mesonet_wind(run, index)


class WindRoseView(QWidget):
    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = None
        self.x_range = None
        self.tower = self.mesonet = None
        self.boom = QComboBox()
        self.boom.setToolTip("the boom whose slot-mean wind is shown")
        self.boom.currentIndexChanged.connect(lambda *_: self.reload())
        self.filtering = QComboBox()
        self.filtering.addItem("filtered (final values)", True)
        self.filtering.addItem("unfiltered (first-layer means)", False)
        self.filtering.setToolTip("unfiltered: the slot means on the first-layer mask - before the direction "
                                  "(shadow sector) and bounce removals and before tertiary's filtering - so the "
                                  "shadow sector shows what was removed")
        self.filtering.currentIndexChanged.connect(lambda *_: self.reload())
        self.bearings = QLineEdit()
        self.bearings.setPlaceholderText("lines at … °")
        self.bearings.setToolTip("bearings (deg, comma-separated) to mark with dashed lines; by default the shadow "
                                 "sector's edges")
        self.bearings.setMaximumWidth(110)
        self.bearings.editingFinished.connect(self.redraw)
        self.same_slots = QCheckBox("mesonet: same slots")
        self.same_slots.setToolTip("draw the mesonet rose from only the slots the tower rose has")
        self.same_slots.toggled.connect(lambda *_: self.redraw())
        self.sectors = QComboBox()
        for n in (8, 16, 36):
            self.sectors.addItem(f"{n} sectors", n)
        self.sectors.setCurrentIndex(2)
        self.sectors.currentIndexChanged.connect(lambda *_: self.redraw())
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.tower_plot, self.meso_plot = RosePlot(), RosePlot()
        labels = _speed_labels(windrose.SPEED_EDGES)
        key = QLabel("speed [m/s]: " + "  ".join(
            f"<span style='color:{_VIRIDIS.map(j / (len(labels) - 1), mode='qcolor').name()}'>■</span> {s}"
            for j, s in enumerate(labels)))
        key.setWordWrap(True)
        row = QHBoxLayout()
        for w in (self.boom, self.filtering, self.sectors):
            row.addWidget(w)
        row.addStretch(1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("lines at"))
        row2.addWidget(self.bearings)
        row2.addWidget(QLabel("°"))
        row2.addWidget(self.same_slots)
        row2.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(row)
        layout.addLayout(row2)
        layout.addWidget(key)
        layout.addWidget(self.tower_plot, stretch=1)
        layout.addWidget(self.meso_plot, stretch=1)
        layout.addWidget(self.info)

    def set_run(self, run, index, booms) -> None:
        self.run, self.index = run, index
        self.tower = self.mesonet = None
        self.boom.blockSignals(True)
        self.boom.clear()
        for b in booms:
            self.boom.addItem(f"boom {b} ({HEIGHTS[b]:g} m)", b)
        default = 7 if 7 in booms else booms[len(booms) // 2]  # near the mesonet's 10 m
        self.boom.setCurrentIndex(max(self.boom.findData(default), 0))
        self.boom.blockSignals(False)
        shadow = run.cfg.qc.second_layer.shadow_sector if run.cfg is not None else ()
        self.bearings.setText(", ".join(f"{b:g}" for b in shadow))

    def set_range(self, x_range) -> None:
        """(x0, x1) unix seconds, or None for the whole period."""
        self.x_range = x_range
        if self.tower is None:
            self.reload()
        else:
            self.redraw()

    def reload(self) -> None:
        if self.run is None or self.boom.currentData() is None:
            return
        key = (self.boom.currentData(), self.filtering.currentData())
        self.info.setText("reading the wind…")
        self.runner.submit(_load, self.run, self.index, *key, key=f"windrose-{id(self)}", label="reading the wind",
                           on_done=lambda result: self._loaded(key, result),
                           on_error=lambda exc, tb: self.info.setText(f"could not read the wind: {exc}"))

    def _loaded(self, key, result) -> None:
        if not shiboken6.isValid(self) or key != (self.boom.currentData(), self.filtering.currentData()):
            return
        self.tower, self.mesonet = result
        self.redraw()

    def redraw(self) -> None:
        if self.tower is None:
            return
        try:
            bearings = windrose.parse_bearings(self.bearings.text())
            self.bearings.setStyleSheet("")
        except ValueError:
            bearings = []
            self.bearings.setStyleSheet("background:#f6c6c6;")
        n = self.sectors.currentData()
        scope = "whole period" if self.x_range is None else "viewed interval"
        tower = windrose.in_range(self.tower, self.x_range)
        which = "filtered" if self.filtering.currentData() else "first layer (unfiltered)"
        b = self.boom.currentData()
        tower_table = windrose.rose(tower.ws, tower.wd, n) if tower.ws.size else None
        meso_table, meso = None, None
        if self.mesonet is not None:
            meso = windrose.in_range(self.mesonet, self.x_range)
            if self.same_slots.isChecked():
                meso = windrose.only_slots(meso, tower.slots)
            meso_table = windrose.rose(meso.ws, meso.wd, n) if meso.ws.size else None
        outer = max(RosePlot.outer(tower_table), RosePlot.outer(meso_table))  # one scale for both
        self.tower_plot.draw(tower_table, f"b{b} ({HEIGHTS[b]:g} m), {which} — {tower.ws.size} slots", outer, bearings)
        parts = [f"{scope}: b{b} mean {tower.ws.mean():.2f} m/s" if tower.ws.size else f"{scope}: no b{b} wind"]
        self.same_slots.setEnabled(meso is not None)
        self.same_slots.setToolTip("draw the mesonet rose from only the slots the tower rose has" if meso is not None
                                   else "this run has no mesonet data")
        if meso is None:
            self.meso_plot.draw(None, "mesonet: not merged into this run")
        else:
            same = ", tower's slots only" if self.same_slots.isChecked() else ""
            self.meso_plot.draw(meso_table, f"mesonet (10 m){same} — {meso.ws.size} slots", outer, bearings)
            if meso.ws.size:
                parts.append(f"mesonet mean {meso.ws.mean():.2f} m/s")
        self.info.setText(" · ".join(parts) + " · both roses share one radial scale · hover a wedge for its share")
