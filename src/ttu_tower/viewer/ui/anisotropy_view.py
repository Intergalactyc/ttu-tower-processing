"""One boom's anisotropy states on the barycentric map over the viewed
interval (or the whole period): a hexbin heatmap as the old plotting code drew
it (viridis counts clipped to the triangle), or the points; one panel per
stability class (or for the brushed selection and the rest) when split; the
share of each class; and a lasso on any panel that selects slots.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import anisotropy
from ttu_tower.viewer.brush import in_polygon
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import in_class
from ttu_tower.viewer.ui import style, tables
from ttu_tower.viewer.ui.inspector.anisotropy_tab import draw_frame
from ttu_tower.viewer.ui.scatter import LassoViewBox
from ttu_tower.viewer.ui.timeline_plot import SELECTED_COLOR

_VARIANTS = (("mrd", "mrd"), ("naive", "naive"), ("mrd_unexcised", "mrd unexcised"))
_VIRIDIS = pg.colormap.get("viridis")
_POINT = "#1f4e9c"
_REST = "#7a7a7a"


class HexbinItem(pg.GraphicsObject):
    """Filled hexagons coloured by count (viridis), clipped to the map's triangle."""

    def __init__(self, cx, cy, counts, hexagon, log: bool = False):
        super().__init__()
        values = np.log10(counts + 1.0) if log else counts.astype(float)
        self.top = float(values.max()) if values.size else 0.0
        scaled = values / self.top if self.top > 0 else np.zeros_like(values)
        colors = _VIRIDIS.map(scaled, mode="qcolor")
        self.hexes = [(QPolygonF([QPointF(x + dx, y + dy) for dx, dy in hexagon]), c)
                      for x, y, c in zip(cx, cy, colors)]
        corners = anisotropy.CORNERS
        self.clip = QPainterPath()
        self.clip.addPolygon(QPolygonF([QPointF(*corners[k]) for k in ("1C", "2C", "3C")]))
        self.clip.closeSubpath()

    def paint(self, painter, *_):
        painter.setClipPath(self.clip)
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        for polygon, color in self.hexes:
            painter.setBrush(QBrush(color))
            painter.drawPolygon(polygon)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, 1.0, anisotropy.SQRT3 / 2)


class AnisotropyMapView(QWidget):
    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = None
        self.x_range = None
        self.stability = None
        self.selection = np.empty(0, dtype=np.int64)
        self.states = None  # (slots, x, y) of the chosen boom and variant, whole period
        self.panels: list = []  # (name, PlotItem, mask) as drawn
        self.brushed = None  # set by the main window: callable(slots, mode)
        self.boom = QComboBox()
        self.variant = QComboBox()
        for key, label in _VARIANTS:
            self.variant.addItem(label, key)
        for w in (self.boom, self.variant):
            w.currentIndexChanged.connect(lambda *_: self.reload())
        self.display = QComboBox()
        for label in ("heatmap", "points"):
            self.display.addItem(label, label)
        self.display.setToolTip("heatmap: slots per hexagon (40 across, as the old plotting code), viridis")
        self.split = QComboBox()
        self.split.addItem("all together", "none")
        self.split.addItem("one panel per stability class", "stability")
        self.split.addItem("brushed selection | the rest", "selection")
        self.split.setToolTip("one triangle for every slot, one per stability class, or two: the slots selected by "
                              "linked brushing (Shift-drag on a timeline, scatter or this map; a Diurnal cell; "
                              "Find's Select on plots) beside all the others")
        self.log_color = QCheckBox("log colour")
        self.regions = QCheckBox("class regions")
        for w in (self.display, self.split):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        for w in (self.log_color, self.regions):
            w.toggled.connect(lambda *_: self.redraw())
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setToolTip("Shift-drag (or drag in brush mode) to lasso slots on any panel; Ctrl adds, "
                                 "Alt removes")
        self.table = tables.new_table()
        self.table.setToolTip("share of each anisotropy class (the pipeline's regions) per panel, %")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        for w in (self.boom, self.variant, self.display, self.split):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        row = QHBoxLayout()
        for w in (self.log_color, self.regions):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.graphics, stretch=1)
        layout.addWidget(self.info)
        layout.addWidget(self.table)
        tables.fit_rows(self.table)

    def sizeHint(self) -> QSize:
        return QSize(600, 600)

    def set_run(self, run, index, booms) -> None:
        self.run, self.index, self.states = run, index, None
        self.boom.blockSignals(True)
        self.boom.clear()
        for b in booms:
            self.boom.addItem(f"boom {b} ({HEIGHTS[b]:g} m)", b)
        self.boom.setCurrentIndex(max(self.boom.findData(5), 0))
        self.boom.blockSignals(False)

    def set_range(self, x_range, stability) -> None:
        self.x_range, self.stability = x_range, stability
        if self.states is None:
            self.reload()
        else:
            self.redraw()

    def set_selection(self, slots) -> None:
        self.selection = slots
        if slots.size and self.split.currentData() == "none":
            self.split.setCurrentIndex(self.split.findData("selection"))
        else:
            self.redraw()

    def reload(self) -> None:
        if self.run is None or self.boom.currentData() is None:
            return
        key = (self.boom.currentData(), self.variant.currentData())
        self.info.setText("reading the boom's anisotropy…")
        self.runner.submit(anisotropy.boom_map, self.run, self.index, *key, key=f"anisomap-{id(self)}",
                           label="reading anisotropy", on_done=lambda states: self._loaded(key, states),
                           on_error=lambda exc, tb: self.info.setText(f"couldn't read the anisotropy: {exc}"))

    def _loaded(self, key, states) -> None:
        if not shiboken6.isValid(self) or key != (self.boom.currentData(), self.variant.currentData()):
            return
        self.states = states
        self.redraw()

    def shown(self):
        """(slots, x, y) in range."""
        slots, x, y = self.states
        if self.x_range is not None:
            t = slot_to_unix(slots)
            keep = (t >= self.x_range[0]) & (t < self.x_range[1])
            slots, x, y = slots[keep], x[keep], y[keep]
        return slots, x, y

    def groups(self, slots) -> list:
        """(name, mask, colour) per panel."""
        by = self.split.currentData()
        if by == "stability" and self.stability is not None:
            names = self.stability[1]
            return [(n, in_class(self.stability, slots, n), style.stability_color(n, i).name())
                    for i, n in enumerate(names)]
        if by == "selection" and self.selection.size:
            chosen = np.isin(slots, self.selection)
            return [("selected", chosen, SELECTED_COLOR), ("the rest", ~chosen, _REST)]
        return [("all slots", np.ones(slots.size, dtype=bool), _POINT)]

    def _enable_controls(self) -> None:
        model = self.split.model()
        model.item(1).setEnabled(self.stability is not None)
        model.item(1).setToolTip("" if self.stability is not None else "the run has no stability classes")
        model.item(2).setEnabled(self.selection.size > 0)
        model.item(2).setToolTip("" if self.selection.size else "needs a brushed selection first: Shift-drag on a "
                                 "timeline, the scatter or this map (or click a Diurnal cell)")
        if not model.item(self.split.currentIndex()).isEnabled():
            self.split.blockSignals(True)
            self.split.setCurrentIndex(0)
            self.split.blockSignals(False)
        heatmap = self.display.currentData() == "heatmap"
        self.log_color.setEnabled(heatmap)
        self.regions.setEnabled(not heatmap)
        self.regions.setToolTip("the heatmap covers the map; the regions show under points" if heatmap
                                else "the pipeline's anisotropy classes, shaded under the points")

    def redraw(self) -> None:
        self._enable_controls()
        self.graphics.clear()
        self.graphics.ci.setContentsMargins(0, 0, 0, 0)
        self.graphics.ci.setSpacing(0)
        self.panels = []
        if self.states is None:
            return
        slots, x, y = self.shown()
        groups = self.groups(slots)
        heatmap = self.display.currentData() == "heatmap"
        columns = 1 if len(groups) == 1 else 2
        for i, (name, mask, color) in enumerate(groups):
            cell = self.graphics.addLayout(row=i // columns, col=i % columns)
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(0)
            vb = LassoViewBox(lambda px, py, mode, m=mask: self._on_lasso(px, py, mode, m))
            plot = cell.addPlot(viewBox=vb)
            plot.setContentsMargins(0, 0, 0, 0)
            plot.setAspectLocked(True)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.hideButtons()
            plot.setTitle(f"{name} ({int(mask.sum())})", size="8pt")
            short = len(groups) > 1
            draw_frame(plot, self.regions.isChecked() and not heatmap, short_labels=short)
            if heatmap:
                hexes = HexbinItem(*anisotropy.hexbin(x[mask], y[mask]), log=self.log_color.isChecked())
                hexes.setZValue(-8)
                plot.addItem(hexes)
                bar = pg.ColorBarItem(values=(0, max(hexes.top, 1)), colorMap=_VIRIDIS, interactive=False, width=8)
                cell.addItem(bar, row=0, col=1)
            elif mask.any():
                c = pg.mkColor(color)
                c.setAlpha(170)
                plot.addItem(pg.ScatterPlotItem(x[mask], y[mask], size=3 if mask.sum() > 5000 else 5, pen=None,
                                                brush=pg.mkBrush(c)))
            margin = 0.05 if short else 0.1  # room for the corner labels, and no more
            plot.setRange(xRange=(-margin, 1 + margin), yRange=(-0.1, anisotropy.SQRT3 / 2 + 0.1), padding=0)
            self.panels.append((name, plot, mask))
        scope = "whole period" if self.x_range is None else "visible range"
        colour = ("colour: log10(slots + 1) per hexagon" if self.log_color.isChecked() else
                  "colour: slots per hexagon") if heatmap else ""
        self.info.setText(f"{slots.size} slots ({scope}) · {self.boom.currentText()} · {self.variant.currentText()}"
                          + (f" · {colour}" if colour else ""))
        self._shares(x, y, groups)

    def _shares(self, x, y, groups) -> None:
        cases = anisotropy.classify(x, y)
        rows = {}
        for name, mask, _ in groups:
            n = int(mask.sum())
            if n:
                counts = pd.Series(cases[mask]).value_counts()
                rows[f"{name} ({n})"] = {c: 100.0 * counts.get(c, 0) / n for c in anisotropy.CASES}
        frame = pd.DataFrame.from_dict(rows, orient="index")
        tables.fill(self.table, frame.loc[:, (frame.fillna(0) > 0).any()] if not frame.empty else frame, index=True,
                    fmt=lambda col, v: f"{v:.1f}" if isinstance(v, float) and np.isfinite(v) else None)
        tables.fit_rows(self.table)

    def _on_lasso(self, px, py, mode: str, mask=None) -> None:
        """The slots inside a lasso drawn on one panel (among that panel's slots)."""
        if self.states is None or self.brushed is None:
            return
        slots, x, y = self.shown()
        inside = in_polygon(x, y, px, py)
        if mask is not None and mask.size == inside.size:
            inside &= mask
        self.brushed(slots[inside], mode)

    def figure_spec(self) -> tuple[str, dict] | None:
        if self.states is None:
            return None
        slots, x, y = self.shown()
        heatmap = self.display.currentData() == "heatmap"
        return "anisotropy", {
            "panels": [{"title": f"{name} ({int(mask.sum())})", "color": color, "x": x[mask], "y": y[mask]}
                       for name, mask, color in self.groups(slots)],
            "heatmap": heatmap, "log": self.log_color.isChecked(), "gridsize": anisotropy.HEX_GRIDSIZE,
            "regions": anisotropy.region_triangles() if self.regions.isChecked() and not heatmap else {},
            "region_colors": anisotropy.CASE_COLORS, "corners": anisotropy.CORNERS,
            "strain": anisotropy.plane_strain_line(), "title": self.info.text()}
