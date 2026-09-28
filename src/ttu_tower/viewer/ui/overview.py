"""Strips under the timelines: the stability class along the viewed interval
(x-linked to the timelines), and the whole-period availability overview,
whose draggable box shows and sets the viewed interval.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ttu_tower.viewer.timeaxis import SLOT_SECONDS, display_offset_s, slot_to_unix
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.curves import FixedXViewBox


def _color_rows(codes: np.ndarray, colors: dict) -> np.ndarray:
    rgba = np.zeros(codes.shape + (4,), dtype=np.uint8)
    for code, color in colors.items():
        rgba[codes == code] = color
    return rgba


def _on_grid(slots: np.ndarray, curve) -> np.ndarray:
    codes = np.full(slots.size, -1, dtype=np.int64)
    if curve.slots.size and slots.size:
        idx = curve.slots - slots[0]
        ok = (idx >= 0) & (idx < slots.size) & np.isfinite(curve.y)
        codes[idx[ok]] = curve.y[ok].astype(np.int64)
    return codes


class StabilityBand(QWidget):
    """The stability class of every slot, as a colored band that pans and
    zooms with the timelines it is x-linked to.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vb = FixedXViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.setMinimumHeight(18)
        self.plot.hideAxis("bottom")
        left = self.plot.getPlotItem().getAxis("left")
        left.setTicks([[(0.5, "stability")]])
        left.setWidth(60)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getPlotItem().setMenuEnabled(False)
        self.plot.hideButtons()
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.plot.setYRange(0, 1, padding=0)
        self.legend = QLabel()
        self.legend.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row = QHBoxLayout()
        row.addWidget(self.legend)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.plot, stretch=1)  # resizing the strip resizes the band, not the legend

    def set_axis_width(self, width: int) -> None:
        self.plot.getPlotItem().getAxis("left").setWidth(width)

    def link_to(self, plot_item) -> None:
        self.plot.getPlotItem().setXLink(plot_item)

    def set_classes(self, slots: np.ndarray, stability) -> None:
        if stability is None or slots.size == 0:
            self.image.clear()
            self.legend.setText("stability: unavailable (needs a tertiary stage)")
            return
        curve, names = stability
        colors = {-1: (255, 255, 255, 0)}
        colors.update({i: style.stability_color(name, i).getRgb() for i, name in enumerate(names)})
        rgba = _color_rows(_on_grid(slots, curve)[None, :], colors)
        self.image.setImage(rgba)
        self.image.setRect(slot_to_unix(slots[0]), 0, slots.size * SLOT_SECONDS, 1)
        self.legend.setText("stability: " + "  ".join(
            f"<span style='color:{style.stability_color(name, i).name()}'>■</span> {name}"
            for i, name in enumerate(names)))


class OverviewStrip(pg.PlotWidget):
    rangeRequested = Signal(float, float)

    def __init__(self, tz: str = "Etc/GMT+6", parent=None):
        self.axis = pg.DateAxisItem(orientation="bottom", utcOffset=display_offset_s(tz))
        super().__init__(parent=parent, axisItems={"bottom": self.axis})
        self.setMinimumHeight(110)
        self.setMaximumHeight(200)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().getAxis("left").setWidth(60)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image)
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(60, 120, 220, 50))
        self.region.setZValue(10)
        self.addItem(self.region)
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        self._quiet = False

    def set_timezone(self, tz: str) -> None:
        self.axis.utcOffset = display_offset_s(tz)
        self.axis.picture = None
        self.axis.update()

    def set_axis_width(self, width: int) -> None:
        self.getPlotItem().getAxis("left").setWidth(width)

    def set_availability(self, slots: np.ndarray, availability: np.ndarray, booms: list[int]) -> None:
        """availability: (booms × slots) status codes."""
        rgba = _color_rows(availability, style.AVAILABILITY_COLORS)
        self.image.setImage(rgba)
        x0 = slot_to_unix(slots[0]) if slots.size else 0.0
        self.image.setRect(x0, 0, slots.size * SLOT_SECONDS, rgba.shape[0])
        self.getPlotItem().getAxis("left").setTicks([[(i + 0.5, f"b{b}") for i, b in enumerate(booms)]])
        self.setYRange(0, rgba.shape[0], padding=0)
        self.setXRange(x0, x0 + slots.size * SLOT_SECONDS, padding=0)

    def show_range(self, x0: float, x1: float) -> None:
        self._quiet = True
        self.region.setRegion((x0, x1))
        self._quiet = False

    def _region_moved(self) -> None:
        if not self._quiet:
            x0, x1 = self.region.getRegion()
            self.rangeRequested.emit(x0, x1)
