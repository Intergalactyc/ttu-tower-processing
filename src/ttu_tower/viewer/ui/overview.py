"""Strips under the timelines: the stability class and each boom's
anisotropy state along the viewed interval (x-linked to the timelines), and
the whole-period availability overview, whose draggable box shows and sets
the viewed interval.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ttu_tower.viewer.anisotropy import CASE_COLORS, CASES
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, display_offset_s, slot_to_unix, unix_to_slot
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
        self.legend.setWordWrap(True)
        self.legend.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row = QHBoxLayout()
        row.addWidget(self.legend, stretch=1)
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


class AnisotropyBand(QWidget):
    """Each chosen boom's anisotropy class per slot, one row per boom (the
    lowest at the bottom), x-linked to the timelines. A click on a cell asks
    for that slot and boom.
    """

    variantChanged = Signal(str)
    cellClicked = Signal(int, int)  # slot, boom

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vb = FixedXViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.setMinimumHeight(30)
        self.plot.hideAxis("bottom")
        self.plot.getPlotItem().getAxis("left").setWidth(60)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getPlotItem().setMenuEnabled(False)
        self.plot.hideButtons()
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.variant = QComboBox()
        for v in ("mrd", "naive", "mrd_unexcised"):
            self.variant.addItem(v, v)
        self.variant.currentIndexChanged.connect(lambda *_: self.variantChanged.emit(self.variant.currentData()))
        self.boom_buttons: dict[int, QToolButton] = {}
        self.boom_row = QHBoxLayout()
        self.boom_row.setSpacing(1)
        legend = QLabel("anisotropy: " + "  ".join(
            f"<span style='color:{CASE_COLORS[c]}'>■</span> {c}" for c in CASES))
        legend.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        legend.setWordWrap(True)  # wraps rather than widening the window
        head = QHBoxLayout()
        head.addWidget(legend, stretch=1)
        head.addWidget(self.variant)
        head.addLayout(self.boom_row)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.plot, stretch=1)
        self.slots = np.empty(0, dtype=np.int64)
        self.codes = None  # booms x slots
        self.booms: list[int] = []
        self.rows: list[int] = []  # the booms drawn, bottom row first

    def set_axis_width(self, width: int) -> None:
        self.plot.getPlotItem().getAxis("left").setWidth(width)

    def link_to(self, plot_item) -> None:
        self.plot.getPlotItem().setXLink(plot_item)

    def set_booms(self, booms, shown=None) -> None:
        for btn in self.boom_buttons.values():
            btn.deleteLater()
        self.boom_buttons = {}
        shown = set(booms if shown is None else shown)
        for b in booms:
            btn = QToolButton()
            btn.setText(str(b))
            btn.setCheckable(True)
            btn.setChecked(b in shown)
            btn.setToolTip(f"show boom {b}")
            btn.toggled.connect(lambda *_: self._draw())
            self.boom_buttons[b] = btn
            self.boom_row.addWidget(btn)

    def set_classes(self, slots: np.ndarray, codes: np.ndarray, booms) -> None:
        self.slots, self.codes, self.booms = slots, codes, list(booms)
        self._draw()

    def _draw(self) -> None:
        if self.codes is None or self.slots.size == 0:
            self.image.clear()
            return
        self.rows = [b for b in self.booms if self.boom_buttons.get(b) is None or self.boom_buttons[b].isChecked()]
        if not self.rows:
            self.image.clear()
            return
        index = {b: i for i, b in enumerate(self.booms)}
        codes = self.codes[[index[b] for b in self.rows]]
        colors = {-1: (255, 255, 255, 0)}
        colors.update({i: pg.mkColor(CASE_COLORS[c]).getRgb() for i, c in enumerate(CASES)})
        self.image.setImage(_color_rows(codes.astype(np.int64), colors))
        self.image.setRect(slot_to_unix(self.slots[0]), 0, self.slots.size * SLOT_SECONDS, len(self.rows))
        self.plot.getPlotItem().getAxis("left").setTicks([[(i + 0.5, f"b{b}") for i, b in enumerate(self.rows)]])
        self.plot.setYRange(0, len(self.rows), padding=0)

    def _on_click(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or not self.rows:
            return
        if not self.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = self.vb.mapSceneToView(ev.scenePos())
        row = int(np.floor(pos.y()))
        if 0 <= row < len(self.rows):
            self.cellClicked.emit(int(unix_to_slot(pos.x())), self.rows[row])


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
        self.selection_image = pg.ImageItem(axisOrder="row-major")  # a row above the booms: the brushed slots
        self.selection_image.setZValue(5)
        self.addItem(self.selection_image)
        self._quiet = False
        self._slots = np.empty(0, dtype=np.int64)
        self._booms: list[int] = []

    def set_timezone(self, tz: str) -> None:
        self.axis.utcOffset = display_offset_s(tz)
        self.axis.picture = None
        self.axis.update()

    def set_axis_width(self, width: int) -> None:
        self.getPlotItem().getAxis("left").setWidth(width)

    def set_availability(self, slots: np.ndarray, availability: np.ndarray, booms: list[int]) -> None:
        """availability: (booms × slots) status codes."""
        rgba = _color_rows(availability, style.AVAILABILITY_COLORS)
        self._slots, self._booms = slots, list(booms)
        self.image.setImage(rgba)
        x0 = slot_to_unix(slots[0]) if slots.size else 0.0
        self.image.setRect(x0, 0, slots.size * SLOT_SECONDS, rgba.shape[0])
        self.getPlotItem().getAxis("left").setTicks([[(i + 0.5, f"b{b}") for i, b in enumerate(booms)]])
        self.setYRange(0, rgba.shape[0], padding=0)
        self.setXRange(x0, x0 + slots.size * SLOT_SECONDS, padding=0)

    def set_selection(self, selected: np.ndarray) -> None:
        """Mark the brushed slots in a row above the booms, to show where in the
        whole period they fall.
        """
        slots, rows = self._slots, len(self._booms)
        ticks = [(i + 0.5, f"b{b}") for i, b in enumerate(self._booms)]
        if selected.size == 0 or slots.size == 0:
            self.selection_image.clear()
            self.setYRange(0, rows, padding=0)
        else:
            on = np.isin(slots, selected)
            rgba = np.zeros((1, slots.size, 4), dtype=np.uint8)
            rgba[0, on] = pg.mkColor(style.SELECTED_COLOR).getRgb()
            self.selection_image.setImage(rgba)
            self.selection_image.setRect(slot_to_unix(slots[0]), rows, slots.size * SLOT_SECONDS, 1)
            self.setYRange(0, rows + 1, padding=0)
            ticks.append((rows + 0.5, "sel"))
        self.getPlotItem().getAxis("left").setTicks([ticks])

    def show_range(self, x0: float, x1: float) -> None:
        self._quiet = True
        self.region.setRegion((x0, x1))
        self._quiet = False

    def _region_moved(self) -> None:
        if not self._quiet:
            x0, x1 = self.region.getRegion()
            self.rangeRequested.emit(x0, x1)
