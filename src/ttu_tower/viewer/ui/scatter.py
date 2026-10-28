"""One quantity against another, slot by slot: by default panel A's against
panel B's, one boom (or pair) each, over the viewed interval or the whole
period. Points or a density map, optionally colored by stability;
correlation statistics; trend, binned central line, y = x; and any of the
curve fits (with limits and binning), fitted only on request; and a function
typed by hand (piecewise), drawn but not fitted. A lasso
(Shift-drag, or any drag in brush mode) selects slots for linked brushing.

A third quantity z (a panel's, any of the run's, or the boom height, which
stacks every boom both panels have) can color the points, and is the third
axis of the 3D view.
"""
import html
import itertools
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import brush, curvefits, expression
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import in_class, joined
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.axes import SwitchLogAxis
from ttu_tower.viewer.ui.function_overlay import FunctionOverlay
from ttu_tower.viewer.ui.panel import _VARIANT_LABELS
from ttu_tower.viewer.ui.timeline_plot import brush_mode
from ttu_tower.viewer.ui.search_combo import SearchCombo
from ttu_tower.viewer.ui.view_state import LateCombo

_POINT = "#1f4e9c"
_FIT = "#2ca02c"
_FIT_COLORS = (_FIT, "#9467bd", "#e377c2", "#17becf", "#bcbd22", "#8c564b", "#ff7f0e")
_SUBSETS_LABELS = (("to all points shown", "all"), ("to the selected only", "selected"),
                   ("to all but the selected", "unselected"))
_fit_ids = itertools.count(1)
_PCT = "#6a3d9a"
_OVERLAY = "#8c564b"
_NO_Z = "#bbbbbb"
Z_MAPS = {"sequential": "viridis", "diverging": "RdBu_r"}  # matplotlib names, for the figures too
HEIGHT = 3  # the z source that stacks the booms, z their heights
HEIGHT_VARIABLES = {"zA": "the height of panel A's boom shown, m", "zB": "of panel B's"}  # for the overlay


def z_levels(v: np.ndarray, centred: bool) -> tuple[float, float]:
    """Color limits: the 2nd-98th percentiles, or ± the 98th of |v| about 0."""
    if v.size == 0:
        return 0.0, 1.0
    if centred:
        lim = float(np.percentile(np.abs(v), 98)) or 1.0
        return -lim, lim
    lo, hi = (float(p) for p in np.percentile(v, [2, 98]))
    return (lo, hi) if hi > lo else (lo - 0.5, lo + 0.5)


def z_fractions(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


class LassoViewBox(pg.ViewBox):
    """A brushing drag draws a lasso and hands its polygon (view coordinates)
    to `lasso(px, py, mode)`; other drags pan and zoom as usual.
    """

    def __init__(self, lasso, **kwargs):
        super().__init__(**kwargs)
        self.lasso = lasso
        self.brush_always = False
        self._path: list = []
        self._line = None

    def mouseDragEvent(self, ev, axis=None):
        mode = brush_mode(ev, self.brush_always) if axis is None else None
        if mode is None:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        if ev.isStart() or self._line is None:
            self._path = [self.mapToView(ev.buttonDownPos())]
            self._line = pg.PlotCurveItem(pen=pg.mkPen(style.SELECTED_COLOR, width=1.5))
            self.addItem(self._line, ignoreBounds=True)
        self._path.append(self.mapToView(ev.pos()))
        px, py = np.array([p.x() for p in self._path]), np.array([p.y() for p in self._path])
        self._line.setData(np.append(px, px[0]), np.append(py, py[0]))
        if ev.isFinish():
            self.removeItem(self._line)
            self._line = None
            self.lasso(px, py, mode)


@dataclass
class FitSpec:
    """One fit's curve and options."""
    name: str | None = None
    pct: tuple = (0.0, 100.0)
    bins: int = 0
    bin_method: str = "mean"
    bin_log: bool = False
    min_count: int = 1
    settings: dict = field(default_factory=dict)  # the curve's own settings, as typed
    subset: str = "all"  # "all" the points shown, only the "selected", or all but them ("unselected")


class FitEntry:
    def __init__(self, spec: FitSpec, color: str):
        self.spec, self.color = spec, color
        self.shown = True
        self.result = self.inputs = None  # the fit and what it was fitted to
        self.failed = None  # inputs it failed on: not retried by "auto"
        self.error = ""
        self.id = next(_fit_ids)


class _FitsState:
    """Every fit's options, for a saved view."""

    def __init__(self, view):
        self.view = view

    def get_state(self):
        v = self.view
        v.current_entry.spec = v._spec_from_form()
        return {"fits": [dict(vars(e.spec), pct=list(e.spec.pct), shown=e.shown) for e in v.fits],
                "current": v.current}

    def put_state(self, value) -> bool:
        fits = (value or {}).get("fits") or []
        fields = set(FitSpec.__dataclass_fields__)
        specs = [FitSpec(**{k: (tuple(v) if k == "pct" else v) for k, v in f.items() if k in fields}) for f in fits]
        self.view.set_fits(specs, int((value or {}).get("current", 0)), [bool(f.get("shown", True)) for f in fits])
        return True


def _fit_job(name, x, y, opts, settings=()):
    """Runs on a worker thread."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return curvefits.fit(name, x, y, opts, **dict(settings))


class ScatterView(QWidget):
    STATE_FIELDS = ("source.0", "member.0", "source.1", "member.1", "source.2", "z_quantity", "z_variant", "member.2",
                    "display", "color_by", "log_x", "sides.x", "log_y", "sides.y", "log_z", "sides.z", "z_map",
                    "points_shown", "ring", "only_class", "trend", "mid", "mid_method", "mid_bins", "unit", "pct.0", "pct.1",
                    "pct_show", "fit_bins", "fit_bin_method", "fit_bin_log", "fit_min_count", "fit_name", "fit_auto",
                    "fit_settings", "fit_subset", "fits_state", "overlay", "overlay_on")  # saved in a view, restored in this order

    pointClicked = Signal(int, object)  # slot, the point's member (its boom, by height)
    brushed = Signal(object, str)  # the lassoed slots, and "replace" / "add" / "remove"

    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.datas = [None, None]
        self.catalog = None
        self.z_data = None  # z's own quantity, when it isn't a panel's
        self.z_loader = None  # (quantity, variant, on_done): loads z's own quantity, every member
        self._z_request = None
        self.x_range = None  # None: the whole period
        self.stability = None  # (Curve of class codes, names)
        self.slots = np.empty(0, dtype=np.int64)
        self.x = self.y = self.z = np.empty(0)
        self.point_member = None  # each point's boom, by height
        self.empty_reason = None
        self.fits: list[FitEntry] = [FitEntry(FitSpec(), _FIT)]
        self.current = 0  # the fit the form edits
        self._pending: set[int] = set()  # fits being fitted
        self._loading_form = False
        self.fits_state = _FitsState(self)
        self.window3d = None

        self.source = [QComboBox(), QComboBox(), QComboBox()]
        self.member = [QComboBox(), QComboBox(), LateCombo()]
        for axis in (0, 1, 2):
            src = self.source[axis]
            src.addItem("panel A", 0)
            src.addItem("panel B", 1)
            if axis == 2:
                src.addItem("a quantity", 2)
                src.addItem("boom height", HEIGHT)
                src.setToolTip("z: a panel's quantity, any of the run's, or the boom height (every boom both panels "
                               "have, one point per slot and boom; a slot quantity's value is at every boom)")
            src.setCurrentIndex(axis)
            src.currentIndexChanged.connect(lambda *_, a=axis: self._sync_members(a))
            self.member[axis].currentIndexChanged.connect(lambda *_: self.recompute())
        swap = QPushButton("⇄")
        swap.setToolTip("swap x and y")
        swap.setMaximumWidth(30)
        swap.clicked.connect(self._swap)
        self.z_quantity = SearchCombo()
        self.z_quantity.addItem("—", None)
        self.z_quantity.setToolTip("z's quantity (type to search)")
        self.z_quantity.currentIndexChanged.connect(self._on_z_quantity)
        self.z_variant = QComboBox()
        self.z_variant.currentIndexChanged.connect(self._load_z)
        self.z_pick = QWidget()
        pick = QHBoxLayout(self.z_pick)
        pick.setContentsMargins(0, 0, 0, 0)
        pick.addWidget(self.z_quantity, stretch=1)
        pick.addWidget(self.z_variant)
        self.log_z = QCheckBox("log z")
        self.z_map = QComboBox()
        self.z_map.addItem("sequential colors", "sequential")
        self.z_map.addItem("diverging about 0", "diverging")
        self.z_map.setToolTip("z's color scale (2nd-98th percentiles; diverging: symmetric about 0)")
        self.view3d_button = QPushButton("3D view…")
        self.view3d_button.setToolTip("x, y and z in a window of their own, to turn with the mouse")
        self.view3d_button.clicked.connect(self.open_3d)

        self.display = QComboBox()
        for label in ("points", "density", "density (log)"):
            self.display.addItem(label, label)
        self.color_by = QComboBox()
        for label in ("single color", "stability", "z"):
            self.color_by.addItem(label, label)
        self.points_shown = QComboBox()
        self.points_shown.addItem("all slots", "all")
        self.points_shown.addItem("selection only", "selected")
        self.points_shown.setToolTip("with slots brushed: every slot, or only the brushed ones (the statistics, "
                                     "lines and fits then use only those too)")
        self.ring = QCheckBox("ring the selection")
        self.ring.setChecked(True)
        self.ring.setToolTip("circle the brushed slots' points")
        self.points_shown.currentIndexChanged.connect(lambda *_: self.redraw())
        self.only_class = LateCombo()  # its classes come with the stability criterion
        self.only_class.addItem("all classes", None)
        self.only_class.setToolTip("only the slots of one stability class (the statistics, lines and fits use only "
                                   "those too)")
        self.only_class.currentIndexChanged.connect(lambda *_: self.redraw())
        self.only_class.setEnabled(False)  # until the run has stability classes
        self.ring.toggled.connect(lambda *_: self.redraw())
        self.log_x, self.log_y = QCheckBox("log x"), QCheckBox("log y")
        self.sides = {}
        for axis in ("x", "y", "z"):
            side = QComboBox()
            side.addItem(f"all {axis}", 0)
            side.addItem(f"{axis} > 0", 1)
            side.addItem(f"{axis} < 0", -1)
            side.setToolTip(f"every {axis}, or only one side of 0 (the statistics, lines and fits use the same "
                            f"points); a log axis shows one side, the negative one as −{axis} (its magnitudes)")
            side.currentIndexChanged.connect(lambda *_: self.redraw())
            self.sides[axis] = side
        self.trend, self.mid, self.unit = QCheckBox("trend (linear)"), QCheckBox("binned"), QCheckBox("y = x")
        self.mid_method = QComboBox()
        self.mid_method.addItems(["median", "mean"])
        self.mid_bins = QSpinBox()
        self.mid_bins.setRange(3, 200)
        self.mid_bins.setValue(25)
        for w in (self.display, self.color_by, self.mid_method, self.z_map):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        for axis, box in (("x", self.log_x), ("y", self.log_y), ("z", self.log_z)):
            box.toggled.connect(lambda on, a=axis: self._on_log(a, on))
        for w in (self.trend, self.mid, self.unit):
            w.toggled.connect(lambda *_: self.redraw())
        self.mid_bins.valueChanged.connect(lambda *_: self.redraw())

        self.fit_name = QComboBox()
        self.fit_name.addItem("none", None)
        for name in curvefits.FITS:
            self.fit_name.addItem(name, name)
        self.pct = [QDoubleSpinBox(), QDoubleSpinBox()]
        for box, value in zip(self.pct, (0.0, 100.0)):
            box.setRange(0.0, 100.0)
            box.setValue(value)
            box.setSuffix(" %")
            box.setDecimals(1)
        self.pct_show = QCheckBox("show")
        self.pct_show.setToolTip("mark these percentiles of x (of the points shown) with dashed lines")
        self.pct_show.toggled.connect(lambda *_: self.redraw())
        self.fit_bins = QSpinBox()
        self.fit_bins.setRange(0, 500)
        self.fit_bins.setSpecialValueText("no binning")
        self.fit_bin_method = QComboBox()
        self.fit_bin_method.addItems(["mean", "median"])
        self.fit_bin_log = QCheckBox("log bins")
        self.fit_bin_log.setToolTip("bins equal in log x (geometric centres), so each decade of x weighs alike; "
                                    "only positive x is binned")
        self.fit_bin_log.toggled.connect(lambda *_: self._show_stats())
        self.fit_min_count = QSpinBox()
        self.fit_min_count.setRange(1, 10_000)
        for w in (self.fit_name, self.fit_bin_method):
            w.currentIndexChanged.connect(lambda *_: (self._show_stats(), self._enable_controls()))
        for w in (*self.pct, self.fit_bins, self.fit_min_count):
            w.valueChanged.connect(lambda *_: (self._show_stats(), self._enable_controls()))
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("fit the chosen curve to the points shown now (the fit stays put when the "
                                   "interval or settings change, until pressed again)")
        self.fit_button.clicked.connect(lambda *_: self.run_fit())
        self.fit_subset = QComboBox()
        for label, key in _SUBSETS_LABELS:
            self.fit_subset.addItem(label, key)
        self.fit_subset.setToolTip("fit to every point shown, only the selected ones (Shift-drag a lasso), or all "
                                   "but the selected (e.g. without a group of outliers)")
        self.fit_subset.currentIndexChanged.connect(lambda *_: self._show_stats())
        self.fit_list = QListWidget()
        self.fit_list.setToolTip("the fits drawn together: tick to show, pick one to edit it below")
        self.fit_list.currentRowChanged.connect(self._on_fit_row)
        self.fit_list.itemChanged.connect(self._on_fit_checked)
        self.add_fit_button = QPushButton("+ fit")
        self.add_fit_button.setToolTip("another fit, starting as a copy of this one: change its curve or options to "
                                       "compare them")
        self.add_fit_button.clicked.connect(lambda *_: self.add_fit())
        self.remove_fit = QPushButton("remove")
        self.remove_fit.setToolTip("remove the fit being edited")
        self.remove_fit.clicked.connect(lambda *_: self.remove_current_fit())
        self.fit_auto = QCheckBox("auto")
        self.fit_auto.setToolTip("refit by itself whenever what's fitted changes: the quantities or booms, the "
                                 "interval, the axes, the slots shown, or the fit's settings")
        self.fit_auto.toggled.connect(lambda *_: self._maybe_autofit())
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(300)  # one fit after a burst of changes
        self._auto_timer.timeout.connect(self._autofit)
        self.fit_status = QLabel()
        self.fit_status.setTextFormat(Qt.TextFormat.RichText)
        self.fit_status.setWordWrap(True)
        self.fit_settings: dict[str, QLineEdit] = {}  # the chosen fit's own settings
        self.fit_settings_row = QHBoxLayout()
        self.fit_name.currentIndexChanged.connect(lambda *_: self._build_fit_settings())
        self.fit_name.currentIndexChanged.connect(self._on_fit_choice)

        self.stats = QLabel()
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.stats.setWordWrap(True)
        self.vb = LassoViewBox(self._on_lasso)
        self.axes = {side: SwitchLogAxis(orientation=side) for side in ("bottom", "left")}
        self.plot = pg.PlotWidget(viewBox=self.vb, axisItems=self.axes)
        self.plot.getPlotItem().export_layout = "per curve"  # CSV: x and y columns per curve
        self.plot.setToolTip("Shift-drag (or drag in brush mode) to lasso slots; Ctrl adds, Alt removes")
        self.selection = np.empty(0, dtype=np.int64)
        self.plot.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        self.legend = self.plot.getPlotItem().addLegend(offset=(5, 5), labelTextSize="8pt")
        self.zbar_axis = SwitchLogAxis(orientation="right")
        self.zbar = pg.PlotWidget(axisItems={"right": self.zbar_axis})
        bar = self.zbar.getPlotItem()
        bar.hideAxis("left")
        bar.hideAxis("bottom")
        bar.showAxis("right")
        bar.setMouseEnabled(x=False, y=False)
        bar.hideButtons()
        bar.setMenuEnabled(False)
        self.zbar_image = pg.ImageItem(np.linspace(0.0, 1.0, 256)[None, :], axisOrder="col-major", levels=(0.0, 1.0))
        bar.addItem(self.zbar_image)
        self.zbar.setFixedWidth(95)
        self.zbar.hide()

        axes = QGridLayout()
        for row, label in enumerate(("x", "y", "z")):
            axes.addWidget(QLabel(label), row, 0)
            axes.addWidget(self.source[row], row, 1)
            axes.addWidget(self.member[row], row, 2)
        axes.addWidget(self.z_pick, 3, 1, 1, 2)
        axes.addWidget(swap, 0, 3, 2, 1)
        axes.setColumnStretch(2, 1)
        look = QHBoxLayout()
        for w in (self.display, self.color_by, self.log_x, self.sides["x"], self.log_y, self.sides["y"]):
            look.addWidget(w)
        look.addStretch(1)
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("z"))
        for w in (self.z_map, self.log_z, self.sides["z"]):
            z_row.addWidget(w)
        z_row.addStretch(1)
        z_row.addWidget(self.view3d_button)
        chosen = QHBoxLayout()
        chosen.addWidget(QLabel("show"))
        chosen.addWidget(self.points_shown)
        chosen.addWidget(self.ring)
        chosen.addWidget(self.only_class)
        chosen.addStretch(1)
        lines = QHBoxLayout()
        for w in (self.trend, self.mid, self.mid_method, self.mid_bins, self.unit):
            lines.addWidget(w)
        lines.addStretch(1)
        fit_page = QWidget()
        form = QFormLayout(fit_page)
        list_row = QHBoxLayout()
        list_row.addWidget(self.fit_list, stretch=1)
        list_buttons = QVBoxLayout()
        list_buttons.addWidget(self.add_fit_button)
        list_buttons.addWidget(self.remove_fit)
        list_buttons.addStretch(1)
        list_row.addLayout(list_buttons)
        form.addRow("fits", list_row)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.fit_name, stretch=1)
        fit_row.addWidget(self.fit_subset, stretch=1)
        fit_row.addWidget(self.fit_button)
        fit_row.addWidget(self.fit_auto)
        form.addRow("fit", fit_row)
        pct_row = QHBoxLayout()
        pct_row.addWidget(self.pct[0])
        pct_row.addWidget(QLabel("to"))
        pct_row.addWidget(self.pct[1])
        pct_row.addWidget(self.pct_show)
        form.addRow("x percentiles", pct_row)
        bin_row = QHBoxLayout()
        bin_row.addWidget(self.fit_bins)
        bin_row.addWidget(self.fit_bin_method)
        bin_row.addWidget(self.fit_bin_log)
        bin_row.addWidget(QLabel("min per bin"))
        bin_row.addWidget(self.fit_min_count)
        form.addRow("bins", bin_row)
        form.addRow(self.fit_settings_row)
        form.addRow(self.fit_status)
        self.overlay_on = QCheckBox("overlay a function (drawn, not fitted)")
        self.overlay_on.setToolTip("draw y(x) typed by hand, in pieces over x ranges if need be; in the values "
                                   "plotted (−x where a log axis shows the negative side)")
        self.overlay = FunctionOverlay(HEIGHT_VARIABLES)
        self.overlay_on.toggled.connect(lambda on: self.lower_tabs.setTabText(1, "overlay (on)" if on else "overlay"))
        self.overlay_on.toggled.connect(lambda *_: self.redraw())
        overlay_page = QWidget()
        overlay_layout = QVBoxLayout(overlay_page)
        overlay_layout.addWidget(self.overlay_on)
        overlay_layout.addWidget(self.overlay)
        overlay_layout.addStretch(1)
        self.lower_tabs = QTabWidget()  # the fit and the overlay share the space under the plot
        self.lower_tabs.addTab(fit_page, "curve fit")
        self.lower_tabs.addTab(overlay_page, "overlay")
        self.overlay.changed.connect(self.redraw)
        self.overlay.from_fit.clicked.connect(self._overlay_from_fit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(axes)
        layout.addLayout(look)
        layout.addLayout(z_row)
        layout.addLayout(chosen)
        layout.addLayout(lines)
        plots = QHBoxLayout()
        plots.setSpacing(0)
        plots.addWidget(self.plot, stretch=1)
        plots.addWidget(self.zbar)
        layout.addLayout(plots, stretch=1)
        layout.addWidget(self.stats)
        layout.addWidget(self.lower_tabs)
        self._items: list = []
        self.drawn_lines: list = []  # what's drawn, for figure_spec
        self.drawn_groups: list = []
        self.drawn_vlines: list = []
        self.drawn_z = None  # z's coloring: (lo, hi) as plotted, colormap name
        self._drawn_slots = np.empty(0, dtype=np.int64)  # what a clicked point's index refers to
        self._drawn_members = None
        self._enable_controls()

    # --- sources -----------------------------------------------------------------------------------

    def set_sources(self, datas, x_range, stability=None) -> None:
        """`x_range`: (x0, x1) unix seconds, or None for the whole period."""
        changed = [d is not o for d, o in zip(datas, self.datas)]
        self.datas, self.x_range, self.stability = list(datas), x_range, stability
        self._fill_classes()
        for axis in (0, 1, 2):
            source = self.source[axis].currentData()
            if source in (0, 1) and changed[source]:
                self._sync_members(axis, recompute=False)
        self.recompute()

    def _data(self, axis: int):
        source = self.source[axis].currentData()
        if source == 2:
            return self.z_data
        return None if source == HEIGHT else self.datas[source]

    def by_height(self) -> bool:
        return self.source[2].currentData() == HEIGHT

    def _sync_members(self, axis: int, recompute: bool = True) -> None:
        box, data = self.member[axis], self._data(axis)
        current = box.currentData()
        box.blockSignals(True)
        box.clear()
        if data is not None:
            for i, member in enumerate(data.curves):
                box.addItem(style.member_label(member) or "value", member)
            other = self.member[0 if axis == 2 else 1 - axis].currentData()
            i = box.findData(current)
            if i < 0 and other is not None:
                i = box.findData(other)  # the same boom on every axis by default
            box.setCurrentIndex(max(i, 0))
            if axis == 2:
                box.take_pending()
        box.blockSignals(False)
        if recompute:
            self.recompute()

    # --- z's own quantity --------------------------------------------------------------------------

    def set_catalog(self, catalog) -> None:
        """The run's quantities, for z to choose from (loaded here, every member)."""
        self.catalog = catalog
        self.z_quantity.fill_quantities([q for qs in catalog.grouped().values() for q in qs if not q.categorical])
        self._on_z_quantity()

    def _z_quantity(self):
        key = self.z_quantity.currentData()
        return self.catalog[key] if self.catalog is not None and key in self.catalog.quantities else None

    def _on_z_quantity(self, *_) -> None:
        q, variant = self._z_quantity(), self.z_variant.currentData()
        self.z_variant.blockSignals(True)
        self.z_variant.clear()
        for v in q.variants if q is not None else ():
            self.z_variant.addItem(_VARIANT_LABELS.get(v, v), v)
        i = self.z_variant.findData(variant)
        self.z_variant.setCurrentIndex(i if i >= 0 else max(self.z_variant.findData("mrd"), 0))
        self.z_variant.setEnabled(self.z_variant.count() > 1)
        self.z_variant.blockSignals(False)
        self._load_z()

    def _load_z(self, *_) -> None:
        q, variant = self._z_quantity(), self.z_variant.currentData()
        self.z_data, self._z_request = None, (q.key, variant) if q is not None else None
        if q is not None and self.z_loader is not None:
            request = self._z_request
            self.z_loader(q, variant, lambda data: self._z_loaded(request, data))
        if self.source[2].currentData() == 2:
            self._sync_members(2)

    def _z_loaded(self, request, data) -> None:
        if not shiboken6.isValid(self) or request != self._z_request:
            return
        self.z_data = data
        if self.source[2].currentData() == 2:
            self._sync_members(2)

    def _z_at(self, slots: np.ndarray) -> np.ndarray:
        """z at these slots (NaN where it has none)."""
        z = np.full(slots.size, np.nan)
        data, member = self._data(2), self.member[2].currentData()
        if data is None or data.quantity.categorical or member not in data.curves:
            return z
        curve = data.curves[member]
        _, ic, iz = np.intersect1d(curve.slots, slots, assume_unique=True, return_indices=True)
        z[iz] = curve.y[ic]
        return z

    def _stacked(self, dx, dy):
        """(slots, x, y, booms): every boom both panels have, one after another;
        a slot quantity's value is at every boom. None where neither is per boom.
        """
        kinds = (dx.quantity.kind, dy.quantity.kind)
        if "pair" in kinds or "boom" not in kinds:
            return None
        sets = [set(d.curves) for d in (dx, dy) if d.quantity.kind == "boom"]
        parts = []
        for b in sorted(set.intersection(*sets)):
            cx, cy = (d.curves[b] if d.quantity.kind == "boom" else next(iter(d.curves.values())) for d in (dx, dy))
            slots, x, y = joined(cx, cy)
            parts.append((slots, x, y, np.full(slots.size, b)))
        if not parts:
            return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)
        return tuple(np.concatenate(p) for p in zip(*parts))

    def _swap(self) -> None:
        a, b = self.source[0].currentIndex(), self.source[1].currentIndex()
        ma, mb = self.member[0].currentData(), self.member[1].currentData()
        for w in self.source + self.member:
            w.blockSignals(True)
        self.source[0].setCurrentIndex(b)
        self.source[1].setCurrentIndex(a)
        self._sync_members(0, recompute=False)
        self._sync_members(1, recompute=False)
        self.member[0].setCurrentIndex(max(self.member[0].findData(mb), 0))
        self.member[1].setCurrentIndex(max(self.member[1].findData(ma), 0))
        for w in self.source + self.member:
            w.blockSignals(False)
        self.recompute()

    def recompute(self) -> None:
        dx, dy = self._data(0), self._data(1)
        mx, my = self.member[0].currentData(), self.member[1].currentData()
        self.slots, self.x, self.y, self.z = np.empty(0, dtype=np.int64), np.empty(0), np.empty(0), np.empty(0)
        self.point_member, self.empty_reason = None, None
        if dx is None or dy is None:
            self.redraw()
            return
        if dx.quantity.categorical or dy.quantity.categorical:
            self.empty_reason = "categorical quantities can't be scattered"
            self.redraw()
            return
        members = None
        if self.by_height():
            stacked = self._stacked(dx, dy)
            if stacked is None:
                self.empty_reason = "z = boom height needs x or y per boom (and neither per boom pair)"
                self.redraw()
                return
            slots, x, y, members = stacked
            z = np.array([HEIGHTS[int(b)] for b in members], dtype=float)
        else:
            if mx not in dx.curves or my not in dy.curves:
                self.redraw()
                return
            slots, x, y = joined(dx.curves[mx], dy.curves[my])
            z = self._z_at(slots)
        ok = np.isfinite(x) & np.isfinite(y)
        if self.x_range is not None:
            t = slot_to_unix(slots)
            ok &= (t >= self.x_range[0]) & (t < self.x_range[1])
        self.slots, self.x, self.y, self.z = slots[ok], x[ok], y[ok], z[ok]
        self.point_member = members[ok] if members is not None else None
        self.redraw()

    # --- drawing -----------------------------------------------------------------------------------

    def class_shown(self) -> str | None:
        """The one stability class shown, or None for all."""
        name = self.only_class.currentData()
        return name if name is not None and self.stability is not None and name in self.stability[1] else None

    def _fill_classes(self) -> None:
        names = list(self.stability[1]) if self.stability is not None else []
        if [self.only_class.itemData(i) for i in range(self.only_class.count())] == [None] + names:
            return
        current = self.only_class.currentData()
        self.only_class.blockSignals(True)
        self.only_class.clear()
        self.only_class.addItem("all classes", None)
        for name in names:
            self.only_class.addItem(name, name)
        self.only_class.setCurrentIndex(max(self.only_class.findData(current), 0))
        self.only_class.take_pending()
        self.only_class.setEnabled(bool(names))
        self.only_class.blockSignals(False)

    def selected_only(self) -> bool:
        return self.points_shown.currentData() == "selected" and self.selection.size > 0

    def _keep(self) -> np.ndarray:
        """The points on the chosen side of 0 of x and y (one side where an axis is
        log), only the brushed ones if so chosen.
        """
        keep = np.ones(self.x.size, dtype=bool)
        if self.selected_only():
            keep &= np.isin(self.slots, self.selection)
        if self.class_shown() is not None:
            keep &= in_class(self.stability, self.slots, self.class_shown())
        for axis, values in (("x", self.x), ("y", self.y)):
            if self.side(axis):
                keep &= values * self.side(axis) > 0
        return keep

    def _shown(self):
        keep = self._keep()
        return self.slots[keep], (self.x * self.sign("x"))[keep], (self.y * self.sign("y"))[keep]

    def shown_z(self) -> np.ndarray:
        """z at the shown points (−z on a log z's negative side), NaN off z's chosen side."""
        z = self.z[self._keep()]
        side = self.side("z")
        return np.where(z * side > 0, z * self.sign("z"), np.nan) if side else z

    def _z_view(self, z: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(z) if self.log_z.isChecked() else z

    def _log_box(self, axis: str) -> QCheckBox:
        return {"x": self.log_x, "y": self.log_y, "z": self.log_z}[axis]

    def side(self, axis: str) -> int:
        """The side of 0 shown: 1, -1, or 0 for both (never on a log axis)."""
        side = self.sides[axis].currentData()
        return side or (1 if self._log_box(axis).isChecked() else 0)

    def sign(self, axis: str) -> int:
        """-1 where a log axis shows its negative side (as −x), else 1."""
        return -1 if self._log_box(axis).isChecked() and self.side(axis) < 0 else 1

    def _on_log(self, axis: str, on: bool) -> None:
        """A log axis shows one side: the positive one, unless it has only negative values."""
        box = self.sides[axis]
        if on and box.currentData() == 0:
            values = {"x": self.x, "y": self.y, "z": self.z}[axis]
            box.blockSignals(True)
            box.setCurrentIndex(box.findData(1 if (values > 0).any() or not (values < 0).any() else -1))
            box.blockSignals(False)
        self.redraw()

    def has_z(self) -> bool:
        return bool(np.isfinite(self.z).any())

    def colors_by_z(self) -> bool:
        return self.color_by.currentData() == "z" and self.display.currentData() == "points" and self.has_z()

    def z_colormap(self) -> str:
        centred = self.z_map.currentData() == "diverging" and not self.log_z.isChecked()
        return Z_MAPS["diverging" if centred else "sequential"]

    def axis_title(self, axis: int) -> str:
        """x, y or z as plotted (−q on a log axis's negative side)."""
        if axis == 2 and self.by_height():
            return "height [m]"
        data = self._data(axis)
        if data is None:
            return ""
        name = "xyz"[axis]
        minus = "−" if self.sign(name) < 0 else ""
        member = "each boom" if self.by_height() else self.member[axis].currentText()
        side = self.side(name)
        only = f" ({'>' if side > 0 else '<'} 0 only)" if side and not self._log_box(name).isChecked() else ""
        return f"{minus}{data.quantity.title} · {member}{only}"

    def _to_view(self, x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log10(x) if self.log_x.isChecked() else x), (np.log10(y) if self.log_y.isChecked() else y)

    def _enable_controls(self) -> None:
        """Grey out what can't apply to what's shown."""
        points = self.display.currentData() == "points"
        brushed = self.selection.size > 0
        self.points_shown.setEnabled(brushed)
        self.ring.setEnabled(brushed)
        self.color_by.setEnabled(points)
        self.color_by.setToolTip("density maps aren't colored by class" if not points else "color the points")
        self.color_by.model().item(1).setEnabled(self.stability is not None)
        self.color_by.model().item(2).setEnabled(self.has_z())
        z_source = self.source[2].currentData()
        self.z_pick.setVisible(z_source == 2)
        for box in self.member:
            box.setEnabled(not self.by_height())
            box.setToolTip("by boom height: every boom both panels have" if self.by_height() else "")
        self.z_map.setEnabled(self.colors_by_z())
        self.z_map.model().item(1).setEnabled(not self.log_z.isChecked())
        self.view3d_button.setEnabled(self.has_z() or self.shows_3d())
        self.mid_method.setEnabled(self.mid.isChecked())
        self.mid_bins.setEnabled(self.mid.isChecked())
        for axis, box, values in (("x", self.log_x, self.x), ("y", self.log_y, self.y), ("z", self.log_z, self.z)):
            box.setEnabled(bool((np.isfinite(values) & (values != 0)).any()) or box.isChecked())
            box.setToolTip("no nonzero values to show on a log axis" if not box.isEnabled()
                           else "a log axis, of the side of 0 chosen beside it")
            side = self.sides[axis]
            side.model().item(0).setEnabled(not box.isChecked())  # a log axis shows one side
            for i, has in enumerate((bool((values > 0).any()), bool((values < 0).any())), start=1):
                side.model().item(i).setEnabled(has or side.currentIndex() == i)
        binned = self.fit_bins.value() > 0
        self.fit_bin_method.setEnabled(binned)
        self.fit_min_count.setEnabled(binned)
        self.fit_bin_log.setEnabled(binned and bool((self.x > 0).any()))
        curves = self.fit_name.currentData() is not None or any(e.spec.name for e in self.fits)
        self.fit_button.setEnabled(curves and self.x.size > 1 and not self._fitting)
        self.overlay.from_fit.setEnabled(self.fit_result is not None)
        chosen = self.selection.size > 0
        for i in (1, 2):
            self.fit_subset.model().item(i).setEnabled(chosen or self.fit_subset.currentIndex() == i)

    def redraw(self) -> None:
        self._redraw()
        if self.shows_3d():
            self.window3d.refresh()

    def _redraw(self) -> None:
        self._enable_controls()
        self.drawn_lines, self.drawn_groups, self.drawn_vlines, self.drawn_z = [], [], [], None
        self.plot.clear()
        self.legend.clear()
        self.zbar.hide()
        dx, dy = self._data(0), self._data(1)
        for side, box in (("bottom", self.log_x), ("left", self.log_y)):
            self.axes[side].set_log10(box.isChecked())
        if dx is not None and dy is not None:
            log = {True: " (log scale)", False: ""}
            for side, axis, box in (("bottom", 0, self.log_x), ("left", 1, self.log_y)):
                self.plot.setLabel(side, html.escape(self.axis_title(axis) + log[box.isChecked()]))  # labels are HTML
        if self.x.size == 0:
            self.stats.setText(self.empty_reason or "no slots with both values in range")
            return
        keep = self._keep()
        slots, x, y = self._shown()
        self._drawn_slots = slots
        self._drawn_members = self.point_member[keep] if self.point_member is not None else None
        if slots.size == 0:
            where = " in the class shown" if self.class_shown() is not None else ""
            self.stats.setText(f"no brushed slots with both values in range{where}" if self.selected_only()
                               else f"no points to show on these axes{where}")
            return
        vx, vy = self._to_view(x, y)
        mode = self.display.currentData()
        if mode.startswith("density"):
            self._density(vx, vy, log=mode.endswith("(log)"))
        elif self.colors_by_z():
            self._points_by_z(vx, vy)
        else:
            self._points(slots, vx, vy)
        self._highlight(slots, vx, vy)
        lo, hi = (np.percentile(x, [0.5, 99.5]) if x.size > 1 else (x.min(), x.max()))
        grid = np.geomspace(max(lo, 1e-12), hi, 200) if self.log_x.isChecked() else np.linspace(lo, hi, 200)
        stats = curvefits.correlation(x, y)
        parts = [f"N {stats['N']}"] + [f"{k} {stats[k]:.4f}" if np.isfinite(stats[k]) else f"{k} —"
                                       for k in ("r", "ρ", "κ")]
        if self.trend.isChecked():
            try:
                res = curvefits.fit("linear", x, y)
                self._line(grid, res.func(grid), "#ff7f0e", f"trend {res.label}")
            except ValueError:
                pass
        if self.mid.isChecked():
            log_bins = self.log_x.isChecked()  # equal bins in what's shown: log bins on a log x axis
            cx, cy = curvefits.bin_average(x, y, self.mid_bins.value(), self.mid_method.currentText(),
                                           bin_range=(lo, hi), log=log_bins)
            self._line(cx, cy, "#d62728", f"{self.mid_method.currentText()}, {self.mid_bins.value()}"
                                          f"{' log' if log_bins else ''} bins", width=2)
        if self.unit.isChecked():
            self._line(grid, grid, "#555555", "y = x", dash=True)
        several = len(self.fits) > 1
        for i, entry in enumerate(self.fits):
            if entry.result is None or not entry.shown:
                continue
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                fy = entry.result.func(grid)
            stale = entry.inputs != self._inputs_for(entry.spec)
            label = (f"{i + 1}. {self._describe(entry)}" if several else entry.inputs[0]) + (" (earlier)" if stale else "")
            self._line(grid, fy, entry.color, label, width=2.8 if several and i == self.current else 2.2, dash=stale)
        if self.overlay_on.isChecked():
            self._draw_overlay(x, y)
        if self.pct_show.isChecked():
            pct = [self.pct[0].value(), self.pct[1].value()]
            for p, at in zip(pct, np.percentile(x, pct)):
                self._vline(float(at), f"{p:g} %")
        self._show_stats()
        if self.class_shown() is not None:
            parts.insert(0, f"{self.class_shown()} only")
        self.stats.setText(" · ".join(parts))
        self.stats.setToolTip("r: Pearson; ρ: Spearman; κ: Cohen's kappa on the signs of x and y "
                              "(— where both never change sign)")
        # like the paper's plots: frame the 1st-99th percentiles, so a few outliers don't squash the rest
        (x0, x1), (y0, y1) = (np.percentile(v, [1, 99]) for v in (vx, vy))
        pad_x, pad_y = 0.05 * (x1 - x0 or 1.0), 0.05 * (y1 - y0 or 1.0)
        self.plot.setRange(xRange=(x0 - pad_x, x1 + pad_x), yRange=(y0 - pad_y, y1 + pad_y), padding=0)

    def class_groups(self, slots) -> list:
        """[(class name, mask, color)] for these slots' stability classes; one group without them."""
        if self.stability is None:
            return [("", np.ones(slots.size, dtype=bool), pg.mkColor(_POINT))]
        curve, names = self.stability
        codes = pd.Series(curve.y, index=curve.slots).reindex(slots).to_numpy()
        groups = [(n, codes == i, style.stability_color(n, i)) for i, n in enumerate(names)]
        groups.append(("unclassified", ~np.isin(codes, range(len(names))), pg.mkColor(_NO_Z)))
        return groups

    def _points(self, slots, vx, vy) -> None:
        if self.color_by.currentData() == "stability":
            groups = self.class_groups(slots)
        else:
            groups = [("", np.ones(slots.size, dtype=bool), pg.mkColor(_POINT))]
        self.drawn_groups = [(name, mask, pg.mkColor(color).name()) for name, mask, color in groups]
        for name, mask, color in groups:
            if not mask.any():
                continue
            color = pg.mkColor(color)
            color.setAlpha(150)
            self._scatter_item(vx, vy, mask, pg.mkBrush(color), name)

    def _scatter_item(self, vx, vy, mask, brush, name="") -> None:
        """Points data: their index among those shown (see `_on_click`)."""
        size = 3 if vx.size > 5000 else 5
        item = pg.ScatterPlotItem(vx[mask], vy[mask], data=np.flatnonzero(mask), size=size, pen=None, brush=brush,
                                  name=name or None)
        item.sigClicked.connect(self._on_click)
        self.plot.addItem(item)

    def _points_by_z(self, vx, vy) -> None:
        """Colored by z, over its 2nd-98th percentiles; points without z in grey."""
        v = self._z_view(self.shown_z())
        have = np.isfinite(v)
        name = self.z_colormap()
        lo, hi = z_levels(v[have], name == Z_MAPS["diverging"])
        self.drawn_z = (lo, hi, name)
        self.drawn_groups = [("no z", ~have, _NO_Z), ("", have, None)]
        if (~have).any():
            grey = pg.mkColor(_NO_Z)
            grey.setAlpha(150)
            self._scatter_item(vx, vy, ~have, pg.mkBrush(grey), "no z")
        cmap = pg.colormap.getFromMatplotlib(name)
        colors = cmap.getLookupTable(0.0, 1.0, 256, mode="qcolor")
        for c in colors:
            c.setAlpha(170)
        brushes = np.array([pg.mkBrush(c) for c in colors], dtype=object)
        index = np.round(z_fractions(v[have], lo, hi) * 255).astype(int)
        self._scatter_item(vx, vy, have, list(brushes[index]))
        bar = self.zbar.getPlotItem()
        self.zbar_image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.zbar_image.setRect(0.0, lo, 1.0, hi - lo)
        bar.setXRange(0.0, 1.0, padding=0)
        bar.setYRange(lo, hi, padding=0)
        self.zbar_axis.set_log10(self.log_z.isChecked())
        bar.setLabel("right", html.escape(self.axis_title(2) + (" (log scale)" if self.log_z.isChecked() else "")))
        self.zbar.show()

    def _vline(self, x: float, label: str) -> None:
        self.drawn_vlines.append({"x": x, "label": label})
        at = np.log10(x) if self.log_x.isChecked() else x
        line = pg.InfiniteLine(at, angle=90, pen=pg.mkPen(_PCT, width=1.2, style=Qt.PenStyle.DashLine), label=label,
                               labelOpts={"position": 0.96, "color": _PCT})
        self.plot.addItem(line)

    def _density(self, vx, vy, log: bool) -> None:
        counts, ex, ey = np.histogram2d(vx, vy, bins=80, range=[np.percentile(vx, [0.5, 99.5]),
                                                                  np.percentile(vy, [0.5, 99.5])])
        img = np.log10(counts, where=counts > 0, out=np.full(counts.shape, np.nan)) if log else counts
        img = np.where(counts > 0, img, np.nan)
        item = pg.ImageItem(img, axisOrder="col-major")
        item.setColorMap(pg.colormap.get("viridis"))
        item.setRect(ex[0], ey[0], ex[-1] - ex[0], ey[-1] - ey[0])
        self.plot.addItem(item)

    def _line(self, x, y, color, label, width: float = 1.8, dash: bool = False, gaps: bool = False) -> None:
        """`gaps`: break the line at NaN (else the finite points are joined)."""
        self.drawn_lines.append({"label": label, "color": pg.mkColor(color).name(), "x": np.asarray(x, dtype=float),
                                 "y": np.asarray(y, dtype=float), "dashed": dash})
        vx, vy = self._to_view(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        ok = np.isfinite(vx) & np.isfinite(vy)
        pen = pg.mkPen(color, width=width, style=Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)
        if gaps:
            self.plot.plot(np.where(ok, vx, np.nan), np.where(ok, vy, np.nan), pen=pen, name=label, connect="finite")
        else:
            self.plot.plot(vx[ok], vy[ok], pen=pen, name=label)

    def overlay_heights(self) -> dict:
        """zA, zB: the heights of the booms shown from panel A and panel B (x's if both axes use one
        panel; NaN where none is a single boom).
        """
        out = dict.fromkeys(HEIGHT_VARIABLES, np.nan)
        for axis in (1, 0):
            source, member = self.source[axis].currentData(), self.member[axis].currentData()
            if source in (0, 1) and isinstance(member, (int, np.integer)):
                out["zA" if source == 0 else "zB"] = HEIGHTS[int(member)]
        return out

    def _overlay_curve(self, func, ranges, lo: float, hi: float, values: dict):
        """The function over [lo, hi], a NaN between pieces so they aren't joined."""
        grid = np.geomspace(max(lo, 1e-12), hi, 600) if self.log_x.isChecked() else np.linspace(lo, hi, 600)
        edges = [b for r in ranges for b in r if lo < b < hi]
        grid = np.unique(np.concatenate([grid, edges, np.nextafter(edges, -np.inf)]))  # each piece to its edge
        gy = func(grid, **values)
        breaks = np.flatnonzero(np.diff(expression.piece_of(grid, ranges)) != 0) + 1
        return np.insert(grid, breaks, np.nan), np.insert(gy, breaks, np.nan)

    def _draw_overlay(self, x, y) -> None:
        """The typed function over the shown points' x span (by boom height, a curve per boom at its own
        height), and how far the points lie from it.
        """
        made = self.overlay.function()
        if made is None:
            return
        func, ranges = made
        unread = bool(self.overlay.status.text())  # pieces that couldn't be read: drawn without them
        if self.by_height() and self._drawn_members is not None:
            heights = np.array([HEIGHTS[int(b)] for b in self._drawn_members], dtype=float)
            at = func(x, zA=heights, zB=heights)  # each point at its own boom's height
            for i, b in enumerate(sorted(set(int(b) for b in self._drawn_members), key=HEIGHTS.get)):
                mine = self._drawn_members == b
                gx, gy = self._overlay_curve(func, ranges, float(x[mine].min()), float(x[mine].max()),
                                             {"zA": HEIGHTS[b], "zB": HEIGHTS[b]})
                self._line(gx, gy, style.boom_color(b), "overlay, each boom" if i == 0 else None, width=2.0,
                           gaps=True)
        else:
            values = self.overlay_heights()
            missing = [n for n in sorted(func.names) if not np.isfinite(values[n])]
            if missing:
                panels = " and ".join(f"panel {n[1]}" for n in missing)
                note = f"{', '.join(missing)}: Scatter shows no single boom from {panels}"
                errors = self.overlay.status.text()
                self.overlay.status.setText(errors.replace("</span>", f"; {note}</span>") if errors
                                            else f"<span style='color:#b00'>{note}</span>")
                return
            gx, gy = self._overlay_curve(func, ranges, float(x.min()), float(x.max()), values)
            self._line(gx, gy, _OVERLAY, "overlay", width=2.2, gaps=True)
            at = func(x, **values)
        ok = np.isfinite(at)
        if unread:
            return  # the status says what couldn't be read
        if ok.any():
            rmse = float(np.sqrt(np.mean((y[ok] - at[ok]) ** 2)))
            self.overlay.status.setText(f"RMSE {rmse:.3g} over the {int(ok.sum())} points where it's defined")
        else:
            self.overlay.status.setText("not defined at any point shown")

    def _overlay_from_fit(self) -> None:
        if self.fit_result is None:
            return
        name, settings = self.fit_inputs[0], dict(self.fit_inputs[-1] or ())
        self.overlay.set_pieces(curvefits.fit_formula(name, self.fit_result.params, settings))
        self.overlay_on.setChecked(True)
        self.lower_tabs.setCurrentIndex(1)

    # --- fitting, on request: several fits, each its own curve and options --------------------------

    @property
    def current_entry(self) -> "FitEntry":
        return self.fits[self.current]

    @property
    def fit_result(self):
        """The fit being edited's result (None until fitted)."""
        return self.current_entry.result

    @property
    def fit_inputs(self):
        return self.current_entry.inputs

    @property
    def _fitting(self) -> bool:
        return bool(self._pending)

    def _spec_from_form(self) -> "FitSpec":
        return FitSpec(self.fit_name.currentData(), (self.pct[0].value(), self.pct[1].value()), self.fit_bins.value(),
                       self.fit_bin_method.currentText(), self.fit_bin_log.isChecked(), self.fit_min_count.value(),
                       {arg: edit.text() for arg, edit in self.fit_settings.items()},
                       self.fit_subset.currentData() or "all")

    def _form_from_spec(self, spec: "FitSpec") -> None:
        """Show a fit's options in the form (which then edits that fit)."""
        self._loading_form = True
        widgets = (self.fit_name, *self.pct, self.fit_bins, self.fit_bin_method, self.fit_bin_log, self.fit_min_count,
                   self.fit_subset)
        for w in widgets:
            w.blockSignals(True)
        self.fit_name.setCurrentIndex(max(self.fit_name.findData(spec.name), 0))
        self.pct[0].setValue(spec.pct[0])
        self.pct[1].setValue(spec.pct[1])
        self.fit_bins.setValue(spec.bins)
        self.fit_bin_method.setCurrentText(spec.bin_method)
        self.fit_bin_log.setChecked(spec.bin_log)
        self.fit_min_count.setValue(spec.min_count)
        self.fit_subset.setCurrentIndex(max(self.fit_subset.findData(spec.subset), 0))
        for w in widgets:
            w.blockSignals(False)
        self._build_fit_settings(spec.settings)
        self._loading_form = False

    def _options_for(self, spec: "FitSpec") -> curvefits.FitOptions:
        return curvefits.FitOptions(pct_x=spec.pct if tuple(spec.pct) != (0.0, 100.0) else None,
                                    bin_count=spec.bins or None, bin_method=spec.bin_method,
                                    bin_min_count=spec.min_count,
                                    bin_log=spec.bin_log and spec.bins > 0 and bool((self.x > 0).any()))

    def _options(self) -> curvefits.FitOptions:
        return self._options_for(self._spec_from_form())

    def _build_fit_settings(self, values: dict | None = None) -> None:
        while self.fit_settings_row.count():
            item = self.fit_settings_row.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self.fit_settings = {}
        for arg, (label, default) in curvefits.FIT_SETTINGS.get(self.fit_name.currentData(), {}).items():
            edit = QLineEdit((values or {}).get(arg, f"{default:g}"))
            edit.setMaximumWidth(70)
            edit.textChanged.connect(lambda *_: self._show_stats())
            self.fit_settings[arg] = edit
            self.fit_settings_row.addWidget(QLabel(label))
            self.fit_settings_row.addWidget(edit)
        self.fit_settings_row.addStretch(1)
        if not self._loading_form:
            self._show_stats()

    @staticmethod
    def _settings_for(spec: "FitSpec") -> tuple:
        """A fit's own settings as (argument, value) pairs (ValueError if unreadable)."""
        out = []
        for arg, text in spec.settings.items():
            try:
                out.append((arg, float(text)))
            except ValueError:
                raise ValueError(f"{arg}: not a number") from None
        return tuple(out)

    def _settings(self) -> tuple:
        return self._settings_for(self._spec_from_form())

    def _inputs_for(self, spec: "FitSpec") -> tuple:
        """What a fit depends on: the curve, its options, the axes and the points (and the selection,
        where it fits to or without it).
        """
        span = (int(self.slots[0]), int(self.slots[-1])) if self.slots.size else None
        try:
            settings = self._settings_for(spec)
        except ValueError:
            settings = None
        quantities = tuple((d.quantity.key, d.variant, d.quantity.extra) if d is not None else None
                           for d in (self._data(0), self._data(1)))  # the same slots can hold another quantity
        chosen = (int(self.selection.size), hash(self.selection.tobytes())) if spec.subset != "all" else None
        return (spec.name or "", self._options_for(spec), quantities, self.source[0].currentData(),
                self.member[0].currentData(), self.source[1].currentData(), self.member[1].currentData(),
                self.by_height(), self.class_shown(), int(self.slots.size), span, self.log_x.isChecked(),
                self.log_y.isChecked(), spec.subset, chosen, self.side("x"), self.side("y"),
                # settings last: a fit reads them as inputs[-1]
                int(self._shown()[0].size) if self.selected_only() else None, settings)

    def _inputs(self) -> tuple:
        return self._inputs_for(self._spec_from_form())

    def _fit_points(self, spec: "FitSpec"):
        """The points shown, or only (or all but) the selected ones."""
        slots, x, y = self._shown()
        if spec.subset == "all":
            return x, y
        chosen = np.isin(slots, self.selection)
        keep = chosen if spec.subset == "selected" else ~chosen
        return x[keep], y[keep]

    def _on_fit_choice(self, *_) -> None:
        entry = self.current_entry
        if self.fit_name.currentData() is None and entry.result is not None:  # "none" takes the fit away
            entry.result = entry.inputs = None
            self.redraw()

    def run_fit(self, only_stale: bool = False) -> None:
        """Fit the fit being edited (unless `only_stale`) and every other fit whose inputs changed."""
        if not self._loading_form:
            self.current_entry.spec = self._spec_from_form()
        cleared = False
        for entry in list(self.fits):
            if entry.spec.name is None:
                if entry.result is not None:
                    entry.result = entry.inputs = None
                    cleared = True
                continue
            now = self._inputs_for(entry.spec)
            if entry is self.current_entry and not only_stale:
                self._fit_entry(entry, now)
            elif now not in (entry.inputs, entry.failed):
                self._fit_entry(entry, now)
        if cleared or not self._pending:
            self.redraw()
        else:
            self._enable_controls()
            self._show_stats()

    def _fit_entry(self, entry: "FitEntry", inputs: tuple) -> None:
        name = entry.spec.name
        if inputs[-1] is None:
            self._fit_failed(entry, inputs, ValueError("a fit setting isn't a number"))
            return
        if entry.spec.subset != "all" and self.selection.size == 0:
            self._fit_failed(entry, inputs, ValueError("nothing is selected (lasso some points first)"))
            return
        x, y = self._fit_points(entry.spec)
        self._pending.add(entry.id)
        if self.runner is None:
            try:
                self._fitted(entry, inputs, _fit_job(name, x, y, inputs[1], inputs[-1]))
            except (ValueError, RuntimeError, TypeError) as exc:
                self._fit_failed(entry, inputs, exc)
            return
        self.runner.submit(_fit_job, name, x, y, inputs[1], inputs[-1], key=f"scatterfit-{id(self)}-{entry.id}",
                           label=f"fitting {name}",
                           on_done=lambda res, e=entry, i=inputs: self._fitted(e, i, res),
                           on_error=lambda exc, tb, e=entry, i=inputs: self._fit_failed(e, i, exc))

    def _fitted(self, entry: "FitEntry", inputs, result) -> None:
        if not shiboken6.isValid(self):
            return
        self._pending.discard(entry.id)
        entry.result, entry.inputs, entry.failed, entry.error = result, inputs, None, ""
        if entry in self.fits:
            self.redraw()

    def _fit_failed(self, entry: "FitEntry", inputs, exc) -> None:
        if not shiboken6.isValid(self):
            return
        self._pending.discard(entry.id)
        entry.failed, entry.error = inputs, f"{entry.spec.name} fit failed: {exc}"
        self._enable_controls()
        self._show_stats()

    def _stale(self) -> list:
        return [e for e in self.fits if e.spec.name is not None
                and self._inputs_for(e.spec) not in (e.inputs, e.failed)]

    def _maybe_autofit(self) -> None:
        if self.fit_auto.isChecked() and self.x.size > 1 and self._stale():
            self._auto_timer.start()

    def _autofit(self) -> None:
        if not shiboken6.isValid(self) or not self.fit_auto.isChecked():
            return
        if self._fitting:  # once they're done, their redraw looks again
            return
        if self._stale():
            self.run_fit(only_stale=True)

    def _describe(self, entry: "FitEntry") -> str:
        spec = entry.spec
        parts = [spec.name or "none"]
        if tuple(spec.pct) != (0.0, 100.0):
            parts.append(f"x {spec.pct[0]:g}–{spec.pct[1]:g} %")
        if spec.bins:
            parts.append(f"{spec.bins} {spec.bin_method}{' log' if spec.bin_log else ''} bins")
        if spec.subset != "all":
            parts.append("selected only" if spec.subset == "selected" else "without the selection")
        parts += [f"{k} {v}" for k, v in spec.settings.items()]
        return ", ".join(parts)

    def _fill_fit_list(self) -> None:
        self.fit_list.blockSignals(True)
        self.fit_list.clear()
        for i, entry in enumerate(self.fits):
            text = f"{i + 1}. {self._describe(entry)}"
            if entry.result is not None:
                stale = entry.inputs != self._inputs_for(entry.spec)
                text += f": RMSE {entry.result.rmse:.3g}" + (" (earlier)" if stale else "")
            elif entry.error:
                text += ": failed"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if entry.shown else Qt.CheckState.Unchecked)
            item.setForeground(pg.mkColor(entry.color))
            item.setToolTip(entry.result.label if entry.result is not None else entry.error or "not fitted yet")
            self.fit_list.addItem(item)
        self.fit_list.setCurrentRow(self.current)
        self.fit_list.blockSignals(False)
        rows = min(len(self.fits), 4)  # as tall as its rows, up to four, then it scrolls
        height = sum(self.fit_list.sizeHintForRow(i) for i in range(rows)) + 2 * self.fit_list.frameWidth() + 2
        if height != self.fit_list.height():  # a new height moves the plot: only when it changes
            self.fit_list.setFixedHeight(height)
        self.remove_fit.setEnabled(len(self.fits) > 1)

    def _on_fit_row(self, row: int) -> None:
        if row < 0 or row == self.current or row >= len(self.fits):
            return
        self.current_entry.spec = self._spec_from_form()
        self.current = row
        self._form_from_spec(self.current_entry.spec)
        self._enable_controls()
        self.redraw()

    def _on_fit_checked(self, item) -> None:
        row = self.fit_list.row(item)
        if 0 <= row < len(self.fits):
            self.fits[row].shown = item.checkState() == Qt.CheckState.Checked
            self.redraw()

    def add_fit(self) -> None:
        """Another fit, starting as a copy of the one being edited."""
        self.current_entry.spec = self._spec_from_form()
        spec = replace(self.current_entry.spec, settings=dict(self.current_entry.spec.settings))
        used = {e.color for e in self.fits}
        color = next((c for c in _FIT_COLORS if c not in used), _FIT_COLORS[len(self.fits) % len(_FIT_COLORS)])
        self.fits.append(FitEntry(spec, color))
        self.current = len(self.fits) - 1
        self._enable_controls()
        self.redraw()

    def remove_current_fit(self) -> None:
        if len(self.fits) < 2:
            return
        self._pending.discard(self.current_entry.id)
        del self.fits[self.current]
        self.current = max(self.current - 1, 0)
        self._form_from_spec(self.current_entry.spec)
        self._enable_controls()
        self.redraw()

    def set_fits(self, specs, current: int = 0, shown=None) -> None:
        """Replace the fits (e.g. from a saved view); none of them fitted yet."""
        specs = list(specs) or [FitSpec()]
        self.fits = [FitEntry(spec, _FIT_COLORS[i % len(_FIT_COLORS)]) for i, spec in enumerate(specs)]
        for entry, on in zip(self.fits, shown or []):
            entry.shown = on
        self._pending.clear()
        self.current = min(max(current, 0), len(self.fits) - 1)
        self._form_from_spec(self.current_entry.spec)
        self._enable_controls()
        self.redraw()

    def _show_stats(self) -> None:
        if not self._loading_form:
            self.current_entry.spec = self._spec_from_form()
        self._maybe_autofit()
        self._fill_fit_list()
        entry = self.current_entry
        res = entry.result
        now = self._inputs_for(entry.spec)
        if entry.id in self._pending:
            self.fit_status.setText(f"fitting {entry.spec.name}…")
            return
        failed_now = entry.error and entry.failed == now
        if res is None:
            if failed_now:
                self.fit_status.setText(f"<span style='color:#b00'>{entry.error}</span>")
            else:
                self.fit_status.setText("choose a curve and press Fit" if entry.spec.name else "")
            return
        params = ", ".join(f"{k} {v:.4g}" for k, v in res.params.items())
        opts = entry.inputs[1]
        text = (f"<b>{res.label}</b> ({params}; fitted to {res.n} {'bins' if opts.bin_count else 'points'}, "
                f"RMSE {res.rmse:.3g})")
        if entry.inputs != now:
            if failed_now:
                text += (f" — <span style='color:#b00'>{entry.error}; the curve is the earlier one</span>"
                         if not self.fit_auto.isChecked() else
                         " — <span style='color:#b00'>this fit failed on what's shown now; the curve is the earlier "
                         "one</span>")
            elif not self.fit_auto.isChecked():
                text += " — <span style='color:#a60'>fitted to earlier data or settings; press Fit to refit</span>"
            else:
                text += " — <span style='color:#a60'>refitting…</span>"
        self.fit_status.setText(text)

    def figure_spec(self) -> tuple[str, dict] | None:
        """What's drawn, for a matplotlib figure (in data units; log axes stay log)."""
        if self.x.size == 0:
            return None
        slots, x, y = self._shown()
        groups = self.drawn_groups or [("", np.ones(slots.size, dtype=bool), _POINT)]
        ringed = self.selection.size and self.ring.isChecked()
        chosen = np.isin(slots, self.selection) if ringed else np.zeros(slots.size, dtype=bool)
        (x0, x1), (y0, y1) = self.plot.getPlotItem().getViewBox().viewRange()
        unview = (lambda v: 10 ** np.asarray(v)) if self.log_x.isChecked() else np.asarray
        unview_y = (lambda v: 10 ** np.asarray(v)) if self.log_y.isChecked() else np.asarray
        z = self.shown_z()
        spec = {
            "xlabel": self.axis_title(0), "ylabel": self.axis_title(1),
            "groups": [{"label": n, "color": c, "x": x[m], "y": y[m]} | ({"c": z[m]} if c is None else {})
                       for n, m, c in groups if m.any()],
            "lines": self.drawn_lines, "vlines": self.drawn_vlines,
            "selected": {"x": x[chosen], "y": y[chosen]} if chosen.any() else None,
            "note": self.stats.text(), "log_x": self.log_x.isChecked(), "log_y": self.log_y.isChecked(),
            "xlim": tuple(unview([x0, x1])), "ylim": tuple(unview_y([y0, y1]))}
        if self.drawn_z is not None:
            lo, hi, name = self.drawn_z
            log = self.log_z.isChecked()
            spec["z"] = {"label": self.axis_title(2), "cmap": name, "log": log,
                         "lim": (10 ** lo, 10 ** hi) if log else (lo, hi)}
        return "scatter", spec

    # --- brushing ----------------------------------------------------------------------------------

    def _on_lasso(self, px, py, mode: str) -> None:
        slots, x, y = self._shown()
        vx, vy = self._to_view(x, y)
        self.brushed.emit(slots[brush.in_polygon(vx, vy, px, py)], mode)

    def set_selection(self, slots: np.ndarray) -> None:
        self.selection = slots
        self.redraw()

    def _highlight(self, slots, vx, vy) -> None:
        """Ring the brushed slots' points (unless that's switched off)."""
        if self.selection.size == 0 or not self.ring.isChecked():
            return
        on = np.isin(slots, self.selection)
        if on.any():
            pen = pg.mkPen(style.SELECTED_COLOR, width=1.5)
            item = pg.ScatterPlotItem(vx[on], vy[on], size=10, brush=None, pen=pen,
                                      name=f"selected ({int(on.sum())})")
            item.setZValue(10)
            self.plot.addItem(item)

    def _on_click(self, _item, points, _ev) -> None:
        if len(points):
            i = int(points[0].data())
            member = (int(self._drawn_members[i]) if self._drawn_members is not None
                      else self.member[0].currentData())
            self.pointClicked.emit(int(self._drawn_slots[i]), member)

    # --- the 3D view -------------------------------------------------------------------------------

    def open_3d(self) -> None:
        if self.window3d is None:
            try:
                from ttu_tower.viewer.ui.scatter3d import Scatter3DWindow
            except ImportError as exc:  # PyOpenGL missing
                self.stats.setText(f"the 3D view needs PyOpenGL ({exc})")
                return
            self.window3d = Scatter3DWindow(self)
        self.window3d.show()
        self.window3d.raise_()
        self.window3d.activateWindow()
        self.window3d.refresh()

    def shows_3d(self) -> bool:
        return self.window3d is not None and shiboken6.isValid(self.window3d) and self.window3d.isVisible()

    def points3d(self):
        """(slots, x, y, z) of the shown points that have z, as plotted (log10 on a log axis)."""
        slots, x, y = self._shown()
        z = self.shown_z()
        have = np.isfinite(z)
        vx, vy = self._to_view(x[have], y[have])
        return slots[have], vx, vy, self._z_view(z[have])
