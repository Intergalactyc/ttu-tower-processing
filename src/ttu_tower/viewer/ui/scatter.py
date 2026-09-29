"""One quantity against another, slot by slot: by default panel A's against
panel B's, one boom (or pair) each, over the viewed interval or the whole
period. Points or a density map, optionally colored by stability;
correlation statistics; trend, binned central line, y = x; and any of the
curve fits (with limits and binning), fitted only on request. A lasso
(Shift-drag, or any drag in brush mode) selects slots for linked brushing.
"""
import warnings

import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ttu_tower.viewer import brush, curvefits
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import joined
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.timeline_plot import brush_mode

_POINT = "#1f4e9c"
_FIT = "#2ca02c"
_STALE = "#8fbf8f"


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


def _fit_job(name, x, y, opts, settings=()):
    """Runs on a worker thread."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return curvefits.fit(name, x, y, opts, **dict(settings))


class ScatterView(QWidget):
    pointClicked = Signal(int, object)  # slot, the x axis's member
    brushed = Signal(object, str)  # the lassoed slots, and "replace" / "add" / "remove"

    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.datas = [None, None]
        self.x_range = None  # None: the whole period
        self.stability = None  # (Curve of class codes, names)
        self.slots = np.empty(0, dtype=np.int64)
        self.x = self.y = np.empty(0)
        self.fit_result = None
        self.fit_inputs = None  # what fit_result was fitted to
        self._fitting = False

        self.source = [QComboBox(), QComboBox()]
        self.member = [QComboBox(), QComboBox()]
        for axis, (src, mem) in enumerate(zip(self.source, self.member)):
            src.addItem("panel A", 0)
            src.addItem("panel B", 1)
            src.setCurrentIndex(axis)
            src.currentIndexChanged.connect(lambda *_, a=axis: self._sync_members(a))
            mem.currentIndexChanged.connect(lambda *_: self.recompute())
        swap = QPushButton("⇄")
        swap.setToolTip("swap x and y")
        swap.setMaximumWidth(30)
        swap.clicked.connect(self._swap)

        self.display = QComboBox()
        for label in ("points", "density", "density (log)"):
            self.display.addItem(label, label)
        self.color_by = QComboBox()
        for label in ("single color", "stability"):
            self.color_by.addItem(label, label)
        self.log_x, self.log_y = QCheckBox("log x"), QCheckBox("log y")
        self.trend, self.mid, self.unit = QCheckBox("trend (linear)"), QCheckBox("binned"), QCheckBox("y = x")
        self.mid_method = QComboBox()
        self.mid_method.addItems(["median", "mean"])
        self.mid_bins = QSpinBox()
        self.mid_bins.setRange(3, 200)
        self.mid_bins.setValue(25)
        for w in (self.display, self.color_by, self.mid_method):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        for w in (self.log_x, self.log_y, self.trend, self.mid, self.unit):
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
        self.fit_bins = QSpinBox()
        self.fit_bins.setRange(0, 500)
        self.fit_bins.setSpecialValueText("no binning")
        self.fit_bin_method = QComboBox()
        self.fit_bin_method.addItems(["mean", "median"])
        self.fit_min_count = QSpinBox()
        self.fit_min_count.setRange(1, 10_000)
        for w in (self.fit_name, self.fit_bin_method):
            w.currentIndexChanged.connect(lambda *_: (self._show_stats(), self._enable_controls()))
        for w in (*self.pct, self.fit_bins, self.fit_min_count):
            w.valueChanged.connect(lambda *_: (self._show_stats(), self._enable_controls()))
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("fit the chosen curve to the points shown now (the fit stays put when the "
                                   "interval or settings change, until pressed again)")
        self.fit_button.clicked.connect(self.run_fit)
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
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.setToolTip("Shift-drag (or drag in brush mode) to lasso slots; Ctrl adds, Alt removes")
        self.selection = np.empty(0, dtype=np.int64)
        self.plot.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        for side in ("left", "bottom"):
            self.plot.getPlotItem().getAxis(side).enableAutoSIPrefix(False)
        self.legend = self.plot.getPlotItem().addLegend(offset=(5, 5), labelTextSize="8pt")

        axes = QGridLayout()
        for row, label in enumerate(("x", "y")):
            axes.addWidget(QLabel(label), row, 0)
            axes.addWidget(self.source[row], row, 1)
            axes.addWidget(self.member[row], row, 2)
        axes.addWidget(swap, 0, 3, 2, 1)
        axes.setColumnStretch(2, 1)
        look = QHBoxLayout()
        for w in (self.display, self.color_by, self.log_x, self.log_y):
            look.addWidget(w)
        look.addStretch(1)
        lines = QHBoxLayout()
        for w in (self.trend, self.mid, self.mid_method, self.mid_bins, self.unit):
            lines.addWidget(w)
        lines.addStretch(1)
        fit_box = QGroupBox("curve fit")
        form = QFormLayout(fit_box)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.fit_name, stretch=1)
        fit_row.addWidget(self.fit_button)
        form.addRow("fit", fit_row)
        pct_row = QHBoxLayout()
        pct_row.addWidget(self.pct[0])
        pct_row.addWidget(QLabel("to"))
        pct_row.addWidget(self.pct[1])
        form.addRow("x percentiles", pct_row)
        bin_row = QHBoxLayout()
        bin_row.addWidget(self.fit_bins)
        bin_row.addWidget(self.fit_bin_method)
        bin_row.addWidget(QLabel("min per bin"))
        bin_row.addWidget(self.fit_min_count)
        form.addRow("bins", bin_row)
        form.addRow(self.fit_settings_row)
        form.addRow(self.fit_status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(axes)
        layout.addLayout(look)
        layout.addLayout(lines)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.stats)
        layout.addWidget(fit_box)
        self._items: list = []

    # --- sources -----------------------------------------------------------------------------------

    def set_sources(self, datas, x_range, stability=None) -> None:
        """`x_range`: (x0, x1) unix seconds, or None for the whole period."""
        changed = [d is not o for d, o in zip(datas, self.datas)]
        self.datas, self.x_range, self.stability = list(datas), x_range, stability
        for axis in (0, 1):
            if changed[self.source[axis].currentData()]:
                self._sync_members(axis, recompute=False)
        self.recompute()

    def _data(self, axis: int):
        return self.datas[self.source[axis].currentData()]

    def _sync_members(self, axis: int, recompute: bool = True) -> None:
        box, data = self.member[axis], self._data(axis)
        current = box.currentData()
        box.blockSignals(True)
        box.clear()
        if data is not None:
            for i, member in enumerate(data.curves):
                box.addItem(style.member_label(member) or "value", member)
            other = self.member[1 - axis].currentData()
            i = box.findData(current)
            if i < 0 and other is not None:
                i = box.findData(other)  # the same boom on both axes by default
            box.setCurrentIndex(max(i, 0))
        box.blockSignals(False)
        if recompute:
            self.recompute()

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
        self.slots, self.x, self.y = np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
        if dx is None or dy is None or mx not in dx.curves or my not in dy.curves:
            self.redraw()
            return
        if dx.quantity.categorical or dy.quantity.categorical:
            self.stats.setText("categorical quantities can't be scattered")
            self.redraw()
            return
        slots, x, y = joined(dx.curves[mx], dy.curves[my])
        ok = np.isfinite(x) & np.isfinite(y)
        if self.x_range is not None:
            t = slot_to_unix(slots)
            ok &= (t >= self.x_range[0]) & (t < self.x_range[1])
        self.slots, self.x, self.y = slots[ok], x[ok], y[ok]
        self.redraw()

    # --- drawing -----------------------------------------------------------------------------------

    def _shown(self):
        """The points drawable on the current axes (positive where an axis is log)."""
        keep = np.ones(self.x.size, dtype=bool)
        if self.log_x.isChecked():
            keep &= self.x > 0
        if self.log_y.isChecked():
            keep &= self.y > 0
        return self.slots[keep], self.x[keep], self.y[keep]

    def _to_view(self, x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log10(x) if self.log_x.isChecked() else x), (np.log10(y) if self.log_y.isChecked() else y)

    def _enable_controls(self) -> None:
        """Grey out what can't apply to what's shown."""
        points = self.display.currentData() == "points"
        self.color_by.setEnabled(points)
        self.color_by.setToolTip("density maps aren't colored by class" if not points else "color the points")
        self.color_by.model().item(1).setEnabled(self.stability is not None)
        self.mid_method.setEnabled(self.mid.isChecked())
        self.mid_bins.setEnabled(self.mid.isChecked())
        for box, values in ((self.log_x, self.x), (self.log_y, self.y)):
            box.setEnabled(bool((values > 0).any()) or box.isChecked())
            box.setToolTip("no positive values to show on a log axis" if not box.isEnabled()
                           else "a log axis (values ≤ 0 are left out)")
        binned = self.fit_bins.value() > 0
        self.fit_bin_method.setEnabled(binned)
        self.fit_min_count.setEnabled(binned)
        self.fit_button.setEnabled(self.fit_name.currentData() is not None and self.x.size > 1 and not self._fitting)

    def redraw(self) -> None:
        self._enable_controls()
        self.plot.clear()
        self.legend.clear()
        dx, dy = self._data(0), self._data(1)
        if dx is not None and dy is not None:
            self.plot.setLabel("bottom", f"{dx.quantity.title} · {self.member[0].currentText()}")
            self.plot.setLabel("left", f"{dy.quantity.title} · {self.member[1].currentText()}")
        if self.x.size == 0:
            if not self.stats.text().startswith("categorical"):
                self.stats.setText("no slots with both values in range")
            return
        slots, x, y = self._shown()
        vx, vy = self._to_view(x, y)
        mode = self.display.currentData()
        if mode.startswith("density"):
            self._density(vx, vy, log=mode.endswith("(log)"))
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
            cx, cy = curvefits.bin_average(x, y, self.mid_bins.value(), self.mid_method.currentText(),
                                           bin_range=(lo, hi))
            self._line(cx, cy, "#d62728", f"{self.mid_method.currentText()}, {self.mid_bins.value()} bins", width=2)
        if self.unit.isChecked():
            self._line(grid, grid, "#555555", "y = x", dash=True)
        if self.fit_result is not None:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                fy = self.fit_result.func(grid)
            stale = self.fit_inputs != self._inputs()
            self._line(grid, fy, _STALE if stale else _FIT, self.fit_inputs[0] + (" (earlier)" if stale else ""),
                       width=2.2, dash=stale)
        self._show_stats()
        self.stats.setText(" · ".join(parts))
        self.stats.setToolTip("r: Pearson; ρ: Spearman; κ: Cohen's kappa on the signs of x and y "
                              "(— where both never change sign)")
        # like the paper's plots: frame the 1st-99th percentiles, so a few outliers don't squash the rest
        (x0, x1), (y0, y1) = (np.percentile(v, [1, 99]) for v in (vx, vy))
        pad_x, pad_y = 0.05 * (x1 - x0 or 1.0), 0.05 * (y1 - y0 or 1.0)
        self.plot.setRange(xRange=(x0 - pad_x, x1 + pad_x), yRange=(y0 - pad_y, y1 + pad_y), padding=0)

    def _points(self, slots, vx, vy) -> None:
        by = self.color_by.currentData()
        groups = [("", np.ones(slots.size, dtype=bool), pg.mkColor(_POINT))]
        if by == "stability" and self.stability is not None:
            curve, names = self.stability
            codes = pd.Series(curve.y, index=curve.slots).reindex(slots).to_numpy()
            groups = [(n, codes == i, style.stability_color(n, i)) for i, n in enumerate(names)]
            groups.append(("unclassified", ~np.isin(codes, range(len(names))), pg.mkColor("#bbbbbb")))
        size = 3 if slots.size > 5000 else 5
        for name, mask, color in groups:
            if not mask.any():
                continue
            color = pg.mkColor(color)
            color.setAlpha(150)
            item = pg.ScatterPlotItem(vx[mask], vy[mask], data=slots[mask], size=size, pen=None,
                                      brush=pg.mkBrush(color), name=name or None)
            item.sigClicked.connect(self._on_click)
            self.plot.addItem(item)

    def _density(self, vx, vy, log: bool) -> None:
        counts, ex, ey = np.histogram2d(vx, vy, bins=80, range=[np.percentile(vx, [0.5, 99.5]),
                                                                  np.percentile(vy, [0.5, 99.5])])
        img = np.log10(counts, where=counts > 0, out=np.full(counts.shape, np.nan)) if log else counts
        img = np.where(counts > 0, img, np.nan)
        item = pg.ImageItem(img, axisOrder="col-major")
        item.setColorMap(pg.colormap.get("viridis"))
        item.setRect(ex[0], ey[0], ex[-1] - ex[0], ey[-1] - ey[0])
        self.plot.addItem(item)

    def _line(self, x, y, color, label, width: float = 1.8, dash: bool = False) -> None:
        vx, vy = self._to_view(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        ok = np.isfinite(vx) & np.isfinite(vy)
        self.plot.plot(vx[ok], vy[ok], pen=pg.mkPen(color, width=width,
                                                  style=Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine),
                       name=label)

    # --- fitting, on request ------------------------------------------------------------------------

    def _options(self) -> curvefits.FitOptions:
        pct = (self.pct[0].value(), self.pct[1].value())
        return curvefits.FitOptions(pct_x=pct if pct != (0.0, 100.0) else None,
                                    bin_count=self.fit_bins.value() or None,
                                    bin_method=self.fit_bin_method.currentText(),
                                    bin_min_count=self.fit_min_count.value())

    def _build_fit_settings(self) -> None:
        while self.fit_settings_row.count():
            item = self.fit_settings_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.fit_settings = {}
        for arg, (label, default) in curvefits.FIT_SETTINGS.get(self.fit_name.currentData(), {}).items():
            edit = QLineEdit(f"{default:g}")
            edit.setMaximumWidth(70)
            edit.textChanged.connect(lambda *_: self._show_stats())
            self.fit_settings[arg] = edit
            self.fit_settings_row.addWidget(QLabel(label))
            self.fit_settings_row.addWidget(edit)
        self.fit_settings_row.addStretch(1)

    def _settings(self) -> tuple:
        """The chosen fit's own settings as (argument, value) pairs (ValueError if unreadable)."""
        out = []
        for arg, edit in self.fit_settings.items():
            try:
                out.append((arg, float(edit.text())))
            except ValueError:
                raise ValueError(f"{arg}: not a number") from None
        return tuple(out)

    def _inputs(self) -> tuple:
        """What a fit depends on: the curve, its settings, the axes and the points."""
        span = (int(self.slots[0]), int(self.slots[-1])) if self.slots.size else None
        try:
            settings = self._settings()
        except ValueError:
            settings = None
        return (self.fit_name.currentData() or "", self._options(), self.source[0].currentData(),
                self.member[0].currentData(), self.source[1].currentData(), self.member[1].currentData(),
                int(self.slots.size), span, self.log_x.isChecked(), self.log_y.isChecked(), settings)

    def _on_fit_choice(self, *_) -> None:
        if self.fit_name.currentData() is None and self.fit_result is not None:  # "none" takes the fit away
            self.run_fit()

    def run_fit(self) -> None:
        name = self.fit_name.currentData()
        if name is None:
            self.fit_result = self.fit_inputs = None
            self.redraw()
            return
        _, x, y = self._shown()
        inputs = self._inputs()
        if inputs[-1] is None:
            self.fit_status.setText("<span style='color:#b00'>a fit setting isn't a number</span>")
            return
        self.fit_status.setText(f"fitting {name} to {x.size} points…")
        self._fitting = True
        self.fit_button.setEnabled(False)
        if self.runner is None:
            try:
                self._fitted(inputs, _fit_job(name, x, y, inputs[1], inputs[-1]))
            except (ValueError, RuntimeError, TypeError) as exc:
                self._fit_failed(name, exc)
            return
        self.runner.submit(_fit_job, name, x, y, inputs[1], inputs[-1], key=f"scatterfit-{id(self)}",
                           label=f"fitting {name}",
                           on_done=lambda res: self._fitted(inputs, res),
                           on_error=lambda exc, tb: self._fit_failed(name, exc))

    def _fitted(self, inputs, result) -> None:
        if not shiboken6.isValid(self):
            return
        self._fitting = False
        self.fit_result, self.fit_inputs = result, inputs
        self.redraw()

    def _fit_failed(self, name, exc) -> None:
        if shiboken6.isValid(self):
            self._fitting = False
            self._enable_controls()
            self.fit_status.setText(f"<span style='color:#b00'>{name} fit failed: {exc}</span>")

    def _show_stats(self) -> None:
        res = self.fit_result
        if res is None:
            self.fit_status.setText("choose a curve and press Fit" if self.fit_name.currentData() else "")
            return
        params = ", ".join(f"{k} {v:.4g}" for k, v in res.params.items())
        opts = self.fit_inputs[1]
        text = (f"<b>{res.label}</b> ({params}; fitted to {res.n} {'bins' if opts.bin_count else 'points'}, "
                f"RMSE {res.rmse:.3g})")
        if self.fit_inputs != self._inputs():
            text += " — <span style='color:#a60'>fitted to earlier data or settings; press Fit to refit</span>"
        self.fit_status.setText(text)

    # --- brushing ----------------------------------------------------------------------------------

    def _on_lasso(self, px, py, mode: str) -> None:
        slots, x, y = self._shown()
        vx, vy = self._to_view(x, y)
        self.brushed.emit(slots[brush.in_polygon(vx, vy, px, py)], mode)

    def set_selection(self, slots: np.ndarray) -> None:
        self.selection = slots
        self.redraw()

    def _highlight(self, slots, vx, vy) -> None:
        """Ring the brushed slots' points."""
        if self.selection.size == 0:
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
            self.pointClicked.emit(int(points[0].data()), self.member[0].currentData())
