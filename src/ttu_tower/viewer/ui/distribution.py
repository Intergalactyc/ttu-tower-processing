"""Histograms of what a timeline panel shows - over its visible range, or the
whole period, in every stability class or one - with a summary table per boom,
and fitted distributions.
"""
import warnings

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ttu_tower.math.polar import yamartino_std
from ttu_tower.viewer import distfits
from ttu_tower.viewer.decimate import visible_slice
from ttu_tower.viewer.timeline import in_class
from ttu_tower.viewer.ui import style, tables

_STATS = ("N", "NaN %", "mean", "median", "σ", "p5", "p95")
TABLE_SHARE = 0.3  # the table's default share of the height it shares with its histogram


def summary(values: np.ndarray, circular: bool = False) -> list[float]:
    """N, NaN %, mean, median, σ, p5, p95. For a direction: the vector mean of
    unit vectors and the Yamartino σ; median and percentiles aren't defined.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0, 100.0 if values.size else np.nan] + [np.nan] * 5
    nan_pct = 100.0 * (1 - finite.size / values.size)
    if circular:
        s, c = np.sin(np.radians(finite)).mean(), np.cos(np.radians(finite)).mean()
        mean = np.degrees(np.arctan2(s, c)) % 360.0
        return [finite.size, nan_pct, mean, np.nan, float(yamartino_std(s, c)), np.nan, np.nan]
    p5, p50, p95 = np.percentile(finite, [5, 50, 95])
    return [finite.size, nan_pct, finite.mean(), p50, finite.std(), p5, p95]


def _fit_all(values: dict, name: str) -> dict:
    """Runs on a worker thread."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m: distfits.fit(name, v) for m, v in values.items()}


def stability_combo(tip: str) -> QComboBox:
    combo = QComboBox()
    combo.addItem("all", None)
    combo.setToolTip(tip)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    return combo


def set_stability_names(combo: QComboBox, stability) -> None:
    """The classes of the run's stability scheme (disabled without one)."""
    names = stability[1] if stability is not None else []
    if [combo.itemData(i) for i in range(1, combo.count())] == names:
        return
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem("all", None)
    for n in names:
        combo.addItem(n, n)
    combo.setCurrentIndex(max(combo.findData(current), 0))
    combo.setEnabled(bool(names))
    combo.blockSignals(False)


class DistributionView(QWidget):
    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.log_bins = QCheckBox("log bins")
        self.log_bins.toggled.connect(lambda *_: self.refresh())
        self.fit_box = QComboBox()
        self.fit_box.setToolTip("fit a distribution to each boom's values (maximum likelihood; dashed)")
        self.fit_box.currentIndexChanged.connect(lambda *_: self.refresh())
        self.stability_box = stability_combo("only the slots of one stability class (Ri_b of the configured pair)")
        self.stability_box.setEnabled(False)
        self.stability_box.currentIndexChanged.connect(lambda *_: self.refresh())
        self.fits: dict = {}
        self.plot = pg.PlotWidget()
        self.plot.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        for side in ("left", "bottom"):
            self.plot.getPlotItem().getAxis(side).enableAutoSIPrefix(False)
        self.table = tables.new_table()
        self.table.setToolTip("fit columns: KS is the Kolmogorov–Smirnov statistic (smaller fits better); "
                              "AIC, lower is better")
        self.note = QLabel()
        self.note.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.title)
        head = QHBoxLayout()
        head.addWidget(QLabel("stability"))
        head.addWidget(self.stability_box)
        head.addStretch(1)
        head.addWidget(QLabel("fit"))
        head.addWidget(self.fit_box)
        head.addWidget(self.log_bins)
        layout.addLayout(head)
        # the table scrolls, and the handle above it resizes it; by default it takes at most
        # TABLE_SHARE of the height, less when its rows need less
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.plot)
        self.split.addWidget(self.table)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.splitterMoved.connect(self._user_sized)
        self._sized_by_user = False
        layout.addWidget(self.split, stretch=1)
        layout.addWidget(self.note)
        self._args = None
        self._stats: pd.DataFrame | None = None
        self._log = False
        self._edges = np.linspace(0.0, 1.0, 2)
        self._fit_choices(False)
        self._size_table()

    def _fit_choices(self, circular: bool) -> None:
        names = [distfits.CIRCULAR] if circular else [n for n in distfits.available(np.array([1.0]), False)]
        current = self.fit_box.currentData()
        if [self.fit_box.itemData(i) for i in range(1, self.fit_box.count())] == names:
            return
        self.fit_box.blockSignals(True)
        self.fit_box.clear()
        self.fit_box.addItem("none", None)
        for n in names:
            self.fit_box.addItem(n, n)
        i = self.fit_box.findData(current)
        self.fit_box.setCurrentIndex(max(i, 0))
        self.fit_box.blockSignals(False)

    def set_data(self, data, members, x_range, stability=None) -> None:
        """`x_range`: (x0, x1) unix seconds, or None for the whole period;
        `stability`: (class-code Curve, class names), for the class subset.
        """
        set_stability_names(self.stability_box, stability)
        self._args = (data, list(members), x_range, stability)
        self.refresh()

    def values_by_member(self) -> dict:
        data, members, x_range, stability = self._args
        name = self.stability_box.currentData()
        out = {}
        for m in members:
            curve = data.curves.get(m)
            if curve is None:
                continue
            sl = slice(None) if x_range is None else visible_slice(curve.x, x_range[0], x_range[1], margin=0)
            y = curve.y[sl]
            out[m] = y if name is None else y[in_class(stability, curve.slots[sl], name)]
        return out

    def refresh(self) -> None:
        self.plot.clear()
        self.plot.getPlotItem().getAxis("bottom").setTicks(None)
        self.plot.setLabel("left", "")
        self.plot.setLabel("bottom", "")
        self.note.hide()
        if self._args is None or self._args[0] is None:
            self.title.setText("")
            self._show_table(pd.DataFrame())
            return
        data = self._args[0]
        q = data.quantity
        values = self.values_by_member()
        scope = "whole period" if self._args[2] is None else "visible range"
        name = self.stability_box.currentData()
        self.title.setText(f"<b>{q.title}</b> — {scope}" + (f", {name} only" if name else ""))
        self._fit_choices(q.circular)
        self._enable_controls(q, values)
        if q.categorical:
            self._bars(values, data.categories or [])
        else:
            self._histograms(values, q.unit, q.circular)
        self._stats = self._summary_frame(values, q.categorical, q.circular)
        self._show_table(self._stats)
        self._start_fits(values, q)

    def _enable_controls(self, q, values: dict) -> None:
        """Grey out what can't apply: fits and log bins to categories, log bins
        to directions or values that aren't all positive.
        """
        self.fit_box.setEnabled(not q.categorical)
        self.fit_box.setToolTip("categories aren't fitted" if q.categorical else
                                "fit a distribution to each boom's values (maximum likelihood; dashed)")
        finite = [v[np.isfinite(v)] for v in values.values()]
        positive = any(v.size for v in finite) and all((v > 0).all() for v in finite)
        why = ("categories have no bins" if q.categorical else "directions are binned over 0–360°" if q.circular
               else None if positive else "needs every value positive")
        self.log_bins.setEnabled(why is None)
        self.log_bins.setToolTip(why or "logarithmically spaced bins, on a log axis")

    # --- the table -----------------------------------------------------------------------------

    def _summary_frame(self, values: dict, categorical: bool, circular: bool) -> pd.DataFrame:
        rows = {}
        for member, v in values.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stats = summary(v, circular) if not categorical else summary(v)[:2] + [np.nan] * 5
            rows[style.member_label(member) or "value"] = stats
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(_STATS))
        return frame.dropna(axis=1, how="all")

    def _show_table(self, frame: pd.DataFrame) -> None:
        tables.fill(self.table, frame, index=True,
                    fmt=lambda col, v: f"{v:.0f}" if col == "N" and np.isfinite(v) else None)
        self._size_table()

    def _user_sized(self, *_) -> None:
        self._sized_by_user = True

    def _size_table(self) -> None:
        """Never taller than its rows; at least a row or two; by default no more than
        TABLE_SHARE of the space (it scrolls), unless its handle has been dragged.
        """
        rows = self.table.rowCount()
        self.table.setMinimumHeight(tables.rows_height(self.table, min(rows, 2)))
        self.table.setMaximumHeight(tables.rows_height(self.table, rows))
        if self._sized_by_user:
            return
        total = sum(self.split.sizes())
        if total <= 0:
            return
        table = min(self.table.maximumHeight(), max(self.table.minimumHeight(), int(TABLE_SHARE * total)))
        self.split.setSizes([total - table, table])

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._size_table)  # once the splitter has its new height

    # --- fits ---------------------------------------------------------------------------------

    def _start_fits(self, values: dict, q) -> None:
        name = self.fit_box.currentData()
        self.fits = {}
        if name is None or q.categorical:
            return
        values = {m: v[np.isfinite(v)] for m, v in values.items()}
        if name != distfits.CIRCULAR and name not in distfits.available(
                np.concatenate(list(values.values())) if values else np.empty(0), False):
            self.note.setText(f"{name} needs every value positive")
            self.note.show()
            return
        args = self._args
        if self.runner is None:
            self._fits_ready(args, _fit_all(values, name))
        else:
            self.runner.submit(_fit_all, values, name, key=f"distfit-{id(self)}", label=f"fitting {name}",
                               on_done=lambda fits: self._fits_ready(args, fits))

    def _fits_ready(self, args, fits: dict) -> None:
        if args is not self._args or self._stats is None:
            return
        self.fits = fits
        rows = {}
        for i, (member, f) in enumerate(fits.items()):
            if f is None:
                continue
            rows[style.member_label(member) or "value"] = {**f.params, "KS": f.ks, "AIC": f.aic}
            lo, hi = self._edges[0], self._edges[-1]
            x = np.geomspace(lo, hi, 300) if self._log else np.linspace(lo, hi, 300)
            with np.errstate(all="ignore"):
                y = f.pdf(x)
            self.plot.addItem(pg.PlotDataItem(np.log10(x) if self._log else x, y, pen=pg.mkPen(
                style.member_color(member, i), width=1.6, style=Qt.PenStyle.DashLine)))
        fitted = pd.DataFrame.from_dict(rows, orient="index")
        self._show_table(self._stats.join(fitted) if not fitted.empty else self._stats)

    # --- plots --------------------------------------------------------------------------------

    def _histograms(self, values: dict, unit: str, circular: bool = False) -> None:
        pooled = np.concatenate([v[np.isfinite(v)] for v in values.values()]) if values else np.empty(0)
        self.plot.setLogMode(x=False)
        if pooled.size < 2:
            return
        log = self.log_bins.isEnabled() and self.log_bins.isChecked()
        self._log = log
        if log:
            edges = np.logspace(np.log10(pooled.min()), np.log10(pooled.max()), 61)
        else:
            lo, hi = np.percentile(pooled, [0.1, 99.9]) if not circular else (0.0, 360.0)
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, 61)
        self._edges = edges
        for i, (member, v) in enumerate(values.items()):
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            counts, _ = np.histogram(v, bins=edges, density=True)
            x = np.log10(edges) if log else edges
            item = pg.PlotDataItem(x, counts, stepMode="center", pen=pg.mkPen(style.member_color(member, i), width=1.5))
            self.plot.addItem(item)
        self.plot.setLabel("bottom", ("log10 " if log else "") + (unit or "value"))
        self.plot.setLabel("left", "density")

    def _bars(self, values: dict, categories: list[str]) -> None:
        n = max(len(values), 1)
        width = 0.8 / n
        for i, (member, v) in enumerate(values.items()):
            codes = v[np.isfinite(v)].astype(int)
            counts = np.bincount(codes, minlength=len(categories)) if codes.size else np.zeros(len(categories))
            frac = counts / max(counts.sum(), 1)
            x = np.arange(len(categories)) - 0.4 + width * (i + 0.5)
            self.plot.addItem(pg.BarGraphItem(x=x, height=frac, width=width, brush=style.member_color(member, i)))
        self.plot.getPlotItem().getAxis("bottom").setTicks([[(i, c) for i, c in enumerate(categories)]])
        self.plot.setLabel("left", "fraction")
