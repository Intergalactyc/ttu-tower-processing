"""Histograms of what a timeline panel shows - over its visible range, or the
whole period - with a summary table per boom.
"""
import warnings

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ttu_tower.math.polar import yamartino_std
from ttu_tower.viewer.decimate import visible_slice
from ttu_tower.viewer.ui import style

_STATS = ("N", "NaN %", "mean", "median", "σ", "p5", "p95")


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


class DistributionView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QLabel()
        self.log_bins = QCheckBox("log bins")
        self.log_bins.toggled.connect(lambda *_: self.refresh())
        self.plot = pg.PlotWidget()
        self.plot.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        for side in ("left", "bottom"):
            self.plot.getPlotItem().getAxis(side).enableAutoSIPrefix(False)
        self.table = QTableWidget(0, len(_STATS))
        self.table.setHorizontalHeaderLabels(_STATS)
        self.table.verticalHeader().setDefaultSectionSize(18)
        self.table.setMaximumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        head = QHBoxLayout()
        head.addWidget(self.title, stretch=1)
        head.addWidget(self.log_bins)
        layout.addLayout(head)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.table)
        self._args = None

    def set_data(self, data, members, x_range) -> None:
        """`x_range`: (x0, x1) unix seconds, or None for the whole period."""
        self._args = (data, list(members), x_range)
        self.refresh()

    def values_by_member(self) -> dict:
        data, members, x_range = self._args
        out = {}
        for m in members:
            curve = data.curves.get(m)
            if curve is None:
                continue
            if x_range is None:
                out[m] = curve.y
            else:
                sl = visible_slice(curve.x, x_range[0], x_range[1], margin=0)
                out[m] = curve.y[sl]
        return out

    def refresh(self) -> None:
        self.plot.clear()
        self.plot.getPlotItem().getAxis("bottom").setTicks(None)
        self.plot.setLabel("left", "")
        self.plot.setLabel("bottom", "")
        self.table.setRowCount(0)
        if self._args is None or self._args[0] is None:
            self.title.setText("")
            return
        data = self._args[0]
        q = data.quantity
        values = self.values_by_member()
        scope = "whole period" if self._args[2] is None else "visible range"
        self.title.setText(f"<b>{q.title}</b> — {scope}")
        if q.categorical:
            self._bars(values, data.categories or [])
        else:
            self._histograms(values, q.unit, q.circular)
        self._fill_table(values, q.categorical, q.circular)

    def _histograms(self, values: dict, unit: str, circular: bool = False) -> None:
        pooled = np.concatenate([v[np.isfinite(v)] for v in values.values()]) if values else np.empty(0)
        self.plot.setLogMode(x=False)
        if pooled.size < 2:
            return
        log = self.log_bins.isChecked() and (pooled > 0).all()
        if log:
            edges = np.logspace(np.log10(pooled.min()), np.log10(pooled.max()), 61)
        else:
            lo, hi = np.percentile(pooled, [0.1, 99.9]) if not circular else (0.0, 360.0)
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, 61)
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

    def _fill_table(self, values: dict, categorical: bool, circular: bool) -> None:
        self.table.setRowCount(len(values))
        labels = []
        for row, (member, v) in enumerate(values.items()):
            labels.append(style.member_label(member) or "value")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stats = summary(v, circular) if not categorical else summary(v)[:2] + [np.nan] * 5
            for col, s in enumerate(stats):
                text = "" if isinstance(s, float) and np.isnan(s) else (f"{s:.0f}" if col == 0 else f"{s:.4g}")
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.setVerticalHeaderLabels(labels)
