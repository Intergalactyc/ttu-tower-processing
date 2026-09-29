"""Vertical profiles of what each timeline panel shows, over the viewed
interval (or the whole period): each boom's mean or median, optionally with
its spread, over all slots or per stability class, with a fitted curve.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ttu_tower.viewer import profiles
from ttu_tower.viewer.ui import style, tables

_ALL_COLOR = "#1f4e99"
OFFSET_PX = 3  # between stability classes' points at one height, so their bars don't overlap
CAP_PX = 6  # error-bar cap length


class ProfilePlot(QWidget):
    """One panel's profile: the plot, its fit choice and the fitted parameters."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.quantity_key = None
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.fit_box = QComboBox()
        self.fit_box.addItem("no fit", None)
        for f in profiles.FITS:
            self.fit_box.addItem(f, f)
        self.fit_box.setToolTip("a curve of the value against height, fitted to each profile's points (dashed)")
        self.plot = pg.PlotWidget()
        item = self.plot.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.15)
        for side in ("left", "bottom"):
            item.getAxis(side).enableAutoSIPrefix(False)
        item.setLabel("left", "height [m]")
        self.table = tables.new_table()
        self.stats: pd.DataFrame | None = None
        self.fits: dict = {}
        self._marks: list = []  # (points item, bars item or None, x, height, spread, class position, classes)
        self._log_z = False
        vb = self.plot.getViewBox()
        vb.sigYRangeChanged.connect(lambda *_: self._place())
        vb.sigResized.connect(lambda *_: self._place())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(self.title, stretch=1)
        head.addWidget(self.fit_box)
        layout.addLayout(head)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.table)
        tables.fit_rows(self.table)

    def set_quantity(self, q) -> None:
        """A newly chosen quantity takes its default fit (power law for wind speed and TI)."""
        key = q.key if q is not None else None
        if key != self.quantity_key:
            self.quantity_key = key
            self.fit_box.blockSignals(True)
            self.fit_box.setCurrentIndex(max(self.fit_box.findData(profiles.default_fit(q)), 0))
            self.fit_box.blockSignals(False)
        self.fit_box.setEnabled(profiles.relevant(q) and not q.circular)

    def clear(self, message: str) -> None:
        self.plot.clear()
        self._marks = []
        self.title.setText(message)
        self.stats, self.fits = None, {}
        tables.fill(self.table, pd.DataFrame())
        tables.fit_rows(self.table)

    def draw(self, q, stats: pd.DataFrame, method: str, error_bars: bool, lines: bool, log_z: bool, scope: str,
             stability) -> None:
        self.plot.clear()
        self._marks = []
        self._log_z = log_z
        self.stats = stats
        self.title.setText(f"<b>{q.title}</b> — {method}, {scope}")
        item = self.plot.getPlotItem()
        item.setLabel("bottom", q.unit or q.label)
        item.setLogMode(y=log_z)
        names = list(stability[1]) if stability is not None else []
        fit_name = self.fit_box.currentData() if self.fit_box.isEnabled() else None
        self.fits = {}
        rows = {}
        groups = [(g, part[np.isfinite(part["centre"])]) for g, part in stats.groupby("group", sort=False)]
        groups = [(g, part) for g, part in groups if not part.empty]
        for position, (g, part) in enumerate(groups):
            color = pg.mkColor(_ALL_COLOR) if g == profiles.ALL else style.stability_color(g, names.index(g))
            x, z = part["centre"].to_numpy(), part["height"].to_numpy()
            curve = pg.PlotDataItem(x, z, pen=pg.mkPen(color, width=1.5) if lines else None, symbol="o",
                                    symbolSize=7, symbolBrush=color, symbolPen=None)
            self.plot.addItem(curve)
            spread = part["spread"].to_numpy()
            bars = None
            if error_bars:
                bars = pg.ErrorBarItem(x=x, y=z, left=spread, right=spread, pen=pg.mkPen(color, width=1))
                self.plot.addItem(bars)
            self._marks.append((curve, bars, x, z, spread, position, len(groups)))
            rows[g] = {"slots (median)": int(part["n"].median())}
            if fit_name is not None:
                result, shown = profiles.fit_profile(fit_name, z, x)
                if result is None:
                    rows[g]["note"] = "too few booms or no fit"
                    continue
                self.fits[g] = result
                zz = np.geomspace(z.min(), z.max(), 100)
                with np.errstate(all="ignore"):
                    xx = result.func(zz)
                self.plot.addItem(pg.PlotDataItem(xx, zz, pen=pg.mkPen(
                    color, width=1.3, style=Qt.PenStyle.DashLine)))
                rows[g].update({**shown, "booms": result.n, "rmse": result.rmse})
        frame = pd.DataFrame.from_dict(rows, orient="index")
        tables.fill(self.table, frame, index=True)
        tables.fit_rows(self.table)
        # heights fixed in view, so the pixel offsets below can't feed back into the y range
        v = np.log10(stats["height"].to_numpy()) if log_z else stats["height"].to_numpy()
        pad = 0.05 * (v.max() - v.min() or 1.0)
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=True)
        vb.setYRange(v.min() - pad, v.max() + pad, padding=0)
        self._place()

    def _place(self) -> None:
        """Shift each class's points and bars a few pixels up or down (in view
        units, so again after a zoom or resize) and size the bars' caps.
        """
        if not self._marks:
            return
        py = self.plot.getViewBox().viewPixelSize()[1]
        py = py if np.isfinite(py) and py > 0 else 0.0
        for curve, bars, x, z, spread, position, count in self._marks:
            v = np.log10(z) if self._log_z else z
            shifted = v + (position - (count - 1) / 2) * OFFSET_PX * py
            # points go through log mode themselves; error bars don't, so they get view units
            curve.setData(x, 10 ** shifted if self._log_z else shifted)
            if bars is not None:
                bars.setData(x=x, y=shifted, left=spread, right=spread, beam=2 * CAP_PX * py)


class ProfilesView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.method = QComboBox()
        self.method.addItem("mean", "mean")
        self.method.addItem("median", "median")
        self.error_bars = QCheckBox("spread")
        self.lines = QCheckBox("lines")
        self.log_z = QCheckBox("log height")
        self.grouping = QComboBox()
        self.grouping.addItem("all slots", False)
        self.grouping.addItem("by stability", True)
        self.grouping.setToolTip("one profile over every slot, or one per stability class (Ri_b of the "
                                 "configured pair)")
        for w in (self.method, self.grouping):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        for w in (self.error_bars, self.lines, self.log_z):
            w.toggled.connect(lambda *_: self.redraw())
        self.key = QLabel()
        self.key.setTextFormat(Qt.TextFormat.RichText)
        self.key.setWordWrap(True)
        self.plots = [ProfilePlot("Panel A"), ProfilePlot("Panel B")]
        for p in self.plots:
            p.fit_box.currentIndexChanged.connect(lambda *_: self.redraw())
        self.sources = [None, None]  # (data, members) per panel
        self.x_range = None
        self.stability = None

        controls = QHBoxLayout()
        for w in (self.method, self.grouping, self.error_bars, self.lines, self.log_z):
            controls.addWidget(w)
        controls.addStretch(1)
        row = QHBoxLayout()
        for p in self.plots:
            row.addWidget(p, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(controls)
        layout.addWidget(self.key)
        layout.addLayout(row, stretch=1)
        self._enable_controls()

    def sizeHint(self) -> QSize:
        return QSize(600, 600)  # two plots' worth would widen the whole dock

    def set_sources(self, sources, x_range, stability) -> None:
        """`sources`: (TimelineData or None, members shown) per panel."""
        self.sources, self.x_range, self.stability = list(sources), x_range, stability
        self.redraw()

    def _quantities(self) -> list:
        return [s[0].quantity if s is not None and s[0] is not None else None for s in self.sources]

    def _enable_controls(self) -> None:
        """Median and MAD have no meaning for directions; per-class profiles need a stability scheme."""
        shown = [q for q in self._quantities() if profiles.relevant(q)]
        circular_only = bool(shown) and all(q.circular for q in shown)
        self.method.model().item(1).setEnabled(not circular_only)
        if circular_only and self.method.currentData() == "median":
            self.method.blockSignals(True)
            self.method.setCurrentIndex(0)
            self.method.blockSignals(False)
        self.method.setToolTip("directions: the vector mean and Yamartino σ (they have no median)" if circular_only
                               else "each boom's mean (spread: σ) or median (spread: MAD)")
        self.grouping.setEnabled(self.stability is not None)
        spread = "σ" if self.method.currentData() == "mean" else "MAD"
        self.error_bars.setText(f"± {spread}")
        what = "standard deviation" if spread == "σ" else "median absolute deviation"
        self.error_bars.setToolTip(f"horizontal bars: ± the {what}")
        for w in (self.method, self.grouping, self.error_bars, self.lines, self.log_z):
            w.setEnabled(bool(shown) and (w is not self.grouping or self.stability is not None))

    def redraw(self) -> None:
        self._enable_controls()
        scope = "whole period" if self.x_range is None else "visible range"
        by_stability = bool(self.grouping.currentData()) and self.stability is not None
        names = list(self.stability[1]) if by_stability else []
        self.key.setText(" ".join(f"<span style='color:{style.stability_color(n, i).name()}'>■ {n}</span>"
                                  for i, n in enumerate(names)))
        self.key.setVisible(by_stability)
        for plot, source, q in zip(self.plots, self.sources, self._quantities()):
            plot.set_quantity(q)
            if q is None:
                plot.clear(f"{plot.name}: nothing selected")
                continue
            if not profiles.relevant(q):
                plot.clear(f"{plot.name}: {q.title} has no per-boom profile")
                continue
            data, members = source
            method = "mean" if q.circular else self.method.currentData()
            stats = profiles.profile_stats(data, members, self.x_range, self.stability, method, by_stability)
            if stats.empty or not np.isfinite(stats["centre"]).any():
                plot.clear(f"<b>{q.title}</b>: no values in range")
                continue
            plot.draw(q, stats, method, self.error_bars.isChecked(), self.lines.isChecked(), self.log_z.isChecked(),
                      scope, self.stability)
