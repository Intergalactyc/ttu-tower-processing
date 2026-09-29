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
        self.sel_table = tables.new_table()
        self.sel_table.setToolTip(self.table.toolTip())
        self.captions = [QLabel(), QLabel()]
        for c in self.captions:
            c.setStyleSheet("color:#555;")
            c.hide()
        self.tables_box = QWidget()
        box = QVBoxLayout(self.tables_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(1)
        for w in (self.captions[0], self.table, self.captions[1], self.sel_table):
            box.addWidget(w)
        self.sel_table.hide()
        self.split.addWidget(self.plot)
        self.split.addWidget(self.tables_box)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.splitterMoved.connect(self._user_sized)
        self._sized_by_user = False
        layout.addWidget(self.split, stretch=1)
        layout.addWidget(self.note)
        self._args = None
        self.selection = np.empty(0, dtype=np.int64)
        self._stats: dict = {}  # part ("all", or "rest" and "selected") -> summary frame
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

    def slots_and_values(self) -> dict:
        """member -> (slots, values) in range (and in the chosen class)."""
        data, members, x_range, stability = self._args
        name = self.stability_box.currentData()
        out = {}
        for m in members:
            curve = data.curves.get(m)
            if curve is None:
                continue
            sl = slice(None) if x_range is None else visible_slice(curve.x, x_range[0], x_range[1], margin=0)
            slots, y = curve.slots[sl], curve.y[sl]
            if name is not None:
                keep = in_class(stability, slots, name)
                slots, y = slots[keep], y[keep]
            out[m] = (slots, y)
        return out

    def values_by_member(self) -> dict:
        return {m: y for m, (_, y) in self.slots_and_values().items()}

    def split_values(self, selected: bool) -> dict:
        """member -> values of the brushed slots (or of the rest)."""
        out = {}
        for m, (slots, y) in self.slots_and_values().items():
            chosen = np.isin(slots, self.selection)
            out[m] = y[chosen if selected else ~chosen]
        return out

    def set_selection(self, slots: np.ndarray) -> None:
        """Brushed slots: each boom's histogram splits into the selection
        (filled) and the rest (lines), each normalized on its own.
        """
        self.selection = slots
        self.refresh()

    def refresh(self) -> None:
        self.plot.clear()
        self.plot.getPlotItem().getAxis("bottom").setTicks(None)
        self.plot.setLabel("left", "")
        self.plot.setLabel("bottom", "")
        self.note.hide()
        if self._args is None or self._args[0] is None:
            self.title.setText("")
            self._stats = {}
            self._show_tables()
            return
        data = self._args[0]
        q = data.quantity
        values = self.values_by_member()
        scope = "whole period" if self._args[2] is None else "visible range"
        name = self.stability_box.currentData()
        split = self.selection.size > 0 and not q.categorical
        self.title.setText(f"<b>{q.title}</b> — {scope}" + (f", {name} only" if name else "")
                           + (" · filled: brushed selection, lines: the rest (each its own density)" if split else ""))
        self._fit_choices(q.circular)
        self._enable_controls(q, values)
        if q.categorical:
            self._bars(values, data.categories or [])
        elif split:
            self._histograms(self.split_values(False), q.unit, q.circular, bins_from=values)
            self._selected_histograms()
        else:
            self._histograms(values, q.unit, q.circular)
        parts = {"rest": self.split_values(False), "selected": self.split_values(True)} if split else {"all": values}
        self._stats = {part: self._summary_frame(v, q.categorical, q.circular) for part, v in parts.items()}
        self._show_tables()
        self._start_fits(parts, q)

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

    def figure_spec(self) -> dict | None:
        """This panel's histograms (or bars) and fits, for a matplotlib figure."""
        if self._args is None or self._args[0] is None:
            return None
        data = self._args[0]
        q = data.quantity
        split = self.selection.size > 0 and not q.categorical
        values = self.split_values(False) if split else self.values_by_member()
        panel = {"title": self.title.text().replace("<b>", "").replace("</b>", ""), "hists": [], "bars": [],
                 "fits": []}
        if q.categorical:
            categories = data.categories or []
            width = 0.8 / max(len(values), 1)
            for i, (m, v) in enumerate(values.items()):
                codes = v[np.isfinite(v)].astype(int)
                counts = np.bincount(codes, minlength=len(categories)) if codes.size else np.zeros(len(categories))
                panel["bars"].append({"label": style.member_label(m) or "value",
                                      "color": pg.mkColor(style.member_color(m, i)).name(),
                                      "x": np.arange(len(categories)) - 0.4 + width * (i + 0.5),
                                      "height": counts / max(counts.sum(), 1), "width": width})
            panel.update(xlabel="", ylabel="fraction", xticks=list(enumerate(categories)))
            return panel
        if self._edges.size < 2:
            return panel | {"xlabel": q.unit or "value", "ylabel": "density", "log_x": False}
        for i, (m, v) in enumerate(values.items()):
            v = v[np.isfinite(v)]
            if v.size:
                density, _ = np.histogram(v, bins=self._edges, density=True)
                label = style.member_label(m) or "value"
                panel["hists"].append({"label": f"{label}, the rest" if split else label,
                                       "color": pg.mkColor(style.member_color(m, i)).name(), "edges": self._edges,
                                       "density": density})
        if split:
            for i, (m, v) in enumerate(self.split_values(True).items()):
                v = v[np.isfinite(v)]
                if v.size:
                    density, _ = np.histogram(v, bins=self._edges, density=True)
                    panel["hists"].append({"label": f"{style.member_label(m) or 'value'}, selected",
                                           "color": pg.mkColor(style.member_color(m, i)).name(),
                                           "edges": self._edges, "density": density, "filled": True})
        lo, hi = self._edges[0], self._edges[-1]
        grid = np.geomspace(lo, hi, 300) if self._log else np.linspace(lo, hi, 300)
        members = list(self._args[1])
        for (part, m), f in self.fits.items():
            if f is not None:
                i = members.index(m) if m in members else 0
                with np.errstate(all="ignore"):
                    panel["fits"].append({"label": m, "color": pg.mkColor(style.member_color(m, i)).name(),
                                          "x": grid, "pdf": f.pdf(grid), "dashdot": part == "selected"})
        panel.update(xlabel=q.unit or "value", ylabel="density", log_x=bool(self._log))
        return panel

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

    def _show_tables(self, fitted: dict | None = None) -> None:
        """The summary per part, with its fit's parameters appended: one table
        for all slots, or the rest's and the brushed selection's.
        """
        fitted = fitted or {}
        split = "selected" in self._stats
        shown = ([("rest", self.table, "the rest (lines)"), ("selected", self.sel_table, "brushed selection (filled)")]
                 if split else [("all", self.table, "")])
        for part, table, _ in shown:
            frame = self._stats.get(part, pd.DataFrame())
            extra = fitted.get(part)
            if extra is not None and not extra.empty:
                # a normal fit's σ beside the sample σ
                frame = frame.join(extra.rename(columns={c: f"{c} (fit)" for c in extra.columns if c in frame.columns}))
            tables.fill(table, frame, index=True,
                        fmt=lambda col, v: f"{v:.0f}" if col == "N" and np.isfinite(v) else None)
        self.sel_table.setVisible(split)
        for label, (_, _, text) in zip(self.captions, shown if split else []):
            label.setText(text)
        for label in self.captions:
            label.setVisible(split)
        self._size_table()

    def _user_sized(self, *_) -> None:
        self._sized_by_user = True

    def _size_table(self) -> None:
        """Never taller than their rows; at least a row or two each; by default no
        more than TABLE_SHARE of the space (they scroll), unless the handle has
        been dragged.
        """
        shown = [t for t in (self.table, self.sel_table) if not t.isHidden()]
        captions = sum(c.sizeHint().height() + 1 for c in self.captions if not c.isHidden())
        for t in shown:
            t.setMinimumHeight(tables.rows_height(t, min(t.rowCount(), 2)))
            t.setMaximumHeight(tables.rows_height(t, t.rowCount()))
        low = captions + sum(t.minimumHeight() for t in shown)
        high = captions + sum(t.maximumHeight() for t in shown)
        self.tables_box.setMinimumHeight(low)
        self.tables_box.setMaximumHeight(high)
        if self._sized_by_user:
            return
        total = sum(self.split.sizes())
        if total <= 0:
            return
        share = min(high, max(low, int(TABLE_SHARE * total * (1.5 if len(shown) > 1 else 1.0))))
        self.split.setSizes([total - share, share])

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._size_table)  # once the splitter has its new height

    # --- fits ---------------------------------------------------------------------------------

    def _start_fits(self, parts: dict, q) -> None:
        """Fit each part (all slots, or the rest and the selection) of each boom."""
        name = self.fit_box.currentData()
        self.fits = {}
        if name is None or q.categorical:
            return
        values = {(part, m): v[np.isfinite(v)] for part, by_member in parts.items() for m, v in by_member.items()}
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
        """`fits`: (part, member) -> DistFit; PDFs dashed (the selection's dash-dotted)."""
        if args is not self._args or not self._stats:
            return
        self.fits = fits
        members = list(self._args[1])
        rows: dict = {}
        for (part, member), f in fits.items():
            if f is None:
                continue
            rows.setdefault(part, {})[style.member_label(member) or "value"] = {**f.params, "KS": f.ks, "AIC": f.aic}
            lo, hi = self._edges[0], self._edges[-1]
            x = np.geomspace(lo, hi, 300) if self._log else np.linspace(lo, hi, 300)
            with np.errstate(all="ignore"):
                y = f.pdf(x)
            dash = Qt.PenStyle.DashDotLine if part == "selected" else Qt.PenStyle.DashLine
            i = members.index(member) if member in members else 0
            curve = pg.PlotDataItem(np.log10(x) if self._log else x, y, pen=pg.mkPen(
                style.member_color(member, i), width=1.6, style=dash))
            part_label = {"rest": ", the rest", "selected": ", brushed selection"}.get(part, "")
            curve.export_label = f"{style.member_label(member) or 'value'}{part_label}: {self.fit_box.currentText()} PDF"
            self.plot.addItem(curve)
        self._show_tables({part: pd.DataFrame.from_dict(r, orient="index") for part, r in rows.items()})

    # --- plots --------------------------------------------------------------------------------

    def _histograms(self, values: dict, unit: str, circular: bool = False, bins_from: dict | None = None) -> None:
        """`bins_from`: the values the bins span (default `values`), so a split's parts share bins."""
        spanned = values if bins_from is None else bins_from
        pooled = np.concatenate([v[np.isfinite(v)] for v in spanned.values()]) if spanned else np.empty(0)
        self.plot.setLogMode(x=False)
        if pooled.size < 2:
            self._edges = np.empty(0)
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
            label = style.member_label(member) or "value"
            item.export_label = f"{label}: the rest" if self.selection.size else label
            self.plot.addItem(item)
        self.plot.setLabel("bottom", ("log10 " if log else "") + (unit or "value"))
        self.plot.setLabel("left", "density")

    def _selected_histograms(self) -> None:
        """The brushed slots' histogram per boom, filled, as a density of the
        selection alone (so its shape compares with the rest's).
        """
        if self._edges.size < 2:
            return
        for i, (member, y) in enumerate(self.split_values(True).items()):
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            share, _ = np.histogram(y, bins=self._edges, density=True)
            color = pg.mkColor(style.member_color(member, i))
            color.setAlpha(110)
            x = np.log10(self._edges) if self._log else self._edges
            item = pg.PlotDataItem(x, share, stepMode="center", fillLevel=0, brush=pg.mkBrush(color),
                                   pen=pg.mkPen(style.SELECTED_COLOR, width=1))
            item.export_label = f"{style.member_label(member) or 'value'}: brushed selection"
            self.plot.addItem(item)

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
