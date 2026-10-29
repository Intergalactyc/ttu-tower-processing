"""Histograms (or cumulative distributions) of what a timeline panel shows -
over its visible range, or the whole period, in every stability class or one -
with a summary table per boom, and fitted distributions.
"""
import html
import warnings

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ttu_tower.math.polar import yamartino_std
from ttu_tower.math.stats import moments
from ttu_tower.viewer import distfits
from ttu_tower.viewer.decimate import visible_slice
from ttu_tower.viewer.timeline import class_choices, in_class
from ttu_tower.viewer.ui import style, tables
from ttu_tower.viewer.ui.axes import SwitchLogAxis

_STATS = ("N", "NaN %", "mean", "σ", "skew", "kurt", "min", "p5", "p25", "median", "p75", "p95", "max")
_QUANTILES = (0, 5, 25, 50, 75, 95, 100)  # min to max, as percentiles
# column -> its label's tooltip (for a direction, _ABOUT_CIRCULAR's where it differs)
_ABOUT = {"N": "number of values", "NaN %": "share of values missing", "mean": "mean",
          "σ": "standard deviation", "skew": "skewness", "kurt": "kurtosis (3 for a normal)",
          "min": "smallest value", "p5": "5th percentile", "p25": "25th percentile (lower quartile)",
          "median": "median", "p75": "75th percentile (upper quartile)", "p95": "95th percentile",
          "max": "largest value", "KS": "Kolmogorov–Smirnov statistic: lower fits better",
          "AIC": "Akaike information criterion: lower fits better"}
_ABOUT_CIRCULAR = {"mean": "mean direction of unit vectors", "σ": "Yamartino standard deviation"}
TABLE_SHARE = 0.3  # the table's default share of the height it shares with its histogram
BINS = 60  # the default bin count
CDF_POINTS = 1000  # where the cumulative view evaluates each boom's distribution
# with slots brushed: the selection beside the rest, the selection alone, or every slot
SHOW = (("both", "selection and the rest"), ("selected", "selection only"), ("all", "all slots"))


def summary(values: np.ndarray, circular: bool = False) -> list[float]:
    """The _STATS columns: N, NaN %, mean, σ, skewness, kurtosis, then min, p5,
    p25, median, p75, p95, max. For a direction: the vector mean of unit vectors
    and the Yamartino σ; the rest aren't defined.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0, 100.0 if values.size else np.nan] + [np.nan] * (len(_STATS) - 2)
    nan_pct = 100.0 * (1 - finite.size / values.size)
    if circular:
        s, c = np.sin(np.radians(finite)).mean(), np.cos(np.radians(finite)).mean()
        mean = np.degrees(np.arctan2(s, c)) % 360.0
        return [finite.size, nan_pct, mean, float(yamartino_std(s, c))] + [np.nan] * (len(_STATS) - 4)
    skew, kurt = moments(finite)
    return [finite.size, nan_pct, finite.mean(), finite.std(), skew, kurt] + list(np.percentile(finite, _QUANTILES))


def ecdf(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The share of the finite values at or below each x."""
    v = np.sort(values[np.isfinite(values)])
    return np.searchsorted(v, x, side="right") / max(v.size, 1)


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
    names = class_choices(stability[1]) if stability is not None else []
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
    STATE_FIELDS = ("stability_box", "show_box", "cumulative", "side", "fit_box", "bin_count", "bin_auto",
                    "bin_lo", "bin_hi", "log_bins")  # saved in a view, restored in this order

    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.log_bins = QCheckBox("log bins")
        self.log_bins.toggled.connect(lambda *_: self.refresh())
        self.side = QComboBox()
        for label, key in (("all values", "all"), ("x > 0 only", "positive"), ("x < 0 only, as −x", "negative"),
                           ("|x|", "abs")):
            self.side.addItem(label, key)
        self.side.setToolTip("all the values, one side of 0 (the negative side as −x, its magnitudes), or the "
                             "absolute values: a side or |x| can take log bins and positive-only fits")
        self.side.currentIndexChanged.connect(lambda *_: self.refresh())
        self.fit_box = QComboBox()
        self.fit_box.setToolTip("fit a distribution to each boom's values (maximum likelihood; dashed)")
        self.fit_box.currentIndexChanged.connect(lambda *_: self.refresh())
        self.stability_box = stability_combo("only the slots of one stability class (as classified under Controls → Stability)")
        self.stability_box.setEnabled(False)
        self.stability_box.currentIndexChanged.connect(lambda *_: self.refresh())
        self.show_box = QComboBox()
        for key, label in SHOW:
            self.show_box.addItem(label, key)
        self.show_box.setToolTip("with slots brushed: the selection (filled) beside the rest (lines), the "
                                 "selection alone, or every slot as if nothing were brushed")
        self.show_box.setEnabled(False)
        self.show_box.currentIndexChanged.connect(lambda *_: self.refresh())
        self.cumulative = QCheckBox("cumulative")
        self.cumulative.toggled.connect(lambda *_: self.refresh())
        self.bin_count = QSpinBox()
        self.bin_count.setRange(5, 500)
        self.bin_count.setValue(BINS)
        self.bin_count.setToolTip("how many bins span the range")
        self.bin_auto = QCheckBox("auto range")
        self.bin_auto.setChecked(True)
        self.bin_lo, self.bin_hi = QDoubleSpinBox(), QDoubleSpinBox()
        for box in (self.bin_lo, self.bin_hi):
            box.setRange(-1e9, 1e9)
            box.setDecimals(3)
            box.setEnabled(False)
            box.setKeyboardTracking(False)  # redraw once a value is entered, not on every keystroke
            box.valueChanged.connect(lambda *_: self.refresh())
        self.bin_count.valueChanged.connect(lambda *_: self.refresh())
        self.bin_auto.toggled.connect(self._on_bin_auto)
        self.fits: dict = {}
        self._fitted = None  # (args, inputs, fits) of the last fit, reused while only the view changes
        self.plot = pg.PlotWidget(axisItems={"bottom": SwitchLogAxis(orientation="bottom")})
        self.plot.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        self.table = tables.new_table()
        self.note = QLabel()
        self.note.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.title)
        head = QHBoxLayout()
        head.addWidget(QLabel("stability"))
        head.addWidget(self.stability_box)
        head.addWidget(QLabel("show"))
        head.addWidget(self.show_box)
        head.addWidget(self.cumulative)
        head.addWidget(self.side)
        head.addStretch(1)
        head.addWidget(QLabel("fit"))
        head.addWidget(self.fit_box)
        layout.addLayout(head)
        bins = QHBoxLayout()
        bins.addWidget(QLabel("bins"))
        bins.addWidget(self.bin_count)
        bins.addWidget(self.bin_auto)
        bins.addWidget(QLabel("from"))
        bins.addWidget(self.bin_lo)
        bins.addWidget(QLabel("to"))
        bins.addWidget(self.bin_hi)
        bins.addWidget(self.log_bins)
        bins.addStretch(1)
        layout.addLayout(bins)
        # the table scrolls, and the handle above it resizes it; by default it takes at most
        # TABLE_SHARE of the height, less when its rows need less
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.sel_table = tables.new_table()
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
        names = distfits.names(circular)
        current = self.fit_box.currentData()
        if [self.fit_box.itemData(i) for i in range(1, self.fit_box.count())] == names:
            return
        self.fit_box.blockSignals(True)
        self.fit_box.clear()
        self.fit_box.addItem("none", None)
        for n in names:
            self.fit_box.addItem(n, n)
            self.fit_box.setItemData(self.fit_box.count() - 1, distfits.formula(n), Qt.ItemDataRole.ToolTipRole)
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
            out[m] = self._side(slots, y)
        return out

    def state_skip(self) -> set:
        """With auto range on, the from/to boxes show what auto chose: not a setting to keep."""
        return {"bin_lo", "bin_hi"} if self.bin_auto.isChecked() else set()

    def side_key(self) -> str:
        """Which values: "all", "positive", "negative" (as −x) or "abs"; all for directions and categories."""
        data = self._args[0] if self._args is not None else None
        q = data.quantity if data is not None else None
        return "all" if q is None or q.categorical or q.circular else self.side.currentData()

    def _side(self, slots, y):
        key = self.side_key()
        if key == "positive":
            keep = y > 0
            return slots[keep], y[keep]
        if key == "negative":
            keep = y < 0
            return slots[keep], -y[keep]
        if key == "abs":
            return slots, np.abs(y)
        return slots, y

    def x_label(self, unit: str) -> str:
        what = {"all": "", "positive": "x > 0 · ", "negative": "−x (x < 0) · ", "abs": "|x| · "}[self.side_key()]
        return f"{what}{unit or 'value'}" + (" (log scale)" if self._log else "")

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
        """Brushed slots: by default each boom's histogram splits into the
        selection (filled) and the rest (lines), each normalized on its own.
        """
        self.selection = slots
        self.refresh()

    def showing(self) -> str:
        """"both" (the selection and the rest), "selected", or "all"; "all" when nothing is brushed."""
        return self.show_box.currentData() if self.selection.size else "all"

    def refresh(self) -> None:
        self.plot.clear()
        self._log = False
        self.plot.getPlotItem().getAxis("bottom").setTicks(None)
        self.plot.getPlotItem().getAxis("bottom").set_log10(False)
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
        showing = self.showing()
        split = showing == "both" and not q.categorical
        cumulative = self.cumulative.isChecked() and not q.categorical
        both = ("highlighted: brushed selection, plain: the rest (each its own fraction)" if cumulative
                else "filled: brushed selection, lines: the rest (each its own density)")
        note = {"both": f" · {both}", "selected": " · the brushed selection only", "all": ""}[showing]
        side = {"all": "", "positive": ", values > 0 only", "negative": ", values < 0 only, as −x",
                "abs": ", absolute values"}[self.side_key()]
        self.title.setText(f"<b>{q.title}</b>{html.escape(side)} — {scope}" + (f", {name} only" if name else "")
                           + (note if split or showing == "selected" else ""))
        self._fit_choices(q.circular)
        self.show_box.setEnabled(self.selection.size > 0)
        if showing == "selected":
            values = self.split_values(True)
        self._enable_controls(q, values)
        if q.categorical:
            self._bars(values, data.categories or [])
        elif cumulative:
            if self._set_edges(values, q.circular):
                if split:
                    self._cumulatives(self.split_values(False), "the rest")
                    self._cumulatives(self.split_values(True), "brushed selection", selected=True)
                else:
                    self._cumulatives(values, "brushed selection" if showing == "selected" else "")
                self.plot.setLabel("bottom", html.escape(self.x_label(q.unit)))
                self.plot.setLabel("left", "cumulative fraction")
        elif split:
            self._histograms(self.split_values(False), q.unit, q.circular, bins_from=values, part="the rest")
            self._selected_histograms()
        else:
            self._histograms(values, q.unit, q.circular, part="brushed selection" if showing == "selected" else "")
        if split:
            parts = {"rest": self.split_values(False), "selected": self.split_values(True)}
        else:
            parts = {"selected" if showing == "selected" else "all": values}
        self._stats = {part: self._summary_frame(v, q.categorical, q.circular) for part, v in parts.items()}
        self._show_tables()
        self._start_fits(parts, q)

    def _on_bin_auto(self, auto: bool) -> None:
        """A range of one's own starts from the automatic one."""
        for box in (self.bin_lo, self.bin_hi):
            box.setEnabled(not auto)
        self.refresh()

    def _enable_controls(self, q, values: dict) -> None:
        """Grey out what can't apply: fits and log bins to categories, log bins
        to directions or values that aren't all positive.
        """
        self.fit_box.setEnabled(not q.categorical)
        fit = self.fit_box.currentData()
        self.fit_box.setToolTip(
            "categories aren't fitted" if q.categorical else
            f"<b>{fit}</b>, fitted to each boom's values by maximum likelihood (dashed)<br>{distfits.formula(fit)}"
            if fit is not None else "fit a distribution to each boom's values (maximum likelihood; dashed)")
        cumulative = self.cumulative.isChecked() and not q.categorical
        self.cumulative.setEnabled(not q.categorical)
        self.cumulative.setToolTip("categories have no order to accumulate" if q.categorical else
                                   "each boom's cumulative distribution, from all its values, with the fit's CDF")
        finite = [v[np.isfinite(v)] for v in values.values()]
        positive = any(v.size for v in finite) and all((v > 0).all() for v in finite)
        why = ("categories have no bins" if q.categorical else "directions are binned over 0–360°" if q.circular
               else None if positive else "needs every value positive: choose x > 0, x < 0 or |x| beside "
                                                  "\"cumulative\"")
        self.log_bins.setEnabled(why is None)
        self.side.setEnabled(not q.categorical and not q.circular)
        self.side.setToolTip("directions and categories are shown whole" if not self.side.isEnabled()
                             else self.side.toolTip())
        self.log_bins.setToolTip(why or ("a log axis" if cumulative else "logarithmically spaced bins, on a log axis"))
        self.bin_auto.setEnabled(not q.categorical)
        self.bin_auto.setToolTip(
            "the view spans the 0.1st to 99.9th percentiles (0–360° for directions); untick to set the range"
            if cumulative else "the bins span the 0.1st to 99.9th percentiles (0–360° for directions); untick to "
            "set the range (values outside it are left out)")
        self.bin_count.setEnabled(not q.categorical and not cumulative)
        self.bin_count.setToolTip("the cumulative view isn't binned" if cumulative else "how many bins span the range")
        for box in (self.bin_lo, self.bin_hi):
            box.setEnabled(not q.categorical and not self.bin_auto.isChecked())

    def figure_spec(self) -> dict | None:
        """This panel's histograms (or bars) and fits, for a matplotlib figure."""
        if self._args is None or self._args[0] is None:
            return None
        data = self._args[0]
        q = data.quantity
        showing = self.showing()
        split = showing == "both" and not q.categorical
        if split:
            values = self.split_values(False)
        else:
            values = self.split_values(True) if showing == "selected" else self.values_by_member()
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
        cumulative = self.cumulative.isChecked()
        ylabel = "cumulative fraction" if cumulative else "density"
        if self._edges.size < 2:
            return panel | {"xlabel": q.unit or "value", "ylabel": ylabel, "log_x": False}
        cdf_x = self._grid(CDF_POINTS)
        parts = [(values, False)] + ([(self.split_values(True), True)] if split else [])
        for by_member, selected in parts:
            for i, (m, v) in enumerate(by_member.items()):
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                label = style.member_label(m) or "value"
                suffix = ", selected" if selected or showing == "selected" else ", the rest" if split else ""
                entry = {"label": f"{label}{suffix}", "color": pg.mkColor(style.member_color(m, i)).name()}
                if cumulative:
                    panel.setdefault("cdfs", []).append(entry | {"x": cdf_x, "y": ecdf(v, cdf_x), "selected": selected})
                else:
                    density, _ = np.histogram(v, bins=self._edges, density=True)
                    panel["hists"].append(entry | {"edges": self._edges, "density": density, "filled": selected})
        grid = self._grid()
        members = list(self._args[1])
        for (part, m), f in self.fits.items():
            if f is not None:
                i = members.index(m) if m in members else 0
                with np.errstate(all="ignore"):
                    curve = {"cdf": f.cdf(grid)} if cumulative else {"pdf": f.pdf(grid)}
                panel["fits"].append({"label": m, "color": pg.mkColor(style.member_color(m, i)).name(), "x": grid,
                                      "dashdot": part == "selected"} | curve)
        panel.update(xlabel=self.x_label(q.unit).replace(" (log scale)", ""), ylabel=ylabel, log_x=bool(self._log))
        return panel

    # --- the table -----------------------------------------------------------------------------

    def _summary_frame(self, values: dict, categorical: bool, circular: bool) -> pd.DataFrame:
        rows = {}
        for member, v in values.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stats = summary(v, circular) if not categorical else summary(v)[:2] + [np.nan] * (len(_STATS) - 2)
            rows[style.member_label(member) or "value"] = stats
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(_STATS))
        return frame.dropna(axis=1, how="all")

    def _show_tables(self, fitted: dict | None = None, formulas: dict | None = None) -> None:
        """The summary per part, with its fit's parameters appended: one table
        for all slots, or the rest's and the brushed selection's. `formulas`:
        part -> row label -> the fitted PDF, for the row label's tooltip.
        """
        fitted, formulas = fitted or {}, formulas or {}
        split = "rest" in self._stats
        if split:
            plain, marked = ("plain", "highlighted") if self.cumulative.isChecked() else ("lines", "filled")
            shown = [("rest", self.table, f"the rest ({plain})"),
                     ("selected", self.sel_table, f"brushed selection ({marked})")]
        else:
            shown = [("selected" if "selected" in self._stats else "all", self.table, "")]
        for part, table, _ in shown:
            frame = self._stats.get(part, pd.DataFrame())
            extra = fitted.get(part)
            if extra is not None and not extra.empty:
                # a normal fit's σ beside the sample σ
                frame = frame.join(extra.rename(columns={c: f"{c} (fit)" for c in extra.columns if c in frame.columns}))
            tables.fill(table, frame, index=True,
                        fmt=lambda col, v: f"{v:.0f}" if col == "N" and np.isfinite(v) else None)
            self._header_tips(table)
            for i, label in enumerate(frame.index):
                if label in formulas.get(part, {}):
                    table.verticalHeaderItem(i).setToolTip(f"<b>{label}</b>, {self.fit_box.currentText()} fit<br>"
                                                           f"{formulas[part][label]}")
        self.sel_table.setVisible(split)
        for label, (_, _, text) in zip(self.captions, shown if split else []):
            label.setText(text)
        for label in self.captions:
            label.setVisible(split)
        self._size_table()

    def _header_tips(self, table) -> None:
        """Each column label's tooltip says what that column is; the cells have none."""
        q = self._args[0].quantity if self._args is not None and self._args[0] is not None else None
        fit = self.fit_box.currentData()
        for j in range(table.columnCount()):
            item = table.horizontalHeaderItem(j)
            label = item.text()
            if fit is not None and (label.endswith(" (fit)") or label not in _ABOUT):
                words = distfits.about(fit, label.removesuffix(" (fit)"))
                tip = f"{fit} fit: {words}" if words else ""
            else:
                tip = (_ABOUT_CIRCULAR.get(label) if q is not None and q.circular else None) or _ABOUT.get(label, "")
            item.setToolTip(tip)

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
            self.note.setText(f"{name} needs {distfits.needs(name)}")
            self.note.show()
            return
        args = self._args
        # the same values and fit as last time (only the view changed): no need to fit again
        inputs = (name, self.stability_box.currentData(), self.showing(), self.selection.tobytes())
        if self._fitted is not None and self._fitted[0] is args and self._fitted[1] == inputs:
            self._fits_ready(args, self._fitted[2])
            return

        def done(fits):
            self._fitted = (args, inputs, fits)
            self._fits_ready(args, fits)

        if self.runner is None:
            done(_fit_all(values, name))
        else:
            self.runner.submit(_fit_all, values, name, key=f"distfit-{id(self)}", label=f"fitting {name}",
                               on_done=done)

    def _fits_ready(self, args, fits: dict) -> None:
        """`fits`: (part, member) -> DistFit; PDFs (or CDFs) dashed, the selection's dash-dotted."""
        if args is not self._args or not self._stats:
            return
        self.fits = fits
        members = list(self._args[1])
        cumulative = self.cumulative.isChecked()
        rows: dict = {}
        formulas: dict = {}
        for (part, member), f in fits.items():
            if f is None:
                continue
            label = style.member_label(member) or "value"
            rows.setdefault(part, {})[label] = {**f.params, "KS": f.ks, "AIC": f.aic}
            formulas.setdefault(part, {})[label] = distfits.filled(f)
            x = self._grid()
            with np.errstate(all="ignore"):
                y = f.cdf(x) if cumulative else f.pdf(x)
            dash = Qt.PenStyle.DashDotLine if part == "selected" else Qt.PenStyle.DashLine
            i = members.index(member) if member in members else 0
            curve = pg.PlotDataItem(np.log10(x) if self._log else x, y, pen=pg.mkPen(
                style.member_color(member, i), width=1.6, style=dash))
            part_label = {"rest": ", the rest", "selected": ", brushed selection"}.get(part, "")
            curve.export_label = (f"{style.member_label(member) or 'value'}{part_label}: {self.fit_box.currentText()} "
                                  + ("CDF" if cumulative else "PDF"))
            self.plot.addItem(curve)
        self._show_tables({part: pd.DataFrame.from_dict(r, orient="index") for part, r in rows.items()}, formulas)

    # --- plots --------------------------------------------------------------------------------

    def _grid(self, n: int = 300) -> np.ndarray:
        """`n` points across the range, log-spaced with log bins."""
        lo, hi = self._edges[0], self._edges[-1]
        return np.geomspace(lo, hi, n) if self._log else np.linspace(lo, hi, n)

    def _cumulatives(self, values: dict, part: str = "", selected: bool = False) -> None:
        """Each boom's empirical CDF across the range, from all its values (the
        range only frames it); the brushed selection's highlighted.
        """
        x = self._grid(CDF_POINTS)
        for i, (member, v) in enumerate(values.items()):
            if not np.isfinite(v).any():
                continue
            halo = pg.mkPen(style.SELECTED_COLOR, width=4.5) if selected else None
            item = pg.PlotDataItem(np.log10(x) if self._log else x, ecdf(v, x), shadowPen=halo,
                                   pen=pg.mkPen(style.member_color(member, i), width=1.8 if selected else 1.5))
            label = style.member_label(member) or "value"
            item.export_label = f"{label}: {part}" if part else label
            self.plot.addItem(item)

    def _set_edges(self, spanned: dict, circular: bool) -> bool:
        """The bins (the cumulative view's range) for `spanned`'s values; False
        when there are too few to span.
        """
        pooled = np.concatenate([v[np.isfinite(v)] for v in spanned.values()]) if spanned else np.empty(0)
        self.plot.setLogMode(x=False)
        if pooled.size < 2:
            self._edges = np.empty(0)
            return False
        log = self.log_bins.isEnabled() and self.log_bins.isChecked()
        self._log = log
        self.plot.getPlotItem().getAxis("bottom").set_log10(log)  # the axis reads the values, not log10
        n = self.bin_count.value() + 1
        if self.bin_auto.isChecked():
            # the 0.1st-99.9th percentiles: a side of 0 always has a few tiny magnitudes to stretch a log axis
            lo, hi = np.percentile(pooled, [0.1, 99.9]) if not circular else (0.0, 360.0)
            for box, v in ((self.bin_lo, lo), (self.bin_hi, hi)):  # what auto chose, to start a range from
                box.blockSignals(True)
                box.setValue(float(v))
                box.blockSignals(False)
        else:
            lo, hi = self.bin_lo.value(), self.bin_hi.value()
            if log and lo <= 0:
                self.note.setText("log bins need a range above 0: the lower end was raised to the smallest value")
                self.note.show()
                lo = pooled.min()
        if hi <= lo:
            lo, hi = (lo / 2, lo * 2) if log else (lo - 0.5, lo + 0.5)
        self._edges = np.logspace(np.log10(lo), np.log10(hi), n) if log else np.linspace(lo, hi, n)
        return True

    def _histograms(self, values: dict, unit: str, circular: bool = False, bins_from: dict | None = None,
                    part: str = "") -> None:
        """`bins_from`: the values the bins span (default `values`), so a split's parts share bins;
        `part`: what the values are, for the CSV export ("the rest", "brushed selection").
        """
        if not self._set_edges(values if bins_from is None else bins_from, circular):
            return
        edges, log = self._edges, self._log
        for i, (member, v) in enumerate(values.items()):
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            counts, _ = np.histogram(v, bins=edges, density=True)
            x = np.log10(edges) if log else edges
            item = pg.PlotDataItem(x, counts, stepMode="center", pen=pg.mkPen(style.member_color(member, i), width=1.5))
            label = style.member_label(member) or "value"
            item.export_label = f"{label}: {part}" if part else label
            self.plot.addItem(item)
        self.plot.setLabel("bottom", html.escape(self.x_label(unit)))
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
