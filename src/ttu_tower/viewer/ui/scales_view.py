"""Integral scales over the viewed interval (or the whole period): per boom,
the median (or mean) ITS or ILS (optionally e-folding corrected) of u, v, w or θv at every ladder rung, with
interquartile (or ± SD) bars, for every slot, one stability class or the
brushed selection; and the values as a table.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ttu_tower.constants import HEIGHTS
from ttu_tower.math.stats import EFOLDING_TOTAL_FACTOR
from ttu_tower.viewer import composite
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import in_class
from ttu_tower.viewer.ui import style, tables
from ttu_tower.viewer.ui.axes import Log10Axis
from ttu_tower.viewer.ui.timeline_plot import LegendBar

_VARIANTS = (("mrd", "mrd"), ("mrd_unexcised", "mrd unexcised"))
_CENTERS = (("median", "median, interquartile bars"), ("mean", "mean, ± SD bars"))


def _load(run, index, booms, variant) -> dict:
    """Runs on a worker thread: boom -> (slots, rungs, {quantity: values})."""
    return {b: composite.scales_matrices(run, index, b, variant) for b in booms}


class ScalesView(QWidget):
    STATE_FIELDS = ("quantity", "corrected", "variant", "group", "center", "bars", "log_y", "boom_bar")  # saved in a view, restored in this order

    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = self.cfg = None
        self.booms: list[int] = []
        self.x_range = None
        self.stability = None
        self.selection = np.empty(0, dtype=np.int64)
        self.matrices: dict = {}
        self.summaries: dict = {}  # boom -> DataFrame, as drawn
        self._loaded_key = None

        self.quantity = QComboBox()
        for key, label, unit in composite.SCALES:
            self.quantity.addItem(f"{label} [{unit}]", key)
        self.quantity.setToolTip("ITS: the integral time scale from the pooled autocorrelation at each rung; "
                                 "ILS: ITS × |u mean| at that rung")
        self.quantity.currentIndexChanged.connect(lambda *_: self.redraw())
        self.corrected = QCheckBox("corrected")
        self.corrected.setToolTip(f"× e/(e−1) ≈ {EFOLDING_TOTAL_FACTOR:.3f}: the full integral of an exponential "
                                  "ACF over its integral up to the 1/e crossing")
        self.corrected.toggled.connect(lambda *_: self.redraw())
        self.variant = QComboBox()
        for key, label in _VARIANTS:
            self.variant.addItem(label, key)
        self.variant.currentIndexChanged.connect(lambda *_: self.reload())
        self.group = QComboBox()
        self.group.setToolTip("which slots: all, one stability class, or the brushed selection (or the rest)")
        self.group.currentIndexChanged.connect(lambda *_: self.redraw())
        self.center = QComboBox()
        for key, label in _CENTERS:
            self.center.addItem(label, key)
        self.center.currentIndexChanged.connect(lambda *_: self.redraw())
        self.bars = QCheckBox("bars")
        self.bars.setChecked(True)
        self.log_y = QCheckBox("log y")
        self.log_y.setChecked(True)
        for w in (self.bars, self.log_y):
            w.toggled.connect(lambda *_: self.redraw())
        self.boom_bar = LegendBar()
        self.boom_bar.toggled.connect(self.redraw)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.plot = pg.PlotWidget(axisItems={"bottom": Log10Axis(orientation="bottom")})
        item = self.plot.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.15)
        item.setLabel("bottom", "ladder rung [s]")
        self.table = tables.new_table()
        self.table.setToolTip("the plotted value per boom and rung, and the slots behind it")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        for w in (self.quantity, self.corrected, self.variant, self.group):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        row = QHBoxLayout()
        for w in (self.center, self.bars, self.log_y):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.boom_bar)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.plot)
        split.addWidget(self.table)
        split.setSizes([450, 180])
        layout.addWidget(split, stretch=1)
        layout.addWidget(self.info)
        self._fill_groups()

    def sizeHint(self) -> QSize:
        return QSize(600, 600)

    def figure_spec(self) -> tuple[str, dict] | None:
        if not self.summaries:
            return None
        key = self.quantity.currentData()
        lines = [{"label": f"{HEIGHTS[b]:g} m", "color": style.boom_color(b).name(), "scale": s["scale_s"].to_numpy(),
                  "mid": s["mid"].to_numpy(), "lo": s["lo"].to_numpy(), "hi": s["hi"].to_numpy()}
                 for b, s in self.summaries.items()]
        label, unit = self._label(key)
        return "composite", {"title": self.info.text().split(" · ")[0], "ylabel": f"{label} ({unit})",
                             "lines": lines, "refs": (), "bars": self.bars.isChecked(),
                             "log_y": self.log_y.isChecked(), "xlabel": "ladder rung (s)"}

    def _label(self, key: str) -> tuple[str, str]:
        _, label, unit = next(q for q in composite.SCALES if q[0] == key)
        return (f"{label} corrected" if self.corrected.isChecked() else label), unit

    def set_run(self, run, index, booms) -> None:
        self.run, self.index, self.booms = run, index, list(booms)
        self.cfg = run.cfg
        self.matrices, self._loaded_key = {}, None
        colors = [style.boom_color(b) for b in self.booms]
        self.boom_bar.set_members(self.booms, [style.member_short(b) for b in self.booms], colors,
                                  [style.member_label(b) for b in self.booms])

    def set_range(self, x_range, stability) -> None:
        self.x_range, self.stability = x_range, stability
        self._fill_groups()
        if self._loaded_key != self._key():
            self.reload()
        else:
            self.redraw()

    def set_selection(self, slots) -> None:
        self.selection = slots
        self._fill_groups()
        self.redraw()

    def _fill_groups(self) -> None:
        current = self.group.currentData()
        items = [("all slots", "all")]
        if self.stability is not None:
            items += [(n, f"class:{n}") for n in self.stability[1]]
        if self.selection.size:
            items += [("selected slots", "selected"), ("unselected slots", "rest")]
        if [self.group.itemData(i) for i in range(self.group.count())] == [d for _, d in items]:
            return
        self.group.blockSignals(True)
        self.group.clear()
        for label, data in items:
            self.group.addItem(label, data)
        self.group.setCurrentIndex(max(self.group.findData(current), 0))
        self.group.blockSignals(False)

    def _key(self):
        return (self.variant.currentData(), tuple(self.booms))

    def reload(self) -> None:
        if self.run is None:
            return
        key = self._key()
        self.info.setText("reading every boom's integral scales (once per variant; a few seconds)…")
        self.runner.submit(_load, self.run, self.index, self.booms, key[0], key=f"scales-{id(self)}",
                           label="reading integral scales", on_done=lambda m: self._loaded(key, m),
                           on_error=lambda exc, tb: self.info.setText(f"couldn't read the scales: {exc}"))

    def _loaded(self, key, matrices) -> None:
        if not shiboken6.isValid(self) or key != self._key():
            return
        self.matrices, self._loaded_key = matrices, key
        self.redraw()

    def _keep(self, slots: np.ndarray) -> np.ndarray:
        keep = np.ones(slots.size, dtype=bool)
        if self.x_range is not None:
            t = slot_to_unix(slots)
            keep &= (t >= self.x_range[0]) & (t < self.x_range[1])
        kind = self.group.currentData() or "all"
        if kind.startswith("class:"):
            keep &= in_class(self.stability, slots, kind[len("class:"):])
        elif kind == "selected":
            keep &= np.isin(slots, self.selection)
        elif kind == "rest":
            keep &= ~np.isin(slots, self.selection)
        return keep

    def redraw(self) -> None:
        item = self.plot.getPlotItem()
        item.clear()
        self.summaries = {}
        log_y = self.log_y.isChecked()
        item.setLogMode(y=log_y)
        if not self.matrices or self._loaded_key != self._key():
            return
        key, center = self.quantity.currentData(), self.center.currentData()
        label, unit = self._label(key)
        factor = EFOLDING_TOTAL_FACTOR if self.corrected.isChecked() else 1.0
        shown = self.boom_bar.visible()
        counts, cells = [], {}
        for b in self.booms:
            if b not in shown or b not in self.matrices:
                continue
            slots, rungs, values = self.matrices[b]
            if slots.size == 0 or key not in values:
                continue
            s = composite.summarize(slots, rungs, values[key] * factor, self._keep(slots), center=center)
            s = s[np.isfinite(s["mid"])]
            if s.empty:
                continue
            self.summaries[b] = s
            counts.append(int(s["n"].max()))
            color = style.boom_color(b)
            x = np.log10(s["scale_s"].to_numpy())
            mid, lo, hi = (s[c].to_numpy() for c in ("mid", "lo", "hi"))
            item.plot(x, mid, pen=pg.mkPen(color, width=1.6), symbol="o", symbolSize=5, symbolBrush=color,
                      symbolPen=None)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log10(np.where(mid > 0, mid, np.nan)) if log_y else mid
                if self.bars.isChecked():
                    if log_y:  # error bars don't follow log mode; a bar running below 0 is cut off at the axis
                        top = np.log10(np.where(hi > 0, hi, np.nan)) - y
                        bottom = y - np.log10(np.where(lo > 0, lo, np.nan))
                        bottom = np.where(np.isfinite(bottom), bottom, 2.0)
                    else:
                        top, bottom = hi - mid, mid - lo
                    item.addItem(pg.ErrorBarItem(x=x, y=y, top=top, bottom=bottom, beam=0.04,
                                                 pen=pg.mkPen(color, width=0.8)))
            cells[f"b{b} ({HEIGHTS[b]:g} m)"] = {f"{r:g} s": v for r, v in zip(s["scale_s"], mid)} | {
                "N": int(s["n"].max())}
        guide = key.startswith("its") and self.cfg is not None and not self.corrected.isChecked()
        if guide and self.summaries:
            ratio = self.cfg.secondary.min_tau_its_ratio
            rungs = np.log10(np.asarray(composite_rungs(self.summaries)))
            item.plot(rungs, 10 ** rungs / ratio, pen=pg.mkPen("#999999", style=Qt.PenStyle.DashLine))
        item.setLabel("left", f"{label} [{unit}]")
        frame = pd.DataFrame.from_dict(cells, orient="index")
        if not frame.empty:
            frame = frame[["N"] + [c for c in frame.columns if c != "N"]]
        tables.fill(self.table, frame, index=True,
                    fmt=lambda col, v: None if col == "N" else ("—" if not np.isfinite(v) else f"{v:.3g}"))
        scope = "whole period" if self.x_range is None else "visible range"
        n = (f"{min(counts)}–{max(counts)}" if min(counts) != max(counts) else f"{counts[0]}") if counts else "0"
        what = "median, bars 25–75 %" if center == "median" else "mean, bars ± 1 SD"
        guide_note = (f" · dashed: ITS = rung/{self.cfg.secondary.min_tau_its_ratio:g} (an ITS above it counts as "
                      "short at that τ)") if guide else ""
        cut = " (bars below 0 are cut off on the log axis)" if log_y and center == "mean" else ""
        self.info.setText(f"{self.group.currentText()}, {scope}: N = {n} slots per boom · {what}{cut}{guide_note}")


def composite_rungs(summaries: dict) -> np.ndarray:
    """Every rung any drawn boom has."""
    return np.unique(np.concatenate([s["scale_s"].to_numpy() for s in summaries.values()]))
