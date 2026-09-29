"""Composite MRD spectra over the viewed interval (or the whole period): per
boom, the median over slots at each averaging scale with interquartile bars,
for every slot, one stability class, or the brushed selection.
"""
import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import composite
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import in_class
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.axes import Log10Axis
from ttu_tower.viewer.ui.timeline_plot import LegendBar

_VARIANTS = (("mrd", "mrd"), ("mrd_unexcised", "mrd unexcised"))


def _load(run, index, booms, variant, spectrum) -> dict:
    """Runs on a worker thread: boom -> (slots, scales, values)."""
    return {b: composite.spectra_matrix(run, index, b, variant, spectrum) for b in booms}


class CompositeSpectraView(QWidget):
    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = None
        self.booms: list[int] = []
        self.x_range = None
        self.stability = None
        self.selection = np.empty(0, dtype=np.int64)
        self.matrices: dict = {}
        self.summaries: dict = {}  # boom -> DataFrame, as drawn
        self._loaded_key = None

        self.spectrum = QComboBox()
        for key, label in composite.SPECTRA:
            self.spectrum.addItem(label, key)
        self.spectrum.setCurrentIndex(self.spectrum.findData("uw"))
        self.variant = QComboBox()
        for key, label in _VARIANTS:
            self.variant.addItem(label, key)
        for w in (self.spectrum, self.variant):
            w.currentIndexChanged.connect(lambda *_: self.reload())
        self.group = QComboBox()
        self.group.setToolTip("which slots: all, one stability class, or the brushed selection (or the rest)")
        self.group.currentIndexChanged.connect(lambda *_: self.redraw())
        self.normalize = QCheckBox("normalize")
        self.normalize.setToolTip("divide each slot's spectrum by its sum over scales (the within-block "
                                  "(co)variance), so slots weigh alike whatever their intensity")
        self.bars = QCheckBox("IQR bars")
        self.bars.setChecked(True)
        self.log_y = QCheckBox("log y")
        for w in (self.normalize, self.bars, self.log_y):
            w.toggled.connect(lambda *_: self.redraw())
        self.boom_bar = LegendBar()
        self.boom_bar.toggled.connect(self.redraw)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.plot = pg.PlotWidget(axisItems={"bottom": Log10Axis(orientation="bottom")})
        item = self.plot.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.15)
        item.setLabel("bottom", "averaging scale τ [s]")
        for side in ("left", "bottom"):
            item.getAxis(side).enableAutoSIPrefix(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        for w in (self.spectrum, self.variant, self.group):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        row = QHBoxLayout()
        for w in (self.normalize, self.bars, self.log_y):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.boom_bar)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.info)
        self._fill_groups()

    def sizeHint(self) -> QSize:
        return QSize(600, 600)

    def figure_spec(self) -> tuple[str, dict] | None:
        if not self.summaries:
            return None
        spectrum = self.spectrum.currentData()
        lines = [{"label": f"{HEIGHTS[b]:g} m", "color": style.boom_color(b).name(), "scale": s["scale_s"].to_numpy(),
                  "mid": s["mid"].to_numpy(), "lo": s["lo"].to_numpy(), "hi": s["hi"].to_numpy()}
                 for b, s in self.summaries.items()]
        label = dict(composite.SPECTRA)[spectrum] + (" (normalized)" if self.normalize.isChecked() else "")
        return "composite", {"title": self.info.text().split(" · dashed")[0], "ylabel": label, "lines": lines,
                             "refs": composite.REFERENCE_SCALES, "bars": self.bars.isChecked(),
                             "log_y": spectrum in composite.VARIANCES and self.log_y.isChecked()}

    def set_run(self, run, index, booms) -> None:
        self.run, self.index, self.booms = run, index, list(booms)
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
        items = [("all slots", "all")]  # item data: "all", "selected", "rest" or "class:<name>"
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
        return (self.variant.currentData(), self.spectrum.currentData(), tuple(self.booms))

    def reload(self) -> None:
        if self.run is None:
            return
        key = self._key()
        self.info.setText("reading every boom's spectra (once per spectrum; a few seconds)…")
        self.runner.submit(_load, self.run, self.index, self.booms, key[0], key[1], key=f"composite-{id(self)}",
                           label="reading spectra", on_done=lambda m: self._loaded(key, m),
                           on_error=lambda exc, tb: self.info.setText(f"couldn't read the spectra: {exc}"))

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
        self.plot.clear()
        self.summaries = {}
        spectrum = self.spectrum.currentData()
        variance = spectrum in composite.VARIANCES
        self.log_y.setEnabled(variance)
        self.log_y.setToolTip("" if variance else "cospectra change sign")
        log_y = variance and self.log_y.isChecked()
        self.plot.getPlotItem().setLogMode(y=log_y)
        if not self.matrices or self._loaded_key != self._key():
            return
        shown = self.boom_bar.visible()
        counts = []
        for b in self.booms:
            if b not in shown or b not in self.matrices:
                continue
            slots, scales, values = self.matrices[b]
            if slots.size == 0:
                continue
            s = composite.summarize(slots, scales, values, self._keep(slots), self.normalize.isChecked())
            s = s[np.isfinite(s["mid"])]
            if s.empty:
                continue
            self.summaries[b] = s
            counts.append(int(s["n"].max()))
            color = style.boom_color(b)
            x = np.log10(s["scale_s"].to_numpy())
            mid = s["mid"].to_numpy()
            self.plot.plot(x, mid, pen=pg.mkPen(color, width=1.6), symbol="o", symbolSize=5, symbolBrush=color,
                           symbolPen=None)
            if self.bars.isChecked():
                lo, hi = s["lo"].to_numpy(), s["hi"].to_numpy()
                if log_y:  # error bars don't follow log mode
                    with np.errstate(divide="ignore", invalid="ignore"):
                        bars = pg.ErrorBarItem(x=x, y=np.log10(mid), top=np.log10(hi) - np.log10(mid),
                                               bottom=np.log10(mid) - np.log10(np.where(lo > 0, lo, np.nan)),
                                               beam=0.04, pen=pg.mkPen(color, width=0.8))
                else:
                    bars = pg.ErrorBarItem(x=x, y=mid, top=hi - mid, bottom=mid - lo, beam=0.04,
                                           pen=pg.mkPen(color, width=0.8))
                self.plot.addItem(bars)
        for v in composite.REFERENCE_SCALES:
            self.plot.addItem(pg.InfiniteLine(np.log10(v), angle=90, pen=pg.mkPen("#555555", width=1,
                                                                                 style=Qt.PenStyle.DashLine)))
        if not variance:
            self.plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#bbbbbb", width=1)))
        label = dict(composite.SPECTRA)[spectrum]
        self.plot.getPlotItem().setLabel("left", f"{label}" + (" (normalized)" if self.normalize.isChecked() else ""))
        scope = "whole period" if self.x_range is None else "visible range"
        n = (f"{min(counts)}–{max(counts)}" if min(counts) != max(counts) else f"{counts[0]}") if counts else "0"
        self.info.setText(f"{self.group.currentText()}, {scope}: N = {n} slots per boom · median, bars 25–75 % · "
                          f"dashed: {', '.join(f'{v:g}' for v in composite.REFERENCE_SCALES)} s")
