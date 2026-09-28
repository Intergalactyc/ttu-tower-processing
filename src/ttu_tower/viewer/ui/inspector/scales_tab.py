"""Integral time and length scales at every rung of the ladder, with the
slot's tau choices marked, and on demand the pooled autocorrelations each ITS
integrates (recomputed from the raw files).
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.primary.rawrung import RUNG_SECONDS
from ttu_tower.viewer.slotdata import rung_scales, stored_selection
from ttu_tower.viewer.ui.axes import Log10Axis

_VARS = (("u", "#1f4e9c"), ("v", "#b3123f"), ("w", "#2a8a4a"), ("vpts", "#8a5a00"))
_TAU_COLORS = {"selected": "#000000", "heat": "#d62728", "momentum": "#2a8a4a"}


class ScalesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bundle = self.cfg = self.reprocessor = self.runner = None
        self.variant = "mrd"
        self.focus_vars: tuple[str, ...] = ()
        self.acf_trace = None

        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.table = QTableWidget()
        self.table.verticalHeader().setDefaultSectionSize(18)
        self.its_plot = pg.PlotWidget(axisItems={"bottom": Log10Axis(orientation="bottom"),
                                                 "left": Log10Axis(orientation="left")})
        self.ils_plot = pg.PlotWidget(axisItems={"bottom": Log10Axis(orientation="bottom"),
                                                 "left": Log10Axis(orientation="left")})
        for plot, label, unit in ((self.its_plot, "ITS", "s"), (self.ils_plot, "ILS", "m")):
            plot.setLabel("left", f"{label} [{unit}]")
            plot.setLabel("bottom", "rung (averaging time) [s]")
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.addLegend(offset=(5, 5))

        self.rung = QComboBox()
        for r in RUNG_SECONDS:
            self.rung.addItem(f"{r:g} s", r)
        self.rung.currentIndexChanged.connect(lambda *_: self._draw_acf())
        self.acf_button = QPushButton("Show ACFs (reprocess)")
        self.acf_button.clicked.connect(self.load_acf)
        self.acf_status = QLabel()
        self.acf_plot = pg.PlotWidget()
        self.acf_plot.setLabel("left", "autocorrelation")
        self.acf_plot.setLabel("bottom", "lag [s]")
        self.acf_plot.showGrid(x=True, y=True, alpha=0.15)
        self.acf_plot.addLegend(offset=(5, 5))

        plots = QSplitter()
        plots.addWidget(self.its_plot)
        plots.addWidget(self.ils_plot)
        acf_box = QWidget()
        acf_layout = QVBoxLayout(acf_box)
        row = QHBoxLayout()
        row.addWidget(self.acf_button)
        row.addWidget(QLabel("rung"))
        row.addWidget(self.rung)
        row.addWidget(self.acf_status, stretch=1)
        acf_layout.addLayout(row)
        acf_layout.addWidget(self.acf_plot, stretch=1)
        plots.addWidget(acf_box)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(plots)
        split.addWidget(self.table)
        split.setSizes([500, 220])
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(split, stretch=1)

    def set_bundle(self, bundle, variant, cfg, reprocessor, runner, focus_vars=(), auto_acf: bool = False) -> None:
        changed = self.bundle is None or (bundle.slot, bundle.boom) != (self.bundle.slot, self.bundle.boom) \
            or variant != self.variant
        self.bundle, self.variant, self.cfg = bundle, variant, cfg
        self.reprocessor, self.runner, self.focus_vars = reprocessor, runner, tuple(focus_vars)
        if changed:
            self.acf_trace = None
            self.acf_plot.clear()
            self.acf_status.setText("")
        self._select_rung()
        self.redraw()
        if auto_acf and self.acf_trace is None:
            self.load_acf()

    def _taus(self) -> dict[str, float]:
        taus = {}
        sel = stored_selection(self.bundle, self.variant)
        if sel is not None and np.isfinite(sel["tau_s"]):
            taus["selected"] = sel["tau_s"]
        tau = self.bundle.table("tau")
        spec_variant = "mrd" if self.variant == "naive" else self.variant
        for row in tau[tau["variant"] == spec_variant].itertuples():
            if row.status in ("found", "capped") and np.isfinite(row.tau_s):
                taus[row.cospectrum] = row.tau_s
        return taus

    def _select_rung(self) -> None:
        tau = self._taus().get("selected")
        if tau is not None:
            i = self.rung.findData(tau)
            if i >= 0:
                self.rung.blockSignals(True)
                self.rung.setCurrentIndex(i)
                self.rung.blockSignals(False)

    def redraw(self) -> None:
        for plot in (self.its_plot, self.ils_plot):
            plot.clear()
        if self.bundle is None:
            return
        scales = rung_scales(self.bundle, self.variant)
        self._fill_table(scales)
        if scales.empty:
            self.info.setText("no ladder rows for this slot")
            return
        taus = self._taus()
        x = np.log10(scales.index.to_numpy(float))
        for var, color in _VARS:
            for plot, prefix in ((self.its_plot, "its"), (self.ils_plot, "ils")):
                col = f"{prefix}_{var}"
                if col not in scales.columns:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = np.log10(scales[col].to_numpy(float))
                width = 2.4 if var in self.focus_vars else 1.2
                plot.plot(x, y, pen=pg.mkPen(color, width=width), symbol="o", symbolSize=5, symbolBrush=color,
                          symbolPen=None, name=var)
        ratio = self.cfg.secondary.min_tau_its_ratio
        self.its_plot.plot(x, x - np.log10(ratio), pen=pg.mkPen("#999999", style=Qt.PenStyle.DashLine),
                           name=f"ITS = τ/{ratio:g}")
        for name, tau in taus.items():
            for plot in (self.its_plot, self.ils_plot):
                plot.addItem(pg.InfiniteLine(np.log10(tau), label=f"{name} τ", labelOpts={"position": 0.95},
                                             pen=pg.mkPen(_TAU_COLORS[name], width=2 if name == "selected" else 1.2)))
        parts = [f"{name} τ {tau:g} s" for name, tau in taus.items()]
        self.info.setText(f"variant {self.variant} · " + (" · ".join(parts) if parts else "no τ") +
                          f" · the dashed line is ITS = τ/{ratio:g}: an ITS above it at the selected τ counts as short")

    def _fill_table(self, scales) -> None:
        self.table.clear()
        self.table.setRowCount(len(scales))
        self.table.setColumnCount(len(scales.columns))
        self.table.setHorizontalHeaderLabels(list(scales.columns))
        self.table.setVerticalHeaderLabels([f"{r:g} s" for r in scales.index])
        for i, (_, row) in enumerate(scales.iterrows()):
            for j, v in enumerate(row):
                text = v if isinstance(v, str) else ("" if v is None or np.isnan(v) else f"{v:.4g}")
                self.table.setItem(i, j, QTableWidgetItem(text))

    # --- ACFs ------------------------------------------------------------------------------------

    def load_acf(self) -> None:
        if self.reprocessor is None or self.bundle is None:
            self.acf_status.setText("reprocessing is disabled for this run")
            return
        self.acf_status.setText("reprocessing…")
        variant = "mrd" if self.variant == "naive" else self.variant
        self.runner.submit(self.reprocessor.acf, self.bundle.boom, self.bundle.slot, variant, key=f"acf-{id(self)}",
                           label="computing autocorrelations", on_done=self._acf_loaded,
                           on_error=lambda exc, tb: self.acf_status.setText(f"failed: {exc}"))

    def _acf_loaded(self, result) -> None:
        values, trace = result
        self.acf_trace = (values, trace)
        self.acf_status.setText("")
        self._draw_acf()

    def _draw_acf(self) -> None:
        self.acf_plot.clear()
        if self.acf_trace is None:
            return
        values, trace = self.acf_trace
        rung = self.rung.currentData()
        by_var = trace.get("acf", {}).get(rung, {})
        its = {var: v for r, var, stat, v in values if r == rung and stat == "its"}
        self.acf_plot.addItem(pg.InfiniteLine(1 / np.e, angle=0, pen=pg.mkPen("#999999", style=Qt.PenStyle.DashLine),
                                              label="1/e", labelOpts={"position": 0.95}))
        self.acf_plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#cccccc")))
        missing = []
        for var, color in _VARS:
            rho = by_var.get(var)
            if rho is None:
                missing.append(var)
                continue
            lag = np.arange(rho.size) / SAMPLE_HZ
            width = 2.4 if var in self.focus_vars else 1.2
            self.acf_plot.plot(lag, rho, pen=pg.mkPen(color, width=width), name=f"{var}: ITS {its.get(var, np.nan):.3g} s")
            if np.isfinite(its.get(var, np.nan)):
                self.acf_plot.addItem(pg.InfiniteLine(its[var], pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine)))
        note = f"rung {rung:g} s; pooled over the rung's fully usable blocks"
        if missing:
            note += f"; no fully usable block for {', '.join(missing)}"
        self.acf_status.setText(note)
