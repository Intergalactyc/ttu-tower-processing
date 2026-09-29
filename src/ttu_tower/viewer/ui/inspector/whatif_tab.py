"""tau what-if: edit the run's detection and selection parameters and see,
for every boom of the slot, which taus would change - and for this boom, the
detection redrawn on its heat and momentum cospectra.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from ttu_tower.secondary.detect import detect
from ttu_tower.viewer import whatif
from ttu_tower.viewer.slotdata import DETECTED, SPECTRUM_LABELS, rerun_detection, spectrum, spectrum_variant
from ttu_tower.viewer.ui import tables
from ttu_tower.viewer.ui.axes import Log10Axis

_STORED = "#000000"
_UNUSED = "#8c8c8c"  # the smoothing detection would use, where it didn't run
_WHATIF = "#c2185b"
_PRIORITIES = (("heat", "momentum"), ("momentum", "heat"))
_COLUMNS = {"boom": "boom", "variant": "variant", "heat_stored": "heat (run)", "heat_new": "heat (what-if)",
            "momentum_stored": "momentum (run)", "momentum_new": "momentum (what-if)",
            "selected_tau_stored": "τ (run)", "selected_tau_new": "τ (what-if)", "source_stored": "source (run)",
            "source_new": "source (what-if)", "ustar_stored": "u* (run)", "ustar_new": "u* (what-if)",
            "sigma_w_stored": "σw (run)", "sigma_w_new": "σw (what-if)"}


def _rung_box() -> QComboBox:
    box = QComboBox()
    for rung in whatif.LADDER_RUNGS:
        box.addItem(f"{rung:g} s", rung)
    return box


class _MiniBundle:
    """The slice of a slot bundle `rerun_detection` reads, for any boom of a SlotAcross."""

    def __init__(self, across, boom):
        self.slot, self.boom, self.mrd = across.slot, boom, across.mrd
        self._across = across

    def table(self, name):
        df = self._across.table(name)
        return df[df["boom"] == self.boom] if "boom" in df.columns else df


class WhatIfTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.across = self.cfg = None
        self.variant, self.boom = "mrd", None
        self.result = None  # the last comparison

        self.fields = {
            "peak_significance_se": QDoubleSpinBox(), "min_scale_s": QDoubleSpinBox(),
            "min_tau_s": _rung_box(), "max_tau_s": _rung_box(), "rule": QComboBox(), "priority": QComboBox(),
            "fallback_tau_s": _rung_box(),
        }
        f = self.fields
        f["peak_significance_se"].setRange(0.0, 50.0)
        f["peak_significance_se"].setSingleStep(0.25)
        f["peak_significance_se"].setToolTip("a peak (or a reversal that would clip tau) must stand this many SE clear")
        f["min_scale_s"].setRange(0.05, 9.0)
        f["min_scale_s"].setSingleStep(0.1)
        f["min_scale_s"].setDecimals(3)
        for rule in ("priority", "longest_significant"):
            f["rule"].addItem(rule, rule)
        for p in _PRIORITIES:
            f["priority"].addItem(", ".join(p), p)
        for w in f.values():
            signal = w.valueChanged if isinstance(w, QDoubleSpinBox) else w.currentIndexChanged
            signal.connect(lambda *_: self.recompute())

        form = QFormLayout()
        form.addRow(QLabel("<b>detection</b>"))
        form.addRow("peak significance [SE]", f["peak_significance_se"])
        form.addRow("min scale", f["min_scale_s"])
        form.addRow("min τ", f["min_tau_s"])
        form.addRow("max τ", f["max_tau_s"])
        form.addRow(QLabel("<b>selection</b>"))
        form.addRow("rule", f["rule"])
        form.addRow("priority", f["priority"])
        form.addRow("fallback τ", f["fallback_tau_s"])
        reset = QPushButton("Reset to the run's config")
        reset.clicked.connect(self.reset)
        self.problem = QLabel()
        self.problem.setStyleSheet("color:#b00")
        self.problem.setWordWrap(True)
        form_box = QGroupBox("What if")
        form_layout = QVBoxLayout(form_box)
        form_layout.addLayout(form)
        form_layout.addWidget(reset)
        form_layout.addWidget(self.problem)
        form_layout.addStretch(1)
        form_box.setMaximumWidth(320)

        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        self.changed_only = QCheckBox("changed rows only")
        self.changed_only.toggled.connect(lambda *_: self._show_table())
        self.table = tables.new_table()
        self.graphics = pg.GraphicsLayoutWidget()
        top = QHBoxLayout()
        top.addWidget(self.summary, stretch=1)
        top.addWidget(self.changed_only)
        right = QSplitter(Qt.Orientation.Vertical)
        table_box = QWidget()
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addLayout(top)
        table_layout.addWidget(self.table, stretch=1)
        right.addWidget(table_box)
        plots_box = QWidget()
        plots_layout = QVBoxLayout(plots_box)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        key = QLabel("blue: the stored (co)spectrum ± SE · <span style='color:#c2185b'>dashed: the 1-2-1 smoothed "
                     "cospectrum over the searched range · band: ± k·SE of it (a peak counts where the band excludes "
                     "0) · ▲ peak · ✕ reversal</span> · dotted grey: min/max τ · τ lines: solid = run, dashed = what-if")
        key.setTextFormat(Qt.TextFormat.RichText)
        key.setWordWrap(True)
        plots_layout.addWidget(key)
        plots_layout.addWidget(self.graphics, stretch=1)
        right.addWidget(plots_box)
        layout = QHBoxLayout(self)
        layout.addWidget(form_box)
        layout.addWidget(right, stretch=1)
        self.plots: dict[str, pg.PlotItem] = {}

    # --- parameters -------------------------------------------------------------------------------

    def set_config(self, cfg) -> None:
        first = self.cfg is None
        self.cfg = cfg
        if first:
            self.reset()

    def reset(self) -> None:
        if self.cfg is None:
            return
        det, sel = self.cfg.secondary.detection, self.cfg.secondary.selection
        f = self.fields
        for w in f.values():
            w.blockSignals(True)
        f["peak_significance_se"].setValue(det.peak_significance_se)
        f["min_scale_s"].setValue(det.min_scale_s)
        for key, value in (("min_tau_s", det.min_tau_s), ("max_tau_s", det.max_tau_s),
                           ("fallback_tau_s", sel.fallback_tau_s), ("rule", sel.rule), ("priority", tuple(sel.priority))):
            f[key].setCurrentIndex(max(f[key].findData(value), 0))
        for w in f.values():
            w.blockSignals(False)
        self.recompute()

    def edited_config(self):
        f = self.fields
        return whatif.edited(
            self.cfg.secondary,
            detection={"peak_significance_se": f["peak_significance_se"].value(), "min_scale_s": f["min_scale_s"].value(),
                       "min_tau_s": f["min_tau_s"].currentData(), "max_tau_s": f["max_tau_s"].currentData()},
            selection={"rule": f["rule"].currentData(), "priority": tuple(f["priority"].currentData()),
                       "fallback_tau_s": f["fallback_tau_s"].currentData()},
        )

    def is_default(self) -> bool:
        return self.cfg is not None and self.edited_config() == self.cfg.secondary

    # --- results ----------------------------------------------------------------------------------

    def set_across(self, across, variant: str, boom: int) -> None:
        self.across, self.variant, self.boom = across, variant, boom
        self.recompute()

    def recompute(self) -> None:
        if self.cfg is None or self.across is None:
            return
        cfg = self.edited_config()
        issues = whatif.problems(cfg)
        self.problem.setText("; ".join(issues))
        if issues:
            return
        self.result = whatif.compare(self.across, cfg)
        self._show_table()
        self._draw(cfg)

    def _show_table(self) -> None:
        r = self.result
        if r is None or r.empty:
            self.table.clear()
            self.summary.setText("no spectra stored for this slot")
            return
        n = int(r["changed"].sum())
        default = " (the run's own parameters)" if self.is_default() else ""
        self.summary.setText(f"<b>{n}</b> of {len(r)} (boom, variant) selections change{default}")
        shown = r[r["changed"]] if self.changed_only.isChecked() else r
        changed = shown["changed"].to_numpy()
        shown = shown[[c for c in _COLUMNS if c in shown.columns]].rename(columns=_COLUMNS)
        tables.fill(self.table, shown.reset_index(drop=True))
        for i, c in enumerate(changed):
            if c:
                for j in range(self.table.columnCount()):
                    self.table.item(i, j).setBackground(tables.WARN)

    def _draw(self, cfg) -> None:
        self.graphics.clear()
        self.plots = {}
        if self.boom is None or self.across.mrd is None:
            return
        variant = "mrd" if self.variant == "naive" else self.variant
        mini = _MiniBundle(self.across, self.boom)
        views, new_sel = rerun_detection(mini, variant, cfg)
        stored_sel = mini.table("tau_selected")
        stored_sel = stored_sel[stored_sel["variant"] == variant]
        selected = [(new_sel[0], _WHATIF, "what-if", Qt.PenStyle.DashLine)]
        if not stored_sel.empty:
            selected.insert(0, (float(stored_sel["tau_s"].iloc[0]), _STORED, "run", Qt.PenStyle.SolidLine))
        mrd_variant = spectrum_variant(mini, variant)
        for i, (family, name) in enumerate(DETECTED.items()):
            plot = self.graphics.addPlot(row=0, col=i, axisItems={"bottom": Log10Axis(orientation="bottom")})
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setTitle(f"b{self.boom} {family}: {SPECTRUM_LABELS[name]} ({variant})", size="9pt")
            plot.setLabel("bottom", "scale [s]")
            plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#999999")))
            self.plots[family] = plot
            spec = spectrum(self.across.mrd, self.across.slot, self.boom, mrd_variant, name)
            if spec is None or not np.isfinite(spec.value).any():
                note = pg.TextItem(f"no {SPECTRUM_LABELS[name]} values in this slot's detection window "
                                   f"({family}: no usable samples), so there is nothing to detect on",
                                   color=_UNUSED, anchor=(0.5, 0.5))
                plot.addItem(note)
                plot.setRange(xRange=(0, 3), yRange=(-1, 1))
                note.setPos(1.5, 0.3)
                continue
            x = np.log10(spec.scale_s)
            ok = np.isfinite(spec.value) & np.isfinite(spec.se)
            if ok.any():
                plot.addItem(pg.ErrorBarItem(x=x[ok], y=spec.value[ok], top=spec.se[ok], bottom=spec.se[ok],
                                             beam=0.03, pen=pg.mkPen((31, 78, 156, 140))))
            if np.isfinite(spec.value).any():
                plot.plot(x, spec.value, pen=pg.mkPen("#1f4e9c", width=1.4), symbol="o", symbolSize=4,
                          symbolBrush="#1f4e9c", symbolPen=None)
            for bound in (cfg.detection.min_tau_s, cfg.detection.max_tau_s):
                plot.addItem(pg.InfiniteLine(np.log10(bound), pen=pg.mkPen("#aaaaaa", width=1, style=Qt.PenStyle.DotLine)))
            view = views[family]
            det, stored = view.detection, view.stored
            if det.status == "no_data" and np.isfinite(spec.value).any():
                # the slot has no usable samples of this family, so detection stops before smoothing; show
                # what it would have seen, greyed, since it plays no part in tau
                trace: dict = {}
                detect(spec.value, spec.se, spec.n_pairs, True, cfg.detection, trace=trace)
                self._draw_trace(plot, trace, x, color=_UNUSED)
                note = pg.TextItem(f"{family}: no_data (no usable {family} samples in the slot) - detection didn't "
                                   "run; grey: what it would see", color=_UNUSED, anchor=(0, 0))
                note.setPos(x[0], np.nanmax(spec.value + np.nan_to_num(spec.se)))
                plot.addItem(note)
            else:
                self._draw_trace(plot, view.trace, x)
            family_taus = [(det.status, det.tau_s, det.tau_lb_s, _WHATIF, "what-if", Qt.PenStyle.DashLine, 0.8)]
            if stored is not None:
                family_taus.insert(0, (stored["status"], stored["tau_s"], stored["tau_lb_s"], _STORED, "run",
                                       Qt.PenStyle.SolidLine, 0.92))
            for status, tau_s, tau_lb_s, color, label, dash, pos in family_taus:
                tau = tau_s if status in ("found", "capped") else tau_lb_s
                if np.isfinite(tau):
                    plot.addItem(pg.InfiniteLine(np.log10(tau), label=f"{label} {family} τ",
                                                 pen=pg.mkPen(color, width=1.2, style=dash),
                                                 labelOpts={"position": pos, "color": color}))
            for tau, color, label, dash in selected:
                if np.isfinite(tau):
                    plot.addItem(pg.InfiniteLine(np.log10(tau), label=f"{label} selected",
                                                 pen=pg.mkPen(color, width=2.5, style=dash),
                                                 labelOpts={"position": 0.12 if label == "run" else 0.04, "color": color}))

    @staticmethod
    def _draw_trace(plot, t, x, color: str = _WHATIF) -> None:
        """The smoothed cospectrum over the searched range, its ±k·SE significance
        band, the peak (▲) and the reversal (✕).
        """
        if not t.get("d_smooth"):
            return
        fill = pg.mkColor(color)
        fill.setAlpha(35)
        modes = sorted(t["d_smooth"])
        xs = np.array([x[i - 1] for i in modes])
        ds = np.array([t["d_smooth"][i] for i in modes])
        half = t["k_se"] * np.array([t["se_smooth"][i] for i in modes])
        upper = pg.PlotCurveItem(xs, ds + half, pen=pg.mkPen(None))
        lower = pg.PlotCurveItem(xs, ds - half, pen=pg.mkPen(None))
        plot.addItem(upper)
        plot.addItem(lower)
        plot.addItem(pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(fill)))
        plot.plot(xs, ds, pen=pg.mkPen(color, width=1.2, style=Qt.PenStyle.DashLine))
        if t.get("peak") is not None:
            p = t["peak"]
            plot.addItem(pg.ScatterPlotItem([x[p - 1]], [t["d_smooth"][p]], symbol="t1", size=12,
                                            brush=pg.mkBrush(color), pen=pg.mkPen("k")))
        if t.get("reversal"):
            r, _ = t["reversal"]
            plot.addItem(pg.ScatterPlotItem([x[r - 1]], [t["d_smooth"][r]], symbol="x", size=13,
                                            brush=pg.mkBrush(color), pen=pg.mkPen(color, width=2)))
