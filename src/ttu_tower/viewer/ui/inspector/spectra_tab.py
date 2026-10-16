"""The slot's MRD variance spectra and cospectra, with how tau was found:
the detection rerun on the stored spectra (smoothed curve, significance band,
searched range, peak and reversal) and the run's stored tau lines; beside
them, the momentum and heat transport efficiencies at each ladder rung.
"""
import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ttu_tower.constants import BOOMS
from ttu_tower.viewer.composite import TE_TIP
from ttu_tower.viewer.slotdata import (
    COSPECTRA, DETECTED, NEIGHBOURS, NORMALIZE_TIP, SPECTRA, SPECTRUM_LABELS, SPECTRUM_UNITS, normalized,
    normalizing_factor, rerun_detection, spectrum, spectrum_variant, stored_selection, transport_efficiency,
)
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.axes import Log10Axis

_MAIN = "#1f4e9c"
_UNEXCISED = "#b3123f"
_TAU_PENS = {"heat": ("#d62728", "heat τ"), "momentum": ("#2a8a4a", "momentum τ")}
GROUPS = (  # key, check box label, the plots it shows
    ("heat_momentum", "heat & momentum", ("wvpts", "uw")),
    ("other_covariances", "other covariances", ("vw", "uv")),
    ("variances", "variances", ("uu", "vv", "ww", "tsts")),
    ("transport", "transport efficiencies", ("te_uw", "te_wvpts")),
)


def plot_rows(shown: set[str]) -> list[list[str]]:
    """The checked groups' plots, by row: covariances above variances with the
    transport efficiencies beside them; otherwise a row per group, or one row
    for four plots or fewer.
    """
    plots = {key: list(names) for key, _, names in GROUPS if key in shown}
    covs = plots.get("heat_momentum", []) + plots.get("other_covariances", [])
    variances, te = plots.get("variances", []), plots.get("transport", [])
    if covs and variances:
        return [covs + te[:1], variances + te[1:]]
    rows = [r for r in (covs, variances, te) if r]
    return [sum(rows, [])] if rows and sum(map(len, rows)) <= 4 else rows


def _fmt_s(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:g} s"


class SpectraTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bundle = self.cfg = None
        self.variant = "mrd"
        self.emphasis: str | None = None
        self.mode = QComboBox()
        self.mode.addItem("value per mode", "value")
        self.mode.addItem("cumulative (ogive)", "ogive")
        self.mode.setToolTip("cumulative: the sum of modes up to each scale = the mean within-block (co)variance")
        self.boxes = {}
        for key, label, on in (("se", "± SE", True), ("detection", "detection details", True),
                               ("log", "log y (variances)", True), ("unexcised", "unexcised", False),
                               ("booms", "all booms", False), ("neighbours", f"slots ±{NEIGHBOURS}", False)):
            box = QCheckBox(label)
            box.setChecked(on)
            box.toggled.connect(lambda *_: self.redraw())
            self.boxes[key] = box
        self.groups = {}
        for key, label, names in GROUPS:
            box = QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(TE_TIP if key == "transport" else
                           ", ".join(SPECTRUM_LABELS[n] for n in names) + " (co)spectra")
            box.toggled.connect(lambda *_: self.redraw())
            self.groups[key] = box
        self.normalize = {}
        for name in SPECTRA:
            box = QCheckBox(SPECTRUM_LABELS[name])
            box.setToolTip(NORMALIZE_TIP)
            box.toggled.connect(lambda *_: self.redraw())
            self.normalize[name] = box
        self.mode.currentIndexChanged.connect(lambda *_: self.redraw())
        controls = QHBoxLayout()
        controls.addWidget(self.mode)
        for box in self.boxes.values():
            controls.addWidget(box)
        controls.addStretch(1)
        shown = QHBoxLayout()
        shown.addWidget(QLabel("plots:"))
        for box in self.groups.values():
            shown.addWidget(box)
        shown.addSpacing(16)
        shown.addWidget(QLabel("normalize:"))
        for box in self.normalize.values():
            shown.addWidget(box)
        shown.addStretch(1)
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.graphics = pg.GraphicsLayoutWidget()
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(shown)
        layout.addWidget(self.info)
        layout.addWidget(self.graphics, stretch=1)
        self.plots: dict[str, pg.PlotItem] = {}
        self._columns = 0  # of the grid last drawn

    def shown_groups(self) -> set[str]:
        return {key for key, box in self.groups.items() if box.isChecked()}

    def set_bundle(self, bundle, variant: str, cfg, emphasis: str | None = None) -> None:
        self.bundle, self.variant, self.cfg, self.emphasis = bundle, variant, cfg, emphasis
        self.redraw()

    def _enable_controls(self, mrd_variant: str) -> None:
        """Grey out what the current view can't show."""
        ogive = self.mode.currentData() == "ogive"
        for key in ("se", "detection"):
            self.boxes[key].setEnabled(not ogive)
            self.boxes[key].setToolTip("drawn on the per-mode values only" if ogive else "")
        rows = sum(plot_rows(self.shown_groups()), [])
        for name, box in self.normalize.items():
            box.setVisible(name in rows)
        variances = "variances" in self.shown_groups()
        self.boxes["log"].setEnabled(variances)
        self.boxes["log"].setToolTip("" if variances else "no variances shown")
        self.boxes["unexcised"].setEnabled(mrd_variant == "mrd")
        self.boxes["unexcised"].setToolTip("" if mrd_variant == "mrd" else "these already are the unexcised spectra")

    # --- drawing -------------------------------------------------------------------------------

    def redraw(self) -> None:
        self.graphics.clear()
        self.plots = {}
        b = self.bundle
        if b is None or b.mrd is None or self.cfg is None:
            self.info.setText("no spectra for this slot")
            return
        mrd_variant = spectrum_variant(b, self.variant)
        self._enable_controls(mrd_variant)
        views, rerun_sel = rerun_detection(b, self.variant, self.cfg.secondary)
        stored_sel = stored_selection(b, self.variant)
        self._describe(views, rerun_sel, stored_sel, mrd_variant)

        rows = plot_rows(self.shown_groups())
        if not rows:
            self.info.setText(self.info.text() + " · <i>no plots chosen</i>")
            return
        # rows of different lengths share one grid of equal columns, each plot spanning a share
        columns = math.lcm(*(len(r) for r in rows))
        grid = self.graphics.ci.layout
        for c in range(max(columns, self._columns)):
            grid.setColumnStretchFactor(c, 1 if c < columns else 0)
        self._columns = columns
        for r, names in enumerate(rows):
            span = columns // len(names)
            last_row = r == len(rows) - 1
            for c, name in enumerate(names):
                if name.startswith("te_"):
                    var = name[len("te_"):]
                    plot = self._new_plot(r, c * span, span, False)
                    plot.setTitle(f"{SPECTRUM_LABELS[var]} transport efficiency", color="#222")
                    if last_row:
                        plot.setLabel("bottom", "ladder rung [s]")
                    self.plots[name] = plot
                    self._draw_te(plot, var, mrd_variant, views, stored_sel)
                    continue
                plot = self._new_plot(r, c * span, span, name not in COSPECTRA and self.boxes["log"].isChecked())
                title = (f"{SPECTRUM_LABELS[name]} (normalized)" if self.normalize[name].isChecked()
                         else f"{SPECTRUM_LABELS[name]} [{SPECTRUM_UNITS[name]}]")
                emphasised = name == self.emphasis
                plot.setTitle(f"<b>{title}</b>" if emphasised else title, color="#b3123f" if emphasised else "#222")
                if last_row:
                    plot.setLabel("bottom", "scale [s]")
                self.plots[name] = plot
                self._draw_spectrum(plot, name, mrd_variant, views, stored_sel)

    def _new_plot(self, row: int, col: int, span: int, log_y: bool) -> pg.PlotItem:
        axes = {"bottom": Log10Axis(orientation="bottom")}
        if log_y:
            axes["left"] = Log10Axis(orientation="left")
        plot = self.graphics.addPlot(row=row, col=col, colspan=span, axisItems=axes)
        plot.setPreferredWidth(1)  # widths from the equal column stretches, not from each plot's axes
        plot.showGrid(x=True, y=True, alpha=0.15)
        return plot

    def _describe(self, views, rerun_sel, stored_sel, mrd_variant) -> None:
        parts = []
        for family, view in views.items():
            det, stored = view.detection, view.stored
            tau = det.tau_s if det.status in ("found", "capped") else det.tau_lb_s
            text = f"<b>{family}</b>: {det.status}"
            if det.status in ("found", "capped", "unresolved"):
                text += f", τ {'≥ ' if det.status == 'unresolved' else ''}{_fmt_s(tau)}"
            if det.reversal_type:
                text += f" (peak {_fmt_s(det.peak_scale_s)}, {det.reversal_type} reversal at {_fmt_s(det.reversal_scale_s)})"
            if not view.agrees:
                text += (f" <span style='color:#b00'>— the run stored {stored['status']}, "
                         f"τ {_fmt_s(stored['tau_s'])}: detection code or config differs</span>")
            parts.append(text)
        if stored_sel is not None:
            parts.append(f"<b>selected ({self.variant})</b>: τ {_fmt_s(stored_sel['tau_s'])} from "
                         f"{stored_sel['source']} ({stored_sel['source_status']})")
            if self.variant != "naive" and (rerun_sel[1], rerun_sel[2]) != (stored_sel["source"], stored_sel["source_status"]):
                parts.append(f"<span style='color:#b00'>rerun now selects τ {_fmt_s(rerun_sel[0])} from {rerun_sel[1]}</span>")
        if mrd_variant != self.variant and self.variant == "mrd_unexcised":
            parts.append("mrd_unexcised equals mrd here (nothing was excised)")
        self.info.setText(" · ".join(parts))

    def _values(self, spec, log_y: bool):
        ogive = self.mode.currentData() == "ogive"
        y = spec.ogive if ogive else spec.value
        se = None if ogive else spec.se
        if log_y:
            with np.errstate(divide="ignore", invalid="ignore"):
                ly = np.where(y > 0, np.log10(y), np.nan)
                if se is not None:
                    top = np.log10(y + se) - ly
                    bottom = ly - np.log10(np.where(y - se > 0, y - se, np.nan))
                    bottom = np.where(np.isfinite(bottom), bottom, 1.0)  # the bar runs off the bottom
                    return ly, (top, bottom)
            return ly, None
        return y, (se, se) if se is not None else None

    def _spectrum(self, name: str, slot: int, boom: int, variant: str):
        spec = spectrum(self.bundle.mrd, slot, boom, variant, name)
        return normalized(spec) if self.normalize[name].isChecked() else spec

    def _draw_spectrum(self, plot, name, mrd_variant, views, stored_sel) -> None:
        b = self.bundle
        log_y = name not in COSPECTRA and self.boxes["log"].isChecked()
        if name in COSPECTRA:
            plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#999999", width=1)))

        if self.boxes["neighbours"].isChecked():
            for dk in range(-NEIGHBOURS, NEIGHBOURS + 1):
                other = self._spectrum(name, b.slot + dk, b.boom, mrd_variant) if dk else None
                if other is not None:
                    y, _ = self._values(other, log_y)
                    plot.plot(np.log10(other.scale_s), y, pen=pg.mkPen((120, 120, 120, 70), width=1))
        if self.boxes["booms"].isChecked():
            for boom in BOOMS:
                if boom == b.boom:
                    continue
                other = self._spectrum(name, b.slot, boom, mrd_variant)
                if other is not None:
                    y, _ = self._values(other, log_y)
                    c = style.boom_color(boom)
                    c.setAlpha(170)
                    plot.plot(np.log10(other.scale_s), y, pen=pg.mkPen(c, width=1))
        if self.boxes["unexcised"].isChecked() and mrd_variant == "mrd":
            other = self._spectrum(name, b.slot, b.boom, "mrd_unexcised")
            if other is not None:
                y, _ = self._values(other, log_y)
                plot.plot(np.log10(other.scale_s), y, pen=pg.mkPen(_UNEXCISED, width=1.2, style=Qt.PenStyle.DashLine))

        raw = spectrum(b.mrd, b.slot, b.boom, mrd_variant, name)
        spec = self._spectrum(name, b.slot, b.boom, mrd_variant)
        if spec is None:
            return
        factor = normalizing_factor(raw) if self.normalize[name].isChecked() else 1.0
        x = np.log10(spec.scale_s)
        y, bars = self._values(spec, log_y)
        if self.boxes["se"].isChecked() and bars is not None:
            top, bottom = bars
            ok = np.isfinite(y) & np.isfinite(top) & np.isfinite(bottom)
            plot.addItem(pg.ErrorBarItem(x=x[ok], y=y[ok], top=top[ok], bottom=bottom[ok], beam=0.03,
                                         pen=pg.mkPen((31, 78, 156, 140))))
        if np.isfinite(y).any():
            plot.plot(x, y, pen=pg.mkPen(_MAIN, width=1.6), symbol="o", symbolSize=5, symbolBrush=_MAIN,
                      symbolPen=None)

        for family, view in views.items():
            if view.spectrum == name and self.boxes["detection"].isChecked() and self.mode.currentData() == "value":
                self._draw_detection(plot, view, x, factor)
        self._draw_tau_lines(plot, name, views, stored_sel)

    def _draw_te(self, plot, var, mrd_variant, views, stored_sel) -> None:
        """TE at each rung, with the same overlays and tau lines as the spectra."""
        b = self.bundle
        if self.boxes["neighbours"].isChecked():
            for dk in range(-NEIGHBOURS, NEIGHBOURS + 1):
                other = transport_efficiency(b.te, b.slot + dk, b.boom, mrd_variant, var) if dk else None
                if other is not None:
                    plot.plot(np.log10(other[0]), other[1], pen=pg.mkPen((120, 120, 120, 70), width=1))
        if self.boxes["booms"].isChecked():
            for boom in BOOMS:
                other = transport_efficiency(b.te, b.slot, boom, mrd_variant, var) if boom != b.boom else None
                if other is not None:
                    c = style.boom_color(boom)
                    c.setAlpha(170)
                    plot.plot(np.log10(other[0]), other[1], pen=pg.mkPen(c, width=1))
        if self.boxes["unexcised"].isChecked() and mrd_variant == "mrd":
            other = transport_efficiency(b.te, b.slot, b.boom, "mrd_unexcised", var)
            if other is not None:
                plot.plot(np.log10(other[0]), other[1], pen=pg.mkPen(_UNEXCISED, width=1.2, style=Qt.PenStyle.DashLine))
        own = transport_efficiency(b.te, b.slot, b.boom, mrd_variant, var)
        if own is not None and np.isfinite(own[1]).any():
            plot.plot(np.log10(own[0]), own[1], pen=pg.mkPen(_MAIN, width=1.6), symbol="o", symbolSize=5,
                      symbolBrush=_MAIN, symbolPen=None)
        plot.setYRange(0.0, 1.0, padding=0.03)
        self._draw_tau_lines(plot, var, views, stored_sel)

    def _draw_detection(self, plot, view, x, factor: float = 1.0) -> None:
        """`factor` scales the smoothed curve and band as the spectrum was (1 / its sum, when normalized)."""
        t = view.trace
        if "d_smooth" not in t or not t["d_smooth"]:
            return
        modes = sorted(t["d_smooth"])
        xs = np.array([x[i - 1] for i in modes])
        ds = factor * np.array([t["d_smooth"][i] for i in modes])
        half = abs(factor) * t["k_se"] * np.array([t["se_smooth"][i] for i in modes])
        upper = pg.PlotCurveItem(xs, ds + half, pen=pg.mkPen(None))
        lower = pg.PlotCurveItem(xs, ds - half, pen=pg.mkPen(None))
        plot.addItem(upper)
        plot.addItem(lower)
        plot.addItem(pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(255, 150, 40, 45)))
        plot.plot(xs, ds, pen=pg.mkPen("#ff7f0e", width=1.4, style=Qt.PenStyle.DashLine))
        region = pg.LinearRegionItem((x[t["lo"] - 1], x[t["hi"] - 1]), movable=False, brush=pg.mkBrush(0, 0, 0, 12),
                                     pen=pg.mkPen(None))
        region.setZValue(-100)
        plot.addItem(region)
        if t.get("peak") is not None:
            p = t["peak"]
            plot.addItem(pg.ScatterPlotItem([x[p - 1]], [factor * t["d_smooth"][p]], symbol="t1", size=13,
                                            brush=pg.mkBrush("#ff7f0e"), pen=pg.mkPen("k")))
        if t.get("reversal"):
            r, kind = t["reversal"]
            plot.addItem(pg.ScatterPlotItem([x[r - 1]], [factor * t["d_smooth"][r]], symbol="x", size=14,
                                            brush=pg.mkBrush("#d62728"), pen=pg.mkPen("#d62728", width=2)))
            label = pg.TextItem(f"{kind} reversal", color="#d62728", anchor=(0, 1))
            label.setPos(x[r - 1], factor * t["d_smooth"][r])
            plot.addItem(label)

    def _draw_tau_lines(self, plot, name, views, stored_sel) -> None:
        det_cfg = self.cfg.secondary.detection
        if name in DETECTED.values():
            for bound in (det_cfg.min_tau_s, det_cfg.max_tau_s):
                plot.addItem(pg.InfiniteLine(np.log10(bound), pen=pg.mkPen("#aaaaaa", width=1,
                                                                           style=Qt.PenStyle.DotLine)))
        for family, view in views.items():
            if view.spectrum != name or view.stored is None:
                continue
            stored = view.stored
            color, label = _TAU_PENS[family]
            value = stored["tau_s"] if stored["status"] in ("found", "capped") else stored["tau_lb_s"]
            if value is not None and np.isfinite(value):
                dashed = stored["status"] == "unresolved"
                plot.addItem(pg.InfiniteLine(np.log10(value), label=label + (" ≥" if dashed else ""),
                                             labelOpts={"position": 0.9, "color": color},
                                             pen=pg.mkPen(color, width=1.5,
                                                          style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)))
        if stored_sel is not None and np.isfinite(stored_sel["tau_s"]):
            plot.addItem(pg.InfiniteLine(np.log10(stored_sel["tau_s"]), label="selected τ",
                                         labelOpts={"position": 0.08, "color": "#000"},
                                         pen=pg.mkPen("#000000", width=2)))
