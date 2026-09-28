"""The slot's MRD variance spectra and cospectra, with how tau was found:
the detection rerun on the stored spectra (smoothed curve, significance band,
searched range, peak and reversal) and the run's stored tau lines.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ttu_tower.constants import BOOMS
from ttu_tower.viewer.slotdata import (
    COSPECTRA, DETECTED, NEIGHBOURS, SPECTRA, SPECTRUM_LABELS, SPECTRUM_UNITS, rerun_detection, spectrum,
    spectrum_variant, stored_selection,
)
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.axes import Log10Axis

_MAIN = "#1f4e9c"
_UNEXCISED = "#b3123f"
_TAU_PENS = {"heat": ("#d62728", "heat τ"), "momentum": ("#2a8a4a", "momentum τ")}


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
        self.mode.currentIndexChanged.connect(lambda *_: self.redraw())
        controls = QHBoxLayout()
        controls.addWidget(self.mode)
        for box in self.boxes.values():
            controls.addWidget(box)
        controls.addStretch(1)
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.graphics = pg.GraphicsLayoutWidget()
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.info)
        layout.addWidget(self.graphics, stretch=1)
        self.plots: dict[str, pg.PlotItem] = {}

    def set_bundle(self, bundle, variant: str, cfg, emphasis: str | None = None) -> None:
        self.bundle, self.variant, self.cfg, self.emphasis = bundle, variant, cfg, emphasis
        self.redraw()

    # --- drawing -------------------------------------------------------------------------------

    def redraw(self) -> None:
        self.graphics.clear()
        self.plots = {}
        b = self.bundle
        if b is None or b.mrd is None or self.cfg is None:
            self.info.setText("no spectra for this slot")
            return
        mrd_variant = spectrum_variant(b, self.variant)
        views, rerun_sel = rerun_detection(b, self.variant, self.cfg.secondary)
        stored_sel = stored_selection(b, self.variant)
        self._describe(views, rerun_sel, stored_sel, mrd_variant)

        for i, name in enumerate(SPECTRA):
            axes = {"bottom": Log10Axis(orientation="bottom")}
            if name not in COSPECTRA and self.boxes["log"].isChecked():
                axes["left"] = Log10Axis(orientation="left")
            plot = self.graphics.addPlot(row=i // 4, col=i % 4, axisItems=axes)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").enableAutoSIPrefix(False)
            title = f"{SPECTRUM_LABELS[name]} [{SPECTRUM_UNITS[name]}]"
            emphasised = name == self.emphasis
            plot.setTitle(f"<b>{title}</b>" if emphasised else title, color="#b3123f" if emphasised else "#222")
            if i // 4 == 1:
                plot.setLabel("bottom", "scale [s]")
            self.plots[name] = plot
            self._draw_spectrum(plot, name, mrd_variant, views, stored_sel)

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

    def _draw_spectrum(self, plot, name, mrd_variant, views, stored_sel) -> None:
        b = self.bundle
        log_y = name not in COSPECTRA and self.boxes["log"].isChecked()
        if name in COSPECTRA:
            plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#999999", width=1)))

        if self.boxes["neighbours"].isChecked():
            for dk in range(-NEIGHBOURS, NEIGHBOURS + 1):
                other = spectrum(b.mrd, b.slot + dk, b.boom, mrd_variant, name) if dk else None
                if other is not None:
                    y, _ = self._values(other, log_y)
                    plot.plot(np.log10(other.scale_s), y, pen=pg.mkPen((120, 120, 120, 70), width=1))
        if self.boxes["booms"].isChecked():
            for boom in BOOMS:
                if boom == b.boom:
                    continue
                other = spectrum(b.mrd, b.slot, boom, mrd_variant, name)
                if other is not None:
                    y, _ = self._values(other, log_y)
                    c = style.boom_color(boom)
                    c.setAlpha(170)
                    plot.plot(np.log10(other.scale_s), y, pen=pg.mkPen(c, width=1))
        if self.boxes["unexcised"].isChecked() and mrd_variant == "mrd":
            other = spectrum(b.mrd, b.slot, b.boom, "mrd_unexcised", name)
            if other is not None:
                y, _ = self._values(other, log_y)
                plot.plot(np.log10(other.scale_s), y, pen=pg.mkPen(_UNEXCISED, width=1.2, style=Qt.PenStyle.DashLine))

        spec = spectrum(b.mrd, b.slot, b.boom, mrd_variant, name)
        if spec is None:
            return
        x = np.log10(spec.scale_s)
        y, bars = self._values(spec, log_y)
        if self.boxes["se"].isChecked() and bars is not None:
            top, bottom = bars
            ok = np.isfinite(y) & np.isfinite(top) & np.isfinite(bottom)
            plot.addItem(pg.ErrorBarItem(x=x[ok], y=y[ok], top=top[ok], bottom=bottom[ok], beam=0.03,
                                         pen=pg.mkPen((31, 78, 156, 140))))
        plot.plot(x, y, pen=pg.mkPen(_MAIN, width=1.6), symbol="o", symbolSize=5, symbolBrush=_MAIN, symbolPen=None)

        for family, view in views.items():
            if view.spectrum == name and self.boxes["detection"].isChecked() and self.mode.currentData() == "value":
                self._draw_detection(plot, view, x)
        self._draw_tau_lines(plot, name, views, stored_sel)

    def _draw_detection(self, plot, view, x) -> None:
        t = view.trace
        if "d_smooth" not in t or not t["d_smooth"]:
            return
        modes = sorted(t["d_smooth"])
        xs = np.array([x[i - 1] for i in modes])
        ds = np.array([t["d_smooth"][i] for i in modes])
        half = t["k_se"] * np.array([t["se_smooth"][i] for i in modes])
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
            plot.addItem(pg.ScatterPlotItem([x[p - 1]], [t["d_smooth"][p]], symbol="t1", size=13,
                                            brush=pg.mkBrush("#ff7f0e"), pen=pg.mkPen("k")))
        if t.get("reversal"):
            r, kind = t["reversal"]
            plot.addItem(pg.ScatterPlotItem([x[r - 1]], [t["d_smooth"][r]], symbol="x", size=14,
                                            brush=pg.mkBrush("#d62728"), pen=pg.mkPen("#d62728", width=2)))
            label = pg.TextItem(f"{kind} reversal", color="#d62728", anchor=(0, 1))
            label.setPos(x[r - 1], t["d_smooth"][r])
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
