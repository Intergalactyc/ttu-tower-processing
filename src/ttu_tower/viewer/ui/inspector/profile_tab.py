"""The slot's vertical profiles: each boom's value against height, the
values tertiary filtered ghosted at what they were, the stored profile fits
drawn through them, and the boom-pair quantities.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ttu_tower.viewer import profiledata
from ttu_tower.viewer.ui import style, tables
from ttu_tower.viewer.ui.axes import Log10Axis

_GHOST = "#9a9a9a"
_FIT_PENS = {"alpha": ("#d62728", Qt.PenStyle.SolidLine), "loglaw": ("#1f77b4", Qt.PenStyle.DashLine),
             "loglaw_c": ("#17becf", Qt.PenStyle.DotLine), "gamma": ("#d62728", Qt.PenStyle.SolidLine),
             "wdgamma": ("#d62728", Qt.PenStyle.SolidLine)}
_COLUMNS = 5


class ProfileTab(QWidget):
    boomClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.across = self.cfg = None
        self.variant, self.boom = "mrd", None
        self.log_z = QCheckBox("log height")
        self.log_z.toggled.connect(lambda *_: self.redraw())
        self.fits_box = QCheckBox("fits")
        self.fits_box.setChecked(True)
        self.fits_box.toggled.connect(lambda *_: self.redraw())
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.pairs = tables.new_table()
        self.fit_table = tables.new_table()

        controls = QHBoxLayout()
        controls.addWidget(self.log_z)
        controls.addWidget(self.fits_box)
        controls.addWidget(QLabel(f"<span style='color:{_GHOST}'>○ filtered by tertiary (value before filtering)"
                                  "</span> · click a point to inspect that boom"))
        controls.addStretch(1)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self.pairs)
        bottom.addWidget(self.fit_table)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.graphics)
        split.addWidget(bottom)
        split.setSizes([700, 180])
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.info)
        layout.addWidget(split, stretch=1)
        self.plots: dict[str, pg.PlotItem] = {}
        self.points: dict = {}

    def set_across(self, across, variant: str, boom: int, cfg) -> None:
        self.across, self.variant, self.boom, self.cfg = across, variant, boom, cfg
        self.redraw()

    def clear(self, message: str = "") -> None:
        self.across = None
        self.graphics.clear()
        self.plots = {}
        self.info.setText(message)

    def redraw(self) -> None:
        self.graphics.clear()
        self.plots, self.points = {}, {}
        a = self.across
        if a is None:
            return
        if "boom_final" not in a.tables:
            self.info.setText("this run has no tertiary stage, so there are no profiles")
            return
        log_z = self.log_z.isChecked()
        first = None
        for i, pq in enumerate(profiledata.PROFILE_QUANTITIES):
            axes = {"left": Log10Axis(orientation="left")} if log_z else {}
            plot = self.graphics.addPlot(row=i // _COLUMNS, col=i % _COLUMNS, axisItems=axes)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            variant = f" · {self.variant}" if pq.per_variant else ""
            plot.setTitle(f"{pq.label}{f' [{pq.unit}]' if pq.unit else ''}{variant}", size="9pt")
            if i % _COLUMNS == 0:
                plot.setLabel("left", "height [m]")
            if first is None:
                first = plot
            else:
                plot.setYLink(first)
            self.plots[pq.key] = plot
            points = profiledata.profile_points(a, pq, self.variant)
            self.points[pq.key] = points
            self._draw(plot, pq, points, log_z)
        self._describe()

    def _z(self, z, log_z: bool):
        return np.log10(z) if log_z else z

    def _draw(self, plot, pq, points, log_z: bool) -> None:
        z = self._z(points["height"].to_numpy(float), log_z)
        v = points["value"].to_numpy(float)
        ok = np.isfinite(v)
        plot.plot(v[ok], z[ok], pen=pg.mkPen("#444444", width=1))
        spots = []
        for (boom, value, pre, filtered, reasons), zi in zip(
                points[["boom", "value", "prefilter", "filtered", "reasons"]].itertuples(index=False), z):
            if filtered:
                spots.append({"pos": (pre, zi), "symbol": "o", "size": 9, "brush": None, "pen": pg.mkPen(_GHOST, width=1.5),
                              "data": (int(boom), f"b{boom} filtered ({reasons}); before filtering {pre:.4g}")})
            elif np.isfinite(value):
                ring = boom == self.boom
                tip = f"b{boom}: {value:.4g}" + (f" · {reasons}" if reasons else "")
                spots.append({"pos": (value, zi), "symbol": "o", "size": 11 if ring else 8,
                              "brush": pg.mkBrush(style.boom_color(boom)),
                              "pen": pg.mkPen("#000000", width=2) if ring else None, "data": (int(boom), tip)})
        scatter = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=lambda x, y, data: data[1])
        scatter.sigClicked.connect(lambda _item, pts, _ev: self.boomClicked.emit(pts[0].data()[0]) if len(pts) else None)
        plot.addItem(scatter)

        if self.fits_box.isChecked() and self.cfg is not None:
            for fit in profiledata.profile_fits(self.across, pq.key, points, self.variant, self.cfg.tertiary.fits):
                color, dash = _FIT_PENS.get(fit.name, ("#d62728", Qt.PenStyle.SolidLine))
                ok = np.isfinite(fit.y)
                plot.plot(fit.y[ok], self._z(fit.z[ok], log_z), pen=pg.mkPen(color, width=1.5, style=dash),
                          name=fit.text)

    def _describe(self) -> None:
        a, cfg = self.across, self.cfg
        parts = []
        if cfg is not None:
            for key in ("ws", "ti", "wd_std"):
                for fit in profiledata.profile_fits(a, key, self.points[key], self.variant, cfg.tertiary.fits):
                    parts.append(fit.text)
        self.info.setText(" · ".join(parts) if parts else "no profile fits for this slot")
        pair = cfg.post.stability.pair if cfg is not None else (0, 0)
        pairs = profiledata.pair_table(a, pair)

        def color(i, col, v):
            return tables.GOOD if i == 0 and len(pairs) and (pairs.iloc[0]["boom"], pairs.iloc[0]["boom2"]) == tuple(pair) else None

        tables.fill(self.pairs, pairs.rename(columns={"rib": "Ri_b", "lapse_vpt": "dθv/dz [K/m]"}), color=color)
        self.pairs.setToolTip(f"every boom pair; the stability pair b{pair[0]}–b{pair[1]} first")
        prof = a.table("profile")
        prof = prof[prof["variant"].isin(["none", self.variant])][["variant", "variable", "value"]]
        tables.fill(self.fit_table, prof.reset_index(drop=True))
