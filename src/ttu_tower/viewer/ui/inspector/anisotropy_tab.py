"""The slot's anisotropy states on the barycentric map: every boom (the
inspected one ringed, filtered ones hollow), the inspected boom's track
through the slots around this one, and its whole-period density behind.
"""
import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QGraphicsPolygonItem, QHBoxLayout, QLabel, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ttu_tower.viewer import anisotropy, timeline
from ttu_tower.viewer.catalog import VARIANTS, Quantity
from ttu_tower.viewer.ui import style, tables

_TRACK = "#444444"


def _eigen_quantity(variable: str) -> Quantity:
    return Quantity(key=f"boom_final|{variable}|", table="boom_final", variable=variable, stat=None, kind="boom",
                    variants=VARIANTS, label=variable, unit="", group="Turbulence")


def whole_period(run, index, boom: int, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Map positions of every final state of one boom. Runs on a worker thread."""
    l2 = timeline.load(run, index, _eigen_quantity("aniso_l2"), variant, [boom]).curves[boom]
    l3 = timeline.load(run, index, _eigen_quantity("aniso_l3"), variant, [boom]).curves[boom]
    _, a, b = timeline.joined(l2, l3)
    x, y = anisotropy.barycentric_xy(a, b)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


class AnisotropyTab(QWidget):
    boomClicked = Signal(int)
    slotClicked = Signal(int)

    def __init__(self, run, index, runner, parent=None):
        super().__init__(parent)
        self.run, self.index, self.runner = run, index, runner
        self.across = None
        self.variant, self.boom = "mrd", None
        self.states = None
        self.track = None
        self.density = None
        self.regions = QCheckBox("class regions")
        self.regions.setChecked(True)
        self.regions.toggled.connect(lambda *_: self.redraw())
        self.span = QSpinBox()
        self.span.setRange(0, 72)
        self.span.setValue(6)
        self.span.setPrefix("track ±")
        self.span.setSuffix(" slots")
        self.span.setToolTip("the inspected boom's states in the slots around this one (click one to go there)")
        self.span.valueChanged.connect(lambda *_: self._load_track())
        self.whole = QCheckBox("whole-period density (this boom)")
        self.whole.toggled.connect(lambda *_: self._load_density())
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setXRange(-0.1, 1.1)
        self.plot.setYRange(-0.12, 0.98)
        self.table = tables.new_table()

        controls = QHBoxLayout()
        for w in (self.regions, self.span, self.whole):
            controls.addWidget(w)
        controls.addWidget(QLabel("  " + "  ".join(
            f"<span style='color:{anisotropy.CASE_COLORS[c]}'>■</span> {c}" for c in anisotropy.CASES)))
        controls.addStretch(1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.plot)
        split.addWidget(self.table)
        split.setSizes([750, 550])
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.info)
        layout.addWidget(split, stretch=1)

    def clear(self, message: str = "") -> None:
        self.across = None
        self.plot.clear()
        self.info.setText(message)

    def set_across(self, across, variant: str, boom: int) -> None:
        moved = self.across is None or (across.slot, variant, boom) != (self.across.slot, self.variant, self.boom)
        new_boom = (boom, variant) != (self.boom, self.variant)
        self.across, self.variant, self.boom = across, variant, boom
        self.states = anisotropy.slot_states(across, variant)
        if moved:
            self.track = None
            self._load_track()
        if new_boom:
            self.density = None
            self._load_density()
        self.redraw()

    # --- background loads ---------------------------------------------------------------------------

    def _load_track(self) -> None:
        a = self.across
        if a is None or self.span.value() == 0:
            self.track = None
            self.redraw()
            return
        n = self.span.value()
        key = (a.slot, self.boom, self.variant, n)
        self.runner.submit(anisotropy.boom_track, self.index, self.boom, self.variant, a.slot - n, a.slot + n,
                           key=f"aniso-track-{id(self)}", label="reading the anisotropy track",
                           on_done=lambda track: self._track_ready(key, track))

    def _track_ready(self, key, track) -> None:
        a = self.across
        if not shiboken6.isValid(self) or a is None or key != (a.slot, self.boom, self.variant, self.span.value()):
            return
        self.track = track
        self.redraw()

    def _load_density(self) -> None:
        if not self.whole.isChecked() or self.boom is None:
            self.density = None
            self.redraw()
            return
        key = (self.boom, self.variant)
        self.runner.submit(whole_period, self.run, self.index, self.boom, self.variant, key=f"aniso-whole-{id(self)}",
                           label="reading the boom's anisotropy", on_done=lambda xy: self._density_ready(key, xy))

    def _density_ready(self, key, xy) -> None:
        if not shiboken6.isValid(self) or key != (self.boom, self.variant):
            return
        self.density = xy
        self.redraw()

    # --- drawing ------------------------------------------------------------------------------------

    def _polygon(self, points, fill, pen=None, z: float = -10) -> None:
        item = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in points]))
        item.setBrush(QBrush(fill))
        item.setPen(pen if pen is not None else QPen(Qt.PenStyle.NoPen))
        item.setZValue(z)
        self.plot.addItem(item)

    def redraw(self) -> None:
        self.plot.clear()
        if self.states is None:
            return
        if self.regions.isChecked():
            for name, tris in anisotropy.region_triangles().items():
                color = QColor(anisotropy.CASE_COLORS[name])
                color.setAlpha(45)
                for tri in tris:
                    self._polygon(tri, color)
        if self.whole.isChecked() and self.density is not None and self.density[0].size:
            self._draw_density(*self.density)
        corners = anisotropy.CORNERS
        outline = [corners["1C"], corners["2C"], corners["3C"], corners["1C"]]
        self.plot.plot([p[0] for p in outline], [p[1] for p in outline], pen=pg.mkPen("k", width=2))
        (ax, ay), (bx, by) = anisotropy.plane_strain_line()
        self.plot.plot([ax, bx], [ay, by], pen=pg.mkPen("#333333", width=1, style=Qt.PenStyle.DashLine))
        for name, (x, y) in corners.items():
            label = pg.TextItem({"1C": "1C (linear)", "2C": "2C (planar)", "3C": "3C (isotropic)"}[name],
                                color="#000", anchor=(0.5, 0 if y == 0 else 1))
            label.setPos(x, y - 0.02 if y == 0 else y + 0.02)
            self.plot.addItem(label)
        self._draw_track()
        self._draw_booms()
        self._describe()

    def _draw_density(self, x, y) -> None:
        counts, ex, ey = np.histogram2d(x, y, bins=60, range=[[0, 1], [0, anisotropy.SQRT3 / 2]])
        cx, cy = np.meshgrid((ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2, indexing="ij")
        inside = (cy <= anisotropy.SQRT3 * cx) & (cy <= anisotropy.SQRT3 * (1 - cx))
        img = np.where((counts > 0) & inside, np.log10(np.maximum(counts, 1)), np.nan)
        item = pg.ImageItem(img, axisOrder="col-major")
        item.setColorMap(pg.ColorMap([0.0, 1.0], [(236, 236, 244), (70, 70, 120)]))
        item.setRect(ex[0], ey[0], ex[-1] - ex[0], ey[-1] - ey[0])
        item.setOpacity(0.8)
        item.setZValue(-5)
        self.plot.addItem(item)

    def _draw_track(self) -> None:
        t = self.track
        if t is None or t.empty:
            return
        t = t[np.isfinite(t["x"]) & np.isfinite(t["y"])]
        self.plot.plot(t["x"].to_numpy(), t["y"].to_numpy(), pen=pg.mkPen(_TRACK, width=1))
        k = self.across.slot
        spots = [{"pos": (r.x, r.y), "data": int(r.slot), "size": 6, "pen": None,
                  "brush": pg.mkBrush(style.category_color(0) if r.slot < k else style.category_color(1))}
                 for r in t.itertuples(index=False) if r.slot != k]
        item = pg.ScatterPlotItem(spots=spots, hoverable=True,
                                  tip=lambda x, y, data: f"slot {data} ({data - k:+d}); click to go there")
        item.sigClicked.connect(lambda _i, pts, _e: self.slotClicked.emit(int(pts[0].data())) if len(pts) else None)
        self.plot.addItem(item)

    def _draw_booms(self) -> None:
        spots = []
        for r in self.states.itertuples(index=False):
            if not (np.isfinite(r.x) and np.isfinite(r.y)):
                continue
            color = style.boom_color(r.boom)
            ring = r.boom == self.boom
            tip = f"b{r.boom}: {r.state}" + (" (filtered by tertiary)" if r.filtered else "")
            spots.append({"pos": (r.x, r.y), "data": (int(r.boom), tip), "size": 14 if ring else 10,
                          "brush": None if r.filtered else pg.mkBrush(color),
                          "pen": pg.mkPen("#000", width=2.5) if ring else pg.mkPen(color, width=2 if r.filtered else 1)})
        item = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=lambda x, y, data: data[1])
        item.setZValue(10)
        item.sigClicked.connect(lambda _i, pts, _e: self.boomClicked.emit(pts[0].data()[0]) if len(pts) else None)
        self.plot.addItem(item)

    def _describe(self) -> None:
        s = self.states
        shown = s[["boom", "height", "state", "aniso_l1", "aniso_l2", "aniso_l3", "aniso_beta", "aniso_phi", "filtered"]]
        shown = shown.rename(columns={"aniso_l1": "λ1", "aniso_l2": "λ2", "aniso_l3": "λ3", "aniso_beta": "β [deg]",
                                      "aniso_phi": "φ [deg]", "height": "z [m]"})

        def color(i, col, v):
            if col == "state" and isinstance(v, str):
                c = QColor(anisotropy.CASE_COLORS[v])
                c.setAlpha(90)
                return c
            return None

        tables.fill(self.table, shown.reset_index(drop=True), color=color)
        row = s[s["boom"] == self.boom]
        if row.empty or not np.isfinite(row["x"].iloc[0]):
            self.info.setText(f"b{self.boom} has no anisotropy state in this slot ({self.variant})")
        else:
            r = row.iloc[0]
            self.info.setText(f"<b>b{self.boom}</b> ({self.variant}): <b>{r['state']}</b>"
                              + (" (filtered by tertiary: shown hollow, as it was before)" if r["filtered"] else "")
                              + f" · β {r['aniso_beta']:.3g}°, φ {r['aniso_phi']:.3g}°"
                              + " · filled: final values; hollow: filtered; ringed: the inspected boom")
