"""Scatter's x, y and z in 3D, in a window of its own that follows Scatter
(the points shown, log axes and their sides, the selection). Drag to turn,
Ctrl-drag or middle-drag to pan, the wheel to zoom.

Each axis spans its 1st-99th percentiles, scaled to a unit cube; points
outside fall outside the box. Grids sit on the three walls at the back of the
starting view.
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.axis_units import factor_text, power_of_ten
from ttu_tower.viewer.ui.scatter import _POINT, Z_MAPS, z_fractions, z_levels

_FRAME = (0.35, 0.35, 0.35, 1.0)
_GRID = (0.82, 0.82, 0.82, 1.0)
_TEXT = (20, 20, 20, 255)
_EDGES = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
          ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
          ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]
VIEWS = {"turned": (25, -60), "x–y": (90, -90), "x–z": (0, -90), "y–z": (0, 0)}  # (elevation, azimuth)


def frame(v: np.ndarray) -> tuple[float, float]:
    """The 1st-99th percentiles, padded 5%."""
    lo, hi = (float(p) for p in np.percentile(v, [1, 99]))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def ticks(lo: float, hi: float, log: bool) -> tuple[list[float], list[str], int]:
    """(positions as plotted, labels, the power of ten factored out of them)."""
    if log:
        decades = [float(d) for d in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo <= d <= hi]
        if len(decades) < 2:
            decades = sorted(d + np.log10(m) for d in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)
                             for m in (1, 2, 5) if lo <= d + np.log10(m) <= hi)
        return decades, [f"{float(f'{10 ** t:.3g}'):g}" for t in decades], 0
    span = hi - lo
    step = 10.0 ** np.floor(np.log10(span / 5))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= 7:
            step *= m
            break
    at = [float(k * step) for k in range(int(np.ceil(lo / step)), int(np.floor(hi / step)) + 1)]  # whole steps
    power = power_of_ten(lo, hi)
    return at, [f"{t / 10.0 ** power + 0.0:.6g}" for t in at], power  # + 0.0: no "-0"


def _rgba(color, alpha: float = 0.8) -> tuple:
    c = pg.mkColor(color)
    return c.redF(), c.greenF(), c.blueF(), alpha


class Scatter3DWindow(QWidget):
    def __init__(self, scatter, parent=None):
        super().__init__(parent or scatter, Qt.WindowType.Window)
        self.scatter = scatter
        self.setWindowTitle("3D scatter")
        self.resize(900, 760)
        self.color_by = QComboBox()
        self.color_by.addItem("color by z", "z")
        self.color_by.addItem("color by stability", "stability")
        self.color_by.addItem("single color", "single color")
        self.color_by.currentIndexChanged.connect(lambda *_: self.refresh())
        self.point_size = QSpinBox()
        self.point_size.setRange(1, 12)
        self.point_size.setValue(3)
        self.point_size.setPrefix("size ")
        self.point_size.valueChanged.connect(lambda *_: self.refresh())
        self.status = QLabel()
        self.status.setWordWrap(True)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("w")
        self.view.opts["fov"] = 30  # nearly flat, so the face-on views read like 2D plots
        self.grid = gl.GLLinePlotItem(mode="lines", color=_GRID, width=1, antialias=True)
        self.box = gl.GLLinePlotItem(pos=np.array(_EDGES, dtype=float).reshape(-1, 3), mode="lines", color=_FRAME,
                                     width=1.2, antialias=True)
        self.points = gl.GLScatterPlotItem(pxMode=True)
        self.chosen = gl.GLScatterPlotItem(pxMode=True)
        for item in (self.points, self.chosen):
            item.setGLOptions("translucent")
        for item in (self.grid, self.box, self.points, self.chosen):
            self.view.addItem(item)
        self.labels: list = []

        row = QHBoxLayout()
        row.addWidget(self.color_by)
        row.addWidget(self.point_size)
        row.addStretch(1)
        row.addWidget(QLabel("view"))
        for name in VIEWS:
            button = QPushButton(name)
            button.setToolTip("the starting view, turned" if name == "turned" else f"face-on to the {name} plane")
            button.clicked.connect(lambda *_, n=name: self.look(n))
            row.addWidget(button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(row)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.status)
        self.look("turned")

    def look(self, name: str) -> None:
        elevation, azimuth = VIEWS[name]
        self.view.setCameraPosition(pos=pg.Vector(0.5, 0.5, 0.5), distance=4.4, elevation=elevation, azimuth=azimuth)

    def refresh(self) -> None:
        s = self.scatter
        self.color_by.model().item(1).setEnabled(s.stability is not None)
        slots, vx, vy, vz = s.points3d()
        titles = [s.axis_title(i) for i in range(3)]
        self.setWindowTitle("3D scatter · " + " · ".join(t for t in titles if t))
        if slots.size == 0:
            for item in (self.points, self.chosen, self.grid):
                item.setVisible(False)
            self._set_labels([])
            self.status.setText(s.empty_reason or ("no points with x, y and z on these axes" if s.has_z()
                                                   else "z has no values at these points: choose z in Scatter"))
            return
        logs = (s.log_x.isChecked(), s.log_y.isChecked(), s.log_z.isChecked())
        ranges = [frame(v) for v in (vx, vy, vz)]
        pos = np.column_stack([(v - lo) / (hi - lo) for v, (lo, hi) in zip((vx, vy, vz), ranges)])
        size = self.point_size.value()
        self.points.setData(pos=pos, color=self._colors(slots, vz), size=size)
        self.points.setVisible(True)
        on = np.isin(slots, s.selection) if s.selection.size and s.ring.isChecked() else np.zeros(slots.size, bool)
        if on.any():
            self.chosen.setData(pos=pos[on], color=_rgba(style.SELECTED_COLOR, 0.95), size=size + 4)
        self.chosen.setVisible(bool(on.any()))
        self._draw_axes(ranges, logs, titles)
        left_out = s._shown()[0].size - slots.size
        self.status.setText(f"{slots.size} points" + (f" ({left_out} without z left out)" if left_out else "")
                            + "; the box spans each axis's 1st-99th percentiles · drag to turn, Ctrl-drag or "
                              "middle-drag to pan, wheel to zoom")

    def _colors(self, slots, vz) -> np.ndarray:
        by = self.color_by.currentData()
        if by == "z":
            name = self.scatter.z_colormap()
            lo, hi = z_levels(vz, name == Z_MAPS["diverging"])
            colors = pg.colormap.getFromMatplotlib(name).map(z_fractions(vz, lo, hi), mode="float")
            colors[:, 3] = 0.8
            return colors
        colors = np.tile(_rgba(_POINT), (slots.size, 1))
        if by == "stability" and self.scatter.stability is not None:
            for _, mask, color in self.scatter.class_groups(slots):
                colors[mask] = _rgba(color)
        return colors

    def _draw_axes(self, ranges, logs, titles) -> None:
        grid, labels = [], []
        for axis, ((lo, hi), log, title) in enumerate(zip(ranges, logs, titles)):
            at, texts, power = ticks(lo, hi, log)
            for t, text in zip(at, texts):
                u = (t - lo) / (hi - lo)
                if axis == 0:
                    grid += [(u, 0, 0), (u, 1, 0), (u, 1, 0), (u, 1, 1)]
                    labels.append(((u, -0.08, 0), text))
                elif axis == 1:
                    grid += [(0, u, 0), (1, u, 0), (0, u, 0), (0, u, 1)]
                    labels.append(((1.06, u, 0), text))
                else:
                    grid += [(0, 0, u), (0, 1, u), (0, 1, u), (1, 1, u)]
                    labels.append(((-0.06, -0.06, u), text))
            name = title + (f" ({factor_text(power)})" if power else "") + (" (log scale)" if log else "")
            labels.append((((0.5, -0.24, 0), (1.22, 0.5, 0), (-0.08, -0.08, 1.12))[axis], name))
        self.grid.setData(pos=np.array(grid, dtype=float).reshape(-1, 3))
        self.grid.setVisible(bool(grid))
        self._set_labels(labels)

    def _set_labels(self, labels) -> None:
        for item in self.labels:
            self.view.removeItem(item)
        font = QFont()
        font.setPointSize(9)
        centre = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        self.labels = [gl.GLTextItem(pos=np.array(at, dtype=float), text=text, color=_TEXT, font=font, alignment=centre)
                       for at, text in labels]
        for item in self.labels:
            self.view.addItem(item)
