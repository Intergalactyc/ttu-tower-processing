"""Every boom's wind rose, stacked by height in a window of its own: drag to
turn, Ctrl-drag or middle-drag to pan, the wheel to zoom. Follows the Wind
rose tab (filtering, sectors, interval, lines of constant bearing); all roses
share one radial scale. North is +y, east +x.
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import windrose

_VIRIDIS = pg.colormap.get("viridis")
_RING = (0.75, 0.75, 0.75, 1.0)
_AXIS = (0.45, 0.45, 0.45, 1.0)
_BEARING = (0.84, 0.15, 0.16, 1.0)
_TEXT = (20, 20, 20, 255)
_MESONET = "mesonet"
GAP = 0.45  # between evenly spaced layers, in rose radii
VIEWS = {"turned": (28, -60), "from above": (90, -90), "from the south": (0, -90)}  # (elevation, azimuth)


def load_all(run, index, booms, filtered: bool):
    """Runs on a worker thread: {boom: Wind}, and the mesonet's (or None)."""
    return {b: windrose.tower_wind(run, index, b, filtered) for b in booms}, windrose.mesonet_wind(run, index)


def layer_heights(heights: list[float], spacing: str) -> np.ndarray:
    """Each layer's z: evenly spaced, or by height or log height over the same overall span."""
    n = len(heights)
    span = GAP * max(n - 1, 1)
    if spacing == "even" or n < 2:
        return np.arange(n) * GAP
    h = np.log10(heights) if spacing == "log height" else np.asarray(heights, dtype=float)
    return (h - h.min()) / ((h.max() - h.min()) or 1.0) * span


def wedges(table: np.ndarray, scale: float, z: float, arc_points: int = 8):
    """Flat stacked wedges of one rose at height z: (vertices, faces, face colors)."""
    sectors, bins = table.shape
    width = 360.0 / sectors
    verts, faces, colors = [], [], []
    for i in range(sectors):
        arc = np.radians(np.linspace(i * width - 0.45 * width, i * width + 0.45 * width, arc_points))
        sin, cos = np.sin(arc), np.cos(arc)
        r0 = 0.0
        for j in range(bins):
            r1 = r0 + table[i, j] / scale
            if r1 > r0:
                start = len(verts)
                verts += [(r0 * s, r0 * c, z) for s, c in zip(sin, cos)]
                verts += [(r1 * s, r1 * c, z) for s, c in zip(sin, cos)]
                color = _VIRIDIS.map(j / max(bins - 1, 1), mode="float")
                for k in range(arc_points - 1):
                    a, b, c, d = start + k, start + k + 1, start + arc_points + k, start + arc_points + k + 1
                    faces += [(a, b, d), (a, d, c)]
                    colors += [color, color]
            r0 = r1
    return (np.array(verts, dtype=float).reshape(-1, 3), np.array(faces, dtype=np.int64).reshape(-1, 3),
            np.array(colors, dtype=float).reshape(-1, 4))


def circle(r: float, z: float, n: int = 97) -> np.ndarray:
    """A ring as line segments (for mode="lines")."""
    t = np.linspace(0, 2 * np.pi, n)
    pts = np.column_stack([r * np.sin(t), r * np.cos(t), np.full(n, z)])
    return np.repeat(pts, 2, axis=0)[1:-1]


class WindRose3DWindow(QWidget):
    def __init__(self, rose_view, parent=None):
        super().__init__(parent or rose_view, Qt.WindowType.Window)
        self.rose_view = rose_view
        self.winds = None  # {boom: Wind}
        self.mesonet = None
        self.loaded_key = None  # (booms, filtered)
        self.setWindowTitle("3D wind rose")
        self.resize(900, 800)
        self.spacing = QComboBox()
        for label in ("even", "height", "log height"):
            self.spacing.addItem(f"layers: {label}", label)
        self.spacing.setToolTip("how the layers are spaced: evenly, or in proportion to the booms' heights")
        self.common = QCheckBox("only slots every boom has")
        self.common.setToolTip("every layer from the same slots (those with wind at every boom shown, and at the "
                               "mesonet if it's shown)")
        self.with_mesonet = QCheckBox("mesonet layer (10 m)")
        self.with_mesonet.setChecked(True)
        for w in (self.spacing,):
            w.currentIndexChanged.connect(lambda *_: self.refresh())
        for w in (self.common, self.with_mesonet):
            w.toggled.connect(lambda *_: self.refresh())
        self.status = QLabel()
        self.status.setWordWrap(True)
        labels = windrose_speed_labels()
        key = QLabel("speed [m/s]: " + "  ".join(
            f"<span style='color:{_VIRIDIS.map(j / (len(labels) - 1), mode='qcolor').name()}'>■</span> {s}"
            for j, s in enumerate(labels)))
        key.setWordWrap(True)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("w")
        self.view.opts["fov"] = 30
        self.mesh = gl.GLMeshItem(smooth=False, drawEdges=False)
        self.mesh.setVisible(False)  # until it has a mesh to draw
        self.lines = gl.GLLinePlotItem(mode="lines", color=_RING, width=1, antialias=True)
        self.bearing_lines = gl.GLLinePlotItem(mode="lines", color=_BEARING, width=1.2, antialias=True)
        for item in (self.mesh, self.lines, self.bearing_lines):
            self.view.addItem(item)
        self.labels: list = []

        row = QHBoxLayout()
        for w in (self.spacing, self.common, self.with_mesonet):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(QLabel("view"))
        for name in VIEWS:
            button = QPushButton(name)
            button.clicked.connect(lambda *_, n=name: self.look(n))
            row.addWidget(button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(row)
        layout.addWidget(key)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.status)
        self.look("turned")

    def look(self, name: str) -> None:
        elevation, azimuth = VIEWS[name]
        top = self._top() if self.winds is not None else 2.0
        self.view.setCameraPosition(pos=pg.Vector(0.0, 0.0, top / 2), distance=3.5 + 2.1 * top,
                                    elevation=elevation, azimuth=azimuth)

    def _top(self) -> float:
        return GAP * max(len(self.winds or ()) - 1 + self.with_mesonet.isChecked(), 1)

    # --- data -----------------------------------------------------------------------------------

    def reload(self) -> None:
        v = self.rose_view
        if v.run is None:
            return
        booms = tuple(v.boom.itemData(i) for i in range(v.boom.count()))
        key = (booms, v.filtering.currentData())
        if key == self.loaded_key or not booms:
            self.refresh()
            return
        self.status.setText(f"reading the wind at {len(booms)} booms…")
        v.runner.submit(load_all, v.run, v.index, booms, key[1], key=f"windrose3d-{id(self)}",
                        label="reading the wind at every boom", on_done=lambda result: self._loaded(key, result),
                        on_error=lambda exc, tb: self.status.setText(f"could not read the wind: {exc}"))

    def _loaded(self, key, result) -> None:
        if not shiboken6.isValid(self):
            return
        first = self.winds is None
        self.winds, self.mesonet = result
        self.loaded_key = key
        self.with_mesonet.setEnabled(self.mesonet is not None)
        self.refresh()
        if first:
            self.look("turned")  # framed on the layers now known

    def layers(self) -> list[tuple[str, float, windrose.Wind]]:
        """(label, height, wind) bottom to top, over the tab's interval (and common slots if chosen)."""
        x_range = self.rose_view.x_range
        out = [(f"b{b} ({HEIGHTS[b]:g} m)", HEIGHTS[b], windrose.in_range(w, x_range)) for b, w in self.winds.items()]
        if self.mesonet is not None and self.with_mesonet.isChecked():
            out.append(("mesonet (10 m)", 10.0, windrose.in_range(self.mesonet, x_range)))
        out.sort(key=lambda layer: (layer[1], layer[0] == "mesonet (10 m)"))
        if self.common.isChecked() and out:
            slots = out[0][2].slots
            for _, _, w in out[1:]:
                slots = np.intersect1d(slots, w.slots, assume_unique=True)
            out = [(label, h, windrose.only_slots(w, slots)) for label, h, w in out]
        return out

    def refresh(self) -> None:
        if self.winds is None:
            self.reload()
            return
        v = self.rose_view
        n = v.sectors.currentData()
        layers = self.layers()
        tables = [windrose.rose(w.ws, w.wd, n) if w.ws.size else None for _, _, w in layers]
        outer = max((t.sum(axis=1).max() for t in tables if t is not None and t.sum()), default=0.0)
        if not outer:
            self.mesh.setVisible(False)
            self._set_labels([])
            self.status.setText("no wind data in range")
            return
        step = next(s for s in (1, 2, 2.5, 5, 10, 20, 25, 50, 100) if outer / s <= 4)
        rim = float(np.arange(step, outer + step, step)[-1])
        zs = layer_heights([h for _, h, _ in layers], self.spacing.currentData())
        verts, faces, colors, rings, labels = [], [], [], [], []
        base = 0
        for (label, _, wind), table, z in zip(layers, tables, zs):
            if table is not None:
                vv, ff, cc = wedges(table, rim, z)
                verts.append(vv)
                faces.append(ff + base)
                colors.append(cc)
                base += len(vv)
            rings.append(circle(1.0, z))
            side = -1.0 if label.startswith("mesonet") else 1.0  # apart from b4's, at nearly the same height
            labels.append(((1.15 * side, -0.35, z), f"{label}: {wind.ws.size}"))
        for ring in np.arange(step, rim + step / 2, step):
            rings.append(circle(ring / rim, zs[-1]))
            labels.append(((np.sin(np.radians(22.5)) * ring / rim, np.cos(np.radians(22.5)) * ring / rim, zs[-1]),
                           f"{ring:g} %"))
        top = zs[-1]
        for name, bearing in {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}.items():
            t = np.radians(bearing)
            rings.append(np.array([(0, 0, 0), (np.sin(t), np.cos(t), 0)], dtype=float))
            rings.append(np.array([(0, 0, top), (np.sin(t), np.cos(t), top)], dtype=float))
            labels.append(((1.15 * np.sin(t), 1.15 * np.cos(t), 0), name))
        rings.append(np.array([(0, 0, 0), (0, 0, top)], dtype=float))
        bearing_segments = []
        try:
            bearings = windrose.parse_bearings(v.bearings.text())
        except ValueError:
            bearings = []
        for bearing in bearings:
            t = np.radians(bearing)
            for z in zs:
                bearing_segments += [(0, 0, z), (1.02 * np.sin(t), 1.02 * np.cos(t), z)]
        if verts:
            mesh = gl.MeshData(vertexes=np.concatenate(verts), faces=np.concatenate(faces),
                               faceColors=np.concatenate(colors))
            self.mesh.setMeshData(meshdata=mesh)
        self.mesh.setVisible(bool(verts))
        self.lines.setData(pos=np.concatenate(rings))
        self.bearing_lines.setData(pos=np.array(bearing_segments, dtype=float).reshape(-1, 3))
        self.bearing_lines.setVisible(bool(bearing_segments))
        self._set_labels(labels)
        which = "filtered" if v.filtering.currentData() else "first layer (unfiltered)"
        scope = "whole period" if v.x_range is None else "viewed interval"
        self.status.setText(f"{scope}, {which}, {n} sectors; one radial scale (outer ring {rim:g} %); slot counts "
                            "beside each layer · drag to turn, Ctrl-drag or middle-drag to pan, wheel to zoom")

    def _set_labels(self, labels) -> None:
        for item in self.labels:
            self.view.removeItem(item)
        font = QFont()
        font.setPointSize(9)
        centre = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        self.labels = [gl.GLTextItem(pos=np.array(at, dtype=float), text=text, color=_TEXT, font=font,
                                     alignment=centre) for at, text in labels]
        for item in self.labels:
            self.view.addItem(item)


def windrose_speed_labels() -> list[str]:
    edges = windrose.SPEED_EDGES
    return [f"{a:g}–{b:g}" if np.isfinite(b) else f"≥ {a:g}" for a, b in zip(edges[:-1], edges[1:])]
