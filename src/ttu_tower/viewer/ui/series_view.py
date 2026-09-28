"""The slot viewer: the 50-Hz series behind a clicked slot - as measured, QC
applied, or unexcised - in the earth, streamwise or sonic frame, with the
samples each test removed ghosted and a rug per test. Neighbouring slots can
be stepped to, or loaded alongside to see several slots in a row.
"""
import numpy as np
import pyarrow.dataset as pa_dataset
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ttu_tower.constants import HEIGHTS, SAMPLE_HZ
from ttu_tower.flags import TESTS
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer.catalog import variable_label
from ttu_tower.viewer.reprocess import removed_by_test, to_streamwise
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, display_offset_s, format_slot, sample_to_unix, slot_to_unix
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.curves import FixedXViewBox, PlotDecimator

MODES = (("unexcised", "unexcised"), ("QC applied", "qc"), ("as measured", "as_measured"),
         ("QC over as measured", "overlay"))
FRAMES = (("earth (ue, vn, w)", "earth"), ("streamwise, per slot (u, v, w)", "streamwise"),
          ("sonic instrument (u, v, w)", "sonic"))
MAX_SLOTS = 36  # six hours
_SERIES_COLORS = ("#1f4e9c", "#b3123f", "#2a8a4a", "#8a5a00", "#6a3d9a", "#00798c", "#555555", "#aa4499")
SAMPLE_TESTS = ("bounds", "spike", "coupled")  # removed sample by sample: ghosted at their measured value
_RUG_GROUPS = (("sonic", ("ue", "vn", "w", "u", "v")), ("ts", ("ts", "vpts")), ("t", ("t",)), ("rh", ("rh",)),
               ("p", ("p",)))
_RUG_ROW_PX = 16
_EARTH_TO_STREAM = {"ue": "u", "vn": "v"}


def _intervals(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def rug_rows(removed: dict, filled: dict | None = None) -> dict[str, np.ndarray]:
    """"test · group" -> union mask, for every test and variable group present."""
    rows = {}
    sources = dict(removed)
    if filled:
        sources["filled"] = filled
    order = list(TESTS) + ["coupled", "filled"]
    for test, per_var in sorted(sources.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)):
        for group, members in _RUG_GROUPS:
            masks = [m for v, m in per_var.items() if v in members]
            if masks:
                union = np.logical_or.reduce(masks)
                if union.any():
                    rows[f"{test} · {group}"] = union
    return rows


class SeriesWindow(QWidget):
    closed = Signal(object)

    def __init__(self, reprocessor, runner, run, index=None, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.reprocessor, self.runner, self.run, self.index = reprocessor, runner, run, index
        self.boom = self.slot = None
        self.lo = self.hi = None  # the loaded slots, inclusive
        self.variables: list[str] = []
        self.hidden: set[str] = set()
        self.context = ""
        self.result = None
        self._closed = False

        self.header = QLabel()
        self.header.setTextFormat(Qt.TextFormat.RichText)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.mode = QComboBox()
        for label, key in MODES:
            self.mode.addItem(label, key)
        self.mode.currentIndexChanged.connect(lambda *_: self.reload())
        self.frame = QComboBox()
        for label, key in FRAMES:
            self.frame.addItem(label, key)
        self.frame.currentIndexChanged.connect(lambda *_: self.reload())
        self.wide = QCheckBox("±35-min context")
        self.wide.setToolTip("show 35 min either side of the loaded slots (as far as an MRD detection window reaches)")
        self.wide.toggled.connect(lambda *_: self.reload())
        self.band = QCheckBox("despike band")
        self.band.setChecked(True)
        self.band.toggled.connect(lambda *_: self.redraw())
        self.pin = QCheckBox("pin")
        self.pin.setToolTip("keep this window; new clicks open a new one")

        nav = QHBoxLayout()
        self.nav_buttons = {}
        for name, label, tip, handler in (
            ("prev", "◀ slot", "show the previous slot instead", lambda: self.shift(-1)),
            ("next", "slot ▶", "show the next slot instead", lambda: self.shift(1)),
            ("earlier", "+ earlier", "load the previous slot alongside", lambda: self.extend(-1)),
            ("later", "+ later", "load the next slot alongside", lambda: self.extend(1)),
            ("single", "single slot", "back to just the focus slot", self.collapse),
        ):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            self.nav_buttons[name] = btn
            nav.addWidget(btn)
        nav.addStretch(1)

        self.series_boxes_row = QHBoxLayout()
        self.series_boxes_row.setContentsMargins(0, 0, 0, 0)
        self.series_boxes: dict[str, QCheckBox] = {}
        boxes = QWidget()
        boxes.setLayout(self.series_boxes_row)

        self.graphics = pg.GraphicsLayoutWidget()
        controls = QHBoxLayout()
        for w in (self.mode, self.frame, self.wide, self.band, self.pin):
            controls.addWidget(w)
        controls.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(controls)
        layout.addLayout(nav)
        layout.addWidget(boxes)
        layout.addWidget(self.status)
        layout.addWidget(self.graphics, stretch=1)
        self.resize(1250, 900)
        self._decimators: list[PlotDecimator] = []
        self.plots: dict[str, pg.PlotItem] = {}

    @property
    def key(self) -> str:
        return f"series-{id(self)}"

    # --- what is shown ------------------------------------------------------------------------

    def show_slot(self, boom: int, slot: int, mode: str, variables: list[str], context: str = "") -> None:
        self.boom, self.slot, self.lo, self.hi = boom, slot, slot, slot
        self.variables, self.context = list(variables), context
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex([k for _, k in MODES].index(mode))
        self.mode.blockSignals(False)
        self._build_series_boxes()
        self.reload()
        self.show()
        self.raise_()

    def shift(self, step: int) -> None:
        if self.slot is None:
            return
        self.slot, self.lo, self.hi = self.slot + step, self.lo + step, self.hi + step
        self.reload()

    def extend(self, side: int) -> None:
        if self.slot is None:
            return
        if self.hi - self.lo + 1 >= MAX_SLOTS:
            self.status.setText(f"at most {MAX_SLOTS} slots at once")
            return
        if side < 0:
            self.lo -= 1
        else:
            self.hi += 1
        self.reload()

    def collapse(self) -> None:
        if self.slot is not None:
            self.lo = self.hi = self.slot
            self.reload()

    def current_mode(self) -> str:
        return self.mode.currentData()

    def window_samples(self) -> tuple[int, int]:
        margin = round((35.0 if self.wide.isChecked() else 10.0) * 60 * SAMPLE_HZ)
        g0 = self.lo * SAMPLES_PER_SLOT - margin
        return g0, (self.hi - self.lo + 1) * SAMPLES_PER_SLOT + 2 * margin

    def _build_series_boxes(self) -> None:
        while self.series_boxes_row.count():
            item = self.series_boxes_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.series_boxes = {}
        self.series_boxes_row.addWidget(QLabel("series:"))
        for var in self.variables:
            box = QCheckBox(variable_label(var)[0])
            box.setChecked(var not in self.hidden)
            box.toggled.connect(lambda on, v=var: self._toggle_series(v, on))
            self.series_boxes[var] = box
            self.series_boxes_row.addWidget(box)
        self.series_boxes_row.addStretch(1)

    def _toggle_series(self, var: str, on: bool) -> None:
        if on:
            self.hidden.discard(var)
        else:
            self.hidden.add(var)
        self.redraw()

    def reload(self) -> None:
        if self.boom is None:
            return
        mode = self.current_mode()
        self.frame.model().item(2).setEnabled(mode == "as_measured")  # the sonic frame exists only as measured
        g0, n = self.window_samples()
        self._update_header()
        self.status.setText("reprocessing from the raw files…")
        self.runner.submit(compute_series, self.reprocessor, self.index, self.boom, g0, n, mode,
                           self.frame.currentData(), key=self.key, label=f"reprocessing boom {self.boom}",
                           on_done=self._loaded, on_error=self._failed)

    def _update_header(self) -> None:
        tz = self.run.timezone
        if self.lo == self.hi:
            span = f"slot {format_slot(self.slot, tz)}"
        else:
            span = (f"{self.hi - self.lo + 1} slots from {format_slot(self.lo, tz)[:16]}"
                    f" · focus {format_slot(self.slot, tz)}")
        self.header.setText(f"<b>Boom {self.boom}</b> ({HEIGHTS[self.boom]:g} m) · {span} · "
                            f"<b>{self.mode.currentText()}</b>" + (f" · {self.context}" if self.context else ""))

    def _failed(self, exc, tb) -> None:
        if not self._closed:
            self.status.setText(f"<span style='color:#b00'>could not reprocess: {exc}</span>")

    def _loaded(self, result) -> None:
        if self._closed:  # the window went away while its job ran
            return
        self.result = result
        notes = []
        missing = sorted(set(result["missing"]))
        if missing:
            notes.append(f"no usable file for half-hours {missing}")
        if result["mode"] == "as_measured" and result["removed"]:
            notes.append("flag rug: the run's stored flags")
        if result["bearings"]:
            wind = ", ".join(f"{format_slot(k, self.run.timezone)[11:16]} from {b:.0f}°"
                             for k, b in sorted(result["bearings"].items()) if self.lo <= k <= self.hi)
            notes.append(f"streamwise: each slot rotated to its own mean wind ({wind})")
        if self.frame.currentData() == "sonic" and result["frame"] != "sonic":
            notes.append("the sonic frame exists only as measured; showing the earth frame")
        self.status.setText(" · ".join(notes))
        self.redraw()

    # --- drawing -------------------------------------------------------------------------------

    def redraw(self) -> None:
        self.graphics.clear()
        self._decimators, self.plots = [], {}
        if self.result is None:
            return
        am, qc, mode, frame = self.result["am"], self.result["qc"], self.result["mode"], self.result["frame"]
        base = qc if qc is not None else am
        x = sample_to_unix(base.g0 + np.arange(base.n))
        rename = {} if frame == "earth" else _EARTH_TO_STREAM
        available = set(am.series if mode in ("as_measured", "overlay") else ()) | set(qc.series if qc else ())
        names = [rename.get(v, v) for v in self.variables if v not in self.hidden]
        names = [v for v in names if v in available]

        first = None
        for row, var in enumerate(names):
            plot = self._new_plot(row, first)
            first = first or plot
            self.plots[var] = plot
            label, unit = variable_label(var)
            plot.setLabel("left", f"{label} [{unit}]" if unit else label)
            self._shade_slots(plot)
            dec = PlotDecimator(plot)
            self._decimators.append(dec)
            self._draw_variable(plot, dec, var, x, am, qc, mode, _SERIES_COLORS[row % len(_SERIES_COLORS)])
            plot.getAxis("bottom").setStyle(showValues=False)

        self._draw_rug(x, qc, self.result["removed"], len(names), first)
        self._fit_x()
        for dec in self._decimators:
            dec.redraw()
        for plot in self.plots.values():
            plot.enableAutoRange(axis=pg.ViewBox.YAxis)

    def _new_plot(self, row: int, link):
        vb = FixedXViewBox()
        axis = pg.DateAxisItem(orientation="bottom", utcOffset=display_offset_s(self.run.timezone))
        plot = self.graphics.addPlot(row=row, col=0, viewBox=vb, axisItems={"bottom": axis})
        plot.showGrid(x=True, y=True, alpha=0.15)
        left = plot.getAxis("left")
        left.enableAutoSIPrefix(False)
        left.setWidth(72)
        vb.setAutoVisible(y=True)
        vb.fitXRequested.connect(self._fit_x)
        if link is not None:
            plot.setXLink(link)
        return plot

    def _fit_x(self) -> None:
        if self.result is None:
            return
        base = self.result["qc"] or self.result["am"]
        items = self.graphics.ci.items
        if items:
            plot = next(iter(items))
            plot.setXRange(sample_to_unix(base.g0), sample_to_unix(base.g0 + base.n), padding=0.01)

    def _shade_slots(self, plot) -> None:
        for k in range(self.lo, self.hi + 1):
            region = pg.LinearRegionItem((slot_to_unix(k), slot_to_unix(k) + SLOT_SECONDS), movable=False,
                                         brush=pg.mkBrush(255, 220, 90, 70 if k == self.slot else 30),
                                         pen=pg.mkPen(None))
            region.setZValue(-100)
            plot.addItem(region, ignoreBounds=True)
        for k in range(self.lo, self.hi + 2):
            line = pg.InfiniteLine(slot_to_unix(k), pen=pg.mkPen("#d0b060", width=1, style=Qt.PenStyle.DashLine))
            line.setZValue(-90)
            plot.addItem(line, ignoreBounds=True)

    def _draw_variable(self, plot, dec, var, x, am, qc, mode, color) -> None:
        raw = am.series.get(var) if am is not None else None
        if mode in ("as_measured", "overlay") and raw is not None:
            item = pg.PlotDataItem(pen=pg.mkPen("#b8b8b8" if mode == "overlay" else color, width=1))
            plot.addItem(item)
            dec.add(item, x, raw)
        if mode == "as_measured" or qc is None or var not in qc.series:
            return

        series = qc.series[var]
        usable = qc.masks.get(var, np.isfinite(series))
        if mode != "overlay":  # what window/slot tests excluded, faint, under the usable part
            excluded = pg.PlotDataItem(pen=pg.mkPen("#c8c8c8", width=1))
            plot.addItem(excluded)
            dec.add(excluded, x, np.where(usable, np.nan, series))
        item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.2))
        plot.addItem(item)
        dec.add(item, x, np.where(usable, series, np.nan))

        if self.band.isChecked() and var in qc.despike_band:
            for bound in qc.despike_band[var]:
                edge = pg.PlotDataItem(pen=pg.mkPen("#8c8c8c", width=1, style=Qt.PenStyle.DashLine))
                plot.addItem(edge)
                dec.add(edge, x, bound)

        if raw is None:
            return
        for test in SAMPLE_TESTS:
            mask = qc.removed.get(test, {}).get(var)
            if mask is None:
                continue
            idx = np.flatnonzero(mask & np.isfinite(raw))
            if idx.size > 20_000:
                idx = idx[:: idx.size // 20_000 + 1]
            if idx.size:
                plot.addItem(pg.ScatterPlotItem(x[idx], raw[idx], size=6, pen=None, symbol="x",
                                                brush=pg.mkBrush(style.TEST_COLORS.get(test, "#000000"))))

    def _draw_rug(self, x, qc, removed, row, link):
        rug = self._new_plot(row, link)
        rug.setMouseEnabled(y=False)
        rug.setLabel("bottom", f"time ({self.run.timezone})")
        rows = rug_rows(removed, qc.filled if qc is not None else None)
        height = max(len(rows), 1) * _RUG_ROW_PX + 45
        rug.setMaximumHeight(height)
        rug.setMinimumHeight(height)
        unusable = set(self.run.unusable_tests) | {"bounds", "spike", "coupled"}
        ticks = []
        for i, (name, mask) in enumerate(rows.items()):
            test = name.split(" · ")[0]
            starts, ends = _intervals(mask)
            xs = np.empty(2 * starts.size)
            xs[0::2] = x[starts]
            xs[1::2] = x[np.minimum(ends, x.size) - 1] + 1 / SAMPLE_HZ
            recorded_only = test in TESTS and TESTS[test].kind == "record"
            pen = pg.mkPen(style.TEST_COLORS.get(test, "#000"), width=8 if test in unusable else 3,
                           style=Qt.PenStyle.DotLine if recorded_only else Qt.PenStyle.SolidLine)
            rug.addItem(pg.PlotDataItem(xs, np.full(xs.size, i), pen=pen, connect="pairs"))
            ticks.append((i, name))
        rug.getAxis("left").setTicks([ticks])
        rug.setYRange(-0.7, max(len(ticks), 1) - 0.3, padding=0)
        return rug

    def closeEvent(self, ev) -> None:
        self._closed = True
        self.closed.emit(self)
        super().closeEvent(ev)


def _stored_slot_means(index, boom, g0, n) -> dict:
    """k -> (ue, vn) mean of every slot the window touches, from the run's `means` table."""
    k0, k1 = g0 // SAMPLES_PER_SLOT, (g0 + n - 1) // SAMPLES_PER_SLOT
    f = pa_dataset.field
    df = index.read("means", booms=[boom], half_hours=range(k0 // 3, k1 // 3 + 1),
                    filter=(f("boom") == boom) & (f("slot") >= k0) & (f("slot") <= k1) & (f("stat") == "mean")
                    & f("variable").isin(["ue", "vn"]))
    if df.empty:
        return {}
    df = df.assign(variable=df["variable"].astype(str))
    wide = df.pivot_table(index="slot", columns="variable", values="value")
    if not {"ue", "vn"} <= set(wide.columns):
        return {}
    return {int(k): (row["ue"], row["vn"]) for k, row in wide.iterrows()}


def compute_series(reprocessor, index, boom, g0, n, mode, frame) -> dict:
    """Runs on a worker thread. As-measured mode needs no Stage B, so its flag
    rug comes from the run's stored flags instead.
    """
    sonic = frame == "sonic" and mode == "as_measured"
    am = reprocessor.as_measured(boom, g0, n, frame="sonic" if sonic else "earth")
    qc = None
    if mode in ("qc", "overlay", "unexcised"):
        qc = reprocessor.qc_applied(boom, g0, n, variant="mrd_unexcised" if mode == "unexcised" else "mrd")
        removed = qc.removed
    elif index is not None:
        removed = removed_by_test(index.flags_window(boom, g0, g0 + n), boom, g0, n)
    else:
        removed = {}
    missing = am.missing_half_hours + (qc.missing_half_hours if qc is not None else [])

    bearings, shown_frame = {}, "sonic" if sonic else "earth"
    if frame == "streamwise":
        means = _stored_slot_means(index, boom, g0, n) if index is not None else {}
        am, bearings = to_streamwise(am, means)
        if qc is not None:
            qc, bearings = to_streamwise(qc, means)
            removed = qc.removed
        else:
            removed = {test: {_EARTH_TO_STREAM.get(v, v): m for v, m in per_var.items()}
                       for test, per_var in removed.items()}
        shown_frame = "streamwise"
    return {"am": am, "qc": qc, "mode": mode, "frame": shown_frame, "missing": missing, "removed": removed,
            "bearings": bearings}
