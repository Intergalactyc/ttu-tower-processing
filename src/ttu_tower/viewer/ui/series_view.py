"""The Series tab of the Slot Inspector: the 50-Hz series behind a slot - as
measured, QC applied, or unexcised - in the earth, streamwise or sonic frame,
with the samples each test removed ghosted and a rug per test. Neighbouring
slots can be loaded alongside, and the ladder's tau blocks overlaid. A QC
what-if reprocesses the loaded slots with edited [qc] settings and shows
what that changes.
"""
import numpy as np
import pyarrow.dataset as pa_dataset
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.flags import TESTS
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer import qcwhatif
from ttu_tower.viewer.catalog import variable_label
from ttu_tower.viewer.reprocess import HFWindow, removed_by_test, tau_block_ids, to_streamwise
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, display_offset_s, format_slot, sample_to_unix, slot_to_unix
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.curves import FixedXViewBox, PlotDecimator
from ttu_tower.viewer.ui.qc_whatif_panel import QCWhatIfPanel

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
SHOWN_BY_DEFAULT = ("ue", "vn", "w", "vpts")  # the other series start unchecked
WHATIF_ADDED = "#e7298a"  # flagged only with the what-if settings
WHATIF_DROPPED = "#1b9e77"  # flagged only with the run's


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


def rug_differences(stored: dict[str, np.ndarray], whatif: dict[str, np.ndarray], n: int) -> dict:
    """"test · group" -> (flagged only with the what-if, flagged only with the
    run's), for every rug row where the two differ.
    """
    out = {}
    empty = np.zeros(n, dtype=bool)
    for name in list(stored) + [k for k in whatif if k not in stored]:
        s, w = stored.get(name, empty), whatif.get(name, empty)
        added, dropped = w & ~s, s & ~w
        if added.any() or dropped.any():
            out[name] = (added, dropped)
    return out


class SeriesPanel(QWidget):
    loaded = Signal()

    def __init__(self, reprocessor, runner, run, index=None, parent=None):
        super().__init__(parent)
        self.reprocessor, self.runner, self.run, self.index = reprocessor, runner, run, index
        self.boom = self.slot = None
        self.lo = self.hi = None  # the loaded slots, inclusive
        self.variables: list[str] = []
        self.hidden: set[str] = set()  # filled with the non-default series when the variables are first set
        self._defaults_applied = False
        self.context = ""
        self.result = None
        self.selected_tau: float | None = None
        self.whatif_rp = None  # a Reprocessor with the what-if QC, while one is applied
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
        self.blocks = QComboBox()
        self.blocks.setToolTip("the ladder's τ blocks, which tile each slot from its start")
        self.blocks.currentIndexChanged.connect(lambda *_: self.redraw())
        self.fluct = QCheckBox("fluctuations")
        self.fluct.toggled.connect(lambda *_: self.redraw())
        self.rug = QCheckBox("flag rug")
        self.rug.setToolTip("a row under the series for each test's flagged samples (and the what-if's differences)")
        self.rug.toggled.connect(lambda *_: self.redraw())
        self._fill_blocks()
        self.whatif_button = QPushButton("QC what-if…")
        self.whatif_button.setCheckable(True)
        self.whatif_button.setToolTip("edit the run's [qc] settings and reprocess these slots with them")
        self.whatif = QCWhatIfPanel(run.cfg) if run.cfg is not None and reprocessor is not None else None
        if self.whatif is None:
            self.whatif_button.setEnabled(False)
        else:
            self.whatif.hide()
            self.whatif_button.toggled.connect(self.whatif.setVisible)
            self.whatif.applied.connect(self.set_whatif)
            self.whatif.tauRequested.connect(self._whatif_tau)

        nav = QHBoxLayout()
        self.nav_buttons = {}
        for name, label, tip, handler in (
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
        for w in (self.mode, self.frame, self.wide, self.band, QLabel("τ blocks"), self.blocks, self.fluct, self.rug):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(self.whatif_button)
        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self.graphics)
        if self.whatif is not None:
            body.addWidget(self.whatif)
            body.setStretchFactor(0, 3)
            body.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(controls)
        layout.addLayout(nav)
        layout.addWidget(boxes)
        layout.addWidget(self.status)
        layout.addWidget(body, stretch=1)
        self.resize(1250, 900)
        self._decimators: list[PlotDecimator] = []
        self.plots: dict[str, pg.PlotItem] = {}

    @property
    def key(self) -> str:
        return f"series-{id(self)}"

    # --- what is shown ------------------------------------------------------------------------

    def set_focus(self, boom: int, slot: int, mode: str | None = None, variables: list[str] | None = None,
                  context: str | None = None, frame: str | None = None, show=()) -> None:
        """`show`: series to check whatever else is hidden (those behind a clicked value)."""
        """Show `slot` (keeping any neighbours loaded alongside, shifted with it)."""
        if self.slot is not None and self.boom == boom:
            span_lo, span_hi = self.slot - self.lo, self.hi - self.slot
        else:
            span_lo = span_hi = 0
        self.boom, self.slot, self.lo, self.hi = boom, slot, slot - span_lo, slot + span_hi
        if context is not None:
            self.context = context
        for combo, value, items in ((self.mode, mode, MODES), (self.frame, frame, FRAMES)):
            if value is not None:
                combo.blockSignals(True)
                combo.setCurrentIndex([k for _, k in items].index(value))
                combo.blockSignals(False)
        if variables is not None and not self._defaults_applied:
            self.hidden = {v for v in variables if v not in SHOWN_BY_DEFAULT}
            self._defaults_applied = True
        if show:
            self.hidden -= set(show)
        if variables is not None and (list(variables) != self.variables or show):
            self.variables = list(variables)
            self._build_series_boxes()
        self.reload()

    def set_selected_tau(self, tau_s: float | None) -> None:
        self.selected_tau = tau_s if tau_s is not None and np.isfinite(tau_s) else None
        self._fill_blocks()

    def _fill_blocks(self) -> None:
        current = self.blocks.currentData()
        self.blocks.blockSignals(True)
        self.blocks.clear()
        self.blocks.addItem("none", None)
        if self.selected_tau is not None and self.selected_tau <= 600:
            self.blocks.addItem(f"selected τ ({self.selected_tau:g} s)", "selected")
        for r in (9.375, 18.75, 37.5, 75.0, 150.0, 300.0, 600.0):
            self.blocks.addItem(f"{r:g} s", r)
        i = self.blocks.findData(current)
        self.blocks.setCurrentIndex(i if i >= 0 else 0)
        self.blocks.blockSignals(False)

    def block_rung(self) -> float | None:
        data = self.blocks.currentData()
        return self.selected_tau if data == "selected" else data

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

    def set_whatif(self, cfg) -> None:
        """Reprocess with a what-if QC config (None: back to the run's own)."""
        if self.whatif_rp is not None:
            self.whatif_rp.close()
        self.whatif_rp = qcwhatif.reprocessor(self.reprocessor, cfg) if cfg is not None else None
        if cfg is not None:
            self.rug.setChecked(True)  # where the what-if's flag differences are drawn
        self.reload()

    def reload(self) -> None:
        if self.boom is None:
            return
        mode = self.current_mode()
        self.frame.model().item(2).setEnabled(mode == "as_measured")  # the sonic frame exists only as measured
        g0, n = self.window_samples()
        self._update_header()
        self.status.setText("reprocessing from the raw files…" + (" (and with the what-if QC)" if self.whatif_rp else ""))
        self.runner.submit(compute_series, self.reprocessor, self.index, self.boom, g0, n, mode,
                           self.frame.currentData(), self.whatif_rp, key=self.key,
                           label=f"reprocessing boom {self.boom}", on_done=self._loaded, on_error=self._failed,
                           priority=10)

    def _compare_slots(self) -> None:
        rp, boom, slots = self.whatif_rp, self.boom, range(self.lo, self.hi + 1)
        self.runner.submit(qcwhatif.slot_comparison, self.index, rp, boom, slots, key=f"qcwhatif-{id(self)}",
                           label="comparing slot values",
                           on_done=lambda df: not self._closed and rp is self.whatif_rp and self.whatif.show_comparison(
                               df, lambda k: format_slot(k, self.run.timezone)),
                           on_error=lambda exc, tb: not self._closed and self.whatif.comparison_note.setText(
                               f"could not compare: {exc}"))

    def _whatif_tau(self) -> None:
        if self.whatif_rp is None or self.slot is None:
            return
        rp = self.whatif_rp
        self.whatif.tau_note.setText("recomputing the focus slot's spectra and τ…")
        self.runner.submit(qcwhatif.slot_tau, self.index, rp, self.boom, self.slot, self.run.cfg.secondary,
                           key=f"qcwhatif-tau-{id(self)}", label="recomputing τ with the what-if QC",
                           on_done=lambda df: not self._closed and rp is self.whatif_rp and self.whatif.show_tau(df),
                           on_error=lambda exc, tb: not self._closed and self.whatif.tau_note.setText(
                               f"could not recompute τ: {exc}"))

    def _update_header(self) -> None:
        tz = self.run.timezone
        span = "" if self.lo == self.hi else f"{self.hi - self.lo + 1} slots from {format_slot(self.lo, tz)[:16]} · "
        self.header.setText(span + f"<b>{self.mode.currentText()}</b>" + (f" · {self.context}" if self.context else ""))

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
        if result.get("whatif") is not None:
            notes.append("<b>QC what-if</b>: series and ghosts use the what-if settings; rug rows marked "
                         f"<span style='color:{WHATIF_ADDED}'>■ flagged only with the what-if</span> / "
                         f"<span style='color:{WHATIF_DROPPED}'>■ only with the run's</span>")
            self._compare_slots()
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.status.setText(" · ".join(notes))
        self.redraw()
        self.loaded.emit()

    # --- drawing -------------------------------------------------------------------------------

    def _enable_controls(self) -> None:
        """Grey out what the current view can't show."""
        as_measured = self.current_mode() == "as_measured"
        self.band.setEnabled(not as_measured)
        self.band.setToolTip("as measured has no despiking" if as_measured
                             else "the despike test's bounds: the running median ± z·MAD/0.6745")
        no_blocks = self.blocks.currentData() is None
        self.fluct.setEnabled(not no_blocks)
        self.fluct.setToolTip("choose τ blocks first" if no_blocks
                              else "subtract each τ block's mean (usable samples): what the ladder's variances see")

    def redraw(self) -> None:
        self._enable_controls()
        self.graphics.clear()
        self._decimators, self.plots = [], {}
        if self.result is None:
            return
        am, qc, mode, frame = self.result["am"], self.result["qc"], self.result["mode"], self.result["frame"]
        stored_qc = qc
        whatif = self.result.get("whatif")
        if whatif is not None and mode != "as_measured":
            qc = whatif
        base = qc if qc is not None else am
        x = sample_to_unix(base.g0 + np.arange(base.n))
        rename = {} if frame == "earth" else _EARTH_TO_STREAM
        available = set(am.series if mode in ("as_measured", "overlay") else ()) | set(qc.series if qc else ())
        names = [rename.get(v, v) for v in self.variables if v not in self.hidden]
        names = [v for v in names if v in available]

        rung = self.block_rung()
        blocks = tau_block_ids(base.g0, base.n, rung) if rung is not None else None
        fluct = blocks is not None and self.fluct.isChecked()
        if fluct:
            am, qc = _fluctuations(am, blocks), _fluctuations(qc, blocks)
        diffs = None
        if whatif is not None:
            if stored_qc is not None:
                stored_rows = rug_rows(self.result["removed"], stored_qc.filled)
                whatif_rows = rug_rows(whatif.removed, whatif.filled)
            else:  # as measured, the run's rug is its stored flags, which hold no fills or coupling
                stored_rows = rug_rows(self.result["removed"])
                whatif_rows = rug_rows({t: m for t, m in whatif.removed.items() if t != "coupled"})
            diffs = rug_differences(stored_rows, whatif_rows, base.n)

        first = None
        for row, var in enumerate(names):
            plot = self._new_plot(row, first)
            first = first or plot
            self.plots[var] = plot
            label, unit = variable_label(var)
            label = f"{label}'" if fluct else label
            plot.setLabel("left", f"{label} [{unit}]" if unit else label)
            self._shade_slots(plot)
            if blocks is not None:
                self._draw_blocks(plot, x, blocks)
            dec = PlotDecimator(plot)
            self._decimators.append(dec)
            self._draw_variable(plot, dec, var, x, am, qc, mode, _SERIES_COLORS[row % len(_SERIES_COLORS)])
            plot.getAxis("bottom").setStyle(showValues=False)

        if self.rug.isChecked():
            self._draw_rug(x, stored_qc, self.result["removed"], len(names), first, diffs)
        elif names:  # the last series carries the time axis instead
            last = self.plots[names[-1]]
            last.getAxis("bottom").setStyle(showValues=True)
            last.setLabel("bottom", f"time ({self.run.timezone})")
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

    def _draw_blocks(self, plot, x, blocks) -> None:
        inside = (x >= slot_to_unix(self.lo)) & (x < slot_to_unix(self.hi + 1))
        edges = np.flatnonzero(np.diff(blocks) != 0) + 1
        edges = edges[inside[edges]]
        if edges.size > 400:
            return  # too many to draw usefully
        for i in edges:
            line = pg.InfiniteLine(x[i], pen=pg.mkPen((90, 90, 160, 110), width=1))
            line.setZValue(-80)
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

    def _draw_rug(self, x, qc, removed, row, link, diffs=None):
        rug = self._new_plot(row, link)
        rug.setMouseEnabled(y=False)
        rug.setLabel("bottom", f"time ({self.run.timezone})")
        rows = rug_rows(removed, qc.filled if qc is not None else None)
        diffs = diffs or {}
        height = (max(len(rows), 1) + len(diffs)) * _RUG_ROW_PX + 45
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
        for name, (added, dropped) in diffs.items():
            i = len(ticks)
            for mask, color in ((added, WHATIF_ADDED), (dropped, WHATIF_DROPPED)):
                if not mask.any():
                    continue
                starts, ends = _intervals(mask)
                xs = np.empty(2 * starts.size)
                xs[0::2] = x[starts]
                xs[1::2] = x[np.minimum(ends, x.size) - 1] + 1 / SAMPLE_HZ
                rug.addItem(pg.PlotDataItem(xs, np.full(xs.size, i), pen=pg.mkPen(color, width=8), connect="pairs"))
            ticks.append((i, f"{name} what-if"))
        rug.getAxis("left").setTicks([ticks])
        rug.setYRange(-0.7, max(len(ticks), 1) - 0.3, padding=0)
        return rug

    def shutdown(self) -> None:
        self._closed = True
        if self.whatif_rp is not None:
            self.whatif_rp.close()


def _fluctuations(win: HFWindow | None, blocks: np.ndarray) -> HFWindow | None:
    """`win` with each τ block's mean (over its usable samples) subtracted from
    every series; the despike band shifts with it.
    """
    if win is None:
        return None
    out = HFWindow(**{**win.__dict__, "series": {}, "despike_band": {}})
    ids = blocks - blocks.min()
    nb = int(ids.max()) + 1
    for var, y in win.series.items():
        ok = np.isfinite(y) & win.masks.get(var, np.ones(y.size, bool))
        sums = np.bincount(ids[ok], weights=y[ok], minlength=nb)
        counts = np.bincount(ids[ok], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)
        out.series[var] = y - means[ids]
        if var in win.despike_band:
            lo, hi = win.despike_band[var]
            out.despike_band[var] = (lo - means[ids], hi - means[ids])
    return out


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


def compute_series(reprocessor, index, boom, g0, n, mode, frame, whatif_rp=None) -> dict:
    """Runs on a worker thread. As-measured mode needs no Stage B, so its flag
    rug comes from the run's stored flags instead. With `whatif_rp`, the same
    window QC'd with the what-if settings comes back as "whatif".
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
    whatif = None
    if whatif_rp is not None:
        whatif = whatif_rp.qc_applied(boom, g0, n, variant="mrd_unexcised" if mode == "unexcised" else "mrd")

    bearings, shown_frame = {}, "sonic" if sonic else "earth"
    if frame == "streamwise":
        means = _stored_slot_means(index, boom, g0, n) if index is not None else {}
        am, bearings = to_streamwise(am, means)
        if whatif is not None:
            whatif, _ = to_streamwise(whatif, means)
        if qc is not None:
            qc, bearings = to_streamwise(qc, means)
            removed = qc.removed
        else:
            removed = {test: {_EARTH_TO_STREAM.get(v, v): m for v, m in per_var.items()}
                       for test, per_var in removed.items()}
        shown_frame = "streamwise"
    return {"am": am, "qc": qc, "mode": mode, "frame": shown_frame, "missing": missing, "removed": removed,
            "bearings": bearings, "whatif": whatif}
