"""The jets in aggregate (what the events look like, when they happen, their
composite profile and direction) and the diagnostics for tuning the detection
(the status of every slot, and how the counts move as one setting sweeps its
range).
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.jets.analysis import JetResult
from ttu_tower.jets.events import EventParams
from ttu_tower.jets.identify import ALGORITHMS, Profiles, describe
from ttu_tower.viewer import diurnal, windrose
from ttu_tower.timegrid import slot_to_time
from ttu_tower.viewer.timeline import Curve
from ttu_tower.viewer.ui import tables
from ttu_tower.viewer.ui.jets_event import duration_text
from ttu_tower.viewer.ui.windrose_view import RosePlot

_BAR = (60, 110, 180, 170)
_SHARE_MAP = pg.colormap.get("viridis")
_THRESHOLD_PEN = pg.mkPen("#d62728", width=1.5, style=Qt.PenStyle.DashLine)
_STATUS_TEXT = {"no_data": "no data", "no_peak": "no peak", "too_slow": "too slow", "weak_falloff": "weak falloff",
                "jet": "jet"}
_PLOT_HEIGHT = 210
_NOSE_COLUMNS = ["n_detected", "peak_slot", "peak_boom", "z_nose_median", "z_nose_min", "z_nose_max", "u_nose_max",
                 "u_nose_mean", "z_est_median", "z_est_min", "z_est_max", "u_est_max", "u_est_mean", "falloff_max",
                 "falloff_mean", "wd_nose"]
# observed peaks or estimated noses: (event height, event speed, slot height, slot speed, words)
_KINDS = {"observed": ("z_nose_median", "u_nose_max", "z_nose", "u_nose", "observed peak"),
          "estimated": ("z_est_median", "u_est_max", "z_nose_est", "u_nose_est", "estimated nose")}


def nearest_boom(height: float) -> int:
    return min(HEIGHTS, key=lambda b: abs(HEIGHTS[b] - height))


def _plot(title: str, xlabel: str = "", unit: str = "", ylabel: str = "density") -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.setMinimumHeight(_PLOT_HEIGHT)
    w.setTitle(title, size="9pt")
    w.getPlotItem().export_title = title
    w.setLabel("bottom", xlabel, units=unit or None)
    w.setLabel("left", ylabel)
    w.showGrid(y=True, alpha=0.2)
    return w


def _histogram(plot: pg.PlotWidget, values, step: float, label: str, brush=_BAR, start=None,
               bins: int | None = None) -> np.ndarray:
    """A density histogram (it integrates to 1) in `step`-wide bins, or in
    `bins` equal bins spanning the values (from `start`, if lower).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([])
    if bins:
        lo = values.min() if start is None else min(start, values.min())
        hi = values.max()
        if hi <= lo:
            lo, hi = lo - 0.5, lo + 0.5
        edges = np.linspace(lo, hi, bins + 1)
        step = edges[1] - edges[0]
    else:
        lo = np.floor((values.min() if start is None else min(start, values.min())) / step) * step
        hi = (np.floor(values.max() / step) + 1) * step
        edges = np.arange(lo, hi + step / 2, step)
    density = np.histogram(values, edges)[0] / (values.size * step)
    item = pg.PlotDataItem(edges, density, stepMode="center", fillLevel=0, brush=brush, pen=pg.mkPen(brush[:3]))
    item.export_label = label
    plot.addItem(item)
    return edges


def _bars(plot: pg.PlotWidget, x, counts, labels, label: str, brush=_BAR) -> None:
    """Each category's fraction of the total."""
    counts = np.asarray(counts, float)
    item = pg.BarGraphItem(x=np.asarray(x, float), height=counts / max(counts.sum(), 1.0), width=0.8, brush=brush,
                           pen=pg.mkPen(brush[:3]))
    item.export_label = label
    plot.addItem(item)
    plot.getPlotItem().getAxis("bottom").setTicks([list(zip(x, labels))])


def _fit(plot: pg.PlotWidget) -> None:
    """Back to automatic ranges, around what's drawn now."""
    plot.enableAutoRange()
    plot.autoRange()


def _grid(widgets, columns: int = 2) -> QScrollArea:
    content = QWidget()
    grid = QGridLayout(content)
    for i, w in enumerate(widgets):
        grid.addWidget(w, i // columns, i % columns)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    return scroll


# --- aggregate ---------------------------------------------------------------------------------

class AggregateView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scope = QComboBox()
        self.scope.addItem("all events, one value each", "events")
        self.scope.addItem("the picked event, slot by slot", "event")
        self.scope.setToolTip("the events together, or the distributions within the event picked last "
                              "(in the Events list or on the jets band)")
        self.scope.currentIndexChanged.connect(lambda *_: self._on_scope())
        self.which = QComboBox()
        self.which.addItem("events the filters keep", True)
        self.which.addItem("all events", False)
        self.which.currentIndexChanged.connect(lambda *_: self.redraw())
        self.kind = QComboBox()
        self.kind.addItem("observed peaks", "observed")
        self.kind.addItem("estimated noses", "estimated")
        self.kind.setToolTip("heights and speeds of the observed peak (the fastest boom), or of the estimated "
                             "nose: the maximum of the quadratic through the peak and the booms just below and "
                             "above it")
        self.kind.currentIndexChanged.connect(lambda *_: self.redraw())
        self.normalize = QCheckBox("composite relative to each slot's peak speed")
        self.normalize.setToolTip("each slot's profile divided by its observed peak or estimated nose speed "
                                  "(as chosen beside)")
        self.normalize.toggled.connect(lambda *_: self.redraw())
        self.bins = QSpinBox()
        self.bins.setRange(0, 200)
        self.bins.setSpecialValueText("auto")
        self.bins.setToolTip("bins per histogram (speed, falloff, duration, estimated nose height): auto keeps "
                             "fixed widths (1 m/s, 0.5 m/s, 0.5 h, 10 m; half that slot by slot), a number splits "
                             "each one's range into that many equal bins")
        self.bins.setKeyboardTracking(False)  # redraw once a value is entered, not on every keystroke
        self.bins.valueChanged.connect(lambda *_: self.redraw())
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        head = QHBoxLayout()
        head.addWidget(self.scope)
        head.addWidget(self.which)
        head.addWidget(self.kind)
        head.addStretch(1)
        head2 = QHBoxLayout()
        head2.addWidget(self.normalize)
        head2.addWidget(QLabel("bins"))
        head2.addWidget(self.bins)
        head2.addStretch(1)
        self.height_plot = _plot("observed peak height (event median)", "height", "m", ylabel="fraction of events")
        self.speed_plot = _plot("max observed peak speed", "speed", "m/s", ylabel="density (per m/s)")
        self.falloff_plot = _plot("max falloff above the peak", "falloff", "m/s", ylabel="density (per m/s)")
        self.duration_plot = _plot("duration", "length", "h", ylabel="density (per h)")
        self.hour_plot = _plot("start hour", "hour", ylabel="fraction of events")
        self.share_plot = _plot("share of slots (with data) in a jet, by hour and month", "month", ylabel="hour")
        self.share_image = pg.ImageItem(axisOrder="row-major")
        self.share_plot.addItem(self.share_image)
        self.share_bar = pg.ColorBarItem(values=(0, 1), colorMap=_SHARE_MAP, label="%", interactive=False)
        self.share_bar.setImageItem(self.share_image, insert_in=self.share_plot.getPlotItem())
        self.share_plot.getPlotItem().getAxis("bottom").setTicks([[(m + 0.5, diurnal.MONTHS[m][0]) for m in range(12)]])
        self.share_plot.getPlotItem().getAxis("left").setTicks([[(h + 0.5, str(h)) for h in range(0, 24, 3)]])
        self.share_plot.setMouseEnabled(x=False, y=False)
        self.composite_plot = _plot("composite profile of jet slots (median, interquartile band)", "wind speed",
                                    "m/s", ylabel="height (m)")
        self.composite_plot.getPlotItem().export_layout = "per curve"
        self.rose_plot = RosePlot()
        self.rose_plot.setMinimumHeight(_PLOT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(head)
        layout.addLayout(head2)
        layout.addWidget(self.summary)
        layout.addWidget(_grid([self.height_plot, self.speed_plot, self.falloff_plot, self.duration_plot,
                                self.hour_plot, self.share_plot, self.composite_plot, self.rose_plot]), stretch=1)
        self.result: JetResult | None = None
        self.profiles: Profiles | None = None
        self.slot_range = None
        self.tz = "UTC"
        self.nose_from = "jets"
        self.event_id: int | None = None

    def sizeHint(self):
        return QSize(520, 600)

    def set_data(self, result, profiles, slot_range, tz, nose_from: str = "jets", event_id: int | None = None) -> None:
        self.result, self.profiles, self.slot_range, self.tz = result, profiles, slot_range, tz
        self.nose_from, self.event_id = nose_from, event_id
        self.redraw()

    def _on_scope(self) -> None:
        self.which.setEnabled(self.scope.currentData() == "events")
        self.redraw()

    def _titles(self, one_event: bool) -> None:
        unit = "slots" if one_event else "events"
        what = "of its slots" if one_event else "(event median)"
        words = _KINDS[self.kind.currentData()][4]
        self.height_plot.setTitle(f"{words} height {what}", size="9pt")
        self.height_plot.setLabel("left", f"fraction of {unit}" if words.startswith("observed") else "density (per m)")
        self.speed_plot.setTitle(f"{words} speed of its slots" if one_event else f"max {words} speed", size="9pt")
        self.falloff_plot.setTitle("falloff of its slots" if one_event else "max falloff above the peak", size="9pt")
        self.hour_plot.setTitle("hour of its slots" if one_event else "start hour", size="9pt")
        self.hour_plot.setLabel("left", f"fraction of {unit}")
        self.duration_plot.setTitle("duration", size="9pt")

    def redraw(self) -> None:
        plots = (self.height_plot, self.speed_plot, self.falloff_plot, self.duration_plot, self.hour_plot,
                 self.composite_plot)
        for plot in plots:
            plot.clear()
        self.share_image.clear()
        r, p = self.result, self.profiles
        if r is None or p is None:
            self.summary.setText("no detection yet")
            self.rose_plot.draw(None, "direction at the peak")
            return
        one_event = self.scope.currentData() == "event"
        self._titles(one_event)
        if one_event:
            self._one_event(r, p, plots)
            return
        events = r.events_in(self.slot_range, passed_only=self.which.currentData())
        if self.nose_from != "jets":
            events = events.drop(columns=_NOSE_COLUMNS).join(r.table(self.nose_from)[_NOSE_COLUMNS])
        detected = r.detected_in(events.index, self.nose_from)
        members = r.slots_in(events.index)
        where = "in the whole period" if self.slot_range is None else "starting in the visible range"
        plural = "s" if len(events) != 1 else ""
        noses = "jet slots" if self.nose_from == "jets" else "slots with a nose"
        self.summary.setText(f"{len(events)} event{plural} ({len(members)} slots, {len(detected)} {noses}) {where}")
        height, speed = _KINDS[self.kind.currentData()][:2]
        nosed = events[np.isfinite(events[height].to_numpy(float))]  # events with a peak to describe
        self._heights(p, nosed[height], "events")
        bins = self.bins.value() or None
        _histogram(self.speed_plot, nosed[speed], 1.0, "events", bins=bins)
        _histogram(self.falloff_plot, nosed["falloff_max"], 0.5, "events", start=0.0, bins=bins)
        _histogram(self.duration_plot, events["duration_min"] / 60.0, 0.5, "events", start=0.0, bins=bins)
        hours = events["start_time"].dt.tz_convert(self.tz).dt.hour if len(events) else pd.Series([], dtype=int)
        _bars(self.hour_plot, range(24), np.bincount(hours.to_numpy(int), minlength=24), [str(h) if h % 3 == 0 else ""
              for h in range(24)], "events")
        self.hour_plot.setLabel("bottom", f"hour ({self.tz})")
        self._share(r, detected if self.nose_from == "jets" else members.to_frame())
        self._composite(p, members, detected)
        self._rose(p, detected)
        for plot in plots:
            _fit(plot)

    def _one_event(self, r: JetResult, p: Profiles, plots) -> None:
        """The distributions within the picked event: one value per slot."""
        if self.event_id is None or self.event_id not in r.events.table.index:
            self.summary.setText("pick an event (in the Events list or on the jets band)")
            self.rose_plot.draw(None, "direction at the peak")
            return
        e = r.events.table.loc[self.event_id]
        detected = r.detected_in([self.event_id], self.nose_from)
        members = r.slots_in([self.event_id])
        noses = "jet slots" if self.nose_from == "jets" else "slots with a nose"
        start = e.start_time.tz_convert(self.tz)
        self.summary.setText(f"jet {self.event_id}, {start:%Y-%m-%d %H:%M}: {len(members)} slots, "
                             f"{len(detected)} {noses}")
        height, speed = _KINDS[self.kind.currentData()][2:4]
        self._heights(p, detected[height], "slots")
        bins = self.bins.value() or None
        _histogram(self.speed_plot, detected[speed], 0.5, "slots", bins=bins)
        _histogram(self.falloff_plot, detected["falloff_abs"], 0.25, "slots", start=0.0, bins=bins)
        hours = pd.DatetimeIndex(slot_to_time(members.index.to_numpy())).tz_convert(self.tz).hour
        _bars(self.hour_plot, range(24), np.bincount(np.asarray(hours, int), minlength=24),
              [str(h) if h % 3 == 0 else "" for h in range(24)], "slots")
        self.hour_plot.setLabel("bottom", f"hour ({self.tz})")
        self.duration_plot.setTitle(f"duration: one event, {duration_text(int(e.duration_min))} h", size="9pt")
        self.share_plot.setTitle("share by hour and month: all events only", size="9pt")
        self._composite(p, members, detected)
        self._rose(p, detected)
        for plot in plots:
            _fit(plot)

    def _heights(self, p: Profiles, heights: pd.Series, label: str) -> None:
        """Observed peaks sit at booms: a bar each. Estimated noses lie between them: a histogram."""
        heights = heights.dropna()
        if self.kind.currentData() == "observed":
            at = [nearest_boom(z) for z in heights]
            booms = list(p.booms)
            _bars(self.height_plot, range(len(booms)), [at.count(b) for b in booms],
                  [f"{HEIGHTS[b]:g}" for b in booms], label)
        else:
            self.height_plot.getPlotItem().getAxis("bottom").setTicks(None)
            _histogram(self.height_plot, heights, 10.0, label, start=0.0, bins=self.bins.value() or None)

    def _share(self, r: JetResult, detected: pd.DataFrame) -> None:
        self.share_plot.setTitle("share of slots (with data) in a jet, by hour and month", size="9pt")
        detection = r.detection
        if self.slot_range is not None:
            detection = detection.loc[self.slot_range[0]:self.slot_range[1] - 1]
        y = np.where(detection["status"] == "no_data", np.nan, 0.0)
        y[np.isin(detection.index.to_numpy(), detected.index.to_numpy())] = 1.0
        values, counts = diurnal.grid(Curve(detection.index.to_numpy(np.int64), y), self.tz, stat="mean")
        share = 100.0 * values
        top = float(np.nanmax(share)) if np.isfinite(share).any() else 1.0
        self.share_bar.setLevels((0.0, max(top, 0.1)))
        self.share_image.setImage(np.nan_to_num(share), autoLevels=False, levels=(0.0, max(top, 0.1)))
        self.share_image.setRect(0, 0, 12, 24)
        self.share_plot.setRange(xRange=(0, 12), yRange=(0, 24), padding=0)
        self.share_plot.setToolTip(f"hours in {self.tz}; slots with data per cell: {counts.min()}–{counts.max()}")

    def _composite(self, p: Profiles, members: pd.Series, detected: pd.DataFrame) -> None:
        """Every slot of the events; relative to the nose speed, only slots that have a nose."""
        relative = self.normalize.isChecked()
        slots = detected.index if relative else members.index
        if len(slots) == 0:
            return
        rows = np.searchsorted(p.slots, slots.to_numpy())
        ws = p.ws[rows]
        words = _KINDS[self.kind.currentData()][4]
        if relative:
            ws = ws / detected[_KINDS[self.kind.currentData()][3]].to_numpy(float)[:, None]
        with np.errstate(all="ignore"):
            q25, q50, q75 = (np.nanpercentile(ws, q, axis=0) for q in (25, 50, 75))
        z = p.heights
        ok = np.isfinite(q50)
        low = pg.PlotDataItem(q25[ok], z[ok], pen=pg.mkPen((60, 110, 180, 90)))
        high = pg.PlotDataItem(q75[ok], z[ok], pen=pg.mkPen((60, 110, 180, 90)))
        low.export_label, high.export_label = "25th percentile", "75th percentile"
        band = pg.FillBetweenItem(low, high, brush=(60, 110, 180, 50))
        for item in (low, high, band):
            self.composite_plot.addItem(item)
        median = self.composite_plot.plot(q50[ok], z[ok], pen=pg.mkPen("#1f3d7a", width=2), symbol="o",
                                          symbolSize=5, symbolBrush="#1f3d7a")
        median.export_label = "median"
        self.composite_plot.setLabel("bottom", f"wind speed / {words} speed" if relative else "wind speed",
                                     units=None if relative else "m/s")
        self.composite_plot.setTitle(f"composite profile of {len(slots)} slots (median, interquartile band)",
                                     size="9pt")

    def _rose(self, p: Profiles, detected: pd.DataFrame) -> None:
        wd = p.extra.get("wd")
        if detected.empty or wd is None:
            self.rose_plot.draw(None, "direction at the peak")
            return
        rows = np.searchsorted(p.slots, detected.index.to_numpy())
        cols = [p.column(int(b)) for b in detected["nose_boom"]]
        direction, speed = wd[rows, cols], detected["u_nose"].to_numpy(float)
        ok = np.isfinite(direction) & np.isfinite(speed)
        table = windrose.rose(speed[ok], direction[ok], sectors=16) if ok.any() else None
        self.rose_plot.draw(table, f"direction and speed at the peak, {int(ok.sum())} slots")


# --- diagnostics -------------------------------------------------------------------------------

def sweepable(settings) -> list[tuple[str, str, dict]]:
    """(part, field name, field) of every numeric setting: the algorithm's, then the events'."""
    out = [("params", f["name"], f) for f in describe(ALGORITHMS[settings.algorithm].params)
           if f["kind"] in ("float", "optional_float", "int")]
    return out + [("events", f["name"], f) for f in describe(EventParams) if f["kind"] == "int"]


class DiagnosticsView(QWidget):
    sweepRequested = Signal(str, str)  # part, field name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = tables.new_table()
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.sweep_choice = QComboBox()
        self.sweep_choice.currentIndexChanged.connect(lambda *_: self._request_sweep())
        self.sweep_note = QLabel()
        self.sweep_note.setWordWrap(True)
        self.slots_plot = _plot("jet slots as the setting varies", ylabel="jet slots")
        self.events_plot = _plot("events (and those kept) as the setting varies", ylabel="events")
        self.events_plot.addLegend(offset=(-10, 10))
        sweep_head = QHBoxLayout()
        sweep_head.addWidget(QLabel("sweep"))
        sweep_head.addWidget(self.sweep_choice, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.summary)
        layout.addWidget(self.status)
        layout.addLayout(sweep_head)
        layout.addWidget(self.sweep_note)
        layout.addWidget(_grid([self.slots_plot, self.events_plot]), stretch=1)
        self.result: JetResult | None = None
        self.slot_range = None
        self._choices: list[tuple[str, str, dict]] = []

    def sizeHint(self):
        return QSize(520, 600)

    def set_data(self, result: JetResult | None, slot_range) -> None:
        self.result, self.slot_range = result, slot_range
        self._fill_choices()
        self.redraw()
        self._request_sweep()

    def _fill_choices(self) -> None:
        if self.result is None:
            return
        choices = sweepable(self.result.settings)
        if [(a, b) for a, b, _ in choices] == [(a, b) for a, b, _ in self._choices]:
            return
        current = self.sweep_choice.currentData()
        self._choices = choices
        self.sweep_choice.blockSignals(True)
        self.sweep_choice.clear()
        for part, name, f in choices:
            unit = f" ({f['unit']})" if f.get("unit") else ""
            self.sweep_choice.addItem(f"{f['label']}{unit}", f"{part}:{name}")
        i = self.sweep_choice.findData(current)
        self.sweep_choice.setCurrentIndex(i if i >= 0 else 0)
        self.sweep_choice.blockSignals(False)

    def _request_sweep(self) -> None:
        data = self.sweep_choice.currentData()
        if self.result is not None and data:
            part, name = data.split(":", 1)
            self.sweep_note.setText("sweeping…")
            self.sweepRequested.emit(part, name)

    def current(self, part: str, name: str):
        s = self.result.settings
        return getattr(s.algorithm_params() if part == "params" else s.events, name)

    def redraw(self) -> None:
        r = self.result
        if r is None:
            self.status.setRowCount(0)
            self.summary.setText("no detection yet")
            return
        detection = r.detection
        if self.slot_range is not None:
            detection = detection.loc[self.slot_range[0]:self.slot_range[1] - 1]
        counts = detection["status"].value_counts(sort=False)
        total = int(counts.sum())
        frame = pd.DataFrame({"slots": counts.to_numpy(), "% of slots": 100.0 * counts.to_numpy() / max(total, 1)},
                             index=[_STATUS_TEXT.get(str(s), str(s)) for s in counts.index])
        tables.fill(self.status, frame, index=True, fmt=lambda col, v: f"{v:.2f}" if col == "% of slots" else None)
        tables.fit_rows(self.status, max_rows=len(frame))
        where = "the whole period" if self.slot_range is None else "the visible range"
        self.summary.setText(f"{total:,} slots in {where}, by the first rule each slot fails")

    def set_sweep(self, part: str, name: str, frame: pd.DataFrame) -> None:
        if self.result is None or self.sweep_choice.currentData() != f"{part}:{name}":
            return
        f = dict(next(f for p, n, f in self._choices if (p, n) == (part, name)))
        label = f"{f['label']}" + (f" ({f['unit']})" if f.get("unit") else "")
        current = self.current(part, name)
        for plot in (self.slots_plot, self.events_plot):
            plot.clear()
            plot.setLabel("bottom", label)
        if self.events_plot.getPlotItem().legend is not None:
            self.events_plot.getPlotItem().legend.clear()
        x = frame["value"].to_numpy(float)
        a = self.slots_plot.plot(x, frame["jet_slots"].to_numpy(float), pen=pg.mkPen("#d62728", width=2), symbol="o",
                                 symbolSize=4)
        a.export_label = "jet slots"
        b = self.events_plot.plot(x, frame["events"].to_numpy(float), pen=pg.mkPen("#3c6eb4", width=2), symbol="o",
                                  symbolSize=4, name="events")
        c = self.events_plot.plot(x, frame["kept"].to_numpy(float), pen=pg.mkPen("#2ca02c", width=2), symbol="s",
                                  symbolSize=4, name="kept by the filters")
        b.export_label, c.export_label = "events", "kept by the filters"
        for plot in (self.slots_plot, self.events_plot):
            if current is not None:
                plot.addItem(pg.InfiniteLine(current, pen=_THRESHOLD_PEN))
            _fit(plot)
        where = "the whole period" if self.slot_range is None else "the visible range"
        note = f"the rest as set; counts over {where}, without manual edits; dashed: the current value"
        if current is None:
            note += ". This threshold is off now: the sweep switches it on"
        self.sweep_note.setText(note)


def sweep_field(settings, part: str, name: str) -> dict:
    fields = describe(ALGORITHMS[settings.algorithm].params) if part == "params" else describe(EventParams)
    return next(f for f in fields if f["name"] == name)
