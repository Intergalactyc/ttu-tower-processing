"""One jet event: a time-height map of the wind speed around it (observed
peaks marked, the event outlined), two profile panels of each of its slots (any of
wind speed, VPT, direction, TI, TKE, σw, wd std and the fluxes), and its slots'
status; and the manual edits: new boundaries (dragged on the map), reject, undo.
"""
import warnings

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.jets.analysis import JetResult
from ttu_tower.jets.identify import Profiles
from ttu_tower.timegrid import slot_to_time
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, display_offset_s, slot_to_unix, unix_to_slot
from ttu_tower.viewer.ui import tables

_SPEED_MAP = pg.colormap.get("viridis")
_TIME_MAP = pg.colormap.get("plasma")
_PAD_SLOTS = 18  # at least 3 h of context either side
_EDIT_PAD_SLOTS = 72  # 12 h while editing, room to drag the boundaries out
_SOURCE_TEXT = {"edited": "boundaries edited by hand", "manual": "made by hand"}
# key -> (label, unit, what boom_final holds it as: (variable, stat, variant); None: in the profiles)
PROFILE_OPTIONS = {
    "ws": ("wind speed", "m/s", None),
    "vpt": ("VPT", "K", ("vpt", None, "none")),
    "wd": ("direction (FROM)", "°", None),
    "ti": ("TI", "", ("ti", None, "mrd")),
    "tke": ("TKE", "m²/s²", ("tke", None, "mrd")),
    "sigma_w": ("σw", "m/s", ("sigma_w", None, "mrd")),
    "wd_std": ("wd std", "°", ("wd", "std", "mrd")),
    "uw": ("w'u'", "m²/s²", ("uw", "cov", "mrd")),
    "vw": ("w'v'", "m²/s²", ("vw", "cov", "mrd")),
    "wvpts": ("w'θv'", "K m/s", ("wvpts", "cov", "mrd")),
    "zeta": ("ζ = z/L", "", ("zeta", None, "mrd")),
}
PANEL_DEFAULTS = ("ws", "vpt")
_ESTIMATE = "#d62728"  # the estimated nose
PANEL_MODES = (("slots", "every slot"), ("mean", "mean ± SD"), ("median", "median, IQR"))
_SUMMARY = "#1f3d7a"


def summarize(rows: np.ndarray, how: str, circular: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per boom (column) over the slots (rows): (centre, low, high) - the mean
    and mean ± SD, or the median and quartiles. A direction is taken relative
    to its circular mean, so the spread doesn't break at north.
    """
    rows = np.asarray(rows, dtype=float)
    shift = np.zeros(rows.shape[1])
    if circular:
        with np.errstate(invalid="ignore"):
            rad = np.deg2rad(rows)
            shift = np.rad2deg(np.arctan2(np.nanmean(np.sin(rad), axis=0), np.nanmean(np.cos(rad), axis=0)))
        rows = (rows - shift + 180.0) % 360.0 - 180.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a boom with no value in any slot
        if how == "mean":
            centre = np.nanmean(rows, axis=0)
            n = np.sum(np.isfinite(rows), axis=0)
            spread = np.where(n > 1, np.nanstd(rows, axis=0, ddof=1), np.nan)
            low, high = centre - spread, centre + spread
        else:
            low, centre, high = np.nanpercentile(rows, [25, 50, 75], axis=0)
    if circular:  # back to bearings, unwrapped up the profile so it doesn't jump at north
        absolute = (centre + shift) % 360.0
        previous = None
        for j in range(absolute.size):
            if not np.isfinite(absolute[j]):
                continue
            if previous is not None:
                absolute[j] += 360.0 * np.round((previous - absolute[j]) / 360.0)
            previous = absolute[j]
        low, high = absolute + (low - centre), absolute + (high - centre)
        centre = absolute
    return centre, low, high
_STATUS_TEXT = {"no_data": "no data", "no_peak": "no peak", "too_slow": "too slow", "weak_falloff": "weak falloff",
                "jet": "jet"}


def duration_text(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}"


def event_label(result: JetResult, event_id: int, tz: str) -> str:
    e = result.events.table.loc[event_id]
    if np.isfinite(e.u_nose_max):
        est_z = f" (est. nose {e.z_est_median:.0f} m)" if np.isfinite(e.z_est_median) else ""
        est_u = f" (est. {e.u_est_max:.1f})" if np.isfinite(e.u_est_max) else ""
        nose = (f"peak at {e.z_nose_median:g} m{est_z}, up to {e.u_nose_max:.1f} m/s{est_u}, "
                f"falloff up to {e.falloff_max:.1f} m/s")
    else:
        nose = "no jet slot"
    if e.rejected:
        verdict = " (rejected by hand)"
    else:
        verdict = "" if result.passed.loc[event_id] else " (removed by the filters)"
    source = f" [{_SOURCE_TEXT[e.source]}]" if e.source in _SOURCE_TEXT else ""
    return (f"jet {event_id}: {e.start_time.tz_convert(tz):%Y-%m-%d %H:%M}, {duration_text(int(e.duration_min))} h, "
            f"{nose}{verdict}{source}")


def _time_color(i: int, n: int):
    return _TIME_MAP.map(0.85 * i / max(n - 1, 1), mode="qcolor")


class _ProfilePlot(pg.PlotWidget):
    """A quantity against height: one curve per slot coloured by time, or the
    event's mean ± SD or median and interquartile range.
    """

    def __init__(self, xlabel: str, unit: str):
        super().__init__()
        self.setLabel("left", "height", units="m")
        self.set_quantity(xlabel, unit)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMinimumSize(160, 160)
        self.plot_item = self.getPlotItem()
        self.plot_item.export_layout = "per curve"

    def set_quantity(self, label: str, unit: str) -> None:
        self.setLabel("bottom", label, units=unit or None)

    def draw(self, heights: np.ndarray, rows: np.ndarray, times: list[str], filled: np.ndarray, noses=None,
             title: str = "", how: str = "slots", circular: bool = False, estimates=None) -> None:
        """`noses`, `estimates`: (height, speed) per slot of the observed peaks
        (stars) and the estimated noses (diamonds), NaN where none.
        """
        self.clear()
        self.setTitle(title, size="9pt")
        if how != "slots":
            self._summary(heights, rows, how, circular)
        else:
            self._slots(heights, rows, times, filled)
        if noses is not None:
            ok = np.isfinite(noses[:, 0])
            if ok.any():
                marks = pg.ScatterPlotItem(noses[ok, 1], noses[ok, 0], symbol="star", size=13,
                                           brush=pg.mkBrush("#ffffff"), pen=pg.mkPen("#000000", width=1))
                marks.export_skip = True
                self.addItem(marks)
        if estimates is not None:
            ok = np.isfinite(estimates[:, 0])
            if ok.any():
                marks = pg.ScatterPlotItem(estimates[ok, 1], estimates[ok, 0], symbol="d", size=8,
                                           brush=pg.mkBrush(_ESTIMATE), pen=pg.mkPen("#000000", width=0.5))
                marks.export_skip = True
                self.addItem(marks)
        self.enableAutoRange()
        self.autoRange()

    def _summary(self, heights, rows, how: str, circular: bool) -> None:
        centre, low, high = summarize(rows, how, circular)
        ok = np.isfinite(centre)
        if not ok.any():
            return
        spread = ok & np.isfinite(low) & np.isfinite(high)
        names = ("mean − SD", "mean + SD") if how == "mean" else ("25th percentile", "75th percentile")
        if spread.sum() > 1:
            lo = pg.PlotDataItem(low[spread], heights[spread], pen=pg.mkPen((60, 110, 180, 90)))
            hi = pg.PlotDataItem(high[spread], heights[spread], pen=pg.mkPen((60, 110, 180, 90)))
            lo.export_label, hi.export_label = names
            for item in (lo, hi, pg.FillBetweenItem(lo, hi, brush=(60, 110, 180, 50))):
                self.addItem(item)
        line = self.plot(centre[ok], heights[ok], pen=pg.mkPen(_SUMMARY, width=2), symbol="o", symbolSize=5,
                         symbolBrush=_SUMMARY, symbolPen=None)
        line.export_label = "mean" if how == "mean" else "median"

    def _slots(self, heights, rows, times, filled) -> None:
        n = len(rows)
        for i, y in enumerate(rows):
            ok = np.isfinite(y)
            if not ok.any():
                continue
            color = _time_color(i, n)
            pen = pg.mkPen(color, width=1.4, style=Qt.PenStyle.DashLine if filled[i] else Qt.PenStyle.SolidLine)
            item = self.plot(y[ok], heights[ok], pen=pen, symbol="o", symbolSize=4, symbolBrush=color,
                             symbolPen=None)
            item.export_label = times[i] + (" (bridged)" if filled[i] else "")


class EventView(QWidget):
    stepRequested = Signal(int)  # -1 earlier event, +1 later
    inspectRequested = Signal(int, int)  # slot, boom
    valuesWanted = Signal(str)  # a PROFILE_OPTIONS key whose values aren't loaded yet
    spanEdited = Signal(int, int, int)  # event id, first slot, last slot
    selectRequested = Signal(int)  # brush this event's slots
    rejectToggled = Signal(int)
    revertRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("pick an event in the Events list or on the jets band")
        self.label.setWordWrap(True)
        self.prev_button, self.next_button = QPushButton("◀ event"), QPushButton("event ▶")
        self.prev_button.clicked.connect(lambda: self.stepRequested.emit(-1))
        self.next_button.clicked.connect(lambda: self.stepRequested.emit(1))
        self.select_button = QPushButton("Select event slots")
        self.select_button.setToolTip("brush every slot of this event (bridged and hand-added ones too), so the "
                                      "timelines and Summary tabs show it")
        self.select_button.clicked.connect(
            lambda: self.event_id is not None and self.selectRequested.emit(self.event_id))
        head = QHBoxLayout()
        head.addWidget(self.prev_button)
        head.addWidget(self.next_button)
        head.addWidget(self.label, stretch=1)
        head.addWidget(self.select_button)

        self.edit_button = QPushButton("Edit boundaries")
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip("drag the event's edges on the map (12 h either side shown), then Save")
        self.save_button, self.cancel_button = QPushButton("Save boundaries"), QPushButton("Cancel")
        self.reject_button = QPushButton("Reject")
        self.reject_button.setCheckable(True)
        self.reject_button.setToolTip("not a jet: the event stays listed but drops out of the kept events")
        self.revert_button = QPushButton("Undo manual edits")
        self.revert_button.setToolTip("remove every manual edit of this event")
        self.edit_note = QLabel()
        self.edit_button.toggled.connect(self._on_edit)
        self.save_button.clicked.connect(self._save_span)
        self.cancel_button.clicked.connect(lambda: self.edit_button.setChecked(False))
        self.reject_button.clicked.connect(lambda: self.event_id is not None and self.rejectToggled.emit(self.event_id))
        self.revert_button.clicked.connect(
            lambda: self.event_id is not None and self.revertRequested.emit(self.event_id))
        edits = QHBoxLayout()
        for w in (self.edit_button, self.save_button, self.cancel_button, self.reject_button, self.revert_button):
            edits.addWidget(w)
        edits.addWidget(self.edit_note, stretch=1)
        self.save_button.hide()
        self.cancel_button.hide()

        self.axis = pg.DateAxisItem(orientation="bottom", utcOffset=0)
        self.map = pg.PlotWidget(axisItems={"bottom": self.axis})
        self.map.setMinimumHeight(170)
        self.map.setMouseEnabled(x=True, y=False)
        self.map.getPlotItem().setLabel("left", "height", units="m")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.map.addItem(self.image)
        self.colorbar = pg.ColorBarItem(values=(0, 20), colorMap=_SPEED_MAP, label="wind speed (m/s)",
                                        interactive=False)
        self.colorbar.setImageItem(self.image, insert_in=self.map.getPlotItem())
        self.span = pg.LinearRegionItem(movable=False, brush=(0, 0, 0, 0), pen=pg.mkPen("#e6007e", width=2))
        self.span.sigRegionChangeFinished.connect(self._snap)
        self.map.addItem(self.span)
        self.noses = pg.ScatterPlotItem(symbol="star", size=11, brush=pg.mkBrush("#ffffff"),
                                        pen=pg.mkPen("#000000", width=1))
        self.bridged = pg.ScatterPlotItem(symbol="x", size=9, brush=pg.mkBrush("#ffffff"), pen=pg.mkPen("#ffffff"))
        for item in (self.noses, self.bridged):
            item.export_skip = True
            self.map.addItem(item)
        self.map.scene().sigMouseClicked.connect(self._on_map_click)
        self.map.setToolTip("double-click a cell to open that slot and boom in the Slot Inspector")

        self.key = QLabel()
        self.key.setWordWrap(True)
        self.choices, self.modes, self.panels = [], [], []
        profiles = QWidget()
        column = QVBoxLayout(profiles)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.key)
        row = QHBoxLayout()
        for default in PANEL_DEFAULTS:
            choice = QComboBox()
            for key, (label, unit, source) in PROFILE_OPTIONS.items():
                choice.addItem(label, key)
                if source is not None and source[2] == "mrd":
                    choice.setItemData(choice.count() - 1, "at the selected τ (mrd)", Qt.ItemDataRole.ToolTipRole)
            choice.setCurrentIndex(choice.findData(default))
            choice.currentIndexChanged.connect(lambda *_: self.redraw())
            mode = QComboBox()
            for key, label in PANEL_MODES:
                mode.addItem(label, key)
            mode.setToolTip("every slot of the event, or the whole event summarized per boom")
            mode.currentIndexChanged.connect(lambda *_: self.redraw())
            label, unit, _ = PROFILE_OPTIONS[default]
            panel = _ProfilePlot(label, unit)
            pick = QHBoxLayout()
            pick.addWidget(choice, stretch=1)
            pick.addWidget(mode)
            box = QVBoxLayout()
            box.addLayout(pick)
            box.addWidget(panel, stretch=1)
            row.addLayout(box)
            self.choices.append(choice)
            self.modes.append(mode)
            self.panels.append(panel)
        column.addLayout(row, stretch=1)

        self.table = tables.new_table()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.map)
        splitter.addWidget(profiles)
        splitter.addWidget(self.table)
        splitter.setSizes([220, 300, 160])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(head)
        layout.addLayout(edits)
        layout.addWidget(splitter, stretch=1)

        self.result: JetResult | None = None
        self.profiles: Profiles | None = None
        self.event_id: int | None = None
        self.tz = "UTC"
        self.values: dict[str, pd.DataFrame] = {}  # PROFILE_OPTIONS key -> slots x booms, loaded on demand
        self.wanted: set[str] = set()
        self.rows: list[int] = []  # the booms of the map's rows, bottom first
        self.nose_from = "jets"  # or "candidates": every slot's best candidate counts
        self.editing = False

    def sizeHint(self):
        return QSize(520, 600)

    def set_event(self, result: JetResult | None, profiles: Profiles | None, event_id: int | None, tz: str,
                  nose_from: str = "jets") -> None:
        self.result, self.profiles, self.tz, self.nose_from = result, profiles, tz, nose_from
        self.event_id = event_id if result is not None and event_id in result.events.table.index else None
        if self.editing:
            self.edit_button.setChecked(False)  # also redraws
        else:
            self.redraw()

    # --- editing ---------------------------------------------------------------------------------

    def _on_edit(self, on: bool) -> None:
        self.editing = on and self.event_id is not None
        self.span.setMovable(self.editing)
        self.span.setBrush(pg.mkBrush(230, 0, 126, 40) if self.editing else pg.mkBrush(0, 0, 0, 0))
        for w in (self.save_button, self.cancel_button):
            w.setVisible(self.editing)
        self.redraw()

    def _span_slots(self) -> tuple[int, int]:
        x0, x1 = self.span.getRegion()
        first = int(round(x0 / SLOT_SECONDS)) * SLOT_SECONDS
        last = int(round(x1 / SLOT_SECONDS)) * SLOT_SECONDS
        a = int(unix_to_slot(first + 1))
        return a, max(int(unix_to_slot(last - 1)), a)

    def _snap(self) -> None:
        if not self.editing:
            return
        a, b = self._span_slots()
        self.span.blockSignals(True)
        self.span.setRegion((slot_to_unix(a), slot_to_unix(b + 1)))
        self.span.blockSignals(False)
        times = slot_to_time(np.array([a, b + 1])).tz_convert(self.tz)
        self.edit_note.setText(f"new: {times[0]:%m-%d %H:%M} → {times[1]:%m-%d %H:%M}, "
                               f"{duration_text(10 * (b - a + 1))} h")

    def _save_span(self) -> None:
        if self.event_id is None:
            return
        a, b = self._span_slots()
        event_id = self.event_id
        self.edit_button.setChecked(False)
        self.spanEdited.emit(event_id, a, b)

    def set_values(self, key: str, frame: pd.DataFrame | None) -> None:
        """A quantity's per-slot, per-boom values (None: forget them all, e.g. for another run)."""
        if frame is None:
            self.values, self.wanted = {}, set()
            return
        self.values[key] = frame
        self.wanted.discard(key)
        if key in (c.currentData() for c in self.choices):
            self.redraw()

    def _profile_rows(self, key: str, slots: np.ndarray, index: np.ndarray, p: Profiles):
        """(slots x booms values, or None while loading)."""
        if key == "ws":
            return p.ws[index]
        if key == "wd":
            wd = p.extra.get("wd")
            return wd[index] if wd is not None else np.full((len(index), len(p.booms)), np.nan)
        if key not in self.values:
            if key not in self.wanted:
                self.wanted.add(key)
                self.valuesWanted.emit(key)
            return None
        return self.values[key].reindex(index=slots, columns=list(p.booms)).to_numpy(float)

    def redraw(self) -> None:
        r, p, event_id = self.result, self.profiles, self.event_id
        if r is None or p is None or event_id is None:
            for plot in self.panels:
                plot.clear()
            self.image.clear()
            self.noses.setData([], [])
            self.bridged.setData([], [])
            self.span.hide()
            self.table.setRowCount(0)
            self.label.setText("pick an event in the Events list or on the jets band")
            for w in (self.edit_button, self.reject_button, self.revert_button, self.select_button):
                w.setEnabled(False)
            self.edit_note.setText("")
            return
        self.label.setText(event_label(r, event_id, self.tz))
        e = r.events.table.loc[event_id]
        a, b = int(e.start_slot), int(e.end_slot)
        for w in (self.edit_button, self.reject_button, self.select_button):
            w.setEnabled(True)
        self.revert_button.setEnabled(bool(e.edits))
        self.reject_button.setChecked(bool(e.rejected))
        self.reject_button.setText("Rejected (undo)" if e.rejected else "Reject")
        if not self.editing:
            self.edit_note.setText(f"manual: {_SOURCE_TEXT[e.source]}" if e.source in _SOURCE_TEXT else "")
        pad = max(_EDIT_PAD_SLOTS if self.editing else _PAD_SLOTS, b - a + 1)
        lo, hi = max(a - pad, int(p.slots[0])), min(b + pad, int(p.slots[-1]))
        window = slice(lo - int(p.slots[0]), hi - int(p.slots[0]) + 1)
        ws = p.ws[window]
        self.rows = list(p.booms)
        self.map.getPlotItem().export_tz = self.tz
        self.axis.utcOffset = display_offset_s(self.tz)
        self.axis.picture = None
        self.axis.update()

        finite = ws[np.isfinite(ws)]
        top = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        self.colorbar.setLevels((0.0, max(top, 1.0)))
        self.image.setImage(ws.T, autoLevels=False, levels=(0.0, max(top, 1.0)))
        self.image.setRect(slot_to_unix(lo), 0, (hi - lo + 1) * SLOT_SECONDS, len(self.rows))
        ticks = [(i + 0.5, f"{HEIGHTS[bm]:g}") for i, bm in enumerate(self.rows)]
        self.map.getPlotItem().getAxis("left").setTicks([ticks])
        self.map.setYRange(0, len(self.rows), padding=0)
        self.map.setXRange(slot_to_unix(lo), slot_to_unix(hi + 1), padding=0)
        self.span.setRegion((slot_to_unix(a), slot_to_unix(b + 1)))
        self.span.show()

        slots = np.arange(a, b + 1)
        detection = r.detection.loc[a:b]
        filled = r.events.slots.loc[a:b, "filled"].to_numpy(bool)
        added = r.events.slots.loc[a:b, "manual"].to_numpy(bool)
        if self.nose_from == "jets":
            detected = detection["is_jet"].to_numpy(bool)
        else:
            detected = detection["u_nose"].notna().to_numpy(bool)
        row_of = {bm: i for i, bm in enumerate(self.rows)}
        boom = detection["nose_boom"]
        x = slot_to_unix(slots) + SLOT_SECONDS / 2
        y = np.array([row_of.get(int(v), np.nan) + 0.5 if pd.notna(v) else np.nan for v in boom])
        self.noses.setData(x[detected], y[detected])
        self.bridged.setData(x[filled], np.full(filled.sum(), len(self.rows) - 0.5))

        times = [f"{t:%H:%M}" for t in slot_to_time(slots).tz_convert(self.tz)]
        index = slots - int(p.slots[0])
        def at_peaks(z: str, u: str) -> np.ndarray:
            return np.column_stack([np.where(detected, detection[z].to_numpy(float), np.nan),
                                    np.where(detected, detection[u].to_numpy(float), np.nan)])
        noses = at_peaks("z_nose", "u_nose")
        estimates = at_peaks("z_nose_est", "u_nose_est") if "z_nose_est" in detection else None
        first, last = _time_color(0, len(times)).name(), _time_color(len(times) - 1, len(times)).name()
        peaks = "jet slots'" if self.nose_from == "jets" else "slots'"
        self.key.setText(f"each slot of the event, <span style='color:{first}'>■</span> {times[0]} → "
                         f"<span style='color:{last}'>■</span> {times[-1]} ({self.tz}); ☆ {peaks} observed peak, "
                         f"<span style='color:{_ESTIMATE}'>◆</span> estimated nose; dashed: bridged or added by hand")
        dashed = filled | added
        for choice, mode, panel in zip(self.choices, self.modes, self.panels):
            key = choice.currentData()
            label, unit, _ = PROFILE_OPTIONS[key]
            panel.set_quantity(label, unit)
            rows = self._profile_rows(key, slots, index, p)
            if rows is None:
                panel.clear()
                panel.setTitle(f"loading {label}…", size="9pt")
                continue
            how = mode.currentData()
            ws = key == "ws"
            panel.draw(p.heights, rows, times, dashed, noses if ws else None, label, how, circular=key == "wd",
                       estimates=estimates if ws else None)

        frame = pd.DataFrame({
            "time": times,
            "status": [_STATUS_TEXT.get(str(s), str(s)) + (" (bridged)" if f else " (added by hand)" if m else "")
                       for s, f, m in zip(detection["status"], filled, added)],
            "obs. peak (m)": detection["z_nose"].to_numpy(), "obs. speed (m/s)": detection["u_nose"].to_numpy(),
            "est. nose (m)": detection.get("z_nose_est", pd.Series(np.nan, index=detection.index)).to_numpy(),
            "est. speed (m/s)": detection.get("u_nose_est", pd.Series(np.nan, index=detection.index)).to_numpy(),
            "falloff (m/s)": detection["falloff_abs"].to_numpy(),
            "slowest above (m)": detection["z_min_above"].to_numpy(),
        })
        tables.fill(self.table, frame)
        self.prev_button.setEnabled(event_id > r.events.table.index.min())
        self.next_button.setEnabled(event_id < r.events.table.index.max())

    def _on_map_click(self, ev) -> None:
        if not ev.double() or self.event_id is None or not self.rows:
            return
        vb = self.map.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = vb.mapSceneToView(ev.scenePos())
        row = int(np.floor(pos.y()))
        if 0 <= row < len(self.rows):
            self.inspectRequested.emit(int(unix_to_slot(pos.x())), self.rows[row])
