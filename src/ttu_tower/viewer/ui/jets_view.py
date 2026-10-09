"""The Jets summary tab: the events of the current detection as a sortable
table (selecting one shows it on the timelines and in the Event sub-tab; a
double click opens it in the Slot Inspector), one event in detail, the events
in aggregate, and the diagnostics for tuning.
"""
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from ttu_tower.jets.analysis import JetResult
from ttu_tower.viewer.ui import tables
from ttu_tower.viewer.ui.jets_event import EventView, duration_text
from ttu_tower.viewer.ui.jets_stats import AggregateView, DiagnosticsView

_FILTER_TEXT = {"time_of_day": "time of day", "night": "night", "inversion": "ΔVPT", "richardson": "Ri_b"}
_SOURCE_TEXT = {"auto": "detected", "edited": "edited", "manual": "by hand"}
MANUAL_COLOR = QColor("#fff0d9")  # rows of events edited or made by hand


class _Item(QTableWidgetItem):
    """Sorts by its key (a number, or text), not by the text shown."""

    def __init__(self, text: str, key):
        super().__init__(text)
        self.key = key
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other):
        a, b = self.key, getattr(other, "key", None)
        a_nan = a is None or (isinstance(a, float) and np.isnan(a))
        b_nan = b is None or (isinstance(b, float) and np.isnan(b))
        if a_nan or b_nan:
            return not a_nan and b_nan  # blanks sort last
        return a < b


class EventsTable(QWidget):
    eventSelected = Signal(int)
    eventActivated = Signal(int)
    addRequested = Signal()  # an event from the brushed selection
    editsRequested = Signal()  # the list of manual edits
    selectRequested = Signal(int)  # brush this event's slots

    COLUMNS = ("#", "source", "start", "length", "obs. peak (m)", "est. nose (m)", "obs. peak speed (m/s)",
               "est. nose speed (m/s)", "max falloff (m/s)", "dir. at peak (°)", "ΔVPT (K)", "Ri_b", "bridged",
               "kept")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kept_only = QCheckBox("only events the filters keep")
        self.kept_only.setChecked(True)
        self.kept_only.toggled.connect(lambda *_: self._fill())
        self.count = QLabel()
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.cellDoubleClicked.connect(lambda row, _: self._emit_row(row, self.eventActivated))
        self.add_button = QPushButton("Add event from selection")
        self.add_button.setToolTip("make an event by hand from the first to the last brushed slot")
        self.add_button.clicked.connect(self.addRequested)
        self.edits_button = QPushButton("Manual edits…")
        self.edits_button.setToolTip("every manual edit of this run: what it does now, and removing some or all")
        self.edits_button.clicked.connect(self.editsRequested)
        self.select_button = QPushButton("Select event slots")
        self.select_button.setToolTip("brush every slot of this event (bridged and hand-added ones too), so the "
                                      "timelines and Summary tabs show it")
        self.select_button.setEnabled(False)
        self.select_button.clicked.connect(lambda: self._emit_selected(self.selectRequested))
        head = QHBoxLayout()
        head.addWidget(self.kept_only)
        head.addStretch(1)
        head.addWidget(self.count)
        buttons = QHBoxLayout()
        buttons.addWidget(self.select_button)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.edits_button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(head)
        layout.addWidget(self.table, stretch=1)
        layout.addLayout(buttons)
        self.result: JetResult | None = None
        self.message = "no detection yet"
        self.nose_from = "jets"

    def sizeHint(self):
        return QSize(520, 400)

    def set_result(self, result: JetResult | None, message: str = "") -> None:
        self.result, self.message = result, message
        self._fill()

    def _fill(self) -> None:
        selected = self.selected_event()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        r = self.result
        if r is None:
            self.count.setText(self.message)
            return
        table = r.table(self.nose_from)
        if self.kept_only.isChecked():
            table = table[table["passed"]]
        edits = r.edit_report if r.edit_report is not None else []
        self.edits_button.setText(f"Manual edits ({len(edits)})…")
        self.count.setText(f"{len(table)} of {len(r.events.table)} events")
        f = r.settings.filters
        lower, upper = f.inversion_pair()
        a, b = sorted(f.rib_booms)
        noses = "jet slots" if self.nose_from == "jets" else "slots with a peak"
        heads = {"ΔVPT (K)": f"vpt(b{upper}) − vpt(b{lower}), the mean over every slot of the event",
                 "Ri_b": f"Ri_b(b{a}, b{b}), the median over every slot of the event",
                 "obs. peak (m)": f"the median height of the observed peak (the fastest boom) of the {noses}; "
                                  "the range in the tooltip",
                 "est. nose (m)": f"the median estimated nose height of the {noses}: the maximum of the quadratic "
                                  "through the peak and the booms just below and above it; the range in the tooltip",
                 "obs. peak speed (m/s)": f"the fastest observed peak of the {noses}",
                 "est. nose speed (m/s)": f"the fastest estimated nose of the {noses}",
                 "max falloff (m/s)": "the largest drop from the observed peak to the slowest boom above it",
                 "dir. at peak (°)": f"circular mean of the wind direction (FROM) at the peak boom of the {noses}",
                 "bridged": "rejected slots counted into the event between jet slots",
                 "source": "detected, its boundaries edited by hand, or made by hand",
                 "kept": "✓, or why not: the filters it fails, or rejected by hand",
                 "length": "h:mm, bridged slots included", "start": f"in {r.events.table['start_time'].dt.tz}"
                 if len(r.events.table) else ""}
        for j, name in enumerate(self.COLUMNS):
            self.table.horizontalHeaderItem(j).setToolTip(heads.get(name, ""))
        self.table.setRowCount(len(table))
        for i, (event_id, e) in enumerate(table.iterrows()):
            failed = ["rejected by hand"] if e.rejected else []
            failed += [_FILTER_TEXT[k] for k in _FILTER_TEXT if not e[k]]
            cells = [
                (str(event_id), int(event_id)),
                (_SOURCE_TEXT.get(e.source, str(e.source)), str(e.source)),
                (f"{e.start_time:%Y-%m-%d %H:%M}", e.start_slot),
                (duration_text(int(e.duration_min)), int(e.duration_min)),
                (tables.text(e.z_nose_median), float(e.z_nose_median)),
                (tables.text(e.z_est_median, 3), float(e.z_est_median)),
                (tables.text(e.u_nose_max, 3), float(e.u_nose_max)),
                (tables.text(e.u_est_max, 3), float(e.u_est_max)),
                (tables.text(e.falloff_max, 3), float(e.falloff_max)),
                (tables.text(e.wd_nose, 3), float(e.wd_nose)),
                (tables.text(e.dvpt_mean, 3), float(e.dvpt_mean)),
                (tables.text(e.rib_median, 3), float(e.rib_median)),
                (str(int(e.n_filled)), int(e.n_filled)),
                ("✓" if e.passed else "✗ " + ", ".join(failed), int(e.passed)),
            ]
            for j, (text, key) in enumerate(cells):
                item = _Item(text, key)
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, int(event_id))
                if j == 4 and np.isfinite(e.z_nose_min):
                    item.setToolTip(f"{e.z_nose_min:g}–{e.z_nose_max:g} m")
                if j == 5 and np.isfinite(e.z_est_min):
                    item.setToolTip(f"{e.z_est_min:.3g}–{e.z_est_max:.3g} m")
                if j == 2 and e.truncated:
                    item.setToolTip("touches the start or end of the data: may be longer")
                if not e.passed:
                    item.setForeground(Qt.GlobalColor.gray)
                if e.source != "auto":
                    item.setBackground(QBrush(MANUAL_COLOR))
                    font = QFont(item.font())
                    font.setItalic(True)
                    item.setFont(font)
                self.table.setItem(i, j, item)
        self.table.setSortingEnabled(True)
        if selected is not None:
            self.select_event(selected, emit=False)

    def _row_event(self, row: int) -> int | None:
        item = self.table.item(row, 0)
        return None if item is None else int(item.data(Qt.ItemDataRole.UserRole))

    def _emit_row(self, row: int, signal) -> None:
        event_id = self._row_event(row)
        if event_id is not None:
            signal.emit(event_id)

    def selected_event(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._row_event(rows[0].row()) if rows else None

    def _emit_selected(self, signal) -> None:
        event_id = self.selected_event()
        if event_id is not None:
            signal.emit(event_id)

    def _on_selection(self) -> None:
        event_id = self.selected_event()
        self.select_button.setEnabled(event_id is not None)
        if event_id is not None:
            self.eventSelected.emit(event_id)

    def select_event(self, event_id: int, emit: bool = True) -> bool:
        for row in range(self.table.rowCount()):
            if self._row_event(row) == event_id:
                self.table.blockSignals(not emit)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                self.table.scrollToItem(self.table.item(row, 0))
                return True
        return False

    def visible_events(self) -> list[int]:
        """The events listed, in the table's current order."""
        return [e for e in (self._row_event(r) for r in range(self.table.rowCount())) if e is not None]


class JetsView(QWidget):
    """The right-dock Jets tab: the event list, one event, the events in
    aggregate, and the diagnostics. Only the sub-tab showing is drawn; the
    others catch up when shown.
    """

    eventSelected = Signal(int)
    eventActivated = Signal(int)
    eventStep = Signal(int)
    inspectRequested = Signal(int, int)  # slot, boom
    valuesWanted = Signal(str)
    sweepRequested = Signal(str, str)
    spanEdited = Signal(int, int, int)  # event id, first slot, last slot
    rejectToggled = Signal(int)
    revertRequested = Signal(int)
    addRequested = Signal()
    editsRequested = Signal()
    selectRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.events = EventsTable()
        self.single = EventView()
        self.aggregate = AggregateView()
        self.diagnostics = DiagnosticsView()
        self.events.eventSelected.connect(self.eventSelected)
        self.events.eventActivated.connect(self.eventActivated)
        self.single.stepRequested.connect(self.eventStep)
        self.single.inspectRequested.connect(self.inspectRequested)
        self.single.valuesWanted.connect(self.valuesWanted)
        self.diagnostics.sweepRequested.connect(self.sweepRequested)
        self.single.spanEdited.connect(self.spanEdited)
        self.single.rejectToggled.connect(self.rejectToggled)
        self.single.revertRequested.connect(self.revertRequested)
        self.events.addRequested.connect(self.addRequested)
        self.events.editsRequested.connect(self.editsRequested)
        self.events.selectRequested.connect(self.selectRequested)
        self.single.selectRequested.connect(self.selectRequested)
        self.every_slot = QCheckBox("every slot of an event is a jet slot")
        self.every_slot.setToolTip(
            "off: an event's nose height, speed, falloff and direction come from its jet slots only; on: from "
            "each slot's best candidate nose, so slots added by hand count too. Profiles, ΔVPT and Ri_b always "
            "use every slot of the event.")
        self.every_slot.toggled.connect(self._on_every_slot)
        self.tabs = QTabWidget()
        for widget, label, tip in (
                (self.events, "Events", "every event of the detection (the whole period)"),
                (self.single, "Event", "the event picked last: its time-height map and profiles"),
                (self.aggregate, "Aggregate", "the events together (the whole period or the visible range)"),
                (self.diagnostics, "Diagnostics", "how the detection's rules and thresholds act (the whole period "
                                                  "or the visible range)")):
            self.tabs.addTab(widget, label)
            self.tabs.setTabToolTip(self.tabs.count() - 1, tip)
        self.tabs.currentChanged.connect(lambda *_: self._refresh())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.every_slot)
        layout.addWidget(self.tabs)
        self.result: JetResult | None = None
        self.profiles = None
        self.tz = "UTC"
        self.slot_range = None
        self.event_id: int | None = None
        self._stale: set = set()

    def sizeHint(self):
        return QSize(520, 400)

    def set_result(self, result: JetResult | None, message: str = "", profiles=None, tz: str | None = None) -> None:
        self.result, self.profiles = result, profiles
        if tz is not None:
            self.tz = tz
        self.events.set_result(result, message)
        if result is None or self.event_id not in result.events.table.index:
            self.event_id = None
        self._stale = {self.single, self.aggregate, self.diagnostics}
        self._refresh()

    @property
    def nose_from(self) -> str:
        return "candidates" if self.every_slot.isChecked() else "jets"

    def _on_every_slot(self, *_) -> None:
        self.events.nose_from = self.nose_from
        self.events._fill()
        self._stale |= {self.single, self.aggregate}
        self._refresh()

    def set_range(self, slot_range) -> None:
        if slot_range != self.slot_range:
            self.slot_range = slot_range
            self._stale |= {self.aggregate, self.diagnostics}
            self._refresh()

    def show_event(self, event_id: int | None) -> None:
        self.event_id = event_id
        self._stale |= {self.single, self.aggregate}
        self._refresh()

    def _refresh(self) -> None:
        shown = self.tabs.currentWidget()
        if shown not in self._stale:
            return
        self._stale.discard(shown)
        if shown is self.single:
            self.single.set_event(self.result, self.profiles, self.event_id, self.tz, self.nose_from)
        elif shown is self.aggregate:
            self.aggregate.set_data(self.result, self.profiles, self.slot_range, self.tz, self.nose_from,
                                    self.event_id)
        elif shown is self.diagnostics:
            self.diagnostics.set_data(self.result, self.slot_range)

    def select_event(self, event_id: int) -> None:
        if not self.events.select_event(event_id, emit=False) and self.events.kept_only.isChecked():
            self.events.kept_only.setChecked(False)  # a removed event picked on the band: show it
            self.events.select_event(event_id, emit=False)
