"""The run's manual jet edits: what each is, what it does now, and removing
some or all of them.
"""
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from ttu_tower.timegrid import slot_to_time

_KIND_TEXT = {"span": "new boundaries", "add": "made by hand", "reject": "rejected"}


class EditsDialog(QDialog):
    removeRequested = Signal(list)  # edit ids
    clearRequested = Signal()
    gotoRequested = Signal(int)  # event id

    COLUMNS = ("edit", "from", "to", "event now", "does now", "made", "note")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual jet edits")
        self.setModal(False)
        self.where = QLabel()
        self.where.setWordWrap(True)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda row, _: self._goto(row))
        self.remove_button = QPushButton("Remove selected")
        self.clear_button = QPushButton("Remove all")
        self.goto_button = QPushButton("Show event")
        close = QPushButton("Close")
        self.remove_button.clicked.connect(self._remove)
        self.clear_button.clicked.connect(self._clear)
        self.goto_button.clicked.connect(lambda: self._goto(self.table.currentRow()))
        close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        for b in (self.goto_button, self.remove_button, self.clear_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.where)
        layout.addWidget(self.table, stretch=1)
        layout.addLayout(buttons)
        self.resize(820, 360)
        self.ids: list[str] = []
        self.events: list = []

    def show_report(self, report: pd.DataFrame | None, tz: str, where: str) -> None:
        report = report if report is not None else pd.DataFrame(columns=["id"])
        self.where.setText(where)
        self.table.setRowCount(len(report))
        self.ids, self.events = [], []
        for i, r in enumerate(report.itertuples(index=False)):
            first, last = slot_to_time([r.start, r.end + 1]).tz_convert(tz)
            made = pd.Timestamp(r.created).tz_convert(tz) if r.created else None
            cells = (_KIND_TEXT.get(r.kind, r.kind), f"{first:%Y-%m-%d %H:%M}", f"{last:%Y-%m-%d %H:%M}",
                     "" if pd.isna(r.event_id) else str(int(r.event_id)), r.status,
                     f"{made:%Y-%m-%d %H:%M}" if made is not None else "", r.note or "")
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j == 4 and r.status != "applied":
                    item.setForeground(Qt.GlobalColor.darkRed)
                self.table.setItem(i, j, item)
            self.ids.append(r.id)
            self.events.append(None if pd.isna(r.event_id) else int(r.event_id))
        self.table.resizeColumnsToContents()
        for b in (self.remove_button, self.clear_button):
            b.setEnabled(bool(self.ids))

    def _selected(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.ids[r] for r in rows if r < len(self.ids)]

    def _remove(self) -> None:
        ids = self._selected()
        if ids:
            self.removeRequested.emit(ids)

    def _clear(self) -> None:
        answer = QMessageBox.question(self, "Remove all", f"Remove all {len(self.ids)} manual edits of this run?")
        if answer == QMessageBox.StandardButton.Yes:
            self.clearRequested.emit()

    def _goto(self, row: int) -> None:
        if 0 <= row < len(self.events) and self.events[row] is not None:
            self.gotoRequested.emit(self.events[row])
