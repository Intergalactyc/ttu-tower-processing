"""Bookmarks: the saved slots of this run (or every run), by list, with their
notes editable in place; opened and stepped through in the Slot Inspector,
selected on the plots, deleted or exported.
"""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer.bookmarks import DEFAULT_LIST, BookmarkStore
from ttu_tower.viewer.timeaxis import format_slot
from ttu_tower.viewer.ui import tables

_ALL_LISTS = "(every list)"
_COLUMNS = ("time", "boom", "variant", "note", "list", "run")


class BookmarkDialog(QDialog):
    """A note and a list for a new bookmark (or a set of them)."""

    def __init__(self, title: str, lists: list[str], list_name: str = DEFAULT_LIST, note: str = "",
                 parent=None, with_note: bool = True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.note = QLineEdit(note)
        self.note.setPlaceholderText("what's worth seeing here")
        self.list_box = QComboBox()
        self.list_box.setEditable(True)
        self.list_box.addItems(lists)
        self.list_box.setCurrentText(list_name)
        self.list_box.setToolTip("choose a list or type a new one's name")
        form = QFormLayout(self)
        if with_note:
            form.addRow("note", self.note)
        form.addRow("list", self.list_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[str, str]:
        return self.note.text().strip(), self.list_box.currentText().strip() or DEFAULT_LIST


class BookmarksPanel(QWidget):
    inspectRequested = Signal(object, int, str, str)  # [(slot, boom)], where to start, label, variant
    selectRequested = Signal(object)  # the listed slots, for linked brushing

    def __init__(self, store: BookmarkStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.run_tag: str | None = None
        self.tz = "UTC"
        self.shown = []  # the bookmarks listed, in table order
        self.list_filter = QComboBox()
        self.list_filter.currentIndexChanged.connect(lambda *_: self.refresh())
        self.all_runs = QCheckBox("every run")
        self.all_runs.setToolTip("list other runs' bookmarks too (they open only in their own run)")
        self.all_runs.toggled.connect(lambda *_: self.refresh())
        self.table = tables.new_table()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._double_clicked)
        self.table.itemChanged.connect(self._edited)
        self.where = QLabel()
        self.where.setWordWrap(True)
        self.where.setStyleSheet("color:#666;")
        buttons = []
        for label, tip, slot in (("Open", "step through the listed bookmarks from the selected one",
                                  self._open_current),
                                 ("Select on plots", "make the listed bookmarks' slots the brushed selection",
                                  self._select),
                                 ("Delete", "delete the selected bookmarks", self._delete),
                                 ("Export…", "save the listed bookmarks as JSON or CSV", self._export)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.append(b)
        self.open_button, self.select_button, self.delete_button, self.export_button = buttons

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Bookmarks</b>"))
        row = QHBoxLayout()
        row.addWidget(QLabel("list"))
        row.addWidget(self.list_filter, stretch=1)
        row.addWidget(self.all_runs)
        layout.addLayout(row)
        layout.addWidget(self.table, stretch=1)
        row = QHBoxLayout()
        for b in buttons:
            row.addWidget(b)
        layout.addLayout(row)
        layout.addWidget(self.where)
        self.refresh()

    def set_run(self, tag: str, tz: str) -> None:
        self.run_tag, self.tz = tag, tz
        self.refresh()

    def refresh(self) -> None:
        run = None if self.all_runs.isChecked() else self.run_tag
        current = self.list_filter.currentText()
        self.list_filter.blockSignals(True)
        self.list_filter.clear()
        self.list_filter.addItem(_ALL_LISTS)
        self.list_filter.addItems(self.store.lists(run))
        i = self.list_filter.findText(current)
        self.list_filter.setCurrentIndex(max(i, 0))
        self.list_filter.blockSignals(False)
        chosen = self.list_filter.currentText()
        self.shown = sorted((b for b in self.store.for_run(run) if chosen == _ALL_LISTS or b.list == chosen),
                            key=lambda b: (b.run, b.slot, b.boom))
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setRowCount(len(self.shown))
        for i, b in enumerate(self.shown):
            cells = (format_slot(b.slot, self.tz), f"b{b.boom} ({HEIGHTS.get(b.boom, 0):g} m)", b.variant, b.note,
                     b.list, b.run)
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if _COLUMNS[j] not in ("note", "list"):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed)
        self.table.setToolTip("double-click a note or list to edit it; double-click elsewhere to open")
        path = self.store.path
        self.where.setText(f"kept in {path}" if path is not None else "no ttu-tower home: bookmarks can't be saved")
        for b in (self.open_button, self.select_button, self.delete_button, self.export_button):
            b.setEnabled(bool(self.shown))

    def _edited(self, item: QTableWidgetItem) -> None:
        column = _COLUMNS[item.column()]
        if column in ("note", "list") and item.row() < len(self.shown):
            value = item.text().strip() or (DEFAULT_LIST if column == "list" else "")
            self.store.update(self.shown[item.row()].id, **{column: value})
            if column == "list":
                self.refresh()

    def _this_run(self) -> list:
        return [b for b in self.shown if b.run == self.run_tag]

    def _double_clicked(self, row: int, column: int) -> None:
        if _COLUMNS[column] not in ("note", "list"):  # a double-click there edits instead
            self._open_row(row)

    def _open_current(self) -> None:
        self._open_row(max(self.table.currentRow(), 0))

    def _open_row(self, row: int) -> None:
        if not self.shown:
            return
        b = self.shown[min(row, len(self.shown) - 1)]
        if b.run != self.run_tag:
            QMessageBox.information(self, "ttu-view", f"That bookmark belongs to run '{b.run}'; open that run first.")
            return
        mine = self._this_run()
        items = [(x.slot, x.boom) for x in mine]
        chosen = self.list_filter.currentText()
        label = "bookmarks" if chosen == _ALL_LISTS else f"list: {chosen}"
        self.inspectRequested.emit(items, mine.index(b), label, b.variant)

    def _select(self) -> None:
        slots = np.unique([b.slot for b in self._this_run()]).astype(np.int64)
        self.selectRequested.emit(slots)

    def _delete(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        plural = "s" if len(rows) != 1 else ""
        answer = QMessageBox.question(self, "ttu-view", f"Delete {len(rows)} bookmark{plural}?")
        if answer == QMessageBox.StandardButton.Yes:
            self.store.remove(self.shown[r].id for r in rows)
            self.refresh()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export bookmarks", "bookmarks.csv", "CSV (*.csv);;JSON (*.json)")
        if path:
            self.store.export(path, self.shown, describe=lambda k: format_slot(int(k), self.tz))
