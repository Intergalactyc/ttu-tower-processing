"""Every stored row for the slot, table by table, searchable and copyable;
and Verify, which recomputes the slot's primary products from the raw files
and compares them with what the run stored.
"""
import numpy as np
import pandas as pd
from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSplitter, QTableView, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ttu_tower.viewer.ui import tables


def _text(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return tables.number(v, 6)
    return str(v)


class NumbersTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bundle = self.reprocessor = self.runner = None
        self.table_box = QComboBox()
        self.table_box.currentIndexChanged.connect(lambda *_: self._show_table())
        self.search = QLineEdit()
        self.search.setPlaceholderText("filter rows…")
        self.model = QStandardItemModel()
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.verticalHeader().setDefaultSectionSize(18)
        copy = QPushButton("Copy")
        copy.setToolTip("copy the selected rows (or all shown rows) as tab-separated text")
        copy.clicked.connect(self.copy)
        QShortcut(QKeySequence.StandardKey.Copy, self.view, activated=self.copy)
        self.verify_button = QPushButton("Verify against raw")
        self.verify_button.setToolTip("recompute this slot's primary products from the raw files and compare")
        self.verify_button.clicked.connect(self.verify)
        self.verify_status = QLabel()
        self.verify_table = QTableWidget()
        self.verify_table.verticalHeader().setVisible(False)
        self.verify_table.verticalHeader().setDefaultSectionSize(18)

        row = QHBoxLayout()
        row.addWidget(QLabel("table"))
        row.addWidget(self.table_box)
        row.addWidget(self.search, stretch=1)
        row.addWidget(copy)
        row.addWidget(self.verify_button)
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addLayout(row)
        top_layout.addWidget(self.view, stretch=1)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.verify_status)
        bottom_layout.addWidget(self.verify_table, stretch=1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([600, 160])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

    def set_bundle(self, bundle, reprocessor, runner) -> None:
        changed = self.bundle is None or (bundle.slot, bundle.boom) != (self.bundle.slot, self.bundle.boom)
        self.bundle, self.reprocessor, self.runner = bundle, reprocessor, runner
        current = self.table_box.currentData()
        self.table_box.blockSignals(True)
        self.table_box.clear()
        for name, df in self._tables().items():
            self.table_box.addItem(f"{name} ({len(df)})", name)
        i = self.table_box.findData(current)
        if i < 0:
            i = max(self.table_box.findData("boom_final"), self.table_box.findData("means"), 0)
        self.table_box.setCurrentIndex(i)
        self.table_box.blockSignals(False)
        self._show_table()
        if changed:
            self.verify_table.clear()
            self.verify_table.setRowCount(0)
            self.verify_status.setText("")
        self.verify_button.setEnabled(reprocessor is not None)

    def _tables(self) -> dict[str, pd.DataFrame]:
        tables = {name: df for name, df in self.bundle.tables.items() if not df.empty}
        if self.bundle.mrd is not None:
            mrd = self.bundle.mrd
            tables["mrd"] = mrd[(mrd["slot"] == self.bundle.slot) & (mrd["boom"] == self.bundle.boom)]
        return tables

    def current_frame(self) -> pd.DataFrame:
        name = self.table_box.currentData()
        return self._tables().get(name, pd.DataFrame()) if name else pd.DataFrame()

    def _show_table(self) -> None:
        df = self.current_frame()
        self.model.clear()
        self.model.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for values in df.itertuples(index=False):
            self.model.appendRow([QStandardItem(_text(v)) for v in values])
        self.view.resizeColumnsToContents()

    def shown_rows(self) -> int:
        return self.proxy.rowCount()

    def copy(self) -> None:
        rows = sorted({i.row() for i in self.view.selectionModel().selectedIndexes()}) or range(self.proxy.rowCount())
        cols = range(self.proxy.columnCount())
        header = "\t".join(self.model.headerData(c, Qt.Orientation.Horizontal) for c in cols)
        lines = ["\t".join(self.proxy.index(r, c).data() or "" for c in cols) for r in rows]
        QGuiApplication.clipboard().setText("\n".join([header] + lines))

    def verify(self) -> None:
        if self.reprocessor is None or self.bundle is None:
            return
        self.verify_status.setText("recomputing this slot from the raw files…")
        stored = self._tables()
        self.runner.submit(self.reprocessor.verify, self.bundle.boom, self.bundle.slot, stored, key=f"verify-{id(self)}",
                           label="verifying against raw", on_done=self._verified,
                           on_error=lambda exc, tb: self.verify_status.setText(f"verify failed: {exc}"))

    def _verified(self, report: pd.DataFrame) -> None:
        self.verify_table.clear()
        self.verify_table.setRowCount(len(report))
        self.verify_table.setColumnCount(len(report.columns))
        self.verify_table.setHorizontalHeaderLabels(list(report.columns))
        for i, row in enumerate(report.itertuples(index=False)):
            for j, v in enumerate(row):
                self.verify_table.setItem(i, j, QTableWidgetItem(_text(v)))
        exact = bool(((report["max abs diff"] == 0) & (report["only recomputed"] == 0) & (report["only stored"] == 0)).all())
        self.verify_status.setText(
            "<span style='color:#1a7f37'><b>identical</b>: the installed code reproduces every stored row exactly</span>"
            if exact else "<span style='color:#b00'><b>differences</b> — see the table (the installed code may differ "
                          "from the run's)</span>")
        self.last_report = report
