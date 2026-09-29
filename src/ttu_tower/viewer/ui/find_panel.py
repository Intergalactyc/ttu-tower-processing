"""Find: a condition over any catalog quantities (one row per slot and boom),
its matches listed, stepped through in the Slot Inspector, selected on the
plots, or saved as a bookmark list.
"""
import numpy as np
import pandas as pd
import shiboken6
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QSpinBox, QTableWidget,
    QToolButton, QVBoxLayout, QWidget,
)

from ttu_tower.viewer import find
from ttu_tower.viewer.timeaxis import format_slot
from ttu_tower.viewer.ui import tables

MAX_SHOWN = 5000  # rows listed (every match is still stepped through and selected)
_VARIANTS = (("mrd", "mrd (selected τ)"), ("naive", "naive (10 min)"), ("mrd_unexcised", "mrd unexcised"))


class NamesDialog(QDialog):
    """Every column a query can use, searchable; double-click puts one in the query."""

    chosen = Signal(str)

    def __init__(self, known: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find: column names")
        self.resize(640, 600)
        self.search = QLineEdit()
        self.search.setPlaceholderText("search names and descriptions…")
        self.table = tables.new_table()
        frame = pd.DataFrame({"name": list(known), "what": [n.description for n in known.values()]})
        tables.fill(self.table, frame)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda row, _col: self.chosen.emit(self.table.item(row, 0).text()))
        self.search.textChanged.connect(self._filter)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Per-boom quantities take the boom's value on each row; per-slot ones (and pairs, "
                                "e.g. rib_2_4) repeat on every boom's row. Double-click a name to insert it."))
        layout.addWidget(self.search)
        layout.addWidget(self.table)

    def _filter(self, text: str) -> None:
        text = text.lower().strip()
        for row in range(self.table.rowCount()):
            line = " ".join(self.table.item(row, c).text() for c in range(2)).lower()
            self.table.setRowHidden(row, bool(text) and text not in line)


class FindPanel(QWidget):
    inspectRequested = Signal(object, int, str)  # [(slot, boom)] to step through, where to start, a label
    selectRequested = Signal(object)  # the matching slots, for linked brushing
    saveRequested = Signal(object, str)  # [(slot, boom)], the query

    def __init__(self, runner, context, parent=None):
        """`context()` -> dict(run, index, catalog, stability, tz, x_range, booms) now."""
        super().__init__(parent)
        self.runner, self.context = runner, context
        self.result: pd.DataFrame | None = None
        self.columns: list[str] = []
        self._expr = ""

        self.query = QLineEdit()
        self.query.setPlaceholderText("e.g. ws_mean > 12 and boom == 9 and stability == 'stable'")
        self.query.returnPressed.connect(self.run_query)
        self.query.setToolTip("a pandas query over one row per slot and boom; Names… lists the columns")
        presets = QToolButton()
        presets.setText("Presets")
        presets.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(presets)
        for label, expr, sample in find.PRESETS:
            act = QAction(label, menu)
            act.setToolTip(expr)
            act.triggered.connect(lambda _=False, e=expr, n=sample: self._preset(e, n))
            menu.addAction(act)
        presets.setMenu(menu)
        names = QPushButton("Names…")
        names.clicked.connect(self._show_names)
        self.variant = QComboBox()
        for key, label in _VARIANTS:
            self.variant.addItem(label, key)
        self.variant.setToolTip("the variant of variant-specific quantities (u*, TI, τ, …) and of the filtering")
        self.in_view = QCheckBox("viewed interval only")
        self.sample = QSpinBox()
        self.sample.setRange(0, 100_000)
        self.sample.setSpecialValueText("all matches")
        self.sample.setPrefix("random ")
        self.sample.setToolTip("keep a random sample of this many matches (all matches: 0)")
        self.run_button = QPushButton("Find")
        self.run_button.clicked.connect(self.run_query)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.table = tables.new_table()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(lambda row, _col: self._inspect(row))
        self.step_button = QPushButton("Step through")
        self.step_button.setToolTip("open the Slot Inspector at the selected (or first) match; its match ◀ ▶ "
                                    "buttons step through the rest")
        self.step_button.clicked.connect(lambda: self._inspect(max(self.table.currentRow(), 0)))
        self.select_button = QPushButton("Select on plots")
        self.select_button.setToolTip("make the matching slots the brushed selection")
        self.select_button.clicked.connect(
            lambda: self.selectRequested.emit(np.unique(self.result["slot"].to_numpy(dtype=np.int64))))
        self.save_button = QPushButton("Save as list…")
        self.save_button.setToolTip("keep the matches as a bookmark list")
        self.save_button.clicked.connect(lambda: self.saveRequested.emit(self._items(), self._expr))

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Find slots</b>"))
        layout.addWidget(self.query)
        row = QHBoxLayout()
        for w in (presets, names, self.variant):
            row.addWidget(w)
        row.addStretch(1)
        layout.addLayout(row)
        row = QHBoxLayout()
        for w in (self.in_view, self.sample, self.run_button):
            row.addWidget(w)
        layout.addLayout(row)
        layout.addWidget(self.status)
        layout.addWidget(self.table, stretch=1)
        row = QHBoxLayout()
        for w in (self.step_button, self.select_button, self.save_button):
            row.addWidget(w)
        layout.addLayout(row)
        self._enable()

    def _enable(self) -> None:
        has = self.result is not None and len(self.result) > 0
        for w in (self.step_button, self.select_button, self.save_button):
            w.setEnabled(has)

    def _preset(self, expr: str, sample: int) -> None:
        self.query.setText(expr)
        self.sample.setValue(sample)
        self.query.setFocus()

    def _show_names(self) -> None:
        ctx = self.context()
        if ctx is None or ctx["catalog"] is None:
            return
        dialog = NamesDialog(find.names(ctx["catalog"]), self)
        dialog.chosen.connect(self.query.insert)
        dialog.show()

    def run_query(self) -> None:
        ctx = self.context()
        if ctx is None or ctx["catalog"] is None:
            self.status.setText("the run's quantities are still loading")
            return
        expr = self.query.text()
        slot_range = ctx["slot_range"] if self.in_view.isChecked() else None
        self.status.setText("reading the columns the query names…")
        self.run_button.setEnabled(False)
        self._expr = expr
        seed = int(np.random.default_rng().integers(2**31))
        self.runner.submit(find.find, ctx["run"], ctx["index"], ctx["catalog"], expr, self.variant.currentData(),
                           ctx["booms"], slot_range, ctx["stability"], ctx["tz"], self.sample.value(), seed,
                           key=f"find-{id(self)}", label="finding slots",
                           on_done=lambda result: self._found(expr, ctx, result), on_error=self._failed)

    def _failed(self, exc, tb) -> None:
        if not shiboken6.isValid(self):
            return
        self.run_button.setEnabled(True)
        message = str(exc) if isinstance(exc, find.FindError) else f"{type(exc).__name__}: {exc}"
        self.status.setText(f"<span style='color:#b00'>{message}</span>")

    def _found(self, expr: str, ctx, result) -> None:
        if not shiboken6.isValid(self):
            return
        self.run_button.setEnabled(True)
        self.result, self.columns = result
        frame, tz = self.result, ctx["tz"]
        slots = frame["slot"].nunique()
        n = self.sample.value()
        sampled = f", a random {n}" if n and len(frame) == n else ""
        self.status.setText(f"{len(frame)} match{'es' if len(frame) != 1 else ''} ({slots} slots{sampled})"
                            + (f"; the first {MAX_SHOWN} listed" if len(frame) > MAX_SHOWN else ""))
        shown = frame.head(MAX_SHOWN)
        listed = pd.DataFrame({"time": [format_slot(int(k), tz) for k in shown["slot"]], "boom": shown["boom"]})
        for c in self.columns:
            if c not in ("slot", "boom", "time"):
                listed[c] = shown[c].to_numpy()
        tables.fill(self.table, listed)
        self._enable()

    def _items(self) -> list[tuple[int, int]]:
        if self.result is None:
            return []
        return list(zip(self.result["slot"].astype(int), self.result["boom"].astype(int)))

    def _inspect(self, row: int) -> None:
        items = self._items()
        if not items:
            return
        self.inspectRequested.emit(items, min(max(row, 0), len(items) - 1), f"find: {self._expr}")
