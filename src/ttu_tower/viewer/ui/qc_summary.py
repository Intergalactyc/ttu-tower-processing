"""QC statistics per boom over the viewed interval (or the whole period):
availability, the samples left at each QC layer, what each flag test flagged,
what tertiary filtered and how tau was chosen.
"""
import numpy as np
import pandas as pd
import shiboken6
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ttu_tower.viewer import qcsummary
from ttu_tower.viewer.catalog import variable_label
from ttu_tower.viewer.timeaxis import unix_to_slot
from ttu_tower.viewer.ui import tables

_VARIANTS = (("mrd", "mrd (selected τ)"), ("naive", "naive (10 min)"), ("mrd_unexcised", "mrd unexcised"))
_FLAGGED_TIP = ("% of the variable's samples each test flagged, over the slots with a file; excursion only "
                "records (it removes nothing), and boom-level tests count toward ue/vn/w/ts")


def _grade(value, bad: float, warn: float, low_is_bad: bool = False):
    if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
        return None
    if low_is_bad:
        return tables.BAD if value < bad else tables.WARN if value < warn else None
    return tables.BAD if value >= bad else tables.WARN if value >= warn else None


def _percent(col, value):
    if not isinstance(value, float) or not np.isfinite(value):
        return None
    return "<0.001" if 0 < value < 0.001 else f"{value:.3g}"


class QCSummaryView(QWidget):
    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = None
        self.booms: list[int] = []
        self.x_range = None
        self.availability = None  # (slots, status image) from the main window's overlays
        self.stored: qcsummary.Stored | None = None
        self.flags: dict | None = None
        self._loading = False

        self.variant = QComboBox()
        for key, label in _VARIANTS:
            self.variant.addItem(label, key)
        self.variant.setToolTip("the variant the filtering and τ tables describe")
        self.coverage_var = QComboBox()
        for v in qcsummary.COVERAGE_CHOICES:
            self.coverage_var.addItem(variable_label(v)[0] if v not in ("momentum", "heat") else f"{v} family", v)
        self.flag_var = QComboBox()
        for v in qcsummary.FLAG_CHOICES:
            self.flag_var.addItem(variable_label(v)[0], v)
        self.flag_var.setCurrentIndex(self.flag_var.findData("w"))
        for combo in (self.variant, self.coverage_var, self.flag_var):
            combo.currentIndexChanged.connect(lambda *_: self.refresh())

        self.scope = QLabel()
        self.availability_table = tables.new_table()
        self.availability_table.setToolTip("% of the slots in range by status; the usable columns are the mean "
                                           "fraction of each family's samples left, over computed slots")
        self.coverage_table = tables.new_table()
        self.coverage_table.setToolTip("the mean fraction of the variable's samples left at each QC layer, over the "
                                       "slots with a file: present (raw) → filled (bounds and spikes out, short gaps "
                                       "filled) → usable_l1 (first-layer tests) → usable (direction and bounce "
                                       "too); unexcised: before excision")
        self.flag_table = tables.new_table()
        self.flag_table.setToolTip(_FLAGGED_TIP)
        self.flag_note = QLabel()
        self.flag_note.hide()
        self.filter_table = tables.new_table()
        self.filter_table.setToolTip("% of computed slots where tertiary filtered each quantity group (hover a cell "
                                     "for the split by criterion); '(means)' groups are the slot means'")
        self.tau_table = tables.new_table()
        self.tau_table.setToolTip("% of computed slots by how the selected τ came about, and from which "
                                  "cospectrum (momentum u'w' or heat w'θv') where it came from one")

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(2, 2, 2, 2)
        column.addWidget(self.scope)
        column.addWidget(QLabel("<b>Availability</b>"))
        column.addWidget(self.availability_table)
        column.addLayout(_titled("<b>Samples left at each QC layer</b>", self.coverage_var))
        column.addWidget(self.coverage_table)
        column.addLayout(_titled("<b>Samples flagged, by test (%)</b>", self.flag_var))
        column.addWidget(self.flag_note)
        column.addWidget(self.flag_table)
        column.addLayout(_titled("<b>Filtered by tertiary (% of computed slots)</b>", self.variant))
        column.addWidget(self.filter_table)
        column.addWidget(QLabel("<b>How τ was chosen (% of computed slots)</b>"))
        column.addWidget(self.tau_table)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        for table in self._tables():
            tables.fit_rows(table)

    def _tables(self):
        return (self.availability_table, self.coverage_table, self.flag_table, self.filter_table, self.tau_table)

    def set_run(self, run, index, booms) -> None:
        self.run, self.index, self.booms = run, index, list(booms)
        self.availability = self.stored = self.flags = None
        self._loading = False
        for table in self._tables():
            table.setRowCount(0)
        has_secondary = "secondary" in run.stages
        self.variant.setEnabled(has_secondary)
        self.scope.setText("")

    def set_availability(self, slots, image) -> None:
        self.availability = (slots, image)
        self.refresh()

    def set_range(self, x_range) -> None:
        self.x_range = x_range
        self._load()
        self.refresh()

    def _load(self) -> None:
        """The stored records, then every boom's flags (slower), each read once per run."""
        if self.run is None or self._loading or (self.stored is not None and self.flags is not None):
            return
        self._loading = True
        run, index, booms = self.run, self.index, tuple(self.booms)

        def stored_ready(data):
            if not shiboken6.isValid(self) or run is not self.run:
                return
            self.stored = data
            self.refresh()
            self._note("reading every boom's flags (once per run; about 3 s a boom)…")
            self.runner.submit(qcsummary.flag_counts, run, index, booms, key=f"qcsum-flags-{id(self)}",
                               label="counting flagged samples", on_done=flags_ready, on_error=failed)

        def flags_ready(counts):
            if not shiboken6.isValid(self) or run is not self.run:
                return
            self.flags, self._loading = counts, False
            self._note("")
            self.refresh()

        def failed(exc, tb):
            if shiboken6.isValid(self):
                self._loading = False
                self._note(f"couldn't read the QC records: {exc}")

        self.runner.submit(qcsummary.stored, run, index, booms, key=f"qcsum-{id(self)}",
                           label="reading QC records", on_done=stored_ready, on_error=failed)

    def _note(self, text: str) -> None:
        self.flag_note.setText(text)
        self.flag_note.setVisible(bool(text))

    def slot_range(self):
        if self.x_range is None:
            return None
        return int(unix_to_slot(self.x_range[0])), int(unix_to_slot(self.x_range[1])) + 1

    def refresh(self) -> None:
        if self.run is None or self.availability is None:
            return
        slots, image = self.availability
        rng = self.slot_range()
        n = int(qcsummary.in_range(slots, rng).sum())
        self.scope.setText(("whole period" if rng is None else "visible range") + f": {n} slots")
        if self.stored is None:
            self.scope.setText(self.scope.text() + " · reading the QC records…")
            return
        data, booms = self.stored, self.booms
        self._fill(self.availability_table, qcsummary.availability_table(slots, image, booms, data, rng),
                   lambda r, c, v: _grade(v, 0.5, 0.8, True) if "usable" in c
                   else _grade(v, 50, 90, True) if c == "computed %" else None)
        self._fill(self.coverage_table,
                   qcsummary.coverage_table(slots, image, booms, data, self.coverage_var.currentData(), rng),
                   lambda r, c, v: _grade(v, 0.5, 0.8, True))
        if self.flags is not None:
            self._fill(self.flag_table,
                       qcsummary.flag_table(slots, image, booms, self.flags, self.flag_var.currentData(), rng),
                       lambda r, c, v: None if c == "excursion" else _grade(v, 10, 1))
        variant = self.variant.currentData()
        frame, why = qcsummary.filter_table(slots, image, booms, data, variant, rng)
        self._fill(self.filter_table, frame, lambda r, c, v: _grade(v, 50, 10))
        for i, row in enumerate(frame.index):
            for j, col in enumerate(frame.columns):
                item = self.filter_table.item(i, j)
                if item is not None and (row, col) in why:
                    item.setToolTip(why[(row, col)])
        self._fill(self.tau_table, qcsummary.tau_table(slots, image, booms, data, variant, rng),
                   lambda r, c, v: None if c.startswith(("found", "fixed")) else _grade(v, 25, 5))

    def _fill(self, table, frame: pd.DataFrame, color) -> None:
        tables.fill(table, frame, index=True, color=color,
                    fmt=lambda col, v: f"{v:.0f}" if col == "slots" else _percent(col, v))
        tables.fit_rows(table)


def _titled(title: str, control) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(title))
    row.addStretch(1)
    row.addWidget(control)
    return row
