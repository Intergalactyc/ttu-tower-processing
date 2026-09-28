"""The Series tab's QC what-if side panel: every `[qc]` setting of the run,
editable, applied on request; then what changed - per slot coverage and
means, and on request the focus slot's tau.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

from ttu_tower.viewer import qcwhatif
from ttu_tower.viewer.ui import tables

_CHANGED = "background:#fff0b3;"


def _text(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


class QCWhatIfPanel(QWidget):
    applied = Signal(object)  # the what-if Config, or None for the run's own
    tauRequested = Signal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.fields = qcwhatif.fields(cfg.qc)
        self.editors: dict[tuple, object] = {}
        self.active = None  # the applied what-if config

        form_host = QWidget()
        form_layout = QVBoxLayout(form_host)
        form_layout.setContentsMargins(2, 2, 2, 2)
        boxes: dict[str, QFormLayout] = {}
        for f in self.fields:
            if f.section not in boxes:
                box = QGroupBox(qcwhatif.section_title(f.section))
                boxes[f.section] = QFormLayout(box)
                form_layout.addWidget(box)
            boxes[f.section].addRow(f.label or f.name, self._editor(f))
        form_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("reprocess the loaded slots with these settings (the run's own stay untouched)")
        self.apply_button.clicked.connect(self.apply)
        reset = QPushButton("Reset")
        reset.setToolTip("back to the run's settings, and drop the what-if")
        reset.clicked.connect(self.reset)
        self.state = QLabel("showing the run's own QC")
        self.state.setWordWrap(True)
        self.state.setTextFormat(Qt.TextFormat.RichText)
        self.comparison = tables.new_table()
        self.comparison_note = QLabel("apply a what-if to compare slot coverage and means")
        self.comparison_note.setWordWrap(True)
        self.tau_button = QPushButton("τ for the focus slot")
        self.tau_button.setToolTip("recompute the focus slot's spectra and ladder with the what-if QC, then detect and "
                                   "select τ with the run's own secondary settings (a few seconds)")
        self.tau_button.setEnabled(False)
        self.tau_button.clicked.connect(self.tauRequested.emit)
        self.tau_table = tables.new_table()
        self.tau_note = QLabel()
        self.tau_note.setWordWrap(True)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(scroll, stretch=1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(reset)
        top_layout.addLayout(buttons)
        top_layout.addWidget(self.state)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.comparison_note)
        bottom_layout.addWidget(self.comparison, stretch=2)
        bottom_layout.addWidget(self.tau_button)
        bottom_layout.addWidget(self.tau_note)
        bottom_layout.addWidget(self.tau_table, stretch=1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([520, 420])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        self.setMinimumWidth(380)

    # --- the form ------------------------------------------------------------------------------

    def _editor(self, f: qcwhatif.Field) -> QWidget:
        value = qcwhatif.get(self.cfg.qc, f.path)
        if f.kind == "tests":
            host = QWidget()
            grid = QGridLayout(host)
            grid.setContentsMargins(0, 0, 0, 0)
            boxes = {}
            for i, test in enumerate(qcwhatif.QUALITY_TESTS):
                box = QCheckBox(test)
                box.setChecked(test in value)
                box.toggled.connect(lambda *_: self._mark())
                boxes[test] = box
                grid.addWidget(box, i // 3, i % 3)
            self.editors[f.path] = boxes
            host.setToolTip("tests whose flagged samples become unusable (the others are only recorded)")
            return host
        if f.kind == "pair":
            host = QWidget()
            row = QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            edits = [QLineEdit(_text(v)) for v in value]
            for e in edits:
                e.textChanged.connect(lambda *_: self._mark())
                row.addWidget(e)
            self.editors[f.path] = edits
            return host
        edit = QLineEdit(_text(value))
        edit.textChanged.connect(lambda *_: self._mark())
        self.editors[f.path] = edit
        return edit

    def _read(self, f: qcwhatif.Field):
        editor = self.editors[f.path]
        if f.kind == "tests":
            return tuple(t for t in qcwhatif.QUALITY_TESTS if editor[t].isChecked())
        cast = int if f.kind == "int" else float
        if f.kind == "pair":
            return tuple(cast(e.text()) for e in editor)
        return cast(editor.text())

    def values(self) -> dict[tuple, object]:
        """Every field whose editor differs from the run (ValueError on unreadable input)."""
        out = {}
        for f in self.fields:
            try:
                value = self._read(f)
            except ValueError:
                raise ValueError(f"{f.name}: not a number") from None
            if value != qcwhatif.get(self.cfg.qc, f.path):
                out[f.path] = value
        return out

    def _mark(self) -> None:
        for f in self.fields:
            editor = self.editors[f.path]
            try:
                changed = self._read(f) != qcwhatif.get(self.cfg.qc, f.path)
            except ValueError:
                changed = True
            widgets = editor.values() if isinstance(editor, dict) else (editor if isinstance(editor, list) else [editor])
            for w in widgets:
                w.setStyleSheet(_CHANGED if changed else "")

    def edited_config(self):
        return qcwhatif.with_values(self.cfg, self.values())

    def apply(self) -> None:
        try:
            values = self.values()
        except ValueError as exc:
            self.state.setText(f"<span style='color:#b00'>{exc}</span>")
            return
        if not values:
            self.reset()
            return
        cfg = qcwhatif.with_values(self.cfg, values)
        issues = qcwhatif.problems(cfg)
        if issues:
            self.state.setText(f"<span style='color:#b00'>{'; '.join(issues)}</span>")
            return
        self.active = cfg
        changed = qcwhatif.changes(self.cfg.qc, cfg.qc)
        self.state.setText("<b>what-if applied</b>: " + "; ".join(f"{name} {_text(a)} → {_text(b)}"
                                                                    for name, a, b in changed))
        self.tau_button.setEnabled(True)
        self.applied.emit(cfg)

    def reset(self) -> None:
        for f in self.fields:
            editor, value = self.editors[f.path], qcwhatif.get(self.cfg.qc, f.path)
            if f.kind == "tests":
                for test, box in editor.items():
                    box.setChecked(test in value)
            elif f.kind == "pair":
                for e, v in zip(editor, value):
                    e.setText(_text(v))
            else:
                editor.setText(_text(value))
        was_active, self.active = self.active is not None, None
        self.state.setText("showing the run's own QC")
        self.tau_button.setEnabled(False)
        self.comparison.clear()
        self.comparison.setRowCount(0)
        self.tau_table.clear()
        self.tau_table.setRowCount(0)
        self.comparison_note.setText("apply a what-if to compare slot coverage and means")
        self.tau_note.setText("")
        if was_active:
            self.applied.emit(None)

    # --- results ---------------------------------------------------------------------------------

    def show_comparison(self, df, format_slot) -> None:
        shown = df.copy()
        shown["slot"] = shown["slot"].map(lambda k: format_slot(k)[11:16])
        changed = (shown["change"].abs() > 0) | (shown["run"].isna() != shown["what-if"].isna())
        tables.fill(self.comparison, shown.reset_index(drop=True),
                    fmt=lambda c, v: f"{v:+.4g}" if c == "change" and isinstance(v, float) and v == v else None)
        for i, c in enumerate(changed):
            if c:
                for j in range(self.comparison.columnCount()):
                    self.comparison.item(i, j).setBackground(tables.WARN)
        n = int(changed.sum())
        self.comparison_note.setText(f"{n} of {len(shown)} slot values change (the run's stored rows vs the what-if)")

    def show_tau(self, df) -> None:
        tables.fill(self.tau_table, df)
        self.tau_note.setText("the focus slot's τ, stored vs with the what-if QC (u*, σw read off the ladder at τ)")
