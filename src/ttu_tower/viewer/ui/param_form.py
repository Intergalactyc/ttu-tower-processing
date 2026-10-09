"""A form built from a parameter dataclass's field metadata (label, unit, kind,
limits, help), so a new jet algorithm needs no GUI code of its own.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QSpinBox, QToolButton, QWidget,
)

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.jets.identify import describe


class BoomPicker(QWidget):
    """One toggle per boom."""

    changed = Signal()

    def __init__(self, booms=BOOMS, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QToolButton:checked { background: #cfe0fb; border: 1px solid #2a5db0; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)
        self.buttons = {}
        for b in booms:
            btn = QToolButton()
            btn.setText(str(b))
            btn.setCheckable(True)
            btn.setToolTip(f"boom {b}, {HEIGHTS[b]:g} m")
            btn.setFixedWidth(24)
            btn.toggled.connect(lambda *_: self.changed.emit())
            self.buttons[b] = btn
            row.addWidget(btn)
        row.addStretch(1)

    def booms(self) -> tuple[int, ...]:
        return tuple(b for b, btn in self.buttons.items() if btn.isChecked())

    def set_booms(self, booms) -> None:
        for b, btn in self.buttons.items():
            btn.blockSignals(True)
            btn.setChecked(b in booms)
            btn.blockSignals(False)


def _spin(meta) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    lo, hi = meta.get("lo"), meta.get("hi")
    spin.setRange(-1e6 if lo is None else lo, 1e6 if hi is None else hi)
    small = hi is not None and hi <= 1
    spin.setDecimals(3 if small else 2)
    spin.setSingleStep(0.05 if small else 0.5)
    unit = meta.get("unit", "")
    if unit and len(unit) <= 4:
        spin.setSuffix(f" {unit}")
    return spin


def _label(meta) -> str:
    unit = meta.get("unit", "")
    return f"{meta['label']} ({unit})" if unit and len(unit) > 4 else meta["label"]


class ParamForm(QWidget):
    """Edits one parameter dataclass. `params()` builds an instance from the
    widgets (validation is the caller's).
    """

    changed = Signal()

    def __init__(self, params_cls, parent=None):
        super().__init__(parent)
        self.params_cls = params_cls
        self.fields = describe(params_cls)
        self.widgets = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for f in self.fields:
            kind = f["kind"]
            if kind == "booms":
                w = BoomPicker()
                w.changed.connect(self.changed)
                form.addRow(_label(f), w)
            elif kind == "int":
                w = QSpinBox()
                w.setRange(f.get("lo") or 0, f.get("hi") if f.get("hi") is not None else 10_000)
                if f.get("unit") and len(f["unit"]) <= 4:
                    w.setSuffix(f" {f['unit']}")
                w.valueChanged.connect(lambda *_: self.changed.emit())
                form.addRow(_label(f), w)
            elif kind == "float":
                w = _spin(f)
                w.valueChanged.connect(lambda *_: self.changed.emit())
                form.addRow(_label(f), w)
            elif kind == "bool":
                w = QCheckBox(_label(f))
                w.toggled.connect(lambda *_: self.changed.emit())
                form.addRow(w)
            else:  # optional_float: a check box switches it on
                box = QCheckBox(_label(f))
                spin = _spin(f)
                box.toggled.connect(spin.setEnabled)
                box.toggled.connect(lambda *_: self.changed.emit())
                spin.valueChanged.connect(lambda *_: self.changed.emit())
                w = (box, spin)
                form.addRow(box, spin)
            for part in (w if isinstance(w, tuple) else (w,)):
                part.setToolTip(f.get("help", ""))
            self.widgets[f["name"]] = w
        self.set_params(params_cls())

    def set_params(self, params) -> None:
        self.blockSignals(True)
        for f in self.fields:
            w, value = self.widgets[f["name"]], getattr(params, f["name"])
            if f["kind"] == "booms":
                w.set_booms(value)
            elif f["kind"] == "optional_float":
                box, spin = w
                box.setChecked(value is not None)
                spin.setEnabled(value is not None)
                shown = value if value is not None else f.get("suggested")
                if shown is not None:
                    spin.setValue(shown)
            elif f["kind"] == "bool":
                w.setChecked(bool(value))
            else:
                w.setValue(value)
        self.blockSignals(False)

    def params(self):
        values = {}
        for f in self.fields:
            w = self.widgets[f["name"]]
            if f["kind"] == "booms":
                values[f["name"]] = w.booms()
            elif f["kind"] == "optional_float":
                box, spin = w
                values[f["name"]] = spin.value() if box.isChecked() else None
            elif f["kind"] == "bool":
                values[f["name"]] = w.isChecked()
            else:
                values[f["name"]] = w.value()
        return self.params_cls(**values)
