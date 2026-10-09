"""The Stability controls: which variable classifies stability (Ri_b between
two booms, or ζ at one boom), the four boundaries between the five classes,
and how many slots each class gets. Every view that uses stability follows.
"""
import numpy as np
import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.viewer.stability import NAMES, VARIABLES, ZETA_EDGES, StabilitySpec
from ttu_tower.viewer.ui import style, tables


def _boom_combo() -> QComboBox:
    box = QComboBox()
    for b in BOOMS:
        box.addItem(f"b{b} ({HEIGHTS[b]:g} m)", b)
    return box


class StabilityPanel(QWidget):
    changed = Signal()  # a valid criterion was set
    resetRequested = Signal()  # back to the run's own

    def __init__(self, parent=None):
        super().__init__(parent)
        self.variable = QComboBox()
        for key, label in VARIABLES.items():
            self.variable.addItem(label, key)
        self.boom_a, self.boom_b = _boom_combo(), _boom_combo()
        self.variant = QComboBox()
        for v in ("mrd", "naive", "mrd_unexcised"):
            self.variant.addItem(v, v)
        self.variant.setToolTip("ζ is τ-dependent: which variant's value classifies")
        self.edges = []
        for _ in range(4):
            spin = QDoubleSpinBox()
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(3)
            spin.setKeyboardTracking(False)  # reclassify once a value is entered, not on every keystroke
            self.edges.append(spin)
        self.errors = QLabel()
        self.errors.setStyleSheet("color:#b00020;")
        self.errors.setWordWrap(True)
        self.errors.hide()
        self.in_use = QLabel()
        self.in_use.setWordWrap(True)
        self.reset = QPushButton("The run's own")
        self.reset.setToolTip("back to the run's configured classification (post.stability)")
        self.counts = tables.new_table()

        criterion = QGroupBox("Classify by")
        grid = QGridLayout(criterion)
        grid.addWidget(QLabel("variable"), 0, 0)
        grid.addWidget(self.variable, 0, 1)
        self.boom_label = QLabel()
        grid.addWidget(self.boom_label, 1, 0)
        booms = QHBoxLayout()
        booms.addWidget(self.boom_a)
        self.and_label = QLabel("and")
        booms.addWidget(self.and_label)
        booms.addWidget(self.boom_b)
        booms.addWidget(self.variant)
        booms.addStretch(1)
        grid.addLayout(booms, 1, 1)

        bounds = QGroupBox("Classes, and the boundaries between them")
        rows = QGridLayout(bounds)
        for i, name in enumerate(NAMES):  # a class, then the boundary to the next above it
            rows.addWidget(QLabel(f"<span style='color:{style.stability_color(name, i).name()}'>■</span> {name}"),
                           2 * i, 0, 1, 2)
            if i < 4:
                rows.addWidget(QLabel("    boundary"), 2 * i + 1, 0)
                rows.addWidget(self.edges[i], 2 * i + 1, 1)
        bounds.setToolTip("each class runs from its lower boundary (included) up to the next (excluded)")

        layout = QVBoxLayout(self)
        layout.addWidget(criterion)
        layout.addWidget(bounds)
        layout.addWidget(self.errors)
        layout.addWidget(self.in_use)
        layout.addWidget(self.reset)
        layout.addWidget(QLabel("slots per class, the whole period:"))
        layout.addWidget(self.counts)
        layout.addStretch(1)

        self._edges_for = {"rib": (-0.05, -0.01, 0.01, 0.1), "zeta": ZETA_EDGES}  # the last used, per variable
        self._quiet = False
        self.variable.currentIndexChanged.connect(self._on_variable)
        for w in (self.boom_a, self.boom_b, self.variant):
            w.currentIndexChanged.connect(lambda *_: self._emit())
        for spin in self.edges:
            spin.valueChanged.connect(lambda *_: self._emit())
        self.reset.clicked.connect(self.resetRequested)
        self.set_spec(StabilitySpec())

    def _on_variable(self, *_) -> None:
        if not self._quiet:
            self._set_edges(self._edges_for[self.variable.currentData()])
        self._show_variable()
        self._emit()

    def _show_variable(self) -> None:
        rib = self.variable.currentData() == "rib"
        self.boom_label.setText("between" if rib else "at")
        self.and_label.setVisible(rib)
        self.boom_b.setVisible(rib)
        self.variant.setVisible(not rib)
        for spin in self.edges:
            spin.setSingleStep(0.01 if rib else 0.05)

    def _set_edges(self, edges) -> None:
        for spin, v in zip(self.edges, edges):
            spin.blockSignals(True)
            spin.setValue(float(v))
            spin.blockSignals(False)

    def spec(self) -> StabilitySpec:
        rib = self.variable.currentData() == "rib"
        booms = (self.boom_a.currentData(), self.boom_b.currentData() if rib else self.boom_a.currentData())
        return StabilitySpec(self.variable.currentData(), booms, self.variant.currentData(),
                             tuple(spin.value() for spin in self.edges))

    def set_spec(self, spec: StabilitySpec) -> None:
        self._quiet = True
        self.variable.setCurrentIndex(max(self.variable.findData(spec.variable), 0))
        a, b = sorted(spec.booms) if spec.variable == "rib" else spec.booms
        self.boom_a.setCurrentIndex(max(self.boom_a.findData(a), 0))
        self.boom_b.setCurrentIndex(max(self.boom_b.findData(b), 0))
        self.variant.setCurrentIndex(max(self.variant.findData(spec.variant), 0))
        self._set_edges(spec.edges)
        self._edges_for[spec.variable] = tuple(spec.edges)
        self._show_variable()
        self.errors.hide()
        self._quiet = False

    def _emit(self) -> None:
        if self._quiet:
            return
        spec = self.spec()
        self._edges_for[spec.variable] = tuple(spec.edges)
        errors = spec.errors()
        self.errors.setText("; ".join(errors))
        self.errors.setVisible(bool(errors))
        if not errors:
            self.changed.emit()

    def show_in_use(self, text: str) -> None:
        self.in_use.setText(text)

    def show_counts(self, stability) -> None:
        if stability is None:
            self.counts.setRowCount(0)
            return
        curve, names = stability
        finite = np.isfinite(curve.y)
        n = np.bincount(curve.y[finite].astype(int), minlength=len(names))
        frame = pd.DataFrame({"slots": n, "%": 100.0 * n / max(int(finite.sum()), 1)}, index=list(names))
        tables.fill(self.counts, frame, index=True, fmt=lambda col, v: f"{v:.1f}" if col == "%" else None)
        tables.fit_rows(self.counts, max_rows=len(names))
