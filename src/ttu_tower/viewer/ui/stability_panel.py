"""The Stability controls: which variable classifies stability (Ri_b between
two booms, or ζ at one boom), the four boundaries between the five classes,
and how many slots each class gets. Every view that uses stability follows.
Schemes can be saved under a name and loaded again, on any run.
"""
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.viewer.stability import NAMES, VARIABLES, ZETA_EDGES, SchemeStore, StabilitySpec
from ttu_tower.viewer.ui import style, tables, variant_combo


def _boom_combo() -> QComboBox:
    box = QComboBox()
    for b in BOOMS:
        box.addItem(f"b{b} ({HEIGHTS[b]:g} m)", b)
    return box


class StabilityPanel(QWidget):
    changed = Signal()  # a valid criterion was set
    resetRequested = Signal()  # back to the run's own

    def __init__(self, schemes: SchemeStore | None = None, parent=None):
        super().__init__(parent)
        self.schemes = schemes if schemes is not None else SchemeStore.open()
        self.scheme = QComboBox()
        self.scheme.setToolTip("saved schemes (every run): pick one to classify by it")
        self.scheme.activated.connect(self._on_scheme_picked)
        self.save_scheme_button = QPushButton("Save…")
        self.save_scheme_button.setToolTip("save what's set here as a named scheme")
        self.save_scheme_button.clicked.connect(lambda *_: self._ask_save())
        self.delete_scheme_button = QPushButton("Delete")
        self.delete_scheme_button.setToolTip("delete the scheme chosen beside")
        self.delete_scheme_button.clicked.connect(lambda *_: self.delete_scheme())
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
            spin.setDecimals(5)
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

        schemes = QHBoxLayout()
        schemes.addWidget(QLabel("scheme"))
        schemes.addWidget(self.scheme, stretch=1)
        schemes.addWidget(self.save_scheme_button)
        schemes.addWidget(self.delete_scheme_button)
        layout = QVBoxLayout(self)
        layout.addLayout(schemes)
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
        self._fill_schemes()

    def set_variants(self, extra) -> None:
        """The viewer's τ variants (Controls → τ) after the run's own."""
        variant_combo.refill(self.variant, [(v, v) for v in ("mrd", "naive", "mrd_unexcised")], extra)

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
        self._fill_schemes()

    def _emit(self) -> None:
        if self._quiet:
            return
        spec = self.spec()
        self._edges_for[spec.variable] = tuple(spec.edges)
        errors = spec.errors()
        self._fill_schemes()
        self.errors.setText("; ".join(errors))
        self.errors.setVisible(bool(errors))
        if not errors:
            self.changed.emit()

    # --- named schemes ------------------------------------------------------------------------

    def _fill_schemes(self) -> None:
        """The saved schemes, showing the one that is what's set (if any)."""
        if not hasattr(self, "scheme"):
            return
        current = self.schemes.name_of(self.spec())
        self.scheme.blockSignals(True)
        self.scheme.clear()
        self.scheme.addItem("(unsaved changes)" if self.schemes.items else "(no saved schemes)", None)
        for name in sorted(self.schemes.items, key=str.lower):
            self.scheme.addItem(name, name)
            self.scheme.setItemData(self.scheme.count() - 1, self.schemes.items[name].describe(),
                                    Qt.ItemDataRole.ToolTipRole)
        self.scheme.setCurrentIndex(max(self.scheme.findData(current), 0) if current else 0)
        self.scheme.blockSignals(False)
        self.delete_scheme_button.setEnabled(current is not None)

    def _on_scheme_picked(self, *_) -> None:
        name = self.scheme.currentData()
        if name is not None:
            self.load_scheme(name)

    def load_scheme(self, name: str) -> None:
        """Classify by a saved scheme (every view follows)."""
        spec = self.schemes.items[name]
        self.set_spec(spec)
        self._emit()

    def save_scheme(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("a scheme needs a name")
        errors = self.spec().errors()
        if errors:
            raise ValueError("; ".join(errors))
        self.schemes.put(name, self.spec())
        self._fill_schemes()

    def _ask_save(self) -> None:
        offered = self.scheme.currentData() or ""
        name, ok = QInputDialog.getText(self, "Save the stability scheme", "name:", text=offered)
        name = name.strip()
        if not ok or not name:
            return
        if name in self.schemes.items and self.schemes.items[name] != self.spec():
            answer = QMessageBox.question(self, "Save the stability scheme",
                                          f"Replace the saved scheme “{name}”?\n\n{self.schemes.items[name].describe()}")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.save_scheme(name)
        except ValueError as exc:
            self.errors.setText(str(exc))
            self.errors.show()

    def delete_scheme(self, name: str | None = None) -> None:
        name = name or self.scheme.currentData()
        if name is not None:
            self.schemes.remove(name)
            self._fill_schemes()

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
