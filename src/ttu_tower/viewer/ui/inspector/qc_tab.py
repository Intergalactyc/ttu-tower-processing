"""The slot's QC at a glance: its coverage ladder, what each test flagged,
its slot-level statistics against their limits, and tertiary's filtering
gates with the values they tested.
"""
import numpy as np
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ttu_tower.viewer import slotqc
from ttu_tower.viewer.ui import tables


def _box(title: str, *widgets) -> QGroupBox:
    """A titled box; the last widget takes the spare height."""
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 4, 4, 4)
    for i, w in enumerate(widgets):
        layout.addWidget(w, stretch=1 if i == len(widgets) - 1 else 0)
    return box


class QCTab(QWidget):
    def __init__(self, index, runner, parent=None):
        super().__init__(parent)
        self.index, self.runner = index, runner
        self.bundle = self.cfg = None
        self.flags = None  # the boom's flags over the detection window
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)

        self.coverage = tables.new_table()
        self.window_box = QComboBox()
        for name in slotqc.windows(0):
            self.window_box.addItem(name, name)
        self.window_box.currentIndexChanged.connect(lambda *_: self._show_flags())
        self.flag_table = tables.new_table()
        self.flag_note = QLabel("fraction of the window each test flagged; boom-level tests count on ue/vn/w/ts")
        self.slot_qc = tables.new_table()
        self.gates = tables.new_table()
        self.gate_note = QLabel()
        self.gate_note.setTextFormat(Qt.TextFormat.RichText)
        self.gate_note.setWordWrap(True)

        window_row = QWidget()
        row = QHBoxLayout(window_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("over the"))
        row.addWidget(self.window_box)
        row.addStretch(1)
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(_box("Coverage (fraction of the slot)", self.coverage))
        left.addWidget(_box("Slot statistics and their limits", self.slot_qc))
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(_box("Flagged", window_row, self.flag_note, self.flag_table))
        right.addWidget(_box("Tertiary filtering", self.gate_note, self.gates))
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(split, stretch=1)

    def set_bundle(self, bundle, cfg) -> None:
        if bundle is self.bundle:  # only the variant changed; nothing here depends on it
            return
        self.bundle, self.cfg, self.flags = bundle, cfg, None
        self._show_coverage()
        self._show_slot_qc()
        self.flag_table.clear()
        self.gates.clear()
        self.gate_note.setText("reading flags…")
        g_lo, g_hi = slotqc.windows(bundle.slot)["detection window (80 min)"]
        self.runner.submit(self.index.flags_window, bundle.boom, g_lo, g_hi, key=f"qc-{id(self)}",
                           label="reading flags", on_done=lambda flags, b=bundle: self._flags_loaded(b, flags),
                           on_error=lambda exc, tb: shiboken6.isValid(self) and self.gate_note.setText(
                               f"could not read flags: {exc}"))

    def _flags_loaded(self, bundle, flags) -> None:
        if not shiboken6.isValid(self) or bundle is not self.bundle:  # the inspector closed, or moved on
            return
        self.flags = flags
        self._show_flags()
        self._show_gates()

    # --- sections -------------------------------------------------------------------------------

    def _show_coverage(self) -> None:
        b, cfg = self.bundle, self.cfg
        ladder = slotqc.coverage_ladder(b.table("coverage"))
        limit = cfg.tertiary.min_coverage if cfg is not None else None

        def color(var, layer, v):
            if np.isnan(v):
                return tables.MUTED
            if layer == "usable" and limit is not None and v < limit:
                return tables.BAD
            return None

        tables.fill(self.coverage, ladder, index=True, color=color, fmt=lambda c, v: "" if np.isnan(v) else f"{v:.3f}")
        status = b.table("slot_boom")
        parts = [f"slot status <b>{status['status'].iloc[0]}</b>" if not status.empty else "no slot_boom row"]
        if cfg is not None:
            parts.append(f"usable coverage below tertiary's min_coverage {cfg.tertiary.min_coverage:g} is red; "
                         f"primary computes a slot's statistics above qc.min_coverage {cfg.qc.min_coverage:g}")
        self.info.setText(" · ".join(parts))

    def _show_slot_qc(self) -> None:
        if self.cfg is None:
            return
        checks = slotqc.slot_qc_checks(self.bundle.table("slot_qc"), self.cfg.qc)
        tables.fill(self.slot_qc, checks)
        for i, r in enumerate(checks.itertuples(index=False)):
            if r.tripped:
                for j in range(self.slot_qc.columnCount()):
                    self.slot_qc.item(i, j).setBackground(tables.BAD if r.removes else tables.WARN)
        self.slot_qc.setToolTip("red: tripped a test that makes the slot's samples unusable; "
                                "orange: tripped a test the run only records")

    def _show_flags(self) -> None:
        if self.flags is None or self.bundle is None:
            return
        start, end = slotqc.windows(self.bundle.slot)[self.window_box.currentData()]
        fractions = slotqc.flag_fractions(self.flags, self.bundle.boom, start, end)
        unusable = set(self.cfg.qc.unusable_tests) if self.cfg is not None else set()

        def color(test, var, v):
            if np.isnan(v):
                return tables.MUTED
            if v > 0:
                return tables.BAD if test in unusable or test in ("bounds", "spike") else tables.WARN
            return None

        def fmt(_, v):
            return "" if np.isnan(v) else ("0" if v == 0 else f"{100 * v:.3g} %")

        tables.fill(self.flag_table, fractions, index=True, color=color, fmt=fmt)

    def _show_gates(self) -> None:
        b, cfg = self.bundle, self.cfg
        if cfg is None or "boom_stats" not in b.tables:
            self.gate_note.setText("this run has no secondary/tertiary stage")
            return
        gates = slotqc.filter_gates(b.slot, b.boom, b.table("means"), b.table("slow"), b.table("boom_stats"),
                                    b.table("coverage"), b.table("tau_selected"), self.flags, cfg.tertiary)
        shown = gates.copy()
        shown["result"] = np.where(shown["failed"], "FAIL", "ok")
        shown = shown.drop(columns=["failed"])

        def fmt(col, v):
            return f"{v:.4g}" if col in ("value", "limit") and isinstance(v, float) else None

        tables.fill(self.gates, shown, fmt=fmt)
        for i, failed in enumerate(gates["failed"]):
            if failed:
                for j in range(self.gates.columnCount()):
                    self.gates.item(i, j).setBackground(tables.BAD)
        recomputed = slotqc.logged_failures(gates)
        stored = slotqc.stored_failures(b.table("filter_log"), b.boom)
        n_fail = len({(v, g) for v, g, _, _ in recomputed})
        text = (f"{n_fail} (variant, group) value set{'s' if n_fail != 1 else ''} filtered; "
                f"limits: coverage ≥ {cfg.tertiary.min_coverage:g}, bounds ≤ {cfg.tertiary.max_bounds_fraction:g}, "
                f"spike ≤ {cfg.tertiary.max_spike_fraction:g} of the slot (±5 min where τ = 1200 s)")
        if "filter_log" in b.tables and recomputed != stored:
            text += " · <span style='color:#b00'>differs from the stored filter_log: " \
                    f"only recomputed {sorted(recomputed - stored, key=str)}, only stored {sorted(stored - recomputed, key=str)}</span>"
        self.gate_note.setText(text)
