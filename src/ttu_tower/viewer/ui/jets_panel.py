"""The Jets controls: which algorithm and its parameters (the form is built
from the algorithm's own description), how slots join into events, the event
filters, and what the result was - slots per rule, events, and how many the
filters keep.
"""
from PySide6.QtCore import QTime, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QTimeEdit, QVBoxLayout, QWidget,
)

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.jets.analysis import JetSettings
from ttu_tower.jets.events import EventParams
from ttu_tower.jets.filters import JetFilters
from ttu_tower.jets.identify import ALGORITHMS
from ttu_tower.viewer.ui.param_form import ParamForm

_STATUS_TEXT = {"no_data": "no data", "no_peak": "no peak", "too_slow": "too slow", "weak_falloff": "weak falloff",
                "jet": "jet"}
_FILTER_TEXT = {"time_of_day": "time of day", "night": "night", "inversion": "ΔVPT", "richardson": "Ri_b"}


def _boom_combo(boom: int) -> QComboBox:
    box = QComboBox()
    for b in BOOMS:
        box.addItem(f"b{b}", b)
        box.setItemData(box.count() - 1, f"{HEIGHTS[b]:g} m", Qt.ItemDataRole.ToolTipRole)
    box.setCurrentIndex(box.findData(boom))
    return box


def _value_spin(lo, hi, step, decimals, suffix="") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    return spin


class FiltersBox(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Filters (on whole events)", parent)
        grid = QGridLayout(self)
        self.time_box = QCheckBox("time of day, at the start")
        self.time_from, self.time_to = QTimeEdit(QTime(18, 0)), QTimeEdit(QTime(9, 0))
        for t in (self.time_from, self.time_to):
            t.setDisplayFormat("HH:mm")
        times = QHBoxLayout()
        times.addWidget(self.time_from)
        times.addWidget(QLabel("–"))
        times.addWidget(self.time_to)
        times.addStretch(1)
        grid.addWidget(self.time_box, 0, 0)
        grid.addLayout(times, 0, 1)
        self.night_box = QCheckBox("night only (sun down at the start)")
        grid.addWidget(self.night_box, 1, 0, 1, 2)

        self.inv_box = QCheckBox("ΔVPT, event mean ≥")
        self.inv_value = _value_spin(-20.0, 40.0, 0.5, 2, " K")
        self.inv_value.setValue(1.0)
        self.inv_low, self.inv_high = _boom_combo(2), _boom_combo(10)
        inv_booms = QHBoxLayout()
        inv_booms.addWidget(QLabel("from"))
        inv_booms.addWidget(self.inv_low)
        inv_booms.addWidget(QLabel("to"))
        inv_booms.addWidget(self.inv_high)
        inv_booms.addStretch(1)
        grid.addWidget(self.inv_box, 2, 0)
        grid.addWidget(self.inv_value, 2, 1)
        grid.addLayout(inv_booms, 3, 1)

        self.rib_low, self.rib_high = _boom_combo(2), _boom_combo(4)
        rib_booms = QHBoxLayout()
        rib_booms.addWidget(QLabel("from"))
        rib_booms.addWidget(self.rib_low)
        rib_booms.addWidget(QLabel("to"))
        rib_booms.addWidget(self.rib_high)
        rib_booms.addStretch(1)
        self.rib_min_box, self.rib_max_box = QCheckBox("Ri_b median ≥"), QCheckBox("Ri_b median ≤")
        self.rib_min, self.rib_max = _value_spin(-10.0, 10.0, 0.05, 2), _value_spin(-10.0, 10.0, 0.05, 2)
        self.rib_max.setValue(0.25)
        grid.addWidget(QLabel("Ri_b"), 4, 0)
        grid.addLayout(rib_booms, 4, 1)
        grid.addWidget(self.rib_min_box, 5, 0)
        grid.addWidget(self.rib_min, 5, 1)
        grid.addWidget(self.rib_max_box, 6, 0)
        grid.addWidget(self.rib_max, 6, 1)

        self.time_box.setToolTip("keep events that start inside this clock window (in the timelines' time zone); "
                                 "it may wrap midnight")
        self.inv_box.setToolTip("vpt(upper) − vpt(lower), slow-sensor virtual potential temperature, averaged "
                                "over the event's detected slots; positive is stable")
        tip = "bulk Richardson number between the two booms, the median over the event's detected slots"
        for w in (self.rib_min_box, self.rib_max_box):
            w.setToolTip(tip)
        for box, parts in ((self.time_box, (self.time_from, self.time_to)), (self.inv_box, (self.inv_value,)),
                           (self.rib_min_box, (self.rib_min,)), (self.rib_max_box, (self.rib_max,))):
            for p in parts:
                box.toggled.connect(p.setEnabled)
                p.setEnabled(False)
        for w in (self.time_box, self.night_box, self.inv_box, self.rib_min_box, self.rib_max_box):
            w.toggled.connect(lambda *_: self.changed.emit())
        for w in (self.time_from, self.time_to):
            w.timeChanged.connect(lambda *_: self.changed.emit())
        for w in (self.inv_value, self.rib_min, self.rib_max):
            w.valueChanged.connect(lambda *_: self.changed.emit())
        for w in (self.inv_low, self.inv_high, self.rib_low, self.rib_high):
            w.currentIndexChanged.connect(lambda *_: self.changed.emit())

    def filters(self) -> JetFilters:
        return JetFilters(
            time_window=(self.time_from.time().toString("HH:mm"), self.time_to.time().toString("HH:mm"))
            if self.time_box.isChecked() else None,
            night_only=self.night_box.isChecked(),
            inversion_booms=(self.inv_low.currentData(), self.inv_high.currentData()),
            min_inversion=self.inv_value.value() if self.inv_box.isChecked() else None,
            rib_booms=(self.rib_low.currentData(), self.rib_high.currentData()),
            rib_min=self.rib_min.value() if self.rib_min_box.isChecked() else None,
            rib_max=self.rib_max.value() if self.rib_max_box.isChecked() else None,
        )

    def set_filters(self, f: JetFilters) -> None:
        self.blockSignals(True)
        self.time_box.setChecked(f.time_window is not None)
        if f.time_window is not None:
            self.time_from.setTime(QTime.fromString(f.time_window[0], "HH:mm"))
            self.time_to.setTime(QTime.fromString(f.time_window[1], "HH:mm"))
        self.night_box.setChecked(f.night_only)
        self.inv_box.setChecked(f.min_inversion is not None)
        if f.min_inversion is not None:
            self.inv_value.setValue(f.min_inversion)
        for box, b in ((self.inv_low, f.inversion_booms[0]), (self.inv_high, f.inversion_booms[1]),
                       (self.rib_low, f.rib_booms[0]), (self.rib_high, f.rib_booms[1])):
            box.setCurrentIndex(max(box.findData(b), 0))
        for box, spin, v in ((self.rib_min_box, self.rib_min, f.rib_min), (self.rib_max_box, self.rib_max, f.rib_max)):
            box.setChecked(v is not None)
            if v is not None:
                spin.setValue(v)
        self.blockSignals(False)


class JetsPanel(QWidget):
    changed = Signal()  # any setting; the window reruns the detection
    selectRequested = Signal()
    stepRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.algorithm = QComboBox()
        for key, algorithm in ALGORITHMS.items():
            self.algorithm.addItem(algorithm.label, key)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color:#555;")
        self.params_box = QGroupBox("Detection, per slot")
        self.params_layout = QVBoxLayout(self.params_box)
        self.form: ParamForm | None = None
        self.event_form = ParamForm(EventParams)
        events_box = QGroupBox("Events")
        QVBoxLayout(events_box).addWidget(self.event_form)
        self.filters = FiltersBox()
        self.errors = QLabel()
        self.errors.setWordWrap(True)
        self.errors.setStyleSheet("color:#b00020;")
        self.errors.hide()
        self.result = QLabel("open a run with a tertiary stage")
        self.result.setWordWrap(True)
        result_box = QGroupBox("Result")
        QVBoxLayout(result_box).addWidget(self.result)
        self.select_button = QPushButton("Select jet slots")
        self.select_button.setToolTip("brush the slots of the events the filters keep, so every Summary tab "
                                      "compares them with the rest")
        self.step_button = QPushButton("Step through events")
        self.step_button.setToolTip("open the Slot Inspector's Profile tab on each kept event's fastest slot, "
                                    "at its peak boom")
        self.defaults_button = QPushButton("Defaults")
        self.defaults_button.setToolTip("this algorithm's default parameters, default events, no filters")
        buttons = QGridLayout()
        buttons.addWidget(self.select_button, 0, 0)
        buttons.addWidget(self.step_button, 0, 1)
        buttons.addWidget(self.defaults_button, 1, 1)

        content = QWidget()
        layout = QVBoxLayout(content)
        head = QHBoxLayout()
        head.addWidget(QLabel("algorithm"))
        head.addWidget(self.algorithm, stretch=1)
        layout.addLayout(head)
        layout.addWidget(self.description)
        for w in (self.params_box, events_box, self.filters, self.errors, result_box):
            layout.addWidget(w)
        layout.addLayout(buttons)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # rows wrap instead
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._set_algorithm_form(self.algorithm.currentData())
        self.algorithm.currentIndexChanged.connect(self._on_algorithm)
        self.event_form.changed.connect(self.changed)
        self.filters.changed.connect(self.changed)
        self.select_button.clicked.connect(self.selectRequested)
        self.step_button.clicked.connect(self.stepRequested)
        self.defaults_button.clicked.connect(self._defaults)
        self.show_actions(False)

    def _set_algorithm_form(self, key: str, params=None) -> None:
        if self.form is not None:
            self.params_layout.removeWidget(self.form)
            self.form.hide()
            self.form.deleteLater()
        algorithm = ALGORITHMS[key]
        self.form = ParamForm(algorithm.params)
        if params is not None:
            self.form.set_params(params)
        self.form.changed.connect(self.changed)
        self.params_layout.addWidget(self.form)
        self.description.setText(algorithm.description)

    def _on_algorithm(self, *_) -> None:
        self._set_algorithm_form(self.algorithm.currentData())
        self.changed.emit()

    def _defaults(self) -> None:
        self.set_settings(JetSettings(algorithm=self.algorithm.currentData()))
        self.changed.emit()

    def settings(self) -> JetSettings:
        return JetSettings(algorithm=self.algorithm.currentData(), params=self.form.params(),
                           events=self.event_form.params(), filters=self.filters.filters())

    def set_settings(self, s: JetSettings) -> None:
        self.algorithm.blockSignals(True)
        self.algorithm.setCurrentIndex(max(self.algorithm.findData(s.algorithm), 0))
        self.algorithm.blockSignals(False)
        self._set_algorithm_form(s.algorithm, s.algorithm_params())
        self.event_form.set_params(s.events)
        self.filters.set_filters(s.filters)

    def show_errors(self, errors: list[str]) -> None:
        self.errors.setText("\n".join(errors))
        self.errors.setVisible(bool(errors))

    def show_actions(self, enabled: bool) -> None:
        self.select_button.setEnabled(enabled)
        self.step_button.setEnabled(enabled)

    def show_message(self, text: str) -> None:
        self.result.setText(text)
        self.show_actions(False)

    def show_counts(self, counts: dict) -> None:
        status = counts["status"]
        with_data = sum(n for s, n in status.items() if s != "no_data")
        rejected = " · ".join(f"{_STATUS_TEXT.get(s, s)} {n:,}" for s, n in status.items()
                              if s not in ("no_data", "jet"))
        lines = [f"slots with the data needed: <b>{with_data:,}</b> (no data: {status.get('no_data', 0):,})",
                 f"rejected: {rejected}",
                 f"jet slots <b>{counts['jet_slots']:,}</b> → <b>{counts['events']:,}</b> events"
                 f" → <b>{counts['passed']:,}</b> kept by the filters"]
        for f, n in counts["failed"].items():
            missing = counts["no_value"].get(f, 0)
            lines.append(f"&nbsp;&nbsp;{_FILTER_TEXT[f]} drops {n:,}" + (f" ({missing:,} without a value)"
                                                                       if missing else ""))
        self.result.setText("<br>".join(lines))
        self.show_actions(counts["passed"] > 0)
