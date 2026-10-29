"""The τ controls: named τ schemes - the data (excised or not), a base
selection (the run's own, or what-if detection and selection settings), and
optional overrides by stability class, classified by a stability scheme (per
slot) or locally by each boom's own ζ and heat flux. Every saved scheme, and
a fixed τ at each ladder rung, is a variant in every variant menu; each is
computed for a run the first time it's shown, and kept.
"""
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QMessageBox, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from ttu_tower.secondary.detect import SMOOTHING
from ttu_tower.viewer import stability as criteria
from ttu_tower.viewer import tauvariants as tv
from ttu_tower.viewer import whatif
from ttu_tower.viewer.ui import style, tables, variant_combo
from ttu_tower.viewer.ui.tau_view import _PRIORITIES, _RULES, _rung_box

_CURRENT = "__current__"


def _choice_box() -> QComboBox:
    box = QComboBox()
    for c in tv.choices():
        box.addItem(tv.choice_label(c), c)
    return box


class TauPanel(QWidget):
    schemesChanged = Signal(list)  # names saved or deleted
    computeRequested = Signal(str)  # a variant to compute now
    useRequested = Signal(str)  # a variant to show in every view
    clearRequested = Signal()  # forget this run's computed variants

    def __init__(self, schemes: tv.SchemeStore, stability_schemes=None, current_stability=None, parent=None):
        """`stability_schemes`: the saved stability schemes (viewer.stability.SchemeStore);
        `current_stability`: () -> the StabilitySpec in use (Controls → Stability), or None.
        """
        super().__init__(parent)
        self.schemes = schemes
        self.stability_schemes = stability_schemes
        self.current_stability = current_stability or (lambda: None)
        self.run = self.variants = None
        self._quiet = False

        self.scheme = QComboBox()
        self.scheme.setToolTip("saved τ schemes (every run): each is a variant in every variant menu")
        self.scheme.activated.connect(lambda *_: self._on_scheme_picked())
        self.save_button = QPushButton("Save…")
        self.save_button.setToolTip("save what's set here as a named τ scheme")
        self.save_button.clicked.connect(lambda *_: self._ask_save())
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(lambda *_: self.delete_scheme())

        self.data = QComboBox()
        self.data.addItem("excised (mrd)", "mrd")
        self.data.addItem("unexcised (mrd unexcised)", "mrd_unexcised")
        self.data.setToolTip("which series the statistics come from: the final usable mask, or nothing removed")

        self.base = QComboBox()
        self.base.addItem("the run's selection", "run")
        self.base.addItem("a what-if of the settings below", "whatif")
        self.base.setToolTip("the τ every slot gets unless a stability class overrides it")
        f = self.fields = {"peak_significance_se": QDoubleSpinBox(), "min_scale_s": QDoubleSpinBox(),
                           "min_tau_s": _rung_box(), "max_tau_s": _rung_box()}
        f["peak_significance_se"].setRange(0.0, 50.0)
        f["peak_significance_se"].setSingleStep(0.25)
        f["min_scale_s"].setRange(0.05, 9.0)
        f["min_scale_s"].setSingleStep(0.1)
        f["min_scale_s"].setDecimals(3)
        self.smoothing = QComboBox()
        self.smoothing.setEditable(True)
        for text, weights in whatif.KERNELS:
            self.smoothing.addItem(text, weights)
        self.rule, self.priority, self.fallback = QComboBox(), QComboBox(), _rung_box()
        for r in _RULES:
            self.rule.addItem(r, r)
        for p in _PRIORITIES:
            self.priority.addItem(", ".join(p), ",".join(p))
        self.copy_whatif = QPushButton("Copy from Summary → τ")
        self.copy_whatif.setToolTip("take the detection and selection settings of the Summary's τ what-if")
        self.whatif_box = QGroupBox("What-if settings")
        form = QFormLayout(self.whatif_box)
        form.addRow("peak significance [SE]", f["peak_significance_se"])
        form.addRow("min scale", f["min_scale_s"])
        form.addRow("min τ", f["min_tau_s"])
        form.addRow("max τ", f["max_tau_s"])
        form.addRow("smoothing", self.smoothing)
        form.addRow("rule", self.rule)
        form.addRow("priority", self.priority)
        form.addRow("fallback τ", self.fallback)
        form.addRow(self.copy_whatif)

        self.classifier = QComboBox()
        self.classifier.addItem("none: every slot takes the choice below", "none")
        self.classifier.addItem("a stability scheme (per slot)", "global")
        self.classifier.addItem("local ζ and heat flux (per boom)", "local")
        self.global_source = QComboBox()
        self.global_source.setToolTip("the stability scheme copied into this τ scheme (later edits to it don't "
                                      "change this one until you copy again)")
        self.copy_global = QPushButton("Copy")
        self.copy_global.setToolTip("copy the chosen stability scheme into this τ scheme")
        self.global_note = QLabel()
        self.global_note.setWordWrap(True)
        self._stability_copy: tuple = (None, "")  # (StabilitySpec, where it came from)
        self.local_variable = QComboBox()
        for key, text in tv.LOCAL_VARIABLES.items():
            self.local_variable.addItem(text, key)
        self.local_variable.setToolTip("each boom's own: ζ = z/L split by the heat flux's sign, or the gradient "
                                       "Richardson number Ri_g at the boom, split by its own sign")
        self.local_zeta0, self.local_percent = QRadioButton("neutral where |x| ≤"), QRadioButton(
            "neutral: each boom's share nearest 0:")
        self.local_mode = QButtonGroup(self)
        self.local_mode.addButton(self.local_zeta0)
        self.local_mode.addButton(self.local_percent)
        self.zeta0 = QDoubleSpinBox()
        self.zeta0.setRange(1e-5, 100.0)
        self.zeta0.setDecimals(5)
        self.zeta0.setSingleStep(0.01)
        self.percent = QDoubleSpinBox()
        self.percent.setRange(0.1, 99.9)
        self.percent.setDecimals(1)
        self.percent.setSuffix(" %")
        self.zeta_variant = QComboBox()
        self.zeta_variant.setToolTip("whose ζ and heat flux (w'θv') classify: ζ depends on τ too")
        self.local_note = QLabel("non-neutral: unstable where the heat flux is upward (w'θv' > 0), stable where "
                                 "it's downward")
        self.local_note.setWordWrap(True)

        self.global_row = QWidget()
        g = QHBoxLayout(self.global_row)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(QLabel("scheme"))
        g.addWidget(self.global_source, stretch=1)
        g.addWidget(self.copy_global)
        self.local_rows = QWidget()
        lg = QGridLayout(self.local_rows)
        lg.setContentsMargins(0, 0, 0, 0)
        self.zeta_variant_label = QLabel("ζ and heat flux of")
        lg.addWidget(QLabel("by"), 0, 0)
        lg.addWidget(self.local_variable, 0, 1)
        lg.addWidget(self.zeta_variant_label, 1, 0)
        lg.addWidget(self.zeta_variant, 1, 1)
        lg.addWidget(self.local_zeta0, 2, 0)
        lg.addWidget(self.zeta0, 2, 1)
        lg.addWidget(self.local_percent, 3, 0)
        lg.addWidget(self.percent, 3, 1)
        lg.addWidget(self.local_note, 4, 0, 1, 2)

        self.class_rows = QGridLayout()
        self.class_labels: list[QLabel] = []
        self.class_choices: list[QComboBox] = []
        for i in range(len(criteria.NAMES)):
            lab, box = QLabel(), _choice_box()
            self.class_rows.addWidget(lab, i, 0)
            self.class_rows.addWidget(box, i, 1)
            self.class_labels.append(lab)
            self.class_choices.append(box)
        self.otherwise_label = QLabel("otherwise")
        self.otherwise = _choice_box()
        self.class_rows.addWidget(self.otherwise_label, len(criteria.NAMES), 0)
        self.class_rows.addWidget(self.otherwise, len(criteria.NAMES), 1)

        overrides = QGroupBox("τ by stability class")
        ov = QVBoxLayout(overrides)
        top = QHBoxLayout()
        top.addWidget(QLabel("classify by"))
        top.addWidget(self.classifier, stretch=1)
        ov.addLayout(top)
        ov.addWidget(self.global_row)
        ov.addWidget(self.global_note)
        ov.addWidget(self.local_rows)
        ov.addLayout(self.class_rows)

        self.errors = QLabel()
        self.errors.setStyleSheet("color:#b00020;")
        self.errors.setWordWrap(True)
        self.state = QLabel()
        self.state.setWordWrap(True)
        self.compute_button = QPushButton("Compute now")
        self.compute_button.setToolTip("compute the chosen scheme for this run (otherwise it is computed the "
                                       "first time a view shows it)")
        self.compute_button.clicked.connect(lambda *_: self._request(self.computeRequested))
        self.use_button = QPushButton("Use in all views")
        self.use_button.setToolTip("switch every variant menu that offers it to the chosen scheme")
        self.use_button.clicked.connect(lambda *_: self._request(self.useRequested))
        self.counts = tables.new_table()
        self.counts.setToolTip("% of each boom's rows by the rule that chose its τ, and the τ chosen")
        self.disk = QLabel()
        self.clear_button = QPushButton("Clear computed")
        self.clear_button.setToolTip("delete this run's computed τ variants (they're computed again when shown)")
        self.clear_button.clicked.connect(lambda *_: self.clearRequested.emit())

        schemes_row = QHBoxLayout()
        schemes_row.addWidget(QLabel("scheme"))
        schemes_row.addWidget(self.scheme, stretch=1)
        schemes_row.addWidget(self.save_button)
        schemes_row.addWidget(self.delete_button)
        basics = QFormLayout()
        basics.addRow("data", self.data)
        basics.addRow("base selection", self.base)
        actions = QHBoxLayout()
        actions.addWidget(self.compute_button)
        actions.addWidget(self.use_button)
        actions.addStretch(1)
        cache = QHBoxLayout()
        cache.addWidget(self.disk, stretch=1)
        cache.addWidget(self.clear_button)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Fixed τ at every rung (τ = 9.375 s … 20 min) is in every variant menu; a saved "
                                "scheme is too."))
        layout.itemAt(0).widget().setWordWrap(True)
        layout.addLayout(schemes_row)
        layout.addLayout(basics)
        layout.addWidget(self.whatif_box)
        layout.addWidget(overrides)
        layout.addWidget(self.errors)
        layout.addWidget(self.state)
        layout.addLayout(actions)
        layout.addWidget(self.counts)
        layout.addLayout(cache)
        layout.addStretch(1)

        widgets = [self.data, self.base, *f.values(), self.smoothing, self.rule, self.priority, self.fallback,
                   self.classifier, self.local_variable, self.zeta_variant, *self.class_choices, self.otherwise]
        for w in widgets:
            signal = w.valueChanged if isinstance(w, QDoubleSpinBox) else w.currentIndexChanged
            signal.connect(lambda *_: self._edited())
        self.smoothing.currentTextChanged.connect(lambda *_: self._edited())
        for w in (self.zeta0, self.percent):
            w.valueChanged.connect(lambda *_: self._edited())
        self.local_mode.buttonToggled.connect(lambda *_: self._edited())
        self.copy_global.clicked.connect(lambda *_: self.copy_stability())
        for box in self.findChildren(QComboBox):  # narrow docks: menus shrink rather than widen the panel
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(8)
        self.local_percent.setChecked(True)
        self.zeta0.setValue(0.05)
        self.percent.setValue(20.0)
        self.set_variants([])
        self._fill_global_sources()
        self.set_scheme(tv.TauScheme())
        self._fill_schemes()

    # --- the run ------------------------------------------------------------------------------------

    def set_run(self, run, variants) -> None:
        self.run, self.variants = run, variants
        self._reset_whatif_fields()
        self._edited()

    def set_variants(self, extra) -> None:
        """Variants whose ζ can classify locally: the run's own and the fixed rungs."""
        variant_combo.fill(self.zeta_variant, [(v, variant_combo.label(v)) for v in tv.STORED],
                           [k for k in extra if not k.startswith(tv.SCHEME_PREFIX)])

    def _reset_whatif_fields(self) -> None:
        if self.run is None or self.run.cfg is None:
            return
        det, sel = self.run.cfg.secondary.detection, self.run.cfg.secondary.selection
        self._put_whatif({k: getattr(det, k) for k in tv.DETECTION_FIELDS}, SMOOTHING,
                         {"rule": sel.rule, "priority": ",".join(sel.priority), "fallback_tau_s": sel.fallback_tau_s})

    def _put_whatif(self, detection: dict, kernel, selection: dict) -> None:
        f = self.fields
        self._quiet = True
        if "peak_significance_se" in detection:
            f["peak_significance_se"].setValue(float(detection["peak_significance_se"]))
        if "min_scale_s" in detection:
            f["min_scale_s"].setValue(float(detection["min_scale_s"]))
        for name in ("min_tau_s", "max_tau_s"):
            if name in detection:
                f[name].setCurrentIndex(max(f[name].findData(float(detection[name])), 0))
        text = next((t for t, w in whatif.KERNELS if tuple(w) == tuple(kernel)), whatif.kernel_text(kernel))
        self.smoothing.setEditText(text)
        for box, key in ((self.rule, "rule"), (self.priority, "priority"), (self.fallback, "fallback_tau_s")):
            if key in selection:
                value = float(selection[key]) if key == "fallback_tau_s" else str(selection[key])
                box.setCurrentIndex(max(box.findData(value), 0))
        self._quiet = False

    def copy_from(self, tau_view) -> None:
        """The Summary τ what-if's settings."""
        chosen, problem = tau_view.edited_detection()
        rules = tau_view.what_if_rules()
        if chosen is None:
            self.errors.setText(f"Summary → τ: {problem}")
            self.errors.show()
            return
        det, kernel = chosen
        self.base.setCurrentIndex(self.base.findData("whatif"))
        self._put_whatif({k: getattr(det, k) for k in tv.DETECTION_FIELDS}, kernel,
                         {"rule": rules.rule, "priority": ",".join(rules.priority),
                          "fallback_tau_s": rules.fallback_tau_s})
        self._edited()

    # --- the scheme as set ----------------------------------------------------------------------

    def scheme_now(self) -> tuple[tv.TauScheme | None, str]:
        """(the scheme set here, "") or (None, why it can't be made)."""
        base = self.base.currentData()
        detection, kernel, selection = (), SMOOTHING, ()
        if base == "whatif":
            f = self.fields
            detection = tuple(sorted({"peak_significance_se": f["peak_significance_se"].value(),
                                      "min_scale_s": f["min_scale_s"].value(),
                                      "min_tau_s": f["min_tau_s"].currentData(),
                                      "max_tau_s": f["max_tau_s"].currentData()}.items()))
            kernel, problem = whatif.parse_kernel(self.smoothing.currentText())
            if kernel is None:
                return None, problem
            selection = tuple(sorted({"rule": self.rule.currentData(), "priority": self.priority.currentData(),
                                      "fallback_tau_s": self.fallback.currentData()}.items()))
        classifier = self.classifier.currentData()
        n = len(criteria.NAMES) if classifier == "global" else len(tv.LOCAL_NAMES) if classifier == "local" else 0
        spec, source = self._stability_copy if classifier == "global" else (None, "")
        local = classifier == "local"
        scheme = tv.TauScheme(
            data=self.data.currentData(), base=base, detection=detection, kernel=tuple(kernel), selection=selection,
            classifier=classifier, stability=spec, stability_source=source,
            local_mode=("percent" if self.local_percent.isChecked() else "zeta0") if local else "percent",
            zeta0=self.zeta0.value() if local else 0.05, percent=self.percent.value() if local else 20.0,
            local_variable=self.local_variable.currentData() if local else "zeta",
            zeta_variant=self.zeta_variant.currentData() if local and self.local_variable.currentData() == "zeta"
            else "mrd",
            choices=tuple(b.currentData() for b in self.class_choices[:n]), otherwise=self.otherwise.currentData())
        problems = scheme.errors()
        if base == "whatif" and self.run is not None and self.run.cfg is not None:
            problems += whatif.problems(scheme.secondary_config(self.run.cfg.secondary))
        return (scheme, "") if not problems else (None, "; ".join(problems))

    def set_scheme(self, scheme: tv.TauScheme) -> None:
        self._quiet = True
        self.data.setCurrentIndex(max(self.data.findData(scheme.data), 0))
        self.base.setCurrentIndex(max(self.base.findData(scheme.base), 0))
        if scheme.base == "whatif":
            self._put_whatif(dict(scheme.detection), scheme.kernel, dict(scheme.selection))
            self._quiet = True
        else:
            self._reset_whatif_fields()
            self._quiet = True
        self.classifier.setCurrentIndex(max(self.classifier.findData(scheme.classifier), 0))
        self._stability_copy = (scheme.stability, scheme.stability_source)
        self.local_variable.setCurrentIndex(max(self.local_variable.findData(scheme.local_variable), 0))
        (self.local_zeta0 if scheme.local_mode == "zeta0" else self.local_percent).setChecked(True)
        self.zeta0.setValue(scheme.zeta0)
        self.percent.setValue(scheme.percent)
        i = self.zeta_variant.findData(scheme.zeta_variant)
        if i < 0:
            self.zeta_variant.addItem(variant_combo.label(scheme.zeta_variant), scheme.zeta_variant)
            i = self.zeta_variant.count() - 1
        self.zeta_variant.setCurrentIndex(i)
        for box, c in zip(self.class_choices, scheme.choices):
            box.setCurrentIndex(max(box.findData(c), 0))
        self.otherwise.setCurrentIndex(max(self.otherwise.findData(scheme.otherwise), 0))
        self._quiet = False
        self._edited()

    def _edited(self) -> None:
        if self._quiet:
            return
        classifier = self.classifier.currentData()
        self.whatif_box.setVisible(self.base.currentData() == "whatif")
        self.global_row.setVisible(classifier == "global")
        self.global_note.setVisible(classifier == "global")
        self.local_rows.setVisible(classifier == "local")
        by_zeta = self.local_variable.currentData() == "zeta"
        self.zeta_variant_label.setVisible(by_zeta)
        self.zeta_variant.setVisible(by_zeta)
        self.local_note.setText("non-neutral: unstable where the heat flux is upward (w'θv' > 0), stable where it's "
                                "downward" if by_zeta else "non-neutral: unstable where Ri < 0, stable where Ri > 0")
        names = criteria.NAMES if classifier == "global" else tv.LOCAL_NAMES if classifier == "local" else ()
        for i, (lab, box) in enumerate(zip(self.class_labels, self.class_choices)):
            shown = i < len(names)
            lab.setVisible(shown)
            box.setVisible(shown)
            if shown:
                lab.setText(f"<span style='color:{style.stability_color(names[i], i).name()}'>■</span> {names[i]}")
        self.otherwise_label.setText("otherwise (unclassified)" if names else "every slot")
        spec, source = self._stability_copy
        self.global_note.setText(f"classifies by <b>{spec.describe()}</b>, copied from {source}" if spec is not None
                                 else "no stability scheme copied yet: choose one and press Copy")
        scheme, problem = self.scheme_now()
        self.errors.setText(problem)
        self.errors.setVisible(bool(problem))
        self._fill_schemes()

    # --- stability schemes ------------------------------------------------------------------------

    def _fill_global_sources(self) -> None:
        current = self.global_source.currentData()
        self.global_source.blockSignals(True)
        self.global_source.clear()
        self.global_source.addItem("Controls → Stability, as set now", _CURRENT)
        if self.stability_schemes is not None:
            for name in sorted(self.stability_schemes.items, key=str.lower):
                self.global_source.addItem(f"saved: {name}", name)
        self.global_source.setCurrentIndex(max(self.global_source.findData(current), 0))
        self.global_source.blockSignals(False)

    def copy_stability(self) -> None:
        self._fill_global_sources()
        which = self.global_source.currentData()
        if which == _CURRENT:
            spec, source = self.current_stability(), "Controls → Stability"
        else:
            spec, source = self.stability_schemes.items.get(which), f"the saved scheme “{which}”"
        if spec is None:
            self.errors.setText("that stability classification can't be copied (the run's own classes aren't five "
                                "contiguous intervals)")
            self.errors.show()
            return
        self._stability_copy = (spec, source)
        self._edited()

    # --- named schemes -------------------------------------------------------------------------------

    def _fill_schemes(self) -> None:
        scheme, _ = self.scheme_now()
        current = self.schemes.name_of(scheme) if scheme is not None else None
        self.scheme.blockSignals(True)
        self.scheme.clear()
        self.scheme.addItem("(unsaved changes)" if self.schemes.items else "(no saved schemes)", None)
        for name in sorted(self.schemes.items, key=str.lower):
            self.scheme.addItem(name, name)
            self.scheme.setItemData(self.scheme.count() - 1, self.schemes.items[name].describe(),
                                    Qt.ItemDataRole.ToolTipRole)
        self.scheme.setCurrentIndex(max(self.scheme.findData(current), 0) if current else 0)
        self.scheme.blockSignals(False)
        self.delete_button.setEnabled(current is not None)
        self.compute_button.setEnabled(current is not None and self.variants is not None)
        self.use_button.setEnabled(current is not None and self.variants is not None)
        self.show_state()

    def chosen_key(self) -> str | None:
        name = self.scheme.currentData()
        return tv.scheme_key(name) if name is not None else None

    def _on_scheme_picked(self) -> None:
        name = self.scheme.currentData()
        if name is not None:
            self.set_scheme(self.schemes.items[name])

    def save_scheme(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("a scheme needs a name")
        if '"' in name:
            raise ValueError('a scheme name can\'t hold "')
        scheme, problem = self.scheme_now()
        if scheme is None:
            raise ValueError(problem)
        self.schemes.put(name, scheme)
        self._fill_schemes()
        self.schemesChanged.emit([name])

    def _ask_save(self) -> None:
        offered = self.scheme.currentData() or ""
        name, ok = QInputDialog.getText(self, "Save the τ scheme", "name:", text=offered)
        name = name.strip()
        if not ok or not name:
            return
        scheme, _ = self.scheme_now()
        if name in self.schemes.items and self.schemes.items[name] != scheme:
            answer = QMessageBox.question(self, "Save the τ scheme", f"Replace the saved scheme “{name}”?\n\n"
                                          f"{self.schemes.items[name].describe()}\n\nViews showing it are computed "
                                          "again.")
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
            self.schemesChanged.emit([name])

    # --- what's computed ---------------------------------------------------------------------------

    def _request(self, signal) -> None:
        key = self.chosen_key()
        if key is not None:
            signal.emit(key)

    def show_progress(self, key: str, text: str) -> None:
        if key == self.chosen_key():
            self.state.setText(text)

    def show_state(self) -> None:
        key = self.chosen_key()
        if self.variants is None:
            self.state.setText("")
            self.counts.setRowCount(0)
            return
        mb = self.variants.disk_usage() / 1e6
        self.disk.setText(f"this run's computed τ variants: {mb:.0f} MB on disk")
        if key is None:
            self.state.setText("save the scheme to use it as a variant" if self.scheme.count() else "")
            self.counts.setRowCount(0)
            return
        summary = self.variants.summary(key) if self.variants.definition(key) is not None else None
        if summary is None:
            self.state.setText(f"{tv.label(key)}: not computed for this run yet (computed the first time a view "
                               "shows it, about half a minute a year)")
            self.counts.setRowCount(0)
            return
        thresholds = [f"b{int(b)} {p['threshold']:.4g}" for b, p in summary.get("booms", {}).items()
                      if p.get("threshold") is not None]
        self.state.setText(f"{tv.label(key)}: computed for this run ({summary['seconds']:g} s)"
                           + (f" · neutral up to |ζ| = {', '.join(thresholds)}" if thresholds else ""))
        self.show_counts(summary)

    def show_counts(self, summary: dict) -> None:
        rows = {}
        for boom, part in summary.get("booms", {}).items():
            total = max(part.get("rows", 0), 1)
            for rule, by_tau in (part.get("counts") or {}).items():
                for tau, n in by_tau.items():
                    label = "none" if tau == "-1" else f"{float(tau):g} s"
                    rows.setdefault((rule, label), {})[f"b{int(boom)}"] = 100.0 * n / total
        if not rows:
            self.counts.setRowCount(0)
            return
        frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
        frame.index = [f"{r} → {t}" for r, t in frame.index]
        tables.fill(self.counts, frame, index=True, fmt=lambda col, v: f"{v:.1f}")
        tables.fit_rows(self.counts, max_rows=12)
