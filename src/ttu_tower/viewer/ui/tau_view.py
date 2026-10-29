"""The Summary's tau tab, over the viewed interval (or the whole period):
heat, momentum and selected tau as distributions over the ladder rungs (per
boom, or per stability class for one boom) with statistics; the heat-vs-
momentum joint distribution, whose cells brush their slots; and a what-if of
the detection and selection settings applied to every slot (selection at once,
detection rerun on every slot's spectra when asked).
"""
import dataclasses

import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.secondary.detect import SMOOTHING
from ttu_tower.viewer import taustats, whatif
from ttu_tower.viewer.taustats import RUNGS, SOURCES, VARIANTS
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import class_choices, in_class
from ttu_tower.viewer.ui import style, tables
from ttu_tower.viewer.ui.timeline_plot import LegendBar

_RULES = ("priority", "longest_significant")
_PRIORITIES = (("heat", "momentum"), ("momentum", "heat"))


def _rung_box() -> QComboBox:
    box = QComboBox()
    for r in RUNGS:
        box.addItem(f"{r:g} s", r)
    return box


def _fmt(col, v):
    if isinstance(v, float) and col.startswith("%"):
        return "—" if np.isnan(v) else f"{v:.1f}"
    if isinstance(v, float) and "τ" in col:
        return "—" if np.isnan(v) else f"{v:g}"
    return None


def _category_plot(title: str) -> pg.PlotWidget:
    w = pg.PlotWidget()
    item = w.getPlotItem()
    item.setTitle(title, size="9pt")
    item.showGrid(y=True, alpha=0.2)
    item.setLabel("left", "% of slots")
    item.setMouseEnabled(y=False)
    return w


def _draw_categories(plot: pg.PlotWidget, which: str, lines, legend: bool = False) -> None:
    """`lines`: (values in `labels(which)` order, color, dashed, name). The rungs
    and the statuses past them on one categorical axis; `legend` names the lines.
    """
    item = plot.getPlotItem()
    item.clear()
    if legend and item.legend is None:
        item.addLegend(offset=(-10, 10))
    if item.legend is not None:
        item.legend.setVisible(legend)
    names = taustats.labels(which)
    item.getAxis("bottom").setTicks([list(enumerate(names))])
    item.getAxis("bottom").setLabel("τ [s] (found or capped) · then no τ")
    item.addItem(pg.InfiniteLine(len(RUNGS) - 0.5, pen=pg.mkPen("#999999", style=Qt.PenStyle.DotLine)))
    for values, color, dashed, _name in lines:
        pen = pg.mkPen(color, width=1.8, style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        item.plot(np.arange(len(names)), values, pen=pen, symbol="o", symbolSize=5, symbolBrush=color,
                  symbolPen=None, name=_name if legend else None)
    item.setXRange(-0.5, len(names) - 0.5, padding=0.02)
    item.enableAutoRange(axis="y")


class TauView(QWidget):
    STATE_FIELDS = ("variant", "group", "boom_bar", "which", "lines", "class_boom", "fields.peak_significance_se",
                    "fields.min_scale_s", "fields.min_tau_s", "fields.max_tau_s", "smoothing", "rule", "priority",
                    "fallback", "compare_which", "tabs")  # saved in a view, restored in this order

    selectRequested = Signal(object)  # slots to brush

    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = self.data = None
        self.booms: list[int] = []
        self.x_range = None
        self.stability = None
        self.selection = np.empty(0, dtype=np.int64)
        self.whatif = None  # (detections, selected tau, source, status) under the what-if settings
        self.applied = None  # (detection config, kernel) recomputed and in use; None: the run's stored
        self._detections: dict = {}  # ((detection config, kernel), boom) -> redetected rows
        self._pending: dict = {}  # boom -> the settings being recomputed for it
        self._joint_rows = None  # the rows each matrix cell holds, for brushing

        self.variant = QComboBox()
        for v in VARIANTS:
            self.variant.addItem(v, v)
        self.variant.setToolTip("which variant's detections and selection")
        self.group = QComboBox()
        self.group.setToolTip("which slots: all, one stability class, or the brushed selection (or the rest)")
        self.boom_bar = LegendBar()
        for w in (self.variant, self.group):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        self.boom_bar.toggled.connect(self.redraw)
        self.info = QLabel()
        self.info.setWordWrap(True)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._distribution_page(), "Distributions")
        self.tabs.addTab(self._joint_page(), "Heat vs momentum")
        self.tabs.addTab(self._whatif_page(), "Selection what-if")
        self.tabs.currentChanged.connect(lambda *_: self.redraw())

        top = QHBoxLayout()
        top.addWidget(self.variant)
        top.addWidget(self.group)
        top.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(top)
        layout.addWidget(self.boom_bar)
        layout.addWidget(self.info)
        layout.addWidget(self.tabs, stretch=1)
        self._fill_groups()

    def sizeHint(self) -> QSize:
        return QSize(600, 700)

    # --- pages ------------------------------------------------------------------------------------

    def _distribution_page(self) -> QWidget:
        self.which = QComboBox()
        for key, label in taustats.WHICH.items():
            self.which.addItem(label, key)
        self.which.setCurrentIndex(self.which.findData("selected"))
        self.lines = QComboBox()
        self.lines.addItem("a line per boom", "boom")
        self.lines.addItem("a line per stability class, for one boom:", "class")
        self.class_boom = QComboBox()
        self.class_boom.setToolTip("the boom whose stability classes are compared")
        for w in (self.which, self.lines, self.class_boom):
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        self.dist_plot = _category_plot("")
        self.dist_table = tables.new_table()
        controls = QHBoxLayout()
        for w in (self.which, self.lines, self.class_boom):
            controls.addWidget(w)
        controls.addStretch(1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.dist_plot)
        split.addWidget(self.dist_table)
        split.setSizes([400, 200])
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(split, stretch=1)
        return page

    def _joint_page(self) -> QWidget:
        self.joint_table = tables.new_table()
        self.joint_table.setToolTip("% of the shown booms' slots with that pair of detections · click a cell to "
                                    "brush its slots")
        self.joint_table.cellClicked.connect(self._joint_clicked)
        self.agree_table = tables.new_table()
        self.agree_table.setToolTip("where both cospectra gave a τ: how often they agree and which is longer "
                                    "(median difference in rungs; a rung is a factor 2)")
        note = QLabel("rows: heat τ · columns: momentum τ · % of slots · click a cell to brush those slots")
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.joint_table)
        split.addWidget(self.agree_table)
        split.setSizes([380, 220])
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.addWidget(note)
        layout.addWidget(split, stretch=1)
        return page

    def _whatif_page(self) -> QWidget:
        f = self.fields = {"peak_significance_se": QDoubleSpinBox(), "min_scale_s": QDoubleSpinBox(),
                           "min_tau_s": _rung_box(), "max_tau_s": _rung_box()}
        f["peak_significance_se"].setRange(0.0, 50.0)
        f["peak_significance_se"].setSingleStep(0.25)
        f["peak_significance_se"].setToolTip("a peak (or a reversal that would clip τ) must stand this many SE clear")
        f["min_scale_s"].setRange(0.05, 9.0)
        f["min_scale_s"].setSingleStep(0.1)
        f["min_scale_s"].setDecimals(3)
        self.smoothing = QComboBox()
        self.smoothing.setEditable(True)
        for label, weights in whatif.KERNELS:
            self.smoothing.addItem(label, weights)
        self.smoothing.setToolTip("the weights smoothing each mode with its neighbours before the peak search "
                                  "(choose one, or type your own, e.g. 1-3-1)")
        for w in f.values():
            signal = w.valueChanged if isinstance(w, QDoubleSpinBox) else w.currentIndexChanged
            signal.connect(lambda *_: self._detection_edited())
        self.smoothing.currentTextChanged.connect(lambda *_: self._detection_edited())
        self.rule, self.priority, self.fallback = QComboBox(), QComboBox(), _rung_box()
        for r in _RULES:
            self.rule.addItem(r, r)
        self.rule.setToolTip("priority: the first of the priority order with a τ; longest_significant: the "
                             "longer of the two")
        for p in _PRIORITIES:
            self.priority.addItem(", ".join(p), ",".join(p))  # a string: Qt turns a tuple's data into a list
        self.fallback.setToolTip("used where neither cospectrum gave a τ (and an unresolved τ's floor)")
        for w in (self.rule, self.priority, self.fallback):
            w.currentIndexChanged.connect(lambda *_: self._apply_whatif())

        detection = QGroupBox("Detection (rerun on every slot's spectra)")
        det_form = QFormLayout(detection)
        det_form.addRow("peak significance [SE]", f["peak_significance_se"])
        det_form.addRow("min scale", f["min_scale_s"])
        det_form.addRow("min τ", f["min_tau_s"])
        det_form.addRow("max τ", f["max_tau_s"])
        det_form.addRow("smoothing", self.smoothing)
        self.recompute = QPushButton("Recompute detection")
        self.recompute.setToolTip("rerun τ detection on every slot's stored heat and momentum cospectra with these "
                                  "settings (in the background, about 10 s a boom for a year; kept once done)")
        self.recompute.clicked.connect(self._recompute)
        det_form.addRow(self.recompute)
        selection = QGroupBox("Selection (applied at once)")
        sel_form = QFormLayout(selection)
        sel_form.addRow("rule", self.rule)
        sel_form.addRow("priority", self.priority)
        sel_form.addRow("fallback τ", self.fallback)
        reset = QPushButton("The run's settings")
        reset.clicked.connect(self.reset_whatif)
        sel_form.addRow(reset)
        boxes = QHBoxLayout()
        boxes.addWidget(detection)
        boxes.addWidget(selection)

        self.problem = QLabel()
        self.problem.setStyleSheet("color:#b00")
        self.problem.setWordWrap(True)
        self.det_note = QLabel()
        self.det_note.setWordWrap(True)
        self.whatif_note = QLabel()
        self.whatif_note.setWordWrap(True)
        self.compare_which = QComboBox()
        for key, label in taustats.WHICH.items():
            self.compare_which.addItem(label, key)
        self.compare_which.setCurrentIndex(self.compare_which.findData("selected"))
        self.compare_which.currentIndexChanged.connect(lambda *_: self.redraw())
        self.brush_changed = QPushButton("Brush changed slots")
        self.brush_changed.setToolTip("select the slots whose shown τ the what-if changes (shown booms, the chosen "
                                      "slots)")
        self.brush_changed.clicked.connect(self._brush_changed)
        compare = QHBoxLayout()
        compare.addWidget(QLabel("compare"))
        compare.addWidget(self.compare_which)
        compare.addWidget(self.brush_changed)
        compare.addStretch(1)
        self.whatif_plot = _category_plot("")
        self.whatif_table = tables.new_table()
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.whatif_plot)
        split.addWidget(self.whatif_table)
        split.setSizes([300, 220])
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.addLayout(boxes)
        layout.addWidget(self.problem)
        layout.addWidget(self.det_note)
        layout.addWidget(self.whatif_note)
        layout.addLayout(compare)
        layout.addWidget(split, stretch=1)
        return page

    # --- inputs -----------------------------------------------------------------------------------

    def set_run(self, run, index, booms) -> None:
        self.run, self.index, self.booms = run, index, list(booms)
        self.data = self.whatif = self.applied = None
        self._detections, self._pending = {}, {}
        colors = [style.boom_color(b) for b in self.booms]
        self.boom_bar.set_members(self.booms, [style.member_short(b) for b in self.booms], colors,
                                  [style.member_label(b) for b in self.booms])
        self.class_boom.blockSignals(True)
        self.class_boom.clear()
        for b in self.booms:
            self.class_boom.addItem(f"b{b} ({HEIGHTS[b]:g} m)", b)
        self.class_boom.setCurrentIndex(max(self.class_boom.findData(4), 0))
        self.class_boom.blockSignals(False)
        self.reset_whatif(apply=False)

    def set_range(self, x_range, stability) -> None:
        self.x_range, self.stability = x_range, stability
        self._fill_groups()
        if self.data is None:
            self.reload()
        else:
            self.redraw()

    def set_selection(self, slots) -> None:
        self.selection = slots
        self._fill_groups()
        self.redraw()

    def _fill_groups(self) -> None:
        current = self.group.currentData()
        items = [("all slots", "all")]
        if self.stability is not None:
            items += [(n, f"class:{n}") for n in class_choices(self.stability[1])]
        if self.selection.size:
            items += [("selected slots", "selected"), ("unselected slots", "rest")]
        if [self.group.itemData(i) for i in range(self.group.count())] == [d for _, d in items]:
            return
        self.group.blockSignals(True)
        self.group.clear()
        for label, data in items:
            self.group.addItem(label, data)
        self.group.setCurrentIndex(max(self.group.findData(current), 0))
        self.group.blockSignals(False)

    def reload(self) -> None:
        if self.run is None:
            return
        self.info.setText("reading every slot's τ (once; a few seconds)…")
        run = self.run
        self.runner.submit(taustats.load, run, self.index, key=f"tau-{id(self)}", label="reading τ",
                           on_done=lambda data: self._loaded(run, data),
                           on_error=lambda exc, tb: self.info.setText(f"couldn't read τ: {exc}"))

    def _loaded(self, run, data) -> None:
        if not shiboken6.isValid(self) or run is not self.run:
            return
        self.data = data
        self._apply_whatif()

    # --- which rows -------------------------------------------------------------------------------

    def _rows(self, booms=None, group: str | None = None) -> np.ndarray:
        """The rows of the chosen variant, booms and slots."""
        d = self.data
        keep = d.variant == VARIANTS.index(self.variant.currentData())
        keep &= np.isin(d.boom, list(self.boom_bar.visible() if booms is None else booms))
        if self.x_range is not None:
            t = slot_to_unix(d.slot)
            keep &= (t >= self.x_range[0]) & (t < self.x_range[1])
        kind = group or self.group.currentData() or "all"
        if kind.startswith("class:"):
            keep[keep] = in_class(self.stability, d.slot[keep], kind[len("class:"):])
        elif kind == "selected":
            keep &= np.isin(d.slot, self.selection)
        elif kind == "rest":
            keep &= ~np.isin(d.slot, self.selection)
        return keep

    # --- drawing ----------------------------------------------------------------------------------

    def redraw(self) -> None:
        if self.data is None:
            return
        scope = "whole period" if self.x_range is None else "visible range"
        self.info.setText(f"{self.variant.currentData()}, {self.group.currentText()}, {scope}")
        page = self.tabs.currentIndex()
        if page == 0:
            self._draw_distributions()
        elif page == 1:
            self._draw_joint()
        else:
            self._draw_whatif()

    def _draw_distributions(self) -> None:
        d, which = self.data, self.which.currentData()
        by_class = self.lines.currentData() == "class"
        self.class_boom.setEnabled(by_class)
        cats_all = taustats.categories(d, which)
        members = []
        if by_class:
            boom = self.class_boom.currentData()
            names = self.stability[1] if self.stability is not None else []
            members = [(name, style.stability_color(name, i), self._rows([boom], "all") &
                        self._class_rows(name)) for i, name in enumerate(names)]
            members.append(("all classes", QColor("#222222"), self._rows([boom], "all")))
            title = f"{taustats.WHICH[which]} at b{boom}, by stability class"
        else:
            keep = self._rows()
            members = [(f"b{b} ({HEIGHTS[b]:g} m)", style.boom_color(b), keep & (d.boom == b))
                       for b in self.booms if b in self.boom_bar.visible()]
            title = f"{taustats.WHICH[which]}, per boom"
        lines, rows = [], {}
        for name, color, rows_mask in members:
            cats = cats_all[rows_mask]
            lines.append((taustats.shares(cats, which), color, False, name))
            rows[name] = taustats.summary(cats, which, source=d.sel_source[rows_mask],
                                          status=d.status[which][rows_mask] if which != "selected" else None)
        self.dist_plot.getPlotItem().setTitle(title, size="9pt")
        _draw_categories(self.dist_plot, which, lines, legend=by_class)
        frame = pd.DataFrame.from_dict(rows, orient="index")
        tables.fill(self.dist_table, frame, index=True, fmt=_fmt)

    def _class_rows(self, name: str) -> np.ndarray:
        keep = np.zeros(self.data.slot.size, dtype=bool)
        if self.stability is not None:
            keep[:] = in_class(self.stability, self.data.slot, name)
        return keep

    def _draw_joint(self) -> None:
        d = self.data
        keep = self._rows()
        heat, mom = taustats.categories(d, "heat"), taustats.categories(d, "momentum")
        counts = taustats.joint(heat[keep], mom[keep])
        total = max(int(counts.sum()), 1)
        names = taustats.labels("heat")
        table = self.joint_table
        table.clear()
        table.setRowCount(len(names))
        table.setColumnCount(len(names))
        table.setHorizontalHeaderLabels([f"m {n}" for n in names])
        table.setVerticalHeaderLabels([f"h {n}" for n in names])
        table.verticalHeader().setVisible(True)
        peak = max(counts.max(), 1)
        for i in range(len(names)):
            for j in range(len(names)):
                pct = 100.0 * counts[i, j] / total
                item = QTableWidgetItem(f"{pct:.1f}" if counts[i, j] else "")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                shade = int(255 - 150 * (counts[i, j] / peak) ** 0.5)
                diagonal = i == j and i < len(RUNGS)
                item.setBackground(QBrush(QColor(shade, shade, 255) if diagonal else QColor(255, shade, shade)))
                item.setToolTip(f"heat {names[i]}, momentum {names[j]}: {counts[i, j]:,} slots ({pct:.2f} %)")
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        self._joint_rows = (keep, heat, mom)
        rows = {}
        for b in self.booms:
            if b in self.boom_bar.visible():
                m = keep & (d.boom == b)
                rows[f"b{b} ({HEIGHTS[b]:g} m)"] = taustats.agreement(heat[m], mom[m])
        rows["all shown"] = taustats.agreement(heat[keep], mom[keep])
        frame = pd.DataFrame.from_dict(rows, orient="index")
        tables.fill(self.agree_table, frame, index=True,
                    fmt=lambda col, v: (f"{v:+.1f}" if col.startswith("median") and np.isfinite(v) else _fmt(col, v)))

    def _joint_clicked(self, row: int, column: int) -> None:
        if self._joint_rows is None:
            return
        keep, heat, mom = self._joint_rows
        rows = keep & (heat == row) & (mom == column)
        slots = np.unique(self.data.slot[rows])
        if slots.size:
            self.selectRequested.emit(slots)

    # --- what-if ---------------------------------------------------------------------------------

    def what_if_rules(self):
        cfg = self.run.cfg.secondary.selection
        return dataclasses.replace(cfg, rule=self.rule.currentData(),
                                   priority=tuple(self.priority.currentData().split(",")),
                                   fallback_tau_s=self.fallback.currentData())

    def edited_detection(self):
        """(detection config, kernel), or (None, the problems) where they can't be used."""
        f = self.fields
        cfg = whatif.edited(self.run.cfg.secondary,
                            detection={"peak_significance_se": f["peak_significance_se"].value(),
                                       "min_scale_s": f["min_scale_s"].value(), "min_tau_s": f["min_tau_s"].currentData(),
                                       "max_tau_s": f["max_tau_s"].currentData()},
                            selection={"fallback_tau_s": self.fallback.currentData()})
        kernel, kernel_problem = whatif.parse_kernel(self.smoothing.currentText())
        problems = whatif.problems(cfg) + ([kernel_problem] if kernel_problem else [])
        return ((cfg.detection, kernel), "") if not problems else (None, "; ".join(problems))

    def _run_detection(self):
        return self.run.cfg.secondary.detection, SMOOTHING

    def reset_whatif(self, apply: bool = True) -> None:
        if self.run is None or self.run.cfg is None:
            return
        det, sel = self.run.cfg.secondary.detection, self.run.cfg.secondary.selection
        widgets = [*self.fields.values(), self.smoothing, self.rule, self.priority, self.fallback]
        for w in widgets:
            w.blockSignals(True)
        self.fields["peak_significance_se"].setValue(det.peak_significance_se)
        self.fields["min_scale_s"].setValue(det.min_scale_s)
        for box, value in ((self.fields["min_tau_s"], det.min_tau_s), (self.fields["max_tau_s"], det.max_tau_s),
                           (self.rule, sel.rule), (self.priority, ",".join(sel.priority)),
                           (self.fallback, sel.fallback_tau_s)):
            box.setCurrentIndex(max(box.findData(value), 0))
        self.smoothing.setCurrentIndex(0)
        self.smoothing.setEditText(self.smoothing.itemText(0))
        for w in widgets:
            w.blockSignals(False)
        self.applied = None  # the run's stored detections
        if apply:
            self._apply_whatif()

    def _detection_edited(self) -> None:
        """Detection edits wait for Recompute; until then the applied detections stand."""
        self._show_detection_state()

    def _show_detection_state(self) -> None:
        if self.run is None or self.run.cfg is None:
            return
        chosen, problem = self.edited_detection()
        self.problem.setText(problem)
        applied = self.applied or self._run_detection()
        pending = [b for b in self.booms if self._pending.get(b) == chosen]
        self.recompute.setEnabled(chosen is not None and chosen != applied and not pending)
        if pending:
            done = len(self.booms) - len(pending)
            self.det_note.setText(f"recomputing detection… {done} of {len(self.booms)} booms done")
        elif self.applied is None:
            note = "detection: the run's (stored)"
            if chosen is not None and chosen != applied:
                note += " — the edited settings apply once you press Recompute"
            self.det_note.setText(note)
        else:
            cfg, kernel = self.applied
            note = (f"detection: recomputed — significance {cfg.peak_significance_se:g} SE, min scale "
                    f"{cfg.min_scale_s:g} s, τ {cfg.min_tau_s:g}–{cfg.max_tau_s:g} s, smoothing "
                    f"{'none' if len(kernel) == 1 else whatif.kernel_text(kernel)}")
            if chosen is not None and chosen != applied:
                note += " · edited since: press Recompute to apply"
            self.det_note.setText(note)

    def _recompute(self) -> None:
        chosen, problem = self.edited_detection()
        if chosen is None or self.data is None:
            return
        if chosen == self._run_detection():
            self.applied = None
            self._apply_whatif()
            return
        run, data = self.run, self.data
        for boom in self.booms:
            if (chosen, boom) in self._detections:
                continue
            self._pending[boom] = chosen
            self.runner.submit(taustats.redetect, self.index, data, boom, chosen[0], chosen[1],
                               key=f"tau-redetect-{id(self)}-{boom}", label=f"redetecting τ, b{boom}",
                               on_done=lambda result, b=boom, c=chosen: self._redetected(run, c, b, result),
                               on_error=lambda exc, tb: self.det_note.setText(f"couldn't recompute: {exc}"))
        self._finish_if_done(chosen)

    def _redetected(self, run, chosen, boom, result) -> None:
        if not shiboken6.isValid(self) or run is not self.run:
            return
        self._detections[(chosen, boom)] = result
        if self._pending.get(boom) == chosen:
            del self._pending[boom]
        self._finish_if_done(chosen)

    def _finish_if_done(self, chosen) -> None:
        if all((chosen, b) in self._detections for b in self.booms):
            self.applied = chosen
            self._apply_whatif()
        else:
            self._show_detection_state()

    def _apply_whatif(self) -> None:
        if self.data is None or self.run is None or self.run.cfg is None:
            self.redraw()
            return
        base = self.data
        if self.applied is not None:
            base = taustats.with_detections(self.data, {b: self._detections[(self.applied, b)] for b in self.booms})
        tau, source, status = taustats.reselect(base, self.what_if_rules())
        self.whatif = (base, tau, source, status)
        self._show_detection_state()
        self.redraw()

    def is_default(self) -> bool:
        return (self.applied is None and self.run is not None
                and self.what_if_rules() == self.run.cfg.secondary.selection)

    def changed(self, which: str = "selected") -> np.ndarray:
        """Rows whose `which` tau the what-if changes."""
        d = self.data
        base, tau, source, _ = self.whatif
        if which == "selected":
            same = (tau == d.sel_tau) | (np.isnan(tau) & np.isnan(d.sel_tau))
            return ~same | (source != d.sel_source)
        same = (base.tau[which] == d.tau[which]) | (np.isnan(base.tau[which]) & np.isnan(d.tau[which]))
        return ~same | (base.status[which] != d.status[which])

    def _whatif_categories(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        base, tau, source, _ = self.whatif
        run = taustats.categories(self.data, which)
        new = taustats.categories(base, which, sel_tau=tau, sel_source=source) if which == "selected" \
            else taustats.categories(base, which)
        return run, new

    def _draw_whatif(self) -> None:
        d = self.data
        if self.whatif is None:
            self.whatif_note.setText("the run's config is unavailable, so its settings can't be edited")
            return
        keep = self._rows()
        which = self.compare_which.currentData()
        base, tau, source, _ = self.whatif
        changed = self.changed(which)
        run_cats, new_cats = self._whatif_categories(which)
        n, n_changed = int(keep.sum()), int((changed & keep).sum())
        own = " — these are the run's own settings" if self.is_default() else ""
        self.whatif_note.setText(f"{taustats.WHICH[which]}: {n_changed:,} of {n:,} change "
                                 f"({100.0 * n_changed / max(n, 1):.1f} %){own}")
        self.brush_changed.setEnabled(n_changed > 0)
        self.whatif_plot.getPlotItem().setTitle(f"{taustats.WHICH[which]}: solid = run, dashed = what-if "
                                                "(the shown booms pooled)", size="9pt")
        _draw_categories(self.whatif_plot, which,
                         [(taustats.shares(run_cats[keep], which), QColor("#222222"), False, "run"),
                          (taustats.shares(new_cats[keep], which), QColor("#c2185b"), True, "what-if")],
                         legend=True)
        changes = {q: self.changed(q) for q in taustats.WHICH}
        sel_run, sel_new = self._whatif_categories("selected")
        rows = {}
        for b in self.booms:
            if b not in self.boom_bar.visible():
                continue
            m = keep & (d.boom == b)
            old = taustats.summary(sel_run[m], "selected", source=d.sel_source[m])
            new = taustats.summary(sel_new[m], "selected", source=source[m])
            row = {"N": old["N"]}
            for q, label in (("heat", "% heat τ changed"), ("momentum", "% momentum τ changed"),
                             ("selected", "% selected changed")):
                row[label] = 100.0 * float(np.mean(changes[q][m])) if m.any() else np.nan
            row["median τ run"], row["median τ what-if"] = old["median τ"], new["median τ"]
            for name in SOURCES[:3]:
                row[f"% {name} run"], row[f"% {name} what-if"] = old[f"% {name}"], new[f"% {name}"]
            rows[f"b{b} ({HEIGHTS[b]:g} m)"] = row
        tables.fill(self.whatif_table, pd.DataFrame.from_dict(rows, orient="index"), index=True, fmt=_fmt)

    def _brush_changed(self) -> None:
        if self.whatif is None:
            return
        slots = np.unique(self.data.slot[self._rows() & self.changed(self.compare_which.currentData())])
        if slots.size:
            self.selectRequested.emit(slots)
