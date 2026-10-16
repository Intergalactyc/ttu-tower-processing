"""The Summary's tau tab, over the viewed interval (or the whole period):
heat, momentum and selected tau as distributions over the ladder rungs (per
boom, or per stability class for one boom) with statistics; the heat-vs-
momentum joint distribution, whose cells brush their slots; and a what-if of
the selection rules applied to every slot.
"""
import dataclasses

import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import taustats
from ttu_tower.viewer.taustats import RUNGS, SOURCES, VARIANTS
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import in_class
from ttu_tower.viewer.ui import style, tables
from ttu_tower.viewer.ui.timeline_plot import LegendBar

_RULES = ("priority", "longest_significant")
_PRIORITIES = (("heat", "momentum"), ("momentum", "heat"))


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
    selectRequested = Signal(object)  # slots to brush

    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = self.data = None
        self.booms: list[int] = []
        self.x_range = None
        self.stability = None
        self.selection = np.empty(0, dtype=np.int64)
        self.whatif = None  # (tau, source, status) under the what-if rules
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
        self.rule, self.priority, self.fallback = QComboBox(), QComboBox(), QComboBox()
        for r in _RULES:
            self.rule.addItem(r, r)
        self.rule.setToolTip("priority: the first of the priority order with a τ; longest_significant: the "
                             "longer of the two")
        for p in _PRIORITIES:
            self.priority.addItem(", ".join(p), ",".join(p))  # a string: Qt turns a tuple's data into a list
        for r in RUNGS:
            self.fallback.addItem(f"{r:g} s", r)
        self.fallback.setToolTip("used where neither cospectrum gave a τ (and an unresolved τ's floor)")
        for w in (self.rule, self.priority, self.fallback):
            w.currentIndexChanged.connect(lambda *_: self._apply_whatif())
        reset = QPushButton("The run's rules")
        reset.clicked.connect(self.reset_whatif)
        self.brush_changed = QPushButton("Brush changed slots")
        self.brush_changed.setToolTip("select the slots whose selection the what-if changes (shown booms, the "
                                      "chosen slots)")
        self.brush_changed.clicked.connect(self._brush_changed)
        form = QFormLayout()
        form.addRow("rule", self.rule)
        form.addRow("priority", self.priority)
        form.addRow("fallback τ", self.fallback)
        box = QGroupBox("Selection rules, applied to every slot")
        box_layout = QVBoxLayout(box)
        box_layout.addLayout(form)
        buttons = QHBoxLayout()
        buttons.addWidget(reset)
        buttons.addWidget(self.brush_changed)
        buttons.addStretch(1)
        box_layout.addLayout(buttons)
        self.whatif_note = QLabel()
        self.whatif_note.setWordWrap(True)
        self.whatif_plot = _category_plot("selected τ: solid = run, dashed = what-if (the shown booms pooled)")
        self.whatif_table = tables.new_table()
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.whatif_plot)
        split.addWidget(self.whatif_table)
        split.setSizes([320, 220])
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.addWidget(box)
        layout.addWidget(self.whatif_note)
        layout.addWidget(split, stretch=1)
        return page

    # --- inputs -----------------------------------------------------------------------------------

    def set_run(self, run, index, booms) -> None:
        self.run, self.index, self.booms = run, index, list(booms)
        self.data = self.whatif = None
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
            items += [(n, f"class:{n}") for n in self.stability[1]]
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

    # --- what-if ----------------------------------------------------------------------------------

    def what_if_rules(self):
        cfg = self.run.cfg.secondary.selection
        return dataclasses.replace(cfg, rule=self.rule.currentData(), priority=tuple(self.priority.currentData().split(",")),
                                   fallback_tau_s=self.fallback.currentData())

    def reset_whatif(self, apply: bool = True) -> None:
        if self.run is None or self.run.cfg is None:
            return
        cfg = self.run.cfg.secondary.selection
        for box, value in ((self.rule, cfg.rule), (self.priority, ",".join(cfg.priority)),
                           (self.fallback, cfg.fallback_tau_s)):
            box.blockSignals(True)
            box.setCurrentIndex(max(box.findData(value), 0))
            box.blockSignals(False)
        if apply:
            self._apply_whatif()

    def _apply_whatif(self) -> None:
        if self.data is None or self.run is None or self.run.cfg is None:
            self.redraw()
            return
        self.whatif = taustats.reselect(self.data, self.what_if_rules())
        self.redraw()

    def changed(self) -> np.ndarray:
        """Rows whose selection the what-if changes."""
        d = self.data
        tau, source, _ = self.whatif
        same_tau = (tau == d.sel_tau) | (np.isnan(tau) & np.isnan(d.sel_tau))
        return ~same_tau | (source != d.sel_source)

    def _draw_whatif(self) -> None:
        d = self.data
        if self.whatif is None:
            self.whatif_note.setText("the run's config is unavailable, so its rules can't be edited")
            return
        keep = self._rows()
        tau, source, _ = self.whatif
        changed = self.changed()
        run_cats = taustats.categories(d, "selected")
        new_cats = taustats.categories(d, "selected", sel_tau=tau, sel_source=source)
        n, n_changed = int(keep.sum()), int((changed & keep).sum())
        default = self.what_if_rules() == self.run.cfg.secondary.selection
        self.whatif_note.setText(f"{n_changed:,} of {n:,} selections change ({100.0 * n_changed / max(n, 1):.1f} %)"
                                 + (" — these are the run's own rules" if default else ""))
        self.brush_changed.setEnabled(n_changed > 0)
        _draw_categories(self.whatif_plot, "selected",
                         [(taustats.shares(run_cats[keep], "selected"), QColor("#222222"), False, "run"),
                          (taustats.shares(new_cats[keep], "selected"), QColor("#c2185b"), True, "what-if")],
                         legend=True)
        rows = {}
        for b in self.booms:
            if b not in self.boom_bar.visible():
                continue
            m = keep & (d.boom == b)
            old = taustats.summary(run_cats[m], "selected", source=d.sel_source[m])
            new = taustats.summary(new_cats[m], "selected", source=source[m])
            row = {"N": old["N"], "% changed": 100.0 * float(np.mean(changed[m])) if m.any() else np.nan,
                   "median τ run": old["median τ"], "median τ what-if": new["median τ"]}
            for name in SOURCES[:3]:
                row[f"% {name} run"], row[f"% {name} what-if"] = old[f"% {name}"], new[f"% {name}"]
            rows[f"b{b} ({HEIGHTS[b]:g} m)"] = row
        tables.fill(self.whatif_table, pd.DataFrame.from_dict(rows, orient="index"), index=True, fmt=_fmt)

    def _brush_changed(self) -> None:
        if self.whatif is None:
            return
        slots = np.unique(self.data.slot[self._rows() & self.changed()])
        if slots.size:
            self.selectRequested.emit(slots)
