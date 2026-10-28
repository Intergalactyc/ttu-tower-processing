"""Hodographs over the viewed interval (or the whole period): each boom's mean
wind vector (u east, v north), joined bottom to top, for every slot, one
stability class, each class overlaid, or the brushed selection (or the rest);
the mesonet's 10 m wind as a point apart. A small group can also show each
slot's own hodograph, faintly.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ttu_tower.viewer import hodograph
from ttu_tower.viewer.hodograph import MESONET
from ttu_tower.viewer.timeaxis import slot_to_unix
from ttu_tower.viewer.timeline import in_class
from ttu_tower.viewer.ui import style, tables

_LINE = "#333333"
EACH_SLOT_LIMIT = 500


def prepare_plot(plot: pg.PlotWidget) -> None:
    item = plot.getPlotItem()
    plot.setAspectLocked(True)
    item.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "u, toward east [m/s]")
    plot.setLabel("left", "v, toward north [m/s]")
    item.addLegend(offset=(5, 5), labelTextSize="8pt")


def draw_hodograph(plot: pg.PlotWidget, rows: pd.DataFrame, color=_LINE, name: str = "", labels="all",
                   width: float = 2.0) -> None:
    """One hodograph from `hodograph.summary` rows: the booms joined bottom to top, each marked in its
    boom's color; the mesonet a star apart. `labels`: "all" heights, the bottom and top ("ends"), or none.
    """
    tower = rows[(rows["member"] != MESONET) & (rows["n"] > 0)]
    plot.plot(tower["u"].to_numpy(float), tower["v"].to_numpy(float), pen=pg.mkPen(color, width=width),
              name=name or None)
    brushes = [pg.mkBrush(style.boom_color(int(m))) for m in tower["member"]]
    plot.addItem(pg.ScatterPlotItem(tower["u"].to_numpy(float), tower["v"].to_numpy(float), size=9,
                                    brush=brushes, pen=pg.mkPen(color, width=1)))
    if labels:
        ends = labels == "ends" and len(tower) > 2
        for _, r in (tower.iloc[[0, -1]] if ends else tower).iterrows():
            text = pg.TextItem(f"{r['height']:g} m", color="#444", anchor=(-0.15, 0.5))
            text.setPos(r["u"], r["v"])
            plot.addItem(text)
    meso = rows[(rows["member"] == MESONET) & (rows["n"] > 0)]
    if not meso.empty:
        plot.addItem(pg.ScatterPlotItem(meso["u"].to_numpy(float), meso["v"].to_numpy(float), size=15, symbol="star",
                                        brush=pg.mkBrush(color), pen=pg.mkPen("#000", width=0.8)))
        if labels == "all":
            text = pg.TextItem("mesonet 10 m", color="#444", anchor=(1.1, 0.5))
            text.setPos(float(meso["u"].iloc[0]), float(meso["v"].iloc[0]))
            plot.addItem(text)


def draw_origin(plot: pg.PlotWidget) -> None:
    for angle in (0, 90):
        plot.addItem(pg.InfiniteLine(0.0, angle=angle, pen=pg.mkPen("#999999", width=1)))


def _load(run, index, booms):
    """Runs on a worker thread."""
    return hodograph.load(run, index, booms)


class HodographView(QWidget):
    STATE_FIELDS = ("group", "common", "each_slot", "labels")  # saved in a view, restored in this order

    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.run = self.index = None
        self.booms: list[int] = []
        self.x_range = None
        self.stability = None
        self.selection = np.empty(0, dtype=np.int64)
        self.uvs = None  # {member: UV} over the whole period
        self._loaded_key = None
        self.drawn: list = []  # [(label, color, summary rows)], for figure_spec and the table

        self.group = QComboBox()
        self.group.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.group.setToolTip("which slots: all, one stability class, each class overlaid, or the brushed "
                              "selection (or the rest)")
        self.common = QCheckBox("only slots every boom has")
        self.common.setChecked(True)
        self.common.setToolTip("every height (and the mesonet) averaged over the same slots, so the hodograph "
                               "isn't bent by where each boom has gaps")
        self.each_slot = QCheckBox(f"each slot (up to {EACH_SLOT_LIMIT})")
        self.each_slot.setToolTip("each slot's own hodograph, faintly behind the mean (one group, at most "
                                  f"{EACH_SLOT_LIMIT} slots)")
        self.labels = QCheckBox("heights")
        self.labels.setChecked(True)
        self.labels.setToolTip("label each boom's height (with classes overlaid, the bottom and top ones)")
        self.group.currentIndexChanged.connect(lambda *_: self.redraw())
        for w in (self.common, self.each_slot, self.labels):
            w.toggled.connect(lambda *_: self.redraw())
        self.plot = pg.PlotWidget()
        self.plot.getPlotItem().export_layout = "per curve"  # CSV: x and y columns per curve
        prepare_plot(self.plot)
        self.table = tables.new_table()
        self.info = QLabel()
        self.info.setWordWrap(True)

        row = QHBoxLayout()
        for w in (self.group, self.common, self.each_slot, self.labels):
            row.addWidget(w)
        row.addStretch(1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.plot)
        split.addWidget(self.table)
        split.setSizes([500, 200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(row)
        layout.addWidget(split, stretch=1)
        layout.addWidget(self.info)
        self._fill_groups()

    def sizeHint(self) -> QSize:
        return QSize(600, 650)

    def set_run(self, run, index, booms) -> None:
        self.run, self.index, self.booms = run, index, list(booms)
        self.uvs, self._loaded_key = None, None

    def set_range(self, x_range, stability) -> None:
        self.x_range, self.stability = x_range, stability
        self._fill_groups()
        if self._loaded_key != tuple(self.booms):
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
            items += [(n, f"class:{n}") for n in self.stability[1]] + [("each class, overlaid", "classes")]
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
        key = tuple(self.booms)
        self.info.setText("reading every boom's mean wind…")
        self.runner.submit(_load, self.run, self.index, self.booms, key=f"hodograph-{id(self)}",
                           label="reading the wind for the hodograph", on_done=lambda uvs: self._loaded(key, uvs),
                           on_error=lambda exc, tb: self.info.setText(f"couldn't read the wind: {exc}"))

    def _loaded(self, key, uvs) -> None:
        if not shiboken6.isValid(self) or key != tuple(self.booms):
            return
        self.uvs, self._loaded_key = uvs, key
        self.redraw()

    # --- which slots ------------------------------------------------------------------------------

    def _in_range(self, slots: np.ndarray) -> np.ndarray:
        if self.x_range is None:
            return np.ones(slots.size, dtype=bool)
        t = slot_to_unix(slots)
        return (t >= self.x_range[0]) & (t < self.x_range[1])

    def groups(self) -> list[tuple[str, object, dict]]:
        """[(label, color, {member: UV})] for the chosen slots."""
        kind = self.group.currentData() or "all"
        if kind == "classes":
            wanted = [(n, style.stability_color(n, i), f"class:{n}") for i, n in enumerate(self.stability[1])]
        else:
            wanted = [(self.group.currentText(), pg.mkColor(_LINE), kind)]
        out = []
        tower = [m for m in self.uvs if m != MESONET]
        for label, color, which in wanted:
            uvs = {}
            for m, w in self.uvs.items():
                keep = self._in_range(w.slots)
                if which.startswith("class:"):
                    keep &= in_class(self.stability, w.slots, which[len("class:"):])
                elif which == "selected":
                    keep &= np.isin(w.slots, self.selection)
                elif which == "rest":
                    keep &= ~np.isin(w.slots, self.selection)
                uvs[m] = w.only(keep)
            if self.common.isChecked() and tower:
                shared = hodograph.common_slots(uvs, tower)
                uvs = {m: w.only(np.isin(w.slots, shared)) for m, w in uvs.items()}
            out.append((label, color, uvs))
        return out

    # --- drawing ----------------------------------------------------------------------------------

    def redraw(self) -> None:
        self.plot.clear()
        self.plot.getPlotItem().legend.clear()
        self.drawn = []
        if self.uvs is None:
            return
        groups = self.groups()
        overlaid = len(groups) > 1
        n_slots = 0
        for label, color, uvs in groups:
            rows = hodograph.summary(uvs)
            self.drawn.append((label, color, rows))
            n_slots = max(n_slots, int(rows["n"].max()) if len(rows) else 0)
        few = n_slots <= EACH_SLOT_LIMIT
        self.each_slot.setEnabled(not overlaid and few and self.common.isChecked())
        self.each_slot.setToolTip("each slot's own hodograph, faintly behind the mean" if self.each_slot.isEnabled()
                                  else f"for one group of at most {EACH_SLOT_LIMIT} slots, with only slots every "
                                       "boom has")
        if self.each_slot.isEnabled() and self.each_slot.isChecked():
            self._each_slot(groups[0][2])
        draw_origin(self.plot)
        for label, color, rows in self.drawn:
            labels = ("ends" if overlaid else "all") if self.labels.isChecked() else None
            draw_hodograph(self.plot, rows, color, label if overlaid else "", labels)
        self._fill_table(overlaid)
        scope = "whole period" if self.x_range is None else "viewed interval"
        common = "the same slots at every height" if self.common.isChecked() else "each height's own slots"
        self.info.setText(f"{self.group.currentText()}, {scope}: up to {n_slots} slots, {common} · mean of the slots' "
                          "vector-mean winds; ★ the mesonet's 10 m wind · steadiness = |mean vector| / mean speed")

    def _each_slot(self, uvs: dict) -> None:
        tower = sorted((m for m in uvs if m != MESONET), key=hodograph.height)
        if not tower:
            return
        slots = uvs[tower[0]].slots
        u = np.full((slots.size, len(tower) + 1), np.nan)  # a NaN column breaks the line between slots
        v = u.copy()
        for j, m in enumerate(tower):
            idx = np.searchsorted(uvs[m].slots, slots)
            u[:, j], v[:, j] = uvs[m].u[idx], uvs[m].v[idx]
        faint = pg.mkColor(_LINE)
        faint.setAlpha(40)
        self.plot.plot(u.ravel(), v.ravel(), pen=pg.mkPen(faint, width=1), connect="finite")

    def _fill_table(self, overlaid: bool) -> None:
        frames = []
        for label, _, rows in self.drawn:
            df = rows.copy()
            df.insert(0, "where", [hodograph.MESONET if m == MESONET else f"b{m}" for m in df["member"]])
            if overlaid:
                df.insert(0, "group", label)
            frames.append(df.drop(columns=["member"]))
        if not frames:
            self.table.clear()
            return
        df = pd.concat(frames, ignore_index=True).rename(columns={
            "height": "height [m]", "n": "N", "u": "ū [m/s]", "v": "v̄ [m/s]", "speed": "|mean| [m/s]",
            "from": "from [°]", "mean_speed": "mean speed [m/s]"})
        tables.fill(self.table, df)

    def figure_spec(self) -> tuple[str, dict] | None:
        if not self.drawn:
            return None
        lines = []
        for label, color, rows in self.drawn:
            tower = rows[(rows["member"] != MESONET) & (rows["n"] > 0)]
            meso = rows[(rows["member"] == MESONET) & (rows["n"] > 0)]
            lines.append({"label": label if len(self.drawn) > 1 else "", "color": pg.mkColor(color).name(),
                          "u": tower["u"].to_numpy(float), "v": tower["v"].to_numpy(float),
                          "heights": tower["height"].to_numpy(float),
                          "mesonet": (float(meso["u"].iloc[0]), float(meso["v"].iloc[0])) if len(meso) else None})
        return "hodograph", {"lines": lines, "labels": self.labels.isChecked(), "title": self.info.text().split(":")[0]}
