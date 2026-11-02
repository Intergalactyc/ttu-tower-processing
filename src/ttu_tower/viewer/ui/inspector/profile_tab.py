"""The slot's vertical profiles: each boom's value against height, the
values tertiary filtered ghosted at what they were, the stored profile fits
drawn through them, and the boom-pair quantities. Further profiles of any
per-boom quantity in the catalog (derived ones too) can be added; they are
read as the timelines read them, in the background.
"""
import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from ttu_tower.viewer import profiledata, timeline
from ttu_tower.viewer.ui import style, tables, variant_combo
from ttu_tower.viewer.ui.axes import Log10Axis
from ttu_tower.viewer.ui.search_combo import SearchCombo
from ttu_tower.viewer.ui.wrap_tabs import FlowLayout

_GHOST = "#9a9a9a"
_FIT_PENS = {"alpha": ("#d62728", Qt.PenStyle.SolidLine), "loglaw": ("#1f77b4", Qt.PenStyle.DashLine),
             "loglaw_c": ("#17becf", Qt.PenStyle.DotLine), "gamma": ("#d62728", Qt.PenStyle.SolidLine),
             "wdgamma": ("#d62728", Qt.PenStyle.SolidLine)}
_COLUMNS = 8


class ProfileTab(QWidget):
    boomClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.across = self.cfg = None
        self.variant, self.boom = "mrd", None
        self.log_z = QCheckBox("log height")
        self.log_z.toggled.connect(lambda *_: self.redraw())
        self.fits_box = QCheckBox("fits")
        self.fits_box.setChecked(True)
        self.fits_box.toggled.connect(lambda *_: self.redraw())
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.pairs = tables.new_table()
        self.fit_table = tables.new_table()
        self.run = self.index = self.runner = None
        self.catalog_of = lambda: None
        self.extras: list[tuple[str, str]] = []  # added profiles: (quantity key, variant; "" follows the inspector's)
        self._filled_for = None
        self._token = 0  # a redraw's: extra profiles read for an earlier one are dropped
        self.add_quantity = SearchCombo()
        self.add_quantity.setToolTip("another profile: any per-boom quantity, derived ones too (type to search)")
        self.add_quantity.setMinimumContentsLength(18)
        self.add_quantity.currentIndexChanged.connect(lambda *_: self._sync_add_variant())
        self.add_variant = QComboBox()
        self.add_variant.setToolTip("its variant; “as the inspector” follows the variant chosen above")
        self.add_button = QPushButton("+ profile")
        self.add_button.clicked.connect(lambda *_: self.add_extra())
        self.chips = QWidget()  # a button per added profile, which removes it
        self.chip_flow = FlowLayout(self.chips)

        controls = QHBoxLayout()
        controls.addWidget(self.log_z)
        controls.addWidget(self.fits_box)
        controls.addWidget(QLabel(f"<span style='color:{_GHOST}'>○ filtered by tertiary (value before filtering)"
                                  "</span> · click a point to inspect that boom"))
        controls.addStretch(1)
        adding = QHBoxLayout()
        adding.addWidget(QLabel("add"))
        adding.addWidget(self.add_quantity, stretch=1)
        adding.addWidget(self.add_variant)
        adding.addWidget(self.add_button)
        adding.addWidget(self.chips, stretch=2)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self.pairs)
        bottom.addWidget(self.fit_table)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.graphics)
        split.addWidget(bottom)
        split.setSizes([700, 180])
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(adding)
        layout.addWidget(self.info)
        layout.addWidget(split, stretch=1)
        self.plots: dict[str, pg.PlotItem] = {}
        self.points: dict = {}

    def attach(self, run, index, runner, catalog_of) -> None:
        """What added profiles are read from: the run, its index, the worker and the viewer's catalog."""
        self.run, self.index, self.runner, self.catalog_of = run, index, runner, catalog_of
        self._fill_add()

    def _quantity(self, key):
        cat = self.catalog_of()
        return cat.quantities.get(key) if cat is not None and key else None

    def _fill_add(self) -> None:
        cat = self.catalog_of()
        stamp = (id(cat), len(cat.quantities)) if cat is not None else None
        if stamp == self._filled_for:
            return
        self._filled_for = stamp
        qs = [q for qs in cat.grouped().values() for q in qs if q.kind == "boom" and not q.categorical] if cat else []
        self.add_quantity.fill_quantities(qs, none_label="— a per-boom quantity —")
        self._sync_add_variant()

    def _sync_add_variant(self) -> None:
        q = self._quantity(self.add_quantity.currentData())
        self.add_variant.clear()
        if q is not None:
            if len(q.variants) > 1:
                self.add_variant.addItem("as the inspector", "")
            for v in q.variants:
                self.add_variant.addItem(variant_combo.label(v), v)
        self.add_variant.setEnabled(self.add_variant.count() > 1)
        self.add_button.setEnabled(q is not None)

    def add_extra(self, key: str | None = None, variant: str | None = None) -> None:
        key = key or self.add_quantity.currentData()
        variant = variant if variant is not None else (self.add_variant.currentData() or "")
        if not key or (key, variant) in self.extras:
            return
        self.extras.append((key, variant))
        self._fill_chips()
        self.redraw()

    def remove_extra(self, i: int) -> None:
        del self.extras[i]
        self._fill_chips()
        self.redraw()

    def _fill_chips(self) -> None:
        while self.chip_flow.count():
            item = self.chip_flow.takeAt(0)
            item.widget().deleteLater()
        for i, (key, variant) in enumerate(self.extras):
            q = self._quantity(key)
            button = QToolButton()
            button.setText(f"✕ {q.title if q is not None else key}{f' · {variant}' if variant else ''}")
            button.setToolTip("remove this profile")
            button.setAutoRaise(True)
            button.clicked.connect(lambda *_, i=i: self.remove_extra(i))
            self.chip_flow.addWidget(button)
        self.chips.updateGeometry()

    def _variant_for(self, q, choice: str) -> str:
        if choice:
            return choice
        if self.variant in q.variants:
            return self.variant
        return "none" if "none" in q.variants else q.variants[0]

    def set_across(self, across, variant: str, boom: int, cfg) -> None:
        self.across, self.variant, self.boom, self.cfg = across, variant, boom, cfg
        self._fill_add()
        self.fits_box.setEnabled(cfg is not None)
        self.fits_box.setToolTip("" if cfg is not None else "the run's config is unavailable")
        self.redraw()

    def clear(self, message: str = "") -> None:
        self.across = None
        self.graphics.clear()
        self.plots = {}
        self.info.setText(message)

    def redraw(self) -> None:
        self.graphics.clear()
        self.plots, self.points = {}, {}
        self._token += 1
        a = self.across
        if a is None:
            return
        if "boom_final" not in a.tables:
            self.info.setText("this run has no tertiary stage, so there are no profiles")
            return
        log_z = self.log_z.isChecked()
        built_in = len(profiledata.PROFILE_QUANTITIES)
        for i, pq in enumerate(profiledata.PROFILE_QUANTITIES):
            plot = self._new_plot(i, log_z)
            variant = f" · {self.variant}" if pq.per_variant else ""
            plot.setTitle(f"{pq.label}{f' [{pq.unit}]' if pq.unit else ''}{variant}", size="9pt")
            self.plots[pq.key] = plot
            points = profiledata.profile_points(a, pq, self.variant)
            self.points[pq.key] = points
            self._draw(plot, pq, points, log_z)
        for j, (key, choice) in enumerate(self.extras):
            plot = self._new_plot(built_in + j, log_z)
            q = self._quantity(key)
            if q is None:
                plot.setTitle(f"{key}: not on this run", size="9pt")
                continue
            variant = self._variant_for(q, choice)
            title = f"{q.title}{f' [{q.unit}]' if q.unit else ''}{'' if variant == 'none' else f' · {variant}'}"
            plot.setTitle(f"{title} (reading…)", size="9pt")
            self.plots[f"extra:{j}"] = plot
            self._read_extra(j, plot, q, variant, title, log_z)
        self._describe()

    def _new_plot(self, i: int, log_z: bool):
        axes = {"left": Log10Axis(orientation="left")} if log_z else {}
        plot = self.graphics.addPlot(row=i // _COLUMNS, col=i % _COLUMNS, axisItems=axes)
        plot.showGrid(x=True, y=True, alpha=0.15)
        if i % _COLUMNS == 0:
            plot.setLabel("left", "height [m]")
        if self.plots:
            plot.setYLink(next(iter(self.plots.values())))
        return plot

    def _read_extra(self, j: int, plot, q, variant: str, title: str, log_z: bool) -> None:
        """Read an added quantity (the whole run, cached as the timelines' are) and draw this slot of it."""
        token, slot, booms = self._token, self.across.slot, list(self.across.booms)

        def done(data):
            if not shiboken6.isValid(self) or token != self._token:
                return
            points = profiledata.points_at(data, slot, booms)
            self.points[f"extra:{j}"] = points
            plot.setTitle(title, size="9pt")
            self._draw(plot, None, points, log_z)

        def failed(exc, *_):
            if shiboken6.isValid(self) and token == self._token:
                plot.setTitle(f"{title}: couldn't be read ({exc})", size="9pt")

        if self.runner is None:
            try:
                done(timeline.load(self.run, self.index, q, variant, booms))
            except Exception as exc:  # noqa: BLE001 - shown on the plot
                failed(exc)
            return
        self.runner.submit(timeline.load, self.run, self.index, q, variant, booms,
                           key=f"profile-extra-{id(self)}-{j}", label=f"reading {q.title}",
                           on_done=done, on_error=failed)

    def _z(self, z, log_z: bool):
        return np.log10(z) if log_z else z

    def _draw(self, plot, pq, points, log_z: bool) -> None:
        z = self._z(points["height"].to_numpy(float), log_z)
        v = points["value"].to_numpy(float)
        ok = np.isfinite(v)
        plot.plot(v[ok], z[ok], pen=pg.mkPen("#444444", width=1))
        spots = []
        for (boom, value, pre, filtered, reasons), zi in zip(
                points[["boom", "value", "prefilter", "filtered", "reasons"]].itertuples(index=False), z):
            if filtered:
                spots.append({"pos": (pre, zi), "symbol": "o", "size": 9, "brush": None, "pen": pg.mkPen(_GHOST, width=1.5),
                              "data": (int(boom), f"b{boom} filtered ({reasons}); before filtering {pre:.4g}")})
            elif np.isfinite(value):
                ring = boom == self.boom
                tip = f"b{boom}: {value:.4g}" + (f" · {reasons}" if reasons else "")
                spots.append({"pos": (value, zi), "symbol": "o", "size": 11 if ring else 8,
                              "brush": pg.mkBrush(style.boom_color(boom)),
                              "pen": pg.mkPen("#000000", width=2) if ring else None, "data": (int(boom), tip)})
        scatter = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=lambda x, y, data: data[1])
        scatter.sigClicked.connect(lambda _item, pts, _ev: self.boomClicked.emit(pts[0].data()[0]) if len(pts) else None)
        plot.addItem(scatter)

        if pq is not None and self.fits_box.isChecked() and self.cfg is not None:
            for fit in profiledata.profile_fits(self.across, pq.key, points, self.variant, self.cfg.tertiary.fits):
                color, dash = _FIT_PENS.get(fit.name, ("#d62728", Qt.PenStyle.SolidLine))
                ok = np.isfinite(fit.y)
                plot.plot(fit.y[ok], self._z(fit.z[ok], log_z), pen=pg.mkPen(color, width=1.5, style=dash),
                          name=fit.text)

    def _describe(self) -> None:
        a, cfg = self.across, self.cfg
        parts = []
        if cfg is not None:
            for key in ("ws", "ti", "wd_std"):
                for fit in profiledata.profile_fits(a, key, self.points[key], self.variant, cfg.tertiary.fits):
                    parts.append(fit.text)
        self.info.setText(" · ".join(parts) if parts else "no profile fits for this slot")
        pair = cfg.post.stability.pair if cfg is not None else (0, 0)
        pairs = profiledata.pair_table(a, pair)

        def color(i, col, v):
            return tables.GOOD if i == 0 and len(pairs) and (pairs.iloc[0]["boom"], pairs.iloc[0]["boom2"]) == tuple(pair) else None

        tables.fill(self.pairs, pairs.rename(columns={"rib": "Ri_b"}), color=color)
        self.pairs.setToolTip(f"every boom pair; the stability pair b{pair[0]}–b{pair[1]} first")
        prof = a.table("profile")
        prof = prof[prof["variant"].isin(["none", self.variant])][["variant", "variable", "value"]]
        tables.fill(self.fit_table, prof.reset_index(drop=True))
