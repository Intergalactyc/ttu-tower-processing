"""The main window: two x-linked timelines of any quantities, their
distributions, a scatter of one against the other and wind roses, stability
and anisotropy bands, a whole-period overview, and click-through to the raw
data.
"""
import logging

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QFileDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QSplitter,
    QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.timegrid import time_to_slot
from ttu_tower.viewer import anisotropy, timeline
from ttu_tower.viewer.catalog import build_catalog
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.provenance import drill_target, ordered_variables, scale_variables
from ttu_tower.viewer.reprocess import Reprocessor
from ttu_tower.viewer.run import RunHandle, list_registered
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, format_slot, parse_time, slot_to_unix
from ttu_tower.viewer.ui.distribution import DistributionView
from ttu_tower.viewer.ui.overview import AnisotropyBand, OverviewStrip, StabilityBand
from ttu_tower.viewer.ui.panel import PanelControls, PanelSpec
from ttu_tower.viewer.ui.inspector.inspector import InspectorWindow
from ttu_tower.viewer.ui.scatter import ScatterView
from ttu_tower.viewer.ui.windrose_view import WindRoseView
from ttu_tower.viewer.ui.timeline_plot import TimelinePlot
from ttu_tower.viewer.ui.worker import JobRunner

_DAY = 86_400.0
_ZOOMS = (("Y", 366 * _DAY), ("M", 31 * _DAY), ("W", 7 * _DAY), ("D", _DAY), ("6h", _DAY / 4))
_HF_VARIABLES = ["ue", "vn", "w", "ts", "vpts", "t", "rh", "p"]
_DEFAULTS = (("boom_final|ws|mean", "none", (1, 5, 10)), ("boom_final|ustar|", "mrd", (1, 5, 10)))
_FALLBACKS = (("coverage|momentum|usable", "none", (1,)), ("slot_qc|ts|skew", "none", (1,)))
_TAB_FOR_TARGET = {"flux": "spectra", "spectra": "spectra", "acf": "scales", "profile": "profile", "none": "numbers"}


class MainWindow(QMainWindow):
    runOpened = Signal()

    def __init__(self, run: RunHandle | None = None, runner: JobRunner | None = None):
        super().__init__()
        self.settings = QSettings("ttu-tower", "ttu-view")
        self.runner = runner or JobRunner(parent=self)
        self.run: RunHandle | None = None
        self.index: FragmentIndex | None = None
        self.reprocessor: Reprocessor | None = None
        self.catalog = None
        self.specs = [PanelSpec(), PanelSpec()]
        self.datas = [None, None]
        self.night = None
        self.stability = None
        self.inspectors: list[InspectorWindow] = []
        self._have_range = False

        self._build_toolbar()
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("background:#fff4c2; padding:4px;")
        self.banner.hide()

        self.plots = [TimelinePlot(), TimelinePlot()]
        self.plots[1].plot_item.setXLink(self.plots[0].plot_item)
        self.stability_band = StabilityBand()
        self.stability_band.link_to(self.plots[0].plot_item)
        self.aniso_band = AnisotropyBand()
        self.aniso_band.link_to(self.plots[0].plot_item)
        self.aniso_band.variantChanged.connect(lambda *_: self._load_aniso_band())
        self.aniso_band.cellClicked.connect(
            lambda slot, boom: QTimer.singleShot(0, lambda: self.inspect(
                boom, slot, variant=self.aniso_band.variant.currentData(), tab="anisotropy")))
        self.overview = OverviewStrip()
        self.overview.rangeRequested.connect(lambda x0, x1: self.set_x_range(x0, x1))
        splitter = QSplitter(Qt.Orientation.Vertical)
        for p in self.plots:
            splitter.addWidget(p)
        splitter.addWidget(self.stability_band)
        splitter.addWidget(self.aniso_band)
        splitter.addWidget(self.overview)
        splitter.setSizes([400, 400, 50, 120, 140])
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.banner)
        layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        self.panels = [PanelControls("Panel A"), PanelControls("Panel B")]
        tabs = QTabWidget()
        for i, panel in enumerate(self.panels):
            tabs.addTab(panel, "Panel A" if i == 0 else "Panel B")
            panel.specChanged.connect(lambda spec, i=i: self._on_spec(i, spec))
        self.night_box = QCheckBox("shade night")
        self.night_box.setChecked(True)
        self.night_box.toggled.connect(self._apply_night)
        self.stability_box = QCheckBox("stability band")
        self.stability_box.setChecked(True)
        self.stability_box.toggled.connect(self.stability_band.setVisible)
        self.aniso_box = QCheckBox("anisotropy band")
        self.aniso_box.setChecked(True)
        self.aniso_box.toggled.connect(self.aniso_band.setVisible)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(tabs, stretch=1)
        left_layout.addWidget(self.night_box)
        left_layout.addWidget(self.stability_box)
        left_layout.addWidget(self.aniso_box)
        self._dock("Controls", left, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.dists = [DistributionView(self.runner), DistributionView(self.runner)]
        self.full_period = QCheckBox("whole period (not just the visible range)")
        self.full_period.setToolTip("distributions, scatter and wind roses use the viewed interval unless this is checked")
        self.full_period.toggled.connect(lambda *_: self._refresh_distributions())
        dist_split = QSplitter(Qt.Orientation.Vertical)
        for d in self.dists:
            dist_split.addWidget(d)
        self.scatter = ScatterView(self.runner)
        self.scatter.pointClicked.connect(lambda slot, member: QTimer.singleShot(0, lambda: self._on_scatter_point(slot, member)))
        self.windrose = WindRoseView(self.runner)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(dist_split, "Distributions")
        self.right_tabs.addTab(self.scatter, "Scatter")
        self.right_tabs.addTab(self.windrose, "Wind rose")
        self.right_tabs.currentChanged.connect(lambda *_: self._refresh_distributions())
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.addWidget(self.full_period)
        right_layout.addWidget(self.right_tabs, stretch=1)
        self._dock("Distributions", right, Qt.DockWidgetArea.RightDockWidgetArea)

        self.hover = QLabel()
        self.busy = QLabel()
        self.warn_label = QLabel()
        self.warn_label.setTextFormat(Qt.TextFormat.RichText)
        self.warn_label.linkActivated.connect(self._open_log)
        self.warn_label.setToolTip("warnings and Qt messages go to the viewer's log file, not the console")
        self.statusBar().addWidget(self.hover, 1)
        self.statusBar().addPermanentWidget(self.warn_label)
        self.statusBar().addPermanentWidget(self.busy)
        from ttu_tower.viewer.ui import logs
        if logs.counter is not None:
            logs.counter.warned.connect(
                lambda n: self.warn_label.setText(f"<a href='log'>{n} warning{'s' if n != 1 else ''} logged</a>"))
        self.runner.busyChanged.connect(self._on_busy)

        self._dist_timer = QTimer(self)
        self._dist_timer.setSingleShot(True)
        self._dist_timer.setInterval(150)
        self._dist_timer.timeout.connect(self._refresh_distributions)
        self.plots[0].vb.sigXRangeChanged.connect(self._on_x_range)
        for i, p in enumerate(self.plots):
            # queued: open the inspector once the click has finished, or Windows can hand focus back to
            # this window and leave the new one hidden behind it
            p.pointClicked.connect(lambda slot, member, i=i: QTimer.singleShot(0, lambda: self._on_point(i, slot, member)))
            p.filteredClicked.connect(
                lambda slot, member, i=i: QTimer.singleShot(0, lambda: self._on_point(i, slot, member, filtered=True)))
            p.vb.fitXRequested.connect(self.zoom_all)
            p.hovered.connect(self._on_hover)
            p.legend.toggled.connect(lambda i=i: self._dist_timer.start())
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._show_all_members)

        self.setWindowTitle("ttu-view")
        self.resize(1600, 950)
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if run is not None:
            self.open_run(run)

    # --- layout helpers ------------------------------------------------------------------------

    def _dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_toolbar(self) -> None:
        bar = QToolBar("main")
        bar.setObjectName("main")
        self.addToolBar(bar)
        self.run_box = QComboBox()
        self.run_box.setMinimumWidth(160)
        self.run_box.activated.connect(self._on_run_chosen)
        bar.addWidget(QLabel(" run "))
        bar.addWidget(self.run_box)
        for label, tip, slot in (("Open from directory…", "open a run by its directory", self._open_directory),
                                 ("Raw data folder", "show the run's raw data folder", self._show_raw_folder),
                                 ("Run folder", "show the run's results folder", self._show_run_folder)):
            act = QAction(label, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            bar.addAction(act)
        bar.addSeparator()
        self.tz_box = QComboBox()
        self.tz_box.currentIndexChanged.connect(self._on_tz)
        bar.addWidget(QLabel(" time zone "))
        bar.addWidget(self.tz_box)
        bar.addSeparator()
        self.goto = QLineEdit()
        self.goto.setPlaceholderText("go to: 2013-11-09 12:10")
        self.goto.setMaximumWidth(180)
        self.goto.returnPressed.connect(self._on_goto)
        bar.addWidget(self.goto)
        self.goto_boom = QComboBox()
        self.goto_boom.setToolTip("the boom 'Inspect' opens")
        for b in sorted(HEIGHTS):
            self.goto_boom.addItem(f"b{b}", b)
        bar.addWidget(self.goto_boom)
        inspect = QAction("Inspect", self)
        inspect.setToolTip("open the Slot Inspector at the go-to time, for the chosen boom")
        inspect.triggered.connect(self._on_inspect_goto)
        bar.addAction(inspect)
        for label, width in _ZOOMS:
            act = QAction(label, self)
            act.setToolTip(f"zoom to {label}")
            act.triggered.connect(lambda _=False, w=width: self.zoom(w))
            bar.addAction(act)
        act = QAction("all", self)
        act.triggered.connect(self.zoom_all)
        bar.addAction(act)
        for label, step in (("◀", -1), ("▶", 1)):
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, s=step: self.step(s))
            bar.addAction(act)
        self.range_label = QLabel()
        bar.addWidget(self.range_label)
        self._refresh_run_list()

    def _refresh_run_list(self) -> None:
        self.run_box.blockSignals(True)
        self.run_box.clear()
        for row in list_registered():
            if row["exists"] and "primary" in row["stages"]:
                self.run_box.addItem(row["tag"], row["tag"])
        self.run_box.blockSignals(False)

    # --- runs -------------------------------------------------------------------------------------

    def _on_run_chosen(self, index: int) -> None:
        tag = self.run_box.itemData(index)
        if tag and (self.run is None or self.run.tag != tag):
            self.open_run(RunHandle.open(tag))

    def _open_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open a run directory")
        if path:
            try:
                self.open_run(RunHandle.from_dir(path))
            except Exception as exc:
                QMessageBox.warning(self, "ttu-view", str(exc))

    def _show_raw_folder(self) -> None:
        if self.run is None:
            return
        folders = sorted({str(p.parent) for p in self.run.accepted.values()})
        if self.run.cfg is not None:
            folders = [d for d in self.run.cfg.paths.raw_dirs if d] or folders
        if not folders:
            self.statusBar().showMessage("this run has no raw files on record", 6000)
        for folder in folders:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(folder)):
                self.statusBar().showMessage(f"can't open {folder}", 8000)

    def _show_run_folder(self) -> None:
        if self.run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run.run_dir)))

    def open_run(self, run: RunHandle) -> None:
        if self.reprocessor is not None:
            self.reprocessor.close()
        self.run = run
        self.index = FragmentIndex(run.run_dir)
        self.reprocessor = Reprocessor(run) if run.cfg is not None else None
        self.catalog, self.datas, self.night, self._have_range = None, [None, None], None, False
        self.stability = None
        self.specs = [PanelSpec(), PanelSpec()]
        self.settings.setValue("last_run", run.tag)
        self.setWindowTitle(f"ttu-view — {run.tag}  ({run.run_dir})")
        i = self.run_box.findData(run.tag)
        if i >= 0:
            self.run_box.setCurrentIndex(i)

        self.tz_box.blockSignals(True)
        self.tz_box.clear()
        self.tz_box.addItem(f"{run.timezone} (run)", run.timezone)
        if run.timezone != "UTC":
            self.tz_box.addItem("UTC", "UTC")
        self.tz_box.blockSignals(False)
        self._on_tz()

        notes = []
        if run.config_problems:
            notes.append("Reprocessing disabled: " + "; ".join(run.config_problems))
        elif run.cfg is None:
            notes.append("Reprocessing disabled: the run's config is unavailable.")
        commit = run.commit_note()
        if commit:
            notes.append("Code differs from the run: " + commit + ". Raw-data views show the installed code's result.")
        self.banner.setText("\n".join(notes))
        self.banner.setVisible(bool(notes))

        for p in self.plots:
            p.clear("loading…")
        self.runner.submit(build_catalog, run, key="catalog", label="reading the run's quantities",
                           on_done=self._catalog_ready, on_error=self._report_error)
        booms = run.booms
        self.runner.submit(_overlays, run, self.index, booms, key="overlays", label="loading overlays",
                           on_done=lambda result: self._overlays_ready(result, booms), on_error=self._report_error)
        self.aniso_band.set_booms(booms)
        self.windrose.set_run(run, self.index, booms)
        self._load_aniso_band()

    def _load_aniso_band(self) -> None:
        if self.run is None:
            return
        run, booms, variant = self.run, self.run.booms, self.aniso_band.variant.currentData()
        self.runner.submit(anisotropy.class_image, run, self.index, variant, booms, key="aniso_band",
                           label="loading anisotropy classes", on_error=self._report_error,
                           on_done=lambda result: run is self.run and self.aniso_band.set_classes(*result, booms))

    def _catalog_ready(self, catalog) -> None:
        self.catalog = catalog
        for panel in self.panels:
            panel.set_catalog(catalog, self.run.booms)
        defaults = [d for d in _DEFAULTS + _FALLBACKS if d[0] in catalog.quantities]
        for panel, (key, variant, members) in zip(self.panels, defaults):
            q = catalog[key]
            members = tuple(b for b in members if b in self.run.booms) or tuple(self.run.booms[:1])
            panel.select(key, variant if variant in q.variants else q.variants[0], members)
        self.runOpened.emit()

    def _overlays_ready(self, result, booms) -> None:
        slots, image, stability, night = result
        self.night, self.stability = night, stability
        self.overview.set_availability(slots, image, booms)
        self.stability_band.set_classes(slots, stability)
        self._apply_night()
        if not self._have_range:
            self.zoom_all()

    def _report_error(self, exc, tb) -> None:
        self.statusBar().showMessage(f"error: {exc}", 15_000)
        logging.getLogger("ttu_view").error(tb)

    def _open_log(self, *_) -> None:
        from ttu_tower.viewer.ui import logs
        if logs.log_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs.log_path)))

    # --- panels -----------------------------------------------------------------------------------

    def _on_spec(self, i: int, spec: PanelSpec) -> None:
        self.specs[i] = spec
        if spec.quantity is None or not spec.members:
            self.plots[i].clear("choose a quantity and at least one boom")
            self.datas[i] = None
            self._refresh_distributions()
            return
        self.plots[i].title.setText(f"loading {spec.quantity.title}…")
        self.runner.submit(timeline.load, self.run, self.index, spec.quantity, spec.variant, spec.members,
                           key=f"panel{i}", label=f"loading {spec.quantity.label}",
                           on_done=lambda data, i=i, spec=spec: self._panel_loaded(i, spec, data),
                           on_error=self._report_error)

    def _panel_loaded(self, i: int, spec: PanelSpec, data) -> None:
        if spec is not self.specs[i]:  # the panel changed (or the run did) while this loaded
            return
        self.datas[i] = data
        self.plots[i].set_data(data, spec)
        self._load_overlays(i, spec)
        self._sync_axis_widths()
        self._apply_night()
        if not self._have_range:
            self.zoom_all()
        self._refresh_distributions()

    def _load_overlays(self, i: int, spec: PanelSpec) -> None:
        """Filtered values, then tau statuses (which sit on them where the final value is gone)."""
        def tau(_=None):
            if spec.show_tau and spec is self.specs[i]:
                self.runner.submit(timeline.tau_status, self.run, self.index, spec.variant, spec.members,
                                   key=f"tau{i}", label="reading τ statuses",
                                   on_done=lambda marks: spec is self.specs[i] and self.plots[i].set_tau_status(marks),
                                   on_error=self._report_error)

        if not spec.show_filtered:
            tau()
            return

        def filtered_ready(overlay):
            if spec is self.specs[i]:
                self.plots[i].set_filtered(overlay)
                tau()

        self.runner.submit(timeline.filtered, self.run, self.index, spec.quantity, spec.variant, spec.members,
                           key=f"filtered{i}", label="reading filtered values", on_done=filtered_ready,
                           on_error=self._report_error)

    def _sync_axis_widths(self) -> None:
        """One left-axis width for everything stacked in the middle, wide enough
        for the longest category label, so their x axes stay lined up.
        """
        width = max(p.axis_width_needed() for p in self.plots)
        for p in self.plots:
            p.plot_item.getAxis("left").setWidth(width)
        self.stability_band.set_axis_width(width)
        self.aniso_band.set_axis_width(width)
        self.overview.set_axis_width(width)

    def _show_all_members(self) -> None:
        for p in self.plots:
            p.legend.show_all()

    def _apply_night(self) -> None:
        for p in self.plots:
            p.set_night(self.night, self.night_box.isChecked())

    # --- ranges -----------------------------------------------------------------------------------

    def x_range(self) -> tuple[float, float]:
        return tuple(self.plots[0].vb.viewRange()[0])

    def set_x_range(self, x0: float, x1: float) -> None:
        self._have_range = True
        self.plots[0].vb.setXRange(x0, x1, padding=0)

    def zoom(self, width: float) -> None:
        x0, x1 = self.x_range()
        center = (x0 + x1) / 2
        self.set_x_range(center - width / 2, center + width / 2)

    def zoom_all(self) -> None:
        if self.run is None:
            return
        a, b = self.run.period
        self.set_x_range(slot_to_unix(a), slot_to_unix(b))

    def step(self, direction: int) -> None:
        x0, x1 = self.x_range()
        shift = direction * (x1 - x0) * 0.8
        self.set_x_range(x0 + shift, x1 + shift)

    def _on_x_range(self, *_):
        x0, x1 = self.x_range()
        self.overview.show_range(x0, x1)
        tz = self.tz_box.currentData() or "UTC"
        a, b = (pd.Timestamp(v, unit="s", tz="UTC").tz_convert(tz) for v in (x0, x1))
        self.range_label.setText(f"  {a:%Y-%m-%d %H:%M} – {b:%Y-%m-%d %H:%M}")
        if not self.full_period.isChecked():
            self._dist_timer.start()

    def _on_tz(self, *_):
        tz = self.tz_box.currentData()
        if tz:
            for p in self.plots:
                p.set_timezone(tz)
            self.overview.set_timezone(tz)

    def _goto_slot(self) -> int | None:
        if self.run is None:
            return None
        try:
            t = parse_time(self.goto.text(), self.tz_box.currentData() or self.run.timezone)
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(f"can't read that time: {exc}", 8000)
            return None
        return int(time_to_slot(t.tz_convert("UTC")))

    def _on_inspect_goto(self) -> None:
        k = self._goto_slot()
        if k is not None:
            self.inspect(self.goto_boom.currentData(), k)

    def _on_goto(self) -> None:
        k = self._goto_slot()
        if k is None:
            return
        x = slot_to_unix(k) + SLOT_SECONDS / 2
        self.set_x_range(x - _DAY / 2, x + _DAY / 2)
        for p in self.plots:
            marker = pg.InfiniteLine(slot_to_unix(k), pen=pg.mkPen("#d62728", width=1, style=Qt.PenStyle.DashLine))
            p.plot_item.addItem(marker)
            QTimer.singleShot(4000, lambda p=p, m=marker: p.plot_item.removeItem(m))

    def _refresh_distributions(self) -> None:
        """Whichever right-hand tab is showing (the others catch up when shown)."""
        x_range = None if self.full_period.isChecked() else self.x_range()
        shown = self.right_tabs.currentWidget()
        if shown is self.scatter:
            self.scatter.set_sources(self.datas, x_range, self.stability)
        elif shown is self.windrose:
            self.windrose.set_range(x_range)
        else:
            for data, plot, dist in zip(self.datas, self.plots, self.dists):
                if data is None:
                    dist.set_data(None, [], None)
                else:
                    dist.set_data(data, plot.visible_members(), x_range)

    def _on_scatter_point(self, slot: int, member) -> None:
        data = self.scatter._data(0)
        q = data.quantity if data is not None else None
        if isinstance(member, tuple):
            boom = member[0]
        elif isinstance(member, int):
            boom = member
        else:
            boom = self.goto_boom.currentData()
        variant = self.specs[self.scatter.source[0].currentData()].variant
        self.inspect(boom, slot, q, variant)

    # --- clicks -----------------------------------------------------------------------------------

    def _on_hover(self, slot: int, values: str) -> None:
        if self.run is not None:
            self.hover.setText(f"{format_slot(slot, self.tz_box.currentData() or self.run.timezone)}"
                               f"  (slot {slot})   {values}")

    def _on_point(self, i: int, slot: int, member, filtered: bool = False) -> None:
        q = self.specs[i].quantity
        if q is None:
            return
        if q.kind == "pair":
            boom = member[0]
        elif q.kind == "slot":
            boom = self.inspectors[-1].boom if self.inspectors else self.goto_boom.currentData()
        else:
            boom = member
        self.inspect(boom, slot, q, self.specs[i].variant, tab="qc" if filtered else None)

    def inspect(self, boom: int, slot: int, q=None, variant: str = "mrd", tab: str | None = None) -> "InspectorWindow":
        """Open (or reuse an unpinned) Slot Inspector on the tab that explains `q`
        (or on `tab`).
        """
        window = next((w for w in self.inspectors if not w.pin.isChecked()), None)
        if window is None:
            reprocessor = self.reprocessor if self.run.can_reprocess else None
            window = InspectorWindow(self.run, self.index, reprocessor, self.runner)
            window.closed.connect(lambda w: self.inspectors.remove(w) if w in self.inspectors else None)
            self.inspectors.append(window)
        explains, frame, emphasis, focus = "series", None, None, ()
        if q is not None:
            target = drill_target(q)
            explains = _TAB_FOR_TARGET.get(target.kind, "series")
            if target.kind == "qc" and (q.table != "flags" or window.reprocessor is None):
                explains = "qc"  # a flag fraction is best seen on the series' flag rug
            elif explains == "series" and window.reprocessor is None:
                explains = "spectra"
            frame = "streamwise" if target.kind == "fluctuation" else None
            emphasis = target.spectrum
            focus = scale_variables(q)
            context = f"from {q.title}" + ("" if variant == "none" else f" ({variant})")
            variables = ordered_variables(target, _HF_VARIABLES)
        else:
            context, variables = "", list(_HF_VARIABLES)
        window.show_slot(boom, slot, tab=tab or explains, mode="unexcised", variables=variables, context=context,
                         frame=frame, emphasis=emphasis, focus_vars=focus, variant=variant)
        return window

    def _on_busy(self, count: int, label: str) -> None:
        self.busy.setText(f"⏳ {label}…" if count else "")

    def closeEvent(self, ev) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        for w in list(self.inspectors):
            w.close()
        self.runner.wait(5000)
        if self.reprocessor is not None:
            self.reprocessor.close()
        super().closeEvent(ev)


def _overlays(run, index, booms):
    """Runs on a worker thread."""
    slots, image = timeline.availability(run, index, booms)
    return slots, image, timeline.stability(run, index), timeline.night(run, index)
