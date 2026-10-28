"""The main window: two x-linked timelines of any quantities, their
distributions, a scatter of one against the other and wind roses, stability
and anisotropy bands, a whole-period overview, and click-through to the raw
data; linked brushing across them, a find over any quantities, and bookmarks;
composite spectra, diurnal cycles and anisotropy over an interval, another run
to compare against, and matplotlib figures of any of it; and low-level jets,
detected live from tunable settings, listed and shown on their own band.
"""
import json
import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QSplitter, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.timegrid import time_to_slot
from ttu_tower.viewer import anisotropy, brush, compare, figures, timeline
from ttu_tower.viewer import stability as criteria
from ttu_tower.viewer.bookmarks import Bookmark, BookmarkStore
from ttu_tower.viewer.views import SavedView, ViewStore
from ttu_tower.viewer import derived
from ttu_tower.viewer.ui.derived_panel import DerivedPanel
from ttu_tower.viewer.catalog import build_catalog
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.provenance import drill_target, ordered_variables, scale_variables
from ttu_tower.viewer.reprocess import Reprocessor
from ttu_tower.viewer.run import RunHandle, list_registered
from ttu_tower.viewer.timeaxis import SLOT_SECONDS, format_slot, parse_time, slot_to_unix, unix_to_slot
from ttu_tower.viewer.ui.anisotropy_view import AnisotropyMapView
from ttu_tower.viewer.ui.bookmarks_panel import BookmarkDialog, BookmarksPanel
from ttu_tower.viewer.ui.compare_view import CompareView
from ttu_tower.viewer.ui.composite_view import CompositeSpectraView
from ttu_tower.viewer.ui.scales_view import ScalesView
from ttu_tower.viewer.ui.tau_view import TauView
from ttu_tower.viewer.ui.diurnal_view import DiurnalView
from ttu_tower.viewer.ui.export_dialog import ExportDialog
from ttu_tower.viewer.ui.distribution import DistributionView
from ttu_tower.viewer.ui.find_panel import FindPanel
from ttu_tower.viewer.ui.hodograph_view import HodographView
from ttu_tower.viewer.ui.hover_readout import HoverReadout
from ttu_tower.viewer.ui.jets_controller import JetsController
from ttu_tower.viewer.ui.stability_panel import StabilityPanel
from ttu_tower.viewer.ui.wrap_tabs import WrapTabWidget
from ttu_tower.viewer.ui.overview import AnisotropyBand, OverviewStrip, StabilityBand
from ttu_tower.viewer.ui import style, view_state
from ttu_tower.viewer.ui.panel import PanelControls, PanelSpec
from ttu_tower.viewer.ui.profiles_view import ProfilesView
from ttu_tower.viewer.ui.qc_summary import QCSummaryView
from ttu_tower.viewer.ui.inspector.inspector import InspectorWindow
from ttu_tower.viewer.ui.scatter import ScatterView
from ttu_tower.viewer.ui.windrose_view import WindRoseView
from ttu_tower.viewer.ui.timeline_plot import TimelinePlot
from ttu_tower.viewer.ui.worker import JobRunner

_DAY = 86_400.0
_ZOOMS = (("Y", 366 * _DAY), ("M", 31 * _DAY), ("W", 7 * _DAY), ("D", _DAY), ("6h", _DAY / 4))
_HF_VARIABLES = ["ue", "vn", "w", "ts", "vpts", "t", "rh", "p"]
_ALL = tuple(sorted(HEIGHTS))
_DEFAULTS = (("boom_final|ws|mean", "mrd", _ALL), ("boom_final|ti|", "mrd", _ALL))
_FALLBACKS = (("coverage|momentum|usable", "none", _ALL), ("slot_qc|ts|skew", "none", _ALL))
_TAB_FOR_TARGET = {"flux": "spectra", "spectra": "spectra", "acf": "scales", "profile": "profile", "none": "numbers"}


class MainWindow(QMainWindow):
    runOpened = Signal()

    def __init__(self, run: RunHandle | None = None, runner: JobRunner | None = None, resume: str = "never"):
        """`resume`: "ask" offers, on opening a run, to resume where it was left (and keeps
        where each run is left); "never" opens on the default panels.
        """
        super().__init__()
        self.resume = resume
        self.settings = QSettings()  # the application's organization and name, set in app.main
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
        self.selection = brush.EMPTY  # the brushed slots, shared by every view
        self.other_run = self.other_index = self.other_catalog = None  # the run compared against
        self.others = [None, None]  # its TimelineData per panel, or why it has none
        self.bookmarks = BookmarkStore()
        self.views = ViewStore()
        try:
            self.derived_store = derived.DerivedStore.open()
        except (OSError, ValueError, KeyError) as exc:  # kept in memory, never written over the unreadable file
            self.derived_store = derived.DerivedStore()
            logging.getLogger("ttu_view").error(f"couldn't read the derived quantities: {exc}")
        self._pending_view = None  # a view to apply once the run's quantities are known
        self._pending_summary = None  # the Summary tabs' part of one, once stability classes are known
        self._pending_legend: dict[int, list] = {}  # panel -> the legend entries shown, once it has loaded
        self._stability_known = False  # the run's stability classes (or that it has none) have arrived
        self._summary_retry: dict[str, dict] = {}  # tab -> settings it couldn't take until it's shown
        self._opening: InspectorWindow | None = None  # the inspector the busy cursor waits on
        self._also_here = ""  # the other curves at the clicked point, said again once the inspector opens
        self._opening_timer = QTimer(self)
        self._opening_timer.setSingleShot(True)
        self._opening_timer.setInterval(60_000)  # never leave the cursor busy for good
        self._opening_timer.timeout.connect(lambda: self._opened(self._opening, True))

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
        self.jets = JetsController(self)
        self.jets.band.link_to(self.plots[0].plot_item)
        self.overview = OverviewStrip()
        self.overview.rangeRequested.connect(lambda x0, x1: self.set_x_range(x0, x1))
        splitter = QSplitter(Qt.Orientation.Vertical)
        for p in self.plots:
            splitter.addWidget(p)
        splitter.addWidget(self.stability_band)
        splitter.addWidget(self.aniso_band)
        splitter.addWidget(self.jets.band)
        splitter.addWidget(self.overview)
        splitter.setSizes([400, 400, 50, 120, 50, 140])
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.banner)
        layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        self.panels = [PanelControls("Panel A"), PanelControls("Panel B")]
        tabs = WrapTabWidget()
        for i, panel in enumerate(self.panels):
            tabs.addTab(panel, "Panel A" if i == 0 else "Panel B")
            panel.specChanged.connect(lambda spec, i=i: self._on_spec(i, spec))
        self.find = FindPanel(self.runner, self._find_context)
        self.find.inspectRequested.connect(
            lambda items, start, label: self.inspect_sequence(items, start, label))
        self.find.selectRequested.connect(lambda slots: self.set_selection(slots, "replace"))
        self.find.saveRequested.connect(self._save_found)
        self.bookmark_panel = BookmarksPanel(self.bookmarks)
        self.bookmark_panel.inspectRequested.connect(
            lambda items, start, label, variant: self.inspect_sequence(items, start, label, variant))
        self.bookmark_panel.selectRequested.connect(lambda slots: self.set_selection(slots, "replace"))
        self.bookmark_panel.changed.connect(self._bookmarks_changed)
        tabs.addTab(self.find, "Find")
        tabs.addTab(self.bookmark_panel, "Bookmarks")
        tabs.addTab(self.jets.panel, "Jets")
        self.stability_panel = StabilityPanel()
        self.stability_panel.changed.connect(lambda: self._stability_timer.start())
        self.stability_panel.resetRequested.connect(self._reset_stability)
        tabs.addTab(self.stability_panel, "Stability")
        tabs.setTabToolTip(tabs.count() - 1, "how stability is classified, for every view that uses it")
        self.derived_panel = DerivedPanel(self.derived_store)
        self.derived_panel.changed.connect(self._derived_changed)
        tabs.addTab(self.derived_panel, "Derived")
        tabs.setTabToolTip(tabs.count() - 1, "quantities defined by a formula of others, for the panels")
        self._stability_timer = QTimer(self)
        self._stability_timer.setSingleShot(True)
        self._stability_timer.setInterval(300)
        self._stability_timer.timeout.connect(lambda: self._set_stability(self.stability_panel.spec()))
        self.stability_spec = None  # None: the run's own classification
        self._overlay_slots = np.empty(0, dtype=np.int64)
        self.left_tabs = tabs
        self.night_box = QCheckBox("shade night")
        self.night_box.setChecked(True)
        self.night_box.toggled.connect(self._apply_night)
        self.stability_box = QCheckBox("stability band")
        self.stability_box.setChecked(True)
        self.stability_box.toggled.connect(self.stability_band.setVisible)
        self.aniso_box = QCheckBox("anisotropy band")
        self.aniso_box.setChecked(True)
        self.aniso_box.toggled.connect(self.aniso_band.setVisible)
        self.jets_box = QCheckBox("jets band")
        self.jets_box.setChecked(True)
        self.jets_box.toggled.connect(self.jets.band.setVisible)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(tabs.container(), stretch=1)
        left_layout.addWidget(self.night_box)
        left_layout.addWidget(self.stability_box)
        left_layout.addWidget(self.aniso_box)
        left_layout.addWidget(self.jets_box)
        self.controls_dock = self._dock("Controls", left, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.dists = [DistributionView(self.runner), DistributionView(self.runner)]
        self.full_period = QCheckBox("whole period (not just the visible range)")
        self.full_period.setToolTip("the summary tabs use the viewed interval unless this is checked")
        self.full_period.toggled.connect(lambda *_: self._refresh_distributions())
        dist_split = QSplitter(Qt.Orientation.Vertical)
        for d in self.dists:
            dist_split.addWidget(d)
        self.scatter = ScatterView(self.runner)
        self.scatter.z_loader = self._load_scatter_z
        self.scatter.pointClicked.connect(lambda slot, member: QTimer.singleShot(0, lambda: self._on_scatter_point(slot, member)))
        self.windrose = WindRoseView(self.runner)
        self.hodograph = HodographView(self.runner)
        self.qc_summary = QCSummaryView(self.runner)
        self.profiles = ProfilesView()
        self.composite = CompositeSpectraView(self.runner)
        self.scales_view = ScalesView(self.runner)
        self.tau_view = TauView(self.runner)
        self.tau_view.selectRequested.connect(lambda slots: self.set_selection(slots, "replace"))
        self.diurnal = DiurnalView()
        self.diurnal.cellSelected.connect(lambda slots: self.set_selection(slots, "replace"))
        self.aniso_map = AnisotropyMapView(self.runner)
        self.aniso_map.brushed = self.set_selection
        self.compare = CompareView()
        self.dist_tab = dist_split
        self.right_tabs = WrapTabWidget()
        for widget, label in ((dist_split, "Distributions"), (self.scatter, "Scatter"), (self.profiles, "Profiles"),
                              (self.composite, "Spectra"), (self.scales_view, "Scales"), (self.tau_view, "τ"),
                              (self.diurnal, "Diurnal"),
                              (self.aniso_map, "Anisotropy"),
                              (self.windrose, "Wind rose"), (self.hodograph, "Hodograph"), (self.qc_summary, "QC"),
                              (self.compare, "Compare"),
                              (self.jets.view, "Jets")):
            self.right_tabs.addTab(widget, label)
        for widget, tip in ((self.composite, "composite MRD spectra: each boom's median over the slots"),
                            (self.scales_view, "integral time and length scales at every ladder rung: each "
                                               "boom's median (or mean) over the slots"),
                            (self.tau_view, "heat, momentum and selected τ over many slots, heat vs momentum, and "
                                            "a what-if of the selection rules over every slot"),
                            (self.diurnal, "a panel's quantity by hour of day and month"),
                            (self.compare, "this run against another (choose it in the toolbar)"),
                            (self.jets.view, "low-level jet events (set the detection on the left, under Jets)")):
            self.right_tabs.setTabToolTip(self.right_tabs.indexOf(widget), tip)
        self.right_tabs.currentChanged.connect(lambda *_: self._refresh_distributions())
        self.summary_hover = HoverReadout(self._on_summary_hover)  # the cursor over a Summary plot
        self.right_tabs.currentChanged.connect(lambda *_: self.summary_hover.watch(self.right_tabs))
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.addWidget(self.full_period)
        right_layout.addWidget(self.right_tabs.container(), stretch=1)
        self.summary_dock = self._dock("Summary", right, Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock_buttons()

        self.hover = QLabel()
        self.busy = QLabel()
        self.warn_label = QLabel()
        self.warn_label.setTextFormat(Qt.TextFormat.RichText)
        self.warn_label.linkActivated.connect(self._open_log)
        self.warn_label.setToolTip("warnings and Qt messages go to the viewer's log file, not the console")
        self.statusBar().addWidget(self.hover, 1)
        self.summary_hover.watch(self.right_tabs)
        self.selection_label = QLabel()
        self.selection_label.setToolTip("brushed slots: Shift-drag on a timeline or scatter (or any drag with Brush "
                                        "on); Ctrl adds, Alt removes; Esc clears")
        self.selection_inspect = QToolButton()
        self.selection_inspect.setText("Step through")
        self.selection_inspect.setToolTip("open the Slot Inspector on the selected slots, for the go-to boom")
        self.selection_inspect.clicked.connect(self._inspect_selection)
        self.selection_clear = QToolButton()
        self.selection_clear.setText("Clear")
        self.selection_clear.clicked.connect(lambda: self.set_selection(brush.EMPTY, "replace"))
        for w in (self.selection_label, self.selection_inspect, self.selection_clear):
            self.statusBar().addPermanentWidget(w)
        self._show_selection()
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
            p.alsoHere.connect(self._on_also_here)
            p.brushed.connect(lambda x0, x1, mode: self.set_selection(brush.slots_in_span(x0, x1), mode))
            p.legend.toggled.connect(lambda i=i: self._dist_timer.start())
        self.scatter.brushed.connect(self.set_selection)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._show_all_members)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=lambda: self.set_selection(brush.EMPTY, "replace"))

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

    def _dock_buttons(self) -> None:
        """Toolbar toggles that show or hide the Controls and Summary panels, so a
        detached panel that was closed can be brought back.
        """
        self.dock_actions = {}
        self.toolbar.insertSeparator(self.range_action)
        for dock in (self.controls_dock, self.summary_dock):
            action = dock.toggleViewAction()
            action.setToolTip(f"show or hide the {dock.windowTitle()} panel (docked or detached)")
            self.toolbar.insertAction(self.range_action, action)
            self.dock_actions[dock.windowTitle()] = action

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
        for label, step in (("◀◀", -1.0), ("◀", -0.5), ("▶", 0.5), ("▶▶", 1.0)):
            act = QAction(label, self)
            act.setToolTip(f"{'back' if step < 0 else 'forward'} {'a whole' if abs(step) == 1 else 'half an'} interval")
            act.triggered.connect(lambda _=False, s=step: self.step(s))
            bar.addAction(act)
        self.brush_action = QAction("Brush", self)
        self.brush_action.setCheckable(True)
        self.brush_action.setShortcut(QKeySequence("B"))
        self.brush_action.setToolTip("brush mode (B): dragging on a timeline selects a time span, on the scatter a "
                                     "lasso; without it, Shift-drag does the same. Ctrl adds, Alt removes, Esc clears")
        self.brush_action.toggled.connect(self._set_brush_mode)
        bar.addAction(self.brush_action)
        bar.addSeparator()
        self.compare_box = QComboBox()
        self.compare_box.setToolTip("another run to compare with: its curves dashed on the timelines, and the "
                                    "Compare tab")
        self.compare_box.activated.connect(self._on_compare_chosen)
        bar.addWidget(QLabel(" compare "))
        bar.addWidget(self.compare_box)
        export = QAction("Export figure…", self)
        export.setToolTip("save the timelines or the showing summary tab as a matplotlib figure (PNG, PDF, SVG)")
        export.triggered.connect(self._export_figure)
        bar.addAction(export)
        self.range_label = QLabel()
        self.toolbar = bar
        self.views_button = QToolButton()
        self.views_button.setText("Views")
        self.views_button.setToolTip("save this view (panels, interval, tabs and their settings, the selection) "
                                     "under a name, or open one saved earlier - from this run or another")
        self.views_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.views_menu = QMenu(self.views_button)
        self.views_menu.aboutToShow.connect(self._fill_views_menu)
        self.views_button.setMenu(self.views_menu)
        bar.addWidget(self.views_button)
        self.range_action = bar.addWidget(self.range_label)
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
        if self.run is not None:
            self._save_session()
        if self.reprocessor is not None:
            self.reprocessor.close()
        self.run = run
        self.index = FragmentIndex(run.run_dir)
        self.reprocessor = Reprocessor(run) if run.cfg is not None else None
        self.catalog, self.datas, self.night, self._have_range = None, [None, None], None, False
        self.stability, self._stability_known = None, False
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
        self._offer_resume(run)  # before anything loads: the dialog's event loop would deliver it meanwhile

        for p in self.plots:
            p.clear("loading…")
        self.runner.submit(build_catalog, run, key="catalog", label="reading the run's quantities",
                           on_done=self._catalog_ready, on_error=self._report_error)
        booms = run.booms
        self.runner.submit(_overlays, run, self.index, booms, key="overlays", label="loading overlays",
                           on_done=lambda result: self._overlays_ready(result, booms), on_error=self._report_error)
        self.aniso_band.set_booms(booms)
        self.bookmark_panel.set_run(run.tag, run.timezone)
        self.set_selection(brush.EMPTY, "replace")
        self.windrose.set_run(run, self.index, booms)
        self.hodograph.set_run(run, self.index, booms)
        self.qc_summary.set_run(run, self.index, booms)
        self.composite.set_run(run, self.index, booms)
        self.tau_view.set_run(run, self.index, booms)
        self.scales_view.set_run(run, self.index, booms)
        self.aniso_map.set_run(run, self.index, booms)
        self._fill_compare_box()
        self._set_other_run(None)
        self._load_aniso_band()
        self.jets.set_run(run)
        self._open_stability(run)

    def _load_aniso_band(self) -> None:
        if self.run is None:
            return
        run, booms, variant = self.run, self.run.booms, self.aniso_band.variant.currentData()
        self.runner.submit(anisotropy.class_image, run, self.index, variant, booms, key="aniso_band",
                           label="loading anisotropy classes", on_error=self._report_error,
                           on_done=lambda result: run is self.run and self.aniso_band.set_classes(*result, booms))

    def _catalog_ready(self, catalog) -> None:
        self.catalog = catalog
        notes = derived.add_to(catalog, self.derived_store.items)
        self.derived_panel.set_catalog(catalog, notes)
        for panel in self.panels:
            panel.set_catalog(catalog, self.run.booms)
        self.scatter.set_catalog(catalog)
        if self._pending_view is not None:
            state, self._pending_view = self._pending_view, None
            self._report_view(self.apply_view_state(state))
            self.runOpened.emit()
            return
        defaults = [d for d in _DEFAULTS + _FALLBACKS if d[0] in catalog.quantities]
        for panel, (key, variant, members) in zip(self.panels, defaults):
            q = catalog[key]
            members = tuple(b for b in members if b in self.run.booms) or tuple(self.run.booms[:1])
            panel.select(key, variant if variant in q.variants else q.variants[0], members)
        self.runOpened.emit()

    def _overlays_ready(self, result, booms) -> None:
        slots, image, stability, night = result
        self.night, self._overlay_slots = night, slots
        self.overview.set_availability(slots, image, booms)
        self.qc_summary.set_availability(slots, image)
        if self.stability_spec is None:  # a criterion of one's own arrives by itself
            self._stability_ready(self.run, None, stability)
        else:
            self.stability_band.set_classes(slots, self.stability, self.stability_spec.describe())
        self._apply_night()
        if not self._have_range:
            self.zoom_all()

    # --- stability ------------------------------------------------------------------------------

    def _settings_key(self) -> str:
        return f"stability/{self.run.tag}"

    def _open_stability(self, run) -> None:
        """The run's own classification, or the one last set for this run."""
        own = criteria.from_config(run.cfg.post) if run.cfg is not None else None
        self.stability_spec, self.stability = None, None
        self.stability_panel.set_spec(own or criteria.StabilitySpec())
        raw = self.settings.value(self._settings_key())
        saved = None
        if raw:
            try:
                saved = criteria.StabilitySpec.from_dict(json.loads(raw))
            except (ValueError, TypeError, AttributeError):
                saved = None
        if saved is not None and not saved.errors() and saved != own:
            self.stability_panel.set_spec(saved)
            self._set_stability(saved)
        else:
            self._describe_stability(own)

    def _describe_stability(self, own) -> None:
        if self.stability_spec is not None:
            text = f"in use: {self.stability_spec.describe()}, set here (the run's outputs are unchanged)"
        elif own is not None:
            text = f"in use: the run's own, {own.describe()}"
        else:
            text = "in use: the run's own (its classes aren't five contiguous intervals, so they can't be shown here)"
        self.stability_panel.show_in_use(text)

    def _set_stability(self, spec) -> None:
        if self.run is None:
            return
        own = criteria.from_config(self.run.cfg.post) if self.run.cfg is not None else None
        if spec == own:
            self._reset_stability()
            return
        self.stability_spec = spec
        self.settings.setValue(self._settings_key(), json.dumps(spec.to_dict()))
        self._describe_stability(own)
        run = self.run
        self.runner.submit(timeline.stability, run, self.index, spec, key="stability",
                           label=f"classifying stability by {spec.describe()}", on_error=self._report_error,
                           on_done=lambda result: self._stability_ready(run, spec, result))

    def _reset_stability(self) -> None:
        if self.run is None:
            return
        own = criteria.from_config(self.run.cfg.post) if self.run.cfg is not None else None
        self.stability_spec = None
        self.settings.remove(self._settings_key())
        self.stability_panel.set_spec(own or criteria.StabilitySpec())
        self._describe_stability(own)
        run = self.run
        self.runner.submit(timeline.stability, run, self.index, key="stability", label="classifying stability",
                           on_error=self._report_error,
                           on_done=lambda result: self._stability_ready(run, None, result))

    def _stability_ready(self, run, spec, result) -> None:
        if run is not self.run or spec != self.stability_spec:
            return
        self.stability, self._stability_known = result, True
        own = criteria.from_config(run.cfg.post) if run.cfg is not None else None
        label = spec.describe() if spec is not None else own.describe() if own is not None else ""
        self.stability_band.set_classes(self._overlay_slots, result, label)
        self.stability_panel.show_counts(result)
        if self._pending_summary is not None:
            summary, self._pending_summary = self._pending_summary, None
            self._report_view(self._restore_summary(summary), quiet=True)
        self._refresh_distributions()

    # --- saved views ------------------------------------------------------------------------------

    def summary_views(self) -> dict:
        """Summary tab name -> the view whose controls a saved view keeps."""
        out = {f"Distributions {'AB'[i]}": d for i, d in enumerate(self.dists)}
        for i in range(self.right_tabs.count()):
            widget = self.right_tabs.widget(i)
            if hasattr(widget, "STATE_FIELDS"):
                out[self.right_tabs.tabText(i)] = widget
        return out

    def view_state(self) -> dict:
        """Everything a saved view reopens, as plain values."""
        tz = self.tz_box.currentData()
        return {
            "version": 1, "run": self.run.tag if self.run is not None else None,
            "panels": [p.state() for p in self.panels],
            "legend": [view_state.get(plot.legend) for plot in self.plots],  # the entries shown
            "x_range": list(self.x_range()) if self._have_range else None,
            "timezone": tz if tz != (self.run.timezone if self.run is not None else None) else None,
            "toggles": {name: box.isChecked() for name, box in self._view_toggles().items()},
            "tabs": {"controls": self.left_tabs.tabText(self.left_tabs.currentIndex()),
                     "summary": self.right_tabs.tabText(self.right_tabs.currentIndex())},
            "stability": self.stability_spec.to_dict() if self.stability_spec is not None else None,
            "selection": view_state.pack_slots(self.selection),
            "summary": {name: view_state.save(view, view.STATE_FIELDS) for name, view in self.summary_views().items()},
            "layout": bytes(self.saveState().toBase64()).decode("ascii"),
        }

    def _view_toggles(self) -> dict:
        return {"night": self.night_box, "stability_band": self.stability_box, "aniso_band": self.aniso_box,
                "jets_band": self.jets_box, "whole_period": self.full_period}

    def apply_view_state(self, state: dict) -> list[str]:
        """Reopen a saved view on the open run, as far as the run allows; what couldn't be."""
        notes = []
        if state.get("layout"):
            self.restoreState(QByteArray.fromBase64(state["layout"].encode("ascii")))
        tz = state.get("timezone")
        if tz:
            i = self.tz_box.findData(tz)
            if i >= 0:
                self.tz_box.setCurrentIndex(i)
            else:
                notes.append(f"time zone {tz}")
        for name, on in state.get("toggles", {}).items():
            if name in self._view_toggles():
                self._view_toggles()[name].setChecked(on)
        spec = state.get("stability")
        stability_changes = False
        if spec is not None and self.run is not None and self.run.cfg is not None:
            chosen = criteria.StabilitySpec.from_dict(spec)
            if chosen.errors() or (chosen.variable == "rib" and not set(chosen.booms) <= set(self.run.booms)):
                notes.append(f"the stability criterion {chosen.describe()}")
            elif chosen != self.stability_spec:
                self.stability_panel.set_spec(chosen)
                self._set_stability(chosen)
                stability_changes = True
        elif spec is None and self.stability_spec is not None:
            self._reset_stability()
            stability_changes = True
        for i, (panel, panel_state) in enumerate(zip(self.panels, state.get("panels", []))):
            if panel_state is None:
                continue
            why = panel.restore(panel_state)
            if why:
                notes.append(f"panel {'AB'[i]}: {why}")
            elif state.get("legend") and state["legend"][i]:
                self._pending_legend[i] = state["legend"][i]
        if state.get("x_range"):
            self.set_x_range(*state["x_range"])
        self.set_selection(np.asarray(view_state.unpack_slots(state.get("selection", [])), dtype=np.int64), "replace")
        for tabs, name in ((self.left_tabs, state.get("tabs", {}).get("controls")),
                           (self.right_tabs, state.get("tabs", {}).get("summary"))):
            if name and not view_state.put(tabs, name):
                notes.append(f"the {name} tab")
        summary = state.get("summary", {})
        if self._stability_known and not stability_changes:
            notes += self._restore_summary(summary)
        else:  # its stability-class choices exist once the classes do
            self._pending_summary = summary
        return notes

    def _restore_summary(self, summary: dict) -> list[str]:
        """Settings a tab can't take yet (its lists fill once it's shown) wait for it to be shown."""
        notes = []
        views = self.summary_views()
        for name, fields in summary.items():
            view = views.get(name)
            if view is None:
                notes.append(f"the {name} tab")
                continue
            missed = view_state.restore(view, fields, view.STATE_FIELDS)
            if missed:
                self._summary_retry[name] = {path: fields[path] for path in missed}
        self._refresh_distributions()
        return notes

    def _retry_summary(self) -> None:
        """The shown tab's settings that had to wait (see `_restore_summary`)."""
        shown = self.right_tabs.currentWidget()
        notes = []
        for name, view in self.summary_views().items():
            if name not in self._summary_retry or not (view is shown or (shown is self.dist_tab and view in self.dists)):
                continue
            fields = self._summary_retry.pop(name)
            notes += [f"{name}: {path}" for path in view_state.restore(view, fields, view.STATE_FIELDS)]
        self._report_view(notes, quiet=True)

    def _report_view(self, notes: list[str], quiet: bool = False) -> None:
        if notes:
            self.statusBar().showMessage("reopened, except: " + "; ".join(notes), 15_000)
        elif not quiet:
            self.statusBar().showMessage("view reopened", 4000)

    def _save_session(self) -> None:
        if self.resume == "never" or self.run is None or self.catalog is None:
            return
        self.settings.setValue(f"session/{self.run.tag}", json.dumps(self.view_state()))

    def _ask_resume(self, tag: str) -> bool:
        """At start-up the main window isn't shown yet, so the question is a window of its own (with a
        taskbar entry), brought to the front with the focus; later it sits on the main window.
        """
        yes, no = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
        box = QMessageBox(QMessageBox.Icon.Question, f"ttu-view — {tag}", f"Resume {tag} where you left off?",
                          yes | no, self if self.isVisible() else None)
        box.setInformativeText("No opens it on the default panels.")
        box.setDefaultButton(yes)
        box.setWindowModality(Qt.WindowModality.ApplicationModal)
        if not self.isVisible():
            box.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        QTimer.singleShot(0, lambda: (box.raise_(), box.activateWindow()))
        box.exec()
        return box.standardButton(box.clickedButton()) == yes

    def _offer_resume(self, run) -> None:
        if self.resume != "ask":
            return
        raw = self.settings.value(f"session/{run.tag}")
        if not raw or not self._ask_resume(run.tag):
            return
        try:
            state = json.loads(raw)
        except ValueError:
            return
        if self.catalog is None:
            self._pending_view = state
        else:  # the quantities came while the dialog was open
            self._report_view(self.apply_view_state(state))

    def open_view(self, view: SavedView) -> None:
        if self.catalog is None:
            self._pending_view = view.state
            return
        notes = self.apply_view_state(view.state)
        if view.run != self.run.tag:
            notes = [f"saved from {view.run}"] + notes if notes else notes
        self._report_view(notes)

    def _fill_views_menu(self) -> None:
        menu = self.views_menu
        menu.clear()
        save = menu.addAction("Save this view…")
        save.setEnabled(self.run is not None and self.catalog is not None)
        save.triggered.connect(self._save_view_dialog)
        run = self.run.tag if self.run is not None else None
        views = self.views.ordered(run)
        if views:
            menu.addSeparator()
        for v in views:
            label = v.name if v.run == run else f"{v.name}  (from {v.run})"
            act = menu.addAction(label)
            act.setToolTip(v.note or f"saved {v.created[:16].replace('T', ' ')} UTC from {v.run}")
            act.triggered.connect(lambda _=False, v=v: self.open_view(v))
        if views:
            delete = menu.addMenu("Delete")
            for v in views:
                act = delete.addAction(v.name if v.run == run else f"{v.name}  (from {v.run})")
                act.triggered.connect(lambda _=False, v=v: self._delete_view(v))
        menu.setToolTipsVisible(True)
        if self.views.path is None:
            menu.addSeparator()
            menu.addAction("(no ttu-tower home: views last only this session)").setEnabled(False)

    def _save_view_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Save this view", "Name:")
        if not ok or not name.strip():
            return
        note, ok = QInputDialog.getText(self, "Save this view", "Note (optional):")
        try:
            self.views.add(SavedView(name.strip(), self.run.tag, self.view_state(), note.strip() if ok else ""))
        except OSError as exc:
            QMessageBox.warning(self, "ttu-view", f"couldn't save the view: {exc}")
            return
        self.statusBar().showMessage(f"saved the view '{name.strip()}'", 6000)

    def _delete_view(self, view: SavedView) -> None:
        answer = QMessageBox.question(self, "ttu-view", f"Delete the view '{view.name}' (from {view.run})?")
        if answer == QMessageBox.StandardButton.Yes:
            self.views.remove(view.id)

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
        if i in self._pending_legend:
            view_state.put(self.plots[i].legend, self._pending_legend.pop(i))
        self._load_other(i, spec)
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
        self.jets.band.set_axis_width(width)
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

    def step(self, fraction: float) -> None:
        """Shift the view by `fraction` of its width (negative: back in time)."""
        x0, x1 = self.x_range()
        shift = fraction * (x1 - x0)
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
            self.jets.set_timezone(tz)

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
        if shown is not self.scatter and self.scatter.shows_3d():  # its 3D window follows with the tab hidden
            self.scatter.set_sources(self.datas, x_range, self.stability)
        if shown is not self.windrose and self.windrose.shows_3d():
            self.windrose.set_range(x_range)
        if shown is self.scatter:
            self.scatter.set_sources(self.datas, x_range, self.stability)
        elif shown is self.windrose:
            self.windrose.set_range(x_range)
        elif shown is self.hodograph:
            self.hodograph.set_range(x_range, self.stability)
        elif shown is self.qc_summary:
            self.qc_summary.set_range(x_range)
        elif shown is self.profiles:
            self.profiles.set_sources([(data, plot.visible_members()) if data is not None else None
                                       for data, plot in zip(self.datas, self.plots)], x_range, self.stability)
        elif shown is self.composite:
            self.composite.set_range(x_range, self.stability)
        elif shown is self.tau_view:
            self.tau_view.set_range(x_range, self.stability)
        elif shown is self.scales_view:
            self.scales_view.set_range(x_range, self.stability)
        elif shown is self.diurnal:
            self.diurnal.set_sources(self.datas, x_range, self.tz_box.currentData() or self.run.timezone)
        elif shown is self.aniso_map:
            self.aniso_map.set_range(x_range, self.stability)
        elif shown is self.jets.view:
            self.jets.set_x_range(x_range)
        elif shown is self.compare:
            self.compare.set_sources(self.datas, self.others, self.other_run.tag if self.other_run else None,
                                     x_range)
        else:
            for data, plot, dist in zip(self.datas, self.plots, self.dists):
                if data is None:
                    dist.set_data(None, [], None)
                else:
                    dist.set_data(data, plot.visible_members(), x_range, self.stability)
        if self._summary_retry:
            self._retry_summary()
        self.summary_hover.watch(self.right_tabs)  # plots a view made as it redrew

    def _load_scatter_z(self, q, variant, on_done) -> None:
        """Scatter's z quantity, every member the run has."""
        run = self.run
        members = tuple(run.booms) if q.kind == "boom" else q.pairs if q.kind == "pair" else (None,)
        self.runner.submit(timeline.load, run, self.index, q, variant, members, key="scatter-z",
                           label=f"loading {q.label}", on_error=self._report_error,
                           on_done=lambda data: run is self.run and on_done(data))

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

    def _on_summary_hover(self, text: str) -> None:
        self.hover.setText(f"{self.right_tabs.tabText(self.right_tabs.currentIndex())}:   {text}")

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

    def inspect(self, boom: int, slot: int, q=None, variant: str = "mrd", tab: str | None = None,
                sequence=None, label: str = "") -> "InspectorWindow":
        """Open (or reuse an unpinned) Slot Inspector on the tab that explains `q`
        (or on `tab`); `sequence`: (slot, boom) pairs its match buttons step through.
        """
        self._opening_feedback(boom, slot)
        window = next((w for w in self.inspectors if not w.pin.isChecked()), None)
        if window is None:
            reprocessor = self.reprocessor if self.run.can_reprocess else None
            window = InspectorWindow(self.run, self.index, reprocessor, self.runner)
            window.closed.connect(lambda w: self.inspectors.remove(w) if w in self.inspectors else None)
            window.closed.connect(lambda w: self._opened(w, False))
            window.loaded.connect(lambda ok, w=window: self._opened(w, ok))
            window.bookmarkRequested.connect(self._add_bookmark)
            window.bookmarks_at = lambda slot, boom: self.bookmarks.at(self.run.tag, slot, boom)
            self.inspectors.append(window)
        window.set_sequence(sequence, label)
        self._opening = window
        explains, frame, emphasis, focus, show = "series", None, None, (), ()
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
            show = target.variables  # a clicked T mean shows T even though it starts unchecked
        else:
            context, variables = "", list(_HF_VARIABLES)
        window.show_slot(boom, slot, tab=tab or explains, mode="unexcised", variables=variables, context=context,
                         frame=frame, emphasis=emphasis, focus_vars=focus, variant=variant, show=show)
        return window

    def inspect_sequence(self, items, start: int, label: str, variant: str = "mrd", tab: str | None = None) -> None:
        slot, boom = items[start]
        self.inspect(boom, slot, variant=variant, tab=tab, sequence=items, label=label)

    # --- linked brushing ------------------------------------------------------------------------

    def set_selection(self, slots, mode: str = "replace") -> None:
        """Combine brushed slots into the shared selection and show it everywhere."""
        self.selection = brush.combine(self.selection, slots, mode)
        for p in self.plots:
            p.set_selection(self.selection)
        self.overview.set_selection(self.selection)
        for d in self.dists:
            d.set_selection(self.selection)
        self.scatter.set_selection(self.selection)
        self.profiles.set_selection(self.selection)
        self.composite.set_selection(self.selection)
        self.tau_view.set_selection(self.selection)
        self.scales_view.set_selection(self.selection)
        self.hodograph.set_selection(self.selection)
        self.aniso_map.set_selection(self.selection)
        self._show_selection()

    def _show_selection(self) -> None:
        has = self.selection.size > 0
        self.selection_label.setText(f"selection: {brush.describe(self.selection)}" if has else "")
        self.selection_inspect.setVisible(has)
        self.selection_clear.setVisible(has)

    def _set_brush_mode(self, on: bool) -> None:
        for p in self.plots:
            p.vb.brush_always = on
        self.scatter.vb.brush_always = on
        self.statusBar().showMessage("brush mode: drag to select; Ctrl adds, Alt removes, Esc clears" if on
                                     else "brush mode off: Shift-drag still selects", 6000)

    def _inspect_selection(self) -> None:
        if self.selection.size == 0:
            return
        boom = self.inspectors[-1].boom if self.inspectors else self.goto_boom.currentData()
        self.inspect_sequence([(int(k), boom) for k in self.selection], 0, f"selection, b{boom}")

    # --- another run ---------------------------------------------------------------------------

    def _fill_compare_box(self) -> None:
        self.compare_box.blockSignals(True)
        self.compare_box.clear()
        self.compare_box.addItem("(none)", None)
        for row in list_registered():
            if row["exists"] and "primary" in row["stages"] and row["tag"] != self.run.tag:
                self.compare_box.addItem(row["tag"], row["tag"])
        self.compare_box.blockSignals(False)

    def _on_compare_chosen(self, index: int) -> None:
        tag = self.compare_box.itemData(index)
        if tag is None:
            self._set_other_run(None)
            return
        try:
            self._set_other_run(RunHandle.open(tag))
        except Exception as exc:
            QMessageBox.warning(self, "ttu-view", f"can't open {tag}: {exc}")
            self.compare_box.setCurrentIndex(0)

    def _set_other_run(self, run) -> None:
        self.other_run = run
        self.other_index = FragmentIndex(run.run_dir) if run is not None else None
        self.other_catalog = None
        self.others = [None, None]
        for p in self.plots:
            p.set_compare(None, None)
        if run is None:
            self._refresh_distributions()
            return
        self.statusBar().showMessage(f"comparing with {run.tag}: reading its quantities…", 6000)
        self.runner.submit(build_catalog, run, key="other-catalog", label=f"reading {run.tag}'s quantities",
                           on_done=lambda catalog: self._other_catalog_ready(run, catalog),
                           on_error=self._report_error)

    def _derived_changed(self) -> None:
        """A definition saved or deleted: the panels' quantities again, each panel kept as it was."""
        if self.catalog is None:
            return
        states = [p.state() for p in self.panels]  # before the catalog changes under them
        notes = derived.add_to(self.catalog, self.derived_store.items)
        self.derived_panel.set_catalog(self.catalog, notes)
        if self.other_catalog is not None:
            derived.add_to(self.other_catalog, self.derived_store.items)
        for panel, state in zip(self.panels, states):
            panel.set_catalog(self.catalog, self.run.booms)
            why = panel.restore(state) if state is not None else None
            if why:  # its derived quantity was deleted
                self.statusBar().showMessage(why, 8000)
        self.scatter.set_catalog(self.catalog)

    def _other_catalog_ready(self, run, catalog) -> None:
        if run is not self.other_run:
            return
        derived.add_to(catalog, self.derived_store.items)
        self.other_catalog = catalog
        for i, spec in enumerate(self.specs):
            if spec.quantity is not None and self.datas[i] is not None:
                self._load_other(i, spec)

    def _load_other(self, i: int, spec: PanelSpec) -> None:
        """The compared run's version of panel i, dashed under this run's."""
        run = self.other_run
        if run is None or spec.quantity is None:
            return
        if self.other_catalog is None:
            self.others[i] = "reading the other run's quantities…"
            return
        other, variant = compare.counterpart(self.other_catalog, spec.quantity, spec.variant)
        if other is None:
            self.others[i] = variant  # the reason
            self._refresh_distributions()
            return
        members = tuple(m for m in spec.members if spec.quantity.kind != "boom" or m in run.booms)

        def ready(data):
            if run is self.other_run and spec is self.specs[i]:
                self.others[i] = data
                self.plots[i].set_compare(data, run.tag)
                self._refresh_distributions()

        self.runner.submit(timeline.load, run, self.other_index, other, variant, members, key=f"other{i}",
                           label=f"loading {run.tag}'s {other.label}", on_done=ready, on_error=self._report_error)

    # --- figures -------------------------------------------------------------------------------

    def timelines_spec(self) -> tuple[str, dict] | None:
        tz = self.tz_box.currentData() or self.run.timezone
        x0, x1 = self.x_range()

        def local(x):
            return pd.to_datetime(x, unit="s", utc=True).tz_convert(tz).tz_localize(None)

        def broken(x, y):
            """Local times and values with a NaN wherever slots are missing, so no line bridges a gap."""
            gaps = np.flatnonzero(np.diff(x) > SLOT_SECONDS * 1.5)
            x, y = np.insert(x, gaps + 1, x[gaps] + SLOT_SECONDS), np.insert(y.astype(float), gaps + 1, np.nan)
            return local(x), y

        panels = []
        for i, (plot, data) in enumerate(zip(self.plots, self.datas)):
            if data is None:
                continue
            q = data.quantity
            curves, others = [], []
            for j, m in enumerate(plot.visible_members()):
                c = data.curves[m]
                keep = (c.x >= x0) & (c.x <= x1)
                color = pg.mkColor(style.member_color(m, j)).name()
                t, y = broken(c.x[keep], c.y[keep])
                curves.append({"label": style.member_label(m) or "value", "color": color, "t": t, "y": y,
                               "points": plot.lines_drawn is False})
                other = self.others[i]
                if not isinstance(other, str) and other is not None and m in other.curves:
                    o = other.curves[m]
                    ok = (o.x >= x0) & (o.x <= x1)
                    t, y = broken(o.x[ok], o.y[ok])
                    others.append({"label": m, "color": color, "t": t, "y": y})
            variant = "" if data.variant == "none" else f" · {data.variant}"
            panels.append({"title": q.title + variant, "ylabel": q.unit, "log_y": plot.log_y,
                           "categories": data.categories if q.categorical else None, "curves": curves,
                           "others": others})
        if not panels:
            return None
        night = []
        if self.night is not None and self.night_box.isChecked():
            for a, b in brush.runs(self.night.slots[np.nan_to_num(self.night.y) > 0.5]):
                t0, t1 = slot_to_unix(a), slot_to_unix(b)
                if t1 >= x0 and t0 <= x1:
                    night.append((local(np.array([t0]))[0], local(np.array([t1]))[0]))
        xlim = tuple(local(np.array([x0, x1])))
        return "timelines", {"panels": panels, "night": night, "xlim": xlim, "xlabel": f"time ({tz})"}

    def summary_spec(self) -> tuple[str, dict] | None:
        shown = self.right_tabs.currentWidget()
        if shown is self.dist_tab:
            panels = [p for p in (d.figure_spec() for d in self.dists) if p is not None]
            return ("distributions", {"panels": panels}) if panels else None
        return shown.figure_spec() if hasattr(shown, "figure_spec") else None

    def _export_figure(self) -> None:
        if self.run is None:
            return
        tab = self.right_tabs.tabText(self.right_tabs.currentIndex())
        dialog = ExportDialog(tab, self.summary_spec() is not None, self.settings, self)
        if not dialog.exec():
            return
        spec = self.timelines_spec() if dialog.target() == "timelines" else self.summary_spec()
        if spec is None:
            QMessageBox.information(self, "ttu-view", "nothing to draw there yet")
            return
        try:
            fig = figures.render(*spec, dialog.size())
            figures.save(fig, dialog.path(), dialog.dpi())
        except Exception as exc:
            QMessageBox.warning(self, "ttu-view", f"couldn't save the figure: {exc}")
            logging.getLogger("ttu_view").exception("figure export")
            return
        self.statusBar().showMessage(f"saved {dialog.path()}", 8000)

    # --- find and bookmarks ---------------------------------------------------------------------

    def _find_context(self) -> dict | None:
        if self.run is None:
            return None
        x0, x1 = self.x_range()
        return {"run": self.run, "index": self.index, "catalog": self.catalog, "stability": self.stability,
                "tz": self.tz_box.currentData() or self.run.timezone, "booms": self.run.booms,
                "slot_range": (int(unix_to_slot(x0)), int(unix_to_slot(x1)) + 1)}

    def _add_bookmark(self, slot: int, boom: int, variant: str) -> None:
        dialog = BookmarkDialog("Bookmark this slot", self.bookmarks.lists(self.run.tag), parent=self.sender())
        if dialog.exec():
            note, name = dialog.values()
            self._store_bookmarks([Bookmark(self.run.tag, slot, boom, variant, note, name)])

    def _save_found(self, items, query: str) -> None:
        dialog = BookmarkDialog(f"Save {len(items)} matches as a list", self.bookmarks.lists(self.run.tag),
                                list_name="find", note=query, parent=self)
        if dialog.exec():
            note, name = dialog.values()
            variant = self.find.variant.currentData()
            self._store_bookmarks([Bookmark(self.run.tag, k, b, variant, note, name) for k, b in items])

    def _bookmarks_changed(self) -> None:
        for window in self.inspectors:
            window.show_bookmarked()

    def _store_bookmarks(self, marks) -> None:
        try:
            self.bookmarks.add(*marks)
        except OSError as exc:
            QMessageBox.warning(self, "ttu-view", f"couldn't save the bookmarks: {exc}")
            return
        self.bookmark_panel.refresh()
        self._bookmarks_changed()
        plural = "s" if len(marks) != 1 else ""
        self.statusBar().showMessage(f"saved {len(marks)} bookmark{plural} to '{marks[0].list}'", 6000)

    def _opening_feedback(self, boom: int, slot: int) -> None:
        """A busy cursor and a status line from the click until the inspector has the slot."""
        if self._opening is None:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._opening = self  # a placeholder until the window exists
        self._opening_timer.start()
        when = format_slot(slot, self.tz_box.currentData() or self.run.timezone)
        self.statusBar().showMessage(f"opening the Slot Inspector: boom {boom}, {when}…")
        self.statusBar().repaint()

    def _on_also_here(self, text: str) -> None:
        self._also_here = text
        self.statusBar().showMessage(text, 8000)

    def _opened(self, window, ok: bool) -> None:
        if self._opening is None or window is not self._opening:
            return
        self._opening = None
        self._opening_timer.stop()
        QApplication.restoreOverrideCursor()
        also, self._also_here = self._also_here, ""
        if ok and also:
            self.statusBar().showMessage(also, 8000)
        elif ok:
            self.statusBar().clearMessage()
        else:
            self.statusBar().showMessage("the Slot Inspector couldn't read that slot (see its summary line)", 8000)

    def _on_busy(self, count: int, label: str) -> None:
        self.busy.setText(f"⏳ {label}…" if count else "")

    def closeEvent(self, ev) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self._save_session()
        self._opened(self._opening, True)
        for w in list(self.inspectors):
            w.close()
        for view in (self.scatter, self.windrose):
            if view.window3d is not None:
                view.window3d.close()
        self.runner.wait(5000)
        self.runner.shutdown()
        if self.reprocessor is not None:
            self.reprocessor.close()
        super().closeEvent(ev)


def _overlays(run, index, booms):
    """Runs on a worker thread."""
    slots, image = timeline.availability(run, index, booms)
    return slots, image, timeline.stability(run, index), timeline.night(run, index)
