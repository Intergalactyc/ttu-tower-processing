"""Ties the Jets controls, the Jets tab and the jets band to the main window:
loads a run's wind profiles and the filters' quantities in the background,
reruns the detection (milliseconds, so on the GUI thread) whenever a setting
or a manual edit changes, keeps the run's manual edits, and turns picks into
timeline ranges, selections and inspectors.
"""
import json
import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QMessageBox

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.jets import data
from ttu_tower.jets.analysis import JetResult, JetSettings, analyse, sweep, sweep_values
from ttu_tower.jets.edits import EditStore, new_edit
from ttu_tower.jets.filters import context_needs
from ttu_tower.viewer import hodograph
from ttu_tower.viewer.timeaxis import slot_to_unix, unix_to_slot
from ttu_tower.viewer.ui import style
from ttu_tower.viewer.ui.jets_panel import JetsPanel
from ttu_tower.viewer.ui.jets_edits_dialog import EditsDialog
from ttu_tower.viewer.ui.jets_event import PROFILE_OPTIONS, event_label
from ttu_tower.viewer.ui.jets_stats import sweep_field
from ttu_tower.viewer.ui.jets_view import JetsView
from ttu_tower.viewer.ui.overview import JetsBand

SETTINGS_KEY = "jets/settings"
_PAD_S = 3 * 3600.0  # at least this much either side of an event shown on the timelines


def nearest_boom(height: float) -> int:
    return min(HEIGHTS, key=lambda b: abs(HEIGHTS[b] - height))


class JetsController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.panel, self.view, self.band = JetsPanel(), JetsView(), JetsBand()
        self.run = None
        self.profiles = None
        self.contexts: dict = {}  # needs -> per-slot frame (None while loading)
        self.result: JetResult | None = None
        self.tz = "UTC"
        self.store: EditStore | None = None  # the run's manual edits
        self.dialog: EditsDialog | None = None
        self._saved = {}
        raw = window.settings.value(SETTINGS_KEY)
        if raw:
            try:
                self._saved = json.loads(raw)
                self.panel.set_settings(JetSettings.from_dict(self._saved))
            except (ValueError, TypeError):
                self._saved = {}
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.recompute)
        self.panel.changed.connect(self._on_changed)
        self.panel.selectRequested.connect(self.select_slots)
        self.panel.stepRequested.connect(lambda: self.step_through(self.view.events.selected_event()))
        self.view.eventSelected.connect(self.show_event)
        self.view.eventActivated.connect(self.step_through)
        self.view.eventStep.connect(self.step_event)
        self.view.inspectRequested.connect(lambda slot, boom: self.window.inspect(boom, slot, tab="profile"))
        self.view.valuesWanted.connect(self._load_values)
        self.view.sweepRequested.connect(self._sweep)
        self.band.eventClicked.connect(self._on_band_click)
        self.band.eventDoubleClicked.connect(self.step_through)
        self.view.spanEdited.connect(self.edit_span)
        self.view.rejectToggled.connect(self.toggle_reject)
        self.view.revertRequested.connect(self.revert)
        self.view.addRequested.connect(self.add_from_selection)
        self.view.editsRequested.connect(self.show_edits)
        self.view.selectRequested.connect(self.select_event_slots)

    # --- inputs ---------------------------------------------------------------------------------

    def set_run(self, run) -> None:
        self.run, self.profiles, self.contexts, self.result = run, None, {}, None
        try:
            self.store = EditStore(run.tag)
        except (OSError, ValueError, TypeError) as exc:  # kept in memory, never written over the unreadable file
            self.store = EditStore(run.tag, load=False)
            self.store.path = None
            self.window.statusBar().showMessage(f"couldn't read this run's manual jet edits: {exc}", 15_000)
        self.view.single.set_values("", None)
        self._show(None, "loading the wind profiles…")
        self.window.runner.submit(data.load_profiles, run.run_dir, key="jets-profiles",
                                  label="loading wind profiles for the jets",
                                  on_done=lambda p: self._profiles_ready(run, p), on_error=self._failed)
        self._ensure_context(self.panel.settings())
        self.view.aggregate.set_mesonet(None)
        self.window.runner.submit(hodograph.mesonet_uv, run, self.window.index, key="jets-mesonet",
                                  label="loading the mesonet wind", on_error=self._failed,
                                  on_done=lambda uv: run is self.run and self.view.aggregate.set_mesonet(uv))

    def set_timezone(self, tz: str) -> None:
        self.tz = tz
        self.timer.start()

    def set_x_range(self, x_range) -> None:
        """The interval the Aggregate and Diagnostics sub-tabs cover (None: the whole period)."""
        slot_range = None if x_range is None else (int(unix_to_slot(x_range[0])), int(unix_to_slot(x_range[1])) + 1)
        self.view.set_range(slot_range)

    def _load_values(self, key: str) -> None:
        """One of the Event sub-tab's profile quantities, every slot and boom."""
        if self.run is None:
            return
        run = self.run
        variable, stat, variant = PROFILE_OPTIONS[key][2]
        self.window.runner.submit(data.load_boom_values, run.run_dir, variable, stat, variant, tuple(BOOMS),
                                  key=f"jets-values-{key}", label=f"loading {PROFILE_OPTIONS[key][0]} profiles",
                                  on_error=self._failed,
                                  on_done=lambda frame: run is self.run and self.view.single.set_values(key, frame))

    def _sweep(self, part: str, name: str) -> None:
        result, profiles = self.result, self.profiles
        if result is None or profiles is None:
            return
        settings = result.settings
        current = getattr(settings.algorithm_params() if part == "params" else settings.events, name)
        values = sweep_values(sweep_field(settings, part, name), current)
        context = self.contexts.get(context_needs(settings.filters))
        slot_range = self.view.slot_range
        self.window.runner.submit(
            sweep, profiles, settings, part, name, values, context, self.tz, slot_range, key="jets-sweep",
            label=f"sweeping {name}", on_error=self._failed,
            on_done=lambda frame: (self.result is not None and self.result.settings == settings
                                   and self.view.diagnostics.set_sweep(part, name, frame)))

    def _profiles_ready(self, run, profiles) -> None:
        if run is self.run:
            self.profiles = profiles
            self.recompute()

    def _ensure_context(self, settings: JetSettings) -> tuple:
        needs = context_needs(settings.filters)
        if needs not in self.contexts and self.run is not None:
            self.contexts[needs] = None
            run = self.run
            self.window.runner.submit(data.load_context, run.run_dir, *needs, key=f"jets-context-{needs}",
                                      label="loading the jets' filter quantities",
                                      on_done=lambda ctx: self._context_ready(run, needs, ctx),
                                      on_error=self._failed)
        return needs

    def _context_ready(self, run, needs, context) -> None:
        if run is self.run:
            self.contexts[needs] = context
            self.recompute()

    def _failed(self, exc, tb) -> None:
        logging.getLogger("ttu_view").error(tb)
        self._show(None, f"couldn't read the jets' inputs: {exc}")

    def _on_changed(self) -> None:
        settings = self.panel.settings()
        errors = settings.errors()
        self.panel.show_errors(errors)
        if not errors:
            self._saved = settings.to_dict(self._saved)
            self.window.settings.setValue(SETTINGS_KEY, json.dumps(self._saved))
            self.timer.start()

    # --- the detection --------------------------------------------------------------------------

    def recompute(self) -> None:
        settings = self.panel.settings()
        errors = settings.errors()
        self.panel.show_errors(errors)
        if errors or self.run is None:
            return
        if self.profiles is None:
            self._show(None, "loading the wind profiles…")
            return
        if self.profiles.slots.size == 0:
            self._show(None, "this run has no tertiary wind speeds to look for jets in")
            return
        context = self.contexts.get(self._ensure_context(settings))
        try:
            result = analyse(self.profiles, settings, context, self.tz,
                             edits=self.store.edits if self.store is not None else ())
        except ValueError as exc:  # e.g. booms this run lacks
            self._show(None, str(exc))
            return
        self._show(result, "" if context is not None else "loading the filter quantities…")

    def _show(self, result: JetResult | None, message: str) -> None:
        self.result = result
        if result is None:
            self.panel.show_message(message)
            self.view.set_result(None, message)
            self.band.set_events(np.empty(0, np.int64), np.empty(0, np.int64), {}, {}, f"jets: {message}")
            return
        self.panel.show_counts(result.counts())
        if message:
            self.panel.result.setText(self.panel.result.text() + f"<br><i>{message}</i>")
        self.view.set_result(result, profiles=self.profiles, tz=self.tz)
        table = result.events.table
        colors, labels, booms = {}, {}, set()
        for event_id, e in table.iterrows():
            if e.rejected:
                colors[event_id] = JetsBand.REJECTED
            elif not result.passed.loc[event_id]:
                colors[event_id] = None
            elif np.isfinite(e.z_nose_median):
                boom = nearest_boom(e.z_nose_median)
                booms.add(boom)
                colors[event_id] = style.boom_color(boom).getRgb()
            else:
                colors[event_id] = JetsBand.NO_NOSE
            labels[event_id] = event_label(result, event_id, self.tz)
        if len(table):
            legend = "jets: " + "  ".join(f"<span style='color:{style.boom_color(b).name()}'>■</span> peak at "
                                          f"{HEIGHTS[b]:g} m" for b in sorted(booms))
            legend += ("  <span style='color:#aaa'>■</span> removed by the filters"
                       "  <span style='color:#964646'>■</span> rejected by hand"
                       "  <span style='color:#f08c00'>▔</span> edited or made by hand")
        else:
            legend = "jets: none"
        event_of = result.events.slots["event_id"].fillna(-1).to_numpy(np.int64)
        marked = set(table.index[table["source"] != "auto"])
        self.band.set_events(result.detection.index.to_numpy(np.int64), event_of, colors, labels, legend, marked)
        if self.dialog is not None and self.dialog.isVisible():
            self._fill_dialog()

    # --- manual edits ---------------------------------------------------------------------------

    def _change_edits(self, remove=(), add=(), show_slot: int | None = None) -> None:
        """Change the run's edits, rerun, and keep showing the event at `show_slot`."""
        if self.store is None:
            return
        try:
            self.store.change(remove, add)
        except OSError as exc:
            self.window.statusBar().showMessage(f"couldn't save the manual jet edits: {exc}", 15_000)
        self.recompute()
        if show_slot is not None and self.result is not None:
            ids = self.result.events.slots["event_id"]
            if show_slot in ids.index and ids.notna().loc[show_slot]:
                event_id = int(ids.loc[show_slot])
                self.view.select_event(event_id)
                self.show_event(event_id)

    def _own(self, event_id: int) -> list:
        e = self.result.events.table.loc[event_id]
        return [x for x in self.store.edits if x.id in e.edits]

    def edit_span(self, event_id: int, first: int, last: int) -> None:
        """New boundaries for an event; they replace any it had."""
        if self.result is None or self.store is None or event_id not in self.result.events.table.index:
            return
        e = self.result.events.table.loc[event_id]
        spans = [x for x in self._own(event_id) if x.kind in ("span", "add")]
        if e.source == "auto":
            edit = new_edit("span", first, last, anchor=int(e.peak_slot))
        else:
            anchored = next((x for x in spans if x.kind == "span"), None)
            edit = new_edit("span", first, last, anchor=anchored.anchor) if anchored else new_edit("add", first, last)
        self._change_edits(remove=[x.id for x in spans], add=[edit], show_slot=first)

    def toggle_reject(self, event_id: int) -> None:
        if self.result is None or self.store is None or event_id not in self.result.events.table.index:
            return
        e = self.result.events.table.loc[event_id]
        rejects = [x for x in self._own(event_id) if x.kind == "reject"]
        if rejects:
            self._change_edits(remove=[x.id for x in rejects], show_slot=int(e.start_slot))
        else:
            edit = new_edit("reject", int(e.start_slot), int(e.end_slot), anchor=int(e.peak_slot))
            self._change_edits(add=[edit], show_slot=int(e.start_slot))

    def revert(self, event_id: int) -> None:
        if self.result is None or event_id not in self.result.events.table.index:
            return
        e = self.result.events.table.loc[event_id]
        self._change_edits(remove=list(e.edits), show_slot=int(e.peak_slot))

    def add_from_selection(self) -> None:
        selection = np.asarray(self.window.selection)
        if selection.size == 0:
            QMessageBox.information(self.window, "Add a jet event",
                                    "Brush the slots first (Shift-drag on a timeline), then add the event: it "
                                    "runs from the first selected slot to the last.")
            return
        first, last = int(selection.min()), int(selection.max())
        self._change_edits(add=[new_edit("add", first, last)], show_slot=first)

    def show_edits(self) -> None:
        if self.dialog is None:
            self.dialog = EditsDialog(self.window)
            self.dialog.removeRequested.connect(lambda ids: self._change_edits(remove=ids))
            self.dialog.clearRequested.connect(
                lambda: self._change_edits(remove=[x.id for x in self.store.edits] if self.store else []))
            self.dialog.gotoRequested.connect(self._goto_event)
        self._fill_dialog()
        self.dialog.show()
        self.dialog.raise_()

    def _fill_dialog(self) -> None:
        report = self.result.edit_report if self.result is not None else None
        where = (f"run {self.run.tag}: kept in {self.store.path}" if self.store is not None and self.store.path
                 else "kept only while the viewer is open (no ttu-tower home)") if self.run is not None else ""
        self.dialog.show_report(report, self.tz, where)

    def _goto_event(self, event_id: int) -> None:
        self.view.select_event(event_id)
        self.view.tabs.setCurrentWidget(self.view.single)
        self.show_event(event_id)

    # --- picks ----------------------------------------------------------------------------------

    def _on_band_click(self, event_id: int) -> None:
        self.view.select_event(event_id)
        self.show_event(event_id)

    def show_event(self, event_id: int) -> None:
        if self.result is None or event_id not in self.result.events.table.index:
            return
        e = self.result.events.table.loc[event_id]
        t0, t1 = slot_to_unix(int(e.start_slot)), slot_to_unix(int(e.end_slot) + 1)
        pad = max(_PAD_S, t1 - t0)
        self.window.set_x_range(t0 - pad, t1 + pad)
        self.band.show_event(event_id)
        self.view.show_event(event_id)
        self.window.statusBar().showMessage(event_label(self.result, event_id, self.tz), 10_000)

    def step_event(self, delta: int) -> None:
        """To the next or previous event (only those the filters keep, if the list shows only those)."""
        if self.result is None:
            return
        ids = list(self.result.events.table.index)
        if self.view.events.kept_only.isChecked():
            ids = [i for i in ids if self.result.passed.loc[i]]
        here = self.view.event_id
        later = [i for i in ids if (here is None or (i > here if delta > 0 else i < here))]
        if later:
            target = later[0] if delta > 0 else later[-1]
            self.view.select_event(target)
            self.show_event(target)

    def select_event_slots(self, event_id: int) -> None:
        """Brush every slot of one event."""
        if self.result is not None and event_id in self.result.events.table.index:
            self.window.set_selection(self.result.event_slots([event_id], passed_only=False), "replace")

    def select_slots(self) -> None:
        if self.result is not None:
            self.window.set_selection(self.result.event_slots(), "replace")

    def step_through(self, event_id: int | None = None) -> None:
        """The Slot Inspector's Profile tab on each kept event's fastest slot,
        starting at `event_id` (or the first).
        """
        if self.result is None:
            return
        table = self.result.events.table
        ids = [i for i in table.index if self.result.passed.loc[i] or i == event_id]
        if not ids:
            return
        fallback = self.window.goto_boom.currentData()  # for an event made by hand with no nose
        items = [(int(table.loc[i, "peak_slot"]),
                  int(table.loc[i, "peak_boom"]) if pd.notna(table.loc[i, "peak_boom"]) else fallback) for i in ids]
        start = ids.index(event_id) if event_id in ids else 0
        self.band.show_event(ids[start])
        self.window.inspect_sequence(items, start, f"jet events ({len(ids)})", tab="profile")
