"""The Slot Inspector: one slot of one boom in depth - its 50-Hz series, its
MRD spectra and tau detection, its integral scales, and every stored number -
with the slot, boom and variant switchable in place.
"""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.post.classify import stability_classes
from ttu_tower.viewer.slotdata import load_slot, stored_selection
from ttu_tower.viewer.timeaxis import format_slot
from ttu_tower.viewer.ui.inspector.numbers_tab import NumbersTab
from ttu_tower.viewer.ui.inspector.scales_tab import ScalesTab
from ttu_tower.viewer.ui.inspector.spectra_tab import SpectraTab
from ttu_tower.viewer.ui.series_view import SeriesPanel

TABS = ("series", "spectra", "scales", "numbers")
_VARIANTS = (("mrd", "mrd (selected τ)"), ("naive", "naive (10 min)"), ("mrd_unexcised", "mrd unexcised"))


def _num(df, **eq):
    for col, value in eq.items():
        df = df[df[col] == value]
    return float(df["value"].iloc[0]) if not df.empty else np.nan


class InspectorWindow(QWidget):
    closed = Signal(object)

    def __init__(self, run, index, reprocessor, runner, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.run, self.index, self.reprocessor, self.runner = run, index, reprocessor, runner
        self.boom = self.slot = None
        self.bundle = None
        self.emphasis: str | None = None
        self.focus_vars: tuple[str, ...] = ()
        self._auto_acf = False
        self._closed = False

        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.boom_box = QComboBox()
        for b in run.booms:
            self.boom_box.addItem(f"boom {b} ({HEIGHTS[b]:g} m)", b)
        self.boom_box.currentIndexChanged.connect(lambda *_: self._on_boom())
        self.variant_box = QComboBox()
        for key, label in _VARIANTS:
            self.variant_box.addItem(label, key)
        self.variant_box.setToolTip("the variant the Spectra and Scales tabs and the summary describe")
        self.variant_box.currentIndexChanged.connect(lambda *_: self._refresh_tabs())
        self.pin = QCheckBox("pin")
        self.pin.setToolTip("keep this window; new clicks open a new one")
        prev_btn, next_btn = QPushButton("◀ slot"), QPushButton("slot ▶")
        prev_btn.clicked.connect(lambda: self.step(-1))
        next_btn.clicked.connect(lambda: self.step(1))
        self.nav_buttons = {"prev": prev_btn, "next": next_btn}
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("background:#f4f6fa; padding:4px;")

        self.series = SeriesPanel(reprocessor, runner, run, index)
        self.spectra = SpectraTab()
        self.scales = ScalesTab()
        self.numbers = NumbersTab()
        self.tabs = QTabWidget()
        for widget, label in ((self.series, "Series"), (self.spectra, "Spectra"), (self.scales, "Scales"),
                              (self.numbers, "Numbers")):
            self.tabs.addTab(widget, label)
        if reprocessor is None:
            self.tabs.setTabToolTip(0, "reprocessing is disabled for this run")

        header = QHBoxLayout()
        header.addWidget(self.title, stretch=1)
        for w in (prev_btn, next_btn, self.boom_box, self.variant_box, self.pin):
            header.addWidget(w)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.summary)
        layout.addWidget(self.tabs, stretch=1)
        self.resize(1400, 950)

    # --- navigation -----------------------------------------------------------------------------

    def show_slot(self, boom: int, slot: int, tab: str = "series", mode: str | None = None, variables=None,
                  context: str = "", frame: str | None = None, emphasis: str | None = None, focus_vars=(),
                  variant: str | None = None) -> None:
        self.boom, self.slot = boom, slot
        self.emphasis, self.focus_vars, self._auto_acf = emphasis, tuple(focus_vars), tab == "scales"
        self._set_combo(self.boom_box, boom)
        if variant is not None:
            self._set_combo(self.variant_box, variant if variant != "none" else "mrd")
        if self.reprocessor is not None:
            self.series.set_focus(boom, slot, mode=mode, variables=variables, context=context, frame=frame)
        self.tabs.setCurrentIndex(TABS.index(tab))
        self._load()
        self.bring_to_front()

    def bring_to_front(self) -> None:
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def step(self, delta: int) -> None:
        if self.slot is None:
            return
        self.slot += delta
        if self.reprocessor is not None:
            self.series.set_focus(self.boom, self.slot)
        self._load()

    def _on_boom(self) -> None:
        boom = self.boom_box.currentData()
        if self.slot is None or boom == self.boom:
            return
        self.boom = boom
        if self.reprocessor is not None:
            self.series.set_focus(boom, self.slot)
        self._load()

    @staticmethod
    def _set_combo(combo, value) -> None:
        i = combo.findData(value)
        if i >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(i)
            combo.blockSignals(False)

    def variant(self) -> str:
        return self.variant_box.currentData()

    # --- stored data ------------------------------------------------------------------------------

    def _load(self) -> None:
        self.title.setText(f"<b>Boom {self.boom}</b> ({HEIGHTS[self.boom]:g} m) · slot "
                           f"{format_slot(self.slot, self.run.timezone)} (k = {self.slot})")
        self.summary.setText("loading…")
        self.runner.submit(load_slot, self.index, self.slot, self.boom, self.run.booms, key=f"inspect-{id(self)}",
                           label="reading the slot", on_done=self._loaded,
                           on_error=lambda exc, tb: self.summary.setText(f"could not read the slot: {exc}"))

    def _loaded(self, bundle) -> None:
        if self._closed or (bundle.slot, bundle.boom) != (self.slot, self.boom):
            return
        self.bundle = bundle
        self._refresh_tabs()

    def _refresh_tabs(self) -> None:
        b = self.bundle
        if b is None:
            return
        variant = self.variant()
        self.summary.setText(self._summary_text(b, variant))
        sel = stored_selection(b, variant)
        self.series.set_selected_tau(sel["tau_s"] if sel else None)
        cfg = self.run.cfg
        if cfg is None:
            self.spectra.info.setText("the run's config is unavailable, so detection can't be rerun")
        else:
            self.spectra.set_bundle(b, variant, cfg, self.emphasis)
            self.scales.set_bundle(b, variant, cfg, self.reprocessor, self.runner, self.focus_vars,
                                   auto_acf=self._auto_acf)
            self._auto_acf = False
        self.numbers.set_bundle(b, self.reprocessor, self.runner)

    def _summary_text(self, b, variant: str) -> str:
        parts = []
        status = b.table("slot_boom")
        if not status.empty:
            s = str(status["status"].iloc[0])
            parts.append(f"status <b>{s}</b>" + (" (unexcised computed)" if bool(status["unexcised_computed"].iloc[0])
                                                  else ""))
        sel = stored_selection(b, variant)
        if sel is not None:
            parts.append(f"τ <b>{sel['tau_s']:g} s</b> from {sel['source']} ({sel['source_status']})"
                         if np.isfinite(sel["tau_s"]) else f"τ none ({sel['source_status']})")
        tau = b.table("tau")
        spec_variant = "mrd" if variant == "naive" else variant
        for row in tau[tau["variant"] == spec_variant].itertuples():
            value = row.tau_s if np.isfinite(row.tau_s) else row.tau_lb_s
            parts.append(f"{row.cospectrum} {row.status}" + (f" {value:g} s" if np.isfinite(value) else ""))
        cov = b.table("coverage")
        for fam in ("momentum", "heat"):
            v = _num(cov.rename(columns={"fraction": "value"}), variable=fam, layer="usable")
            if np.isfinite(v):
                parts.append(f"{fam} coverage {v:.2f}")
        means = b.table("means")
        ws, wd = _num(means, variable="ws", stat="mean"), _num(means, variable="wd", stat="mean")
        if np.isfinite(ws):
            parts.append(f"ws {ws:.2f} m/s" + (f" from {wd:.0f}°" if np.isfinite(wd) else ""))
        stab = self._stability(b)
        if stab:
            parts.append(stab)
        flt = b.table("filter_log")
        failed = flt[flt["variant"].isin([variant, "none"])] if not flt.empty else flt
        if not failed.empty:
            groups = sorted({f"{g}: {c}" for g, c in zip(failed["group"], failed["criterion"])})
            parts.append("<span style='color:#b00'>filtered " + ", ".join(groups) + "</span>")
        return " · ".join(parts)

    def _stability(self, b) -> str:
        cfg = self.run.cfg
        pairs = b.table("pairs")
        if cfg is None or pairs.empty:
            return ""
        boom, boom2 = cfg.post.stability.pair
        classes = stability_classes(pairs, cfg.post)
        rib = pairs[(pairs["variable"] == "rib") & (pairs["boom"] == boom) & (pairs["boom2"] == boom2)]
        if classes.empty or rib.empty:
            return ""
        label = classes.iloc[0]
        value = rib["value"].iloc[0]
        return f"Ri_b({boom}–{boom2}) {value:.3g}" + (f" → {label}" if isinstance(label, str) else "")

    def closeEvent(self, ev) -> None:
        self._closed = True
        self.series.shutdown()
        self.closed.emit(self)
        super().closeEvent(ev)
