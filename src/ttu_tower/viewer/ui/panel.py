"""The controls for one timeline panel: which quantity, variant and booms
(or boom pairs), and how to draw it.
"""
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QStackedWidget, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer.catalog import Catalog, Quantity
from ttu_tower.viewer.timeline import has_filtered_overlay, has_tau_overlay
from ttu_tower.viewer.ui import variant_combo

_VARIANT_LABELS = variant_combo.LABELS  # the run's own; variant_combo.label covers the viewer's too
STYLES = ("lines", "points", "lines + points")


def default_pairs(q: Quantity) -> list:
    """The pairs a pair quantity starts with: veer against its reference boom
    for every other boom, Ri_b over 2-4.
    """
    if q.variable == "veer":
        refs = {b2 for _, b2 in q.pairs}
        return [p for p in q.pairs if p[0] != p[1]] if len(refs) == 1 else list(q.pairs[:1])
    return [p for p in q.pairs if p == (2, 4)] or list(q.pairs[:1])


@dataclass
class PanelSpec:
    quantity: Quantity | None = None
    variant: str = "none"
    members: tuple = ()
    log_y: bool = False
    style: str = "lines + points"
    visible: set = field(default_factory=set)  # members shown (legend toggles); empty = all
    show_filtered: bool = False  # ghost the values tertiary filtered
    show_tau: bool = False  # mark values whose selected tau wasn't simply found


class PanelControls(QWidget):
    specChanged = Signal(object)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.catalog: Catalog | None = None
        self.booms: list[int] = []
        self._quiet = False
        self._pairs_chosen: dict[str, list] = {}  # quantity key -> the pairs last checked for it

        self.search = QLineEdit()
        self.search.setPlaceholderText("search quantities…")
        self.search.textChanged.connect(self._filter_tree)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._on_quantity_changed)

        self.variant = QComboBox()
        self.variant.currentIndexChanged.connect(self._on_variant_changed)
        self._preferred_variant = "mrd"  # kept across quantities that offer it

        self.boom_boxes: dict[int, QCheckBox] = {}
        boom_page = QWidget()
        grid = QGridLayout(boom_page)
        grid.setContentsMargins(0, 0, 0, 0)
        for i, b in enumerate(sorted(HEIGHTS)):
            box = QCheckBox(f"{b}  ({HEIGHTS[b]:g} m)")
            box.toggled.connect(lambda *_: self._emit())
            self.boom_boxes[b] = box
            grid.addWidget(box, i % 5, i // 5)
        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        for label, pick in (("all", lambda b: True), ("none", lambda b: False), ("odd", lambda b: b % 2 == 1),
                            ("even", lambda b: b % 2 == 0), ("1/5/10", lambda b: b in (1, 5, 10))):
            btn = QToolButton()  # a push button's style minimum would hold the panel wide
            btn.setText(label)
            btn.setAutoRaise(True)
            btn.clicked.connect(lambda _=False, p=pick: self.set_booms([b for b in self.booms if p(b)]))
            buttons.addWidget(btn)
        buttons.addStretch(1)
        grid.addLayout(buttons, 5, 0, 1, 2)

        self.pair_list = QListWidget()
        self.pair_list.itemChanged.connect(lambda *_: self._on_pairs_changed())
        self.members = QStackedWidget()
        self.members.addWidget(boom_page)
        self.members.addWidget(self.pair_list)
        self.members.addWidget(QLabel("one value per slot"))

        self.log_y = QCheckBox("log y")
        self.log_y.toggled.connect(lambda *_: self._emit())
        self.style = QComboBox()
        self.style.addItems(STYLES)
        self.style.setCurrentText("lines + points")
        self.style.currentIndexChanged.connect(lambda *_: self._emit())
        self.filtered_box = QCheckBox("filtered values")
        self.filtered_box.setChecked(True)
        self.filtered_box.setToolTip("hollow grey: a value tertiary filtered, drawn as it was before filtering "
                                     "(hover for why; click to open the QC tab)")
        self.filtered_box.toggled.connect(lambda *_: self._emit())
        self.tau_box = QCheckBox("τ status")
        self.tau_box.setToolTip("mark the values whose selected τ was capped, unresolved, a fallback or none")
        self.tau_box.toggled.connect(lambda *_: self._emit())

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(self.search)
        layout.addWidget(self.tree, stretch=1)
        row = QHBoxLayout()
        row.addWidget(QLabel("variant"))
        row.addWidget(self.variant, stretch=1)
        layout.addLayout(row)
        layout.addWidget(self.members)
        row = QHBoxLayout()
        row.addWidget(self.log_y)
        row.addWidget(self.style, stretch=1)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.filtered_box)
        row.addWidget(self.tau_box)
        row.addStretch(1)
        layout.addLayout(row)

    # --- population -------------------------------------------------------------------

    def set_catalog(self, catalog: Catalog, booms: list[int]) -> None:
        self._quiet = True
        self.catalog, self.booms = catalog, list(booms)
        self._pairs_chosen = {}
        self.tree.clear()
        for group, quantities in catalog.grouped().items():
            parent = QTreeWidgetItem([group])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for q in quantities:
                child = QTreeWidgetItem([q.title])
                child.setData(0, Qt.ItemDataRole.UserRole, q.key)
                child.setToolTip(0, f"{q.table}: {q.variable}" + (f" / {q.stat}" if q.stat else ""))
                parent.addChild(child)
            self.tree.addTopLevelItem(parent)
        for b, box in self.boom_boxes.items():
            box.setEnabled(b in self.booms)
            box.setChecked(b in self.booms)
        self._quiet = False

    def select(self, key: str, variant: str | None = None, members=None) -> None:
        """Programmatic selection (defaults, restored state, tests)."""
        self._quiet = True
        item = self._item_for(key)
        if item is not None:
            self.tree.setCurrentItem(item)
            self._sync_quantity_widgets(self.catalog[key])
        if variant is not None:
            i = self.variant.findData(variant)
            if i >= 0:
                self.variant.setCurrentIndex(i)
        if members is not None:
            q = self.catalog[key]
            if q.kind == "boom":
                self.set_booms(members, emit=False)
            elif q.kind == "pair":
                for i in range(self.pair_list.count()):
                    it = self.pair_list.item(i)
                    pair = it.data(Qt.ItemDataRole.UserRole)
                    it.setCheckState(Qt.CheckState.Checked if pair in members else Qt.CheckState.Unchecked)
        self._quiet = False
        self._emit()

    def set_booms(self, booms, emit: bool = True) -> None:
        quiet, self._quiet = self._quiet, True
        for b, box in self.boom_boxes.items():
            box.setChecked(b in booms)
        self._quiet = quiet
        if emit:
            self._emit()

    def _item_for(self, key: str):
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                if parent.child(j).data(0, Qt.ItemDataRole.UserRole) == key:
                    return parent.child(j)
        return None

    def _filter_tree(self, text: str) -> None:
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            shown = 0
            for j in range(parent.childCount()):
                child = parent.child(j)
                match = not text or text in child.text(0).lower() or text in (child.toolTip(0) or "").lower()
                child.setHidden(not match)
                shown += match
            parent.setHidden(shown == 0)
            parent.setExpanded(bool(text) and shown > 0)

    def _sync_quantity_widgets(self, q: Quantity) -> None:
        self.variant.blockSignals(True)
        self.variant.clear()
        for v in q.variants:
            self.variant.addItem(variant_combo.label(v), v)
            self.variant.setItemData(self.variant.count() - 1, variant_combo.tip(v), Qt.ItemDataRole.ToolTipRole)
        i = self.variant.findData(self._preferred_variant)
        self.variant.setCurrentIndex(i if i >= 0 else max(self.variant.findData("mrd"), 0))
        self.variant.setEnabled(len(q.variants) > 1)
        self.variant.blockSignals(False)

        self.members.setCurrentIndex({"boom": 0, "pair": 1, "slot": 2}[q.kind])
        if q.kind == "pair":
            checked = set(self._pairs_chosen.get(q.key) or default_pairs(q))
            self.pair_list.blockSignals(True)
            self.pair_list.clear()
            for pair in q.pairs:
                item = QListWidgetItem(f"b{pair[0]} – b{pair[1]}")
                item.setData(Qt.ItemDataRole.UserRole, pair)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if pair in checked else Qt.CheckState.Unchecked)
                self.pair_list.addItem(item)
            self.pair_list.blockSignals(False)
        self.log_y.setEnabled(not q.categorical)
        self.style.setEnabled(not q.categorical)
        self.style.setToolTip("categories are drawn as points" if q.categorical else "how the curves are drawn")

    def _on_pairs_changed(self) -> None:
        q = self.current_quantity()
        if q is not None and q.kind == "pair":
            self._pairs_chosen[q.key] = self._checked_pairs()
        self._emit()

    def _checked_pairs(self) -> list:
        out = []
        for i in range(self.pair_list.count()):
            it = self.pair_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    # --- the spec ------------------------------------------------------------------------

    def current_quantity(self) -> Quantity | None:
        item = self.tree.currentItem()
        key = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        return self.catalog[key] if key and self.catalog is not None else None

    def spec(self) -> PanelSpec:
        q = self.current_quantity()
        if q is None:
            return PanelSpec()
        if q.kind == "boom":
            members = tuple(b for b, box in self.boom_boxes.items() if box.isChecked() and b in self.booms)
        elif q.kind == "pair":
            members = tuple(self._checked_pairs())
        else:
            members = (None,)
        variant = self.variant.currentData() or q.variants[0]
        can_filter, can_tau = has_filtered_overlay(q), has_tau_overlay(q, variant)
        self.filtered_box.setEnabled(can_filter)
        self.tau_box.setEnabled(can_tau)
        return PanelSpec(quantity=q, variant=variant, members=members,
                         log_y=self.log_y.isChecked() and not q.categorical, style=self.style.currentText(),
                         show_filtered=can_filter and self.filtered_box.isChecked(),
                         show_tau=can_tau and self.tau_box.isChecked())

    def state(self) -> dict | None:
        """The panel's choices as plain values, for a saved view; None with no quantity."""
        spec = self.spec()
        if spec.quantity is None:
            return None
        return {"quantity": spec.quantity.key, "variant": spec.variant,
                "members": [list(m) if isinstance(m, tuple) else m for m in spec.members],
                "log_y": self.log_y.isChecked(), "style": self.style.currentText(),
                "filtered": self.filtered_box.isChecked(), "tau": self.tau_box.isChecked()}

    def restore(self, state: dict) -> str | None:
        """Choose what a saved view had; why not, where this run can't."""
        key = state.get("quantity")
        if self.catalog is None or key not in self.catalog.quantities:
            return f"{key} isn't in this run"
        q = self.catalog[key]
        members = [tuple(m) if isinstance(m, list) else m for m in state.get("members", [])]
        if q.kind == "boom":
            members = [b for b in members if b in self.booms]
            if not members:
                return f"none of {key}'s booms are in this run"
        self._quiet = True
        self.log_y.setChecked(bool(state.get("log_y")))
        if self.style.findText(state.get("style", "")) >= 0:
            self.style.setCurrentText(state["style"])
        self.filtered_box.setChecked(bool(state.get("filtered")))
        self.tau_box.setChecked(bool(state.get("tau")))
        self._quiet = False
        variant = state.get("variant")
        self.select(key, variant if variant in q.variants else None, tuple(members))
        return None

    def _on_variant_changed(self, *_) -> None:
        if self.variant.count() > 1:
            self._preferred_variant = self.variant.currentData()
        self._emit()

    def _on_quantity_changed(self, *_) -> None:
        q = self.current_quantity()
        if q is not None and not self._quiet:
            self._sync_quantity_widgets(q)
        self._emit()

    def _emit(self) -> None:
        if self._quiet:
            return
        self.specChanged.emit(self.spec())
