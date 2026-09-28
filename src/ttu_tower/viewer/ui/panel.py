"""The controls for one timeline panel: which quantity, variant and booms
(or boom pairs), and how to draw it.
"""
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer.catalog import Catalog, Quantity
from ttu_tower.viewer.timeline import has_filtered_overlay, has_tau_overlay

_VARIANT_LABELS = {"none": "—", "mrd": "mrd (selected τ)", "naive": "naive (10 min)",
                   "mrd_unexcised": "mrd unexcised"}
STYLES = ("lines", "points", "lines + points")


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

        self.search = QLineEdit()
        self.search.setPlaceholderText("search quantities…")
        self.search.textChanged.connect(self._filter_tree)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._on_quantity_changed)

        self.variant = QComboBox()
        self.variant.currentIndexChanged.connect(lambda *_: self._emit())

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
        for label, pick in (("all", lambda b: True), ("none", lambda b: False), ("odd", lambda b: b % 2 == 1),
                            ("1/5/10", lambda b: b in (1, 5, 10))):
            btn = QPushButton(label)
            btn.setFlat(True)
            btn.clicked.connect(lambda _=False, p=pick: self.set_booms([b for b in self.booms if p(b)]))
            buttons.addWidget(btn)
        grid.addLayout(buttons, 5, 0, 1, 2)

        self.pair_list = QListWidget()
        self.pair_list.itemChanged.connect(lambda *_: self._emit())
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
        current = self.variant.currentData()
        self.variant.blockSignals(True)
        self.variant.clear()
        for v in q.variants:
            self.variant.addItem(_VARIANT_LABELS.get(v, v), v)
        i = self.variant.findData(current)
        self.variant.setCurrentIndex(i if i >= 0 else 0)
        self.variant.setEnabled(len(q.variants) > 1)
        self.variant.blockSignals(False)

        self.members.setCurrentIndex({"boom": 0, "pair": 1, "slot": 2}[q.kind])
        if q.kind == "pair":
            checked = set(self._checked_pairs())
            self.pair_list.blockSignals(True)
            self.pair_list.clear()
            for pair in q.pairs:
                item = QListWidgetItem(f"b{pair[0]} – b{pair[1]}")
                item.setData(Qt.ItemDataRole.UserRole, pair)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                on = pair in checked or (not checked and pair == q.pairs[0])
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                self.pair_list.addItem(item)
            self.pair_list.blockSignals(False)
        self.log_y.setEnabled(not q.categorical)

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

    def _on_quantity_changed(self, *_) -> None:
        q = self.current_quantity()
        if q is not None and not self._quiet:
            self._sync_quantity_widgets(q)
        self._emit()

    def _emit(self) -> None:
        if self._quiet:
            return
        self.specChanged.emit(self.spec())
