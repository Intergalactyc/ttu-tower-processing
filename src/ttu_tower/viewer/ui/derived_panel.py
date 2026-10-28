"""Controls → Derived: quantities defined by a formula of others (e.g.
ws/ustar), kept in the ttu-tower home and offered in the panels' "Derived"
group on every run that has their variables.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
    QWidget,
)

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import derived
from ttu_tower.viewer.derived import Definition, Variable
from ttu_tower.viewer.ui.panel import _VARIANT_LABELS
from ttu_tower.viewer.ui.search_combo import SearchCombo

_NAMES = ("x", "y", "a", "b", "c", "d", "f", "g", "h", "k")
_EACH = "each"  # a variable at each member of the result


def _drop(widget) -> None:
    widget.hide()  # gone at once; deleteLater waits for the event loop
    widget.deleteLater()


class _Row:
    """One variable: its name, quantity, variant and member."""

    def __init__(self, panel: "DerivedPanel", name: str):
        self.panel = panel
        self.name = QLineEdit(name)
        self.name.setMaximumWidth(60)
        self.name.setToolTip("the variable's name in the formula")
        self.quantity = SearchCombo()
        self.quantity.setToolTip("the quantity (type to search)")
        self.variant = QComboBox()
        self.member = QComboBox()
        self.member.setToolTip("each boom (pair) of the derived quantity, or one fixed boom (pair)")
        self.remove = QPushButton("✕")
        self.remove.setMaximumWidth(26)
        self.quantity.fill_quantities(panel.choices(), none_label="— choose a quantity —")
        self.quantity.currentIndexChanged.connect(lambda *_: self.sync())
        self.sync()

    def widgets(self):
        return (self.name, self.quantity, self.variant, self.member, self.remove)

    def sync(self, variant: str | None = None, member=_EACH) -> None:
        """Variants and members of the chosen quantity, keeping (or setting) the choices where it can."""
        q = self.panel_quantity()
        variant = variant if variant is not None else self.variant.currentData()
        member = member if member is not _EACH else self.member.currentData()
        self.variant.clear()
        self.member.clear()
        if q is None:
            return
        for v in q.variants:
            self.variant.addItem(_VARIANT_LABELS.get(v, v), v)
        i = self.variant.findData(variant)
        self.variant.setCurrentIndex(i if i >= 0 else max(self.variant.findData("mrd"), 0))
        self.variant.setEnabled(len(q.variants) > 1)
        if q.kind == "boom":
            self.member.addItem("each boom", None)
            for b in sorted(HEIGHTS, key=HEIGHTS.get):
                self.member.addItem(f"b{b} ({HEIGHTS[b]:g} m)", b)
        elif q.kind == "pair":
            self.member.addItem("each pair", None)
            for pair in q.pairs:
                self.member.addItem(f"b{pair[0]}–b{pair[1]}", f"{pair[0]},{pair[1]}")  # strings: Qt lists tuples
        else:
            self.member.addItem("the slot's value", None)
        key = f"{member[0]},{member[1]}" if isinstance(member, tuple) else member
        self.member.setCurrentIndex(max(self.member.findData(key), 0))
        self.member.setEnabled(self.member.count() > 1)

    def panel_quantity(self):
        key, catalog = self.quantity.currentData(), self.panel.catalog
        return catalog.quantities.get(key) if key is not None and catalog is not None else None

    def variable(self) -> Variable:
        member = self.member.currentData()
        if isinstance(member, str):
            member = tuple(int(b) for b in member.split(","))
        return Variable(self.name.text().strip(), self.quantity.currentData() or "", self.variant.currentData() or "none",
                        member)


class DerivedPanel(QWidget):
    """`changed` fires when a definition is saved or deleted."""

    changed = Signal()

    def __init__(self, store: derived.DerivedStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.catalog = None
        self.editing: str | None = None  # the saved definition's name being edited
        self.rows: list[_Row] = []
        self.list = QListWidget()
        self.list.setToolTip("the derived quantities (every run); pick one to edit it")
        self.list.currentItemChanged.connect(lambda *_: self._open_current())
        new, delete = QPushButton("New"), QPushButton("Delete")
        new.clicked.connect(self.new)
        delete.clicked.connect(self.delete)
        self.name = QLineEdit()
        self.name.setPlaceholderText("name, e.g. u_ustar_ratio")
        self.unit = QLineEdit()
        self.unit.setPlaceholderText("unit (optional)")
        self.unit.setMaximumWidth(110)
        self.grid = QGridLayout()
        self.add_button = QPushButton("+ variable")
        self.add_button.clicked.connect(lambda: self.add_row())
        self.formula = QLineEdit()
        self.formula.setPlaceholderText("formula of the variables, e.g. x/y")
        self.formula.setToolTip("numbers, the variables' names, + - * / ^, brackets, pi, e, exp, ln, log10, sqrt, abs, "
                                "tanh, sin, cos, tan, arctan, sinh, cosh; z is the boom's height (m) for a per-boom "
                                "quantity")
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)

        buttons = QHBoxLayout()
        buttons.addWidget(new)
        buttons.addWidget(delete)
        buttons.addStretch(1)
        head = QHBoxLayout()
        head.addWidget(self.name, stretch=1)
        head.addWidget(self.unit)
        formula = QHBoxLayout()
        formula.addWidget(QLabel("="))
        formula.addWidget(self.formula, stretch=1)
        formula.addWidget(self.save_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list, stretch=1)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("<b>definition</b>"))
        layout.addLayout(head)
        layout.addLayout(self.grid)
        layout.addWidget(self.add_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(formula)
        layout.addWidget(self.status)
        self._fill_list()
        self.new()

    # --- the catalog and the list ---------------------------------------------------------------

    def set_catalog(self, catalog, notes=()) -> None:
        """The run's quantities to choose from; `notes`: definitions this run can't resolve, and why."""
        self.catalog = catalog
        for row in self.rows:
            current = row.variable()
            row.quantity.fill_quantities(self.choices(), none_label="— choose a quantity —")
            row.sync(current.variant, current.member)
        self._fill_list(notes)

    def choices(self) -> list:
        if self.catalog is None:
            return []
        return [q for qs in self.catalog.grouped().values() for q in qs
                if not q.categorical and q.table != derived.TABLE]

    def _fill_list(self, notes=()) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        unresolved = {n.split(":", 1)[0]: n for n in notes}
        for d in sorted(self.store.items, key=lambda d: d.name.lower()):
            item = QListWidgetItem(f"{d.name} = {d.formula}" + (f"  [{d.unit}]" if d.unit else ""))
            item.setData(Qt.ItemDataRole.UserRole, d.name)
            if d.name in unresolved:
                item.setForeground(Qt.GlobalColor.gray)
                item.setToolTip(f"not on this run: {unresolved[d.name]}")
            self.list.addItem(item)
            if d.name == self.editing:
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)

    def _open_current(self) -> None:
        item = self.list.currentItem()
        name = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        d = next((d for d in self.store.items if d.name == name), None)
        if d is not None:
            self.open(d)

    # --- the editor ---------------------------------------------------------------------------

    def add_row(self, name: str | None = None) -> _Row:
        used = {r.name.text() for r in self.rows}
        row = _Row(self, name or next((n for n in _NAMES if n not in used), f"v{len(self.rows) + 1}"))
        row.remove.clicked.connect(lambda *_, r=row: self._remove_row(r))
        self.rows.append(row)
        self._layout_rows()
        return row

    def _remove_row(self, row: _Row) -> None:
        self.rows.remove(row)
        for w in row.widgets():
            _drop(w)
        self._layout_rows()

    def _layout_rows(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        for i, row in enumerate(self.rows):
            for j, w in enumerate(row.widgets()):
                self.grid.addWidget(w, i, j)
        self.grid.setColumnStretch(1, 1)

    def new(self) -> None:
        self.editing = None
        self.list.blockSignals(True)
        self.list.clearSelection()
        self.list.setCurrentItem(None)
        self.list.blockSignals(False)
        self.name.clear()
        self.unit.clear()
        self.formula.clear()
        for row in list(self.rows):
            self._remove_row(row)
        self.add_row()
        self.add_row()
        self.status.setText("a new derived quantity: name it, choose its variables, write its formula, Save")

    def open(self, d: Definition) -> None:
        self.editing = d.name
        self.name.setText(d.name)
        self.unit.setText(d.unit)
        self.formula.setText(d.formula)
        for row in list(self.rows):
            self._remove_row(row)
        for v in d.variables:
            row = self.add_row(v.name)
            row.quantity.setCurrentIndex(max(row.quantity.findData(v.key), 0))
            row.sync(v.variant, v.member)
        self.status.setText(f"editing {d.name}: Save replaces it")

    def definition(self) -> Definition:
        return Definition(self.name.text().strip(), self.formula.text().strip(),
                          tuple(r.variable() for r in self.rows), self.unit.text().strip())

    def save(self) -> bool:
        d = self.definition()
        try:
            if self.catalog is None:
                raise ValueError("the run's quantities are still loading")
            if any(not v.key for v in d.variables):
                raise ValueError("choose a quantity for every variable")
            q = derived.resolve(d, self.catalog)
            if d.name != self.editing and any(x.name == d.name for x in self.store.items):
                raise ValueError(f"there's already a derived quantity named {d.name}")
        except ValueError as exc:
            self.status.setText(f"<span style='color:#b00'>{exc}</span>")
            return False
        self.store.put(d, replacing=self.editing)
        self.editing = d.name
        self._fill_list()
        kind = {"boom": "per boom", "pair": "per boom pair", "slot": "per slot"}[q.kind]
        self.status.setText(f"saved {d.name} ({kind}): in the panels' “Derived” group")
        self.changed.emit()
        return True

    def delete(self) -> None:
        if self.editing is None:
            return
        self.store.remove(self.editing)
        self.new()
        self._fill_list()
        self.changed.emit()
