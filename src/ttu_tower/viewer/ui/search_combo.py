"""A combo for a long list of quantities: type any part of a title to search."""
from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QCompleter

from ttu_tower.viewer.ui.view_state import put_combo


class SearchCombo(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

    def put_state(self, value) -> bool:
        return put_combo(self, value, free_text=False)

    def fill_quantities(self, quantities, none_label: str | None = "—") -> None:
        """The quantities by title (with the table where titles repeat), keeping the current one."""
        key = self.currentData()
        self.blockSignals(True)
        self.clear()
        if none_label is not None:
            self.addItem(none_label, None)
        titles = Counter(q.title for q in quantities)
        for q in quantities:
            self.addItem(q.title if titles[q.title] == 1 else f"{q.title} · {q.table}", q.key)
            self.setItemData(self.count() - 1, f"{q.group}: {q.table} {q.variable}", Qt.ItemDataRole.ToolTipRole)
        self.setCurrentIndex(max(self.findData(key), 0))
        self.blockSignals(False)
