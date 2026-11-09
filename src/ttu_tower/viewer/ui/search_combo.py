"""A combo for a long list of quantities: type to search (by title, symbol
or long name; see viewer.search).
"""
from collections import Counter

from PySide6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import QComboBox, QCompleter

from ttu_tower.viewer import search
from ttu_tower.viewer.ui.view_state import put_combo

TERMS_ROLE = Qt.ItemDataRole.UserRole + 1  # an item's search terms (search.terms); else its text


class _Matches(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query = search.Query("")

    def set_query(self, text: str) -> None:
        if not hasattr(self, "endFilterChange"):  # before Qt 6.10
            self.query = search.Query(text)
            self.invalidateRowsFilter()
            return
        self.beginFilterChange()
        self.query = search.Query(text)
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, row, parent) -> bool:
        if not self.query:
            return True
        index = self.sourceModel().index(row, 0, parent)
        return self.query.matches(index.data(TERMS_ROLE) or search.terms(index.data() or ""))


class SearchCombo(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setCompleter(None)  # the combo's own matches text only
        self._matches = _Matches(self)
        self._matches.setSourceModel(self.model())
        self._completer = QCompleter(self._matches, self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self._completer.setWidget(self.lineEdit())
        self._completer.activated[QModelIndex].connect(self._chosen)
        self.lineEdit().textEdited.connect(self._search)

    def _search(self, text: str) -> None:
        self._matches.set_query(text)
        if not text.strip() or self._matches.rowCount() == 0:
            self._completer.popup().hide()
            return
        self._completer.complete()
        self._completer.popup().setCurrentIndex(self._completer.completionModel().index(0, 0))

    def _chosen(self, index) -> None:
        """A completion picked: the combo's item under it."""
        row = self._matches.mapToSource(self._completer.completionModel().mapToSource(index)).row()
        if row >= 0:
            self.setCurrentIndex(row)
            self.setEditText(self.itemText(row))

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
            self.setItemData(self.count() - 1, search.quantity_terms(q), TERMS_ROLE)
        self.setCurrentIndex(max(self.findData(key), 0))
        self.blockSignals(False)
