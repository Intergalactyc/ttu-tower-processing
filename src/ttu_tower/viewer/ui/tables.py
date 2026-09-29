"""Filling a QTableWidget from a DataFrame, with per-cell text and colors."""
import numpy as np
import pandas as pd
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QSizePolicy, QTableWidget, QTableWidgetItem

BAD = QColor("#f6c6c6")
WARN = QColor("#fbe3c0")
GOOD = QColor("#dff0d8")
MUTED = QColor("#f2f2f2")


def number(v: float, digits: int = 4) -> str:
    """`digits` significant figures, but a whole number (a count held as a
    float, e.g. beside NaNs) in full rather than as 1.01e+04.
    """
    if float(v).is_integer() and abs(v) < 1e12:
        return f"{int(v)}"
    return f"{v:.{digits}g}"


def text(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return number(v, digits)
    return str(v)


def new_table() -> QTableWidget:
    table = QTableWidget()
    table.verticalHeader().setDefaultSectionSize(18)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setAlternatingRowColors(False)
    return table


def fill(table: QTableWidget, df: pd.DataFrame, *, index: bool = False, color=None, fmt=None) -> None:
    """`color(row_label, column, value) -> QColor | None` and `fmt(column, value)
    -> str | None` customise cells; the index becomes the row headers if `index`.
    """
    table.clear()
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    table.verticalHeader().setVisible(index)
    if index:
        table.setVerticalHeaderLabels([str(i) for i in df.index])
    for i, (label, row) in enumerate(df.iterrows()):
        for j, col in enumerate(df.columns):
            value = row[col]
            shown = fmt(col, value) if fmt is not None else None
            item = QTableWidgetItem(shown if shown is not None else text(value))
            c = color(label, col, value) if color is not None else None
            if c is not None:
                item.setBackground(QBrush(c))
            table.setItem(i, j, item)
    table.resizeColumnsToContents()


def rows_height(table: QTableWidget, rows: int) -> int:
    """Pixels that show the header and `rows` rows (and the horizontal scroll bar when needed)."""
    height = table.frameWidth() * 2 + table.verticalHeader().defaultSectionSize() * max(rows, 1)
    if table.horizontalHeader().isVisible():
        height += table.horizontalHeader().sizeHint().height()
    if table.horizontalHeader().length() > table.viewport().width():
        height += table.horizontalScrollBar().sizeHint().height()
    return height + 2


def fit_rows(table: QTableWidget, max_rows: int = 10) -> None:
    """Exactly as tall as its rows (up to `max_rows`, then it scrolls), so a
    short table leaves the rest of the space to the plot above it.
    """
    table.setFixedHeight(rows_height(table, min(table.rowCount(), max_rows)))
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
