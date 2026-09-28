"""Filling a QTableWidget from a DataFrame, with per-cell text and colors."""
import numpy as np
import pandas as pd
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem

BAD = QColor("#f6c6c6")
WARN = QColor("#fbe3c0")
GOOD = QColor("#dff0d8")
MUTED = QColor("#f2f2f2")


def text(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}g}"
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
