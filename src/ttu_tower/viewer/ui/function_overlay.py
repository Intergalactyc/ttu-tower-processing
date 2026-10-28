"""A curve typed by hand to lay over the scatter (drawn, not fitted): one or
more pieces, each y in terms of x (and any variables the caller offers) over an
optional x range.
"""
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ttu_tower.viewer import expression

_BAD = "background:#f6c6c6;"


def _drop(widget) -> None:
    widget.hide()  # gone at once; deleteLater waits for the event loop
    widget.deleteLater()


class _Piece:
    def __init__(self, formula: str = "", lo: str = "", hi: str = "", variables: dict | None = None):
        self.formula, self.lo, self.hi = QLineEdit(formula), QLineEdit(lo), QLineEdit(hi)
        self.formula.setPlaceholderText("y = …, e.g. 0.18*x^0.5")
        names = "".join(f", {name} ({what})" for name, what in (variables or {}).items())
        self.formula.setToolTip(f"y in terms of x: numbers, x{names}, + - * / ^, brackets, pi, e, and "
                                + ", ".join(expression.FUNCTIONS) + "()")
        for edit, which in ((self.lo, "from"), (self.hi, "to")):
            edit.setPlaceholderText("−∞" if which == "from" else "∞")
            edit.setMaximumWidth(70)
            edit.setToolTip(f"x {which} (blank: no bound); the piece covers from ≤ x < to")
        self.remove = QPushButton("✕")
        self.remove.setMaximumWidth(26)
        self.remove.setToolTip("remove this piece")


class FunctionOverlay(QWidget):
    """`changed` fires when a piece is finished editing, added or removed."""

    changed = Signal()

    def __init__(self, variables: dict | None = None, parent=None):
        """`variables`: {name: what it is} that formulas may use besides x."""
        super().__init__(parent)
        self.variables = dict(variables or {})
        self.pieces: list[_Piece] = []
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.add_button = QPushButton("+ piece")
        self.add_button.setToolTip("another piece over another x range (where pieces overlap, the first one wins)")
        self.add_button.clicked.connect(lambda: self.add_piece(emit=True))
        self.from_fit = QPushButton("from fit")
        self.from_fit.setToolTip("replace the pieces with the current fit's curve, to edit")
        self.status = QLabel()
        self.status.setWordWrap(True)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.from_fit)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.grid)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        self.add_piece()

    def add_piece(self, formula: str = "", lo: str = "", hi: str = "", emit: bool = False) -> None:
        piece = _Piece(formula, lo, hi, self.variables)
        for edit in (piece.formula, piece.lo, piece.hi):
            edit.editingFinished.connect(self._edited)
        piece.remove.clicked.connect(lambda *_, p=piece: self._remove(p))
        self.pieces.append(piece)
        self._layout()
        if emit:
            self.changed.emit()

    def _remove(self, piece: _Piece) -> None:
        self.pieces.remove(piece)
        for w in (piece.formula, piece.lo, piece.hi, piece.remove):
            _drop(w)
        if not self.pieces:
            self.add_piece()
        self._layout()
        self.changed.emit()

    def _layout(self) -> None:
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if isinstance(w, QLabel):  # the row labels are made afresh
                _drop(w)
        for row, p in enumerate(self.pieces):
            self.grid.addWidget(QLabel("y ="), row, 0)
            self.grid.addWidget(p.formula, row, 1)
            self.grid.addWidget(QLabel("for x from"), row, 2)
            self.grid.addWidget(p.lo, row, 3)
            self.grid.addWidget(QLabel("to"), row, 4)
            self.grid.addWidget(p.hi, row, 5)
            self.grid.addWidget(p.remove, row, 6)
        self.grid.setColumnStretch(1, 1)

    def _edited(self) -> None:
        self.changed.emit()

    def set_pieces(self, pieces) -> None:
        """[(formula, from, to)] as text, replacing what's there."""
        for p in self.pieces:
            for w in (p.formula, p.lo, p.hi, p.remove):
                _drop(w)
        self.pieces = []
        for formula, lo, hi in pieces or [("", "", "")]:
            self.add_piece(formula, lo, hi)
        self.changed.emit()

    def texts(self) -> list[list[str]]:
        return [[p.formula.text(), p.lo.text(), p.hi.text()] for p in self.pieces]

    def active(self) -> bool:
        return any(p.formula.text().strip() for p in self.pieces)

    def function(self):
        """The overlay's y(x) and its pieces' x ranges, or None with nothing typed; marks what can't be read
        (and says why in the status line)."""
        parts, errors = [], []
        for i, p in enumerate(self.pieces, start=1):
            for edit in (p.formula, p.lo, p.hi):
                edit.setStyleSheet("")
            if not p.formula.text().strip():
                continue
            try:
                f = expression.compile_expression(p.formula.text(), tuple(self.variables))
            except ValueError as exc:
                p.formula.setStyleSheet(_BAD)
                errors.append(f"piece {i}: {exc}")
                continue
            bounds = []
            for edit, default in ((p.lo, -np.inf), (p.hi, np.inf)):
                try:
                    bounds.append(expression.parse_bound(edit.text(), default))
                except ValueError as exc:
                    edit.setStyleSheet(_BAD)
                    errors.append(f"piece {i}: {exc}")
                    bounds.append(None)
            if None not in bounds:
                parts.append((f, *bounds))
        self.status.setText("<span style='color:#b00'>" + "; ".join(errors) + "</span>" if errors else "")
        return (expression.piecewise(parts), [(lo, hi) for _, lo, hi in parts]) if parts else None

    # --- saved views ------------------------------------------------------------------------------

    def get_state(self):
        return self.texts()

    def put_state(self, value) -> bool:
        self.set_pieces([tuple(v) for v in value or []])
        return True
