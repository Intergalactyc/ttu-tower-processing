"""A tab widget whose tabs wrap onto more rows when its panel is narrow,
instead of scrolling or being cut off. It is a QTabWidget with its own tab bar
hidden and a row of tab buttons above it (`container()`), so it is used like
any QTabWidget. Below the buttons the pages scroll where the panel is smaller
than a page needs, so no page can widen the window past the screen (a
window's minimum is the sum of its parts', and a maximized window wider than
the screen runs off its edge) and the buttons always span the visible width.
"""
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QLayout, QScrollArea, QSizePolicy, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

_TAB_STYLE = """
QToolButton { border: 1px solid palette(mid); border-bottom: 2px solid transparent; background: palette(button);
              padding: 3px 9px; border-top-left-radius: 3px; border-top-right-radius: 3px; }
QToolButton:hover { background: palette(light); }
QToolButton:checked { background: palette(base); border-bottom: 2px solid palette(highlight); }
"""


class FlowLayout(QLayout):
    """Lays its items out left to right, starting a new row where one is full."""

    def __init__(self, parent=None, spacing: int = 2):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self) -> QSize:
        """One row of every item."""
        w = sum(i.sizeHint().width() for i in self._items) + self.spacing() * max(len(self._items) - 1, 0)
        h = max((i.sizeHint().height() for i in self._items), default=0)
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self) -> QSize:
        """The widest item: everything else can wrap."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect: QRect, move: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_height = area.x(), area.y(), 0
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:  # full: a new row
                x, y = area.x(), y + row_height + self.spacing()
                row_height = 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self.spacing()
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y() + m.bottom()


class _FillLayout(QVBoxLayout):
    """Reports no height-for-width: a resizable scroll area would otherwise size
    its contents to their preferred height (wrapped labels make a layout
    height-for-width) and scroll where they would fit.
    """

    def hasHeightForWidth(self) -> bool:
        return False


class FitScroll(QScrollArea):
    """Its widget, filling it where there's room and scrolling where there isn't."""

    FLOOR = QSize(240, 150)  # it can't be squeezed below this

    def __init__(self, widget: QWidget):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        layout = _FillLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        self.setWidget(holder)

    def sizeHint(self) -> QSize:
        """At least the widget's minimum: it opens without scrolling wherever there's room."""
        w = self.widget()
        return w.sizeHint().expandedTo(w.minimumSizeHint())

    def minimumSizeHint(self) -> QSize:
        return self.widget().minimumSizeHint().boundedTo(self.FLOOR)


class WrapTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabBar().hide()
        self.bar = QWidget()
        self.bar.setStyleSheet(_TAB_STYLE)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.bar.setSizePolicy(policy)
        self._flow = FlowLayout(self.bar)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idClicked.connect(self.setCurrentIndex)
        self.currentChanged.connect(self._check)
        self._container = None

    def container(self) -> QWidget:
        """The tab buttons above the (scrolling) pages: what goes into a layout."""
        if self._container is None:
            self._container = QWidget()
            layout = QVBoxLayout(self._container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self.bar)
            self.scroll = FitScroll(self)
            layout.addWidget(self.scroll, stretch=1)
        return self._container

    def tabInserted(self, index: int) -> None:
        super().tabInserted(index)
        self._rebuild()

    def tabRemoved(self, index: int) -> None:
        super().tabRemoved(index)
        self._rebuild()

    def setTabText(self, index: int, text: str) -> None:
        super().setTabText(index, text)
        self._rebuild()

    def setTabToolTip(self, index: int, tip: str) -> None:
        super().setTabToolTip(index, tip)
        button = self._group.button(index)
        if button is not None:
            button.setToolTip(tip)

    def buttons(self) -> list[QToolButton]:
        return [self._group.button(i) for i in range(self.count())]

    def _rebuild(self) -> None:
        for button in self._group.buttons():
            self._group.removeButton(button)
            self._flow.removeWidget(button)
            button.deleteLater()
        for i in range(self.count()):
            button = QToolButton()
            button.setText(self.tabText(i))
            button.setToolTip(self.tabToolTip(i))
            button.setCheckable(True)
            button.setAutoRaise(False)
            self._group.addButton(button, i)
            self._flow.addWidget(button)
        self._check(self.currentIndex())
        self.bar.updateGeometry()

    def _check(self, index: int) -> None:
        button = self._group.button(index)
        if button is not None:
            button.setChecked(True)
