"""Choosing what to export as a matplotlib figure, where, and at what size."""
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
    QPushButton, QSpinBox,
)

_FORMATS = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"


class ExportDialog(QDialog):
    def __init__(self, tab: str, tab_ready: bool, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        self.settings = settings
        self.what = QComboBox()
        self.what.addItem("the timelines (panels A and B, the viewed interval)", "timelines")
        self.what.addItem(f"the {tab} tab", "summary")
        self.what.model().item(1).setEnabled(tab_ready)
        if not tab_ready:
            self.what.setItemText(1, f"the {tab} tab (nothing to draw, or no figure for it)")
        self.width = QDoubleSpinBox()
        self.height = QDoubleSpinBox()
        for box, value in ((self.width, float(settings.value("export/width", 6.5))),
                           (self.height, float(settings.value("export/height", 4.0)))):
            box.setRange(1.0, 40.0)
            box.setSingleStep(0.5)
            box.setSuffix(" in")
            box.setValue(value)
        self.dpi_box = QSpinBox()
        self.dpi_box.setRange(50, 1200)
        self.dpi_box.setValue(int(settings.value("export/dpi", 300)))
        self.dpi_box.setToolTip("for PNG; PDF and SVG are vector")
        folder = Path(settings.value("export/folder", str(Path.home())))
        self.path_edit = QLineEdit(str(folder / "ttu-view-figure.png"))
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        form = QFormLayout(self)
        form.addRow("figure of", self.what)
        size = QHBoxLayout()
        size.addWidget(self.width)
        size.addWidget(self.height)
        size.addWidget(self.dpi_box)
        form.addRow("size (w × h), dpi", size)
        path = QHBoxLayout()
        path.addWidget(self.path_edit, stretch=1)
        path.addWidget(browse)
        form.addRow("file (.png, .pdf, .svg)", path)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", self.path_edit.text(), _FORMATS)
        if path:
            self.path_edit.setText(path)

    def _accept(self) -> None:
        if Path(self.path_edit.text()).suffix.lower() not in (".png", ".pdf", ".svg"):
            self.path_edit.setStyleSheet("background:#f6c6c6;")
            return
        self.settings.setValue("export/width", self.width.value())
        self.settings.setValue("export/height", self.height.value())
        self.settings.setValue("export/dpi", self.dpi_box.value())
        self.settings.setValue("export/folder", str(Path(self.path_edit.text()).parent))
        self.accept()

    def target(self) -> str:
        return self.what.currentData()

    def size(self) -> tuple[float, float]:
        return self.width.value(), self.height.value()

    def dpi(self) -> int:
        return self.dpi_box.value()

    def path(self) -> str:
        return self.path_edit.text()
