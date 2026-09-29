"""The viewer's version of pyqtgraph's right-click Export dialog: room for the
export options, a title and axis labels to export with (the plot itself is
left as it was), a "Saved to …" confirmation, and a CSV exporter whose
columns say what they are (time in the display zone, "b5 (16.8 m)", the axis
labels) rather than x0000/y0000.
"""
import contextlib

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import exporters
from pyqtgraph.exporters import CSVExporter
from pyqtgraph.exporters.Exporter import Exporter
from pyqtgraph.GraphicsScene import exportDialog
from pyqtgraph.parametertree import Parameter
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLineEdit, QMessageBox

_LAYOUTS = ("one row per x (curves side by side)", "x and y columns per curve")


# --- what a plot's items are, for CSV ---------------------------------------------------------

def label_text(axis) -> str:
    """An axis's label with its units, as shown."""
    text = axis.labelText or ""
    return f"{text} [{axis.labelUnits}]" if axis.labelUnits else text


def _plot_item(item):
    if isinstance(item, pg.PlotItem):
        return item
    if isinstance(item, pg.ViewBox) and isinstance(item.parentItem(), pg.PlotItem):
        return item.parentItem()
    return None


def _series(item) -> tuple[np.ndarray, np.ndarray] | None:
    """An item's full-resolution data: what it was given, not what's drawn after
    decimation (the viewer's curves keep that as `export_data`).
    """
    data = getattr(item, "export_data", None)
    if data is None:
        if isinstance(item, pg.ScatterPlotItem):
            data = item.getData()
        elif hasattr(item, "getOriginalDataset"):
            data = item.getOriginalDataset()
        elif hasattr(item, "getData"):
            data = item.getData()
    if data is None or data[0] is None or len(data[0]) == 0:
        return None
    x, y = np.asarray(data[0], dtype=float), np.asarray(data[1], dtype=float)
    if x.size == y.size + 1:  # a step histogram's edges: its bins' centres
        x = (x[:-1] + x[1:]) / 2
    return x, y


def plot_frame(plot, joined: bool = True, visible_only: bool = True) -> pd.DataFrame:
    """A plot's data as a table: the index is x (time in the display zone for a
    date axis), each column a curve named by its label.
    """
    bottom, left = plot.getAxis("bottom"), plot.getAxis("left")
    x_name, y_name = label_text(bottom) or "x", label_text(left) or "y"
    date_axis = isinstance(bottom, pg.DateAxisItem)
    (x0, x1), _ = plot.getViewBox().viewRange()
    log_x = plot.ctrl.logXCheck.isChecked() if hasattr(plot, "ctrl") else False
    columns, used = [], set()
    for k, item in enumerate(plot.items):
        if getattr(item, "export_skip", False) or not item.isVisible():
            continue
        if isinstance(item, pg.ErrorBarItem):
            continue
        if not (hasattr(item, "implements") and item.implements("plotData")) and not isinstance(
                item, pg.ScatterPlotItem):
            continue
        data = _series(item)
        if data is None:
            continue
        x, y = data
        if visible_only:
            view_x = np.log10(np.where(x > 0, x, np.nan)) if log_x else x
            keep = (view_x >= x0) & (view_x <= x1)
            x, y = x[keep], y[keep]
        name = getattr(item, "export_label", None) or (item.name() if hasattr(item, "name") else None) \
            or f"{y_name} ({k + 1})"
        while name in used:
            name += "'"
        used.add(name)
        columns.append((name, x, y))
    if not columns:
        return pd.DataFrame()
    index_name = x_name
    if date_axis:
        tz = getattr(plot, "export_tz", None)
        offset = bottom.utcOffset or 0
        index_name = f"time ({tz})" if tz else f"time (UTC{-offset / 3600:+g} h)"

    def to_index(x):
        if not date_axis:
            return pd.Index(x, name=index_name)
        t = pd.to_datetime(np.asarray(x) - (bottom.utcOffset or 0), unit="s")
        return pd.Index(t.strftime("%Y-%m-%d %H:%M:%S"), name=index_name)

    if joined:
        parts = [pd.Series(y, index=to_index(x), name=name).groupby(level=0).first() for name, x, y in columns]
        return pd.concat(parts, axis=1).sort_index()
    frame = pd.concat([pd.DataFrame({f"{name}: {index_name}": to_index(x).to_numpy(), f"{name}: {y_name}": y})
                       for name, x, y in columns], axis=1)
    frame.index.name = "row"
    return frame


class LabelledCSVExporter(Exporter):
    Name = "CSV of plot data (labelled columns)"
    windows = []

    def __init__(self, item):
        super().__init__(item)
        plot = _plot_item(item)
        layout = getattr(plot, "export_layout", "joined") if plot is not None else "joined"
        self.params = Parameter.create(name="params", type="group", children=[
            {"name": "separator", "type": "list", "value": "comma", "limits": ["comma", "tab"]},
            {"name": "significant digits", "type": "int", "value": 10, "limits": [1, 17]},
            {"name": "layout", "type": "list", "limits": list(_LAYOUTS),
             "value": _LAYOUTS[0] if layout == "joined" else _LAYOUTS[1]},
            {"name": "rows", "type": "list", "limits": ["the viewed x range", "all the data"],
             "value": "the viewed x range"},
        ])

    def parameters(self):
        return self.params

    def export(self, fileName=None):
        plot = _plot_item(self.item)
        if plot is None:
            raise TypeError("choose a Plot (not the entire scene) for CSV export")
        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv"])
            return
        frame = plot_frame(plot, joined=self.params["layout"] == _LAYOUTS[0],
                           visible_only=self.params["rows"] == "the viewed x range")
        frame.to_csv(fileName, sep="," if self.params["separator"] == "comma" else "\t",
                     float_format=f"%.{self.params['significant digits']}g", encoding="utf-8")


# --- the dialog ---------------------------------------------------------------------------------

class ViewerExportDialog(exportDialog.ExportDialog):
    """pyqtgraph's Export dialog with a roomier options panel, a title and axis
    labels applied only while exporting, and a confirmation when it's saved.
    """

    def __init__(self, scene):
        super().__init__(scene)
        grid = self.ui.gridLayout
        grid.setSpacing(4)
        for row, stretch in ((1, 2), (3, 2), (5, 4)):  # item tree, formats, export options
            grid.setRowStretch(row, stretch)
        self.ui.paramTree.setMinimumHeight(170)
        self.text = {key: QLineEdit() for key in ("title", "bottom", "left")}
        self.text["title"].setPlaceholderText("none")
        box = QGroupBox("Title and axis labels (for this export only)")
        form = QFormLayout(box)
        for key, label in (("title", "title"), ("bottom", "x axis"), ("left", "y axis")):
            form.addRow(label, self.text[key])
        buttons = [self.ui.copyBtn, self.ui.exportBtn, self.ui.closeBtn]
        for b in buttons:
            grid.removeWidget(b)
        grid.addWidget(box, 6, 0, 1, 3)
        for col, b in enumerate(buttons):
            grid.addWidget(b, 7, col)
        self.resize(460, 720)

    def exportItemChanged(self, item, prev):
        super().exportItemChanged(item, prev)
        plot = _plot_item(item.gitem) if item is not None else None
        for key, edit in self.text.items():
            edit.setEnabled(plot is not None)
        if plot is not None:
            shown = plot.titleLabel.text if plot.titleLabel.isVisible() else ""
            self.text["title"].setText(shown or getattr(plot, "export_title", ""))  # a title drawn outside the plot
            for side in ("bottom", "left"):
                self.text[side].setText(label_text(plot.getAxis(side)))

    @contextlib.contextmanager
    def _texts(self, item):
        """The chosen title and labels on the plot while it's exported, then as before."""
        plot = _plot_item(item)
        if plot is None:
            yield
            return
        title = (plot.titleLabel.text, plot.titleLabel.isVisible())
        axes = {side: (plot.getAxis(side).labelText, plot.getAxis(side).labelUnits, plot.getAxis(side).label.isVisible())
                for side in ("bottom", "left")}
        try:
            text = self.text["title"].text().strip()
            plot.setTitle(text if text else None)
            for side in ("bottom", "left"):
                label = self.text[side].text().strip()
                plot.getAxis(side).setLabel(label if label else None, units=None)
                plot.getAxis(side).showLabel(bool(label))
            yield
        finally:
            plot.setTitle(title[0] if title[1] else None)
            for side, (text, units, shown) in axes.items():
                plot.getAxis(side).setLabel(text, units=units)
                plot.getAxis(side).showLabel(shown)

    def exportClicked(self):
        self.selectBox.hide()
        exporter = self.currentExporter
        if exporter is None:
            return
        original = type(exporter).export
        dialog = self

        def export(*args, **kwargs):
            file_name = kwargs.get("fileName", args[0] if args else None)
            if file_name is None and not kwargs.get("copy") and not kwargs.get("toBytes"):
                return original(exporter, *args, **kwargs)  # asks for a file, then comes back here
            try:
                with dialog._texts(exporter.item):
                    result = original(exporter, *args, **kwargs)
            except Exception as exc:
                QMessageBox.warning(dialog, "Export", f"Couldn't export: {exc}")
                return None
            if file_name:
                QMessageBox.information(dialog, "Export", f"Saved to {file_name}")
            return result

        exporter.export = export
        try:
            exporter.export()
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Couldn't export: {exc}")

    def copyClicked(self):
        self.selectBox.hide()
        with self._texts(self.currentExporter.item):
            self.currentExporter.export(copy=True)


def install() -> None:
    """Use the viewer's Export dialog and labelled CSV in every plot's right-click menu."""
    exportDialog.ExportDialog = ViewerExportDialog
    registered = exporters.Exporter.Exporters
    if CSVExporter in registered:
        registered.remove(CSVExporter)
    if LabelledCSVExporter not in registered:
        registered.insert(0, LabelledCSVExporter)

