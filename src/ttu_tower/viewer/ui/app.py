"""Starting the viewer's Qt application."""
import sys
import warnings

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMessageBox

from ttu_tower.viewer.run import RunHandle, RunNotFound, list_registered


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(background="w", foreground="k", antialias=False, useNumba=False)


def open_initial_run(tag: str | None, run_dir: str | None, test: bool, settings_tag: str | None) -> RunHandle | None:
    if run_dir:
        return RunHandle.from_dir(run_dir)
    if tag:
        return RunHandle.open(tag, test=test)
    registered = [r["tag"] for r in list_registered(test) if r["exists"]]
    if settings_tag in registered:
        return RunHandle.open(settings_tag, test=test)
    return RunHandle.open(registered[0], test=test) if registered else None


def main(tag: str | None = None, run_dir: str | None = None, test: bool = False) -> int:
    # the pipeline's expected all-NaN/empty-window warnings, which primary tallies rather than prints
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("ttu-view")
    configure_pyqtgraph()

    from ttu_tower.viewer.ui.main_window import MainWindow

    window = MainWindow()
    try:
        run = open_initial_run(tag, run_dir, test, window.settings.value("last_run"))
    except RunNotFound as exc:
        QMessageBox.warning(window, "ttu-view", str(exc))
        run = None
    if run is not None:
        window.open_run(run)
    window.show()
    return app.exec()
