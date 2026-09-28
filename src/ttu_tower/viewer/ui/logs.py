"""Where the viewer's diagnostics go: Python warnings and Qt's own messages
into a rotating log file under the ttu-tower home (not the console), errors
into the log and the console. A counter lets the status bar say how many
warnings were logged.
"""
import logging
import sys
import tempfile
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QObject, QtMsgType, Signal, qInstallMessageHandler

from ttu_tower.io.runs import locate_home

LOGGER = "ttu_view"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _Counter(QObject):
    warned = Signal(int)  # warnings logged so far

    def __init__(self):
        super().__init__()
        self.count = 0

    def bump(self) -> None:
        self.count += 1
        self.warned.emit(self.count)


counter: _Counter | None = None
log_path: Path | None = None


class _Counting(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING and counter is not None:
            counter.bump()


def _qt_message(mode, context, message) -> None:
    level = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
             QtMsgType.QtWarningMsg: logging.WARNING}.get(mode, logging.ERROR)
    logging.getLogger(f"{LOGGER}.qt").log(level, message)


def install() -> Path:
    """Route warnings and Qt messages to the log file; returns its path."""
    global counter, log_path
    home = locate_home()
    folder = home / "viewer" if home is not None else Path(tempfile.gettempdir())
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "ttu-view.log"
    counter = _Counter()

    to_file = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    to_file.setFormatter(logging.Formatter(_FORMAT))
    to_console = logging.StreamHandler(sys.stderr)
    to_console.setLevel(logging.ERROR)
    to_console.setFormatter(logging.Formatter(_FORMAT))
    for name in (LOGGER, "py.warnings"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in (to_file, to_console, _Counting()):
            logger.addHandler(handler)

    logging.captureWarnings(True)
    warnings.simplefilter("default")  # each warning once per place it's raised
    # the pipeline's expected all-NaN/empty-window warnings while reprocessing, which primary tallies rather than prints
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")
    qInstallMessageHandler(_qt_message)
    logging.getLogger(LOGGER).info("ttu-view started")
    return log_path


def error(message: str) -> None:
    logging.getLogger(LOGGER).error(message)
