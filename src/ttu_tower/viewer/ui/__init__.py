"""The Qt GUI. Importing this package pins pyqtgraph to PySide6, since PyQt5
may also be installed alongside it.
"""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import PySide6  # noqa: E402,F401  (imported before pyqtgraph, which then binds to it)
