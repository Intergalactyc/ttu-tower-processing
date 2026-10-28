"""Saving and restoring a view's controls by name: each view lists the
attributes that make up its state (`STATE_FIELDS`), and these read and set
them by widget type. A path may step into a list or dict ("source.0",
"sides.x"); a dict of widgets is saved as a dict. Values are plain JSON.
"""
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QTabWidget

from ttu_tower.viewer.ui.timeline_plot import LegendBar


def _resolve(obj, path: str):
    for part in path.split("."):
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            obj = obj[int(part)] if part.isdigit() and int(part) < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    return obj


def _plain(v):
    """`v` as JSON (tuples as lists), or None where it can't be."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        items = [_plain(x) for x in v]
        return None if any(x is None and y is not None for x, y in zip(items, v)) else items
    try:
        return v.item()  # a numpy scalar
    except AttributeError:
        return None


def _member_key(m):
    return list(m) if isinstance(m, tuple) else m


def get(widget):
    if isinstance(widget, dict):
        return {k: get(w) for k, w in widget.items()}
    if hasattr(widget, "get_state"):
        return widget.get_state()
    if isinstance(widget, QComboBox):
        return {"data": _plain(widget.currentData()), "text": widget.currentText()}
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, LegendBar):
        return [_member_key(m) for m in widget.buttons if widget.buttons[m].isChecked()]
    if isinstance(widget, QTabWidget):
        return widget.tabText(widget.currentIndex())
    return None


def put_combo(widget: QComboBox, value, free_text: bool = True) -> bool:
    """`free_text`: an editable combo takes text that none of its items has."""
    data, text = value.get("data"), value.get("text", "")
    i = widget.findData(tuple(data) if isinstance(data, list) else data) if data is not None else -1
    if i < 0:
        i = widget.findText(text)
    if i >= 0:
        widget.setCurrentIndex(i)
        return True
    if free_text and widget.isEditable() and text:
        widget.setEditText(text)
        return True
    return False


class LateCombo(QComboBox):
    """A combo whose items arrive after a view is reopened (they load in the
    background): a saved choice it can't take yet waits for them.
    """

    pending = None

    def put_state(self, value) -> bool:
        self.pending = None if put_combo(self, value) else value
        return True

    def take_pending(self) -> None:
        if self.pending is not None and put_combo(self, self.pending):
            self.pending = None


def put(widget, value) -> bool:
    """Set `widget` to a saved value; False where it can't take it now."""
    if isinstance(widget, dict):
        ok = True
        for k, v in (value or {}).items():
            ok &= k in widget and put(widget[k], v)
        return ok
    if hasattr(widget, "put_state"):
        return widget.put_state(value)
    if isinstance(widget, QComboBox):
        return put_combo(widget, value)
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
        return True
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(value)
        return True
    if isinstance(widget, QLineEdit):
        widget.setText(str(value))
        return True
    if isinstance(widget, LegendBar):
        shown = [tuple(v) if isinstance(v, list) else v for v in value]
        if not any(m in shown for m in widget.buttons):
            return False
        for m, button in widget.buttons.items():
            button.blockSignals(True)
            button.setChecked(m in shown)
            button.blockSignals(False)
        widget.toggled.emit()
        return True
    if isinstance(widget, QTabWidget):
        for i in range(widget.count()):
            if widget.tabText(i) == value:
                widget.setCurrentIndex(i)
                return True
        return False
    return False


def save(obj, fields) -> dict:
    """The fields' values; a view's `state_skip()` names any that show derived values now."""
    skip = obj.state_skip() if hasattr(obj, "state_skip") else set()
    out = {}
    for path in fields:
        if path in skip:
            continue
        widget = _resolve(obj, path)
        if widget is not None:
            out[path] = get(widget)
    return out


def restore(obj, state: dict, fields) -> list[str]:
    """Set the fields in order (a combo's items can depend on an earlier one);
    the paths that couldn't be set.
    """
    missed = []
    for path in fields:
        if path not in state:
            continue
        widget = _resolve(obj, path)
        if widget is None or not put(widget, state[path]):
            missed.append(path)
    return missed


def pack_slots(slots) -> list[list[int]]:
    """Sorted slots as [first, last] runs, to keep a large selection small."""
    runs = []
    for k in sorted(int(s) for s in slots):
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    return runs


def unpack_slots(runs) -> list[int]:
    return [k for a, b in runs for k in range(int(a), int(b) + 1)]
