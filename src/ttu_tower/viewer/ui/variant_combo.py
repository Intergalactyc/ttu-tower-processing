"""Variant names for menus: the run's own (none, mrd, naive, mrd_unexcised)
and the viewer's τ variants (fixed rungs, τ schemes).
"""
from PySide6.QtCore import Qt

from ttu_tower.viewer import tauvariants

LABELS = {"none": "slot (no τ)", "mrd": "mrd (selected τ)", "naive": "naive (10 min)",
          "mrd_unexcised": "mrd unexcised"}
TIPS = {
    "none": "Over the whole 10-min slot on the final usable mask (means, slow sensors and what's "
            "built from them); doesn't depend on τ.",
    "mrd": "Mean over the valid τ-blocks tiling the slot, at the selected τ (τ = 20 min: the "
           "window from 5 min before the slot to 15 min after).",
    "naive": "The 600-s rung of mrd: the slot as a single block.",
    "mrd_unexcised": "As mrd, on the series with nothing removed (gaps ≤ 10 min interpolated).",
}


def label(v: str) -> str:
    return LABELS.get(v) or tauvariants.label(v)


def tip(v: str) -> str:
    return TIPS.get(v) or tauvariants.tip(v)


def fill(combo, items, extra=(), fallback: str | None = None) -> bool:
    """Set `combo` to `items` ((key, text) pairs) then the viewer variants
    `extra`, keeping its choice where it can (else `fallback`, else the first).
    Signals stay blocked; returns whether the choice changed.
    """
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for key, text in list(items) + [(k, label(k)) for k in extra]:
        combo.addItem(text, key)
        combo.setItemData(combo.count() - 1, tip(key), Qt.ItemDataRole.ToolTipRole)
    i = combo.findData(current)
    if i < 0 and fallback is not None:
        i = combo.findData(fallback)
    combo.setCurrentIndex(max(i, 0))
    combo.blockSignals(False)
    return combo.currentData() != current


def refill(combo, items, extra=()) -> None:
    """`fill`, telling the combo's listeners if the choice changed (a variant it showed was deleted)."""
    if fill(combo, items, extra) and combo.count():
        combo.currentIndexChanged.emit(combo.currentIndex())
