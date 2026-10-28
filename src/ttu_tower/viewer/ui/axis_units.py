"""Axis labels keep the computed units. pyqtgraph would otherwise swap in an SI
prefix picked from the visible range (millidegrees for a range under one
degree, mm²/s² for small variances). Instead, where values are very small or
very large, the ticks show the mantissa and the label a power of ten, e.g.
"(×10⁻⁴ m²/s²)".
"""
import numpy as np
import pyqtgraph as pg

from ttu_tower.viewer.ui.axes import Log10Axis

SMALL, LARGE = 1e-2, 1e4  # a largest |value| inside [SMALL, LARGE) is shown as it is
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
_installed = False


def power_of_ten(lo: float, hi: float) -> int:
    """The exponent to factor out of an axis spanning [lo, hi]; 0 for none."""
    m = max(abs(lo), abs(hi))
    if not np.isfinite(m) or m == 0 or SMALL <= m < LARGE:
        return 0
    return int(np.floor(np.log10(m)))


def factor_text(power: int) -> str:
    return f"×10{str(power).translate(_SUPERSCRIPT)}" if power else ""


def _scalable(axis: pg.AxisItem) -> bool:
    """Plain value axes with a label to carry the factor: not times, decades or fixed ticks."""
    decades = isinstance(axis, Log10Axis) and getattr(axis, "log10", True)  # a SwitchLogAxis may be plain
    return (axis.label.isVisible() and not axis.logMode and axis._tickLevels is None
            and not isinstance(axis, pg.DateAxisItem) and not decades)


def _update(self) -> None:
    power = power_of_ten(self.range[0] * self.scale, self.range[1] * self.scale) if _scalable(self) else 0
    self.unit_power = power
    self.autoSIPrefixScale = 10.0 ** -power
    self.labelUnitPrefix = ""
    self._updateLabel()


def _label_string(self) -> str:
    inner = " ".join(p for p in (factor_text(getattr(self, "unit_power", 0)), self.labelUnits) if p)
    text = f"{self.labelText} ({inner})" if inner else self.labelText
    style = ";".join(f"{k}: {v}" for k, v in self.labelStyle.items())
    return f"<span style='{style}'>{text}</span>"


def install() -> None:
    """Patch every pyqtgraph axis, once."""
    global _installed
    if _installed:
        return
    _installed = True
    enable, set_ticks = pg.AxisItem.enableAutoSIPrefix, pg.AxisItem.setTicks

    def enable_always(self, enable_=True):
        enable(self, True)  # the range must be watched for the power of ten whatever a caller asks

    def set_ticks_and_update(self, ticks):
        set_ticks(self, ticks)
        self.updateAutoSIPrefix()  # fixed ticks lose the factor, automatic ones may regain it

    pg.AxisItem.updateAutoSIPrefix = _update
    pg.AxisItem.labelString = _label_string
    pg.AxisItem.enableAutoSIPrefix = enable_always
    pg.AxisItem.setTicks = set_ticks_and_update
