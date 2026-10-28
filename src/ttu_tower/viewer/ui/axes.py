"""An axis for data already plotted as log10(x): ticks at decades (labelled
with the value itself) and minor ticks at 2-9 × each decade.
"""
import numpy as np
import pyqtgraph as pg


class Log10Axis(pg.AxisItem):
    def tickValues(self, minVal, maxVal, size):
        lo, hi = int(np.floor(minVal)), int(np.ceil(maxVal))
        major = [float(d) for d in range(lo, hi + 1) if minVal <= d <= maxVal]
        minor = [d + np.log10(m) for d in range(lo, hi + 1) for m in range(2, 10) if minVal <= d + np.log10(m) <= maxVal]
        if len(major) < 2:  # a narrow range: label the minor ticks too
            return [(1.0, sorted(major + minor))]
        return [(1.0, major), (0.1, minor)]

    def tickStrings(self, values, scale, spacing):
        if spacing < 1.0:  # minor ticks, when there are decades to label instead
            return ["" for _ in values]
        return [f"{float(f'{10 ** v:.3g}'):g}" for v in values]


class SwitchLogAxis(Log10Axis):
    """A plain value axis, or (with `log10` on) a Log10Axis for data plotted as
    log10(x): one axis that follows a "log" check box.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log10 = False

    def set_log10(self, on: bool) -> None:
        if on != self.log10:
            self.log10 = on
            self.picture = None
            self.updateAutoSIPrefix()  # a power-of-ten factor applies to the plain axis only
            self.update()

    def tickValues(self, minVal, maxVal, size):
        return super().tickValues(minVal, maxVal, size) if self.log10 else pg.AxisItem.tickValues(
            self, minVal, maxVal, size)

    def tickStrings(self, values, scale, spacing):
        return super().tickStrings(values, scale, spacing) if self.log10 else pg.AxisItem.tickStrings(
            self, values, scale, spacing)
