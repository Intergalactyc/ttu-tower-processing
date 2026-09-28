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
