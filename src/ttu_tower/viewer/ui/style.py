"""Colors: booms by height (viridis, as in the ttu-windprofiles plots), flag
tests, categories.
"""
import pyqtgraph as pg

from ttu_tower.constants import BOOMS, HEIGHTS

_VIRIDIS = pg.colormap.get("viridis")
_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf"]

TEST_COLORS = {
    "bounds": "#d62728", "spike": "#ff7f0e", "excursion": "#bcbd22", "unchecked": "#7f7f7f",
    "resolution": "#9467bd", "dropouts": "#8c564b", "skew": "#e377c2", "kurt": "#c51b7d",
    "direction": "#17becf", "bounce": "#1f77b4", "coupled": "#ffbb78", "filled": "#2ca02c",
}
AVAILABILITY_COLORS = {-1: (255, 255, 255, 0), 0: (200, 200, 200, 255), 1: (240, 170, 60, 255),
                       2: (80, 160, 90, 255)}
NIGHT_BRUSH = (40, 40, 90, 28)
# the ttu-windprofiles figure palette, so the band reads like the paper's plots
STABILITY_COLORS = {"strongly unstable": "#d62728", "unstable": "#ff7f0e", "neutral": "#9b5445",
                    "stable": "#2ca02c", "strongly stable": "#3b50d6"}


def boom_color(boom: int) -> pg.QtGui.QColor:
    order = sorted(BOOMS, key=lambda b: HEIGHTS[b])
    i = order.index(boom)
    return _VIRIDIS.map(0.92 * i / (len(order) - 1), mode="qcolor")


def member_color(member, i: int = 0) -> pg.QtGui.QColor:
    """A curve's color: boom by height; boom pairs and slot-level curves by position."""
    if isinstance(member, int):
        return boom_color(member)
    return pg.mkColor(_TAB10[i % len(_TAB10)])


def category_color(i: int) -> pg.QtGui.QColor:
    return pg.mkColor(_TAB10[i % len(_TAB10)])


def stability_color(name: str, i: int) -> pg.QtGui.QColor:
    """A stability class's color by name; classes named otherwise in a config fall back by position."""
    return pg.mkColor(STABILITY_COLORS[name]) if name in STABILITY_COLORS else category_color(i + 4)


def member_short(member) -> str:
    """A legend button's text: the height goes in its tooltip."""
    if member is None:
        return ""
    if isinstance(member, tuple):
        return f"b{member[0]}–b{member[1]}"
    return f"b{member}"


def member_label(member) -> str:
    if member is None:
        return ""
    if isinstance(member, tuple):
        return f"b{member[0]}–b{member[1]}"
    return f"b{member} ({HEIGHTS[member]:g} m)"
