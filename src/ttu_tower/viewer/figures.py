"""Publication figures of what the viewer shows, drawn with matplotlib (its
object API with the Agg canvas, never pyplot, so it can't disturb Qt) in the
style of ttu-windprofiles' figures: tab colours, stability colours and markers,
viridis by height.

Each view hands over a plain description (arrays, labels, colours) and
`render(kind, spec, size)` draws it; `save` writes PNG, PDF or SVG by extension.
"""
from pathlib import Path

import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

# ttu-windprofiles' style.MARKERS
STABILITY_MARKERS = {"strongly unstable": "v", "unstable": "^", "neutral": "o", "stable": "D",
                     "strongly stable": "s"}
KINDS = ("timelines", "distributions", "scatter", "profiles", "composite", "diurnal", "anisotropy", "windrose",
         "compare")


def new_figure(size: tuple[float, float], nrows: int = 1, ncols: int = 1, **kwargs):
    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False, **kwargs)
    return fig, axes


def save(fig: Figure, path, dpi: int = 300) -> None:
    fig.savefig(Path(path), dpi=dpi)


def render(kind: str, spec: dict, size: tuple[float, float]) -> Figure:
    if kind not in KINDS:
        raise ValueError(f"no figure for {kind}")
    return globals()[f"_{kind}"](spec, size)


def _legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, frameon=False, **kwargs)


# --- the kinds ---------------------------------------------------------------------------------

def _timelines(spec, size):
    """spec: panels [{title, ylabel, log_y, categories, curves [{label, color, t, y, points}],
    others [{label, color, t, y}] (another run, dashed)}], night [(t0, t1)], xlim (t0, t1)."""
    panels = spec["panels"]
    fig, axes = new_figure(size, len(panels), 1, sharex=True)
    for ax, p in zip(axes[:, 0], panels):
        for t0, t1 in spec.get("night", []):
            ax.axvspan(t0, t1, color="#28285a", alpha=0.08, lw=0)
        for c in p["curves"]:
            if p.get("categories") or c.get("points"):
                ax.plot(c["t"], c["y"], ls="none", marker="o", ms=1.5, color=c["color"], label=c["label"])
            else:
                ax.plot(c["t"], c["y"], lw=0.8, color=c["color"], label=c["label"])
        for c in p.get("others", []):
            ax.plot(c["t"], c["y"], lw=0.8, ls="--", color=c["color"], alpha=0.8)
        if p.get("log_y"):
            ax.set_yscale("log")
        if p.get("categories"):
            ax.set_yticks(range(len(p["categories"])), p["categories"])
        ax.set_title(p["title"], fontsize=9, loc="left")
        ax.set_ylabel(p["ylabel"])
        _legend(ax, ncols=min(len(p["curves"]), 5), loc="upper right")
    if spec.get("xlim"):
        axes[0, 0].set_xlim(*spec["xlim"])
    axes[-1, 0].set_xlabel(spec.get("xlabel", "time"))
    fig.autofmt_xdate()
    return fig


def _distributions(spec, size):
    """spec: panels [{title, xlabel, ylabel, bars (categorical: [{label, color, x, height, width}]) or
    hists [{label, color, edges, density}], fits [{label, color, x, pdf}], xticks [(x, label)]}]."""
    panels = spec["panels"]
    fig, axes = new_figure(size, len(panels), 1)
    for ax, p in zip(axes[:, 0], panels):
        for h in p.get("hists", []):
            if h.get("filled"):
                ax.stairs(h["density"], h["edges"], color=h["color"], fill=True, alpha=0.35, label=h["label"])
            else:
                ax.stairs(h["density"], h["edges"], color=h["color"], lw=1.2, label=h["label"])
        for b in p.get("bars", []):
            ax.bar(b["x"], b["height"], width=b["width"], color=b["color"], label=b["label"])
        for f in p.get("fits", []):
            ax.plot(f["x"], f["pdf"], ls="-." if f.get("dashdot") else "--", lw=1.2, color=f["color"])
        if p.get("log_x"):
            ax.set_xscale("log")
        if p.get("xticks"):
            ax.set_xticks([x for x, _ in p["xticks"]], [t for _, t in p["xticks"]])
        ax.set_title(p["title"], fontsize=9, loc="left")
        ax.set_xlabel(p["xlabel"])
        ax.set_ylabel(p["ylabel"])
        _legend(ax)
    return fig


def _scatter(spec, size):
    """spec: xlabel, ylabel, groups [{label, color, x, y}], lines [{label, color, x, y, dashed}],
    selected {x, y} | None, note, log_x, log_y, xlim, ylim."""
    fig, axes = new_figure(size)
    ax = axes[0, 0]
    n = sum(len(g["x"]) for g in spec["groups"])
    for g in spec["groups"]:
        ax.scatter(g["x"], g["y"], s=4 if n > 5000 else 8, color=g["color"], alpha=0.6, lw=0,
                   marker=STABILITY_MARKERS.get(g["label"], "o"), label=g["label"] or None)
    sel = spec.get("selected")
    if sel is not None and len(sel["x"]):
        ax.scatter(sel["x"], sel["y"], s=30, facecolors="none", edgecolors="#e6007e", lw=0.8, label="selected")
    for line in spec.get("lines", []):
        ax.plot(line["x"], line["y"], color=line["color"], lw=1.4, ls="--" if line.get("dashed") else "-",
                label=line["label"])
    if spec.get("log_x"):
        ax.set_xscale("log")
    if spec.get("log_y"):
        ax.set_yscale("log")
    if spec.get("xlim"):
        ax.set_xlim(*spec["xlim"])
    if spec.get("ylim"):
        ax.set_ylim(*spec["ylim"])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    if spec.get("note"):
        ax.set_title(spec["note"], fontsize=8, loc="left")
    _legend(ax)
    return fig


def _profiles(spec, size):
    """spec: panels [{title, xlabel, log_z, groups [{label, color, x, z, spread, fit (x, z) | None}]}]."""
    panels = spec["panels"]
    fig, axes = new_figure(size, 1, len(panels), sharey=True)
    for ax, p in zip(axes[0], panels):
        for g in p["groups"]:
            marker = STABILITY_MARKERS.get(g["label"], "o")
            if g.get("spread") is not None:
                ax.errorbar(g["x"], g["z"], xerr=g["spread"], fmt="none", ecolor=g["color"], elinewidth=0.8,
                            capsize=2, alpha=0.7)
            ax.plot(g["x"], g["z"], ls="-" if g.get("lines") else "none", marker=marker, ms=4, color=g["color"],
                    label=g["label"])
            if g.get("fit") is not None:
                ax.plot(g["fit"][0], g["fit"][1], ls="--", lw=1, color=g["color"])
        if p.get("log_z"):
            ax.set_yscale("log")
        ax.set_title(p["title"], fontsize=9, loc="left")
        ax.set_xlabel(p["xlabel"])
        _legend(ax)
    axes[0, 0].set_ylabel("height (m)")
    return fig


def _composite(spec, size):
    """spec: title, ylabel, log_y, lines [{label, color, scale, mid, lo, hi}], refs (s), bars."""
    fig, axes = new_figure(size)
    ax = axes[0, 0]
    for line in spec["lines"]:
        ax.plot(line["scale"], line["mid"], color=line["color"], marker="o", ms=3, lw=1.5, label=line["label"])
        if spec.get("bars", True):
            ax.errorbar(line["scale"], line["mid"], yerr=[line["mid"] - line["lo"], line["hi"] - line["mid"]],
                        fmt="none", ecolor=line["color"], elinewidth=0.8, capsize=2, alpha=0.6)
    if not spec.get("log_y"):
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    for v in spec.get("refs", ()):
        ax.axvline(v, color="k", ls="--", lw=0.8)
    # scales after drawing: errorbars added to an already-log axis corrupt its data limits (as in the paper's code)
    ax.set_xscale("log")
    if spec.get("log_y"):
        ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"], fontsize=9, loc="left")
    ax.legend(title="Height", fontsize=7, title_fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    return fig


def _diurnal(spec, size):
    """spec: values[24, 12], title, clabel, cyclic, months (labels)."""
    fig, axes = new_figure(size)
    ax = axes[0, 0]
    values = np.ma.masked_invalid(spec["values"])
    kwargs = {"vmin": 0, "vmax": 360} if spec.get("cyclic") else {}
    image = ax.pcolormesh(np.arange(13), np.arange(25), values, cmap="twilight" if spec.get("cyclic") else "viridis",
                          shading="flat", **kwargs)
    fig.colorbar(image, ax=ax, label=spec.get("clabel", ""))
    ax.set_xticks(np.arange(12) + 0.5, spec["months"])
    ax.set_yticks(range(0, 25, 3))
    ax.set_ylabel(spec.get("ylabel", "hour of day"))
    ax.set_title(spec["title"], fontsize=9, loc="left")
    return fig


_CORNER_LABELS = {"1C": "1-Component\n(Linear)", "2C": "2-Component\n(Planar)", "3C": "3-Component\n(Isotropic)"}


def _anisotropy(spec, size):
    """spec: panels [{title, color, x, y}], heatmap, log, gridsize, regions {name: [triangles]},
    region_colors, corners {name: (x, y)}, strain ((x, y), (x, y)), title.

    As the old plotting code's barycentric heatmaps: a viridis hexbin of counts clipped to the
    triangle with a colour bar per panel, one panel per stability class when split.
    """
    panels = spec["panels"]
    ncols = min(len(panels), 3)
    nrows = -(-len(panels) // ncols)
    fig, axes = new_figure(size, nrows, ncols)
    corners = spec["corners"]
    triangle = np.array([corners["1C"], corners["2C"], corners["3C"]])
    for k, (ax, p) in enumerate(zip(axes.flat, panels)):
        for name, tris in spec.get("regions", {}).items():
            for tri in tris:
                ax.add_patch(Polygon(tri, closed=True, color=spec["region_colors"][name], alpha=0.15, lw=0))
        if spec.get("heatmap", True):
            hb = ax.hexbin(p["x"], p["y"], gridsize=spec.get("gridsize", 40), cmap="viridis", edgecolors="face",
                           linewidths=0.2,
                           extent=(0.0, 1.0, 0.0, np.sqrt(3) / 2), bins="log" if spec.get("log") else None)
            hb.set_clip_path(Polygon(triangle, transform=ax.transData))
            fig.colorbar(hb, ax=ax, shrink=0.8, label="count per bin" if k % ncols == ncols - 1 else None)
        else:
            ax.scatter(p["x"], p["y"], s=3, color=p["color"], alpha=0.6, lw=0,
                       marker=STABILITY_MARKERS.get(p["title"].rsplit(" (", 1)[0], "o"))
        ax.plot(*np.vstack([triangle, triangle[:1]]).T, color="k", lw=1.5)
        (x0, y0), (x1, y1) = spec["strain"]
        # white dashes over a black stroke: visible across the whole viridis range and on white
        ax.plot([x0, x1], [y0, y1], color="white", ls="--", lw=1.2,
                path_effects=[patheffects.Stroke(linewidth=2.6, foreground="black"), patheffects.Normal()])
        for name, (x, y) in corners.items():
            ax.text(x, y + (0.05 if y > 0 else -0.05), _CORNER_LABELS[name] if k == 0 else name, ha="center",
                    va="bottom" if y > 0 else "top", fontsize=7)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.2, 1.0)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(p["title"], fontsize=9)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(spec["title"], fontsize=9)
    return fig


def _windrose(spec, size):
    """spec: roses [{title, table[sectors, bins] (%), bearings}], speed_labels, outer."""
    roses = spec["roses"]
    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    colors = [colormaps["viridis"](j / max(len(spec["speed_labels"]) - 1, 1))
              for j in range(len(spec["speed_labels"]))]
    for i, rose in enumerate(roses):
        ax = fig.add_subplot(1, len(roses), i + 1, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        table = rose["table"]
        if table is not None and table.sum():
            sectors = table.shape[0]
            theta = np.radians(np.arange(sectors) * 360.0 / sectors)
            width = np.radians(360.0 / sectors) * 0.9
            bottom = np.zeros(sectors)
            for j in range(table.shape[1]):
                ax.bar(theta, table[:, j], width=width, bottom=bottom, color=colors[j], lw=0,
                       label=spec["speed_labels"][j] if i == 0 else None)
                bottom += table[:, j]
            for b in rose.get("bearings", ()):
                ax.plot([np.radians(b)] * 2, [0, spec["outer"]], color="#d62728", ls="--", lw=0.8)
            ax.set_ylim(0, spec["outer"])
        ax.set_title(rose["title"], fontsize=8)
        ax.tick_params(labelsize=7)
    fig.legend(title="speed (m/s)", fontsize=7, title_fontsize=7, loc="outside right center", frameon=False)
    return fig


def _compare(spec, size):
    """spec: x, y, xlabel, ylabel, title."""
    fig, axes = new_figure(size)
    ax = axes[0, 0]
    ax.scatter(spec["x"], spec["y"], s=4, color="tab:blue", alpha=0.6, lw=0)
    if len(spec["x"]):
        lo, hi = float(min(np.min(spec["x"]), np.min(spec["y"]))), float(max(np.max(spec["x"]), np.max(spec["y"])))
        ax.plot([lo, hi], [lo, hi], color="0.3", ls="--", lw=0.8, label="y = x")
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"], fontsize=9, loc="left")
    _legend(ax)
    return fig
