import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.viewer import anisotropy, compare, composite, diurnal, figures
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.run import RunHandle
from ttu_tower.viewer.timeaxis import slot_to_unix, unix_to_slot
from ttu_tower.viewer.timeline import Curve

from viewer_fixtures import FULL_BOOMS


def test_composite_is_the_median_and_quartiles_per_scale():
    scales = np.array([1.0, 2.0, 4.0])
    values = np.array([[1.0, 2.0, 1.0], [3.0, 6.0, 3.0], [5.0, 10.0, np.nan], [7.0, 14.0, 7.0]])
    slots = np.arange(4)
    s = composite.summarize(slots, scales, values, np.ones(4, dtype=bool))
    assert list(s["mid"]) == [4.0, 8.0, 3.0] and list(s["n"]) == [4, 4, 3]
    assert s["lo"].iloc[0] == pytest.approx(np.percentile([1, 3, 5, 7], 25))
    half = composite.summarize(slots, scales, values, slots < 2)
    assert list(half["mid"]) == [2.0, 4.0, 2.0]
    normalized = composite.summarize(slots, scales, values[[0, 1, 3]], np.ones(3, dtype=bool), normalize=True)
    assert normalized["mid"].to_numpy() == pytest.approx([0.25, 0.5, 0.25])  # every row is 1:2:1


def _hourly_curve(tz: str = "UTC", days: int = 60) -> Curve:
    """One value per slot: the local hour of day."""
    start = int(unix_to_slot(pd.Timestamp("2014-01-01", tz="UTC").timestamp()))
    slots = np.arange(start, start + days * 144, dtype=np.int64)
    hours = pd.to_datetime(slot_to_unix(slots), unit="s", utc=True).tz_convert(tz).hour
    return Curve(slots=slots, y=np.asarray(hours, dtype=float))


def test_diurnal_cells_hold_their_hour_and_month():
    curve = _hourly_curve("Etc/GMT+6")
    values, counts = diurnal.grid(curve, "Etc/GMT+6", None, "median")
    jan, feb = values[:, 0], values[:, 1]
    assert np.array_equal(jan, np.arange(24)) and np.isfinite(feb).all() and np.isnan(values[:, 5]).all()
    assert counts.sum() == curve.slots.size
    assert counts[:, 11].sum() == 36 and counts[18:, 11].all()  # UTC midnight on Jan 1 is 18:00 on Dec 31 here
    share, _ = diurnal.grid(curve, "Etc/GMT+6", None, category=3)
    assert share[3, 0] == 1.0 and share[4, 0] == 0.0
    cell = diurnal.cell_slots(curve, "Etc/GMT+6", None, 7, 1)
    assert cell.size == counts[7, 0] and (curve.y[np.isin(curve.slots, cell)] == 7).all()


def test_hexbin_bins_like_matplotlib():
    from matplotlib.figure import Figure
    rng = np.random.default_rng(0)
    x, y = rng.random(3000), rng.random(3000) * np.sqrt(3) / 2
    cx, cy, counts, hexagon = anisotropy.hexbin(x, y)
    reference = Figure().subplots().hexbin(x, y, gridsize=anisotropy.HEX_GRIDSIZE, extent=anisotropy.HEX_EXTENT)
    assert np.allclose(np.column_stack([cx, cy]), reference.get_offsets())
    assert np.array_equal(counts, reference.get_array()) and counts.sum() == x.size and hexagon.shape == (6, 2)


def test_diurnal_directions_average_as_vectors():
    slots = np.arange(1000, 1006, dtype=np.int64)
    curve = Curve(slots=slots, y=np.array([350.0, 10.0, 350.0, 10.0, 350.0, 10.0]))
    values, _ = diurnal.grid(curve, "UTC", None, "mean", circular=True)
    finite = values[np.isfinite(values)]
    assert finite.size and np.allclose(np.minimum(finite, 360 - finite), 0.0, atol=1e-9)


def test_comparing_runs_pairs_slots_and_wraps_directions():
    a = Curve(slots=np.arange(5, dtype=np.int64), y=np.array([1.0, 2.0, np.nan, 4.0, 5.0]))
    b = Curve(slots=np.arange(1, 7, dtype=np.int64), y=np.arange(1, 7) + 2.0)  # a + 1 where both exist
    stats = compare.member_stats(a, b)
    assert stats["N both"] == 3 and stats["bias (other − this)"] == pytest.approx(1.0)
    assert stats["RMSE"] == pytest.approx(1.0) and stats["identical %"] == 0.0 and stats["r"] == pytest.approx(1.0)
    wa = Curve(slots=np.arange(2, dtype=np.int64), y=np.array([355.0, 10.0]))
    wb = Curve(slots=np.arange(2, dtype=np.int64), y=np.array([5.0, 350.0]))
    assert list(compare.differences(wa.y, wb.y, True)) == [10.0, -20.0]


def _spec(kind):
    x = np.linspace(1, 10, 20)
    t = pd.date_range("2014-01-01", periods=20, freq="10min")
    table = np.ones((8, 3))
    return {
        "timelines": {"panels": [{"title": "ws", "ylabel": "m/s", "curves": [{"label": "b1", "color": "#123456",
                                                                             "t": t, "y": x}],
                                  "others": [{"label": "b1", "color": "#123456", "t": t, "y": x + 1}]}],
                      "night": [(t[2], t[5])], "xlim": (t[0], t[-1])},
        "distributions": {"panels": [{"title": "ws", "xlabel": "m/s", "ylabel": "density",
                                      "hists": [{"label": "b1", "color": "#123456", "edges": np.arange(4.0),
                                                 "density": np.ones(3) / 3}],
                                      "fits": [{"label": "b1", "color": "#123456", "x": x, "pdf": x / 10}]}]},
        "scatter": {"xlabel": "x", "ylabel": "y", "groups": [{"label": "stable", "color": "#2ca02c", "x": x, "y": x}],
                    "lines": [{"label": "y = x", "color": "#555555", "x": x, "y": x, "dashed": True}],
                    "selected": {"x": x[:3], "y": x[:3]}},
        "profiles": {"panels": [{"title": "ws", "xlabel": "m/s", "groups": [
            {"label": "neutral", "color": "#9b5445", "x": x[:5], "z": x[:5] * 20, "spread": x[:5] / 10,
             "fit": (x[:5], x[:5] * 20)}]}]},
        "composite": {"title": "uw", "ylabel": "u'w'", "refs": (60.0,), "lines": [
            {"label": "10 m", "color": "#123456", "scale": x, "mid": -x, "lo": -x - 1, "hi": -x + 1}]},
        "diurnal": {"values": np.random.default_rng(0).random((24, 12)), "title": "ws", "months": diurnal.MONTHS},
        "anisotropy": {"regions": {}, "region_colors": anisotropy.CASE_COLORS, "heatmap": True,
                       "corners": anisotropy.CORNERS, "strain": anisotropy.plane_strain_line(), "title": "b5",
                       "panels": [{"title": f"{n} (3)", "color": "#1f4e9c", "x": [0.5, 0.4, 0.6], "y": [0.3, 0.2, 0.4]}
                                  for n in ("stable", "neutral", "unstable", "strongly stable")]},
        "windrose": {"roses": [{"title": "b7", "table": table, "bearings": [105.0]}], "speed_labels": ["a", "b", "c"],
                     "outer": 4.0},
        "compare": {"x": x, "y": x, "xlabel": "this", "ylabel": "other", "title": "b1"},
    }[kind]


@pytest.mark.parametrize("kind", figures.KINDS)
def test_every_figure_kind_renders_and_saves(kind, tmp_path):
    fig = figures.render(kind, _spec(kind), (5, 3.5))
    for ext in (".png", ".pdf", ".svg"):
        path = tmp_path / f"{kind}{ext}"
        figures.save(fig, path, dpi=60)
        assert path.stat().st_size > 1000


@pytest.mark.slow
def test_spectra_matrix_and_boom_map_match_the_stored_rows(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    boom = FULL_BOOMS[0]
    slots, scales, values = composite.spectra_matrix(run, index, boom, "mrd", "uw")
    f = pa_dataset.field
    rows = index.read("mrd", booms=[boom], filter=(f("variant") == "mrd") & (f("spectrum") == "uw"))
    assert values.shape == (rows["slot"].nunique(), rows["scale_s"].nunique())
    r = rows.iloc[len(rows) // 2]
    assert values[np.searchsorted(slots, r["slot"]), np.searchsorted(scales, r["scale_s"])] == r["value"]
    map_slots, x, y = anisotropy.boom_map(run, index, boom, "mrd")
    track = anisotropy.boom_track(index, boom, "mrd", int(map_slots.min()), int(map_slots.max()))
    track = track[np.isfinite(track["x"])]
    assert list(track["slot"]) == list(map_slots) and np.allclose(track["x"], x) and np.allclose(track["y"], y)
