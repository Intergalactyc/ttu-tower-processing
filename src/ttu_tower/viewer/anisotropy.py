"""Reynolds-stress anisotropy on the barycentric map (Banerjee et al. 2007):
map coordinates from the stored eigenvalues, the class regions the pipeline
classifies by, and the states of a slot, its neighbours, and a whole period.
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower.constants import HEIGHTS

CASES = ("1c", "2c", "3c", "prolate", "oblate", "ellipse", "mixed")
# the old plotting code's fixed palette
CASE_COLORS = {"1c": "#1f3fd6", "2c": "#e6c700", "3c": "#d62728", "prolate": "#8e44ad", "oblate": "#ff7f0e",
               "ellipse": "#2ca02c", "mixed": "#8c8c8c"}
EDGE_K = 2.0 / 3.0  # the class-edge width secondary classifies with
SQRT3 = np.sqrt(3.0)
CORNERS = {"1C": (0.0, 0.0), "2C": (1.0, 0.0), "3C": (0.5, SQRT3 / 2)}


def barycentric_xy(l2, l3) -> tuple[np.ndarray, np.ndarray]:
    """Map position of eigenvalues l2, l3 (descending l1 ≥ l2 ≥ l3); 1C at the
    origin, 2C at (1, 0), 3C at the apex.
    """
    l2, l3 = np.asarray(l2, dtype=float), np.asarray(l3, dtype=float)
    c2 = 2 * (l2 - l3)
    c3 = 3 * l3 + 1
    return c2 + 0.5 * c3, (SQRT3 / 2) * c3


def plane_strain_line() -> tuple[tuple[float, float], tuple[float, float]]:
    """From the 3C corner to the 1C-2C edge at c1 = 1/3, c2 = 2/3."""
    return CORNERS["3C"], (2.0 / 3.0, 0.0)


def region_triangles(k: float = EDGE_K) -> dict[str, list]:
    """Each named class's area as triangles; what's left of the map is "mixed"."""
    v_1c, v_2c, v_3c = CORNERS["1C"], CORNERS["2C"], CORNERS["3C"]
    v_1c_2c_l, v_1c_2c_r = (0.5 * k, 0.0), (1 - 0.5 * k, 0.0)
    v_1c_3c_b, v_1c_3c_t = (0.25 * k, 0.25 * k * SQRT3), (0.5 - 0.25 * k, SQRT3 * (0.5 - 0.25 * k))
    v_2c_3c_b, v_2c_3c_t = (1 - 0.25 * k, 0.25 * k * SQRT3), (0.5 + 0.25 * k, SQRT3 * (0.5 - 0.25 * k))
    tip_1c, tip_2c = (0.5 * k, k * SQRT3 / 6), (1 - 0.5 * k, k * SQRT3 / 6)
    tip_3c = (0.5, SQRT3 * (0.5 - k / 3))
    return {
        "1c": [(v_1c, v_1c_3c_b, v_1c_2c_l), (v_1c_3c_b, tip_1c, v_1c_2c_l)],
        "2c": [(v_2c, v_1c_2c_r, v_2c_3c_b), (v_1c_2c_r, v_2c_3c_b, tip_2c)],
        "3c": [(v_2c_3c_t, v_1c_3c_t, v_3c), (v_2c_3c_t, v_1c_3c_t, tip_3c)],
        "prolate": [(v_1c_3c_b, v_1c_3c_t, tip_3c), (v_1c_3c_b, tip_3c, tip_1c)],
        "oblate": [(v_2c_3c_b, v_2c_3c_t, tip_3c), (v_2c_3c_b, tip_3c, tip_2c)],
        "ellipse": [(v_1c_2c_l, v_1c_2c_r, tip_2c), (v_1c_2c_l, tip_2c, tip_1c)],
    }


def _in_triangle(x, y, tri) -> np.ndarray:
    (x1, y1), (x2, y2), (x3, y3) = tri
    c1 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    c2 = (x3 - x2) * (y - y2) - (y3 - y2) * (x - x2)
    c3 = (x1 - x3) * (y - y3) - (y1 - y3) * (x - x3)
    return ((c1 <= 0) & (c2 <= 0) & (c3 <= 0)) | ((c1 >= 0) & (c2 >= 0) & (c3 >= 0))


def classify(x, y, k: float = EDGE_K) -> np.ndarray:
    """The class of each map position, first matching region winning (as secondary does)."""
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    out = np.full(x.shape, None, dtype=object)
    for name, tris in region_triangles(k).items():
        inside = np.zeros(x.shape, dtype=bool)
        for tri in tris:
            inside |= _in_triangle(x, y, tri)
        out[(out == None) & inside] = name  # noqa: E711
    out[(out == None) & np.isfinite(x) & np.isfinite(y)] = "mixed"  # noqa: E711
    return out


_EIGEN = ("aniso_l1", "aniso_l2", "aniso_l3", "aniso_beta", "aniso_phi")


def slot_states(across, variant: str) -> pd.DataFrame:
    """Per boom: the eigenvalues, angles, map position and class, from the
    final values, or (flagged `filtered`) from before filtering where
    tertiary removed them.
    """
    variant = "mrd" if variant not in ("mrd", "naive", "mrd_unexcised") else variant
    booms = list(across.booms)
    out = pd.DataFrame({"boom": booms, "height": [HEIGHTS[b] for b in booms]})
    final, pre = across.table("boom_final"), across.table("boom_stats")
    filtered = np.zeros(len(booms), dtype=bool)
    for var in _EIGEN:
        f = final[(final["variant"] == variant) & (final["variable"] == var)].set_index("boom")["value"]
        p = pre[(pre["variant"] == variant) & (pre["variable"] == var)].set_index("boom")["value"]
        fv = out["boom"].map(f).astype(float).to_numpy()
        pv = out["boom"].map(p).astype(float).to_numpy()
        out[var] = np.where(np.isfinite(fv), fv, pv)
        filtered |= ~np.isfinite(fv) & np.isfinite(pv)
    out["filtered"] = filtered
    out["x"], out["y"] = barycentric_xy(out["aniso_l2"], out["aniso_l3"])
    out["state"] = classify(out["x"], out["y"])
    return out


def boom_track(index, boom: int, variant: str, slot_lo: int, slot_hi: int) -> pd.DataFrame:
    """slot, x, y of one boom's final states over [slot_lo, slot_hi]."""
    f = pa_dataset.field
    df = index.read("boom_final", half_hours=range(slot_lo // 3, slot_hi // 3 + 1),
                    columns=["slot", "variable", "value"],
                    filter=(f("boom") == boom) & (f("variant") == variant) & (f("slot") >= slot_lo)
                    & (f("slot") <= slot_hi) & f("variable").isin(["aniso_l2", "aniso_l3"]))
    if df.empty:
        return pd.DataFrame(columns=["slot", "x", "y"])
    wide = df.astype({"variable": object}).pivot_table(index="slot", columns="variable", values="value", aggfunc="first")
    wide = wide.reindex(columns=["aniso_l2", "aniso_l3"])
    x, y = barycentric_xy(wide["aniso_l2"], wide["aniso_l3"])
    return pd.DataFrame({"slot": wide.index.to_numpy(), "x": x, "y": y})


def class_image(run, index, variant: str, booms) -> tuple[np.ndarray, np.ndarray]:
    """(slots, codes): codes[i, j] indexes CASES for booms[i] at slots[j]
    (-1 where there's no class: no data, or filtered).
    """
    booms = tuple(booms)
    return run.cache("timeline", 32).get(("aniso_classes", variant, booms),
                                         lambda: _class_image(run, index, variant, booms))


def _class_image(run, index, variant: str, booms: tuple) -> tuple[np.ndarray, np.ndarray]:
    slot_a, slot_b = run.period
    slots = np.arange(slot_a, slot_b, dtype=np.int64)
    image = np.full((len(booms), slots.size), -1, dtype=np.int8)
    if "tertiary" not in run.stages:
        return slots, image
    f = pa_dataset.field
    df = index.read("boom_labels_final", columns=["slot", "boom", "value"],
                    filter=(f("label") == "aniso_class") & (f("variant") == variant) & f("boom").isin(list(booms)))
    if df.empty:
        return slots, image
    codes = pd.Categorical(df["value"].astype(object), categories=list(CASES)).codes
    col = df["slot"].to_numpy(dtype=np.int64) - slot_a
    row = df["boom"].map({b: i for i, b in enumerate(booms)}).to_numpy()
    ok = (col >= 0) & (col < slots.size) & (codes >= 0)
    image[row[ok].astype(np.int64), col[ok]] = codes[ok]
    return slots, image
