"""Hodographs: the mean wind vector at each height as (u, v) = (east, north),
the "blows-toward" components. A slot's vector is its vector-mean speed and
direction (boom_final ws vector_mean, wd mean); over many slots the vectors
are averaged. The mesonet's 10 m wind (from its slot means) is a point apart.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower.constants import HEIGHTS
from ttu_tower.math.polar import bearing_to_vector, vector_to_bearing
from ttu_tower.viewer import timeline
from ttu_tower.viewer.windrose import _quantity

MESONET = "mesonet"
MESONET_HEIGHT = 10.0
_TOWER = (("ws", "vector_mean"), ("wd", "mean"))
_MESO = ("ws_meso_mean", "wd_meso_mean")


@dataclass
class UV:
    slots: np.ndarray
    u: np.ndarray
    v: np.ndarray

    def only(self, keep: np.ndarray) -> "UV":
        return UV(self.slots[keep], self.u[keep], self.v[keep])


def _uv(slots, ws, wd) -> UV:
    ok = np.isfinite(ws) & np.isfinite(wd)
    u, v = bearing_to_vector(ws[ok], wd[ok])
    return UV(np.asarray(slots)[ok], np.asarray(u, dtype=float), np.asarray(v, dtype=float))


def load(run, index, booms) -> dict:
    """Runs on a worker thread: {boom: UV, "mesonet": UV (if the run merged it)} over the whole period."""
    ws_q, wd_q = (_quantity("boom_final", v, s, "boom") for v, s in _TOWER)
    ws, wd = (timeline.load(run, index, q, "none", list(booms)) for q in (ws_q, wd_q))
    out = {}
    for b in booms:
        if b in ws.curves and b in wd.curves:
            slots, s, d = timeline.joined(ws.curves[b], wd.curves[b])
            out[b] = _uv(slots, s, d)
    meso = mesonet_uv(run, index)
    if meso is not None:
        out[MESONET] = meso
    return out


def mesonet_uv(run, index) -> UV | None:
    """The mesonet's slot-mean wind over the whole period (None if the run merged none)."""
    meso = [timeline.load(run, index, _quantity("slot_final", v, None, "slot"), "none").curves[None] for v in _MESO]
    if meso[0].slots.size == 0:
        return None
    slots, s, d = timeline.joined(*meso)
    return _uv(slots, s, d)


def from_profiles(slots: np.ndarray, booms, ws: np.ndarray, wd: np.ndarray, min_share: float = 0.5) -> dict:
    """{boom: UV} from (slots, booms) arrays of speed and FROM-bearing: the booms with wind in at least
    `min_share` of the slots, over the slots all of those have.
    """
    with np.errstate(invalid="ignore"):
        u, v = bearing_to_vector(ws, wd)
    have = np.isfinite(u) & np.isfinite(v)
    used = [j for j in range(len(booms)) if have[:, j].mean() >= min_share] if len(slots) else []
    every = have[:, used].all(axis=1) if used else np.zeros(len(slots), dtype=bool)
    return {booms[j]: UV(np.asarray(slots)[every], u[every, j], v[every, j]) for j in used}


def slot_window(index, k: int, booms, around: int = 0) -> dict:
    """{boom or "mesonet": UV} for slots k - around .. k + around (one slot's hodograph and its neighbours)."""
    f = pa_dataset.field
    lo, hi = k - around, k + around
    half_hours = list(range(lo // 3, hi // 3 + 1))
    flt = ((f("slot") >= lo) & (f("slot") <= hi) & (f("variant") == "none") & f("boom").isin(list(booms))
           & (((f("variable") == "ws") & (f("stat") == "vector_mean")) | ((f("variable") == "wd") & (f("stat") == "mean"))))
    rows = index.read("boom_final", half_hours=half_hours, columns=["slot", "boom", "variable", "value"], filter=flt)
    out = {}
    if not rows.empty:
        rows["variable"] = rows["variable"].astype(str)
        wide = rows.pivot_table(index=["boom", "slot"], columns="variable", values="value", aggfunc="first")
        for b in booms:
            if b in wide.index.get_level_values(0) and {"ws", "wd"} <= set(wide.columns):
                part = wide.loc[b]
                out[b] = _uv(part.index.to_numpy(dtype=np.int64), part["ws"].to_numpy(float), part["wd"].to_numpy(float))
    try:
        meso = index.read("slot_final", half_hours=half_hours, columns=["slot", "variable", "value"],
                          filter=(f("slot") >= lo) & (f("slot") <= hi) & f("variable").isin(list(_MESO)))
    except (FileNotFoundError, OSError):
        meso = pd.DataFrame()
    if not meso.empty:
        meso["variable"] = meso["variable"].astype(str)
        wide = meso.pivot_table(index="slot", columns="variable", values="value", aggfunc="first")
        if set(_MESO) <= set(wide.columns):
            out[MESONET] = _uv(wide.index.to_numpy(dtype=np.int64), wide[_MESO[0]].to_numpy(float),
                               wide[_MESO[1]].to_numpy(float))
    return out


def height(member) -> float:
    return MESONET_HEIGHT if member == MESONET else HEIGHTS[member]


def common_slots(uvs: dict, members) -> np.ndarray:
    """The slots every one of `members` has."""
    slots = None
    for m in members:
        slots = uvs[m].slots if slots is None else np.intersect1d(slots, uvs[m].slots, assume_unique=True)
    return slots if slots is not None else np.empty(0, dtype=np.int64)


def summary(uvs: dict) -> pd.DataFrame:
    """Per member (bottom to top, the mesonet last): N, the mean vector (u, v), its speed and FROM-bearing,
    the mean speed of the slots' vectors, and steadiness = |mean vector| / mean speed.
    """
    rows = []
    for m in sorted((m for m in uvs if m != MESONET), key=height) + ([MESONET] if MESONET in uvs else []):
        w = uvs[m]
        n = int(w.u.size)
        u, v = (float(w.u.mean()), float(w.v.mean())) if n else (np.nan, np.nan)
        speed, bearing = vector_to_bearing(u, v) if n else (np.nan, np.nan)
        mean_speed = float(np.hypot(w.u, w.v).mean()) if n else np.nan
        rows.append({"member": m, "height": height(m), "n": n, "u": u, "v": v, "speed": speed, "from": bearing,
                     "mean_speed": mean_speed, "steadiness": speed / mean_speed if n and mean_speed else np.nan})
    return pd.DataFrame(rows, columns=["member", "height", "n", "u", "v", "speed", "from", "mean_speed",
                                       "steadiness"])
