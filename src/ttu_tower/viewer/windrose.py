"""Wind roses: a boom's slot-mean wind (tertiary's final values, or every
value before filtering) and the mesonet station's, binned by the direction
the wind comes from and by speed.
"""
from dataclasses import dataclass

import numpy as np

from ttu_tower.viewer import timeline
from ttu_tower.viewer.catalog import Quantity
from ttu_tower.viewer.timeaxis import slot_to_unix

SPEED_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, np.inf)


def _quantity(table: str, variable: str, stat: str | None, kind: str) -> Quantity:
    return Quantity(key=f"{table}|{variable}|{stat or ''}", table=table, variable=variable, stat=stat, kind=kind,
                    variants=("none",), label=variable, unit="", group="Wind",
                    circular=variable.startswith("wd"))


@dataclass
class Wind:
    slots: np.ndarray
    ws: np.ndarray
    wd: np.ndarray  # degrees, the bearing the wind comes from


def tower_wind(run, index, boom: int, filtered: bool = True) -> Wind:
    """A boom's slot means: tertiary's final values if `filtered`, else the
    first-layer means (`slot_qc` mean_l1) - before the second-layer direction
    (shadow) and bounce removals and before tertiary filtering, so the shadow
    sector is populated.
    """
    if filtered:
        qs = [_quantity("boom_final", v, "mean", "boom") for v in ("ws", "wd")]
    else:
        qs = [_quantity("slot_qc", v, "mean_l1", "boom") for v in ("ws", "wd")]
    ws, wd = (timeline.load(run, index, q, "none", [boom]).curves[boom] for q in qs)
    slots, s, d = timeline.joined(ws, wd)
    ok = np.isfinite(s) & np.isfinite(d)
    return Wind(slots[ok], s[ok], d[ok])


def mesonet_wind(run, index) -> Wind | None:
    """None when the run merged no mesonet data."""
    ws, wd = (timeline.load(run, index, _quantity("slot_final", v, None, "slot"), "none").curves[None]
              for v in ("ws_meso_mean", "wd_meso_mean"))
    if ws.slots.size == 0:
        return None
    slots, s, d = timeline.joined(ws, wd)
    ok = np.isfinite(s) & np.isfinite(d)
    return Wind(slots[ok], s[ok], d[ok])


def only_slots(wind: Wind, slots: np.ndarray) -> Wind:
    keep = np.isin(wind.slots, slots)
    return Wind(wind.slots[keep], wind.ws[keep], wind.wd[keep])


def parse_bearings(text: str) -> list[float]:
    """"105, 170" -> [105.0, 170.0] (ValueError on anything else)."""
    return [float(part) % 360.0 for part in text.replace(";", ",").split(",") if part.strip()]


def in_range(wind: Wind, x_range) -> Wind:
    """The slots starting in [x0, x1) unix seconds (all of them for None)."""
    if x_range is None:
        return wind
    t = slot_to_unix(wind.slots)
    keep = (t >= x_range[0]) & (t < x_range[1])
    return Wind(wind.slots[keep], wind.ws[keep], wind.wd[keep])


def rose(ws: np.ndarray, wd: np.ndarray, sectors: int = 16, speed_edges=SPEED_EDGES) -> np.ndarray:
    """sectors x speed-bins percentages of all values; sector 0 is centred on
    north, then clockwise.
    """
    ws, wd = np.asarray(ws, dtype=float), np.asarray(wd, dtype=float)
    width = 360.0 / sectors
    sector = (np.floor(((wd % 360.0) + width / 2) / width) % sectors).astype(np.int64)
    speed = np.clip(np.searchsorted(np.asarray(speed_edges), ws, side="right") - 1, 0, len(speed_edges) - 2)
    counts = np.zeros((sectors, len(speed_edges) - 1))
    np.add.at(counts, (sector, speed), 1)
    return 100.0 * counts / max(ws.size, 1)
