"""Event filters: time of day, night, inversion strength and bulk Richardson
number. They keep or drop whole events. The quantities come from a per-slot
context frame (`data.load_context`) and are summarized over each event's
slots (every slot of its span, bridged and hand-added ones too); an event
without a finite value fails an enabled filter.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttu_tower.constants import BOOMS, HEIGHTS
from ttu_tower.jets.events import Events

FILTERS = ("time_of_day", "night", "inversion", "richardson")


def vpt_column(boom: int) -> str:
    return f"vpt_b{boom}"


def rib_column(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return f"rib_b{a}-b{b}"


def _minutes(clock: str) -> int:
    hh, mm = clock.split(":")
    minutes = int(hh) * 60 + int(mm)
    if not 0 <= minutes < 24 * 60 or not 0 <= int(mm) < 60:
        raise ValueError(f"{clock!r} isn't a time of day")
    return minutes


@dataclass(frozen=True)
class JetFilters:
    time_window: tuple[str, str] | None = None  # ("HH:MM", "HH:MM") in the events' zone; may wrap midnight
    night_only: bool = False
    inversion_booms: tuple[int, int] = (2, 10)
    min_inversion: float | None = None  # K, vpt(upper) - vpt(lower)
    rib_booms: tuple[int, int] = (2, 4)
    rib_min: float | None = None
    rib_max: float | None = None

    def validate(self) -> list[str]:
        errors = []
        if self.time_window is not None:
            try:
                _minutes(self.time_window[0]), _minutes(self.time_window[1])
            except (ValueError, IndexError):
                errors.append("the time window must be two times written HH:MM")
        for name in ("inversion_booms", "rib_booms"):
            pair = getattr(self, name)
            if len(pair) != 2 or pair[0] == pair[1] or any(b not in BOOMS for b in pair):
                errors.append(f"{name.split('_')[0]}: choose two different booms")
        if self.rib_min is not None and self.rib_max is not None and self.rib_min > self.rib_max:
            errors.append("Ri_b: the minimum is above the maximum")
        return errors

    def enabled(self) -> dict[str, bool]:
        return {"time_of_day": self.time_window is not None, "night": self.night_only,
                "inversion": self.min_inversion is not None,
                "richardson": self.rib_min is not None or self.rib_max is not None}

    def inversion_pair(self) -> tuple[int, int]:
        """(lower, upper) by height."""
        return tuple(sorted(self.inversion_booms, key=lambda b: HEIGHTS[b]))

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict) -> JetFilters:
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items() if k in names}
        return cls(**kwargs)


def context_needs(filters: JetFilters) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """The vpt booms and Ri_b pairs the filters' quantities are built from."""
    return tuple(filters.inversion_pair()), (tuple(sorted(filters.rib_booms)),)


def event_quantities(events: Events, detection: pd.DataFrame, context: pd.DataFrame | None,
                     filters: JetFilters) -> pd.DataFrame:
    """Per event: the start's clock minute and night flag, the mean ΔVPT and the
    median Ri_b over its slots.
    """
    table = events.table
    start = table["start_time"]
    out = pd.DataFrame(index=table.index)
    out["start_minute"] = (start.dt.hour * 60 + start.dt.minute).astype(float)
    lower, upper = filters.inversion_pair()
    rib = rib_column(*filters.rib_booms)
    if context is None:
        out[["start_night", "dvpt_mean", "rib_median"]] = np.nan
        return out
    night = context["night"] if "night" in context else pd.Series(np.nan, index=context.index)
    out["start_night"] = night.reindex(table["start_slot"]).to_numpy(float)
    ids = events.slots["event_id"].dropna().astype(int)
    per_slot = pd.DataFrame({"event_id": ids})
    if {vpt_column(lower), vpt_column(upper)} <= set(context.columns):
        per_slot["dvpt"] = (context[vpt_column(upper)] - context[vpt_column(lower)]).reindex(ids.index)
    else:
        per_slot["dvpt"] = np.nan
    per_slot["rib"] = context[rib].reindex(ids.index) if rib in context else np.nan
    g = per_slot.groupby("event_id")
    out["dvpt_mean"] = g["dvpt"].mean().reindex(out.index)
    out["rib_median"] = g["rib"].median().reindex(out.index)
    return out


def apply_filters(quantities: pd.DataFrame, filters: JetFilters) -> pd.DataFrame:
    """One boolean column per filter (True where it passes or is off) and `passed`."""
    errors = filters.validate()
    if errors:
        raise ValueError("; ".join(errors))
    on = filters.enabled()
    out = pd.DataFrame(True, index=quantities.index, columns=list(FILTERS))
    if on["time_of_day"]:
        a, b = (_minutes(t) for t in filters.time_window)
        m = quantities["start_minute"]
        out["time_of_day"] = ((m >= a) & (m < b)) if a <= b else ((m >= a) | (m < b))
    if on["night"]:
        out["night"] = quantities["start_night"] == 1
    if on["inversion"]:
        out["inversion"] = quantities["dvpt_mean"] >= filters.min_inversion
    if on["richardson"]:
        r = quantities["rib_median"]
        ok = r.notna()
        if filters.rib_min is not None:
            ok &= r >= filters.rib_min
        if filters.rib_max is not None:
            ok &= r <= filters.rib_max
        out["richardson"] = ok
    out["passed"] = out[list(FILTERS)].all(axis=1)
    return out
