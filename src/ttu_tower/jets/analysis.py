"""One call from settings to everything about the jets: detection, events,
the filters' quantities and verdicts, and the counts that show which rule
removed what. The settings serialize to plain dicts (saved by the viewer;
ready for an export later).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttu_tower.constants import SOURCE_TIMEZONE
from ttu_tower.jets import identify, noses
from ttu_tower.jets.edits import apply_edits
from ttu_tower.jets.events import EventParams, Events, find_events, nose_rows, nose_stats
from ttu_tower.jets.filters import FILTERS, JetFilters, apply_filters, event_quantities
from ttu_tower.jets.identify import ALGORITHMS, Params, Profiles

DEFAULT_ALGORITHM = "nose"
_FILTER_QUANTITY = {"night": "start_night", "inversion": "dvpt_mean", "richardson": "rib_median"}


@dataclass(frozen=True)
class JetSettings:
    algorithm: str = DEFAULT_ALGORITHM
    params: Params | None = None  # None: the algorithm's defaults
    events: EventParams = EventParams()
    filters: JetFilters = JetFilters()

    def algorithm_params(self) -> Params:
        return ALGORITHMS[self.algorithm].params() if self.params is None else self.params

    def errors(self) -> list[str]:
        if self.algorithm not in ALGORITHMS:
            return [f"no algorithm {self.algorithm!r}"]
        return self.algorithm_params().validate() + self.events.validate() + self.filters.validate()

    def to_dict(self, saved: dict | None = None) -> dict:
        """Plain values; `saved` keeps other algorithms' parameters from an earlier save."""
        params = dict((saved or {}).get("params", {}))
        params[self.algorithm] = identify.to_dict(self.algorithm_params())
        return {"algorithm": self.algorithm, "params": params, "events": identify.to_dict(self.events),
                "filters": self.filters.to_dict()}

    @classmethod
    def from_dict(cls, values: dict) -> JetSettings:
        """From a save, tolerating stale or partial values (anything unusable falls back to its default)."""
        key = values.get("algorithm", DEFAULT_ALGORITHM)
        key = key if key in ALGORITHMS else DEFAULT_ALGORITHM
        try:
            params = identify.from_dict(ALGORITHMS[key].params, values.get("params", {}).get(key, {}))
            events = identify.from_dict(EventParams, values.get("events", {}))
            filters = JetFilters.from_dict(values.get("filters", {}))
        except (TypeError, ValueError, AttributeError):
            return cls(algorithm=key)
        return cls(algorithm=key, params=params, events=events, filters=filters)

    def replace(self, **changes) -> JetSettings:
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class JetResult:
    settings: JetSettings
    detection: pd.DataFrame
    events: Events  # after the manual edits
    quantities: pd.DataFrame  # per event: the filters' quantities
    verdicts: pd.DataFrame  # per event: one column per filter, rejected (by hand), and passed
    edit_report: pd.DataFrame | None = None  # what each manual edit did
    profiles: Profiles | None = None

    @property
    def passed(self) -> pd.Series:
        return self.verdicts["passed"]

    def events_in(self, slot_range=None, passed_only: bool = True) -> pd.DataFrame:
        """The events (with their filter quantities and verdicts) starting in
        `slot_range` ([a, b) slots; None: all), only those passing by default.
        """
        table = self.table()
        if passed_only:
            table = table[table["passed"]]
        if slot_range is not None:
            table = table[(table["start_slot"] >= slot_range[0]) & (table["start_slot"] < slot_range[1])]
        return table

    def detected_in(self, event_ids, nose_from: str = "jets") -> pd.DataFrame:
        """The detection rows whose noses describe these events, with their event
        id: their jet slots, or (`nose_from` "candidates") every slot with a candidate.
        """
        ids = self.events.slots["event_id"]
        return nose_rows(self.detection, ids.where(ids.isin(list(event_ids))), nose_from)

    def slots_in(self, event_ids) -> pd.Series:
        """Every slot of these events' spans, with its event id."""
        ids = self.events.slots["event_id"]
        return ids[ids.isin(list(event_ids))].astype(int)

    def table(self, nose_from: str = "jets") -> pd.DataFrame:
        """The events with their filter quantities and verdicts, the nose
        statistics from `nose_from`'s slots.
        """
        table = self.events.table
        if nose_from != "jets":
            stats = nose_stats(self.detection, self.events.slots["event_id"], self.profiles, nose_from)
            table = table.drop(columns=stats.columns).join(stats)
        return table.join(self.quantities).join(self.verdicts.drop(columns="rejected"))

    def event_slots(self, event_ids=None, passed_only: bool = True) -> np.ndarray:
        """Every slot (detected and bridged) of these events (default: those passing the filters)."""
        ids = self.events.slots["event_id"]
        keep = ids.notna()
        if passed_only:
            keep &= ids.isin(self.passed.index[self.passed])
        if event_ids is not None:
            keep &= ids.isin(list(event_ids))
        return ids.index[keep].to_numpy(np.int64)

    def counts(self, slot_range=None) -> dict:
        """Slots per status (in the algorithm's order), jet slots, events, events
        passing, and per enabled filter how many events it fails and how many of
        those for want of a value. With `slot_range` ([a, b)), only slots in it
        and events starting in it.
        """
        detection = self.detection
        if slot_range is not None:
            detection = detection.loc[slot_range[0]:slot_range[1] - 1]
        events = self.events_in(slot_range, passed_only=False)
        enabled = self.settings.filters.enabled()
        failed = {f: int((~events[f]).sum()) for f in FILTERS if enabled[f]}
        no_value = {f: int((~events[f] & events[_FILTER_QUANTITY[f]].isna()).sum())
                    for f in failed if f in _FILTER_QUANTITY}
        return {"status": {str(c): int(n) for c, n in detection["status"].value_counts(sort=False).items()},
                "jet_slots": int(detection["is_jet"].sum()), "events": len(events),
                "passed": int(events["passed"].sum()), "failed": failed, "no_value": no_value,
                "rejected": int(events["rejected"].sum()), "edited": int((events["source"] == "edited").sum()),
                "manual": int((events["source"] == "manual").sum())}


def analyse(profiles: Profiles, settings: JetSettings, context: pd.DataFrame | None = None,
            tz: str = SOURCE_TIMEZONE, edits=()) -> JetResult:
    """Detection, events (then the manual `edits`), the filters, and their
    verdicts. The detection gains each peak's estimated nose (`noses.estimate`).
    """
    errors = settings.errors()
    if errors:
        raise ValueError("; ".join(errors))
    params = settings.algorithm_params()
    detection = identify.detect(settings.algorithm, profiles, params)
    # the estimated nose, for describing the events only
    detection = detection.join(noses.estimate(detection, profiles, getattr(params, "profile_booms", None)))
    events = find_events(detection, settings.events, profiles, tz)
    events, report = apply_edits(events, detection, edits, profiles, tz)
    quantities = event_quantities(events, detection, context, settings.filters)
    verdicts = apply_filters(quantities, settings.filters)
    verdicts["rejected"] = events.table["rejected"]
    verdicts["passed"] = verdicts["passed"] & ~verdicts["rejected"]
    return JetResult(settings, detection, events, quantities, verdicts, report, profiles)


# --- how the counts depend on one setting ------------------------------------------------------

def sweep_values(field: dict, current) -> np.ndarray:
    """Values to sweep a numeric parameter over: from its lower limit (or 0) to
    about 2.5 times its current value (or its suggested one), within its limits.
    """
    lo = field.get("lo") if field.get("lo") is not None else 0.0
    centre = current if current is not None else field.get("suggested")
    if centre is None or centre <= lo:
        centre = (field.get("hi") or lo + 10.0) / 4
    hi = lo + 2.5 * (centre - lo)
    if field.get("hi") is not None:
        hi = min(hi, field["hi"])
    if field["kind"] == "int":
        return np.arange(int(lo), int(max(hi, lo + 6)) + 1)
    return np.linspace(lo, hi, 26)


def sweep(profiles: Profiles, settings: JetSettings, part: str, name: str, values, context=None,
          tz: str = SOURCE_TIMEZONE, slot_range=None) -> pd.DataFrame:
    """Jet slots, events and events kept as one parameter (`part` "params" or
    "events", field `name`) takes each of `values`, the rest as in `settings`.
    An event parameter reuses one detection.
    """
    rows = []
    detection = None
    for v in values:
        v = v.item() if isinstance(v, np.generic) else v
        if part == "params":
            trial = settings.replace(params=dataclasses.replace(settings.algorithm_params(), **{name: v}))
        else:
            trial = settings.replace(events=dataclasses.replace(settings.events, **{name: v}))
        if trial.errors():
            rows.append({"value": v, "jet_slots": np.nan, "events": np.nan, "kept": np.nan})
            continue
        if part == "params" or detection is None:
            detection = identify.detect(trial.algorithm, profiles, trial.algorithm_params())
        events = find_events(detection, trial.events, profiles, tz)
        quantities = event_quantities(events, detection, context, trial.filters)
        verdicts = apply_filters(quantities, trial.filters)
        verdicts["rejected"] = events.table["rejected"]
        result = JetResult(trial, detection, events, quantities, verdicts)
        c = result.counts(slot_range)
        rows.append({"value": v, "jet_slots": c["jet_slots"], "events": c["events"], "kept": c["passed"]})
    return pd.DataFrame(rows)
