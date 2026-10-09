"""Manual edits to the detected jet events: new boundaries for an event, an
event made by hand, and an event rejected. Edits are kept per run and reapplied
whenever the detection reruns, so they survive changed settings:

- "span": an event's new [start, end], found again by `anchor` (one of its
  detected slots). If no detected event holds the anchor any more, the span
  stays as a manual event.
- "add": an event made by hand over [start, end].
- "reject": the event holding `anchor` is not a jet; nothing happens while no
  event holds it.

Detected events overlapping an edited or hand-made span are absorbed into it,
and overlapping hand-made spans join.
"""
from __future__ import annotations

import dataclasses
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ttu_tower.constants import SOURCE_TIMEZONE
from ttu_tower.io.runs import locate_home
from ttu_tower.jets.events import Events, assemble
from ttu_tower.jets.identify import Profiles

KINDS = ("span", "add", "reject")
_VERSION = 1


@dataclass(frozen=True)
class Edit:
    id: str
    kind: str
    start: int  # slots, inclusive
    end: int
    anchor: int | None = None
    created: str = ""
    note: str = ""


def new_edit(kind: str, start: int, end: int, anchor: int | None = None, note: str = "") -> Edit:
    if kind not in KINDS:
        raise ValueError(f"unknown edit kind {kind!r}")
    if end < start:
        raise ValueError("an edit's end is before its start")
    return Edit(uuid.uuid4().hex[:10], kind, int(start), int(end), None if anchor is None else int(anchor),
                pd.Timestamp.now("UTC").isoformat(timespec="seconds"), note)


def default_path(tag: str) -> Path | None:
    home = locate_home()
    return home / "viewer" / "jet_edits" / f"{tag}.json" if home is not None else None


class EditStore:
    """One run's edits, in <home>/viewer/jet_edits/<tag>.json (written on each
    change; in memory only without a ttu-tower home).
    """

    def __init__(self, tag: str, path: Path | None = None, load: bool = True):
        self.tag = tag
        self.path = Path(path) if path is not None else default_path(tag)
        self.edits: list[Edit] = []
        if load and self.path is not None and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            names = {f.name for f in dataclasses.fields(Edit)}
            self.edits = [Edit(**{k: v for k, v in e.items() if k in names}) for e in raw.get("edits", [])]

    def save(self) -> None:
        if self.path is None:
            raise OSError("no ttu-tower home to keep the edits in")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": _VERSION, "run": self.tag,
                                   "edits": [dataclasses.asdict(e) for e in self.edits]}, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def change(self, remove=(), add=()) -> None:
        """Remove the edits with these ids, add these, and save."""
        gone = set(remove)
        self.edits = [e for e in self.edits if e.id not in gone] + list(add)
        if self.path is not None:
            self.save()

    def clear(self) -> None:
        self.change(remove=[e.id for e in self.edits])


def _overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def apply_edits(events: Events, detection: pd.DataFrame, edits, profiles: Profiles | None = None,
                tz: str = SOURCE_TIMEZONE) -> tuple[Events, pd.DataFrame]:
    """The events after the edits, and a report of what each edit did."""
    slots = detection.index.to_numpy(np.int64)
    report = []
    if slots.size == 0:
        return events, pd.DataFrame(columns=["id", "kind", "start", "end", "anchor", "status", "created", "note"])
    lo, hi = int(slots[0]), int(slots[-1])
    auto = [(int(a), int(b)) for a, b in events.table[["start_slot", "end_slot"]].itertuples(index=False)]
    consumed: set[int] = set()
    spans = []  # dicts: start, end (slots), source, edits

    def status(e: Edit, text: str) -> None:
        report.append({**dataclasses.asdict(e), "status": text})

    for e in sorted((e for e in edits if e.kind in ("span", "add")), key=lambda e: e.created):
        start, end = max(e.start, lo), min(e.end, hi)
        if start > end:
            status(e, "outside the data")
            continue
        hit = None
        if e.kind == "span" and e.anchor is not None:
            hit = next((i for i, span in enumerate(auto) if span[0] <= e.anchor <= span[1]), None)
        if hit is not None:
            consumed.add(hit)
            spans.append({"start": start, "end": end, "source": "edited", "edits": [e.id]})
            status(e, "applied")
        else:
            spans.append({"start": start, "end": end, "source": "manual", "edits": [e.id]})
            status(e, "applied" if e.kind == "add" else "kept as a manual event (its detected event is gone)")

    joined = []
    for span in sorted(spans, key=lambda s: s["start"]):
        if joined and span["start"] <= joined[-1]["end"]:
            last = joined[-1]
            last["end"] = max(last["end"], span["end"])
            last["edits"] += span["edits"]
            if span["source"] == "edited":
                last["source"] = "edited"
        else:
            joined.append(dict(span))
    for i, span in enumerate(auto):
        if i not in consumed and any(_overlaps(span, (j["start"], j["end"])) for j in joined):
            consumed.add(i)
    merged = joined + [{"start": a, "end": b, "source": "auto", "edits": []}
                       for i, (a, b) in enumerate(auto) if i not in consumed]
    merged.sort(key=lambda s: s["start"])
    for span in merged:
        span["rejected"] = False

    for e in (e for e in edits if e.kind == "reject"):
        anchor = e.anchor if e.anchor is not None else e.start
        hit = next((s for s in merged if s["start"] <= anchor <= s["end"]), None)
        if hit is None:
            status(e, "no event there now (inactive)")
            continue
        hit["rejected"] = True
        hit["edits"].append(e.id)
        status(e, "applied")

    starts = np.array([s["start"] - lo for s in merged], dtype=np.int64)
    ends = np.array([s["end"] - lo + 1 for s in merged], dtype=np.int64)
    in_auto = np.zeros(len(slots), bool)
    for i in range(len(auto)):
        a, b = auto[i]
        in_auto[a - lo:b - lo + 1] = True
    filled = events.slots["filled"].to_numpy(bool) & in_auto
    covered = np.zeros(len(slots), bool)
    for a, b in zip(starts, ends):
        covered[a:b] = True
    manual = covered & ~detection["is_jet"].to_numpy(bool) & ~filled
    result = assemble(detection, starts, ends, filled & covered, manual, profiles, tz,
                      source=[s["source"] for s in merged], rejected=[s["rejected"] for s in merged],
                      edits=[s["edits"] for s in merged])
    report = pd.DataFrame(report, columns=["id", "kind", "start", "end", "anchor", "created", "note", "status"])
    report["event_id"] = [next((int(i) for i, ids in result.table["edits"].items() if r in ids), None)
                          for r in report["id"]]
    return result, report
