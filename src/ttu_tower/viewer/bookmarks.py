"""Bookmarked slots: a run, slot, boom and variant with a note, in named lists,
kept in <home>/viewer/bookmarks.json (created on the first save), and
exportable as JSON or CSV.
"""
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ttu_tower.io.runs import locate_home

DEFAULT_LIST = "bookmarks"
_VERSION = 1


@dataclass
class Bookmark:
    run: str
    slot: int
    boom: int
    variant: str = "mrd"
    note: str = ""
    list: str = DEFAULT_LIST
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def default_path() -> Path | None:
    home = locate_home()
    return home / "viewer" / "bookmarks.json" if home is not None else None


class BookmarkStore:
    """Bookmarks in one JSON file; every change is written through."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_path()
        self.items: list[Bookmark] = []
        if self.path is not None and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            fields = set(Bookmark.__dataclass_fields__)
            self.items = [Bookmark(**{k: v for k, v in b.items() if k in fields}) for b in raw.get("bookmarks", [])]

    def save(self) -> None:
        if self.path is None:
            raise OSError("no ttu-tower home to keep bookmarks in")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": _VERSION, "bookmarks": [asdict(b) for b in self.items]}, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, *bookmarks: Bookmark) -> None:
        self.items.extend(bookmarks)
        self.save()

    def remove(self, ids) -> None:
        ids = set(ids)
        self.items = [b for b in self.items if b.id not in ids]
        self.save()

    def update(self, bookmark_id: str, **changes) -> None:
        for b in self.items:
            if b.id == bookmark_id:
                for k, v in changes.items():
                    setattr(b, k, v)
        self.save()

    def for_run(self, run: str | None) -> list[Bookmark]:
        return [b for b in self.items if run is None or b.run == run]

    def lists(self, run: str | None = None) -> list[str]:
        names = {b.list for b in self.for_run(run)} | {DEFAULT_LIST}
        return sorted(names, key=lambda n: (n != DEFAULT_LIST, n.lower()))

    def export(self, path: Path, bookmarks: list[Bookmark], describe=None) -> None:
        """JSON (as stored) or CSV (by extension); `describe(slot)` adds a time column to CSV."""
        path = Path(path)
        rows = [asdict(b) for b in bookmarks]
        if path.suffix.lower() == ".csv":
            frame = pd.DataFrame(rows, columns=list(Bookmark.__dataclass_fields__))
            if describe is not None:
                frame.insert(2, "time", [describe(s) for s in frame["slot"]])
            frame.to_csv(path, index=False, encoding="utf-8")
        else:
            path.write_text(json.dumps({"version": _VERSION, "bookmarks": rows}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
