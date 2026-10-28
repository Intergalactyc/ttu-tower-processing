"""Named views: the viewer's state (panels, interval, tabs and their settings,
the brushed selection), saved under a name to reopen later, on the run it was
saved from or another. Kept in one JSON file in the ttu-tower home.
"""
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ttu_tower.io.runs import locate_home

_VERSION = 1


@dataclass
class SavedView:
    name: str
    run: str  # the run it was saved from
    state: dict
    note: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def default_path() -> Path | None:
    home = locate_home()
    return home / "viewer" / "views.json" if home is not None else None


class ViewStore:
    """Every change is written through; `path` None keeps them in memory only."""

    def __init__(self, path: Path | None = None, load: bool = True):
        self.path = Path(path) if path is not None else (default_path() if load else None)
        self.items: list[SavedView] = []
        if self.path is not None and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            fields = set(SavedView.__dataclass_fields__)
            self.items = [SavedView(**{k: v for k, v in item.items() if k in fields}) for item in raw.get("views", [])]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": _VERSION, "views": [asdict(v) for v in self.items]}, indent=1,
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, view: SavedView) -> None:
        """A view of the same name and run replaces the older one."""
        self.items = [v for v in self.items if not (v.name == view.name and v.run == view.run)] + [view]
        self.save()

    def remove(self, view_id: str) -> None:
        self.items = [v for v in self.items if v.id != view_id]
        self.save()

    def ordered(self, run: str | None) -> list[SavedView]:
        """This run's views first, then the others', each by name."""
        return sorted(self.items, key=lambda v: (v.run != run, v.run, v.name.lower()))
