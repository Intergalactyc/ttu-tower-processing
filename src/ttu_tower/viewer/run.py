"""Opening a run directory read-only: stage metadata, the config the run was
processed with, and the file table - without ever creating a ttu-tower home.
"""
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from ttu_tower import __version__
from ttu_tower.config import Config, ConfigError, config_from_dict, load_config
from ttu_tower.config import hashing
from ttu_tower.io.rawfiles import accepted_half_hours, period_slots_from_table
from ttu_tower.io.runs import link_dir_for, link_path, locate_home
from ttu_tower.io.stage import git_commit, git_dirty, read_run_meta
from ttu_tower.viewer.cache import LRUCache

STAGES = ("primary", "secondary", "tertiary")
_HASHES = {"primary": hashing.primary_hash, "secondary": hashing.secondary_hash, "tertiary": hashing.tertiary_hash}


class RunNotFound(Exception):
    """No run is registered under a tag, or its directory is missing."""


def _read_link(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def list_registered(test: bool = False) -> list[dict]:
    """Every registered run (tag, run_dir, exists, stages, registered, config);
    empty if there is no ttu-tower home.
    """
    home = locate_home()
    if home is None:
        return []
    link_dir = link_dir_for(home, test)
    rows = []
    for path in sorted(link_dir.glob("*.json")) if link_dir.is_dir() else []:
        data = _read_link(path)
        if data is None:
            continue
        run_dir = Path(data["run_dir"])
        stages = [s for s in STAGES if (run_dir / s / "run_meta.json").is_file()]
        rows.append({"tag": data["tag"], "run_dir": run_dir, "exists": run_dir.is_dir(), "stages": stages,
                     "registered": data.get("registered"), "config": data.get("config")})
    return rows


@dataclass(eq=False)
class RunHandle:
    """One run directory, opened read-only."""

    run_dir: Path
    tag: str
    registered_config: str | None = None
    caches: dict = field(default_factory=dict, repr=False)

    @classmethod
    def open(cls, tag: str, test: bool = False) -> "RunHandle":
        home = locate_home()
        data = _read_link(link_path(home, tag, test)) if home is not None else None
        if data is None:
            raise RunNotFound(f"no run registered as '{tag}'")
        run_dir = Path(data["run_dir"])
        if not run_dir.is_dir():
            raise RunNotFound(f"run directory for '{tag}' is missing: {run_dir}")
        return cls(run_dir=run_dir, tag=tag, registered_config=data.get("config"))

    @classmethod
    def from_dir(cls, run_dir, config_path: str | None = None) -> "RunHandle":
        run_dir = Path(run_dir)
        meta = read_run_meta(run_dir / "primary")
        if meta is None:
            raise RunNotFound(f"{run_dir} has no primary stage")
        return cls(run_dir=run_dir, tag=meta.get("tag") or run_dir.name, registered_config=config_path)

    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / stage

    def cache(self, name: str, maxsize: int = 64) -> LRUCache:
        """A cache owned by this run, so closing the run frees its data."""
        if name not in self.caches:
            self.caches[name] = LRUCache(maxsize)
        return self.caches[name]

    @cached_property
    def meta(self) -> dict[str, dict | None]:
        return {s: read_run_meta(self.stage_dir(s)) for s in STAGES}

    @property
    def stages(self) -> list[str]:
        return [s for s in STAGES if self.meta[s] is not None]

    @property
    def timezone(self) -> str:
        return self.meta["primary"].get("timezone", "Etc/GMT+6")

    @property
    def unusable_tests(self) -> tuple[str, ...]:
        return tuple(self.meta["primary"].get("unusable_tests") or ())

    @cached_property
    def _config_resolution(self) -> tuple[Config | None, str | None, list[str]]:
        """(cfg, source, problems). The source is "resolved" (a stage's own
        config.resolved.json) or "registered" (the registry's config path,
        trusted only if it reproduces every present stage's config hash).
        """
        for stage in self.stages:
            path = self.stage_dir(stage) / "config.resolved.json"
            if path.is_file():
                with open(path) as f:
                    return config_from_dict(json.load(f)), "resolved", []

        if not self.registered_config:
            return None, None, ["the run has no config.resolved.json and no registered config path"]
        try:
            cfg = load_config(self.registered_config)
        except (OSError, ConfigError) as exc:
            return None, None, [f"registered config unavailable: {exc}"]
        problems = []
        for stage in self.stages:
            meta = self.meta[stage]
            expected = _HASHES[stage](cfg, meta.get("package_version", __version__))
            if expected != meta.get("config_hash"):
                problems.append(f"{stage}: {self.registered_config} no longer matches the run's config hash")
        return cfg, "registered", problems

    @property
    def cfg(self) -> Config | None:
        return self._config_resolution[0]

    @property
    def config_source(self) -> str | None:
        return self._config_resolution[1]

    @property
    def config_problems(self) -> list[str]:
        return self._config_resolution[2]

    @property
    def can_reprocess(self) -> bool:
        return self.cfg is not None and not self.config_problems

    @cached_property
    def installed_commit(self) -> tuple[str | None, bool | None]:
        return git_commit(), git_dirty()

    def commit_note(self) -> str | None:
        """A short warning when the installed code may differ from the code
        that produced the run (reprocessing shows the installed code's result).
        """
        installed, dirty = self.installed_commit
        notes = []
        for stage in self.stages:
            run_commit = self.meta[stage].get("git_commit")
            if run_commit and installed and run_commit != installed:
                notes.append(f"{stage} at {run_commit[:7]}")
            elif self.meta[stage].get("git_dirty"):
                notes.append(f"{stage} at {(run_commit or '?')[:7]} with uncommitted changes")
        if not notes and not dirty:
            return None
        installed_desc = (installed or "?")[:7] + (" with uncommitted changes" if dirty else "")
        return f"installed code {installed_desc}; run produced by " + (", ".join(notes) or "the same commit")

    @cached_property
    def files(self) -> pd.DataFrame:
        return pd.read_parquet(self.stage_dir("primary") / "files.parquet")

    @cached_property
    def slots(self) -> pd.DataFrame:
        return pd.read_parquet(self.stage_dir("primary") / "slots.parquet")

    @cached_property
    def period(self) -> tuple[int, int]:
        return period_slots_from_table(self.slots)

    @cached_property
    def accepted(self) -> dict[int, Path]:
        return accepted_half_hours(self.files)

    @property
    def booms(self) -> list[int]:
        return list(self.cfg.files.booms) if self.cfg is not None else list(range(1, 11))
