"""The ttu-tower home and the run registry: finding/creating the home
directory, resolving a config's run directory, and registering a run's
tag -> run directory link.
"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from ttu_tower.io.store import write_json_atomic

_SIGNATURE = "ttu-tower-processing home"
_CANDIDATE_RE = re.compile(r"^\.ttu-tower(\d*)$")

_logger = logging.getLogger(__name__)


def _is_ttu_tower_home(path: Path) -> bool:
    sig_path = path / "ttu-tower-home.json"
    if not sig_path.is_file():
        return False
    try:
        with open(sig_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("signature") == _SIGNATURE


def _write_signature(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path / "ttu-tower-home.json", {
        "signature": _SIGNATURE,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


def _candidate_sort_key(name: str) -> int:
    suffix = _CANDIDATE_RE.match(name).group(1)
    return int(suffix) if suffix else -1  # ".ttu-tower" (no suffix) sorts first


def find_home(user_home: Path | None = None) -> Path:
    """Find or create the ttu-tower home."""
    env = os.environ.get("TTU_TOWER_HOME")
    if env:
        path = Path(env)
        if _is_ttu_tower_home(path):
            return path
        if not path.exists() or (path.is_dir() and not any(path.iterdir())):
            _write_signature(path)
            return path
        raise RuntimeError(f"TTU_TOWER_HOME={path} exists and is not a ttu-tower home")

    base = user_home if user_home is not None else Path.home()
    candidates = [e.name for e in base.iterdir() if _CANDIDATE_RE.match(e.name)] if base.is_dir() else []
    for name in sorted(candidates, key=_candidate_sort_key):
        path = base / name
        if _is_ttu_tower_home(path):
            return path

    n = None
    while True:
        name = ".ttu-tower" if n is None else f".ttu-tower{n}"
        path = base / name
        if not path.exists():
            _write_signature(path)
            if name != ".ttu-tower":
                _logger.warning(f"~/.ttu-tower exists but isn't a ttu-tower home; created ~/{name}")
            return path
        n = 0 if n is None else n + 1


def link_dir_for(home: Path, test: bool) -> Path:
    return home / "runs" / "testing" if test else home / "runs"


def link_path(home: Path, tag: str, test: bool) -> Path:
    return link_dir_for(home, test) / f"{tag}.json"


def run_dir_for(cfg, test: bool) -> Path:
    """A config's run directory, without registering it."""
    base = Path(cfg.paths.results_dir) if cfg.paths.results_dir else find_home() / "results"
    if test:
        base = base / "testing"
    return base / cfg.tag


def register_run(cfg, config_path, *, test: bool = False, force: bool = False, package_version: str) -> Path:
    """Resolve and register `cfg`'s run directory. Runs in the main process
    of every command that writes into the run directory, before it writes
    anything.
    """
    run_dir = run_dir_for(cfg, test)
    home = find_home()
    link = link_path(home, cfg.tag, test)

    existing = None
    if link.is_file():
        with open(link) as f:
            existing = json.load(f)

    if existing is not None:
        existing_dir = Path(existing["run_dir"])
        if existing_dir == run_dir:
            return run_dir
        if existing_dir.exists():
            if not force:
                raise RuntimeError(
                    f"tag '{cfg.tag}' is registered to {existing_dir}; use another tag, "
                    "or pass --force to re-register it (the old results are not deleted)"
                )
        else:
            _logger.warning(
                f"tag '{cfg.tag}' was registered to {existing_dir}, which no longer exists; "
                f"re-pointing to {run_dir}"
            )

    write_json_atomic(link, {
        "tag": cfg.tag,
        "run_dir": str(run_dir),
        "config": str(Path(config_path).resolve()),
        "registered": datetime.now().astimezone().isoformat(timespec="seconds"),
        "package_version": package_version,
    })
    return run_dir
