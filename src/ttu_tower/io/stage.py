"""Per-stage lifecycle within an already-registered run directory: the
config-hash guard and run_meta.json/run_summary.json.
"""
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from ttu_tower.io.store import write_json_atomic
from ttu_tower.timegrid import EPOCH


class StageHashMismatch(Exception):
    """A stage's config hash differs from the run directory's recorded one."""


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_run_meta(stage_dir) -> dict | None:
    path = Path(stage_dir) / "run_meta.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def check_hash_guard(stage_dir, config_hash: str, force: bool, fresh: bool = False) -> None:
    """Stop with `StageHashMismatch` if the stage directory has a recorded
    config hash that differs from `config_hash`, unless `force` (in which
    case the stage directory's contents are cleared instead).

    `fresh` clears the stage directory unconditionally, before any hash
    comparison - for a code-only change, which a config hash can never
    detect (it covers config content and the package version, not source),
    so a normal rerun (even with `--force`) would otherwise skip every batch
    whose marker fragment already exists and reuse stale output.
    """
    if fresh:
        clear_stage(stage_dir)
        return
    existing = read_run_meta(stage_dir)
    if existing is None or existing.get("config_hash") == config_hash:
        return
    if not force:
        raise StageHashMismatch(
            f"{stage_dir}: recorded config hash {existing.get('config_hash')} does not match "
            f"the current config's {config_hash}; pass --force to clear this stage's outputs and rerun"
        )
    clear_stage(stage_dir)


def clear_stage(stage_dir) -> None:
    stage_dir = Path(stage_dir)
    if stage_dir.is_dir():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)


def write_run_meta(stage_dir, *, stage: str, tag: str, config_hash: str, upstream_hash: str | None,
                    package_version: str, timezone: str, unusable_tests=None) -> None:
    meta = {
        "stage": stage, "tag": tag, "config_hash": config_hash, "upstream_hash": upstream_hash,
        "package_version": package_version, "git_commit": git_commit(),
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "epoch": EPOCH.isoformat(), "timezone": timezone,
    }
    if unusable_tests is not None:
        meta["unusable_tests"] = list(unusable_tests)
    write_json_atomic(Path(stage_dir) / "run_meta.json", meta)


def write_run_summary(stage_dir, *, stage: str, config_hash: str, package_version: str, started: str,
                       finished: str, status_counts: dict, totals: dict, timings: dict) -> None:
    write_json_atomic(Path(stage_dir) / "run_summary.json", {
        "stage": stage, "config_hash": config_hash, "package_version": package_version,
        "started": started, "finished": finished,
        "status_counts": status_counts, "totals": totals, "timings": timings,
    })
