"""Loading pipeline results by run tag: the public API, re-exported from
`ttu_tower`.
"""
import json
import warnings
from pathlib import Path

import pandas as pd

from ttu_tower import schema
from ttu_tower.io import store
from ttu_tower.io.runs import find_home, link_dir_for, link_path


def find_run(tag: str, *, test: bool = False) -> Path:
    """The run directory registered for `tag`."""
    home = find_home()
    link = link_path(home, tag, test)
    if not link.is_file():
        link_dir = link.parent
        registered = sorted(p.stem for p in link_dir.glob("*.json")) if link_dir.is_dir() else []
        raise RuntimeError(f"unknown tag '{tag}'; registered tags: {registered}")
    with open(link) as f:
        data = json.load(f)
    run_dir = Path(data["run_dir"])
    if not run_dir.is_dir():
        raise RuntimeError(
            f"run directory for tag '{tag}' is missing: {run_dir} "
            f"(edit '{link}''s run_dir if the results were moved)"
        )
    return run_dir


def load_results(tag: str, table: str = "wide", *, test: bool = False, columns=None, filters=None) -> pd.DataFrame:
    """Any output table, by name, for the run registered as `tag`."""
    spec = schema.TABLES.get(table)
    if spec is None:
        raise ValueError(f"unknown table '{table}'; known tables: {sorted(schema.TABLES)}")
    run_dir = find_run(tag, test=test)
    stage_dir = run_dir / spec.stage
    if not (stage_dir / "run_summary.json").is_file():
        warnings.warn(f"'{tag}': {spec.stage}/run_summary.json is missing; the run may not have finished")

    if spec.layout == "single_file":
        return pd.read_parquet(stage_dir / f"{table}.parquet", columns=columns)
    if spec.layout == "fragments":
        return store.read_table(stage_dir / "data" / table, columns=columns, filter=filters)
    if spec.layout == "wide_monthly":
        paths = sorted((stage_dir / "wide").glob("*.parquet"))
        if not paths:
            return pd.DataFrame(columns=columns)
        return pd.concat((pd.read_parquet(p, columns=columns) for p in paths), ignore_index=True)
    raise AssertionError(f"unhandled layout '{spec.layout}'")


def list_runs(*, test: bool = False) -> pd.DataFrame:
    """Every registered run: tag, run directory, whether it exists, stages present, registration time."""
    home = find_home()
    link_dir = link_dir_for(home, test)
    rows = []
    if link_dir.is_dir():
        for path in sorted(link_dir.glob("*.json")):
            with open(path) as f:
                data = json.load(f)
            run_dir = Path(data["run_dir"])
            exists = run_dir.is_dir()
            stages = [s for s in ("primary", "secondary", "tertiary") if (run_dir / s / "run_meta.json").is_file()] if exists else []
            rows.append({
                "tag": data["tag"], "run_dir": str(run_dir), "exists": exists,
                "stages": stages, "registered": data.get("registered"),
            })
    return pd.DataFrame(rows, columns=["tag", "run_dir", "exists", "stages", "registered"])
