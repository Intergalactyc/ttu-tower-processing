"""Fragment write/read, atomic JSON, and the per-unit manifest."""
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq


def write_fragment(table_dir: Path, key: str, df: pd.DataFrame) -> Path:
    """Write `df` as `<table_dir>/<key>.parquet`, atomically (temp file + rename)."""
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / f"{key}.parquet"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp_path)
    os.replace(tmp_path, path)
    return path


def read_table(table_dir: Path, columns: list[str] | None = None, filter=None) -> pd.DataFrame:
    """Read every fragment in `table_dir` as one table.

    Fragments can disagree on a column's type - e.g. a "variable" column is
    entirely null (Arrow type `null`) in a fragment whose only rows are
    boom-level tests, but a populated string dictionary in one with a
    variable-specific row. Letting pyarrow.dataset infer one schema from an
    arbitrary fragment (its default with no explicit schema) silently drops
    real values from every fragment that doesn't match whichever it picked.
    Unifying every fragment's own schema first - null safely promotes to the
    wider type - reads all of them correctly.
    """
    table_dir = Path(table_dir)
    if not table_dir.is_dir():
        return pd.DataFrame(columns=columns)
    paths = sorted(table_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=columns)
    schema = pa.unify_schemas([pq.read_schema(p) for p in paths])
    dataset = pa_dataset.dataset(str(table_dir), format="parquet", schema=schema)
    return dataset.to_table(columns=columns, filter=filter).to_pandas()


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp_path, path)


def write_manifest_entry(manifest_dir: Path, unit: str, entry: dict) -> None:
    write_json_atomic(Path(manifest_dir) / f"{unit}.json", entry)


def read_manifest(manifest_dir: Path) -> dict[str, dict]:
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.is_dir():
        return {}
    result = {}
    for path in sorted(manifest_dir.glob("*.json")):
        with open(path) as f:
            result[path.stem] = json.load(f)
    return result
