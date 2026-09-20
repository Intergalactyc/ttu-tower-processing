"""Raw .csv/.csv.gz/zip-wrapped-.csv file conversion to the canonical column schema."""
import os
import zipfile

import numpy as np
import pandas as pd

from ttu_tower.constants import HEADER_MAP_NEW, HEADER_MAP_OLD, SOURCE_HEADERS_NEW, SOURCE_HEADERS_OLD


def _resolve_zip(filepath: str, temp_dir: str) -> tuple[str, bool]:
    """If `filepath` is a .zip with exactly one member, extracts it into
    `temp_dir` and returns (extracted_path, True); otherwise (filepath, False).
    """
    if str(filepath).endswith(".zip"):
        with zipfile.ZipFile(filepath) as zipped:
            members = zipped.infolist()
            if len(members) != 1:
                raise ValueError(f"zip file must have exactly one member: {filepath}")
            name = members[0].filename
            zipped.extract(name, path=temp_dir)
        return os.path.join(temp_dir, name), True
    return str(filepath), False


def _read_raw_csv(filepath: str, names: list[str] | None) -> pd.DataFrame:
    return pd.read_csv(
        filepath, compression="gzip" if str(filepath).endswith(".gz") else None,
        dtype=np.float32, header=None, index_col=False, names=names, engine="c", thousands=",",
    )


def infer_source_schema(n_columns: int) -> tuple[list[str], dict]:
    """The positional header list and rename map a headerless raw file uses,
    from its column count (the old era carries propeller-boom columns that
    the new era lacks).
    """
    if n_columns == len(SOURCE_HEADERS_OLD):
        return SOURCE_HEADERS_OLD, HEADER_MAP_OLD
    if n_columns == len(SOURCE_HEADERS_NEW):
        return SOURCE_HEADERS_NEW, HEADER_MAP_NEW
    raise ValueError(
        f"unrecognized raw column count {n_columns} (expected {len(SOURCE_HEADERS_OLD)} for "
        f"old-structure or {len(SOURCE_HEADERS_NEW)} for new-structure files)"
    )


def _rename_headers(df: pd.DataFrame, header_map: dict) -> pd.DataFrame:
    rename, drop = {}, []
    for col in df.columns:
        prefix, boom = col.split("_")
        target = header_map.get(prefix)
        if target is None:
            drop.append(col)
        else:
            rename[col] = f"{target}_{boom}"
    return df.drop(columns=drop).rename(columns=rename)


def convert_raw_file(filepath: str, temp_dir: str) -> pd.DataFrame:
    """Reads one raw .csv/.csv.gz/zip-wrapped-.csv file and renames its
    columns to the canonical schema, inferring old- vs. new-structure from
    the column count.
    """
    real_path, cleanup = _resolve_zip(filepath, temp_dir)
    try:
        raw = _read_raw_csv(real_path, names=None)
    finally:
        if cleanup:
            os.remove(real_path)

    # A trailing delimiter on each row (present in some raw files) produces one
    # fully-empty extra column beyond the logical schema.
    while raw.shape[1] and raw.iloc[:, -1].isna().all():
        raw = raw.iloc[:, :-1]

    source_headers, header_map = infer_source_schema(raw.shape[1])
    raw.columns = source_headers
    return _rename_headers(raw, header_map)
