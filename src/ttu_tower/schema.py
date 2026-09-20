"""Output table definitions: ordered columns, dtypes, categorical columns,
stage and on-disk layout, one per table the pipeline writes.
"""
from dataclasses import dataclass

import pandas as pd


class SchemaError(Exception):
    """A DataFrame doesn't match its table's column contract."""


@dataclass(frozen=True)
class TableSpec:
    name: str
    stage: str  # "primary" | "secondary" | "tertiary"
    layout: str  # "single_file" | "fragments" | "wide_monthly"
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    categorical: frozenset[str]


def _spec(name: str, stage: str, layout: str, columns: list[tuple[str, str]]) -> TableSpec:
    return TableSpec(
        name=name,
        stage=stage,
        layout=layout,
        columns=tuple(c for c, _ in columns),
        dtypes={c: dt for c, dt in columns},
        categorical=frozenset(c for c, dt in columns if dt == "category"),
    )


# "timestamp" columns are tz-aware datetimes whose zone is a run's chosen
# output timezone (or the source zone for raw file times); cast() only
# checks they're datetime, since the zone isn't known at schema-definition
# time. "Int8"/"Int16"/"Int32" (capitalized) are pandas nullable integer
# dtypes, for columns that can be legitimately missing.

_TABLE_DEFS = [
    # --- primary ---------------------------------------------------------
    _spec("files", "primary", "single_file", [
        ("path", "string"), ("name", "string"), ("record", "Int32"),
        ("name_time", "timestamp"), ("offset_min", "int16"),
        ("file_start", "timestamp"), ("half_hour", "int64"),
        ("n_rows", "Int32"), ("status", "category"),
    ]),
    _spec("slots", "primary", "single_file", [
        ("slot", "int64"), ("slot_start", "timestamp"), ("half_hour", "int64"),
        ("file_status", "category"), ("record", "Int32"),
    ]),
    _spec("slot_boom", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("status", "category"),
        ("unexcised_computed", "bool"),
    ]),
    _spec("coverage", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variable", "category"),
        ("layer", "category"), ("fraction", "float64"),
    ]),
    _spec("means", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variable", "category"),
        ("stat", "category"), ("value", "float64"),
    ]),
    _spec("slot_qc", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variable", "category"),
        ("stat", "category"), ("value", "float64"),
    ]),
    _spec("flags", "primary", "fragments", [
        ("start", "int64"), ("end", "int64"), ("test", "category"),
        ("kind", "category"), ("boom", "int8"), ("variable", "category"),
    ]),
    _spec("ladder", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("rung_s", "float64"), ("variable", "category"), ("stat", "category"),
        ("value", "float64"),
    ]),
    _spec("ladder_coverage", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("rung_s", "float64"), ("family", "category"),
        ("blocks_used", "int16"), ("blocks_total", "int16"), ("coverage", "float64"),
    ]),
    _spec("mrd", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("spectrum", "category"), ("scale_s", "float64"), ("value", "float64"),
        ("se", "float64"), ("n_pairs", "int32"),
    ]),
    _spec("mrd_frame", "primary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("wd_deg", "float64"), ("coverage_momentum", "float64"), ("coverage_heat", "float64"),
    ]),

    # --- secondary ---------------------------------------------------------
    _spec("tau", "secondary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("cospectrum", "category"), ("status", "category"), ("tau_s", "float64"),
        ("tau_lb_s", "float64"), ("sign", "int8"), ("peak_scale_s", "float64"),
        ("reversal_scale_s", "float64"), ("reversal_type", "category"),
    ]),
    _spec("tau_selected", "secondary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("tau_s", "float64"), ("source", "category"), ("source_status", "category"),
    ]),
    _spec("boom_stats", "secondary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("variable", "category"), ("stat", "category"), ("value", "float64"),
    ]),
    _spec("boom_labels", "secondary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("label", "category"), ("value", "string"),
    ]),
    _spec("slow", "secondary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variable", "category"), ("value", "float64"),
    ]),
    _spec("slot_stats", "secondary", "fragments", [
        ("slot", "int64"), ("sun_elevation", "float64"), ("night", "bool"),
    ]),

    # --- tertiary ---------------------------------------------------------
    _spec("boom_final", "tertiary", "fragments", [
        ("slot", "int64"), ("slot_start", "timestamp"), ("boom", "int8"),
        ("variant", "category"), ("variable", "category"), ("stat", "category"),
        ("value", "float64"),
    ]),
    _spec("tau_final", "tertiary", "fragments", [
        ("slot", "int64"), ("slot_start", "timestamp"), ("boom", "int8"),
        ("variant", "category"), ("tau_s", "float64"), ("source", "category"),
        ("source_status", "category"),
    ]),
    _spec("pairs", "tertiary", "fragments", [
        ("slot", "int64"), ("slot_start", "timestamp"), ("boom", "int8"),
        ("boom2", "int8"), ("variant", "category"), ("variable", "category"),
        ("value", "float64"),
    ]),
    _spec("profile", "tertiary", "fragments", [
        ("slot", "int64"), ("slot_start", "timestamp"), ("variant", "category"),
        ("variable", "category"), ("value", "float64"),
    ]),
    _spec("slot_final", "tertiary", "fragments", [
        ("slot", "int64"), ("slot_start", "timestamp"), ("variable", "category"),
        ("value", "float64"),
    ]),
    _spec("filter_log", "tertiary", "fragments", [
        ("slot", "int64"), ("boom", "int8"), ("variant", "category"),
        ("group", "category"), ("criterion", "category"), ("variable", "category"),
    ]),
    # "wide" is a pivoted, dynamic-width convenience export: only slot_start
    # is guaranteed, so cast() isn't used on it. Its TableSpec exists so
    # load_results can resolve its path.
    _spec("wide", "tertiary", "wide_monthly", [("slot_start", "timestamp")]),
]

TABLES: dict[str, TableSpec] = {t.name: t for t in _TABLE_DEFS}


def cast(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df` has exactly `name`'s columns and cast them to its dtypes."""
    spec = TABLES[name]
    given = set(df.columns)
    expected = set(spec.columns)
    missing = expected - given
    extra = given - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"extra {sorted(extra)}")
        raise SchemaError(f"table '{name}': {'; '.join(parts)}")

    out = df[list(spec.columns)].copy()
    for col in spec.columns:
        if col in spec.categorical:
            out[col] = out[col].astype("category")
            continue
        dtype = spec.dtypes[col]
        if dtype == "timestamp":
            out[col] = pd.to_datetime(out[col])
        else:
            out[col] = out[col].astype(dtype)
    return out
