"""Whole-period series of one catalog quantity, per boom (or boom pair), and
the overlays drawn behind the timelines.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower import schema
from ttu_tower.flags import FlagStore
from ttu_tower.post.classify import stability_classes
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer.catalog import Quantity
from ttu_tower.viewer.fragments import FragmentIndex, alias_bounds_variables
from ttu_tower.viewer.timeaxis import slot_to_unix

SLOT_STATUSES = ("no_file", "no_data", "computed")


@dataclass
class Curve:
    slots: np.ndarray  # int64, sorted
    y: np.ndarray  # float64; category codes for categorical quantities (NaN = missing)

    @property
    def x(self) -> np.ndarray:
        return slot_to_unix(self.slots)


@dataclass
class TimelineData:
    quantity: Quantity
    variant: str
    curves: dict  # boom | (boom, boom2) | None -> Curve
    categories: list[str] | None = None


def _has(table: str, column: str) -> bool:
    return column in schema.TABLES[table].columns


def _filter(q: Quantity, variant: str):
    f = pa_dataset.field
    parts = []
    if q.table not in ("tau_final", "tau", "mrd_frame", "slot_boom", "flags", "boom_labels_final"):
        parts.append(f("variable") == q.variable)
    if q.table == "boom_labels_final":
        parts.append(f("label") == q.variable)
    if _has(q.table, "stat"):
        parts.append(f("stat").is_null() if q.stat is None else f("stat") == q.stat)
    if _has(q.table, "variant") and variant != "none":
        parts.append(f("variant") == variant)
    elif _has(q.table, "variant"):
        parts.append(f("variant") == "none")
    for column, value in q.extra:
        parts.append(f(column) == value)
    expr = parts[0] if parts else None
    for p in parts[1:]:
        expr = expr & p
    return expr


def _is_per_boom_fragments(table: str) -> bool:
    return schema.TABLES[table].stage == "primary"


def load(run, index: FragmentIndex, q: Quantity, variant: str, members=None) -> TimelineData:
    """`members`: booms for a per-boom quantity, (boom, boom2) pairs for a pair
    quantity, ignored for a slot quantity. Cached per run.
    """
    members = tuple(members) if members is not None else ()
    key = (q.key, variant, members)
    return run.cache("timeline", 32).get(key, lambda: _load(run, index, q, variant, members))


def _load(run, index: FragmentIndex, q: Quantity, variant: str, members: tuple) -> TimelineData:
    if q.table == "flags":
        return _load_flag_fractions(run, index, q, members)

    f = pa_dataset.field
    flt = _filter(q, variant)
    columns = ["slot", q.column]
    booms = None
    if q.kind == "boom":
        booms = list(members)
        columns.append("boom")
        boom_flt = f("boom").isin(booms)
        flt = boom_flt if flt is None else flt & boom_flt
    elif q.kind == "pair":
        columns += ["boom", "boom2"]
        flt = flt & f("boom").isin([b for b, _ in members]) & f("boom2").isin([b2 for _, b2 in members])

    df = index.read(q.table, booms=booms if _is_per_boom_fragments(q.table) else None, columns=columns, filter=flt,
                    include_other=q.table == "slot_boom")

    categories = None
    values = df[q.column] if not df.empty else pd.Series(dtype=float)
    if q.categorical:
        cat = values.astype("category") if not df.empty else pd.Categorical([])
        categories = [str(c) for c in (cat.cat.categories if not df.empty else [])]
        if q.table == "slot_boom":
            categories = list(SLOT_STATUSES)
            cat = pd.Categorical(values.astype(object), categories=categories)
            codes = np.asarray(cat.codes, dtype=np.float64)
        else:
            codes = np.asarray(cat.cat.codes, dtype=np.float64) if not df.empty else np.empty(0)
        codes[codes < 0] = np.nan
        y_all = codes
    else:
        y_all = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64) if not df.empty else np.empty(0)

    curves = {}
    slots_all = df["slot"].to_numpy(dtype=np.int64) if not df.empty else np.empty(0, dtype=np.int64)
    if q.kind == "boom":
        boom_col = df["boom"].to_numpy() if not df.empty else np.empty(0)
        for b in members:
            sel = boom_col == b
            curves[b] = _curve(slots_all[sel], y_all[sel])
    elif q.kind == "pair":
        b1 = df["boom"].to_numpy() if not df.empty else np.empty(0)
        b2 = df["boom2"].to_numpy() if not df.empty else np.empty(0)
        for pair in members:
            sel = (b1 == pair[0]) & (b2 == pair[1])
            curves[pair] = _curve(slots_all[sel], y_all[sel])
    else:
        curves[None] = _curve(slots_all, y_all)
    return TimelineData(quantity=q, variant=variant, curves=curves, categories=categories)


def _curve(slots: np.ndarray, y: np.ndarray) -> Curve:
    order = np.argsort(slots, kind="stable")
    return Curve(slots=slots[order], y=y[order])


def _load_flag_fractions(run, index: FragmentIndex, q: Quantity, members: tuple) -> TimelineData:
    """Per-slot fraction of samples a test flagged, over every slot of the period."""
    test = dict(q.extra)["test"]
    slot_a, slot_b = run.period
    slots = np.arange(slot_a, slot_b, dtype=np.int64)
    starts = slots * SAMPLES_PER_SLOT
    f = pa_dataset.field
    curves = {}
    for b in members:
        frame = index.read("flags", booms=[b], filter=(f("test") == test) & (f("boom") == b))
        store = FlagStore(alias_bounds_variables(frame))
        curves[b] = Curve(slots=slots, y=store.fraction(test, b, q.variable, starts, starts + SAMPLES_PER_SLOT))
    return TimelineData(quantity=q, variant="none", curves=curves)


# --- overlays -------------------------------------------------------------------------

def night(run, index: FragmentIndex) -> Curve | None:
    """1.0 at night, from tertiary's slot_final (None without a tertiary stage)."""
    if "tertiary" not in run.stages:
        return None
    df = index.read("slot_final", columns=["slot", "value"], filter=pa_dataset.field("variable") == "night")
    return _curve(df["slot"].to_numpy(dtype=np.int64), df["value"].to_numpy(dtype=np.float64))


def stability(run, index: FragmentIndex) -> tuple[Curve, list[str]] | None:
    """Stability class codes per slot (from the configured Ri_b pair) and the class names."""
    if "tertiary" not in run.stages or run.cfg is None:
        return None
    boom, boom2 = run.cfg.post.stability.pair
    f = pa_dataset.field
    pairs = index.read("pairs", columns=["slot", "boom", "boom2", "variable", "value"],
                       filter=(f("variable") == "rib") & (f("boom") == boom) & (f("boom2") == boom2))
    classes = stability_classes(pairs, run.cfg.post)
    names = [name for name, _ in run.cfg.post.stability.classes]
    codes = pd.Categorical(classes.to_numpy(dtype=object), categories=names).codes.astype(np.float64)
    codes[codes < 0] = np.nan
    return _curve(classes.index.to_numpy(dtype=np.int64), codes), names


def availability(run, index: FragmentIndex, booms) -> tuple[np.ndarray, np.ndarray]:
    """(slots, image): image[i, j] is the status code (index into SLOT_STATUSES,
    or -1 where absent) of booms[i] at slots[j].
    """
    slot_a, slot_b = run.period
    slots = np.arange(slot_a, slot_b, dtype=np.int64)
    df = index.read("slot_boom", booms=list(booms), columns=["slot", "boom", "status"], include_other=True)
    image = np.full((len(booms), slots.size), -1, dtype=np.int8)
    if not df.empty:
        codes = pd.Categorical(df["status"].astype(object), categories=SLOT_STATUSES).codes
        col = df["slot"].to_numpy(dtype=np.int64) - slot_a
        ok = (col >= 0) & (col < slots.size)
        row_of = {b: i for i, b in enumerate(booms)}
        rows = df["boom"].map(row_of).to_numpy()
        ok &= ~pd.isna(rows)
        image[rows[ok].astype(np.int64), col[ok]] = codes[ok]
    return slots, image
