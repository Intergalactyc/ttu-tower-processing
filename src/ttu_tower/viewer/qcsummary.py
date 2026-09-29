"""QC statistics over a range of slots, per boom: which slots have data, how
much of each variable survives each QC layer, what each flag test flagged,
what tertiary filtered, and how tau was chosen.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttu_tower.flags import BOOM_LEVEL_VARIABLES, TESTS
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer import flagcounts
from ttu_tower.viewer.slotqc import COVERAGE_VARIABLES, FLAG_VARIABLES, LAYERS
from ttu_tower.viewer.timeline import SLOT_STATUSES

FILE_PRESENT = (SLOT_STATUSES.index("no_data"), SLOT_STATUSES.index("computed"))
COMPUTED = SLOT_STATUSES.index("computed")
TAU_STATUSES = ("found", "fixed", "capped", "fallback", "unresolved", "none")
TAU_FAMILIES = ("momentum", "heat")  # the cospectrum a found, capped or unresolved tau came from


@dataclass
class Stored:
    """Per-slot QC records of every boom over the whole period."""
    slots: np.ndarray  # the period's slots
    coverage: dict  # boom -> DataFrame indexed by slot, columns (variable, layer), float32
    filter_log: pd.DataFrame  # slot, boom, variant, group, criterion
    tau: pd.DataFrame  # slot, boom, variant, source, source_status


def stored(run, index, booms) -> Stored:
    booms = tuple(booms)
    return run.cache("qcsummary", 4).get(("stored", booms), lambda: _stored(run, index, booms))


def _stored(run, index, booms) -> Stored:
    slot_a, slot_b = run.period
    coverage = {}
    for b in booms:
        df = index.read("coverage", booms=[b], columns=["slot", "variable", "layer", "fraction"])
        wide = df.pivot_table(index="slot", columns=["variable", "layer"], values="fraction", aggfunc="first",
                              observed=True)
        coverage[b] = wide.astype(np.float32)
    has = set(run.stages)
    filter_log = (index.read("filter_log", columns=["slot", "boom", "variant", "group", "criterion"])
                  if "tertiary" in has else pd.DataFrame(columns=["slot", "boom", "variant", "group", "criterion"]))
    columns = ["slot", "boom", "variant", "source", "source_status"]
    tau = index.read("tau_selected", columns=columns) if "secondary" in has else pd.DataFrame(columns=columns)
    return Stored(np.arange(slot_a, slot_b, dtype=np.int64), coverage, filter_log, tau)


def flag_counts(run, index, booms) -> dict:
    """boom -> {(test, variable or None): flagged samples per period slot (int32)}."""
    booms = tuple(booms)
    return run.cache("qcsummary", 4).get(("flags", booms), lambda: {b: flagcounts.boom_counts(run, index, b)
                                                                     for b in booms})


# --- tables over a range ------------------------------------------------------------------

def in_range(slots: np.ndarray, slot_range: tuple[int, int] | None) -> np.ndarray:
    if slot_range is None:
        return np.ones(slots.size, dtype=bool)
    return (slots >= slot_range[0]) & (slots < slot_range[1])


def _label(boom: int) -> str:
    from ttu_tower.constants import HEIGHTS
    return f"b{boom} ({HEIGHTS[boom]:g} m)"


def availability_table(slots, image, booms, data: Stored, slot_range) -> pd.DataFrame:
    """Per boom: slots in range, % computed / without usable data / without a
    file, and the mean usable fraction of each family over computed slots.
    """
    sel = in_range(slots, slot_range)
    rows = {}
    for i, b in enumerate(booms):
        status = image[i, sel]
        n = int(sel.sum())
        row = {"slots": n}
        for name, code in (("computed %", COMPUTED), ("no data %", SLOT_STATUSES.index("no_data")),
                           ("no file %", SLOT_STATUSES.index("no_file"))):
            row[name] = 100.0 * np.count_nonzero(status == code) / n if n else np.nan
        row["missing %"] = 100.0 * np.count_nonzero(status < 0) / n if n else np.nan
        computed = slots[sel][status == COMPUTED]
        wide = data.coverage.get(b)
        for fam in ("momentum", "heat"):
            row[f"{fam} usable"] = _mean_at(wide, (fam, "usable"), computed)
        rows[_label(b)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _mean_at(wide: pd.DataFrame | None, column, slots: np.ndarray) -> float:
    if wide is None or column not in wide.columns or slots.size == 0:
        return np.nan
    values = wide[column].reindex(slots).to_numpy(dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def coverage_table(slots, image, booms, data: Stored, variable: str, slot_range) -> pd.DataFrame:
    """Per boom: the mean fraction of `variable`'s samples left at each QC layer,
    over the slots in range that have a file.
    """
    sel = in_range(slots, slot_range)
    rows = {}
    for i, b in enumerate(booms):
        present = slots[sel][np.isin(image[i, sel], FILE_PRESENT)]
        wide = data.coverage.get(b)
        rows[_label(b)] = {layer: _mean_at(wide, (variable, layer), present) for layer in LAYERS}
    return pd.DataFrame.from_dict(rows, orient="index").dropna(axis=1, how="all")


def flag_table(slots, image, booms, counts: dict, variable: str, slot_range) -> pd.DataFrame:
    """Per boom: % of `variable`'s samples each test flagged, over the slots in
    range that have a file (boom-level tests count toward the sonic series).
    """
    sel = in_range(slots, slot_range)
    applies = [t for t, spec in TESTS.items()
               if variable in (spec.variables if spec.variables is not None else BOOM_LEVEL_VARIABLES)]
    rows = {}
    for i, b in enumerate(booms):
        have = sel & np.isin(image[i], FILE_PRESENT)
        denominator = np.count_nonzero(have) * SAMPLES_PER_SLOT
        row = {}
        for test in applies:
            flagged = 0
            for key in ((test, variable), (test, None)):
                c = counts.get(b, {}).get(key)
                if c is not None and (key[1] is not None or variable in BOOM_LEVEL_VARIABLES):
                    flagged += int(c[have].sum())
            row[test] = 100.0 * flagged / denominator if denominator else np.nan
        rows[_label(b)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def filter_table(slots, image, booms, data: Stored, variant: str, slot_range) -> tuple[pd.DataFrame, dict]:
    """Per boom: % of computed slots in range where tertiary filtered each group
    (the variant's own, then the means'), and each cell's split by criterion.
    """
    sel = in_range(slots, slot_range)
    log = data.filter_log
    log = log[log["variant"].isin([variant, "none"])] if not log.empty else log
    columns = []
    for v in (variant, "none"):
        groups = sorted(log.loc[log["variant"] == v, "group"].astype(str).unique()) if not log.empty else []
        columns += [(v, g) for g in groups]
    rows, why = {}, {}
    for i, b in enumerate(booms):
        computed = slots[sel][image[i, sel] == COMPUTED]
        n = computed.size
        mine = log[(log["boom"] == b) & log["slot"].isin(computed)] if not log.empty else log
        row = {}
        for v, g in columns:
            name = _group_label(v, g)
            part = mine[(mine["variant"] == v) & (mine["group"] == g)] if not mine.empty else mine
            row[name] = 100.0 * part["slot"].nunique() / n if n else np.nan
            if not part.empty:
                by = part.groupby("criterion", observed=True)["slot"].nunique()
                why[(_label(b), name)] = ", ".join(f"{c}: {100.0 * k / n:.3g}%" for c, k in by.items())
        rows[_label(b)] = row
    return pd.DataFrame.from_dict(rows, orient="index"), why


def _group_label(variant: str, group: str) -> str:
    return f"{group} (means)" if variant == "none" and group in ("momentum", "ts") else group


def tau_table(slots, image, booms, data: Stored, variant: str, slot_range) -> pd.DataFrame:
    """Per boom: % of computed slots in range by how the selected tau came
    about, split by the cospectrum it came from where it came from one
    ("found · momentum", "found · heat", ...).
    """
    sel = in_range(slots, slot_range)
    tau = data.tau[data.tau["variant"] == variant] if not data.tau.empty else data.tau
    rows = {}
    for i, b in enumerate(booms):
        computed = slots[sel][image[i, sel] == COMPUTED]
        mine = tau[(tau["boom"] == b) & tau["slot"].isin(computed)] if not tau.empty else tau
        status, source = mine["source_status"].astype(str), mine["source"].astype(str)
        counts = (status + np.where(source.isin(TAU_FAMILIES), " · " + source, "")).value_counts()
        n = computed.size
        rows[_label(b)] = {c: 100.0 * counts.get(c, 0) / n if n else np.nan for c in tau_columns()}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    return frame.loc[:, (frame.fillna(0) > 0).any()] if not frame.empty else frame


def tau_columns() -> list[str]:
    out = []
    for s in TAU_STATUSES:
        out += [f"{s} · {f}" for f in TAU_FAMILIES] if s in ("found", "capped", "unresolved") else [s]
    return out


COVERAGE_CHOICES = COVERAGE_VARIABLES
FLAG_CHOICES = FLAG_VARIABLES
