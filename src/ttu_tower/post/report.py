"""Stability reports: per-class breakdowns of yield, flagging, and tau
source/status, from a run's already-written primary and tertiary output.
"""
import logging
from pathlib import Path

import pandas as pd

from ttu_tower.flags import BOOM_LEVEL_VARIABLES, TESTS
from ttu_tower.post.classify import stability_classes
from ttu_tower.primary.report import flagged_present, predates_flagged_counts
from ttu_tower.timegrid import SAMPLES_PER_SLOT

_logger = logging.getLogger(__name__)

_VARIANTS = ("mrd", "naive", "mrd_unexcised")
_YIELD_QUANTITIES = [("ustar", None), ("wvpts", "cov"), ("ti", None)]


def stability_yield_report(classes: pd.Series, boom_final: pd.DataFrame, slot_boom: pd.DataFrame,
                            variants=_VARIANTS) -> pd.DataFrame:
    """Per stability class x boom x variant x quantity: how many of that
    boom's `computed` slots have a non-NaN (filtered-survived) value, of the
    slots that fall in this class.
    """
    computed = slot_boom[slot_boom["status"] == "computed"][["slot", "boom"]].copy()
    computed["stability_class"] = computed["slot"].map(classes)
    computed = computed.dropna(subset=["stability_class"])

    rows = []
    for variant in variants:
        for variable, stat in _YIELD_QUANTITIES:
            mask = (boom_final["variant"] == variant) & (boom_final["variable"] == variable)
            mask &= boom_final["stat"].isna() if stat is None else (boom_final["stat"] == stat)
            values = boom_final.loc[mask, ["slot", "boom", "value"]]

            merged = computed.merge(values, on=["slot", "boom"], how="left")
            merged["survived"] = merged["value"].notna()
            grouped = merged.groupby(["stability_class", "boom"], observed=True).agg(
                n_computed=("survived", "size"), n_survived=("survived", "sum"),
            ).reset_index()
            grouped["variant"] = variant
            grouped["quantity"] = variable if stat is None else f"{variable}_{stat}"
            grouped["fraction"] = grouped["n_survived"] / grouped["n_computed"]
            rows.append(grouped)

    result = pd.concat(rows, ignore_index=True)
    return result[["stability_class", "boom", "variant", "quantity", "n_computed", "n_survived", "fraction"]]


def stability_flags_report(classes: pd.Series, slot_qc_df: pd.DataFrame, coverage_df: pd.DataFrame, booms) -> pd.DataFrame:
    """Per stability class x boom x test: the fraction of present samples the
    test flagged, within that class. Numerator (`flagged_<test>` rows) and
    denominator (`present` coverage) both sum over exactly the variables the
    test applies to (`BOOM_LEVEL_VARIABLES` for a boom-level test) and count
    present samples only.
    """
    present = coverage_df[coverage_df["layer"] == "present"].copy()
    present["stability_class"] = present["slot"].map(classes)
    present = present.dropna(subset=["stability_class"])
    present["samples"] = present["fraction"] * SAMPLES_PER_SLOT

    flagged = flagged_present(slot_qc_df)
    flagged["stability_class"] = flagged["slot"].map(classes)
    flagged = flagged.dropna(subset=["stability_class"])

    rows = []
    for boom in booms:
        boom_present = present[present["boom"] == boom]
        boom_flagged = flagged[flagged["boom"] == boom]
        for test, fgrp in boom_flagged.groupby("test"):
            test_variables = TESTS[test].variables or BOOM_LEVEL_VARIABLES
            denom_by_class = boom_present[boom_present["variable"].isin(test_variables)].groupby(
                "stability_class", observed=True,
            )["samples"].sum()
            for cls, cgrp in fgrp.groupby("stability_class"):
                denom = denom_by_class.get(cls, 0)
                rows.append((cls, boom, test, cgrp["samples"].sum() / denom if denom else float("nan")))

    return pd.DataFrame(rows, columns=["stability_class", "boom", "test", "fraction"])


def stability_tau_report(classes: pd.Series, tau_final: pd.DataFrame, variants=_VARIANTS) -> pd.DataFrame:
    """Per stability class x boom x variant: the tau source/status
    distribution (count and fraction of that group's slots) and the median
    selected tau, repeated on every row of the group for convenience.
    """
    tf = tau_final[tau_final["variant"].isin(variants)].copy()
    tf["stability_class"] = tf["slot"].map(classes)
    tf = tf.dropna(subset=["stability_class"])

    rows = []
    for (cls, boom, variant), grp in tf.groupby(["stability_class", "boom", "variant"], observed=True):
        median_tau = grp["tau_s"].median()
        total = len(grp)
        for (source, status), count in grp.groupby(["source", "source_status"], observed=True).size().items():
            rows.append((cls, boom, variant, str(source), str(status), int(count), count / total, median_tau))

    return pd.DataFrame(rows, columns=[
        "stability_class", "boom", "variant", "source", "source_status", "count", "fraction", "median_tau_s",
    ])


def write_stability_reports(run_dir: Path, reports_dir: Path, cfg) -> None:
    from ttu_tower.io.store import read_table

    tertiary_dir = Path(run_dir) / "tertiary"
    primary_dir = Path(run_dir) / "primary"

    pairs = read_table(tertiary_dir / "data" / "pairs")
    classes = stability_classes(pairs, cfg.post)

    boom_final = read_table(tertiary_dir / "data" / "boom_final")
    tau_final = read_table(tertiary_dir / "data" / "tau_final")
    slot_boom = read_table(primary_dir / "data" / "slot_boom")
    coverage = read_table(primary_dir / "data" / "coverage")
    slot_qc = read_table(primary_dir / "data" / "slot_qc")
    flags = read_table(primary_dir / "data" / "flags")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stability_yield_report(classes, boom_final, slot_boom).to_csv(reports_dir / "stability_yield.csv", index=False)
    if predates_flagged_counts(slot_qc, flags):
        _logger.warning("stability_flags.csv skipped: this run predates the per-slot flagged_<test> counts; rerun primary")
    else:
        stability_flags_report(classes, slot_qc, coverage, cfg.files.booms).to_csv(reports_dir / "stability_flags.csv",
                                                                                   index=False)
    stability_tau_report(classes, tau_final).to_csv(reports_dir / "stability_tau.csv", index=False)
