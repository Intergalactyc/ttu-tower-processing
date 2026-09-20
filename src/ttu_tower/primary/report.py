"""Primary QC/yield reports: file, slot, flag, coverage and second-layer CSVs."""
import pandas as pd

from ttu_tower.timegrid import SAMPLES_PER_SLOT


def _month(series: pd.Series) -> pd.Series:
    """The calendar month of each (already run-local) timestamp, as 'YYYY-MM'."""
    return series.dt.tz_localize(None).dt.to_period("M").astype(str)


def files_report(files_df: pd.DataFrame) -> pd.DataFrame:
    """File status counts, overall and by month of `file_start`."""
    overall = files_df["status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
    overall.insert(0, "month", "overall")

    dated = files_df.dropna(subset=["file_start"]).copy()
    dated["month"] = _month(dated["file_start"])
    by_month = dated.groupby(["month", "status"], observed=True).size().reset_index(name="count")
    by_month = by_month[by_month["count"] > 0]

    return pd.concat([overall, by_month], ignore_index=True)[["month", "status", "count"]]


def slots_report(slot_boom_df: pd.DataFrame, slots_df: pd.DataFrame) -> pd.DataFrame:
    """slot_boom status counts per boom, overall and by month of `slot_start`."""
    merged = slot_boom_df.merge(slots_df[["slot", "slot_start"]], on="slot", how="left")

    overall = merged.groupby(["boom", "status"], observed=True).size().reset_index(name="count")
    overall.insert(1, "month", "overall")

    dated = merged.dropna(subset=["slot_start"]).copy()
    dated["month"] = _month(dated["slot_start"])
    by_month = dated.groupby(["boom", "month", "status"], observed=True).size().reset_index(name="count")

    return pd.concat([overall, by_month], ignore_index=True)[["boom", "month", "status", "count"]]


def flags_report(flags_df: pd.DataFrame, coverage_df: pd.DataFrame, slots_df: pd.DataFrame) -> pd.DataFrame:
    """Per boom x variable x test: flagged-sample fraction of present samples,
    overall and by month (a flag interval is assigned to the month of its
    start sample).
    """
    present = coverage_df[coverage_df["layer"] == "present"].merge(slots_df[["slot", "slot_start"]], on="slot", how="left")
    present = present.assign(samples=present["fraction"] * SAMPLES_PER_SLOT, month=_month(present["slot_start"]))

    flags = flags_df.copy()
    flags["slot"] = flags["start"] // SAMPLES_PER_SLOT
    flags = flags.merge(slots_df[["slot", "slot_start"]], on="slot", how="left")
    flags["length"] = flags["end"] - flags["start"]
    flags["month"] = _month(flags["slot_start"])
    boom_level = flags["variable"].isna()

    rows = []
    for (boom, variable), grp in present.groupby(["boom", "variable"], observed=True):
        present_total = grp["samples"].sum()
        present_by_month = grp.groupby("month", observed=True)["samples"].sum()

        var_flags = flags[(flags["boom"] == boom) & ((flags["variable"] == variable) | boom_level)]
        for test, fgrp in var_flags.groupby("test", observed=True):
            flagged_total = fgrp["length"].sum()
            rows.append((boom, variable, str(test), "overall", flagged_total / present_total if present_total else float("nan")))
            for month, mgrp in fgrp.groupby("month", observed=True):
                denom = present_by_month.get(month, 0)
                rows.append((boom, variable, str(test), month, mgrp["length"].sum() / denom if denom else float("nan")))

    return pd.DataFrame(rows, columns=["boom", "variable", "test", "month", "fraction"])


def coverage_report(coverage_df: pd.DataFrame, c: float) -> pd.DataFrame:
    """Per boom x variable x layer: mean, 10/50/90% quantiles, and the
    fraction of computed slots with coverage >= c.
    """
    def agg(g: pd.DataFrame) -> pd.Series:
        f = g["fraction"]
        return pd.Series({
            "mean": f.mean(), "q10": f.quantile(0.10), "q50": f.quantile(0.50), "q90": f.quantile(0.90),
            "frac_ge_c": (f >= c).mean(),
        })

    return coverage_df.groupby(["boom", "variable", "layer"], observed=True).apply(agg, include_groups=False).reset_index()


def second_layer_report(flags_df: pd.DataFrame, slots_df: pd.DataFrame, booms) -> pd.DataFrame:
    """`direction` and `bounce` slot fractions per boom, over every slot of the period."""
    total_slots = len(slots_df)
    rows = []
    for boom in booms:
        for test in ("direction", "bounce"):
            n_flagged = len(flags_df[(flags_df["test"] == test) & (flags_df["boom"] == boom)])
            rows.append((boom, test, n_flagged / total_slots if total_slots else float("nan")))
    return pd.DataFrame(rows, columns=["boom", "test", "fraction"])


def write_primary_reports(stage_dir, reports_dir, cfg) -> None:
    from ttu_tower.io.store import read_table

    files_df = pd.read_parquet(stage_dir / "files.parquet")
    slots_df = pd.read_parquet(stage_dir / "slots.parquet")
    slot_boom_df = read_table(stage_dir / "data" / "slot_boom")
    coverage_df = read_table(stage_dir / "data" / "coverage")
    flags_df = read_table(stage_dir / "data" / "flags")

    reports_dir.mkdir(parents=True, exist_ok=True)
    files_report(files_df).to_csv(reports_dir / "primary_files.csv", index=False)
    slots_report(slot_boom_df, slots_df).to_csv(reports_dir / "primary_slots.csv", index=False)
    flags_report(flags_df, coverage_df, slots_df).to_csv(reports_dir / "primary_flags.csv", index=False)
    coverage_report(coverage_df, cfg.qc.min_coverage).to_csv(reports_dir / "primary_coverage.csv", index=False)
    second_layer_report(flags_df, slots_df, cfg.files.booms).to_csv(reports_dir / "primary_second_layer.csv", index=False)
