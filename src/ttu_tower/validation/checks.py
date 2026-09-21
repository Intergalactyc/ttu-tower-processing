"""First-run validation checks: each function returns `{csv_name:
DataFrame}`. `write_validation_reports` runs the requested checks (default:
all) and writes their CSVs under `<run dir>/reports/validation/<check>/`.
"""
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.flags import mask_to_intervals
from ttu_tower.io.load import BadFileError, load_boom
from ttu_tower.io.rawfiles import build_file_table
from ttu_tower.io.store import read_table
from ttu_tower.math.polar import signed_angular_distance
from ttu_tower.math.stats import autocovariance, efolding_integral, linear_detrend
from ttu_tower.post.classify import stability_classes
from ttu_tower.primary.despike import despike
from ttu_tower.primary.gaps import fill_short
from ttu_tower.primary.ladder import RUNG_SECONDS
from ttu_tower.primary.report import flags_report
from ttu_tower.primary.stage_a import stage_a
from ttu_tower.validation.synthetic import correlated_ou, inject_spikes, ou

CHECK_NAMES = (
    "tau_agreement", "despike_calibration", "its_bias", "interp_bias",
    "floor_peak", "yield", "tau_profile", "mrd_vs_naive", "sanity",
)
_DESPIKED_VARS = ("ue", "vn", "w", "ts", "t", "rh", "p")
_RATIO_QUANTITIES = [("ustar", None), ("wvpts", "cov"), ("ti", None),
                      ("sigma_u", None), ("ils_u", None), ("zeta", None)]


def _add_class(df: pd.DataFrame, classes: pd.Series, slot_col: str = "slot") -> pd.DataFrame:
    df = df.copy()
    df["stability_class"] = df[slot_col].map(classes)
    return df.dropna(subset=["stability_class"])


def _quantity_mask(df: pd.DataFrame, variant: str, variable: str, stat: str | None) -> pd.Series:
    mask = (df["variant"] == variant) & (df["variable"] == variable)
    return mask & (df["stat"].isna() if stat is None else df["stat"] == stat)


# --- 3.1 tau_agreement -------------------------------------------------------------


def tau_agreement(tau: pd.DataFrame, tau_selected: pd.DataFrame, classes: pd.Series) -> dict[str, pd.DataFrame]:
    mrd = tau[tau["variant"] == "mrd"]
    heat = mrd[mrd["cospectrum"] == "heat"][["slot", "boom", "status", "tau_s"]]
    heat = heat.rename(columns={"status": "heat_status", "tau_s": "heat_tau"})
    momentum = mrd[mrd["cospectrum"] == "momentum"][["slot", "boom", "status", "tau_s"]]
    momentum = momentum.rename(columns={"status": "momentum_status", "tau_s": "momentum_tau"})
    merged = _add_class(heat.merge(momentum, on=["slot", "boom"], how="inner"), classes)

    active_status = ("found", "capped")
    active = merged[merged["heat_status"].isin(active_status) & merged["momentum_status"].isin(active_status)].copy()
    active["log2_ratio"] = np.log2(active["heat_tau"] / active["momentum_tau"])
    rung_index = {r: i for i, r in enumerate(RUNG_SECONDS)}
    active["rung_diff"] = active["heat_tau"].map(rung_index) - active["momentum_tau"].map(rung_index)
    active["agree_exact"] = active["heat_tau"] == active["momentum_tau"]
    active["agree_within_1"] = active["rung_diff"].abs() <= 1

    agreement = active.groupby(["stability_class", "boom"]).agg(
        n=("log2_ratio", "size"), mean_log2_ratio=("log2_ratio", "mean"),
        median_log2_ratio=("log2_ratio", "median"), frac_exact=("agree_exact", "mean"),
        frac_within_1_rung=("agree_within_1", "mean"),
    ).reset_index()

    sel = _add_class(tau_selected[tau_selected["variant"] == "mrd"], classes)
    source_rates = sel.groupby(["stability_class", "boom", "source"], observed=True).size().rename("count").reset_index()
    source_rates["fraction"] = source_rates["count"] / source_rates.groupby(["stability_class", "boom"])["count"].transform("sum")

    status_dist = _add_class(mrd, classes)
    status_counts = status_dist.groupby(["stability_class", "boom", "cospectrum", "status"], observed=True).size()
    status_counts = status_counts.rename("count").reset_index()
    status_counts["fraction"] = status_counts["count"] / status_counts.groupby(
        ["stability_class", "boom", "cospectrum"], observed=True,
    )["count"].transform("sum")

    return {"agreement": agreement, "selected_source": source_rates, "status_distribution": status_counts}


# --- 3.2 despike_calibration --------------------------------------------------------


def _old_despike(x: np.ndarray, z_threshold: float = 5.0) -> np.ndarray:
    """Old-pipeline-style despiking, reimplemented here for comparison only:
    one whole-file linear detrend, one median/MAD pass, no duration rule
    (every outlier sample is removed however long a run it's part of).
    """
    detrended = linear_detrend(x)
    finite = detrended[np.isfinite(detrended)]
    if finite.size == 0:
        return np.zeros(x.size, dtype=bool)
    median = np.median(finite)
    sigma = max(np.median(np.abs(finite - median)), 1e-12) / 0.6745
    return np.abs(detrended - median) / sigma > z_threshold


def _file_stability_class(half_hour: int, classes: pd.Series):
    labels = [classes.get(3 * half_hour + i) for i in range(3)]
    labels = [label for label in labels if pd.notna(label)]
    return pd.Series(labels).mode().iloc[0] if labels else None


def _stratified_files(file_table: pd.DataFrame, classes: pd.Series, rng: np.random.Generator, n_per_class: int) -> pd.DataFrame:
    accepted = file_table[file_table["status"] == "accepted"].copy()
    accepted["stability_class"] = accepted["half_hour"].map(lambda h: _file_stability_class(int(h), classes))
    accepted = accepted.dropna(subset=["stability_class"])
    chosen = []
    for _, group in accepted.groupby("stability_class"):
        n = min(n_per_class, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        chosen.append(group.loc[idx])
    return pd.concat(chosen, ignore_index=True) if chosen else accepted


def _despike_one_file(task: dict) -> dict[str, list]:
    """One sampled file's worth of `despike_calibration` work, across every
    boom and despiked variable - the unit of work `ProcessPoolExecutor`
    farms out, since files are independent and this is the expensive part.
    """
    cfg, path, file_name, stability_class = task["cfg"], task["path"], task["file_name"], task["stability_class"]
    booms, thresholds, inject_sigmas = task["booms"], task["thresholds"], task["inject_sigmas"]
    rng = np.random.default_rng(task["seed"])

    rate_rows, old_rows, excursion_rows, excursion_examples, injection_rows = [], [], [], [], []

    with tempfile.TemporaryDirectory(prefix="ttu_validate_") as temp_dir:
        for boom in booms:
            try:
                raw = load_boom(path, boom, temp_dir)
            except BadFileError:
                continue
            si = stage_a(raw, boom, cfg.qc)

            for var in _DESPIKED_VARS:
                x = getattr(si, var)
                min_mad = cfg.qc.despike.min_mad[var]
                configured_threshold = cfg.qc.despike.z_threshold[var]

                for t in thresholds:
                    result = despike(x, 0, cfg.qc.despike, min_mad, t, cfg.qc.min_coverage)
                    rate_rows.append((stability_class, boom, var, t,
                                      result.spike.sum() / x.size, result.excursion.sum() / x.size))
                    if t == configured_threshold:
                        starts, ends = mask_to_intervals(result.excursion, g0=0)
                        for s, e in zip(starts, ends):
                            length = int(e - s)
                            excursion_rows.append((stability_class, boom, var, length))
                            if 4 <= length <= 10:
                                lo, hi = max(0, s - 5), min(x.size, e + 5)
                                excursion_examples.append({
                                    "stability_class": stability_class, "boom": boom, "variable": var,
                                    "file": file_name, "start": int(s), "end": int(e), "length": length,
                                    "values": list(x[lo:hi]),
                                })

                old_rows.append((stability_class, boom, var, _old_despike(x).sum() / x.size))

                baseline = despike(x, 0, cfg.qc.despike, min_mad, configured_threshold, cfg.qc.min_coverage)
                clean = np.flatnonzero(~baseline.spike & ~baseline.excursion & ~baseline.unchecked & np.isfinite(x))
                clean = clean[clean < x.size - 3]  # leave room for a 1-3-sample injected run
                if clean.size < 200:
                    continue
                n_inject = min(20, clean.size // 200)
                for amp in inject_sigmas:
                    positions = rng.choice(clean, size=n_inject, replace=False)
                    lengths = rng.integers(1, 4, size=n_inject)
                    injected = inject_spikes(x, positions, lengths, np.full(n_inject, amp))
                    for t in thresholds:
                        result = despike(injected, 0, cfg.qc.despike, min_mad, t, cfg.qc.min_coverage)
                        detected = sum(result.spike[p:p + l].any() for p, l in zip(positions, lengths))
                        outside = np.ones(x.size, dtype=bool)
                        for p, l in zip(positions, lengths):
                            outside[p:p + l] = False
                        injection_rows.append((stability_class, boom, var, amp, t,
                                                detected / n_inject, (result.spike & outside).sum() / outside.sum()))

    return {"rates": rate_rows, "old": old_rows, "excursions": excursion_rows,
            "examples": excursion_examples, "injection": injection_rows}


def despike_calibration(cfg, run_dir: Path, rng: np.random.Generator, n_per_class: int = 30,
                         thresholds=(3.5, 4.0, 4.5, 5.0, 6.0), inject_sigmas=(4.0, 6.0, 8.0, 12.0),
                         nproc: int | None = None) -> dict[str, pd.DataFrame]:
    """Every sampled file is independent work, farmed out across `nproc`
    processes (default: every available core) the same way primary
    parallelizes over (batch, boom) - this check sweeps enough
    threshold/injection combinations per file to be the slowest part of a
    validation run otherwise.
    """
    classes = stability_classes(read_table(Path(run_dir) / "tertiary" / "data" / "pairs"), cfg.post)
    file_table = build_file_table(cfg.paths.raw_dirs, cfg.files)
    sample = _stratified_files(file_table, classes, rng, n_per_class)

    seeds = rng.integers(0, 2**31 - 1, size=len(sample))
    tasks = [
        {
            "cfg": cfg, "path": row.path, "file_name": row.name, "stability_class": row.stability_class,
            "booms": cfg.files.booms, "thresholds": thresholds, "inject_sigmas": inject_sigmas, "seed": int(seed),
        }
        for row, seed in zip(sample.itertuples(index=False), seeds)
    ]

    nproc = nproc or os.cpu_count() or 1
    results: list[dict[str, list]]
    if nproc > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(nproc, len(tasks)), mp_context=ctx) as executor:
            results = list(executor.map(_despike_one_file, tasks))
    else:
        results = [_despike_one_file(task) for task in tasks]

    rate_rows = [row for r in results for row in r["rates"]]
    old_rows = [row for r in results for row in r["old"]]
    excursion_rows = [row for r in results for row in r["excursions"]]
    excursion_examples = [row for r in results for row in r["examples"]]
    injection_rows = [row for r in results for row in r["injection"]]

    return {
        "removal_rates": pd.DataFrame(rate_rows, columns=["stability_class", "boom", "variable", "z_threshold", "spike_fraction", "excursion_fraction"]),
        "old_method": pd.DataFrame(old_rows, columns=["stability_class", "boom", "variable", "old_spike_fraction"]),
        "excursion_lengths": pd.DataFrame(excursion_rows, columns=["stability_class", "boom", "variable", "length"]),
        "excursion_examples": pd.DataFrame(excursion_examples),
        "injection": pd.DataFrame(injection_rows, columns=[
            "stability_class", "boom", "variable", "injected_sigma", "z_threshold", "detection_rate", "false_removal_rate",
        ]),
    }


# --- 3.3 its_bias --------------------------------------------------------------------


def _pooled_its(x: np.ndarray, block_samples: int, max_lag_fraction: float) -> float:
    n_blocks = x.size // block_samples
    if n_blocks == 0:
        return float("nan")
    max_lag = int(np.floor(max_lag_fraction * block_samples))
    pooled = np.zeros(max_lag + 1)
    for b in range(n_blocks):
        block = x[b * block_samples:(b + 1) * block_samples]
        pooled += autocovariance(block - block.mean(), max_lag)
    if pooled[0] == 0:
        return float("nan")
    return (1.0 / SAMPLE_HZ) * efolding_integral(pooled / pooled[0])


def _its_bias_synthetic(cfg, rng: np.random.Generator, T_values, n_realizations: int, n_top_blocks: int) -> pd.DataFrame:
    max_lag_fraction = cfg.ladder.its_max_lag_fraction
    n = n_top_blocks * round(RUNG_SECONDS[-1] * SAMPLE_HZ)
    rows = []
    for T in T_values:
        for _ in range(n_realizations):
            x = ou(n, T, 1.0, rng)
            for rung_s in RUNG_SECONDS:
                its = _pooled_its(x, round(rung_s * SAMPLE_HZ), max_lag_fraction)
                expected = (1.0 - np.exp(-1.0)) * T
                rows.append((T, rung_s, rung_s / T, its, its / expected if expected else np.nan))
    return pd.DataFrame(rows, columns=["T", "rung_s", "tau_over_T", "its", "its_ratio"])


def _its_bias_real(boom_stats: pd.DataFrame, tau_selected: pd.DataFrame, classes: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    mrd_tau = tau_selected[tau_selected["variant"] == "mrd"][["slot", "boom", "tau_s"]]
    its = boom_stats[(boom_stats["variant"] == "mrd") & (boom_stats["stat"] == "its")]
    merged = its.merge(mrd_tau, on=["slot", "boom"], how="inner")
    merged["ratio"] = merged["tau_s"] / merged["value"]
    merged = _add_class(merged, classes)
    ratio_summary = merged.groupby(["stability_class", "boom", "variable"], observed=True)["ratio"].agg(
        n="size", mean="mean", median="median", q10=lambda s: s.quantile(0.1), q90=lambda s: s.quantile(0.9),
    ).reset_index()

    short = boom_stats[(boom_stats["variant"] == "mrd") & boom_stats["variable"].isin(["its_short", "its_short_vpts"]) & boom_stats["stat"].isna()]
    short = _add_class(short, classes)
    short_rates = short.groupby(["stability_class", "boom", "variable"], observed=True)["value"].mean().rename("short_rate").reset_index()
    return ratio_summary, short_rates


def its_bias(cfg, boom_stats: pd.DataFrame, tau_selected: pd.DataFrame, classes: pd.Series, rng: np.random.Generator,
             T_values=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0), n_realizations: int = 20, n_top_blocks: int = 20) -> dict[str, pd.DataFrame]:
    synthetic = _its_bias_synthetic(cfg, rng, T_values, n_realizations, n_top_blocks)
    ratio_summary, short_rates = _its_bias_real(boom_stats, tau_selected, classes)
    return {"synthetic": synthetic, "real_ratio": ratio_summary, "real_short_rates": short_rates}


# --- 3.4 interp_bias -----------------------------------------------------------------


def _block_var_cov(x: np.ndarray, y: np.ndarray, block_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_blocks = x.size // block_samples
    var_x = np.full(n_blocks, np.nan)
    var_y = np.full(n_blocks, np.nan)
    cov_xy = np.full(n_blocks, np.nan)
    for b in range(n_blocks):
        sl = slice(b * block_samples, (b + 1) * block_samples)
        xb, yb = x[sl], y[sl]
        if np.isnan(xb).any() or np.isnan(yb).any():
            continue
        var_x[b], var_y[b] = np.var(xb), np.var(yb)
        cov_xy[b] = np.cov(xb, yb)[0, 1]
    return var_x, var_y, cov_xy


def _blank_random_gaps(x: np.ndarray, fraction: float, rng: np.random.Generator, max_gap: int = 50) -> np.ndarray:
    x = x.copy()
    target = int(fraction * x.size)
    filled, guard = 0, 0
    while filled < target and guard < 20_000:
        guard += 1
        length = int(rng.integers(1, max_gap + 1))
        start = int(rng.integers(1, x.size - length - 1))
        if np.isnan(x[start - 1:start + length + 1]).any():
            continue
        x[start:start + length] = np.nan
        filled += length
    return x


def _interp_bias_from_series(x_true: np.ndarray, y_true: np.ndarray, cfg, rng: np.random.Generator,
                              fractions, n_realizations: int) -> pd.DataFrame:
    rows = []
    for fraction in fractions:
        for _ in range(n_realizations):
            x_filled, _ = fill_short(_blank_random_gaps(x_true, fraction, rng), cfg.qc.max_fill_gap_samples)
            y_filled, _ = fill_short(_blank_random_gaps(y_true, fraction, rng), cfg.qc.max_fill_gap_samples)
            for rung_s in RUNG_SECONDS:
                block_samples = round(rung_s * SAMPLE_HZ)
                var_x_t, var_y_t, cov_t = _block_var_cov(x_true, y_true, block_samples)
                var_x_f, var_y_f, cov_f = _block_var_cov(x_filled, y_filled, block_samples)
                rows.append((fraction, rung_s, np.nanmean(var_x_f - var_x_t),
                             np.nanmean(var_y_f - var_y_t), np.nanmean(cov_f - cov_t)))
    return pd.DataFrame(rows, columns=["gap_fraction", "rung_s", "var_x_bias", "var_y_bias", "cov_bias"])


def interp_bias(cfg, run_dir: Path, rng: np.random.Generator, fractions=(0.001, 0.005, 0.01, 0.02, 0.05),
                 n_realizations: int = 10, n_top_blocks: int = 20) -> dict[str, pd.DataFrame]:
    n = n_top_blocks * round(RUNG_SECONDS[-1] * SAMPLE_HZ)
    synthetic_series = correlated_ou(n, 1.0, [1.0, 1.0], [[1.0, -0.3], [-0.3, 1.0]], rng)
    synthetic = _interp_bias_from_series(synthetic_series[0], synthetic_series[1], cfg, rng, fractions, n_realizations)

    file_table = build_file_table(cfg.paths.raw_dirs, cfg.files)
    accepted = file_table[file_table["status"] == "accepted"]
    real = pd.DataFrame(columns=["gap_fraction", "rung_s", "var_x_bias", "var_y_bias", "cov_bias"])
    if not accepted.empty:
        sample_rows = accepted.sample(n=min(5, len(accepted)), random_state=int(rng.integers(0, 2**31 - 1)))
        real_parts = []
        with tempfile.TemporaryDirectory(prefix="ttu_validate_") as temp_dir:
            for row in sample_rows.itertuples(index=False):
                for boom in cfg.files.booms:
                    try:
                        raw = load_boom(row.path, boom, temp_dir)
                    except BadFileError:
                        continue
                    si = stage_a(raw, boom, cfg.qc)
                    if np.isnan(si.ue).any() or np.isnan(si.w).any():
                        continue
                    real_parts.append(_interp_bias_from_series(si.ue, si.w, cfg, rng, fractions, n_realizations=1))
        if real_parts:
            real = pd.concat(real_parts, ignore_index=True).groupby(["gap_fraction", "rung_s"], as_index=False).mean()

    return {"synthetic": synthetic, "real": real}


# --- 3.5 floor_peak --------------------------------------------------------------------


def floor_peak(mrd: pd.DataFrame, tau: pd.DataFrame, classes: pd.Series, cfg, booms=(1, 2, 3)) -> dict[str, pd.DataFrame]:
    mrd_mrd = mrd[mrd["variant"] == "mrd"]
    smallest = mrd_mrd.groupby(["slot", "boom", "spectrum"], observed=True)["scale_s"].min().reset_index()
    smallest = smallest.rename(columns={"spectrum": "cospectrum", "scale_s": "smallest_scale_s"})

    tau_mrd = _add_class(tau[(tau["variant"] == "mrd") & (tau["boom"].isin(booms))], classes)
    tau_mrd = tau_mrd.merge(smallest, on=["slot", "boom", "cospectrum"], how="left")
    tau_mrd["at_floor"] = tau_mrd["peak_scale_s"] == tau_mrd["smallest_scale_s"]

    peak_dist = tau_mrd.groupby(["stability_class", "boom", "cospectrum"], observed=True)["peak_scale_s"].describe().reset_index()
    floor_rate = tau_mrd.groupby(["stability_class", "boom", "cospectrum"], observed=True)["at_floor"].mean()
    floor_rate = floor_rate.rename("floor_fraction").reset_index()

    found = _add_class(tau[(tau["variant"] == "mrd") & (tau["status"] == "found")], classes).copy()
    found["clipped"] = found["reversal_scale_s"] <= cfg.secondary.detection.min_tau_s + 1e-9
    clip_rate = found.groupby(["stability_class", "boom", "cospectrum"], observed=True)["clipped"].mean()
    clip_rate = clip_rate.rename("clip_fraction").reset_index()

    return {"peak_scale_distribution": peak_dist, "floor_rate": floor_rate, "clip_rate": clip_rate}


# --- 3.6 yield ---------------------------------------------------------------------------


def _survival_by_variant(boom_final: pd.DataFrame, slot_boom: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    computed = _add_class(slot_boom[slot_boom["status"] == "computed"][["slot", "boom"]], classes)
    rows = []
    for variant in boom_final["variant"].unique():
        any_value = boom_final[boom_final["variant"] == variant].groupby(["slot", "boom"])["value"].apply(
            lambda s: bool(s.notna().any()),
        ).rename("survived")
        merged = computed.merge(any_value.reset_index(), on=["slot", "boom"], how="left")
        merged["survived"] = merged["survived"].fillna(False)
        grouped = merged.groupby(["stability_class", "boom"]).agg(
            n_computed=("survived", "size"), n_survived=("survived", "sum"),
        ).reset_index()
        grouped["variant"] = variant
        grouped["fraction"] = grouped["n_survived"] / grouped["n_computed"]
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def yield_report(primary_dir: Path, tertiary_dir: Path, classes: pd.Series) -> dict[str, pd.DataFrame]:
    flags_df = read_table(Path(primary_dir) / "data" / "flags")
    coverage_df = read_table(Path(primary_dir) / "data" / "coverage")
    slots_df = pd.read_parquet(Path(primary_dir) / "slots.parquet")
    slot_boom = read_table(Path(primary_dir) / "data" / "slot_boom")
    boom_final = read_table(Path(tertiary_dir) / "data" / "boom_final")

    return {
        "flag_rates": flags_report(flags_df, coverage_df, slots_df),
        "survival_by_variant": _survival_by_variant(boom_final, slot_boom, classes),
    }


# --- 3.7 tau_profile -----------------------------------------------------------------------


def tau_profile(tau_selected: pd.DataFrame, classes: pd.Series) -> dict[str, pd.DataFrame]:
    mrd = _add_class(tau_selected[tau_selected["variant"] == "mrd"], classes)

    profile = mrd.groupby(["stability_class", "boom"])["tau_s"].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
    ).reset_index()

    source_mix = mrd.groupby(["stability_class", "boom", "source"], observed=True).size().rename("count").reset_index()
    source_mix["fraction"] = source_mix["count"] / source_mix.groupby(["stability_class", "boom"])["count"].transform("sum")

    return {"tau_profile": profile, "source_mix": source_mix}


# --- 3.8 mrd_vs_naive ----------------------------------------------------------------------


def _ratio_table(boom_final: pd.DataFrame, classes: pd.Series, variant_a: str, variant_b: str, quantities) -> pd.DataFrame:
    rows = []
    for variable, stat in quantities:
        a = boom_final.loc[_quantity_mask(boom_final, variant_a, variable, stat), ["slot", "boom", "value"]]
        b = boom_final.loc[_quantity_mask(boom_final, variant_b, variable, stat), ["slot", "boom", "value"]]
        merged = a.rename(columns={"value": "a"}).merge(b.rename(columns={"value": "b"}), on=["slot", "boom"], how="inner")
        merged["ratio"] = merged["a"] / merged["b"]
        merged = _add_class(merged.dropna(subset=["ratio"]), classes)
        if merged.empty:
            continue
        grouped = merged.groupby(["stability_class", "boom"])["ratio"].agg(
            n="size", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
        ).reset_index()
        grouped["quantity"] = variable if stat is None else f"{variable}_{stat}"
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["stability_class", "boom", "n", "median", "q25", "q75", "quantity"],
    )


def mrd_vs_naive(boom_final: pd.DataFrame, slot_boom: pd.DataFrame, classes: pd.Series) -> dict[str, pd.DataFrame]:
    vs_naive = _ratio_table(boom_final, classes, "mrd", "naive", _RATIO_QUANTITIES)

    unexcised_pairs = slot_boom[slot_boom["unexcised_computed"]][["slot", "boom"]]
    boom_final_uc = boom_final.merge(unexcised_pairs, on=["slot", "boom"], how="inner")
    vs_unexcised = _ratio_table(boom_final_uc, classes, "mrd", "mrd_unexcised", _RATIO_QUANTITIES)

    return {"mrd_vs_naive": vs_naive, "mrd_vs_unexcised": vs_unexcised}


# --- 3.9 sanity ----------------------------------------------------------------------------


def sanity(slow: pd.DataFrame, tau_selected: pd.DataFrame, slot_boom: pd.DataFrame, mrd: pd.DataFrame,
           mrd_frame: pd.DataFrame, means: pd.DataFrame, classes: pd.Series) -> dict[str, pd.DataFrame]:
    p_measured = _add_class(slow[slow["variable"] == "p_measured"], classes)
    p_measured_rate = p_measured.groupby(["stability_class", "boom"])["value"].apply(lambda s: (s == 0).mean())
    p_measured_rate = p_measured_rate.rename("p_measured_zero_rate").reset_index()

    p_factor = _add_class(slow[slow["variable"] == "p_factor"], classes)
    p_factor_dist = p_factor.groupby(["stability_class", "boom"])["value"].describe().reset_index()

    mrd_sel = _add_class(tau_selected[tau_selected["variant"] == "mrd"], classes)
    status_rates = mrd_sel.groupby(["stability_class", "boom", "source_status"], observed=True).size().rename("count").reset_index()
    status_rates["fraction"] = status_rates["count"] / status_rates.groupby(["stability_class", "boom"])["count"].transform("sum")

    computed = slot_boom[slot_boom["status"] == "computed"]
    unexcised_rate = computed.groupby("boom")["unexcised_computed"].mean().rename("unexcised_computed_fraction").reset_index()

    mrd_1200 = _add_class(mrd[(mrd["variant"] == "mrd") & (mrd["scale_s"] == 1200.0)], classes)
    n_pairs_dist = mrd_1200.groupby(["stability_class", "boom", "spectrum"], observed=True)["n_pairs"].describe().reset_index()

    wd_slot = means[(means["variable"] == "wd") & (means["stat"] == "mean")][["slot", "boom", "value"]]
    wd_slot = wd_slot.rename(columns={"value": "wd_slot"})
    wd_frame = mrd_frame[mrd_frame["variant"] == "mrd"][["slot", "boom", "wd_deg"]]
    merged = wd_slot.merge(wd_frame, on=["slot", "boom"], how="inner")
    merged["diff_deg"] = [signed_angular_distance(a, b) for a, b in zip(merged["wd_deg"], merged["wd_slot"])]
    merged = _add_class(merged, classes)
    wd_diff = merged.groupby(["stability_class", "boom"])["diff_deg"].describe().reset_index()

    return {
        "p_measured_rate": p_measured_rate, "p_factor_distribution": p_factor_dist,
        "tau_status_rates": status_rates, "unexcised_computed_rate": unexcised_rate,
        "n_pairs_at_20min": n_pairs_dist, "wind_direction_diff": wd_diff,
    }


# --- orchestration ---------------------------------------------------------------------------


def run_check(name: str, run_dir: Path, cfg, rng: np.random.Generator, nproc: int | None = None) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    primary_dir, secondary_dir, tertiary_dir = run_dir / "primary", run_dir / "secondary", run_dir / "tertiary"

    def classes() -> pd.Series:
        return stability_classes(read_table(tertiary_dir / "data" / "pairs"), cfg.post)

    if name == "tau_agreement":
        return tau_agreement(read_table(secondary_dir / "data" / "tau"), read_table(secondary_dir / "data" / "tau_selected"), classes())
    if name == "despike_calibration":
        return despike_calibration(cfg, run_dir, rng, nproc=nproc)
    if name == "its_bias":
        return its_bias(cfg, read_table(secondary_dir / "data" / "boom_stats"), read_table(secondary_dir / "data" / "tau_selected"), classes(), rng)
    if name == "interp_bias":
        return interp_bias(cfg, run_dir, rng)
    if name == "floor_peak":
        return floor_peak(read_table(primary_dir / "data" / "mrd"), read_table(secondary_dir / "data" / "tau"), classes(), cfg)
    if name == "yield":
        return yield_report(primary_dir, tertiary_dir, classes())
    if name == "tau_profile":
        return tau_profile(read_table(secondary_dir / "data" / "tau_selected"), classes())
    if name == "mrd_vs_naive":
        return mrd_vs_naive(read_table(tertiary_dir / "data" / "boom_final"), read_table(primary_dir / "data" / "slot_boom"), classes())
    if name == "sanity":
        return sanity(
            read_table(secondary_dir / "data" / "slow"), read_table(secondary_dir / "data" / "tau_selected"),
            read_table(primary_dir / "data" / "slot_boom"), read_table(primary_dir / "data" / "mrd"),
            read_table(primary_dir / "data" / "mrd_frame"), read_table(primary_dir / "data" / "means"), classes(),
        )
    raise ValueError(f"unknown check '{name}'; known checks: {CHECK_NAMES}")


def write_validation_reports(run_dir: Path, cfg, checks=None, seed: int = 0, nproc: int | None = None) -> None:
    run_dir = Path(run_dir)
    reports_dir = run_dir / "reports" / "validation"
    rng = np.random.default_rng(seed)
    for name in (checks or CHECK_NAMES):
        result = run_check(name, run_dir, cfg, rng, nproc=nproc)
        check_dir = reports_dir / name
        check_dir.mkdir(parents=True, exist_ok=True)
        for csv_name, df in result.items():
            df.to_csv(check_dir / f"{csv_name}.csv", index=False)
