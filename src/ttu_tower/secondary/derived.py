"""tau assembly (detection + selection across every slot/boom/variant),
selected-rung statistics, and every single-boom derived quantity.
"""
import numpy as np
import pandas as pd

from ttu_tower.constants import HEIGHTS, SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.physics import most, turbulence
from ttu_tower.physics.earth import local_gravity
from ttu_tower.secondary.detect import detect
from ttu_tower.secondary.select import select

_COSPECTRA = {"heat": "wvpts", "momentum": "uw"}
_NAIVE_RUNG_S = 600.0
# "naive" borrows the mrd variant's own ladder rows at the 600-s rung; the
# copy from mrd to mrd_unexcised where it wasn't separately computed happens
# at the table level (materialize_missing_unexcised), not here.
_LADDER_SOURCE_VARIANT = {"mrd": "mrd", "naive": "mrd", "mrd_unexcised": "mrd_unexcised"}

_STATS_TABLE_COLUMNS = ["slot", "boom", "variant", "variable", "stat", "value"]
_TAU_COLUMNS = ["slot", "boom", "variant", "cospectrum", "status", "tau_s", "tau_lb_s",
                "sign", "peak_scale_s", "reversal_scale_s", "reversal_type"]
_TAU_SELECTED_COLUMNS = ["slot", "boom", "variant", "tau_s", "source", "source_status"]
_BOOM_LABELS_COLUMNS = ["slot", "boom", "variant", "label", "value"]

_GRAVITY = {boom: local_gravity(SITE_LATITUDE, SITE_ELEVATION + height) for boom, height in HEIGHTS.items()}


def _mode_arrays(spectrum_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = spectrum_rows.sort_values("scale_s")
    return g["value"].to_numpy(dtype=float), g["se"].to_numpy(dtype=float), g["n_pairs"].to_numpy()


def tau_tables(mrd: pd.DataFrame, coverage: pd.DataFrame, cfg_secondary) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`tau` and `tau_selected` rows for every (slot, boom, variant) present in
    `mrd` (variant in {mrd, mrd_unexcised}, whichever primary computed).
    """
    family_has_data: dict[tuple, bool] = {}
    cov = coverage[coverage["variable"].isin(("heat", "momentum")) & coverage["layer"].isin(("usable", "unexcised"))]
    for row in cov.itertuples(index=False):
        variant = "mrd" if row.layer == "usable" else "mrd_unexcised"
        family_has_data[(row.slot, row.boom, variant, row.variable)] = row.fraction > 0

    tau_rows = []
    selected_rows = []
    for (slot, boom, variant), group in mrd.groupby(["slot", "boom", "variant"], observed=True):
        detections = {}
        for family, spectrum in _COSPECTRA.items():
            D, SE, N = _mode_arrays(group[group["spectrum"] == spectrum])
            has_data = family_has_data.get((slot, boom, variant, family), False)
            det = detect(D, SE, N, has_data, cfg_secondary.detection)
            detections[family] = det
            tau_rows.append((slot, boom, variant, family, det.status, det.tau_s, det.tau_lb_s,
                              det.sign, det.peak_scale_s, det.reversal_scale_s, det.reversal_type))

        tau_s, source, source_status = select(detections["heat"], detections["momentum"], cfg_secondary.selection)
        selected_rows.append((slot, boom, variant, tau_s, source, source_status))

    tau_df = pd.DataFrame(tau_rows, columns=_TAU_COLUMNS)
    tau_selected_df = pd.DataFrame(selected_rows, columns=_TAU_SELECTED_COLUMNS)
    return tau_df, tau_selected_df


def naive_tau_selected(tau_selected_mrd: pd.DataFrame) -> pd.DataFrame:
    """(600 s, fixed, fixed) wherever an `mrd` tau_selected row exists."""
    out = tau_selected_mrd[["slot", "boom"]].copy()
    out["variant"] = "naive"
    out["tau_s"] = _NAIVE_RUNG_S
    out["source"] = "fixed"
    out["source_status"] = "fixed"
    return out[_TAU_SELECTED_COLUMNS]


def selected_stats(ladder: pd.DataFrame, ladder_coverage: pd.DataFrame, tau_selected: pd.DataFrame,
                    p_factor: pd.Series) -> pd.DataFrame:
    """`boom_stats` rows (variable, stat, value) at each row's selected rung:
    every `ladder` row at that rung (heat rows converted by the slot's
    `p_factor`, indexed by (slot, boom)), plus the support-coverage rows
    (<family> x coverage/blocks_used/blocks_total). A `naive`-variant row
    reads the `mrd` variant's ladder at the 600-s rung.
    """
    if tau_selected.empty:
        return pd.DataFrame(columns=_STATS_TABLE_COLUMNS)

    sel = tau_selected.copy()
    sel["source_variant"] = sel["variant"].map(_LADDER_SOURCE_VARIANT)

    ladder_rows = sel.merge(
        ladder, left_on=["slot", "boom", "source_variant", "tau_s"],
        right_on=["slot", "boom", "variant", "rung_s"], how="inner", suffixes=("", "_src"),
    )[_STATS_TABLE_COLUMNS].copy()

    p_factor_df = p_factor.rename("p_factor").reset_index()
    ladder_rows = ladder_rows.merge(p_factor_df, on=["slot", "boom"], how="left")
    ladder_rows["p_factor"] = ladder_rows["p_factor"].fillna(1.0)

    is_wvpts_cov = (ladder_rows["variable"] == "wvpts") & (ladder_rows["stat"] == "cov")
    is_vpts_mean = (ladder_rows["variable"] == "vpts") & (ladder_rows["stat"] == "mean")
    is_vpts_var = (ladder_rows["variable"] == "vpts") & (ladder_rows["stat"] == "var")
    scale_once = is_wvpts_cov | is_vpts_mean
    ladder_rows.loc[scale_once, "value"] *= ladder_rows.loc[scale_once, "p_factor"]
    ladder_rows.loc[is_vpts_var, "value"] *= ladder_rows.loc[is_vpts_var, "p_factor"] ** 2
    ladder_rows = ladder_rows[_STATS_TABLE_COLUMNS]

    cov_rows = sel.merge(
        ladder_coverage, left_on=["slot", "boom", "source_variant", "tau_s"],
        right_on=["slot", "boom", "variant", "rung_s"], how="inner", suffixes=("", "_src"),
    )
    coverage_df = cov_rows.melt(
        id_vars=["slot", "boom", "variant", "family"], value_vars=["coverage", "blocks_used", "blocks_total"],
        var_name="stat", value_name="value",
    ).rename(columns={"family": "variable"})
    coverage_df["value"] = coverage_df["value"].astype(float)
    coverage_df = coverage_df[_STATS_TABLE_COLUMNS]

    return pd.concat([ladder_rows, coverage_df], ignore_index=True)


def _short_flag(its_values: list[float], tau_s: float, blocks_used: float, min_ratio: float) -> float:
    if not blocks_used:
        return float("nan")

    def is_short(its_x: float) -> bool:
        return not np.isfinite(its_x) or (tau_s / its_x) < min_ratio

    return 1.0 if any(is_short(x) for x in its_values) else 0.0


def derived(stats: pd.DataFrame, tau_selected: pd.DataFrame, slow: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every scalar derived quantity (TI, TKE, u*, L, zeta, ILS, ITS ratios and
    the anisotropy tensor/class) from `stats` (ladder rows at the selected
    rung, pressure-converted, from `selected_stats`), the selected tau (for
    the ITS ratios) and `slow` (t/rh/p thermodynamics, for the slow-sensor vpt
    `obukhov_length` needs).
    """
    if stats.empty:
        return pd.DataFrame(columns=_STATS_TABLE_COLUMNS), pd.DataFrame(columns=_BOOM_LABELS_COLUMNS)

    vpt_by_group = slow[slow["variable"] == "vpt"].set_index(["slot", "boom"])["value"]
    tau_by_group = tau_selected.set_index(["slot", "boom", "variant"])["tau_s"]

    wide = stats.copy()
    wide["key"] = list(zip(wide["variable"], wide["stat"]))
    wide = wide.pivot_table(index=["slot", "boom", "variant"], columns="key", values="value", aggfunc="first")

    min_ratio = cfg.secondary.min_tau_its_ratio
    stat_rows = []
    label_rows = []
    for (slot, boom, variant), row in wide.iterrows():
        def get(variable: str, stat: str) -> float:
            return row.get((variable, stat), np.nan)

        var_u, var_v, var_w = get("u", "var"), get("v", "var"), get("w", "var")
        cov_uw, cov_vw, cov_uv = get("uw", "cov"), get("vw", "cov"), get("uv", "cov")
        u_mean, w_mean, ws_mean, ws_std = get("u", "mean"), get("w", "mean"), get("ws", "mean"), get("ws", "std")
        wvpts_cov = get("wvpts", "cov")
        its_u, its_v, its_w, its_vpts = get("u", "its"), get("v", "its"), get("w", "its"), get("vpts", "its")
        blocks_its, blocks_its_vpts = get("its", "blocks_used"), get("its_vpts", "blocks_used")
        tau_s = tau_by_group.get((slot, boom, variant), np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            sigma_u, sigma_v, sigma_w = np.sqrt(var_u), np.sqrt(var_v), np.sqrt(var_w)
            ti = ws_std / ws_mean if ws_mean else np.nan
            ti_u = sigma_u / ws_mean if ws_mean else np.nan
            ti_v = sigma_v / ws_mean if ws_mean else np.nan
            ti_w = sigma_w / ws_mean if ws_mean else np.nan
            ti_ratio_vu = ti_v / ti_u if ti_u else np.nan
            ti_ratio_wu = ti_w / ti_u if ti_u else np.nan
            tke = 0.5 * (var_u + var_v + var_w)
            ctke = 0.5 * np.sqrt(cov_uw**2 + cov_vw**2 + cov_uv**2)
            ustar = most.friction_velocity(cov_uw, cov_vw)
            vpt_slow = vpt_by_group.get((slot, boom), np.nan)
            obukhov_length = most.obukhov_length(ustar, vpt_slow, wvpts_cov, gravity=_GRAVITY[boom])
            zeta = HEIGHTS[boom] / obukhov_length if obukhov_length else np.nan
            ils_u = its_u * u_mean
            ils_v = its_v * abs(u_mean)
            ils_w = its_w * abs(u_mean)
            ils_vpts = its_vpts * abs(u_mean)
            ils_ratio_vu = ils_v / ils_u if ils_u else np.nan
            ils_ratio_wu = ils_w / ils_u if ils_u else np.nan
            its_ratio_u = tau_s / its_u if its_u else np.nan
            its_ratio_v = tau_s / its_v if its_v else np.nan
            its_ratio_w = tau_s / its_w if its_w else np.nan
            its_ratio_vpts = tau_s / its_vpts if its_vpts else np.nan
            its_short = _short_flag([its_u, its_v, its_w], tau_s, blocks_its, min_ratio)
            its_short_vpts = _short_flag([its_vpts], tau_s, blocks_its_vpts, min_ratio)
            w_ratio = w_mean / ws_mean if ws_mean else np.nan

        a = turbulence.anisotropy(var_u, var_v, var_w, cov_uv, cov_uw, cov_vw, tke)

        values = {
            "sigma_u": sigma_u, "sigma_v": sigma_v, "sigma_w": sigma_w,
            "ti": ti, "ti_u": ti_u, "ti_v": ti_v, "ti_w": ti_w,
            "ti_ratio_vu": ti_ratio_vu, "ti_ratio_wu": ti_ratio_wu,
            "tke": tke, "ctke": ctke, "ustar": ustar,
            "obukhov_length": obukhov_length, "zeta": zeta,
            "ils_u": ils_u, "ils_v": ils_v, "ils_w": ils_w, "ils_vpts": ils_vpts,
            "ils_ratio_vu": ils_ratio_vu, "ils_ratio_wu": ils_ratio_wu,
            "its_ratio_u": its_ratio_u, "its_ratio_v": its_ratio_v,
            "its_ratio_w": its_ratio_w, "its_ratio_vpts": its_ratio_vpts,
            "its_short": its_short, "its_short_vpts": its_short_vpts,
            "aniso_l1": a.l1, "aniso_l2": a.l2, "aniso_l3": a.l3,
            "aniso_beta": a.beta, "aniso_phi": a.phi,
            "w_ratio": w_ratio,
        }
        for variable, value in values.items():
            stat_rows.append((slot, boom, variant, variable, None, float(value)))
        label_rows.append((slot, boom, variant, "aniso_class", a.aniso_class))

    stat_df = pd.DataFrame(stat_rows, columns=_STATS_TABLE_COLUMNS)
    label_df = pd.DataFrame(label_rows, columns=_BOOM_LABELS_COLUMNS)
    return stat_df, label_df


def materialize_missing_unexcised(tables: dict[str, pd.DataFrame], slot_boom: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """For every (slot, boom) that primary computed without a real
    `mrd_unexcised` (`slot_boom.status == "computed"` and
    `unexcised_computed` is false), copy that (slot, boom)'s already-computed
    `mrd`-variant rows into every table that has a `variant` column, relabeled
    `mrd_unexcised` - the two would be identical, since primary only computes
    a real unexcised counterfactual where the mask actually differs.
    """
    needs_copy = slot_boom[(slot_boom["status"] == "computed") & ~slot_boom["unexcised_computed"]]
    pairs = set(zip(needs_copy["slot"], needs_copy["boom"]))

    out = {}
    for name, df in tables.items():
        if "variant" not in df.columns or not pairs:
            out[name] = df
            continue
        mrd_rows = df[df["variant"] == "mrd"]
        key = list(zip(mrd_rows["slot"], mrd_rows["boom"]))
        mask = [k in pairs for k in key]
        copy = mrd_rows[mask].copy()
        copy["variant"] = "mrd_unexcised"
        out[name] = pd.concat([df, copy], ignore_index=True)
    return out
