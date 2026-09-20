"""Slot-level means, maxima and gusts (variant-independent)."""
import numpy as np

from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.timegrid import SAMPLES_PER_SLOT


def slot_means(series: dict[str, np.ndarray], masks: dict[str, np.ndarray], slow: dict[str, tuple[np.ndarray, np.ndarray]],
               g0: int, k: int, cfg, c: float):
    """The `means` rows for slot k: momentum-family wind statistics, gusts,
    ts/vpts means (ts family), and t/rh/p means (each on its own smoothed
    usability). `slow` maps "t"/"rh"/"p" -> (smoothed_values, usable).
    """
    start = SAMPLES_PER_SLOT * k - g0
    sl = slice(start, start + SAMPLES_PER_SLOT)
    rows = []

    ue, vn, w = series["ue"][sl], series["vn"][sl], series["w"][sl]
    mom_mask = masks["momentum"][sl]

    if mom_mask.any():
        ue_u, vn_u, w_u = ue[mom_mask], vn[mom_mask], w[mom_mask]
        ws_sample = np.hypot(ue_u, vn_u)

        rows.append(("ue", "mean", float(ue_u.mean())))
        rows.append(("vn", "mean", float(vn_u.mean())))
        rows.append(("w", "mean", float(w_u.mean())))
        rows.append(("ws", "mean", float(ws_sample.mean())))

        mean_ue, mean_vn = ue_u.mean(), vn_u.mean()
        vector_speed, wd_mean = vector_to_bearing(mean_ue, mean_vn)
        rows.append(("ws", "vector_mean", float(vector_speed)))
        rows.append(("wd", "mean", float(wd_mean)))

        moving = mom_mask & (np.hypot(ue, vn) > 0)
        if moving.any():
            ws_full = np.hypot(ue, vn)
            unit_ue = (ue[moving] / ws_full[moving]).mean()
            unit_vn = (vn[moving] / ws_full[moving]).mean()
            _, wd_unit_mean = vector_to_bearing(unit_ue, unit_vn)
        else:
            wd_unit_mean = np.nan
        rows.append(("wd", "unit_mean", float(wd_unit_mean)))

        rows.append(("ws", "max", float(ws_sample.max())))
        rows.append(("w", "max", float(w_u.max())))
    else:
        for var, stat in [("ue", "mean"), ("vn", "mean"), ("w", "mean"), ("ws", "mean"),
                           ("ws", "vector_mean"), ("wd", "mean"), ("wd", "unit_mean"),
                           ("ws", "max"), ("w", "max")]:
            rows.append((var, stat, np.nan))

    for p in cfg.gust_periods_s:
        rows += _gust_rows(ue, vn, w, mom_mask, p, c)

    ts, vpts = series["ts"][sl], series["vpts"][sl]
    ts_mask = masks["ts"][sl]
    if ts_mask.any():
        rows.append(("ts", "mean", float(ts[ts_mask].mean())))
        rows.append(("vpts", "mean", float(vpts[ts_mask].mean())))
    else:
        rows.append(("ts", "mean", np.nan))
        rows.append(("vpts", "mean", np.nan))

    for var in ("t", "rh", "p"):
        values, usable = slow[var]
        values_sl, usable_sl = values[sl], usable[sl]
        rows.append((var, "mean", float(values_sl[usable_sl].mean()) if usable_sl.any() else np.nan))

    return rows


def _gust_rows(ue, vn, w, mom_mask, p: int, c: float):
    block_samples = 50 * p
    n_blocks = SAMPLES_PER_SLOT // block_samples
    speeds, ws_means = [], []
    for gi in range(n_blocks):
        gsl = slice(gi * block_samples, (gi + 1) * block_samples)
        block_mask = mom_mask[gsl]
        if block_mask.sum() / block_samples < c:
            continue
        mean_ue, mean_vn = ue[gsl][block_mask].mean(), vn[gsl][block_mask].mean()
        speeds.append(np.hypot(mean_ue, mean_vn))
        ws_means.append(w[gsl][block_mask].mean())

    speeds, ws_means = np.array(speeds), np.array(ws_means)
    if speeds.size == 0:
        return [
            ("ws", f"max_{p}s", np.nan), ("ws", f"std_{p}s", np.nan),
            ("w", f"max_{p}s", np.nan), ("w", f"std_{p}s", np.nan),
        ]
    return [
        ("ws", f"max_{p}s", float(speeds.max())),
        ("ws", f"std_{p}s", float(speeds.std()) if speeds.size >= 2 else np.nan),
        ("w", f"max_{p}s", float(ws_means.max())),
        ("w", f"std_{p}s", float(ws_means.std()) if ws_means.size >= 2 else np.nan),
    ]
