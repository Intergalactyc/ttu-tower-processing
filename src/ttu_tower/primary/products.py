"""Per-file products: assembling multi-file detection and ladder windows from
neighbouring Stage B outputs, and the slot_boom/mrd_unexcised trigger logic.
"""
import time

import numpy as np
import pandas as pd

from ttu_tower import schema
from ttu_tower.constants import DETECTION_WINDOW_MIN, SAMPLE_HZ
from ttu_tower.primary.blocks import FamilySums
from ttu_tower.primary.ladder import rung_statistics
from ttu_tower.primary.mrd import detection_outputs, mode_scales_s
from ttu_tower.primary.rawrung import rung_its_te
from ttu_tower.primary.stage_b import StageBOut
from ttu_tower.timegrid import BlockGrid, SAMPLES_PER_HALF_HOUR, SAMPLES_PER_SLOT

_DETECTION_WINDOW_SAMPLES = DETECTION_WINDOW_MIN * 60 * SAMPLE_HZ  # 240,000; independent of floor_level
_DETECTION_MARGIN_SAMPLES = 35 * 60 * SAMPLE_HZ
_LADDER_WINDOW_SAMPLES = 20 * 60 * SAMPLE_HZ  # 60,000
_LADDER_MARGIN_SAMPLES = 5 * 60 * SAMPLE_HZ
_N_FINEST_WINDOW = 128
_FLOOR_FAMILIES = ("momentum", "heat", "ts")
_FINEST_FAMILIES = ("momentum", "heat", "ts", "direction")
_SERIES_VARS = ("ue", "vn", "w", "vpts")


def _half_hours_overlapping(g0: int, n: int) -> list[int]:
    return list(range(g0 // SAMPLES_PER_HALF_HOUR, (g0 + n - 1) // SAMPLES_PER_HALF_HOUR + 1))


def _floor_blocks_per_half_hour(floor_level: int) -> int:
    return int(round(SAMPLES_PER_HALF_HOUR / BlockGrid.floor(floor_level).length_samples))


def _assemble_family(per_hh: dict[int, FamilySums | None], hh_block0: dict[int, int], blocks_per_hh: int,
                      target_b0: int, target_nb: int, columns: tuple[str, ...]) -> FamilySums:
    n = np.zeros(target_nb)
    sx = np.zeros((target_nb, len(columns)))
    for hh, fam in per_hh.items():
        if fam is None:
            continue
        hh_b0 = hh_block0[hh]
        lo, hi = max(hh_b0, target_b0), min(hh_b0 + blocks_per_hh, target_b0 + target_nb)
        if lo >= hi:
            continue
        src, dst = slice(lo - hh_b0, hi - hh_b0), slice(lo - target_b0, hi - target_b0)
        n[dst] = fam.n[src]
        sx[dst] = fam.sx[src]
    return FamilySums(n=n, sx=sx, columns=columns)


def _assemble_block_window(B: dict[int, StageBOut], half_hours: list[int], attr: str, families: tuple[str, ...],
                            blocks_per_hh: int, grid: BlockGrid, target_g0: int, target_nb: int) -> dict[str, FamilySums]:
    hh_block0 = {hh: SAMPLES_PER_HALF_HOUR * hh * grid.den // grid.num for hh in half_hours}
    target_b0 = target_g0 * grid.den // grid.num
    out = {}
    for fam in families:
        per_hh = {}
        for hh in half_hours:
            sb = B.get(hh)
            store = getattr(sb, attr) if sb is not None and not sb.is_placeholder else None
            per_hh[hh] = store[fam] if store is not None else None
        columns = next((f.columns for f in per_hh.values() if f is not None), ())
        out[fam] = _assemble_family(per_hh, hh_block0, blocks_per_hh, target_b0, target_nb, columns)
    return out


def _assemble_samples(per_hh: dict[int, np.ndarray | None], target_g0: int, target_n: int, is_mask: bool) -> np.ndarray:
    out = np.zeros(target_n, dtype=bool) if is_mask else np.full(target_n, np.nan)
    for hh, arr in per_hh.items():
        if arr is None:
            continue
        hh_g0 = SAMPLES_PER_HALF_HOUR * hh
        lo, hi = max(hh_g0, target_g0), min(hh_g0 + SAMPLES_PER_HALF_HOUR, target_g0 + target_n)
        if lo >= hi:
            continue
        out[lo - target_g0 : hi - target_g0] = arr[lo - hh_g0 : hi - hh_g0]
    return out


def _assemble_series_window(B: dict[int, StageBOut], half_hours: list[int], variant: str,
                             target_g0: int, target_n: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    series_attr = "final_series" if variant == "mrd" else "unexcised_series"
    masks_attr = "final_masks" if variant == "mrd" else "unexcised_masks"

    series = {}
    for var in _SERIES_VARS:
        per_hh = {}
        for hh in half_hours:
            sb = B.get(hh)
            store = getattr(sb, series_attr) if sb is not None and not sb.is_placeholder else None
            per_hh[hh] = store[var] if store is not None else None
        series[var] = _assemble_samples(per_hh, target_g0, target_n, is_mask=False)

    masks = {}
    for fam in ("momentum", "heat", "ts"):
        per_hh = {}
        for hh in half_hours:
            sb = B.get(hh)
            store = getattr(sb, masks_attr) if sb is not None and not sb.is_placeholder else None
            per_hh[hh] = store[fam] if store is not None else None
        masks[fam] = _assemble_samples(per_hh, target_g0, target_n, is_mask=True)

    return series, masks


def _mrd_rows(k: int, boom: int, variant: str, floor: dict[str, FamilySums], floor_level: int, c: float,
              timings: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    spectra, wd_deg, cov_mom, cov_heat = detection_outputs(floor, c)
    timings["detection_outputs"] = timings.get("detection_outputs", 0.0) + (time.perf_counter() - t0)
    scales = mode_scales_s(floor_level, SAMPLE_HZ)
    rows = [
        (k, boom, variant, spec, scale_s, v, s, int(n))
        for spec, (value, se, n_pairs) in spectra.items()
        for scale_s, v, s, n in zip(scales, value, se, n_pairs)
    ]
    mrd_df = pd.DataFrame(rows, columns=["slot", "boom", "variant", "spectrum", "scale_s", "value", "se", "n_pairs"])
    frame_df = pd.DataFrame([(k, boom, variant, wd_deg, cov_mom, cov_heat)],
                             columns=["slot", "boom", "variant", "wd_deg", "coverage_momentum", "coverage_heat"])
    return mrd_df, frame_df


def _ladder_rows(k: int, boom: int, variant: str, finest: dict[str, FamilySums], series: dict, masks: dict,
                  cfg, c: float, timings: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    values, coverage = rung_statistics(finest, c)
    timings["ladder"] = timings.get("ladder", 0.0) + (time.perf_counter() - t0)
    ladder_rows = [(k, boom, variant, rung_s, var, stat, val) for rung_s, var, stat, val in values]
    coverage_rows = [(k, boom, variant, rung_s, fam, used, total, cov) for rung_s, fam, used, total, cov in coverage]

    lad_g0 = SAMPLES_PER_SLOT * k - _LADDER_MARGIN_SAMPLES
    t0 = time.perf_counter()
    its_values, its_coverage = rung_its_te(series, masks, g0=lad_g0, k=k, cfg_ladder=cfg.ladder, c=c)
    timings["rawrung"] = timings.get("rawrung", 0.0) + (time.perf_counter() - t0)
    for rung_s, var, stat, val in its_values:
        ladder_rows.append((k, boom, variant, rung_s, var, stat, val))
    for rung_s, fam, used, total, cov in its_coverage:
        coverage_rows.append((k, boom, variant, rung_s, fam, used, total, cov))

    ladder_df = pd.DataFrame(ladder_rows, columns=["slot", "boom", "variant", "rung_s", "variable", "stat", "value"])
    coverage_df = pd.DataFrame(coverage_rows, columns=["slot", "boom", "variant", "rung_s", "family", "blocks_used", "blocks_total", "coverage"])
    return ladder_df, coverage_df


def _compute_variant(k: int, boom: int, variant: str, B: dict[int, StageBOut], cfg, timings: dict[str, float]) -> dict[str, pd.DataFrame]:
    c = cfg.qc.min_coverage
    floor_level = cfg.mrd.floor_level

    det_g0 = SAMPLES_PER_SLOT * k - _DETECTION_MARGIN_SAMPLES
    floor_attr = "floor" if variant == "mrd" else "floor_unexcised"
    floor = _assemble_block_window(
        B, _half_hours_overlapping(det_g0, _DETECTION_WINDOW_SAMPLES), floor_attr, _FLOOR_FAMILIES,
        _floor_blocks_per_half_hour(floor_level), BlockGrid.floor(floor_level), det_g0, 2 ** floor_level,
    )
    mrd_df, frame_df = _mrd_rows(k, boom, variant, floor, floor_level, c, timings)

    lad_g0 = SAMPLES_PER_SLOT * k - _LADDER_MARGIN_SAMPLES
    lad_half_hours = _half_hours_overlapping(lad_g0, _LADDER_WINDOW_SAMPLES)
    finest_attr = "finest" if variant == "mrd" else "finest_unexcised"
    finest = _assemble_block_window(B, lad_half_hours, finest_attr, _FINEST_FAMILIES, 192, BlockGrid.FINEST, lad_g0, _N_FINEST_WINDOW)
    series, masks = _assemble_series_window(B, lad_half_hours, variant, lad_g0, _LADDER_WINDOW_SAMPLES)
    ladder_df, ladder_cov_df = _ladder_rows(k, boom, variant, finest, series, masks, cfg, c, timings)

    return {"mrd": mrd_df, "mrd_frame": frame_df, "ladder": ladder_df, "ladder_coverage": ladder_cov_df}


def _slot_slice(k: int, h: int) -> slice:
    lo = SAMPLES_PER_SLOT * (k - 3 * h)
    return slice(lo, lo + SAMPLES_PER_SLOT)


def _empty(table: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(schema.TABLES[table].columns))


def file_products(h: int, B: dict[int, StageBOut], boom: int, cfg, timings: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
    """Coverage/means/slot_qc pass-through, mrd/mrd_frame/ladder/ladder_coverage
    (assembled from B's neighbouring half-hours), and slot_boom, for h's three
    slots. `timings` (seconds), if given, accumulates "detection_outputs",
    "ladder" and "rawrung" time across every variant computed here.
    """
    if timings is None:
        timings = {}
    center = B.get(h)
    slots = [3 * h, 3 * h + 1, 3 * h + 2]

    if center is None or center.is_placeholder:
        slot_boom = pd.DataFrame({"slot": slots, "boom": boom, "status": "no_file", "unexcised_computed": False})
        return {"slot_boom": schema.cast("slot_boom", slot_boom)}

    tables = {"coverage": center.coverage, "means": center.means, "slot_qc": center.slot_qc, "flags": center.flags}

    all_mask_differs = {}
    for hh in range(h - 2, h + 3):
        sb = B.get(hh)
        if sb is not None and not sb.is_placeholder:
            all_mask_differs.update(sb.mask_differs)

    mrd_parts, frame_parts, ladder_parts, ladder_cov_parts, slot_boom_rows = [], [], [], [], []
    for k in slots:
        sl = _slot_slice(k, h)
        no_data = not (center.final_masks["momentum"][sl].any() or center.final_masks["ts"][sl].any())

        if not no_data:
            parts = _compute_variant(k, boom, "mrd", B, cfg, timings)
            mrd_parts.append(parts["mrd"])
            frame_parts.append(parts["mrd_frame"])
            ladder_parts.append(parts["ladder"])
            ladder_cov_parts.append(parts["ladder_coverage"])
            triggered = cfg.primary.unexcised and any(all_mask_differs.get(kk, False) for kk in range(k - 4, k + 5))
        else:
            slot_has_unexcised_data = center.unexcised_masks["momentum"][sl].any() or center.unexcised_masks["ts"][sl].any()
            triggered = cfg.primary.unexcised and slot_has_unexcised_data

        if triggered:
            parts = _compute_variant(k, boom, "mrd_unexcised", B, cfg, timings)
            mrd_parts.append(parts["mrd"])
            frame_parts.append(parts["mrd_frame"])
            ladder_parts.append(parts["ladder"])
            ladder_cov_parts.append(parts["ladder_coverage"])

        slot_boom_rows.append((k, boom, "no_data" if no_data else "computed", bool(triggered)))

    tables["slot_boom"] = schema.cast("slot_boom", pd.DataFrame(slot_boom_rows, columns=["slot", "boom", "status", "unexcised_computed"]))
    tables["mrd"] = schema.cast("mrd", pd.concat(mrd_parts, ignore_index=True) if mrd_parts else _empty("mrd"))
    tables["mrd_frame"] = schema.cast("mrd_frame", pd.concat(frame_parts, ignore_index=True) if frame_parts else _empty("mrd_frame"))
    tables["ladder"] = schema.cast("ladder", pd.concat(ladder_parts, ignore_index=True) if ladder_parts else _empty("ladder"))
    tables["ladder_coverage"] = schema.cast("ladder_coverage", pd.concat(ladder_cov_parts, ignore_index=True) if ladder_cov_parts else _empty("ladder_coverage"))
    return tables
