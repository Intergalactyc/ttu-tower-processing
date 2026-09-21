"""Stage B: per-half-hour QC, masking and block-sum products, computed on a
span with margin context so a file's results never depend on how the
timeline is split.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttu_tower import schema
from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.flags import FlagRows, mask_to_intervals
from ttu_tower.primary.blocks import FamilySums, finest_sums as compute_finest_sums, floor_sums as compute_floor_sums
from ttu_tower.primary.despike import despike
from ttu_tower.primary.gaps import fill_short, fill_within_runs
from ttu_tower.primary.masks import family_masks, first_layer, second_layer
from ttu_tower.primary.qc_windows import higher_moments, resolution_dropouts
from ttu_tower.primary.slotstats import slot_means
from ttu_tower.primary.slow import hann_kernel, smooth, vpts as compute_vpts
from ttu_tower.primary.stage_a import StageA, couple_triplet
from ttu_tower.timegrid import BlockGrid, SAMPLES_PER_HALF_HOUR, SAMPLES_PER_SLOT

_SONIC_TS = ("ue", "vn", "w", "ts")  # despiked and gap-filled; QC windows applied
_SLOW = ("t", "rh", "p")  # despiked, then smoothed
_ALL_DESPIKED = _SONIC_TS + _SLOW
_MASK_VARS = ("ue", "vn", "w", "ts", "vpts")  # participate in first/second-layer masking
_N_FINEST = 192


@dataclass
class Span:
    """Stage A arrays for one half-hour's core plus margin context on each
    side, assembled from the neighbouring half-hours (or NaN where one is a
    placeholder). `file_run_id` is -1 for placeholder samples, else a tag
    shared by consecutive accepted half-hours within the span - see
    `fill_within_runs`.
    """

    g0: int
    series: dict[str, np.ndarray]
    file_run_id: np.ndarray
    core_bounds_removed: dict[str, np.ndarray]


@dataclass
class StageBOut:
    coverage: pd.DataFrame
    means: pd.DataFrame
    slot_qc: pd.DataFrame
    flags: pd.DataFrame
    mask_differs: dict[int, bool]
    is_placeholder: bool = False
    final_series: dict[str, np.ndarray] = field(default_factory=dict)  # ue, vn, w, ts, vpts (core)
    final_masks: dict[str, np.ndarray] = field(default_factory=dict)  # momentum, heat, ts (core)
    unexcised_series: dict[str, np.ndarray] = field(default_factory=dict)
    unexcised_masks: dict[str, np.ndarray] = field(default_factory=dict)
    floor: dict[str, FamilySums] | None = None
    floor_unexcised: dict[str, FamilySums] | None = None
    finest: dict[str, FamilySums] | None = None
    finest_unexcised: dict[str, FamilySums] | None = None


def _empty_out(h: int) -> StageBOut:
    """A placeholder for a missing file: NaN series, zero weights, no rows."""
    coverage = schema.cast("coverage", pd.DataFrame(columns=["slot", "boom", "variable", "layer", "fraction"]))
    means = schema.cast("means", pd.DataFrame(columns=["slot", "boom", "variable", "stat", "value"]))
    slot_qc = schema.cast("slot_qc", pd.DataFrame(columns=["slot", "boom", "variable", "stat", "value"]))
    flags = FlagRows().frame()
    mask_differs = {3 * h: False, 3 * h + 1: False, 3 * h + 2: False}
    return StageBOut(coverage=coverage, means=means, slot_qc=slot_qc, flags=flags, mask_differs=mask_differs,
                      is_placeholder=True)


def build_span(stage_a_by_half_hour: dict[int, StageA | None], h: int, margin: int) -> Span:
    """Assemble half-hour `h`'s span from `stage_a_by_half_hour[h-1..h+1]`."""
    total = 2 * margin + SAMPLES_PER_HALF_HOUR
    g0 = SAMPLES_PER_HALF_HOUR * h - margin
    variables = _ALL_DESPIKED
    series = {var: np.full(total, np.nan) for var in variables}
    file_run_id = np.full(total, -1, dtype=np.int64)

    available = {hh: stage_a_by_half_hour.get(hh) is not None for hh in (h - 1, h, h + 1)}
    tags: dict[int, int] = {}
    prev_tag = None
    for hh in (h - 1, h, h + 1):
        if available[hh]:
            tags[hh] = prev_tag if prev_tag is not None else hh
            prev_tag = tags[hh]
        else:
            tags[hh] = -1
            prev_tag = None

    slices = {
        h - 1: (0, margin, SAMPLES_PER_HALF_HOUR - margin, SAMPLES_PER_HALF_HOUR),
        h: (margin, margin + SAMPLES_PER_HALF_HOUR, 0, SAMPLES_PER_HALF_HOUR),
        h + 1: (margin + SAMPLES_PER_HALF_HOUR, total, 0, margin),
    }

    core_bounds_removed = {v: np.zeros(SAMPLES_PER_HALF_HOUR, dtype=bool) for v in variables}
    for hh, (dst_lo, dst_hi, src_lo, src_hi) in slices.items():
        file_run_id[dst_lo:dst_hi] = tags[hh]
        a = stage_a_by_half_hour.get(hh)
        if a is None:
            continue
        for var in variables:
            series[var][dst_lo:dst_hi] = getattr(a, var)[src_lo:src_hi]
        if hh == h:
            core_bounds_removed = a.bounds_removed

    return Span(g0=g0, series=series, file_run_id=file_run_id, core_bounds_removed=core_bounds_removed)


def _slot_flag_intervals(flag: np.ndarray, slots: list[int]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.array([SAMPLES_PER_SLOT * k for k, f in zip(slots, flag) if f], dtype=np.int64)
    return starts, starts + SAMPLES_PER_SLOT


def stage_b(span: Span, h: int, boom: int, cfg) -> StageBOut:
    """The Stage B chain for half-hour `h`'s core, given its span. `cfg` is
    the full run Config. The core's position is derived from `span.g0` and
    `h` directly, not assumed to be centered - `span` need not be symmetric
    (a whole-file-run span, used to prove seam invariance, generally isn't).
    """
    core_g0 = SAMPLES_PER_HALF_HOUR * h
    core_lo = core_g0 - span.g0
    core = slice(core_lo, core_lo + SAMPLES_PER_HALF_HOUR)

    if span.file_run_id[core_lo] == -1:
        return _empty_out(h)

    qc = cfg.qc
    c = qc.min_coverage
    slots = [3 * h, 3 * h + 1, 3 * h + 2]
    flag_rows = FlagRows()

    def clip_add(test, variable, starts, ends):
        a = np.maximum(starts, core_g0)
        b = np.minimum(ends, core_g0 + SAMPLES_PER_HALF_HOUR)
        keep = a < b
        flag_rows.add(test, boom, variable, a[keep], b[keep])

    for var, mask in span.core_bounds_removed.items():
        clip_add("bounds", var, *mask_to_intervals(mask, g0=core_g0))

    # 1. despike every variable; couple the ue/vn/w triplet's spike removals
    despiked = {}
    for var in _ALL_DESPIKED:
        despiked[var] = despike(span.series[var], span.g0, qc.despike, qc.despike.min_mad[var], qc.despike.z_threshold[var], c)
        clip_add("spike", var, *mask_to_intervals(despiked[var].spike, g0=span.g0))
        clip_add("excursion", var, *mask_to_intervals(despiked[var].excursion, g0=span.g0))
        clip_add("unchecked", var, *mask_to_intervals(despiked[var].unchecked, g0=span.g0))

    ue_c, vn_c, w_c = couple_triplet(despiked["ue"].x, despiked["vn"].x, despiked["w"].x)
    despiked_x = {"ue": ue_c, "vn": vn_c, "w": w_c, "ts": despiked["ts"].x,
                  "t": despiked["t"].x, "rh": despiked["rh"].x, "p": despiked["p"].x}

    # 2. fill_short, on every despiked variable including t, rh and p
    filled, filled_mask = {}, {}
    for var in _ALL_DESPIKED:
        filled[var], filled_mask[var] = fill_short(despiked_x[var], qc.max_fill_gap_samples)

    # 3. higher_moments (core slots)
    higher = {}
    for var in _SONIC_TS:
        skew, kurt, skew_flag, kurt_flag = higher_moments(filled[var], span.g0, slots, qc.windows, c)
        higher[var] = (skew, kurt, skew_flag, kurt_flag)
        clip_add("skew", var, *_slot_flag_intervals(skew_flag, slots))
        clip_add("kurt", var, *_slot_flag_intervals(kurt_flag, slots))

    # 4. resolution_dropouts (ue, vn, w, ts)
    resolution_iv, dropout_iv = {}, {}
    for var in _SONIC_TS:
        (res_s, res_e), (drop_s, drop_e) = resolution_dropouts(filled[var], span.g0, qc.windows, c, is_ts=(var == "ts"))
        resolution_iv[var], dropout_iv[var] = (res_s, res_e), (drop_s, drop_e)
        clip_add("resolution", var, res_s, res_e)
        clip_add("dropouts", var, drop_s, drop_e)

    # 5. smooth (t, rh, p), on each one's first-layer usable mask (finite after
    # step 2, outside its "unchecked" interval if that's unusable - the only
    # window test that applies to a slow variable, so this needs nothing from
    # steps 3-4, which are sonic-only).
    finite_slow = {var: np.isfinite(filled[var]) for var in _SLOW}
    intervals_slow = {var: {"unchecked": mask_to_intervals(despiked[var].unchecked, g0=span.g0)} for var in _SLOW}
    l1_slow = first_layer(finite_slow, intervals_slow, list(qc.unusable_tests), span.g0, span.series["ue"].size)
    slow_smoothed = {}
    for var in _SLOW:
        k = hann_kernel(qc.slow.smoothing_width_s[var])
        slow_smoothed[var] = smooth(filled[var], l1_slow[var], k, c)

    # 6. vpts, from the filled ts (and from the raw ts, for the "present" layer)
    vpts_filled = compute_vpts(filled["ts"], boom)
    vpts_present = compute_vpts(span.series["ts"], boom)

    # 7. first-layer masks
    finite = {var: np.isfinite(filled[var]) for var in _SONIC_TS}
    intervals = {
        var: {
            "unchecked": mask_to_intervals(despiked[var].unchecked, g0=span.g0),
            "resolution": resolution_iv[var],
            "dropouts": dropout_iv[var],
            "skew": _slot_flag_intervals(higher[var][2], slots),
            "kurt": _slot_flag_intervals(higher[var][3], slots),
        }
        for var in _SONIC_TS
    }
    l1_masks = first_layer(finite, intervals, list(qc.unusable_tests), span.g0, span.series["ue"].size)
    l1_masks["vpts"] = l1_masks["ts"]

    # 8. second_layer (core slots)
    l1_momentum = l1_masks["ue"] & l1_masks["vn"] & l1_masks["w"]
    wd, ws_mean, ws_std, direction_flag, bounce_flag = second_layer(
        filled["ue"], filled["vn"], l1_momentum, span.g0, slots, qc.second_layer, c
    )
    clip_add("direction", None, *_slot_flag_intervals(direction_flag, slots))
    clip_add("bounce", None, *_slot_flag_intervals(bounce_flag, slots))

    # 9. final masks (core-only): first layer, minus direction/bounce slots
    boom_level_bad = np.zeros(SAMPLES_PER_HALF_HOUR, dtype=bool)
    for i, k in enumerate(slots):
        lo = SAMPLES_PER_SLOT * (k - 3 * h)
        excluded = ("direction" in qc.unusable_tests and direction_flag[i]) or ("bounce" in qc.unusable_tests and bounce_flag[i])
        if excluded:
            boom_level_bad[lo : lo + SAMPLES_PER_SLOT] = True
    final_masks_var = {var: l1_masks[var][core] & ~boom_level_bad for var in _MASK_VARS}
    final_series = {
        "ue": filled["ue"][core], "vn": filled["vn"][core], "w": filled["w"][core],
        "ts": filled["ts"][core], "vpts": vpts_filled[core],
    }
    final_family = family_masks(final_masks_var)

    # 10. the unexcised series and masks
    unexcised = {}
    max_gap = round(cfg.primary.unexcised_max_fill_gap_s * SAMPLE_HZ)
    for var in _SONIC_TS:
        x_filled, _ = fill_within_runs(despiked_x[var], span.file_run_id, max_gap)
        unexcised[var] = x_filled[core]
    unexcised["vpts"] = compute_vpts(unexcised["ts"], boom)
    unexcised_masks_var = {var: np.isfinite(unexcised[var]) for var in _MASK_VARS}
    unexcised_family = family_masks(unexcised_masks_var)

    # 11. floor_sums and finest_sums for both variants
    floor_level = cfg.mrd.floor_level
    floor_grid = BlockGrid.floor(floor_level)
    n_floor = int(round(SAMPLES_PER_HALF_HOUR / floor_grid.length_samples))
    floor_b0 = core_g0 * floor_grid.den // floor_grid.num
    finest_b0 = core_g0 * BlockGrid.FINEST.den // BlockGrid.FINEST.num

    floor = compute_floor_sums(final_series, final_family, core_g0, floor_level, floor_b0, n_floor)
    finest = compute_finest_sums(final_series, final_family, core_g0, finest_b0, _N_FINEST)
    floor_unexcised = compute_floor_sums(unexcised, unexcised_family, core_g0, floor_level, floor_b0, n_floor)
    finest_unexcised = compute_finest_sums(unexcised, unexcised_family, core_g0, finest_b0, _N_FINEST)

    # 12. per-slot coverage, means, slot_qc, mask_differs
    present = {v: np.isfinite(span.series[v][core]) for v in _SONIC_TS}
    present["vpts"] = np.isfinite(vpts_present[core])
    filled_finite = {v: np.isfinite(filled[v][core]) for v in _SONIC_TS}
    filled_finite["vpts"] = np.isfinite(vpts_filled[core])
    l1_core = {v: l1_masks[v][core] for v in _MASK_VARS}
    slow_present = {v: np.isfinite(span.series[v][core]) for v in _SLOW}
    slow_usable = {v: slow_smoothed[v][1][core] for v in _SLOW}

    coverage_rows, slot_qc_rows, mask_differs = [], [], {}
    for i, k in enumerate(slots):
        lo = SAMPLES_PER_SLOT * (k - 3 * h)
        sl = slice(lo, lo + SAMPLES_PER_SLOT)

        for var in _MASK_VARS:
            for layer, mask_dict in (
                ("present", present), ("filled", filled_finite), ("usable_l1", l1_core),
                ("usable", final_masks_var), ("unexcised", unexcised_masks_var),
            ):
                coverage_rows.append((k, boom, var, layer, float(mask_dict[var][sl].mean())))
        for var in _SLOW:
            coverage_rows.append((k, boom, var, "present", float(slow_present[var][sl].mean())))
            coverage_rows.append((k, boom, var, "usable", float(slow_usable[var][sl].mean())))
        for fam in ("momentum", "heat"):
            coverage_rows.append((k, boom, fam, "usable", float(final_family[fam][sl].mean())))
            coverage_rows.append((k, boom, fam, "unexcised", float(unexcised_family[fam][sl].mean())))

        for var in _SONIC_TS:
            slot_qc_rows.append((k, boom, var, "skew", higher[var][0][i]))
            slot_qc_rows.append((k, boom, var, "kurt", higher[var][1][i]))
        slot_qc_rows.append((k, boom, "wd", "mean_l1", wd[i]))
        slot_qc_rows.append((k, boom, "ws", "mean_l1", ws_mean[i]))
        slot_qc_rows.append((k, boom, "ws", "std_l1", ws_std[i]))

        mask_differs[k] = bool(any((unexcised_masks_var[v][sl] & ~final_masks_var[v][sl]).any() for v in _MASK_VARS))

    coverage_df = schema.cast("coverage", pd.DataFrame(coverage_rows, columns=["slot", "boom", "variable", "layer", "fraction"]))
    slot_qc_df = schema.cast("slot_qc", pd.DataFrame(slot_qc_rows, columns=["slot", "boom", "variable", "stat", "value"]))

    means_rows = []
    slow_for_means = {v: (slow_smoothed[v][0][core], slow_smoothed[v][1][core]) for v in _SLOW}
    masks_for_means = {"momentum": final_family["momentum"], "ts": final_masks_var["ts"]}
    for k in slots:
        for var, stat, value in slot_means(final_series, masks_for_means, slow_for_means, core_g0, k, cfg.ladder, c):
            means_rows.append((k, boom, var, stat, value))
    means_df = schema.cast("means", pd.DataFrame(means_rows, columns=["slot", "boom", "variable", "stat", "value"]))

    return StageBOut(
        coverage=coverage_df, means=means_df, slot_qc=slot_qc_df, flags=flag_rows.frame(), mask_differs=mask_differs,
        final_series=final_series, final_masks=final_family,
        unexcised_series=unexcised, unexcised_masks=unexcised_family,
        floor=floor, floor_unexcised=floor_unexcised, finest=finest, finest_unexcised=finest_unexcised,
    )
