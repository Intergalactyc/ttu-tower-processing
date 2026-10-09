# Low-level jet detection: algorithm and default criteria

This document defines exactly what counts as a low-level jet (LLJ) in `ttu_tower.jets`. It covers:
- the data a detector reads
- the common interface every detection algorithm implements
- the first algorithm, **nose** (a generalized, corrected version of the old pipeline's
  `classify_jet_discrete`)
- the time step that turns jet slots into jet events
- the event filters the viewer offers

The plan for building it is `llj-plan.md`.

---

## 1. Input data

| Quantity | Source | Notes |
|---|---|---|
| Wind speed profile | tertiary `boom_final`, `variable = ws`, `stat = mean`, `variant = none` | Scalar mean of the per-sample horizontal speed over the 10-min slot, on the final usable mask. NaN where the boom's `momentum` group failed tertiary filtering (coverage, bounds, spike) or where the boom has no data. |
| Heights | `ttu_tower.constants.HEIGHTS` | 0.9, 2.4, 4.0, 10.1, 16.8, 47.3, 74.7, 116.5, 158.2, 200.0 m for booms 1–10. |
| Wind direction at the nose (event statistics only) | `boom_final`, `wd`, `mean`, `none` | FROM-bearing of the mean horizontal vector. Not used for detection. |
| Inversion strength (filter) | `boom_final`, `vpt`, `none` | Slow-sensor virtual potential temperature (K). NaN where the `vpt` group failed filtering. |
| Richardson number (filter) | tertiary `pairs`, `rib` | Bulk Ri between `boom` < `boom2`; all 45 pairs exist. |
| Time of day (filter) | the slot start | In the viewer's display zone: local standard time (`Etc/GMT+6`) or UTC. |
| Night (filter) | tertiary `slot_final`, `night` | 1 when the sun is below the horizon at the slot. |

**"Present and good"** means a finite value in that table. Tertiary filtering has already
blanked bad values, so no separate QC test is needed.

Detection treats slots as a regular 10-minute grid, with slot k starting at
`EPOCH + 10 min · k`. A slot with no row at all, for example an outage, is treated exactly
like a slot whose booms are all NaN.

---

## 2. The detector interface

Every algorithm implements the same signature. Algorithms can then be swapped, picked from a
list in the GUI, and run through the same time step, filters and statistics.

### 2.1 Input: `Profiles`

```python
@dataclass(frozen=True)
class Profiles:
    slots: np.ndarray      # (n,) int64, ascending, one per 10-min slot (a complete grid)
    booms: tuple[int, ...] # (m,) ascending by height
    heights: np.ndarray    # (m,) metres
    ws: np.ndarray         # (n, m) float64 m/s, NaN = missing or filtered
    extra: Mapping[str, np.ndarray]  # optional (n, m) arrays (e.g. "wd") for algorithms that want them
```

`ttu_tower.jets.data.load_profiles(run_dir, booms=…)` builds one from a run's tertiary output.
Tests and notebooks can also build one directly from arrays.

### 2.2 Parameters

Each algorithm declares a frozen dataclass of its parameters. Every field carries metadata:
- a label
- a unit
- a kind: float, optional float, int, or boom set
- limits
- a one-line help text

The GUI builds its form from that metadata, so a new algorithm needs no GUI code. A parameter
set is plain data: it can be written to and read from JSON, which is what a future export or a
saved preset needs.

### 2.3 Output: one row per slot

```python
def detect(profiles: Profiles, params) -> pd.DataFrame  # indexed by slot, n rows
```

Required columns, the same for every algorithm:

| Column | Type | Meaning |
|---|---|---|
| `is_jet` | bool | This slot shows a jet under these parameters. |
| `status` | category | Why: `jet`, or the first rule the slot failed (algorithm-specific codes, §3.3). |
| `nose_boom` | Int8 | The nose's boom (the jet's, or the best rejected candidate's; see §3.3). |
| `z_nose`, `u_nose` | float | Nose height (m) and speed (m/s). |
| `z_min_above`, `u_min_above` | float | Height and speed of the slowest valid boom above the nose. |
| `falloff_abs` | float | `u_nose − u_min_above` (m/s). |
| `falloff_rel` | float | `falloff_abs / u_nose` (fraction). |

The nose and falloff columns are filled for rejected slots too, whenever a candidate existed.
That's what lets the GUI show near-misses and what each threshold removes.

Algorithms may add columns of their own, such as a fit's residual. The time step and the
filters read only the required columns.

### 2.4 Registry

```python
@register("nose", label="Peak (local maximum with falloff)", params=NoseParams)
def nose(profiles: Profiles, params: NoseParams) -> pd.DataFrame: ...

ALGORITHMS: dict[str, Algorithm]  # key -> (label, params class, function)
```

All of this lives in `ttu_tower/jets/identify.py`. Algorithms must be vectorized over slots,
with no per-row Python loop, so a year reruns in well under a second while the user tunes.

---

## 3. Algorithm `nose`

### 3.1 Parameters and defaults

The viewer labels these with "peak" for what this section calls the nose, the boom the
algorithm finds (§4.2): for example `min_nose_speed` is shown as "min peak speed". The
algorithm's key (`nose`) and the parameter names are unchanged, so saved settings still load.

| Parameter | Default | Meaning |
|---|---|---|
| `profile_booms` | 1–10 | The booms that make up the profile. Nose candidates and the falloff are found among these. |
| `required_booms` | 7–10 (74.7, 116.5, 158.2, 200 m) | Booms that must all be present and good, or the slot is `no_data`. |
| `min_nose_speed` | 10 m/s | The nose speed must be at least this. |
| `min_falloff_abs` | 2 m/s | The nose must be at least this much faster than the slowest boom above it. Can be switched off. |
| `min_falloff_rel` | off | If set (e.g. 0.25), the falloff must also be at least this fraction of the nose speed. Can be switched off. |

At least one of the two falloff thresholds must be on. When both are on, a slot must pass
both. That's the form of Baas et al. (2009), who use 2 m/s and 25 %.

The time-step parameters (§4) aren't part of the algorithm but are tuned alongside it:

| Parameter | Default | Meaning |
|---|---|---|
| `max_gap_slots` | 1 (10 min) | The longest run of rejected slots bridged between two jet slots. Only slots with data that failed the rules count, unless `bridge_missing` is on. |
| `bridge_missing` | off | Also bridge `no_data` slots: an outage, or a slot whose required boom is missing or flagged. |
| `min_event_slots` | 6 (1 h) | The shortest event kept, counted after gap filling. |

The defaults keep the old pipeline's choices, except for:
- the minimum nose speed, raised from 5 to 10 m/s
- the falloff threshold, changed from a 0.005 s⁻¹ rate to an absolute and/or relative drop

### 3.2 Rules, per slot

Let the profile be the valid booms among `profile_booms`, ordered by height, with speeds u_i
at heights z_i.

1. **Required booms.** If any of `required_booms` is missing or bad: `status = no_data`, not a
   jet.
2. **Candidates.** A boom i is a candidate nose if all of these hold:
   - it isn't the lowest or highest boom of `profile_booms`
   - the next boom of `profile_booms` below it and the next one above it are both valid; a
     missing neighbour can't confirm a peak, so the candidate is dropped rather than skipping
     over the gap
   - it is a strict local maximum: u_i > u_below and u_i > u_above

   **This rules out a peak at the top:** the highest boom (200 m by default) is never a
   candidate.
3. **Falloff above the nose.** For each candidate:
   - u_min_above = the minimum over every valid profile boom above it, up to the top
   - falloff_abs = u_i − u_min_above
   - falloff_rel = falloff_abs / u_i

   The falloff is measured above the nose itself, not at some unrelated pair of booms (the old
   method's flaw). The required booms guarantee that the booms above a high nose exist.
4. **Thresholds.** A candidate qualifies if u_i ≥ `min_nose_speed` and passes the falloff
   threshold or thresholds that are on.
5. **The nose.** If any candidate qualifies, the slot is a jet and its nose is the fastest
   qualifying candidate. Ties go to the lower boom.

### 3.3 Status codes and the reported candidate

| `status` | Meaning | Nose columns report |
|---|---|---|
| `no_data` | A required boom is missing or bad. | NaN |
| `no_peak` | No candidate: no strict interior local maximum with valid neighbours. | NaN |
| `too_slow` | Candidates exist, but none reaches `min_nose_speed`. | The fastest candidate |
| `weak_falloff` | Some candidates are fast enough, but none has enough falloff. | The fast-enough candidate with the largest `falloff_abs` |
| `jet` | At least one candidate qualifies. | The chosen nose |

The status is the first rule that failed, in the order above. So a diagnostic count tells you,
for example, how many slots were lost to the falloff threshold alone.

### 3.4 What this fixes in the old `classify_jet_discrete`

| Old behaviour | Now |
|---|---|
| Max over all booms, plus any decreasing pair among the top four. The decrease didn't need to be above the max, so a peak at 200 m, or at 47 m with a dip near 200 m, counted. | The nose must be a strict local maximum, and the falloff is measured above it. |
| Falloff as a rate, 0.005 s⁻¹ across one ~42 m pair: a 0.21 m/s drop, within the noise of a 10-min mean. | An absolute (2 m/s) and/or relative falloff, measured to the slowest boom above. |
| Missing booms skipped, so one bad boom could make or hide a jet. | Required booms must all be present; candidates need valid neighbours. |
| Minimum speed 5 m/s, strict `>`, a module constant. | 10 m/s, `≥`, a parameter. |
| Booms, gap fill and minimum length hard-coded; the threshold argument never passed through. | All are parameters. |
| Returned only `(bool, z, u)`. | Returns a status, the falloff and the candidate for every slot. |
| Row-by-row `df.apply` over the old wide table's column names. | Vectorized arrays built from the new tables. |

### 3.5 Limits that come from the tower, not the algorithm

- **Noses above 158.2 m can't be detected.** A nose needs a boom above it, so with the default
  booms it can only be at 158.2 m or below. A jet whose nose is at or above 200 m appears only
  as speed rising with height.
- **"Falloff above the nose" only reaches 200 m.** A jet whose speed drops by 2 m/s only above
  200 m is missed. Counts are therefore a lower bound on how often LLJs occur.
- **The top-boom requirement costs about a third of the data.** On `oneyear`, 26,739 of the
  40,732 slots with wind data have all of booms 7–10 present and good.

---

## 4. From slots to events (shared by all algorithms)

This lives in `ttu_tower/jets/events.py`. It reads `is_jet` and `status`, plus the nose columns for the
statistics.

1. **Gap filling.** Between two jet slots, a run of at most `max_gap_slots` non-jet slots is
   marked jet and `filled`, but only if every slot in the run is bridgeable.
   - **Bridgeable:** a slot that has data and failed the rules (`no_peak`, `too_slow` or
     `weak_falloff`).
   - **Missing data:** with `bridge_missing` on, a `no_data` slot is bridgeable too, whether
     it's an outage or a required boom is missing or flagged. With it off, missing data
     always ends an event. Filled slots keep their own `status` and
   nose values (no interpolation). Gaps at the start or end of the period are never filled.
2. **Minimum length.** Runs of jet slots (detected plus filled) shorter than `min_event_slots`
   are dropped.
3. **Events.** Each remaining run is an event, numbered in time order. Its columns:
   - `event_id`, `start_slot`, `end_slot`, `n_slots`, `n_filled`, `n_detected`
   - `duration_min` = 10 · `n_slots`
   - `start_time` and `end_time` in the display zone
   - `truncated` if the event touches the first or last slot of the data loaded
4. **Event statistics.** These are computed from the **detected** slots only; filled slots
   contribute nothing:
   - nose height: median, minimum and maximum
   - nose speed: maximum and mean
   - `falloff_abs`: maximum and mean
   - the wind direction at the nose: circular mean of `wd` at each slot's nose boom
   - `peak_slot` and `peak_boom`: the fastest detected slot and its nose boom (where the
     viewer opens the Slot Inspector)
   - the filter quantities in §5

The order (fill first, then drop short runs) is the old pipeline's: a binary closing of width 3,
then an opening of width 6. Two things differ:
- The new code uses explicit run lengths, so events at the ends of the data aren't trimmed,
  which scipy's morphology did.
- The old closing also bridged missing slots. By default only rejected slots are bridged now
  (Elliott, 2026-10-01); `bridge_missing` restores bridging missing data as an option (Elliott,
  2026-10-02).
- **On `oneyear`**, every one-slot gap that splits an event under the defaults is a slot
  missing a single required boom:
  - 10 Dec 2013 04:30, 28 Dec 2013 00:10 and 5 May 2014 17:00, each missing boom 10
  - 25 Jan 2014 03:50, missing boom 7

  With night only and ΔVPT ≥ 1 K:

  | `bridge_missing` | Gap | Events | Kept | Median length |
  |---|---|---|---|---|
  | off | 1 | 20 | 11 | 1.2 h |
  | on | 1 | 19 | 11 | 1.8 h |
  | on | 3 | 19 | 9 | 2.0 h |

  With the option on and gap 1, the 10 Dec jet becomes one 3-h event, 27–28 Dec one of
  2:20 h, and 25 Jan one of 6:10 h.

### 4.1 Manual edits (Elliott, 2026-10-02)

Events can be changed by hand. The edits are kept per run, in
`<home>/viewer/jet_edits/<tag>.json`, and are reapplied after every detection, so they survive
changed settings. The code is in `ttu_tower/jets/edits.py`.

**Kinds of edit:**

| Kind | What it does | How it's found again |
|---|---|---|
| new boundaries (`span`) | the event's new first and last slot | the event holding its `anchor`, one of the event's detected slots |
| made by hand (`add`) | an event over a chosen span | — |
| rejected (`reject`) | the event is not a jet: listed, but not kept | the event holding its `anchor` |

**How edits apply:**
- **Lost anchor:** if no detected event holds a new-boundaries edit's anchor any more, the span
  stays as an event made by hand (`manual`).
- **Absorbing:** detected events that overlap an edited or hand-made span are absorbed into it,
  and overlapping hand-made spans are joined.
- **Inactive rejections:** a rejection matching no event does nothing until one matches again.
- **Labels:** every event has a `source`: `auto`, `edited` or `manual`. Slots added by hand
  (in the span, neither jet nor bridged) are marked `manual` per slot. The edit report says what
  each edit does now: applied, kept as a manual event, inactive, or outside the data.

**Which slots count:**
- **Spans:** the profiles, the composite profile, ΔVPT and Ri_b use every slot of an event's
  span, hand-added and bridged slots included.
- **Noses:** an event's nose statistics come from its jet slots by default. The viewer's
  "every slot of an event is a jet slot" switch takes them from each slot's best candidate
  instead.
- **No nose:** an event made by hand without a jet slot has no nose statistics in the default
  mode.

---

### 4.2 Observed peak and estimated nose (Elliott, 2026-10-02)

**Detection works on observations only.** The nose the algorithm finds is the fastest boom,
which the viewer calls the **observed peak**: its height (`z_nose`) and speed (`u_nose`).

**The estimate, for describing identified events:** `ttu_tower/jets/noses.py` fits a
quadratic in height through the observed peak and the profile booms directly below and above
it. The profile booms are the same ones the detection used, so the two neighbours are exactly
those the local-maximum test compared against. The fit's maximum is the **estimated nose**:
`z_nose_est` and `u_nose_est`.
- The peak is a strict local maximum, so the quadratic opens downward and its maximum lies
  strictly between the two neighbours.
- The estimate is NaN where a neighbour is missing; that can only happen for candidates the
  detection didn't use.

**Where it appears:**
- `analyse` adds the two columns to the detection frame.
- Every event gains `z_est_median`, `z_est_min`, `z_est_max`, `u_est_max` and `u_est_mean`,
  beside the observed-peak `z_nose_*` and `u_nose_*`.
- Sweeps don't compute it.

**On `oneyear` (defaults), all 341 jet slots have an estimate:**

| Observed peak | Slots | Estimated nose, median (range) | Estimated − observed speed, median (max) |
|---|---|---|---|
| 47.3 m | 4 | 58 m (52–61) | 0.18 (0.21) m/s |
| 74.7 m | 86 | 86 m (64–96) | 0.21 (1.04) m/s |
| 116.5 m | 213 | 118 m (96–136) | 0.01 (0.51) m/s |
| 158.2 m | 38 | 154 m (139–163) | 0.04 (1.12) m/s |

Peaks observed at 74.7 m tend to have their estimated nose above that boom, between 74.7 and
116.5 m.

---

## 5. Event filters (viewer)

These apply to **events** after grouping, so a filter keeps or drops whole events and never
splits one. Each filter is off until enabled. They live in `ttu_tower/jets/filters.py`, so a
later export can apply the same filters.

| Filter | Quantity | Default booms | Event statistic | Test |
|---|---|---|---|---|
| Time of day | the event's start, in the display zone | — | start time | inside a clock window, which may wrap midnight (e.g. 18:00–09:00) |
| Night only | `slot_final.night` (sun below the horizon) | — | the start slot | `night == 1` |
| Inversion strength | ΔVPT = vpt(upper) − vpt(lower), K | lower 2, upper 10 | mean over detected slots | ≥ a threshold (positive = stable) |
| Richardson number | Ri_b(lower, upper) | 2 and 4 | median over detected slots | inside [min, max]; either end may be open |

The statistics are taken over every slot of the event's span (§4.1); before manual edits
existed, only its detected slots counted. An event rejected by hand is never kept, whatever
the filters say.

The event table shows these statistics as columns, so they can also just be sorted on. A slot
with a NaN filter quantity doesn't count toward the statistic. An event with no finite value
fails an enabled filter, and the viewer reports how many events were dropped that way.

---

## 6. On `oneyear` (implementation, 2026-10-01)

Elliott wants quality over quantity, a few events that are certain, so the strict defaults
stay.

The detection runs on the run's full slot grid of 52,527 slots, with gap 1 and minimum 6 slots.
The last column applies night only plus ΔVPT(b10 − b2) ≥ 1 K.

| Criteria | Jet slots | Events | Median event length | Night + inversion |
|---|---|---|---|---|
| Defaults: ≥ 10 m/s, falloff ≥ 2 m/s | 341 | 20 | 1.2 h | 11 |
| Defaults plus falloff ≥ 25 % | 97 | 4 | 1.25 h | 4 |
| ≥ 8 m/s, falloff ≥ 2 m/s | 509 | 29 | 1.2 h | 15 |
| ≥ 10 m/s, falloff ≥ 1 m/s | 970 | 52 | 1.4 h | 28 |

Under the defaults the noses were at:

| Nose height | Jet slots |
|---|---|
| 116.5 m | 213 |
| 74.7 m | 86 |
| 158.2 m | 38 |
| 47.3 m | 4 |

The slots sort by status as follows:

| `no_data` | `no_peak` | `too_slow` | `weak_falloff` | `jet` |
|---|---|---|---|---|
| 25,788 | 12,772 | 10,149 | 3,477 | 341 |

**What changed from the prototype:**
- The earlier prototype bridged any one-slot gap and gave 19 events. Not bridging missing data
  splits some events into two shorter ones, for example on 10 Dec 2013 at a one-slot gap with
  no data. That's why there are now 20 events and a shorter median.
- **Daytime events:** several April–May events are daytime events with negative ΔVPT. They have
  a nose and a 2 m/s falloff but come with an unstable lower layer. The night and inversion
  filters remove them.
