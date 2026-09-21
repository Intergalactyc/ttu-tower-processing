# Output schema

The contract between the pipeline stages and for downstream users (`ttu-windprofiles` will be
rebuilt against it). All tables are **tidy long** Parquet with explicit key columns;
wide format appears only in the final convenience export. Nothing here is compatible with
`old-tower-processing` output; a mapping from old column names is at the end.

When implementation forces a change, update this file in the same phase (plan.md §1.3).

---

## 1. Conventions

### 1.1 Time and keys

- **Epoch:** `2000-01-01 00:00` local standard time (`Etc/GMT+6`, i.e. 06:00 UTC).
- **`slot`** (int64): 10-minute slot index since the epoch. Slot k covers
  `[epoch + 10k min, epoch + 10(k+1) min)`.
- **`slot_start`** (timestamp, ns, tz-aware, in `[output].timezone`: `Etc/GMT+6` by default,
  or `UTC`): the slot's start. It appears in slot-level tables and in all tertiary outputs.
  Every timestamp column in every table carries its zone (never naive), and CSV reports write
  ISO 8601 with the offset (e.g. `2013-11-01T00:00:00-06:00`). The epoch and all integer
  indices are the same whichever zone is chosen. Big primary and secondary tables carry `slot`
  only; use `ttu_tower.timegrid.slot_to_time` to convert.
- **`half_hour`** (int64): 30-minute (file) index since the epoch; file k holds slots
  3k, 3k+1, 3k+2.
- **Sample index `g`** (int64): 50 Hz sample index since the epoch; slot k holds samples
  `[30000k, 30000(k+1))`. Flag intervals use sample indices.
- **`boom`**, **`boom2`** (int8): 1–10. Heights in m: 0.9, 2.4, 4.0, 10.1, 16.8, 47.3, 74.7,
  116.5, 158.2, 200.0.

### 1.2 Categorical columns

All string-valued key columns (`variant`, `variable`, `stat`, `test`, `kind`, `spectrum`,
`family`, `status`, …) are Parquet dictionary-encoded strings, read as pandas `category`.
`value` columns are float64. Missing values are NaN (floats) or null (others).

### 1.3 Units and frames

SI throughout: m/s, K, kPa, relative humidity as a fraction, degrees for angles, seconds for
times. Wind directions are **true FROM-bearings** (degrees clockwise from north).

Two frames appear:
- **Earth frame:** `ue` (east), `vn` (north), `w` (up). These are "blows-toward" components
  after tilt correction.
- **Streamwise frame:** `u` (along the mean wind), `v` (horizontal, 90° to the *left* of the
  mean wind, so (u, v, w) is right-handed), `w` (up). Ladder statistics use each τ-block's own
  mean wind. MRD spectra use the detection window's mean wind (`mrd_frame`).

### 1.4 Variants

| `variant` | Meaning | Where |
|---|---|---|
| `mrd` | Excised data (final usable mask), selected τ. The primary result. | primary, secondary, tertiary |
| `mrd_unexcised` | No mask; within-run gaps ≤ 10 min interpolated. Counterfactual. | primary (where computed), secondary and tertiary (complete) |
| `naive` | The 600-s rung of `mrd`. | secondary, tertiary |
| `none` | Variant-independent quantities (means, slow sensors, sun, Ri_b, …). | tertiary |

Primary writes `mrd_unexcised` rows only for (slot, boom) where `slot_boom.unexcised_computed`
is true. For the other `computed` slots they would equal `mrd`, and secondary materializes them
by copying the `mrd` rows. From secondary on, every variant is complete and explicit.

### 1.5 Families (combined masks)

| Family | Mask | Used by |
|---|---|---|
| `momentum` | ue ∧ vn ∧ w | wind components, speed, direction, momentum fluxes, variances of u, v, w |
| `heat` | w ∧ vpts (vpts mask = ts mask; vpts uses a fixed per-boom reference pressure in primary) | w'θv', vpts statistics |
| `ts` | ts | ts statistics |

### 1.6 Statuses and vocabularies

- `slot_boom.status`: `no_file` (no accepted file covers the slot), `no_data` (file present but
  no final-usable sample in either the `momentum` or the `ts` family, whether the data are
  missing or entirely flagged — nothing in this schema is then computable), `computed`.
- τ detection status: `no_data`, `weak`, `found`, `capped`, `unresolved`.
- τ source: `heat`, `momentum`, `fallback` (600 s: neither cospectrum gave a τ or a lower
  bound), `fixed` (naive), `none` (no data in either family).

---

## 2. Directory layout

Results are registered in the **ttu-tower home** (`~/.ttu-tower`, or the first
`~/.ttu-tower<N>` carrying the signature file; plan.md §3.10), and loaded by tag with
`ttu_tower.load_results(tag, table)`:

```
~/.ttu-tower/
  ttu-tower-home.json             signature
  runs/<tag>.json                 link file: {"tag", "run_dir", "config", "registered", "package_version"}
  runs/testing/<tag>.json         the same, for --test runs
  results/                        run directories, when [paths].results_dir is ""
```

A **run directory** is `<R>/<tag>/` (`<R>/testing/<tag>/` for `--test` runs), where R is
`[paths].results_dir`, or the home's `results/` if that is empty:

```
<run dir>/
  primary/
    config.toml                   copy of the config file used
    config.resolved.json          every key, defaults filled in
    run_meta.json                 see §6
    run_summary.json              see §6
    files.parquet
    slots.parquet
    manifest/<unit>.json          one per (batch, boom) work unit, with its summary record
    data/<table>/<unit>.parquet   fragments of the tables in §3
    logs/primary.log
  secondary/
    config.toml, config.resolved.json, run_meta.json, run_summary.json
    data/<table>/<batch>.parquet
    logs/secondary.log
  tertiary/
    config.toml, config.resolved.json, run_meta.json, run_summary.json
    data/<table>/<batch>.parquet
    wide/<YYYY-MM>.parquet        the convenience export, one file per month (§5.7)
    logs/tertiary.log
  reports/
    *.csv
```

`<unit>` = `bat{h_a:07d}-{h_b:07d}_b{boom:02d}`, where `h_a`..`h_b` are the half-hour indices
of the batch's core files; `<batch>` is the same without the boom suffix. Fragments are
written atomically (temp file + rename). A table is read by unifying the schemas of all its
fragments (`ttu_tower.io.store.read_table`).

---

## 3. Primary tables

### 3.1 `files.parquet`: the file table (one row per raw file found)

| Column | Type | Meaning |
|---|---|---|
| `path` | string | Absolute path. |
| `name` | string | File name (`FT2_E07_C03_R…_D…_T….parquet`, or `.csv`/`.csv.gz`/`.zip` for unconverted files). |
| `record` | int32, null | `R` number; null if unparseable. |
| `name_time` | timestamp tz | Time encoded in the name. |
| `offset_min` | int16, null | Minutes after the preceding :00/:30 boundary; null if unparseable. |
| `file_start` | timestamp tz | Rounded-down boundary (set for every parseable file). |
| `half_hour` | int64, null | Half-hour index of `file_start`; null if unparseable. |
| `n_rows` | int32, null | From Parquet metadata; null for unconverted files (checked when loaded, with `--allow-non-parquet`). |
| `status` | category | `accepted`, `unparseable`, `bad_record`, `offset`, `collision`, `bad_length` (checked in that order). |

### 3.2 `slots.parquet`: every slot in the processing period

| Column | Type | Meaning |
|---|---|---|
| `slot`, `slot_start`, `half_hour` | | Keys (§1.1). |
| `file_status` | category | `file` / `no_file`. |
| `record` | int32, null | Record number of the covering file. |

### 3.3 `slot_boom`: every slot × boom in the period

| Column | Type | Meaning |
|---|---|---|
| `slot`, `boom` | | Keys. |
| `status` | category | `no_file`, `no_data` (no final-usable ue/vn/w/ts sample: missing *or* entirely flagged), `computed`. |
| `unexcised_computed` | bool | `mrd_unexcised` rows exist for this slot and boom. This includes `no_data` slots whose data were all flagged: the counterfactual is computed there. |

### 3.4 `coverage`: data availability per slot (every slot with a file: `computed` and `no_data`)

Columns: `slot`, `boom`, `variable`, `layer`, `fraction`.

- `variable` ∈ {`ue`, `vn`, `w`, `ts`, `vpts`, `t`, `rh`, `p`, `momentum`, `heat`}.
- `layer`:
  - `present`: non-NaN in the source file.
  - `filled`: finite after the ≤ 1-s interpolation (the counterpart of `unexcised` below).
  - `usable_l1`: first-layer usable.
  - `usable`: final usable.
  - `unexcised`: available to `mrd_unexcised` (finite after within-run filling).

  The slow sensors (`t`, `rh`, `p`) have `present` and `usable` only.

### 3.5 `means`: variant-independent slot statistics on the final mask

Columns: `slot`, `boom`, `variable`, `stat`, `value`.

| variable | stat | Meaning |
|---|---|---|
| `ue`, `vn`, `w` | `mean` | Earth-frame component means (momentum family). |
| `ws` | `mean` | Scalar mean of per-sample horizontal speed. |
| `ws` | `vector_mean` | Magnitude of the mean horizontal vector. |
| `wd` | `mean` | FROM-bearing of the mean horizontal vector (speed-weighted). |
| `wd` | `unit_mean` | FROM-bearing of the mean unit vector: each usable sample contributes its direction only, so gusts don't dominate. Samples with ws = 0 are left out; NaN if none remain. |
| `ws`, `w` | `max` | Largest single sample. |
| `ws`, `w` | `max_{p}s`, `std_{p}s` | Largest value and standard deviation of p-second block means (p in `gust_periods_s`; blocks with coverage < c skipped; speed = magnitude of the block-mean vector). |
| `ts`, `vpts` | `mean` | Sonic temperature; sonic virtual potential temperature at the boom's reference pressure P_REF (K). The pressure-corrected value is in `slow` (§4.5). |
| `t`, `rh`, `p` | `mean` | Slow sensors (smoothed series, each on its own mask). |

### 3.6 `slot_qc`: QC diagnostics per slot

Columns: `slot`, `boom`, `variable`, `stat`, `value`.

- `ue`, `vn`, `w`, `ts` × `skew`, `kurt`: higher moments (NaN if coverage < c).
- `wd` / `mean_l1`, `ws` / `mean_l1`, `ws` / `std_l1`: the first-layer inputs to the
  `direction` and `bounce` tests.

### 3.7 `flags`: the interval-keyed flag store

| Column | Type | Meaning |
|---|---|---|
| `start`, `end` | int64 | Sample-index interval `[start, end)`. |
| `test` | category | See below. |
| `kind` | category | `quality`, `record` (future: `assumption`). |
| `boom` | int8 | |
| `variable` | category, null | null = boom-level test, applying to `ue`, `vn`, `w`, `ts`. |

| test | kind | Grain | Variables | Meaning |
|---|---|---|---|---|
| `bounds` | quality | sample runs | ue, vn, w (coupled), ts, t, rh, p | Outside `[qc.bounds]`; removed. |
| `spike` | quality | sample runs | same | Outlier run of ≤ `max_spike_samples` samples, after splitting runs where they cross the median; removed. |
| `excursion` | record | sample runs | same | Outlier run longer than that, on one side of the median; kept. Input for a future discontinuity flag. |
| `unchecked` | quality | sample runs | despiked variables | No despiking reference within half a window. |
| `resolution` | quality | 100-s windows | ue, vn, w, ts | Empty-bin fraction > `max_empty_bin_fraction`. |
| `dropouts` | quality | 100-s windows | ue, vn, w, ts | Dropout-pair fraction > `max_dropout_fraction`. |
| `skew`, `kurt` | quality | slot | ue, vn, w, ts | Outside `skew_range` / `kurt_range`. |
| `direction` | quality | slot | null | First-layer wd in `shadow_sector`. |
| `bounce` | quality | slot | null | First-layer ws mean or std over its limit. |

Intervals of one (test, boom, variable) are merged within a work unit's core and may be split
at unit boundaries. Semantics are the union of the intervals. Query through
`ttu_tower.flags.FlagStore` (flagged-sample counts and fractions over arbitrary supports).

### 3.8 `ladder`: statistics at every rung

Columns: `slot`, `boom`, `variant` (`mrd` | `mrd_unexcised`), `rung_s`, `variable`, `stat`,
`value`.

`rung_s` ∈ {9.375, 18.75, 37.5, 75, 150, 300, 600, 1200}. For rung ≤ 600, the value is the
mean over the valid τ-blocks tiling the slot. For rung 1200, it is the single window
`[slot_start − 5 min, slot_start + 15 min)`.

| variable | stat | Family | Meaning |
|---|---|---|---|
| `u`, `v`, `w` | `var` | momentum | Variances, each block rotated to its own mean wind (m²/s²). |
| `uv`, `uw`, `vw` | `cov` | momentum | Covariances, same frame (m²/s²). |
| `ws` | `mean`, `std` | momentum | Scalar speed over the rung's valid blocks (std = mean over blocks of within-block std², square-rooted). |
| `u` | `mean` | momentum | Mean over valid blocks of the block's mean-vector magnitude. |
| `w` | `mean` | momentum | |
| `wd` | `std` | momentum | Yamartino σ_θ (deg): √(mean over blocks of σ_θ,b²). |
| `wvpts` | `cov` | heat | w'θv' (K m/s), at the reference pressure P_REF[b], like every heat quantity in primary; secondary converts it (§4.3). |
| `vpts` | `var`, `mean` | heat | |
| `ts` | `var`, `mean` | ts | |
| `u`, `v`, `w` | `its` | momentum | Integral timescale (s), e-folding, pooled ACF over fully usable blocks; NaN if the ACF never reaches 1/e within the maximum lag. |
| `vpts` | `its` | ts | The same, for the sonic virtual potential temperature, on blocks fully usable in the `ts` family and with no rotation. It equals the ITS of `ts`, which differs from `vpts` only by a constant factor. |
| `uw` | `te` | momentum | Momentum transport efficiency. |
| `wvpts` | `te` | heat | Heat transport efficiency. |

Variances and covariances within a block use the population normalization (1/n, fractional
weights), matching the MRD identity.

### 3.9 `ladder_coverage`

Columns: `slot`, `boom`, `variant`, `rung_s`, `family`, `blocks_used` (int16), `blocks_total`
(int16), `coverage` (float64).

- `family` ∈ {`momentum`, `heat`, `ts`, `its`, `its_vpts`} (the last two count fully usable
  blocks, in the momentum and `ts` families).
- `coverage` = usable sample weight in the support / samples in the support. For `its` and
  `its_vpts`, coverage = `blocks_used` / `blocks_total`, counting only fully usable blocks.

### 3.10 `mrd`: detection outputs

Columns: `slot`, `boom`, `variant`, `spectrum`, `scale_s`, `value`, `se`, `n_pairs` (int32).

- `spectrum` ∈ {`wvpts`, `uw`, `vw`, `uv`, `uu`, `vv`, `ww`, `tsts`}. The u/v spectra are in
  the `mrd_frame` frame.
- `scale_s` = P_i = 0.29296875 × 2^(i−1), i = 1..15 (0.29 s … 4800 s, for K = 15): the mode's
  averaging time scale, i.e. the width of the blocks it is computed in. Each block is a pair of
  halves; mode i has 2^(15−i) blocks in the window.
- `value` = mean over valid blocks of ΔxΔy/4 (Δ = the difference of the two half means). `se`
  = sample standard deviation of the block products / √n_pairs (NaN if n_pairs < 2). `value`
  is NaN if n_pairs = 0. `n_pairs` = the number of valid blocks.
- **Identity:** Σ_{P_i ≤ τ} `value` = mean within-τ-block covariance at floor resolution, when
  every block is valid.
- The `wvpts` spectrum uses the reference-pressure vpts; a constant factor doesn't affect
  detection.

### 3.11 `mrd_frame`

Columns: `slot`, `boom`, `variant`, `wd_deg`, `coverage_momentum`, `coverage_heat`.

`wd_deg` is the FROM-bearing of the detection window's momentum-family mean wind; x points
downwind (toward `wd_deg + 180`). The coverages are the usable fractions of the whole 80-min
window.

---

## 4. Secondary tables

Secondary tables have rows for each (slot, boom, variant) that primary computed (`naive`
wherever `mrd`); `slow` has rows for every slot and boom with `means` rows.

### 4.1 `tau`: per-cospectrum detection results

Columns: `slot`, `boom`, `variant` (`mrd` | `mrd_unexcised`), `cospectrum` (`heat` |
`momentum`), `status`, `tau_s`, `tau_lb_s`, `sign` (int8), `peak_scale_s`,
`reversal_scale_s`, `reversal_type` (`sign` | `increase` | null).

The algorithm is in plan.md Phase 7: peak and reversal are located on the 1–2–1-smoothed
cospectrum, over the modes from `min_scale_s` to 40 min.
- `tau_s` is set for `found` and `capped`; `tau_lb_s` for `unresolved`.
- `sign` is the sign of the smoothed cospectrum at the peak; 0 if no peak was found.
- `peak_scale_s` is the scale of the peak (NaN if none). For `found`, `reversal_scale_s` is the
  scale at which the reversal was seen, and `tau_s` the scale just below it, raised to
  `min_tau_s` if smaller.
- For `capped`, `reversal_scale_s` = 4800 with a `reversal_type` means the single-block 80-min
  mode showed an unconfirmed reversal, which would put the gap at 40 min, beyond the cap.

### 4.2 `tau_selected`

Columns: `slot`, `boom`, `variant` (`mrd` | `naive` | `mrd_unexcised`), `tau_s`, `source`,
`source_status`.

`source_status` is the chosen cospectrum's status (`found`, `capped`, `unresolved`), or
`fallback`, `fixed` or `none`.

### 4.3 `boom_stats`: selected-τ statistics and derived quantities

Columns: `slot`, `boom`, `variant`, `variable`, `stat` (null for scalar derived quantities),
`value`.

It contains:
- every `ladder` row at the selected rung, with the same (variable, stat). The heat rows are
  converted to the measured slot pressure with the slot's `p_factor` (§4.5): `wvpts` cov and
  `vpts` mean × `p_factor`, `vpts` var × `p_factor`²; `wvpts` te is a ratio and unchanged;
- the support coverage: `momentum` / `heat` / `ts` / `its` / `its_vpts` × `coverage`,
  `blocks_used`, `blocks_total`;
- the derived quantities below (stat null), computed from the converted rows.

| variable | Definition |
|---|---|
| `sigma_u`, `sigma_v`, `sigma_w` | √var |
| `ti`, `ti_u`, `ti_v`, `ti_w` | (ws std, σ_u, σ_v, σ_w) / ws mean over the same support |
| `ti_ratio_vu`, `ti_ratio_wu` | ti_v/ti_u, ti_w/ti_u |
| `tke` | ½(var_u + var_v + var_w) |
| `ctke` | ½√(cov_uw² + cov_vw² + cov_uv²) |
| `ustar` | (cov_uw² + cov_vw²)^¼ |
| `obukhov_length` | −u*³ θv / (κ g w'θv'), θv = slow-sensor VPT, g = local gravity, κ = 0.41 |
| `zeta` | z / L |
| `ils_u`, `ils_v`, `ils_w`, `ils_vpts` | ITS × (u mean), with \|u mean\| for v, w and vpts |
| `ils_ratio_vu`, `ils_ratio_wu` | |
| `its_ratio_u`, `its_ratio_v`, `its_ratio_w`, `its_ratio_vpts` | τ / ITS (NaN where the ITS is) |
| `its_short` | 1.0 if any of u, v, w has `its_ratio` < `min_tau_its_ratio`, or has usable blocks but an ITS of NaN (its ACF never crossed 1/e, the extreme case of the same problem); else 0.0; NaN only if none of the three had a usable block |
| `its_short_vpts` | the same for vpts, kept separate so that a long temperature scale doesn't discredit the velocity scales |
| `aniso_l1`, `aniso_l2`, `aniso_l3` | Eigenvalues of b_ij = R_ij/(2k) − δ_ij/3, in descending order |
| `aniso_beta`, `aniso_phi` | Inclination and orientation angles (deg), after Gucci et al. (2025), ported |
| `w_ratio` | w mean / ws mean (tilt diagnostic) |

A zero denominator gives NaN, never inf.

### 4.4 `boom_labels`

Columns: `slot`, `boom`, `variant`, `label`, `value` (string). Currently `aniso_class` ∈ {`1c`,
`2c`, `3c`, `prolate`, `oblate`, `ellipse`, `mixed`} (barycentric map, k = 2/3, ported).

### 4.5 `slow`: slow-sensor thermodynamics (variant-independent)

Columns: `slot`, `boom`, `variable`, `value`. From slot means of the smoothed t (K), rh
(fraction) and p (kPa):

| variable | Definition |
|---|---|
| `t`, `rh`, `p` | the means (copied from primary) |
| `es` | Magnus, AERK coefficients (0.61094·exp(17.625·T/(T + 243.04)), T in °C), kPa |
| `e` | rh·es |
| `r` | mixing ratio 0.622 e/(p − e) |
| `q` | r/(1 + r) |
| `td` | Magnus dewpoint, AERK: 243.04·γ/(17.625 − γ) °C + 273.15, γ = ln(rh) + 17.625·T/(T + 243.04), K |
| `pt` | t (100/p)^(R/c_p) |
| `vt` | t (1 + r/0.622)/(1 + r) |
| `vpt` | pt (1 + r/0.622)/(1 + r) |
| `p_factor` | (P_REF[b]/p)^(R/c_p): converts primary's reference-pressure heat quantities to the measured slot pressure, for every variant and τ (§4.3); 1.0 when the slot's pressure coverage is below c |
| `p_measured` | 1.0 if `p_factor` used the measured pressure, 0.0 if the reference was kept |
| `vpts` | sonic virtual potential temperature slot mean, pressure-corrected (primary's `means` value × `p_factor`) |

### 4.6 `slot_stats`

Columns: `slot`, `sun_elevation` (deg, at the slot center), `night` (bool: slot center before
sunrise or after sunset).

---

## 5. Tertiary tables

All carry `slot_start` as well as `slot`, and have rows for every slot of the period with a
file; values are NaN where inputs are missing or filtered.

### 5.1 `boom_final`: filtered per-boom values

Columns: `slot`, `slot_start`, `boom`, `variant` (`none` | `mrd` | `naive` | `mrd_unexcised`),
`variable`, `stat`, `value`.

- `variant = none`: the rows of `means` and `slow`.
- Other variants: the rows of `boom_stats`.

Values that fail filtering are NaN (§5.6). `boom_labels` (the anisotropy class) isn't part of
this table - see `boom_labels_final` (§5.8).

### 5.2 `tau_final`

`tau_selected` plus `slot_start` (unfiltered, for reference).

### 5.3 `pairs`: multi-boom quantities

Columns: `slot`, `slot_start`, `boom`, `boom2`, `variant`, `variable`, `value`.

| variable | variant | Definition |
|---|---|---|
| `rib` | none | Bulk Richardson number between `boom` < `boom2` (all 45 pairs); vector wind difference from ue/vn means; slow VPT; local g. |
| `lapse_vpt` | none | (VPT₂ − VPT₁)/(z₂ − z₁), K/m, all pairs. |
| `veer` | none | Signed angular difference wd(`boom`) − wd(`boom2`), deg, where `boom2` is the reference boom; positive when `boom`'s direction is clockwise of the reference's. Includes the reference boom's own row (`boom == boom2`, value 0 unless the reference's own `wd` was filtered) - unlike `rib`/`lapse_vpt`, `veer` doesn't follow the `boom < boom2` convention, since `boom2` here is a fixed reference rather than a pair partner. |

### 5.4 `profile`: profile fits per slot

Columns: `slot`, `slot_start`, `variant`, `variable`, `value`.

| variable | variant | Definition |
|---|---|---|
| `alpha`, `alpha_n` | none | Power-law exponent of ws mean vs height (weighted fit); booms used. |
| `gamma`, `gamma_n`, `gamma_n_capped` | mrd / naive / mrd_unexcised | Power-law exponent of TI vs height; booms used; how many of them had a `capped` τ. |
| `wdgamma`, `wdgamma_n`, `wdgamma_n_capped` | as above | Power-law exponent of wd std vs height. |
| `loglaw_ustar`, `loglaw_z0`, `loglaw_n` | none | Unconstrained neutral log-law fit on `loglaw_booms` ws means; `loglaw_n` is the boom count shared by both (one fit produces both values, so there's one count, not two). |
| `loglaw_z0_constrained`, `loglaw_z0_constrained_n`, `loglaw_z0_constrained_n_capped` | mrd / naive / mrd_unexcised | z0 with u* fixed at the median of that variant's `ustar` over `loglaw_booms`; booms used; how many had a `capped` τ - same τ-policy treatment as gamma/wdgamma. |

**Profile-fit rules.** These are all of them. Each is configurable in `[tertiary.fits]`
(config-reference.md):
1. **Booms considered:** `alpha_booms`, `gamma_booms`, `wdgamma_booms`, `loglaw_booms`.
2. **Filtering first:** a boom whose input value was set to NaN by filtering (§5.6) is
   absent.
3. **τ policy**, for the second-moment fits of `mrd` and `mrd_unexcised` only (gamma, wdgamma,
   loglaw_z0_constrained): a boom is excluded when its selected τ's `source_status` or `source`
   is in `exclude_tau` (default `unresolved`, `fallback`). `capped` booms are included and
   counted in `<fit>_n_capped`. `naive` fits and mean-only fits (alpha, loglaw_ustar,
   loglaw_z0) have no τ exclusions.
4. **Minimum:** at least `min_booms` valid booms, else the fit is NaN. `<fit>_n` records how many
   booms were used.
5. **Method:** power laws use the y²-weighted log-space least squares (`math.fits.power_fit`);
   the log law uses `physics.most.neutral_loglaw_fit` and `constrained_neutral_loglaw_fit`.

### 5.5 `slot_final`

Columns: `slot`, `slot_start`, `variable`, `value`.

- `sun_elevation`, `night` (0/1).
- Mesonet (when `mesonet_dir` is set), resampled to 10 min with gaps linearly interpolated:
  `ws_meso_mean`, `ws_meso_vector_mean`, `wd_meso_mean`, `u_meso_mean`, `v_meso_mean`,
  `ws_meso_std`, `wd_meso_std`, `t_meso_mean`, `rh_meso_mean`, `p_meso_mean`, `solar`,
  `precip`, `ti_meso`.

### 5.6 `filter_log`: why values were filtered

Columns: `slot`, `boom`, `variant`, `group`, `criterion`, `variable` (null for coverage).

- `group` ∈ {`momentum`, `heat`, `ts`, `t`, `rh`, `p`, `vpt`}.
- `criterion` ∈ {`coverage`, `bounds`, `spike`} (future: `stationarity`).

Only failures are stored. Every quantity belongs to one group, defined by its inputs:

| group | Input variables | Required families | Quantities |
|---|---|---|---|
| `momentum` | ue, vn, w | momentum | wind means, gusts, u/v/w and uv/uw/vw statistics, ws/wd statistics, the velocity ITS/ILS and their ratios, `its_short`, ustar, TI, TKE, anisotropy, `uw` te, w_ratio |
| `heat` | ue, vn, w, ts | heat, momentum | wvpts cov/te, obukhov_length, zeta, ils_vpts (a ts quantity scaled by the mean wind) |
| `ts` | ts | ts | ts and vpts statistics and means, including the ITS of vpts, `its_ratio_vpts` and `its_short_vpts`; `p_factor`, `p_measured` (they travel with the vpts values they converted) |
| `t`, `rh`, `p` | itself | that variable | the slow mean |
| `vpt` | t, rh, p | t, rh, p | es, e, r, q, td, pt, vt, vpt |

The criteria are evaluated over the value's support: the slot, or the 20-min window when the
selected τ is 1200 s. A value fails if any required family's coverage is below `min_coverage`
(ladder coverage for ladder quantities, `coverage.usable` for means), or if any input
variable's `bounds` or `spike` fraction exceeds `max_bounds_fraction` / `max_spike_fraction`.

A timescale is computed only from blocks with no unusable sample at all, which is stricter than
coverage, so its block counts (`ladder_coverage` families `its` and `its_vpts`) are reported
rather than thresholded: an ITS from few blocks is noisy but not invalid, and a consumer who
wants more can require it.

### 5.7 `wide/<YYYY-MM>.parquet`: convenience export

One file per calendar month (in `[output].timezone`), indexed by `slot_start`; concatenating
the files in order gives the whole period. Column naming:
- per-boom: `{variable}[_{stat}]_b{boom}[_{variant}]` (the variant suffix is omitted for
  `none`);
- pairs: `{variable}_b{boom}-b{boom2}`;
- profile: `{variable}[_{variant}]`;
- slot: `{variable}`;
- τ: `tau_b{boom}_{variant}` and `tau_source_b{boom}_{variant}` (string).

### 5.8 `boom_labels_final`: filtered anisotropy labels

Columns: `slot`, `slot_start`, `boom`, `variant`, `label`, `value`.

Secondary's `boom_labels` (`aniso_class`, from the barycentric-map classification) copied
verbatim, except that `value` is null wherever that (slot, boom, variant)'s momentum group
failed tertiary filtering (§5.6) - the same failure that already NaNs `boom_final`'s
`aniso_l1`/`l2`/`l3`/`aniso_beta`/`aniso_phi` for that row. Without this table, `aniso_class`
could stay a real classification while the eigenvalues it was computed from are NaN'd in
`boom_final`, since secondary's own NaN-guard on `aniso_class` only catches a momentum family
whose *coverage* was too low (`var_u`/`cov_uv`/etc. are already NaN by the time they reach the
anisotropy calculation in that case) - it can't anticipate a later `bounds`/`spike`
flag-fraction failure, which secondary never evaluates.

---

## 6. Run metadata (`run_meta.json`, one per stage)

```json
{
  "stage": "primary",
  "tag": "oneyear",
  "config_hash": "…",              // hash of the sections this stage depends on
  "upstream_hash": null,           // secondary/tertiary: the previous stage's config_hash
  "package_version": "2.0.0",
  "git_commit": "…",               // or null
  "created": "2026-…",
  "epoch": "2000-01-01T00:00:00-06:00",
  "timezone": "Etc/GMT+6",         // [output].timezone: the zone of every timestamp column
  "unusable_tests": ["…"]          // primary only
}
```

A stage refuses to write into a run directory whose recorded hash differs from the current
config's unless `--force` is given. With `--force`, it clears that stage's outputs and every
later stage's before writing.

**`run_summary.json`** (one per stage, written when a run ends; plan.md §3.9): `stage`,
`config_hash`, `package_version`, `started`, `finished`; unit (primary) or batch counts by
status; the totals of the per-unit or per-batch summary records (files loaded and missing,
slots by status, flagged-sample fractions by test, `unexcised_computed` slots, numpy-warning
counts); and wall time in total and per step.

---

## 7. Old → new names (for rebuilding downstream code)

| Old column | New location |
|---|---|
| `ws_{b}_mean` | `means` (ws, mean) |
| `u_{b}_mean` (streamwise) | `means` (ws, vector_mean) |
| `wd_{b}_mean` | `means` (wd, mean) |
| `wd_{b}_unit_mean` | `means` (wd, unit_mean) |
| `u_{b}_rms`, `v_{b}_rms`, `w_{b}_rms` | `boom_stats` sigma_u, sigma_v, sigma_w |
| `ws_{b}_rms` | `boom_stats` (ws, std) |
| `wd_{b}_rms` | `boom_stats` (wd, std) |
| `w'u'_{b}_mean`, `w'v'_{b}_mean`, `u'v'_{b}_mean` | `boom_stats` (uw / vw / uv, cov) |
| `w'vpts'_{b}_mean` | `boom_stats` (wvpts, cov) |
| `ti_{b}`, `tiu_{b}`, … | `boom_stats` ti, ti_u, … |
| `tke_{b}`, `ctke_{b}`, `ustar_{b}` | `boom_stats` tke, ctke, ustar |
| `L_{b}`, `sparam_{b}` | `boom_stats` obukhov_length, zeta |
| `ITS-efolding_{u,v,w}_{b}` | `boom_stats` ({u,v,w}, its) |
| `ILS-efolding_{u,v,w}_{b}` | `boom_stats` ils_u, ils_v, ils_w |
| `te-momt_{b}`, `te-heat_{b}` | `boom_stats` (uw, te), (wvpts, te) |
| `vpt_{b}_mean`, `t_{b}_mean`, … | `slow` vpt, t, … |
| `ws_{b}_max_{p}s`, `ws_{b}_rms_{p}s` | `means` (ws, max_{p}s), (ws, std_{p}s) |
| `Rib_{i}-{j}` | `pairs` rib (i, j) |
| `veer10m_{b}` | `pairs` veer (b, 4) |
| `alpha`, `gamma`, `wdgamma` | `profile` |
| `ustar_neutral_fit`, `z0_neutral_*_fit` | `profile` loglaw_* |
| `l1_{b}`…, `anisotropy_{b}`, `beta_{b}`, `phi_{b}` | `boom_stats` aniso_*, `boom_labels` aniso_class |
| `sun_elevation`, `night` | `slot_stats` / `slot_final` |
| `quality` | removed (boom-level filtering replaces record classes) |
| `tiglobal_{b}`, `Rif_*`, `spoleto`, `pp` flags | removed |
