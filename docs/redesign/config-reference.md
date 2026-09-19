# Configuration reference

Every tunable parameter, with its default and where the value comes from. One TOML file (read
with `tomllib`) describes one pipeline run and is parsed into typed frozen dataclasses (plan.md
Phase 0). Its file stem is the run's **tag** unless `tag` is set.

**Provenance codes**

| Code | Meaning |
|---|---|
| LIT | Literature value (reference given). |
| INST | Instrument specification. |
| PROD | The old production config (`ttu-windprofiles/configs/templates/oneyear_TEMPLATE.ini`): what actually ran. What existed, not a validated value. |
| OLDC | A constant in the old code (`old-tower-processing/src/ttu_tower/definitions.py`). What existed. |
| DES | A design decision (design-reference.md). |

**Provisional** means the value is checked or recalibrated in the first-run validation
(validation.md §3).

**Naming:** `min_…` and `max_…` are inclusive limits on the quantity named; a `…_fraction` is a
fraction of samples, from 0 to 1; units are suffixed (`_s`, `_samples`, `_min`). Where a key
was renamed from the old pipeline, the notes give the old name.

**Which stage a key affects.** Each stage's outputs record a hash of the config sections they
depend on (plan.md §3.8). Changing a key invalidates that stage and every later one:

| Sections | First stage affected |
|---|---|
| `[paths].raw_dirs`, `[output]`, `[period]`, `[files]`, `[qc.*]`, `[mrd]`, `[ladder]`, `[primary]` (except `nproc`) | primary |
| `[secondary.*]` | secondary |
| `[tertiary.*]`, `[paths].mesonet_dir` | tertiary |
| `tag`, `[paths].results_dir`, `[primary].nproc`, `[post.*]` | none (where and how fast, not what; `[post]` is read only by reports) |

**Validation at load:** types, the ranges in the notes, and the cross-key rules at the end.
Each failure raises `ConfigError` naming the key. Unknown keys are an error, which catches
typos.

**Environment variables:**
- `TTU_TOWER_HOME`, if set, is the ttu-tower home (plan.md §3.10). The test suite always sets it
  to a temporary directory.
- `TTU_RAW_DIR`, if set, overrides `raw_dirs` for the test suite's real-data tests only, never
  for pipeline runs.

---

## Top level

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `tag` | str | config file stem | — | The run's identifier: it names the run directory and the link file, and `load_results` takes it (plan.md §3.10). Letters, digits, `.`, `_` and `-` only. |

## `[paths]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `raw_dirs` | list[str] | **required** | — | Directories scanned (non-recursively) for raw files. Converted `.parquet` files are expected; `.csv`, `.csv.gz` and `.zip` files are read only with `--allow-non-parquet`. |
| `results_dir` | str | `""` | — | Where run directories go: `<results_dir>/<tag>/`. `""` = `results/` in the ttu-tower home. Must be absolute (after `~` expansion), so it doesn't depend on where a command is run. A year of output is several GB, so a data drive may suit better than the home directory. |
| `mesonet_dir` | str | `""` | — | Mesonet data directory (with `stationinfo.xls` and `data/`), as in the old pipeline. `""` disables the merge. |

## `[output]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `timezone` | str | `"Etc/GMT+6"` | DES | The zone every output timestamp is labeled in, and in which `[period]` is read: `"Etc/GMT+6"` (local standard time, UTC−6) or `"UTC"`. Zones with DST are rejected. The choice changes labels only, never values or slot boundaries. |

## `[period]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `start` | str | `""` | — | `"YYYY-MM-DD HH:MM"` in `[output].timezone`, inclusive, on a 10-min boundary. `""` = the start of the first accepted file. |
| `end` | str | `""` | — | Exclusive, on a 10-min boundary. `""` = the end of the last accepted file. |

The period selects which slots are emitted. Files just outside it are still used as context, so
a slot's results don't depend on the period's limits.

## `[files]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `bad_records` | list[[int, int]] | **required** | PROD + DES | Inclusive record-number ranges, dropped before anything else. Template value below: the old 19 ranges plus six single records. |
| `max_name_offset_min` | int | `2` | DES | Accept file names 0..N minutes after a :00/:30 boundary. Range 0–29. |
| `booms` | list[int] | `[1, …, 10]` | — | A subset, for test runs. |

## `[qc]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `min_coverage` | float | `0.75` | DES | The coverage threshold *c*: the minimum usable fraction for a partial block or window to count (plan.md §3.5). Range (0, 1]. |
| `max_fill_gap_samples` | int | `50` | PROD | Interior data gaps up to this many samples (1 s) are filled linearly and count as usable. Old: `max_interpolation_gap`. |
| `unusable_tests` | list[str] | `["unchecked", "resolution", "dropouts", "skew", "kurt", "direction", "bounce"]` | DES | Tests whose intervals make samples unusable; each must be a `quality` test. `bounds` and `spike` are allowed but not in the default: their samples are already removed and, if the gap is short, filled. Recorded in `run_meta.json`. |

### `[qc.bounds]` (SI units, applied after unit conversion)

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `sonic` | [float, float] | `[-45.0, 45.0]` | OLDC | m/s, each raw sonic component (north, west, up), before tilt correction. |
| `ts` | [float, float] | `[233.15, 333.15]` | OLDC | K. |
| `t` | [float, float] | `[223.15, 323.15]` | OLDC | K. |
| `rh` | [float, float] | `[0.0, 1.0]` | OLDC | Fraction. |
| `p` | [float, float] | `[50.0, 110.0]` | OLDC | kPa. |

The old code's `ws`, `wd` and propeller bounds are dropped: the derived speed can't exceed
45·√2 < 65 m/s, and the propellers aren't loaded.

### `[qc.despike]` (applied to ue, vn, w, ts, t, rh and p)

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `window_s` | float | `300.0` | LIT (VM97) | Length of the reference window. |
| `stride_s` | float | `30.0` | DES | The reference is evaluated every stride and interpolated in between. |
| `z_threshold` | float | `3.5` | DES, **provisional** | A sample is an outlier if 0.6745·\|x − median\|/MAD exceeds this. Recalibrated in the first run (validation.md §3.2). The old value, 5.0 on a whole-file reference, isn't comparable. Old: `despiking_deviations`. |
| `max_spike_samples` | int | `3` | LIT (VM97) | Outlier runs up to this long are spikes (removed); longer ones are excursions (kept and recorded). Runs are split where they cross the median first (plan.md Phase 2). |
| `min_mad` | table | `{ ue = 0.001, vn = 0.001, w = 0.001, ts = 0.01, t = 0.002, rh = 0.00002, p = 0.0004 }` | DES | Lower bound on the MAD. For ts, t, rh and p: one quantization step, measured on real files. For the winds, whose stored values are effectively continuous: a chosen floor well below the weakest turbulent MADs. |

### `[qc.windows]` (resolution, dropouts, higher moments; applied to ue, vn, w and ts)

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `resolution_window_s` | float | `100.0` | LIT (VM97) / PROD | Test window for resolution and dropouts. Old: `discrete_window_s`. |
| `resolution_stride_s` | float | `50.0` | OLDC | Half-overlapping windows, as in the old code. |
| `resolution_bins` | int | `100` | LIT / PROD | Histogram bins for u, v, w. Old: `discrete_bins`. |
| `ts_resolution_bins` | int | `30` | PROD | Histogram bins for ts. Old: `ts_discrete_bins`. |
| `max_empty_bin_fraction` | float | `0.8` | PROD | A window is flagged `resolution` when more than this fraction of its bins is empty. Old: `resolution_capacity`. |
| `dropout_min_run` | int | `10` | PROD | Minimum run of consecutive same-bin sample pairs that counts as a dropout (u, v, w). Old: `dropout_run_length`. |
| `ts_dropout_min_run` | int | `50` | PROD | The same, for ts. Old: `ts_dropout_run_length`. |
| `ts_repeat_tolerance` | float | `0.05` | PROD | K. For ts, "same value" means \|Δ\| ≤ this. Old: `ts_dropout_tolerance`. |
| `max_dropout_fraction` | float | `0.05` | PROD | A window is flagged `dropouts` when more than this fraction of its sample pairs are dropout pairs. Old: `dropout_capacity`, which applied to a whole 30-min record. |
| `dropout_min_windows` | int | `2` | OLDC | A pair is a dropout only if marked in at least this many overlapping windows. |
| `skew_range` | [float, float] | `[-2.0, 2.0]` | LIT (VM97), **provisional** | Slot skewness (after a linear detrend) outside this range → `skew`. |
| `kurt_range` | [float, float] | `[1.0, 8.0]` | LIT (VM97), **provisional** | Slot kurtosis (Gaussian = 3) outside this range → `kurt`. Old: both ranges were `higher_moments_limits`. |

### `[qc.slow]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `smoothing_width_s` | table | `{ t = 20, rh = 20, p = 2 }` | INST | Full width of the Hann window that smooths each slow-sensor series: twice the sensor response time (RM Young 41382V T/RH ≈ 10 s, RM Young 61302V barometer ≈ 1 s). Its −3 dB point is at 0.72/width. A positive multiple of 0.04 s. |

### `[qc.second_layer]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `shadow_sector` | [float, float] | `[105.0, 170.0]` | OLDC | Open interval, in true FROM-bearing, tested on the slot's first-layer vector-mean wind direction (the tower wake). |
| `bounce_max_ws_mean` | float | `30.0` | OLDC | m/s, the slot's first-layer scalar mean wind speed. Old: `BOUNCE_MAX_WS`. |
| `bounce_max_ws_std` | float | `6.75` | OLDC | m/s, the slot's first-layer standard deviation of wind speed. Old: `BOUNCE_MAX_STD`. |

## `[mrd]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `floor_level` | int | `15` | DES, **provisional** | K: the 80-min detection window is 2^K floor blocks (0.146 s at 15), and there are K modes. The floor must sit below the cospectral peak (validation.md §3.5). |

## `[ladder]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `its_max_lag_fraction` | float | `0.25` | PROD | Maximum ACF lag for the integral timescales, as a fraction of the τ-block length. |
| `gust_periods_s` | list[int] | `[1, 2, 3, 5, 10, 30, 60]` | PROD | Block lengths for gust statistics; each must divide 600. Old: `sample_periods`. |

## `[primary]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `unexcised` | bool | `true` | DES | Compute the `mrd_unexcised` variant where the mask matters. |
| `unexcised_max_fill_gap_s` | int | `600` | DES | The longest within-run data gap `mrd_unexcised` fills. |
| `batch_max_files` | int | `96` | DES | Maximum number of files in a batch (96 = 2 days). Hashed, because batches name the output fragments. |
| `nproc` | int | `0` | — | Primary's worker processes; 0 = CPU count − 1. |

## `[secondary]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `min_tau_its_ratio` | float | `10.0` | DES, **provisional** | `its_short` is set when τ/ITS for u, v or w is below this, `its_short_vpts` likewise for vpts (output-schema.md §4.3, validation.md §3.3). |

### `[secondary.detection]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `peak_significance_se` | float | `2.0` | DES, **provisional** | The turbulence peak must be at least this many standard errors from zero; with no significant peak the status is `weak`. |
| `min_scale_s` | float | `0.5` | LIT (VM06) | Modes with a smaller scale are ignored (with K = 15, only mode 1, 0.29 s). |
| `min_tau_s` | float | `9.375` | DES | Smallest τ; shorter gap scales are clipped to it. A ladder rung. |
| `max_tau_s` | float | `1200.0` | DES | Largest τ (`capped`). A ladder rung. |

### `[secondary.selection]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `rule` | str | `"priority"` | DES | `"priority"` or `"longest_significant"` (plan.md Phase 7). |
| `priority` | list[str] | `["heat", "momentum"]` | DES | The order in which the two cospectra are consulted. |
| `fallback_tau_s` | float | `600.0` | DES | τ when neither cospectrum gives one, and the floor applied to an unresolved lower bound. |

## `[tertiary]`

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `min_coverage` | float | `0.75` | DES | Minimum usable fraction of a value's support. Old: `nan_threshold`, a *maximum* NaN fraction of 0.10. |
| `max_bounds_fraction` | float | `0.01` | PROD | Maximum fraction of a support's samples with `bounds` rows, per input variable. Old: `bounds_flag`. |
| `max_spike_fraction` | float | `0.01` | PROD | The same for `spike` rows. Old: `despiking_flag`. |
| `veer_reference_boom` | int | `4` | OLDC | Veer is relative to this boom (10.1 m). Old: `veer10m`. |

### `[tertiary.fits]`: every profile-fit rule

Filtering runs first, so a boom whose input value was filtered is already absent
(output-schema.md §5.4).

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `exclude_tau` | list[str] | `["unresolved", "fallback"]` | DES | For the second-moment fits of `mrd` and `mrd_unexcised` (gamma, wdgamma, constrained z0): exclude booms whose selected τ has this `source_status` or `source`. Allowed: `found`, `capped`, `unresolved`, `fallback`. Included `capped` booms are counted in `<fit>_n_capped`. `naive` fits have no τ exclusions. Old: `fit_exclude_tau`. |
| `min_booms` | int | `2` | OLDC, **provisional** | Minimum valid booms for any fit, else NaN (a 2-point fit is exact, not a fit). Old: `minimum_present`. |
| `alpha_booms` | list[int] | `[1, …, 10]` | OLDC | Booms for the wind-speed power law. |
| `gamma_booms` | list[int] | `[1, …, 10]` | OLDC | Booms for the TI power law. |
| `wdgamma_booms` | list[int] | `[1, …, 10]` | `ttu-windprofiles` | Booms for the wd-standard-deviation power law. |
| `loglaw_booms` | list[int] | `[1, 2, 3, 4, 5, 6, 7]` | OLDC | Booms for the neutral log-law fits. |

## `[post.stability]` (read only by `ttu-report --stability` and `ttu_tower.post.classify`)

| Key | Type | Default | Prov. | Notes |
|---|---|---|---|---|
| `pair` | [int, int] | `[2, 4]` | OLDC | The boom pair whose Ri_b classifies stability. |
| `classes` | list[[str, str]] | see template | `ttu-windprofiles` `f4d1b57` | (name, interval) pairs, intervals written `(a,b)`, `[a,b)`, with `inf`. |

## Constants (code, not config)

In `ttu_tower/constants.py`:
- booms and heights; site latitude 33.61055°, longitude −102.05056°, elevation 1014 m;
- source timezone `Etc/GMT+6`; 50 Hz and 90,000 rows per file;
- the 80-min detection window and the 10-min Stage B margin;
- tilt tables (Kelly & Ennis); header maps and source units;
- per-boom reference pressures P_REF (ISA at site elevation + boom height).

In `ttu_tower/physics/constants.py`: R, c_p, R/c_p, p0 = 100 kPa, ε = 0.622, standard gravity,
κ = 0.41 and Earth's rotation rate. The Magnus (AERK) and ISA coefficients are in
`physics/thermo.py`, the Businger–Dyer coefficients in `physics/most.py`; local gravity comes
from `physics.earth.local_gravity`. The epoch is in `timegrid.py` (plan.md §3.1).

## Cross-key rules checked at load

- `mrd.floor_level` ≥ 9, so that floor blocks divide the finest (9.375-s) block.
- `qc.despike.window_s / stride_s` is an even integer, and 1800 is a multiple of `stride_s`;
  `resolution_stride_s` divides 1800.
- Every Stage B reach fits in the 10-min margin: `despike.window_s` + `despike.stride_s`,
  `resolution_window_s` + `resolution_stride_s`, max(`smoothing_width_s`)/2 and
  `unexcised_max_fill_gap_s` are each ≤ 600 s.
- `min_tau_s` ≤ `fallback_tau_s` ≤ `max_tau_s` ≤ 1200, all ladder rungs; `min_scale_s` <
  `min_tau_s`.
- Every name in `unusable_tests` is a known `quality` test.
- `veer_reference_boom` ∈ `booms`; every `tertiary.fits.*_booms` ⊆ `booms`.
- `output.timezone` ∈ {`"Etc/GMT+6"`, `"UTC"`}; `results_dir` is `""` or absolute.

---

## Template: `configs/templates/oneyear_TEMPLATE.toml`

Created in Phase 0, then copied to `configs/oneyear.toml` for running. Configs in `configs/`
itself are git-ignored; `configs/templates/` is tracked.

```toml
# One year of TTU 200 m tower data (Nov 2013 - Oct 2014).

[paths]
raw_dirs = ["C:/Users/ellwalke/Data/TTU_200m_Nov2013-Oct2014"]
results_dir = ""          # "" = ~/.ttu-tower/results; otherwise an absolute path
mesonet_dir = "C:/Users/ellwalke/Data/Mesonet_2012-2014"

[output]
timezone = "Etc/GMT+6"

[period]
start = "2013-11-01 00:00"
end = "2014-11-01 00:00"

[files]
bad_records = [
    [2348, 2496], [3727, 3728], [3957, 3967], [4340, 4398], [6677, 6945],
    [7437, 7575], [8154, 8356], [8721, 8759], [9118, 9243], [9370, 9382],
    [9448, 9791], [9872, 9876], [9910, 9912], [11257, 11300], [12570, 12645],
    [12971, 12981], [13139, 13387], [17338, 17358], [18199, 18223],
    [6946, 6946], [15022, 15022], [15028, 15028], [15045, 15045],
    [15061, 15061], [15648, 15648],
]
max_name_offset_min = 2
booms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

[qc]
min_coverage = 0.75
max_fill_gap_samples = 50
unusable_tests = ["unchecked", "resolution", "dropouts", "skew", "kurt", "direction", "bounce"]

[qc.bounds]
sonic = [-45.0, 45.0]
ts = [233.15, 333.15]
t = [223.15, 323.15]
rh = [0.0, 1.0]
p = [50.0, 110.0]

[qc.despike]
window_s = 300.0
stride_s = 30.0
z_threshold = 3.5
max_spike_samples = 3
min_mad = { ue = 0.001, vn = 0.001, w = 0.001, ts = 0.01, t = 0.002, rh = 0.00002, p = 0.0004 }

[qc.windows]
resolution_window_s = 100.0
resolution_stride_s = 50.0
resolution_bins = 100
ts_resolution_bins = 30
max_empty_bin_fraction = 0.8
dropout_min_run = 10
ts_dropout_min_run = 50
ts_repeat_tolerance = 0.05
max_dropout_fraction = 0.05
dropout_min_windows = 2
skew_range = [-2.0, 2.0]
kurt_range = [1.0, 8.0]

[qc.slow]
smoothing_width_s = { t = 20, rh = 20, p = 2 }

[qc.second_layer]
shadow_sector = [105.0, 170.0]
bounce_max_ws_mean = 30.0
bounce_max_ws_std = 6.75

[mrd]
floor_level = 15

[ladder]
its_max_lag_fraction = 0.25
gust_periods_s = [1, 2, 3, 5, 10, 30, 60]

[primary]
unexcised = true
unexcised_max_fill_gap_s = 600
batch_max_files = 96
nproc = 0

[secondary]
min_tau_its_ratio = 10.0

[secondary.detection]
peak_significance_se = 2.0
min_scale_s = 0.5
min_tau_s = 9.375
max_tau_s = 1200.0

[secondary.selection]
rule = "priority"
priority = ["heat", "momentum"]
fallback_tau_s = 600.0

[tertiary]
min_coverage = 0.75
max_bounds_fraction = 0.01
max_spike_fraction = 0.01
veer_reference_boom = 4

[tertiary.fits]
exclude_tau = ["unresolved", "fallback"]
min_booms = 2
alpha_booms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gamma_booms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
wdgamma_booms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loglaw_booms = [1, 2, 3, 4, 5, 6, 7]

[post.stability]
pair = [2, 4]
classes = [
    ["strongly unstable", "(-inf,-0.05)"],
    ["unstable", "[-0.05,-0.01)"],
    ["neutral", "[-0.01,0.01)"],
    ["stable", "[0.01,0.1)"],
    ["strongly stable", "[0.1,inf)"],
]
```
