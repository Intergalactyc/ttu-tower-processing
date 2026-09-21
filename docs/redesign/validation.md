# Validation

How the rewrite is tested and what the first real run must check before anyone relies on its
numbers. The tests of each phase are listed in that phase of plan.md; this file gives the
principles they follow (§1), the synthetic data they use (§2), and the first-run validation
(§3–§4).

---

## 1. Test principles

- **Test the numerics against identities and independent computations, not against the old
  pipeline's outputs**, which are not ground truth. The only old-code regression fixture is for
  Stage A (units, tilt, rotation), whose constants are taken as truth.
- **Synthetic data with known answers** for every estimator: covariances, timescales, spectra,
  injected defects.
- **Synthetic tests verify; they don't validate.** They show that the implementation does what
  the specification says. Where a criterion comes from a prototype of the same algorithm (the
  Phase 7 time-series scenarios), passing shows that the implementation matches the prototype,
  not that the algorithm suits the real data; that evidence comes from §3.
- **Invariances are tests:** seam invariance, batch-partition invariance, parallel = serial,
  translation invariance of every windowed operation, and rotation invariance where the physics
  demands it (plan.md §3.7).
- **Tolerances:** `rtol=1e-12` for invariance tests and `1e-10` relative for "aggregated =
  direct" identities; statistical tests state their sample size and tolerance. Discrete outputs
  (masks, flags, statuses, counts) must match exactly.
- **Determinism:** every random test uses `np.random.default_rng(seed)` with a fixed seed.
- **Never weaken a failing test to make it pass;** report it (plan.md §1.4).
- **Layout:** `tests/unit/<module path>/test_<module>.py`; real-data tests in
  `tests/integration/`, marked `integration`.
- **Isolation:** an autouse fixture sets `TTU_TOWER_HOME` to a temporary directory for every
  test, so the suite never touches the real ttu-tower home. Real-data tests read `TTU_RAW_DIR`
  (default `C:/Users/ellwalke/Data/TTU_200m_Nov2013-Oct2014`), skip if it is absent, and write
  only to pytest's `tmp_path`.

---

## 2. Synthetic data (`ttu_tower/validation/synthetic.py`)

A package module, not test-only, because the first-run checks reuse it. Every generator takes an
`rng` and returns float64 arrays at 50 Hz. Every other test relies on these generators, so each
is tested against its own analytic properties first (the OU autocorrelation and variance, the
`correlated_ou` covariance matrix, the wave periods and amplitudes, the positions of injected
defects), with stated sampling tolerances.

### 2.1 Ornstein–Uhlenbeck processes
- `ou(n, T, sigma, rng)`: the exact discretization x_{k+1} = a·x_k + σ√(1 − a²)·ε_k, with
  a = exp(−Δt/T) and x_0 drawn from the stationary distribution; ACF = exp(−lag/T).
- `correlated_ou(n, T, sigmas, corr, rng)`: several series sharing drivers through a Cholesky
  factor of `corr`, all with timescale T; covariance σ_i σ_j corr_ij.

### 2.2 Turbulence plus waves (a known gap)
`turbulence_with_waves(n, rng, *, T, sigma_w, sigma_u, sigma_theta, rho_wu, rho_wtheta,
wave_periods_s, wave_std_u, wave_std_w, wave_std_theta, coupling)`, with no defaults: every test
passes one of the presets below.
- The turbulent (w, u, θ) come from `correlated_ou` with timescale T.
- The waves stand for non-turbulent motion: for each period, one sinusoid per variable with a
  random phase and amplitude a = std·√(2/n), n the number of periods, so the waves' total
  spread equals the preset's std.
- Over whole periods, a wave's flux between two variables is (a₁a₂/2)·cos Δφ, with Δφ their
  phase difference, and `coupling` sets Δφ:
  - `"random"`: each variable draws its own phases, so each realization's wave flux is random,
    zero on average;
  - `"opposing"`: the variables share phases, with signs chosen so that the waves' flux has the
    *opposite* sign to the turbulent flux of the same pair. The waves then carry their full
    flux, std₁·std₂, in every realization: a systematic contamination of long windows;
  - `"aligned"`: the same, with the *same* sign as the turbulent flux, which makes long windows
    overestimate instead. Both signs occur in the data (the sign of the slow flux is close to a
    coin flip at booms 4–6 on strongly stable nights).
- The components are returned separately, so tests know the turbulence-only answer.

**Presets**, two regimes of the gap at two strengths:

| Preset | Turbulence | Wave periods | Wave spread (std) | Coupling |
|---|---|---|---|---|
| `MESOSCALE_TYPICAL` | T = 1 s; σ_w = 0.5, σ_u = 1.0 m/s, ρ_wu = −0.35 (w'u' = −0.175 m²/s²); σ_θ = 0.3 K, ρ_wθ = 0.3 | 20, 30, 45, 60 min | u 0.56, w 0.052 m/s; θ 0.28 K | random |
| `MESOSCALE_STRESS` | the same | the same | u 0.84, w 0.067 m/s; θ 0.28 K | opposing |
| `STABLE_TYPICAL` | T = 0.5 s; σ_w = 0.14, σ_u = 0.25 m/s, ρ_wu = −0.33 (w'u' = −0.01155 m²/s²); σ_θ = 0.1 K, ρ_wθ = −0.3 | 60, 90, 150, 240 s | u 0.23, w 0.027 m/s; θ 0.07 K | random |
| `STABLE_STRESS` | the same | the same | u 0.37, w 0.041 m/s; θ 0.07 K | opposing |

Adding a 30-s period to a `STABLE` preset (keeping its spread) puts a wave right next to the
turbulence, where the gap closes.

**Where the numbers come from.** The old pipeline's one-year 10-min and 1-min tables at this
tower: measurements summarized by old code, adequate for the orders of magnitude these tests
need and not validated beyond that. Stability classes are Ri_b(2, 4) as in `[post.stability]`.
- `STABLE` turbulence: medians at booms 1–4 in the strongly stable class (σ_w 0.13–0.14,
  σ_u 0.24–0.25 m/s; ρ_wu −0.33 at booms 1–2). T is one timescale for all variables, between the
  e-folding timescales of w (about 0.2 s) and u (about 2 s).
- `MESOSCALE` turbulence: within the interquartile ranges at boom 4 in the neutral class
  (σ_w 0.46–0.71, σ_u 0.90–1.45 m/s, ρ_wu −0.33 to −0.22); T between the timescales of w
  (0.28 s) and u (4 s).
- Wave spreads: the median (typical) and upper quartile (stress) of the spread of slow means:
  1-min means within 20-min windows at booms 2–4 on strongly stable nights (periods of about
  2–20 min) for `STABLE`; 10-min means within 80-min windows at boom 4 in unstable conditions
  (about 20–80 min) for `MESOSCALE`. The `STABLE` periods put that spread at 1–4 min, next to
  the gap: harder than the observed spread over 2–20 min.
- Coupling: the observed correlation of slow u and w is partial (typically ±0.3–0.5) and of
  either sign relative to the turbulent flux. A coupled preset at the stress spread is
  therefore an upper bound: the waves then carry 0.32 (`MESOSCALE_STRESS`) and 1.31
  (`STABLE_STRESS`) times the turbulent flux, inside the observed range (the slow flux's upper
  quartile reaches 0.6–2.9 times the turbulent flux at booms 4–6 on strongly stable nights).
- The θ parameters aren't in those tables and are judgment. Detection is tested on the momentum
  cospectrum; θ matters only to the end-to-end identity test.

Sinusoids give the clean, known gap the detection tests need. Real non-turbulent motion is
intermittent and broadband; the first-run checks (§3.1, §3.5, §3.7) show how detection behaves
on it.

### 2.3 Defect injection
- `inject_spikes(x, positions, lengths, amplitudes)`: runs of 1–6 samples at k × the local robust
  σ; per-sample signed amplitudes give bursts that alternate sides of the median.
- `inject_stuck(x, start, length)`: holds a value constant (dropouts).
- `inject_gaps(x, starts, lengths)`: sets NaN.
- `quantize(x, step)`: rounds to a resolution.
- `level_shift(x, start, delta)`: an excursion longer than a spike.

### 2.4 Synthetic raw files
`write_raw_dataset(dir, half_hours, booms, rng, *, mean_speed=8.0, mean_from_deg=270.0,
waves=MESOSCALE_TYPICAL, missing=(), mutate=None) -> truth` makes a dataset the pipeline reads
like the real one:
1. It generates each boom's series in SI and in the pipeline's own frame: Earth-frame ue, vn, w
   (the mean wind plus §2.2), ts (a mean plus the θ of §2.2), and t, rh and p as slow random
   walks with realistic quantization.
2. `write_raw_files(dir, series, half_hours, booms, missing, mutate)` writes them as the tower
   records them, by inverting Stage A: rotation (u_raw = −vn, v_raw = ue), the boom's tilt (the
   transpose of its orthonormal tilt matrix), then source units (mph, °F, %, inHg), as float32.
   Files are converted-style Parquet named with the real convention
   (`FT2_E07_C03_R…_D…_T….parquet`, record numbers increasing), with 90,000 rows and every
   canonical column, including the propeller columns. `mutate(half_hour, boom, df)` injects
   defects per file. It takes any SI series, so another generator can reuse it.
3. It writes `synthetic.json` next to the files (generator, parameters, seed, numpy and package
   versions) and returns `truth`: the generated SI series per boom, before inversion.

The Phase 6 and Phase 7 tests use it, so they need no real data.

---

## 3. First-run validation

Run on the pilot month (plan.md Phase 10: suggested 2014-04, all booms, all stages), then on the
production year. Each check is `ttu-validate CONFIG --check <name>` and writes CSVs to
`<run dir>/reports/validation/<name>/`.

Stability classes come from `post.classify` with the configured scheme on `rib` for the
configured boom pair. Every table is broken down by class and boom unless stated. **The checks
produce evidence; Elliott makes the calls.** The "reading" notes say what a result would
suggest, not what to change.

### 3.1 `tau_agreement`: heat vs momentum τ
- **Where both** are `found`/`capped` (`mrd`): the distribution of the rung difference
  log2(τ_heat/τ_momentum), and the fractions agreeing exactly and within ±1 rung.
- **Selected source:** the heat / momentum / fallback rates, and each cospectrum's status
  distribution.
- **Informs:** `peak_significance_se`, and keeping heat first vs changing `priority` or `rule`.
- **Reading:** disagreement concentrated near neutral is the risk of putting heat first.

### 3.2 `despike_calibration`: the threshold
- **Removal rates:** on a stratified sample of pilot files (e.g. 30 per class, fixed seed),
  rerun `despike` at thresholds {3.5, 4, 4.5, 5, 6} and report spike and excursion fractions
  per variable. For comparison, also run the old method (whole-file linear detrend,
  median/MAD, z > 5, no duration rule, single pass), reimplemented inside the check.
- **Injection:** into clean segments, inject spikes of 1–3 samples at {4, 6, 8, 12} × robust σ;
  report the detection rate vs threshold, and the false-removal rate on uninjected segments.
- **Excursions:** their length distribution per variable and class, and every excursion of
  4–10 samples listed with its values and surroundings for inspection. A short same-sign noise
  burst can't be told from the start of a genuine extreme by the rule, so how often short
  excursions occur, and what they look like, decides whether the rule needs more.
- **Informs:** `qc.despike.z_threshold`, and whether `max_spike_samples` or the rule itself
  needs revisiting.

### 3.3 `its_bias`: record-length bias and τ/ITS
- **Synthetic:** OU with T ∈ {0.5, 1, 2, 5, 10, 30, 60} s; the ITS at each rung divided by
  (1 − e⁻¹)·T, as a function of τ/T (20 realizations each).
- **Real:** distributions of τ/ITS for u, v, w and vpts at the selected τ; the `its_short` and
  `its_short_vpts` rates.
- **Informs:** `min_tau_its_ratio`, and whether ILS profiles need a caveat.

### 3.4 `interp_bias`: filling gaps of 1 s or less
- **Synthetic:** `correlated_ou` with a known covariance; blank random gaps of 1–50 samples at
  total fractions {0.1, 0.5, 1, 2, 5}%, fill with `fill_short`, and report the bias of
  covariances and variances at each rung vs fraction and gap length.
- **Real:** clean real segments with the same blanking; truth = the unblanked series.
- **Informs:** `max_fill_gap_samples`.

### 3.5 `floor_peak`: the small-scale end
- The distribution of `peak_scale_s` (both cospectra) at booms 1–3, and the fraction of slots
  whose peak is at the smallest analysed mode (0.59 s).
- The fraction of `found` detections clipped to `min_tau_s` (τ itself at or below 9.375 s - a
  reversal one rung above the floor floors τ just the same, so this isn't simply the reversal at
  or below 9.375 s).
- **Informs:** `mrd.floor_level` (16 halves the floor), and whether a 4.69-s rung is worth
  adding.
- **Reading:** a peak often at the floor means the floor may cut it; frequent clipping at the
  low booms on stable nights means the minimum τ includes submesoscale flux there.

### 3.6 `yield`: QC yield
The primary report (plan.md Phase 6) plus the stability report (Phase 9): flag rates by test,
and the fraction of computed slots surviving tertiary filtering per variant; also the
`direction` flag rate against the old pipeline's shadowed fraction, if Elliott supplies the old
report. **Reading:** a class or boom losing far more data than the others suggests
over-aggressive QC there.

### 3.7 `tau_profile`: τ(z) by stability
The median and IQR of the selected τ (`mrd`) and the source mix. Expected physics: τ grows with
height and with instability. This is the most direct evidence for per-boom τ.

### 3.8 `mrd_vs_naive`: what MRD changes
The ratios `mrd`/`naive` (median, IQR) of ustar, (wvpts, cov), ti, sigma_u, ils_u and zeta;
also `mrd`/`mrd_unexcised` where computed (how much excision changes results).

### 3.9 Sanity items
- The `p_measured` = 0 rate (slots whose pressure couldn't be converted) and the distribution of
  `p_factor`.
- `unresolved` and `capped` rates.
- The fraction of slots with `unexcised_computed`.
- Detection `n_pairs` at the 20-min mode.
- The `mrd_frame` wind direction vs the slot's.

---

## 4. Reporting to Elliott

After the pilot, one report: each check's key table (trimmed), a two-line reading, and the
decisions it informs:

| Decision | Informed by |
|---|---|
| `z_threshold` | §3.2 |
| `max_spike_samples`, the spike rule | §3.2 (excursions) |
| `peak_significance_se` | §3.1 and the detection-status rates |
| `min_tau_its_ratio` | §3.3 |
| `max_fill_gap_samples` | §3.4 |
| `floor_level` | §3.5 |
| `priority` / `rule` | §3.1 |

Config values change only on Elliott's decision. Then rerun the affected stages
(secondary-only changes need no primary rerun).
