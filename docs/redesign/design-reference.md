# Design reference: TTU tower pipeline rewrite

What the pipeline does and why each design choice was made. How to build it is in `plan.md`;
parameters are in `config-reference.md`; the output contract is in `output-schema.md`; tests and
first-run checks are in `validation.md`; terms are in `definitions.md`.

Everything here is settled. If implementation exposes a concrete problem with a decision, stop
and raise it with Elliott rather than working around it.

Names: this repository is **`ttu-tower-processing`** 2.0.0 (import package `ttu_tower`,
commands `ttu-*`). The previous pipeline is **`old-tower-processing`** (same internal names,
version 1.0.0), called "the old pipeline" throughout. The library `windprofiles` is no longer
a dependency; the pieces needed are ported.

---

## A. Scope and environment

- **One site: the TTU 200 m tower.** 10 booms at 0.9, 2.4, 4.0, 10.1,
  16.8, 47.3, 74.7, 116.5, 158.2 and 200.0 m, each with a Gill R3-50 sonic anemometer at 50 Hz
  and slow sensors: RM Young 41382V temperature/humidity (response time ≈ 10 s) and RM Young
  61302V barometer (≈ 1 s). 30-min raw files. No generality beyond this tower; it keeps the
  code simple.
    - Primary data target is Nov 2013 - Oct 2014.
- **Environment:** Python 3.13 (`>=3.13,<3.14`, because `wtxmeso` needs < 3.14). No
  `windprofiles`; `wtxmeso` stays an external dependency at `@main`; `arch`/`statsmodels` are
  dropped until the stationarity flag is built (L).
- **Consumers:** `ttu-windprofiles` (the paper) will be rebuilt on this output, out of scope
  here; output-schema.md §7 maps the old column names. `duststorms` stays on the old pipeline.
  There is no compatibility layer. Old and new can't share an environment, so comparisons go
  through output files.
- **Configuration:** one TOML file per run, parsed into typed frozen dataclasses. Every
  uncertain numeric parameter is config; site facts (heights, tilt tables, instrument
  constants) are code. Starting values come from the old production config: what existed, not
  validated values.
- **Results live outside the package** (plan.md §3.10). A ttu-tower home in the user's home
  directory holds a registry of runs and, by default, the results; `[paths].results_dir` puts
  results elsewhere. Downstream code loads results by run tag. The old pipeline wrote results
  inside its repository, which tied data to a code checkout.

## B. Stages

- **Primary** does everything that needs the raw 50-Hz data. **Secondary** computes per-slot,
  per-boom (or boom-independent) quantities from primary's outputs, with no cross-boom or
  external data. **Tertiary** does everything cross-boom, external (mesonet), or needing the
  assembled flags. **Post** is helpers outside the pipeline (stability classification, whose
  thresholds still change often). Each stage can be rerun alone, and everything downstream of
  the raw data can be retuned without reprocessing it.
- **Primary never chooses τ.** It stores MRD detection outputs and statistics at every rung of
  the τ ladder (D). Detection and selection happen in secondary, so every τ choice is retunable
  by rerunning secondary and tertiary only.
  - **Stage A** (per file, as loaded): per-sample transforms only (C).
  - **Stage B** (per file, with context from its neighbours): every windowed operation, from
    despiking to the block sums (E, F, G).
  - **Products** (per slot): detection outputs and the ladder, which need ±35 min of Stage B
    output around the slot.
- **Secondary:** τ detection → selection → selected-τ statistics and derived quantities per
  variant; slow-sensor thermodynamics; sun elevation.
- **Tertiary:** mesonet merge → boom-level filtering → multi-boom quantities → profile fits.
- **Cut from scope:** the flux Richardson number (choice-sensitive); the steady-state/ITC test
  (MRD's gap-based averaging addresses what it tested); the stationarity flag is deferred (L).

## C. Raw data and per-sample processing

- **File-timing rule, in this order:**
  1. drop `bad_records`: 19 hand-curated record ranges plus R06946, R15022, R15028, R15045,
     R15061 and R15648;
  2. accept only file names 0 to +2 min after a :00/:30 boundary, rounding down to it;
  3. if two files map to one half-hour, reject both.

  File-name stamps lag by 0–2 min in naming only (adjacent files are continuous), while
  clock-anomaly episodes produce names hours or days off. Dropping `bad_records` first removes
  those episodes before rounding can misplace them. A few likely-good late-stamped files are
  lost in exchange for near-total timestamp confidence. On the real listing: 16,817 files,
  1,686 bad records, 133 rejected offsets, 0 collisions, **14,998 accepted = 85.6% of the
  17,520 half-hours**.
- **Every file has exactly 90,000 rows**, so sample time = file start + index × 0.02 s, and each
  file is exactly 3 slots.
- **Input format:** converted Parquet. Unconverted `.csv`/`.csv.gz`/`.zip` files are read only
  with `--allow-non-parquet`: each boom's work unit then parses the whole file, which is slow
  and meant for small runs. Propeller anemometer columns are never loaded.
- **Per-sample order: units → bounds (in the raw sonic frame) → triplet coupling → tilt →
  rotation to Earth axes.** Tilt and rotation mix components, so a bad value must be removed
  before it leaks into the others. **Triplet coupling:** a sample that is NaN or removed in one
  of u, v, w is removed in all three, for the same reason; ts, t, rh and p are independent.
- **Tilt correction:** the per-boom constants of Kelly & Ennis (Sandia report on this tower),
  taken as truth. They are residual tilts relative to the datalogger's TRANS channels, the
  stored north/west/up components: applied there, they turn the directional dependence of mean
  w from sinusoidal to near-flat. The old implementation is ported exactly. (Background: in the
  raw files the TRANS channels are an exact, fixed per-boom rotation M_b of the sonic's native
  channels TSU/TSV/TSW, which conversion drops; M_b already contains a 1.3–3.5° tilt.)
- **Wind direction:** true FROM-bearing (degrees clockwise of north) everywhere, with one
  conversion pair (`bearing_to_vector` / `vector_to_bearing`) and no `flip`-style parameter.
  The old pipeline arrived at this after several sign errors; don't rediscover them.
- **Time zones:** file names are in local standard time, `Etc/GMT+6` (no DST). Output
  timestamps are labeled in `Etc/GMT+6` or `UTC`, a config choice; zones with DST are rejected so
  that timestamps are never discontinuous or ambiguous. Slots are labeled by their start; sun
  quantities are evaluated at the slot center.

## D. Cadence, windows and the τ ladder

- **Output cadence: 10-minute slots**, the same for every boom. 60-minute records mix stability
  regimes and smear out the brief near-neutral transitions the study cares about; per-boom record
  lengths would stagger timestamps across booms.
- **Detection and averaging are decoupled.** An **80-minute detection window** centered on each
  slot, recomputed for every slot, is used only to find τ. Statistics use τ-wide blocks centered
  on the slot. Plain 10-minute MRD can't see convective gaps of 15–30 min, which would cause
  systematic data loss and a selection bias in unstable conditions.
- **MRD modes are labeled by averaging time scale** (Vickers & Mahrt's convention). The
  detection window is averaged onto 2^K floor blocks (K = 15: 0.146 s, 4,096 per slot). Mode i
  has scale P_i = 2^i floor blocks = 0.29296875 s × 2^(i−1), from P_1 = 0.29 s to P_15 = 80 min.
  Its value is the mean, over the window's blocks of width P_i, of ΔxΔy/4, where Δ is the
  difference between the block's two half means. Mode i has 2^(15−i) blocks: 2 at 40 min, 1 at
  80 min.
- **Identity:** the sum of the modes with P ≤ τ equals the average within-block covariance over
  blocks of width τ. So a statistic at rung τ contains exactly the modes up to scale τ.
- **τ is the gap scale:** if the cospectrum reverses (changes sign or grows in magnitude) at mode
  r, τ = P_(r−1), the scale just before the reversal (VM06).
- **The τ ladder:** 9.375 s, 18.75 s, 37.5 s, 75 s, 2.5 min, 5 min, 10 min, 20 min, i.e. 80 min
  / 2^j. Powers of two only: anything between rungs breaks exact tiling and has no clean
  meaning in a block decomposition.
  - **Minimum 9.375 s.** Shorter gap scales occur in the weakest turbulence (VM06 report a few
    seconds); how often τ is clipped to the minimum is checked in the first run. Synthetic
    stable cases (validation.md §2.2, turbulence timescale 0.5 s) show both costs of the
    minimum. The 9.375-s blocks truncate the turbulent flux (to 0.91 of it). And when
    non-turbulent motion reaches down to a 30-s period, the gap closes: the reversal falls
    below 9.375 s in 35 of 50 cases, part of the wave stays inside the blocks, and the flux at
    the selected τ is 0.80 of the turbulent flux (0.87 without the 30-s wave). No power-of-two
    rung separates the two there.
  - **Maximum 20 min:** a gap at τ is confirmed by the reversal at the next scale up. 20 min is
    confirmed at the 40-min mode, which has 2 blocks; 40 min would need the 80-min mode, which
    has 1 block, no standard error, and a single sample of the mesoscale.
- **Why 80 minutes** (= 10 min × 2³): it puts a rung at exactly 10 min, and every rung ≤ 10 min
  tiles the slot exactly; with 60 min, no rung ≤ 10 min divides the slot. 40 min is too short
  for convection; 160 min mixes regimes within detection.
- **The naive window is the 10-min rung** (the same estimator with τ fixed), so MRD and naive
  results differ only in τ. It is also the wind-energy convention (IEC 61400-1).
- **Maximum wind speed and gusts** are over the slot: the largest sample, and the maximum and
  standard deviation of 1, 2, 3, 5, 10, 30 and 60-s block means.
- **Two slot-mean wind directions:** the bearing of the mean vector, which weights each sample
  by its speed, and the bearing of the mean unit vector, which weights every sample equally.
  They differ when the direction changes with the speed, as in gusty or meandering conditions;
  the second is the one to average over slots.

## E. Quality control

- **Flags are intervals** (start, end, test, boom, variable) of global sample indices, produced
  once at each test's own grain; any later window asks what fraction of it a test flagged.
  Flags therefore never need rerunning, and a future test is just new rows.
- **Two kinds of test:**
  - *removal tests* (`bounds`, `spike`) set samples to NaN; the gap rules then apply, so short
    removals are filled and usable. The fraction of a value's samples removed is judged later,
    by tertiary (`max_bounds_fraction`, `max_spike_fraction`), because that is a judgment about
    the consumer's window;
  - *window tests* mark intervals unusable: `unchecked`, `resolution`, `dropouts` (100-s
    windows), `skew`, `kurt` (slot), and the second-layer tests `direction` and `bounce` (slot).
    Which window tests make data unusable is config (all of them by default).
- **Variables:** despiking runs on all seven series (ue, vn, w, ts, t, rh, p). Resolution,
  dropouts and higher moments run on the sonic outputs only: the skewness of a slow signal over
  10 min is meaningless.
- **Despiking:** a 5-min median/MAD reference (VM97), evaluated every 30 s and interpolated
  (an exact per-sample rolling MAD at 50 Hz is too slow). A run of consecutive outliers is first
  split where it crosses the median, since at 50 Hz a real signal cannot jump from one side of
  the reference to the other without passing through normal values; a run that does is a noise
  burst, however long. Each piece of ≤ 3 samples is a spike and is removed; a longer one is an
  *excursion*, kept and recorded for a future discontinuity flag. A sliding reference alone
  would clip genuine extremes in intermittent stable turbulence; the duration rule prevents
  that, and splitting instead at every large jump would undo it (in a prototype that removed a
  third of a genuine burst's samples). The threshold (3.5 robust σ) is provisional and
  recalibrated in the first run. The MAD has a floor (`min_mad`): one
  quantization step for ts, t, rh and p, measured on real files; 0.001 m/s for the winds, whose
  stored values are effectively continuous. Samples with no valid reference within half a
  window are flagged `unchecked`.
- **Gap filling:** interior gaps of ≤ 1 s are filled linearly and count as usable. Longer gaps
  are unusable and are never bridged.
- **Higher moments:** per slot, after a linear detrend, with the VM97 limits (skewness
  [−2, 2], kurtosis [1, 8]), provisional.
- **The two-layer mask.** Quality flags give a first-layer mask; first-layer slot means give the
  `direction` flag (slot-mean wind direction in (105°, 170°), the tower wake) and the `bounce`
  flag (slot-mean speed > 30 m/s, or its standard deviation > 6.75 m/s); those give the final
  mask. Without the second layer, wake-affected data would feed τ detection and the
  neighbouring slots' windows. The second layer applies to ue, vn, w and ts (ts comes from the
  same acoustic paths), not to the slow sensors, which the wake doesn't meaningfully affect.
- **Masks per variable, combined per quantity.** A quantity uses the combined mask of its inputs
  (its *family*): momentum = ue ∧ vn ∧ w; heat = w ∧ vpts; ts = ts. A bad ts never costs a
  momentum quantity.
- **Excision plus coverage.** Primary leaves unusable samples out of every calculation and
  records coverage; tertiary filters values by coverage. **One coverage threshold c = 0.75**
  decides whether any partial block or window counts: floor blocks, MRD halves, τ-blocks, the
  20-min window, despiking references, smoothing kernels, gust blocks, higher-moment slots and
  the first-layer means.
- **Slow-sensor smoothing:** each t, rh and p sample is replaced by a NaN-aware weighted mean
  over a Hann window centered on it, of full width twice the sensor's response time (20 s for
  t and rh, 2 s for p).
  - Evaluated at every sample with a window that tapers to zero, the smoothed series has no
    steps. (Block means held constant over each block would step at every block edge.)
  - The −3 dB point is at 0.72/width (0.036 Hz for t/rh, 0.36 Hz for p). The filter still
    passes 94% at the sensor's own −3 dB frequency, 1/(2π·response time), so it removes what
    the sensor can't have measured (noise, quantization flicker) without compounding the
    sensor's response. Its sidelobes are −31 dB.
  - The smoothed series feed the slow-sensor slot means and the pressure factor (I), not any
    flux.
- **QC yield is reported early**, per test and later by stability class. Flags correlate with
  meteorology (resolution ↔ weak turbulence ↔ strongly stable), and an earlier QC retuning of
  the old pipeline had to be reverted within a day.

## F. MRD detection outputs (primary)

- **Pre-averaging:** the whole stream is averaged once onto the global floor grid, with
  fractional weights for samples straddling a block edge; each detection window is a slice of
  2^K floor blocks. Modes at a scale depend only on block means at that scale and above, so
  results above the floor are unchanged, at negligible cost. The floor must stay below the
  cospectral peak (roughly 0.5–6 s at boom 1 in stable conditions); K is config and checked in
  the first run.
- **Valid-pair rule:** block means use usable samples only. At each scale, a block contributes
  only if both halves have coverage ≥ c, judged directly at that scale. Missing data costs
  pairs and never injects values. Detection never uses data interpolated across long gaps:
  linear ramps would inject a spurious cospectral term growing with the square of the scale.
- **Stored per slot, boom and variant, for every mode:** value, standard error (the spread of
  the block products) and valid-block count, for the cospectra w'θv', w'u', w'v', u'v' and the
  variance spectra of u, v, w, ts. The variance spectra show how sensitive σ_u is to a one-rung
  change in τ.
- **Frame:** one fixed rotation to the detection window's mean wind, with its angle stored;
  any other fixed frame is recoverable from w'u', w'v', u'v' and the variances. The heat
  cospectrum doesn't depend on the frame. Per-scale alignment is rejected: at small scales a
  block's "mean" is an eddy.

## G. Statistics ladder (primary)

- **Second-moment statistics at every rung**, instead of at one τ. Primary stores per-9.375-s
  sums of usable samples in a fixed frame; these aggregate exactly to every coarser rung and to
  the 20-min window.
  - For τ ≤ 10 min, τ-blocks tile the slot. Blocks with coverage < c are dropped and the rest
    averaged; blocks used/total are recorded.
  - For τ = 20 min there is one 20-min window centered on the slot, judged against c like any
    τ-block: it has a value or is NaN.
- **Per-block alignment:** each τ-block's covariance tensor is rotated to that block's own mean
  wind (exact, since rotation is linear). The wind-direction standard deviation is Yamartino's
  single-pass estimate, which aggregates through sin/cos sums.
- **Integral timescales at every rung** for u, v, w and vpts, e-folding method, from the raw
  samples (they don't aggregate). Fluctuations are about each τ-block's mean (the rung is the
  high-pass, so no detrend); u and v are in the block's streamwise frame; the ACF is pooled over
  the rung's fully usable blocks; it is integrated to the first 1/e crossing with a maximum lag
  of 0.25 × the block length, and ITS is NaN if 1/e isn't reached. τ/ITS is reported per variable
  and flagged (`its_short`) when any velocity component is below a threshold, or when its ACF
  never crossed 1/e, because a short τ biases ITS low; the longest timescale decides, which in
  this tower's data is u or v, never w. The vpts timescale uses the blocks its own family makes
  usable, so a broken sonic component doesn't cost it; multiplied by the mean wind it gives the
  temperature length scale, the counterpart of the velocity ones. It carries its own flag, so a
  long temperature scale doesn't discredit the velocity scales.
- **Transport efficiencies** of u'w' and w'θv' per rung, also from raw samples, pooled over the
  rung's blocks (Salesky et al. 2017).
- **Denominators:** a variance is normalized by a mean over the same support (the slot for
  τ ≤ 10 min, the 20-min window for τ = 20 min).
- **Variants** (an explicit `variant` key everywhere):
  - `mrd`: excised data at the selected τ; the primary result.
  - `naive`: the 10-min rung of `mrd`.
  - `mrd_unexcised`: the counterfactual "what if nothing had been excised": flagged data are
    kept as measured, and data gaps (missing values, bounds and spike removals) of up to 10 min
    are filled linearly within a file run, as the old pipeline did. Longer gaps stay NaN: the
    old pipeline bridged at most one file, and an unbounded fill would need unbounded context.
    Where no sample within reach differs between the two series, `mrd_unexcised` equals `mrd`,
    so it is computed only where the mask matters and copied elsewhere.

  The variant key propagates to everything derived from second moments (TI, TKE, u*, L, z/L,
  ILS, anisotropy, second-moment profile fits). Mean-only quantities are variant-independent.
- **Storage:** about 0.5 GB/yr per variant for the ladder, and 0.5 GB/yr for detection outputs.

## H. τ detection and selection (secondary)

- **Detection runs separately on the heat (w'θv') and momentum (w'u') cospectra**, from stored
  outputs only. The rule is Vickers & Mahrt's (2006, *BLM* 118:431; "VM06"), plus a
  significance test on the peak (plan.md Phase 7):
  - the cospectrum is smoothed 1–2–1 across modes, and scales below 0.5 s are ignored;
  - the peak is the first local maximum (a decrease in magnitude or a sign change follows it)
    that is at least 2 standard errors from zero; insignificant maxima are passed over;
  - the gap is the next increase in magnitude or sign change after the peak.
- **Why the one addition:** without a significance test every cospectrum has a peak, so pure
  noise gives a short τ, and there is no `weak` status for selection to fall back from. The
  comparison below uses the presets of validation.md §2.2, whose turbulence and wave amplitudes
  come from this tower's data, with 50 realizations per scenario. Each cell is the mean flux at
  the selected τ over the turbulent flux, and the share of realizations within 10% of the
  turbulence-only flux at the same τ. "Opposing" and "aligned" are the sign of the waves' flux
  against the turbulent flux.

  | Scenario | Tests on peak and reversal | VM06 | VM06 + peak test (adopted) |
  |---|---|---|---|
  | mesoscale stress, opposing | 0.99 (100%) | 0.99 (100%) | same as VM06 |
  | mesoscale stress, aligned | 1.11 (66%) | 1.00 (98%) | same as VM06 |
  | mesoscale typical | 1.01 (94%) | 1.00 (100%) | same as VM06 |
  | mesoscale typical, long turbulent timescale (T = 5 s) | 0.99 (78%) | 0.98 (82%) | same as VM06 |
  | stable stress, opposing | 0.87 (96%) | 0.87 (96%) | same as VM06 |
  | stable stress, aligned | 1.08 (44%) | 0.96 (92%) | same as VM06 |
  | stable typical | 1.03 (70%) | 0.98 (80%) | same as VM06 |
  | stable typical, weak flux (ρ = −0.1) | 1.21 (44%) | 0.94 (70%) | same as VM06 |
  | stable typical, very weak flux (ρ = −0.05) | 1.55 (28%) | 1.16 (38%) | same as VM06 |
  | zero flux (mesoscale / stable) | 21 / 1 of 50 `weak` | never `weak`; τ = 9.4 s from noise in 43 / 47 of 50 | 28 / 4 of 50 `weak` |

  Wherever there is a real flux the peak is significant, so the adopted rule and VM06 agree
  exactly; they differ only when there is nothing to find, where the adopted rule reports `weak`
  instead of a τ read from noise. The earlier rule (significance tests on the peak *and* the
  reversal) cuts too late when the waves' flux has the same sign as the turbulent flux, and
  overestimates weak fluxes. The adopted rule also follows the literature. Its own limits show
  in the last rows: weak fluxes are underestimated, very weak ones scatter badly, and a long
  turbulent timescale costs agreement.
- **Statuses** (exactly one per cospectrum): `no_data` (no usable data for that family in the
  slot), `weak` (no significant peak), `found` (τ from a reversal), `capped` (no reversal
  through the 40-min mode: τ = 20 min), `unresolved` (a mode became unusable first: only a
  lower bound). No fallback is substituted at this step.
- **Unresolved lower bound:** if mode u is the first unusable mode and no reversal was seen below
  it, any reversal lies at mode u or above, so τ ≥ P_(u−1).
- **Selection** gives one operative τ per slot, boom and variant, with its source:
  1. the first source in priority order (default heat, then momentum) that is `found` or
     `capped`;
  2. else the first `unresolved` source, at max(lower bound, 10 min): a bound below 10 min
     means the data ran out early, and using it would knowingly truncate the flux;
  3. else, if any data exist, the 10-min fallback.

  `longest_significant` (the largest `found`/`capped` τ) is a config alternative. The one τ
  applies to every quantity of that slot, boom and variant, as in VM06, who applied the heat
  gap scale to momentum too. **Heat first** because its cospectrum is usually better behaved
  (VM06), it doesn't depend on the frame, and it is less exposed to residual tilt (w'u' picks
  up δ·(σ_u² − σ_w²) at large scales, heat only δ·u'θ'). The cost: heat is insignificant across
  much of the near-neutral class, so the source switches there; the first run checks this.
- **Per-boom τ is the consistent definition across heights:** each boom's statistics contain
  exactly its own flux-carrying scales, whereas one common window is a height-dependent
  spectral cut. Second-moment profile fits are computed for every variant, and results say
  which they use. σ_u and TI are more sensitive to rung steps than fluxes (horizontal variance
  often has no gap); the variance spectra quantify this.

## I. Thermodynamics (primary and secondary)

- **vpts** (sonic virtual potential temperature) is computed in primary with a **fixed per-boom
  reference pressure** P_REF[b], the ISA pressure at the site elevation (1014 m) plus the boom
  height (≈ 89.7 kPa at boom 1, 87.6 kPa at boom 10). Pressure therefore never varies inside a
  window, and vpts's mask is ts's: a broken barometer never costs a heat flux.
- **Secondary converts the heat quantities to the measured pressure** with one factor per slot,
  f = (P_REF[b]/p̄)^(R/c_p), p̄ = the slot's mean smoothed pressure: w'θv' and the vpts mean ×
  f, the vpts variance × f². Transport efficiency is a ratio and unchanged; L and z/L use the
  converted flux. If the slot's pressure isn't usable, f = 1 and the value is marked as using
  the reference; that is off by ≲ 1%.
  - **One factor per slot, for every τ:** the slot and the 20-min window are both centered on
    the slot center, so their mean pressures agree to second order in the pressure tendency.
  - **Not per-sample pressure:** per-sample pressure turns the barometric tendency into a vpts
    trend inside each τ-block (~0.08 K per 10 min at 0.5 kPa/h). Its chance correlation with w
    changes a moderate heat flux by ~1% rms and a weak one by 10–15% rms (synthetic check; no
    bias) and inflates the vpts variance. Converting with a mean pressure is also the usual
    eddy-covariance convention.
  - Primary's own tables stay at P_REF; detection is unaffected by a constant factor.
- **Humidity:** the Magnus form with the Alduchov & Eskridge (1996) coefficients over water, for
  both e_s and the dewpoint, so e_s(T_d) = e exactly. The sensors report humidity relative to
  water, so the water form applies below 0 °C too. (It replaces the old Tetens form; the
  difference is ≲ 0.1% in e_s.)
- **Derived-quantity conventions:** a zero denominator gives NaN, never inf. Anisotropy
  (barycentric map, k = 2/3) is ported from the old pipeline and applied to the selected-τ
  tensor.

## J. Tertiary

- **Mesonet merge:** ported from the old pipeline, including its linear interpolation of mesonet
  gaps (which matters more at 10-min cadence).
- **Filtering is boom-level:** a failing value becomes NaN; whole records are never dropped, so
  NaN-safe fits and summaries just work. A value fails if its family's coverage over its
  support is below `min_coverage` (0.75), or if more than `max_bounds_fraction` /
  `max_spike_fraction` of an input variable's support was removed. Filtering precedes every
  multi-boom quantity. There is no record-level quality class (the old
  good/moderate/poor/shadowed labels); one could be a post helper later.
- **Profile fits** (power law of speed, TI and wd standard deviation; neutral log law) follow
  the rules of output-schema.md §5.4, all configurable. Second-moment fits of the MRD variants
  exclude booms whose τ is `unresolved` or the fallback (a naive-style definition mixed into an
  MRD profile), and include `capped` booms, counted separately. The power law uses the ported
  weighted log-space fit.
- **Multi-boom quantities:** bulk Richardson number for every boom pair, lapse rate of VPT, and
  veer relative to boom 4 (10.1 m). TI normalized by boom 10 (the old `tiglobal`) is dropped: it
  mixes τ definitions across booms.
- **Guidance for downstream analysis** (not pipeline code): within-boom ratios are
  τ-consistent; cross-boom second-moment normalizations mix τ definitions, so prefer local
  scaling.

## K. Processing

- **Batches and work units.** The timeline is split into batches of consecutive files; the
  work unit is (batch, boom), reading only that boom's columns. Two or more consecutive missing
  files (≥ 60 min) end a batch freely, since nothing on one side affects the other. Primary runs
  work units in parallel; secondary and tertiary run their batches serially, because they take
  minutes per year.
- **Streaming with context.** Within a unit, files are processed in order with a rolling
  buffer. Stage B processes each file on a span with fixed 10-min margins on each side, so each
  file's results are identical however the timeline is split; margin samples are recomputed as
  a neighbour's own file, which is bounded extra work.
- **Contiguity is not context.** A single missing file ends a *file run*: Stage B filters and
  gap fills never bridge it. MRD windows (±40 min) still use data beyond a 30-min gap. Windows
  that run past the data simply contain unusable samples, handled by the coverage rules, so no
  truncated-window logic is needed.
- **Every slot of the processing period is emitted** with a status: no accepted file, file but
  no usable sonic data, or computed. Files just outside the period are used as context when
  present, so a slot's results don't depend on the period's limits.
- **Logging:** one summary per work unit (also in its manifest entry), per-stage run summaries,
  and numpy warnings counted rather than logged one by one (plan.md §3.9).

## L. Deferred (none blocks the build)

- **Stationarity flag.** It would run per rung in primary (it needs raw data), with secondary
  taking the value at the selected τ. It would use flag kind `assumption` and add a filtering
  criterion of that kind. Room left: interval flags with a `kind`, filtering criteria kept as a
  list, and a per-rung raw-data hook next to the ITS computation. The old repository's ADF→PP
  Monte Carlo study is validated precedent for it.
- **Angle-of-attack correction** (Nakai & Shimoyama 2012). Room left: Stage A is an ordered list
  of per-sample steps, and the correction would go after triplet coupling, before tilt. It
  works in the sonic's native frame, which conversion drops, but native = M_b⁻¹·TRANS with M_b
  fixed per boom (C). M_b can be estimated by least squares from the 44 raw sample files in
  `C:\Users\ellwalke\Data\Sample\` (to ~10⁻⁶); confirm it is constant over the year first.
- **Discontinuity flag** (Haar, VM96): excursions are recorded for it.
- **Cage-shadowing flag:** a lead only (self-shadow centers near 97.5°, 217.5° and 337.5°, from
  coarse 30° bins); the second layer is where it would go.
- **Gust extremes by extreme-value theory** (e.g. a Gumbel/BLUE estimate of the expected slot
  maximum); the p-second block series it needs are produced.
- **PSD calculations**
- **Spectral corrections** (aliasing, path length): no scaffolding needed.
- **Research-grade synthetic data** for QC, ITS and MRD/τ studies and sensitivity analyses:
  turbulence synthesized from target spectra and cospectra (the test generators' OU processes
  have no inertial subrange and one shared timescale), realistic submesoscale motion, and a
  sensor model (path averaging, quantization, realistic defects). Room left:
  `write_raw_files` turns any SI series into raw files the pipeline reads, `truth` keeps the
  components separate, and `synthetic.json` records how a dataset was made (validation.md
  §2.4). It belongs in its own module; the test generators stay as they are, since test
  criteria are tuned to them.
- **Correlated consecutive values** when τ = 20 min: adjacent slots' 20-min windows overlap, a
  caveat for downstream uncertainty estimates.

## M. Pitfalls (don't reintroduce)

- **The tower-wake sector is (105°, 170°) in true FROM-bearing.** A rotated (285°, 350°) seen in
  older notes is wrong.
- **MRD modes are labeled by the width of the block they live in, not of its halves.** Under
  that labeling, τ = 20 min is confirmed at the 40-min mode (2 blocks). Labeling by half-width
  shifts every statement by one mode.
- **Old-pipeline behavior is history, not validation.** Port code where the plan says so, but
  "the old code did X" is not evidence that X is right. Validated precedents: literature values,
  the Kelly & Ennis tilt constants, and the old ADF→PP Monte Carlo study.
- **ITS by e-folding integrates to the first 1/e crossing.**
