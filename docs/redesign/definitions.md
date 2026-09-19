# Definitions

The terms used in these documents, with their exact meaning here. Durations assume 50-Hz
sampling and the default floor level K = 15.

---

## 1. Time intervals

From shortest to longest.

| Term | Length | Meaning |
|---|---|---|
| **sample** | 0.02 s | One row of a raw file, identified by its global **sample index g** = (time − epoch)/0.02 s. |
| **floor block** | 0.146484375 s (7.32 samples) | The unit of the global floor grid onto which the whole stream is averaged once (fractional weights at the edges); MRD is computed from these averages. 4,096 per slot. |
| **half** | 2^(i−1) floor blocks | One of the two halves of a block at MRD mode i. |
| **smoothing kernel** | 2 s (p), 20 s (t, rh) | The Hann window that smooths a slow-sensor series, centered on each sample in turn. Not a block. |
| **gust block** | p s, p ∈ {1, 2, 3, 5, 10, 30, 60} | Blocks tiling a slot, whose mean wind speeds give the gust statistics. |
| **finest block** | 9.375 s (468.75 samples) | The smallest τ-block. Primary stores sums per finest block, from which every coarser rung is aggregated exactly. 64 per slot. |
| **τ-block** | τ | A block of width τ. For τ ≤ 10 min the τ-blocks tile the slot; for τ = 20 min there is one, the 20-min window. Fluctuations are deviations from the τ-block's own mean. |
| **slot** | 10 min (30,000 samples) | **The output interval.** Slot k covers samples [30000k, 30000(k+1)) and is labeled by its start (`slot_start`). |
| **20-min window** | 20 min | [slot start − 5 min, slot start + 15 min): the single τ-block used when τ = 20 min, centered on the slot. |
| **file** (half-hour) | 30 min (90,000 samples) | One raw file, identified by its **half-hour index h**; file h holds slots 3h, 3h+1, 3h+2. |
| **detection window** | 80 min | [slot start − 35 min, slot end + 35 min): the window whose MRD cospectra are used to find a slot's τ. Recomputed for every slot; never used for averaging. |
| **batch** | ≤ `batch_max_files` files | Consecutive half-hours processed together, per boom. Its **core** files are the ones whose slots it emits; up to 3 files of **context** on each side are read only to compute them. |
| **outage** | ≥ 60 min | Two or more consecutive half-hours with no accepted file. Batches never span one. |
| **file run** | variable | A maximal sequence of consecutive accepted files. A single missing file ends it. Stage B filters and gap fills never cross a file-run boundary; detection windows do. |
| **processing period** | configured | [`period.start`, `period.end`): the slots emitted. Data just outside it are used as context. |
| **epoch** | — | 2000-01-01 00:00 local standard time (06:00 UTC). All integer indices count from it. |

**Durations that are values:**
- **τ** (the averaging time, or gap scale when it comes from detection): the width of the
  τ-blocks that define fluctuations for a slot's statistics. Always a **rung** of the ladder:
  9.375, 18.75, 37.5, 75, 150, 300, 600 or 1200 s. "Rung X" means τ = X s. τ_heat and τ_momentum
  are the two detected values; the selected τ is the one used; τ_lb is a lower bound.
- **Support:** the samples a value is computed from: the slot, or the 20-min window when
  τ = 20 min. Coverage and the flag fractions of filtering are evaluated over it.
- **MRD scale P_i** = 0.29296875·2^(i−1) s, i = 1..15 (0.29 s … 80 min): the width of the
  blocks in which mode i is computed. The modes with P ≤ τ make up the statistics at rung τ.

Easily confused: the **detection window** is used only to *find* τ; **τ-blocks** define the
*fluctuations*; the **support** is what the *average* covers. A **batch** is a unit of work
(days long) and never a "block".

## 2. Words with more than one sense

- **block:** always an averaging block (floor, half, gust, finest, τ-block, or an MRD block of
  scale P); never a batch.
- **window:** the detection window, a QC test window (100 s), the despiking reference window
  (5 min), the 20-min window, or the smoothing kernel. The documents always say which.
- **run:** a *file run* (§1); a *run of samples* sharing a property (a NaN run, an outlier run);
  or a *pipeline run*, one execution of the stages for one config and tag.
- **gap:** a *data gap* (consecutive NaN samples), or the *cospectral gap*, the range of scales
  separating turbulence from slower, non-turbulent motion (mesoscale, or submesoscale in stable
  conditions), whose location is τ.
- **pair:** in MRD, a block of scale P is a pair of halves, and `n_pairs` counts valid blocks;
  in the dropout test, a pair is two consecutive samples.
- **record:** the `R` number in a file name (a file counter). The old pipeline also used it for
  an averaging interval; the new design has slots instead.

## 3. Data, masks and QC

- **Raw sonic components:** the stored `u`, `v`, `w` of a converted file: the north and west
  components of the wind's FROM-vector, and up (the datalogger's TRANS channels).
- **Earth frame:** `ue` (east), `vn` (north), `w` (up), "blows-toward", after tilt correction.
- **Streamwise frame:** `u` along a block's mean wind, `v` 90° to its left, `w` up.
- **wd:** wind direction as a true FROM-bearing (degrees clockwise of north). A slot has two
  means: from the mean vector (speed-weighted) and from the mean unit vector.
- **Triplet:** (u, v, w) of one sonic sample, removed together.
- **Slow sensors:** t, rh and p.
- **vpts:** sonic virtual potential temperature, computed in primary with the boom's fixed
  reference pressure **P_REF[b]** and converted to the measured pressure in secondary
  (`p_factor`). **VPT:** virtual potential temperature from the slow sensors.
- **Family:** the combined mask of a group of quantities: `momentum` = ue ∧ vn ∧ w;
  `heat` = w ∧ vpts; `ts` = ts.
- **Usable:** a sample is usable if it is finite after short gap filling and outside every
  interval of the tests in `unusable_tests`. The **first-layer mask** leaves out the
  **second-layer** tests `direction` and `bounce`, which are computed from first-layer slot
  means; the **final mask** includes them. "Usable" means the final mask unless stated.
- **Coverage:** usable weight in a block or window ÷ its nominal length. **c**
  (`qc.min_coverage`) is the threshold for it to count.
- **Excision:** leaving unusable samples out of every calculation (the `mrd` variant).
- **Short fill / within-run fill:** linear interpolation of data gaps ≤ 1 s (all variants) /
  of gaps ≤ 10 min inside a file run (`mrd_unexcised` only).
- **Flag:** a row of the flag store, an interval `[start, end)` where a test failed, for one
  boom and variable (null variable = all four sonic outputs). A test's **kind** is `quality`,
  `record` (excursions) or, later, `assumption`.
- **Removal test / window test:** removal tests (`bounds`, `spike`) set samples to NaN; window
  tests mark intervals unusable.
- **Spike / excursion:** after splitting outlier runs where they cross the median, a piece of
  ≤ 3 samples (removed) / a longer one (kept and recorded). **Unchecked:** a sample with no
  valid despiking reference within half a window.
- **Filter group:** the input variables and families a quantity depends on (output-schema.md
  §5.6); a failure sets every quantity of the group to NaN.

## 4. MRD and τ

- **MRD (multiresolution decomposition):** splits a covariance into dyadic modes (HM97). Mode
  i's value is the mean, over the window's blocks of width P_i, of ΔxΔy/4, where Δ is the
  difference between the block's two half means.
- **Cospectrum / variance spectrum:** the modes of a covariance (w'θv', w'u', w'v', u'v') or of a
  variance (u, v, w, ts). Summing modes up to scale τ gives the statistic at rung τ.
- **Valid block:** both halves have coverage ≥ c. **SE:** the standard deviation of a mode's
  block products ÷ √(number of valid blocks).
- **Peak / reversal:** the first significant local maximum of the smoothed cospectrum / the
  next sign change or increase in magnitude after it, at mode r; then τ = P_(r−1).
- **Detection status:** `no_data`, `weak`, `found`, `capped`, `unresolved` (output-schema.md
  §1.6).
- **Source:** where the selected τ came from: `heat`, `momentum`, `fallback`, `fixed` (naive) or
  `none`.
- **Variant:** `mrd` (excised, selected τ), `naive` (the 600-s rung of `mrd`), `mrd_unexcised`
  (no excision, within-run fill), `none` (variant-independent quantities).
- **Ladder:** statistics at every rung, stored by primary.

## 5. Pipeline

- **Stages:** primary (needs raw data), secondary (per slot and boom, from primary's outputs),
  tertiary (cross-boom and external), post (helpers outside the pipeline).
- **Stage A / Stage B:** primary's per-sample transforms, per file as loaded / its windowed
  operations, per file on a span.
- **Span, core, margin:** Stage B processes a file (the **core**) plus a 10-min **margin** on
  each side (together, the **span**); only core results are kept. **Reach:** how far from a
  sample an operation reads; every reach fits in the margin.
- **Products:** primary's per-slot outputs (detection outputs, ladder, ITS/TE), computed once
  the Stage B outputs around the slot exist.
- **Placeholder:** the all-NaN stand-in for a missing file inside a batch's context.
- **Seam invariance:** results don't depend on where file or batch boundaries fall.
- **Work unit:** (batch, boom), primary's unit of parallelism and checkpointing. **Fragment:**
  one Parquet file of one table, written by one work unit (primary) or batch (secondary,
  tertiary). **Manifest:** per-unit status records, used to resume.
- **Config hash:** a per-stage fingerprint of the config sections the stage depends on.
- **ttu-tower home:** `~/.ttu-tower` (or the first `~/.ttu-tower<N>` with the signature file,
  or `TTU_TOWER_HOME`): holds the run registry and, by default, the results. **Tag:** a run's
  identifier. **Run directory:** where one run's outputs live. **Link file:**
  `<home>/runs/<tag>.json`, pointing a tag to its run directory.
- **Profile fit:** a per-slot fit across booms (power laws, log law), governed by the rules of
  output-schema.md §5.4.

## 6. Abbreviations and references

| Abbreviation | Meaning |
|---|---|
| VM97 | Vickers & Mahrt (1997), *J. Atmos. Oceanic Technol.* 14:512: QC tests (despiking, resolution, dropouts, higher moments). |
| VM06 | Vickers & Mahrt (2006), *Boundary-Layer Meteorol.* 118:431: the gap-scale averaging time. |
| HM97 | Howell & Mahrt (1997), *Boundary-Layer Meteorol.* 83:117: multiresolution flux decomposition. |
| K&E | Kelly & Ennis, the Sandia report on this tower: the tilt constants. |
| AERK | The Magnus coefficients recommended by Alduchov & Eskridge (1996). |
| ISA | International Standard Atmosphere (the reference pressures). |
| MAD | Median absolute deviation; robust σ ≈ MAD/0.6745. |
| ITS, ILS, TE | Integral time scale, integral length scale, transport efficiency. |
| TI, TKE | Turbulence intensity, turbulent kinetic energy. |
| u*, L, ζ | Friction velocity, Obukhov length, z/L (`zeta`). |
| Ri_b | Bulk Richardson number between two booms. |
